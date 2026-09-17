"""Runtime adapter that drives the loaded STEVE-1 policy step-by-step.

We support two policy interfaces (from MineStudio or the native steve1
package) without copying any of their code. The adapter exposes a single
public method, ``act(obs)``, returning an action dict the wrapper can
``step()`` with.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from ..util.image import decode_pov


logger = logging.getLogger("mineevolve.executor.steve_runner")


def _is_minestudio_policy(policy: Any) -> bool:
    """MineStudio >= 1.1 ``SteveOnePolicy``: ``get_action(input={'image', 'condition'}, state_in)``."""
    return hasattr(policy, "prepare_condition") and hasattr(policy, "get_action") and hasattr(policy, "device")


class _MineStudioAdapter:
    """Bridge between raw MineRL observations/actions and MineStudio's STEVE-1.

    * The policy sees a 128x128 RGB frame (VPT native; ``img_shape`` in the
      HF checkpoint config), so the env POV is resized here.
    * The policy emits a VPT factored action ``{'buttons', 'camera'}``; it is
      mapped back to the MineRL key/camera dict with the same
      ``CameraHierarchicalMapping`` + ``ActionTransformer`` MineStudio's own
      simulator uses (``MinecraftSim.agent_action_to_env_action``).
    """

    def __init__(self, policy: Any) -> None:
        from minestudio.simulator.entry import CameraConfig  # type: ignore
        from minestudio.utils.vpt_lib.action_mapping import CameraHierarchicalMapping  # type: ignore
        from minestudio.utils.vpt_lib.actions import ActionTransformer  # type: ignore

        self.policy = policy
        cam = CameraConfig()
        self.action_mapper = CameraHierarchicalMapping(n_camera_bins=cam.n_camera_bins)
        self.action_transformer = ActionTransformer(**cam.action_transformer_kwargs)
        img_shape = None
        try:
            img_shape = policy.net.img_preprocess.inshape  # not always exposed
        except AttributeError:
            pass
        self.img_hw = tuple(int(x) for x in img_shape[:2]) if img_shape else (128, 128)

    def frame(self, obs: Dict[str, Any]) -> np.ndarray:
        import cv2  # type: ignore

        pov = decode_pov(obs.get("pov", obs.get("image")))
        if pov is None:
            raise RuntimeError("STEVE-1 requires an observation with a 'pov'/'image' frame.")
        if pov.shape[:2] != self.img_hw:
            pov = cv2.resize(pov, dsize=(self.img_hw[1], self.img_hw[0]), interpolation=cv2.INTER_LINEAR)
        return pov

    def to_env_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        import torch  # type: ignore

        if isinstance(action, tuple):
            action = {"buttons": action[0], "camera": action[1]}
        if isinstance(action["buttons"], torch.Tensor):
            action = {
                "buttons": action["buttons"].detach().cpu().numpy(),
                "camera": action["camera"].detach().cpu().numpy(),
            }
        factored = self.action_mapper.to_factored(action)
        env_action = self.action_transformer.policy2env(factored)
        out: Dict[str, Any] = {}
        for k, v in env_action.items():
            v = np.asarray(v)
            out[k] = v.astype(np.float32) if k == "camera" else np.array(int(v))
        return out


@dataclass
class SteveCondition:
    text: str
    cond_scale: float = 4.0


class SteveRunner:
    """Step-wise driver around a loaded STEVE-1 policy."""

    def __init__(self, policy: Any) -> None:
        self.policy = policy
        self._condition_obj: Any | None = None
        self._state: Any | None = None
        self._condition_text: str | None = None
        self._adapter = _MineStudioAdapter(policy) if _is_minestudio_policy(policy) else None

    # ------------------------------------------------------------------
    # Condition lifecycle
    # ------------------------------------------------------------------

    def set_condition(self, condition: SteveCondition | str) -> None:
        if isinstance(condition, str):
            condition = SteveCondition(text=condition)

        # Called once per step by the server; only re-embed the prompt and
        # reset the recurrent state when the subgoal text actually changes,
        # otherwise STEVE-1 loses its temporal context every frame.
        key = f"{condition.text}\x00{float(condition.cond_scale)}"
        if self._condition_obj is not None and key == self._condition_text:
            return
        self._condition_text = key

        prepare = getattr(self.policy, "prepare_condition", None)
        if callable(prepare):
            import torch  # type: ignore

            with torch.no_grad():
                self._condition_obj = prepare({
                    "cond_scale": float(condition.cond_scale),
                    "text": str(condition.text),
                })
            initial_state = getattr(self.policy, "initial_state", None)
            if callable(initial_state):
                try:
                    self._state = initial_state(self._condition_obj, batch_size=1)
                except TypeError:
                    self._state = initial_state(condition=self._condition_obj)
            return

        # Native steve1 package path: encoders live on the agent itself.
        if hasattr(self.policy, "set_text_condition"):
            self.policy.set_text_condition(text=str(condition.text), scale=float(condition.cond_scale))
            self._condition_obj = condition
            self._state = None
            return

        # Last-resort: stash and apply per-step.
        self._condition_obj = condition
        initial_state = getattr(self.policy, "initial_state", None)
        if callable(initial_state):
            try:
                self._state = initial_state(condition=condition, batch_size=1)
            except TypeError:
                try:
                    self._state = initial_state(batch_size=1)
                except TypeError:
                    self._state = initial_state()
        else:
            self._state = None

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def act(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        if self._condition_obj is None:
            raise RuntimeError("STEVE-1 condition not set. Call set_condition() first.")

        if self._adapter is not None:
            import torch  # type: ignore

            # Batch the frame as [B=1, T=1, H, W, C] and use the "BT*" path:
            # the "*" path would re-batchify the tensors already inside the
            # prepared condition. Action tensors come back as [B, T, ...].
            frame = torch.from_numpy(self._adapter.frame(obs)).to(self.policy.device)[None, None]
            with torch.no_grad():
                action, self._state = self.policy.get_action(
                    {"image": frame, "condition": self._condition_obj},
                    self._state,
                    input_shape="BT*",
                )
            action = {k: v[0, 0] for k, v in action.items()}
            return self._adapter.to_env_action(action)

        step_fn = getattr(self.policy, "get_steve_action", None)
        if callable(step_fn):
            action, self._state = step_fn(
                self._condition_obj,
                obs,
                self._state,
                input_shape="*",
            )
            return action

        action_fn = getattr(self.policy, "get_action", None)
        if callable(action_fn):
            try:
                return action_fn(obs)
            except TypeError as exc:
                try:
                    result = action_fn(obs, self._state)
                except TypeError:
                    raise exc
                if isinstance(result, tuple) and len(result) == 2:
                    action, self._state = result
                    return action
                return result

        raise RuntimeError("Loaded STEVE-1 policy exposes no recognised action method.")

    def reset_episode(self) -> None:
        self._condition_obj = None
        self._condition_text = None
        self._state = None
        reset = getattr(self.policy, "reset_episode", None)
        if callable(reset):
            reset()
