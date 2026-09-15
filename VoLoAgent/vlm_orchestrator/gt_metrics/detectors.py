# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Passive GT metric detectors.

The detectors in this module consume two existing streams:

* ``obs["gt_state"]`` exported by the simulator/evaluation client.
* Structured orchestrator logs already produced by strategies.

They emit metric events only; they do not mutate observations, prompts,
detector state used by strategies, or recovery behavior.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from vlm_orchestrator.gt_metrics.events import GTMetricEvent, stable_id


COLOR_WORDS = {
    "red", "blue", "green", "yellow", "black", "white", "orange",
    "purple", "grey", "gray", "brown", "pink", "silver", "gold",
}

_OBJECT_NOUN_ALIASES = {
    "blocks": "block",
    "cube": "block",
    "cubes": "block",
    "rubik": "rubiks",
}

_SCENE_CATEGORY_ALIASES = {
    "bin": {"bin", "container"},
    "container": {"bin", "container"},
    "tray": {"bin", "container", "tray"},
}

_SCENE_OBJECT_PHRASE_ALIASES = {
    "bleach_cleanser": {
        "blue bottle",
        "blue cleanser",
        "bleach bottle",
        "cleanser bottle",
    },
    "coffee_can": {
        "blue can",
        "blue coffee can",
        "blue container",
        "blue jar",
        "blue lid jar",
        "blue object",
        "coffee can",
    },
    "corn_can": {"corn can", "yellow can", "corn tin"},
    "ketchup_bottle": {"ketchup bottle", "red bottle"},
    "mustard": {
        "mustard bottle",
        "yellow mustard",
        "yellow mustard bottle",
    },
    "mustard_bottle": {"mustard bottle", "yellow bottle"},
    "rubiks_cube": {"rubik cube", "rubiks cube", "toy cube"},
    "spam_can": {
        "spam can",
        "green can",
        "green tin",
        "green tin can",
        "sardine can",
    },
    "sugar_box": {
        "domino box",
        "domino sugar box",
        "sugar box",
        "yellow sugar box",
    },
    "tuna_can": {"tuna can", "blue can", "blue tuna can"},
    # LH-CS additions — VLMs describe these objects in natural language,
    # rarely use the canonical robolab object_id. Without these aliases
    # vlm_grasp_target_mismatch fires on Claude every time it names the
    # right object descriptively. (Observed 121 false positives in the
    # 4-VLM LH-CS sub+tools run on 2026-05-19; >90% were the two below.)
    "ranch_dressing": {
        "ranch dressing", "ranch bottle",
        "white bottle", "white bottle with green cap",
        "green-capped bottle", "white green-capped bottle",
        "green-capped white bottle", "bottle with green cap",
    },
    "crabbypenholder": {
        "crab toy", "red crab", "red crab toy", "crab penholder",
        "crabby penholder", "the crab", "the red crab",
        "red crab on the table",
        "red starfish", "red star", "red star toy",
    },
}

_PLACEHOLDER_OBJECT_NAMES = {
    "condition", "conditions", "object", "objects", "subtask", "subtasks",
}

_VLM_DECISION_TYPES = {
    "check",
    "vlm_check",
    "subgoal_check",
    "vlm_detect",
    "vlm_detection",
    "next_goal_check",
}

_PLAN_DECISION_TYPES = {
    "decompose",
    "recycle",
}

_TARGET_DECISION_TYPES = _VLM_DECISION_TYPES | {
    "gt_vlm_recovery",
}


def _norm_name(value: str | None) -> str:
    if not value:
        return ""
    return re.sub(r"\s+", "_", value.strip().lower())


def _display_name(value: str) -> str:
    return value.replace("_", " ")


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_]+", text.lower())


def _canonical_object_tokens(text: str) -> list[str]:
    result: list[str] = []
    for token in _tokens(text):
        if token.isdigit() or token == "s":
            continue
        token = re.sub(r"\d+$", "", token)
        if not token:
            continue
        result.append(_OBJECT_NOUN_ALIASES.get(token, token))
    return result


def _core_noun_from_name(name: str) -> str | None:
    words = _tokens(name.replace("_", " "))
    candidates = [
        w for w in words
        if len(w) >= 3 and w not in COLOR_WORDS
    ]
    if not candidates:
        return None
    return max(candidates, key=len)


def _object_colors(name: str) -> set[str]:
    return set(_tokens(name.replace("_", " "))) & COLOR_WORDS


def _gt_state(obs: dict[str, Any]) -> dict[str, Any]:
    gt = obs.get("gt_state")
    return gt if isinstance(gt, dict) else {}


def _subtask(gt: dict[str, Any]) -> dict[str, Any]:
    sub = gt.get("subtask", {})
    return sub if isinstance(sub, dict) else {}


def _task(gt: dict[str, Any]) -> dict[str, Any]:
    task = gt.get("task", {})
    return task if isinstance(task, dict) else {}


def _structured_task_checks(gt: dict[str, Any]) -> bool:
    task = _task(gt)
    return isinstance(task.get("success_checks"), list) or isinstance(
        task.get("invariant_checks"),
        list,
    )


def _conditions(gt: dict[str, Any]) -> list[dict[str, Any]]:
    conditions = _subtask(gt).get("conditions", [])
    return [c for c in conditions if isinstance(c, dict)]


def _condition_object(cond: dict[str, Any]) -> str | None:
    obj = cond.get("object") or cond.get("object_name")
    return str(obj) if obj else None


def _scene_objects(gt: dict[str, Any]) -> list[str]:
    """All objects present in the scene, derived from both the explicit
    ``scene_objects`` field and the ``object_states`` map.

    Some robolab tasks (e.g., KitUtensilSetsTask) export only containers
    in ``scene_objects`` while the full object inventory lives in
    ``object_states``. Without the union, _target_objects_from_gt
    misattributes the grasp target to the container — e.g., subgoal
    "pick the silver spoon and place it in the bowl" gets
    expected_targets=[bowl] because spoon_small isn't in scene_objects.
    """
    out: list[str] = []
    seen: set[str] = set()
    raw = gt.get("scene_objects", [])
    if isinstance(raw, list):
        for obj in raw:
            s = str(obj)
            if s and s not in seen:
                seen.add(s)
                out.append(s)
    obj_states = gt.get("object_states", {})
    if isinstance(obj_states, dict):
        for name in obj_states.keys():
            s = str(name)
            if s and s not in seen:
                seen.add(s)
                out.append(s)
    return out


def _is_real_scene_object(gt: dict[str, Any], obj: str | None) -> bool:
    if not obj:
        return False
    norm = _norm_name(obj)
    if norm in _PLACEHOLDER_OBJECT_NAMES:
        return False
    scene = {_norm_name(name) for name in _scene_objects(gt)}
    return not scene or norm in scene


def _current_subgoal_text(ctx: "MetricContext") -> str:
    return _subgoal_text(ctx, ctx.subgoal_idx)


def _subgoal_text(ctx: "MetricContext", subgoal_idx: int) -> str:
    subgoals = getattr(ctx.state, "subgoals", []) or []
    if isinstance(subgoals, list) and 0 <= subgoal_idx < len(subgoals):
        return str(subgoals[subgoal_idx])
    rewritten = getattr(ctx.state, "rewritten_instruction", None)
    if rewritten:
        return str(rewritten)
    return ""


def _entry_subgoal_idx(ctx: "MetricContext", entry: dict[str, Any]) -> int:
    raw = entry.get("subgoal_idx", ctx.subgoal_idx)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return ctx.subgoal_idx


def _entry_subgoal_text(
    ctx: "MetricContext",
    entry: dict[str, Any],
) -> str:
    subgoal = entry.get("subgoal") or entry.get("instruction")
    if isinstance(subgoal, str) and subgoal.strip():
        return subgoal
    return _subgoal_text(ctx, _entry_subgoal_idx(ctx, entry))


def _decision_kind(source_type: str) -> str:
    if source_type in _PLAN_DECISION_TYPES:
        return "plan"
    if source_type in _VLM_DECISION_TYPES:
        return "scene_success_check"
    if source_type == "gt_vlm_recovery":
        return "recovery_target_choice"
    return "control"


def _decision_id(
    ctx: "MetricContext",
    entry: dict[str, Any],
    *,
    subgoal_idx: int | None = None,
) -> str:
    entry_idx = ctx.subgoal_idx if subgoal_idx is None else subgoal_idx
    source_type = str(entry.get("type", ""))
    payload = (
        ctx.episode_id,
        ctx.episode_step,
        ctx.step_count,
        entry_idx,
        source_type,
        entry.get("status"),
        entry.get("action"),
        entry.get("grasp_target") or entry.get("target"),
        entry.get("subgoal") or entry.get("instruction"),
        entry.get("subgoals") or entry.get("new_subgoals"),
        str(entry.get("reason", ""))[:200],
    )
    return stable_id("vlmd", payload)


def _raw_vlm_field(entry: dict[str, Any], key: str) -> Any:
    raw = entry.get("vlm_raw")
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(rf'"{re.escape(key)}"\s*:\s*"([^"]+)"', raw)
        return match.group(1) if match else None
    if isinstance(parsed, dict):
        return parsed.get(key)
    return None


def _entry_vlm_value(entry: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = entry.get(key)
        if value:
            return value
    for key in keys:
        value = _raw_vlm_field(entry, key)
        if value:
            return value
    return None


def _entry_grasp_target(entry: dict[str, Any]) -> str:
    target = _entry_vlm_value(entry, "grasp_target", "target", "target_object")
    return str(target) if target else ""


def _decision_summary(
    ctx: "MetricContext",
    entry: dict[str, Any],
    *,
    subgoal_idx: int,
    subgoal_text: str,
) -> dict[str, Any]:
    source_type = str(entry.get("type", ""))
    return {
        "decision_id": _decision_id(ctx, entry, subgoal_idx=subgoal_idx),
        "decision_kind": _decision_kind(source_type),
        "source_log_type": source_type,
        "episode_id": ctx.episode_id,
        "episode_step": ctx.episode_step,
        "infer_count": ctx.step_count,
        "subgoal_idx": subgoal_idx,
        "subgoal_text": subgoal_text,
        "vlm_decision": {
            "status": entry.get("status"),
            "action": entry.get("action"),
            "target": _entry_vlm_value(entry, "target"),
            "grasp_target": _entry_vlm_value(entry, "grasp_target"),
            "reason": str(entry.get("reason", ""))[:1000],
            "raw": str(entry.get("vlm_raw", ""))[:1200],
        },
    }


def _objects_mentioned_in_text(
    text: str,
    objects: list[str],
) -> list[str]:
    if not text:
        return []
    text_tokens = _tokens(text.replace("_", " "))
    matches: list[tuple[int, str]] = []
    object_by_norm = {_norm_name(obj): obj for obj in objects}
    for obj in objects:
        obj_tokens = _tokens(obj.replace("_", " "))
        if not obj_tokens:
            continue
        for idx in range(0, len(text_tokens) - len(obj_tokens) + 1):
            if text_tokens[idx:idx + len(obj_tokens)] == obj_tokens:
                matches.append((idx, obj))
                break

    allowed_color_followers = {
        "block",
        "blocks",
        "cube",
        "cubes",
        "on",
        "onto",
        "top",
        "above",
        "over",
        "and",
        "then",
        "after",
    } | COLOR_WORDS
    matched_objects = {obj for _idx, obj in matches}
    for idx, token in enumerate(text_tokens):
        if token not in COLOR_WORDS:
            continue
        next_token = text_tokens[idx + 1] if idx + 1 < len(text_tokens) else ""
        if next_token and next_token not in allowed_color_followers:
            continue
        candidate = object_by_norm.get(f"{token}_block")
        if candidate and candidate not in matched_objects:
            matches.append((idx, candidate))
            matched_objects.add(candidate)

    matches.sort(key=lambda item: item[0])
    return [obj for _pos, obj in matches]


def _is_grab_condition(cond: dict[str, Any]) -> bool:
    predicate = str(cond.get("predicate", "")).lower()
    info = str(cond.get("info", "")).lower()
    return (
        "grab" in predicate
        or info.startswith("grabbed(")
        or cond.get("condition_idx") == 0
    )


def _target_objects_from_gt(
    gt: dict[str, Any],
    *,
    text_hint: str = "",
) -> list[str]:
    seen: set[str] = set()
    targets: list[str] = []

    if text_hint:
        scene_objects = _scene_objects(gt)
        container_relation = _container_relation_from_text(text_hint, scene_objects)
        if container_relation:
            return [container_relation["object"]]
        stack_relation = _stack_relation_from_text(text_hint, scene_objects)
        if stack_relation:
            return [stack_relation["top"]]
        mentioned = _objects_mentioned_in_text(text_hint, scene_objects)
        if mentioned:
            return [mentioned[0]]

    for cond in _conditions(gt):
        obj = _condition_object(cond)
        if (
            not obj
            or obj in seen
            or not _is_real_scene_object(gt, obj)
        ):
            continue
        if cond.get("satisfied") is False or not _is_grab_condition(cond):
            seen.add(obj)
            targets.append(obj)

    if targets:
        return targets

    completed = _subtask(gt).get("object_completed", {})
    if isinstance(completed, dict):
        for obj, done in completed.items():
            obj_str = str(obj)
            if (
                not done
                and obj_str not in seen
                and _is_real_scene_object(gt, obj_str)
            ):
                seen.add(obj_str)
                targets.append(obj_str)
    if targets:
        return targets

    return _scene_objects(gt)


def _target_matches_observed(target: str, expected: str) -> bool:
    norm_target = _norm_name(target)
    norm_expected = _norm_name(expected)
    if not norm_target or not norm_expected:
        return False
    if norm_target == norm_expected:
        return True
    target_tokens = set(_canonical_object_tokens(norm_target.replace("_", " ")))
    for alias in _SCENE_OBJECT_PHRASE_ALIASES.get(norm_expected, set()):
        alias_tokens = set(_canonical_object_tokens(alias))
        if alias_tokens and alias_tokens <= target_tokens:
            return True
    expected_tokens = set(
        _canonical_object_tokens(norm_expected.replace("_", " ")),
    )
    return bool(expected_tokens) and expected_tokens <= target_tokens


def _category_token(name: str) -> str | None:
    candidates = [
        token for token in _canonical_object_tokens(name.replace("_", " "))
        if len(token) > 1 and token not in COLOR_WORDS
    ]
    if not candidates:
        return None
    return candidates[-1]


def _category_aliases(category: str | None) -> set[str]:
    if not category:
        return set()
    return _SCENE_CATEGORY_ALIASES.get(category, {category})


def _resolve_observed_scene_target(
    observed_target: str,
    scene_objects: list[str],
) -> str | None:
    """Resolve a VLM target phrase to a unique scene object when possible."""
    if not observed_target or not scene_objects:
        return None

    direct = [
        obj for obj in scene_objects
        if _target_matches_observed(observed_target, obj)
    ]
    if len(direct) == 1:
        return direct[0]
    if len(direct) > 1:
        return None

    observed_tokens = set(
        _canonical_object_tokens(observed_target.replace("_", " ")),
    )
    if not observed_tokens:
        return None
    observed_colors = observed_tokens & COLOR_WORDS
    category_matches: list[str] = []
    observed_categories = {
        alias
        for token in observed_tokens
        for alias in _category_aliases(token)
    }
    for obj in scene_objects:
        obj_tokens = set(_canonical_object_tokens(obj.replace("_", " ")))
        category = _category_token(obj)
        if not category:
            continue
        scene_categories = _category_aliases(category)
        if scene_categories.isdisjoint(observed_categories):
            continue
        obj_colors = obj_tokens & COLOR_WORDS
        if observed_colors and obj_colors and observed_colors.isdisjoint(obj_colors):
            continue
        category_matches.append(obj)

    return category_matches[0] if len(category_matches) == 1 else None


def _resolve_spatial_scene_target(
    observed_target: str,
    scene_objects: list[str],
    gt: dict[str, Any] | None,
) -> str | None:
    """Resolve phrases like ``right container`` using GT object positions."""
    tokens = set(_canonical_object_tokens(observed_target.replace("_", " ")))
    directions = tokens & {"left", "right"}
    if len(directions) != 1:
        return None
    direction = next(iter(directions))
    observed_categories = {
        alias
        for token in tokens - {"left", "right"}
        for alias in _category_aliases(token)
    }
    if not observed_categories:
        return None

    candidates: list[tuple[float, str]] = []
    for obj in scene_objects:
        obj_tokens = set(_canonical_object_tokens(obj.replace("_", " ")))
        category = _category_token(obj)
        scene_categories = _category_aliases(category) | obj_tokens
        if scene_categories.isdisjoint(observed_categories):
            continue
        pos = _object_position(gt, obj)
        if pos is None:
            continue
        # RoboLab tabletop scenes use +Y as robot-right in the exported
        # frame; the prompt asks VLMs to emit directions in robot frame.
        candidates.append((pos[1], obj))

    if len(candidates) == 1:
        return candidates[0][1]
    if len(candidates) < 2:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[-1][1] if direction == "right" else candidates[0][1]


def _target_matches_scene_object(
    observed_target: str,
    expected: str,
    scene_objects: list[str],
) -> tuple[bool, str | None]:
    if _target_matches_observed(observed_target, expected):
        return True, expected
    resolved = _resolve_observed_scene_target(observed_target, scene_objects)
    return _norm_name(resolved) == _norm_name(expected), resolved


def _target_objects(ctx: "MetricContext") -> list[str]:
    return _target_objects_from_gt(
        ctx.gt_state,
        text_hint=_current_subgoal_text(ctx),
    )


def _failure_target_objects_from_gt(
    gt: dict[str, Any],
    *,
    text_hint: str = "",
) -> list[str]:
    if text_hint:
        stack_relation = _stack_relation_from_text(text_hint, _scene_objects(gt))
        if stack_relation:
            return [stack_relation["bottom"], stack_relation["top"]]
    return _target_objects_from_gt(gt, text_hint=text_hint)


def _score(gt: dict[str, Any]) -> float:
    try:
        return float(_subtask(gt).get("score", 0.0))
    except (TypeError, ValueError):
        return 0.0


def _object_completed(gt: dict[str, Any], obj: str) -> bool:
    completed = _subtask(gt).get("object_completed", {})
    if isinstance(completed, dict) and obj in completed:
        return bool(completed[obj])
    terminal = _terminal_condition(gt, obj)
    return bool(terminal and terminal[1])


def _terminal_condition(
    gt: dict[str, Any],
    obj: str,
) -> tuple[dict[str, Any], bool] | None:
    candidates = [
        cond for cond in _conditions(gt)
        if _condition_object(cond) == obj and not _is_grab_condition(cond)
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda c: int(c.get("condition_idx", 0) or 0))
    cond = candidates[-1]
    return cond, bool(cond.get("satisfied"))


def _subtask_condition_key(
    gt: dict[str, Any],
    ctx: "MetricContext",
    *,
    subgoal_idx: int | None = None,
) -> str:
    sub = _subtask(gt)
    current = subgoal_idx
    if current is None:
        current = sub.get("current_index")
    if current is None:
        current = ctx.subgoal_idx
    try:
        idx = int(current)
    except (TypeError, ValueError):
        idx = ctx.subgoal_idx
    return f"subtask_{idx}"


def _current_gt_subtask_idx(
    gt: dict[str, Any],
    fallback: int,
) -> int:
    try:
        return int(_subtask(gt).get("current_index", fallback))
    except (TypeError, ValueError):
        return fallback


def _current_gt_satisfied(
    gt: dict[str, Any],
    ctx: "MetricContext",
    *,
    target_objects: list[str] | None = None,
    subgoal_idx: int | None = None,
) -> bool | None:
    sub = _subtask(gt)
    targets = target_objects or _target_objects(ctx)
    completed = sub.get("object_completed", {})
    if isinstance(completed, dict) and targets:
        target_statuses = [
            bool(completed[obj])
            for obj in targets
            if obj in completed
        ]
        if target_statuses:
            return all(target_statuses)

    all_checks = sub.get("all_subtask_conditions", {})
    if isinstance(all_checks, dict):
        key = _subtask_condition_key(gt, ctx, subgoal_idx=subgoal_idx)
        if key in all_checks:
            return bool(all_checks[key])

    total = sub.get("total")
    completed_count = sub.get("completed")
    try:
        if int(total) > 0 and int(completed_count) >= int(total):
            return True
    except (TypeError, ValueError):
        pass
    if _score(gt) >= 1.0:
        return True
    return None


def _terminal_condition_for_context(
    ctx: "MetricContext",
    obj: str,
) -> tuple[dict[str, Any], bool] | None:
    terminal = _terminal_condition(ctx.gt_state, obj)
    if terminal is not None:
        return terminal
    satisfied = _current_gt_satisfied(
        ctx.gt_state,
        ctx,
        target_objects=[obj],
    )
    if satisfied is None:
        return None
    return (
        {
            "object": obj,
            "predicate": "Subtask",
            "info": f"subtask_complete({_current_subgoal_text(ctx)})",
        },
        satisfied,
    )


def _terminal_info(cond: dict[str, Any]) -> str:
    if cond.get("info"):
        return str(cond["info"])
    predicate = str(cond.get("predicate", "condition")).lower()
    obj = _condition_object(cond) or "object"
    target = cond.get("target")
    if target:
        return f"{predicate}({obj}, {target})"
    return f"{predicate}({obj})"


def _condition_objects(cond: dict[str, Any]) -> list[str]:
    obj = _condition_object(cond)
    if obj:
        return [obj]
    objects = cond.get("objects")
    if isinstance(objects, list):
        values = [str(item) for item in objects if str(item).strip()]
        if values:
            return values
    if isinstance(objects, str) and objects.strip():
        return [objects]
    targets = cond.get("target_objects")
    if isinstance(targets, list):
        objects = [str(obj) for obj in targets if str(obj).strip()]
        if objects:
            return objects
    return []


def _condition_reference(cond: dict[str, Any]) -> str:
    for key in ("reference", "container", "surface", "reference_object", "target"):
        value = cond.get(key)
        if isinstance(value, str) and value.strip():
            return value

    info = str(cond.get("info", ""))
    for key in ("reference", "container", "surface", "reference_object"):
        match = re.search(
            rf"\b{key}\s*=\s*['\"]?(?P<reference>[a-zA-Z0-9_]+)",
            info,
        )
        if match:
            return match.group("reference")

    match = re.search(r"\([^,]+,\s*['\"]?(?P<reference>[a-zA-Z0-9_]+)", info)
    return match.group("reference") if match else ""


def _is_currently_grabbed(gt: dict[str, Any], obj: str) -> bool:
    robot = gt.get("robot", {})
    if isinstance(robot, dict) and robot.get("grasped_object") == obj:
        return True
    for cond in _conditions(gt):
        if (
            _condition_object(cond) == obj
            and _is_grab_condition(cond)
            and cond.get("satisfied")
        ):
            return True
    return False


def _first_target(entry: dict[str, Any]) -> str | None:
    for key in ("target_object", "target", "object"):
        val = entry.get(key)
        if val:
            return str(val)
    targets = entry.get("target_objects")
    if isinstance(targets, list) and targets:
        return str(targets[0])
    return None


def _entry_failure_type(entry: dict[str, Any]) -> str:
    return str(
        entry.get("failure_type")
        or entry.get("gt_failure_type")
        or "unknown"
    )


def _entry_texts(entry: dict[str, Any]) -> list[str]:
    texts: list[str] = []
    for key in (
        "instruction",
        "rewritten_instruction",
        "original_instruction",
        "subgoal",
        "current_subgoal",
        "target_object",
        "target",
        "vlm_raw",
        "reason",
        "recovery_instruction",
    ):
        val = entry.get(key)
        if isinstance(val, str) and val.strip():
            texts.append(val)
    subgoals = entry.get("subgoals")
    if isinstance(subgoals, list):
        texts.extend(str(sg) for sg in subgoals if str(sg).strip())
    return texts


def _entry_is_complete(entry: dict[str, Any]) -> bool:
    status = str(entry.get("status", "")).lower()
    action = str(entry.get("action", "")).lower()
    if status in {"complete", "completed", "done", "success"}:
        return True
    if action == "next":
        return True
    done = entry.get("done")
    return bool(done) if isinstance(done, bool) else False


def _entry_claims_task_complete(
    ctx: "MetricContext",
    entry: dict[str, Any],
    *,
    entry_idx: int,
    task_done: bool,
) -> bool:
    if not _entry_is_complete(entry):
        return False
    if task_done:
        return True

    subgoals = getattr(ctx.state, "subgoals", []) or []
    if isinstance(subgoals, list) and len(subgoals) > 1 and entry_idx < len(subgoals) - 1:
        text = " ".join(
            str(entry.get(key, ""))
            for key in ("status", "action", "reason", "vlm_raw")
        ).lower()
        task_phrases = (
            "task is complete",
            "task complete",
            "overall task",
            "entire task",
            "all subgoals",
            "all required subgoals",
            "all subtasks",
            "all required subtasks",
            "all items",
            "completing all",
            "sorting is complete",
        )
        return any(
            _has_non_negated_task_complete_phrase(text, phrase)
            for phrase in task_phrases
        )
    return True


def _has_non_negated_task_complete_phrase(text: str, phrase: str) -> bool:
    start = text.find(phrase)
    while start != -1:
        end = start + len(phrase)
        prefix = text[max(0, start - 48):start]
        suffix = text[end:end + 72]
        negated_prefix = re.search(
            r"\b(?:not|no|never|without|n't)\b(?:\W+\w+){0,4}\W*$",
            prefix,
        )
        negated_suffix = re.search(
            r"^\W*(?:is|are|was|were|be|being|remain|remains|"
            r"has|have|still|yet|currently)?\W*"
            r"(?:not|incomplete|unfinished)\b",
            suffix,
        )
        if not negated_prefix and not negated_suffix:
            return True
        start = text.find(phrase, start + 1)
    return False


def _entry_is_failure(entry: dict[str, Any]) -> bool:
    status = str(entry.get("status", "")).lower()
    action = str(entry.get("action", "")).lower()
    return status == "failure" or action in {
        "replan",
        "grasp_tool",
        "grasp_tool_vlm",
    }


def _entry_parse_failed(entry: dict[str, Any]) -> bool:
    reason = str(entry.get("reason", "")).strip().lower()
    fallback = str(entry.get("fallback", "")).strip().lower()
    error = str(entry.get("error", "")).strip().lower()
    return (
        reason == "parse_failure"
        or fallback in {"parse_failed", "parse_failure"}
        or error.startswith("parse_failure")
    )


def _object_position(gt: dict[str, Any] | None, obj: str) -> list[float] | None:
    if not gt:
        return None
    objects = gt.get("objects", {})
    if not isinstance(objects, dict):
        return None
    data = objects.get(obj)
    if not isinstance(data, dict):
        return None
    pos = data.get("pos")
    if pos is None:
        return None
    try:
        return [float(pos[0]), float(pos[1]), float(pos[2])]
    except (TypeError, ValueError, IndexError):
        return None


def _blocking_failures_from_gt(
    gt: dict[str, Any],
    target_objects: list[str],
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    seen: set[str] = set()
    for obj in target_objects:
        if obj in seen or not _is_real_scene_object(gt, obj):
            continue
        seen.add(obj)
        pos = _object_position(gt, obj)
        if pos is None:
            continue
        # Conservative table-top check: a required object with a strongly
        # negative z has fallen below the table plane and is no longer a
        # plausible stack/place target.
        if pos[2] < -0.05:
            failures.append({
                "failure_type": "object_out_of_scene",
                "object": obj,
                "pos": pos,
            })
    return failures


def _gt_summary(ctx: "MetricContext") -> dict[str, Any]:
    return _gt_summary_for(ctx)


def _object_states_snapshot(gt: dict[str, Any]) -> dict[str, Any]:
    objects = gt.get("objects", {})
    if not isinstance(objects, dict):
        return {}
    scene_names = _scene_objects(gt)
    names = scene_names or [str(name) for name in objects.keys()]
    snapshot: dict[str, Any] = {}
    for name in names:
        data = objects.get(name)
        if not isinstance(data, dict):
            continue
        snapshot[name] = {
            "pos": data.get("pos"),
            "quat": data.get("quat"),
            "vel": data.get("vel"),
            "displacement": data.get("displacement"),
            "z_lift": data.get("z_lift"),
            "max_z_lift": data.get("max_z_lift"),
            "lifted": data.get("lifted"),
        }
    return snapshot


def _gt_summary_for(
    ctx: "MetricContext",
    *,
    subgoal_idx: int | None = None,
    target_objects: list[str] | None = None,
) -> dict[str, Any]:
    gt = ctx.gt_state
    sub = _subtask(gt)
    robot = gt.get("robot", {})
    if not isinstance(robot, dict):
        robot = {}
    key = _subtask_condition_key(gt, ctx, subgoal_idx=subgoal_idx)
    subtask_checks = sub.get("all_subtask_checks", {})
    matched_predicate = (
        subtask_checks.get(key)
        if isinstance(subtask_checks, dict)
        else None
    )
    conditions = []
    for cond in _conditions(gt):
        conditions.append({
            "object": _condition_object(cond),
            "condition_idx": cond.get("condition_idx"),
            "predicate": cond.get("predicate"),
            "info": cond.get("info", ""),
            "satisfied": bool(cond.get("satisfied")),
            "target_objects": cond.get("target_objects", []),
        })

    summary = {
        "gt_step": gt.get("step"),
        "score": _score(gt),
        "completed": sub.get("completed", 0),
        "total": sub.get("total", 0),
        "current_index": sub.get("current_index"),
        "current_name": sub.get("current_name"),
        "current_subtask_key": key,
        "current_subtask_satisfied": _current_gt_satisfied(
            gt, ctx,
            target_objects=target_objects,
            subgoal_idx=subgoal_idx,
        ),
        "target_objects": target_objects or _target_objects(ctx),
        "object_completed": sub.get("object_completed", {}),
        "all_subtask_conditions": sub.get("all_subtask_conditions", {}),
        "matched_predicate": matched_predicate,
        "conditions": conditions,
        "object_states": _object_states_snapshot(gt),
        "robot": {
            "grasped_object": robot.get("grasped_object"),
            "objects_in_contact": robot.get("objects_in_contact", []),
            "gripper_width": robot.get("gripper_width"),
        },
    }
    if _structured_task_checks(gt):
        summary["task"] = _task_state_summary(gt)
    return summary


def _task_check_list(gt: dict[str, Any], key: str) -> list[dict[str, Any]]:
    checks = _task(gt).get(key, [])
    return [check for check in checks if isinstance(check, dict)]


def _task_success_checks(gt: dict[str, Any]) -> list[dict[str, Any]]:
    return _task_check_list(gt, "success_checks")


def _task_invariant_checks(gt: dict[str, Any]) -> list[dict[str, Any]]:
    return _task_check_list(gt, "invariant_checks")


def _task_check_satisfied(check: dict[str, Any]) -> bool:
    return bool(check.get("satisfied"))


def _task_check_name(check: dict[str, Any]) -> str:
    name = str(check.get("name", "")).strip()
    if name:
        return name
    predicate = str(check.get("predicate", "check")).strip() or "check"
    objects = "_".join(_task_check_objects(check)) or "object"
    return f"{predicate}:{objects}"


def _task_check_objects(check: dict[str, Any]) -> list[str]:
    objects = check.get("objects", check.get("object"))
    if isinstance(objects, list):
        return [str(obj) for obj in objects if str(obj).strip()]
    if isinstance(objects, str) and objects.strip():
        return [objects]
    return []


def _task_check_reference(check: dict[str, Any]) -> str:
    for key in ("reference", "container", "surface", "reference_object"):
        value = check.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _task_check_info(check: dict[str, Any]) -> str:
    predicate = str(check.get("predicate", "check")).strip() or "check"
    reference = _task_check_reference(check)
    objects = _task_check_objects(check)
    subject = ", ".join(objects) if objects else "object"
    if reference:
        return f"{predicate}({subject}, {reference})"
    return f"{predicate}({subject})"


def _failed_task_checks(checks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [check for check in checks if not _task_check_satisfied(check)]


def _task_success_satisfied(gt: dict[str, Any]) -> bool | None:
    task = _task(gt)
    raw = task.get("success_satisfied")
    if isinstance(raw, bool):
        return raw
    if raw is not None:
        return bool(raw)
    success_checks = _task_success_checks(gt)
    invariant_checks = _task_invariant_checks(gt)
    if success_checks or invariant_checks:
        return all(
            _task_check_satisfied(check)
            for check in [*success_checks, *invariant_checks]
        )
    return None


def _task_state_summary(gt: dict[str, Any]) -> dict[str, Any]:
    success_checks = _task_success_checks(gt)
    invariant_checks = _task_invariant_checks(gt)
    failed_success = _failed_task_checks(success_checks)
    failed_invariants = _failed_task_checks(invariant_checks)
    return {
        "task_success_satisfied": _task_success_satisfied(gt),
        "success_checks": success_checks,
        "invariant_checks": invariant_checks,
        "failed_success_checks": [
            _task_check_name(check) for check in failed_success
        ],
        "failed_invariants": [
            _task_check_name(check) for check in failed_invariants
        ],
    }


def _stack_relation_from_text(
    text: str,
    scene_objects: list[str],
) -> dict[str, str] | None:
    if not text:
        return None
    lowered = text.lower()
    if "stack" not in lowered and "on top" not in lowered and " on " not in lowered:
        return None
    objects = _objects_mentioned_in_text(text, scene_objects)
    if len(objects) < 2:
        return None
    top, bottom = objects[0], objects[1]
    if not top.endswith("_block") or not bottom.endswith("_block"):
        return None
    return {"bottom": bottom, "top": top}


def _strip_target_phrase(text: str) -> str:
    cleaned = re.sub(r"[.;,].*$", "", text.strip().lower())
    cleaned = re.sub(r"^(?:the|a|an)\s+", "", cleaned)
    return cleaned.strip()


def _strip_container_phrase(text: str) -> str:
    cleaned = _strip_target_phrase(text)
    cleaned = re.split(
        r"\s+(?:upright|beside|next\s+to|near|alongside|with)\b",
        cleaned,
        maxsplit=1,
    )[0]
    return cleaned.strip()


def _container_relation_from_text(
    text: str,
    scene_objects: list[str],
    gt: dict[str, Any] | None = None,
) -> dict[str, str] | None:
    if not text:
        return None
    lowered = text.lower()
    patterns = [
        (
            r"(?:pick\s+up|pick|grab|grasp|move)\s+"
            r"(?P<object>.+?)\s+"
            r"(?:from|out\s+of|out\s+from)\s+.+?\s+and\s+"
            r"(?:place|put|move|deposit|drop)\s+(?:it|them)?\s*"
            r"(?:in|into|on|onto|to)\s+(?P<container>.+)"
        ),
        (
            r"(?:pick\s+up|pick|grab|grasp|move)\s+"
            r"(?P<object>.+?)\s+and\s+"
            r"(?:place|put|move|deposit|drop)\s+(?:it|them)?\s*"
            r"(?:in|into|on|onto|to)\s+(?P<container>.+)"
        ),
        (
            r"(?:place|put|move|deposit|drop|transport)\s+"
            r"(?P<object>.+?)\s+"
            r"(?:in|into|on|onto|to)\s+(?P<container>.+)"
        ),
        (
            r"(?:transport|move|carry)\s+"
            r"(?P<object>.+?)\s+(?:over\s+)?"
            r"(?:in|into|on|onto|to)\s+(?P<container>.+?)"
            r"(?:\s+and\s+(?:release|drop|place)|$)"
        ),
    ]
    match = None
    for pattern in patterns:
        match = re.search(pattern, lowered)
        if match:
            break
    if not match:
        return None

    object_phrase = _strip_target_phrase(match.group("object"))
    container_phrase = _strip_container_phrase(match.group("container"))
    obj = _resolve_observed_scene_target(object_phrase, scene_objects)
    container = _resolve_observed_scene_target(container_phrase, scene_objects)
    if not container:
        container = _resolve_spatial_scene_target(
            container_phrase,
            scene_objects,
            gt,
        )
    if not obj or not container or obj == container:
        return None
    return {"object": obj, "container": container}


def _outside_container_relation_from_text(
    text: str,
    scene_objects: list[str],
) -> dict[str, str] | None:
    if not text:
        return None
    lowered = text.lower()
    patterns = [
        (
            r"(?:pick\s+up|pick|grab|grasp|move|take|remove)\s+"
            r"(?P<object>.+?)\s+"
            r"(?:from|out\s+of|out\s+from)\s+(?P<container>.+?)"
            r"(?:\s+and\s+|$)"
        ),
        (
            r"(?:take|remove)\s+"
            r"(?P<object>.+?)\s+"
            r"(?:from|out\s+of|out\s+from)\s+(?P<container>.+?)"
            r"(?:\s+and\s+|$)"
        ),
    ]
    match = None
    for pattern in patterns:
        match = re.search(pattern, lowered)
        if match:
            break
    if not match:
        return None

    object_phrase = _strip_target_phrase(match.group("object"))
    container_phrase = _strip_target_phrase(match.group("container"))
    obj = _resolve_observed_scene_target(object_phrase, scene_objects)
    container = _resolve_observed_scene_target(container_phrase, scene_objects)
    if not obj or not container or obj == container:
        return None
    return {"object": obj, "container": container}


def _expected_plan_objects(
    expected: list[dict[str, Any]],
    *,
    relation_type: str,
) -> set[str]:
    if relation_type in {"container", "outside_container"}:
        return {
            str(row.get("object"))
            for row in expected
            if row.get("object")
        }
    return set()


def _mechanical_plan_objects(
    text: str,
    scene_objects: list[str],
) -> list[str]:
    lowered = text.lower()
    patterns = [
        (
            r"\b(?:grasp|grab|pick(?:\s+up)?)\s+(?P<object>.+?)"
            r"(?:\s+(?:from|out\s+of|and|then|to|in|into|on|onto)\b|$)"
        ),
        (
            r"\b(?:place|put|drop|release)\s+(?P<object>.+?)\s+"
            r"(?:on|onto|to|in|into)\b"
        ),
    ]
    resolved: list[str] = []
    for pattern in patterns:
        for match in re.finditer(pattern, lowered):
            phrase = _strip_target_phrase(match.group("object"))
            if phrase in {"it", "them", "object", "item", "thing"}:
                continue
            obj = _resolve_observed_scene_target(phrase, scene_objects)
            if obj and obj not in resolved:
                resolved.append(obj)
    if resolved:
        return resolved
    if re.search(r"\b(?:grasp|grab|pick(?:\s+up)?)\b", lowered):
        return _objects_mentioned_in_text(text, scene_objects)
    return resolved


def _is_ignorable_plan_prelude(
    text: str,
    *,
    relation_type: str,
    expected_objects: set[str],
    scene_objects: list[str],
) -> bool:
    """Return True for VLM subgoals that are only grasp/release mechanics.

    Plans for container/recovery tasks often split a semantic move into
    "grasp X" and "place/release it".  Those rows are not independently
    checkable against a placement predicate, so the plan QA should not become
    unchecked as long as the same plan also contains parseable GT relations.
    """
    if relation_type not in {"container", "outside_container"}:
        return False
    lowered = text.lower()
    mentioned_objects = _mechanical_plan_objects(text, scene_objects)
    if mentioned_objects:
        if not expected_objects.issuperset(mentioned_objects):
            return False
    elif not re.search(r"\b(?:it|them|object|item|thing)\b", lowered):
        return False

    if re.search(r"\b(?:grasp|grab|pick(?:\s+up)?)\b", lowered):
        if not re.search(r"\b(?:place|put|move|deposit|drop|transport)\b", lowered):
            return True
    if relation_type == "outside_container":
        return bool(re.search(r"\b(?:place|put|drop|release)\b.+\btable\b", lowered))
    return False


def _expected_stack_relations(gt: dict[str, Any]) -> list[dict[str, Any]]:
    sub = _subtask(gt)
    checks = sub.get("all_subtask_checks", {})
    if not isinstance(checks, dict):
        return []
    rows: list[tuple[int, dict[str, Any]]] = []
    for key, check in checks.items():
        if not isinstance(check, dict):
            continue
        try:
            idx = int(check.get("subtask_idx", str(key).split("_")[-1]))
        except (TypeError, ValueError):
            continue
        conditions = check.get("conditions", [])
        if not isinstance(conditions, list):
            continue
        for cond in conditions:
            if not isinstance(cond, dict):
                continue
            predicate = str(cond.get("predicate", "")).lower()
            info = str(cond.get("info", "")).lower()
            targets = cond.get("target_objects", [])
            if (
                "stack" not in predicate
                and "stacked(" not in info
            ):
                continue
            if not isinstance(targets, list) or len(targets) < 2:
                continue
            bottom = str(targets[-2])
            top = str(targets[-1])
            if not bottom.endswith("_block") or not top.endswith("_block"):
                continue
            rows.append((idx, {
                "subtask_idx": idx,
                "bottom": bottom,
                "top": top,
                "predicate": cond.get("info") or cond.get("predicate"),
            }))
            break
    rows.sort(key=lambda item: item[0])
    return [row for _idx, row in rows]


def _incomplete_objects(gt: dict[str, Any]) -> set[str]:
    completed = _subtask(gt).get("object_completed", {})
    if not isinstance(completed, dict):
        return set()
    return {
        str(obj) for obj, done in completed.items()
        if not done
    }


def _expected_subtask_condition_relations(
    gt: dict[str, Any],
    *,
    predicate_name: str,
) -> list[dict[str, Any]]:
    sub = _subtask(gt)
    checks = sub.get("all_subtask_checks", {})
    if not isinstance(checks, dict):
        return []

    incomplete_objects = _incomplete_objects(gt)
    rows: list[tuple[int, int, dict[str, Any]]] = []
    for key, check in checks.items():
        if not isinstance(check, dict):
            continue
        try:
            subtask_idx = int(check.get("subtask_idx", str(key).split("_")[-1]))
        except (TypeError, ValueError):
            continue
        conditions = check.get("conditions", [])
        if not isinstance(conditions, list):
            continue
        for condition_idx, cond in enumerate(conditions):
            if not isinstance(cond, dict):
                continue
            predicate = str(cond.get("predicate", "")).lower()
            if predicate != predicate_name:
                continue
            reference = _condition_reference(cond)
            if not _is_real_scene_object(gt, reference):
                continue
            for obj in _condition_objects(cond):
                if incomplete_objects and obj not in incomplete_objects:
                    continue
                if not _is_real_scene_object(gt, obj):
                    continue
                rows.append((
                    subtask_idx,
                    condition_idx,
                    {
                        "object": obj,
                        "container": reference,
                        "predicate": _terminal_info(cond),
                    },
                ))
    rows.sort(key=lambda item: (item[0], item[1]))
    return [row for _subtask_idx, _condition_idx, row in rows]


def _expected_container_relations(gt: dict[str, Any]) -> list[dict[str, Any]]:
    incomplete_objects = _incomplete_objects(gt)

    rows: list[dict[str, Any]] = []
    for check in _task_success_checks(gt):
        predicate = str(check.get("predicate", "")).lower()
        if predicate != "object_in_container":
            continue
        reference = _task_check_reference(check)
        if not _is_real_scene_object(gt, reference):
            continue
        for obj in _task_check_objects(check):
            if incomplete_objects and obj not in incomplete_objects:
                continue
            if not _is_real_scene_object(gt, obj):
                continue
            rows.append({
                "object": obj,
                "container": reference,
                "predicate": _task_check_info(check),
            })
    if rows:
        return rows
    return _expected_subtask_condition_relations(
        gt,
        predicate_name="object_in_container",
    )


def _expected_outside_container_relations(
    gt: dict[str, Any],
) -> list[dict[str, Any]]:
    incomplete_objects = _incomplete_objects(gt)

    rows: list[dict[str, Any]] = []
    for check in _task_success_checks(gt):
        predicate = str(check.get("predicate", "")).lower()
        if predicate != "object_outside_of":
            continue
        reference = _task_check_reference(check)
        if not _is_real_scene_object(gt, reference):
            continue
        for obj in _task_check_objects(check):
            if incomplete_objects and obj not in incomplete_objects:
                continue
            if not _is_real_scene_object(gt, obj):
                continue
            rows.append({
                "object": obj,
                "container": reference,
                "predicate": _task_check_info(check),
            })
    if rows:
        return rows
    return _expected_subtask_condition_relations(
        gt,
        predicate_name="object_outside_of",
    )


def _plan_subgoals(entry: dict[str, Any]) -> list[str]:
    for key in ("subgoals", "new_subgoals"):
        value = entry.get(key)
        if isinstance(value, list):
            return [str(item) for item in value if str(item).strip()]
    instruction = entry.get("instruction")
    if isinstance(instruction, str) and instruction.strip():
        return [instruction]
    return []


@dataclass
class MetricContext:
    obs: dict[str, Any]
    gt_state: dict[str, Any]
    state: Any
    control_entries: list[dict[str, Any]]
    episode_id: int
    step_count: int
    episode_step: int
    subgoal_idx: int


class GTMetricDetector:
    """Base class for passive ground-truth metric detectors."""

    metric_type = "base"

    def reset(self) -> None:
        pass

    def detect(self, ctx: MetricContext) -> list[GTMetricEvent]:
        raise NotImplementedError

    @staticmethod
    def _finish(
        ctx: MetricContext,
        event: GTMetricEvent,
        *,
        subgoal_idx: int | None = None,
    ) -> GTMetricEvent:
        if event.step_count == 0:
            event.step_count = ctx.step_count
        if event.episode_step is None:
            event.episode_step = ctx.episode_step
        event.subgoal_idx = ctx.subgoal_idx if subgoal_idx is None else subgoal_idx
        return event

    @classmethod
    def _finish_decision(
        cls,
        ctx: MetricContext,
        event: GTMetricEvent,
        entry: dict[str, Any],
        *,
        subgoal_idx: int,
    ) -> GTMetricEvent:
        source_type = str(entry.get("type", ""))
        event.decision_id = event.decision_id or _decision_id(
            ctx, entry, subgoal_idx=subgoal_idx,
        )
        event.decision_kind = event.decision_kind or _decision_kind(source_type)
        return cls._finish(ctx, event, subgoal_idx=subgoal_idx)


class VLMPerceptionMismatchDetector(GTMetricDetector):
    """Detect color/object phrase mismatches between VLM logs and GT."""

    metric_type = "perception"
    _SOURCE_TYPES = {
        "decompose",
        "next_goal_episode_start",
        "next_goal_check",
        "vlm_check",
        "vlm_detect",
        "vlm_detection",
        "subgoal_check",
        "recycle_applied",
        "recycle_rejected",
        "scene_edit_detection",
        "scene_edit_vlm_detection",
    }

    def __init__(self) -> None:
        self._seen: set[tuple[int, int, str, str, str]] = set()

    def reset(self) -> None:
        self._seen.clear()

    def detect(self, ctx: MetricContext) -> list[GTMetricEvent]:
        if not ctx.gt_state:
            return []
        events: list[GTMetricEvent] = []
        default_targets = _target_objects(ctx)

        for entry in ctx.control_entries:
            source_type = str(entry.get("type", ""))
            if source_type not in self._SOURCE_TYPES:
                continue
            for text in _entry_texts(entry):
                mentioned_targets = _target_objects_from_gt(
                    ctx.gt_state,
                    text_hint=text,
                )
                if default_targets:
                    targets = list(dict.fromkeys(
                        [*default_targets, *mentioned_targets],
                    ))
                else:
                    targets = mentioned_targets
                if not targets:
                    continue
                mismatch = self._find_mismatch(
                    text,
                    targets,
                    scene_objects=_scene_objects(ctx.gt_state),
                )
                if mismatch is None:
                    continue
                obj, expected, observed = mismatch
                key = (
                    ctx.episode_id, ctx.subgoal_idx, obj,
                    observed, source_type,
                )
                if key in self._seen:
                    continue
                self._seen.add(key)
                entry_idx = _entry_subgoal_idx(ctx, entry)
                entry_text = _entry_subgoal_text(ctx, entry)
                evidence = _decision_summary(
                    ctx, entry,
                    subgoal_idx=entry_idx,
                    subgoal_text=entry_text,
                )
                evidence.update({
                    "source_text": text[:500],
                })
                events.append(self._finish_decision(ctx, GTMetricEvent(
                    metric="vlm_perception_mismatch",
                    category="perception",
                    subject_object=obj,
                    expected=expected,
                    observed=observed,
                    evidence=evidence,
                    causal_role="perception_error",
                ), entry, subgoal_idx=entry_idx))
        return events

    @staticmethod
    def _find_mismatch(
        text: str,
        target_objects: list[str],
        *,
        scene_objects: list[str] | None = None,
    ) -> tuple[str, str, str] | None:
        words = _tokens(text)
        if not words:
            return None
        real_scene_objects = {
            _norm_name(obj) for obj in (scene_objects or [])
        }
        for obj in target_objects:
            noun = _core_noun_from_name(obj)
            colors = _object_colors(obj)
            if not noun or not colors:
                continue
            for idx, word in enumerate(words):
                if word not in (noun, f"{noun}s"):
                    continue
                nearby = words[max(0, idx - 3):idx]
                observed_colors = [w for w in nearby if w in COLOR_WORDS]
                for color in reversed(observed_colors):
                    if color not in colors:
                        observed_name = _norm_name(f"{color}_{noun}")
                        if observed_name in real_scene_objects:
                            continue
                        return (
                            obj,
                            _display_name(obj),
                            f"{color} {noun}",
                        )
        return None


class VLMCompletionMismatchDetector(GTMetricDetector):
    """Detect VLM completion claims that contradict GT subtask checks."""

    metric_type = "perception"

    def __init__(self) -> None:
        self._seen: set[tuple[int, int, int, str]] = set()

    def reset(self) -> None:
        self._seen.clear()

    def detect(self, ctx: MetricContext) -> list[GTMetricEvent]:
        if not ctx.gt_state:
            return []
        events: list[GTMetricEvent] = []

        for entry in ctx.control_entries:
            source_type = str(entry.get("type", ""))
            if source_type not in _VLM_DECISION_TYPES:
                continue
            if not _entry_is_complete(entry):
                continue
            entry_idx = _entry_subgoal_idx(ctx, entry)
            entry_text = _entry_subgoal_text(ctx, entry)
            targets = _target_objects_from_gt(
                ctx.gt_state,
                text_hint=entry_text,
            )
            gt_done = _current_gt_satisfied(
                ctx.gt_state,
                ctx,
                target_objects=targets,
                subgoal_idx=entry_idx,
            )
            if gt_done is True:
                continue
            key = (
                ctx.episode_id,
                ctx.step_count,
                entry_idx,
                str(entry.get("status", "")),
                str(entry.get("action", "")),
                str(entry.get("reason", ""))[:120],
            )
            if key in self._seen:
                continue
            self._seen.add(key)
            evidence = _gt_summary_for(
                ctx,
                subgoal_idx=entry_idx,
                target_objects=targets,
            )
            evidence.update(_decision_summary(
                ctx, entry,
                subgoal_idx=entry_idx,
                subgoal_text=entry_text,
            ))
            evidence.update({
                "source_log_type": source_type,
                "source_status": entry.get("status"),
                "source_action": entry.get("action"),
                "source_reason": str(entry.get("reason", ""))[:500],
                "source_subgoal": entry_text,
            })
            events.append(self._finish_decision(ctx, GTMetricEvent(
                metric="vlm_completion_mismatch",
                category="perception",
                subject_object=targets[0] if targets else None,
                expected="gt_subtask_satisfied",
                observed="vlm_complete_but_gt_incomplete",
                evidence=evidence,
                causal_role="perception_error",
            ), entry, subgoal_idx=entry_idx))
        return events


class VLMResponseFormatDetector(GTMetricDetector):
    """Detect VLM decisions that could not be parsed into the expected JSON."""

    metric_type = "format_qa"
    _SOURCE_TYPES = _VLM_DECISION_TYPES | _PLAN_DECISION_TYPES | {
        "gt_vlm_recovery",
    }

    def __init__(self) -> None:
        self._seen: set[tuple[int, int, int, str, str]] = set()

    def reset(self) -> None:
        self._seen.clear()

    def detect(self, ctx: MetricContext) -> list[GTMetricEvent]:
        events: list[GTMetricEvent] = []
        for entry in ctx.control_entries:
            source_type = str(entry.get("type", ""))
            if source_type not in self._SOURCE_TYPES:
                continue
            if not _entry_parse_failed(entry):
                continue

            entry_idx = _entry_subgoal_idx(ctx, entry)
            entry_text = _entry_subgoal_text(ctx, entry)
            key = (
                ctx.episode_id,
                ctx.step_count,
                entry_idx,
                source_type,
                str(entry.get("vlm_raw", ""))[:120],
            )
            if key in self._seen:
                continue
            self._seen.add(key)

            evidence = _decision_summary(
                ctx, entry,
                subgoal_idx=entry_idx,
                subgoal_text=entry_text,
            )
            evidence.update({
                "parse_failed": True,
                "reason": str(entry.get("reason", ""))[:1000],
                "fallback": entry.get("fallback"),
                "error": str(entry.get("error", ""))[:1000],
                "raw": str(entry.get("vlm_raw", ""))[:1200],
            })
            events.append(self._finish_decision(ctx, GTMetricEvent(
                metric="vlm_response_parse_failure",
                category="format_qa",
                subject_object=None,
                expected="valid_json_response",
                observed="parse_failure",
                evidence=evidence,
                causal_role="response_format_error",
                confidence=float(entry.get("confidence", 1.0) or 1.0),
            ), entry, subgoal_idx=entry_idx))
        return events


class VLMSceneQADetector(GTMetricDetector):
    """Record VLM visual-state judgments alongside matched GT predicates."""

    metric_type = "scene_qa"

    def __init__(self) -> None:
        self._seen: set[tuple[int, int, int, str, str, str]] = set()

    def reset(self) -> None:
        self._seen.clear()

    def detect(self, ctx: MetricContext) -> list[GTMetricEvent]:
        if not ctx.gt_state:
            return []

        events: list[GTMetricEvent] = []
        for entry in ctx.control_entries:
            source_type = str(entry.get("type", ""))
            if source_type not in _VLM_DECISION_TYPES:
                continue

            entry_idx = _entry_subgoal_idx(ctx, entry)
            entry_text = _entry_subgoal_text(ctx, entry)
            status = str(entry.get("status", "")).lower()
            action = str(entry.get("action", "")).lower()
            reason = str(entry.get("reason", ""))
            key = (
                ctx.episode_id,
                ctx.step_count,
                entry_idx,
                status,
                action,
                reason[:120],
            )
            if key in self._seen:
                continue
            self._seen.add(key)

            targets = _target_objects_from_gt(
                ctx.gt_state,
                text_hint=entry_text,
            )
            gt_done = _current_gt_satisfied(
                ctx.gt_state,
                ctx,
                target_objects=targets,
                subgoal_idx=entry_idx,
            )
            claims_complete = _entry_is_complete(entry)
            parse_failed = _entry_parse_failed(entry)
            if parse_failed:
                comparison = "unchecked_parse_failure"
            else:
                comparison = self._comparison_label(
                    claims_complete=claims_complete,
                    gt_done=gt_done,
                )
            evidence = _gt_summary_for(
                ctx,
                subgoal_idx=entry_idx,
                target_objects=targets,
            )
            evidence.update(_decision_summary(
                ctx, entry,
                subgoal_idx=entry_idx,
                subgoal_text=entry_text,
            ))
            evidence.update({
                "vlm": {
                    "source_log_type": source_type,
                    "status": entry.get("status"),
                    "action": entry.get("action"),
                    "reason": reason[:1000],
                    "raw": str(entry.get("vlm_raw", ""))[:1200],
                    "subgoal_idx": entry_idx,
                    "subgoal": entry_text,
                    "confidence": entry.get("confidence"),
                    "grasp_target": entry.get("grasp_target"),
                },
                "comparison": comparison,
                "parse_failed": parse_failed,
                "vlm_claims_complete": claims_complete,
                "gt_predicate_satisfied": gt_done,
            })

            is_failure = comparison in {
                "false_complete",
                "missed_complete",
            }
            metric = (
                "vlm_scene_qa_failure"
                if is_failure
                else "vlm_scene_qa"
            )
            events.append(self._finish_decision(ctx, GTMetricEvent(
                metric=metric,
                category="scene_qa",
                subject_object=targets[0] if targets else None,
                expected=(
                    "vlm_state_matches_gt_predicate"
                    if not is_failure
                    else "vlm_state_should_match_gt_predicate"
                ),
                observed=comparison,
                evidence=evidence,
                causal_role=(
                    "perception_error"
                    if is_failure
                    else (
                        "visual_state_unchecked"
                        if parse_failed
                        else "visual_state_audit"
                    )
                ),
                confidence=float(entry.get("confidence", 1.0) or 1.0),
            ), entry, subgoal_idx=entry_idx))
        return events

    @staticmethod
    def _comparison_label(
        *,
        claims_complete: bool,
        gt_done: bool | None,
    ) -> str:
        if gt_done is None:
            return "unchecked"
        if claims_complete and gt_done:
            return "confirmed_complete"
        if claims_complete and not gt_done:
            return "false_complete"
        if not claims_complete and gt_done:
            return "missed_complete"
        return "aligned_incomplete"


class VLMFailureStateQADetector(GTMetricDetector):
    """Check whether the VLM reacts to obvious GT-observable failure states."""

    metric_type = "failure_qa"

    def __init__(self) -> None:
        self._seen: set[tuple[int, int, int, str, str]] = set()

    def reset(self) -> None:
        self._seen.clear()

    def detect(self, ctx: MetricContext) -> list[GTMetricEvent]:
        if not ctx.gt_state:
            return []

        events: list[GTMetricEvent] = []
        for entry in ctx.control_entries:
            source_type = str(entry.get("type", ""))
            if source_type not in _VLM_DECISION_TYPES:
                continue

            entry_idx = _entry_subgoal_idx(ctx, entry)
            entry_text = _entry_subgoal_text(ctx, entry)
            targets = _failure_target_objects_from_gt(
                ctx.gt_state,
                text_hint=entry_text,
            )
            failures = _blocking_failures_from_gt(ctx.gt_state, targets)
            if not failures:
                continue

            observed_failure = _entry_is_failure(entry)
            observed = (
                "failure_detected"
                if observed_failure
                else "missed_blocking_failure"
            )
            key = (
                ctx.episode_id,
                ctx.step_count,
                entry_idx,
                source_type,
                observed,
            )
            if key in self._seen:
                continue
            self._seen.add(key)

            evidence = _gt_summary_for(
                ctx,
                subgoal_idx=entry_idx,
                target_objects=targets,
            )
            evidence.update(_decision_summary(
                ctx, entry,
                subgoal_idx=entry_idx,
                subgoal_text=entry_text,
            ))
            evidence.update({
                "blocking_failures": failures,
                "vlm_claims_failure": observed_failure,
                "parse_failed": _entry_parse_failed(entry),
            })
            is_failure = not observed_failure
            events.append(self._finish_decision(ctx, GTMetricEvent(
                metric=(
                    "vlm_failure_missed"
                    if is_failure
                    else "vlm_failure_qa"
                ),
                category="failure_qa",
                subject_object=failures[0].get("object"),
                expected="vlm_detects_blocking_gt_failure",
                observed=observed,
                evidence=evidence,
                causal_role=(
                    "failure_detection_error"
                    if is_failure
                    else "failure_detection_audit"
                ),
                confidence=float(entry.get("confidence", 1.0) or 1.0),
            ), entry, subgoal_idx=entry_idx))
        return events


class VLMTaskInvariantQADetector(GTMetricDetector):
    """Compare VLM task-complete claims with structured GT task checks."""

    metric_type = "task_qa"

    def __init__(self) -> None:
        self._seen: set[tuple[int, int, int, str, str, str]] = set()

    def reset(self) -> None:
        self._seen.clear()

    def detect(self, ctx: MetricContext) -> list[GTMetricEvent]:
        if not ctx.gt_state or not _structured_task_checks(ctx.gt_state):
            return []

        events: list[GTMetricEvent] = []
        for entry in ctx.control_entries:
            source_type = str(entry.get("type", ""))
            if source_type not in _VLM_DECISION_TYPES:
                continue
            if _entry_parse_failed(entry):
                continue

            entry_idx = _entry_subgoal_idx(ctx, entry)
            entry_text = _entry_subgoal_text(ctx, entry)
            claims_subgoal_complete = _entry_is_complete(entry)
            task_done = _task_success_satisfied(ctx.gt_state)
            if task_done is None:
                continue
            claims_task_complete = _entry_claims_task_complete(
                ctx,
                entry,
                entry_idx=entry_idx,
                task_done=task_done,
            )

            status = str(entry.get("status", "")).lower()
            action = str(entry.get("action", "")).lower()
            reason = str(entry.get("reason", ""))[:120]
            key = (
                ctx.episode_id,
                ctx.step_count,
                entry_idx,
                status,
                action,
                reason,
            )
            if key in self._seen:
                continue
            self._seen.add(key)

            task_summary = _task_state_summary(ctx.gt_state)
            evidence = _gt_summary_for(ctx, subgoal_idx=entry_idx)
            evidence.update(_decision_summary(
                ctx, entry,
                subgoal_idx=entry_idx,
                subgoal_text=entry_text,
            ))
            evidence.update(task_summary)
            evidence.update({
                "vlm_claims_subgoal_complete": claims_subgoal_complete,
                "vlm_claims_task_complete": claims_task_complete,
                "parse_failed": False,
            })

            comparison = self._task_comparison(
                claims_complete=claims_task_complete,
                task_done=task_done,
            )
            task_failure = comparison in {
                "false_task_success",
                "missed_task_success",
            }
            subject = self._subject_for_task(ctx.gt_state)
            events.append(self._finish_decision(ctx, GTMetricEvent(
                metric=(
                    "vlm_task_success_qa_failure"
                    if task_failure else "vlm_task_success_qa"
                ),
                category="task_qa",
                subject_object=subject,
                expected="vlm_task_success_matches_gt_task_success",
                observed=comparison,
                evidence=evidence,
                causal_role=(
                    "task_success_error"
                    if task_failure else "task_success_audit"
                ),
                confidence=float(entry.get("confidence", 1.0) or 1.0),
            ), entry, subgoal_idx=entry_idx))

            events.extend(self._invariant_events(
                ctx,
                entry,
                entry_idx=entry_idx,
                entry_text=entry_text,
                claims_complete=claims_task_complete,
                base_evidence=evidence,
            ))
        return events

    @staticmethod
    def _task_comparison(
        *,
        claims_complete: bool,
        task_done: bool,
    ) -> str:
        if claims_complete and task_done:
            return "confirmed_task_success"
        if claims_complete and not task_done:
            return "false_task_success"
        if not claims_complete and task_done:
            return "missed_task_success"
        return "aligned_task_incomplete"

    @staticmethod
    def _subject_for_task(gt: dict[str, Any]) -> str | None:
        for check in [
            *_failed_task_checks(_task_success_checks(gt)),
            *_failed_task_checks(_task_invariant_checks(gt)),
            *_task_success_checks(gt),
            *_task_invariant_checks(gt),
        ]:
            objects = _task_check_objects(check)
            if objects:
                return objects[0]
        return None

    def _invariant_events(
        self,
        ctx: MetricContext,
        entry: dict[str, Any],
        *,
        entry_idx: int,
        entry_text: str,
        claims_complete: bool,
        base_evidence: dict[str, Any],
    ) -> list[GTMetricEvent]:
        if not claims_complete:
            return []

        invariant_checks = _task_invariant_checks(ctx.gt_state)
        if not invariant_checks:
            return []

        failed = _failed_task_checks(invariant_checks)
        if not failed:
            return [self._finish_decision(ctx, GTMetricEvent(
                metric="vlm_invariant_qa",
                category="task_qa",
                subject_object=None,
                expected="gt_task_invariants_hold",
                observed="invariants_satisfied_when_claimed_complete",
                evidence=base_evidence,
                causal_role="task_invariant_audit",
                confidence=float(entry.get("confidence", 1.0) or 1.0),
            ), entry, subgoal_idx=entry_idx)]

        events: list[GTMetricEvent] = []
        for check in failed:
            evidence = dict(base_evidence)
            evidence["failed_invariant"] = check
            objects = _task_check_objects(check)
            events.append(self._finish_decision(ctx, GTMetricEvent(
                metric="vlm_invariant_qa_failure",
                category="task_qa",
                subject_object=objects[0] if objects else None,
                expected=_task_check_info(check),
                observed="missed_invariant_violation",
                evidence=evidence,
                causal_role="task_invariant_error",
                confidence=float(entry.get("confidence", 1.0) or 1.0),
            ), entry, subgoal_idx=entry_idx))
        return events


class VLMGraspTargetQADetector(GTMetricDetector):
    """Check VLM-selected grasp targets against current GT targets."""

    metric_type = "target_qa"

    def __init__(self) -> None:
        self._seen: set[tuple[int, int, int, str, str]] = set()

    def reset(self) -> None:
        self._seen.clear()

    def detect(self, ctx: MetricContext) -> list[GTMetricEvent]:
        if not ctx.gt_state:
            return []

        events: list[GTMetricEvent] = []
        for entry in ctx.control_entries:
            source_type = str(entry.get("type", ""))
            if source_type not in _TARGET_DECISION_TYPES:
                continue
            action = str(entry.get("action", "")).lower()
            if action not in {
                "grasp_tool",
                "grasp_tool_vlm",
                "grasp",
                "grab",
                "pick",
                "pick_up",
                "pick_and_stack",
            }:
                continue

            entry_idx = _entry_subgoal_idx(ctx, entry)
            entry_text = _entry_subgoal_text(ctx, entry)
            observed_target = _entry_grasp_target(entry)
            targets = _target_objects_from_gt(
                ctx.gt_state,
                text_hint=entry_text,
            )
            key = (
                ctx.episode_id,
                ctx.step_count,
                entry_idx,
                action,
                observed_target,
                str(entry.get("status", "")).lower(),
                str(entry.get("reason", ""))[:120],
            )
            if key in self._seen:
                continue
            self._seen.add(key)

            scene_objects = _scene_objects(ctx.gt_state)
            observed_scene_target: str | None = None
            if not observed_target:
                observed = "missing_grasp_target"
                is_failure = True
            elif not targets:
                observed = "unchecked"
                is_failure = False
            elif any(
                (
                    match := _target_matches_scene_object(
                        observed_target,
                        target,
                        scene_objects,
                    )
                )[0]
                for target in targets
            ):
                observed_scene_target = match[1]
                observed = "target_aligned"
                is_failure = False
            else:
                observed_scene_target = _resolve_observed_scene_target(
                    observed_target,
                    scene_objects,
                )
                observed = "target_mismatch"
                is_failure = True

            evidence = _gt_summary_for(
                ctx,
                subgoal_idx=entry_idx,
                target_objects=targets,
            )
            evidence.update(_decision_summary(
                ctx, entry,
                subgoal_idx=entry_idx,
                subgoal_text=entry_text,
            ))
            evidence.update({
                "expected_targets": targets,
                "observed_target": observed_target or None,
                "observed_scene_target": observed_scene_target,
            })

            metric = (
                "vlm_grasp_target_mismatch"
                if is_failure
                else "vlm_grasp_target_qa"
            )
            events.append(self._finish_decision(ctx, GTMetricEvent(
                metric=metric,
                category="target_qa",
                subject_object=observed_target or (targets[0] if targets else None),
                expected="one_of_current_gt_targets",
                observed=observed,
                evidence=evidence,
                causal_role=(
                    "target_selection_error"
                    if is_failure
                    else "target_selection_audit"
                ),
                confidence=float(entry.get("confidence", 1.0) or 1.0),
            ), entry, subgoal_idx=entry_idx))
        return events


class VLMPlanAlignmentDetector(GTMetricDetector):
    """Compare parseable Color-Stack VLM plans against GT stack order."""

    metric_type = "plan_qa"

    def __init__(self) -> None:
        self._seen: set[tuple[int, int, str, str]] = set()

    def reset(self) -> None:
        self._seen.clear()

    def detect(self, ctx: MetricContext) -> list[GTMetricEvent]:
        if not ctx.gt_state:
            return []

        scene_objects = _scene_objects(ctx.gt_state)
        events: list[GTMetricEvent] = []
        for entry in ctx.control_entries:
            source_type = str(entry.get("type", ""))
            if source_type not in _PLAN_DECISION_TYPES:
                continue
            subgoals = _plan_subgoals(entry)
            entry_idx = _entry_subgoal_idx(ctx, entry)
            entry_text = _entry_subgoal_text(ctx, entry)
            key = (
                ctx.episode_id,
                ctx.step_count,
                source_type,
                "|".join(subgoals),
            )
            if key in self._seen:
                continue
            self._seen.add(key)

            relation_type = "stack"
            expected_all = _expected_stack_relations(ctx.gt_state)
            parser = _stack_relation_from_text
            if not expected_all:
                relation_type = "container"
                expected_all = _expected_container_relations(ctx.gt_state)
                def parser(text, objects, gt=ctx.gt_state):
                    return _container_relation_from_text(text, objects, gt)
            if not expected_all:
                relation_type = "outside_container"
                expected_all = _expected_outside_container_relations(
                    ctx.gt_state,
                )
                parser = _outside_container_relation_from_text

            start_idx = (
                0 if (
                    source_type == "decompose"
                    or relation_type in {"container", "outside_container"}
                )
                else _current_gt_subtask_idx(ctx.gt_state, entry_idx)
            )
            expected = expected_all[start_idx:start_idx + len(subgoals)]
            parsed = [
                parser(subgoal, scene_objects)
                for subgoal in subgoals
            ]
            expected_objects = _expected_plan_objects(
                expected,
                relation_type=relation_type,
            )
            ignorable_unparsed = [
                row is None and _is_ignorable_plan_prelude(
                    subgoal,
                    relation_type=relation_type,
                    expected_objects=expected_objects,
                    scene_objects=scene_objects,
                )
                for subgoal, row in zip(subgoals, parsed, strict=True)
            ]

            observed, metric, causal_role = self._verdict(
                expected,
                parsed,
                relation_type=relation_type,
                ignorable_unparsed=ignorable_unparsed,
            )
            evidence = _gt_summary_for(ctx, subgoal_idx=entry_idx)
            evidence.update(_decision_summary(
                ctx, entry,
                subgoal_idx=entry_idx,
                subgoal_text=entry_text,
            ))
            evidence.update({
                "source_log_type": source_type,
                "ordered": entry.get("ordered"),
                "subgoals": subgoals,
                "expected_relation_type": relation_type,
                "expected_relations": expected,
                "parsed_relations": parsed,
                "expected_plan_objects": sorted(expected_objects),
                "unchecked_reason": self._unchecked_reason(
                    expected,
                    parsed,
                    relation_type=relation_type,
                    ignorable_unparsed=ignorable_unparsed,
                ),
                "ignored_plan_preludes": [
                    subgoal
                    for subgoal, ignored in zip(
                        subgoals,
                        ignorable_unparsed,
                        strict=True,
                    )
                    if ignored
                ],
            })
            events.append(self._finish_decision(ctx, GTMetricEvent(
                metric=metric,
                category="plan_qa",
                subject_object=(
                    expected[0].get("top") or expected[0].get("object")
                    if expected else None
                ),
                expected=f"gt_{relation_type}_plan_relations",
                observed=observed,
                evidence=evidence,
                causal_role=causal_role,
            ), entry, subgoal_idx=entry_idx))
        return events

    @staticmethod
    def _verdict(
        expected: list[dict[str, Any]],
        parsed: list[dict[str, str] | None],
        *,
        relation_type: str,
        ignorable_unparsed: list[bool] | None = None,
    ) -> tuple[str, str, str]:
        reason = VLMPlanAlignmentDetector._unchecked_reason(
            expected,
            parsed,
            relation_type=relation_type,
            ignorable_unparsed=ignorable_unparsed,
        )
        if reason:
            return "unchecked", "vlm_plan_qa", "plan_unchecked"
        expected_pairs = VLMPlanAlignmentDetector._relation_pairs(
            expected,
            relation_type=relation_type,
        )
        parsed_pairs = VLMPlanAlignmentDetector._relation_pairs(
            [row for row in parsed if row is not None],
            relation_type=relation_type,
        )
        if relation_type == "container":
            aligned = set(expected_pairs) == set(parsed_pairs)
        elif relation_type == "outside_container":
            aligned = set(expected_pairs) == set(parsed_pairs)
        else:
            aligned = expected_pairs == parsed_pairs
        if aligned:
            return "plan_aligned", "vlm_plan_qa", "plan_audit"
        return "plan_mismatch", "vlm_plan_mismatch", "plan_error"

    @staticmethod
    def _unchecked_reason(
        expected: list[dict[str, Any]],
        parsed: list[dict[str, str] | None],
        *,
        relation_type: str,
        ignorable_unparsed: list[bool] | None = None,
    ) -> str | None:
        if not expected:
            return f"no_parseable_gt_{relation_type}_relations"
        if not parsed:
            return "no_vlm_subgoals"
        ignorable = ignorable_unparsed or [False] * len(parsed)
        if any(row is None and not ignored for row, ignored in zip(
            parsed,
            ignorable,
            strict=True,
        )):
            return f"unparseable_vlm_{relation_type}_relation"
        if not any(row is not None for row in parsed):
            return f"no_parseable_vlm_{relation_type}_relation"
        return None

    @staticmethod
    def _relation_pairs(
        rows: list[dict[str, Any]],
        *,
        relation_type: str,
    ) -> list[tuple[str, str]]:
        if relation_type in {"container", "outside_container"}:
            return [
                (str(row["object"]), str(row["container"]))
                for row in rows
            ]
        return [
            (str(row["bottom"]), str(row["top"]))
            for row in rows
        ]


class GTStateCheckDetector(GTMetricDetector):
    """Log compact GT state/check snapshots at orchestrator decision points."""

    metric_type = "gt_state"
    _SOURCE_TYPES = _VLM_DECISION_TYPES | {
        "decompose",
        "gt_subgoal_complete",
        "gt_failure_detected",
        "recycle",
        "recycle_rejected",
        "replan_failed",
    }

    def __init__(self) -> None:
        self._seen: set[tuple[int, int, int, str, str, str]] = set()

    def reset(self) -> None:
        self._seen.clear()

    def detect(self, ctx: MetricContext) -> list[GTMetricEvent]:
        if not ctx.gt_state:
            return []
        events: list[GTMetricEvent] = []
        for entry in ctx.control_entries:
            source_type = str(entry.get("type", ""))
            if source_type not in self._SOURCE_TYPES:
                continue
            entry_idx = _entry_subgoal_idx(ctx, entry)
            entry_text = _entry_subgoal_text(ctx, entry)
            targets = _target_objects_from_gt(
                ctx.gt_state,
                text_hint=entry_text,
            )
            key = (
                ctx.episode_id,
                ctx.step_count,
                entry_idx,
                str(entry.get("status", "")),
                str(entry.get("action", "")),
                entry_text[:120],
            )
            if key in self._seen:
                continue
            self._seen.add(key)
            evidence = _gt_summary_for(
                ctx,
                subgoal_idx=entry_idx,
                target_objects=targets,
            )
            evidence.update(_decision_summary(
                ctx, entry,
                subgoal_idx=entry_idx,
                subgoal_text=entry_text,
            ))
            evidence.update({
                "source_log_type": source_type,
                "source_status": entry.get("status"),
                "source_action": entry.get("action"),
                "source_subgoal": entry_text,
            })
            events.append(self._finish_decision(ctx, GTMetricEvent(
                metric="gt_state_check",
                category="gt_state",
                subject_object=targets[0] if targets else None,
                expected="current_gt_state_recorded",
                observed=(
                    "subtask_satisfied"
                    if evidence["current_subtask_satisfied"]
                    else "subtask_not_satisfied"
                ),
                evidence=evidence,
                causal_role="gt_context",
            ), entry, subgoal_idx=entry_idx))
        return events


class GTTaskStateCheckDetector(GTMetricDetector):
    """Log structured task success/invariant checks at decision points."""

    metric_type = "gt_state"
    _SOURCE_TYPES = GTStateCheckDetector._SOURCE_TYPES

    def __init__(self) -> None:
        self._seen: set[tuple[int, int, int, str, str, str]] = set()

    def reset(self) -> None:
        self._seen.clear()

    def detect(self, ctx: MetricContext) -> list[GTMetricEvent]:
        if not ctx.gt_state or not _structured_task_checks(ctx.gt_state):
            return []

        events: list[GTMetricEvent] = []
        for entry in ctx.control_entries:
            source_type = str(entry.get("type", ""))
            if source_type not in self._SOURCE_TYPES:
                continue
            entry_idx = _entry_subgoal_idx(ctx, entry)
            entry_text = _entry_subgoal_text(ctx, entry)
            key = (
                ctx.episode_id,
                ctx.step_count,
                entry_idx,
                str(entry.get("status", "")),
                str(entry.get("action", "")),
                entry_text[:120],
            )
            if key in self._seen:
                continue
            self._seen.add(key)

            task_done = _task_success_satisfied(ctx.gt_state)
            task_summary = _task_state_summary(ctx.gt_state)
            evidence = _gt_summary_for(ctx, subgoal_idx=entry_idx)
            evidence.update(_decision_summary(
                ctx, entry,
                subgoal_idx=entry_idx,
                subgoal_text=entry_text,
            ))
            evidence.update(task_summary)
            evidence.update({
                "source_log_type": source_type,
                "source_status": entry.get("status"),
                "source_action": entry.get("action"),
                "source_subgoal": entry_text,
            })
            events.append(self._finish_decision(ctx, GTMetricEvent(
                metric="gt_task_state_check",
                category="task_state",
                subject_object=VLMTaskInvariantQADetector._subject_for_task(
                    ctx.gt_state,
                ),
                expected="structured_gt_task_state_recorded",
                observed=(
                    "task_complete"
                    if task_done else "task_incomplete"
                ),
                evidence=evidence,
                causal_role="gt_task_context",
            ), entry, subgoal_idx=entry_idx))
        return events


class PlacementOutcomeDetector(GTMetricDetector):
    """Detect releases that do not satisfy the target placement predicate."""

    metric_type = "placement"

    def __init__(self, confirm_steps: int = 5) -> None:
        self.confirm_steps = max(1, int(confirm_steps))
        self._states: dict[tuple[int, int, str], dict[str, Any]] = {}

    def reset(self) -> None:
        self._states.clear()

    def detect(self, ctx: MetricContext) -> list[GTMetricEvent]:
        if not ctx.gt_state:
            return []
        events: list[GTMetricEvent] = []
        targets = _target_objects(ctx)
        if targets:
            targets = [targets[0]]
        for obj in targets:
            terminal = _terminal_condition_for_context(ctx, obj)
            if terminal is None:
                continue
            cond, terminal_satisfied = terminal
            key = (ctx.episode_id, ctx.subgoal_idx, obj)
            state = self._states.setdefault(key, {
                "was_grabbed": False,
                "pending_release_step": None,
                "emitted_for_release": None,
            })
            grabbed = _is_currently_grabbed(ctx.gt_state, obj)

            if terminal_satisfied:
                state["pending_release_step"] = None
                state["emitted_for_release"] = None
                continue

            if grabbed:
                state["was_grabbed"] = True
                state["pending_release_step"] = None
                state["emitted_for_release"] = None
                continue

            if not state["was_grabbed"]:
                continue

            release_step = state["pending_release_step"]
            if release_step is None:
                state["pending_release_step"] = ctx.step_count
                continue

            if state["emitted_for_release"] == release_step:
                continue
            if ctx.step_count - release_step < self.confirm_steps - 1:
                continue

            state["emitted_for_release"] = release_step
            events.append(self._finish(ctx, GTMetricEvent(
                metric="placement_failed",
                category="placement",
                subject_object=obj,
                expected=_terminal_info(cond),
                observed="released_without_terminal_condition",
                evidence={
                    "release_step": release_step,
                    "confirm_steps": self.confirm_steps,
                    "terminal_condition": _terminal_info(cond),
                },
                causal_role="placement_outcome_failure",
            )))
        return events


@dataclass
class _FailureRecord:
    event: GTMetricEvent
    step_count: int
    subgoal_idx: int
    target_objects: set[str]
    failure_type: str


@dataclass
class _ToolAttempt:
    event: GTMetricEvent
    start_step: int
    subgoal_idx: int
    target_object: str
    start_score: float
    prior_failure: _FailureRecord | None
    outcome_event: GTMetricEvent | None = None
    outcome_step: int | None = None


class ToolCausalityDetector(GTMetricDetector):
    """Attribute grasp tool use to prior failures and later outcomes."""

    metric_type = "tool_causality"
    _FAILURE_TYPES = {
        "gt_failure_detected",
        "hitl_failure",
    }
    _OUTCOME_TYPES = {
        "grasp_tool_done",
        "grasp_tool_failed",
    }

    def __init__(self, attribution_window_steps: int = 10) -> None:
        self.attribution_window_steps = max(1, int(attribution_window_steps))
        self._failures: list[_FailureRecord] = []
        self._active_tools: list[_ToolAttempt] = []
        self._finished_tools: list[_ToolAttempt] = []

    def reset(self) -> None:
        self._failures.clear()
        self._active_tools.clear()
        self._finished_tools.clear()

    def detect(self, ctx: MetricContext) -> list[GTMetricEvent]:
        events: list[GTMetricEvent] = []
        for entry in ctx.control_entries:
            entry_type = str(entry.get("type", ""))
            if entry_type in self._FAILURE_TYPES:
                event = self._record_failure(ctx, entry)
                events.append(event)
                continue
            if self._is_tool_invocation(entry):
                events.extend(self._record_invocation(ctx, entry))
                continue
            if entry_type in self._OUTCOME_TYPES:
                event = self._record_outcome(ctx, entry)
                if event is not None:
                    events.append(event)
        return events

    def _record_failure(
        self,
        ctx: MetricContext,
        entry: dict[str, Any],
    ) -> GTMetricEvent:
        failure_type = _entry_failure_type(entry)
        targets = entry.get("target_objects")
        if isinstance(targets, list):
            target_objects = {str(t) for t in targets}
        else:
            target = _first_target(entry)
            target_objects = {target} if target else set()
        recent_tool = self._recent_tool_for_failure(
            ctx, failure_type, target_objects,
        )
        causal_role = "root_failure"
        parent_event_id = None
        evidence: dict[str, Any] = {
            "source_log_type": entry.get("type"),
            "failure_type": failure_type,
            "reason": entry.get("reason") or entry.get("failure_reason", ""),
            "target_objects": sorted(target_objects),
        }
        if recent_tool is not None:
            parent = recent_tool.outcome_event or recent_tool.event
            parent_event_id = parent.event_id
            if (
                recent_tool.prior_failure is not None
                and failure_type == recent_tool.prior_failure.failure_type
                and self._targets_overlap(
                    target_objects, recent_tool.prior_failure.target_objects,
                )
            ):
                causal_role = "persistent_failure_after_tool"
            else:
                causal_role = "failure_after_tool"
            evidence["preceding_tool_event_id"] = parent.event_id
            evidence["preceding_tool_target"] = recent_tool.target_object

        event = self._finish(ctx, GTMetricEvent(
            metric="failure_detected",
            category="failure",
            subject_object=next(iter(target_objects), None),
            observed=failure_type,
            evidence=evidence,
            parent_event_id=parent_event_id,
            causal_role=causal_role,
        ))
        record = _FailureRecord(
            event=event,
            step_count=ctx.step_count,
            subgoal_idx=ctx.subgoal_idx,
            target_objects=target_objects,
            failure_type=failure_type,
        )
        self._failures.append(record)
        self._trim_history(ctx.step_count)
        return event

    def _record_invocation(
        self,
        ctx: MetricContext,
        entry: dict[str, Any],
    ) -> list[GTMetricEvent]:
        target = _first_target(entry) or ""
        prior_failure = self._nearest_failure(ctx, target)
        synthetic_failure_event = None
        if prior_failure is None and (
            entry.get("failure_reason")
            or entry.get("failure_type")
            or entry.get("gt_failure_type")
        ):
            prior_failure = self._synthetic_failure_from_invocation(ctx, entry)
            synthetic_failure_event = prior_failure.event

        causal_role = (
            "recovery_attempt_after_failure"
            if prior_failure is not None
            else "tool_used_without_known_prior_failure"
        )
        evidence = {
            "source_log_type": entry.get("type"),
            "action": entry.get("action"),
            "target_object": target,
            "start_score": _score(ctx.gt_state),
        }
        if prior_failure is not None:
            evidence["prior_failure_type"] = prior_failure.failure_type
            evidence["prior_failure_step"] = prior_failure.step_count

        event = self._finish(ctx, GTMetricEvent(
            metric="grasp_tool_invocation",
            category="tool_causality",
            subject_object=target or None,
            observed="grasp_tool_called",
            evidence=evidence,
            parent_event_id=(
                prior_failure.event.event_id if prior_failure is not None
                else None
            ),
            causal_role=causal_role,
            confidence=float(entry.get("confidence", 1.0) or 1.0),
        ))
        self._active_tools.append(_ToolAttempt(
            event=event,
            start_step=ctx.step_count,
            subgoal_idx=ctx.subgoal_idx,
            target_object=target,
            start_score=_score(ctx.gt_state),
            prior_failure=prior_failure,
        ))
        self._trim_history(ctx.step_count)
        if synthetic_failure_event is not None:
            return [synthetic_failure_event, event]
        return [event]

    def _record_outcome(
        self,
        ctx: MetricContext,
        entry: dict[str, Any],
    ) -> GTMetricEvent | None:
        target = _first_target(entry) or ""
        attempt = self._pop_active_tool(ctx, target)
        if attempt is None:
            return None

        current_score = _score(ctx.gt_state)
        score_delta = round(current_score - attempt.start_score, 6)
        entry_type = str(entry.get("type", ""))
        completed = (
            bool(target)
            and _object_completed(ctx.gt_state, _norm_name(target))
        )
        if entry_type == "grasp_tool_failed":
            causal_role = "tool_internal_failure"
            observed = "grasp_tool_failed"
        elif score_delta > 0.0 or completed:
            causal_role = "corrective_success"
            observed = "grasp_tool_completed_with_progress"
        elif attempt.prior_failure is not None:
            causal_role = "failed_to_recover"
            observed = "grasp_tool_completed_without_progress"
        else:
            causal_role = "tool_completed_without_known_gt_progress"
            observed = "grasp_tool_completed_without_progress"

        evidence = {
            "source_log_type": entry_type,
            "target_object": target,
            "start_score": attempt.start_score,
            "end_score": current_score,
            "score_delta": score_delta,
            "object_completed": completed,
        }
        if entry.get("reason"):
            evidence["reason"] = entry["reason"]

        event = self._finish(ctx, GTMetricEvent(
            metric="grasp_tool_outcome",
            category="tool_causality",
            subject_object=target or attempt.target_object or None,
            observed=observed,
            evidence=evidence,
            parent_event_id=attempt.event.event_id,
            causal_role=causal_role,
            confidence=float(entry.get("confidence", 1.0) or 1.0),
        ))
        attempt.outcome_event = event
        attempt.outcome_step = ctx.step_count
        self._finished_tools.append(attempt)
        self._trim_history(ctx.step_count)
        return event

    @staticmethod
    def _is_tool_invocation(entry: dict[str, Any]) -> bool:
        entry_type = str(entry.get("type", ""))
        if entry_type in {"gt_grasp_escalation", "hitl_grasp_tool"}:
            return True
        return (
            entry_type == "failure_recovery"
            and entry.get("action") in {"grasp_escalation", "grasp_tool_vlm"}
        )

    def _nearest_failure(
        self,
        ctx: MetricContext,
        target: str,
    ) -> _FailureRecord | None:
        candidates = []
        norm_target = _norm_name(target)
        for failure in self._failures:
            if failure.subgoal_idx != ctx.subgoal_idx:
                continue
            if ctx.step_count - failure.step_count > self.attribution_window_steps:
                continue
            if norm_target and failure.target_objects:
                norm_failure_targets = {_norm_name(t) for t in failure.target_objects}
                if norm_target not in norm_failure_targets:
                    continue
            candidates.append(failure)
        if not candidates:
            return None
        return max(candidates, key=lambda f: f.step_count)

    def _synthetic_failure_from_invocation(
        self,
        ctx: MetricContext,
        entry: dict[str, Any],
    ) -> _FailureRecord:
        target = _first_target(entry)
        failure_type = _entry_failure_type(entry)
        event = self._finish(ctx, GTMetricEvent(
            metric="failure_detected",
            category="failure",
            subject_object=target,
            observed=failure_type,
            evidence={
                "source_log_type": entry.get("type"),
                "reason": entry.get("failure_reason", ""),
                "synthetic_parent_for_tool": True,
            },
            causal_role="root_failure",
        ))
        record = _FailureRecord(
            event=event,
            step_count=ctx.step_count,
            subgoal_idx=ctx.subgoal_idx,
            target_objects={target} if target else set(),
            failure_type=failure_type,
        )
        self._failures.append(record)
        return record

    def _pop_active_tool(
        self,
        ctx: MetricContext,
        target: str,
    ) -> _ToolAttempt | None:
        norm_target = _norm_name(target)
        matching: list[tuple[int, _ToolAttempt]] = []
        for idx, attempt in enumerate(self._active_tools):
            if attempt.subgoal_idx != ctx.subgoal_idx:
                continue
            if norm_target and _norm_name(attempt.target_object) != norm_target:
                continue
            matching.append((idx, attempt))
        if not matching:
            for idx, attempt in enumerate(self._active_tools):
                if attempt.subgoal_idx == ctx.subgoal_idx:
                    matching.append((idx, attempt))
        if not matching:
            return None
        idx, attempt = max(matching, key=lambda item: item[1].start_step)
        del self._active_tools[idx]
        return attempt

    def _recent_tool_for_failure(
        self,
        ctx: MetricContext,
        failure_type: str,
        target_objects: set[str],
    ) -> _ToolAttempt | None:
        candidates: list[_ToolAttempt] = []
        for attempt in self._finished_tools:
            if attempt.subgoal_idx != ctx.subgoal_idx:
                continue
            step = attempt.outcome_step or attempt.start_step
            if ctx.step_count - step > self.attribution_window_steps:
                continue
            if target_objects and attempt.target_object:
                norm_targets = {_norm_name(t) for t in target_objects}
                if _norm_name(attempt.target_object) not in norm_targets:
                    continue
            candidates.append(attempt)
        if not candidates:
            return None
        return max(candidates, key=lambda a: a.outcome_step or a.start_step)

    @staticmethod
    def _targets_overlap(left: set[str], right: set[str]) -> bool:
        if not left or not right:
            return False
        return bool({_norm_name(v) for v in left} & {_norm_name(v) for v in right})

    def _trim_history(self, current_step: int) -> None:
        keep_after = current_step - (self.attribution_window_steps * 4)
        self._failures = [
            failure for failure in self._failures
            if failure.step_count >= keep_after
        ]
        self._finished_tools = [
            attempt for attempt in self._finished_tools
            if (attempt.outcome_step or attempt.start_step) >= keep_after
        ]
