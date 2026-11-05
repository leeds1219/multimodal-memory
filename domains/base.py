"""What a domain provides to the method (the Designer and the learning-from-failure loop, experiments/ours_loop.py).

The method never reads a domain's files or runs its episodes itself; it calls these. A new domain is a folder
domains/<name>/ with a module domain.py whose DOMAIN is an instance of a Domain subclass, plus whatever that domain
needs (its world, its Executor agent, its log reader). Select it with MM_DOMAIN=<name> (paths.py puts its folders on
the import path) and `--domain <name>` where a script takes one.

An episode is a folder. Whatever is inside is the domain's business; the method only gets:
  result(ep)                 {"task", "seed", "instruction", "success", "steps", "horizon_steps"}
  turns(ep, until)           one dict per agent turn that started before clock `until`: {"turn", "t0", "t1", "cmd",
                             "thought", "outcome", "effect"}; t0/t1 = the domain's clock (e.g. game steps)
  commands()                 the commands the agent can send, as its prompt describes them
  log(parts, milestones, markers)
                             the log the Designer reads (designer/terms.py LOG_LAYOUT): parts [(episode, until, marker)]
                             of a timeline, grouped by segment when milestones are given, markers {turn: [lines]}
  checkpoints(ep)            the clocks of ep's saves
  save_before(ep, turn)      (save, clock) of the last save at or before the start of `turn`; ("", 0) when the turn
                             comes before the first save; None when `turn` is not a turn of ep
  state_line(ep, clock, window)
                             the restart point as the Designer is told it (clock, window, state)
  plan(ep)                   the agent's initial plan in ep, or None (planning off)
  run_episode(...)           play one episode (see Domain.run_episode)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Vocabulary:
    """The domain's words in the Designer's prompts (designer/*.py)."""
    name: str  # "a <name> task", "a <name> agent", "in <name>"


class Domain:
    name = "base"
    vocabulary: Vocabulary
    env_label = ""  # "env" field of the LLM call context (ledger, call logs)

    def result(self, ep: Path) -> dict:
        raise NotImplementedError

    def turns(self, ep: Path, until: int | None = None) -> list[dict]:
        raise NotImplementedError

    def commands(self) -> str:
        raise NotImplementedError

    def log(self, parts: list, milestones: list | None = None, markers: dict | None = None) -> str:
        raise NotImplementedError

    def milestones_reached(self, parts: list, milestones: list) -> int:
        """How far the parts (as in log) got: 1 + the order of the furthest milestone reached, 0 when none."""
        raise NotImplementedError

    def checkpoints(self, ep: Path) -> list[int]:
        raise NotImplementedError

    def save_before(self, ep: Path, turn: int) -> tuple[str | Path, int] | None:
        raise NotImplementedError

    def state_line(self, ep: Path, clock: int, window: int) -> str:
        raise NotImplementedError

    def plan(self, ep: Path) -> str | None:
        """The sub-goals the agent wrote before its first turn in ep, one per line, or None when it did not plan."""
        return None

    def llm_config(self, opts: dict) -> str:
        """The LLM config file (configs/llm_*.yaml) the domain's episodes run with."""
        raise NotImplementedError

    def run_episode(self, task: str, seed: int, ep: Path, start_save: str | Path, extras_file: Path,
                    opts: dict) -> dict:
        """Play `task` in the world of `seed` into the new folder `ep`, from `start_save` ("" = from the beginning).
        `extras_file` (JSON: {"hypothesis", "knowledge", "from_start"}) is what the method adds to the agent
        (executor/extras.py). `opts`: the loop's run options (domain-specific ones such as world rules, plus gpu/port,
        mock, save_every and max_steps: the episode ends that many clock units after its start). Returns at least
        {"status", "success", "steps"}; status "finished" or "anomaly" = the episode counts, anything else = crashed,
        and the loop retries it."""
        raise NotImplementedError
