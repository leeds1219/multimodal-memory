"""What a domain provides to the method (System 2 and the learning-from-failure loop, experiments/ours_loop.py).

The method never reads a domain's files or runs its episodes itself; it calls these. A new domain is a folder
domains/<name>/ with a module domain.py whose DOMAIN is an instance of a Domain subclass, plus whatever that domain
needs (its world, its System 1 agent, its log reader). Select it with MM_DOMAIN=<name> (paths.py puts its folders on
the import path) and `--domain <name>` where a script takes one.

An episode is a folder. Whatever is inside is the domain's business; the method only gets:
  result(ep)            {"task", "seed", "instruction", "success", "steps", "horizon_steps"}
  turns(ep)             one dict per agent turn: {"turn", "t0", "t1", "cmd", "thought", "outcome", "effect"};
                        t0/t1 = the domain's clock (e.g. game steps) at the turn's start/end
  compress(ep)          the text System 2 reads: the episode as the agent played it, compressed
  raw_window(ep, a, b)  turns a..b as logged, for the context around a failure point
  save_before(ep, turn) (save, clock) of the last saved state at or before the start of `turn`; ("", 0) when the
                        turn comes before the first save (restart the same world from the beginning); None when
                        `turn` is not a turn of ep
  state_at(ep, turn)    the agent's state at the start of `turn`, as System 2 is told it (task, turn, clock, ...)
  run_episode(...)      play one episode (see Domain.run_episode)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Vocabulary:
    """The domain's words in System 2's prompts (system2/*.py)."""
    name: str  # "a failed attempt at a <name> task", "You improve a <name> agent", "an experiment in <name>"
    log_shows: str  # what the episode log records that a prediction can be checked against


class Domain:
    name = "base"
    vocabulary: Vocabulary
    env_label = ""  # "env" field of the LLM call context (ledger, call logs)

    def result(self, ep: Path) -> dict:
        raise NotImplementedError

    def turns(self, ep: Path) -> list[dict]:
        raise NotImplementedError

    def compress(self, ep: Path) -> str:
        raise NotImplementedError

    def raw_window(self, ep: Path, t_from: int, t_to: int) -> list[dict]:
        raise NotImplementedError

    def save_before(self, ep: Path, turn: int) -> tuple[str | Path, int] | None:
        raise NotImplementedError

    def state_at(self, ep: Path, turn: int) -> dict:
        raise NotImplementedError

    def llm_config(self, opts: dict) -> str:
        """The LLM config file (configs/llm_*.yaml) the domain's episodes run with."""
        raise NotImplementedError

    def run_episode(self, task: str, seed: int, ep: Path, start_save: str | Path, extras_file: Path,
                    opts: dict) -> dict:
        """Play `task` in the world of `seed` into the new folder `ep`, from `start_save` ("" = from the beginning).
        `extras_file` (JSON: {"hypothesis", "knowledge", "from_start"}) is what the method adds to the agent
        (system1/extras.py). `opts`: the loop's run options (domain-specific ones such as world rules, plus gpu/port
        and mock). Returns at least {"status", "success", "steps"}; status "finished" or "anomaly" = the episode
        counts, anything else = crashed, and the loop retries it."""
        raise NotImplementedError
