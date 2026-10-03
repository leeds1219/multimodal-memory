"""Run one chain: (env, method, variant, task order) over a list of tasks.

Resumable and crash-safe:
  * an episode whose ``result.json`` has status finished/anomaly is skipped;
  * the method's memory is restored from the snapshot taken after the last
    finished episode before every episode, so a crashed episode leaves no
    trace in memory; a crashed episode is retried up to ``--retries`` times
    and then recorded as ``crashed_final``;
  * after every finished episode the memory is snapshotted to
    ``memory_snapshots/{env}/{chain}/{order}/{idx:02d}_{task}/``;
  * the method's server is (re)started for every episode so it always loads
    memory from disk;
  * watchdog: kill the episode if its heartbeat stalls for ``--stall-min``;
  * stops the whole chain if free disk < 5 GB or the global LLM cap is hit.

Outputs: runs/{env}/{chain}/{order}/{task}/{seed}/ (result.json, trajectory,
llm/calls.jsonl + images, client.log, server.log, artifacts).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

REPRO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPRO / "envs"))
sys.path.insert(0, str(REPRO / "llm"))

from common import MIN_FREE_GB, ORDERS, RUNS_ROOT, SEEDS, TASKS, disk_free_gb, write_ctx  # noqa: E402

PY = "/opt/conda/envs/mcagent/bin/python"
DONE = {"finished", "anomaly", "crashed_final"}
MINUTE = 1200


def log(msg: str) -> None:
    print(time.strftime("%m-%d %H:%M:%S"), msg, flush=True)


def wait_port(port: int, timeout: float = 300) -> bool:
    import urllib.request
    t = time.time()
    while time.time() - t < timeout:
        for path in ("/status", "/docs"):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=5)
                return True
            except Exception:
                pass
        time.sleep(2)
    return False


def kill_tree(p: subprocess.Popen) -> None:
    try:
        os.killpg(os.getpgid(p.pid), signal.SIGKILL)
    except Exception:
        pass


def free_port(port: int) -> None:
    subprocess.run([PY, str(REPRO / "scripts" / "kill_port.py"), str(port)], capture_output=True)


class Chain:
    def __init__(self, a):
        self.a = a
        self.chain_id = a.method + (f"-{a.variant}" if a.variant else "")
        self.runs = RUNS_ROOT / "runs" / a.env / self.chain_id / a.order
        self.state = RUNS_ROOT / "state" / a.env / self.chain_id / a.order
        self.snaps = RUNS_ROOT / "memory_snapshots" / a.env / self.chain_id / a.order
        for d in (self.runs, self.state, self.snaps):
            d.mkdir(parents=True, exist_ok=True)
        self.ctx_file = self.state / "ctx.json"
        self.env_vars = {**os.environ, "LLM_CTX_FILE": str(self.ctx_file), "METHOD": a.method}
        if a.mock:
            self.env_vars["LLM_MOCK"] = "1"
        # A variant token t with configs/llm_<t>.yaml selects that planner model (e.g. "g38").
        for tok in (a.variant or "").split("-"):
            if tok and (REPRO / "configs" / f"llm_{tok}.yaml").exists():
                self.env_vars["LLM_CONFIG"] = str(REPRO / "configs" / f"llm_{tok}.yaml")
        if "cond6" in (a.variant or ""):  # sensitivity: Env M STEVE-1 at the official 6.0
            self.env_vars["MINEEVOLVE_STEVE_COND_SCALE"] = "6.0"
        # Which server provides STEVE-1: the method's own (Stage A) or, for the
        # Stage B ports, the evaluation env's native one (MineEvolve server for
        # Env M, Optimus-1 server for Env O); their LLM parts stay unused.
        self.server_kind = a.method if a.method in ("mineevolve", "optimus1") else (
            "mineevolve" if a.env == "M" else "optimus1")
        # The env's native STEVE-1 server (D16/D32): every method in an env gets
        # its low-level actions from the same wrapper. Stage A cross pairs
        # (MineEvolve in O, Optimus-1 in M) need it as a second server.
        native = "mineevolve" if a.env == "M" else "optimus1"
        self.steve_kind = native if (a.method in ("mineevolve", "optimus1") and a.method != native) else None
        self.steve_port = a.port + 1000
        if a.method == "optimus1":
            from optimus_workdir import make_workdir
            self.wd = make_workdir(self.state / "wd", (a.variant or "empty").split("-")[0])
            self.memory = self.wd / "src" / "optimus1" / "memories" / "v1"
            if "goalfix" in (a.variant or ""):
                self.env_vars["OPTIMUS_GOALFIX"] = "1"
            if "isoworld" in (a.variant or ""):  # D36
                self.env_vars["OPTIMUS_ISOWORLD"] = "1"
            if "logfix" in (a.variant or ""):  # D33
                self.env_vars["OPTIMUS_LOGFIX"] = "1"
            if "craftfix" in (a.variant or ""):  # D37
                self.env_vars["OPTIMUS_CRAFTFIX"] = "1"
            if "tagfix" in (a.variant or ""):  # D40
                self.env_vars["OPTIMUS_TAGFIX"] = "1"
            if "replanfix" in (a.variant or ""):  # D41
                self.env_vars["OPTIMUS_REPLANFIX"] = "1"
            if "escapefix" in (a.variant or ""):  # D42
                self.env_vars["OPTIMUS_ESCAPEFIX"] = "1"
            if "promptfix" in (a.variant or ""):  # D43; promptfix2 = D45
                self.env_vars["OPTIMUS_PROMPTFIX"] = "2" if "promptfix2" in a.variant else "1"
            if "forest" in (a.variant or ""):  # D38
                self.env_vars["OPTIMUS_BIOME"] = "forest"
            if "memfix" in (a.variant or "") and not (self.state / "memfix_done").exists():  # D35
                from optimus_workdir import fix_memory_typos
                n = fix_memory_typos(self.memory)
                (self.state / "memfix_done").write_text(str(n))
                log(f"memfix: rewrote {n} 'craft chest' steps in this chain's memory")
        elif a.method == "mineevolve":
            self.memory = self.state / "kb"
            self.memory.mkdir(exist_ok=True)
        elif a.method in ("deps", "jarvis1", "steve1"):  # no cross-task memory
            self.memory = self.state / "no_memory"
            self.memory.mkdir(exist_ok=True)
            if self.server_kind == "optimus1":
                from optimus_workdir import make_workdir
                self.wd = make_workdir(self.state / "wd", "empty")
            else:
                self.kb_unused = self.state / "kb_unused"
        else:
            raise SystemExit(f"unknown method {a.method}")
        if self.steve_kind == "optimus1":
            from optimus_workdir import make_workdir
            self.steve_wd = make_workdir(self.state / "steve_wd", "empty")
        elif self.steve_kind == "mineevolve":
            self.steve_kb = self.state / "steve_kb_unused"
        self.initial = self.snaps / "00_initial"
        if not self.initial.exists():
            shutil.copytree(self.memory, self.initial, copy_function=self._link_or_copy)

    # ---------------------------------------------------------------- memory
    # Snapshots are deltas against 00_initial (every file whose content differs
    # or that is new), so a 200k-file pre-built memory is not copied per episode.
    # Restore = initial + latest delta, done only when memory may be dirty
    # (an episode started after the last clean snapshot).
    @staticmethod
    def _link_or_copy(src, dst):
        if str(src).endswith(".json"):  # methods rewrite JSON in place: never share inodes
            return shutil.copy2(src, dst)
        return os.link(src, dst)

    def last_snapshot(self) -> Path:
        snaps = sorted(p for p in self.snaps.iterdir() if p.is_dir() and not p.name.startswith("00_initial"))
        return snaps[-1] if snaps else self.initial

    def _clean_marker(self) -> Path:
        return self.state / "memory_clean"

    def restore_memory(self) -> None:
        last = self.last_snapshot()
        m = self._clean_marker()
        if m.exists() and m.read_text() == last.name and self.memory.exists():
            m.unlink()  # memory == last snapshot; it becomes dirty from here on
            return
        if m.exists():
            m.unlink()
        shutil.rmtree(self.memory, ignore_errors=True)
        shutil.copytree(self.initial, self.memory, copy_function=self._link_or_copy)
        if last != self.initial:
            for f in last.rglob("*"):
                if f.is_file():
                    dst = self.memory / f.relative_to(last)
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    if dst.exists():
                        dst.unlink()
                    shutil.copy2(f, dst)

    def snapshot(self, idx: int, task: str) -> Path:
        dst = self.snaps / f"{idx:02d}_{task}"
        if dst.exists():
            shutil.rmtree(dst)
        dst.mkdir(parents=True)
        for f in self.memory.rglob("*"):
            if not f.is_file():
                continue
            rel = f.relative_to(self.memory)
            base = self.initial / rel
            if base.exists() and base.stat().st_size == f.stat().st_size and (
                    not str(f).endswith(".json") or base.read_bytes() == f.read_bytes()):
                continue
            (dst / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dst / rel)
        self._clean_marker().write_text(dst.name)
        return dst

    # ---------------------------------------------------------------- server
    def start_steve_server(self, ep: Path):
        if self.steve_kind is None:
            return None
        free_port(self.steve_port)
        if self.steve_kind == "mineevolve":
            cmd = [str(REPRO / "scripts" / "start_mineevolve_server.sh"), str(self.a.gpu), str(self.steve_port),
                   str(self.steve_kb), str(self.ctx_file), str(ep / "steve_server.log")]
        else:
            cmd = [str(REPRO / "scripts" / "start_optimus_server.sh"), str(self.a.gpu), str(self.steve_port),
                   str(self.steve_wd), str(self.ctx_file), str(ep / "steve_server.log")]
        p = subprocess.Popen(cmd, env=self.env_vars, start_new_session=True,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if not wait_port(self.steve_port):
            kill_tree(p)
            raise RuntimeError("STEVE-1 server did not come up")
        return p

    def start_server(self, ep: Path) -> subprocess.Popen:
        free_port(self.a.port)
        if self.server_kind == "mineevolve":
            kb = self.memory if self.a.method == "mineevolve" else self.kb_unused
            cmd = [str(REPRO / "scripts" / "start_mineevolve_server.sh"), str(self.a.gpu), str(self.a.port),
                   str(kb), str(self.ctx_file), str(ep / "server.log")]
        else:
            cmd = [str(REPRO / "scripts" / "start_optimus_server.sh"), str(self.a.gpu), str(self.a.port),
                   str(self.wd), str(self.ctx_file), str(ep / "server.log")]
        p = subprocess.Popen(cmd, env=self.env_vars, start_new_session=True,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if not wait_port(self.a.port):
            kill_tree(p)
            raise RuntimeError("server did not come up")
        return p

    # --------------------------------------------------------------- episode
    def run_episode(self, task: str, ep: Path) -> dict:
        seed = (SEEDS[task] + self.a.seed_offset)
        from cross_glue import env_m_group, env_o_group
        genv = (env_m_group if self.a.env == "M" else env_o_group)(TASKS[task]["group"])
        write_ctx(self.ctx_file, {"env": self.a.env, "method": self.a.method, "chain": self.chain_id,
                                  "horizon_steps": int(genv["max_minutes"]) * MINUTE,
                                  "order_id": self.a.order, "task": task, "seed": seed,
                                  "instruction": TASKS[task]["instruction"], "episode_dir": str(ep),
                                  "run_id": f"{self.a.env}/{self.chain_id}/{self.a.order}/{task}"})
        server = self.start_server(ep)
        try:
            steve_server = self.start_steve_server(ep)
        except Exception:
            kill_tree(server)
            raise
        script = {"mineevolve": "mineevolve_episode.py", "optimus1": "optimus_episode.py"}.get(
            self.a.method, "stageb_episode.py")
        cmd = ["xvfb-run", "-a", PY, str(REPRO / "envs" / script), "--env", self.a.env, "--task", task,
               "--seed", str(seed), "--order-id", self.a.order, "--episode-dir", str(ep), "--port", str(self.a.port)]
        if self.a.method == "optimus1":
            cmd += ["--workdir", str(self.wd)]
        if self.a.method in ("deps", "jarvis1", "steve1"):
            cmd += ["--method", self.a.method]
        if self.steve_kind is not None:
            cmd += ["--steve-port", str(self.steve_port)]
            if self.steve_kind == "optimus1":
                cmd += ["--steve-workdir", str(self.steve_wd)]
        t0 = time.time()
        with open(ep / "launcher.log", "a") as lf:
            p = subprocess.Popen(cmd, env=self.env_vars, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)
            reason = None
            while p.poll() is None:
                time.sleep(20)
                hb = ep / "heartbeat"
                last = hb.stat().st_mtime if hb.exists() else t0
                if time.time() - last > self.a.stall_min * 60:
                    reason = f"stalled {self.a.stall_min} min"
                if (ep / "ANOMALY").exists() and time.time() - (ep / "ANOMALY").stat().st_mtime > 120:
                    reason = reason or "anomaly marker (episode did not stop)"
                if reason:
                    log(f"  killing episode: {reason}")
                    kill_tree(p)
                    break
        kill_tree(server)
        free_port(self.a.port)
        if steve_server is not None:
            kill_tree(steve_server)
            free_port(self.steve_port)
        res_f = ep / "result.json"
        if res_f.exists():
            return json.loads(res_f.read_text())
        status = "anomaly" if (ep / "ANOMALY").exists() else "crashed"
        return {"status": status, "error": reason or f"launcher exit {p.returncode}"}

    # ------------------------------------------------------------------ main
    def run(self, tasks: list[str]) -> int:
        from gemini_client import CFG, ledger_total
        for idx, task in enumerate(tasks, start=1):
            ep = self.runs / task / str((SEEDS[task] + self.a.seed_offset))
            res_f = ep / "result.json"
            if res_f.exists() and json.loads(res_f.read_text()).get("status") in DONE:
                continue
            if disk_free_gb() < MIN_FREE_GB:
                log(f"STOP: disk free {disk_free_gb():.1f} GB < {MIN_FREE_GB}")
                (RUNS_ROOT / "STOP_DISK").write_text(time.ctime())
                return 2
            if not self.a.mock and ledger_total() >= CFG["global_cap_usd"]:
                log("STOP: global LLM cap reached")
                return 2
            stop_budget = Path(CFG["log_root"]) / "STOP_BUDGET"  # global, shared by every suite
            if not self.a.mock and stop_budget.exists():
                log("STOP: STOP_BUDGET present (tripwire) -> " + stop_budget.read_text().strip())
                return 2
            for attempt in range(1, self.a.retries + 2):
                self.restore_memory()
                if ep.exists():
                    shutil.move(str(ep), str(ep) + f".crash{int(time.time())}")
                ep.mkdir(parents=True)
                log(f"[{idx}/{len(tasks)}] {self.a.env}/{self.chain_id}/{self.a.order} {task} attempt {attempt}")
                try:
                    res = self.run_episode(task, ep)
                except Exception as e:
                    res = {"status": "crashed", "error": repr(e)}
                log(f"  -> {res.get('status')} success={res.get('success')} steps={res.get('steps')} "
                    f"calls={res.get('llm_calls')} cost={res.get('cost_usd')} wall={res.get('wall_time_s')}")
                if res.get("status") in DONE:
                    break
            else:
                res = {**res, "status": "crashed_final", "task": task, "env": self.a.env,
                       "method": self.a.method, "order_id": self.a.order, "success": False}
                (ep / "result.json").write_text(json.dumps(res, indent=2))
            if res.get("status") in ("finished", "anomaly"):
                self.snapshot(idx, task)
        return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True, choices=["O"])
    ap.add_argument("--method", required=True, choices=["optimus1"])
    ap.add_argument("--variant", default="", help="optimus1: empty | prebuilt")
    ap.add_argument("--order", default="order0")
    ap.add_argument("--tasks", default="", help="comma list; default = the whole order")
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--stall-min", type=float, default=60.0,
                    help="kill an episode whose env took no step for this many wall minutes (infra hang); above the worst-case LLM wait (300 s x 7 attempts)")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--seed-offset", type=int, default=0, help="extra-seed runs: seed = seeds.yaml + offset")
    a = ap.parse_args()
    if already_running(a):
        log(f"chain {a.env}/{a.method}/{a.variant}/{a.order} already running -> exit 4")
        return 4
    tasks = a.tasks.split(",") if a.tasks else ORDERS[a.order]
    return Chain(a).run(tasks)


def already_running(a) -> bool:
    """True if another live chain.py has the same (env, method, variant, order),
    so a restarted supervisor / watchdog can never launch a duplicate."""
    import psutil
    me = os.getpid()
    want = {"--env": a.env, "--method": a.method, "--variant": a.variant or "", "--order": a.order}
    for p in psutil.process_iter(["pid", "cmdline"]):
        c = p.info["cmdline"] or []
        # only a python interpreter running chain.py (not wrappers like `timeout`)
        if p.info["pid"] == me or len(c) < 2 or "python" not in os.path.basename(c[0]) \
                or not c[1].endswith("scripts/chain.py"):
            continue
        got = {k: (c[c.index(k) + 1] if k in c and c.index(k) + 1 < len(c) else "") for k in want}
        if got == want:
            return True
    return False


if __name__ == "__main__":
    sys.exit(main())
