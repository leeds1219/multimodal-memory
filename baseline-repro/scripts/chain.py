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
        # Which server provides STEVE-1: the method's own (Stage A) or, for the
        # Stage B ports, the evaluation env's native one (MineEvolve server for
        # Env M, Optimus-1 server for Env O); their LLM parts stay unused.
        self.server_kind = a.method if a.method in ("mineevolve", "optimus1") else (
            "mineevolve" if a.env == "M" else "optimus1")
        if a.method == "optimus1":
            from optimus_workdir import make_workdir
            self.wd = make_workdir(self.state / "wd", a.variant or "empty")
            self.memory = self.wd / "src" / "optimus1" / "memories" / "v1"
        elif a.method == "mineevolve":
            self.memory = self.state / "kb"
            self.memory.mkdir(exist_ok=True)
        elif a.method in ("deps", "jarvis1"):  # no cross-task memory
            self.memory = self.state / "no_memory"
            self.memory.mkdir(exist_ok=True)
            if self.server_kind == "optimus1":
                from optimus_workdir import make_workdir
                self.wd = make_workdir(self.state / "wd", "empty")
            else:
                self.kb_unused = self.state / "kb_unused"
        else:
            raise SystemExit(f"unknown method {a.method}")
        self.initial = self.snaps / "00_initial"
        if not self.initial.exists():
            shutil.copytree(self.memory, self.initial)

    # ---------------------------------------------------------------- memory
    def last_snapshot(self) -> Path:
        snaps = sorted(p for p in self.snaps.iterdir() if p.is_dir() and not p.name.startswith("00_initial"))
        return snaps[-1] if snaps else self.initial

    def restore_memory(self) -> None:
        src = self.last_snapshot()
        shutil.rmtree(self.memory, ignore_errors=True)
        shutil.copytree(src, self.memory, copy_function=os.link)  # hardlinks: cheap
        # hardlinks are shared with the snapshot: break them before writing
        for f in self.memory.rglob("*.json"):
            data = f.read_bytes(); f.unlink(); f.write_bytes(data)

    def snapshot(self, idx: int, task: str) -> Path:
        dst = self.snaps / f"{idx:02d}_{task}"
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(self.memory, dst, copy_function=os.link)
        for f in self.memory.rglob("*.json"):  # keep snapshot immutable
            data = f.read_bytes(); f.unlink(); f.write_bytes(data)
        return dst

    # ---------------------------------------------------------------- server
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
        seed = SEEDS[task]
        write_ctx(self.ctx_file, {"env": self.a.env, "method": self.a.method, "chain": self.chain_id,
                                  "order_id": self.a.order, "task": task, "seed": seed,
                                  "instruction": TASKS[task]["instruction"], "episode_dir": str(ep),
                                  "run_id": f"{self.a.env}/{self.chain_id}/{self.a.order}/{task}"})
        server = self.start_server(ep)
        script = {"mineevolve": "mineevolve_episode.py", "optimus1": "optimus_episode.py"}.get(
            self.a.method, "stageb_episode.py")
        cmd = ["xvfb-run", "-a", PY, str(REPRO / "envs" / script), "--env", self.a.env, "--task", task,
               "--seed", str(seed), "--order-id", self.a.order, "--episode-dir", str(ep), "--port", str(self.a.port)]
        if self.a.method == "optimus1":
            cmd += ["--workdir", str(self.wd)]
        if self.a.method in ("deps", "jarvis1"):
            cmd += ["--method", self.a.method]
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
        res_f = ep / "result.json"
        if res_f.exists():
            return json.loads(res_f.read_text())
        status = "anomaly" if (ep / "ANOMALY").exists() else "crashed"
        return {"status": status, "error": reason or f"launcher exit {p.returncode}"}

    # ------------------------------------------------------------------ main
    def run(self, tasks: list[str]) -> int:
        from gemini_client import CFG, ledger_total
        for idx, task in enumerate(tasks, start=1):
            ep = self.runs / task / str(SEEDS[task])
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
    ap.add_argument("--env", required=True, choices=["M", "O"])
    ap.add_argument("--method", required=True, choices=["mineevolve", "optimus1", "deps", "jarvis1"])
    ap.add_argument("--variant", default="", help="optimus1: empty | prebuilt")
    ap.add_argument("--order", default="order0")
    ap.add_argument("--tasks", default="", help="comma list; default = the whole order")
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--stall-min", type=float, default=20.0)
    ap.add_argument("--mock", action="store_true")
    a = ap.parse_args()
    tasks = a.tasks.split(",") if a.tasks else ORDERS[a.order]
    return Chain(a).run(tasks)


if __name__ == "__main__":
    sys.exit(main())
