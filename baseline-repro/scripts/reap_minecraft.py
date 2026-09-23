"""Find (and with --kill, terminate) Minecraft instances no episode is using.

A MineRL Java instance is in use iff some Python process holds an ESTABLISHED
TCP connection to its --envPort. Instances younger than --min-age minutes are
left alone (they may still be starting up).
"""
import argparse, re, time, psutil
ap = argparse.ArgumentParser(); ap.add_argument("--kill", action="store_true"); ap.add_argument("--min-age", type=float, default=10)
a = ap.parse_args()
used = set()
for c in psutil.net_connections(kind="tcp"):
    if c.status == psutil.CONN_ESTABLISHED and c.raddr and c.pid:
        try:
            if "python" in psutil.Process(c.pid).name():
                used.add(c.raddr.port)
        except psutil.Error:
            pass
now = time.time()
for p in psutil.process_iter(["pid", "name", "cmdline", "create_time"]):
    c = " ".join(p.info["cmdline"] or [])
    m = re.search(r"mcprec-6\.13\.jar --envPort=(\d+)", c)
    if not m or p.info["name"] != "java":
        continue
    port, age = int(m[1]), (now - p.info["create_time"]) / 60
    state = "in-use" if port in used else ("young" if age < a.min_age else "LEAKED")
    print(f"{state:7} pid={p.info['pid']} port={port} age={age:.0f}min")
    if state == "LEAKED" and a.kill:
        for q in [p] + [psutil.Process(p.ppid())] if psutil.pid_exists(p.ppid()) else [p]:
            try:
                if "java" in q.name() or "launchClient" in " ".join(q.cmdline()):
                    q.kill()
            except psutil.Error:
                pass
