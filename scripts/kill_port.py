"""Kill the process listening on a TCP port (safe: never matches our shell)."""
import sys, psutil
port = int(sys.argv[1])
for c in psutil.net_connections(kind="tcp"):
    if c.laddr and c.laddr.port == port and c.status == psutil.CONN_LISTEN and c.pid:
        p = psutil.Process(c.pid)
        print("killing", c.pid, " ".join(p.cmdline())[:120])
        p.kill()
