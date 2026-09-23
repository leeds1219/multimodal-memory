"""Block until no chain.py process whose args contain every given token is alive."""
import sys, time, psutil
tokens = sys.argv[1:]
def alive():
    n = 0
    for p in psutil.process_iter(["cmdline"]):
        c = " ".join(p.info["cmdline"] or [])
        if "scripts/chain.py" in c and all(t in c for t in tokens) and "wait_chains" not in c:
            n += 1
    return n
while alive():
    time.sleep(30)
print("all done")
