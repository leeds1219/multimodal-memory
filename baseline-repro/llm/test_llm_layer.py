"""Debug check 2: shared LLM layer with real Gemini calls (no game).

Each scenario runs in a subprocess with its own config overrides so the
guards can be tested with tiny limits without touching the real ledger.
"""
import base64, glob, json, os, subprocess, sys, tempfile, textwrap, time
from pathlib import Path
import yaml

HERE = Path(__file__).resolve().parent
BASE = yaml.safe_load(open(HERE.parent / "configs" / "llm.yaml"))
TMP = Path(tempfile.mkdtemp(prefix="llmtest_", dir="/home/rag/data/repro_runs"))


def run(name, overrides, body, extra_env=None):
    cfg = {**BASE, "ledger_path": str(TMP / name / "ledger.jsonl"), "log_root": str(TMP / name), **overrides}
    (TMP / name).mkdir(parents=True, exist_ok=True)
    cfgf = TMP / name / "llm.yaml"; cfgf.write_text(yaml.safe_dump(cfg))
    ctxf = TMP / name / "ctx.json"
    ctxf.write_text(json.dumps({"method": "llmtest", "run_id": name, "episode_dir": str(TMP / name / "ep")}))
    code = "import sys; sys.path.insert(0, %r)\nfrom gemini_client import *\nc = GeminiClient()\n" % str(HERE) + textwrap.dedent(body)
    env = {**os.environ, "LLM_CONFIG": str(cfgf), "LLM_CTX_FILE": str(ctxf), **(extra_env or {})}
    t = time.time()
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    print(f"--- {name} ({time.time()-t:.1f}s) rc={r.returncode}\n{r.stdout.strip()}\n{r.stderr.strip()[-400:]}")
    return TMP / name


img = sorted(glob.glob("/home/rag/data/official/NeurIPS24-Optimus-1/src/optimus1/memories/v1/reflection/img/*"))[0]


d = run("real_text", {}, """
r = c.chat.completions.create(model="gpt-4o", messages=[{"role":"user","content":"Name the tool needed to mine iron ore in Minecraft. One line."}], max_tokens=2000)
print("reply:", r.choices[0].message.content.strip()[:120])
""")
d2 = run("real_image", {}, f"""
import base64
b64 = base64.b64encode(open({img!r}, "rb").read()).decode()
r = c.chat.completions.create(model="gpt-4o", messages=[{{"role":"user","content":[{{"type":"text","text":"What biome is this? One word."}},{{"type":"image_url","image_url":{{"url":"data:image/jpeg;base64," + b64}}}}]}}], max_tokens=2000)
print("reply:", r.choices[0].message.content.strip()[:80])
""")
for dd in (d, d2):
    rec = json.loads(open(dd / "ep/llm/calls.jsonl").readline())
    print(dd.name, {k: rec[k] for k in ("model", "requested_model", "tokens_in", "tokens_out_visible", "tokens_out_billed", "cost_usd", "latency_s", "attempts", "finish_reason", "caller")},
          "images saved:", len(list((dd / "ep/llm/images").glob("*"))) if (dd / "ep/llm/images").exists() else 0)

run("backoff", {"base_url": "http://127.0.0.1:9/v1/", "max_retries": 3}, """
import time
t=time.time()
try:
    c.chat.completions.create(model="x", messages=[{"role":"user","content":"hi"}])
except LLMFailed as e:
    print("LLMFailed after %.1fs:" % (time.time()-t), str(e)[:80])
""")
rec = json.loads(open(TMP / "backoff/ep/llm/calls.jsonl").readline())
print("backoff logged attempts:", rec["attempts"], "error:", rec.get("error", "")[:60])

run("global_cap", {"global_cap_usd": 1e-9}, """
c.chat.completions.create(model="x", messages=[{"role":"user","content":"Say OK."}])
print("first call ok (ledger now > cap)")
import gemini_client; gemini_client._TOTAL_CACHE["t"] = 0
try:
    c.chat.completions.create(model="x", messages=[{"role":"user","content":"Say OK."}])
    print("ERROR: cap not enforced")
except BudgetExceeded as e:
    print("BudgetExceeded:", e)
""")

run("episode_guard", {"episode_mean_calls": {"llmtest": 0.1}, "episode_call_multiplier": 10.0}, """
for i in range(3):
    try:
        c.chat.completions.create(model="x", messages=[{"role":"user","content":"Say OK."}])
        print("call", i+1, "ok")
    except EpisodeAnomaly as e:
        print("EpisodeAnomaly at call", i+1, ":", e)
""")
print("ANOMALY marker:", (TMP / "episode_guard/ep/ANOMALY").read_text().strip())

key = open("/home/rag/data/env.yaml").read().split("=", 1)[1].strip().strip('"\'')
leaks = subprocess.run(["grep", "-rl", key, str(TMP)], capture_output=True, text=True).stdout.strip()
print("key found in logs:", bool(leaks))
print("TMP", TMP)
