"""Repair the corrupt JSON files in the authors' released Optimus-1 memory
(DEVIATIONS: kind b). Keeps every complete record; originals are stashed.

  "Extra data"      two writes concatenated -> keep the first JSON document
  truncated/broken  keep the longest prefix that parses once its open
                    brackets are closed (drops only the incomplete tail)
"""
import json, shutil, sys
from pathlib import Path


def closers(prefix: str) -> str:
    stack, in_str, esc = [], False, False
    for ch in prefix:
        if in_str:
            esc = (ch == "\\" and not esc)
            if ch == '"' and not esc:
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "[{":
            stack.append("]" if ch == "[" else "}")
        elif ch in "]}" and stack:
            stack.pop()
    return None if in_str else "".join(reversed(stack))


def repair(text: str):
    fixes = 0
    while True:  # lost "," between two values: re-insert it (up to 50 times)
        try:
            json.loads(text)
            return (None, "ok") if fixes == 0 else (json.loads(text), f"{fixes} missing comma(s) re-inserted")
        except json.JSONDecodeError as e:
            if e.msg == "Expecting ',' delimiter" and fixes < 50 and text[e.pos:e.pos + 1] in '[{"':
                text = text[:e.pos] + "," + text[e.pos:]
                fixes += 1
                continue
            err = e
            break
    e = err
    if True:
        if e.msg == "Extra data":
            obj, _ = json.JSONDecoder().raw_decode(text)
            return obj, f"extra data after char {e.pos} dropped"
        pos = e.pos
    cut = pos
    while cut > 0:
        cut = max(text.rfind("}", 0, cut), text.rfind("]", 0, cut))
        if cut < 0:
            break
        head = text[: cut + 1]
        cl = closers(head)
        if cl is not None:
            try:
                return json.loads(head + cl), f"kept first {cut + 1} of {len(text)} chars (error at {pos})"
            except json.JSONDecodeError:
                pass
    raise ValueError("unrepairable")


def count(o):
    if isinstance(o, dict):
        return sum(count(v) for v in o.values()) if any(isinstance(v, (list, dict)) for v in o.values()) else 1
    if isinstance(o, list):
        return len(o)
    return 1


if __name__ == "__main__":
    root = Path(sys.argv[1]); stash = Path(sys.argv[2]); stash.mkdir(parents=True, exist_ok=True)
    for f in sorted(root.rglob("*.json")):
        text = f.read_text()
        obj, how = repair(text)
        if obj is None:
            continue
        rel = f.relative_to(root)
        (stash / rel).parent.mkdir(parents=True, exist_ok=True)
        if not (stash / rel).exists():
            shutil.copy2(f, stash / rel)
        tmp = f.with_suffix(".json.tmp"); tmp.write_text(json.dumps(obj, indent=2)); tmp.replace(f)
        print(f"{rel}: {how}; records kept ~{count(obj)}")
