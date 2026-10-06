"""Run the benchmark against any OpenAI-compatible chat endpoint (llama.cpp server, vLLM, ...).

    python -m benchmarks.run_llama --url http://127.0.0.1:8001 --model qwen --lang ken --n 3
    python -m benchmarks.run_llama --url http://127.0.0.1:8001 --model qwen --lang py  --n 3 --think off
    python -m benchmarks.run_llama --model qwen --lang ken --think off --set unseen --n 1

``--lang ken`` gives the model the Ken reference; ``--lang py`` asks for the same tasks in Python
(pandas allowed), which is the baseline for "does Ken help a model that has never seen it?".
``--think off`` sends ``chat_template_kwargs: {"enable_thinking": false}``.

``--set`` picks the tasks, the same three sets that ``training/evaluate.py`` scores: ``bench`` (the seven
hand-written tasks), ``gen`` (held-out generated tasks) and ``unseen`` (generated tasks about a table of a
different shape). Every prompt carries the description of its own table. Programs run in a separate process
in an otherwise empty directory that contains only that table.
"""
import argparse
import json
import pathlib
import subprocess
import sys
import time
import urllib.request

from kenlang.llm import build_prompt, extract_code
from training.tables import TABLES

from .scoring import check_csv_spec, run_isolated, same_out, scratch_dir

PY_PROMPT = """# Data

{data}

# Environment

Python 3. The standard library and pandas are available. The program reads the CSV file described above from
the current directory and prints its result to standard output.

# Task

{task}

Answer with code only.
"""


def load_items(name, only):
    from training.evaluate import bench_items, gen_items, unseen_items
    items = {"bench": bench_items, "gen": gen_items, "unseen": unseen_items}[name]()
    if only:
        items = [i for i in items if i["id"] in {f"bench{n}" for n in only}]
    return items


def make_prompt(lang, task, description):
    if lang == "ken":
        return build_prompt(task, description)
    return PY_PROMPT.format(data=description.strip(), task=task.replace("Ken program", "Python program"))


def chat(url, model, prompt, think, max_tokens):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0.6,
            "max_tokens": max_tokens, "chat_template_kwargs": {"enable_thinking": think}}
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.time()
    res = json.load(urllib.request.urlopen(req, timeout=3600))
    msg = res["choices"][0]["message"]
    return (msg["content"] or "", len(msg.get("reasoning_content") or ""), res["usage"]["completion_tokens"],
            time.time() - t0, res["choices"][0]["finish_reason"])


def run_python(code, work):
    (pathlib.Path(work) / "prog.py").write_text(code, encoding="utf-8")
    r = subprocess.run([sys.executable, "prog.py"], cwd=work, capture_output=True, text=True, timeout=120,
                       encoding="utf-8")
    if r.returncode != 0:
        return False, (r.stderr.strip().splitlines() or ["error"])[-1][:200]
    return True, r.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8001")
    ap.add_argument("--model", required=True)
    ap.add_argument("--lang", choices=["ken", "py"], default="ken")
    ap.add_argument("--think", choices=["on", "off"], default="off")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--set", dest="set_name", choices=["bench", "gen", "unseen"], default="bench")
    ap.add_argument("--only", default="", help="bench only: comma-separated task numbers")
    ap.add_argument("--limit", type=int, default=0, help="use only the first N tasks of the set")
    ap.add_argument("--out", default="", help="append result rows (JSON lines) to this file")
    a = ap.parse_args()
    think = a.think == "on"
    only = {int(x) for x in a.only.split(",") if x}
    rows = []
    items = load_items(a.set_name, only)
    if a.limit:
        items = items[:a.limit]
    for t in items:
        exp, spec = t["expected"], t["csv"]
        if isinstance(exp, list):
            exp = "\n".join(exp)
        prompt = make_prompt(a.lang, t["task"], TABLES[t["table"]].description)
        for k in range(1, a.n + 1):
            raw, reasoning_chars, toks, secs, finish = chat(a.url, a.model, prompt, think, 36000 if think else 6000)
            code = extract_code(raw)
            with scratch_dir(TABLES[t["table"]].csv) as work:
                ok_run, out = run_isolated(code, work) if a.lang == "ken" else run_python(code, work)
                ok = ok_run and same_out(out, exp)
                note = "" if ok_run else out
                if ok and spec:
                    ok, note = check_csv_spec(work / spec["file"], spec)
                elif ok_run and not ok:
                    note = "different output"
            row = dict(lang=a.lang, think=a.think, set=a.set_name, id=t["id"], attempt=k, ok=bool(ok), tokens=toks,
                       reasoning_chars=reasoning_chars, sec=round(secs, 1), finish=finish, note=note)
            rows.append(row)
            if a.out:
                with open(a.out, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(row) + "\n")
            verdict = "OK" if ok else "FAIL"
            print(f"{a.lang} think={a.think} {t['id']} #{k}: {verdict} ({toks} tok, {secs:.0f}s) {note[:80]}",
                  flush=True)
    good = sum(r["ok"] for r in rows)
    print(f"\n{a.lang} think={a.think}: {good}/{len(rows)} ({100 * good / max(len(rows), 1):.0f}%)")


if __name__ == "__main__":
    main()
