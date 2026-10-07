"""Evaluate a model (base or with a LoRA adapter) on Ken.

    python -m training.evaluate --adapter none --n 3 --tag base
    python -m training.evaluate --adapter training/out/adapter --n 3 --tag finetuned
    python -m training.evaluate --adapter none --lang py --n 3 --tag base_python    # the same model, asked for Python

``--lang py`` gives the model the Python prompt that ``benchmarks/run_llama.py`` gives other models (the table
description, "Python 3, pandas available", the task) instead of the Ken reference, and runs the program with Python.

Three sets:

* ``bench``: the seven tasks of ``benchmarks/tasks.json``, all about ``orders.csv``;
* ``gen``: held-out synthetic tasks from the generator, about the tables used in training;
* ``unseen``: synthetic tasks about ``grades.csv``, a table of a different shape that was never used in training.

Each prompt carries the description of its own table. Programs are run in a separate process in an otherwise
empty directory that contains only that table (and ``orders.csv``).
Sampling: temperature 0.6, top-p 0.95, top-k 64, reasoning off.
"""
import argparse
import json
import pathlib
import sys
import time

from benchmarks.expected import expected, tasks
from benchmarks.run_llama import make_prompt, run_python
from benchmarks.scoring import check_csv_spec, run_isolated, same_out, scratch_dir
from kenlang.llm import extract_code

from .tables import TABLES
from .train import render_prompt

HERE = pathlib.Path(__file__).resolve().parent


def bench_items():
    items = []
    for t in tasks():
        exp, spec = expected(t["id"])
        items.append(dict(id=f"bench{t['id']}", set="bench", table="orders", task=t["task"], expected=exp, csv=spec))
    return items


def generated_items(name, tag):
    rows = [json.loads(line) for line in open(HERE / "data" / f"{name}.jsonl", encoding="utf-8")]
    return [dict(id=r["id"], set=tag, table=r.get("table", "orders"), task=r["task"], expected=r["expected"],
                 csv=r["csv"]) for r in rows]


def gen_items():
    return generated_items("test", "gen")


def unseen_items():
    return generated_items("test_unseen_table", "unseen")


def render(tok, item, lang):
    """The prompt of one item, as chat text with reasoning off."""
    description = TABLES[item["table"]].description
    if lang == "ken":
        return render_prompt(tok, item["task"], description)
    return tok.apply_chat_template([{"role": "user", "content": make_prompt("py", item["task"], description)}],
                                   tokenize=False, add_generation_prompt=True, enable_thinking=False)


def score(code, item, lang="ken"):
    with scratch_dir(TABLES[item["table"]].csv) as work:
        if lang == "ken":
            ok_run, out = run_isolated(code, work, python=sys.executable)
        else:
            ok_run, out = run_python(code, work)
        if not ok_run:
            return False, out
        if not same_out(out, item["expected"]):
            return False, "different output"
        if item["csv"]:
            return check_csv_spec(work / item["csv"]["file"], item["csv"])
        return True, ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="none")
    ap.add_argument("--base", default="unsloth/gemma-4-12b-it")
    ap.add_argument("--n", type=int, default=3, help="attempts per task")
    ap.add_argument("--sets", default="bench,gen,unseen")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default="eval")
    ap.add_argument("--lang", choices=["ken", "py"], default="ken")
    ap.add_argument("--max-new", type=int, default=0, help="default: 400 tokens for Ken, 800 for Python")
    a = ap.parse_args()
    max_new = a.max_new or (400 if a.lang == "ken" else 800)

    from unsloth import FastModel  # noqa: I001  (must be imported before transformers/trl)
    import torch

    model, processor = FastModel.from_pretrained(
        model_name=a.base if a.adapter == "none" else a.adapter, max_seq_length=4096, load_in_4bit=True)
    FastModel.for_inference(model)
    tok = getattr(processor, "tokenizer", processor)

    items = [i for i in bench_items() + gen_items() + unseen_items() if i["set"] in a.sets.split(",")]
    if a.limit:
        items = items[:a.limit]
    out_dir = HERE / "out"
    out_dir.mkdir(exist_ok=True)
    res_path = out_dir / f"eval_{a.tag}.jsonl"
    res_path.write_text("", encoding="utf-8")
    results = []
    for it in items:
        enc = tok(render(tok, it, a.lang), return_tensors="pt", add_special_tokens=False).to(model.device)
        for k in range(1, a.n + 1):
            t0 = time.time()
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=max_new, do_sample=True, temperature=0.6,
                                     top_p=0.95, top_k=64, pad_token_id=tok.pad_token_id)
            raw = tok.decode(gen[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            code = extract_code(raw)
            ok, note = score(code, it, a.lang)
            row = dict(id=it["id"], set=it["set"], table=it["table"], lang=a.lang, attempt=k, ok=bool(ok),
                       sec=round(time.time() - t0, 1),
                       tokens=int(gen.shape[1] - enc["input_ids"].shape[1]), note=note, code=code)
            results.append(row)
            with open(res_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            verdict = "OK" if ok else "FAIL"
            print(f"{it['id']} #{k}: {verdict} ({row['tokens']} tok, {row['sec']}s) {note[:90]}", flush=True)

    print("\n=== SUMMARY", a.tag, "===")
    for s in ("bench", "gen", "unseen"):
        sel = [r for r in results if r["set"] == s]
        if sel:
            print(f"{s}: {sum(r['ok'] for r in sel)}/{len(sel)} ({100 * sum(r['ok'] for r in sel) / len(sel):.0f}%)")
    for i in range(1, 8):
        sel = [r for r in results if r["id"] == f"bench{i}"]
        if sel:
            print(f"  task {i}: {sum(r['ok'] for r in sel)}/{len(sel)}")


if __name__ == "__main__":
    main()
