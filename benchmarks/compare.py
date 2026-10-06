"""Put the results of several runs side by side in one Markdown table.

    python -m benchmarks.compare "Gemma 4 12B|Ken|off=training/out/eval_base.jsonl" \\
                                 "Qwen3.8-27B|Python|on=benchmarks/results/qwen_py_on_bench.jsonl"

Each argument is ``model|language|reasoning=path``. A path is a JSON-lines file written by
``training/evaluate.py`` or by ``benchmarks/run_llama.py`` (both put ``set``, ``ok`` and ``tokens`` on every row);
several files can be joined with ``+`` (for example one file per set). A column is left empty when a run did not
cover that set.
"""
import json
import pathlib
import sys

SETS = (("bench", "Hand-written tasks"), ("gen", "Generated tasks"), ("unseen", "Table never seen"))
HEAD = ["Model", "Language", "Reasoning"] + [name for _, name in SETS]
HEAD += ["Tokens per answer", "Tokens per correct answer"]


def load(spec):
    rows = []
    for part in spec.split("+"):
        rows += [json.loads(line) for line in open(part, encoding="utf-8") if line.strip()]
    return rows


def share(rows):
    ok = sum(r["ok"] for r in rows)
    return f"{ok}/{len(rows)} ({100 * ok / len(rows):.0f}%)"


def summarize(rows):
    """Per set: correct/attempts. Plus the mean answer length and the tokens spent per correct answer, both
    measured on the hand-written tasks (on all rows if the run did not include them)."""
    out = {}
    for key, _ in SETS:
        sel = [r for r in rows if r["set"] == key]
        out[key] = share(sel) if sel else ""
    used = [r for r in rows if r["set"] == "bench"] or rows
    out["mean_tokens"] = sum(r["tokens"] for r in used) / len(used) if used else None
    good = sum(r["ok"] for r in used)
    out["per_correct"] = sum(r["tokens"] for r in used) / good if good else None
    return out


def fmt(number, none):
    return none if number is None else f"{number:,.0f}"


def table(runs):
    """``runs``: list of (model, language, reasoning, rows). Returns the Markdown table."""
    lines = ["| " + " | ".join(HEAD) + " |", "|" + "---|" * len(HEAD)]
    for model, language, reasoning, rows in runs:
        s = summarize(rows)
        cells = [model, language, reasoning] + [s[k] for k, _ in SETS]
        cells += [fmt(s["mean_tokens"], ""), fmt(s["per_correct"], "none correct")]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv):
    runs = []
    for arg in argv:
        label, _, spec = arg.partition("=")
        parts = label.split("|")
        if len(parts) != 3 or not spec or not all(pathlib.Path(p).exists() for p in spec.split("+")):
            sys.exit(f"expected model|language|reasoning=path[+path...] with existing files, got: {arg}")
        runs.append((*parts, load(spec)))
    print(table(runs))


if __name__ == "__main__":
    main(sys.argv[1:])
