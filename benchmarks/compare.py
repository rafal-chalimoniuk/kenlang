"""Put the results of several runs side by side in one Markdown table.

    python -m benchmarks.compare "Gemma 4 12B|Ken|off=training/out/eval_base.jsonl" \\
                                 "Qwen3.8-27B|Python|on=benchmarks/results/qwen_python_on_bench.jsonl"

Each argument is ``model|language|reasoning=path``. A path is a JSON-lines file written by
``training/evaluate.py`` or by ``benchmarks/run_llama.py`` (both put ``set``, ``id``, ``ok`` and ``tokens`` on every
row); several files can be joined with ``+`` (for example one file per set). A column is left empty when a run did
not cover that set.

Some generated questions have a correct output that depends on how ties are broken: "the 3 products with the most
items" is not determined when the third and fourth have the same number. Ken keeps the order of first appearance,
pandas sorts the keys, and a program in another language need not do either. The data generator marks such questions
``ambiguous``; the tables leave them out (``--strict`` counts them) so that no language is marked down for a choice
the question did not fix.
"""
import json
import pathlib
import sys

DATA = pathlib.Path(__file__).resolve().parent.parent / "training" / "data"
SETS = (("bench", "Hand-written tasks"), ("gen", "Generated tasks"), ("unseen", "Table never seen"))
HEAD = ["Model", "Language", "Reasoning"] + [name for _, name in SETS]
HEAD += ["Tokens per answer", "Tokens per correct answer"]


def load(spec):
    rows = []
    for part in spec.split("+"):
        rows += [json.loads(line) for line in open(part, encoding="utf-8") if line.strip()]
    return rows


def ambiguous_ids():
    """Ids of the generated test questions whose correct output depends on tie-breaking."""
    ids = set()
    for name in ("test", "test_unseen_table"):
        path = DATA / f"{name}.jsonl"
        if path.exists():
            ids |= {r["id"] for r in map(json.loads, open(path, encoding="utf-8")) if r.get("ambiguous")}
    return ids


def fair(rows):
    """``rows`` without the questions whose correct output depends on tie-breaking."""
    skip = ambiguous_ids()
    return [r for r in rows if r.get("id") not in skip]


def label_ids():
    """Ids of the generated questions that say "first print the text `X`".

    A program that prints the text and the value on one line does what the words allow, but the expected output has
    them on two lines, so a model that was not taught the convention is marked down for it."""
    ids = set()
    for name in ("test", "test_unseen_table"):
        path = DATA / f"{name}.jsonl"
        if path.exists():
            for r in map(json.loads, open(path, encoding="utf-8")):
                if "first print the text" in r["task"]:
                    ids.add(r["id"])
    return ids


def label_split(rows):
    """(share on the questions without a label, share on those with one), over the generated questions that count."""
    labelled = label_ids()
    counted = [r for r in fair(rows) if r["set"] in ("gen", "unseen")]
    without = [r for r in counted if r["id"] not in labelled]
    with_label = [r for r in counted if r["id"] in labelled]
    return (share(without) if without else ""), (share(with_label) if with_label else "")


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


def table(runs, strict=False):
    """``runs``: list of (model, language, reasoning, rows). Returns the Markdown table. Questions that depend on
    tie-breaking are left out unless ``strict``."""
    lines = ["| " + " | ".join(HEAD) + " |", "|" + "---|" * len(HEAD)]
    for model, language, reasoning, rows in runs:
        s = summarize(rows if strict else fair(rows))
        cells = [model, language, reasoning] + [s[k] for k, _ in SETS]
        cells += [fmt(s["mean_tokens"], ""), fmt(s["per_correct"], "none correct")]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv):
    strict = "--strict" in argv
    runs = []
    for arg in [a for a in argv if a != "--strict"]:
        label, _, spec = arg.partition("=")
        parts = label.split("|")
        if len(parts) != 3 or not spec or not all(pathlib.Path(p).exists() for p in spec.split("+")):
            sys.exit(f"expected model|language|reasoning=path[+path...] with existing files, got: {arg}")
        runs.append((*parts, load(spec)))
    print(table(runs, strict))


if __name__ == "__main__":
    main(sys.argv[1:])
