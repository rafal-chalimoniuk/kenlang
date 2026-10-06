"""How long are the reference solutions in Ken, plain Python and pandas?

    python -m benchmarks.length                          # tokens (Gemma 4 tokenizer), characters and lines
    python -m benchmarks.length --tokenizer none         # characters and lines only, no `transformers` needed

All three versions of each of the seven benchmark tasks give the same answers (``tests/test_benchmark.py`` runs
them). The Python and pandas versions are written plainly, the way a competent programmer or model would write them:
neither golfed nor padded. Tokens are counted with the tokenizer of the model that was trained on Ken, so the figure
is what a model of that family pays to write the program.
"""
import argparse
import json
import pathlib

REFERENCE = pathlib.Path(__file__).resolve().parent / "reference"
VERSIONS = (("Ken", "task{}.ken", REFERENCE), ("Python", "task{}.py", REFERENCE / "python"),
            ("pandas", "task{}.py", REFERENCE / "pandas"))
TASKS = range(1, 8)


def source(version, task):
    _, pattern, folder = next(v for v in VERSIONS if v[0] == version)
    return (folder / pattern.format(task)).read_text(encoding="utf-8").strip()


def measure(count_tokens=None):
    """``{version: [(tokens or None, characters, lines) for each task]}``."""
    out = {}
    for version, _, _ in VERSIONS:
        rows = []
        for task in TASKS:
            text = source(version, task)
            rows.append((count_tokens(text) if count_tokens else None, len(text), len(text.splitlines())))
        out[version] = rows
    return out


def table(measured, with_tokens):
    """A Markdown table with one row per task and a total, plus how many times longer Python and pandas are."""
    head = ["Task"]
    for version, _, _ in VERSIONS:
        head += [f"{version} tokens"] if with_tokens else [f"{version} characters"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    index = 0 if with_tokens else 1
    totals = {version: 0 for version, _, _ in VERSIONS}
    for position, task in enumerate(TASKS):
        cells = [str(task)]
        for version, _, _ in VERSIONS:
            value = measured[version][position][index]
            totals[version] += value
            cells.append(f"{value:,}")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("| **All 7** | " + " | ".join(f"**{totals[v]:,}**" for v, _, _ in VERSIONS) + " |")
    ken = totals["Ken"]
    ratios = " | ".join(f"{totals[v] / ken:.1f}×" for v in ("Python", "pandas"))
    lines.append(f"| **Compared with Ken** | 1.0× | {ratios} |")
    return "\n".join(lines)


def load(path):
    """Read a file written by ``--json``: returns (tokenizer name, measured)."""
    data = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    return data["tokenizer"], {v: [tuple(row) for row in rows] for v, rows in data["measured"].items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="unsloth/gemma-4-12b-it", help="a Hugging Face tokenizer, or 'none'")
    ap.add_argument("--json", default="", help="also write the numbers to this file (used by model/assemble.py)")
    a = ap.parse_args()
    count = None
    if a.tokenizer != "none":
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.tokenizer)
        count = lambda text: len(tok(text, add_special_tokens=False)["input_ids"])  # noqa: E731
    measured = measure(count)
    print(table(measured, with_tokens=count is not None))
    lines = {v: sum(row[2] for row in rows) for v, rows in measured.items()}
    print("\nLines of code, all 7 tasks: " + ", ".join(f"{v} {n}" for v, n in lines.items()))
    if a.json:
        pathlib.Path(a.json).write_text(json.dumps({"tokenizer": a.tokenizer, "measured": measured}, indent=1),
                                        encoding="utf-8")


if __name__ == "__main__":
    main()
