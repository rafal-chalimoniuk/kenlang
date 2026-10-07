"""Assemble the Hugging Face upload folder for the fine-tuned adapter.

    python -m model.assemble --repo-id <user>/kenlang-gemma-4-12b-it-lora-v2 --repo-url https://github.com/<user>/kenlang

Reads the adapter, training log/config and the evaluation results from ``training/out`` and writes a
ready-to-upload folder outside the repository: by default ``<repo>/../project_hf/<model name>``, where the model
name is the last part of ``--repo-id`` (the weights do not belong in git, and every model version gets its own
folder, named like its repository on the Hub).
Every number on the model card is computed from the result files; nothing is typed in by hand.
Nothing is uploaded: see ``model/upload.py``.
"""
import argparse
import json
import pathlib
import re
import shutil
import sys

from benchmarks import compare, length
from benchmarks.expected import tasks

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = ROOT / "training" / "out"
DEFAULT_OUT = ROOT.parent / "project_hf"
QWEN_RESULTS = ROOT / "benchmarks" / "results"
LENGTH_JSON = QWEN_RESULTS / "length.json"
GEMMA = "Gemma 4 12B (4-bit)"
QWEN = "Qwen3.8-27B (Q5_K_XL)"
ADAPTER_FILES = ["adapter_config.json", "adapter_model.safetensors", "tokenizer.json",
                 "tokenizer_config.json", "chat_template.jinja"]


def load_rows(results, tag):
    path = results / f"eval_{tag}.jsonl"
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def share(rows):
    ok = sum(r["ok"] for r in rows)
    return f"{ok}/{len(rows)} ({100 * ok / len(rows):.0f}%)" if rows else "n/a"


def check_adapter(adapter):
    missing = [f for f in ADAPTER_FILES if not (adapter / f).exists()]
    if missing:
        sys.exit(f"adapter folder {adapter} lacks: {', '.join(missing)}")
    if (adapter / "adapter_model.safetensors").stat().st_size < 1_000_000:
        sys.exit("adapter_model.safetensors is suspiciously small")
    cfg = json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))
    base = str(cfg.get("base_model_name_or_path", ""))
    if re.match(r"^([A-Za-z]:|/|\\\\)", base) or "\\" in base:
        sys.exit(f"adapter_config.json points at a local path ({base}); it must name a Hub repository")
    return cfg


def qwen_rows(folder, language, reasoning):
    """Rows of ``qwen_<language>_<reasoning>_<set>.jsonl`` files written by ``benchmarks/run_llama.py``."""
    rows = []
    for path in sorted(pathlib.Path(folder).glob(f"qwen_{language.lower()}_{reasoning}_*.jsonl")):
        rows += [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    return rows


def comparison(base, ft, qwen_folder, base_python=None):
    """The table that puts the adapter next to Qwen (if its results exist), and two sentences computed from it."""
    adapter_label = f"{GEMMA} + this adapter"
    runs = [(GEMMA, "Ken", "off", base), (adapter_label, "Ken", "off", ft)]
    if base_python:                         # the same model without any training, asked for Python
        runs.insert(1, (GEMMA, "Python", "off", base_python))
    for language, reasoning in (("Ken", "on"), ("Python", "on"), ("Ken", "off"), ("Python", "off")):
        rows = qwen_rows(qwen_folder, language, reasoning)
        if rows:
            runs.append((QWEN, language, reasoning, rows))
    stats = {(model, language, reasoning): compare.summarize(compare.fair(rows))
             for model, language, reasoning, rows in runs}
    adapter, notes = stats[(adapter_label, "Ken", "off")], []
    ken_on, ken_off, py_off = (stats.get((QWEN, "Ken", "on")), stats.get((QWEN, "Ken", "off")),
                               stats.get((QWEN, "Python", "off")))
    if ken_on and ken_on["per_correct"] and adapter["per_correct"]:
        notes.append(f"- With reasoning on, {QWEN} spends about {ken_on['per_correct']:,.0f} tokens per correct Ken "
                     f"answer; this adapter spends about {adapter['per_correct']:,.0f}. That is "
                     f"{ken_on['per_correct'] / adapter['per_correct']:.1f} times the tokens.")
    if ken_off and py_off and ken_off["mean_tokens"] and py_off["mean_tokens"]:
        notes.append(f"- With reasoning off, {QWEN} writes {ken_off['mean_tokens']:,.0f} tokens per answer in Ken and "
                     f"{py_off['mean_tokens']:,.0f} in Python, and gets {ken_off['bench']} right in Ken against "
                     f"{py_off['bench']} in Python on the hand-written tasks.")
    if py_off and py_off["per_correct"] and adapter["per_correct"]:
        py_on = stats.get((QWEN, "Python", "on"))
        with_reasoning = f" ({py_on['per_correct']:,.0f} with reasoning on)" if py_on and py_on["per_correct"] else ""
        notes.append(f"- In Python, {QWEN} needs about {py_off['per_correct']:,.0f} tokens per correct answer with "
                     f"reasoning off{with_reasoning}. This adapter needs about {adapter['per_correct']:,.0f} for Ken, "
                     f"which is {py_off['per_correct'] / adapter['per_correct']:.1f} times shorter than the Python "
                     "answer.")
    gemma_py = stats.get((GEMMA, "Python", "off"))
    if gemma_py and gemma_py["per_correct"] and adapter["per_correct"]:
        notes.append(
            f"- Without any training, {GEMMA} writing Python gets {gemma_py['bench']} on the hand-written tasks, "
            f"{gemma_py['gen']} on the generated ones and {gemma_py['unseen']} on the table it never saw, and needs "
            f"about {gemma_py['per_correct']:,.0f} tokens per correct answer. The adapter writing Ken gets "
            f"{adapter['bench']}, {adapter['gen']} and {adapter['unseen']} with about {adapter['per_correct']:,.0f}, "
            f"which is {gemma_py['per_correct'] / adapter['per_correct']:.1f} times shorter.")
    python_rows = next((rows for model, language, reasoning, rows in runs
                        if (model, language, reasoning) == (QWEN, "Python", "off")), None)
    if python_rows:
        py_plain, py_label = compare.label_split(python_rows)
        ad_plain, ad_label = compare.label_split(ft)
        gp_plain, gp_label = compare.label_split(base_python) if base_python else ("", "")
        if py_plain and py_label and ad_plain and ad_label:
            notes.append(
                "- Some generated questions ask to print a label first (\"first print the text ...\"). In "
                f"Python, {QWEN} gets {py_plain} on the questions without a label and {py_label} on those with one; "
                "in the cases I looked at it printed the label and the value on one line, where the questions expect "
                f"two. The adapter was trained on that convention and gets {ad_plain} and {ad_label}."
                + (f" Untrained {GEMMA} in Python gets {gp_plain} and {gp_label}." if gp_plain and gp_label else ""))
    table = compare.table(runs)
    prompt_note = ("- The prompt is about 1,800 tokens in every Ken run: the language reference is about 1,500 of "
                   "them. The comparison is about the answer, which is the slow and costly part.")
    if not any(run[0] == QWEN for run in runs):      # no Qwen results: say only what the table shows
        return ("Every token a model writes costs time and money. The table shows the mean answer length and the "
                "tokens spent per correct answer, both measured on the hand-written tasks.\n\n"
                f"{table}\n\n{prompt_note}")
    return ("Every token a model writes costs time and money, and a model that reasons first can write thousands of "
            f"tokens for one program. The table puts this adapter next to {QWEN.split(' (')[0]} (a "
            "27-billion-parameter model, 5-bit GGUF, run with llama.cpp, temperature 0.6). Qwen was given the same "
            "prompts: the Ken "
            "reference and the table description for the Ken rows, and the table description for the Python rows, "
            "where it was asked for a Python program that uses pandas. The two right-hand columns are measured on "
            f"the hand-written tasks.\n\n{table}\n\n" + "\n".join(notes) + f"\n\n{prompt_note}\n"
            "- The sets differ in size, and Qwen's runs with reasoning on cover only the hand-written tasks, where "
            f"each question is worth {100 / 7:.0f} percentage points; read small differences there with care.")


def length_section(path):
    """How long the reference solutions are in Ken, Python and pandas ('' if the numbers were not measured)."""
    if not pathlib.Path(path).exists():
        return ""
    tokenizer, measured = length.load(path)
    return ("## How long are the programs?\n\n"
            "A separate question from whether a model can write Ken is how much it has to write. The source "
            "repository holds three correct solutions of each of the seven hand-written tasks: Ken, plain Python "
            "(standard library only) and pandas. The tests run all of them against the same independent answers. "
            f"They are written plainly, neither shortened nor padded. Length in tokens of the {tokenizer} "
            f"tokenizer:\n\n{length.table(measured, with_tokens=True)}\n\n"
            "This compares correct programs, not what models write. I wrote the Python and pandas versions myself, "
            "so read the ratios as an estimate. The seven tasks are tasks on tables, so the ratios say nothing "
            "about other kinds of programs.\n")


def left_out(all_rows, kept, n_attempts):
    """The sentence that says how many questions are not counted because their correct output depends on ties."""
    parts = []
    for key, what in (("gen", "generated questions"), ("unseen", "questions about the unseen table")):
        total = sum(r["set"] == key for r in all_rows) // n_attempts
        left = total - sum(r["set"] == key for r in kept) // n_attempts
        if left:
            parts.append(f"{left} of the {total} {what}")
    if not parts:
        return ""
    return (" and ".join(parts) + " are not counted in the numbers on this card: the correct output of such a "
            "question depends on how ties between equal values are broken (for example the 3 products with the most "
            "items, when the third and fourth have the same number), and the question does not say. "
            "The complete results are in `evaluation/`. Counting them, the adapter gets "
            f"{share([r for r in all_rows if r['set'] == 'gen'])} on the generated questions and "
            f"{share([r for r in all_rows if r['set'] == 'unseen'])} on the unseen table; some of its wrong answers "
            "there are real mistakes, not tie-breaking.")


def prepare_output(dist):
    """Create ``dist`` empty. An earlier assembled folder is replaced; any other non-empty folder is refused."""
    if dist.exists() and any(dist.iterdir()):
        if not ((dist / "README.md").exists() and (dist / "adapter_config.json").exists()):
            sys.exit(f"{dist} exists and does not look like an assembled model folder; refusing to delete it")
        shutil.rmtree(dist)
    dist.mkdir(parents=True, exist_ok=True)


def source_banner(url):
    """A line under the card title that points to the source repository (empty without a URL)."""
    if not url:
        return ""
    label = url.rstrip("/").split("github.com/")[-1]
    return ("\n**Source code** — everything about the Ken language, and the tools used to train this model: "
            f"[{label}]({url})\n")


def render(template, values):
    for key, value in values.items():
        template = template.replace("{{" + key + "}}", str(value))
    left = re.findall(r"\{\{[A-Z_]+\}\}", template)
    if left:
        sys.exit(f"unfilled placeholders in the model card: {sorted(set(left))}")
    return template


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", default="<your-username>/kenlang-gemma-4-12b-it-lora-v2")
    ap.add_argument("--repo-url", default="", help="source repository URL; leave empty to omit the link")
    ap.add_argument("--results", default=str(OUT), help="folder with the adapter, logs and evaluation results")
    ap.add_argument("--adapter", default="", help="adapter folder (default: <results>/adapter)")
    ap.add_argument("--out", default="", help="folder to create (default: <DEFAULT_OUT>/<model name>)")
    ap.add_argument("--length-json", default=str(LENGTH_JSON),
                    help="numbers written by `python -m benchmarks.length --json`; the section is left out if absent")
    ap.add_argument("--qwen-results", default=str(QWEN_RESULTS),
                    help="folder with the Qwen runs (qwen_<language>_<on|off>_<set>.jsonl); absent runs are skipped")
    a = ap.parse_args()

    results = pathlib.Path(a.results)
    adapter = pathlib.Path(a.adapter or results / "adapter")
    dist = pathlib.Path(a.out) if a.out else DEFAULT_OUT / a.repo_id.split("/")[-1]
    check_adapter(adapter)
    all_base, all_ft = load_rows(results, "base"), load_rows(results, "finetuned")
    base, ft = compare.fair(all_base), compare.fair(all_ft)       # the rows the numbers on the card are made of
    cfg = json.loads((results / "train_config.json").read_text(encoding="utf-8"))
    log = json.loads((results / "train_log.json").read_text(encoding="utf-8"))
    losses = [e["loss"] for e in log if "loss" in e]
    final_loss = f"{sum(float(x) for x in losses[-5:]) / len(losses[-5:]):.3f}" if losses else "n/a"

    def subset(rows, name):
        return [r for r in rows if r["set"] == name]

    def per_task(rows, i):
        sel = [r for r in rows if r["id"] == f"bench{i}"]
        return f"{sum(r['ok'] for r in sel)}/{len(sel)}"

    task_rows = [f"| {t['id']} | {t['title']} | {per_task(base, t['id'])} | {per_task(ft, t['id'])} |"
                 for t in tasks()]
    n_attempts = len(subset(ft, "bench")) // 7
    mean = lambda rows: f"{sum(r['tokens'] for r in rows) / len(rows):.0f}"  # noqa: E731
    steps = max((e.get("step", 0) for e in log), default=0)
    python_file = results / "eval_base_python.jsonl"
    base_python = compare.fair(load_rows(results, "base_python")) if python_file.exists() else None
    section = comparison(base, ft, a.qwen_results, base_python)

    values = {
        "MODEL_NAME": a.repo_id.split("/")[-1], "REPO_ID": a.repo_id,
        "REPO_LINK": f" ([GitHub]({a.repo_url}))" if a.repo_url else "",
        "REPO_BANNER": source_banner(a.repo_url),
        "ATTEMPTS": n_attempts, "GEN_TASKS": len(subset(ft, "gen")) // n_attempts,
        "UNSEEN_TASKS": len(subset(ft, "unseen")) // n_attempts,
        "LEFT_OUT": left_out(all_ft, ft, n_attempts),
        "BASE_BENCH": share(subset(base, "bench")), "FT_BENCH": share(subset(ft, "bench")),
        "BASE_GEN": share(subset(base, "gen")), "FT_GEN": share(subset(ft, "gen")),
        "BASE_UNSEEN": share(subset(base, "unseen")), "FT_UNSEEN": share(subset(ft, "unseen")),
        "BASE_TOKENS": mean(base), "FT_TOKENS": mean(ft), "TASK_ROWS": "\n".join(task_rows),
        "COMPARISON_SECTION": section, "LENGTH_SECTION": length_section(a.length_json),
        "TRAIN_TABLES": ", ".join(f"`{k}`" for k in cfg.get("train_tables", [])),
        "TRAIN_EXAMPLES": cfg["train_examples"], "LORA_RANK": cfg["lora_rank"], "EPOCHS": cfg["epochs"],
        "STEPS": steps, "BATCH": cfg["gradient_accumulation_steps"] * cfg["per_device_batch_size"],
        "LR": cfg["learning_rate"], "OPTIMIZER": cfg["optimizer"], "FINAL_LOSS": final_loss,
        "LIBRARIES": ", ".join(f"{k} {v}" for k, v in cfg["libraries"].items()),
        "ONE_TASK": f"{100 / 7:.0f}",
    }
    card = render((HERE / "MODEL_CARD.template.md").read_text(encoding="utf-8"), values)

    prepare_output(dist)
    for f in ADAPTER_FILES:
        shutil.copy(adapter / f, dist / f)
    (dist / "README.md").write_text(card, encoding="utf-8")
    shutil.copy(ROOT / "src" / "kenlang" / "reference.md", dist / "kenlang_reference.md")
    shutil.copy(results / "train_config.json", dist / "training_config.json")
    shutil.copy(results / "train_log.json", dist / "training_log.json")
    (dist / "evaluation").mkdir()
    for tag in ("base", "finetuned"):
        shutil.copy(results / f"eval_{tag}.jsonl", dist / "evaluation" / f"{tag}.jsonl")
    if python_file.exists():
        shutil.copy(python_file, dist / "evaluation" / "base_python.jsonl")
    for path in sorted(pathlib.Path(a.qwen_results).glob("qwen_*.jsonl")):
        shutil.copy(path, dist / "evaluation" / path.name)
    total = sum(f.stat().st_size for f in dist.rglob("*") if f.is_file())
    print(f"assembled {dist} ({total / 1e6:.0f} MB)")
    for f in sorted(dist.rglob("*")):
        if f.is_file():
            print(f"  {f.relative_to(dist)}  {f.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
