"""The committed training data must be valid, leak-free and consistent with the runtime."""
import json
import pathlib
import re

import pandas as pd
import pytest

from benchmarks.expected import expected
from benchmarks.scoring import check_csv_spec, run_in_process, same_out, scratch_dir
from training.tables import HELD_OUT_TABLE, TABLES, TRAIN_TABLES, expected_text

DATA = pathlib.Path(__file__).resolve().parent.parent / "training" / "data"
SETS = ("train", "val", "test", "test_unseen_table")


def load(name):
    return [json.loads(line) for line in open(DATA / f"{name}.jsonl", encoding="utf-8")]


def blocks(program):
    return re.findall(r"\[(.*?)\]", program)


@pytest.fixture(scope="module")
def splits():
    return {s: load(s) for s in SETS}


def test_sizes(splits):
    assert len(splits["train"]) >= 1500
    assert len(splits["val"]) >= 50 and len(splits["test"]) >= 50 and len(splits["test_unseen_table"]) >= 50


def test_splits_do_not_overlap(splits):
    """No task text and no program is shared between sets (the same program worded differently would leak)."""
    names = list(SETS)
    for key in ("task", "program"):
        seen = {s: {r[key] for r in rows} for s, rows in splits.items()}
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                assert not seen[a] & seen[b], (key, a, b)
        for s, rows in splits.items():
            assert len(seen[s]) == len(rows), f"duplicate {key} inside {s}"


def test_the_table_that_is_held_out_never_appears_in_training(splits):
    for split in ("train", "val", "test"):
        assert {r["table"] for r in splits[split]} <= set(TRAIN_TABLES)
        assert not any(HELD_OUT_TABLE in r["program"] for r in splits[split])
    assert {r["table"] for r in splits["test_unseen_table"]} == {HELD_OUT_TABLE}
    assert all(HELD_OUT_TABLE + ".csv" in r["program"] for r in splits["test_unseen_table"])


def test_every_training_table_is_used(splits):
    assert {r["table"] for r in splits["train"]} == set(TRAIN_TABLES)


def test_constructs_that_v01_left_out_are_now_taught(splits):
    """`if`, `def` and `as` inside a block each appear in a meaningful share of the training programs."""
    train = [r["program"] for r in splits["train"]]
    with_if = sum(bool(re.search(r"(^|\s)if\s", p)) for p in train)
    with_def = sum(bool(re.search(r"(^|\s)def\s", p)) for p in train)
    with_as = sum(any(" as " in f" {b} " for b in blocks(p)) for p in train)
    for name, n in (("if", with_if), ("def", with_def), ("as in a block", with_as)):
        assert n >= 0.08 * len(train), f"{name}: only {n} of {len(train)} programs"


def test_no_example_reproduces_a_benchmark_answer(splits):
    answers = [expected(i)[0] for i in range(1, 8)]
    for rows in splits.values():
        for r in rows:
            exp = "\n".join(r["expected"]) if isinstance(r["expected"], list) else r["expected"]
            assert not any(same_out(exp, a) for a in answers), r["task"]


def test_tasks_are_english_and_well_formed(splits):
    for rows in splits.values():
        for r in rows[:200]:
            assert r["task"].startswith("Write a Ken program that")
            assert not re.search(r"[ąćęłńóśźż]", r["task"] + r["program"], re.I)
            assert r["table"] in TABLES and f"{TABLES[r['table']].file}" in r["program"]


def test_a_program_never_names_a_variable_like_a_command(splits):
    """`as sub` once slipped in: `sub` is the subtraction command, so the program could not be parsed."""
    from kenlang.compiler import ARITY
    for r in splits["train"]:
        for name in re.findall(r"\bas (\w+)", r["program"]):
            assert name not in ARITY or name == "o", r["program"]
        for name in re.findall(r"\bdef (\w+)", r["program"]):
            assert name not in ARITY, r["program"]


@pytest.mark.parametrize("split,count", [("train", 120), ("val", 40), ("test", 40), ("test_unseen_table", 40)])
def test_programs_reproduce_expected_output(splits, split, count):
    """Re-run a sample: the stored expectation was computed independently with pandas."""
    rows = splits[split]
    step = max(len(rows) // count, 1)
    with scratch_dir(*(t.csv for t in TABLES.values())) as work:
        keep = {t.file for t in TABLES.values()} | {"orders.csv"}
        for r in rows[::step][:count]:
            for f in work.iterdir():
                if f.name not in keep:
                    f.unlink()
            ok, out = run_in_process(r["program"], work)
            assert ok, (r["program"], out)
            exp = r["expected"] if isinstance(r["expected"], str) else "\n".join(r["expected"])
            assert same_out(out, exp), r["program"]
            if r["csv"]:
                assert check_csv_spec(work / r["csv"]["file"], r["csv"])[0], r["program"]


@pytest.mark.parametrize("key", [k for k in TABLES if k != "orders"])
def test_generated_tables_match_their_generator(key):
    """The committed CSV files are exactly what ``training/tables.py`` produces."""
    assert TABLES[key].csv.read_text(encoding="utf-8").replace("\r\n", "\n") == expected_text(key)


@pytest.mark.parametrize("key", list(TABLES))
def test_table_descriptions_match_the_files(key):
    t = TABLES[key]
    header = list(pd.read_csv(t.csv, nrows=0).columns)
    assert list(t.cols) == header
    described = re.findall(r"^- `(\w+)`", t.description, re.M)
    assert described == header
    for column in (t.date, t.state, t.name_col, *t.dims, *(n.name for n in (t.qty, t.price, t.stat) if n)):
        assert column in header
    assert set(t.frame()[t.state].unique()) == set(t.states)
