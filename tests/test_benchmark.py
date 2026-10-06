"""The benchmark must be self-consistent: reference solutions agree with independent pandas results."""
import json
import pathlib
import subprocess
import sys

import pytest

from benchmarks.expected import expected, tasks
from benchmarks.scoring import check_csv_spec, run_in_process, same_line, same_out, scratch_dir

BENCH = pathlib.Path(__file__).resolve().parent.parent / "benchmarks"


def test_task_file_is_well_formed():
    ts = tasks()
    assert [t["id"] for t in ts] == list(range(1, 8))
    for t in ts:
        assert t["task"].startswith("Write a Ken program that")
        assert t["title"].strip() and len(t["title"]) < 90   # a plain-language title for readers
        assert isinstance(t["seen_in_training"], bool)
        assert (BENCH / "reference" / f"task{t['id']}.ken").exists()


@pytest.mark.parametrize("task_id", range(1, 8))
def test_reference_solution_matches_pandas(task_id):
    src = (BENCH / "reference" / f"task{task_id}.ken").read_text(encoding="utf-8")
    exp, spec = expected(task_id)
    with scratch_dir() as work:
        ok, out = run_in_process(src, work)
        assert ok, out
        assert same_out(out, exp), out
        if spec:
            assert check_csv_spec(work / spec["file"], spec) == (True, "")


@pytest.mark.parametrize("flavour", ["python", "pandas"])
@pytest.mark.parametrize("task_id", range(1, 8))
def test_python_and_pandas_solutions_give_the_same_answers(flavour, task_id):
    """The Python and pandas versions of every task, used to compare program length, are as correct as the Ken ones."""
    script = BENCH / "reference" / flavour / f"task{task_id}.py"
    exp, spec = expected(task_id)
    with scratch_dir() as work:
        run = subprocess.run([sys.executable, str(script)], cwd=work, capture_output=True, text=True, timeout=120)
        assert run.returncode == 0, run.stderr
        assert same_out(run.stdout, exp), run.stdout
        if spec:
            assert check_csv_spec(work / spec["file"], spec) == (True, "")


class TestScoring:
    def test_numbers_compare_with_tolerance(self):
        assert same_line("a: 71089", "a: 71089.0")
        assert same_line("x 1.004", "x 1.0")
        assert not same_line("x 1.5", "x 1.0")
        assert not same_line("a: 1", "b: 1")

    def test_same_out_ignores_blank_lines_and_edges(self):
        assert same_out("\n1\n\n2\n", "1\n2")
        assert not same_out("1\n2\n3", "1\n2")

    def test_csv_spec_detects_wrong_order_and_rows(self, tmp_path):
        spec = dict(header=["a"], rows=[["1"], ["2"]], sort_col=0, desc=False)
        good = tmp_path / "g.csv"
        good.write_text("a\n1\n2\n", encoding="utf-8")
        assert check_csv_spec(good, spec) == (True, "")
        bad = tmp_path / "b.csv"
        bad.write_text("a\n2\n1\n", encoding="utf-8")
        assert check_csv_spec(bad, spec) == (False, "wrong order")
        other = tmp_path / "o.csv"
        other.write_text("a\n1\n3\n", encoding="utf-8")
        assert check_csv_spec(other, spec) == (False, "different rows")
        assert not check_csv_spec(tmp_path / "missing.csv", spec)[0]
        assert json.dumps(spec)  # specs are serialisable
