"""Every example program must reproduce its golden output, and the numbers must be right."""
import pathlib
import shutil

import pandas as pd
import pytest

import kenlang

EXAMPLES = pathlib.Path(__file__).resolve().parent.parent / "examples"
PROGRAMS = sorted(p.stem for p in EXAMPLES.glob("*.ken"))


def run_example(name, tmp_path, capsys, monkeypatch):
    work = tmp_path / "examples"
    shutil.copytree(EXAMPLES, work)
    monkeypatch.chdir(work)
    kenlang.run((work / f"{name}.ken").read_text(encoding="utf-8"))
    return capsys.readouterr().out, work


@pytest.mark.parametrize("name", PROGRAMS)
def test_golden_output(name, tmp_path, capsys, monkeypatch):
    if name == "libraries":
        pytest.importorskip("scipy")
        pytest.importorskip("matplotlib")
    out, work = run_example(name, tmp_path, capsys, monkeypatch)
    assert out == (EXAMPLES / "expected" / f"{name}.out").read_text(encoding="utf-8")
    if name == "libraries":
        assert (work / "chart.png").stat().st_size > 0


def test_every_example_has_a_golden_file():
    assert PROGRAMS and all((EXAMPLES / "expected" / f"{n}.out").exists() for n in PROGRAMS)


def test_sales_report_numbers_match_pandas(tmp_path, capsys, monkeypatch):
    out, _ = run_example("sales_report", tmp_path, capsys, monkeypatch)
    df = pd.read_csv(EXAMPLES / "data" / "orders.csv")
    done = df[df.status == "done"]
    rev = done.qty * done.price
    lines = out.splitlines()
    assert float(lines[lines.index("Revenue:") + 1]) == pytest.approx(round(rev.sum(), 2))
    assert int(lines[lines.index("Completed orders:") + 1]) == len(done)
    returned_pct = round(100 * (df.status == "returned").mean(), 1)
    assert float(lines[lines.index("Returned %:") + 1]) == pytest.approx(returned_pct)
    by_cat = rev.groupby(done.category).sum().round(2).sort_values(ascending=False)
    start = lines.index("Revenue by category:") + 1
    assert lines[start:start + len(by_cat)] == [f"{k}: {v}" for k, v in by_cat.items()]
