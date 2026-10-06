import pathlib

import pytest

import kenlang

ROOT = pathlib.Path(__file__).resolve().parent.parent
ORDERS = ROOT / "examples" / "data" / "orders.csv"


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    """A scratch directory that is the current directory and holds a copy of orders.csv."""
    (tmp_path / "orders.csv").write_text(ORDERS.read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def run(capsys, workdir):
    """run("ken source") -> stdout text."""
    def _run(src):
        kenlang.run(src)
        return capsys.readouterr().out
    return _run
