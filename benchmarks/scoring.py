"""Scoring helpers shared by the benchmark, the data generator and the evaluation scripts."""
import contextlib
import csv
import io
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

NUM = re.compile(r"-?\d+(?:\.\d+)?")
ORDERS = pathlib.Path(__file__).resolve().parent.parent / "examples" / "data" / "orders.csv"


def same_line(a, b):
    """Lines match if they are equal after replacing numbers by a marker and the numbers agree
    within 0.01 (so ``71089`` equals ``71089.0``)."""
    na, nb = NUM.findall(a), NUM.findall(b)
    if NUM.sub("#", a.strip()) != NUM.sub("#", b.strip()) or len(na) != len(nb):
        return False
    return all(abs(float(x) - float(y)) < 0.01 for x, y in zip(na, nb))


def same_out(got, expected):
    g = [line for line in got.strip().splitlines() if line.strip()]
    e = [line for line in expected.strip().splitlines() if line.strip()]
    return len(g) == len(e) and all(same_line(x, y) for x, y in zip(g, e))


def check_csv_spec(path, spec):
    """``spec``: {header, rows, sort_col, desc}. Same header, same multiset of rows, and the sort
    column ordered as asked (ties may appear in any order). Returns (ok, note)."""
    path = pathlib.Path(path)
    if not path.exists():
        return False, f"missing file {path.name}"
    with open(path, encoding="utf-8", newline="") as fh:
        got = list(csv.reader(fh))
    if not got or got[0] != spec["header"]:
        return False, f"header {got[0] if got else None}"
    body = got[1:]
    if sorted(map(tuple, body)) != sorted(map(tuple, spec["rows"])):
        return False, "different rows"
    col = spec["sort_col"]

    def key(v):
        try:
            return (0, float(v), "")
        except ValueError:
            return (1, 0.0, v)
    keys = [key(r[col]) for r in body]
    ok = keys == sorted(keys, reverse=spec.get("desc", False))
    return ok, "" if ok else "wrong order"


def run_in_process(src, workdir):
    """Run trusted Ken source (reference solutions, generated data) in this process."""
    import kenlang
    old = os.getcwd()
    buf = io.StringIO()
    os.chdir(workdir)
    try:
        with contextlib.redirect_stdout(buf):
            kenlang.run(src)
        return True, buf.getvalue()
    except BaseException as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"
    finally:
        os.chdir(old)


def run_isolated(src, workdir, python=None, timeout=120):
    """Run untrusted Ken source (written by a model) in a separate process."""
    workdir = pathlib.Path(workdir)
    (workdir / "prog.ken").write_text(src, encoding="utf-8")
    r = subprocess.run([python or sys.executable, "-m", "kenlang", "run", "prog.ken"], cwd=workdir,
                       capture_output=True, text=True, encoding="utf-8", timeout=timeout,
                       env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"})
    if r.returncode != 0:
        return False, (r.stderr.strip().splitlines() or ["error"])[-1][:200]
    return True, r.stdout


@contextlib.contextmanager
def scratch_dir(*extra):
    """A temporary working directory that contains ``orders.csv`` and any ``extra`` files given."""
    work = pathlib.Path(tempfile.mkdtemp())
    for f in (ORDERS, *extra):
        shutil.copy(f, work)
    try:
        yield work
    finally:
        shutil.rmtree(work, ignore_errors=True)
