import subprocess
import sys

import kenlang


def cli(*args, stdin=None):
    return subprocess.run([sys.executable, "-m", "kenlang", *args], input=stdin, capture_output=True,
                          text=True, encoding="utf-8")


def test_version():
    r = cli("--version")
    assert r.returncode == 0 and r.stdout.strip() == f"kenlang {kenlang.__version__}"


def test_run_expression():
    r = cli("run", "-e", "upto 3 sum print")
    assert r.returncode == 0 and r.stdout.strip() == "6"


def test_run_file(tmp_path):
    f = tmp_path / "p.ken"
    f.write_text('"hi" upper print', encoding="utf-8")
    r = cli("run", str(f))
    assert r.returncode == 0 and r.stdout.strip() == "HI"


def test_run_from_stdin():
    r = cli("run", "-", stdin="upto 2 print")
    assert r.returncode == 0 and r.stdout.split() == ["1", "2"]


def test_py_prints_python():
    r = cli("py", "-e", "upto 3 print")
    assert r.returncode == 0
    assert "import kenlang.runtime as rt" in r.stdout and "rt.upto(_, V, 3)" in r.stdout
    compile(r.stdout, "<generated>", "exec")


def test_syntax_error_exit_code():
    r = cli("run", "-e", "map [ add 1")
    assert r.returncode == 2 and "syntax error" in r.stderr and "missing closing ]" in r.stderr


def test_missing_file_exit_code(tmp_path):
    r = cli("run", str(tmp_path / "nope.ken"))
    assert r.returncode == 1 and "nope.ken" in r.stderr


def test_no_command_is_an_error():
    assert cli().returncode == 2
