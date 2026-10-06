"""The documentation must stay in sync with the implementation."""
import pathlib
import re

import pytest

import kenlang
from kenlang.compiler import ARITY
from kenlang.llm import build_prompt, extract_code, reference

ROOT = pathlib.Path(__file__).resolve().parent.parent
SPEC = (ROOT / "docs" / "language-reference.md").read_text(encoding="utf-8")


@pytest.mark.parametrize("name", sorted(ARITY))
def test_every_command_is_documented(name):
    assert f"`{name}`" in SPEC, f"{name} missing from docs/language-reference.md"
    assert f"`{name}`" in reference(), f"{name} missing from src/kenlang/reference.md"


def test_documented_arities_match_the_compiler():
    """The LLM reference lists each command's arity as `name` (N) or in a group like `a` `b` (N)."""
    ref = reference()
    for name, arity in ARITY.items():
        row = next((line for line in ref.splitlines() if line.startswith("|") and f"`{name}`" in line), None)
        assert row is not None, name
        # find the text after the command name up to the next separator
        m = re.search(rf"`{re.escape(name)}`((?:\s*`[a-z]+`)*)\s*\((\d)\)", row)
        if m:
            assert int(m.group(2)) == arity, f"{name}: docs say {m.group(2)}, compiler says {arity}"


def readings_fixtures(tmp_path):
    (tmp_path / "notes.txt").write_text(
        "I saw a cat. The cat saw the dog. The dog saw I and the cat.", encoding="utf-8")
    (tmp_path / "readings.csv").write_text(
        "city,country,temp\nPrague,CZ,24\nHamburg,DE,19\nKrakow,PL,27\nBrno,CZ,22\nKrakow,PL,25\nMagdeburg,DE,21\n",
        encoding="utf-8")


def code_blocks(text):
    return re.findall(r"```\n(.*?)```", text, re.S)


@pytest.mark.parametrize("i", range(len(code_blocks(reference()))))
def test_examples_in_the_llm_reference_run(i, tmp_path, monkeypatch, capsys):
    pytest.importorskip("numpy")
    readings_fixtures(tmp_path)
    monkeypatch.chdir(tmp_path)
    kenlang.run(code_blocks(reference())[i])
    assert capsys.readouterr().out.strip()


def test_block_scoped_variables_do_not_leak(run):
    assert run("1 as x upto 2 map [ 5 as x ] drop null x print").strip() == "1"


def test_variable_shadows_text_argument(run):
    """The pitfall documented in the language reference: a variable wins over text."""
    shadowed = int(run("csv orders.csv as o o where product has o count print"))
    literal_fragment = int(run("csv orders.csv where product has o count print"))
    assert shadowed == 0          # `o` resolved to the table, which is not a fragment of any name
    assert literal_fragment > 0   # without the variable, `o` is the text "o"


def test_build_prompt_layout():
    prompt = build_prompt("Count the orders.", "The file `orders.csv` has columns a and b.")
    assert prompt.startswith("# The Ken language")
    assert "# Data" in prompt and "columns a and b" in prompt
    assert prompt.rstrip().endswith("Answer with code only.")
    assert "Count the orders." in prompt


@pytest.mark.parametrize("answer,code", [
    ("upto 3 print", "upto 3 print"),
    ("```\nupto 3 print\n```", "upto 3 print"),
    ("Here you go:\n```ken\nupto 3\nprint\n```\nDone.", "upto 3\nprint"),
])
def test_extract_code(answer, code):
    assert extract_code(answer) == code
