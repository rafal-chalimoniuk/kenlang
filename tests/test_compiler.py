import re

import pytest

import kenlang
from kenlang import runtime
from kenlang.compiler import ARITY, PYNAME, Parser, guess_arity, literal, split_arity


def py(src):
    return kenlang.translate(src)


class TestTokenizing:
    def test_comments_are_dropped(self):
        p = Parser('1 # a comment with "quotes" and [brackets]\n2')
        assert p.t == ["1", "2"]

    def test_quoted_text_keeps_spaces_and_hash(self):
        p = Parser('"two words # not a comment" print')
        assert p.t == ['"two words # not a comment"', "print"]

    def test_brackets_are_separate_tokens(self):
        assert Parser("map [ add 1 ]").t == ["map", "[", "add", "1", "]"]

    @pytest.mark.parametrize("text,value", [
        ('"hi"', "hi"), ("null", None), ("3", 3), ("2.5", 2.5), ("abc", "abc"), ("2026-03", "2026-03"),
        ('"a\\nb"', "a\nb"),
    ])
    def test_literal(self, text, value):
        assert literal(text) == value

    def test_split_arity(self):
        assert split_arity("np.percentile:1") == ("np.percentile", 1)
        assert split_arity("np.std") == ("np.std", None)


class TestParsing:
    def test_command_consumes_its_arity(self):
        code = py("csv orders.csv where status eq done count")
        assert "rt.where(_, V, 'status', 'eq', 'done')" in code
        assert code.rstrip().endswith("_ = rt.count(_, V)")

    def test_zero_arity_commands_take_nothing(self):
        assert py("1 2 sum").count("rt.") == 1

    def test_blocks_become_functions(self):
        code = py("map [ add 1 mul 2 ]")
        assert "def _b1(_, V):" in code
        assert "rt.add(_, V, 1)" in code and "return _" in code

    def test_nested_blocks(self):
        code = py("map [ if [ ge 1 ] [ add 1 ] [ add 2 ] ]")
        assert code.count("def _b") == 4

    def test_as_stores_in_variable_table(self):
        assert "V['x'] = _" in py("5 as x")

    def test_def_defines_callable_word(self):
        code = py("def double [ mul 2 ] 4 double")
        assert "def t_double(_, V):" in code and "_ = t_double(_, V)" in code

    def test_bare_word_is_variable_or_text(self):
        assert "rt.load(V, 'abc')" in py("abc")

    def test_use_adds_import_and_alias(self):
        code = py("use statistics use numpy as np")
        assert "import statistics" in code and "import numpy as np" in code

    def test_library_call_arity_from_signature(self):
        assert guess_arity(__import__("statistics"), "statistics.median") == 0
        assert "statistics.median(_)" in py("use statistics 1 statistics.median")

    def test_explicit_arity_and_keywords(self):
        code = py("use numpy as np 1 np.percentile:1 90")
        assert "np.percentile(_, rt.arg(V, 90))" in code or "np.percentile(_, 90)" in code
        code = py(".sort_values:1 temp ascending=0")
        assert "rt.dot(_, V, 'sort_values', ['temp'], {'ascending': 0})" in code

    def test_method_without_arguments(self):
        assert "rt.dot(_, V, 'sum', [], {})" in py(".sum")

    def test_every_command_has_a_runtime_function(self):
        for name in ARITY:
            assert hasattr(runtime, PYNAME.get(name, name)), name

    def test_output_is_valid_python(self):
        compile(py('csv orders.csv where status eq done set rev [ qty mul price ] by category [ each rev sum ] print'),
                "<test>", "exec")


class TestErrors:
    @pytest.mark.parametrize("src,fragment", [
        ("map [ add 1", "missing closing ]"),
        ("1 ]", "unexpected ]"),
        ("where status eq", "unexpected end of program"),
        ("def f mul 2", "expected ["),
        ("def", "unexpected end of program"),
        ("as", "unexpected end of program"),
        ("use no_such_module_xyz", "no_such_module_xyz"),
    ])
    def test_syntax_errors(self, src, fragment):
        with pytest.raises(kenlang.KenSyntaxError, match=re.escape(fragment)):
            kenlang.translate(src)

    def test_error_is_a_syntax_error(self):
        assert issubclass(kenlang.KenSyntaxError, SyntaxError)
