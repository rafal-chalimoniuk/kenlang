"""Behaviour of every built-in command, on lists, dicts and pandas tables."""

import pytest


def lines(out):
    return out.strip().splitlines()


class TestSources:
    def test_upto_and_list(self, run):
        assert lines(run("upto 3 print")) == ["1", "2", "3"]
        assert lines(run("list 1,2.5,x print")) == ["1", "2.5", "x"]

    def test_read_lines_words(self, run, workdir):
        (workdir / "t.txt").write_text("a b\nc d", encoding="utf-8")
        assert lines(run("read t.txt lines print")) == ["a b", "c d"]
        assert lines(run("read t.txt words count print")) == ["4"]

    def test_json(self, run, workdir):
        (workdir / "d.json").write_text('[1, 2, 3]', encoding="utf-8")
        assert lines(run("d.json json sum print")) == ["6"]
        assert lines(run('"[4, 5]" json sum print')) == ["9"]

    def test_csv(self, run):
        assert lines(run("csv orders.csv count print")) == ["400"]


class TestSelecting:
    def test_where_comparisons(self, run):
        assert lines(run("csv orders.csv where status eq cancelled count print")) == ["19"]
        assert lines(run("csv orders.csv where qty gt 3 count print"))[0] == str(
            sum(1 for q in __import__("pandas").read_csv("orders.csv").qty if q > 3))

    def test_where_in_nin_has_on_table(self, run):
        n_in = int(run("csv orders.csv where category in books,clothing count print"))
        n_nin = int(run("csv orders.csv where category nin books,clothing count print"))
        assert n_in + n_nin == 400
        assert int(run("csv orders.csv where product has ea count print")) > 0

    def test_where_on_list_of_dicts(self, run):
        import pandas as pd
        head = pd.read_csv("orders.csv").head(40)
        out = run("csv orders.csv first 40 .to_dict:1 records where qty ge 2 count print")
        assert int(out) == int((head.qty >= 2).sum())

    def test_filter_and_drop(self, run):
        assert lines(run("upto 10 filter [ mod 2 eq 0 ] print")) == ["2", "4", "6", "8", "10"]
        assert lines(run("list 1,2,1,3 drop 1 print")) == ["2", "3"]

    def test_pick_and_get(self, run):
        assert lines(run("csv orders.csv first 1 pick date,qty .columns.tolist print")) == ["date", "qty"]
        assert lines(run("list 5,6,7 get 1 print")) == ["6"]

    def test_each_set_map(self, run):
        assert lines(run("upto 3 map [ mul 10 ] print")) == ["10", "20", "30"]
        out = run("csv orders.csv first 2 set total [ qty mul price ] each total print")
        assert len(lines(out)) == 2


class TestGrouping:
    def test_group_and_by_agree(self, run):
        a = run("csv orders.csv group category map [ each qty sum ] pairs sort 0 map [ fmt \"{0}={1}\" ] print")
        b = run("csv orders.csv by category [ each qty sum ] pairs sort 0 map [ fmt \"{0}={1}\" ] print")
        assert a == b and len(lines(a)) == 4

    def test_top_from_dict_and_list(self, run):
        import pandas as pd
        busiest = pd.read_csv("orders.csv").category.value_counts().idxmax()
        assert lines(run('csv orders.csv by category [ count ] top 1 map [ fmt "{0}" ] print')) == [busiest]
        assert lines(run("list 3,9,1 top 2 print")) == ["9", "3"]

    def test_pairs_keys_vals(self, run):
        out = run("csv orders.csv by status [ count ] keys sort it print")
        assert lines(out) == ["cancelled", "done", "returned"]
        assert int(run("csv orders.csv by status [ count ] vals sum print")) == 400

    def test_nuniq_and_uniq(self, run):
        assert int(run("csv orders.csv each customer nuniq print")) == 10
        assert lines(run("list 1,2,1,3,2 uniq print")) == ["1", "2", "3"]
        assert int(run("list 1,2,1,3,2 nuniq print")) == 3

    def test_freq(self, run):
        out = run("list a,b,a,c,a,b freq map [ fmt \"{0} {1}\" ] print")
        assert lines(out) == ["a 3", "b 2", "c 1"]


class TestOrdering:
    def test_sort_rsort_first_last_rev(self, run):
        assert lines(run("list 3,1,2 sort it print")) == ["1", "2", "3"]
        assert lines(run("list 3,1,2 rsort it print")) == ["3", "2", "1"]
        assert lines(run("upto 5 first 2 print")) == ["1", "2"]
        assert lines(run("upto 5 last 2 print")) == ["4", "5"]
        assert lines(run("upto 3 rev print")) == ["3", "2", "1"]

    def test_sort_table(self, run):
        out = run("csv orders.csv sort date first 1 each date print")
        assert lines(out)[0].endswith("2026-01-01")

    def test_slice(self, run):
        assert lines(run('list abcdef slice 1 3 print')) == ["bc"]
        assert lines(run("csv orders.csv first 1 each date slice 0 7 print"))[0].endswith("2026-02")


class TestAggregatesAndMath:
    def test_aggregates(self, run):
        assert lines(run("list 1,2,3,4 sum print")) == ["10"]
        assert lines(run("list 1,2,3,4 mean print")) == ["2.5"]
        assert lines(run("list 1,2,3,4 max print")) == ["4"]
        assert lines(run("list 1,2,3,4 min print")) == ["1"]
        assert lines(run("list 1,2,3,4 count print")) == ["4"]

    def test_arithmetic_on_scalars_lists_and_columns(self, run):
        assert lines(run("7 add 3 sub 1 mul 2 div 3 print")) == ["6"]
        assert lines(run("7 mod 4 print")) == ["3"]
        assert lines(run("2 pow 10 print")) == ["1024"]
        assert lines(run("upto 3 add 1 print")) == ["2", "3", "4"]
        doubled = int(run("csv orders.csv set v [ qty mul 2 ] each v sum print"))
        assert doubled == 2 * int(run("csv orders.csv each qty sum print"))

    def test_comparison_commands(self, run):
        assert lines(run("5 gt 3 print")) == ["True"]
        assert lines(run("upto 3 le 2 print")) == ["True", "True", "False"]

    def test_round(self, run):
        assert lines(run("3.14159 round 2 print")) == ["3.14"]
        assert lines(run("list 1.234,5.678 round 1 print")) == ["1.2", "5.7"]

    def test_float_printing_strips_zeros(self, run):
        assert lines(run("2.50 print")) == ["2.5"]
        assert lines(run("4 div 2 print")) == ["2"]


class TestText:
    def test_case_split_join(self, run):
        assert lines(run('"AbC" lower print')) == ["abc"]
        assert lines(run('"AbC" upper print')) == ["ABC"]
        assert lines(run('"a-b-c" split "-" print')) == ["a", "b", "c"]
        assert lines(run('list a,b,c join "+" print')) == ["a+b+c"]

    def test_fmt_positional_named_and_spec(self, run):
        assert lines(run('list x,2.5 fmt "{0}: {1}" print')) == ["x: 2.5"]
        assert lines(run('list x,2.5 fmt "{1:.2f}" print')) == ["2.50"]
        assert run('list x,3 fmt "|{1:>4}|" print').strip() == "|   3|"
        out = run('csv orders.csv first 1 .to_dict:1 records map [ fmt "{product} {qty}" ] print')
        assert lines(out) == ["hat 1"]


class TestControl:
    def test_if(self, run):
        assert lines(run('5 if [ gt 3 ] [ "big" ] [ "small" ] print')) == ["big"]
        assert lines(run('upto 3 map [ if [ gt 1 ] [ "y" ] [ "n" ] ] print')) == ["n", "y", "y"]

    def test_def_and_as(self, run):
        assert lines(run("def double [ mul 2 ] 21 double print")) == ["42"]
        assert lines(run("5 as x 9 print x print")) == ["9", "5"]

    def test_print_passes_value_on(self, run):
        assert lines(run("3 print add 1 print")) == ["3", "4"]

    def test_print_formats(self, run):
        out = run("csv orders.csv by status [ count ] print")
        assert set(lines(out)) == {"done: 351", "returned: 30", "cancelled: 19"}
        out = run("csv orders.csv first 2 pick date,qty print")
        assert "date" in out and "qty" in out

    def test_save_table_list_and_text(self, run, workdir):
        run("csv orders.csv first 3 pick date,qty save out.csv")
        assert (workdir / "out.csv").read_text(encoding="utf-8").splitlines()[0] == "date,qty"
        run("upto 3 save list.txt")
        assert (workdir / "list.txt").read_text(encoding="utf-8").splitlines() == ["1", "2", "3"]
        run("42 save one.txt")
        assert (workdir / "one.txt").read_text(encoding="utf-8") == "42"


class TestLibraries:
    def test_library_function(self, run):
        assert lines(run("use statistics list 1,2,9 statistics.median print")) == ["2"]

    def test_library_with_explicit_arity_and_alias(self, run):
        pytest.importorskip("numpy")
        assert lines(run("use numpy as np list 1,2,3,4,5 np.percentile:1 50 print")) == ["3"]

    def test_method_chain_with_keywords(self, run):
        out = run("csv orders.csv .sort_values:1 price ascending=0 .head:1 1 each price print")
        assert lines(out)[0].endswith("3499")
