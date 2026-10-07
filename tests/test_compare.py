"""The comparison table is computed from result files, never typed in."""
import json

import pytest

from benchmarks import compare


def write(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return str(path)


def rows(set_name, oks, tokens):
    return [dict(set=set_name, ok=ok, tokens=t) for ok, t in zip(oks, tokens)]


def test_summary_counts_and_tokens():
    s = compare.summarize(rows("bench", [True, False, True, True], [40, 60, 40, 60]) + rows("gen", [True], [10]))
    assert s["bench"] == "3/4 (75%)" and s["gen"] == "1/1 (100%)" and s["unseen"] == ""
    assert s["mean_tokens"] == 50                   # mean over the benchmark rows only
    assert round(s["per_correct"]) == 67            # 200 tokens spent for 3 correct answers


def test_nothing_correct_is_reported_as_such():
    s = compare.summarize(rows("bench", [False, False], [5, 5]))
    assert s["per_correct"] is None
    line = compare.table([("M", "Ken", "off", rows("bench", [False], [5]))]).splitlines()[-1]
    assert line.endswith("| none correct |")


def test_runs_without_a_benchmark_use_all_rows():
    s = compare.summarize(rows("unseen", [True, True], [30, 50]))
    assert s["mean_tokens"] == 40 and s["per_correct"] == 40


def test_table_names_model_language_and_reasoning_in_every_row():
    t = compare.table([("Qwen3.8-27B", "Python", "on", rows("bench", [True], [9000])),
                       ("Gemma 4 12B", "Ken", "off", rows("bench", [False], [20]))]).splitlines()
    assert len(t) == 4 and t[0].startswith("| Model | Language | Reasoning | Hand-written tasks |")
    assert t[1] == "|" + "---|" * 8
    assert t[2].startswith("| Qwen3.8-27B | Python | on | 1/1 (100%) |") and "| 9,000 | 9,000 |" in t[2]
    assert t[3].startswith("| Gemma 4 12B | Ken | off | 0/1 (0%) |")


def test_several_files_can_be_joined(tmp_path, capsys):
    a = write(tmp_path / "a.jsonl", rows("bench", [True, True], [10, 10]))
    b = write(tmp_path / "b.jsonl", rows("unseen", [True, False], [20, 20]))
    compare.main([f"M|Ken|off={a}+{b}"])
    line = capsys.readouterr().out.splitlines()[-1]
    assert "2/2 (100%)" in line and "1/2 (50%)" in line


def test_a_bad_argument_is_an_error(tmp_path):
    with pytest.raises(SystemExit, match="expected model"):
        compare.main([f"M|Ken|off={tmp_path / 'nope.jsonl'}"])
    with pytest.raises(SystemExit, match="expected model"):
        compare.main(["only a label=x"])


def test_questions_that_depend_on_tie_breaking_are_left_out(monkeypatch):
    monkeypatch.setattr(compare, "ambiguous_ids", lambda: {"q2"})
    data = [dict(set="gen", id="q1", ok=True, tokens=10), dict(set="gen", id="q2", ok=False, tokens=10)]
    fair_row = compare.table([("M", "Python", "off", data)]).splitlines()[-1]
    strict_row = compare.table([("M", "Python", "off", data)], strict=True).splitlines()[-1]
    assert "1/1 (100%)" in fair_row and "1/2 (50%)" in strict_row


def test_the_generated_test_sets_mark_their_ambiguous_questions():
    ids = compare.ambiguous_ids()
    assert 0 < len(ids) < 100                     # some, but far from all, of the 155 generated test questions
    assert all(i.startswith(("test", "test_unseen_table")) for i in ids)


def test_label_split_separates_questions_that_ask_for_a_label_first(monkeypatch):
    monkeypatch.setattr(compare, "ambiguous_ids", lambda: {"q3"})
    monkeypatch.setattr(compare, "label_ids", lambda: {"q2", "q3"})
    data = [dict(set="gen", id="q1", ok=True, tokens=1), dict(set="gen", id="q2", ok=False, tokens=1),
            dict(set="unseen", id="q3", ok=False, tokens=1), dict(set="unseen", id="q4", ok=True, tokens=1),
            dict(set="bench", id="bench1", ok=False, tokens=1)]
    # q3 is dropped as a tie, the benchmark row is not a generated question
    assert compare.label_split(data) == ("2/2 (100%)", "0/1 (0%)")


def test_the_data_marks_questions_with_a_label():
    ids = compare.label_ids()
    assert 0 < len(ids) < 60
