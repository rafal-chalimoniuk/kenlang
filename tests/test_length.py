"""The program-length comparison reads all 21 reference solutions and reports honest totals."""
from benchmarks import length


def words(text):
    return len(text.split())


def test_every_reference_solution_exists():
    for version, _, _ in length.VERSIONS:
        for task in length.TASKS:
            assert length.source(version, task), (version, task)


def test_measure_covers_every_version_and_task():
    measured = length.measure(words)
    assert set(measured) == {"Ken", "Python", "pandas"}
    for rows in measured.values():
        assert len(rows) == 7
        assert all(tokens > 0 and characters > 0 and lines > 0 for tokens, characters, lines in rows)


def test_without_a_tokenizer_only_characters_and_lines_are_counted():
    measured = length.measure()
    assert all(tokens is None for rows in measured.values() for tokens, _, _ in rows)
    text = length.table(measured, with_tokens=False)
    assert "Ken characters" in text and "tokens" not in text


def test_table_totals_and_ratios_are_computed_from_the_rows():
    measured = {"Ken": [(10, 1, 1)] * 7, "Python": [(30, 1, 1)] * 7, "pandas": [(20, 1, 1)] * 7}
    lines = length.table(measured, with_tokens=True).splitlines()
    assert lines[0] == "| Task | Ken tokens | Python tokens | pandas tokens |"
    assert lines[-2] == "| **All 7** | **70** | **210** | **140** |"
    assert lines[-1] == "| **Compared with Ken** | 1.0× | 3.0× | 2.0× |"
