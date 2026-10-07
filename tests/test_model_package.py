"""The Hugging Face packaging scripts, exercised on synthetic results (no GPU, no network)."""
import json
import sys

import pytest

from benchmarks import compare
from benchmarks.expected import tasks
from model import assemble, upload


@pytest.fixture
def results(tmp_path):
    """A fake training/out folder: an adapter, a training log/config and two evaluation files."""
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"\0" * 2_000_000)
    (adapter / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": "unsloth/gemma-4-12b-it"}))
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        (adapter / name).write_text("{}")

    def rows(ok_task, tokens):
        out = []
        for t in range(1, 8):
            out += [dict(id=f"bench{t}", set="bench", attempt=k, ok=t in ok_task, tokens=tokens, sec=1.0, note="")
                    for k in (1, 2, 3)]
        out += [dict(id=f"test{i}", set="gen", attempt=k, ok=bool(ok_task), tokens=tokens, sec=1.0, note="")
                for i in range(4) for k in (1, 2, 3)]
        out += [dict(id=f"unseen{i}", set="unseen", attempt=k, ok=i == 0 and bool(ok_task), tokens=tokens, sec=1.0,
                     note="") for i in range(2) for k in (1, 2, 3)]
        return out
    for tag, ok, toks in (("base", set(), 120), ("finetuned", {1, 2, 4, 6, 7}, 70)):
        (tmp_path / f"eval_{tag}.jsonl").write_text("\n".join(json.dumps(r) for r in rows(ok, toks)))
    (tmp_path / "train_log.json").write_text(json.dumps(
        [{"loss": 0.9, "step": 5}]
        + [{"loss": v, "step": 140 + 5 * i} for i, v in enumerate([0.04, 0.03, 0.02, 0.02, 0.03])]))
    (tmp_path / "train_config.json").write_text(json.dumps({
        "epochs": 1.0, "learning_rate": 0.0002, "lora_rank": 16, "gradient_accumulation_steps": 8,
        "per_device_batch_size": 1, "train_examples": 1400, "optimizer": "adamw_8bit",
        "train_tables": ["bookings", "orders"],
        "libraries": {"torch": "2.11.0", "transformers": "5.17.0"}}))
    return tmp_path


def run_assemble(monkeypatch, results, out, *extra, ambiguous=frozenset()):
    no_qwen = str(results / "no_qwen_results")      # the real benchmarks/results folder must not leak into tests
    no_length = str(results / "no_length.json")
    monkeypatch.setattr(compare, "ambiguous_ids", lambda: set(ambiguous))   # nor the real list of tied questions
    monkeypatch.setattr(sys, "argv", ["assemble", "--results", str(results), "--out", str(out),
                                      "--qwen-results", no_qwen, "--length-json", no_length,
                                      "--repo-id", "someone/model", "--repo-url", "https://example.org/tok", *extra])
    assemble.main()


def test_assemble_builds_a_complete_folder(monkeypatch, results, tmp_path):
    dist = tmp_path / "dist"
    run_assemble(monkeypatch, results, dist)
    names = {f.name for f in dist.rglob("*") if f.is_file()}
    assert {"README.md", "adapter_model.safetensors", "adapter_config.json", "kenlang_reference.md",
            "training_config.json", "base.jsonl", "finetuned.jsonl"} <= names
    card = (dist / "README.md").read_text(encoding="utf-8")
    assert "{{" not in card
    assert card.startswith("---\nbase_model: unsloth/gemma-4-12b-it\nlibrary_name: peft\nlicense: apache-2.0")
    assert "someone/model" in card and "https://example.org/tok" in card


def test_card_numbers_are_computed_from_the_results(monkeypatch, results, tmp_path):
    dist = tmp_path / "dist"
    run_assemble(monkeypatch, results, dist)
    card = (dist / "README.md").read_text(encoding="utf-8")
    assert "| 0/21 (0%) | **15/21 (71%)** |" in card                      # 5 of 7 tasks solved 3/3
    assert "| 0/12 (0%) | **12/12 (100%)** |" in card                      # generated set
    assert "| 0/6 (0%) | **3/6 (50%)** |" in card                         # unseen table: 1 of 2 questions
    flat = " ".join(card.split())                       # the card wraps its lines
    assert "Mean answer length over all three sets: 120 tokens for Gemma 4 12B alone, 70 tokens with the" in flat
    assert "last five logged steps was 0.028" in card
    assert "about one of three tables: `bookings`, `orders`" in card
    titles = {t["id"]: t["title"] for t in tasks()}
    assert f"| 3 | {titles[3]} | 0/3 | 0/3 |" in card
    assert f"| 1 | {titles[1]} | 0/3 | 3/3 |" in card


def test_comparison_names_model_language_and_reasoning(monkeypatch, results, tmp_path):
    dist = tmp_path / "dist"
    run_assemble(monkeypatch, results, dist)
    card = (dist / "README.md").read_text(encoding="utf-8")
    assert "| Model | Language | Reasoning | Hand-written tasks |" in card
    assert "| Gemma 4 12B (4-bit) | Ken | off | 0/21 (0%) |" in card
    assert "| Gemma 4 12B (4-bit) + this adapter | Ken | off | 15/21 (71%) |" in card
    assert "| Qwen3.8-27B (Q5_K_XL) |" not in card                       # no Qwen rows without Qwen results
    section = card.split("## Tokens per correct answer")[1].split("## Training")[0]
    assert "Qwen" not in section                                           # and no sentence about Qwen either


def test_qwen_rows_are_added_from_result_files(monkeypatch, results, tmp_path):
    qwen = tmp_path / "qwen"
    qwen.mkdir()

    def write(name, language, reasoning, oks, tokens):
        rows = [dict(lang=language, think=reasoning, set="bench", id=f"bench{i}", ok=ok, tokens=tokens)
                for i, ok in enumerate(oks)]
        (qwen / name).write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    write("qwen_ken_on_bench.jsonl", "ken", "on", [True] * 7, 9000)
    write("qwen_python_on_bench.jsonl", "py", "on", [True] * 7, 4000)
    write("qwen_ken_off_bench.jsonl", "ken", "off", [True, False, False, False, False, False, False], 50)
    write("qwen_python_off_bench.jsonl", "py", "off", [True] * 7, 120)
    dist = tmp_path / "dist"
    run_assemble(monkeypatch, results, dist, "--qwen-results", str(qwen))
    card = (dist / "README.md").read_text(encoding="utf-8")
    assert "| Qwen3.8-27B (Q5_K_XL) | Ken | on | 7/7 (100%) |" in card
    assert "| Qwen3.8-27B (Q5_K_XL) | Python | on | 7/7 (100%) |" in card
    assert "| Qwen3.8-27B (Q5_K_XL) | Ken | off | 1/7 (14%) |" in card
    assert "| Qwen3.8-27B (Q5_K_XL) | Python | off | 7/7 (100%) |" in card
    assert ("spends about 9,000 tokens per correct Ken answer; this adapter spends about 98. "
            "That is 91.8 times the tokens") in card         # 21 attempts x 70 tokens for 15 correct
    assert ("writes 50 tokens per answer in Ken and 120 in Python, and gets 1/7 (14%) right in Ken "
            "against 7/7 (100%)") in card
    assert ("needs about 120 tokens per correct answer with reasoning off (4,000 with reasoning on). "
            "This adapter needs about 98 for Ken, which is 1.2 times shorter than the Python answer.") in card
    assert (dist / "evaluation" / "qwen_ken_on_bench.jsonl").exists()


def test_questions_that_depend_on_tie_breaking_are_not_counted_and_the_card_says_so(monkeypatch, results, tmp_path):
    dist = tmp_path / "dist"
    run_assemble(monkeypatch, results, dist, ambiguous={"test0", "unseen1"})
    card = " ".join((dist / "README.md").read_text(encoding="utf-8").split())
    assert "1 of the 4 generated questions and 1 of the 2 questions about the unseen table are not counted" in card
    assert "| 0/9 (0%) | **9/9 (100%)** |" in card          # generated set: 3 of 4 questions left, 3 attempts each
    # counting every question the adapter gets all 12 generated attempts and 3 of 6 unseen-table attempts
    assert "Counting them, the adapter gets 12/12 (100%) on the generated questions and 3/6 (50%) on the unseen" in card
    assert (dist / "evaluation" / "finetuned.jsonl").read_text(encoding="utf-8").count('"test0"') == 3  # raw data kept


def test_without_tied_questions_the_card_has_no_such_sentence(monkeypatch, results, tmp_path):
    dist = tmp_path / "dist"
    run_assemble(monkeypatch, results, dist)
    assert "not counted" not in (dist / "README.md").read_text(encoding="utf-8")


def test_length_section_is_computed_from_the_numbers_file(monkeypatch, results, tmp_path):
    numbers = tmp_path / "length.json"
    row = [10, 100, 1]
    numbers.write_text(json.dumps({"tokenizer": "some/tokenizer", "measured": {
        "Ken": [row] * 7, "Python": [[30, 300, 3]] * 7, "pandas": [[20, 200, 2]] * 7}}), encoding="utf-8")
    dist = tmp_path / "dist"
    run_assemble(monkeypatch, results, dist, "--length-json", str(numbers))
    card = (dist / "README.md").read_text(encoding="utf-8")
    assert "## How long are the programs?" in card and "some/tokenizer" in card
    assert "| **All 7** | **70** | **210** | **140** |" in card
    assert "| **Compared with Ken** | 1.0× | 3.0× | 2.0× |" in card


def test_without_the_numbers_file_the_length_section_is_left_out(monkeypatch, results, tmp_path):
    dist = tmp_path / "dist"
    run_assemble(monkeypatch, results, dist)
    assert "How long are the programs" not in (dist / "README.md").read_text(encoding="utf-8")


def test_adapter_pointing_at_a_local_path_is_rejected(monkeypatch, results, tmp_path):
    (results / "adapter" / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": r"D:\models\gemma"}))
    with pytest.raises(SystemExit, match="local path"):
        run_assemble(monkeypatch, results, tmp_path / "dist")


def test_missing_adapter_file_is_rejected(monkeypatch, results, tmp_path):
    (results / "adapter" / "tokenizer.json").unlink()
    with pytest.raises(SystemExit, match="tokenizer.json"):
        run_assemble(monkeypatch, results, tmp_path / "dist")


def test_unfilled_placeholder_is_an_error():
    with pytest.raises(SystemExit, match="unfilled"):
        assemble.render("hello {{NAME}} {{OTHER}}", {"NAME": "x"})


def test_upload_is_a_dry_run_by_default(monkeypatch, results, tmp_path, capsys):
    dist = tmp_path / "dist"
    run_assemble(monkeypatch, results, dist)
    monkeypatch.setattr(sys, "argv", ["upload", "--repo-id", "someone/model", "--folder", str(dist)])
    upload.main()
    out = capsys.readouterr().out
    assert "Would upload" in out and "dry run" in out


def test_upload_refuses_placeholder_repo_id(monkeypatch, results, tmp_path):
    dist = tmp_path / "dist"
    run_assemble(monkeypatch, results, dist)
    monkeypatch.setattr(sys, "argv", ["upload", "--repo-id", "<your-username>/model", "--folder", str(dist), "--yes"])
    with pytest.raises(SystemExit, match="your-username"):
        upload.main()


def test_assemble_refuses_to_wipe_an_unrelated_folder(monkeypatch, results, tmp_path):
    precious = tmp_path / "precious"
    precious.mkdir()
    (precious / "thesis.docx").write_text("do not delete")
    with pytest.raises(SystemExit, match="refusing to delete"):
        run_assemble(monkeypatch, results, precious)
    assert (precious / "thesis.docx").exists()


def test_assemble_replaces_a_previous_assembly(monkeypatch, results, tmp_path):
    dist = tmp_path / "dist"
    run_assemble(monkeypatch, results, dist)
    (dist / "stale.txt").write_text("left over")
    run_assemble(monkeypatch, results, dist)
    assert not (dist / "stale.txt").exists() and (dist / "README.md").exists()


def test_default_output_is_outside_the_repository():
    repo = assemble.ROOT
    assert assemble.DEFAULT_OUT == repo.parent / "project_hf"
    assert repo not in assemble.DEFAULT_OUT.parents
    assert upload.DEFAULT_ROOT == assemble.DEFAULT_OUT


def test_each_model_gets_its_own_folder_named_like_its_repository(monkeypatch, results, tmp_path, capsys):
    """Without --out the package goes to <container>/<model name>, and upload finds it there without --folder."""
    container = tmp_path / "container"
    monkeypatch.setattr(assemble, "DEFAULT_OUT", container)
    monkeypatch.setattr(upload, "DEFAULT_ROOT", container)
    optional = ["--qwen-results", str(results / "none"), "--length-json", str(results / "no.json")]
    for name in ("model-v7", "model-v8"):
        monkeypatch.setattr(sys, "argv", ["assemble", "--results", str(results), "--repo-id", f"someone/{name}",
                                          *optional])
        assemble.main()
    # two versions side by side: the second did not overwrite the first
    assert (container / "model-v7" / "README.md").exists() and (container / "model-v8" / "README.md").exists()
    assert not (container / "README.md").exists()          # nothing is put directly into the container
    monkeypatch.setattr(sys, "argv", ["upload", "--repo-id", "someone/model-v7"])
    upload.main()                                          # a dry run: it must find the folder on its own
    assert "Would upload" in capsys.readouterr().out


def test_card_without_a_source_link(monkeypatch, results, tmp_path):
    dist = tmp_path / "dist"
    monkeypatch.setattr(sys, "argv", ["assemble", "--results", str(results), "--out", str(dist), "--repo-id", "a/b"])
    assemble.main()
    card = (dist / "README.md").read_text(encoding="utf-8")
    assert "{{" not in card
    assert "**Source code**" not in card and "[GitHub](" not in card   # no link without --repo-url
    assert "not on PyPI yet" in card


def test_card_shows_the_source_repository_prominently(monkeypatch, results, tmp_path):
    dist = tmp_path / "dist"
    run_assemble(monkeypatch, results, dist)  # passes --repo-url https://example.org/tok
    lines = (dist / "README.md").read_text(encoding="utf-8").split("# model", 1)[1].splitlines()
    assert any(line.startswith("**Source code**") and "(https://example.org/tok)" in line for line in lines[:4])


def test_card_reports_how_a_python_program_does_on_questions_that_ask_for_a_label(monkeypatch, results, tmp_path):
    monkeypatch.setattr(compare, "label_ids", lambda: {"test1"})
    qwen = tmp_path / "qwen_py"
    qwen.mkdir()
    rows = [dict(set="gen", id=f"test{i}", ok=i != 1, tokens=100) for i in range(4) for _ in range(3)]
    (qwen / "qwen_python_off_gen.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    dist = tmp_path / "dist"
    run_assemble(monkeypatch, results, dist, "--qwen-results", str(qwen))
    card = " ".join((dist / "README.md").read_text(encoding="utf-8").split())
    assert ("gets 9/9 (100%) on the questions without a label and 0/3 (0%) on those with one" in card)
    # the adapter's fake rows also cover the unseen table (one of its two questions is wrong): 12 of 15, and 3 of 3
    assert "The adapter was trained on that convention and gets 12/15 (80%) and 3/3 (100%)." in card


def test_untrained_gemma_in_python_is_added_when_its_results_exist(monkeypatch, results, tmp_path):
    monkeypatch.setattr(compare, "label_ids", lambda: {"test1"})
    qwen = tmp_path / "qwen_for_python"
    qwen.mkdir()
    rows = [dict(set="gen", id=f"test{i}", ok=True, tokens=100) for i in range(4) for _ in range(3)]
    (qwen / "qwen_python_off_gen.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    base_python = [dict(set="bench", id=f"bench{i}", ok=i != 1, tokens=200) for i in range(1, 8) for _ in range(3)]
    base_python += [dict(set="gen", id=f"test{i}", ok=i != 1, tokens=200) for i in range(4) for _ in range(3)]
    (results / "eval_base_python.jsonl").write_text("\n".join(json.dumps(r) for r in base_python), encoding="utf-8")
    dist = tmp_path / "dist"
    run_assemble(monkeypatch, results, dist, "--qwen-results", str(qwen))
    card = " ".join((dist / "README.md").read_text(encoding="utf-8").split())
    assert "| Gemma 4 12B (4-bit) | Python | off | 18/21 (86%) | 9/12 (75%) |" in card
    assert "Without any training, Gemma 4 12B (4-bit) writing Python gets 18/21 (86%) on the hand-written tasks" in card
    # the question with a label ("test1") is the one it gets wrong; the other three of the four are right
    assert "Untrained Gemma 4 12B (4-bit) in Python gets 9/9 (100%) and 0/3 (0%)." in card
    assert (dist / "evaluation" / "base_python.jsonl").exists()
    # the Python run sits between the untrained and the trained Ken rows
    order = [card.index(label) for label in ("| Gemma 4 12B (4-bit) | Ken |", "| Gemma 4 12B (4-bit) | Python |",
                                              "| Gemma 4 12B (4-bit) + this adapter |")]
    assert order == sorted(order)


def test_without_the_gemma_python_results_the_card_does_not_mention_them(monkeypatch, results, tmp_path):
    dist = tmp_path / "dist"
    run_assemble(monkeypatch, results, dist)
    card = (dist / "README.md").read_text(encoding="utf-8")
    assert "Without any training" not in card and not (dist / "evaluation" / "base_python.jsonl").exists()
