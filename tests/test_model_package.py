"""The Hugging Face packaging scripts, exercised on synthetic results (no GPU, no network)."""
import json
import sys

import pytest

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


def run_assemble(monkeypatch, results, out, *extra):
    no_qwen = str(results / "no_qwen_results")      # the real benchmarks/results folder must not leak into tests
    no_length = str(results / "no_length.json")
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
    assert (dist / "evaluation" / "qwen_ken_on_bench.jsonl").exists()


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
