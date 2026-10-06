"""QLoRA fine-tuning of Gemma 4 12B on Ken, with Unsloth.

    python -m training.train                  # full run (1 epoch)
    python -m training.train --max-steps 5    # smoke test of the whole pipeline
    python -m training.train --dry            # print one training example, no GPU needed

The prompt is exactly ``kenlang.llm.build_prompt(task, data_description)`` rendered with the model's
chat template and reasoning switched off (the same as at evaluation time); the data description is the one of
the table the example is about (``training/tables.py``). The loss is computed on the answer only: the program
followed by the end-of-turn token.
"""
import argparse
import json
import pathlib

from kenlang.llm import build_prompt

from .tables import TABLES

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
DATA_DESCRIPTION = TABLES["orders"].description      # the table of the benchmark tasks


def read_jsonl(path):
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def render_prompt(tokenizer, task, data_description=DATA_DESCRIPTION):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": build_prompt(task, data_description)}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)


def build_pairs(tokenizer, rows):
    return [{"prompt": render_prompt(tokenizer, r["task"], TABLES[r.get("table", "orders")].description),
             "completion": r["program"] + tokenizer.eos_token} for r in rows]


def write_config(path, args, n_train, n_val, tables):
    """Record what was trained, with which settings and library versions (used for the model card)."""
    import importlib.metadata as md
    libs = {name: md.version(name) for name in
            ("torch", "transformers", "trl", "peft", "unsloth", "unsloth_zoo", "bitsandbytes", "accelerate")}
    path.write_text(json.dumps({"base_model": args.model, "epochs": args.epochs, "max_steps": args.max_steps,
                                "learning_rate": args.lr, "lora_rank": args.rank, "lora_alpha": args.rank,
                                "max_seq_length": args.max_len, "gradient_accumulation_steps": args.accum,
                                "per_device_batch_size": 1, "train_examples": n_train, "val_examples": n_val,
                                "train_tables": tables,
                                "optimizer": "adamw_8bit", "quantization": "bnb 4-bit (QLoRA)",
                                "libraries": libs}, indent=1), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unsloth/gemma-4-12b-it")
    ap.add_argument("--out", default=str(HERE / "out" / "adapter"))
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=3072)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()

    if a.dry:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.model)
        pair = build_pairs(tok, read_jsonl(HERE / "data" / "train.jsonl")[:1])[0]
        print(pair["prompt"][-700:], "\n=== completion ===\n", pair["completion"])
        return

    from unsloth import FastModel  # noqa: I001  (must be imported before trl and transformers)
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    model, processor = FastModel.from_pretrained(
        model_name=a.model, max_seq_length=a.max_len, load_in_4bit=True, full_finetuning=False)
    tok = getattr(processor, "tokenizer", processor)
    model = FastModel.get_peft_model(
        model, finetune_vision_layers=False, finetune_language_layers=True,
        finetune_attention_modules=True, finetune_mlp_modules=True,
        r=a.rank, lora_alpha=a.rank, lora_dropout=0.0, bias="none", random_state=3407)

    train = Dataset.from_list(build_pairs(tok, read_jsonl(HERE / "data" / "train.jsonl")))
    val = Dataset.from_list(build_pairs(tok, read_jsonl(HERE / "data" / "val.jsonl")))

    cfg = SFTConfig(
        output_dir=str(HERE / "out" / "trainer"),
        per_device_train_batch_size=1, per_device_eval_batch_size=1,
        gradient_accumulation_steps=a.accum,
        num_train_epochs=a.epochs, max_steps=a.max_steps,
        learning_rate=a.lr, lr_scheduler_type="linear", warmup_steps=10,
        optim="adamw_8bit", weight_decay=0.01, bf16=True,
        logging_steps=5, eval_strategy="steps", eval_steps=50, save_strategy="no",
        max_length=a.max_len, completion_only_loss=True, packing=False,
        dataset_num_proc=1, report_to="none", seed=3407)
    trainer = SFTTrainer(model=model, processing_class=tok, train_dataset=train, eval_dataset=val, args=cfg)
    trainer.train()

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tok.save_pretrained(out)
    (out.parent / "train_log.json").write_text(json.dumps(trainer.state.log_history, indent=1), encoding="utf-8")
    tables = sorted({r.get("table", "orders") for r in read_jsonl(HERE / "data" / "train.jsonl")})
    write_config(out.parent / "train_config.json", a, len(train), len(val), tables)
    print("SAVED adapter:", out)


if __name__ == "__main__":
    main()
