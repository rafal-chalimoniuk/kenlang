# Publishing the model

The project has **two repositories**, and they are deliberately separate folders:

| | What it holds | Where it lives |
|---|---|---|
| **GitHub** – this repository | the language, docs, tests, benchmark, training code and the scripts in this folder; **no weights** | `kenlang/` |
| **Hugging Face** – the model | the LoRA adapter, tokenizer, model card, the exact language reference, training log, evaluation results | `project_hf/` next to this repository |

`model/` contains the tooling that builds the second from the first. It never writes into this repository.

```
python -m model.assemble --repo-id <user>/kenlang-gemma-4-12b-it-lora-v2 --repo-url https://github.com/<user>/kenlang
python -m model.upload   --repo-id <user>/kenlang-gemma-4-12b-it-lora-v2            # dry run: lists the files
python -m model.upload   --repo-id <user>/kenlang-gemma-4-12b-it-lora-v2 --yes      # creates a PRIVATE repo and uploads
```

Both commands default to the folder `<this repository>/../project_hf`; pass `--out` / `--folder` to change it.

| File | Purpose |
|---|---|
| `MODEL_CARD.template.md` | the model card; every `{{PLACEHOLDER}}` is filled by `assemble.py` |
| `assemble.py` | validates the adapter, computes all numbers from `training/out/eval_*.jsonl`, writes the model folder |
| `upload.py` | uploads that folder; a dry run unless `--yes`, private unless `--public` |

What the model folder contains: the LoRA adapter (`adapter_model.safetensors`, `adapter_config.json`), the tokenizer
and chat template, the model card (`README.md`), the exact language reference used in training (`kenlang_reference.md`),
the training configuration and log, and the raw evaluation results of the base model and the adapter.

Checks made by `assemble.py` before it writes anything: all adapter files exist, the weights file is not empty,
`adapter_config.json` names a Hub repository rather than a local path, and no placeholder is left in the card.
It also refuses to delete a non-empty output folder that is not a previous assembled model. `upload.py` refuses a
repository id that still contains `<your-username>`.

Things the scripts deliberately leave to you: the repository owner and name, and whether the repository is public.
