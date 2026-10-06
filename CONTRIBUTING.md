# Contributing

Thank you for helping with Ken.

## Setup

```
python -m venv .venv && . .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install -e ".[dev]"
python -m pytest
ruff check .
```

## Where things live

| You want to change | Edit | Also update |
|---|---|---|
| a command's behaviour | `src/kenlang/runtime.py` | tests in `tests/test_runtime.py`, `docs/language-reference.md` |
| syntax | `src/kenlang/compiler.py` | `tests/test_compiler.py`, the grammar in `docs/language-reference.md` |
| add a command | `ARITY` in `compiler.py` **and** a function of the same name in `runtime.py` | `docs/language-reference.md`, `src/kenlang/reference.md`, a test |
| the LLM prompt | `src/kenlang/reference.md` | retrain: the model learns this exact text (see `docs/llm.md`) |

`tests/test_docs.py` fails if a command is missing from either document, or if its arity is documented
wrongly, and it runs every example in `src/kenlang/reference.md`.

## Language changes and the model

The released model was trained on language version 0.1 and on the exact prompt in `src/kenlang/reference.md`.
A change that alters the meaning of an existing program, or any edit of the reference text, makes the
published model's behaviour stale. Therefore:

- additive changes (a new command that does not clash with names or variables in common use) are a minor
  version and need the training data regenerated before the next model release;
- changes of existing behaviour are a major version.

State in the pull request which of the two it is.

## Licensing of contributions

Ken is MIT-licensed and copyright © 2026 Rafał Chalimoniuk. By opening a pull request you agree that your
contribution is released under the same MIT license as the rest of the project. You keep the copyright to your
own contribution; add yourself to the pull request description if you want to be credited.

## Style

- Python 3.9 compatible; `ruff check .` must pass.
- Comments and documentation in English. Keep runtime functions short and free of hidden state.
- Tests for observable behaviour, not for implementation details. Prefer a real program (`run("...")`)
  over mocking.
