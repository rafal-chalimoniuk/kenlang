# Changelog

All notable changes are recorded here. The format follows [Keep a Changelog](https://keepachangelog.com/),
and the project follows [Semantic Versioning](https://semver.org/). Because a language model is trained on a
specific version of the language, **a change in the meaning of existing programs is a major-version change.**

## [0.1.0] - unreleased

First release.

### Language
- Postfix pipeline language with a current value, a variable table, blocks, `def`, `as`, `use`.
- 56 built-in commands: sources, selecting, shaping, ordering, aggregates, arithmetic, comparison, text, control.
- Python interop: `alias.function`, `.method`, `:N` explicit arity, `name=value` keyword arguments.
- `#` comments. `where` operators `in`, `nin`, `has`. `fmt` accepts Python format specs.

### Tooling
- `kenlang run`, `kenlang py`, `python -m kenlang`; package `kenlang` with `translate()` and `run()`.
- `KenSyntaxError` with readable messages (unbalanced brackets, missing arguments, unknown module).
- `kenlang.llm`: language reference for prompts, `build_prompt()`, `extract_code()`.
- Benchmark of seven tasks with independently computed answers; synthetic training-data generator over three
  tables, with a fourth held out to test generalisation; Unsloth fine-tuning and evaluation scripts;
  Hugging Face packaging for the fine-tuned adapter.
