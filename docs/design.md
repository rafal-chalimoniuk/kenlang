# Design notes

## Goals

1. **Short programs.** Typical tasks (in the experiments so far, tasks on tables such as "revenue per month, top
   five customers") should take one or two lines, and fewer tokens than the equivalent Python code.
2. **Easy for small language models to write.** The whole language fits in a ~1,500-token reference, and the
   grammar is so regular that a model needs to learn only *words and their arities*, not precedence,
   operators or nesting rules.
3. **No new ecosystem.** Ken compiles to Python; anything Python can import, Ken can call.

Version 0.1 is deliberately small: it has no loops, no user-defined arguments and no recursion. It is not a
sandbox.

## Why postfix with fixed arity

Every command has a fixed number of arguments that follow it directly. That gives:

- **No parser state.** A single left-to-right pass with no precedence: read a word, look up its arity,
  read that many values (a value is a literal or a bracketed block).
- **A natural data flow.** The current value is implicit, so a pipeline reads in the order it executes.
  `ok each rev sum round 2 print` is what happens, step by step.
- **Predictable errors.** A model that forgets an argument or adds one produces a program that fails
  at compile time (`unexpected end of program`) or at the next command, not a silently different one.

The price is that arguments are *literal*: they are numbers, texts, blocks or variable names, never
expressions. A computed value must be stored with `as` or `set` first. See the pitfalls in the
[language reference](language-reference.md#pitfalls).

## How a program is compiled

```
source ──tokenize──▶ words ──Parser──▶ tree ──Gen──▶ Python source ──exec──▶ output
         (regex)              (arity table)        (one statement per word)
```

1. **Tokenize** (`compiler.TOKEN_RE`): quoted text, comments, brackets, everything else by whitespace.
2. **Parse** (`compiler.Parser`): `word()` decides what a word is — `use`, `def`, `as`, a `.method`, a
   library call (`alias.function`), a built-in command (looked up in `ARITY`), or a bare value — and reads
   the arguments that follow. Blocks recurse. Library calls whose arity is not given as `:N` read it from
   the function's signature (`inspect`), which is why `use` imports the module at parse time.
3. **Generate** (`compiler.Gen`): every node becomes one statement `_ = rt.command(_, V, args...)`;
   a block becomes a nested `def _bN(_, V)` that returns its final `_`; `def name [...]` becomes
   `def t_name(_, V)`. Run `kenlang py file.ken` to see the result.
4. **Execute** (`compiler.run`): `exec` of the generated source.

## The runtime

`kenlang/runtime.py` holds one function per command, with the signature `command(it, V, *args)`.
Most commands accept three kinds of values and dispatch on the type:

| Value | Used for | Example |
|---|---|---|
| pandas `DataFrame` / `Series` | data loaded with `csv`; vectorised | `where`, `set`, `each`, `sum` |
| Python list / dict | results of `group`, `pairs`, `freq`, `list` | `map`, `sort`, `top` |
| scalar | numbers and text | `add`, `round`, `fmt` |

Blocks are called as `block(element, scope)`, where `scope` is a **copy** of the variable table extended
with the element's fields (`row_env`). That is why variables created inside a block do not leak and why
columns can be used by name inside `set` and `filter`.

## Extending the language

Adding a command takes three steps, and the test suite enforces the last two:

1. Add `"name": arity` to `ARITY` in `compiler.py` and a function `name(it, V, ...)` to `runtime.py`
   (add it to `PYNAME` if the name clashes with a Python keyword).
2. Document it in `docs/language-reference.md` and in `src/kenlang/reference.md`
   (`tests/test_docs.py` fails otherwise, and also checks the documented arity).
3. Add a test in `tests/test_runtime.py` that exercises it on a list, a dict and a table where it applies.

### Compatibility and the model

A language model trained on Ken learns the *exact* text of `reference.md` and the behaviour of version 0.1.
Edits to the reference or to existing behaviour invalidate the released model; see
[CONTRIBUTING.md](../CONTRIBUTING.md#language-changes-and-the-model).

## Known design debts (candidates for 0.2)

These were found by watching a small model fail, and are recorded in `docs/llm.md`:

- `group` and `by` look alike but take different arguments; `group category [ ... ]` is a common mistake.
  A candidate fix is to let `group field [block]` mean `by`.
- A variable named like a text argument wins (`has o` when a table is called `o`). Possible fix: resolve
  names in argument position only when they were declared with `as`, or require quotes for text that
  collides.
- `if` cannot be used in `set`/`filter` on a table, which pushes people to `map` over a column.
- Error messages from the runtime are Python's; they could be translated into Ken terms.
