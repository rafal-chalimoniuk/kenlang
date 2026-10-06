# Ken and language models

Ken is an experiment in designing a language that a small language model can write from a short reference. This
document says how to prompt a model, records what was measured, what models get wrong, and how to fine-tune one.
What is planned next is in the [roadmap](../ROADMAP.md).

## Prompting

`kenlang.llm.build_prompt(task, data_description)` returns the prompt used throughout this repository: the language
reference ([`src/kenlang/reference.md`](../src/kenlang/reference.md), about 1,500 tokens), a description of the data
(file name, column names and meanings, no values), the task, and the instruction *Answer with code only.*
`kenlang.llm.extract_code(answer)` pulls the program out of a reply (the first fenced block, or the whole text).

Notes from practice:

- Describe the **columns**, not the values. Models write what the data allows; they do not need rows.
- Put the argument order in the prompt: *arguments follow the command* (`get 0`, not `0 get`). Models trained
  mostly on Python and Forth-like languages sometimes reverse it; the reference says so in its first paragraph.
- Say what a placeholder in an output format means. `product: units` in a task means "the number of units", not the
  text `units`. Models trained on data where that was ambiguous appended the word to every line.
- Run what comes back in an isolated process (`benchmarks.scoring.run_isolated`), never in your interpreter.
  See [SECURITY.md](../SECURITY.md).

## What was measured

All scores are the share of attempts whose program ran and printed the correct output. Correct outputs are computed
independently with pandas (see [benchmarks/](../benchmarks/README.md)). Sampling temperature was 0.6, three attempts
per question, no reasoning phase. All prompts are English.

### Fine-tuning Gemma 4 12B

Gemma 4 12B, loaded in 4 bits, before and after QLoRA training on 2,020 generated examples
([training/](../training/README.md)):

| Questions | Gemma 4 12B | After training |
|---|---|---|
| 7 written by hand, about `orders.csv` | 1/21 (5%) | **21/21 (100%)** |
| 95 generated, about the three training tables | 2/285 (1%) | **279/285 (98%)** |
| 60 generated, about a table it never saw (`grades.csv`) | 7/180 (4%) | **176/180 (98%)** |

| Task | Skills | Gemma 4 12B | After training |
|---|---|---|---|
| 1 | `by`, `top`, `fmt` | 0/3 | 3/3 |
| 2 | `set`, `slice`, `by` | 0/3 | 3/3 |
| 3 | `as` inside a block, arithmetic | 0/3 | 3/3 |
| 4 | `by`, `filter`, `get` | 0/3 | 3/3 |
| 5 | `def`, nested `if`, `freq` | 0/3 | 3/3 |
| 6 | `where in`, `where has` | 0/3 | 3/3 |
| 7 | `pick`, `save` | 1/3 | 3/3 |

Mean answer length: 172 tokens for the untrained model, 52 after training.

How to read this:

- **The unseen table is the informative row.** The model never saw `grades.csv`; it has other column names and no
  price column. 98% there says the model learned the language and not one dataset.
- **The hand-written tasks say less than they used to.** Their skills (`if`, `def`, `as` inside a block) are in the
  training data, although none of the tasks or answers is, and the table is one the model trained on. Seven tasks is
  also a small sample: one task is 14 percentage points.
- **The generated tasks share their templates with the training data**, so they are the easiest to pass.

What the 10 wrong answers (out of 465) look like:

- **Wrong thresholds.** In a classification task the model picks class limits that do not match the question.
- **`if` on a whole column.** `set class [ if ... ]` on a table does not work (pitfall 3 in the
  [reference](language-reference.md#pitfalls)); `if` belongs inside `map` over a list.
- **A value kept with `as` inside a block** is divided by the wrong thing, or the variable is read where it does not
  exist.
- **A second part of a two-part program** that is right in structure but wrong in one detail.

### A lesson about the data

An earlier round of training data taught the model to append a unit to every output line, because in the training
tasks a word such as `units` in an output format sometimes was a literal unit and sometimes only a description of the
number. The hand-written benchmark, which writes `product: units` to mean "the number of units", caught it: two
tasks went from 3/3 to 0/3 while the generated test set, built with the same convention as the training data, still
showed 99%. The generator now writes `X pcs` when literal text follows the number, and a bare descriptive word is
never printed. A test set generated from the same templates as the training data cannot find this kind of mistake.

### How long are the programs?

A separate question from "can a model write it" is how much it has to write. [`benchmarks/reference/`](../benchmarks/reference)
holds three correct solutions of each of the seven benchmark tasks: Ken, plain Python (standard library only) and
pandas. All 21 are run by the tests against the same independent answers. The Python and pandas versions are written
plainly, the way a competent programmer or model would write them, neither golfed nor padded. Length in tokens of the
Gemma 4 tokenizer (`python -m benchmarks.length`):

| Task | Ken | Python | pandas |
|---|---|---|---|
| 1 | 51 | 125 | 86 |
| 2 | 51 | 101 | 78 |
| 3 | 48 | 142 | 76 |
| 4 | 39 | 96 | 63 |
| 5 | 72 | 139 | 97 |
| 6 | 23 | 80 | 62 |
| 7 | 22 | 125 | 66 |
| **All 7** | **306** | **808** | **528** |
| **Compared with Ken** | 1.0× | 2.6× | 1.7× |

In lines of code the totals are 13 for Ken, 46 for pandas and 86 for plain Python; in characters 1,053, 1,570 and
2,578.

How to read this:

- It is a static comparison of **correct programs**, not of what models write. For reference, the trained model's
  answers to these seven tasks average 45 tokens, close to the 44 of the Ken solutions above; what an untrained model
  writes in Python is a separate measurement that is not made yet.
- I wrote the Python and pandas versions myself. Someone else would write them somewhat shorter or longer, so treat
  the ratios as an estimate, not a constant.
- The seven tasks are table tasks, the domain Ken was designed around so far. The ratios say nothing about other kinds
  of programs; the [roadmap](../ROADMAP.md) plans a benchmark that does.
- Tokens are counted with the tokenizer of the model that was trained on Ken. A different tokenizer gives different
  numbers, but the characters and lines above point the same way.

## What models get wrong

Collected from every failure seen so far. They are the best guide to improving the language and the data:

| Mistake | Example | Why |
|---|---|---|
| Argument before the command | `pair 0 get` | habit from Forth/stack languages |
| `group` with a block | `group product [ each qty sum ]` | looks like `by`, which does take a block |
| Text command on a column | `date split "-"` | `split` works on one text, not on a column |
| Expressions as arguments | `where qty gt price`, `(a / b) * 100` | arguments are literal in Ken |
| `if` inside `set`/`filter` on a table | `set class [ if [ ge 1000 ] ... ]` | the block receives the whole column |
| Variable shadows text | `as o ... has o` | a variable name beats text in argument position |
| Variable named like a command | `as sub` | `sub` is the subtraction command |
| Unbalanced brackets in nested blocks | `[ if [ ... ] [ ... ] [ if ...` | counting brackets is hard without reasoning |
| Degenerate repetition | `by category [ ... ] by category [ ... ] ...` | a base model without enough signal about when to stop |

## Fine-tuning

[`training/`](../training/README.md) holds the whole recipe: a generator for verified synthetic data, QLoRA training with
Unsloth, and an evaluation that treats base and fine-tuned models identically. Design decisions worth knowing:

- **Verified data.** Each program is executed and compared with a result computed independently with pandas; examples
  that disagree are dropped. This found real problems: a variable named like a text argument, a variable named like a
  command, and integers written as `3.0` by a CSV check.
- **Honest splits.** No task text and no program appears in two sets (the same program worded differently once did),
  no benchmark answer appears in the data, and one table is kept out of training entirely.
- **More than one table.** Three tables with different columns are used for training, so the model cannot get by on
  one schema.
- **The prompt is part of the model.** Training uses exactly the prompt `build_prompt` produces, including the
  ~1,500-token reference. Change the reference and the trained model is out of date; see [CONTRIBUTING.md](../CONTRIBUTING.md).
