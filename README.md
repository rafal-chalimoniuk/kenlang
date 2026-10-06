# Ken

[![CI](https://github.com/rafal-chalimoniuk/kenlang/actions/workflows/ci.yml/badge.svg)](https://github.com/rafal-chalimoniuk/kenlang/actions/workflows/ci.yml)

**Ken is an experimental programming language.** The experiment asks one question: can a language be designed
so that small AI models write it correctly, in few tokens? The current version (0.1) is a pipeline language: a
program is a chain of steps, and Ken turns it into Python and runs it, so anything Python can do, Ken can use too.

The first experiments use tasks on tables of data, because their answers are easy to check automatically. Here is
a complete program. It finds the five best-selling products among the orders that were completed:

```
csv orders.csv where status eq done by product [ each qty sum ] top 5 map [ fmt "{0}: {1}" ] print
```

And this is what it prints:

```
laptop: 117
shoes: 114
headphones: 94
coffee: 92
mouse: 91
```

You can read it almost like a sentence: *load the orders file, keep the orders whose status is "done", for each
product add up the quantities, keep the top five, format each line, and print.*

For comparison, here is the same thing written in Python with pandas:

```python
import pandas as pd
df = pd.read_csv("orders.csv")
top = df[df.status == "done"].groupby("product").qty.sum().nlargest(5)
for product, units in top.items():
    print(f"{product}: {units}")
```

Both give exactly the same answer. One takes a single line, the other five. Over the seven benchmark tasks the Ken
solutions come to 306 tokens, against 528 for pandas and 808 for plain Python (see
[docs/llm.md](https://github.com/rafal-chalimoniuk/kenlang/blob/main/docs/llm.md#how-long-are-the-programs) for the
method and the caveats).

## Why does it exist?

Small AI models are good at short, simple things and easily lose their way in long, fiddly code. Ken is built
to suit them. It has only a few rules, and they are always the same. The whole language can be explained in a
couple of pages, so a model can read the instructions every time it writes a program. And the programs
come out short, so there is less room for mistakes.

This repository holds the language itself, a small test that measures how well an AI model can write it, and the
tools I used to teach one. The trained model is on [Hugging Face](https://huggingface.co/djchali/kenlang-gemma-4-12b-it-lora-v2).

## Try it

You need Python 3.9 or newer.

```
git clone https://github.com/rafal-chalimoniuk/kenlang.git
cd kenlang
python -m pip install -e .
kenlang run -e 'upto 10 filter [ mod 3 eq 0 ] print'
```

That last line keeps the numbers from 1 to 10 that divide evenly by 3, and prints `3`, `6` and `9`.
(A plain `pip install kenlang` is planned for the first public release.)

The example programs read their data from a `data` folder, so run them from inside `examples`:

```
cd examples
kenlang run sales_report.ken       # run a program from a file
kenlang py hello.ken               # see the Python that Ken writes for it
```

You can also use it from Python:

```python
import kenlang

kenlang.run('"Hello, Ken!" print')
print(kenlang.translate("upto 3 sum print"))   # the Python code, as text
```

## A first look at the language

- **A program is a list of words, read from left to right.** Each word does one thing to *the current value*
  (whatever you are holding at that moment, such as a table of orders or a list of numbers) and hands the result to the
  next word.
- **Some words need a few details after them.** `where status eq done` means "keep the rows where the status equals
  done". `round 2` means "round to two decimal places".
- **Square brackets hold a little recipe.** In `by product [ each qty sum ]` the recipe in brackets is run once for each
  product: "add up the quantities".
- **You can name things.** `as total` saves the current value under the name `total`. `def` lets you make your own
  word. Everything after `#` on a line is a comment.
- **Python is always within reach.** `use numpy as np` brings in a library, and `.sort_values:1 price` calls a method.

There are 56 built-in words. They cover reading data, picking rows, grouping, sorting, simple maths, working with
text and printing. The full list, with the exact rules, is in the
**[language reference](https://github.com/rafal-chalimoniuk/kenlang/blob/main/docs/language-reference.md)**.

## Examples

Each of these lives in the [examples](https://github.com/rafal-chalimoniuk/kenlang/tree/main/examples) folder, and each has a saved copy of its
correct output that the tests check.

| Program | What it does |
|---|---|
| [hello.ken](https://github.com/rafal-chalimoniuk/kenlang/blob/main/examples/hello.ken) | The smallest possible program |
| [numbers.ken](https://github.com/rafal-chalimoniuk/kenlang/blob/main/examples/numbers.ken) | Picks numbers from a list and adds them up |
| [word_frequency.ken](https://github.com/rafal-chalimoniuk/kenlang/blob/main/examples/word_frequency.ken) | Finds the most common words in a text |
| [grades.ken](https://github.com/rafal-chalimoniuk/kenlang/blob/main/examples/grades.ken) | Makes a new word that decides pass or retake |
| [sales_report.ken](https://github.com/rafal-chalimoniuk/kenlang/blob/main/examples/sales_report.ken) | A small sales report: revenue, best customers, monthly totals |
| [libraries.ken](https://github.com/rafal-chalimoniuk/kenlang/blob/main/examples/libraries.ken) | Uses numpy, scipy and matplotlib, and draws a chart |

## How well can an AI write it?

I measured it. A model gets a task in English and a description of a table, answers with a program, and the
program is run. An answer counts only if it runs and prints exactly the correct result, which I computed
separately with pandas. Every question was tried three times, temperature 0.6, no reasoning phase.

Gemma 4 (12 billion parameters, squeezed to 4 bits), before and after training on 2,020 generated examples:

| Questions | Gemma 4 12B | After training |
|---|---|---|
| 7 written by hand, about `orders.csv` | 1/21 (5%) | **21/21 (100%)** |
| 95 generated, about the three training tables | 2/285 (1%) | **279/285 (98%)** |
| 60 generated, about a table it never saw (`grades.csv`) | 7/180 (4%) | **176/180 (98%)** |

The last row is the one to look at. The model never saw that table during training, and its columns and shape are
different, so the result shows that the model learned the language and not one dataset. The hand-written questions
are about a table it did see, and the skills they need are in the training data (the tasks and answers are not), so
that row says less. Seven questions is also a small test.

The trained model writes short answers: 52 tokens on average, against 172 for the untrained one. A *token* is a
small piece of text, roughly three-quarters of a word.

The 10 wrong answers out of 465 are all in the harder tasks: two-part programs, wrongly chosen class thresholds,
a value kept with `as` inside a block, or `if` applied to a whole column (which the language does not support). The
full story, including what models tend to get wrong, is in
[docs/llm.md](https://github.com/rafal-chalimoniuk/kenlang/blob/main/docs/llm.md).

## What is in this repository

| Folder | What you will find |
|---|---|
| `src/kenlang` | The language itself: the part that reads your program and the part that runs it |
| `examples` | Programs you can run, with sample data |
| `docs` | The full language reference and notes on how it is built |
| `benchmarks` | The seven questions, the correct answers, and the scoring |
| `training` | The tools that create practice examples and train a model |
| `model` | The tools that prepare the trained model for Hugging Face |
| `tests` | Automatic checks that the language, the examples and the documentation still work |

More guides: [design notes](https://github.com/rafal-chalimoniuk/kenlang/blob/main/docs/design.md) ·
[how to benchmark a model](https://github.com/rafal-chalimoniuk/kenlang/blob/main/benchmarks/README.md) ·
[how to train one](https://github.com/rafal-chalimoniuk/kenlang/blob/main/training/README.md) ·
[what I plan to try next](https://github.com/rafal-chalimoniuk/kenlang/blob/main/ROADMAP.md).

## Good to know

- **Only run programs you trust.** A Ken program can do anything a Python program can, including deleting
  files. That also goes for programs written by an AI. Read
  [SECURITY.md](https://github.com/rafal-chalimoniuk/kenlang/blob/main/SECURITY.md) before running code from someone else.
- **It is an early version (0.1).** It works and is tested, but it has rough edges, and I list the ones I know
  about in the [design notes](https://github.com/rafal-chalimoniuk/kenlang/blob/main/docs/design.md#known-design-debts-candidates-for-02).
- **About the name.** The language is called Ken, its program files end in `.ken`, and the Python package is
  called `kenlang`, because the short name `ken` was already taken on PyPI.

## Want to help?

Ideas, questions and bug reports are welcome. The
[contributing guide](https://github.com/rafal-chalimoniuk/kenlang/blob/main/CONTRIBUTING.md) explains how to set things up and what to keep in
mind.

To check that everything works on your machine:

```
python -m pip install -e ".[dev]"
python -m pytest
```

## License

Copyright © 2026 Rafał Chalimoniuk. The language and its tools are free to use under the MIT license
([LICENSE](https://github.com/rafal-chalimoniuk/kenlang/blob/main/LICENSE)). The trained model on Hugging Face uses the Apache 2.0 license, like the
model it was built from.
