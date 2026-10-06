# Roadmap

Ken is an experiment. It asks whether a programming language can be designed so that small AI models write it
correctly and in few tokens. This page lists what I plan to do next to answer that question. It is a plan, not a
promise: the order and the contents will change with what the measurements show, and there are no dates.

## Where things stand

- **Language:** version 0.1, a small postfix pipeline language that compiles to Python
  ([reference](docs/language-reference.md)).
- **Evidence so far:** one domain only, tasks on tables of data, chosen because the answers are easy to check
  automatically. A fine-tuned Gemma 4 12B writes Ken for these tasks without a reasoning phase, and is compared with
  Qwen3.8-27B writing Ken and Python with and without reasoning ([details](docs/llm.md)).
- **What this does not show yet:** whether the language helps outside that domain, and whether a model that has
  never been trained on Ken writes it better or more cheaply than Python.

## 1. Publish the current experiment

- Evaluate the fine-tuned model on all three test sets (hand-written tasks, generated tasks, a table never seen in
  training) and compare it with Qwen3.8-27B.
- Update the documentation and the model card with these results, computed from the result files.
- Publish the model on Hugging Face and the `kenlang` package on PyPI, and make the repository public.

## 2. Test the idea beyond tables

This is the main question of the experiment.

- **A benchmark with several domains.** Hand-written tasks with automatic checks, for example text and file
  processing, small algorithms (counting, sorting, searching), JSON and nested structures, and tasks with several
  steps. Every task also gets a Python version, so both languages are measured on accuracy and on tokens.
- **Models that have not been trained on Ken.** The fairest test of the design is a model that sees Ken for the
  first time, through its reference only. If it writes Ken more reliably or more cheaply than Python, that is a
  property of the language and not of fine-tuning.
- **Report the results as they come,** including the domains where Ken does worse than Python.

## 3. Let the measurements shape the language

- Collect the mistakes models make in each domain (see the list in [docs/llm.md](docs/llm.md)) and change the
  language only where the data points to a problem: missing constructs, confusing words, rules that models get wrong.
- Measure every change: does it make models write Ken better, worse, or not differently?
- Known candidates from the current results are listed in the
  [design notes](docs/design.md#known-design-debts-candidates-for-02).
- A change that alters the meaning of existing programs is a new language version, and a model trained on an older
  version is out of date (see [CONTRIBUTING.md](CONTRIBUTING.md)).

## 4. Smaller models

- Train and evaluate a model of about 4 billion parameters or less. The idea behind Ken is "a small model plus a
  simple language", so this tests the hypothesis directly.
- Try a shorter language reference in the prompt: a trained model may need much less than the current ~1,500 tokens.

## 5. Make the results easy to reproduce

- Export the fine-tuned model to GGUF, so that it runs in llama.cpp, Ollama or LM Studio, and measure how quantisation
  changes the results.
- Keep every number in the documentation computed from result files that are published with it.

## Ideas for a later version

These are ideas to try once the steps above give a clearer picture, not commitments.

- **A small verifier model.** Train a small classifier (for example the encoder model
  [Laya](https://huggingface.co/convaiinnovations/laya)) to judge a program written by a generating model: does it do
  what the task asks, and if not, what kind of mistake is it? Such a verifier could pick the best of several
  candidate programs or hand a low-confidence case to a larger model.
  - It would not check syntax. The compiler already does that exactly.
  - Training data can be produced automatically: correct programs from the data generator, wrong ones made by
    deliberately breaking them (swapped arguments, a missing step, a changed field or number) and by collecting real
    model mistakes, with each label confirmed by running the program.
  - The test is whether it beats the plain rule "the program ran without an error", and it is most informative with a
    weaker generating model, since the current model is already right 98% of the time on its own tasks.

## How to take part

Ideas, tasks for the benchmark and reports of what a model got wrong are all useful. See
[CONTRIBUTING.md](CONTRIBUTING.md).
