# Ken benchmark

Seven natural-language data tasks over `examples/data/orders.csv` (400 orders: `date, customer, product,
category, qty, price, status`). A model sees the Ken reference, the column description and one task, and
must answer with code only.

| # | Task | Skills | In training data |
|---|---|---|---|
| 1 | cancelled count + top 5 products by items | `by`, `top`, `fmt` | yes |
| 2 | revenue per month | `set`, `slice`, `by`, rounding | yes |
| 3 | % returned per category | `as` inside a block, arithmetic | yes |
| 4 | customers with ≥ 5 returned items | `by`, `filter`, `get` | yes |
| 5 | classify orders into large/medium/small | `def`, nested `if`, `freq` | yes |
| 6 | items in clothing/groceries with a name fragment | `where in`, `where has` | yes |
| 7 | write a CSV and print its row count | `pick`, `save` | yes |

*Yes* means the skill (not the task) occurs in the generated training data; no benchmark task or answer does.
Generalisation to unseen material is measured with a held-out table instead (see
[training/README.md](../training/README.md)).

## Scoring

`expected.py` computes the correct output of every task with plain pandas — no Ken involved — and
`reference/taskN.ken` are reference solutions that `tests/test_benchmark.py` checks against it. A model's
program passes if it runs without error in an isolated empty directory and prints the same lines
(numbers compared within 0.01); task 7 additionally checks the CSV (header, rows, order).

## Run it

```
# any OpenAI-compatible server, e.g. llama.cpp's llama-server
python -m benchmarks.run_llama --url http://127.0.0.1:8001 --model my-model --lang ken --n 3
python -m benchmarks.run_llama --url http://127.0.0.1:8001 --model my-model --lang py  --n 3   # Python baseline
python -m benchmarks.run_llama ... --think off      # disable reasoning (chat_template_kwargs.enable_thinking)

# a local model with Unsloth (base or adapter)
python -m training.evaluate --adapter training/out/adapter --n 3 --tag finetuned
```

Programs written by a model run on your machine; see [SECURITY.md](../SECURITY.md).

## Length of the reference solutions

`reference/` holds three correct solutions of every task: Ken (`taskN.ken`), plain Python (`python/taskN.py`, standard
library only) and pandas (`pandas/taskN.py`). `tests/test_benchmark.py` runs all 21 and checks them against the same
independent answers. The Python and pandas versions are written plainly, neither golfed nor padded.

```
python -m benchmarks.length                    # tokens (Gemma 4 tokenizer), needs `transformers`
python -m benchmarks.length --tokenizer none   # characters and lines only
```

The result is quoted in [docs/llm.md](../docs/llm.md#how-long-are-the-programs).
