"""Expected results of the seven benchmark tasks, computed independently with pandas.

These do not use Ken at all, so they can validate both the reference solutions and a model's
programs. ``expected(task_id)`` returns ``(stdout_text, csv_spec_or_None)``.
"""
import json
import pathlib

import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
ORDERS = HERE.parent / "examples" / "data" / "orders.csv"


def _df():
    df = pd.read_csv(ORDERS)
    df["rev"] = df.qty * df.price
    return df


def _lines(pairs, sep=": ", suffix=""):
    return "\n".join(f"{k}{sep}{v}{suffix}" for k, v in pairs) + "\n"


def expected(task_id):
    df = _df()
    done = df[df.status == "done"]
    csv_spec = None
    if task_id == 1:
        top = done.groupby("product", sort=False).qty.sum().sort_values(ascending=False, kind="stable").head(5)
        out = f"Cancelled:\n{(df.status == 'cancelled').sum()}\nBest sellers:\n" + _lines(top.items())
    elif task_id == 2:
        month = done.date.str[:7]
        rev = done.rev.groupby(month).sum().round(2)
        out = _lines(sorted(rev.items()))
    elif task_id == 3:
        share = (df.status == "returned").groupby(df.category).mean() * 100
        ranked = sorted(((k, round(v, 1)) for k, v in share.items()), key=lambda kv: kv[1], reverse=True)
        out = _lines(ranked, suffix="%")
    elif task_id == 4:
        ret = df[df.status == "returned"].groupby("customer").qty.sum()
        out = _lines(sorted((k, int(v)) for k, v in ret.items() if v >= 5))
    elif task_id == 5:
        tier = done.rev.map(lambda v: "large" if v >= 1000 else "medium" if v >= 200 else "small")
        out = _lines(tier.value_counts().items())
    elif task_id == 6:
        sel = done[done.category.isin(["clothing", "groceries"]) & done["product"].str.contains("e", regex=False)]
        out = f"{int(sel.qty.sum())}\n"
    elif task_id == 7:
        sel = df[df.status == "returned"].sort_values("date", kind="stable")[["date", "customer", "product"]]
        out = f"{len(sel)}\n"
        csv_spec = dict(file="returns.csv", header=list(sel.columns),
                        rows=[[str(v) for v in r] for r in sel.values.tolist()], sort_col=0, desc=False)
    else:
        raise ValueError(task_id)
    return out, csv_spec


def tasks():
    return json.loads((HERE / "tasks.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    for t in range(1, 8):
        print(f"--- task {t}")
        print(expected(t)[0], end="")
