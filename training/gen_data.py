"""Synthetic training data for Ken: (task in English, Ken program) pairs.

Every program is *executed*, and its output is compared with a result computed independently
with pandas; examples that disagree are discarded. Examples whose output equals the output of a
benchmark task are dropped as well.

The data

* covers three tables with different columns (``orders``, ``bookings``, ``shipments``; see ``tables.py``),
  and a fourth, ``grades``, that is used only for a test set and never for training;
* teaches ``if`` inside a block, ``def`` and ``as`` inside a block, in several program shapes (shares, ratios,
  labels, classes, scaled values);
* words the tasks in more than one way.

No task text and no program appears in two sets. The model is judged on the generated test sets, on the unseen
table and on the hand-written benchmark; the benchmark's skills are taught, but none of its tasks or answers is.

    python -m training.gen_data [n_train]    # writes training/data/{train,val,test,test_unseen_table}.jsonl
"""
import hashlib
import json
import operator
import pathlib
import random
import sys

import pandas as pd

from benchmarks.expected import expected
from benchmarks.scoring import check_csv_spec, run_in_process, same_out, scratch_dir
from kenlang.compiler import ARITY

from .tables import HELD_OUT_TABLE, TABLES, TRAIN_TABLES

HERE = pathlib.Path(__file__).resolve().parent

LABELS = ["Result:", "Summary:", "Overview:", "Data:", "Report:", "Results:"]
# Names a program uses for its own variables and words. A text argument equal to one of them would be
# read as that variable, so fragments for `has` are never one of these.
VARIABLES = {"o", "g", "n", "r", "q", "a", "b", "x", "y", "t", "grp", "num", "den", "total", "rows", "part",
             "size", "base", "whole", "full", "tot", "grand", "rev", "month"}
SINGLE_LETTERS = set("eiukcdflmhjwzv")
HIGH_LOW = [("high", "low"), ("large", "small"), ("good", "poor"), ("above", "below"), ("heavy", "light")]
HML = [("high", "medium", "low"), ("large", "medium", "small"), ("top", "middle", "bottom"),
       ("gold", "silver", "bronze"), ("heavy", "medium", "light")]
DEF_NAMES = ["tier", "level", "band", "bucket", "grade", "kind", "class_of", "mark", "bracket"]
SCALE_NAMES = ["markup", "adjust", "scaled", "with_tax", "bump", "uplift"]
OPS = {"ge": operator.ge, "gt": operator.gt, "le": operator.le, "lt": operator.lt}
OP_TEXT = {"ge": "is at least", "gt": "is greater than", "le": "is at most", "lt": "is less than"}


assert not VARIABLES & set(DEF_NAMES + SCALE_NAMES)
assert not (VARIABLES | set(DEF_NAMES + SCALE_NAMES)) & (set(ARITY) - {"rev"}), "a variable named like a command"


class Skip(Exception):
    """The random draw produced an uninteresting example; try again."""


def pick(rnd, *alts):
    return rnd.choice(alts)


# ---------- what a table offers ----------

def singular(t):
    return t.rows[:-1]


_FRAGMENTS = {}


def fragments(t):
    """Fragments of the values of ``t.name_col`` that match some, but not all, of them."""
    if t.key in _FRAGMENTS:
        return _FRAGMENTS[t.key]
    vals = sorted(t.frame()[t.name_col].unique())
    found = set()
    for v in vals:
        for n in (1, 2, 3):
            for i in range(len(v) - n + 1):
                f = v[i:i + n]
                if f.isalpha() and (n > 1 or f in SINGLE_LETTERS):
                    found.add(f)
    reserved = set(ARITY) | VARIABLES | {"as", "def", "use", "if"} | set(t.cols)
    _FRAGMENTS[t.key] = [f for f in sorted(found - reserved)
                         if 1 <= sum(f in v for v in vals) <= max(1, int(0.85 * len(vals)))]
    return _FRAGMENTS[t.key]


def group_keys(t):
    """Columns one can group by: key -> {each phrases, noun, sort word, prefix}."""
    keys = {c: dict(each=d["each"], noun=d["noun"], key=c, sort="alphabetically", prefix="")
            for c, d in t.dims.items()}
    keys[t.state] = dict(each=[f"each {t.state}"], noun=t.state, key=t.state, sort="alphabetically", prefix="")
    keys["month"] = dict(each=["each month"], noun="month", key="month", sort="chronologically",
                         prefix=f"set month [ {t.date} slice 0 7 ]")
    return keys


def measures(t):
    """The things one can compute for a group of rows. ``raw`` is the unrounded block, ``fraw`` its value."""
    m = {"count": dict(phr=f"the number of {t.rows}", blk="count", raw="count", f=len, fraw=len,
                       val="count", unit="")}
    q, p, s = t.qty, t.price, t.stat
    if q:
        c = q.name
        m["qty"] = dict(phr="the total " + q.noun[4:], blk=f"each {c} sum", raw=f"each {c} sum", val="total",
                        f=lambda d, c=c: int(d[c].sum()), fraw=lambda d, c=c: float(d[c].sum()), unit=q.unit)
        m["meanqty"] = dict(phr=f"the average {q.noun[4:]} per {singular(t)}", blk=f"each {c} mean round 2",
                            f=lambda d, c=c: round(float(d[c].mean()), 2), val="average", unit="")
    if q and p:
        qc, pc = q.name, p.name
        both = f"(`{qc}` × `{pc}`)"
        m["rev"] = dict(phr=f"the total {t.money['noun']} {both}", blk="each rev sum round 2", raw="each rev sum",
                        rev=True, f=lambda d: round(float(d.rev.sum()), 2), fraw=lambda d: float(d.rev.sum()),
                        val=t.money["noun"], unit=t.money["unit"])
        m["avg"] = dict(phr=f"{t.money['avg']} {both}", blk="each rev mean round 2", rev=True,
                        f=lambda d: round(float(d.rev.mean()), 2), val="average", unit=t.money["unit"])
        m["maxrev"] = dict(phr=f"{t.money['max']} {both}", blk="each rev max round 2", rev=True,
                           f=lambda d: round(float(d.rev.max()), 2), val="maximum", unit=t.money["unit"])
    if s:
        c = s.name
        m["smean"] = dict(phr=f"the average {s.noun}", blk=f"each {c} mean round 2", val="average", unit=s.unit,
                          f=lambda d, c=c: round(float(d[c].mean()), 2))
        m["smax"] = dict(phr=f"the highest {s.noun}", blk=f"each {c} max", val="maximum", unit=s.unit,
                         f=lambda d, c=c: float(d[c].max()))
        m["smin"] = dict(phr=f"the lowest {s.noun}", blk=f"each {c} min", val="minimum", unit=s.unit,
                         f=lambda d, c=c: float(d[c].min()))
        if s.additive:
            m["ssum"] = dict(phr=f"the total {s.noun}", blk=f"each {c} sum round 1", raw=f"each {c} sum", unit=s.unit,
                             val="total", f=lambda d, c=c: round(float(d[c].sum()), 1),
                             fraw=lambda d, c=c: float(d[c].sum()))
    for c, d in t.dims.items():
        m[f"n_{c}"] = dict(phr=f"the number of distinct {d['plural']}", blk=f"each {c} nuniq", raw=f"each {c} nuniq",
                           f=lambda df, c=c: int(df[c].nunique()), fraw=lambda df, c=c: float(df[c].nunique()),
                           val="distinct", unit="", nuniq=c)
    return m


def scalar_extras(t):
    out = {}
    if t.price:
        c = t.price.name
        out["maxprice"] = dict(phr=f"the highest {t.price.noun}", blk=f"each {c} max",
                               f=lambda d, c=c: float(d[c].max()))
        out["minprice"] = dict(phr=f"the lowest {t.price.noun}", blk=f"each {c} min",
                               f=lambda d, c=c: float(d[c].min()))
    return out


# ---------- filters: (task text, Ken `where`, pandas mask) ----------

def numeric_filter(num, rnd):
    kind, n = rnd.choice(list(OPS)), rnd.choice(num.steps)
    return (f"where `{num.name}` {OP_TEXT[kind]} {n}", f"where {num.name} {kind} {n}",
            lambda d: OPS[kind](d[num.name], n))


def make_filter(t, field, rnd):
    if field == "state":
        x = rnd.choice(t.states)
        if rnd.random() < 0.25:
            return f"with {t.state} other than `{x}`", f"where {t.state} ne {x}", lambda d: d[t.state] != x
        return f"with {t.state} `{x}`", f"where {t.state} eq {x}", lambda d: d[t.state] == x
    if field in t.dims:
        values = sorted(t.frame()[field].unique())
        noun = t.dims[field]["noun"]
        if rnd.random() < 0.3:
            a, b = sorted(rnd.sample(values, 2))
            return (f"with {noun} `{a}` or `{b}`", f"where {field} in {a},{b}", lambda d: d[field].isin([a, b]))
        x = rnd.choice(values)
        return f"with {noun} `{x}`", f"where {field} eq {x}", lambda d: d[field] == x
    if field == "name":
        fr = rnd.choice(fragments(t))
        noun = t.dims[t.name_col]["noun"]
        return (f"whose {noun} name contains the fragment `{fr}`", f"where {t.name_col} has {fr}",
                lambda d: d[t.name_col].str.contains(fr, regex=False))
    if field == "month":
        m = rnd.choice(sorted(t.frame().month.unique()))
        return f"from the month `{m}`", f"where {t.date} has {m}", lambda d: d[t.date].str.contains(m, regex=False)
    num = {"qty": t.qty, "price": t.price, "stat": t.stat}[field]
    return numeric_filter(num, rnd)


def filter_fields(t):
    fields = ["state", "name", "month"] + list(t.dims)
    fields += [f for f, num in (("qty", t.qty), ("price", t.price), ("stat", t.stat)) if num]
    return fields


def sample_filters(t, rnd, k, avoid=()):
    fields = [f for f in filter_fields(t) if f not in avoid]
    chosen = [make_filter(t, f, rnd) for f in rnd.sample(fields, k)]
    df = t.frame()
    mask = pd.Series(True, index=df.index)
    for _, _, m in chosen:
        mask &= m(df)
    return [x for x, _, _ in chosen], [w for _, w, _ in chosen], df[mask]


def considering(t, texts):
    return f", considering only {t.rows} {' and '.join(texts)}" if texts else ""


def scope_text(t, texts):
    return f"among {t.rows} {' and '.join(texts)}" if texts else f"over all {t.rows}"


def label_part(rnd):
    if rnd.random() < 0.4:
        lab = rnd.choice(LABELS)
        return lab, f" (first print the text `{lab}`)", f'"{lab}" print '
    return None, "", ""


def prefix_parts(t, start, wheres, gi=None, rev=False):
    parts = [start] + wheres
    if rev:
        parts.append(f"set rev [ {t.qty.name} mul {t.price.name} ]")
    if gi and gi["prefix"]:
        parts.append(gi["prefix"])
    return parts


def round_nice(x, integer=False):
    """A threshold that reads well in a task text."""
    x = float(x)
    if integer:
        return int(round(x))
    if abs(x) >= 100:
        return int(round(x, -1)) if abs(x) >= 300 else int(round(x))
    if x.is_integer() or abs(x) >= 10:
        return int(round(x))
    return round(x, 1)


# ---------- output shapes shared by the grouped pieces ----------

def choose_shape(rnd, items, gi, sep, unit, hint_val):
    """How the (key, value) pairs are ordered and printed: returns pipe, text, expected pairs and format."""
    shape = rnd.choice(["sorted_key", "top", "bottom", "rsort_all", "threshold", "top"])
    show = rnd.choice(["prints", "prints", "outputs", "displays"])
    fmtstr = "{0}" + sep + "{1}" + unit
    hint = f"{gi['key']}{sep}{hint_val}{unit}"
    if shape == "sorted_key":
        return ["pairs sort 0"], f"{show} the results {gi['sort']}, one per line in the format `{hint}`", \
            sorted(items, key=lambda kv: kv[0]), fmtstr
    if shape == "top":
        n = rnd.randint(1, min(5, len(items) - 1))
        what = "the single entry" if n == 1 else f"the {n} entries"
        return [f"top {n}"], \
            f"{show} {what} with the largest value (descending), one per line in the format `{hint}`", \
            sorted(items, key=lambda kv: kv[1], reverse=True)[:n], fmtstr
    if shape == "bottom":
        n = rnd.randint(1, min(4, len(items) - 1))
        what = "the single entry" if n == 1 else f"the {n} entries"
        return [f"pairs sort 1 first {n}"], \
            f"{show} {what} with the smallest value (ascending), one per line in the format `{hint}`", \
            sorted(items, key=lambda kv: kv[1])[:n], fmtstr
    if shape == "rsort_all":
        return ["pairs rsort 1"], \
            f"{show} all results from the largest value to the smallest, in the format `{hint}`", \
            sorted(items, key=lambda kv: kv[1], reverse=True), fmtstr
    vs = sorted(v for _, v in items)
    cut = round_nice(vs[len(vs) // 2])
    out = sorted([kv for kv in items if kv[1] >= cut], key=lambda kv: kv[0])
    if not out or len(out) == len(items):
        raise Skip
    return [f"pairs filter [ get 1 ge {cut} ] sort 0"], \
        f"{show} {gi['sort']} only the entries whose value is at least {cut}, in the format `{hint}`", out, fmtstr


def grouped_items(sub, gi, value, min_groups=3):
    items = [(key, value(grp)) for key, grp in sub.groupby(gi["key"], sort=False)]
    if len(items) < min_groups:
        raise Skip
    return items


def format_lines(out, sep, unit):
    return [f"{k}{sep}{v}{unit}" for k, v in out]


def choose_group(t, rnd):
    keys = group_keys(t)
    return keys[rnd.choice(list(keys))]


def by_or_group(rnd, gi, block):
    if rnd.random() < 0.65:
        return f"by {gi['key']} [ {block} ]"
    return f"group {gi['key']} map [ {block} ]"


def per_clause(rnd, gi, phr, cons, out_txt, lab_txt, verb=None):
    verb = verb or pick(rnd, "computes", "computes", "calculates", "works out", "determines")
    each = rnd.choice(gi["each"])
    return pick(rnd,
                f"for {each} {verb} {phr}{cons}, and then {out_txt}{lab_txt}",
                f"{verb} {phr} for {each}{cons}, and then {out_txt}{lab_txt}",
                f"{verb} {phr} for every {gi['noun']}{cons}, then {out_txt}{lab_txt}")


# ---------- pieces of a program ----------
# Each returns dict(clause, tok, exp, pre=[def lines]); `start` is "csv file" or the variable `o`.

def piece_grouped(rnd, t, start):
    gi, ms = choose_group(t, rnd), measures(t)
    m = rnd.choice([k for k, v in ms.items() if v.get("nuniq") != gi["key"]])
    mi = ms[m]
    texts, wheres, sub = sample_filters(t, rnd, rnd.choice([0, 0, 1, 1, 2]))
    if len(sub) < 12:
        raise Skip
    items = grouped_items(sub, gi, mi["f"])
    sep = rnd.choice([": ", ": ", " - ", "; "])
    unit = mi["unit"] if rnd.random() < 0.5 else ""
    # `X` stands for the number when literal text follows it (`X pcs`). A descriptive word (`total`, `units`)
    # is only a placeholder and is never printed: a task such as `product: units` means "the number of units".
    word = "X" if unit else mi["val"]
    if not unit and m == "qty" and rnd.random() < 0.5:
        word = t.qty.unit.strip()
    pipe, out_txt, out, fmtstr = choose_shape(rnd, items, gi, sep, unit, word)
    lab, lab_txt, lab_tok = label_part(rnd)
    parts = prefix_parts(t, start, wheres, gi, mi.get("rev"))
    parts.append(by_or_group(rnd, gi, mi["blk"]))
    parts += pipe + [f'map [ fmt "{fmtstr}" ] print']
    return dict(clause=per_clause(rnd, gi, mi["phr"], considering(t, texts), out_txt, lab_txt),
                tok=lab_tok + " ".join(parts), exp=([lab] if lab else []) + format_lines(out, sep, unit), pre=[])


def piece_scalar(rnd, t, start):
    pool = {**measures(t), **scalar_extras(t)}
    mi = pool[rnd.choice(list(pool))]
    texts, wheres, sub = sample_filters(t, rnd, rnd.choice([0, 1, 1, 2, 2]))
    if len(sub) < 8:
        raise Skip
    v = mi["f"](sub)
    lab, lab_txt, lab_tok = label_part(rnd)
    parts = prefix_parts(t, start, wheres, rev=mi.get("rev")) + [mi["blk"], "print"]
    verb = pick(rnd, "computes", "determines", "calculates", "works out")
    clause = f"{verb} {mi['phr']} {scope_text(t, texts)} and prints just the value{lab_txt}"
    return dict(clause=clause, tok=lab_tok + " ".join(parts), exp=([lab] if lab else []) + [str(v)], pre=[])


def piece_distinct(rnd, t, start):
    col = rnd.choice(list(t.dims))
    texts, wheres, sub = sample_filters(t, rnd, rnd.choice([1, 2]))
    vals = sorted(sub[col].unique())
    if len(sub) < 6 or len(vals) < 2:
        raise Skip
    lab, lab_txt, lab_tok = label_part(rnd)
    parts = [start] + wheres + [f"each {col} uniq sort it print"]
    clause = (f"prints, alphabetically and without duplicates, the values of the column `{col}` "
              f"for {t.rows} {' and '.join(texts)}{lab_txt}")
    return dict(clause=clause, tok=lab_tok + " ".join(parts), exp=([lab] if lab else []) + vals, pre=[])


def piece_save(rnd, t, start):
    texts, wheres, sub = sample_filters(t, rnd, rnd.choice([1, 2]))
    if len(sub) < 6:
        raise Skip
    cols = rnd.sample(list(t.cols), rnd.choice([2, 3, 4]))
    sc = rnd.choice(cols)
    desc = rnd.random() < 0.4
    fname = rnd.choice(["result.csv", "export.csv", "orders_out.csv", "data_out.csv", "list.csv", "report_out.csv"])
    sub = sub.sort_values(sc, ascending=not desc, kind="stable")
    rows = [[str(v) for v in r] for r in sub[cols].astype(object).values.tolist()]
    cols_txt = ", ".join(f"`{c}`" for c in cols[:-1]) + f" and `{cols[-1]}`"
    clause = (f"writes to the file `{fname}` the columns {cols_txt} (in this order) of {t.rows} "
              f"{' and '.join(texts)}, sorted {'descending' if desc else 'ascending'} by the column `{sc}`. "
              f"The file must have a header and no index column. Then the program prints the number of rows written")
    tok = " ".join([start] + wheres + [f"{'rsort' if desc else 'sort'} {sc}", f"pick {','.join(cols)}",
                                      f"save {fname}", "count print"])
    return dict(clause=clause, tok=tok, exp=[str(len(sub))], pre=[],
                csv=dict(file=fname, header=cols, rows=rows, sort_col=cols.index(sc), desc=desc))


# --- `as` inside a block ---

def piece_share(rnd, t, start):
    """Percentage of the rows of each group that meet a condition: a table is stored with `as` inside the block."""
    gi = choose_group(t, rnd)
    cond_field = rnd.choice(["state", "state", "qty", "name"] if t.qty else ["state"])
    cond_text, cond_where, cond_mask = make_filter(t, cond_field, rnd)
    texts, wheres, sub = sample_filters(t, rnd, rnd.choice([0, 0, 1]), avoid=(cond_field,))
    if len(sub) < 40:
        raise Skip

    def value(grp):
        return round(float(cond_mask(grp).sum()) / len(grp) * 100, 1)
    items = grouped_items(sub, gi, value)
    sep = rnd.choice([": ", ": ", " - "])
    pipe, out_txt, out, fmtstr = choose_shape(rnd, items, gi, sep, "%", "X")
    g, n = rnd.choice(["g", "grp", "rows", "part"]), rnd.choice(["n", "size", "tot"])
    block = f"as {g} {g} count as {n} {g} {cond_where} count div {n} mul 100 round 1"
    parts = prefix_parts(t, start, wheres, gi) + [f"by {gi['key']} [ {block} ]"] + pipe
    parts.append(f'map [ fmt "{fmtstr}" ] print')
    each = rnd.choice(gi["each"])
    clause = (f"computes, for {each}, the percentage of {t.rows} {cond_text} among all {t.rows} of that "
              f"{gi['noun']}{considering(t, texts)}, rounded to 1 decimal place, and then {out_txt}")
    return dict(clause=clause, tok=" ".join(parts), exp=format_lines(out, sep, "%"), pre=[])


def piece_ratio(rnd, t, start):
    """Ratio of two measures per group: both are stored with `as` inside the block."""
    gi, ms = choose_group(t, rnd), measures(t)
    top = [k for k, v in ms.items() if "raw" in v and v.get("nuniq") != gi["key"]]
    bottom = [k for k in top if k == "count" or k == "qty" or ms[k].get("nuniq")]
    a, b = rnd.choice(top), rnd.choice(bottom)
    if a == b:
        raise Skip
    A, B = ms[a], ms[b]
    texts, wheres, sub = sample_filters(t, rnd, rnd.choice([0, 1, 1]))
    if len(sub) < 40:
        raise Skip
    items = grouped_items(sub, gi, lambda d: round(A["fraw"](d) / B["fraw"](d), 2))
    sep = rnd.choice([": ", ": ", " - "])
    pipe, out_txt, out, fmtstr = choose_shape(rnd, items, gi, sep, "", "ratio")
    g, x, y = rnd.choice(["g", "grp", "rows", "part"]), rnd.choice(["a", "x", "num"]), rnd.choice(["b", "y", "den"])
    block = f"as {g} {g} {A['raw']} as {x} {g} {B['raw']} as {y} {x} div {y} round 2"
    rev = A.get("rev") or B.get("rev")
    parts = prefix_parts(t, start, wheres, gi, rev) + [f"by {gi['key']} [ {block} ]"] + pipe + \
        [f'map [ fmt "{fmtstr}" ] print']
    clause = per_clause(rnd, gi, f"the ratio of {A['phr']} to {B['phr']}",
                        f"{considering(t, texts)}, rounded to 2 decimal places", out_txt, "",
                        verb=pick(rnd, "computes", "calculates"))
    return dict(clause=clause, tok=" ".join(parts), exp=format_lines(out, sep, ""), pre=[])


def piece_total_share(rnd, t, start):
    """Each group's share of a total that was stored earlier with `as` and is used inside the block."""
    gi, ms = choose_group(t, rnd), measures(t)
    m = rnd.choice([k for k, v in ms.items() if "raw" in v and not v.get("nuniq")])
    mi = ms[m]
    texts, wheres, sub = sample_filters(t, rnd, rnd.choice([0, 1, 1]))
    if len(sub) < 40:
        raise Skip
    total = mi["fraw"](sub)
    items = grouped_items(sub, gi, lambda d: round(mi["fraw"](d) / total * 100, 1))
    sep = rnd.choice([": ", ": ", " - "])
    pipe, out_txt, out, fmtstr = choose_shape(rnd, items, gi, sep, "%", "X")
    s, tot = rnd.choice(["base", "whole", "full"]), rnd.choice(["total", "tot", "grand"])
    parts = prefix_parts(t, start, wheres, gi, mi.get("rev")) + [
        f"as {s}", f"{s} {mi['raw']} as {tot}",
        f"{s} by {gi['key']} [ {mi['raw']} div {tot} mul 100 round 1 ]"] + pipe + \
        [f'map [ fmt "{fmtstr}" ] print']
    each = rnd.choice(gi["each"])
    clause = (f"computes, for {each}, its share of {mi['phr']} {scope_text(t, texts)}, as a percentage rounded to "
              f"1 decimal place, and then {out_txt}")
    return dict(clause=clause, tok=" ".join(parts), exp=format_lines(out, sep, "%"), pre=[])


# --- `if` inside a block, and `def` ---

def if_chain(ops, names):
    """``if [ ge 1000 ] [ "large" ] [ if [ ge 200 ] [ "medium" ] [ "small" ] ]`` for (op, threshold) pairs."""
    (op, cut), rest = ops[0], ops[1:]
    otherwise = if_chain(rest, names[1:]) if rest else f'"{names[-1]}"'
    return f'if [ {op} {cut} ] [ "{names[0]}" ] [ {otherwise} ]'


def classify(value, ops, names):
    for (op, cut), name in zip(ops, names):
        if OPS[op](value, cut):
            return name
    return names[-1]


def class_rule(rnd, cuts):
    """Thresholds and class names for 2 or 3 classes: returns (ops, names, rule text)."""
    k = len(cuts) + 1
    names = rnd.choice(HIGH_LOW if k == 2 else HML)
    mode = rnd.choice(["ge", "ge", "gt"]) if k == 2 else rnd.choice(["ge", "lt"])
    if mode == "lt":                     # smallest class first
        names = names[::-1]
        ops = [("lt", c) for c in sorted(cuts)]
        words = [f"`{n}` if it is less than {c}" for n, (_, c) in zip(names, ops)]
    else:
        ops = [(mode, c) for c in sorted(cuts, reverse=True)]
        ph = "at least" if mode == "ge" else "greater than"
        words = [f"`{n}` if it is {ph} {c}" for n, (_, c) in zip(names, ops)]
    text = ", ".join(words) + (", and " if k > 2 else " and ") + f"`{names[-1]}` otherwise"
    return ops, list(names), text


def maybe_def(rnd, rule, names_pool):
    """Either an inline ``if`` or a ``def`` line plus a call. Returns (pre lines, block code, extra clause text)."""
    if rnd.random() < 0.6:
        word = rnd.choice(names_pool)
        extra = f", using your own word `{word}` for the rule" if rnd.random() < 0.3 else ""
        return [f"def {word} [ {rule} ]"], word, extra
    return [], rule, ""


def piece_classify(rnd, t, start):
    """Sorts the rows into classes by a value (if, often inside a def) and counts them with freq."""
    kind = rnd.choice(["rev", "rev", "price", "qty", "stat"])
    num = {"rev": None, "price": t.price, "qty": t.qty, "stat": t.stat}[kind]
    if kind == "rev" and not (t.qty and t.price):
        raise Skip
    if kind != "rev" and num is None:
        raise Skip
    texts, wheres, sub = sample_filters(t, rnd, rnd.choice([0, 1, 1, 2]))
    if len(sub) < 40:
        raise Skip
    series = sub.rev if kind == "rev" else sub[num.name]
    k = rnd.choice([2, 3, 3])
    whole = pd.api.types.is_integer_dtype(series)
    cuts = {round_nice(series.quantile(q), whole) for q in ([0.55] if k == 2 else [0.3, 0.75])}
    if len(cuts) != k - 1:
        raise Skip
    ops, names, rule_text = class_rule(rnd, list(cuts))
    counts = {}
    for v in series:
        counts[classify(v, ops, names)] = counts.get(classify(v, ops, names), 0) + 1
    if len(counts) != k:
        raise Skip
    by_count = rnd.random() < 0.6
    if by_count and len(set(counts.values())) != k:
        raise Skip
    pre, call, extra = maybe_def(rnd, if_chain(ops, names), DEF_NAMES)
    out = sorted(counts.items(), key=(lambda kv: -kv[1]) if by_count else (lambda kv: kv[0]))
    sep = rnd.choice([": ", ": ", " - "])
    parts = [start] + wheres
    if kind == "rev":
        parts += [f"set rev [ {t.qty.name} mul {t.price.name} ]", "each rev"]
        value_txt = f"its value `{t.qty.name}` × `{t.price.name}`"
    else:
        parts.append(f"each {num.name}")
        value_txt = f"its `{num.name}`"
    parts += [f"map [ {call} ]", "freq"] + ([] if by_count else ["sort 0"]) + [f'map [ fmt "{{0}}{sep}{{1}}" ] print']
    order = "most numerous first" if by_count else "alphabetically by class"
    hint = f"class{sep}count"
    scope = f" {' and '.join(texts)}" if texts else ""
    clause = (f"classifies every one of the {t.rows}{scope} by {value_txt}: {rule_text}{extra}. "
              f"Print the number of {t.rows} in each class, {order}, in the format `{hint}`")
    return dict(clause=clause, tok=" ".join(parts), exp=format_lines(out, sep, ""), pre=pre)


def piece_label(rnd, t, start):
    """Computes a value for each group and labels it with an `if` (inline or in a def)."""
    gi, ms = choose_group(t, rnd), measures(t)
    m = rnd.choice([k for k, v in ms.items() if v.get("nuniq") != gi["key"]])
    mi = ms[m]
    texts, wheres, sub = sample_filters(t, rnd, rnd.choice([0, 1, 1]))
    if len(sub) < 40:
        raise Skip
    items = grouped_items(sub, gi, mi["f"], min_groups=4)
    vs = sorted(v for _, v in items)
    k = rnd.choice([2, 3])
    cuts = {round_nice(vs[len(vs) * q // 10]) for q in ([5] if k == 2 else [3, 7])}
    if len(cuts) != k - 1:
        raise Skip
    ops, names, rule_text = class_rule(rnd, list(cuts))
    labelled = [(key, classify(v, ops, names)) for key, v in items]
    if len({lab for _, lab in labelled}) != k:
        raise Skip
    out = sorted(labelled, key=lambda kv: kv[0])
    pre, call, extra = maybe_def(rnd, if_chain(ops, names), DEF_NAMES)
    sep = rnd.choice([": ", ": ", " - "])
    parts = prefix_parts(t, start, wheres, gi, mi.get("rev")) + [
        by_or_group(rnd, gi, mi["blk"]), f"map [ {call} ]", "pairs sort 0", f'map [ fmt "{{0}}{sep}{{1}}" ] print']
    each = rnd.choice(gi["each"])
    clause = (f"computes {mi['phr']} for {each}{considering(t, texts)} and labels each result: {rule_text}{extra}. "
              f"Print the labels {gi['sort']} by {gi['noun']}, in the format `{gi['key']}{sep}label`")
    return dict(clause=clause, tok=" ".join(parts), exp=format_lines(out, sep, ""), pre=pre)


def piece_scaled(rnd, t, start):
    """A value per group multiplied by a factor with a word of your own (def), or inline."""
    nums = [n for n in (t.price, t.stat) if n]
    num = rnd.choice(nums)
    gi = choose_group(t, rnd)
    agg, aggword = rnd.choice([("max", "highest"), ("min", "lowest"), ("mean", "average")])
    factor = rnd.choice([1.1, 1.2, 1.25, 1.5, 0.9, 0.8, 2])
    texts, wheres, sub = sample_filters(t, rnd, rnd.choice([0, 1, 1]))
    if len(sub) < 40:
        raise Skip

    def value(d):
        return round(float(getattr(d[num.name], agg)()) * factor, 2)
    items = grouped_items(sub, gi, value)
    sep = rnd.choice([": ", ": ", " - "])
    pipe, out_txt, out, fmtstr = choose_shape(rnd, items, gi, sep, "", "value")
    step = f"mul {factor} round 2"
    word = rnd.choice(SCALE_NAMES)
    use_def = rnd.random() < 0.6
    pre = [f"def {word} [ {step} ]"] if use_def else []
    extra = ""
    if use_def and rnd.random() < 0.3:
        extra = f" (define your own word `{word}` for the multiplication and rounding)"
    parts = prefix_parts(t, start, wheres, gi) + [
        by_or_group(rnd, gi, f"each {num.name} {agg} {word if use_def else step}")] + pipe + \
        [f'map [ fmt "{fmtstr}" ] print']
    phr = f"the {aggword} {num.noun} multiplied by {factor} and rounded to 2 decimal places{extra}"
    clause = per_clause(rnd, gi, phr, considering(t, texts), out_txt, "", verb=pick(rnd, "computes", "calculates"))
    return dict(clause=clause, tok=" ".join(parts), exp=format_lines(out, sep, ""), pre=pre)


# ---------- assembling examples ----------

OLD_PIECES = [(piece_grouped, 3.0), (piece_scalar, 1.5), (piece_distinct, 1.0)]
NEW_PIECES = [(piece_share, 1.0), (piece_ratio, 1.0), (piece_total_share, 0.8), (piece_classify, 1.4),
              (piece_label, 1.2), (piece_scaled, 0.8)]
PIECES = OLD_PIECES + NEW_PIECES


def draw_piece(rnd):
    pieces, weights = zip(*PIECES)
    return rnd.choices(pieces, weights)[0]


def build_example(rnd, t):
    r = rnd.random()
    if r < 0.09:
        kinds = [piece_save]
    elif r < 0.40:
        kinds = [draw_piece(rnd) for _ in range(2)]
    else:
        kinds = [draw_piece(rnd)]
    single = len(kinds) == 1
    use_o = (not single) or rnd.random() < 0.35
    start = "o" if use_o else f"csv {t.file}"
    pieces = [k(rnd, t, start) for k in kinds]
    defs = [line for p in pieces for line in p["pre"]]
    if len({line.split()[1] for line in defs}) != len(defs):
        raise Skip                      # two pieces would define the same word
    if single:
        clause = pieces[0]["clause"]
        if rnd.random() < 0.2 and pieces[0].get("csv") is None:
            clause = f"reads `{t.file}` and {clause}"
        task = f"Write a Ken program that {clause}."
    else:
        items = ";\n".join(f"{i}. {p['clause']}" for i, p in enumerate(pieces, 1))
        task = f"Write a Ken program that:\n{items}."
    lines = defs + ([f"csv {t.file} as o"] if use_o else []) + [p["tok"] for p in pieces]
    return dict(task=task, program="\n".join(lines), exp="\n".join(line for p in pieces for line in p["exp"]),
                csv=next((p["csv"] for p in pieces if "csv" in p), None), table=t.key)


def generate(rnd, tables, banned, seen, stats, work, want, accept):
    """Draw examples for ``tables`` until ``accept`` has been called ``want`` times."""
    done = 0
    while done < want:
        t = TABLES[rnd.choices(tables, [3 if k == "orders" else 2 for k in tables])[0]]
        try:
            ex = build_example(rnd, t)
        except Skip:
            stats["skip"] += 1
            continue
        # one example per task text and per program: the same program worded differently must not end up in
        # both the training set and a test set
        if ex["task"] in seen or ex["program"] in seen:
            stats["dup"] += 1
            continue
        if any(same_out(ex["exp"], b) for b in banned):
            stats["leak"] += 1
            continue
        keep = {TABLES[k].file for k in TABLES} | {"orders.csv"}
        for f in work.iterdir():
            if f.name not in keep:
                f.unlink()
        ok, out = run_in_process(ex["program"], work)
        if not ok:
            stats["crash"] += 1
            print("CRASH", ex["program"], "->", out, file=sys.stderr)
            continue
        good = same_out(out, ex["exp"])
        if good and ex["csv"]:
            good = check_csv_spec(work / ex["csv"]["file"], ex["csv"])[0]
        if not good:
            stats["mismatch"] += 1
            print("MISMATCH", ex["program"], "\n got:", out[:200], "\n exp:", ex["exp"][:200], file=sys.stderr)
            continue
        seen.update((ex["task"], ex["program"]))
        accept(ex)
        done += 1


def main():
    n_train = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    rnd = random.Random(20261005)
    banned = [expected(i)[0] for i in range(1, 8)]
    seen, rows, unseen = set(), [], []
    stats = dict(skip=0, dup=0, leak=0, mismatch=0, crash=0)

    def split_of(ex):
        h = int(hashlib.md5(ex["task"].encode()).hexdigest(), 16) % 100
        return "test" if h < 4 else "val" if h < 8 else "train"

    with scratch_dir(*(t.csv for t in TABLES.values())) as work:
        def add(ex):
            ex["split"] = split_of(ex)
            rows.append(ex)
        # draw until enough training examples exist (val and test come along by hash of the task)
        while sum(r["split"] == "train" for r in rows) < n_train:
            generate(rnd, list(TRAIN_TABLES), banned, seen, stats, work, 50, add)
        generate(rnd, [HELD_OUT_TABLE], banned, seen, stats, work, 60, unseen.append)
    out_dir = HERE / "data"
    out_dir.mkdir(exist_ok=True)
    sets = {s: [r for r in rows if r["split"] == s] for s in ("train", "val", "test")}
    sets["test_unseen_table"] = unseen
    for name, sel in sets.items():
        with open(out_dir / f"{name}.jsonl", "w", encoding="utf-8", newline="\n") as fh:
            for i, r in enumerate(sel):
                fh.write(json.dumps(dict(id=f"{name}{i}", table=r["table"], task=r["task"], program=r["program"],
                                         expected=r["exp"], csv=r["csv"]), ensure_ascii=False) + "\n")
        print(name, len(sel))
    print("stats:", stats)


if __name__ == "__main__":
    main()
