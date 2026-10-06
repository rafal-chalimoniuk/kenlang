"""Ken runtime: the commands that generated code calls.

Every command takes the current pipeline value ``it`` and the variable table ``V``, followed by
the command's own arguments, and returns the new pipeline value. Commands work on plain Python
lists and dicts as well as on pandas ``DataFrame`` and ``Series`` objects (pandas is optional,
but required for ``csv``). Library functions of user-imported modules are called directly by the
generated code and never pass through this module.
"""
import builtins as B
import json as _json
import os
import re
import string
from collections import Counter

os.environ.setdefault("MPLBACKEND", "Agg")   # render plots to files, never open a window

try:
    import pandas as pd
    DF, SER = pd.DataFrame, pd.Series
except ImportError:                           # Ken also works without pandas
    pd = None
    DF = SER = type("_None", (), {})


# ---------- helpers ----------

def arg(V, x):
    """An argument that names a variable is replaced by that variable's value."""
    return V[x] if isinstance(x, str) and x in V else x


def load(V, x):
    return arg(V, x)


def field(x, f):
    """Select field ``f`` of an element: ``it`` is the element itself, an int indexes a pair or
    list, a dotted path walks nested containers."""
    if f == "it":
        return x
    if isinstance(f, int):
        return x[f]
    for part in str(f).split("."):
        x = x[int(part)] if isinstance(x, (list, tuple)) else x[part]
    return x


def row_env(V, x):
    """Inside a block, the fields of a row (or the columns of a table) are visible as variables."""
    if isinstance(x, dict):
        return {**V, **x}
    if isinstance(x, DF):
        return {**V, **{c: x[c] for c in x.columns}}
    return dict(V)


def auto(v):
    """Convert numeric-looking text to int or float."""
    if isinstance(v, str):
        for conv in (int, float):
            try:
                return conv(v)
            except ValueError:
                pass
    return v


def _items(v):
    """List argument of ``in``/``nin``/``pick``: a list is used as is, "a,b,c" is split on commas."""
    return v if isinstance(v, list) else [auto(x) for x in str(v).split(",")]


CMP = {
    "eq": lambda a, b: a == b, "ne": lambda a, b: a != b,
    "gt": lambda a, b: a > b, "ge": lambda a, b: a >= b,
    "lt": lambda a, b: a < b, "le": lambda a, b: a <= b,
}


def elementwise(fn):
    """On a list: element by element. On a Series or a number: the operator directly
    (pandas applies it vectorised)."""
    def op(it, V, n):
        n = arg(V, n)
        if isinstance(it, list):
            return [fn(x, n) for x in it]
        return fn(it, n)
    return op


# ---------- sources ----------

def read(it, V, p):
    return open(arg(V, p), encoding="utf-8").read()


def csv(it, V, p):
    return pd.read_csv(arg(V, p))


def json(it, V):
    return _json.load(open(it)) if str(it).endswith(".json") else _json.loads(it)


def upto(it, V, n):
    return list(range(1, arg(V, n) + 1))


def list_(it, V, s):
    return [auto(x) for x in str(s).split(",")]


def lines(it, V):
    return it.splitlines()


def words(it, V):
    return re.findall(r"[\w']+", it)


# ---------- selecting and shaping ----------

def where(it, V, f, op, v):
    v = arg(V, v)
    if isinstance(it, DF):
        col = it[f]
        if op in ("in", "nin"):
            mask = col.isin(_items(v))
            return it[~mask if op == "nin" else mask]
        if op == "has":
            return it[col.astype(str).str.contains(str(v), regex=False)]
        return it[CMP[op](col, v)]
    if op in ("in", "nin"):
        vs = _items(v)
        return [x for x in it if (field(x, f) in vs) == (op == "in")]
    if op == "has":
        return [x for x in it if str(v) in str(field(x, f))]
    return [x for x in it if CMP[op](field(x, f), v)]


def filter(it, V, blk):
    if isinstance(it, DF):
        return it[blk(it, row_env(V, it))]           # mask computed vectorised
    return [x for x in it if blk(x, row_env(V, x))]


def drop(it, V, v):
    if isinstance(it, (DF, SER)):
        return it.dropna() if v is None else it[it != v]
    return [x for x in it if x != v]


def each(it, V, f):
    if isinstance(it, DF):
        return it[f]
    return [field(x, f) for x in it]


def set(it, V, f, blk):
    if isinstance(it, DF):
        return it.assign(**{f: blk(it, row_env(V, it))})
    return [{**x, f: blk(x, row_env(V, x))} for x in it]


def map(it, V, blk):
    if isinstance(it, dict):
        return {k: blk(v, row_env(V, v)) for k, v in it.items()}
    if isinstance(it, SER):
        return it.map(lambda x: blk(x, V))
    return [blk(x, row_env(V, x)) for x in it]


def group(it, V, f):
    if isinstance(it, DF):
        return {k: g for k, g in it.groupby(f, sort=False)}
    out = {}
    for x in it:
        out.setdefault(field(x, f), []).append(x)
    return out


def by(it, V, f, blk):
    """``group`` followed by ``map``: a dict from group key to the block's result for that group."""
    return map(group(it, V, f), V, blk)


def top(it, V, n):
    """The n largest: [key, value] pairs for a dict or Series, elements for a list."""
    if isinstance(it, (dict, SER)):
        return [list(p) for p in sorted(it.items(), key=lambda p: p[1], reverse=True)[:n]]
    return sorted(it, reverse=True)[:n]


def pick(it, V, cols):
    cols = _items(cols)
    if isinstance(it, DF):
        return it[cols]
    return [{c: x[c] for c in cols} for x in it]


def get(it, V, f):
    return field(it, f)


def nuniq(it, V):
    if isinstance(it, SER):
        return it.nunique()
    if isinstance(it, DF):
        return len(it.drop_duplicates())
    return len(B.set(it))


def save(it, V, p):
    """Write the value to a file and pass it on: a table as CSV, a dict or Series as
    ``key,value`` lines, a list as lines, anything else as text."""
    p = arg(V, p)
    if isinstance(it, DF):
        it.to_csv(p, index=False)
        return it
    if isinstance(it, (dict, SER)):
        text = "".join(f"{k},{fmt_value(v)}\n" for k, v in it.items())
    elif isinstance(it, list):
        text = "".join(fmt_value(x) + "\n" for x in it)
    else:
        text = fmt_value(it)
    with open(p, "w", encoding="utf-8") as fh:
        fh.write(text)
    return it


def _sort(it, f, rev):
    if isinstance(it, DF):
        return it.sort_values(f, ascending=not rev)
    if isinstance(it, SER):
        return it.sort_values(ascending=not rev)
    return sorted(it, key=lambda x: field(x, f), reverse=rev)


def sort(it, V, f):
    return _sort(it, f, False)


def rsort(it, V, f):
    return _sort(it, f, True)


def first(it, V, n):
    return it.head(n) if isinstance(it, (DF, SER)) else it[:n]


def last(it, V, n):
    return it.tail(n) if isinstance(it, (DF, SER)) else it[-n:]


def rev(it, V):
    return it[::-1]


def uniq(it, V):
    return list(dict.fromkeys(it))


def freq(it, V):
    return [list(p) for p in Counter(it).most_common()]


def pairs(it, V):
    return [list(p) for p in it.items()]


def keys(it, V):
    return list(it.keys())


def vals(it, V):
    return list(it.values())


def slice(it, V, a, b):
    if isinstance(it, SER):
        return it.str[a:b]
    if isinstance(it, list):
        return [x[a:b] for x in it]
    return it[a:b]


# ---------- aggregates ----------

def _agg(name, fallback):
    def op(it, V):
        if isinstance(it, (DF, SER)):
            return getattr(it, name)()
        return fallback(it)
    return op


sum = _agg("sum", B.sum)
mean = _agg("mean", lambda xs: __import__("statistics").fmean(xs))
max = _agg("max", B.max)
min = _agg("min", B.min)


def count(it, V):
    return len(it)


# ---------- arithmetic and comparison ----------

add = elementwise(lambda a, b: a + b)
sub = elementwise(lambda a, b: a - b)
mul = elementwise(lambda a, b: a * b)
div = elementwise(lambda a, b: a / b)
mod = elementwise(lambda a, b: a % b)
pow = elementwise(lambda a, b: a ** b)
eq, ne, gt, ge, lt, le = (elementwise(CMP[k]) for k in ("eq", "ne", "gt", "ge", "lt", "le"))


def round(it, V, n):
    if isinstance(it, list):
        return [B.round(x, n) for x in it]
    if isinstance(it, (DF, SER)):
        return it.round(n)
    return B.round(float(it), n)


# ---------- text ----------

def lower(it, V):
    return it.lower() if isinstance(it, str) else it.str.lower() if isinstance(it, SER) else [s.lower() for s in it]


def upper(it, V):
    return it.upper() if isinstance(it, str) else it.str.upper() if isinstance(it, SER) else [s.upper() for s in it]


def split(it, V, s):
    return it.split(s)


def join(it, V, s):
    return s.join(fmt_value(x) for x in it)


class _Fmt(string.Formatter):
    """``{0}`` renders like ``print``; with a format spec (``{0:.2f}``) the raw value is formatted."""

    def format_field(self, value, spec):
        return format(value, spec) if spec else fmt_value(value)


def fmt(it, V, t):
    if isinstance(it, dict):
        return _Fmt().vformat(t, (), it)
    if isinstance(it, (list, tuple)):
        return _Fmt().vformat(t, it, {})
    return _Fmt().vformat(t, (it,), {})


# ---------- control flow and output ----------

def if_(it, V, cond, a, b):
    return a(it, V) if cond(it, row_env(V, it)) else b(it, V)


def dot(it, V, path, args, kw):
    """A method or attribute of the pipeline value: ``.groupby:1 city``, ``.plot.bar``.
    An attribute yields its value; a method is called."""
    obj = it
    for name in path.split("."):
        obj = getattr(obj, name)
    if callable(obj):
        return obj(*[arg(V, a) for a in args], **{k: arg(V, v) for k, v in kw.items()})
    return obj


def call(it, V, fn, args, kw):
    """A library function: the pipeline value is passed as the first argument."""
    return fn(it, *[arg(V, a) for a in args], **{k: arg(V, v) for k, v in kw.items()})


def fmt_value(x):
    if isinstance(x, float):
        return f"{x:.6f}".rstrip("0").rstrip(".")
    if isinstance(x, (list, tuple)):
        return " ".join(fmt_value(v) for v in x)
    return str(x)


def print_(it, V):
    if isinstance(it, DF):
        default = isinstance(it.index, pd.RangeIndex)
        print(it.to_string(index=not default))
    elif isinstance(it, (dict, SER)):
        for k, v in it.items():
            print(f"{k}: {fmt_value(v)}")
    elif isinstance(it, list):
        for x in it:
            print(fmt_value(x))
    elif it is not None:
        print(fmt_value(it))
    return it
