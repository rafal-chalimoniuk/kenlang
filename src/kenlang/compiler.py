"""Ken compiler: translates a Ken program into plain Python source.

A program is a sequence of whitespace-separated words that act on one *current value*. Every
word either replaces that value or transforms it. The compiler turns each word into one Python
statement of the form ``_ = rt.command(_, V, args...)`` where ``_`` is the current value and ``V``
the variable table; blocks (``[ ... ]``) become nested functions.
"""
import importlib
import inspect
import re

from .errors import KenSyntaxError

# `#` starts a comment that runs to the end of the line.
TOKEN_RE = re.compile(r'"[^"]*"|#[^\n]*|\[|\]|\S+')
KW_RE = re.compile(r"^[A-Za-z_]\w*=")

# Built-in commands: name -> number of arguments that follow the command.
ARITY = {
    "read": 1, "csv": 1, "json": 0, "lines": 0, "words": 0, "upto": 1, "list": 1,
    "where": 3, "filter": 1, "drop": 1, "each": 1, "set": 2, "map": 1, "group": 1,
    "sort": 1, "rsort": 1, "first": 1, "last": 1, "rev": 0, "uniq": 0, "freq": 0,
    "pairs": 0, "keys": 0, "vals": 0, "slice": 2,
    "by": 2, "top": 1, "pick": 1, "get": 1, "nuniq": 0, "save": 1,
    "sum": 0, "mean": 0, "max": 0, "min": 0, "count": 0, "round": 1,
    "add": 1, "sub": 1, "mul": 1, "div": 1, "mod": 1, "pow": 1,
    "eq": 1, "ne": 1, "gt": 1, "ge": 1, "lt": 1, "le": 1,
    "lower": 0, "upper": 0, "split": 1, "join": 1, "fmt": 1,
    "if": 3, "print": 0,
}
# Commands whose names clash with Python keywords or builtins that the runtime shadows.
PYNAME = {"list": "list_", "if": "if_", "print": "print_"}


def literal(t):
    """Interpret a token as a value: quoted text, ``null``, a number, or bare text."""
    if t.startswith('"'):
        return t[1:-1].replace("\\n", "\n")
    if t == "null":
        return None
    for conv in (int, float):
        try:
            return conv(t)
        except ValueError:
            pass
    return t


def split_arity(w):
    """``name:N`` -> (name, N); a word without ``:N`` -> (word, None)."""
    if ":" in w:
        path, n = w.rsplit(":", 1)
        return path, int(n)
    return w, None


def guess_arity(mod, path):
    """How many positional arguments, beyond the pipeline value, a library function requires."""
    obj = mod
    for part in path.split(".")[1:]:
        obj = getattr(obj, part)
    try:
        params = inspect.signature(obj).parameters.values()
    except (TypeError, ValueError):
        return 0
    required = [p for p in params
                if p.default is p.empty and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    return max(len(required) - 1, 0)


class Parser:
    def __init__(self, src):
        self.t = [t for t in TOKEN_RE.findall(src) if not t.startswith("#")]
        self.i = 0
        self.aliases = {}          # alias -> imported module (used to read function signatures)

    def next(self, context=None):
        if self.i >= len(self.t):
            where = f" while reading {context}" if context else ""
            raise KenSyntaxError(f"unexpected end of program{where}")
        tok = self.t[self.i]
        self.i += 1
        return tok

    def peek(self):
        return self.t[self.i] if self.i < len(self.t) else None

    def parse(self, stop=None):
        nodes = []
        while self.peek() is not None:
            w = self.next()
            if w == stop:
                return nodes
            if w == "]":
                raise KenSyntaxError("unexpected ]")
            nodes.append(self.word(w))
        if stop:
            raise KenSyntaxError("missing closing ]")
        return nodes

    def value(self, context=None):
        tok = self.next(context)
        if tok == "[":
            return ("block", self.parse(stop="]"))
        return literal(tok)

    def kwargs(self):
        kw = {}
        while self.peek() and KW_RE.match(self.peek()):
            k, v = self.next().split("=", 1)
            kw[k] = literal(v)
        return kw

    def word(self, w):
        if w == "use":
            mod = self.next("'use'")
            alias = mod.split(".")[0]
            if self.peek() == "as":
                self.next()
                alias = self.next("'use ... as'")
            try:
                self.aliases[alias] = importlib.import_module(mod)
            except ImportError as e:
                raise KenSyntaxError(f"use {mod}: {e}") from None
            return ("use", mod, alias)
        if w == "def":
            name = self.next("'def'")
            if self.next(f"'def {name}'") != "[":
                raise KenSyntaxError(f"def {name}: expected [")
            return ("def", name, ("block", self.parse(stop="]")))
        if w == "as":
            return ("as", self.next("'as'"))
        if w.startswith(".") and len(w) > 1:            # method or attribute of the pipeline value
            path, n = split_arity(w[1:])
            args = [self.value(f"'{w}'") for _ in range(n or 0)]
            return ("dot", path, args, self.kwargs())
        head = w.split(".")[0].split(":")[0]
        if head in self.aliases and "." in w:          # library function
            path, n = split_arity(w)
            if n is None:
                n = guess_arity(self.aliases[head], path)
            args = [self.value(f"'{w}'") for _ in range(n)]
            return ("call", path, args, self.kwargs())
        if w in ARITY:
            return ("op", w, [self.value(f"'{w}'") for _ in range(ARITY[w])])
        return ("load", literal(w))


class Gen:
    """Turns the syntax tree into Python source lines."""

    def __init__(self):
        self.n = 0
        self.defs = set()
        self.imports = []

    def expr(self, v, out, ind):
        """Python expression for an argument; a block becomes a nested function."""
        if isinstance(v, tuple) and v[0] == "block":
            self.n += 1
            name = f"_b{self.n}"
            out.append(f"{ind}def {name}(_, V):")
            out += self.body(v[1], ind + "    ")
            out.append(f"{ind}    return _")
            return name
        return repr(v)   # variable names are resolved by the runtime where the argument is used

    def library_arg(self, a, out, ind):
        """Argument of a library call: a name is resolved at run time, anything else as in `expr`."""
        return f"rt.arg(V, {a!r})" if isinstance(a, str) else self.expr(a, out, ind)

    def body(self, nodes, ind):
        out = []
        for node in nodes:
            kind = node[0]
            if kind == "use":
                _, mod, alias = node
                self.imports.append(f"import {mod}" + (f" as {alias}" if alias != mod.split('.')[0] else ""))
            elif kind == "def":
                _, name, blk = node
                self.defs.add(name)
                out.append(f"{ind}def t_{name}(_, V):")
                out += self.body(blk[1], ind + "    ")
                out.append(f"{ind}    return _")
            elif kind == "as":
                out.append(f"{ind}V[{node[1]!r}] = _")
            elif kind == "load":
                v = node[1]
                if v in self.defs:
                    out.append(f"{ind}_ = t_{v}(_, V)")
                elif isinstance(v, str):
                    out.append(f"{ind}_ = rt.load(V, {v!r})")
                else:
                    out.append(f"{ind}_ = {v!r}")
            elif kind == "op":
                _, name, args = node
                exprs = [self.expr(a, out, ind) for a in args]
                out.append(f"{ind}_ = rt.{PYNAME.get(name, name)}(_, V{''.join(', ' + e for e in exprs)})")
            elif kind == "call":
                _, path, args, kw = node
                exprs = [self.library_arg(a, out, ind) for a in args]
                exprs += [f"{k}={self.library_arg(v, out, ind)}" for k, v in kw.items()]
                out.append(f"{ind}_ = {path}(_{''.join(', ' + e for e in exprs)})")
            elif kind == "dot":
                _, path, args, kw = node
                a = "[" + ", ".join(repr(x) for x in args) + "]"
                out.append(f"{ind}_ = rt.dot(_, V, {path!r}, {a}, {kw!r})")
        return out


def translate(src):
    """Translate Ken source into Python source."""
    nodes = Parser(src).parse()
    g = Gen()
    body = g.body(nodes, "")
    head = ["import kenlang.runtime as rt", *g.imports, "", "V = {}", "_ = None"]
    return "\n".join(head + body) + "\n"


def run(src):
    """Translate and execute a Ken program."""
    code = translate(src)
    exec(compile(code, "<ken>", "exec"), {"__name__": "__ken__"})
