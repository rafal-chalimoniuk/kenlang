"""Command-line interface: ``kenlang run``, ``kenlang py``."""
import argparse
import pathlib
import sys

from . import __version__
from .compiler import run, translate
from .errors import KenSyntaxError


def _source(args):
    if args.expr is not None:
        return args.expr
    if args.file in (None, "-"):
        return sys.stdin.read()
    return pathlib.Path(args.file).read_text(encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="kenlang", description="Ken: an experimental programming language that compiles to Python.")
    parser.add_argument("--version", action="version", version=f"kenlang {__version__}")
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name, help_ in (("run", "run a Ken program"), ("py", "print the Python a Ken program compiles to")):
        p = sub.add_parser(name, help=help_)
        p.add_argument("file", nargs="?", help="path to a .ken file, or - for standard input")
        p.add_argument("-e", "--expr", help="program text given on the command line")
    args = parser.parse_args(argv)
    if args.expr is None and args.file is None and sys.stdin.isatty():
        parser.error("give a file, -e CODE, or pipe a program to standard input")
    try:
        src = _source(args)
        if args.cmd == "py":
            sys.stdout.write(translate(src))
        else:
            run(src)
    except KenSyntaxError as e:
        print(f"kenlang: syntax error: {e}", file=sys.stderr)
        return 2
    except OSError as e:
        print(f"kenlang: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
