# Ken language reference (v0.1)

Ken is a postfix pipeline language. A program is a flat sequence of words; each word acts on a
single **current value**. The compiler translates a program to Python, so every Python
library is available (see [Interop](#interop-with-python)).

> This document is the specification of version 0.1. The fine-tuned model released with this
> repository was trained on exactly this version. Behaviour described here is covered by the test suite.

Contents: [Execution model](#execution-model) · [Lexical structure](#lexical-structure) ·
[Values and variables](#values-and-variables) · [Blocks](#blocks) · [Commands](#commands) ·
[Definitions](#definitions) · [Interop with Python](#interop-with-python) ·
[Output formatting](#output-formatting) · [Errors](#errors) · [Pitfalls](#pitfalls) ·
[Security](#security) · [Grammar](#grammar)

## Execution model

At any moment a program has

- a **current value** `_` (initially `null`), and
- a **variable table** `V` (initially empty).

Each word, read left to right, either *replaces* the current value or *transforms* it. A command
reads the current value and takes a fixed number of arguments that follow it **after** the command
(`get 0`, never `0 get`). Its result becomes the new current value.

```
upto 10 filter [ mod 2 eq 0 ] sum print
```

| Word | Current value afterwards |
|---|---|
| `upto 10` | `[1, 2, ..., 10]` |
| `filter [ mod 2 eq 0 ]` | `[2, 4, 6, 8, 10]` |
| `sum` | `30` |
| `print` | `30` (and prints it) |

`kenlang py program.ken` shows the generated Python: one statement `_ = rt.command(_, V, args...)`
per word.

## Lexical structure

- Tokens are separated by whitespace; line breaks have no meaning.
- `#` starts a comment that runs to the end of the line (unless it is inside a quoted string).
- `"..."` is a text literal. It may contain spaces and `#` but not another `"`. The two characters
  `\n` inside it become a line break.
- `[` and `]` delimit a block and must be separate tokens (surround them with spaces).
- Everything else is a *word*: a number, `null`, a command name, a variable name, a bare text value,
  or a library reference (`np.std`, `.sort_values:1`, `title=Revenue`).

## Values and variables

A literal is interpreted as follows:

| Token | Value |
|---|---|
| `42`, `-3` | integer |
| `2.5` | float |
| `null` | `None` |
| `"two words"` | text |
| anything else | text, **unless** it names a command or a variable (below) |

A *bare word* that is not a command is looked up: if a variable of that name exists, its value
becomes the current value; otherwise the word is text (`abc` is the same as `"abc"`). The same
resolution applies to **arguments** of commands: `where status eq done` compares with the text
`done` unless a variable named `done` exists (see [Pitfalls](#pitfalls)).

- A lone value replaces the current value. This is how a new pipeline starts:
  `"Heading:" print`, `all where status eq done`.
- `as name` stores the current value under `name`; the pipeline continues unchanged.
- Variables stored **inside a block** are visible only inside that block (the block works on a copy
  of the table).

## Blocks

`[ ... ]` is code that a command runs later, once per element or once per table. Inside a block
the current value is the element, and, if the element is a row (dict) or a table, its fields
(columns) are available as variables. What a block receives depends on the command:

| Command | Block runs | Block receives |
|---|---|---|
| `map` on a list or Series | once per element | the element |
| `map` on a dict | once per value; keys are kept | the value (e.g. a group table) |
| `filter` on a list | once per element; keeps those where the result is truthy | the element |
| `filter` on a table | once | the whole table; result must be a boolean column |
| `set` on a table | once | the whole table; result becomes the new column |
| `set` on a list of dicts | once per row | the row |
| `by field [block]` | once per group | the group as a table |
| `if [cond] [then] [else]` | `cond` once, then one of the other two | the current value |

Consequences: arithmetic and comparisons work inside `set` and `filter` on a table (they act on
whole columns), but `if` does not (it needs a single truth value). To classify rows, extract a column
first and `map` over it.

## Commands

The number in parentheses is the number of arguments. "table" means a pandas `DataFrame`, "series"
a `Series`. Commands also work on plain Python lists and dicts unless noted.

### Sources

| Command | Description |
|---|---|
| `csv` (1) | Read a CSV file with a header into a table. Requires pandas. |
| `read` (1) | Read a text file into one string. |
| `lines` (0) | Split text into a list of lines. |
| `words` (0) | Split text into a list of words (letters, digits, `_` and `'`). |
| `json` (0) | Parse JSON text, or read the file if the value ends with `.json`. |
| `upto` (1) | `upto n` gives `[1, 2, ..., n]`. |
| `list` (1) | `list 1,2,x` gives `[1, 2, "x"]`; numeric-looking items become numbers. |

### Selecting

| Command | Description |
|---|---|
| `where` (3) | `where field operator value` keeps matching rows. Operators: `eq ne gt ge lt le`, `in` / `nin` (value is a list `a,b,c`), `has` (the field's text contains the fragment). |
| `filter` (1) | Keep elements for which the block is true (see [Blocks](#blocks)). |
| `drop` (1) | Remove every occurrence of a value; `drop null` removes missing values. |
| `pick` (1) | `pick a,b` keeps only those columns (table) or keys (list of dicts), in that order. |
| `get` (1) | Field or index of one element: `get 0` on a pair, `get name` on a row. |

### Shaping

| Command | Description |
|---|---|
| `each` (1) | `each field` extracts a column (table) or one field of every element (list). |
| `set` (2) | `set name [block]` adds or replaces a column. |
| `map` (1) | Transform every element or dict value with a block. |
| `group` (1) | `group field` gives a dict from group key to the rows of that group, in order of first appearance. |
| `by` (2) | `by field [block]` is `group field map [block]`: a dict from key to the block's result. |
| `pairs` (0) | Dict (or series) to a list of `[key, value]` pairs. |
| `keys`, `vals` (0) | Keys or values of a dict. |
| `slice` (2) | `slice a b` takes `[a:b]` of a text, of every text in a list, or of a text column. |

### Ordering

| Command | Description |
|---|---|
| `sort` (1), `rsort` (1) | Ascending / descending by a field (table, list of dicts) or by index (list of pairs). Use `it` for the elements themselves: `sort it`. The sort is stable. |
| `top` (1) | The `n` largest. From a dict or series it yields `[key, value]` pairs, largest value first; from a list it yields the elements. |
| `first`, `last` (1) | The first or last `n` elements or rows. |
| `rev` (0) | Reverse a list. |
| `uniq` (0) | Remove duplicates, keeping the first occurrence of each element. |
| `freq` (0) | List of `[element, count]` pairs, most frequent first (ties keep first-seen order). |

### Aggregates

| Command | Description |
|---|---|
| `sum`, `mean`, `max`, `min` (0) | Aggregate a list, series or column. |
| `count` (0) | Number of elements or rows. |
| `nuniq` (0) | Number of distinct values (of a series, list, or distinct rows of a table). |
| `round` (1) | Round a number, a list of numbers, or a column to `n` decimals. |

### Arithmetic and comparison

`add` `sub` `mul` `div` `mod` `pow` and `eq` `ne` `gt` `ge` `lt` `le` (1 each) combine the current value with one
argument. They work on a number, on a list (element by element) and on a table column (vectorised).
`div` is true division.

### Text

| Command | Description |
|---|---|
| `lower`, `upper` (0) | Change case of a text, a list of texts, or a text column. |
| `split` (1) | Split a text on a separator (not for columns). |
| `join` (1) | Join a list into one text with a separator. |
| `fmt` (1) | Format with a template. A list or pair fills `{0} {1} ...`; a row fills `{field}`. Python format specs work: `{1:.2f}`, `{0:>8}`. Without a spec, numbers are printed as by `print`. |

### Control and output

| Command | Description |
|---|---|
| `if` (3) | `if [cond] [then] [else]` evaluates `cond` on the current value and returns the result of `then` or `else` (each also run on the current value). |
| `print` (0) | Print the value and pass it on. See [Output formatting](#output-formatting). |
| `save` (1) | Write the value to a file and pass it on. A table is saved as CSV without index, a list line by line, a dict or series as `key,value` lines, anything else as text. |

## Definitions

`def name [ ... ]` defines a new word. It takes no arguments and acts on the current value, like a
command; the body is an ordinary block. A word must be defined before it is used.

```
def verdict [ if [ ge 50 ] [ "pass" ] [ "retake" ] ]
list 72,38,50,91 map [ verdict ] print
```

## Interop with Python

| Syntax | Meaning |
|---|---|
| `use module` / `use module as alias` | `import module [as alias]`. |
| `alias.function` | Call `alias.function(_, ...)`: the current value is the **first** argument. The number of extra positional arguments is read from the function's signature (required parameters minus one). |
| `alias.function:N` | State the number of extra positional arguments explicitly: `np.percentile:1 90`. |
| `.method` | Call a method of the current value, or read an attribute if it is not callable: `.sum`, `.plot.barh`, `.columns`. Arguments: 0 by default. |
| `.method:N` | Take `N` positional arguments: `.groupby:1 category`, `.head:1 5`. |
| `name=value` | Keyword argument; must follow directly after an `alias.function` or `.method` and its positional arguments. No spaces around `=`. Use `0`/`1` for booleans. |

```
use numpy as np
ok each rev np.percentile:1 90 round 2 print
all .pivot_table values=qty index=category columns=status aggfunc=count fill_value=0 print
```

## Output formatting

`print` shows the value according to its type:

- a **table** is printed with its columns (the row index is shown only if it is not the default);
- a **dict** or **series** prints one `key: value` line per entry;
- a **list** prints one element per line; a nested list (a pair) prints its items separated by a space;
- a **float** is printed with up to six decimals and trailing zeros removed (`2.50` prints `2.5`,
  `4 div 2` prints `2`);
- `null` prints nothing.

## Errors

- **Compile time** — `KenSyntaxError` (a `SyntaxError`) for unbalanced brackets, a stray `]`, a
  command that runs out of program before getting its arguments, `def` without `[`, and `use` of a
  module that cannot be imported. The CLI prints `tok: syntax error: ...` and exits with status 2.
- **Run time** — ordinary Python exceptions from the generated code (a missing column is a
  `KeyError`, calling `split` on a table is an `AttributeError`, and so on).

## Pitfalls

These are properties of version 0.1 that surprise newcomers (and models).

1. **Arguments are literal.** A command argument is a number, a text, a block or a variable name; it
   is never an expression. `where qty gt price` does not compare two columns. Compute a column with
   `set` first.
2. **A variable name wins over text.** In `o where product has o`, the last `o` is the variable `o`
   (the table), not the text `o`. Avoid variable names that are also values you pass as arguments.
3. **`if` does not work in `set` or `filter` on a table**, because the block gets the whole table
   (see [Blocks](#blocks)). Extract the column and `map` over it instead.
4. **`group` needs `map`; `by` does not.** `group category [ ... ]` is wrong; write
   `by category [ ... ]` or `group category map [ ... ]`.
5. **Text commands on columns.** `split` and `join` work on single texts, not on columns.
   Use `slice`, `lower`, `upper` for columns.
6. **Order of groups** is the order of first appearance in the data, not sorted. Sort explicitly.
7. **Ties.** `top`, `sort`, `rsort` and `freq` are stable: equal values keep their original order.

## Security

A Ken program is a Python program in disguise: `use` imports any module and `.method` calls any
method of any object, so a program can do anything Python can. **Never run Ken code from an
untrusted source** (including code written by a language model) outside a sandbox. The evaluation
tools in this repository run model-written programs in a separate process in an empty temporary
directory; that is isolation of mistakes, not a security boundary.

## Grammar

```
program   = { word } ;
word      = definition | import | store | command | call | method | value ;
definition = "def" NAME "[" program "]" ;
import    = "use" MODULE [ "as" ALIAS ] ;
store     = "as" NAME ;
command   = NAME { argument }            (* as many arguments as the command's arity *)
call      = ALIAS "." PATH [ ":" INT ] { argument } { KEYWORD "=" literal } ;
method    = "." PATH [ ":" INT ] { argument } { KEYWORD "=" literal } ;
argument  = literal | block ;
block     = "[" program "]" ;
literal   = NUMBER | "null" | TEXT | WORD ;
comment   = "#" { any character except newline } ;
```

`NAME` is a built-in command, a user-defined word or a variable. The arity of every built-in
command is the number in parentheses in [Commands](#commands).
