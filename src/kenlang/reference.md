# The Ken language

Ken is a pipeline language that is translated to Python. A program is a sequence of words separated by whitespace; line breaks do not matter. Everything from `#` to the end of the line is a comment. A program has one **current value**. Every word either sets it or transforms it. A command takes the current value and a fixed number of arguments that follow it directly **after** the command (`get 0`, not `0 get`). The result of a command becomes the new current value.

## Values

- Numbers `3`, `2.5`, text in double quotes `"two words"` (no `"` inside), and `null`.
- A word that is not a command is the value of the variable of that name if one exists, and text otherwise (`abc` is the same as `"abc"`).
- A lone value replaces the current value. This is how a new pipeline starts, e.g. `"Heading:" print`.

## Built-in commands (number of arguments in parentheses)

| Group | Commands |
|---|---|
| Sources | `csv` (1) file → pandas table · `read` (1) file → text · `lines` (0) · `words` (0) · `json` (0) · `upto` (1) n → [1..n] · `list` (1) `"a,b,c"` → list |
| Selecting | `where` (3) field operator value · `filter` (1) [block] · `drop` (1) removes a value, `drop null` removes empty ones · `pick` (1) `"a,b"` → only those fields/columns · `get` (1) field or index of a single element |
| Shaping | `each` (1) field → column/list · `set` (2) field [block] → new field · `map` (1) [block] · `group` (1) field → dict group→rows · `by` (2) field [block] = `group` + `map`: dict group→result of the block · `pairs` (0) dict → [key, value] pairs · `keys` `vals` (0) · `slice` (2) from to |
| Ordering | `sort` (1), `rsort` (1) ascending/descending by field or index · `top` (1) the n largest; from a dict it yields [key, value] pairs · `first` `last` (1) · `rev` `uniq` (0) · `freq` (0) → [element, count] pairs, most frequent first |
| Aggregates | `sum` `mean` `max` `min` `count` `nuniq` (number of distinct values) (0) · `round` (1) |
| Arithmetic | `add` `sub` `mul` `div` `mod` `pow` (1) |
| Comparison | `eq` `ne` `gt` `ge` `lt` `le` (1) |
| Text | `lower` `upper` (0) · `split` `join` (1) · `fmt` (1) template: `{0} {1}` for a list/pair, `{field}` for a row; Python format specs work: `{1:.2f}` |
| Control | `if` (3) [condition] [then] [else] · `print` (0) prints and passes the value on · `save` (1) file: writes it (a table as CSV, a list line by line) and passes it on |

The operators of `where` are the comparisons plus `in`/`nin` (the value is / is not in the list `a,b,c`) and `has` (the field's text contains the given fragment). The field `it` means the element itself, and a number as a field is an index into a pair or list. `print` prints a list one element per line, and a dict or a series as `key: value` lines. Arithmetic and comparisons work on a number, on a list (element by element) and on a table column.

## Syntax

- `[ ... ]` is a block, i.e. code that runs later. Inside a block the current value is the element. If the element is a row or a table, its fields (columns) are available as variables. Separate brackets with spaces.
  - `map` and `filter` run the block once per element of a list or series; `map` on a dict transforms the values and keeps the keys.
  - `set` and `filter` on a table receive the whole table at once (columns as series), so arithmetic works there but `if` does not.
  - `by`, and `map` after `group`, receive each group as a separate table.
- `as name` stores the current value in a variable and the pipeline continues. A variable stored inside a block is visible only there.
- `def name [ ... ]` defines a new word. It takes no arguments and acts on the current value.
- `use module` / `use module as alias` imports a Python module. `alias.function` calls the function with the current value as its first argument. The number of additional positional arguments is read from the function's signature.
- `.method` calls a method or reads an attribute of the current value (also `.plot.bar`).
- `:N` after a name (`np.round:1`, `.head:1`) states the number of positional arguments explicitly. For `.method` the default is 0.
- `name=value` right after an `alias.function` or `.method` is a keyword argument (no spaces around `=`; use `0`/`1` instead of False/True).
- Arguments of commands are taken literally (a number, text, a block or a variable name) and are not evaluated. Store an intermediate result with `as` first and pass its name.

## Examples

```
upto 30 filter [ mod 7 eq 0 ] as multiples   # multiples of 7
multiples print
"Sum:" print multiples sum print
```

```
read notes.txt lower words drop i freq first 3 map [ fmt "{0} ({1}x)" ] print
```

```
def verdict [ if [ ge 50 ] [ "pass" ] [ "retake" ] ]
list 72,38,50,91 map [ verdict ] print
```

```
# readings.csv: city, country, temp
use numpy as np
csv readings.csv as p
p where country in PL,CZ by city [ each temp max ] top 2 map [ fmt "{0} {1}°C" ] print
p where city has burg pick city,temp save burg.csv count print
p each temp np.percentile:1 75 round 1 print
p .sort_values:1 temp ascending=0 .head:1 2 print
```
