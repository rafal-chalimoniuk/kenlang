# Security policy

## Ken programs run arbitrary Python

A Ken program compiles to Python. `use module` imports any importable module and `.method` calls any
method, so a Ken program can read and delete files, open network connections and run commands.

- **Do not run Ken code from sources you do not trust**, including code written by a language model,
  except inside a sandbox (container, VM or an account without sensitive data).
- `training/evaluate.py` runs model-written programs in a separate process in an empty temporary
  directory. This isolates *mistakes* (a wrong program cannot overwrite your files by accident); it is
  **not** a security boundary.

## Reporting a vulnerability

If you find a way for the compiler or runtime itself (not a program you chose to run) to execute code
or access files unexpectedly, please report it privately to the maintainers instead of opening a
public issue. Include a minimal program and the output of `kenlang --version`.
