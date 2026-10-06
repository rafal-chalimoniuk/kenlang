"""Helpers for asking a language model to write Ken.

``reference()`` returns the language reference that is meant to go into the model's prompt;
``build_prompt()`` appends a data description and the task. The same function is used to build
the training prompts and the evaluation prompts, so a model trained here sees exactly the text
that ``build_prompt`` produces at inference time.
"""
from importlib import resources

ANSWER_INSTRUCTION = "Answer with code only."


def reference() -> str:
    """The Ken language reference (``reference.md``) as prompt text."""
    return resources.files("kenlang").joinpath("reference.md").read_text(encoding="utf-8")


def build_prompt(task: str, data_description: str = "") -> str:
    parts = [reference().rstrip()]
    if data_description:
        parts.append("# Data\n\n" + data_description.strip())
    parts.append(f"# Task\n\n{task.strip()}\n\n{ANSWER_INSTRUCTION}")
    return "\n\n".join(parts) + "\n"


def extract_code(answer: str) -> str:
    """Pull the program out of a model answer: the first fenced block, or the whole text."""
    import re
    m = re.search(r"```[^\n]*\n(.*?)```", answer, re.S)
    return (m.group(1) if m else answer).strip()
