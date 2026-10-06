"""Exceptions raised by the Ken compiler."""


class KenSyntaxError(SyntaxError):
    """The program is not valid Ken (unbalanced brackets, missing arguments, unknown import)."""
