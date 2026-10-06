"""Ken: an experimental programming language that compiles to Python."""
from .compiler import ARITY, run, translate
from .errors import KenSyntaxError

__version__ = "0.1.0"
__all__ = ["ARITY", "KenSyntaxError", "__version__", "run", "translate"]
