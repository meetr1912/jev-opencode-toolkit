"""A lean TypeSafe Jev toolkit for opencode.

The one call worth remembering::

    from jev_toolkit import ask
    result = ask(state, [question, ...])  # one request, validated answers

Submodules (``cache``, ``concurrency``, ``pinned``, ``calibrate``) stay
importable on their own for startup-light use by the MCP server.
"""

from .jev import BatchResult, JevError, Question, ask, question_from_dict, validate_answer

__version__ = "0.1.0"

__all__ = [
    "BatchResult",
    "JevError",
    "Question",
    "ask",
    "question_from_dict",
    "validate_answer",
    "__version__",
]
