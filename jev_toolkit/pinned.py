"""Pinned question files: fixed rubrics in JSON with ``{$slot}`` placeholders.

Keeping the rubric in a file and binding only the caller's facts means the model
can never rewrite the criteria between calls, which buys accuracy and saves the
tokens that rewriting would cost. The file is JSON (never YAML) so no parser
dependency is needed, and its raw bytes are hashed so drift is detectable.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

try:  # pragma: no cover - the fallback only matters while jev.py is absent
    from .jev import JevError
except ImportError:  # pragma: no cover

    class JevError(RuntimeError):
        """Fallback error type used until ``jev_toolkit.jev`` is importable."""


if TYPE_CHECKING:
    from .jev import Question

DEFAULT_SLOT_PATTERN = r"\{\$([A-Za-z_][A-Za-z0-9_]*)\}"
_PINNED_VERSION = 1
_SLOT_RE = re.compile(DEFAULT_SLOT_PATTERN)


@dataclass
class PinnedQuestions:
    """A parsed pinned file: the shared state template plus question specs."""

    state: str
    questions: dict[str, dict]
    sha256: str
    path: str | None = None

    @property
    def qids(self) -> list[str]:
        return list(self.questions)

    def bind(self, facts: dict[str, object]) -> tuple[str, list[Question]]:
        return bind_questions(self, facts)


@dataclass
class _FallbackQuestion:
    """Stand-in used only when ``jev_toolkit.jev`` is not importable."""

    qid: str
    form: str
    instructions: str
    criteria: object = None
    state: str = ""
    truth: object = None
    truth_dist: object = None
    meta: dict = field(default_factory=dict)


_jev: object | None = None


def _load_jev():
    """Import and cache ``jev_toolkit.jev`` lazily so this module imports alone."""

    global _jev
    if _jev is None:
        from . import jev

        _jev = jev
    return _jev


def substitute(value: str, facts: dict[str, object]) -> str:
    """Replace every ``{$name}`` slot in ``value`` with ``str(facts[name])``."""

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in facts:
            raise JevError(f"missing fact {name!r} for slot {{${name}}} in template {value!r}")
        return str(facts[name])

    return _SLOT_RE.sub(replace, value)


def _substitute_deep(value: object, facts: dict[str, object]) -> object:
    if isinstance(value, str):
        return substitute(value, facts)
    if isinstance(value, dict):
        return {_substitute_deep(key, facts): _substitute_deep(item, facts) for key, item in value.items()}
    if isinstance(value, list):
        return [_substitute_deep(item, facts) for item in value]
    if isinstance(value, tuple):
        return tuple(_substitute_deep(item, facts) for item in value)
    return value


def _iter_strings(value: object):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _iter_strings(key)
            yield from _iter_strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_strings(item)


def _slot_contexts(pinned: PinnedQuestions) -> dict[str, str]:
    contexts: dict[str, str] = {}

    def scan(value: object, context: str) -> None:
        for text in _iter_strings(value):
            for name in _SLOT_RE.findall(text):
                contexts.setdefault(name, context)

    scan(pinned.state, "state")
    for qid, spec in pinned.questions.items():
        scan(spec, f"question {qid!r}")
    return contexts


def _build_question(qid: str, spec: dict, state: str) -> Question:
    form = spec.get("type")
    instructions = spec.get("instructions")
    if not isinstance(form, str) or not isinstance(instructions, str):
        raise JevError(f"question {qid!r} is missing a string 'type' or 'instructions'")
    try:
        jev = _load_jev()
    except ImportError:
        jev = None
    if jev is not None:
        helper = getattr(jev, "question_from_dict", None)
        if callable(helper):
            return helper(qid, spec, state=state)
        question_cls = getattr(jev, "Question")
        return question_cls(
            qid=qid,
            form=form,
            instructions=instructions,
            criteria=spec.get("criteria"),
            state=state,
        )
    return _FallbackQuestion(
        qid=qid,
        form=form,
        instructions=instructions,
        criteria=spec.get("criteria"),
        state=state,
    )


def load_pinned(path: str | Path) -> PinnedQuestions:
    """Read and validate a pinned file, hashing its raw bytes for drift checks."""

    location = str(path)
    source = Path(path)
    if source.suffix.lower() in (".yaml", ".yml"):
        raise JevError(f"{location}: pinned questions must be JSON-only; YAML is not supported")
    try:
        raw = source.read_bytes()
    except OSError as exc:
        raise JevError(f"{location}: cannot read pinned questions: {exc}") from exc
    digest = hashlib.sha256(raw).hexdigest()
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise JevError(f"{location}: invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise JevError(f"{location}: top level must be a JSON object")
    version = data.get("version")
    if isinstance(version, bool) or not isinstance(version, int):
        raise JevError(f"{location}: 'version' must be an int, got {version!r}")
    if version != _PINNED_VERSION:
        raise JevError(f"{location}: unsupported version {version!r} (expected {_PINNED_VERSION})")
    state = data.get("state")
    if not isinstance(state, str):
        raise JevError(f"{location}: 'state' must be a string")
    questions = data.get("questions")
    if not isinstance(questions, dict) or not questions:
        raise JevError(f"{location}: 'questions' must be a non-empty object")
    for qid, spec in questions.items():
        if not isinstance(spec, dict):
            raise JevError(f"{location}: question {qid!r} must be an object")
    return PinnedQuestions(state=state, questions=questions, sha256=digest, path=location)


def bind_questions(pinned: PinnedQuestions, facts: dict[str, object]) -> tuple[str, list[Question]]:
    """Bind ``facts`` into the state and every question, rejecting stray facts."""

    contexts = _slot_contexts(pinned)
    missing = [name for name in contexts if name not in facts]
    if missing:
        name = missing[0]
        raise JevError(f"missing fact {name!r} for slot {{${name}}} in {contexts[name]}")
    unused = [name for name in facts if name not in contexts]
    if unused:
        where = pinned.path or "the pinned questions"
        named = ", ".join(repr(name) for name in unused)
        raise JevError(f"unused fact(s) {named}: not referenced by any slot in {where}")
    bound_state = substitute(pinned.state, facts)
    # The shared state is returned separately and passed to ``ask``; do not also
    # prepend it per question or the two would double up in the request body.
    bound = [_build_question(qid, _substitute_deep(spec, facts), "") for qid, spec in pinned.questions.items()]
    return bound_state, bound
