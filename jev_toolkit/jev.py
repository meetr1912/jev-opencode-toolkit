"""The heart: many independent typed questions over one state, in one request.

TypeSafe Jev exposes a single endpoint that accepts a *map* of questions. Sending
N questions in one body is roughly an order of magnitude cheaper and faster than
N calls (``docs.typesafe.ai/cookbooks/parallel_questions``), so :func:`ask`
defaults to exactly that: one request, every question, strict validation of every
answer.

The module is deliberately small and synchronous: the network call is a single
``httpx.Client.post``. Bounded concurrency across batches lives in
``concurrency.py``; content-hash caching lives in ``cache.py``; rubric files live
in ``pinned.py``.

Three jaggedness guards are enforced here rather than documented away:

* every requested ``qid`` must come back exactly once (no silent omissions),
* every distribution must be finite, cover exactly the offered labels, and sum
  to 1 (tolerance ``SUM_TOLERANCE``),
* a ``choice`` answer must be the argmax of its own distribution.

``noul`` questions carry no ``confidence``; ``choice``/``score`` must carry one.
"""

from __future__ import annotations

import datetime as _dt
import email.utils
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

ENDPOINT = "https://api.typesafe.ai/v1/systemone"
DEFAULT_MODEL = "jev-latest"
DEFAULT_TIMEOUT = 60.0
DEFAULT_ATTEMPTS = 3
RETRY_STATUS = {429, 503, 529}
SUM_TOLERANCE = 0.02
FORMS = ("choice", "score", "noul")

__all__ = [
    "ENDPOINT",
    "RETRY_STATUS",
    "SUM_TOLERANCE",
    "BatchResult",
    "JevError",
    "Question",
    "ask",
    "question_from_dict",
    "validate_answer",
]


class JevError(RuntimeError):
    """The TypeSafe call failed or returned an unusable answer."""


@dataclass
class BatchResult:
    """Validated answers for one :func:`ask` call.

    ``requests`` counts successful upstream round trips (``0`` on a cache hit),
    which is what the batching benchmark reports against a single-question
    baseline.
    """

    answers: dict[str, dict]
    model: str
    latency_ms: int
    input_tokens: int = 0
    output_tokens: int = 0
    requests: int = 0
    cache_hit: bool = False
    raw_usage: dict = field(default_factory=dict)

    @property
    def tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class Question:
    """One typed Jev question. ``criteria`` shapes the answer space.

    * ``choice``: ``criteria`` is a ``{label: description}`` mapping.
    * ``score``: ``criteria`` is an ordered list of level descriptions; the
      offered labels are ``"0".."n-1"`` (matching ``jev-arena``).
    * ``noul``: no criteria; the answer is a single probability.
    """

    qid: str
    form: str
    instructions: str
    criteria: Any = None
    state: str = ""
    truth: Any = None
    truth_dist: Any = None
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.qid:
            raise JevError("question is missing a qid")
        if self.form not in FORMS:
            raise JevError(f"{self.qid}: unknown form {self.form!r} (expected one of {FORMS})")
        if self.form in ("choice", "score") and not self.labels:
            raise JevError(f"{self.qid}: {self.form} question needs criteria")

    @property
    def labels(self) -> list[str]:
        """The offered answer labels, in the order they will be sent."""

        if self.form == "choice":
            if isinstance(self.criteria, dict):
                return [str(label) for label in self.criteria]
            if isinstance(self.criteria, (list, tuple)):
                return [str(label) for label in self.criteria]
            return []
        if self.form == "score":
            if isinstance(self.criteria, (list, tuple, dict)):
                return [str(index) for index in range(len(self.criteria))]
            return []
        return []

    def to_request(self) -> dict:
        """The per-question payload TypeSafe expects."""

        instructions = self.instructions
        if self.state:
            instructions = f"{self.state}\n\n{instructions}"
        payload: dict = {"type": self.form, "instructions": instructions}
        if self.form == "choice":
            if isinstance(self.criteria, dict):
                payload["criteria"] = dict(self.criteria)
            else:
                payload["criteria"] = {str(label): str(label) for label in self.criteria}
        elif self.form == "score":
            if isinstance(self.criteria, (list, tuple)):
                payload["criteria"] = list(self.criteria)
            elif isinstance(self.criteria, dict):
                payload["criteria"] = list(self.criteria.values())
            else:
                payload["criteria"] = []
        return payload


def question_from_dict(qid: str, spec: Any, state: str = "") -> Question:
    """Build a :class:`Question` from a pinned-file/JSON spec object."""

    if isinstance(spec, Question):
        return spec
    if not isinstance(spec, dict):
        raise JevError(f"question {qid!r} is not an object")
    form = spec.get("type") or spec.get("form")
    instructions = spec.get("instructions")
    if not isinstance(form, str) or not isinstance(instructions, str):
        raise JevError(f"question {qid!r} is missing a string 'type' or 'instructions'")
    return Question(
        qid=qid,
        form=form,
        instructions=instructions,
        criteria=spec.get("criteria"),
        state=spec.get("state", state) or state,
        truth=spec.get("truth"),
        truth_dist=spec.get("truth_dist"),
        meta=spec.get("meta") or {},
    )


def _coerce_questions(questions: Any) -> list[Question]:
    if isinstance(questions, dict):
        return [question_from_dict(str(qid), spec) for qid, spec in questions.items()]
    if isinstance(questions, (list, tuple)):
        out: list[Question] = []
        for index, spec in enumerate(questions):
            if isinstance(spec, Question):
                out.append(spec)
            elif isinstance(spec, dict):
                qid = str(spec.get("qid") or spec.get("id") or index)
                out.append(question_from_dict(qid, spec))
            else:
                raise JevError(f"question at index {index} is not a Question or object")
        return out
    raise JevError("questions must be a list or a {qid: spec} object")


def _unit(value: Any) -> bool:
    """True for a real finite probability in ``[0, 1]`` (``bool`` excluded)."""

    return type(value) in (int, float) and math.isfinite(value) and 0.0 <= float(value) <= 1.0


def validate_answer(question: Question, answer: Any) -> None:
    """Reject a malformed answer before it can reach a caller. Raises :class:`JevError`."""

    if not isinstance(answer, dict):
        raise JevError(f"{question.qid}: answer is not an object")
    if question.form == "noul":
        if not _unit(answer.get("noul")):
            raise JevError(f"{question.qid}: noul answer missing or not a probability in [0, 1]")
        return

    probabilities = answer.get("probabilities")
    labels = set(question.labels)
    if not isinstance(probabilities, dict) or set(probabilities) != labels:
        raise JevError(f"{question.qid}: probabilities do not cover exactly the offered criteria")
    if not all(_unit(value) for value in probabilities.values()):
        raise JevError(f"{question.qid}: probabilities contain a non-finite or out-of-range value")
    total = sum(float(value) for value in probabilities.values())
    if abs(total - 1.0) >= SUM_TOLERANCE:
        raise JevError(f"{question.qid}: probabilities sum to {total:.4f}, not 1")
    if not _unit(answer.get("confidence")):
        raise JevError(f"{question.qid}: {question.form} answer is missing a valid confidence")
    if question.form == "choice":
        chosen = answer.get("choice")
        if chosen not in labels:
            raise JevError(f"{question.qid}: choice {chosen!r} was not offered")
        best = max(question.labels, key=lambda label: probabilities[label])
        if str(chosen) != best:
            raise JevError(f"{question.qid}: choice {chosen!r} is not the argmax {best!r}")


def api_key() -> str:
    """Read the key from the process environment only. Never from a file."""

    key = os.environ.get("TYPESAFE_API_KEY")
    if not key:
        raise JevError("TYPESAFE_API_KEY is not set (live calls need it; use the offline Stub otherwise)")
    return key


def _retry_after(response: httpx.Response) -> float | None:
    """Honour ``Retry-After`` as seconds or an HTTP date; ``None`` if unusable."""

    value = response.headers.get("retry-after")
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        when = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if when is None:
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=_dt.timezone.utc)
    return max(0.0, (when - _dt.datetime.now(_dt.timezone.utc)).total_seconds())


def _sleep_for(response: httpx.Response | None, attempt: int, base: float) -> None:
    delay = _retry_after(response) if response is not None else None
    if delay is None:
        delay = base * (2**attempt)
    time.sleep(delay)


def _chunks(ordered: list[Question], size: int | None) -> list[list[Question]]:
    if not size or size >= len(ordered):
        return [ordered]
    return [ordered[start : start + size] for start in range(0, len(ordered), size)]


def _ask_chunk(
    client: httpx.Client,
    state: Any,
    chunk: list[Question],
    model: str,
    attempts: int,
    retry_base: float,
) -> tuple[dict[str, dict], int, int, str, dict]:
    body = {"model": model, "state": state, "questions": {question.qid: question.to_request() for question in chunk}}
    expected = set(body["questions"])
    key = api_key()
    last_error: Exception | None = None
    for attempt in range(attempts):
        response: httpx.Response | None = None
        try:
            response = client.post(ENDPOINT, json=body, headers={"Authorization": f"Bearer {key}"})
        except httpx.HTTPError as error:
            last_error = error
            _sleep_for(None, attempt, retry_base)
            continue
        if response.status_code in RETRY_STATUS and attempt < attempts - 1:
            last_error = JevError(f"TypeSafe returned retryable HTTP {response.status_code}")
            _sleep_for(response, attempt, retry_base)
            continue
        if response.is_error:
            raise JevError(f"TypeSafe returned HTTP {response.status_code}: {response.text[:200]}")
        try:
            payload = response.json()
            answers = payload["answers"]
            if not isinstance(answers, dict) or set(answers) != expected:
                raise JevError("answer keys do not match the requested question keys")
            by_id = {question.qid: question for question in chunk}
            for qid, answer in answers.items():
                validate_answer(by_id[qid], answer)
        except (KeyError, TypeError, ValueError, JevError) as error:
            last_error = error
            _sleep_for(None, attempt, retry_base)
            continue
        usage = payload.get("usage") or {}
        return (
            answers,
            int(usage.get("input_tokens", 0) or 0),
            int(usage.get("output_tokens", 0) or 0),
            str(payload.get("model") or model),
            payload,
        )
    raise JevError(f"no usable Jev answer after {attempts} attempts: {last_error}")


def ask(
    state: Any,
    questions: Any,
    *,
    model: str | None = None,
    batch_size: int | None = None,
    attempts: int = DEFAULT_ATTEMPTS,
    client: httpx.Client | None = None,
    cache: Any = None,
    retry_base: float = 0.5,
    timeout: float = DEFAULT_TIMEOUT,
) -> BatchResult:
    """Ask many independent typed questions about ``state`` in as few requests as possible.

    ``batch_size=None`` (the default) sends every question in **one** request.
    Set it smaller to split into multiple sequential chunks. Pass ``cache`` to
    make a repeated identical call free.
    """

    ordered = _coerce_questions(questions)
    if not ordered:
        raise JevError("no questions to ask")
    resolved_model = model or os.environ.get("TYPESAFE_MODEL", DEFAULT_MODEL)

    cache_key: str | None = None
    if cache is not None:
        cache_key = cache.key(state, ordered, resolved_model)
        hit = cache.get(cache_key)
        if hit is not None:
            return hit

    owns_client = client is None
    if owns_client:
        client = httpx.Client(timeout=timeout)
    started = time.perf_counter()
    answers: dict[str, dict] = {}
    total_in = total_out = requests = 0
    raw: dict = {}
    try:
        for chunk in _chunks(ordered, batch_size):
            chunk_answers, input_tokens, output_tokens, chunk_model, payload = _ask_chunk(
                client, state, chunk, resolved_model, attempts, retry_base
            )
            answers.update(chunk_answers)
            total_in += input_tokens
            total_out += output_tokens
            requests += 1
            resolved_model = chunk_model
            raw = payload
    finally:
        if owns_client:
            client.close()

    result = BatchResult(
        answers=answers,
        model=resolved_model,
        latency_ms=round((time.perf_counter() - started) * 1000),
        input_tokens=total_in,
        output_tokens=total_out,
        requests=requests,
        cache_hit=False,
        raw_usage=raw.get("usage") or {},
    )
    if cache is not None and cache_key is not None:
        cache.put(cache_key, result)
    return result
