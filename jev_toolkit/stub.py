"""Deterministic offline stub for the live ``jev.ask`` call shape.

The stub exists so CI, ``--offline`` runs, and the benchmarks are green with no
API key and no paid calls. It mirrors the live call shape exactly:

* :meth:`Stub.ask` accepts the same keyword arguments as ``jev.ask`` (including
  ``client``, ``cache`` and ``retry_base``) and ignores what it does not need, so
  it can be passed as ``asker=stub.ask`` interchangeably with ``asker=jev.ask``.
* Batching mirrors the live client: ``questions`` are split into chunks of
  ``batch_size`` and each chunk is one simulated round trip, so the batching
  benchmark measures one latency per request rather than one per question.

Three modes:

* ``oracle`` -- returns each question's exact ground truth. This is the accuracy
  reference and yields a perfect Brier/ECE of 0.
* ``miscalibrated`` -- an overconfident forecaster: it sharpens the truth with
  temperature ``T = 1.5`` and adds a ``+0.45`` bias to ``noul`` (clipped to
  ``[0, 1]``). It must trip the ECE detector; that is the "teeth test".
* ``corrupt`` -- returns one malformed answer for the first question so strict
  validation raises :class:`~jev_toolkit.jev.JevError`. This is the fixture that
  proves validation has teeth.

The stub is deterministic: it uses no randomness, so results are reproducible.
"""

from __future__ import annotations

import math
import time
from dataclasses import fields, is_dataclass
from typing import Any

from .jev import BatchResult, JevError, Question, question_from_dict, validate_answer

SHARPNESS = 1.5
BIAS = 0.45

_MODES = ("oracle", "miscalibrated", "corrupt")

__all__ = ["Stub", "make_stub", "stub_truth"]


def _nonneg(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise JevError(f"probability is not numeric: {value!r}")
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise JevError(f"probability is not finite and non-negative: {value!r}")
    return number


def _normalize(dist: dict) -> dict:
    cleaned = {label: _nonneg(value) for label, value in dist.items()}
    total = sum(cleaned.values())
    if total <= 0.0:
        raise JevError("distribution sums to zero")
    return {label: round(value / total, 6) for label, value in cleaned.items()}


def _sharpen_dist(dist: dict) -> dict:
    powered = {label: value**SHARPNESS for label, value in dist.items()}
    total = sum(powered.values())
    if total <= 0.0:
        return dict(dist)
    return {label: round(value / total, 6) for label, value in powered.items()}


def _sharpen_noul(probability: float) -> float:
    if probability <= 0.0:
        sharp = 0.0
    elif probability >= 1.0:
        sharp = 1.0
    else:
        low = probability**SHARPNESS
        high = (1.0 - probability) ** SHARPNESS
        sharp = low / (low + high)
    return min(max(sharp + BIAS, 0.0), 1.0)


def _argmax(dist: dict) -> Any:
    return max(dist, key=lambda label: dist[label])


def _score_label(criteria: object, key: object) -> object:
    if isinstance(criteria, dict):
        return criteria.get(key, key)
    if isinstance(criteria, (list, tuple)):
        try:
            index = int(key)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            index = -1
        if 0 <= index < len(criteria):
            return criteria[index]
    return key


def _coerce_questions(questions: object) -> list[Question]:
    """Mirror ``jev._coerce_questions`` so the stub is a true drop-in."""

    if isinstance(questions, dict):
        return [question_from_dict(str(qid), spec) for qid, spec in questions.items()]
    if isinstance(questions, (list, tuple)):
        out: list[Question] = []
        for index, spec in enumerate(questions):
            if isinstance(spec, Question) or hasattr(spec, "form"):
                out.append(spec)  # type: ignore[arg-type]
            elif isinstance(spec, dict):
                qid = str(spec.get("qid") or spec.get("id") or index)
                out.append(question_from_dict(qid, spec))
            else:
                raise JevError(f"question at index {index} is not a Question or spec object")
        return out
    return []


def _batch_result(**kwargs: Any) -> BatchResult:
    if is_dataclass(BatchResult):
        allowed = {field.name for field in fields(BatchResult)}
        kwargs = {key: value for key, value in kwargs.items() if key in allowed}
    return BatchResult(**kwargs)


class Stub:
    """An offline stand-in for a live Jev client with the same call shape."""

    def __init__(
        self,
        truth: dict[str, dict] | None = None,
        *,
        mode: str = "oracle",
        seed: int = 0,
        latency_s: float = 0.0,
        model: str = "stub-jev",
    ) -> None:
        if mode not in _MODES:
            raise ValueError(f"unknown stub mode: {mode!r}")
        self.truth = dict(truth) if truth else {}
        self.mode = mode
        self.seed = int(seed)
        self.latency_s = float(latency_s)
        self.model = str(model)
        self._corrupt_used = False

    def _values(self, q: Question) -> tuple[float | None, dict | None]:
        record = self.truth.get(getattr(q, "qid", None))
        if q.form == "noul":
            if record and record.get("truth") is not None:
                return float(record["truth"]), None
            if getattr(q, "truth", None) is not None:
                return float(q.truth), None
            raise JevError(f"{q.qid}: noul question has no truth")
        if record and record.get("truth_dist"):
            return None, dict(record["truth_dist"])
        if getattr(q, "truth_dist", None):
            return None, dict(q.truth_dist)
        raise JevError(f"{q.qid}: {q.form} question has no truth_dist")

    def _corrupt(self, q: Question) -> dict:
        if q.form == "noul":
            return {"type": "noul", "noul": float("nan")}
        return {"type": q.form, "probabilities": {}}

    def answer(self, q: Question) -> dict:
        if self.mode == "corrupt" and not self._corrupt_used:
            self._corrupt_used = True
            return self._corrupt(q)
        if self.mode not in ("oracle", "miscalibrated"):
            raise ValueError(f"unknown stub mode: {self.mode!r}")

        probability, dist = self._values(q)
        if q.form == "noul":
            value = probability if probability is not None else float("nan")
            if self.mode == "miscalibrated":
                value = _sharpen_noul(value)
            if not math.isfinite(value):
                raise JevError(f"{q.qid}: stub produced a non-finite noul")
            return {"type": "noul", "noul": min(max(float(value), 0.0), 1.0)}

        normalized = _normalize(dist or {})
        if self.mode == "miscalibrated":
            normalized = _sharpen_dist(normalized)
        best = _argmax(normalized)
        confidence = normalized[best]
        if self.mode == "miscalibrated":
            confidence = min(0.99, round(confidence + 0.2, 6))
        if q.form == "choice":
            return {
                "type": "choice",
                "choice": best,
                "probabilities": normalized,
                "confidence": confidence,
            }
        return {
            "type": "score",
            "score": _score_label(getattr(q, "criteria", None), best),
            "probabilities": normalized,
            "confidence": confidence,
        }

    def ask(
        self,
        state: Any,
        questions: Any,
        *,
        model: str | None = None,
        batch_size: int | None = None,
        attempts: int = 3,
        client: Any = None,
        cache: Any = None,
        retry_base: float = 0.5,
        **_: Any,
    ) -> BatchResult:
        del attempts, client, retry_base

        ordered = _coerce_questions(questions)
        if not ordered:
            raise JevError("no questions to ask")
        size = int(batch_size) if batch_size else len(ordered)
        if size <= 0:
            size = len(ordered)
        resolved_model = model or self.model

        cache_key: str | None = None
        if cache is not None:
            cache_key = cache.key(state, ordered, resolved_model)
            hit = cache.get(cache_key)
            if hit is not None:
                return hit

        self._corrupt_used = False
        answers: dict[str, dict] = {}
        requests = 0
        started = time.perf_counter()
        for start in range(0, len(ordered), size):
            chunk = ordered[start : start + size]
            if self.latency_s > 0:
                time.sleep(self.latency_s)
            for question in chunk:
                answer = self.answer(question)
                validate_answer(question, answer)
                answers[question.qid] = answer
            requests += 1

        result = _batch_result(
            answers=answers,
            model=resolved_model,
            latency_ms=round((time.perf_counter() - started) * 1000),
            input_tokens=0,
            output_tokens=0,
            requests=requests,
            cache_hit=False,
            raw_usage={},
        )
        if cache is not None and cache_key is not None:
            cache.put(cache_key, result)
        return result


def stub_truth(questions: list[Question]) -> dict[str, dict]:
    """Extract each question's truth into ``{qid: {"truth"|"truth_dist": ...}}``."""

    truth: dict[str, dict] = {}
    for question in questions:
        if question.form == "noul":
            if getattr(question, "truth", None) is None:
                raise JevError(f"{question.qid}: noul question has no truth")
            truth[question.qid] = {"truth": float(question.truth)}
        else:
            dist = getattr(question, "truth_dist", None)
            if not dist:
                raise JevError(f"{question.qid}: {question.form} question has no truth_dist")
            truth[question.qid] = {"truth_dist": {label: float(value) for label, value in dist.items()}}
    return truth


def make_stub(
    questions: list[Question] | dict | None = None,
    *,
    mode: str = "oracle",
    seed: int = 0,
    latency_s: float = 0.0,
) -> Stub:
    """Build a :class:`Stub`, accepting either questions or a truth mapping."""

    if questions is None:
        truth: dict[str, dict] | None = None
    elif isinstance(questions, dict):
        truth = questions
    else:
        truth = stub_truth(list(questions))
    return Stub(truth, mode=mode, seed=seed, latency_s=latency_s)
