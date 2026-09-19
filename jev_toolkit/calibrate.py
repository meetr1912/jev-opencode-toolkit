"""Calibration lock — the accuracy lever.

Calibration metrics are copied verbatim (math and definitions) from the proven
``jev_arena.metrics`` module so the two stay comparable:

* **Brier** ``= mean((p - y)^2)`` for binary events.
* **Log loss** ``= -mean(y*log(p) + (1-y)*log(1-p))`` with ``p`` clipped to
  ``[EPS, 1-EPS]``.
* **ECE** (expected calibration error) with equal-width bins:
  ``sum_b (n_b/N) * |mean_pred_b - empirical_rate_b|``.
* **Reliability table** — per-bin predicted mean, empirical rate, count.
* **Multiclass Brier / cross-entropy** for categorical distributions.

On top of the metrics this module fits per-question decisiveness thresholds on
labeled samples, emits a small ``decisions.lock.json``, and verifies that lock on
later runs. ``check`` fails when the pinned question hash drifts or when the
recomputed global metrics no longer match the declared ones — the CI teeth that
catch an un-recalibrated model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

EPS = 1e-12
DEFAULT_BINS = [round(0.1 * i, 1) for i in range(11)]


@dataclass
class Sample:
    qid: str
    probability: float
    gate: float
    correct: bool


@dataclass
class ReliabilityBin:
    lower: float
    upper: float
    mean_predicted: float
    empirical_rate: float
    count: int

    @property
    def gap(self) -> float:
        return self.mean_predicted - self.empirical_rate


def _clip(p: float) -> float:
    return min(max(p, EPS), 1.0 - EPS)


def _in_bin(p: float, lower: float, upper: float, edges: list[float]) -> bool:
    if upper == edges[-1]:
        return lower <= p <= upper
    return lower <= p < upper


def brier_score(pairs: list[tuple[float, float]]) -> float:
    """Binary Brier over ``(probability, outcome)`` pairs. Empty -> 0.0."""

    if not pairs:
        return 0.0
    return sum((p - y) ** 2 for p, y in pairs) / len(pairs)


def log_loss(pairs: list[tuple[float, float]]) -> float:
    """Binary log loss (natural log) over ``(probability, outcome)`` pairs."""

    if not pairs:
        return 0.0
    total = 0.0
    for p, y in pairs:
        q = _clip(p)
        total -= y * math.log(q) + (1.0 - y) * math.log(1.0 - q)
    return total / len(pairs)


def multiclass_brier(dists: list[tuple[dict[str, float], dict[str, float]]]) -> float:
    """Mean over events of ``sum_c (p_c - y_c)^2`` for ``(pred, truth)`` dists."""

    if not dists:
        return 0.0
    total = 0.0
    for pred, truth in dists:
        total += sum((pred.get(label, 0.0) - truth.get(label, 0.0)) ** 2 for label in truth)
    return total / len(dists)


def cross_entropy(dists: list[tuple[dict[str, float], dict[str, float]]]) -> float:
    """Mean over events of ``-sum_c y_c log p_c`` for ``(pred, truth)`` dists."""

    if not dists:
        return 0.0
    total = 0.0
    for pred, truth in dists:
        total -= sum(y * math.log(_clip(pred.get(label, 0.0))) for label, y in truth.items())
    return total / len(dists)


def expected_calibration_error(pairs: list[tuple[float, float]], bins: list[float] | None = None) -> float:
    """ECE over equal-width bins (default deciles). Empty -> 0.0."""

    if not pairs:
        return 0.0
    edges = bins or DEFAULT_BINS
    n = len(pairs)
    ece = 0.0
    for lower, upper in zip(edges[:-1], edges[1:], strict=False):
        bucket = [y for p, y in pairs if _in_bin(p, lower, upper, edges)]
        if not bucket:
            continue
        mean_pred = sum(p for p, y in pairs if _in_bin(p, lower, upper, edges)) / len(bucket)
        rate = sum(bucket) / len(bucket)
        ece += (len(bucket) / n) * abs(mean_pred - rate)
    return ece


def reliability_table(pairs: list[tuple[float, float]], bins: list[float] | None = None) -> list[ReliabilityBin]:
    """Per-bin predicted mean, empirical rate, and count."""

    edges = bins or DEFAULT_BINS
    rows: list[ReliabilityBin] = []
    for lower, upper in zip(edges[:-1], edges[1:], strict=False):
        bucket = [(p, y) for p, y in pairs if _in_bin(p, lower, upper, edges)]
        if not bucket:
            rows.append(ReliabilityBin(lower, upper, 0.0, 0.0, 0))
            continue
        mean_pred = sum(p for p, _ in bucket) / len(bucket)
        rate = sum(y for _, y in bucket) / len(bucket)
        rows.append(ReliabilityBin(lower, upper, mean_pred, rate, len(bucket)))
    return rows


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _fit_qid_threshold(rows: list[Sample], target_accuracy: float) -> dict:
    """Largest gate prefix (max coverage) whose running accuracy clears target."""

    ordered = sorted(rows, key=lambda s: s.gate, reverse=True)
    n = len(ordered)
    best_k = 0
    best_accuracy = 0.0
    hits = 0
    for i, sample in enumerate(ordered, start=1):
        hits += 1 if sample.correct else 0
        accuracy = hits / i
        if accuracy >= target_accuracy:
            best_k = i
            best_accuracy = accuracy
    if best_k == 0:
        return {"threshold": 1.0, "coverage": 0.0, "accuracy": 0.0, "n": n}
    return {
        "threshold": ordered[best_k - 1].gate,
        "coverage": best_k / n,
        "accuracy": best_accuracy,
        "n": n,
    }


def _global_metrics(samples: list[Sample]) -> dict:
    pairs = [(s.probability, int(s.correct)) for s in samples]
    n = len(samples)
    return {
        "brier": brier_score(pairs),
        "log_loss": log_loss(pairs),
        "ece": expected_calibration_error(pairs),
        "accuracy": (sum(1 for s in samples if s.correct) / n) if n else 0.0,
        "n": n,
    }


def fit_lock(
    samples: list[Sample],
    *,
    target_accuracy: float = 0.9,
    question_file: str | Path | None = None,
    model: str = "jev-latest",
    created: str | None = None,
) -> dict:
    """Fit a JSON-serializable calibration lock over labeled ``samples``."""

    if created is None:
        created = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    by_qid: dict[str, list[Sample]] = {}
    for sample in samples:
        by_qid.setdefault(sample.qid, []).append(sample)

    thresholds = {qid: _fit_qid_threshold(by_qid[qid], target_accuracy) for qid in sorted(by_qid)}
    questions_sha256 = _sha256(question_file) if question_file is not None else None

    return {
        "version": 1,
        "created": created,
        "model": model,
        "target_accuracy": target_accuracy,
        "questions_sha256": questions_sha256,
        "metrics": _global_metrics(samples),
        "thresholds": thresholds,
    }


_THRESHOLD_KEYS = ("threshold", "coverage", "accuracy", "n")
_METRIC_KEYS = ("brier", "log_loss", "ece", "accuracy")


def check_lock(
    lock: dict,
    *,
    question_file: str | Path | None = None,
    samples: list[Sample] | None = None,
    tol: float = 1e-6,
) -> list[str]:
    """Return a list of human-readable problems; empty means the lock passes."""

    if not isinstance(lock, dict):
        return ["lock is not a JSON object"]

    problems: list[str] = []

    if lock.get("version") != 1:
        problems.append(f"unsupported or missing version: {lock.get('version')!r} (expected 1)")

    metrics = lock.get("metrics")
    thresholds = lock.get("thresholds")
    if not isinstance(metrics, dict):
        problems.append("missing metrics")
    if not isinstance(thresholds, dict):
        problems.append("missing thresholds")

    if isinstance(thresholds, dict):
        for qid, value in thresholds.items():
            if not isinstance(value, dict):
                problems.append(f"threshold {qid!r} is not an object")
                continue
            missing = [key for key in _THRESHOLD_KEYS if key not in value]
            if missing:
                problems.append(f"threshold {qid!r} missing keys: {', '.join(missing)}")

    declared_hash = lock.get("questions_sha256")
    if question_file is not None:
        path = Path(question_file)
        if not path.exists():
            problems.append(f"questions file not found: {path}")
        else:
            actual_hash = _sha256(path)
            if declared_hash is None:
                problems.append("missing calibration hash")
            elif declared_hash != actual_hash:
                problems.append(f"questions hash drift: lock={declared_hash} file={actual_hash}")

    if samples is not None and isinstance(metrics, dict):
        actual = _global_metrics(samples)
        for key in _METRIC_KEYS:
            if key not in metrics:
                problems.append(f"metric drift: {key} missing from lock")
                continue
            try:
                if abs(float(metrics[key]) - actual[key]) > tol:
                    problems.append(f"metric drift: {key} lock={metrics[key]} actual={actual[key]}")
            except (TypeError, ValueError):
                problems.append(f"metric drift: {key} is not numeric: {metrics[key]!r}")
        if "n" not in metrics:
            problems.append("metric drift: n missing from lock")
        elif metrics["n"] != actual["n"]:
            problems.append(f"metric drift: n lock={metrics['n']} actual={actual['n']}")

    return problems


def _load_samples(path: str | Path) -> list[Sample]:
    rows = json.loads(Path(path).read_text())
    if not isinstance(rows, list):
        raise ValueError("samples JSON must be a list")
    return [
        Sample(
            qid=str(row["qid"]),
            probability=float(row["probability"]),
            gate=float(row["gate"]),
            correct=bool(row["correct"]),
        )
        for row in rows
    ]


def _cmd_fit(args: argparse.Namespace) -> int:
    samples_path = Path(args.samples)
    if not samples_path.exists():
        print(f"error: samples file not found: {samples_path}", file=sys.stderr)
        return 1
    try:
        samples = _load_samples(samples_path)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"error: cannot read samples {samples_path}: {exc}", file=sys.stderr)
        return 1

    lock = fit_lock(
        samples,
        target_accuracy=args.target_accuracy,
        question_file=args.questions,
        model=args.model,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(lock, indent=2) + "\n")
    print(f"OK: wrote calibration lock to {out_path}")
    return 0


def _cmd_check(args: argparse.Namespace) -> int:
    lock_path = Path(args.lock)
    if not lock_path.exists():
        print(f"error: lock file not found: {lock_path}", file=sys.stderr)
        return 1
    try:
        lock = json.loads(lock_path.read_text())
    except (OSError, ValueError) as exc:
        print(f"error: cannot read lock {lock_path}: {exc}", file=sys.stderr)
        return 1

    samples = None
    if args.samples is not None:
        samples_path = Path(args.samples)
        if not samples_path.exists():
            print(f"error: samples file not found: {samples_path}", file=sys.stderr)
            return 1
        try:
            samples = _load_samples(samples_path)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            print(f"error: cannot read samples {samples_path}: {exc}", file=sys.stderr)
            return 1

    problems = check_lock(lock, question_file=args.questions, samples=samples)
    if problems:
        for problem in problems:
            print(f"FAIL: {problem}", file=sys.stderr)
        return 1
    print("OK: calibration lock verified")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="jev-lock", description="Fit or verify the calibration lock.")
    sub = parser.add_subparsers(dest="command", required=True)

    fit_parser = sub.add_parser("fit", help="fit a calibration lock")
    fit_parser.add_argument("--samples", required=True, help="labeled samples JSON")
    fit_parser.add_argument("--out", required=True, help="lock output path")
    fit_parser.add_argument("--questions", default=None, help="pinned questions file to hash")
    fit_parser.add_argument("--model", default="jev-latest")
    fit_parser.add_argument("--target-accuracy", type=float, default=0.9)

    check_parser = sub.add_parser("check", help="verify a calibration lock")
    check_parser.add_argument("--lock", required=True)
    check_parser.add_argument("--questions", default=None)
    check_parser.add_argument("--samples", default=None)

    args = parser.parse_args(argv)
    if args.command == "fit":
        return _cmd_fit(args)
    return _cmd_check(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
