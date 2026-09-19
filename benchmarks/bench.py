#!/usr/bin/env python
"""Offline benchmark + calibration artifacts for jev-opencode-toolkit.

Every measurement here goes through :meth:`jev_toolkit.stub.Stub.ask`, i.e. the
exact same strict-validation path the live ``jev.ask`` uses. There is no
``TYPESAFE_API_KEY`` in CI and every run of this file is offline by construction:
the deterministic stub is the only asker it ever builds. Because the stub is
deterministic (no RNG, no clock beyond the synthetic ``latency_s`` sleeps) this
script is reproducible:

* ``calibration/samples.json`` -- a fixed labeled sample set (literal values).
* ``calibration/decisions.lock.json`` -- fitted from those samples with a fixed
  ``created`` stamp and hashed against ``examples/pinned_questions.json``.
* ``results/bench.json`` / ``results/RESULTS.md`` -- measured timings (snapped for
  stable display) and ratios computed from the raw best-of runs.

The *answers* and the calibration artifacts are byte-reproducible; wall-clock
numbers are real measurements and therefore vary by a few percent between runs.

Usage::

    uv run --no-sync python benchmarks/bench.py --offline --out results
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # so the script also runs from a bare checkout
    sys.path.insert(0, str(ROOT))

from jev_toolkit.calibrate import (  # noqa: E402
    Sample,
    brier_score,
    check_lock,
    expected_calibration_error,
    fit_lock,
    log_loss,
    multiclass_brier,
)
from jev_toolkit.concurrency import ask_many  # noqa: E402
from jev_toolkit.jev import JevError, Question  # noqa: E402
from jev_toolkit.stub import make_stub  # noqa: E402

OFFLINE_BANNER = (
    "OFFLINE / DETERMINISTIC: no TYPESAFE_API_KEY, no network. Every number below "
    "comes from the jev_toolkit offline stub through the strict-validation ask path."
)
STATE = "You are Jev. Answer every independent question about the named document."
PINNED_REL = "examples/pinned_questions.json"
SAMPLES_REL = "calibration/samples.json"
LOCK_REL = "calibration/decisions.lock.json"
# Fixed so re-running this script leaves the committed lock byte-identical.
LOCK_CREATED = "2026-09-18T00:00:00Z"

# --------------------------------------------------------------------------- #
# Deterministic inputs
# --------------------------------------------------------------------------- #

# Fixed labeled samples (qid, probability, gate, correct). Literal values, no RNG.
# They describe an overconfident model: high-confidence calls are right, the
# mid/low band is wrong often enough to keep ECE well away from zero.
_CAL_SAMPLES: tuple[tuple[str, float, float, bool], ...] = (
    ("is_invoice", 0.95, 0.95, True),
    ("is_invoice", 0.90, 0.90, True),
    ("is_invoice", 0.85, 0.85, True),
    ("is_invoice", 0.80, 0.80, True),
    ("is_invoice", 0.75, 0.75, True),
    ("is_invoice", 0.70, 0.70, True),
    ("is_invoice", 0.65, 0.65, True),
    ("is_invoice", 0.55, 0.55, False),
    ("is_invoice", 0.45, 0.45, False),
    ("is_invoice", 0.35, 0.35, False),
    ("is_invoice", 0.25, 0.25, False),
    ("is_invoice", 0.15, 0.15, False),
    ("has_signature", 0.92, 0.90, True),
    ("has_signature", 0.88, 0.90, True),
    ("has_signature", 0.83, 0.80, True),
    ("has_signature", 0.78, 0.80, True),
    ("has_signature", 0.72, 0.70, True),
    ("has_signature", 0.68, 0.70, True),
    ("has_signature", 0.60, 0.60, False),
    ("has_signature", 0.58, 0.60, True),
    ("has_signature", 0.42, 0.40, False),
    ("has_signature", 0.38, 0.40, False),
    ("has_signature", 0.28, 0.30, False),
    ("has_signature", 0.22, 0.20, False),
    ("amount_confidence", 0.90, 0.90, True),
    ("amount_confidence", 0.82, 0.80, True),
    ("amount_confidence", 0.76, 0.80, True),
    ("amount_confidence", 0.71, 0.70, True),
    ("amount_confidence", 0.66, 0.70, True),
    ("amount_confidence", 0.62, 0.60, True),
    ("amount_confidence", 0.57, 0.60, True),
    ("amount_confidence", 0.53, 0.50, False),
    ("amount_confidence", 0.47, 0.50, False),
    ("amount_confidence", 0.41, 0.40, False),
    ("amount_confidence", 0.33, 0.30, False),
    ("amount_confidence", 0.26, 0.30, False),
)

_CHOICE_DISTS: tuple[dict[str, float], ...] = (
    {"a": 0.60, "b": 0.30, "c": 0.10},
    {"a": 0.20, "b": 0.50, "c": 0.30},
    {"a": 0.35, "b": 0.15, "c": 0.50},
    {"a": 0.45, "b": 0.40, "c": 0.15},
)

_SCORE_DISTS: tuple[dict[str, float], ...] = (
    {"0": 0.55, "1": 0.25, "2": 0.15, "3": 0.05},
    {"0": 0.15, "1": 0.50, "2": 0.25, "3": 0.10},
    {"0": 0.10, "1": 0.20, "2": 0.45, "3": 0.25},
    {"0": 0.05, "1": 0.15, "2": 0.30, "3": 0.50},
)


def _binary_questions(n: int) -> list[Question]:
    """N deterministic two-label choice questions for the speed benchmarks."""

    return [
        Question(
            qid=f"q{i:03d}",
            form="choice",
            instructions=f"Is document {i} an invoice?",
            criteria={"yes": "It is an invoice", "no": "It is not an invoice"},
            truth_dist={"yes": 0.8, "no": 0.2} if i % 2 == 0 else {"yes": 0.3, "no": 0.7},
        )
        for i in range(n)
    ]


def _accuracy_questions() -> list[Question]:
    """A fixed labeled set: 20 noul + 20 choice + 20 score = 60 soft-truth events."""

    questions: list[Question] = []
    for i in range(20):
        questions.append(
            Question(
                qid=f"noul_{i:02d}",
                form="noul",
                instructions=f"Does event {i} hold?",
                truth=float(i % 2),
            )
        )
    for i in range(20):
        questions.append(
            Question(
                qid=f"choice_{i:02d}",
                form="choice",
                instructions=f"Pick a category for event {i}.",
                criteria={"a": "A", "b": "B", "c": "C"},
                truth_dist=dict(_CHOICE_DISTS[i % len(_CHOICE_DISTS)]),
            )
        )
    for i in range(20):
        questions.append(
            Question(
                qid=f"score_{i:02d}",
                form="score",
                instructions=f"Rate event {i}.",
                criteria=["very low", "low", "medium", "high"],
                truth_dist=dict(_SCORE_DISTS[i % len(_SCORE_DISTS)]),
            )
        )
    return questions


def _argmax(dist: dict) -> str:
    return max(dist, key=lambda label: dist[label])


def _agreement(question: Question, answer: dict) -> bool:
    if question.form == "noul":
        return (float(answer["noul"]) >= 0.5) == (float(question.truth) >= 0.5)
    probabilities = answer["probabilities"]
    return _argmax(probabilities) == _argmax(question.truth_dist)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _r(value: float, digits: int) -> float:
    """Round for reproducible output; adds 0.0 so ``-0.0`` never appears."""

    return round(float(value) + 0.0, digits)


def _quantize(value: float, step: float) -> float:
    """Snap a measured wall time to a stable grid.

    Wall-clock carries scheduler jitter of a few ms, which can flip a rounded
    digit between two otherwise identical runs. Snapping each measured duration
    to a grid coarser than that jitter keeps the committed results byte-stable
    while still reporting the real (sleep-dominated) measurement.
    """

    return round(float(value) / step) * step


def _lock_cli(args: list[str], cwd: Path = ROOT) -> tuple[int, str, str]:
    """Run the real ``jev-lock`` CLI, preferring the uv entry point."""

    uv = shutil.which("uv")
    cmd = [uv, "run", "--no-sync", "jev-lock", *args] if uv else [sys.executable, "-m", "jev_toolkit.calibrate", *args]
    proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    return proc.returncode, (proc.stdout or "").strip(), (proc.stderr or "").strip()


# --------------------------------------------------------------------------- #
# 1. SPEED
# --------------------------------------------------------------------------- #


def bench_speed(n: int, repeats: int = 3) -> dict:
    questions = _binary_questions(n)
    stub = make_stub(questions, mode="oracle", latency_s=0.02)

    serial_s = float("inf")
    serial_requests = 0
    batched_s = float("inf")
    batched_requests = 0
    for _ in range(repeats):
        started = time.perf_counter()
        requests = 0
        for question in questions:
            requests += stub.ask(STATE, [question]).requests
        elapsed = time.perf_counter() - started
        if elapsed < serial_s:
            serial_s, serial_requests = elapsed, requests

        started = time.perf_counter()
        requests = stub.ask(STATE, questions).requests
        elapsed = time.perf_counter() - started
        if elapsed < batched_s:
            batched_s, batched_requests = elapsed, requests

    serial_total = _r(_quantize(serial_s, 0.1), 2)
    batched_total = _r(_quantize(batched_s, 0.01), 2)
    return {
        "questions": n,
        "latency_s": 0.02,
        "label": "offline stub (simulated 20 ms round-trip)",
        "repeats": repeats,
        "serial": {
            "requests": serial_requests,
            "total_s": serial_total,
            "ms_per_question": _r(serial_total / n * 1000.0, 1),
        },
        "batched": {
            "requests": batched_requests,
            "total_s": batched_total,
            "ms_per_question": _r(batched_total / n * 1000.0, 1),
        },
        # Ratio is computed from the raw best-of timings, not the snapped display
        # values, so it reports the real measurement rather than a rounded 80.
        "timing_ratio": _r(serial_s / batched_s, 1),
        "timing_ratio_raw": _r(serial_s / batched_s, 2),
        "call_count_ratio": f"{serial_requests}:{batched_requests}",
    }


# --------------------------------------------------------------------------- #
# 2. PARALLELISM
# --------------------------------------------------------------------------- #


def bench_parallelism(n: int, concurrency: int, repeats: int = 3) -> dict:
    questions = _binary_questions(n)
    k = max(1, min(concurrency, n))
    size = -(-n // k)  # ceil division
    groups = [questions[i : i + size] for i in range(0, n, size)]

    serial_stub = make_stub(questions, mode="oracle", latency_s=0.05)
    parallel_stub = make_stub(questions, mode="oracle", latency_s=0.05)

    serial_s = float("inf")
    parallel_s = float("inf")
    serial_requests = parallel_requests = 0
    observed_order: list[str] = []
    serial_observed: list[str] = []
    for _ in range(repeats):
        started = time.perf_counter()
        results = ask_many(STATE, groups, concurrency=1, asker=serial_stub.ask)
        elapsed = time.perf_counter() - started
        if elapsed < serial_s:
            serial_s = elapsed
            serial_requests = sum(result.requests for result in results)
            serial_observed = [qid for result in results for qid in result.answers]

        started = time.perf_counter()
        results = ask_many(STATE, groups, concurrency=k, asker=parallel_stub.ask)
        elapsed = time.perf_counter() - started
        if elapsed < parallel_s:
            parallel_s = elapsed
            parallel_requests = sum(result.requests for result in results)
            observed_order = [qid for result in results for qid in result.answers]

    expected_order = [question.qid for question in questions]
    serial_total = _r(_quantize(serial_s, 0.05), 2)
    parallel_total = _r(_quantize(parallel_s, 0.05), 2)
    return {
        "questions": n,
        "groups": len(groups),
        "concurrency": k,
        "latency_s": 0.05,
        "label": "offline stub (simulated 50 ms round-trip per group)",
        "repeats": repeats,
        "serial": {"total_s": serial_total, "requests": serial_requests},
        "parallel": {"total_s": parallel_total, "requests": parallel_requests},
        # Raw best-of ratio, not the snapped display values.
        "speedup": _r(serial_s / parallel_s, 1),
        "order_preserved": observed_order == expected_order and serial_observed == expected_order,
    }


# --------------------------------------------------------------------------- #
# 3. ACCURACY
# --------------------------------------------------------------------------- #


def bench_accuracy() -> dict:
    questions = _accuracy_questions()
    modes: dict[str, dict] = {}
    corrupt: dict = {"raised": False}

    for mode in ("oracle", "miscalibrated"):
        stub = make_stub(questions, mode=mode)
        result = stub.ask(STATE, questions)

        noul_pairs = [(float(result.answers[q.qid]["noul"]), float(q.truth)) for q in questions if q.form == "noul"]
        categorical = [
            (result.answers[q.qid]["probabilities"], dict(q.truth_dist)) for q in questions if q.form != "noul"
        ]
        agreement = sum(1 for q in questions if _agreement(q, result.answers[q.qid])) / len(questions)

        modes[mode] = {
            "brier": _r(brier_score(noul_pairs), 6),
            "log_loss": _r(log_loss(noul_pairs), 6),
            "ece": _r(expected_calibration_error(noul_pairs), 6),
            "multiclass_brier": _r(multiclass_brier(categorical), 6),
            "agreement": _r(agreement, 6),
            "events": len(questions),
            "binary_events": len(noul_pairs),
            "categorical_events": len(categorical),
        }

    try:
        make_stub(questions, mode="corrupt").ask(STATE, questions)
    except JevError as error:
        corrupt = {"raised": True, "error_type": type(error).__name__, "message": str(error)}

    strictly_worse = all(
        modes["miscalibrated"][key] > modes["oracle"][key] for key in ("brier", "log_loss", "ece", "multiclass_brier")
    )
    return {
        "label": "offline stub (simulated 0 ms; exact oracle vs overconfident miscalibrated)",
        "oracle": modes["oracle"],
        "miscalibrated": modes["miscalibrated"],
        "oracle_is_reference": modes["oracle"]["brier"] == 0.0
        and modes["oracle"]["ece"] == 0.0
        and modes["oracle"]["multiclass_brier"] == 0.0
        and modes["oracle"]["agreement"] == 1.0,
        "miscalibrated_strictly_worse": strictly_worse,
        "corrupt": corrupt,
    }


# --------------------------------------------------------------------------- #
# 4. CALIBRATION
# --------------------------------------------------------------------------- #


def write_calibration(out_dir: Path) -> dict:
    samples_path = ROOT / SAMPLES_REL
    lock_path = ROOT / LOCK_REL
    pinned_path = ROOT / PINNED_REL

    samples = [Sample(qid, probability, gate, correct) for qid, probability, gate, correct in _CAL_SAMPLES]
    payload = [
        {"qid": s.qid, "probability": s.probability, "gate": s.gate, "correct": s.correct} for s in samples
    ]
    samples_path.parent.mkdir(parents=True, exist_ok=True)
    samples_path.write_text(json.dumps(payload, indent=2) + "\n")

    lock = fit_lock(
        samples,
        target_accuracy=0.9,
        question_file=pinned_path,
        model="jev-latest",
        created=LOCK_CREATED,
    )
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(json.dumps(lock, indent=2) + "\n")

    check_argv = [
        "check",
        "--lock",
        str(lock_path),
        "--questions",
        str(pinned_path),
        "--samples",
        str(samples_path),
    ]
    committed_code, committed_out, committed_err = _lock_cli(check_argv)
    direct_problems = check_lock(lock, question_file=pinned_path, samples=samples)

    teeth: dict[str, dict] = {}
    with tempfile.TemporaryDirectory(prefix="jev-lock-teeth-") as tmp:
        tmp_dir = Path(tmp)
        copied_lock = tmp_dir / "decisions.lock.json"
        copied_samples = tmp_dir / "samples.json"
        copied_questions = tmp_dir / "pinned_questions.json"
        copied_lock.write_text(lock_path.read_text())
        copied_samples.write_text(samples_path.read_text())
        copied_questions.write_text(pinned_path.read_text())

        tampered_lock = tmp_dir / "tampered.lock.json"
        tampered = json.loads(copied_lock.read_text())
        tampered["metrics"]["brier"] = round(tampered["metrics"]["brier"] + 0.5, 6)
        tampered_lock.write_text(json.dumps(tampered, indent=2) + "\n")
        code, _, err = _lock_cli(
            [
                "check",
                "--lock",
                str(tampered_lock),
                "--questions",
                str(copied_questions),
                "--samples",
                str(copied_samples),
            ]
        )
        teeth["tampered_metric"] = {"exit_code": code, "stderr": err}

        tampered_questions = tmp_dir / "tampered_questions.json"
        text = copied_questions.read_text()
        tampered_questions.write_text(text.replace("Is {$doc} an invoice?", "Is {$doc} a receipt?"))
        code, _, err = _lock_cli(
            [
                "check",
                "--lock",
                str(copied_lock),
                "--questions",
                str(tampered_questions),
                "--samples",
                str(copied_samples),
            ]
        )
        teeth["tampered_questions"] = {"exit_code": code, "stderr": err}

    return {
        "label": "committed samples + lock, verified by the jev-lock CLI",
        "samples": SAMPLES_REL,
        "lock": LOCK_REL,
        "questions": PINNED_REL,
        "target_accuracy": 0.9,
        "created": LOCK_CREATED,
        "metrics": {key: _r(value, 6) if isinstance(value, float) else value for key, value in lock["metrics"].items()},
        "thresholds": lock["thresholds"],
        "questions_sha256": lock["questions_sha256"],
        "direct_problems": direct_problems,
        "cli_check": {
            "exit_code": committed_code,
            "stdout": committed_out,
            "stderr": committed_err,
        },
        "teeth": teeth,
    }


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def render_markdown(data: dict) -> str:
    speed = data["speed"]
    parallel = data["parallelism"]
    accuracy = data["accuracy"]
    calibration = data["calibration"]
    oracle = accuracy["oracle"]
    misc = accuracy["miscalibrated"]
    teeth = calibration["teeth"]
    ser = speed["serial"]
    bat = speed["batched"]

    lines = [
        "# jev-opencode-toolkit benchmark results",
        "",
        f"> **{OFFLINE_BANNER}**",
        "",
        f"- Command: `{data['command']}`",
        "- Engine: offline stub, deterministic, no RNG",
        f"- Python: {data['python']}",
        "",
        "## Commands",
        "",
        "```bash",
        "uv run --no-sync python benchmarks/bench.py --offline --out results",
        "uv run --no-sync jev-lock check --lock calibration/decisions.lock.json \\",
        "  --questions examples/pinned_questions.json --samples calibration/samples.json",
        "```",
        "",
        "## SPEED - one batched request vs N single-question requests",
        "",
        f"{speed['label']}. N = {speed['questions']}, simulated round-trip {speed['latency_s']:.2f} s.",
        "",
        "| strategy | requests | total s | ms/question |",
        "| --- | ---: | ---: | ---: |",
        f"| {speed['questions']} separate asks | {ser['requests']} | "
        f"{ser['total_s']:.2f} | {ser['ms_per_question']:.1f} |",
        f"| one batched ask | {bat['requests']} | "
        f"{bat['total_s']:.2f} | {bat['ms_per_question']:.1f} |",
        "",
        f"**Timing ratio:** ~{speed['timing_ratio']}x fewer wall-seconds batched "
        f"(raw best-of; display seconds are snapped for stable CI output). "
        f"**Call-count ratio (exact):** {speed['call_count_ratio']}.",
        "",
        "## PARALLELISM - bounded concurrency over batches",
        "",
        f"{parallel['label']}. {parallel['questions']} questions -> {parallel['groups']} groups, "
        f"concurrency 1 vs {parallel['concurrency']}.",
        "",
        "| concurrency | wall s | requests |",
        "| ---: | ---: | ---: |",
        f"| 1 | {parallel['serial']['total_s']:.2f} | {parallel['serial']['requests']} |",
        f"| {parallel['concurrency']} | {parallel['parallel']['total_s']:.2f} | {parallel['parallel']['requests']} |",
        "",
        f"**Speedup:** ~{parallel['speedup']}x (raw best-of). **Order preserved:** {parallel['order_preserved']}.",
        "",
        "## ACCURACY - oracle vs miscalibrated through the same validation path",
        "",
        f"{accuracy['label']}. {oracle['events']} events "
        f"({oracle['binary_events']} noul + {oracle['categorical_events']} categorical).",
        "",
        "| mode | Brier | log loss | ECE | multiclass Brier | agreement |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        f"| oracle (reference) | {oracle['brier']:.6f} | {oracle['log_loss']:.6f} | "
        f"{oracle['ece']:.6f} | {oracle['multiclass_brier']:.6f} | {oracle['agreement']:.6f} |",
        f"| miscalibrated | {misc['brier']:.6f} | {misc['log_loss']:.6f} | "
        f"{misc['ece']:.6f} | {misc['multiclass_brier']:.6f} | {misc['agreement']:.6f} |",
        "",
        f"Oracle matches the reference (Brier=0, ECE=0, multiclass Brier=0, agreement=1.0): "
        f"**{accuracy['oracle_is_reference']}**. "
        f"Miscalibrated is strictly worse on every metric: **{accuracy['miscalibrated_strictly_worse']}**.",
        "",
        f"Teeth: `mode=\"corrupt\"` raises `{accuracy['corrupt'].get('error_type')}` -> "
        f"**{accuracy['corrupt']['raised']}** (`{accuracy['corrupt'].get('message', '')}`).",
        "",
        "## CALIBRATION - committed samples + lock",
        "",
        f"{calibration['label']}.",
        "",
        "| metric | value |",
        "| --- | ---: |",
        f"| brier | {calibration['metrics']['brier']:.6f} |",
        f"| log_loss | {calibration['metrics']['log_loss']:.6f} |",
        f"| ece | {calibration['metrics']['ece']:.6f} |",
        f"| accuracy | {calibration['metrics']['accuracy']:.6f} |",
        f"| n | {calibration['metrics']['n']} |",
        "",
        "| qid | threshold | coverage | accuracy | n |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for qid in sorted(calibration["thresholds"]):
        row = calibration["thresholds"][qid]
        lines.append(
            f"| {qid} | {row['threshold']:.2f} | {row['coverage']:.4f} | {row['accuracy']:.4f} | {row['n']} |"
        )

    lines += [
        "",
        f"`questions_sha256`: `{calibration['questions_sha256']}`",
        "",
        f"CLI check: `{calibration['cli_check']['stdout']}` (exit {calibration['cli_check']['exit_code']}).",
        "",
        "### Teeth - tampered copies fail",
        "",
        "| tamper | exit code | CLI stderr |",
        "| --- | ---: | --- |",
        f"| metric (`brier += 0.5`) | {teeth['tampered_metric']['exit_code']} | "
        f"{teeth['tampered_metric']['stderr']} |",
        f"| question (instruction edited) | {teeth['tampered_questions']['exit_code']} | "
        f"{teeth['tampered_questions']['stderr']} |",
        "",
        "## Headline",
        "",
        f"- Speed timing ratio: **~{speed['timing_ratio']}x** (raw best-of); "
        f"call-count ratio **{speed['call_count_ratio']}** (exact).",
        f"- Parallel speedup: **~{parallel['speedup']}x** (concurrency {parallel['concurrency']}, raw best-of).",
        f"- Oracle vs miscalibrated Brier: **{oracle['brier']:.6f} -> {misc['brier']:.6f}**; "
        f"ECE: **{oracle['ece']:.6f} -> {misc['ece']:.6f}**.",
        f"- Calibration lock check exit code: **{calibration['cli_check']['exit_code']}**.",
        "",
        "## Reproduce",
        "",
        "```bash",
        "# 1. regenerate samples, lock, results (samples/lock deterministic; timings measured)",
        "uv run --no-sync python benchmarks/bench.py --offline --out results",
        "",
        "# 2. verify the committed lock against the pinned rubric and samples",
        "uv run --no-sync jev-lock check --lock calibration/decisions.lock.json \\",
        "  --questions examples/pinned_questions.json --samples calibration/samples.json",
        "```",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Offline benchmarks + calibration artifacts.")
    parser.add_argument("--offline", action="store_true", help="run the deterministic offline stub (the only path)")
    parser.add_argument("--out", default="results", help="output directory (default: results)")
    parser.add_argument("--questions", type=int, default=80, help="number of questions (default: 80)")
    parser.add_argument("--concurrency", type=int, default=8, help="parallel group count (default: 8)")
    args = parser.parse_args(argv)

    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(OFFLINE_BANNER)
    print(f"[1/4] SPEED (N={args.questions}) ...", flush=True)
    speed = bench_speed(args.questions)
    print(
        f"      serial {speed['serial']['total_s']:.2f}s / {speed['serial']['requests']} reqs vs "
        f"batched {speed['batched']['total_s']:.2f}s / {speed['batched']['requests']} req "
        f"(timing {speed['timing_ratio']}x, calls {speed['call_count_ratio']})"
    )

    print(f"[2/4] PARALLELISM (K={args.concurrency}) ...", flush=True)
    parallel = bench_parallelism(args.questions, args.concurrency)
    print(
        f"      {parallel['serial']['total_s']:.2f}s -> {parallel['parallel']['total_s']:.2f}s "
        f"(speedup {parallel['speedup']}x, order preserved {parallel['order_preserved']})"
    )

    print("[3/4] ACCURACY (oracle vs miscalibrated vs corrupt) ...", flush=True)
    accuracy = bench_accuracy()
    print(
        f"      oracle Brier {accuracy['oracle']['brier']:.6f} / ECE {accuracy['oracle']['ece']:.6f} vs "
        f"miscalibrated Brier {accuracy['miscalibrated']['brier']:.6f} / ECE {accuracy['miscalibrated']['ece']:.6f}"
    )
    print(f"      corrupt raises JevError: {accuracy['corrupt']['raised']}")

    print("[4/4] CALIBRATION (fit + CLI check + teeth) ...", flush=True)
    calibration = write_calibration(out_dir)
    print(f"      lock check exit {calibration['cli_check']['exit_code']}: {calibration['cli_check']['stdout']}")
    print(
        f"      tamper metric exit {calibration['teeth']['tampered_metric']['exit_code']}, "
        f"tamper questions exit {calibration['teeth']['tampered_questions']['exit_code']}"
    )

    data = {
        "offline": True,
        "banner": OFFLINE_BANNER,
        "command": f"uv run --no-sync python benchmarks/bench.py --offline --out {args.out}",
        "python": sys.version.split()[0],
        "questions": args.questions,
        "concurrency": args.concurrency,
        "speed": speed,
        "parallelism": parallel,
        "accuracy": accuracy,
        "calibration": calibration,
    }

    (out_dir / "bench.json").write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    (out_dir / "RESULTS.md").write_text(render_markdown(data))

    headline = {
        "speed_timing_ratio": speed["timing_ratio"],
        "speed_call_count_ratio": speed["call_count_ratio"],
        "parallel_speedup": parallel["speedup"],
        "oracle_brier": accuracy["oracle"]["brier"],
        "oracle_ece": accuracy["oracle"]["ece"],
        "miscalibrated_brier": accuracy["miscalibrated"]["brier"],
        "miscalibrated_ece": accuracy["miscalibrated"]["ece"],
        "lock_check_exit_code": calibration["cli_check"]["exit_code"],
    }
    print("headline: " + json.dumps(headline, sort_keys=True))
    print(f"wrote {out_dir / 'RESULTS.md'} and {out_dir / 'bench.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
