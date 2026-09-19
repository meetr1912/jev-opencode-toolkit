"""Offline test suite for jev-opencode-toolkit.

Everything here runs with only the standard library and pytest: no network, no
``TYPESAFE_API_KEY``, no sleeps longer than 50 ms. Live coverage lives behind the
``live`` marker and is skipped unless a real key is present.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from jev_toolkit import calibrate
from jev_toolkit.cache import Cache
from jev_toolkit.calibrate import Sample
from jev_toolkit.concurrency import ask_many, ask_many_async
from jev_toolkit.jev import (
    BatchResult,
    JevError,
    Question,
    ask,
    question_from_dict,
    validate_answer,
)
from jev_toolkit.pinned import load_pinned, substitute
from jev_toolkit.stub import make_stub, stub_truth

REPO_ROOT = Path(__file__).resolve().parents[1]
PINNED_EXAMPLE = REPO_ROOT / "examples" / "pinned_questions.json"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _mk_questions(n: int, *, a: float = 0.8, b: float = 0.2) -> list[Question]:
    """N deterministic choice questions whose truth_dist makes the Stub answerable."""

    return [
        Question(
            qid=f"q{i}",
            form="choice",
            instructions=f"pick for {i}",
            criteria={"a": "A", "b": "B"},
            truth_dist={"a": a, "b": b},
        )
        for i in range(n)
    ]


def _noul_questions(truths: list[float]) -> list[Question]:
    return [Question(qid=f"n{i}", form="noul", instructions="?", truth=t) for i, t in enumerate(truths)]


def _pairs_from_noul(result: BatchResult, truths: list[float]) -> list[tuple[float, float]]:
    return [(result.answers[f"n{i}"]["noul"], t) for i, t in enumerate(truths)]


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload
        self.status_code = 200
        self.is_error = False
        self.text = ""

    def json(self) -> dict:
        return self._payload


class _FakeClient:
    """Duck-typed httpx.Client that records request bodies and returns valid answers."""

    def __init__(self, builder) -> None:
        self.calls: list[dict] = []
        self._builder = builder

    def post(self, url, *, json=None, headers=None):  # noqa: A002 - mirrors httpx signature
        self.calls.append(json)
        return _FakeResponse(self._builder(json))

    def close(self) -> None:
        pass


# --------------------------------------------------------------------------- #
# 1. Question labels / to_request / construction guards
# --------------------------------------------------------------------------- #


def test_question_labels_choice_score_noul():
    choice = Question("c", "choice", "i", criteria={"a": "A", "b": "B"})
    score = Question("s", "score", "i", criteria=["x", "y", "z"])
    noul = Question("n", "noul", "i")

    assert choice.labels == ["a", "b"]
    assert score.labels == ["0", "1", "2"]
    assert noul.labels == []

    assert choice.to_request() == {"type": "choice", "instructions": "i", "criteria": {"a": "A", "b": "B"}}
    assert score.to_request() == {"type": "score", "instructions": "i", "criteria": ["x", "y", "z"]}
    assert noul.to_request() == {"type": "noul", "instructions": "i"}


def test_question_state_is_prepended_to_instructions():
    q = Question("q", "noul", "INSTR", state="CTX")
    assert q.to_request()["instructions"] == "CTX\n\nINSTR"

    # No state -> instructions untouched.
    bare = Question("q2", "noul", "INSTR")
    assert bare.to_request()["instructions"] == "INSTR"


def test_question_rejects_unknown_form_and_missing_criteria():
    with pytest.raises(JevError):
        Question("q", "weird", "i")
    with pytest.raises(JevError):
        Question("q", "choice", "i")
    with pytest.raises(JevError):
        Question("q", "score", "i")
    with pytest.raises(JevError):
        Question("", "noul", "i")


def test_question_from_dict():
    q = question_from_dict("q", {"type": "choice", "instructions": "i", "criteria": {"a": "A"}, "state": "S"})
    assert q.qid == "q" and q.form == "choice" and q.state == "S"
    assert q.labels == ["a"]

    with pytest.raises(JevError):
        question_from_dict("q", {"instructions": "i"})
    with pytest.raises(JevError):
        question_from_dict("q", "not-an-object")


# --------------------------------------------------------------------------- #
# 2. validate_answer teeth
# --------------------------------------------------------------------------- #


def test_validate_answer_accepts_valid_shapes():
    choice = Question("c", "choice", "i", criteria={"a": "A", "b": "B"})
    score = Question("s", "score", "i", criteria=["x", "y"])
    noul = Question("n", "noul", "i")

    validate_answer(choice, {"probabilities": {"a": 0.7, "b": 0.3}, "choice": "a", "confidence": 0.7})
    validate_answer(score, {"probabilities": {"0": 0.6, "1": 0.4}, "confidence": 0.6})
    validate_answer(noul, {"noul": 0.42})


def test_validate_answer_rejects_wrong_keyset():
    q = Question("c", "choice", "i", criteria={"a": "A", "b": "B"})
    with pytest.raises(JevError):
        validate_answer(q, {"probabilities": {"a": 0.7, "c": 0.3}, "choice": "a", "confidence": 0.7})
    with pytest.raises(JevError):
        validate_answer(q, {"probabilities": {"a": 0.5, "b": 0.4, "c": 0.1}, "choice": "a", "confidence": 0.5})
    with pytest.raises(JevError):
        validate_answer(q, {"probabilities": {}})


def test_validate_answer_rejects_non_finite_and_bool():
    q = Question("s", "score", "i", criteria=["x", "y"])
    for bad in (float("nan"), float("inf"), float("-inf"), True, False):
        with pytest.raises(JevError):
            validate_answer(q, {"probabilities": {"0": bad, "1": 0.5}, "confidence": 0.5})


def test_validate_answer_rejects_out_of_range_and_sum():
    q = Question("s", "score", "i", criteria=["x", "y"])
    with pytest.raises(JevError):
        validate_answer(q, {"probabilities": {"0": 1.5, "1": -0.5}, "confidence": 0.5})
    with pytest.raises(JevError):
        validate_answer(q, {"probabilities": {"0": 0.5, "1": 0.4}, "confidence": 0.5})
    with pytest.raises(JevError):
        validate_answer(q, {"probabilities": {"0": 0.5, "1": 0.6}, "confidence": 0.5})


def test_validate_answer_rejects_bad_confidence():
    q = Question("s", "score", "i", criteria=["x", "y"])
    valid_probs = {"0": 0.6, "1": 0.4}
    with pytest.raises(JevError):
        validate_answer(q, {"probabilities": dict(valid_probs)})
    for bad in (True, 1.2, -0.1, float("nan")):
        with pytest.raises(JevError):
            validate_answer(q, {"probabilities": dict(valid_probs), "confidence": bad})


def test_validate_answer_choice_must_equal_argmax():
    q = Question("c", "choice", "i", criteria={"a": "A", "b": "B"})
    with pytest.raises(JevError):
        validate_answer(q, {"probabilities": {"a": 0.3, "b": 0.7}, "choice": "a", "confidence": 0.7})
    with pytest.raises(JevError):
        validate_answer(q, {"probabilities": {"a": 0.7, "b": 0.3}, "choice": "c", "confidence": 0.7})
    # The argmax itself is accepted.
    validate_answer(q, {"probabilities": {"a": 0.7, "b": 0.3}, "choice": "a", "confidence": 0.7})


def test_validate_answer_noul_missing_or_non_finite():
    q = Question("n", "noul", "i")
    with pytest.raises(JevError):
        validate_answer(q, {})
    with pytest.raises(JevError):
        validate_answer(q, {"noul": float("inf")})
    with pytest.raises(JevError):
        validate_answer(q, {"noul": True})
    validate_answer(q, {"noul": 0.0})
    validate_answer(q, {"noul": 1.0})


def test_validate_answer_sum_boundary():
    q = Question("s", "score", "i", criteria=["x", "y"])
    # Off by 0.019 -> passes.
    validate_answer(q, {"probabilities": {"0": 0.5, "1": 0.519}, "confidence": 0.519})
    # Off by 0.021 -> fails.
    with pytest.raises(JevError):
        validate_answer(q, {"probabilities": {"0": 0.5, "1": 0.521}, "confidence": 0.521})


# --------------------------------------------------------------------------- #
# 3. Batching (live ask path with a fake client + Stub path)
# --------------------------------------------------------------------------- #


def test_ask_batching_with_fake_client(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    questions = _mk_questions(5)
    by_id = {q.qid: q for q in questions}
    stub = make_stub(questions)

    def builder(body):
        answers = {qid: stub.answer(by_id[qid]) for qid in body["questions"]}
        return {"answers": answers, "model": "fake-model", "usage": {"input_tokens": 3, "output_tokens": 4}}

    client = _FakeClient(builder)
    result = ask("state", questions, client=client, model="m")
    assert result.requests == 1
    assert len(client.calls) == 1
    assert list(result.answers) == [q.qid for q in questions]
    assert result.model == "fake-model"
    assert result.input_tokens == 3 and result.output_tokens == 4
    assert result.tokens == 7

    chunked = _FakeClient(builder)
    result2 = ask("state", questions, client=chunked, model="m", batch_size=2)
    assert result2.requests == 3 == len(chunked.calls)
    assert list(result2.answers) == [q.qid for q in questions]


def test_stub_batching():
    questions = _mk_questions(5)
    one = make_stub(questions).ask("s", questions)
    assert one.requests == 1
    assert list(one.answers) == [q.qid for q in questions]

    chunked = make_stub(questions).ask("s", questions, batch_size=2)
    assert chunked.requests == 3
    assert list(chunked.answers) == [q.qid for q in questions]
    assert set(chunked.answers) == {q.qid for q in questions}


# --------------------------------------------------------------------------- #
# 4. Cache
# --------------------------------------------------------------------------- #


def test_cache_memory_hit():
    cache = Cache()
    questions = _mk_questions(2)
    first = make_stub(questions).ask("state", questions, cache=cache)
    second = make_stub(questions).ask("state", questions, cache=cache)

    assert first.cache_hit is False and first.requests == 1
    assert second.cache_hit is True and second.requests == 0
    assert cache.hits == 1 and cache.misses == 1


def test_cache_disk_round_trip(tmp_path):
    questions = _mk_questions(3)
    path = tmp_path / "cache"

    first_cache = Cache(path)
    first = make_stub(questions).ask("state", questions, cache=first_cache)
    assert first.cache_hit is False and first.requests == 1
    assert first_cache.misses == 1

    fresh_cache = Cache(path)
    second = make_stub(questions).ask("state", questions, cache=fresh_cache)
    assert second.cache_hit is True and second.requests == 0
    assert second.answers == first.answers
    assert fresh_cache.hits == 1


def test_cache_disabled_is_inert(tmp_path):
    path = tmp_path / "cache"
    cache = Cache(path, enabled=False)
    questions = _mk_questions(2)

    first = make_stub(questions).ask("state", questions, cache=cache)
    second = make_stub(questions).ask("state", questions, cache=cache)
    assert first.requests == 1 and first.cache_hit is False
    assert second.requests == 1 and second.cache_hit is False

    key = cache.key("state", questions, "m")
    assert isinstance(key, str)
    cache.put(key, BatchResult(answers={}, model="m", latency_ms=0))
    assert cache.get(key) is None
    assert not path.exists()


# --------------------------------------------------------------------------- #
# 5. Concurrency
# --------------------------------------------------------------------------- #


def test_ask_many_preserves_order_and_caps_in_flight():
    questions = [Question(f"q{i}", "noul", "?") for i in range(6)]
    groups = [[q] for q in questions]
    lock = threading.Lock()
    stats = {"current": 0, "max": 0}

    def asker(_state, group, **kwargs):
        with lock:
            stats["current"] += 1
            stats["max"] = max(stats["max"], stats["current"])
        time.sleep(0.02)
        with lock:
            stats["current"] -= 1
        return group[0].qid

    results = ask_many("state", groups, concurrency=2, asker=asker)
    assert results == [q.qid for q in questions]
    assert stats["max"] <= 2
    assert stats["max"] >= 1


def test_ask_many_empty_groups():
    def asker(*_args, **_kwargs):  # pragma: no cover - must never run
        raise AssertionError("asker called for an empty group list")

    assert ask_many("state", [], concurrency=3, asker=asker) == []
    assert asyncio.run(ask_many_async("state", [], concurrency=3, asker=asker)) == []


# --------------------------------------------------------------------------- #
# 6. Pinned questions
# --------------------------------------------------------------------------- #


def test_pinned_example_bind():
    pinned = load_pinned(PINNED_EXAMPLE)
    assert set(pinned.qids) == {"is_invoice", "has_signature", "amount_confidence"}
    assert pinned.path == str(PINNED_EXAMPLE)

    state, questions = pinned.bind({"doc": "INV-42"})
    assert isinstance(state, str)
    by_id = {q.qid: q for q in questions}
    assert "INV-42" in by_id["is_invoice"].instructions
    assert "INV-42" in by_id["has_signature"].instructions
    assert "INV-42" in by_id["amount_confidence"].instructions
    # No double-state: bind must not leak the shared state onto each question.
    assert all(q.state == "" for q in questions)


def test_pinned_missing_and_unused_facts_raise():
    pinned = load_pinned(PINNED_EXAMPLE)
    with pytest.raises(JevError):
        pinned.bind({})
    with pytest.raises(JevError):
        pinned.bind({"doc": "x", "extra": "unused"})


def test_pinned_rejects_yaml(tmp_path):
    yaml_path = tmp_path / "pinned.yaml"
    yaml_path.write_text(PINNED_EXAMPLE.read_text())
    with pytest.raises(JevError):
        load_pinned(yaml_path)
    yml_path = tmp_path / "pinned.yml"
    yml_path.write_text(PINNED_EXAMPLE.read_text())
    with pytest.raises(JevError):
        load_pinned(yml_path)


def test_pinned_sha256_stable():
    first = load_pinned(PINNED_EXAMPLE)
    second = load_pinned(PINNED_EXAMPLE)
    assert first.sha256 == second.sha256
    assert len(first.sha256) == 64


def test_substitute_direct():
    assert substitute("a {$x} b {$y}", {"x": 1, "y": "z"}) == "a 1 b z"
    with pytest.raises(JevError):
        substitute("{$missing}", {})


# --------------------------------------------------------------------------- #
# 7. Calibration
# --------------------------------------------------------------------------- #


def test_calibration_metrics_hand_computed():
    pairs = [(0.8, 1.0), (0.2, 0.0)]
    assert calibrate.brier_score(pairs) == pytest.approx(0.04)
    assert calibrate.log_loss(pairs) == pytest.approx(-math.log(0.8))

    ece = calibrate.expected_calibration_error([(0.9, 1.0), (0.9, 1.0), (0.1, 0.0)])
    assert ece == pytest.approx(0.1)

    assert calibrate.brier_score([]) == 0.0
    assert calibrate.log_loss([]) == 0.0
    assert calibrate.expected_calibration_error([]) == 0.0


def test_calibration_multiclass_metrics_and_table():
    dists = [({"a": 0.7, "b": 0.3}, {"a": 1.0, "b": 0.0})]
    assert calibrate.multiclass_brier(dists) == pytest.approx(0.18)
    assert calibrate.cross_entropy(dists) == pytest.approx(-math.log(0.7))
    assert calibrate.multiclass_brier([]) == 0.0
    assert calibrate.cross_entropy([]) == 0.0

    table = calibrate.reliability_table([(0.95, 1.0), (0.85, 1.0), (0.05, 0.0)])
    assert len(table) == 10
    assert sum(row.count for row in table) == 3


def _samples() -> list[Sample]:
    return [
        Sample("q1", 0.9, 0.9, True),
        Sample("q1", 0.6, 0.6, False),
        Sample("q2", 0.8, 0.8, True),
        Sample("q2", 0.4, 0.4, False),
    ]


def test_fit_lock_then_check_lock_is_clean(tmp_path):
    questions = tmp_path / "questions.json"
    questions.write_text('{"version": 1}')
    samples = _samples()
    lock = calibrate.fit_lock(samples, question_file=questions)
    assert calibrate.check_lock(lock, question_file=questions, samples=samples) == []


def test_check_lock_drift_cases(tmp_path):
    questions = tmp_path / "questions.json"
    questions.write_text('{"version": 1}')
    samples = _samples()
    lock = calibrate.fit_lock(samples, question_file=questions)

    tampered_metric = json.loads(json.dumps(lock))
    tampered_metric["metrics"]["brier"] += 0.5
    assert calibrate.check_lock(tampered_metric, samples=samples)

    tampered_hash = json.loads(json.dumps(lock))
    tampered_hash["questions_sha256"] = "0" * 64
    assert calibrate.check_lock(tampered_hash, question_file=questions)

    questions.write_text('{"version": 2}')
    assert calibrate.check_lock(lock, question_file=questions)

    missing_metrics = json.loads(json.dumps(lock))
    del missing_metrics["metrics"]
    assert calibrate.check_lock(missing_metrics)

    missing_version = json.loads(json.dumps(lock))
    del missing_version["version"]
    assert calibrate.check_lock(missing_version)


def test_calibration_cli_fit_then_check(tmp_path):
    samples_path = tmp_path / "samples.json"
    samples_path.write_text(
        json.dumps(
            [
                {"qid": "q1", "probability": 0.9, "gate": 0.9, "correct": True},
                {"qid": "q1", "probability": 0.6, "gate": 0.6, "correct": False},
                {"qid": "q2", "probability": 0.8, "gate": 0.8, "correct": True},
            ]
        )
    )
    questions_path = tmp_path / "questions.json"
    questions_path.write_text('{"version": 1}')
    out_path = tmp_path / "decisions.lock.json"

    fit_argv = ["fit", "--samples", str(samples_path), "--out", str(out_path), "--questions", str(questions_path)]
    check_argv = ["check", "--lock", str(out_path), "--questions", str(questions_path), "--samples", str(samples_path)]

    assert calibrate.main(fit_argv) == 0
    assert out_path.exists()
    assert calibrate.main(check_argv) == 0

    tampered = json.loads(out_path.read_text())
    tampered["metrics"]["ece"] += 1.0
    out_path.write_text(json.dumps(tampered))
    assert calibrate.main(check_argv) == 1


# --------------------------------------------------------------------------- #
# 8. Stub
# --------------------------------------------------------------------------- #


def test_stub_oracle_is_exact_and_stub_truth():
    choice = Question("c", "choice", "i", criteria={"yes": "Y", "no": "N"}, truth_dist={"yes": 0.8, "no": 0.2})
    noul = Question("n", "noul", "i", truth=0.37)
    score = Question("s", "score", "i", criteria=["low", "mid", "high"], truth_dist={"0": 0.1, "1": 0.3, "2": 0.6})

    result = make_stub([choice, noul, score]).ask("state", [choice, noul, score])
    assert result.answers["c"]["probabilities"] == {"yes": 0.8, "no": 0.2}
    assert result.answers["c"]["choice"] == "yes"
    assert result.answers["n"]["noul"] == 0.37
    assert result.answers["s"]["probabilities"] == {"0": 0.1, "1": 0.3, "2": 0.6}

    truth = stub_truth([choice, noul, score])
    assert truth["c"] == {"truth_dist": {"yes": 0.8, "no": 0.2}}
    assert truth["n"] == {"truth": 0.37}

    with pytest.raises(JevError):
        stub_truth([Question("bad", "noul", "i")])


def test_stub_miscalibrated_is_strictly_worse():
    truths = [1.0, 0.0, 1.0, 0.0]
    questions = _noul_questions(truths)

    oracle = make_stub(questions, mode="oracle").ask("state", questions)
    miscalibrated = make_stub(questions, mode="miscalibrated").ask("state", questions)

    oracle_pairs = _pairs_from_noul(oracle, truths)
    misc_pairs = _pairs_from_noul(miscalibrated, truths)

    oracle_brier = calibrate.brier_score(oracle_pairs)
    misc_brier = calibrate.brier_score(misc_pairs)
    oracle_ece = calibrate.expected_calibration_error(oracle_pairs)
    misc_ece = calibrate.expected_calibration_error(misc_pairs)

    assert oracle_brier == pytest.approx(0.0)
    assert oracle_ece == pytest.approx(0.0)
    assert misc_brier > oracle_brier
    assert misc_ece > oracle_ece


def test_stub_corrupt_raises():
    choice = Question("c", "choice", "i", criteria={"a": "A", "b": "B"}, truth_dist={"a": 1.0, "b": 0.0})
    with pytest.raises(JevError):
        make_stub([choice], mode="corrupt").ask("state", [choice])

    noul = Question("n", "noul", "i", truth=0.5)
    with pytest.raises(JevError):
        make_stub([noul], mode="corrupt").ask("state", [noul])


def test_stub_is_deterministic():
    questions = _mk_questions(4) + _noul_questions([0.2, 0.8])
    first = make_stub(questions).ask("state", questions)
    second = make_stub(questions).ask("state", questions)
    assert json.dumps(first.answers, sort_keys=True) == json.dumps(second.answers, sort_keys=True)


def test_stub_unknown_mode_rejected():
    with pytest.raises(ValueError):
        make_stub([], mode="nope")


# --------------------------------------------------------------------------- #
# 9. MCP server end-to-end over a subprocess
# --------------------------------------------------------------------------- #


def _write_offline_pinned(tmp_path: Path) -> Path:
    data = {
        "version": 1,
        "state": "Offline rubric for {$doc}.",
        "questions": {
            "q_choice": {
                "type": "choice",
                "instructions": "Is {$doc} an invoice?",
                "criteria": {"yes": "Y", "no": "N"},
                "truth_dist": {"yes": 1.0, "no": 0.0},
            },
            "q_noul": {
                "type": "noul",
                "instructions": "Does {$doc} have a signature?",
                "truth": 0.0,
            },
            "q_score": {
                "type": "score",
                "instructions": "How confident are you in {$doc}?",
                "criteria": ["low", "high"],
                "truth_dist": {"0": 0.8, "1": 0.2},
            },
        },
    }
    path = tmp_path / "pinned.json"
    path.write_text(json.dumps(data))
    return path


def test_mcp_server_end_to_end(tmp_path):
    pinned = _write_offline_pinned(tmp_path)

    env = dict(os.environ)
    env.pop("TYPESAFE_API_KEY", None)
    env.pop("JEV_CACHE_DIR", None)
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")

    messages = [
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "pytest", "version": "1"},
            },
        },
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "does_not_exist", "arguments": {}},
        },
        {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {
                "name": "jev_ask",
                "arguments": {"state": "s", "questions": {"q1": {"type": "noul", "instructions": "?"}}},
            },
        },
        {
            "jsonrpc": "2.0",
            "id": 5,
            "method": "tools/call",
            "params": {
                "name": "jev_ask_file",
                "arguments": {"path": str(pinned), "facts": {"doc": "INV-1"}, "offline": True},
            },
        },
    ]
    stdin_text = "".join(json.dumps(message) + "\n" for message in messages)

    proc = subprocess.Popen(
        [sys.executable, "-m", "jev_toolkit.mcp_server"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        env=env,
        cwd=str(REPO_ROOT),
    )
    stdout, _stderr = proc.communicate(stdin_text, timeout=30)
    assert proc.returncode == 0

    responses: dict[int, dict] = {}
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        message = json.loads(line)
        if message.get("id") is not None:
            responses[message["id"]] = message

    assert responses[1]["result"]["serverInfo"]["name"] == "jev-opencode-toolkit"
    assert responses[1]["result"]["protocolVersion"] == "2024-11-05"

    tools = responses[2]["result"]["tools"]
    assert [tool["name"] for tool in tools] == ["jev_ask", "jev_ask_file", "jev_calibrate"]

    # Unknown tool -> tool-level error result, not a transport crash.
    assert responses[3]["result"].get("isError") is True

    # Live jev_ask without a key -> isError result (no network attempted).
    assert responses[4]["result"].get("isError") is True

    # Offline pinned ask -> success with the stub's deterministic answers.
    offline = json.loads(responses[5]["result"]["content"][0]["text"])
    assert offline["mode"] == "offline"
    assert set(offline["qids"]) == {"q_choice", "q_noul", "q_score"}
    assert set(offline["answers"]) == {"q_choice", "q_noul", "q_score"}
    assert offline["requests"] == 1


# --------------------------------------------------------------------------- #
# Live marker (skipped without a real key)
# --------------------------------------------------------------------------- #


@pytest.mark.live
def test_live_ask_smoke():
    if not os.environ.get("TYPESAFE_API_KEY"):
        pytest.skip("TYPESAFE_API_KEY not set; live test skipped")
    question = Question("q", "noul", "Is the sky blue during a clear day?")
    result = ask("You are concise and truthful.", [question])
    assert "q" in result.answers
