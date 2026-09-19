"""Minimal MCP (Model Context Protocol) server over stdio for the Jev toolkit.

This is a *thin* transport shim: it speaks newline-delimited JSON-RPC 2.0 on
stdin/stdout and exposes exactly three tools (``ask``, ``ask_file``,
``calibrate``) so opencode can drive the toolkit. opencode prefixes MCP tool
names with the server name, so wiring this as the ``jev`` server surfaces them
as ``jev_ask``, ``jev_ask_file`` and ``jev_calibrate``. It deliberately uses only
the standard library — no ``mcp`` package, no HTTP server, no web UI — so the
runtime dependency surface stays exactly what ``jev`` itself needs.

stdout is reserved for protocol JSON only. Every diagnostic goes to stderr.
"""

from __future__ import annotations

import inspect
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable

from jev_toolkit import __version__

SERVER_NAME = "jev-opencode-toolkit"
DEFAULT_PROTOCOL_VERSION = "2024-11-05"
JSONRPC_VERSION = "2.0"

PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603


class ToolError(Exception):
    """A bad tool argument or a failed tool execution, surfaced as ``isError``."""


def log(message: str) -> None:
    """Write a diagnostic line to stderr (never stdout)."""

    print(f"[{SERVER_NAME}] {message}", file=sys.stderr, flush=True)


# --------------------------------------------------------------------------- #
# JSON-RPC plumbing
# --------------------------------------------------------------------------- #


def _write(out: Any, payload: dict[str, Any]) -> None:
    out.write(json.dumps(payload, default=str) + "\n")
    out.flush()


def _result(out: Any, msg_id: Any, result: Any) -> None:
    _write(out, {"jsonrpc": JSONRPC_VERSION, "id": msg_id, "result": result})


def _error(out: Any, msg_id: Any, code: int, message: str) -> None:
    _write(out, {"jsonrpc": JSONRPC_VERSION, "id": msg_id, "error": {"code": code, "message": message}})


def _initialize_result(params: dict[str, Any]) -> dict[str, Any]:
    protocol_version = params.get("protocolVersion") or DEFAULT_PROTOCOL_VERSION
    return {
        "protocolVersion": str(protocol_version),
        "capabilities": {"tools": {}},
        "serverInfo": {"name": SERVER_NAME, "version": __version__},
    }


def _handle_message(out: Any, message: Any) -> None:
    if not isinstance(message, dict) or message.get("jsonrpc") not in (None, JSONRPC_VERSION):
        _error(out, None, INVALID_REQUEST, "invalid JSON-RPC request")
        return

    method = message.get("method")
    has_id = "id" in message and message.get("id") is not None
    msg_id = message.get("id")
    params = message.get("params")
    if not isinstance(params, dict):
        params = {}

    # Notifications (no id) never receive a reply, including unknown methods.
    if method == "notifications/initialized":
        return

    try:
        if method == "initialize":
            result: Any = _initialize_result(params)
        elif method == "ping":
            result = {}
        elif method == "tools/list":
            result = {"tools": _tool_definitions()}
        elif method == "tools/call":
            result = _call_tool(params)
        else:
            if not has_id:
                log(f"ignoring unknown notification: {method!r}")
                return
            _error(out, msg_id, METHOD_NOT_FOUND, f"method not found: {method!r}")
            return
    except Exception as exc:  # noqa: BLE001 - a handler bug must not kill the loop
        log(f"error handling {method!r}: {type(exc).__name__}: {exc}")
        if has_id:
            _error(out, msg_id, INTERNAL_ERROR, f"{type(exc).__name__}: {exc}")
        return

    if has_id:
        _result(out, msg_id, result)


# --------------------------------------------------------------------------- #
# Tool definitions
# --------------------------------------------------------------------------- #

_STATE_SCHEMA: dict[str, Any] = {
    "type": ["string", "object"],
    "description": "Shared system state / framing passed to Jev for every question.",
}
_QUESTION_MAP_SCHEMA: dict[str, Any] = {
    "type": "object",
    "description": "Mapping of question id -> {type: choice|score|noul, instructions, criteria?}.",
    "additionalProperties": {
        "type": "object",
        "properties": {
            "type": {"type": "string", "enum": ["choice", "score", "noul"]},
            "instructions": {"type": "string"},
            "criteria": {},
        },
        "required": ["type", "instructions"],
    },
}
_MODEL_SCHEMA: dict[str, Any] = {"type": "string", "description": "Optional Jev model override."}
_BATCH_SIZE_SCHEMA: dict[str, Any] = {
    "type": "integer",
    "minimum": 1,
    "description": "Optional max questions per upstream request.",
}


def _tool_definitions() -> list[dict[str, Any]]:
    return [
        {
            "name": "ask",
            "description": "Send many independent typed questions to TypeSafe Jev in ONE batched request.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "state": _STATE_SCHEMA,
                    "questions": _QUESTION_MAP_SCHEMA,
                    "model": _MODEL_SCHEMA,
                    "batch_size": _BATCH_SIZE_SCHEMA,
                    "concurrency": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "Run consecutive question chunks concurrently.",
                    },
                },
                "required": ["state", "questions"],
            },
        },
        {
            "name": "ask_file",
            "description": "Bind facts into a pinned question file and ask the whole rubric.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to a pinned questions JSON file."},
                    "facts": {
                        "type": "object",
                        "description": "Facts substituted into the pinned instructions.",
                        "default": {},
                    },
                    "model": _MODEL_SCHEMA,
                    "batch_size": _BATCH_SIZE_SCHEMA,
                    "offline": {
                        "type": "boolean",
                        "description": "Force the deterministic Stub instead of the live API.",
                    },
                },
                "required": ["path"],
            },
        },
        {
            "name": "calibrate",
            "description": "Check or fit the calibration lock.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["check", "fit"]},
                    "lock": {"type": "string", "description": "Lock JSON path (check)."},
                    "questions": {"type": "string", "description": "Pinned questions path (hashed / drift check)."},
                    "samples": {"type": "string", "description": "Labeled samples JSON path."},
                    "out": {"type": "string", "description": "Output path for a fitted lock."},
                    "target_accuracy": {"type": "number", "description": "Fit target accuracy (default 0.9)."},
                },
                "required": ["action"],
            },
        },
    ]


def _call_tool(params: dict[str, Any]) -> dict[str, Any]:
    name = params.get("name")
    arguments = params.get("arguments")
    if not isinstance(arguments, dict):
        arguments = {}

    handlers: dict[str, Callable[[dict[str, Any]], str]] = {
        # Exposed names (opencode prefixes the server name -> jev_ask, ...).
        "ask": _tool_jev_ask,
        "ask_file": _tool_jev_ask_file,
        "calibrate": _tool_jev_calibrate,
        # Legacy aliases kept so existing callers/configs keep working.
        "jev_ask": _tool_jev_ask,
        "jev_ask_file": _tool_jev_ask_file,
        "jev_calibrate": _tool_jev_calibrate,
    }
    handler = handlers.get(name)
    if handler is None:
        return _tool_error(f"unknown tool: {name!r}")

    try:
        text = handler(arguments)
    except Exception as exc:  # noqa: BLE001 - tool failures are results, not crashes
        return _tool_error(f"{type(exc).__name__}: {exc}")
    return {"content": [{"type": "text", "text": text}]}


def _tool_error(message: str) -> dict[str, Any]:
    payload = {"status": "error", "error": message}
    return {"content": [{"type": "text", "text": json.dumps(payload)}], "isError": True}


# --------------------------------------------------------------------------- #
# Helpers shared by the tools
# --------------------------------------------------------------------------- #


def _supported_kwargs(func: Callable[..., Any], kwargs: dict[str, Any]) -> dict[str, Any]:
    """Drop kwargs a callable does not accept (unless it takes ``**kwargs``)."""

    try:
        params = inspect.signature(func).parameters
    except (TypeError, ValueError):
        return dict(kwargs)
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return dict(kwargs)
    return {key: value for key, value in kwargs.items() if key in params}


def _make_cache() -> Any:
    """A disk-backed cache when ``JEV_CACHE_DIR`` is set, else in-memory only."""

    from jev_toolkit.cache import Cache

    cache_dir = os.environ.get("JEV_CACHE_DIR")
    return Cache(cache_dir) if cache_dir else Cache()


def _tokens(result: Any) -> int:
    value = getattr(result, "tokens", None)
    if isinstance(value, bool):  # guard against a stray bool property
        value = None
    if callable(value):
        value = value()
    if isinstance(value, (int, float)):
        return int(value)
    return int(getattr(result, "input_tokens", 0) or 0) + int(getattr(result, "output_tokens", 0) or 0)


def _chunks(questions: Any, groups: int) -> list[Any]:
    """Split ordered questions into ``groups`` contiguous chunks of like type."""

    if isinstance(questions, dict):
        items: list[Any] = list(questions.items())
        return [dict(chunk) for chunk in _split(items, groups)]
    if isinstance(questions, (list, tuple)):
        return _split(list(questions), groups)
    return []


def _split(items: list[Any], groups: int) -> list[list[Any]]:
    if not items:
        return []
    groups = max(1, min(int(groups), len(items)))
    size = -(-len(items) // groups)  # ceil division
    return [items[i : i + size] for i in range(0, len(items), size)]


def _merge(results: list[Any]) -> dict[str, Any]:
    answers: dict[str, Any] = {}
    model: Any = None
    requests = 0
    tokens = 0
    cache_hit = True
    for result in results:
        if result is None:
            continue
        answers.update(getattr(result, "answers", {}) or {})
        model = model or getattr(result, "model", None)
        requests += int(getattr(result, "requests", 0) or 0)
        tokens += _tokens(result)
        cache_hit = cache_hit and bool(getattr(result, "cache_hit", False))
    return {"answers": answers, "model": model, "requests": requests, "tokens": tokens, "cache_hit": cache_hit}


def _result_payload(result: Any) -> dict[str, Any]:
    return {
        "answers": getattr(result, "answers", {}) or {},
        "model": getattr(result, "model", None),
        "requests": int(getattr(result, "requests", 0) or 0),
        "tokens": _tokens(result),
        "cache_hit": bool(getattr(result, "cache_hit", False)),
    }


def _run_ask(jev_mod: Any, state: Any, questions: Any, *, model: Any, batch_size: Any, concurrency: Any, cache: Any):
    kwargs: dict[str, Any] = {}
    if model is not None:
        kwargs["model"] = model
    if batch_size is not None:
        kwargs["batch_size"] = batch_size
    kwargs["cache"] = cache
    kwargs = _supported_kwargs(jev_mod.ask, kwargs)

    if concurrency and int(concurrency) > 1:
        group_count = min(int(concurrency), len(questions) if hasattr(questions, "__len__") else int(concurrency))
        groups = _chunks(questions, group_count)
        if len(groups) > 1:
            from jev_toolkit.concurrency import ask_many

            ask_many_kwargs = _supported_kwargs(ask_many, {**kwargs, "concurrency": int(concurrency)})
            return _merge(ask_many(state, groups, **ask_many_kwargs))

    return jev_mod.ask(state, questions, **kwargs)


def _offline_ask(stub_mod: Any, state: Any, questions: list[Any], *, model: Any, batch_size: Any):
    """Run the deterministic Stub built from the questions' own truth fields."""

    stub_truth = getattr(stub_mod, "stub_truth", None)
    if stub_truth is None:
        raise ToolError("jev_toolkit.stub.stub_truth is unavailable")
    truth = _call_flexible(stub_truth, questions)

    make_stub = getattr(stub_mod, "make_stub", None)
    if make_stub is not None:
        stub = _call_flexible(make_stub, truth)
    else:
        stub_cls = getattr(stub_mod, "Stub", None)
        if stub_cls is None:
            raise ToolError("jev_toolkit.stub exposes neither make_stub nor Stub")
        stub = stub_cls(truth)

    asker = getattr(stub, "ask", None)
    if asker is None:
        if callable(stub):
            asker = stub
        else:
            raise ToolError("stub has no callable .ask")

    kwargs: dict[str, Any] = {}
    if model is not None:
        kwargs["model"] = model
    if batch_size is not None:
        kwargs["batch_size"] = batch_size
    kwargs = _supported_kwargs(asker, kwargs)
    return asker(state, questions, **kwargs)


def _call_flexible(func: Callable[..., Any], arg: Any) -> Any:
    """Call ``func`` with one positional arg, adding state= if it wants two."""

    try:
        params = inspect.signature(func).parameters
    except (TypeError, ValueError):
        return func(arg)
    positional = [p for p in params.values() if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    if len(positional) >= 2 and not any(p.kind is p.VAR_POSITIONAL for p in params.values()):
        return func(arg, arg)
    return func(arg)


def _load_json(path: str) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except OSError as exc:
        raise ToolError(f"cannot read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ToolError(f"invalid JSON in {path}: {exc}") from exc


# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #


def _tool_jev_ask(arguments: dict[str, Any]) -> str:
    from jev_toolkit import jev as jev_mod

    state = arguments.get("state")
    if state is None:
        raise ToolError("'state' is required")
    questions = arguments.get("questions")
    if not isinstance(questions, dict) or not questions:
        raise ToolError("'questions' must be a non-empty object mapping qid -> spec")

    result = _run_ask(
        jev_mod,
        state,
        questions,
        model=arguments.get("model"),
        batch_size=arguments.get("batch_size"),
        concurrency=arguments.get("concurrency"),
        cache=_make_cache(),
    )
    return json.dumps(_result_payload(result), default=str)


def _tool_jev_ask_file(arguments: dict[str, Any]) -> str:
    from jev_toolkit import jev as jev_mod
    from jev_toolkit import pinned as pinned_mod
    from jev_toolkit import stub as stub_mod

    path = arguments.get("path")
    if not path:
        raise ToolError("'path' is required")
    facts = arguments.get("facts") or {}
    if not isinstance(facts, dict):
        raise ToolError("'facts' must be an object")

    pinned = pinned_mod.load_pinned(path)
    state, questions = pinned.bind(facts)
    qids = [getattr(question, "qid", None) for question in questions]

    offline = bool(arguments.get("offline")) or not os.environ.get("TYPESAFE_API_KEY")
    if offline:
        result = _offline_ask(
            stub_mod,
            state,
            questions,
            model=arguments.get("model"),
            batch_size=arguments.get("batch_size"),
        )
        mode = "offline"
    else:
        result = _run_ask(
            jev_mod,
            state,
            questions,
            model=arguments.get("model"),
            batch_size=arguments.get("batch_size"),
            concurrency=None,
            cache=_make_cache(),
        )
        mode = "live"

    payload = _result_payload(result)
    payload["mode"] = mode
    payload["qids"] = qids
    return json.dumps(payload, default=str)


def _tool_jev_calibrate(arguments: dict[str, Any]) -> str:
    from jev_toolkit.calibrate import check_lock, fit_lock

    action = arguments.get("action")
    if action not in ("check", "fit"):
        raise ToolError("'action' must be 'check' or 'fit'")
    questions = arguments.get("questions")

    if action == "check":
        lock_path = arguments.get("lock")
        if not lock_path:
            raise ToolError("'lock' is required for action='check'")
        samples_path = arguments.get("samples")
        samples = _load_samples(samples_path) if samples_path else None
        problems = check_lock(_load_json(lock_path), question_file=questions, samples=samples)
        return json.dumps({"ok": not problems, "problems": problems}, default=str)

    samples_path = arguments.get("samples")
    if not samples_path:
        raise ToolError("'samples' is required for action='fit'")
    fit_kwargs: dict[str, Any] = {"question_file": questions}
    if arguments.get("target_accuracy") is not None:
        fit_kwargs["target_accuracy"] = float(arguments["target_accuracy"])
    lock = fit_lock(_load_samples(samples_path), **fit_kwargs)

    out = arguments.get("out")
    if out:
        target = Path(out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    return json.dumps(lock, default=str)


def _load_samples(path: str) -> list[Any]:
    from jev_toolkit.calibrate import Sample

    rows = _load_json(path)
    if not isinstance(rows, list):
        raise ToolError(f"samples file {path} must contain a JSON list")
    samples: list[Any] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ToolError(f"samples file {path} has a non-object row")
        samples.append(
            Sample(
                qid=str(row["qid"]),
                probability=float(row["probability"]),
                gate=float(row["gate"]),
                correct=bool(row["correct"]),
            )
        )
    return samples


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def main() -> int:
    out = sys.stdout
    for line in sys.stdin:
        stripped = line.strip()
        if not stripped:
            continue
        try:
            message = json.loads(stripped)
        except json.JSONDecodeError as exc:
            log(f"malformed JSON line ignored: {exc}")
            _error(out, None, PARSE_ERROR, f"parse error: {exc}")
            continue
        try:
            _handle_message(out, message)
        except KeyboardInterrupt:
            raise
        except Exception as exc:  # noqa: BLE001 - never let one message kill the server
            log(f"unhandled error for message: {type(exc).__name__}: {exc}")
            _error(out, None, INTERNAL_ERROR, f"{type(exc).__name__}: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
