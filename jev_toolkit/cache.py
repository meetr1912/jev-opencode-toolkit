"""Content-hash cache for Jev batches.

Jev calls are deterministic given ``(state, questions, model)``, so identical
requests should be free on a replay. This module hashes that triple into a
sha256 key and keeps results in memory, optionally spilling one JSON file per
key to disk so a fresh process still gets the hit.

Only the stdlib is used. ``BatchResult`` is imported lazily inside ``get`` to
keep this module free of an import-cycle with ``jev`` (``jev`` must not import
``cache`` at module load).
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .jev import BatchResult

__all__ = ["Cache", "cache_key"]

_PERSISTED = ("answers", "model", "input_tokens", "output_tokens", "requests", "raw_usage")


def cache_key(state: Any, questions: list[Any], model: str) -> str:
    """Return the sha256 hex key for a ``(state, questions, model)`` call.

    Canonical JSON (sorted keys, no whitespace, ``default=str``) makes the key
    stable across processes and dict orderings, so a byte-identical request
    always maps to the same cache entry.
    """
    payload = {
        "state": state,
        "model": model,
        "questions": [{**dict(q.to_request() or {}), "qid": q.qid} for q in questions],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class Cache:
    """In-memory (and optionally on-disk) result cache keyed by content hash.

    ``enabled=False`` makes the cache inert: ``key`` still computes (callers may
    want it for logs) but ``get`` always misses and ``put`` is a no-op. That lets
    callers pass a cache unconditionally and let an env flag decide.
    """

    def __init__(self, path: str | Path | None = None, *, enabled: bool = True) -> None:
        self.enabled = enabled
        self.path = Path(path) if path is not None else None
        self._mem: dict[str, BatchResult] = {}
        self._written: set[str] = set()
        self._hits = 0
        self._misses = 0

    @property
    def hits(self) -> int:
        """Number of ``get`` calls that returned a result."""
        return self._hits

    @property
    def misses(self) -> int:
        """Number of ``get`` calls that found nothing."""
        return self._misses

    def key(self, state: Any, questions: list[Any], model: str) -> str:
        """Content hash for this call (always computed, even when disabled)."""
        return cache_key(state, questions, model)

    def get(self, key: str) -> BatchResult | None:
        """Return a cached result, checking memory before disk.

        A corrupt or unreadable disk file is treated as a miss, never an error:
        a bad cache entry must not break a run.
        """
        if not self.enabled:
            return None
        if key in self._mem:
            self._hits += 1
            return self._as_hit(self._mem[key])
        result = self._load_disk(key)
        if result is not None:
            self._mem[key] = result
            self._hits += 1
            return result
        self._misses += 1
        return None

    def put(self, key: str, result: BatchResult) -> None:
        """Store a result in memory, and on disk when a path was configured."""
        if not self.enabled:
            return
        self._mem[key] = result
        if self.path is not None:
            self._write_disk(key, result)

    def clear(self) -> None:
        """Drop all in-memory entries and every disk file this instance touched."""

        keys = set(self._mem) | set(self._written)
        self._mem.clear()
        for key in keys:
            if self.path is None:
                continue
            try:
                self._file(key).unlink()
            except FileNotFoundError:
                pass
            except OSError:
                pass
        self._written.clear()

    def _file(self, key: str) -> Path:
        assert self.path is not None
        return self.path / key[:2] / f"{key}.json"

    def _load_disk(self, key: str) -> BatchResult | None:
        if self.path is None:
            return None
        target = self._file(key)
        try:
            data = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(data, dict):
            return None
        try:
            return self._build(data)
        except (TypeError, ValueError, KeyError):
            return None

    @staticmethod
    def _as_hit(result: BatchResult) -> BatchResult:
        from .jev import BatchResult

        return BatchResult(
            answers=result.answers,
            model=result.model,
            latency_ms=0,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            requests=0,
            cache_hit=True,
            raw_usage=getattr(result, "raw_usage", {}) or {},
        )

    @staticmethod
    def _build(data: dict[str, Any]) -> BatchResult:
        from .jev import BatchResult

        return BatchResult(
            answers=data["answers"],
            model=data["model"],
            latency_ms=0,
            input_tokens=data.get("input_tokens", 0),
            output_tokens=data.get("output_tokens", 0),
            requests=0,
            cache_hit=True,
            raw_usage=data.get("raw_usage", {}),
        )

    def _write_disk(self, key: str, result: BatchResult) -> None:
        assert self.path is not None
        target = self._file(key)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            payload = {name: getattr(result, name, None) for name in _PERSISTED}
            fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=f".{key[:8]}.", suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    json.dump(payload, fh, default=str)
                os.replace(tmp, target)
            except BaseException:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
                raise
        except OSError:
            return
        self._written.add(key)
