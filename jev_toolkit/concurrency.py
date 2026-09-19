"""Bounded concurrency across Jev batches.

A "batch" is one request unit (a list of questions). Running them one at a
time wastes wall-clock on network latency; running them all at once trips
provider rate limits. A semaphore with a small default cap occupies the middle
ground while keeping results in input order regardless of completion order.

The heavy lifting is delegated to ``asyncio.to_thread`` around a synchronous
``asker`` (``jev.ask`` by default), so the sync core stays sync and benchmarks
can inject a stub.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from .jev import BatchResult

__all__ = ["DEFAULT_CONCURRENCY", "ask_many", "ask_many_async"]

DEFAULT_CONCURRENCY = 4


async def ask_many_async(
    state: Any,
    question_groups: list[list[Any]],
    *,
    concurrency: int = DEFAULT_CONCURRENCY,
    asker: Callable[..., Any] | None = None,
    **kwargs: Any,
) -> list[BatchResult]:
    """Ask several question groups concurrently, returning results in input order.

    At most ``concurrency`` groups run at once; every group is a separate call
    to ``asker(state, group, **kwargs)``. ``asker`` defaults to ``jev.ask`` and
    is resolved lazily so this module never imports ``jev`` at load time.
    """
    if not question_groups:
        return []
    concurrency = max(1, int(concurrency))
    if asker is None:
        from .jev import ask as asker

    sem = asyncio.Semaphore(concurrency)
    results: list[Any] = [None] * len(question_groups)

    async def run(index: int, group: list[Any]) -> None:
        async with sem:
            results[index] = await asyncio.to_thread(asker, state, group, **kwargs)

    await asyncio.gather(*(run(i, group) for i, group in enumerate(question_groups)))
    return results


def ask_many(
    state: Any,
    question_groups: list[list[Any]],
    *,
    concurrency: int = DEFAULT_CONCURRENCY,
    asker: Callable[..., Any] | None = None,
    **kwargs: Any,
) -> list[BatchResult]:
    """Synchronous wrapper around :func:`ask_many_async`.

    If an event loop is already running, callers must ``await ask_many_async``
    instead; ``asyncio.run`` cannot nest loops.
    """
    return asyncio.run(
        ask_many_async(state, question_groups, concurrency=concurrency, asker=asker, **kwargs)
    )
