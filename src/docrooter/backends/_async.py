"""Run a coroutine to completion from synchronous code."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from concurrent.futures import ThreadPoolExecutor
from typing import Any, TypeVar

T = TypeVar("T")


def run_sync(coro: Coroutine[Any, Any, T]) -> T:
    """Run ``coro`` on a fresh event loop and return its result.

    When this thread already runs an event loop (e.g. a server calling a backend from async code)
    ``asyncio.run`` would raise, so the coroutine runs on a short-lived worker thread instead and
    the calling thread blocks until it finishes.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()
