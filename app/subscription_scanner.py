"""Background scanner for subscribed alert symbols."""
from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable

logger = logging.getLogger("optionsganster")


class SubscriptionScanner:
    def __init__(
        self,
        *,
        get_symbols: Callable[[], Awaitable[list[str]]],
        scan_symbol: Callable[[str], Awaitable[None]],
        interval_seconds: int = 60,
        max_concurrency: int = 3,
    ):
        self._get_symbols = get_symbols
        self._scan_symbol = scan_symbol
        self._interval_seconds = interval_seconds
        self._max_concurrency = max(1, max_concurrency)
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()

    async def start(self) -> None:
        if self._task and not self._task.done():
            return
        self._stop.clear()
        self._task = asyncio.create_task(self._loop(), name="subscription-scanner")

    async def stop(self) -> None:
        self._stop.set()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                symbols = await self._get_symbols()
                if symbols:
                    logger.info("[SubScanner] scanning %d subscribed symbol(s)", len(symbols))
                    sem = asyncio.Semaphore(self._max_concurrency)

                    async def _scan(sym: str) -> None:
                        async with sem:
                            try:
                                await self._scan_symbol(sym)
                            except Exception as exc:
                                logger.warning("[SubScanner] scan failed for %s: %s", sym, exc)

                    await asyncio.gather(*(_scan(symbol) for symbol in symbols))
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("[SubScanner] loop error: %s", exc)

            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self._interval_seconds)
            except asyncio.TimeoutError:
                continue
