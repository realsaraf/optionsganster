"""
DataLayer – DB-ready abstraction over PolygonClient caches.
============================================================
Wraps data retrieval with in-memory dict caches keyed by (symbol, date).
Designed so swapping to MongoDB later requires only changing the
_get / _set helpers (~30 lines).
"""
from datetime import date, timedelta
from typing import Optional
import pandas as pd

from app.polygon_client import PolygonClient


class DataLayer:
    """
    Central data access layer with daily-keyed caching.
    All cache entries auto-partition by (symbol, today) so stale
    cross-day data is never served.

    Future MongoDB migration:
      Replace _cache_get / _cache_set with Mongo find/upsert.
      Replace self._caches dicts with a MongoCollection reference.
    """

    def __init__(self, polygon: PolygonClient):
        self._poly = polygon
        # In-memory caches – keyed by (symbol, date.today(), *extra)
        self._daily_bars_cache: dict[tuple, pd.DataFrame] = {}
        self._intraday_bars_cache: dict[tuple, pd.DataFrame] = {}
        self._sr_levels_cache: dict[tuple, dict] = {}

    # ── Cache helpers (swap these for MongoDB) ───────────────

    def _cache_get(self, store: dict, key: tuple):
        """Retrieve from cache. Returns None on miss."""
        return store.get(key)

    def _cache_set(self, store: dict, key: tuple, value):
        """Store in cache."""
        store[key] = value

    def _cache_clear(self, store: dict, symbol: str | None = None):
        """Clear cache, optionally filtered to a symbol."""
        if symbol is None:
            store.clear()
        else:
            keys_to_del = [k for k in store if k[0] == symbol.upper()]
            for k in keys_to_del:
                del store[k]

    # ── Daily bars (full OHLCV, 90 trading days) ─────────────

    async def get_daily_bars(
        self,
        symbol: str,
        num_days: int = 90,
    ) -> pd.DataFrame:
        """
        Fetch daily OHLCV bars for the underlying.
        Returns DataFrame with columns: date, open, high, low, close, volume.
        Cached by (symbol, today, num_days).
        """
        key = (symbol.upper(), date.today(), num_days)
        cached = self._cache_get(self._daily_bars_cache, key)
        if cached is not None:
            return cached

        df = await self._poly.get_stock_daily_bars(symbol, num_days=num_days)
        self._cache_set(self._daily_bars_cache, key, df)
        return df

    # ── Intraday bars (5-min, last N days) ───────────────────

    async def get_intraday_bars(
        self,
        symbol: str,
        lookback_days: int = 20,
        interval_min: int = 5,
    ) -> pd.DataFrame:
        """
        Fetch intraday OHLCV bars for volume profile / POC.
        Re-uses PolygonClient.get_stock_ohlcv() under the hood.
        Cached by (symbol, today, lookback_days, interval_min).
        """
        key = (symbol.upper(), date.today(), lookback_days, interval_min)
        cached = self._cache_get(self._intraday_bars_cache, key)
        if cached is not None:
            return cached

        end_dt = date.today()
        start_dt = end_dt - timedelta(days=int(lookback_days * 1.5))

        df = await self._poly.get_stock_ohlcv(
            symbol=symbol.upper(),
            start_date=start_dt,
            end_date=end_dt,
            interval_min=interval_min,
        )
        self._cache_set(self._intraday_bars_cache, key, df)
        return df

    # ── S/R levels (computed, cached daily) ──────────────────

    def get_cached_sr(self, symbol: str) -> Optional[dict]:
        """Get previously computed S/R levels for today."""
        key = (symbol.upper(), date.today())
        return self._cache_get(self._sr_levels_cache, key)

    def set_cached_sr(self, symbol: str, levels: dict):
        """Store computed S/R levels for today."""
        key = (symbol.upper(), date.today())
        self._cache_set(self._sr_levels_cache, key, levels)

    # ── Cache management ─────────────────────────────────────

    def clear_all(self, symbol: str | None = None):
        """Clear all caches, optionally for a specific symbol."""
        self._cache_clear(self._daily_bars_cache, symbol)
        self._cache_clear(self._intraday_bars_cache, symbol)
        self._cache_clear(self._sr_levels_cache, symbol)
