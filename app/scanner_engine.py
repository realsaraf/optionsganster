"""
Scanner Engine — On-demand multi-symbol scan.

Scores each symbol on:
  clean_trend      (0-25)  – slope consistency, ADX-like
  compression      (0-25)  – ATR squeeze / Bollinger squeeze
  breakout_prox    (0-25)  – distance to key S/R level
  positioning_edge (0-25)  – GEX / IV rank / put-call skew

Returns a sorted list of ScanResult.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field, asdict
from typing import Optional

log = logging.getLogger(__name__)

DEFAULT_SYMBOLS = ["QQQ", "SPY", "NVDA", "TSLA", "AMD", "META"]


@dataclass
class ScanResult:
    symbol: str
    score: int = 0                  # 0-100
    clean_trend: int = 0
    compression: int = 0
    breakout_prox: int = 0
    positioning_edge: int = 0
    regime: str = ""
    spot: float = 0.0
    atr: float = 0.0
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class ScannerEngine:
    """Scores multiple symbols for trade-readiness."""

    async def scan(
        self,
        symbols: list[str] | None = None,
        polygon_client=None,
        data_layer=None,
    ) -> list[ScanResult]:
        """Run a scan across symbols. Returns sorted by score desc."""
        syms = symbols or DEFAULT_SYMBOLS
        tasks = [self._score_symbol(s, polygon_client, data_layer) for s in syms]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        out: list[ScanResult] = []
        for r in results:
            if isinstance(r, ScanResult):
                out.append(r)
            elif isinstance(r, Exception):
                log.warning("Scanner error: %s", r)

        out.sort(key=lambda r: r.score, reverse=True)
        return out

    async def _score_symbol(
        self,
        symbol: str,
        polygon_client,
        data_layer,
    ) -> ScanResult:
        """Score a single symbol."""
        res = ScanResult(symbol=symbol)
        try:
            # Try to get recent bars from polygon
            bars = None
            if polygon_client:
                try:
                    bars = await polygon_client.get_bars(
                        symbol, timespan="minute", limit=120
                    )
                except Exception:
                    pass

            if not bars or len(bars) < 30:
                # Fallback: try data_layer cache
                if data_layer and hasattr(data_layer, "get_cached_bars"):
                    bars = data_layer.get_cached_bars(symbol)

            if not bars or len(bars) < 20:
                res.note = "Insufficient data"
                return res

            closes = [b.get("c", b.get("close", 0)) for b in bars if b]
            highs = [b.get("h", b.get("high", 0)) for b in bars if b]
            lows = [b.get("l", b.get("low", 0)) for b in bars if b]
            volumes = [b.get("v", b.get("volume", 0)) for b in bars if b]

            if len(closes) < 20:
                res.note = "Insufficient data"
                return res

            spot = closes[-1]
            res.spot = spot

            # ── 1. Clean Trend (0-25) ────────────────────────────
            # Slope consistency: count consecutive same-direction 5-bar average moves
            sma5 = _rolling_mean(closes, 5)
            sma20 = _rolling_mean(closes, 20)

            if sma5 and sma20 and len(sma5) >= 5:
                # Direction consistency: fraction of recent 5-bar SMA segments trending same way
                diffs = [sma5[i] - sma5[i - 1] for i in range(1, len(sma5))]
                recent = diffs[-min(20, len(diffs)):]
                if recent:
                    pos = sum(1 for d in recent if d > 0)
                    neg = sum(1 for d in recent if d < 0)
                    consistency = max(pos, neg) / len(recent)
                    res.clean_trend = int(consistency * 25)

                # Bonus: price above/below SMA20
                if sma20:
                    if spot > sma20[-1] and res.clean_trend > 10:
                        res.clean_trend = min(25, res.clean_trend + 3)
                    elif spot < sma20[-1] and res.clean_trend > 10:
                        res.clean_trend = min(25, res.clean_trend + 3)

            # ── 2. Compression (0-25) ────────────────────────────
            # ATR squeeze: compare recent ATR to longer-term ATR
            recent_ranges = [highs[i] - lows[i] for i in range(-min(10, len(highs)), 0)]
            older_ranges = [highs[i] - lows[i] for i in range(-min(30, len(highs)), -10)] if len(highs) > 10 else recent_ranges

            recent_atr = sum(recent_ranges) / len(recent_ranges) if recent_ranges else 1
            older_atr = sum(older_ranges) / len(older_ranges) if older_ranges else 1
            res.atr = round(recent_atr, 4)

            if older_atr > 0:
                compression_ratio = recent_atr / older_atr
                if compression_ratio < 0.5:
                    res.compression = 25    # extreme squeeze
                elif compression_ratio < 0.7:
                    res.compression = 20
                elif compression_ratio < 0.85:
                    res.compression = 15
                elif compression_ratio < 1.0:
                    res.compression = 10
                else:
                    res.compression = 5     # expanding = less compression

            # ── 3. Breakout Proximity (0-25) ──────────────────────
            # Distance to recent high/low as a percentage of ATR
            recent_high = max(highs[-20:]) if len(highs) >= 20 else max(highs)
            recent_low = min(lows[-20:]) if len(lows) >= 20 else min(lows)

            dist_to_high = abs(spot - recent_high)
            dist_to_low = abs(spot - recent_low)
            nearest_level = min(dist_to_high, dist_to_low)

            if recent_atr > 0:
                prox_atr = nearest_level / recent_atr
                if prox_atr < 0.3:
                    res.breakout_prox = 25  # right at level
                elif prox_atr < 0.7:
                    res.breakout_prox = 20
                elif prox_atr < 1.0:
                    res.breakout_prox = 15
                elif prox_atr < 1.5:
                    res.breakout_prox = 10
                else:
                    res.breakout_prox = 5

            # ── 4. Positioning Edge (0-25) ────────────────────────
            # Volume trend: recent vs average
            if volumes and len(volumes) >= 20:
                recent_vol = sum(volumes[-10:]) / 10
                avg_vol = sum(volumes) / len(volumes)
                if avg_vol > 0:
                    vol_ratio = recent_vol / avg_vol
                    if vol_ratio > 1.5:
                        res.positioning_edge = 20  # volume surge
                    elif vol_ratio > 1.2:
                        res.positioning_edge = 15
                    elif vol_ratio > 0.8:
                        res.positioning_edge = 10
                    else:
                        res.positioning_edge = 5   # volume drying up
            else:
                res.positioning_edge = 8  # default

            # ── Regime label ─────────────────────────────────────
            if sma5 and sma20 and sma5[-1] > sma20[-1]:
                res.regime = "UPTREND" if res.clean_trend > 15 else "CHOPPY_UP"
            elif sma5 and sma20:
                res.regime = "DOWNTREND" if res.clean_trend > 15 else "CHOPPY_DOWN"
            else:
                res.regime = "UNKNOWN"

            # ── Total ────────────────────────────────────────────
            res.score = res.clean_trend + res.compression + res.breakout_prox + res.positioning_edge
            res.note = f"ATR={recent_atr:.2f}"

        except Exception as exc:
            log.warning("Scanner error for %s: %s", symbol, exc)
            res.note = f"Error: {exc}"

        return res


def _rolling_mean(data: list[float], window: int) -> list[float]:
    """Simple rolling mean."""
    if len(data) < window:
        return []
    out = []
    for i in range(window - 1, len(data)):
        out.append(sum(data[i - window + 1:i + 1]) / window)
    return out


# Module-level singleton
scanner_engine = ScannerEngine()
