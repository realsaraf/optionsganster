"""
Support / Resistance Engine
============================
Computes key price levels from daily OHLCV bars:
  1. Swing-point S/R (local highs/lows clustered)
  2. Fibonacci retracement + extension levels
  3. Volume Profile POC (Point of Control) from intraday bars
  4. Round-number magnets ($5 / $10 increments)

All computation is pure — no API calls.  Data fetching is handled
by the DataLayer, which caches daily.
"""
from dataclasses import dataclass, field
from typing import Optional
import math
import numpy as np
import pandas as pd


# ── Data classes ─────────────────────────────────────────────

@dataclass
class SRLevel:
    """A single support or resistance level."""
    price: float
    kind: str          # "support" | "resistance"
    source: str        # "swing" | "fib_ret" | "fib_ext" | "poc" | "round"
    strength: float    # 0.0-1.0 (how many touches / how significant)
    label: str = ""    # Human-readable label, e.g. "Fib 61.8%"


@dataclass
class SRResult:
    """Complete S/R analysis result."""
    levels: list[SRLevel]
    poc: Optional[float] = None                # Volume Point of Control
    vah: Optional[float] = None                # Value Area High (70% vol)
    val: Optional[float] = None                # Value Area Low  (70% vol)
    fib_high: Optional[float] = None           # Swing high used for Fib
    fib_low: Optional[float] = None            # Swing low used for Fib
    proximity_score: float = 0.0               # -1 to +1 for greeks engine
    proximity_detail: str = ""
    nearest_support: Optional[float] = None
    nearest_resistance: Optional[float] = None


# ── Engine ───────────────────────────────────────────────────

class SREngine:
    """
    Support / Resistance analysis engine.
    Pure computation — no I/O.  Feed it DataFrames.
    """

    # Fibonacci levels
    FIB_RETRACEMENT = [0.236, 0.382, 0.500, 0.618, 0.786]
    FIB_EXTENSIONS  = [1.0, 1.272, 1.618, 2.0]

    # Volume profile bins
    VP_NUM_BINS = 50

    # Clustering tolerance (% of price)
    CLUSTER_PCT = 0.003   # 0.3%

    # ── Public API ───────────────────────────────────────────

    def analyze(
        self,
        daily_bars: pd.DataFrame,
        underlying_price: float,
        intraday_bars: pd.DataFrame | None = None,
    ) -> SRResult:
        """
        Run full S/R analysis.

        Parameters
        ----------
        daily_bars : DataFrame with columns [date, open, high, low, close, volume]
                     90 trading days of daily candles.
        underlying_price : current price of the underlying.
        intraday_bars : (optional) DataFrame with 5-min bars for volume profile.
                        columns [datetime, open, high, low, close, volume]

        Returns
        -------
        SRResult with all levels, POC, proximity score.
        """
        if daily_bars.empty:
            return SRResult(levels=[], proximity_detail="No daily bar data available")

        all_levels: list[SRLevel] = []

        # 1. Swing-point S/R
        swing_levels = self._find_swing_levels(daily_bars, underlying_price)
        all_levels.extend(swing_levels)

        # 2. Fibonacci retracement + extensions
        fib_levels, fib_high, fib_low = self._compute_fibonacci(daily_bars, underlying_price)
        all_levels.extend(fib_levels)

        # 3. Volume Profile POC (if intraday data available)
        poc = None
        vah = None
        val = None
        if intraday_bars is not None and not intraday_bars.empty:
            poc, vah, val, vp_levels = self._compute_volume_profile(
                intraday_bars, underlying_price
            )
            all_levels.extend(vp_levels)

        # 4. Round-number magnets
        round_levels = self._round_number_levels(underlying_price)
        all_levels.extend(round_levels)

        # 5. Cluster nearby levels to avoid clutter
        all_levels = self._cluster_levels(all_levels)

        # 6. Sort by proximity to current price
        all_levels.sort(key=lambda l: abs(l.price - underlying_price))

        # 7. Score proximity for greeks engine
        nearest_sup, nearest_res = self._find_nearest(all_levels, underlying_price)
        prox_score, prox_detail = self._score_proximity(
            underlying_price, nearest_sup, nearest_res, all_levels
        )

        return SRResult(
            levels=all_levels,
            poc=poc,
            vah=vah,
            val=val,
            fib_high=fib_high,
            fib_low=fib_low,
            proximity_score=prox_score,
            proximity_detail=prox_detail,
            nearest_support=nearest_sup,
            nearest_resistance=nearest_res,
        )

    # ── Swing detection ──────────────────────────────────────

    def _find_swing_levels(
        self, df: pd.DataFrame, current_price: float, lookback: int = 5
    ) -> list[SRLevel]:
        """
        Identify swing highs and lows using a rolling window.
        A swing high: bar's high > high of `lookback` bars on each side.
        A swing low:  bar's low  < low  of `lookback` bars on each side.
        Then cluster nearby swings and assign strength by touch count.
        """
        highs = df["high"].values
        lows = df["low"].values
        n = len(df)

        raw_swing_highs: list[float] = []
        raw_swing_lows: list[float] = []

        for i in range(lookback, n - lookback):
            # Swing high
            if highs[i] == max(highs[i - lookback: i + lookback + 1]):
                raw_swing_highs.append(float(highs[i]))
            # Swing low
            if lows[i] == min(lows[i - lookback: i + lookback + 1]):
                raw_swing_lows.append(float(lows[i]))

        levels: list[SRLevel] = []

        # Cluster swing highs
        for cluster_price, count in self._cluster_prices(raw_swing_highs):
            kind = "resistance" if cluster_price >= current_price else "support"
            strength = min(1.0, count * 0.35)
            levels.append(SRLevel(
                price=round(cluster_price, 2),
                kind=kind,
                source="swing",
                strength=strength,
                label=f"Swing {'High' if kind == 'resistance' else 'Low'} ({count}x)",
            ))

        # Cluster swing lows
        for cluster_price, count in self._cluster_prices(raw_swing_lows):
            kind = "support" if cluster_price <= current_price else "resistance"
            strength = min(1.0, count * 0.35)
            levels.append(SRLevel(
                price=round(cluster_price, 2),
                kind=kind,
                source="swing",
                strength=strength,
                label=f"Swing {'Low' if kind == 'support' else 'High'} ({count}x)",
            ))

        return levels

    def _cluster_prices(self, prices: list[float]) -> list[tuple[float, int]]:
        """Cluster a list of prices within CLUSTER_PCT tolerance.
        Returns list of (avg_price, touch_count)."""
        if not prices:
            return []

        sorted_p = sorted(prices)
        clusters: list[list[float]] = [[sorted_p[0]]]

        for p in sorted_p[1:]:
            cluster_avg = sum(clusters[-1]) / len(clusters[-1])
            if abs(p - cluster_avg) / cluster_avg < self.CLUSTER_PCT:
                clusters[-1].append(p)
            else:
                clusters.append([p])

        return [(sum(c) / len(c), len(c)) for c in clusters]

    # ── Fibonacci ────────────────────────────────────────────

    def _compute_fibonacci(
        self, df: pd.DataFrame, current_price: float
    ) -> tuple[list[SRLevel], float, float]:
        """
        Compute Fibonacci retracements (23.6%-78.6%) and extensions (100%-200%)
        from the 90-day swing high/low range.
        """
        swing_high = float(df["high"].max())
        swing_low = float(df["low"].min())
        price_range = swing_high - swing_low

        if price_range < 0.01:
            return [], swing_high, swing_low

        levels: list[SRLevel] = []

        # Determine trend: if current price is in upper half → downtrend fib
        # if in lower half → uptrend fib
        midpoint = (swing_high + swing_low) / 2
        trend_up = current_price >= midpoint

        # Retracement levels
        for ratio in self.FIB_RETRACEMENT:
            if trend_up:
                # Uptrend: retrace from high → low
                price = swing_high - ratio * price_range
            else:
                # Downtrend: retrace from low → high
                price = swing_low + ratio * price_range

            kind = "support" if price < current_price else "resistance"
            pct_label = f"{ratio * 100:.1f}%"

            levels.append(SRLevel(
                price=round(price, 2),
                kind=kind,
                source="fib_ret",
                strength=0.7 if ratio in (0.382, 0.618) else 0.5,
                label=f"Fib {pct_label}",
            ))

        # Extension levels
        for ratio in self.FIB_EXTENSIONS:
            if trend_up:
                price = swing_high + (ratio - 1.0) * price_range
            else:
                price = swing_low - (ratio - 1.0) * price_range

            # Extensions are always beyond the range
            kind = "resistance" if price > current_price else "support"
            pct_label = f"{ratio * 100:.1f}%"

            levels.append(SRLevel(
                price=round(price, 2),
                kind=kind,
                source="fib_ext",
                strength=0.5 if ratio <= 1.272 else 0.35,
                label=f"Fib Ext {pct_label}",
            ))

        return levels, swing_high, swing_low

    # ── Volume Profile ───────────────────────────────────────

    def _compute_volume_profile(
        self,
        intraday_df: pd.DataFrame,
        current_price: float,
    ) -> tuple[Optional[float], Optional[float], Optional[float], list[SRLevel]]:
        """
        Build a volume profile from intraday bars.
        Returns (POC, VAH, VAL, levels).

        POC = price level with the highest traded volume.
        Value Area = 70% of total volume, centered on POC.
        """
        if intraday_df.empty:
            return None, None, None, []

        # Use VWAP-style: distribute bar volume across the bar's price range
        price_min = float(intraday_df["low"].min())
        price_max = float(intraday_df["high"].max())
        price_range = price_max - price_min

        if price_range < 0.01:
            return None, None, None, []

        bin_size = price_range / self.VP_NUM_BINS
        bins = np.zeros(self.VP_NUM_BINS)
        bin_prices = np.array([price_min + (i + 0.5) * bin_size for i in range(self.VP_NUM_BINS)])

        for _, bar in intraday_df.iterrows():
            bar_low = float(bar["low"])
            bar_high = float(bar["high"])
            bar_vol = float(bar["volume"])
            if bar_vol <= 0 or bar_high <= bar_low:
                continue

            # Determine which bins this bar spans
            low_bin = max(0, int((bar_low - price_min) / bin_size))
            high_bin = min(self.VP_NUM_BINS - 1, int((bar_high - price_min) / bin_size))
            n_bins = high_bin - low_bin + 1
            vol_per_bin = bar_vol / n_bins if n_bins > 0 else 0

            for b in range(low_bin, high_bin + 1):
                bins[b] += vol_per_bin

        # POC = bin with highest volume
        poc_idx = int(np.argmax(bins))
        poc_price = float(bin_prices[poc_idx])

        # Value Area (70% of total volume, expanding from POC)
        total_vol = float(np.sum(bins))
        if total_vol <= 0:
            return round(poc_price, 2), None, None, []

        target_vol = total_vol * 0.70
        va_vol = float(bins[poc_idx])
        va_low_idx = poc_idx
        va_high_idx = poc_idx

        while va_vol < target_vol:
            # Expand the side with more volume
            expand_low = bins[va_low_idx - 1] if va_low_idx > 0 else 0
            expand_high = bins[va_high_idx + 1] if va_high_idx < self.VP_NUM_BINS - 1 else 0

            if expand_low >= expand_high and va_low_idx > 0:
                va_low_idx -= 1
                va_vol += expand_low
            elif va_high_idx < self.VP_NUM_BINS - 1:
                va_high_idx += 1
                va_vol += expand_high
            else:
                break

        vah = float(bin_prices[va_high_idx]) + bin_size / 2
        val = float(bin_prices[va_low_idx]) - bin_size / 2

        # Build levels
        levels: list[SRLevel] = []

        levels.append(SRLevel(
            price=round(poc_price, 2),
            kind="support" if poc_price <= current_price else "resistance",
            source="poc",
            strength=0.85,
            label="POC (Volume)",
        ))

        levels.append(SRLevel(
            price=round(vah, 2),
            kind="resistance" if vah >= current_price else "support",
            source="poc",
            strength=0.65,
            label="VAH (70%)",
        ))

        levels.append(SRLevel(
            price=round(val, 2),
            kind="support" if val <= current_price else "resistance",
            source="poc",
            strength=0.65,
            label="VAL (70%)",
        ))

        return round(poc_price, 2), round(vah, 2), round(val, 2), levels

    # ── Round numbers ────────────────────────────────────────

    def _round_number_levels(self, current_price: float) -> list[SRLevel]:
        """
        Generate round-number magnets: $5 and $10 increments
        within ±5% of current price.
        """
        levels: list[SRLevel] = []
        band = current_price * 0.05  # ±5%

        # Determine increment based on price level
        if current_price > 200:
            increments = [10, 25]    # $10 and $25 rounds for high-priced
        elif current_price > 50:
            increments = [5, 10]     # $5 and $10 rounds
        else:
            increments = [1, 5]      # $1 and $5 rounds

        for inc in increments:
            start = int((current_price - band) / inc) * inc
            end = int((current_price + band) / inc + 1) * inc

            for price in range(start, end + 1, inc):
                price_f = float(price)
                if abs(price_f - current_price) < band and price_f > 0:
                    kind = "support" if price_f < current_price else "resistance"
                    # Bigger rounds get more strength
                    strength = 0.45 if inc >= 10 else 0.30
                    levels.append(SRLevel(
                        price=price_f,
                        kind=kind,
                        source="round",
                        strength=strength,
                        label=f"${price}",
                    ))

        return levels

    # ── Clustering ───────────────────────────────────────────

    def _cluster_levels(self, levels: list[SRLevel]) -> list[SRLevel]:
        """
        Merge levels that are within CLUSTER_PCT of each other.
        Keeps the level with the highest strength; combines labels.
        """
        if not levels:
            return []

        sorted_levels = sorted(levels, key=lambda l: l.price)
        merged: list[SRLevel] = [sorted_levels[0]]

        for lvl in sorted_levels[1:]:
            prev = merged[-1]
            if prev.price > 0 and abs(lvl.price - prev.price) / prev.price < self.CLUSTER_PCT:
                # Merge: keep the stronger one, boost strength
                if lvl.strength > prev.strength:
                    lvl.strength = min(1.0, lvl.strength + 0.15)
                    if prev.label and prev.label not in lvl.label:
                        lvl.label = f"{lvl.label} + {prev.label}"
                    merged[-1] = lvl
                else:
                    prev.strength = min(1.0, prev.strength + 0.15)
                    if lvl.label and lvl.label not in prev.label:
                        prev.label = f"{prev.label} + {lvl.label}"
            else:
                merged.append(lvl)

        return merged

    # ── Proximity helpers ────────────────────────────────────

    @staticmethod
    def _find_nearest(
        levels: list[SRLevel], current_price: float
    ) -> tuple[Optional[float], Optional[float]]:
        """Find nearest support (below) and resistance (above) current price."""
        supports = [l.price for l in levels if l.price < current_price and l.kind == "support"]
        resistances = [l.price for l in levels if l.price > current_price and l.kind == "resistance"]

        nearest_sup = max(supports) if supports else None
        nearest_res = min(resistances) if resistances else None

        return nearest_sup, nearest_res

    def _score_proximity(
        self,
        current_price: float,
        nearest_support: Optional[float],
        nearest_resistance: Optional[float],
        all_levels: list[SRLevel],
    ) -> tuple[float, str]:
        """
        Score how price relates to key S/R levels for the greeks engine.

        Returns (score, detail) where score is -1.0 to +1.0:
          +0.4 to +0.8 : price at strong support → bullish bounce expected
          -0.4 to -0.8 : price at strong resistance → bearish rejection expected
          near 0       : price in no-man's land, no S/R edge
        """
        if not nearest_support and not nearest_resistance:
            return 0.0, "No S/R levels identified"

        score = 0.0
        details: list[str] = []

        # Distance to nearest support/resistance as % of price
        if nearest_support:
            sup_dist_pct = (current_price - nearest_support) / current_price * 100
        else:
            sup_dist_pct = 99.0

        if nearest_resistance:
            res_dist_pct = (nearest_resistance - current_price) / current_price * 100
        else:
            res_dist_pct = 99.0

        # Very close to support (< 0.5%) → bullish
        if sup_dist_pct < 0.5:
            # Find strength of nearest support level
            sup_levels = [l for l in all_levels if abs(l.price - nearest_support) < 0.01]
            strength = max((l.strength for l in sup_levels), default=0.5)
            score += 0.4 + strength * 0.4   # 0.4 to 0.8
            details.append(
                f"At support ${nearest_support:.2f} ({sup_dist_pct:.1f}% away, "
                f"strength {strength:.0%}) → bounce expected"
            )
        elif sup_dist_pct < 1.5:
            score += 0.2
            details.append(f"Near support ${nearest_support:.2f} ({sup_dist_pct:.1f}% away)")

        # Very close to resistance (< 0.5%) → bearish
        if res_dist_pct < 0.5:
            res_levels = [l for l in all_levels if abs(l.price - nearest_resistance) < 0.01]
            strength = max((l.strength for l in res_levels), default=0.5)
            score -= 0.4 + strength * 0.4   # -0.4 to -0.8
            details.append(
                f"At resistance ${nearest_resistance:.2f} ({res_dist_pct:.1f}% away, "
                f"strength {strength:.0%}) → rejection expected"
            )
        elif res_dist_pct < 1.5:
            score -= 0.2
            details.append(f"Near resistance ${nearest_resistance:.2f} ({res_dist_pct:.1f}% away)")

        # If between support and resistance with room, neutral
        if not details:
            details.append(
                f"Between S${nearest_support:.2f if nearest_support else 0:.2f} "
                f"and R${nearest_resistance:.2f if nearest_resistance else 0:.2f} — "
                f"no immediate S/R edge"
            )

        # Clamp
        score = max(-1.0, min(1.0, score))

        return round(score, 3), " | ".join(details)


# Global engine instance
sr_engine = SREngine()
