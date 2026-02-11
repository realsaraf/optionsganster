"""
Volume Price Analysis (VPA) Engine – v2
========================================
Detects price-volume patterns to predict direction.

Improvements over v1
────────────────────
* Multi-bar pattern detection (confirmation, no-demand/no-supply follow-through)
* Upper/lower wick ratio analysis for pin-bar detection
* Time-decay weighted bias calculation (recent signals matter more)
* Configurable thresholds via constructor
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import pandas as pd
import numpy as np


# ── Signal taxonomy ─────────────────────────────────────────

class VPASignal(Enum):
    STRONG_BULLISH = "strong_bullish"
    STRONG_BEARISH = "strong_bearish"
    WEAK_UP = "weak_up"           # Low-volume rally → may reverse down
    WEAK_DOWN = "weak_down"       # Low-volume selloff → may bounce up
    CLIMAX_TOP = "climax_top"     # Extreme volume at highs → reversal risk
    CLIMAX_BOTTOM = "climax_bottom"  # Extreme volume at lows → bounce risk
    ACCUMULATION = "accumulation" # Smart money buying
    DISTRIBUTION = "distribution" # Smart money selling
    TEST_SUPPORT = "test_support"       # Testing lows with low volume → bullish
    TEST_RESISTANCE = "test_resistance" # Testing highs with low volume → bearish
    NO_DEMAND = "no_demand"       # Low-volume up bar after rally → bearish
    NO_SUPPLY = "no_supply"       # Low-volume down bar after selloff → bullish
    PIN_BAR_BULL = "pin_bar_bull" # Long lower wick rejection → bullish
    PIN_BAR_BEAR = "pin_bar_bear" # Long upper wick rejection → bearish
    CONFIRMED_REVERSAL_UP = "confirmed_reversal_up"   # Multi-bar reversal pattern
    CONFIRMED_REVERSAL_DOWN = "confirmed_reversal_down"
    NEUTRAL = "neutral"


@dataclass
class VPAResult:
    signal: VPASignal
    confidence: float       # 0.0 – 1.0
    description: str
    bar_index: int
    datetime: str
    price: float
    volume: int
    volume_ratio: float     # vs rolling average


# ── Engine ───────────────────────────────────────────────────

@dataclass
class VPAThresholds:
    """Tuneable thresholds – pass to VPAEngine to customise."""
    volume_avg_period: int = 10
    high_volume: float = 1.5
    very_high_volume: float = 2.5
    low_volume: float = 0.7
    very_low_volume: float = 0.5
    close_near_high: float = 0.7
    close_near_low: float = 0.3
    wick_ratio_pin: float = 2.0   # wick ≥ 2× body → pin bar
    bias_lookback: int = 10
    bias_decay: float = 0.85      # exponential decay factor


class VPAEngine:
    """
    Volume Price Analysis Engine

    Analyses price and volume together to detect:
    - Strong / weak / climax moves
    - Accumulation & distribution
    - Support / resistance tests
    - Pin bars (wick rejection)
    - Multi-bar confirmation patterns (no-demand, no-supply, confirmed reversals)
    """

    def __init__(self, thresholds: VPAThresholds | None = None):
        self.t = thresholds or VPAThresholds()

    # ── public API ───────────────────────────────────────────

    def analyze(self, df: pd.DataFrame) -> list[VPAResult]:
        """
        Analyse OHLCV data.

        Parameters
        ----------
        df : DataFrame with columns [datetime, open, high, low, close, volume]

        Returns
        -------
        list[VPAResult] for every bar starting from where we have enough history.
        """
        min_bars = max(3, self.t.volume_avg_period + 1)
        if len(df) < min_bars:
            return []

        df = df.copy()

        # ── derived columns ──────────────────────────────────
        df["price_change"] = df["close"] - df["open"]
        df["bar_range"] = df["high"] - df["low"]
        df["body"] = (df["close"] - df["open"]).abs()

        df["close_position"] = np.where(
            df["bar_range"] > 0,
            (df["close"] - df["low"]) / df["bar_range"],
            0.5,
        )

        # Wick ratios (relative to body; body 0 → treat as doji)
        safe_body = df["body"].replace(0, np.nan)
        df["upper_wick"] = df["high"] - df[["open", "close"]].max(axis=1)
        df["lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]
        df["upper_wick_ratio"] = (df["upper_wick"] / safe_body).fillna(0)
        df["lower_wick_ratio"] = (df["lower_wick"] / safe_body).fillna(0)

        actual_period = min(self.t.volume_avg_period, len(df) - 1)
        df["volume_avg"] = (
            df["volume"].rolling(window=actual_period, min_periods=1).mean()
        )
        df["volume_ratio"] = df["volume"] / df["volume_avg"].replace(0, 1)

        # ── per-bar analysis ─────────────────────────────────
        results: list[VPAResult] = []
        start_idx = min(actual_period, len(df) - 1)
        for i in range(start_idx, len(df)):
            result = self._analyze_bar(df, i)
            results.append(result)

        # ── multi-bar overlay (upgrades signals in-place) ────
        self._apply_multi_bar_patterns(results, df, start_idx)

        return results

    # ── single-bar analysis ──────────────────────────────────

    def _analyze_bar(self, df: pd.DataFrame, idx: int) -> VPAResult:
        row = df.iloc[idx]
        prev_rows = df.iloc[max(0, idx - 5): idx]

        price_change = row["price_change"]
        close_pos = row["close_position"]
        vol_ratio = row["volume_ratio"]
        upper_wick_r = row["upper_wick_ratio"]
        lower_wick_r = row["lower_wick_ratio"]

        is_up = price_change > 0
        is_down = price_change < 0

        hi_vol = vol_ratio > self.t.high_volume
        vhi_vol = vol_ratio > self.t.very_high_volume
        lo_vol = vol_ratio < self.t.low_volume
        vlo_vol = vol_ratio < self.t.very_low_volume

        near_high = close_pos > self.t.close_near_high
        near_low = close_pos < self.t.close_near_low

        signal = VPASignal.NEUTRAL
        confidence = 0.5
        description = "No clear signal"

        # ── CLIMAX PATTERNS (highest volume priority) ────────
        if vhi_vol:
            if is_up and near_high:
                signal = VPASignal.CLIMAX_TOP
                confidence = 0.80
                description = "Climax buying – potential top, watch for reversal"
            elif is_down and near_low:
                signal = VPASignal.CLIMAX_BOTTOM
                confidence = 0.80
                description = "Climax selling – potential bottom, watch for bounce"

        # ── STRONG MOVES ─────────────────────────────────────
        elif hi_vol:
            if is_up and near_high:
                signal = VPASignal.STRONG_BULLISH
                confidence = 0.75
                description = "Strong buying – high volume, close near high"
            elif is_down and near_low:
                signal = VPASignal.STRONG_BEARISH
                confidence = 0.75
                description = "Strong selling – high volume, close near low"
            elif is_up and near_low:
                signal = VPASignal.DISTRIBUTION
                confidence = 0.70
                description = "Distribution – up bar but close near low, selling into strength"
            elif is_down and near_high:
                signal = VPASignal.ACCUMULATION
                confidence = 0.70
                description = "Accumulation – down bar but close near high, buying the dip"

        # ── PIN BARS (wick rejection – only when volume is normal) ─
        elif lower_wick_r >= self.t.wick_ratio_pin and near_high:
            signal = VPASignal.PIN_BAR_BULL
            confidence = 0.72
            description = "Bullish pin bar – long lower wick rejection"
        elif upper_wick_r >= self.t.wick_ratio_pin and near_low:
            signal = VPASignal.PIN_BAR_BEAR
            confidence = 0.72
            description = "Bearish pin bar – long upper wick rejection"

        # ── WEAK MOVES (potential reversals) ─────────────────
        elif lo_vol:
            if is_up:
                signal = VPASignal.WEAK_UP
                confidence = 0.60
                description = "Weak up move – low volume, may reverse down"
            elif is_down:
                signal = VPASignal.WEAK_DOWN
                confidence = 0.60
                description = "Weak down move – low volume, may bounce up"

        # ── TEST PATTERNS (very-low-volume probes) ───────────
        if vlo_vol and len(prev_rows) > 0:
            recent_lows = prev_rows["low"].min()
            recent_highs = prev_rows["high"].max()

            if row["low"] <= recent_lows * 1.002 and near_high:
                signal = VPASignal.TEST_SUPPORT
                confidence = 0.70
                description = "Testing support with low volume – no sellers, bullish"
            elif row["high"] >= recent_highs * 0.998 and near_low:
                signal = VPASignal.TEST_RESISTANCE
                confidence = 0.70
                description = "Testing resistance with low volume – no buyers, bearish"

        return VPAResult(
            signal=signal,
            confidence=confidence,
            description=description,
            bar_index=idx,
            datetime=str(row["datetime"]),
            price=float(row["close"]),
            volume=int(row["volume"]),
            volume_ratio=float(vol_ratio),
        )

    # ── multi-bar pattern overlay ────────────────────────────

    def _apply_multi_bar_patterns(
        self, results: list[VPAResult], df: pd.DataFrame, start_idx: int
    ) -> None:
        """
        Walk through results and upgrade signals when multi-bar
        sequences confirm a pattern.
        """
        if len(results) < 2:
            return

        for i in range(1, len(results)):
            prev = results[i - 1]
            curr = results[i]

            # ── NO DEMAND: after a rally bar, next bar is weak up ────
            if prev.signal in (VPASignal.STRONG_BULLISH, VPASignal.CLIMAX_TOP):
                if curr.signal == VPASignal.WEAK_UP:
                    curr.signal = VPASignal.NO_DEMAND
                    curr.confidence = 0.73
                    curr.description = (
                        "No demand – low-volume up bar after rally, buyers exhausted"
                    )

            # ── NO SUPPLY: after a selloff bar, next bar is weak down ─
            if prev.signal in (VPASignal.STRONG_BEARISH, VPASignal.CLIMAX_BOTTOM):
                if curr.signal == VPASignal.WEAK_DOWN:
                    curr.signal = VPASignal.NO_SUPPLY
                    curr.confidence = 0.73
                    curr.description = (
                        "No supply – low-volume down bar after selloff, sellers exhausted"
                    )

            # ── CONFIRMED REVERSAL UP ────────────────────────────────
            # Climax bottom → followed by strong bullish or accumulation
            if prev.signal == VPASignal.CLIMAX_BOTTOM:
                if curr.signal in (VPASignal.STRONG_BULLISH, VPASignal.ACCUMULATION):
                    curr.signal = VPASignal.CONFIRMED_REVERSAL_UP
                    curr.confidence = 0.85
                    curr.description = (
                        "Confirmed reversal up – climax selling followed by strong buying"
                    )

            # ── CONFIRMED REVERSAL DOWN ──────────────────────────────
            # Climax top → followed by strong bearish or distribution
            if prev.signal == VPASignal.CLIMAX_TOP:
                if curr.signal in (VPASignal.STRONG_BEARISH, VPASignal.DISTRIBUTION):
                    curr.signal = VPASignal.CONFIRMED_REVERSAL_DOWN
                    curr.confidence = 0.85
                    curr.description = (
                        "Confirmed reversal down – climax buying followed by strong selling"
                    )

    # ── bias with time-decay weighting ───────────────────────

    def get_bias(self, results: list[VPAResult], lookback: int | None = None) -> dict:
        """
        Compute overall bias from recent signals using exponential
        time-decay so recent bars weigh more.

        Returns
        -------
        dict with keys: bias, strength (0-1), reason
        """
        lookback = lookback or self.t.bias_lookback
        if not results:
            return {"bias": "neutral", "strength": 0, "reason": "No data"}

        recent = results[-lookback:] if len(results) >= lookback else results

        bullish_set = {
            VPASignal.STRONG_BULLISH,
            VPASignal.ACCUMULATION,
            VPASignal.WEAK_DOWN,
            VPASignal.TEST_SUPPORT,
            VPASignal.CLIMAX_BOTTOM,
            VPASignal.NO_SUPPLY,
            VPASignal.PIN_BAR_BULL,
            VPASignal.CONFIRMED_REVERSAL_UP,
        }
        bearish_set = {
            VPASignal.STRONG_BEARISH,
            VPASignal.DISTRIBUTION,
            VPASignal.WEAK_UP,
            VPASignal.TEST_RESISTANCE,
            VPASignal.CLIMAX_TOP,
            VPASignal.NO_DEMAND,
            VPASignal.PIN_BAR_BEAR,
            VPASignal.CONFIRMED_REVERSAL_DOWN,
        }

        n = len(recent)
        decay = self.t.bias_decay

        bullish_score = 0.0
        bearish_score = 0.0
        for i, r in enumerate(recent):
            weight = decay ** (n - 1 - i)  # most recent bar → weight 1.0
            if r.signal in bullish_set:
                bullish_score += r.confidence * weight
            elif r.signal in bearish_set:
                bearish_score += r.confidence * weight

        total = bullish_score + bearish_score
        if total == 0:
            return {"bias": "neutral", "strength": 0, "reason": "No significant signals"}

        max_possible = sum(decay ** (n - 1 - i) for i in range(n)) * 0.7
        if bullish_score > bearish_score * 1.2:
            strength = min(1.0, bullish_score / max_possible) if max_possible else 0
            return {
                "bias": "bullish",
                "strength": round(strength, 2),
                "reason": (
                    f"Bullish signals dominating "
                    f"({bullish_score:.1f} vs {bearish_score:.1f} weighted)"
                ),
            }
        elif bearish_score > bullish_score * 1.2:
            strength = min(1.0, bearish_score / max_possible) if max_possible else 0
            return {
                "bias": "bearish",
                "strength": round(strength, 2),
                "reason": (
                    f"Bearish signals dominating "
                    f"({bearish_score:.1f} vs {bullish_score:.1f} weighted)"
                ),
            }
        else:
            return {
                "bias": "neutral",
                "strength": 0.3,
                "reason": "Mixed signals, no clear direction",
            }


# Global engine instance
vpa_engine = VPAEngine()
