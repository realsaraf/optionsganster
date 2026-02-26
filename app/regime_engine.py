"""
Market Regime Engine
=====================
Classifies the current market environment into one of 5 regimes
based on price structure, VWAP relationship, RSI, ATR, and volume.

Regimes:
    TREND_UP           — Clear uptrend: HH/HL structure, above VWAP
    TREND_DOWN         — Clear downtrend: LL/LH structure, below VWAP
    RANGE_CHOP         — Oscillating around VWAP, low ATR expansion
    BREAKOUT_ATTEMPT   — Price pressing against key level with rising vol
    REVERSAL_EXHAUSTION — Extended move with divergence / climax volume

Regime is frozen for 15 minutes after classification to prevent
whipsawing and false invalidation on noisy prints.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional

import pandas as pd

from app.indicators import (
    atr_value,
    ema,
    rsi_value,
    slope,
    vwap,
    vwap_band_values,
    higher_highs_lows,
    lower_lows_highs,
    volume_ratio,
)


# ── Regime taxonomy ──────────────────────────────────────────

class MarketRegime(Enum):
    TREND_UP             = "TREND_UP"
    TREND_DOWN           = "TREND_DOWN"
    RANGE_CHOP           = "RANGE_CHOP"
    BREAKOUT_ATTEMPT     = "BREAKOUT_ATTEMPT"
    REVERSAL_EXHAUSTION  = "REVERSAL_EXHAUSTION"


@dataclass
class RegimeResult:
    """Current market regime with confidence and freeze window."""
    regime: MarketRegime
    confidence: float           # 0.0 – 1.0
    frozen_until: Optional[datetime] = None  # None = not frozen
    detail: str = ""

    # Supporting indicators for downstream engines
    vwap_current: float = 0.0
    vwap_upper: float = 0.0     # VWAP + 1σ
    vwap_lower: float = 0.0     # VWAP - 1σ
    atr_current: float = 0.0
    rsi_current: float = 50.0
    ema_9: float = 0.0
    ema_20: float = 0.0
    price_vs_vwap: str = "at"   # "above" | "below" | "at"

    @property
    def is_directional(self) -> bool:
        return self.regime in (MarketRegime.TREND_UP, MarketRegime.TREND_DOWN)

    @property
    def is_frozen(self) -> bool:
        if self.frozen_until is None:
            return False
        return datetime.utcnow() < self.frozen_until


# ── Engine ───────────────────────────────────────────────────

class RegimeEngine:
    """
    Classify market regime from intraday OHLCV bars.

    Usage
    -----
    engine = RegimeEngine()
    result = engine.classify(intraday_df)

    The engine maintains state: once a regime is classified, it is
    frozen for FREEZE_MINUTES to prevent rapid oscillation.
    """

    FREEZE_MINUTES = 15
    EMA_FAST = 9
    EMA_SLOW = 20
    ATR_PERIOD = 14
    RSI_PERIOD = 14

    # Thresholds
    VWAP_PROXIMITY_PCT    = 0.001   # Within 0.1% of VWAP = "at VWAP"
    SLOPE_TREND_THRESHOLD = 0.02    # VWAP slope per bar for "trending"
    MIN_HH_HL_RATIO       = 0.55    # 55% of bars must be HH or HL
    VOLUME_SPIKE_RATIO    = 1.8     # Volume spike = 1.8× average

    def __init__(self):
        self._last_results: dict[str, RegimeResult] = {}  # per-symbol freeze

    def classify(
        self,
        df: pd.DataFrame,
        force_reclassify: bool = False,
        symbol: str = "QQQ",
    ) -> RegimeResult:
        """
        Classify the current market regime.

        Parameters
        ----------
        df : intraday OHLCV DataFrame (1 or 5-min bars), today's session
        force_reclassify : bypass freeze window
        symbol : ticker symbol (used to keep freeze per-symbol)

        Returns
        -------
        RegimeResult
        """
        sym_key = symbol.upper()
        last = self._last_results.get(sym_key)

        # Return frozen result if within freeze window for THIS symbol
        if (
            not force_reclassify
            and last is not None
            and last.is_frozen
        ):
            return last

        if df is None or len(df) < 10:
            result = RegimeResult(
                regime=MarketRegime.RANGE_CHOP,
                confidence=0.30,
                detail="Insufficient data — defaulting to RANGE_CHOP",
            )
            self._last_results[sym_key] = result
            return result

        result = self._compute_regime(df)

        # Set freeze window
        result.frozen_until = datetime.utcnow() + timedelta(
            minutes=self.FREEZE_MINUTES
        )

        self._last_results[sym_key] = result
        return result

    def _compute_regime(self, df: pd.DataFrame) -> RegimeResult:
        """Core regime computation — no state, pure analysis."""
        # ── Base indicators ──────────────────────────────
        current_price = float(df["close"].iloc[-1])

        atr_val   = atr_value(df, self.ATR_PERIOD)
        rsi_val   = rsi_value(df, self.RSI_PERIOD)
        vwap_ser  = vwap(df)
        vwap_val  = float(vwap_ser.iloc[-1]) if len(vwap_ser) > 0 else current_price
        vwap_up, vwap_lo = vwap_band_values(df, num_stdev=1.0)

        ema9_ser  = ema(df["close"], self.EMA_FAST)
        ema20_ser = ema(df["close"], self.EMA_SLOW)
        ema9_val  = float(ema9_ser.iloc[-1])
        ema20_val = float(ema20_ser.iloc[-1])

        vwap_slope_val = slope(vwap_ser, window=15)

        hh, hl     = higher_highs_lows(df, window=20)
        ll, lh     = lower_lows_highs(df, window=20)
        vol_ratio  = volume_ratio(df, period=20)

        # Price position relative to VWAP
        vwap_pct_diff = (current_price - vwap_val) / vwap_val if vwap_val > 0 else 0.0
        above_vwap = vwap_pct_diff > self.VWAP_PROXIMITY_PCT
        below_vwap = vwap_pct_diff < -self.VWAP_PROXIMITY_PCT
        price_vs_vwap = "above" if above_vwap else ("below" if below_vwap else "at")

        # Structural counts (over last 20 bars)
        total_bars = min(20, len(df) - 1)
        hh_hl_ratio = (hh + hl) / (2 * total_bars) if total_bars > 0 else 0.0
        ll_lh_ratio = (ll + lh) / (2 * total_bars) if total_bars > 0 else 0.0

        # EMAs aligned
        ema_bullish = ema9_val > ema20_val
        ema_bearish = ema9_val < ema20_val

        # RSI extremes
        rsi_overbought  = rsi_val > 70
        rsi_oversold    = rsi_val < 30
        rsi_extreme     = rsi_overbought or rsi_oversold

        # Breakout: pressing against session extreme with rising volume
        session_high = float(df["high"].max())
        session_low  = float(df["low"].min())
        pct_from_high = (session_high - current_price) / session_high if session_high > 0 else 1
        pct_from_low  = (current_price - session_low) / session_low if session_low > 0 else 1

        pressing_high = pct_from_high < 0.002   # within 0.2% of session high
        pressing_low  = pct_from_low  < 0.002   # within 0.2% of session low
        volume_rising = vol_ratio > self.VOLUME_SPIKE_RATIO

        # ── Scoring ──────────────────────────────────────
        scores: dict[str, float] = {
            MarketRegime.TREND_UP.value:            0.0,
            MarketRegime.TREND_DOWN.value:          0.0,
            MarketRegime.RANGE_CHOP.value:          0.0,
            MarketRegime.BREAKOUT_ATTEMPT.value:    0.0,
            MarketRegime.REVERSAL_EXHAUSTION.value: 0.0,
        }

        # TREND_UP evidence
        if above_vwap:      scores[MarketRegime.TREND_UP.value] += 0.25
        if ema_bullish:     scores[MarketRegime.TREND_UP.value] += 0.20
        if hh_hl_ratio > self.MIN_HH_HL_RATIO:
                            scores[MarketRegime.TREND_UP.value] += 0.30
        if vwap_slope_val > self.SLOPE_TREND_THRESHOLD:
                            scores[MarketRegime.TREND_UP.value] += 0.15
        if rsi_val > 55:    scores[MarketRegime.TREND_UP.value] += 0.10

        # TREND_DOWN evidence
        if below_vwap:      scores[MarketRegime.TREND_DOWN.value] += 0.25
        if ema_bearish:     scores[MarketRegime.TREND_DOWN.value] += 0.20
        if ll_lh_ratio > self.MIN_HH_HL_RATIO:
                            scores[MarketRegime.TREND_DOWN.value] += 0.30
        if vwap_slope_val < -self.SLOPE_TREND_THRESHOLD:
                            scores[MarketRegime.TREND_DOWN.value] += 0.15
        if rsi_val < 45:    scores[MarketRegime.TREND_DOWN.value] += 0.10

        # RANGE_CHOP evidence (price oscillates around VWAP, flat ema spread)
        ema_spread_pct = abs(ema9_val - ema20_val) / ema20_val if ema20_val > 0 else 0
        if price_vs_vwap == "at":
                            scores[MarketRegime.RANGE_CHOP.value] += 0.20
        if ema_spread_pct < 0.002:
                            scores[MarketRegime.RANGE_CHOP.value] += 0.30
        if abs(vwap_slope_val) < self.SLOPE_TREND_THRESHOLD / 2:
                            scores[MarketRegime.RANGE_CHOP.value] += 0.25
        if hh_hl_ratio < 0.35 and ll_lh_ratio < 0.35:
                            scores[MarketRegime.RANGE_CHOP.value] += 0.25

        # BREAKOUT_ATTEMPT
        if (pressing_high or pressing_low) and volume_rising:
                            scores[MarketRegime.BREAKOUT_ATTEMPT.value] += 0.50
        if pressing_high and above_vwap:
                            scores[MarketRegime.BREAKOUT_ATTEMPT.value] += 0.30
        if pressing_low and below_vwap:
                            scores[MarketRegime.BREAKOUT_ATTEMPT.value] += 0.30

        # REVERSAL_EXHAUSTION
        if rsi_extreme and volume_rising:
                            scores[MarketRegime.REVERSAL_EXHAUSTION.value] += 0.40
        if rsi_overbought and below_vwap:
                            scores[MarketRegime.REVERSAL_EXHAUSTION.value] += 0.30
        if rsi_oversold and above_vwap:
                            scores[MarketRegime.REVERSAL_EXHAUSTION.value] += 0.30
        # Exhaustion candle: large spike then reversal
        if len(df) >= 3:
            last3 = df.tail(3)
            range3 = float((last3["high"] - last3["low"]).max())
            avg_range = atr_val if atr_val > 0 else range3
            if range3 > avg_range * 2.0:
                                scores[MarketRegime.REVERSAL_EXHAUSTION.value] += 0.30

        # ── Select winning regime ─────────────────────────
        best_regime_name = max(scores, key=lambda k: scores[k])
        best_score       = scores[best_regime_name]
        best_regime      = MarketRegime(best_regime_name)

        # Normalise confidence: best_score is raw sum of evidence weights
        # Max possible is ~1.0 for most regimes
        confidence = min(0.95, best_score)

        # Build detail string
        details = [
            f"RSI={rsi_val:.0f}",
            f"VWAP={price_vs_vwap}",
            f"slope={vwap_slope_val:+.4f}",
            f"EMA {'bull' if ema_bullish else 'bear'}",
            f"HH/HL={hh}/{hl}",
        ]
        detail = f"{best_regime.value} | " + " | ".join(details)

        return RegimeResult(
            regime=best_regime,
            confidence=round(confidence, 2),
            detail=detail,
            vwap_current=round(vwap_val, 2),
            vwap_upper=round(vwap_up, 2) if vwap_up else 0.0,
            vwap_lower=round(vwap_lo, 2) if vwap_lo else 0.0,
            atr_current=round(atr_val, 4),
            rsi_current=round(rsi_val, 1),
            ema_9=round(ema9_val, 2),
            ema_20=round(ema20_val, 2),
            price_vs_vwap=price_vs_vwap,
        )

    def reset(self, symbol: str | None = None):
        """Clear frozen state (e.g., new session start)."""
        if symbol:
            self._last_results.pop(symbol.upper(), None)
        else:
            self._last_results.clear()


# ── Module-level singleton ───────────────────────────────────
regime_engine = RegimeEngine()
