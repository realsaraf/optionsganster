"""
Setup Engine
=============
Detects 7 high-probability trade setups from the market regime,
technical indicators, S/R levels, and options chain data.

Setups:
    1. ORB Breakout          — Opening Range Breakout (first 15 min)
    2. VWAP Reclaim          — Price reclaims VWAP after being below/above
    3. Pullback Continuation — Deep pullback to VWAP/EMA in a trend
    4. Exhaustion Fade       — Extended move with RSI divergence + rejection wick
    5. Range Mean Reversion  — Touch of VWAP band in choppy market
    6. Level Break Retest    — Break key S/R → pull back → hold 2-3 bars
    7. Dealer Hedge Zone     — High-OI strike cluster acting as magnet/wall

Each setup returns a SetupAlert dataclass or None if conditions not met.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional

import pandas as pd

from app.indicators import (
    atr_value,
    ema,
    is_rejection_wick_down,
    is_rejection_wick_up,
    opening_range_today,
    rsi_value,
    slope,
    vwap,
    vwap_bands,
    volume_ratio,
)
from app.regime_engine import MarketRegime, RegimeResult
from app.chain_analytics import ChainMetrics
from app.sr_engine import SRResult


# ── Data class ───────────────────────────────────────────────

@dataclass
class SetupAlert:
    """A detected trade setup."""
    name: str                       # "ORB Breakout", "VWAP Reclaim", etc.
    direction: str                  # "CALL" or "PUT"
    trigger_price: float            # Exact level where setup fires

    # Entry / exit plan
    entry_condition: str            # e.g. "1m close above 610.20"
    stop_price: float
    target_1: float
    target_2: Optional[float]
    time_stop_minutes: int          # Exit if not working within N minutes

    # Context
    regime: str                     # MarketRegime.value
    reasons: list[str]              # 3-5 bullet points explaining the setup
    confidence: float               # 0.0-1.0 raw setup score
    is_advanced: bool = False       # Requires more skill (e.g. Exhaustion Fade)

    # Derived (set after edge scoring)
    detected_at: Optional[str] = None  # ISO timestamp


# ── Engine ───────────────────────────────────────────────────

class SetupEngine:
    """
    Detects trade setups by evaluating conditions against the current
    regime, indicators, S/R levels, and chain metrics.

    All detect_* methods return a SetupAlert or None.
    detect_all() runs all 7 and returns a list of valid setups,
    sorted by confidence descending.
    """

    # Minimum VWAP closes for reclaim confirmation
    VWAP_RECLAIM_CONSECUTIVE = 2

    # Breakout volume confirmation multiplier
    ORB_VOLUME_MIN = 1.5

    # Pullback: must be within this % of VWAP or EMA20 to count as pullback
    PULLBACK_PROXIMITY_PCT = 0.003   # 0.3%

    # Allowed regimes per setup
    _ORB_REGIMES = {
        MarketRegime.TREND_UP,
        MarketRegime.TREND_DOWN,
        MarketRegime.BREAKOUT_ATTEMPT,
    }
    _TREND_REGIMES = {
        MarketRegime.TREND_UP,
        MarketRegime.TREND_DOWN,
    }

    def detect_all(
        self,
        df: pd.DataFrame,
        regime: RegimeResult,
        sr: Optional[SRResult],
        chain: Optional[ChainMetrics],
        symbol: str = "",
    ) -> list[SetupAlert]:
        """
        Run all 7 setup detectors and return valid setups,
        sorted by confidence descending.

        Parameters
        ----------
        df     : today's intraday OHLCV bars (1-min or 5-min)
        regime : RegimeResult from RegimeEngine
        sr     : SRResult from SREngine (may be None)
        chain  : ChainMetrics from chain_analytics (may be None)
        """
        if df is None or len(df) < 10:
            return []

        detectors = [
            self.detect_orb,
            self.detect_vwap_reclaim,
            self.detect_pullback_continuation,
            self.detect_exhaustion_fade,
            self.detect_range_mean_reversion,
            self.detect_level_break_retest,
            self.detect_dealer_hedge_zone,
        ]

        alerts: list[SetupAlert] = []
        for detector in detectors:
            try:
                alert = detector(df, regime, sr, chain)
                if alert is not None:
                    alert.detected_at = datetime.utcnow().isoformat()
                    alerts.append(alert)
            except Exception as e:
                # Never let one detector crash the whole pipeline
                import logging
                logging.getLogger("optionsganster").debug(
                    f"Setup detector {detector.__name__} error: {e}"
                )

        alerts.sort(key=lambda a: a.confidence, reverse=True)
        return alerts

    # ── Setup 1: ORB Breakout ────────────────────────────────

    def detect_orb(
        self,
        df: pd.DataFrame,
        regime: RegimeResult,
        sr: Optional[SRResult],
        chain: Optional[ChainMetrics],
    ) -> Optional[SetupAlert]:
        """
        Opening Range Breakout.
        Fires when price closes above/below the first-15-min high/low
        with VWAP confirmation and volume >= 1.5× average.
        Allowed in: TREND_UP, TREND_DOWN, BREAKOUT_ATTEMPT.
        """
        if regime.regime not in self._ORB_REGIMES:
            return None

        orh, orl = opening_range_today(df, minutes=15)
        if orh <= 0 or orl <= 0:
            return None

        current_close = float(df["close"].iloc[-1])
        current_bar   = df.iloc[-1]
        vwap_ser      = vwap(df)
        vwap_now      = float(vwap_ser.iloc[-1])
        vol_ratio_val = volume_ratio(df, period=20)
        atr_val       = atr_value(df, 14)

        if vol_ratio_val < self.ORB_VOLUME_MIN:
            return None

        # Bullish ORB
        if current_close > orh and current_close > vwap_now:
            stop     = max(orh - atr_val * 0.5, orl)
            target_1 = orh + (orh - orl)             # 1× range extension
            target_2 = orh + (orh - orl) * 2         # 2× range extension
            reasons  = [
                f"Closed above ORH {orh:.2f} | range was {orh - orl:.2f} pts",
                f"Above VWAP {vwap_now:.2f} — structural support",
                f"Volume {vol_ratio_val:.1f}× avg — conviction present",
                f"Regime: {regime.regime.value} (trend-aligned)",
            ]
            if regime.regime == MarketRegime.TREND_DOWN:
                # Counter-trend — reduce confidence
                confidence = 0.55
                reasons.append("⚠️ Counter to overall TREND_DOWN — smaller size")
            else:
                confidence = 0.78 + min(0.10, (vol_ratio_val - 1.5) * 0.04)

            return SetupAlert(
                name="ORB Breakout",
                direction="CALL",
                trigger_price=orh,
                entry_condition=f"1m close above ORH {orh:.2f}",
                stop_price=round(stop, 2),
                target_1=round(target_1, 2),
                target_2=round(target_2, 2),
                time_stop_minutes=10,
                regime=regime.regime.value,
                reasons=reasons,
                confidence=round(min(0.95, confidence), 2),
            )

        # Bearish ORB
        if current_close < orl and current_close < vwap_now:
            stop     = min(orl + atr_val * 0.5, orh)
            target_1 = orl - (orh - orl)
            target_2 = orl - (orh - orl) * 2
            reasons  = [
                f"Closed below ORL {orl:.2f} | range was {orh - orl:.2f} pts",
                f"Below VWAP {vwap_now:.2f} — structural resistance",
                f"Volume {vol_ratio_val:.1f}× avg — conviction present",
                f"Regime: {regime.regime.value}",
            ]
            if regime.regime == MarketRegime.TREND_UP:
                confidence = 0.55
                reasons.append("⚠️ Counter to overall TREND_UP — smaller size")
            else:
                confidence = 0.78 + min(0.10, (vol_ratio_val - 1.5) * 0.04)

            return SetupAlert(
                name="ORB Breakdown",
                direction="PUT",
                trigger_price=orl,
                entry_condition=f"1m close below ORL {orl:.2f}",
                stop_price=round(stop, 2),
                target_1=round(target_1, 2),
                target_2=round(target_2, 2),
                time_stop_minutes=10,
                regime=regime.regime.value,
                reasons=reasons,
                confidence=round(min(0.95, confidence), 2),
            )

        return None

    # ── Setup 2: VWAP Reclaim ────────────────────────────────

    def detect_vwap_reclaim(
        self,
        df: pd.DataFrame,
        regime: RegimeResult,
        sr: Optional[SRResult],
        chain: Optional[ChainMetrics],
    ) -> Optional[SetupAlert]:
        """
        VWAP Reclaim: price dips below VWAP then reclaims it with 2
        consecutive closes above VWAP and a positive slope.
        Strong continuation signal when the broader trend is UP.
        """
        if len(df) < 5:
            return None

        vwap_ser = vwap(df)
        closes   = df["close"]
        atr_val  = atr_value(df, 14)

        # Need at least one recent bar that was BELOW VWAP before the reclaim
        recent_closes = closes.iloc[-5:]
        recent_vwap   = vwap_ser.iloc[-5:]

        was_below = any(
            float(c) < float(v)
            for c, v in zip(recent_closes.iloc[:3], recent_vwap.iloc[:3])
        )
        now_above_2 = all(
            float(c) > float(v)
            for c, v in zip(recent_closes.iloc[-2:], recent_vwap.iloc[-2:])
        )

        if not (was_below and now_above_2):
            return None

        vwap_slope_val = slope(vwap_ser, window=10)
        if vwap_slope_val <= 0:
            return None  # VWAP must be sloping up for bullish reclaim

        current_price = float(df["close"].iloc[-1])
        vwap_now      = float(vwap_ser.iloc[-1])

        stop     = vwap_now - atr_val * 0.5
        target_1 = current_price + atr_val * 1.5
        target_2 = current_price + atr_val * 3.0

        confidence = 0.72
        if regime.regime == MarketRegime.TREND_UP:
            confidence += 0.10  # Bonus for trend alignment

        reasons = [
            f"Price dipped below VWAP then reclaimed {vwap_now:.2f}",
            f"2 consecutive closes above VWAP — confirmed hold",
            f"VWAP slope +{vwap_slope_val:.4f} — trend intact",
            f"Regime: {regime.regime.value}",
        ]

        return SetupAlert(
            name="VWAP Reclaim",
            direction="CALL",
            trigger_price=round(vwap_now, 2),
            entry_condition=f"2nd close above VWAP {vwap_now:.2f}",
            stop_price=round(stop, 2),
            target_1=round(target_1, 2),
            target_2=round(target_2, 2),
            time_stop_minutes=8,
            regime=regime.regime.value,
            reasons=reasons,
            confidence=round(min(0.90, confidence), 2),
        )

    # ── Setup 3: Pullback Continuation ───────────────────────

    def detect_pullback_continuation(
        self,
        df: pd.DataFrame,
        regime: RegimeResult,
        sr: Optional[SRResult],
        chain: Optional[ChainMetrics],
    ) -> Optional[SetupAlert]:
        """
        Pullback Continuation: in a trend, price pulls back to VWAP or EMA20,
        then shows a reversal candle + higher low (bullish) or lower high (bearish).
        Only valid in TREND_UP or TREND_DOWN.
        """
        if regime.regime not in self._TREND_REGIMES:
            return None

        if len(df) < 10:
            return None

        current_price = float(df["close"].iloc[-1])
        vwap_ser = vwap(df)
        vwap_now = float(vwap_ser.iloc[-1])
        ema20    = float(ema(df["close"], 20).iloc[-1])
        atr_val  = atr_value(df, 14)

        is_trend_up   = regime.regime == MarketRegime.TREND_UP
        is_trend_down = regime.regime == MarketRegime.TREND_DOWN

        # Check if current price is near VWAP or EMA20 (pulled back)
        vwap_proximity = abs(current_price - vwap_now) / vwap_now < self.PULLBACK_PROXIMITY_PCT
        ema_proximity  = abs(current_price - ema20) / ema20 < self.PULLBACK_PROXIMITY_PCT * 2

        near_support = vwap_proximity or ema_proximity

        if not near_support:
            return None

        current_bar   = df.iloc[-1]
        prev_bar      = df.iloc[-2]

        # Bullish continuation (trend up, pulled back to VWAP, reversal candle)
        if is_trend_up:
            reversal_candle = (
                is_rejection_wick_down(current_bar)
                or (float(current_bar["close"]) > float(current_bar["open"]))  # bullish close
            )
            higher_low = float(current_bar["low"]) > float(prev_bar["low"])

            if reversal_candle and higher_low:
                stop     = current_price - atr_val * 1.0
                target_1 = current_price + atr_val * 2.0
                target_2 = current_price + atr_val * 4.0
                support_level = vwap_now if vwap_proximity else ema20

                reasons = [
                    f"{'VWAP' if vwap_proximity else 'EMA20'} pullback in TREND_UP at {support_level:.2f}",
                    f"Higher low confirmation: {prev_bar['low']:.2f} → {current_bar['low']:.2f}",
                    f"Reversal candle: {'lower wick rejection' if is_rejection_wick_down(current_bar) else 'bullish close'}",
                    f"ATR={atr_val:.3f} — stop {atr_val:.2f} below entry",
                ]

                return SetupAlert(
                    name="Pullback Continuation",
                    direction="CALL",
                    trigger_price=round(current_price, 2),
                    entry_condition=f"Bullish reversal at {'VWAP' if vwap_proximity else 'EMA20'} {support_level:.2f}",
                    stop_price=round(stop, 2),
                    target_1=round(target_1, 2),
                    target_2=round(target_2, 2),
                    time_stop_minutes=12,
                    regime=regime.regime.value,
                    reasons=reasons,
                    confidence=0.74,
                )

        # Bearish continuation (trend down, pulled back to VWAP, rejection candle)
        if is_trend_down:
            reversal_candle = (
                is_rejection_wick_up(current_bar)
                or (float(current_bar["close"]) < float(current_bar["open"]))  # bearish close
            )
            lower_high = float(current_bar["high"]) < float(prev_bar["high"])

            if reversal_candle and lower_high:
                stop     = current_price + atr_val * 1.0
                target_1 = current_price - atr_val * 2.0
                target_2 = current_price - atr_val * 4.0
                resistance_level = vwap_now if vwap_proximity else ema20

                reasons = [
                    f"{'VWAP' if vwap_proximity else 'EMA20'} pullback in TREND_DOWN at {resistance_level:.2f}",
                    f"Lower high confirmation: {prev_bar['high']:.2f} → {current_bar['high']:.2f}",
                    f"Rejection candle: {'upper wick' if is_rejection_wick_up(current_bar) else 'bearish close'}",
                    f"ATR={atr_val:.3f}",
                ]

                return SetupAlert(
                    name="Pullback Continuation",
                    direction="PUT",
                    trigger_price=round(current_price, 2),
                    entry_condition=f"Bearish rejection at {'VWAP' if vwap_proximity else 'EMA20'} {resistance_level:.2f}",
                    stop_price=round(stop, 2),
                    target_1=round(target_1, 2),
                    target_2=round(target_2, 2),
                    time_stop_minutes=12,
                    regime=regime.regime.value,
                    reasons=reasons,
                    confidence=0.74,
                )

        return None

    # ── Setup 4: Exhaustion Fade ─────────────────────────────

    def detect_exhaustion_fade(
        self,
        df: pd.DataFrame,
        regime: RegimeResult,
        sr: Optional[SRResult],
        chain: Optional[ChainMetrics],
    ) -> Optional[SetupAlert]:
        """
        Exhaustion Fade (Advanced):
        Price at VWAP+1σ or VWAP-1σ with rejection wick, RSI divergence
        (price new high but RSI lower), and volume spike.
        Marked as is_advanced=True.
        """
        if len(df) < 20:
            return None

        vwap_ser  = vwap(df)
        upper, lower = vwap_bands(df, num_stdev=1.0)
        vwap_now  = float(vwap_ser.iloc[-1])
        upper_now = float(upper.iloc[-1]) if len(upper) > 0 else vwap_now * 1.01
        lower_now = float(lower.iloc[-1]) if len(lower) > 0 else vwap_now * 0.99

        current_price = float(df["close"].iloc[-1])
        current_bar   = df.iloc[-1]
        atr_val       = atr_value(df, 14)

        rsi_now   = rsi_value(df, 14)
        rsi_prev  = rsi_value(df.iloc[:-3], 14)  # 3 bars ago
        vol_ratio_val = volume_ratio(df, period=20)

        # Bearish exhaustion: price at/above upper band
        at_upper = current_price >= upper_now * 0.999
        if (
            at_upper
            and is_rejection_wick_up(current_bar)
            and rsi_now < rsi_prev        # RSI divergence
            and vol_ratio_val >= 1.8
        ):
            stop     = upper_now + atr_val * 0.3
            target_1 = vwap_now
            target_2 = lower_now

            reasons = [
                f"Price {current_price:.2f} at VWAP+1σ ({upper_now:.2f}) — upper band hit",
                f"Upper wick rejection — buyers exhausted",
                f"RSI divergence: was {rsi_prev:.0f}, now {rsi_now:.0f} — weakening momentum",
                f"Volume spike {vol_ratio_val:.1f}× — potential climax",
                "⚠️ Advanced setup — confirm with 2nd bearish bar",
            ]

            return SetupAlert(
                name="Exhaustion Fade",
                direction="PUT",
                trigger_price=round(upper_now, 2),
                entry_condition=f"Rejection wick at VWAP+1σ {upper_now:.2f}",
                stop_price=round(stop, 2),
                target_1=round(target_1, 2),
                target_2=round(target_2, 2),
                time_stop_minutes=7,
                regime=regime.regime.value,
                reasons=reasons,
                confidence=0.68,
                is_advanced=True,
            )

        # Bullish exhaustion: price at/below lower band
        at_lower = current_price <= lower_now * 1.001
        if (
            at_lower
            and is_rejection_wick_down(current_bar)
            and rsi_now > rsi_prev
            and vol_ratio_val >= 1.8
        ):
            stop     = lower_now - atr_val * 0.3
            target_1 = vwap_now
            target_2 = upper_now

            reasons = [
                f"Price {current_price:.2f} at VWAP-1σ ({lower_now:.2f}) — lower band touch",
                f"Lower wick rejection — sellers exhausted",
                f"RSI divergence: was {rsi_prev:.0f}, now {rsi_now:.0f}",
                f"Volume spike {vol_ratio_val:.1f}×",
                "⚠️ Advanced setup",
            ]

            return SetupAlert(
                name="Exhaustion Fade",
                direction="CALL",
                trigger_price=round(lower_now, 2),
                entry_condition=f"Rejection wick at VWAP-1σ {lower_now:.2f}",
                stop_price=round(stop, 2),
                target_1=round(target_1, 2),
                target_2=round(target_2, 2),
                time_stop_minutes=7,
                regime=regime.regime.value,
                reasons=reasons,
                confidence=0.68,
                is_advanced=True,
            )

        return None

    # ── Setup 5: Range Mean Reversion ────────────────────────

    def detect_range_mean_reversion(
        self,
        df: pd.DataFrame,
        regime: RegimeResult,
        sr: Optional[SRResult],
        chain: Optional[ChainMetrics],
    ) -> Optional[SetupAlert]:
        """
        Range Mean Reversion: only in RANGE_CHOP.
        Price touches VWAP±1σ with contracting volume.
        Fade the touch back toward VWAP.
        """
        if regime.regime != MarketRegime.RANGE_CHOP:
            return None

        if len(df) < 15:
            return None

        vwap_ser  = vwap(df)
        upper, lower = vwap_bands(df, num_stdev=1.0)
        vwap_now  = float(vwap_ser.iloc[-1])
        upper_now = float(upper.iloc[-1]) if len(upper) > 0 else vwap_now * 1.005
        lower_now = float(lower.iloc[-1]) if len(lower) > 0 else vwap_now * 0.995

        current_price = float(df["close"].iloc[-1])
        atr_val       = atr_value(df, 14)
        vol_ratio_val = volume_ratio(df, period=20)

        # Volume should be contracting (range environment)
        if vol_ratio_val > 1.5:
            return None  # Too much volume = not a range setup

        # Fade upper band
        if current_price >= upper_now * 0.998:
            stop     = upper_now + atr_val * 0.5
            target_1 = vwap_now

            reasons = [
                f"RANGE_CHOP: price {current_price:.2f} at upper band {upper_now:.2f}",
                f"Contracting volume ({vol_ratio_val:.1f}× — no breakout energy)",
                f"Fade target: VWAP {vwap_now:.2f}",
                f"ATR={atr_val:.3f}",
            ]

            return SetupAlert(
                name="Range Mean Reversion",
                direction="PUT",
                trigger_price=round(upper_now, 2),
                entry_condition=f"Touch of VWAP+1σ {upper_now:.2f} with low volume",
                stop_price=round(stop, 2),
                target_1=round(target_1, 2),
                target_2=None,
                time_stop_minutes=6,
                regime=regime.regime.value,
                reasons=reasons,
                confidence=0.65,
            )

        # Fade lower band
        if current_price <= lower_now * 1.002:
            stop     = lower_now - atr_val * 0.5
            target_1 = vwap_now

            reasons = [
                f"RANGE_CHOP: price {current_price:.2f} at lower band {lower_now:.2f}",
                f"Contracting volume ({vol_ratio_val:.1f}×)",
                f"Bounce target: VWAP {vwap_now:.2f}",
                f"ATR={atr_val:.3f}",
            ]

            return SetupAlert(
                name="Range Mean Reversion",
                direction="CALL",
                trigger_price=round(lower_now, 2),
                entry_condition=f"Touch of VWAP-1σ {lower_now:.2f} with low volume",
                stop_price=round(stop, 2),
                target_1=round(target_1, 2),
                target_2=None,
                time_stop_minutes=6,
                regime=regime.regime.value,
                reasons=reasons,
                confidence=0.65,
            )

        return None

    # ── Setup 6: Level Break Retest ──────────────────────────

    def detect_level_break_retest(
        self,
        df: pd.DataFrame,
        regime: RegimeResult,
        sr: Optional[SRResult],
        chain: Optional[ChainMetrics],
    ) -> Optional[SetupAlert]:
        """
        Level Break + Retest:
        Price breaks a key S/R level, then pulls back to retest
        and holds for 2-3 bars (former resistance = new support).
        """
        if sr is None or not sr.levels:
            return None

        if len(df) < 10:
            return None

        current_price   = float(df["close"].iloc[-1])
        prev_close      = float(df["close"].iloc[-2])
        atr_val         = atr_value(df, 14)
        vwap_ser        = vwap(df)
        vwap_now        = float(vwap_ser.iloc[-1])

        # Consider nearest strong S/R levels
        strong_levels = [
            l for l in sr.levels
            if l.strength >= 0.4 and abs(l.price - current_price) < atr_val * 3
        ]

        for level in strong_levels:
            lp = level.price
            # Bullish flip: was resistance, now broken & held
            if (
                level.kind == "resistance"
                and prev_close < lp           # Was below recently
                and current_price > lp        # Now above
                and current_price < lp + atr_val  # Still close to level
            ):
                stop     = lp - atr_val * 0.5
                target_1 = lp + atr_val * 2.0
                target_2 = lp + atr_val * 4.0
                vol_r    = volume_ratio(df, period=20)

                reasons = [
                    f"Broke resistance {lp:.2f} ({level.label}) — now support",
                    f"Retesting broken level from above",
                    f"VWAP {vwap_now:.2f} {'above' if vwap_now < current_price else 'below'} price",
                    f"Volume {vol_r:.1f}× avg on break",
                ]

                return SetupAlert(
                    name="Level Break Retest",
                    direction="CALL",
                    trigger_price=round(lp, 2),
                    entry_condition=f"Hold above broken resistance {lp:.2f}",
                    stop_price=round(stop, 2),
                    target_1=round(target_1, 2),
                    target_2=round(target_2, 2),
                    time_stop_minutes=8,
                    regime=regime.regime.value,
                    reasons=reasons,
                    confidence=0.71,
                )

            # Bearish flip: was support, now broken & held below
            if (
                level.kind == "support"
                and prev_close > lp
                and current_price < lp
                and current_price > lp - atr_val
            ):
                stop     = lp + atr_val * 0.5
                target_1 = lp - atr_val * 2.0
                target_2 = lp - atr_val * 4.0
                vol_r    = volume_ratio(df, period=20)

                reasons = [
                    f"Broke support {lp:.2f} ({level.label}) — now resistance",
                    f"Retesting broken level from below",
                    f"VWAP {vwap_now:.2f} reference",
                    f"Volume {vol_r:.1f}× avg on break",
                ]

                return SetupAlert(
                    name="Level Break Retest",
                    direction="PUT",
                    trigger_price=round(lp, 2),
                    entry_condition=f"Hold below broken support {lp:.2f}",
                    stop_price=round(stop, 2),
                    target_1=round(target_1, 2),
                    target_2=round(target_2, 2),
                    time_stop_minutes=8,
                    regime=regime.regime.value,
                    reasons=reasons,
                    confidence=0.71,
                )

        return None

    # ── Setup 7: Dealer Hedge Zone ───────────────────────────

    def detect_dealer_hedge_zone(
        self,
        df: pd.DataFrame,
        regime: RegimeResult,
        sr: Optional[SRResult],
        chain: Optional[ChainMetrics],
    ) -> Optional[SetupAlert]:
        """
        Dealer Hedge Zone:
        When price breaches the highest-OI call or put strike:
          - Break of call wall → dealers forced to buy → continuation CALL
          - Pin at call wall → price likely to chop (no alert — just chop warning)
          - Break of put wall below → continuation PUT
        """
        if chain is None:
            return None

        if chain.call_wall <= 0 and chain.put_wall <= 0:
            return None

        current_price = float(df["close"].iloc[-1])
        atr_val       = atr_value(df, 14)
        vwap_ser      = vwap(df)
        vwap_now      = float(vwap_ser.iloc[-1])
        vol_ratio_val = volume_ratio(df, period=20)

        # Break above call wall with volume
        if chain.call_wall > 0:
            dist_above = current_price - chain.call_wall
            if 0 < dist_above < atr_val and vol_ratio_val >= 1.5:
                stop     = chain.call_wall - atr_val * 0.3
                target_1 = chain.call_wall + atr_val * 2
                target_2 = chain.call_wall + atr_val * 4

                reasons = [
                    f"Breached call wall at {chain.call_wall:.2f} — dealer short gamma",
                    f"Dealers forced to delta-hedge by buying → momentum squeezes higher",
                    f"GEX regime: {chain.gex_regime}",
                    f"Volume {vol_ratio_val:.1f}× — accumulation above wall",
                ]

                return SetupAlert(
                    name="Dealer Hedge Zone",
                    direction="CALL",
                    trigger_price=round(chain.call_wall, 2),
                    entry_condition=f"Hold above call wall {chain.call_wall:.2f}",
                    stop_price=round(stop, 2),
                    target_1=round(target_1, 2),
                    target_2=round(target_2, 2),
                    time_stop_minutes=8,
                    regime=regime.regime.value,
                    reasons=reasons,
                    confidence=0.70,
                )

        # Break below put wall with volume
        if chain.put_wall > 0:
            dist_below = chain.put_wall - current_price
            if 0 < dist_below < atr_val and vol_ratio_val >= 1.5:
                stop     = chain.put_wall + atr_val * 0.3
                target_1 = chain.put_wall - atr_val * 2
                target_2 = chain.put_wall - atr_val * 4

                reasons = [
                    f"Breached put wall at {chain.put_wall:.2f} — dealers short gamma",
                    f"Forced delta-hedge selling → accelerates downside",
                    f"GEX regime: {chain.gex_regime}",
                    f"Volume {vol_ratio_val:.1f}×",
                ]

                return SetupAlert(
                    name="Dealer Hedge Zone",
                    direction="PUT",
                    trigger_price=round(chain.put_wall, 2),
                    entry_condition=f"Hold below put wall {chain.put_wall:.2f}",
                    stop_price=round(stop, 2),
                    target_1=round(target_1, 2),
                    target_2=round(target_2, 2),
                    time_stop_minutes=8,
                    regime=regime.regime.value,
                    reasons=reasons,
                    confidence=0.70,
                )

        return None


# ── Module-level singleton ───────────────────────────────────
setup_engine = SetupEngine()
