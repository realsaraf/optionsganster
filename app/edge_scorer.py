"""
Edge Scorer
============
Converts a raw SetupAlert + OptionPick into a normalized 0-100 Edge Score
and assigns a tier label.

Score Components
----------------
| Component      | Max | Logic                                          |
|----------------|-----|------------------------------------------------|
| Structure      |  35 | Level proximity, room to target, VWAP align    |
| Regime Fit     |  20 | Is setup allowed in this regime? Confidence?   |
| Momentum/Vol   |  15 | Volume ratio, breakout candle confirmation     |
| Options Quality|  20 | Liquidity, delta, IV vs RV                     |
| Risk           |  10 | R:R ratio, time of day, DTE quality            |

Tiers
-----
A+  80-100 — Premium setup, full size
A   65-79  — Strong setup, normal size
B   50-64  — Acceptable setup, reduced size
---
< 50 → No alert emitted
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, time
from typing import Optional
from zoneinfo import ZoneInfo

import pandas as pd

from app.indicators import atr_value, volume_ratio, vwap, vwap_band_values
from app.regime_engine import MarketRegime, RegimeResult
from app.setup_engine import SetupAlert
from app.option_picker import OptionPick

_ET = ZoneInfo("America/New_York")


# ── Output ───────────────────────────────────────────────────

@dataclass
class EdgeResult:
    """Scored setup ready for the alert manager."""
    edge_score: int             # 0-100
    tier: str                   # "A+", "A", "B", or "NO_EDGE"
    setup: SetupAlert
    pick: Optional[OptionPick]

    # Component breakdown (for UI)
    structure_score: int        # /35
    regime_score: int           # /20
    momentum_score: int         # /15
    options_score: int          # /20
    risk_score: int             # /10

    detail: str = ""            # Summary of scoring


# ── Scorer ───────────────────────────────────────────────────

class EdgeScorer:
    """
    Compute the Edge Score for a setup + option pick.
    """

    MIN_SCORE_FOR_ALERT = 50

    # Regime fit: some setups are better in certain regimes
    _REGIME_FIT: dict[str, set] = {
        "ORB Breakout":         {MarketRegime.TREND_UP, MarketRegime.TREND_DOWN, MarketRegime.BREAKOUT_ATTEMPT},
        "ORB Breakdown":        {MarketRegime.TREND_UP, MarketRegime.TREND_DOWN, MarketRegime.BREAKOUT_ATTEMPT},
        "VWAP Reclaim":         {MarketRegime.TREND_UP, MarketRegime.BREAKOUT_ATTEMPT},
        "Pullback Continuation":{MarketRegime.TREND_UP, MarketRegime.TREND_DOWN},
        "Exhaustion Fade":      {MarketRegime.REVERSAL_EXHAUSTION, MarketRegime.RANGE_CHOP},
        "Range Mean Reversion": {MarketRegime.RANGE_CHOP},
        "Level Break Retest":   {MarketRegime.TREND_UP, MarketRegime.TREND_DOWN, MarketRegime.BREAKOUT_ATTEMPT},
        "Dealer Hedge Zone":    {MarketRegime.TREND_UP, MarketRegime.TREND_DOWN, MarketRegime.BREAKOUT_ATTEMPT},
    }

    def score(
        self,
        setup: SetupAlert,
        regime: RegimeResult,
        option: Optional[OptionPick],
        df: pd.DataFrame,
        rv: float = 0.0,           # Realized Vol (annualized)
    ) -> EdgeResult:
        """
        Compute Edge Score.

        Parameters
        ----------
        setup  : SetupAlert to score
        regime : current RegimeResult
        option : OptionPick or None (if None, options_score capped at 0)
        df     : intraday OHLCV bars (for volume + VWAP)
        rv     : realized volatility annualized (from rolling_rv)
        """
        s = 0.0  # running raw score (0-100 scale built from components)

        # ── 1. Structure (max 35) ────────────────────────
        structure = self._score_structure(setup, df)          # 0-1
        structure_pts = round(structure * 35)

        # ── 2. Regime Fit (max 20) ───────────────────────
        regime_fit = self._score_regime_fit(setup, regime)    # 0-1
        regime_pts = round(regime_fit * 20)

        # ── 3. Momentum / Volume (max 15) ────────────────
        momentum = self._score_momentum(df)                   # 0-1
        momentum_pts = round(momentum * 15)

        # ── 4. Options Quality (max 20) ──────────────────
        options_q = self._score_options_quality(option, rv)   # 0-1
        options_pts = round(options_q * 20)

        # ── 5. Risk / R:R (max 10) ───────────────────────
        risk = self._score_risk(setup, option)                # 0-1
        risk_pts = round(risk * 10)

        total = structure_pts + regime_pts + momentum_pts + options_pts + risk_pts

        # Tier assignment
        if total >= 80:
            tier = "A+"
        elif total >= 65:
            tier = "A"
        elif total >= self.MIN_SCORE_FOR_ALERT:
            tier = "B"
        else:
            tier = "NO_EDGE"

        detail_parts = [
            f"Structure={structure_pts}/35",
            f"Regime={regime_pts}/20",
            f"Momentum={momentum_pts}/15",
            f"Options={options_pts}/20",
            f"Risk={risk_pts}/10",
        ]

        return EdgeResult(
            edge_score=total,
            tier=tier,
            setup=setup,
            pick=option,
            structure_score=structure_pts,
            regime_score=regime_pts,
            momentum_score=momentum_pts,
            options_score=options_pts,
            risk_score=risk_pts,
            detail=" | ".join(detail_parts),
        )

    # ── Component scorers ────────────────────────────────────

    def _score_structure(self, setup: SetupAlert, df: pd.DataFrame) -> float:
        """
        Score based on:
        - Distance of trigger_price to nearby S/R (are we at a real level?)
        - Room to target relative to ATR (is there space?)
        - VWAP alignment
        """
        score = 0.0
        atr_val = atr_value(df, 14)
        if atr_val <= 0:
            atr_val = 0.01

        # Target room factor: how many ATRs to target_1?
        if setup.target_1 and setup.trigger_price:
            room = abs(setup.target_1 - setup.trigger_price)
            room_atrs = room / atr_val
            if room_atrs >= 2.0:
                score += 0.50
            elif room_atrs >= 1.0:
                score += 0.30
            else:
                score += 0.10

        # Stop distance (closer stop = better R:R → better structure)
        if setup.stop_price and setup.trigger_price:
            stop_dist = abs(setup.trigger_price - setup.stop_price)
            stop_atrs = stop_dist / atr_val
            if stop_atrs <= 0.5:
                score += 0.30
            elif stop_atrs <= 1.0:
                score += 0.20
            else:
                score += 0.10

        # VWAP alignment
        if not df.empty:
            vwap_ser = vwap(df)
            vwap_val = float(vwap_ser.iloc[-1])
            current  = float(df["close"].iloc[-1])
            if vwap_val > 0:
                if setup.direction == "CALL" and current > vwap_val:
                    score += 0.20
                elif setup.direction == "PUT" and current < vwap_val:
                    score += 0.20
                else:
                    score += 0.05

        return min(1.0, score)

    def _score_regime_fit(
        self, setup: SetupAlert, regime: RegimeResult
    ) -> float:
        """Score 0-1: how well does the setup match the current regime."""
        allowed = self._REGIME_FIT.get(setup.name, set())
        if regime.regime in allowed:
            base = 0.70 + regime.confidence * 0.30
        else:
            # Setup fires in a suboptimal regime — penalise
            base = 0.30 + regime.confidence * 0.10

        return round(min(1.0, base), 3)

    def _score_momentum(self, df: pd.DataFrame) -> float:
        """Volume and candle momentum confirmation."""
        if df.empty:
            return 0.40

        vol_r = volume_ratio(df, period=20)
        if vol_r >= 2.5:
            return 0.95
        elif vol_r >= 1.8:
            return 0.80
        elif vol_r >= 1.3:
            return 0.60
        elif vol_r >= 1.0:
            return 0.40
        else:
            return 0.20  # Low volume = weak momentum

    def _score_options_quality(
        self, option: Optional[OptionPick], rv: float
    ) -> float:
        """Score option contract quality: liquidity, delta, IV vs RV."""
        if option is None:
            return 0.0

        score = 0.0

        # Liquidity
        score += option.liquidity_score * 0.40

        # Delta zone
        da = abs(option.delta)
        if 0.40 <= da <= 0.60:
            score += 0.35
        elif 0.35 <= da < 0.40 or 0.60 < da <= 0.70:
            score += 0.25
        elif da > 0.70:
            score += 0.15  # Deep ITM — high prob but low leverage
        elif da >= 0.25:
            score += 0.10
        else:
            score += 0.00  # Far OTM — no credit

        # IV vs RV (cheap vs expensive)
        if option.iv > 0 and rv > 0:
            iv_rv_ratio = option.iv / rv
            if iv_rv_ratio <= 1.1:
                score += 0.25   # Cheap IV — buy cheaply
            elif iv_rv_ratio <= 1.5:
                score += 0.15
            else:
                score += 0.05   # Expensive IV

        return min(1.0, score)

    def _score_risk(
        self, setup: SetupAlert, option: Optional[OptionPick]
    ) -> float:
        """R:R ratio and time-of-day risk."""
        score  = 0.0

        # R:R ratio
        if setup.stop_price and setup.trigger_price and setup.target_1:
            reward = abs(setup.target_1 - setup.trigger_price)
            risk   = abs(setup.trigger_price - setup.stop_price)
            if risk > 0:
                rr = reward / risk
                if rr >= 3.0:
                    score += 0.60
                elif rr >= 2.0:
                    score += 0.45
                elif rr >= 1.5:
                    score += 0.30
                else:
                    score += 0.10

        # Time of day (avoid first 30 min and last 30 min)
        now_t = datetime.now(_ET).time()
        pre_market  = now_t < time(9, 30)
        early       = time(9, 30) <= now_t < time(10, 0)
        late        = now_t >= time(15, 30)
        if pre_market or late:
            score = score * 0.5   # Cut score in half outside sweet spot
        elif early:
            score = score * 0.7

        # DTE risk
        if option is not None:
            if option.dte == 0:
                score *= 0.80    # 0DTE has extra risk
            elif option.dte == 1:
                score *= 0.90

        return min(1.0, max(0.0, score))


# ── Module-level singleton ───────────────────────────────────
edge_scorer = EdgeScorer()
