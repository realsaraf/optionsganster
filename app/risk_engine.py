"""
Risk Engine
===========
Builds a complete TradePlan from a scored setup + option pick.

The TradePlan includes:
  - Underlying entry / stop / target levels
  - Option entry / stop (as % of premium)
  - Reward:Risk ratio
  - Time stop (exit after N minutes if trade hasn't moved)
  - Kill-switch conditions evaluated per tick

Capital Mode — session-level discipline layer:
  NORMAL   — full position sizing
  REDUCED  — 0.5× after 2 consecutive losses, RANGE regime, or LOW volume
  LOCKED   — 0× for 30 min after 3 consecutive losses

This module is pure computation — no API calls.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Optional

from app.indicators import atr_value
from app.setup_engine import SetupAlert
from app.option_picker import OptionPick
from app.edge_scorer import EdgeResult

import pandas as pd


# ── Data classes ─────────────────────────────────────────────

@dataclass
class TradePlan:
    """Complete trade plan for a setup."""
    # Underlying levels (for structure reference)
    entry_price: float
    stop_price: float
    target_1: float
    target_2: Optional[float]

    # Option-specific
    option_entry_est: float       # Estimated fill price (midpoint)
    option_stop_pct: float        # Exit if option drops this % (e.g. -0.35)
    option_stop_dollar: float     # Dollar value of stop (option_entry × (1 + stop_pct))

    # Timing
    time_stop_minutes: int        # Exit if not working within N minutes
    expires_at: Optional[str]     # ISO timestamp of time stop

    # Metrics
    reward_risk_ratio: float      # Underlying-level R:R
    edge_score: int
    tier: str

    # Kill-switch conditions (human-readable, evaluated each tick)
    kill_switch_conditions: list[str] = field(default_factory=list)

    # Context
    setup_name: str = ""
    direction: str = ""           # "CALL" | "PUT"
    regime: str = ""
    reasons: list[str] = field(default_factory=list)


# ── Engine ───────────────────────────────────────────────────

class RiskEngine:
    """
    Build a TradePlan from an EdgeResult.
    Also tracks session-level capital mode (loss streaks, lock-outs).
    """

    # Capital mode thresholds
    CONSECUTIVE_LOSSES_REDUCED = 2   # → REDUCED after 2 in a row
    CONSECUTIVE_LOSSES_LOCKED  = 3   # → LOCKED after 3 in a row
    LOCK_MINUTES               = 30

    def __init__(self):
        self._losses_today: int = 0
        self._wins_today: int = 0
        self._consecutive_losses: int = 0
        self._win_streak: int = 0
        self._locked_until: Optional[datetime] = None
        self._daily_reset_date: Optional[date] = None

    # ── Capital Mode ──────────────────────────────────────────

    def record_outcome(self, is_win: bool) -> None:
        """Record a trade outcome; updates loss/win streaks and lock timer."""
        self._ensure_daily_reset()
        if is_win:
            self._wins_today += 1
            self._win_streak += 1
            self._consecutive_losses = 0
        else:
            self._losses_today += 1
            self._win_streak = 0
            self._consecutive_losses += 1
            if self._consecutive_losses >= self.CONSECUTIVE_LOSSES_LOCKED:
                self._locked_until = datetime.utcnow() + timedelta(minutes=self.LOCK_MINUTES)

    def get_capital_mode(
        self,
        regime_name: str = "",
        vol_regime: str = "",
    ) -> dict:
        """
        Compute current capital mode.

        Returns
        -------
        dict with keys: mode, reason, max_size_mult, locked_until
        """
        self._ensure_daily_reset()

        # LOCKED: 3 consecutive losses → no new trades for 30 min
        if self._locked_until and datetime.utcnow() < self._locked_until:
            return {
                "mode": "LOCKED",
                "reason": f"{self._consecutive_losses} consecutive losses — locked for {self.LOCK_MINUTES} min",
                "max_size_mult": 0.0,
                "locked_until": self._locked_until.isoformat(),
            }

        reasons = []

        # REDUCED: 2 consecutive losses
        if self._consecutive_losses >= self.CONSECUTIVE_LOSSES_REDUCED:
            reasons.append(f"{self._consecutive_losses} consecutive losses")

        # REDUCED: RANGE regime
        r = regime_name.upper() if regime_name else ""
        if "RANGE" in r or "CHOP" in r:
            reasons.append("Range/chop regime")

        # REDUCED: LOW volume
        if vol_regime and vol_regime.upper() == "LOW":
            reasons.append("Low volume regime")

        if reasons:
            return {
                "mode": "REDUCED",
                "reason": " · ".join(reasons),
                "max_size_mult": 0.5,
                "locked_until": None,
            }

        return {
            "mode": "NORMAL",
            "reason": "",
            "max_size_mult": 1.0,
            "locked_until": None,
        }

    def get_session_stats(self) -> dict:
        """Return today's session stats for the frontend."""
        self._ensure_daily_reset()
        return {
            "wins_today": self._wins_today,
            "losses_today": self._losses_today,
            "consecutive_losses": self._consecutive_losses,
            "win_streak": self._win_streak,
            "locked_until": self._locked_until.isoformat() if self._locked_until and datetime.utcnow() < self._locked_until else None,
        }

    def _ensure_daily_reset(self):
        """Auto-reset counters when a new trading day starts."""
        today = date.today()
        if self._daily_reset_date != today:
            self._losses_today = 0
            self._wins_today = 0
            self._consecutive_losses = 0
            self._win_streak = 0
            self._locked_until = None
            self._daily_reset_date = today

    def build_plan(
        self,
        edge_result: EdgeResult,
        df: Optional[pd.DataFrame] = None,
    ) -> Optional[TradePlan]:
        """
        Build a TradePlan.

        Parameters
        ----------
        edge_result : EdgeResult from EdgeScorer
        df          : intraday bars for ATR reference (optional)

        Returns
        -------
        TradePlan or None if edge_score < minimum
        """
        if edge_result.edge_score < 50:
            return None

        setup  = edge_result.setup
        option = edge_result.pick

        # Levels from setup
        entry  = setup.trigger_price
        stop   = setup.stop_price
        tgt1   = setup.target_1
        tgt2   = setup.target_2

        if entry <= 0 or stop <= 0:
            return None

        # R:R
        reward = abs(tgt1 - entry) if tgt1 else 0.0
        risk   = abs(entry - stop)
        rr = round(reward / risk, 2) if risk > 0 else 0.0

        # Option stop
        opt_entry = option.entry_premium_est if option else 0.0
        opt_stop_pct = option.option_stop_pct if option else -0.35
        opt_stop_dollar = round(
            opt_entry * (1 + opt_stop_pct), 2
        ) if opt_entry > 0 else 0.0

        # Time stop timestamp
        now = datetime.utcnow()
        expires_at = (
            now + timedelta(minutes=setup.time_stop_minutes)
        ).isoformat()

        # Kill-switch conditions
        kill_switches = _build_kill_switches(setup, option, entry, stop)

        return TradePlan(
            entry_price=round(entry, 2),
            stop_price=round(stop, 2),
            target_1=round(tgt1, 2) if tgt1 else 0.0,
            target_2=round(tgt2, 2) if tgt2 else None,
            option_entry_est=round(opt_entry, 2),
            option_stop_pct=opt_stop_pct,
            option_stop_dollar=opt_stop_dollar,
            time_stop_minutes=setup.time_stop_minutes,
            expires_at=expires_at,
            reward_risk_ratio=rr,
            edge_score=edge_result.edge_score,
            tier=edge_result.tier,
            kill_switch_conditions=kill_switches,
            setup_name=setup.name,
            direction=setup.direction,
            regime=setup.regime,
            reasons=setup.reasons,
        )


def _build_kill_switches(
    setup: SetupAlert,
    option: Optional[OptionPick],
    entry: float,
    stop: float,
) -> list[str]:
    """Build a list of human-readable kill-switch conditions."""
    conditions = []

    # Price-based
    if setup.direction == "CALL":
        conditions.append(
            f"Underlying closes back below {stop:.2f} (setup invalidated)"
        )
        conditions.append(
            f"Underlying closes below VWAP for 2 consecutive bars"
        )
    else:
        conditions.append(
            f"Underlying closes back above {stop:.2f} (setup invalidated)"
        )
        conditions.append(
            f"Underlying closes above VWAP for 2 consecutive bars"
        )

    # Option premium stop
    if option and option.entry_premium_est > 0:
        stop_prem = round(option.entry_premium_est * (1 + option.option_stop_pct), 2)
        conditions.append(
            f"Option premium drops to ${stop_prem:.2f} "
            f"({abs(int(option.option_stop_pct * 100))}% stop)"
        )

    # Time stop
    conditions.append(
        f"Time stop: exit after {setup.time_stop_minutes} minutes regardless of P&L"
    )

    # Regime shift
    conditions.append(
        "Market regime flips to RANGE_CHOP or REVERSAL_EXHAUSTION — exit immediately"
    )

    return conditions


# ── Module-level singleton ───────────────────────────────────
risk_engine = RiskEngine()
