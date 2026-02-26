"""
Option Picker
=============
Delta-based strike selection with liquidity gates and time-of-day filters.

Given a directional setup (CALL or PUT) and the current options chain,
selects the best-fit contract based on:
  - Target delta range (0.35-0.55 ideal)
  - Spread% gate (≤10%)
  - Minimum liquidity (volume ≥ 200 OR OI ≥ 500)
  - Time-of-day filter (after 14:30 prefer ATM/ITM)
  - DTE filter (avoid 0DTE OTM)

Returns OptionPick dataclass or None if no contract passes all gates.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime, time
from typing import Optional
from zoneinfo import ZoneInfo

from app.chain_analytics import compute_liquidity_score, compute_spread_pct

_ET = ZoneInfo("America/New_York")


# ── Data class ───────────────────────────────────────────────

@dataclass
class OptionPick:
    """Selected option contract for a setup."""
    ticker: str             # OCC format e.g. "O:QQQ260225C00610000"
    strike: float
    expiration: date
    option_type: str        # "call" | "put"
    delta: float
    spread_pct: float
    liquidity_score: float
    iv: float
    entry_premium_est: float   # midpoint price
    option_stop_pct: float     # e.g. -0.35 = exit if option drops 35%
    dte: int
    notes: str


# ── Picker ───────────────────────────────────────────────────

class OptionPicker:
    """
    Selects the optimal option contract from the chain snapshot
    given a directional setup signal.
    """

    # Delta target ranges
    DELTA_IDEAL_LO   = 0.35
    DELTA_IDEAL_HI   = 0.55
    DELTA_ATM_LO     = 0.45  # After 14:30 prefer closer to ATM
    DELTA_LATE_MIN   = 0.40  # After 14:30 minimum delta (no OTM)

    # Hard gates
    MAX_SPREAD_PCT   = 10.0
    MIN_VOLUME       = 200
    MIN_OI           = 500

    # Edge score gate (imposed by caller)
    MIN_EDGE_FOR_OTM = 65   # Edge score needed to allow delta < 0.35

    # Time gates (Eastern)
    BROAD_OPEN_TIME  = time(9, 30)
    MID_SESSION_TIME = time(10, 30)
    LATE_SESSION_TIME = time(14, 30)

    def pick(
        self,
        chain: list[dict],
        direction: str,
        spot: float,
        edge_score: float = 0.0,
        expiration_date: Optional[date] = None,
    ) -> Optional[OptionPick]:
        """
        Select best option contract.

        Parameters
        ----------
        chain      : raw chain snapshot list from Polygon
        direction  : "CALL" or "PUT"
        spot       : current underlying price
        edge_score : 0-100 from EdgeScorer (gates OTM selection)
        expiration_date : preferred expiration; None = nearest available

        Returns
        -------
        OptionPick or None if no contract passes gates
        """
        now_et      = datetime.now(_ET)
        now_time    = now_et.time()
        today       = now_et.date()

        # Filter direction
        ct_filter   = direction.lower()
        candidates  = [
            c for c in chain
            if (c.get("contract_type") or "").lower() == ct_filter
        ]

        if not candidates:
            return None

        # Filter to preferred expiration if given
        if expiration_date:
            candidates = [
                c for c in candidates
                if c.get("expiration_date") == expiration_date.isoformat()
            ]
            if not candidates:
                return None

        # Compute DTE per contract
        scored: list[tuple[float, dict]] = []
        for c in candidates:
            exp_str = c.get("expiration_date", "")
            try:
                exp_dt  = date.fromisoformat(exp_str)
                dte_val = (exp_dt - today).days
            except (ValueError, TypeError):
                continue

            if dte_val < 0:
                continue

            delta_raw = (c.get("greeks") or {}).get("delta", 0) or 0
            delta_abs = abs(delta_raw)

            # Time-of-day gate
            if now_time >= self.LATE_SESSION_TIME:
                # After 14:30: require near-ATM (delta≥0.40)
                if delta_abs < self.DELTA_LATE_MIN:
                    continue
                # 0DTE OTM hard block
                if dte_val == 0 and delta_abs < 0.45:
                    continue

            # Broad early session (first 30 min): require confirmed edge
            if now_time < self.MID_SESSION_TIME and edge_score < self.MIN_EDGE_FOR_OTM:
                if delta_abs < self.DELTA_IDEAL_LO:
                    continue

            # Liquidity gate
            liq = compute_liquidity_score(c)
            if not liq.passes_gate:
                continue

            # Score: distance from ideal delta range
            if self.DELTA_IDEAL_LO <= delta_abs <= self.DELTA_IDEAL_HI:
                delta_score = 1.0
            elif delta_abs > self.DELTA_IDEAL_HI:
                delta_score = 0.9  # Slightly ITM — still good
            elif delta_abs >= 0.25:
                delta_score = 0.7  # Slightly OTM — acceptable
            else:
                delta_score = 0.4  # Far OTM — penalised

            # Prefer near-term expiration (lower DTE = cheaper, faster movement)
            dte_score = max(0.2, 1.0 - dte_val / 30.0)  # 30 DTE → 0.2

            # Combined score
            composite_score = (
                0.50 * delta_score
                + 0.30 * liq.score
                + 0.20 * dte_score
            )

            scored.append((composite_score, c, dte_val, liq))

        if not scored:
            return None

        scored.sort(key=lambda x: x[0], reverse=True)
        best_score, best, dte_val, liq = scored[0]

        # Extract fields
        delta_raw  = (best.get("greeks") or {}).get("delta", 0) or 0
        iv_val     = float(best.get("iv", 0) or 0)
        bid        = float(best.get("bid", 0) or 0)
        ask        = float(best.get("ask", 0) or 0)
        last_price = float(best.get("last_price", 0) or 0)
        mid        = (bid + ask) / 2 if bid > 0 and ask > 0 else last_price

        # Dynamic stop: deeper OTM = tighter stop
        delta_abs = abs(delta_raw)
        if delta_abs >= 0.45:
            option_stop_pct = -0.30   # ATM/ITM: allow 30% drawdown
        elif delta_abs >= 0.35:
            option_stop_pct = -0.35
        else:
            option_stop_pct = -0.40   # OTM: tighter

        # Build ticker (may already be present)
        ticker = best.get("ticker", "")

        exp_str = best.get("expiration_date", "")
        try:
            exp_dt = date.fromisoformat(exp_str)
        except (ValueError, TypeError):
            exp_dt = date.today()

        # Notes
        notes_parts = []
        if liq.spread_pct > 5:
            notes_parts.append(f"Wide spread {liq.spread_pct:.1f}%")
        if dte_val == 0:
            notes_parts.append("0DTE — monitor closely")
        if delta_abs < self.DELTA_IDEAL_LO:
            notes_parts.append(f"OTM (delta={delta_abs:.2f}) — requires edge≥{self.MIN_EDGE_FOR_OTM}")

        return OptionPick(
            ticker=ticker,
            strike=float(best.get("strike", 0) or 0),
            expiration=exp_dt,
            option_type=ct_filter,
            delta=round(float(delta_raw), 3),
            spread_pct=round(liq.spread_pct, 2),
            liquidity_score=round(liq.score, 3),
            iv=round(iv_val, 4),
            entry_premium_est=round(mid, 2),
            option_stop_pct=option_stop_pct,
            dte=dte_val,
            notes=" | ".join(notes_parts) if notes_parts else "OK",
        )


# ── Module-level singleton ───────────────────────────────────
option_picker = OptionPicker()
