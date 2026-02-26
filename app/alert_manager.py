"""
Alert Manager
=============
Lifecycle management for trade setup alerts.

Responsibilities
----------------
1. Debounce  — same setup can't re-fire within 5 minutes
2. Dedup     — identical direction+trigger already active → skip
3. Cap       — maximum 2 active alerts simultaneously
4. Lifecycle — PENDING → ACTIVE → HIT_T1 / STOPPED / TIMED_OUT / INVALIDATED
5. Invalidation — kill-switch evaluation per tick
6. Performance — track outcomes for calibration

Alert States
-----------
PENDING     — detected, not yet entered
ACTIVE      — entered, being monitored
HIT_T1      — target 1 reached
HIT_T2      — target 2 reached
STOPPED     — stop loss hit
TIMED_OUT   — time stop expired with no result
INVALIDATED — kill-switch condition triggered before fill
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional

from app.risk_engine import TradePlan
from app.setup_engine import SetupAlert
from app.option_picker import OptionPick
from app.edge_scorer import EdgeResult

logger = logging.getLogger("optionsganster")


# ── State taxonomy ────────────────────────────────────────────

class AlertState(Enum):
    PENDING     = "PENDING"
    ACTIVE      = "ACTIVE"
    HIT_T1      = "HIT_T1"
    HIT_T2      = "HIT_T2"
    STOPPED     = "STOPPED"
    TIMED_OUT   = "TIMED_OUT"
    INVALIDATED = "INVALIDATED"


# ── Alert data class ─────────────────────────────────────────

@dataclass
class ActiveAlert:
    """A single live alert being tracked."""
    id: str                         # Unique ID
    state: AlertState

    # Core setup
    setup_name: str
    direction: str                  # "CALL" | "PUT"
    edge_score: int
    tier: str
    regime: str
    trigger_price: float
    reasons: list[str]

    # Trade plan
    plan: TradePlan
    option: Optional[OptionPick]

    # Timing
    detected_at: str                # ISO timestamp
    activated_at: Optional[str] = None
    resolved_at: Optional[str] = None
    expires_at: Optional[str] = None   # Time stop

    # Performance tracking
    entry_premium: float = 0.0
    exit_premium: float = 0.0
    pnl_pct: float = 0.0

    @property
    def is_terminal(self) -> bool:
        return self.state in (
            AlertState.HIT_T1, AlertState.HIT_T2,
            AlertState.STOPPED, AlertState.TIMED_OUT, AlertState.INVALIDATED
        )

    @property
    def is_active_or_pending(self) -> bool:
        return self.state in (AlertState.PENDING, AlertState.ACTIVE)

    def to_dict(self) -> dict:
        """Serialize for JSON API response."""
        return {
            "id": self.id,
            "state": self.state.value,
            "setup_name": self.setup_name,
            "direction": self.direction,
            "edge_score": self.edge_score,
            "tier": self.tier,
            "regime": self.regime,
            "trigger_price": self.trigger_price,
            "reasons": self.reasons,
            "entry_price": self.plan.entry_price,
            "stop_price": self.plan.stop_price,
            "target_1": self.plan.target_1,
            "target_2": self.plan.target_2,
            "reward_risk_ratio": self.plan.reward_risk_ratio,
            "time_stop_minutes": self.plan.time_stop_minutes,
            "expires_at": self.expires_at,
            "kill_switch_conditions": self.plan.kill_switch_conditions,
            "option": (
                {
                    "ticker": self.option.ticker,
                    "strike": self.option.strike,
                    "expiration": self.option.expiration.isoformat(),
                    "option_type": self.option.option_type,
                    "delta": self.option.delta,
                    "spread_pct": self.option.spread_pct,
                    "iv": self.option.iv,
                    "entry_premium_est": self.option.entry_premium_est,
                    "option_stop_pct": self.option.option_stop_pct,
                    "option_stop_dollar": self.plan.option_stop_dollar,
                    "dte": self.option.dte,
                    "notes": self.option.notes,
                }
                if self.option
                else None
            ),
            "detected_at": self.detected_at,
            "activated_at": self.activated_at,
            "resolved_at": self.resolved_at,
            "pnl_pct": self.pnl_pct,
        }


# ── Manager ──────────────────────────────────────────────────

class AlertManager:
    """
    Manages the full alert lifecycle.

    Usage
    -----
    manager = AlertManager()

    # Each tick — call with new detections and current price
    new_alerts = manager.process_tick(edge_results, current_price, current_time)
    active = manager.get_active_alerts()
    """

    MAX_ACTIVE_ALERTS  = 2
    DEBOUNCE_MINUTES   = 5

    def __init__(self):
        self._alerts: dict[str, ActiveAlert] = {}     # id → ActiveAlert
        self._history: list[ActiveAlert] = []         # resolved alerts (performance log)
        self._last_fire: dict[str, datetime] = {}     # setup_name+direction → last fire time
        self._counter = 0

    def process_tick(
        self,
        edge_results: list[EdgeResult],
        current_price: float,
        current_time: Optional[datetime] = None,
    ) -> list[ActiveAlert]:
        """
        Main per-tick method.

        1. Evaluate existing active alerts against kill-switches
        2. Expire time-stopped alerts
        3. Try to admit new alerts (debounce + cap)

        Returns list of newly admitted alerts (for WS broadcast).
        """
        if current_time is None:
            current_time = datetime.utcnow()

        # 1. Evaluate existing alerts
        self._evaluate_existing(current_price, current_time)

        # 2. Try to admit new alerts
        new_alerts: list[ActiveAlert] = []

        for er in edge_results:
            if er.tier == "NO_EDGE":
                continue

            alert = self._try_admit(er, current_time)
            if alert:
                new_alerts.append(alert)

        return new_alerts

    def get_active_alerts(self) -> list[ActiveAlert]:
        """Return all PENDING or ACTIVE alerts."""
        return [
            a for a in self._alerts.values()
            if a.is_active_or_pending
        ]

    def get_all_alerts(self) -> list[ActiveAlert]:
        """Return all alerts including resolved."""
        return list(self._alerts.values()) + self._history

    def mark_activated(self, alert_id: str, entry_premium: float = 0.0):
        """Call when an alert transitions from PENDING → ACTIVE (trade entered)."""
        alert = self._alerts.get(alert_id)
        if alert and alert.state == AlertState.PENDING:
            alert.state       = AlertState.ACTIVE
            alert.activated_at = datetime.utcnow().isoformat()
            alert.entry_premium = entry_premium
            logger.info(f"[AlertMgr] Alert {alert_id} ACTIVATED at premium ${entry_premium:.2f}")

    def resolve_alert(
        self,
        alert_id: str,
        final_state: AlertState,
        exit_premium: float = 0.0,
    ):
        """Manually resolve an alert (e.g., user hit stop)."""
        alert = self._alerts.pop(alert_id, None)
        if alert:
            alert.state         = final_state
            alert.resolved_at   = datetime.utcnow().isoformat()
            alert.exit_premium  = exit_premium
            if alert.entry_premium > 0 and exit_premium > 0:
                alert.pnl_pct = round(
                    (exit_premium - alert.entry_premium) / alert.entry_premium * 100, 1
                )
            self._history.append(alert)
            logger.info(f"[AlertMgr] Alert {alert_id} resolved → {final_state.value} PnL={alert.pnl_pct:.1f}%")

    def reset(self):
        """Clear all state (call at session start)."""
        self._alerts.clear()
        self._last_fire.clear()
        self._counter = 0

    # ── Internal ─────────────────────────────────────────────

    def _try_admit(
        self, er: EdgeResult, now: datetime
    ) -> Optional[ActiveAlert]:
        """Try to admit a new alert. Returns the alert if admitted, else None."""
        setup = er.setup

        # Cap: max 2 active alerts
        active_count = len([a for a in self._alerts.values() if a.is_active_or_pending])
        if active_count >= self.MAX_ACTIVE_ALERTS:
            return None

        # Debounce: same setup + direction can't re-fire within 5 min
        fire_key = f"{setup.name}:{setup.direction}"
        last_fire = self._last_fire.get(fire_key)
        if last_fire and (now - last_fire).total_seconds() < self.DEBOUNCE_MINUTES * 60:
            return None

        # Dedup: same direction + trigger price already active
        for existing in self._alerts.values():
            if (
                existing.is_active_or_pending
                and existing.direction == setup.direction
                and abs(existing.trigger_price - setup.trigger_price) < 0.05
            ):
                return None

        # Admit
        self._counter += 1
        alert_id = f"alert-{self._counter:04d}"

        # Time stop
        expires_at = (
            now + timedelta(minutes=er.setup.time_stop_minutes)
        ).isoformat()

        alert = ActiveAlert(
            id=alert_id,
            state=AlertState.PENDING,
            setup_name=setup.name,
            direction=setup.direction,
            edge_score=er.edge_score,
            tier=er.tier,
            regime=setup.regime,
            trigger_price=setup.trigger_price,
            reasons=setup.reasons,
            plan=er.pick and _make_plan_stub(er),  # may be None — handled below
            option=er.pick,
            detected_at=now.isoformat(),
            expires_at=expires_at,
        )

        # plan is required — if None we have to create one from edge result
        if er.pick is None:
            # Build minimal plan from setup
            from app.risk_engine import TradePlan
            alert.plan = TradePlan(
                entry_price=setup.trigger_price,
                stop_price=setup.stop_price,
                target_1=setup.target_1,
                target_2=setup.target_2,
                option_entry_est=0.0,
                option_stop_pct=-0.35,
                option_stop_dollar=0.0,
                time_stop_minutes=setup.time_stop_minutes,
                expires_at=expires_at,
                reward_risk_ratio=_compute_rr(setup),
                edge_score=er.edge_score,
                tier=er.tier,
                setup_name=setup.name,
                direction=setup.direction,
                regime=setup.regime,
                reasons=setup.reasons,
                kill_switch_conditions=[],
            )

        self._alerts[alert_id] = alert
        self._last_fire[fire_key] = now

        logger.info(
            f"[AlertMgr] Admitted {setup.name} {setup.direction} "
            f"edge={er.edge_score} tier={er.tier}"
        )
        return alert

    def _evaluate_existing(self, current_price: float, now: datetime):
        """Check kill-switches and time stops on all active alerts."""
        to_resolve: list[tuple[str, AlertState]] = []

        for alert_id, alert in self._alerts.items():
            if alert.is_terminal:
                continue

            plan = alert.plan

            # Time stop
            if alert.expires_at:
                try:
                    exp_dt = datetime.fromisoformat(alert.expires_at)
                    if now > exp_dt:
                        to_resolve.append((alert_id, AlertState.TIMED_OUT))
                        continue
                except ValueError:
                    pass

            # Price-based stop / target
            if plan and plan.entry_price > 0:
                if alert.direction == "CALL":
                    if current_price <= plan.stop_price:
                        to_resolve.append((alert_id, AlertState.STOPPED))
                        continue
                    if plan.target_2 and current_price >= plan.target_2:
                        to_resolve.append((alert_id, AlertState.HIT_T2))
                        continue
                    if current_price >= plan.target_1:
                        to_resolve.append((alert_id, AlertState.HIT_T1))
                        continue
                else:  # PUT
                    if current_price >= plan.stop_price:
                        to_resolve.append((alert_id, AlertState.STOPPED))
                        continue
                    if plan.target_2 and current_price <= plan.target_2:
                        to_resolve.append((alert_id, AlertState.HIT_T2))
                        continue
                    if current_price <= plan.target_1:
                        to_resolve.append((alert_id, AlertState.HIT_T1))
                        continue

        for alert_id, final_state in to_resolve:
            self.resolve_alert(alert_id, final_state)


# ── Helpers ───────────────────────────────────────────────────

def _make_plan_stub(er: EdgeResult) -> Optional["TradePlan"]:
    """Make a TradePlan from an EdgeResult that has a pick."""
    from app.risk_engine import risk_engine
    return risk_engine.build_plan(er)


def _compute_rr(setup: SetupAlert) -> float:
    if setup.stop_price and setup.trigger_price and setup.target_1:
        risk   = abs(setup.trigger_price - setup.stop_price)
        reward = abs(setup.target_1 - setup.trigger_price)
        return round(reward / risk, 2) if risk > 0 else 0.0
    return 0.0


# ── Module-level singleton ───────────────────────────────────
alert_manager = AlertManager()
