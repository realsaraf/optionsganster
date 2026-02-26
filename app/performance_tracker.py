"""
Performance Tracker — MongoDB-backed trade logging and analytics.

Collections:
  trade_log    – one doc per resolved alert
  daily_session – daily win/loss/streak aggregates (optional cache)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta
from typing import Optional

from .mongo import get_db

log = logging.getLogger(__name__)

# ── helpers ──────────────────────────────────────────────────────────
def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class PerformanceTracker:
    """Async MongoDB-backed trade performance tracker."""

    # ── logging trades ───────────────────────────────────────────────
    async def log_trade(
        self,
        *,
        symbol: str,
        direction: str,           # "CALL" | "PUT"
        setup_type: str,          # idea_type / alert reason
        entry_price: float,
        exit_price: float,
        stop_price: float = 0.0,
        target_price: float = 0.0,
        pnl: float = 0.0,
        alert_id: str = "",
        score: int = 0,
        score_breakdown: Optional[dict] = None,
        extra: Optional[dict] = None,
    ) -> str:
        """Insert a trade doc and return its _id as string."""
        db = get_db()

        risk = abs(entry_price - stop_price) if stop_price else 0.0
        reward = abs(exit_price - entry_price) if exit_price else 0.0
        r_multiple = round(reward / risk, 2) if risk > 0 else 0.0
        is_win = pnl > 0

        doc = {
            "symbol": symbol,
            "direction": direction,
            "setup_type": setup_type,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "stop_price": stop_price,
            "target_price": target_price,
            "pnl": round(pnl, 2),
            "r_multiple": r_multiple,
            "is_win": is_win,
            "alert_id": alert_id,
            "score": score,
            "score_breakdown": score_breakdown or {},
            "closed_at": _utcnow(),
            **(extra or {}),
        }

        result = await db.trade_log.insert_one(doc)
        log.info("Logged trade %s  pnl=%.2f  R=%.2f", result.inserted_id, pnl, r_multiple)
        return str(result.inserted_id)

    # ── aggregate stats ──────────────────────────────────────────────
    async def get_stats(
        self,
        symbol: str = "",
        days: int = 30,
    ) -> dict:
        """Return performance stats for a symbol (or all) over last N days."""
        db = get_db()
        cutoff = _utcnow() - timedelta(days=days)
        query: dict = {"closed_at": {"$gte": cutoff}}
        if symbol:
            query["symbol"] = symbol.upper()

        cursor = db.trade_log.find(query).sort("closed_at", -1)
        trades = await cursor.to_list(length=500)

        if not trades:
            return {
                "total_trades": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0.0,
                "avg_r": 0.0,
                "total_pnl": 0.0,
                "expectancy": 0.0,
                "best_r": 0.0,
                "worst_r": 0.0,
                "avg_score": 0,
                "recent_trades": [],
            }

        wins = [t for t in trades if t.get("is_win")]
        losses = [t for t in trades if not t.get("is_win")]
        rs = [t.get("r_multiple", 0) for t in trades]
        pnls = [t.get("pnl", 0) for t in trades]
        scores = [t.get("score", 0) for t in trades if t.get("score")]

        win_rate = len(wins) / len(trades) if trades else 0
        avg_win_r = (sum(t.get("r_multiple", 0) for t in wins) / len(wins)) if wins else 0
        avg_loss_r = (sum(abs(t.get("r_multiple", 0)) for t in losses) / len(losses)) if losses else 0
        expectancy = (win_rate * avg_win_r) - ((1 - win_rate) * avg_loss_r)

        # Recent trades for the table (last 10)
        recent = []
        for t in trades[:10]:
            recent.append({
                "symbol": t.get("symbol", ""),
                "direction": t.get("direction", ""),
                "setup_type": t.get("setup_type", ""),
                "pnl": t.get("pnl", 0),
                "r_multiple": t.get("r_multiple", 0),
                "is_win": t.get("is_win", False),
                "score": t.get("score", 0),
                "closed_at": t.get("closed_at", "").isoformat() if hasattr(t.get("closed_at", ""), "isoformat") else str(t.get("closed_at", "")),
            })

        return {
            "total_trades": len(trades),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": round(win_rate, 3),
            "avg_r": round(sum(rs) / len(rs), 2) if rs else 0,
            "total_pnl": round(sum(pnls), 2),
            "expectancy": round(expectancy, 3),
            "best_r": round(max(rs), 2) if rs else 0,
            "worst_r": round(min(rs), 2) if rs else 0,
            "avg_score": round(sum(scores) / len(scores)) if scores else 0,
            "recent_trades": recent,
        }

    async def get_stats_by_setup(self, days: int = 30) -> list[dict]:
        """Aggregate win-rate / avg-R grouped by setup_type."""
        db = get_db()
        cutoff = _utcnow() - timedelta(days=days)

        pipeline = [
            {"$match": {"closed_at": {"$gte": cutoff}}},
            {"$group": {
                "_id": "$setup_type",
                "total": {"$sum": 1},
                "wins": {"$sum": {"$cond": ["$is_win", 1, 0]}},
                "total_pnl": {"$sum": "$pnl"},
                "avg_r": {"$avg": "$r_multiple"},
                "avg_score": {"$avg": "$score"},
            }},
            {"$sort": {"total": -1}},
        ]
        results = await db.trade_log.aggregate(pipeline).to_list(length=50)
        out = []
        for r in results:
            total = r["total"]
            out.append({
                "setup_type": r["_id"] or "unknown",
                "total_trades": total,
                "wins": r["wins"],
                "win_rate": round(r["wins"] / total, 3) if total else 0,
                "avg_r": round(r["avg_r"], 2),
                "total_pnl": round(r["total_pnl"], 2),
                "avg_score": round(r.get("avg_score") or 0),
            })
        return out


# Module-level singleton
perf_tracker = PerformanceTracker()
