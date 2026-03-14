"""Mongo-backed alert history storage."""
from __future__ import annotations

import logging
from datetime import date, datetime, time, timedelta, timezone
from typing import Optional

from app.alert_manager import ActiveAlert
from app.mongo import get_db

logger = logging.getLogger("optionsganster")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _parse_iso(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


class AlertStore:
    """Persist actionable alerts and their state transitions."""

    @staticmethod
    def _alert_key(alert: ActiveAlert) -> str:
        symbol = (alert.symbol or "UNKNOWN").upper()
        return f"{symbol}|{alert.detected_at}|{alert.id}"

    def _get_db(self):
        try:
            return get_db()
        except RuntimeError:
            return None

    def _serialize_alert(self, alert: ActiveAlert) -> dict:
        payload = alert.to_dict()
        payload.update(
            {
                "alert_key": self._alert_key(alert),
                "alert_id": alert.id,
                "symbol": (alert.symbol or "").upper(),
                "entry_condition": alert.entry_condition,
                "is_actionable": True,
                "updated_at": _utcnow(),
                "detected_at_dt": _parse_iso(alert.detected_at),
                "activated_at_dt": _parse_iso(alert.activated_at),
                "resolved_at_dt": _parse_iso(alert.resolved_at),
            }
        )
        return payload

    async def handle_alert_event(self, event_type: str, alert: ActiveAlert) -> None:
        """Upsert an alert document and append a lifecycle transition."""
        db = self._get_db()
        if db is None:
            return

        doc = self._serialize_alert(alert)
        transition_at = (
            doc.get("resolved_at_dt")
            or doc.get("activated_at_dt")
            or doc.get("detected_at_dt")
            or _utcnow()
        )
        transition = {
            "event": event_type,
            "state": alert.state.value,
            "at": transition_at,
        }

        try:
            await db.alerts_log.update_one(
                {"alert_key": doc["alert_key"]},
                {
                    "$set": doc,
                    "$setOnInsert": {"transitions": []},
                    "$push": {"transitions": transition},
                },
                upsert=True,
            )
        except Exception as exc:
            logger.warning("[AlertStore] failed to persist alert %s: %s", doc["alert_key"], exc)

    async def get_day_history(self, symbol: str, alert_day: date) -> list[dict]:
        """Return actionable alerts for one symbol/day, oldest first."""
        db = self._get_db()
        if db is None:
            return []

        start = datetime.combine(alert_day, time.min, tzinfo=timezone.utc)
        end = start + timedelta(days=1)
        query = {
            "symbol": symbol.upper(),
            "is_actionable": True,
            "detected_at_dt": {"$gte": start, "$lt": end},
        }
        try:
            cursor = db.alerts_log.find(query, {"_id": 0}).sort("detected_at_dt", 1)
            return await cursor.to_list(length=500)
        except Exception as exc:
            logger.warning("[AlertStore] failed to load history for %s %s: %s", symbol, alert_day, exc)
            return []


alert_store = AlertStore()
