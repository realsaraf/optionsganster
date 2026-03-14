"""Per-user alert subscription settings and notification storage."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from app.alert_manager import ActiveAlert
from app.mongo import get_db

UTC = timezone.utc
DEFAULT_ALERT_TYPES = {
    "actionable": True,
    "activated": False,
    "hit_t1": False,
    "hit_t2": False,
    "stopped": False,
    "timed_out": False,
    "invalidated": False,
}
VALID_ALERT_MODES = {"shared", "per_symbol"}


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _alert_key(alert: ActiveAlert) -> str:
    symbol = (alert.symbol or "UNKNOWN").upper()
    return f"{symbol}|{alert.detected_at}|{alert.id}"


def _event_preference_key(event_type: str, alert: ActiveAlert) -> Optional[str]:
    if event_type == "admitted":
        return "actionable"
    if event_type == "activated":
        return "activated"
    if event_type == "resolved":
        return alert.state.value.lower()
    return None


def _title_for(preference_key: str, alert: ActiveAlert) -> str:
    symbol = (alert.symbol or "").upper()
    base = f"{symbol} {alert.direction} {alert.setup_name}".strip()
    labels = {
        "actionable": "Actionable Alert",
        "activated": "Trade Activated",
        "hit_t1": "Target 1 Hit",
        "hit_t2": "Target 2 Hit",
        "stopped": "Stopped Out",
        "timed_out": "Timed Out",
        "invalidated": "Invalidated",
    }
    return f"{labels.get(preference_key, 'Alert')}: {base}"


def _message_for(preference_key: str, alert: ActiveAlert) -> str:
    trigger = f"trigger {alert.trigger_price:.2f}" if alert.trigger_price else "trigger pending"
    if preference_key == "actionable":
        return f"{alert.tier} tier setup detected with {trigger}. Entry condition: {alert.entry_condition}."
    if preference_key == "activated":
        return f"Trade was marked entered. Estimated option entry premium: {alert.entry_premium:.2f}."
    if preference_key == "hit_t1":
        return f"Price reached target 1. Review the trade for a trim or stop adjustment."
    if preference_key == "hit_t2":
        return f"Price reached target 2."
    if preference_key == "stopped":
        return f"The setup hit its stop condition."
    if preference_key == "timed_out":
        return f"The setup expired on time stop without reaching a target."
    if preference_key == "invalidated":
        return f"The setup invalidated before entry."
    return f"Alert event {preference_key} recorded for {alert.symbol}."


class UserAlertStore:
    def _db(self):
        return get_db()

    @staticmethod
    def normalize_email(email: str) -> str:
        return (email or "").strip().lower()

    @staticmethod
    def normalize_symbols(symbols: list[str] | None) -> list[str]:
        cleaned: list[str] = []
        seen: set[str] = set()
        for symbol in symbols or []:
            normalized = (symbol or "").strip().upper()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            cleaned.append(normalized)
        return cleaned

    @staticmethod
    def merge_alert_types(alert_types: Optional[dict]) -> dict:
        merged = dict(DEFAULT_ALERT_TYPES)
        for key, value in (alert_types or {}).items():
            if key in merged:
                merged[key] = bool(value)
        return merged

    @classmethod
    def normalize_mode(cls, mode: Optional[str]) -> str:
        return mode if mode in VALID_ALERT_MODES else "shared"

    @classmethod
    def normalize_symbol_settings(cls, symbol_settings: list[dict] | None, symbols: list[str] | None = None, fallback: Optional[dict] = None) -> list[dict]:
        allowed = set(cls.normalize_symbols(symbols)) if symbols is not None else None
        cleaned: list[dict] = []
        seen: set[str] = set()
        for item in symbol_settings or []:
            symbol = (item or {}).get("symbol", "")
            normalized = cls.normalize_symbols([symbol])
            if not normalized:
                continue
            key = normalized[0]
            if allowed is not None and key not in allowed:
                continue
            if key in seen:
                continue
            seen.add(key)
            cleaned.append({
                "symbol": key,
                "alert_types": cls.merge_alert_types((item or {}).get("alert_types") or fallback),
            })
        return cleaned

    @classmethod
    def effective_alert_types(cls, settings_doc: dict, symbol: str) -> dict:
        mode = cls.normalize_mode(settings_doc.get("alert_mode"))
        shared = cls.merge_alert_types(settings_doc.get("shared_alert_types") or settings_doc.get("alert_types"))
        if mode != "per_symbol":
            return shared
        normalized = cls.normalize_symbols([symbol])
        if not normalized:
            return shared
        lookup = normalized[0]
        for item in settings_doc.get("symbol_settings", []) or []:
            if (item.get("symbol") or "").upper() == lookup:
                return cls.merge_alert_types(item.get("alert_types"))
        return shared

    async def get_settings(self, email: str) -> dict:
        normalized = self.normalize_email(email)
        doc = await self._db().user_alert_settings.find_one(
            {"email": normalized},
            {"_id": 0},
        )
        if not doc:
            return {
                "email": normalized,
                "symbols": [],
                "alert_mode": "shared",
                "shared_alert_types": dict(DEFAULT_ALERT_TYPES),
                "symbol_settings": [],
                "created_at": None,
                "updated_at": None,
            }
        doc["symbols"] = self.normalize_symbols(doc.get("symbols", []))
        doc["alert_mode"] = self.normalize_mode(doc.get("alert_mode"))
        doc["shared_alert_types"] = self.merge_alert_types(doc.get("shared_alert_types") or doc.get("alert_types"))
        doc["symbol_settings"] = self.normalize_symbol_settings(doc.get("symbol_settings"), doc["symbols"], doc["shared_alert_types"])
        doc["alert_types"] = dict(doc["shared_alert_types"])
        return doc

    async def update_settings(self, email: str, symbols: list[str], alert_mode: str, shared_alert_types: dict, symbol_settings: list[dict] | None = None) -> dict:
        normalized = self.normalize_email(email)
        now = _utcnow()
        normalized_symbols = self.normalize_symbols(symbols)
        normalized_mode = self.normalize_mode(alert_mode)
        merged_shared = self.merge_alert_types(shared_alert_types)
        normalized_symbol_settings = self.normalize_symbol_settings(symbol_settings, normalized_symbols, merged_shared)
        if normalized_mode == "per_symbol":
            existing = {item["symbol"]: item for item in normalized_symbol_settings}
            normalized_symbol_settings = [existing.get(symbol, {"symbol": symbol, "alert_types": dict(merged_shared)}) for symbol in normalized_symbols]
        else:
            normalized_symbol_settings = []
        update = {
            "email": normalized,
            "symbols": normalized_symbols,
            "alert_mode": normalized_mode,
            "shared_alert_types": merged_shared,
            "symbol_settings": normalized_symbol_settings,
            "alert_types": merged_shared,
            "updated_at": now,
        }
        await self._db().user_alert_settings.update_one(
            {"email": normalized},
            {
                "$set": update,
                "$setOnInsert": {"created_at": now},
            },
            upsert=True,
        )
        return await self.get_settings(normalized)

    async def get_all_subscribed_symbols(self) -> list[str]:
        db = self._db()
        authorized_emails = await db.users.distinct("email", {"is_authorized": True})
        if not authorized_emails:
            return []
        cursor = db.user_alert_settings.find(
            {
                "email": {"$in": authorized_emails},
                "symbols.0": {"$exists": True},
            },
            {"_id": 0, "symbols": 1},
        )
        docs = await cursor.to_list(length=500)
        merged: list[str] = []
        seen: set[str] = set()
        for doc in docs:
            for symbol in self.normalize_symbols(doc.get("symbols", [])):
                if symbol in seen:
                    continue
                seen.add(symbol)
                merged.append(symbol)
        return merged

    async def handle_alert_event(self, event_type: str, alert: ActiveAlert) -> None:
        preference_key = _event_preference_key(event_type, alert)
        if not preference_key:
            return

        db = self._db()
        symbol = (alert.symbol or "").upper()
        cursor = db.user_alert_settings.find(
            {"symbols": symbol},
            {"_id": 0},
        )
        subscribers = await cursor.to_list(length=500)
        if not subscribers:
            return

        alert_payload = alert.to_dict()
        for subscriber in subscribers:
            effective_types = self.effective_alert_types(subscriber, symbol)
            if not effective_types.get(preference_key):
                continue
            email = subscriber.get("email", "")
            notification_key = f"{email}|{_alert_key(alert)}|{preference_key}"
            created_at = _utcnow()
            doc = {
                "notification_key": notification_key,
                "email": email,
                "symbol": symbol,
                "event_type": preference_key,
                "alert_key": _alert_key(alert),
                "alert_id": alert.id,
                "state": alert.state.value,
                "title": _title_for(preference_key, alert),
                "message": _message_for(preference_key, alert),
                "created_at": created_at,
                "is_read": False,
                "read_at": None,
                "alert": alert_payload,
            }
            await db.user_alert_notifications.update_one(
                {"notification_key": notification_key},
                {
                    "$setOnInsert": doc,
                    "$set": {
                        "symbol": symbol,
                        "state": alert.state.value,
                        "title": doc["title"],
                        "message": doc["message"],
                        "alert": alert_payload,
                    },
                },
                upsert=True,
            )

    async def list_notifications(self, email: str, limit: int = 25) -> list[dict]:
        normalized = self.normalize_email(email)
        cursor = self._db().user_alert_notifications.find(
            {"email": normalized},
            {"_id": 0},
        ).sort("created_at", -1).limit(max(1, min(limit, 100)))
        return await cursor.to_list(length=max(1, min(limit, 100)))

    async def unread_count(self, email: str) -> int:
        normalized = self.normalize_email(email)
        return await self._db().user_alert_notifications.count_documents(
            {"email": normalized, "is_read": False}
        )

    async def mark_notifications_read(self, email: str, notification_ids: list[str] | None = None) -> int:
        normalized = self.normalize_email(email)
        query = {"email": normalized, "is_read": False}
        if notification_ids:
            query["notification_key"] = {"$in": notification_ids}
        result = await self._db().user_alert_notifications.update_many(
            query,
            {"$set": {"is_read": True, "read_at": _utcnow()}},
        )
        return int(result.modified_count)


user_alert_store = UserAlertStore()
