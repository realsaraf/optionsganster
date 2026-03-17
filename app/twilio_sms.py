"""Twilio SMS alert fan-out with per-user phone numbers + Verify OTP."""
from __future__ import annotations

import logging

import httpx

from app.alert_manager import ActiveAlert
from app.config import settings
from app.mongo import get_db

logger = logging.getLogger("optionsganster")

_KNOWN_ALERT_TYPES = {
    "actionable",
    "activated",
    "hit_t1",
    "hit_t2",
    "stopped",
    "timed_out",
    "invalidated",
}


def _event_preference_key(event_type: str, alert: ActiveAlert) -> str | None:
    if event_type == "admitted":
        return "actionable"
    if event_type == "activated":
        return "activated"
    if event_type == "resolved":
        return alert.state.value.lower()
    return None


def _sms_body(preference_key: str, alert: ActiveAlert) -> str:
    symbol = (alert.symbol or "").upper()
    setup = (alert.setup_name or "setup").strip()
    direction = (alert.direction or "").upper()
    trigger = f"{alert.trigger_price:.2f}" if alert.trigger_price else "pending"
    stop = f"{alert.plan.stop_price:.2f}" if alert.plan and alert.plan.stop_price else "-"
    target_1 = f"{alert.plan.target_1:.2f}" if alert.plan and alert.plan.target_1 else "-"
    label_map = {
        "actionable": "ACTIONABLE",
        "activated": "ENTERED",
        "hit_t1": "T1 HIT",
        "hit_t2": "T2 HIT",
        "stopped": "STOPPED",
        "timed_out": "TIMEOUT",
        "invalidated": "INVALID",
    }
    header = label_map.get(preference_key, preference_key.upper())
    if preference_key == "actionable":
        body = f"OG {header}: {symbol} {direction} {setup}. Trigger {trigger}. Stop {stop}. T1 {target_1}."
    elif preference_key == "activated":
        premium = f"{alert.entry_premium:.2f}" if alert.entry_premium else "-"
        body = f"OG {header}: {symbol} {direction} {setup}. Entry premium {premium}. Stop {stop}. T1 {target_1}."
    else:
        pnl = f"{alert.pnl_pct:.1f}%" if alert.pnl_pct else "0.0%"
        body = f"OG {header}: {symbol} {direction} {setup}. State {alert.state.value}. PnL {pnl}."
    return body[:320]


# ── OTP via Twilio Verify ───────────────────────────────────

async def send_sms_otp(phone_number: str, email: str) -> bool:
    """Send an OTP via Twilio Verify API (SMS) to verify phone ownership."""
    verify_sid = settings.TWILIO_VERIFY_SERVICE_SID
    if not verify_sid or not settings.TWILIO_ACCOUNT_SID or not settings.TWILIO_AUTH_TOKEN:
        logger.warning("[SMS] Verify service not configured")
        return False

    db = get_db()
    await db.sms_otp.update_one(
        {"email": email},
        {"$set": {"email": email, "phone": phone_number}},
        upsert=True,
    )

    url = f"https://verify.twilio.com/v2/Services/{verify_sid}/Verifications"
    auth = (settings.TWILIO_ACCOUNT_SID, settings.TWILIO_AUTH_TOKEN)
    clean_phone = "+" + phone_number.lstrip("+").lstrip("0")

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(url, auth=auth, data={
                "To": clean_phone,
                "Channel": "sms",
            })
            resp.raise_for_status()
            data = resp.json()
            logger.info("[SMS] Verify OTP sent to %s (sid=%s)", phone_number, data.get("sid", ""))
            return True
    except Exception as exc:
        logger.warning("[SMS] Failed to send Verify OTP to %s: %s", phone_number, exc)
        return False


async def verify_sms_otp(phone_number: str, code: str, email: str) -> bool:
    """Check OTP via Twilio Verify API and mark SMS as verified."""
    verify_sid = settings.TWILIO_VERIFY_SERVICE_SID
    if not verify_sid or not settings.TWILIO_ACCOUNT_SID or not settings.TWILIO_AUTH_TOKEN:
        return False

    url = f"https://verify.twilio.com/v2/Services/{verify_sid}/VerificationCheck"
    auth = (settings.TWILIO_ACCOUNT_SID, settings.TWILIO_AUTH_TOKEN)
    clean_phone = "+" + phone_number.lstrip("+").lstrip("0")

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(url, auth=auth, data={
                "To": clean_phone,
                "Code": code,
            })
            resp.raise_for_status()
            data = resp.json()
            if data.get("status") != "approved":
                return False
    except Exception as exc:
        logger.warning("[SMS] Verify check failed for %s: %s", phone_number, exc)
        return False

    db = get_db()
    await db.user_alert_settings.update_one(
        {"email": email},
        {
            "$set": {
                "sms_number": phone_number,
                "sms_verified": True,
                "sms_enabled": True,
            }
        },
        upsert=True,
    )
    return True


# ── Alert fan-out ───────────────────────────────────────────

class TwilioSMSNotifier:
    @staticmethod
    def is_enabled() -> bool:
        return all([
            settings.TWILIO_ACCOUNT_SID,
            settings.TWILIO_AUTH_TOKEN,
            settings.TWILIO_SMS_FROM,
        ])

    async def handle_alert_event(self, event_type: str, alert: ActiveAlert) -> None:
        preference_key = _event_preference_key(event_type, alert)
        if not preference_key:
            return
        if not self.is_enabled():
            return

        body = _sms_body(preference_key, alert)
        symbol = (alert.symbol or "").upper()

        db = get_db()
        from app.user_alert_store import UserAlertStore
        cursor = db.user_alert_settings.find(
            {
                "symbols": symbol,
                "sms_enabled": True,
                "sms_verified": True,
                "sms_number": {"$ne": ""},
            },
            {"_id": 0},
        )
        subscribers = await cursor.to_list(length=500)
        if not subscribers:
            return

        url = f"https://api.twilio.com/2010-04-01/Accounts/{settings.TWILIO_ACCOUNT_SID}/Messages.json"
        auth = (settings.TWILIO_ACCOUNT_SID, settings.TWILIO_AUTH_TOKEN)

        async with httpx.AsyncClient(timeout=15.0) as client:
            for subscriber in subscribers:
                effective_types = UserAlertStore.effective_alert_types(subscriber, symbol)
                if not effective_types.get(preference_key):
                    continue

                phone = subscriber.get("sms_number", "")
                if not phone:
                    continue

                try:
                    response = await client.post(
                        url,
                        auth=auth,
                        data={
                            "From": settings.TWILIO_SMS_FROM,
                            "To": phone,
                            "Body": body,
                        },
                    )
                    response.raise_for_status()
                    payload = response.json()
                    logger.info("[SMS] Sent %s alert for %s to %s (%s)", preference_key, symbol, phone, payload.get("sid", ""))
                except Exception as exc:
                    logger.warning("[SMS] Failed to send %s alert for %s to %s: %s", preference_key, symbol, phone, exc)


twilio_sms_notifier = TwilioSMSNotifier()