"""Twilio WhatsApp alert fan-out using Content Templates + Verify API for OTP."""
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


def _whatsapp_from() -> str:
    return settings.TWILIO_WHATSAPP_FROM or ""


def _format_expiration(alert: ActiveAlert) -> str:
    """Format option expiration as 'Mar 16'."""
    if alert.option and alert.option.expiration:
        return alert.option.expiration.strftime("%b %d")
    return "-"


def _format_contract(alert: ActiveAlert) -> str:
    """Format as 'QQQ C 600'."""
    symbol = (alert.symbol or "").upper()
    if alert.option:
        opt_type = "C" if (alert.option.option_type or "").lower() == "call" else "P"
        strike = f"{alert.option.strike:.0f}" if alert.option.strike else "-"
        return f"{symbol} {opt_type} {strike}"
    direction = "C" if (alert.direction or "").upper() == "CALL" else "P"
    return f"{symbol} {direction}"


def _build_template_vars(preference_key: str, alert: ActiveAlert) -> dict:
    """Build Content Template variables for og_trade_alert_v2.

    Template body:
    OptionGangster Trade Alert

    Signal: {{1}}
    Contract: {{2}}
    Expiration: {{3}}
    Details: {{4}}

    Manage alerts at optiongangster.com
    """
    label_map = {
        "actionable": "ACTIONABLE",
        "activated": "ACTIVATED",
        "hit_t1": "T1 HIT",
        "hit_t2": "T2 HIT",
        "stopped": "STOPPED",
        "timed_out": "TIMEOUT",
        "invalidated": "INVALID",
    }
    header = label_map.get(preference_key, preference_key.upper())
    contract = _format_contract(alert)
    expiration = _format_expiration(alert)

    trigger = f"{alert.trigger_price:.0f}" if alert.trigger_price else "-"
    stop = f"{alert.plan.stop_price:.0f}" if alert.plan and alert.plan.stop_price else "-"
    t1 = f"{alert.plan.target_1:.0f}" if alert.plan and alert.plan.target_1 else "-"
    t2 = f"{alert.plan.target_2:.0f}" if alert.plan and alert.plan.target_2 else "-"

    if preference_key == "actionable":
        details = f"E {trigger} | T1 {t1} | T2 {t2} | S {stop}"
    elif preference_key == "activated":
        entry = f"{alert.entry_premium:.2f}" if alert.entry_premium else trigger
        details = f"Filled near {entry} | T1 {t1} | T2 {t2} | S {stop}"
    elif preference_key == "hit_t1":
        details = f"{t1} reached | T2 {t2} | S {stop}"
    elif preference_key == "stopped":
        details = f"Stop {stop} hit"
    elif preference_key == "hit_t2":
        details = f"{t2} reached"
    else:
        pnl = f"{alert.pnl_pct:.1f}%" if alert.pnl_pct else "0.0%"
        details = f"PnL {pnl}"

    return {"1": header, "2": contract, "3": expiration, "4": details}


async def _send_content_template(to_number: str, content_sid: str, variables: dict | None = None) -> dict | None:
    """Send a WhatsApp message using a Twilio Content Template."""
    wa_from = _whatsapp_from()
    if not wa_from or not settings.TWILIO_ACCOUNT_SID or not settings.TWILIO_AUTH_TOKEN:
        return None

    url = f"https://api.twilio.com/2010-04-01/Accounts/{settings.TWILIO_ACCOUNT_SID}/Messages.json"
    auth = (settings.TWILIO_ACCOUNT_SID, settings.TWILIO_AUTH_TOKEN)
    data = {
        "From": wa_from,
        "To": f"whatsapp:+{to_number.lstrip('+')}",
        "ContentSid": content_sid,
    }
    if variables:
        import json
        data["ContentVariables"] = json.dumps(variables)

    async with httpx.AsyncClient(timeout=15.0) as client:
        response = await client.post(url, auth=auth, data=data)
        response.raise_for_status()
        return response.json()


async def send_otp(phone_number: str, email: str) -> bool:
    """Send an OTP via Twilio Verify API (SMS channel) to verify phone ownership."""
    verify_sid = settings.TWILIO_VERIFY_SERVICE_SID
    if not verify_sid or not settings.TWILIO_ACCOUNT_SID or not settings.TWILIO_AUTH_TOKEN:
        logger.warning("[WhatsApp] Verify service not configured")
        return False

    # Store phone→email mapping so verify_otp can look it up
    db = get_db()
    await db.whatsapp_otp.update_one(
        {"email": email},
        {"$set": {"email": email, "phone": phone_number}},
        upsert=True,
    )

    url = f"https://verify.twilio.com/v2/Services/{verify_sid}/Verifications"
    auth = (settings.TWILIO_ACCOUNT_SID, settings.TWILIO_AUTH_TOKEN)
    # Normalize phone to E.164
    clean_phone = "+" + phone_number.lstrip("+").lstrip("0")

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(url, auth=auth, data={
                "To": clean_phone,
                "Channel": "sms",
            })
            resp.raise_for_status()
            data = resp.json()
            logger.info("[WhatsApp] Verify OTP sent to %s (sid=%s, channel=%s)",
                        phone_number, data.get("sid", ""), data.get("channel", ""))
            return True
    except Exception as exc:
        logger.warning("[WhatsApp] Failed to send Verify OTP to %s: %s", phone_number, exc)
        return False


async def verify_otp(phone_number: str, code: str, email: str) -> bool:
    """Check OTP via Twilio Verify API and mark WhatsApp as verified."""
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
        logger.warning("[WhatsApp] Verify check failed for %s: %s", phone_number, exc)
        return False

    # Mark verified in user alert settings
    db = get_db()
    await db.user_alert_settings.update_one(
        {"email": email},
        {
            "$set": {
                "whatsapp_number": phone_number,
                "whatsapp_verified": True,
                "whatsapp_enabled": True,
            }
        },
        upsert=True,
    )
    return True


class TwilioWhatsAppNotifier:
    @staticmethod
    def is_enabled() -> bool:
        return all([
            settings.TWILIO_ACCOUNT_SID,
            settings.TWILIO_AUTH_TOKEN,
            _whatsapp_from(),
            settings.TWILIO_WA_ALERT_TEMPLATE_SID,
        ])

    async def handle_alert_event(self, event_type: str, alert: ActiveAlert) -> None:
        preference_key = _event_preference_key(event_type, alert)
        if not preference_key:
            return
        if not self.is_enabled():
            return

        template_sid = settings.TWILIO_WA_ALERT_TEMPLATE_SID
        variables = _build_template_vars(preference_key, alert)
        symbol = (alert.symbol or "").upper()

        # Find all users subscribed to this symbol with WhatsApp enabled
        db = get_db()
        from app.user_alert_store import UserAlertStore
        cursor = db.user_alert_settings.find(
            {
                "symbols": symbol,
                "whatsapp_enabled": True,
                "whatsapp_verified": True,
                "whatsapp_number": {"$ne": ""},
            },
            {"_id": 0},
        )
        subscribers = await cursor.to_list(length=500)
        if not subscribers:
            return

        for subscriber in subscribers:
            effective_types = UserAlertStore.effective_alert_types(subscriber, symbol)
            if not effective_types.get(preference_key):
                continue

            phone = subscriber.get("whatsapp_number", "")
            if not phone:
                continue

            try:
                result = await _send_content_template(phone, template_sid, variables)
                logger.info(
                    "[WhatsApp] Sent %s alert for %s to %s (%s)",
                    preference_key, symbol, phone,
                    result.get("sid", "") if result else "",
                )
            except Exception as exc:
                logger.warning(
                    "[WhatsApp] Failed to send %s alert for %s to %s: %s",
                    preference_key, symbol, phone, exc,
                )


twilio_whatsapp_notifier = TwilioWhatsAppNotifier()
