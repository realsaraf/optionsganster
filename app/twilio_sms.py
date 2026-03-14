"""Twilio SMS alert fan-out for actionable trade events."""
from __future__ import annotations

import logging

import httpx

from app.alert_manager import ActiveAlert
from app.config import settings

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


def _configured_recipients() -> list[str]:
    recipients: list[str] = []
    seen: set[str] = set()
    for raw in (settings.TWILIO_SMS_TO or "").split(","):
        number = raw.strip()
        if not number or number in seen:
            continue
        seen.add(number)
        recipients.append(number)
    return recipients


def _configured_alert_types() -> set[str]:
    configured = {
        item.strip().lower()
        for item in (settings.TWILIO_SMS_ALERT_TYPES or "actionable").split(",")
        if item.strip()
    }
    matched = configured & _KNOWN_ALERT_TYPES
    return matched or {"actionable"}


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


class TwilioSMSNotifier:
    @staticmethod
    def is_enabled() -> bool:
        return all(
            [
                settings.TWILIO_ACCOUNT_SID,
                settings.TWILIO_AUTH_TOKEN,
                settings.TWILIO_SMS_FROM,
                _configured_recipients(),
            ]
        )

    async def handle_alert_event(self, event_type: str, alert: ActiveAlert) -> None:
        preference_key = _event_preference_key(event_type, alert)
        if not preference_key:
            return
        if not self.is_enabled():
            return
        if preference_key not in _configured_alert_types():
            return

        body = _sms_body(preference_key, alert)
        url = f"https://api.twilio.com/2010-04-01/Accounts/{settings.TWILIO_ACCOUNT_SID}/Messages.json"
        auth = (settings.TWILIO_ACCOUNT_SID, settings.TWILIO_AUTH_TOKEN)
        async with httpx.AsyncClient(timeout=15.0) as client:
            for recipient in _configured_recipients():
                try:
                    response = await client.post(
                        url,
                        auth=auth,
                        data={
                            "From": settings.TWILIO_SMS_FROM,
                            "To": recipient,
                            "Body": body,
                        },
                    )
                    response.raise_for_status()
                    payload = response.json()
                    logger.info("[TwilioSMS] Sent %s alert for %s to %s (%s)", preference_key, alert.symbol, recipient, payload.get("sid", ""))
                except Exception as exc:
                    logger.warning("[TwilioSMS] failed to send %s alert for %s to %s: %s", preference_key, alert.symbol, recipient, exc)


twilio_sms_notifier = TwilioSMSNotifier()