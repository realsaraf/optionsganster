"""
LLM Narrator
============
Optional OpenAI-powered narrative layer for the Idea Engine.

If OPENAI_API_KEY is set in the environment / .env file, this module
produces plain-English trading narratives for the DailyBriefing and
individual TradeIdeas.  If the key is absent, all functions return None
gracefully — the Idea Engine works fine without it.

Model used: gpt-4o-mini (fast + cheap for structured prompts)
Temperature: 0.4 (creative but grounded)
Max tokens: 220 per call
"""
from __future__ import annotations

import logging
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from app.idea_engine import DailyBriefing, TradeIdea

logger = logging.getLogger("optionsganster.llm")

# ── Lazy client ───────────────────────────────────────────────

_client = None
_key_checked = False


def _get_client():
    """Return an openai.OpenAI client if the key is configured, else None."""
    global _client, _key_checked
    if _key_checked:
        return _client
    _key_checked = True
    try:
        from app.config import settings
        api_key = getattr(settings, "OPENAI_API_KEY", "")
        if not api_key:
            return None
        import openai
        _client = openai.OpenAI(api_key=api_key)
        logger.info("[LLM] OpenAI client initialised (model: gpt-4o-mini)")
    except Exception as e:
        logger.warning(f"[LLM] Could not init OpenAI client: {e}")
        _client = None
    return _client


def is_available() -> bool:
    """True if the LLM narrator is configured and ready."""
    return _get_client() is not None


# ── Prompt builders ────────────────────────────────────────────

def _call(system: str, user: str, max_tokens: int = 220) -> Optional[str]:
    client = _get_client()
    if client is None:
        return None
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.4,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
        )
        text = response.choices[0].message.content or ""
        return text.strip() or None
    except Exception as e:
        logger.warning(f"[LLM] API call failed: {e}")
        return None


_SYS_BRIEFING = (
    "You are a seasoned day trader writing a concise morning briefing for options traders. "
    "Your tone is direct, professional, and actionable. "
    "Avoid generic advice. Reference the specific data points provided. "
    "Write 3-4 sentences max. No bullet points. No markdown headers. "
    "Start with the market bias, explain why, and end with what to watch for."
)

_SYS_IDEA = (
    "You are a seasoned options trader explaining a specific trade idea. "
    "Be concise (2-3 sentences), reference the entry/stop/target levels given, "
    "and explain the logic without generic disclaimers. "
    "End with a one-sentence risk note."
)


def narrate_briefing(briefing: "DailyBriefing") -> Optional[str]:
    """
    Generate a top-level narrative paragraph for the DailyBriefing.
    Returns None if LLM is not configured or fails.
    """
    if not is_available():
        return None

    b = briefing
    bias_dir = b.bias.direction.upper()
    bias_str = b.bias.strength
    regime   = b.regime_name
    rsi      = b.rsi
    vwap_pos = b.vwap_position
    day_type = b.day_type.most_likely.value if b.day_type else "UNKNOWN"
    bullets  = "; ".join(b.bias.bullets[:3]) if b.bias.bullets else "N/A"
    themes   = ", ".join(t.title for t in b.themes[:2]) if b.themes else "none"
    sym      = b.symbol
    price    = b.underlying_price

    user_prompt = (
        f"Symbol: {sym} at ${price:.2f}\n"
        f"Regime: {regime}, RSI: {rsi:.0f}, Price vs VWAP: {vwap_pos}\n"
        f"Bias: {bias_dir} ({int(bias_str*100)}%)\n"
        f"Day type: {day_type}\n"
        f"Key signals: {bullets}\n"
        f"Top themes: {themes}\n\n"
        f"Write the morning briefing narrative."
    )

    return _call(_SYS_BRIEFING, user_prompt, max_tokens=220)


def narrate_idea(idea: "TradeIdea") -> Optional[str]:
    """
    Generate a short narrative explanation for a single TradeIdea.
    Returns None if LLM is not configured or fails.
    """
    if not is_available():
        return None

    user_prompt = (
        f"Idea: {idea.headline}\n"
        f"Type: {idea.idea_type.value}, Direction: {idea.direction}\n"
        f"Entry: {idea.entry_level:.2f}, Stop: {idea.stop_level:.2f}, "
        f"T1: {idea.target_1:.2f}, T2: {idea.target_2:.2f}, R:R {idea.reward_risk:.1f}x\n"
        f"Option hint: {idea.option_hint}\n"
        f"Existing rationale: {idea.rationale}\n\n"
        f"Write a 2-3 sentence trader narrative for this idea."
    )

    return _call(_SYS_IDEA, user_prompt, max_tokens=160)
