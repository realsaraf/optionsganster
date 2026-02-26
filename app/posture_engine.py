"""
Posture Engine — Decision Clarity State Machine
=================================================
Reduces ALL market signals to ONE clear trade posture.

6 posture states (can never be ambiguous — exactly one fires):
  BUY_PULLBACKS    – Trend up, not extended, dip-buy mode
  SHORT_RALLIES    – Trend down, not compressed, sell-rally mode
  BREAKOUT_WATCH   – Near key level, momentum building; wait for confirm
  WAIT_EXHAUSTION  – Price extended >1.5σ from VWAP; no new trades
  MEAN_REVERSION   – Range/chop; fade the extremes, harvest premium
  STAND_ASIDE      – High-volume / high-risk day; reduce or skip

Priority order (first match wins):
  1. HIGH_RISK volume  → STAND_ASIDE
  2. REVERSAL_EXHAUSTION regime → WAIT_EXHAUSTION
  3. BREAKOUT_ATTEMPT regime → BREAKOUT_WATCH
  4. RANGE_CHOP → MEAN_REVERSION
  5. TREND_UP + extended → WAIT_EXHAUSTION
  6. TREND_UP → BUY_PULLBACKS
  7. TREND_DOWN + compressed → WAIT_EXHAUSTION
  8. TREND_DOWN → SHORT_RALLIES
  9. fallback → MEAN_REVERSION
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd


# ── State labels ─────────────────────────────────────────────
class PostureState:
    BUY_PULLBACKS   = "BUY_PULLBACKS"
    SHORT_RALLIES   = "SHORT_RALLIES"
    BREAKOUT_WATCH  = "BREAKOUT_WATCH"
    WAIT_EXHAUSTION = "WAIT_EXHAUSTION"
    MEAN_REVERSION  = "MEAN_REVERSION"
    STAND_ASIDE     = "STAND_ASIDE"
    TRANSITION      = "TRANSITION"
    VOL_EXPANSION   = "VOL_EXPANSION"


_STATE_META = {
    PostureState.BUY_PULLBACKS: {
        "label": "Buy Pullbacks",
        "color": "green",
        "icon": "📈",
        "direction": "bullish",
        "allowed": ["Pullback to VWAP", "Breakout retest", "Momentum continuation"],
        "avoid":   ["Countertrend shorts", "Fades at highs"],
    },
    PostureState.SHORT_RALLIES: {
        "label": "Short Rallies",
        "color": "red",
        "icon": "📉",
        "direction": "bearish",
        "allowed": ["Rally fade", "Breakdown continuation", "Overhead resistance rejection"],
        "avoid":   ["Buying dips", "Catching bottoms"],
    },
    PostureState.BREAKOUT_WATCH: {
        "label": "Breakout Watch",
        "color": "blue",
        "icon": "🔥",
        "direction": "neutral",
        "allowed": ["Breakout confirmation", "Failed breakout fade"],
        "avoid":   ["Premature entry before confirmation", "Fading with size"],
    },
    PostureState.WAIT_EXHAUSTION: {
        "label": "Wait — Exhaustion",
        "color": "orange",
        "icon": "⏸",
        "direction": "neutral",
        "allowed": ["Exhaustion reversal with confirmation", "Tight scalp only"],
        "avoid":   ["Trend continuation", "Breakouts", "Momentum plays"],
    },
    PostureState.MEAN_REVERSION: {
        "label": "Mean Reversion",
        "color": "purple",
        "icon": "🔄",
        "direction": "neutral",
        "allowed": ["Range top fade", "Range bottom buy", "VWAP revert"],
        "avoid":   ["Momentum plays", "Breakouts (fade them)", "Wide stops"],
    },
    PostureState.STAND_ASIDE: {
        "label": "Stand Aside",
        "color": "gray",
        "icon": "🚫",
        "direction": "neutral",
        "allowed": ["Paper trade only", "0.25× size max if must trade"],
        "avoid":   ["Full size", "New setups", "Momentum chasing"],
    },
    PostureState.TRANSITION: {
        "label": "Transition",
        "color": "yellow",
        "icon": "⚠",
        "direction": "neutral",
        "allowed": [],
        "avoid":   ["All directional trades", "New positions"],
    },
    PostureState.VOL_EXPANSION: {
        "label": "Vol Expansion",
        "color": "cyan",
        "icon": "💥",
        "direction": "neutral",
        "allowed": ["Breakout continuation", "Level break retest"],
        "avoid":   ["Fades", "Mean reversion", "Counter-trend"],
    },
}


# ── Output dataclass ──────────────────────────────────────────
@dataclass
class TradePosture:
    state: str                      # PostureState constant
    state_label: str                # human-readable label
    icon: str                       # emoji
    color: str                      # "green" | "red" | "blue" | "orange" | "purple" | "gray"
    direction: str                  # "bullish" | "bearish" | "neutral"
    headline: str                   # single punchy line
    best_play: str                  # 2-4 sentence tactical description
    best_play_bullets: list[str] = field(default_factory=list)  # ➡ action bullets
    allowed_setups: list[str]       = field(default_factory=list)
    avoid: list[str]                = field(default_factory=list)
    entry_zone_low: Optional[float] = None   # price zone low
    entry_zone_high: Optional[float]= None   # price zone high
    stop_level: Optional[float]     = None
    targets: list[float]            = field(default_factory=list)
    option_action: str              = ""     # what to do with options
    kill_switch: str                = ""     # what invalidates this posture
    confidence: float               = 0.0
    stats_note: str                 = ""     # e.g. "23.5% WR on high-vol days"
    capital_mode: str               = "NORMAL"       # NORMAL | REDUCED | LOCKED
    capital_mode_reason: str        = ""
    max_size_mult: float            = 1.0
    locked_until: Optional[str]     = None           # ISO timestamp


# ── Engine ────────────────────────────────────────────────────
class PostureEngine:
    """
    Reduces regime + SR + VPA + chain + volume regime → TradePosture.
    Pure computation — no I/O.
    """

    # Extension thresholds (in ATR multiples from VWAP)
    EXT_WARN  = 1.5   # flag as approaching exhaustion
    EXT_STOP  = 2.0   # clear exhaustion — no new trades

    def compute(
        self,
        *,
        regime,           # RegimeResult or None
        sr,               # SRResult or None
        chain_metrics,    # ChainMetrics or None
        vpa_bias: dict,   # from VPAEngine.get_bias()
        vol_regime: dict, # from VPAEngine.get_volume_regime()
        underlying_price: float,
        df=None,          # intraday DataFrame (optional — for σ calc)
    ) -> TradePosture:
        """Main entry point."""

        # ── Extract raw fields ────────────────────────────────
        regime_name = ""
        vwap = 0.0
        atr  = 0.0
        rsi  = 50.0
        vwap_pos = "at"
        regime_conf = 0.5

        if regime:
            rn = getattr(regime, "regime", "")
            regime_name = rn.value if hasattr(rn, "value") else str(rn)
            vwap = float(getattr(regime, "vwap_current", 0) or 0)
            atr  = float(getattr(regime, "atr_current",  0) or 0)
            rsi  = float(getattr(regime, "rsi_current",  50) or 50)
            vwap_pos = getattr(regime, "price_vs_vwap", "at") or "at"
            regime_conf = float(getattr(regime, "confidence", 0.5) or 0.5)

        vol_regime_name = (vol_regime or {}).get("regime", "NORMAL")
        vol_ratio = float((vol_regime or {}).get("ratio", 1.0) or 1.0)
        vol_detail = (vol_regime or {}).get("detail", "")

        gex_regime  = ""
        max_pain    = 0.0
        if chain_metrics:
            gex_regime = getattr(chain_metrics, "gex_regime", "") or ""
            max_pain   = float(getattr(chain_metrics, "max_pain", 0) or 0)

        prox_score = 0.0
        nearest_sup = 0.0
        nearest_res = 0.0
        if sr:
            prox_score  = float(getattr(sr, "proximity_score", 0) or 0)
            nearest_sup = float(getattr(sr, "nearest_support", 0) or getattr(sr, "nearest_support", underlying_price * 0.998))
            nearest_res = float(getattr(sr, "nearest_resistance", 0) or getattr(sr, "nearest_resistance", underlying_price * 1.002))
            if not nearest_sup and sr.levels:
                sups = [l.price for l in sr.levels if l.kind == "support" and l.price < underlying_price]
                nearest_sup = max(sups) if sups else underlying_price - atr
            if not nearest_res and sr.levels:
                ress = [l.price for l in sr.levels if l.kind == "resistance" and l.price > underlying_price]
                nearest_res = min(ress) if ress else underlying_price + atr

        # σ extension: distance from VWAP in ATR units
        sigma = 0.0
        if vwap > 0 and atr > 0:
            sigma = (underlying_price - vwap) / atr

        # VPA bias direction
        vpa_dir = (vpa_bias or {}).get("bias", "neutral") or "neutral"

        # Impulse candle count (consecutive same-dir candles with body > 1.5× ATR)
        impulse_count = 0
        if df is not None and not df.empty and atr > 0:
            try:
                last_dir = None
                for i in range(len(df) - 1, max(len(df) - 10, -1), -1):
                    row = df.iloc[i]
                    body = abs(float(row["close"]) - float(row["open"]))
                    is_bull = float(row["close"]) > float(row["open"])
                    if body > atr * 1.5:
                        d = "up" if is_bull else "down"
                        if last_dir is None:
                            last_dir = d
                        if d == last_dir:
                            impulse_count += 1
                        else:
                            break
                    else:
                        break
            except Exception:
                pass

        # ATR compression ratio for breakout detection
        atr_compression = 1.0
        if df is not None and len(df) >= 30 and atr > 0:
            try:
                recent_range = df.iloc[-10:]["high"].max() - df.iloc[-10:]["low"].min()
                older_range = df.iloc[-30:-10]["high"].max() - df.iloc[-30:-10]["low"].min()
                if older_range > 0:
                    atr_compression = recent_range / older_range
            except Exception:
                pass

        # Proximity to nearest resistance (for breakout detection)
        res_dist_pct = 999.0
        if nearest_res > 0 and underlying_price > 0:
            res_dist_pct = abs(nearest_res - underlying_price) / underlying_price

        # ── State machine ─────────────────────────────────────
        state = self._classify_state(
            regime_name=regime_name,
            vwap_pos=vwap_pos,
            sigma=sigma,
            rsi=rsi,
            vol_regime_name=vol_regime_name,
            gex_regime=gex_regime,
            prox_score=prox_score,
            vpa_dir=vpa_dir,
            regime_conf=regime_conf,
            vol_ratio=vol_ratio,
            impulse_count=impulse_count,
            atr_compression=atr_compression,
            res_dist_pct=res_dist_pct,
        )

        meta = _STATE_META[state]
        p    = underlying_price
        atr_safe = atr if atr > 0 else p * 0.001

        # ── Build concrete levels ─────────────────────────────
        ez_low = ez_high = stop = None
        targets = []
        option_action = ""
        kill_switch   = ""

        if state == PostureState.BUY_PULLBACKS:
            # Entry: pullback toward VWAP or nearest support
            ez_high = round(vwap + atr_safe * 0.2, 2) if vwap > 0 else round(p, 2)
            ez_low  = round(vwap - atr_safe * 0.3, 2) if vwap > 0 else round(p * 0.998, 2)
            if nearest_sup > 0 and nearest_sup > ez_low:
                ez_low = round(nearest_sup, 2)
            stop    = round(ez_low - atr_safe * 0.5, 2)
            t1 = round(p + atr_safe * 1.0, 2)
            t2 = round(p + atr_safe * 2.0, 2)
            if nearest_res > p:
                t1 = round(min(t1, nearest_res * 0.998), 2)
            targets = [t1, t2]
            option_action = "Buy ATM or 1-strike OTM calls on pullback confirmation. Δ 0.40–0.55."
            kill_switch = f"Posture flips if price breaks below VWAP ({_fp(vwap)}) on volume."

        elif state == PostureState.SHORT_RALLIES:
            ez_low  = round(vwap - atr_safe * 0.2, 2) if vwap > 0 else round(p, 2)
            ez_high = round(vwap + atr_safe * 0.3, 2) if vwap > 0 else round(p * 1.002, 2)
            if nearest_res > 0 and nearest_res < ez_high:
                ez_high = round(nearest_res, 2)
            stop    = round(ez_high + atr_safe * 0.5, 2)
            t1 = round(p - atr_safe * 1.0, 2)
            t2 = round(p - atr_safe * 2.0, 2)
            if nearest_sup > 0:
                t1 = round(max(t1, nearest_sup * 1.002), 2)
            targets = [t1, t2]
            option_action = "Buy ATM or 1-strike OTM puts on rally failure. Δ −0.40 to −0.55."
            kill_switch = f"Posture flips if price reclaims VWAP ({_fp(vwap)}) with bullish volume."

        elif state == PostureState.BREAKOUT_WATCH:
            key_level = nearest_res if prox_score > 0 else nearest_sup
            if key_level == 0:
                key_level = p
            direction_hint = "above" if prox_score > 0 else "below"
            ez_low  = round(key_level * 0.9993, 2)
            ez_high = round(key_level * 1.0007, 2)
            stop    = round(key_level - atr_safe * 0.4, 2) if prox_score > 0 else round(key_level + atr_safe * 0.4, 2)
            t1 = round(key_level + atr_safe * 1.2, 2) if prox_score > 0 else round(key_level - atr_safe * 1.2, 2)
            t2 = round(key_level + atr_safe * 2.5, 2) if prox_score > 0 else round(key_level - atr_safe * 2.5, 2)
            targets = [t1, t2]
            option_action = f"Wait for a 1-bar close {direction_hint} {_fp(key_level)}, then buy ATM calls." if prox_score > 0 else f"Wait for confirmed break below {_fp(key_level)}, then buy ATM puts."
            kill_switch = f"No entry until clean close {direction_hint} {_fp(key_level)}. False breakout → fade it."

        elif state == PostureState.WAIT_EXHAUSTION:
            dir_w = "above" if sigma > 0 else "below"
            revert_target = round(vwap, 2) if vwap > 0 else round(p, 2)
            stop    = round(p + atr_safe * 0.5, 2) if sigma > 0 else round(p - atr_safe * 0.5, 2)
            targets = [revert_target]
            option_action = "Scalp only — 0.25× normal size. No new trend trades."
            kill_switch = f"Price is {'>' if sigma>0 else '<'} {abs(sigma):.1f}σ from VWAP. If it reverts to VWAP ({_fp(vwap)}), posture resets."

        elif state == PostureState.MEAN_REVERSION:
            half_range = atr_safe * 1.5
            ez_low  = round(p - half_range * 0.2, 2)
            ez_high = round(p + half_range * 0.2, 2)
            stop    = round(nearest_sup - atr_safe * 0.3, 2) if nearest_sup > 0 else round(p - atr_safe * 0.7, 2)
            revert  = round(vwap, 2) if vwap > 0 else round(p, 2)
            targets = [revert, round(revert + atr_safe * 0.5, 2)]
            option_action = "Sell premium (spreads/IC) or buy near-expiry ATM straddles if IV is low."
            kill_switch = f"Range breaks directionally if price closes beyond nearest S/R with volume."

        else:  # STAND_ASIDE
            kill_switch = f"Volume is {vol_ratio:.1f}× average. Historically 23.5% win rate. Sit out or trade 0.25× max."
            option_action = "Paper trade or 0.25× size maximum."

        # TRANSITION and VOL_EXPANSION – extra level logic
        if state == PostureState.TRANSITION:
            kill_switch = "Regime confidence too low. Wait until regime confidence rises above 50%."
            option_action = "No new positions. Re-evaluate when regime clarifies."

        elif state == PostureState.VOL_EXPANSION:
            dir_hint = "bullish" if prox_score >= 0 else "bearish"
            key_level = nearest_res if prox_score >= 0 else nearest_sup
            ez_low  = round(key_level * 0.9993, 2) if key_level else None
            ez_high = round(key_level * 1.0007, 2) if key_level else None
            stop    = round(key_level - atr_safe * 0.5, 2) if prox_score >= 0 else round(key_level + atr_safe * 0.5, 2)
            t1      = round(underlying_price + atr_safe * 1.5, 2) if prox_score >= 0 else round(underlying_price - atr_safe * 1.5, 2)
            t2      = round(underlying_price + atr_safe * 3.0, 2) if prox_score >= 0 else round(underlying_price - atr_safe * 3.0, 2)
            targets = [t1, t2]
            option_action = f"Breakout confirmed — enter {'calls' if prox_score>=0 else 'puts'} on first pullback to breakout level."
            kill_switch = f"If price falls back below breakout level ({_fp(key_level)}), exit immediately."

        # ── Build headline & bullets ──────────────────────────
        headline, bullets, best_play, stats_note = self._build_narrative(
            state=state,
            regime_name=regime_name,
            vwap_pos=vwap_pos,
            sigma=sigma,
            rsi=rsi,
            vwap=vwap,
            atr=atr_safe,
            underlying_price=underlying_price,
            ez_low=ez_low,
            ez_high=ez_high,
            gex_regime=gex_regime,
            vol_regime_name=vol_regime_name,
            vol_ratio=vol_ratio,
            nearest_sup=nearest_sup,
            nearest_res=nearest_res,
            targets=targets,
            max_pain=max_pain,
            option_action=option_action,
            regime_conf=regime_conf,
        )

        # ── Confidence ────────────────────────────────────────
        confidence = self._calc_confidence(
            state=state,
            regime_conf=regime_conf,
            sigma=sigma,
            rsi=rsi,
            vol_regime_name=vol_regime_name,
            vpa_dir=vpa_dir,
        )

        # ── Capital Mode ───────────────────────────────────────
        from app.risk_engine import risk_engine
        cap = risk_engine.get_capital_mode(
            regime_name=regime_name,
            vol_regime=vol_regime_name,
        )

        return TradePosture(
            state=state,
            state_label=meta["label"],
            icon=meta["icon"],
            color=meta["color"],
            direction=meta["direction"],
            headline=headline,
            best_play=best_play,
            best_play_bullets=bullets,
            allowed_setups=meta["allowed"],
            avoid=meta["avoid"],
            entry_zone_low=ez_low,
            entry_zone_high=ez_high,
            stop_level=stop,
            targets=targets,
            option_action=option_action,
            kill_switch=kill_switch,
            confidence=round(confidence, 2),
            stats_note=stats_note,
            capital_mode=cap["mode"],
            capital_mode_reason=cap["reason"],
            max_size_mult=cap["max_size_mult"],
            locked_until=cap["locked_until"],
        )

    # ── State classifier ─────────────────────────────────────
    def _classify_state(
        self, *, regime_name, vwap_pos, sigma, rsi,
        vol_regime_name, gex_regime, prox_score, vpa_dir,
        regime_conf=0.5, vol_ratio=1.0, impulse_count=0,
        atr_compression=1.0, res_dist_pct=999.0,
    ) -> str:
        r = regime_name.upper() if regime_name else ""

        # 0. Low confidence → TRANSITION (sit out while regime is unclear)
        if regime_conf < 0.50:
            return PostureState.TRANSITION

        # 1. High-volume risk day → stand aside
        if vol_regime_name == "HIGH_RISK":
            return PostureState.STAND_ASIDE

        # 1b. VOL_EXPANSION: breakout candle > 1.8× avg volume OR confirmed level break
        if vol_ratio > 1.8 and (res_dist_pct < 0.002 or "BREAKOUT" in r):
            return PostureState.VOL_EXPANSION

        # 2. Reversal exhaustion → wait (add stricter check with σ + RSI + impulse)
        if "REVERSAL" in r:
            return PostureState.WAIT_EXHAUSTION

        # 2b. Exhaustion check: σ > 2, RSI extreme, 3+ impulse candles
        if abs(sigma) > 2.0 and (rsi > 75 or rsi < 25) and impulse_count >= 3:
            return PostureState.WAIT_EXHAUSTION

        # 3. Explicit breakout regime + compression + near resistance
        if "BREAKOUT" in r:
            if atr_compression < 0.7 and res_dist_pct < 0.002:
                return PostureState.VOL_EXPANSION  # compressed and AT the level
            return PostureState.BREAKOUT_WATCH

        # 3b. Compression + near resistance even without BREAKOUT regime
        if atr_compression < 0.7 and res_dist_pct < 0.002:
            return PostureState.BREAKOUT_WATCH

        # 4. Range/chop → mean reversion
        if "RANGE" in r or "CHOP" in r:
            return PostureState.MEAN_REVERSION

        # 5-6. Trend up
        if "UP" in r or "TREND_UP" in r:
            if sigma >= self.EXT_STOP or rsi >= 72:
                return PostureState.WAIT_EXHAUSTION
            if sigma >= self.EXT_WARN or rsi >= 68:
                # approaching exhaustion but GEX positive → still buy pullbacks, flag as weak
                if gex_regime == "positive":
                    return PostureState.BUY_PULLBACKS
                return PostureState.WAIT_EXHAUSTION
            # Tightened: require extension < 1.5σ and volume not LOW
            if vol_regime_name == "LOW":
                return PostureState.MEAN_REVERSION
            return PostureState.BUY_PULLBACKS

        # 7-8. Trend down
        if "DOWN" in r or "TREND_DOWN" in r:
            if sigma <= -self.EXT_STOP or rsi <= 28:
                return PostureState.WAIT_EXHAUSTION
            return PostureState.SHORT_RALLIES

        # 9. VPA override — strong directional bias even without regime match
        if "strong_bull" in vpa_dir or "bull" in vpa_dir:
            return PostureState.BUY_PULLBACKS
        if "strong_bear" in vpa_dir or "bear" in vpa_dir:
            return PostureState.SHORT_RALLIES

        return PostureState.MEAN_REVERSION

    # ── Narrative builder ─────────────────────────────────────
    def _build_narrative(
        self, *, state, regime_name, vwap_pos, sigma, rsi, vwap, atr,
        underlying_price, ez_low, ez_high, gex_regime, vol_regime_name,
        vol_ratio, nearest_sup, nearest_res, targets, max_pain, option_action,
        regime_conf=0.5,
    ):
        p = underlying_price
        r = regime_name.replace("_", " ").title() if regime_name else "Unknown"
        sigma_str = f"{abs(sigma):.1f}σ {'above' if sigma>0 else 'below'} VWAP"
        vwap_str  = _fp(vwap) if vwap > 0 else "N/A"
        stats_note = ""

        if state == PostureState.BUY_PULLBACKS:
            headline  = f"Trend Up — Buy Dips to VWAP Zone"
            best_play = (
                f"Regime is {r}. Price is {sigma_str} ({_fp(p)}). "
                f"GEX is {gex_regime or 'neutral'} — market makers {_gex_hint(gex_regime)}. "
                f"RSI {rsi:.0f} — {'momentum healthy' if rsi<65 else 'watch for fade at new highs'}. "
                f"Wait for any dip toward {_fp(ez_low)}–{_fp(ez_high)} (VWAP zone)."
            )
            bullets = [
                f"➡ Wait for pullback into {_fp(ez_low)}–{_fp(ez_high)} (VWAP zone)",
                f"➡ {option_action}",
                f"➡ Targets: {', '.join(_fp(t) for t in targets) if targets else 'ATH re-test'}",
                f"➡ Stop below entry zone if price reverses with volume",
            ]

        elif state == PostureState.SHORT_RALLIES:
            headline  = f"Trend Down — Sell Rips to VWAP"
            best_play = (
                f"Regime is {r}. Price is {sigma_str} ({_fp(p)}). "
                f"GEX is {gex_regime or 'neutral'}. RSI {rsi:.0f}. "
                f"Rallies toward {_fp(ez_low)}–{_fp(ez_high)} are sell opportunities. "
                f"Do not buy dips — the path of least resistance is lower."
            )
            bullets = [
                f"➡ Wait for dead-cat rally toward {_fp(ez_low)}–{_fp(ez_high)}",
                f"➡ {option_action}",
                f"➡ Targets: {', '.join(_fp(t) for t in targets) if targets else 'new low sweep'}",
                f"➡ Stop above VWAP ({vwap_str}) if reclaimed with volume",
            ]

        elif state == PostureState.BREAKOUT_WATCH:
            key = _fp(nearest_res) if nearest_res > p else _fp(nearest_sup)
            dir_hint = "above" if nearest_res > p else "below"
            headline  = f"Breakout Watch — Key Level {key}"
            best_play = (
                f"Price is approaching key level {key}. "
                f"Regime: {r}. GEX: {gex_regime or 'neutral'}. "
                f"Wait for a confirmed close {dir_hint} {key} before entering. "
                f"False breaks are common — let price prove itself."
            )
            bullets = [
                f"➡ Watch for a 1-bar close {dir_hint} {key} on above-average volume",
                f"➡ {option_action}",
                f"➡ Targets: {', '.join(_fp(t) for t in targets) if targets else '1–2× ATR extension'}",
                f"➡ No entry on the approach — entry on confirmation only",
            ]

        elif state == PostureState.WAIT_EXHAUSTION:
            dir_w = "above" if sigma >= 0 else "below"
            headline  = f"Exhaustion — Pause, No New Trend Trades"
            best_play = (
                f"Price is {sigma_str} from VWAP ({vwap_str}). RSI {rsi:.0f}. "
                f"This is a statistically unfavorable entry zone for trend continuation. "
                f"Wait for reversion toward VWAP before re-entering. "
                f"If you must trade, scalp only with 0.25× normal size."
            )
            bullets = [
                f"➡ No new trend entries — price {sigma_str}",
                f"➡ RSI {rsi:.0f} — {'overbought territory' if rsi>70 else 'oversold territory' if rsi<30 else 'mid-range'}",
                f"➡ Wait for price to mean-revert toward VWAP ({vwap_str})",
                f"➡ Scalp only if conviction is very high — 0.25× size",
            ]
            stats_note = "Extended moves from VWAP have historically lower continuation probability."

        elif state == PostureState.MEAN_REVERSION:
            headline  = f"Range Bound — Fade Extremes, Harvest Premium"
            best_play = (
                f"Regime is {r}. Price ({_fp(p)}) is oscillating between "
                f"{_fp(nearest_sup)} support and {_fp(nearest_res)} resistance. "
                f"Do not chase breakouts. "
                f"Fade moves to the extremes and target VWAP ({vwap_str}) as mean."
            )
            bullets = [
                f"➡ Sell strength near {_fp(nearest_res)}, buy weakness near {_fp(nearest_sup)}",
                f"➡ VWAP ({vwap_str}) is the mean — target it on both sides",
                f"➡ {option_action}",
                f"➡ Tight stops — any directional break ends the range",
            ]

        else:  # STAND_ASIDE
            headline  = f"Stand Aside — High-Volume Risk Day"
            best_play = (
                f"Volume is {vol_ratio:.1f}× average. "
                f"High-volume days historically produce 23.5% win rate on directional trades. "
                f"Institutional players dominate. The edge is to not play. "
                f"If forced, trade 0.25× normal size only."
            )
            bullets = [
                f"➡ Volume {vol_ratio:.1f}× average — institutional activity dominant",
                f"➡ Historical win rate: ~23.5% on high-vol days",
                f"➡ Paper trade or 0.25× maximum",
                f"➡ Re-evaluate posture when volume normalizes",
            ]
            stats_note = "23.5% historical win rate on HIGH_RISK volume days."

        # Override headline/bullets for new states
        if state == PostureState.TRANSITION:
            headline  = f"⚠ Transition — Regime Unclear"
            best_play = (
                f"Regime confidence is only {int(regime_conf*100) if regime_conf else 0}%. "
                f"Market structure is ambiguous — no edge for directional bets. "
                f"Wait until the regime clarifies before committing capital."
            )
            bullets = [
                f"➡ Regime confidence {int(regime_conf*100) if regime_conf else 0}% — below 50% threshold",
                f"➡ No directional trades until regime stabilizes",
                f"➡ Watch VWAP ({vwap_str}) for direction clue",
                f"➡ Re-evaluate in 15–30 minutes",
            ]
            stats_note = "Sub-50% regime confidence → sit out and wait."

        elif state == PostureState.VOL_EXPANSION:
            dir_w = "up" if sigma >= 0 else "down"
            headline  = f"💥 Vol Expansion — Breakout Confirmed"
            best_play = (
                f"Volume spike ({vol_ratio:.1f}×) with price pushing through key level. "
                f"This is a confirmed expansion move. "
                f"Enter on the first retest of the breakout level, not on the initial spike."
            )
            bullets = [
                f"➡ Volume {vol_ratio:.1f}× average — breakout fuel confirmed",
                f"➡ {option_action}",
                f"➡ Targets: {', '.join(_fp(t) for t in targets) if targets else '1.5–3× ATR extension'}",
                f"➡ Do NOT fade this move — momentum is real",
            ]
            stats_note = "Volume-confirmed breakouts have higher follow-through probability."

        if max_pain > 0 and abs(p - max_pain) / p < 0.01:
            bullets.append(f"ℹ️  Near max pain ({_fp(max_pain)}) — expect magnetic pull, pin risk")

        return headline, bullets, best_play, stats_note

    # ── Confidence calculator ─────────────────────────────────
    def _calc_confidence(
        self, *, state, regime_conf, sigma, rsi, vol_regime_name, vpa_dir,
    ) -> float:
        base = regime_conf

        # TRANSITION = very low confidence by definition
        if state == PostureState.TRANSITION:
            return max(0.1, regime_conf * 0.6)

        # Reduce confidence if volume-adverse
        if vol_regime_name == "HIGH_RISK":
            base *= 0.5
        elif vol_regime_name == "LOW":
            base *= 0.8

        # VPA alignment boost
        if state == PostureState.BUY_PULLBACKS and ("bull" in vpa_dir):
            base = min(base + 0.1, 1.0)
        if state == PostureState.SHORT_RALLIES and ("bear" in vpa_dir):
            base = min(base + 0.1, 1.0)
        if state == PostureState.VOL_EXPANSION:
            base = min(base + 0.15, 1.0)  # breakout + volume = higher confidence

        # Extension penalty
        if abs(sigma) > 2.0:
            base *= 0.7
        elif abs(sigma) > 1.5:
            base *= 0.85

        # RSI extreme penalty for trend states
        if state in (PostureState.BUY_PULLBACKS, PostureState.SHORT_RALLIES):
            if rsi > 72 or rsi < 28:
                base *= 0.75

        return max(0.1, min(0.99, base))


# ── Helpers ───────────────────────────────────────────────────
def _fp(v) -> str:
    try:
        return f"{float(v):.2f}"
    except Exception:
        return "—"


def _gex_hint(gex: str) -> str:
    if gex == "positive":
        return "suppress volatility (pin / range likely)"
    if gex == "negative":
        return "amplify moves (fuel for breakout)"
    return "neutral — no strong vol bias"


# ── Singleton ─────────────────────────────────────────────────
posture_engine = PostureEngine()
