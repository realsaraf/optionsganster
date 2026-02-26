"""
Decision Engine — 0DTE Trade Decision Machine
===============================================
Reduces ALL market signals to a single 3-choice output:
  ✅ BUY CALLS
  ✅ BUY PUTS
  ⏸  WAIT

Deterministic scoring. No vibes. No AI magic.

Six scoring layers (user-specified weights):
  1. Structure   (max 30)  — Regime, VWAP position, VWAP slope
  2. Momentum    (max 20)  — HH/LL, breakout, volume regime
  3. Extension   (max 15)  — Sigma from VWAP, RSI extremes (can go negative)
  4. Positioning (max 15)  — GEX, S/R proximity, max pain
  5. Options     (max 10)  — Spread, IV rank, delta
  6. Time of Day (max 10)  — Session-aware adjustments

Decision rule:
  max(call, put) < 60        → WAIT
  callScore - putScore >= 12  → BUY CALLS
  putScore - callScore >= 12  → BUY PUTS
  else                        → WAIT (unclear edge)

Hard guards (override to WAIT):
  • Volume LOW + RANGE_CHOP
  • Extension > 2.5σ from VWAP
  • Capital mode LOCKED
  • Spread > 15%
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

import pandas as pd


# ── Decision enum ────────────────────────────────────────────
class Decision(str, Enum):
    BUY_CALLS = "BUY_CALLS"
    BUY_PUTS  = "BUY_PUTS"
    WAIT      = "WAIT"


# ── Sub-scores ───────────────────────────────────────────────
@dataclass
class SubScores:
    structure: int = 0
    momentum: int = 0
    extension: int = 0
    positioning: int = 0
    option_quality: int = 0
    time_adj: int = 0


# ── Output dataclass ─────────────────────────────────────────
@dataclass
class TradeDecision:
    decision: str                            # BUY_CALLS / BUY_PUTS / WAIT
    call_score: int = 0                      # 0–100
    put_score: int = 0                       # 0–100
    wait_reason: str = ""                    # why we're waiting (if WAIT)
    trigger: str = ""                        # what must happen to enter
    invalidation: str = ""                   # what cancels it
    because: list[str] = field(default_factory=list)  # max 3 bullets
    entry_zone_low: Optional[float] = None
    entry_zone_high: Optional[float] = None
    stop_level: Optional[float] = None
    targets: list[float] = field(default_factory=list)
    confidence: float = 0.0                  # 0–1
    hard_guard_active: bool = False
    hard_guard_reason: str = ""

    # Capital mode (from RiskEngine)
    capital_mode: str = "NORMAL"
    capital_mode_reason: str = ""
    max_size_mult: float = 1.0
    locked_until: Optional[str] = None

    # Sub-scores for transparency
    call_sub: SubScores = field(default_factory=SubScores)
    put_sub: SubScores = field(default_factory=SubScores)

    # ── Action Engine fields ──────────────────────────────────
    wait_until: list[dict] = field(default_factory=list)
    # [{"direction": "above"/"below", "level": float, "label": str, "action": "BUY_CALLS"/"BUY_PUTS"}]

    execution_mode: str = "STAND_ASIDE"   # SCALP_ONLY / BREAKOUT / FULL / STAND_ASIDE
    execution_mode_reason: str = ""

    wait_confidence: str = "Moderate"     # Strong / Moderate / Weak

    next_move_map: list[dict] = field(default_factory=list)
    # [{"scenario": str, "outcome": str}]


# ── Engine ────────────────────────────────────────────────────
class DecisionEngine:
    """
    Deterministic call/put/wait scoring engine for 0DTE.
    Pure computation — no I/O, no state.
    """

    def compute(
        self,
        *,
        regime,            # RegimeResult or None
        sr,                # SRResult or None
        chain_metrics,     # ChainMetrics or None
        vpa_bias: dict,    # {bias, strength, reason}
        vol_regime: dict,  # {regime, ratio, detail}
        underlying_price: float,
        df=None,           # intraday DataFrame
        fv_verdict: str = "",          # "CHEAP" / "EXPENSIVE" / "FAIR"
        spread_pct: float = 0.0,      # nearest ATM spread %
        delta: float = 0.0,           # nearest ATM delta
        iv_rank: float = 50.0,        # 0–100
    ) -> TradeDecision:

        # ── Extract fields ────────────────────────────────────
        regime_name = ""
        vwap = 0.0
        atr = 0.0
        rsi = 50.0
        vwap_pos = "at"
        regime_conf = 0.5
        ema_9 = 0.0
        ema_20 = 0.0

        if regime:
            rn = getattr(regime, "regime", "")
            regime_name = rn.value if hasattr(rn, "value") else str(rn)
            vwap = float(getattr(regime, "vwap_current", 0) or 0)
            atr = float(getattr(regime, "atr_current", 0) or 0)
            rsi = float(getattr(regime, "rsi_current", 50) or 50)
            vwap_pos = getattr(regime, "price_vs_vwap", "at") or "at"
            regime_conf = float(getattr(regime, "confidence", 0.5) or 0.5)
            ema_9 = float(getattr(regime, "ema_9", 0) or 0)
            ema_20 = float(getattr(regime, "ema_20", 0) or 0)

        vol_regime_name = (vol_regime or {}).get("regime", "NORMAL")
        vol_ratio = float((vol_regime or {}).get("ratio", 1.0) or 1.0)

        gex_regime = ""
        max_pain = 0.0
        if chain_metrics:
            gex_regime = (getattr(chain_metrics, "gex_regime", "") or "").lower()
            max_pain = float(getattr(chain_metrics, "max_pain", 0) or 0)
            if iv_rank == 50.0:  # default → pull from chain
                iv_rank = float(getattr(chain_metrics, "iv_rank", 50) or 50)

        nearest_sup = 0.0
        nearest_res = 0.0
        if sr:
            nearest_sup = float(getattr(sr, "nearest_support", 0) or 0)
            nearest_res = float(getattr(sr, "nearest_resistance", 0) or 0)
            if not nearest_sup and hasattr(sr, "levels") and sr.levels:
                sups = [l.price for l in sr.levels if l.kind == "support" and l.price < underlying_price]
                nearest_sup = max(sups) if sups else 0.0
            if not nearest_res and hasattr(sr, "levels") and sr.levels:
                ress = [l.price for l in sr.levels if l.kind == "resistance" and l.price > underlying_price]
                nearest_res = min(ress) if ress else 0.0

        p = underlying_price
        atr_safe = atr if atr > 0 else p * 0.001

        # Fallback S/R from VWAP ± ATR if still not found
        if not nearest_sup and vwap > 0 and atr_safe > 0:
            nearest_sup = round(vwap - atr_safe * 1.5, 2)
        if not nearest_res and vwap > 0 and atr_safe > 0:
            nearest_res = round(vwap + atr_safe * 1.5, 2)

        # σ from VWAP in ATR units
        sigma = 0.0
        if vwap > 0 and atr_safe > 0:
            sigma = (p - vwap) / atr_safe

        # VWAP slope (from detail string or estimate)
        vwap_slope = self._estimate_vwap_slope(regime)

        # HH/HL detection from bars
        hh_count, ll_count = self._count_hh_ll(df)

        # Break above resistance / below support with volume
        break_above = self._detect_break(df, nearest_res, "above", vol_ratio)
        break_below = self._detect_break(df, nearest_sup, "below", vol_ratio)

        # ── SCORING ───────────────────────────────────────────

        # 1. STRUCTURE (max 30)
        cs_struct, ps_struct = self._score_structure(regime_name, vwap_pos, vwap_slope)

        # 2. MOMENTUM (max 20)
        cs_mom, ps_mom = self._score_momentum(
            hh_count, ll_count, break_above, break_below,
            vol_regime_name, vol_ratio,
        )

        # 3. EXTENSION (max 15 / can go negative)
        cs_ext, ps_ext = self._score_extension(sigma, rsi)

        # 4. POSITIONING (max 15)
        cs_pos, ps_pos = self._score_positioning(
            gex_regime, vwap_pos, nearest_sup, nearest_res, p, max_pain,
        )

        # 5. OPTION QUALITY (max 10)
        cs_opt, ps_opt = self._score_option_quality(spread_pct, iv_rank, delta, fv_verdict)

        # 6. TIME OF DAY (max 10)
        cs_time, ps_time = self._score_time_of_day(regime_name)

        # ── TOTALS ────────────────────────────────────────────
        call_score = cs_struct + cs_mom + cs_ext + cs_pos + cs_opt + cs_time
        put_score = ps_struct + ps_mom + ps_ext + ps_pos + ps_opt + ps_time

        # Clamp 0–100
        call_score = max(0, min(100, call_score))
        put_score = max(0, min(100, put_score))

        call_sub = SubScores(cs_struct, cs_mom, cs_ext, cs_pos, cs_opt, cs_time)
        put_sub = SubScores(ps_struct, ps_mom, ps_ext, ps_pos, ps_opt, ps_time)

        # ── HARD GUARDS ──────────────────────────────────────
        hard_guard = False
        guard_reason = ""

        from app.risk_engine import risk_engine
        cap = risk_engine.get_capital_mode(
            regime_name=regime_name, vol_regime=vol_regime_name,
        )

        if cap["mode"] == "LOCKED":
            hard_guard = True
            guard_reason = f"Capital LOCKED — {cap['reason']}"
        elif vol_regime_name == "LOW" and "RANGE" in regime_name.upper():
            hard_guard = True
            guard_reason = "Low volume + Range Chop — no directional edge"
        elif abs(sigma) > 2.5:
            hard_guard = True
            guard_reason = f"Price extended {abs(sigma):.1f}σ from VWAP — chasing risk"
        elif spread_pct > 15:
            hard_guard = True
            guard_reason = f"Spread {spread_pct:.0f}% — too wide for 0DTE"

        # ── DECISION RULE ─────────────────────────────────────
        if hard_guard:
            decision = Decision.WAIT
            wait_reason = guard_reason
        elif max(call_score, put_score) < 60:
            decision = Decision.WAIT
            wait_reason = "Scores below threshold (60) — no clear edge"
        elif call_score - put_score >= 12:
            decision = Decision.BUY_CALLS
            wait_reason = ""
        elif put_score - call_score >= 12:
            decision = Decision.BUY_PUTS
            wait_reason = ""
        else:
            decision = Decision.WAIT
            wait_reason = f"Scores too close ({call_score} vs {put_score}) — unclear edge"

        # ── ENTRY / STOP / TARGETS ────────────────────────────
        ez_low, ez_high, stop, targets = self._compute_levels(
            decision, p, vwap, atr_safe, sigma,
            nearest_sup, nearest_res,
        )

        # ── TRIGGER / INVALIDATION / BECAUSE ──────────────────
        trigger = self._generate_trigger(decision, regime_name, vwap, vwap_pos, p, nearest_sup, nearest_res)
        invalidation = self._generate_invalidation(decision, vwap, p, nearest_sup, nearest_res)
        because = self._generate_because(
            call_score, put_score, call_sub, put_sub,
            regime_name, vwap_pos, vwap_slope, rsi, vol_regime_name,
            gex_regime, iv_rank, fv_verdict, sigma, vpa_bias,
        )

        # ── CONFIDENCE ────────────────────────────────────────
        top = max(call_score, put_score)
        gap = abs(call_score - put_score)
        base_conf = top / 100.0
        if gap < 12:
            base_conf *= 0.6
        if hard_guard:
            base_conf *= 0.3
        if regime_conf < 0.5:
            base_conf *= 0.5
        confidence = max(0.05, min(0.99, base_conf))

        # ── WAIT UNTIL (conditional triggers) ─────────────────
        wait_until = self._generate_wait_until(
            decision, regime_name, vwap, vwap_pos, p,
            nearest_sup, nearest_res, atr_safe,
        )

        # ── EXECUTION MODE ────────────────────────────────────
        exec_mode, exec_reason = self._determine_execution_mode(
            decision, regime_name, vol_regime_name, hard_guard,
            sigma, cap["mode"],
        )

        # ── WAIT CONFIDENCE ───────────────────────────────────
        wait_conf = self._classify_wait_confidence(
            decision, hard_guard, call_score, put_score,
            regime_conf, vol_regime_name,
        )

        # ── NEXT MOVE MAP ─────────────────────────────────────
        next_moves = self._generate_next_move_map(
            decision, regime_name, vwap, p,
            nearest_sup, nearest_res, atr_safe, max_pain,
        )

        return TradeDecision(
            decision=decision.value,
            call_score=call_score,
            put_score=put_score,
            wait_reason=wait_reason,
            trigger=trigger,
            invalidation=invalidation,
            because=because,
            entry_zone_low=ez_low,
            entry_zone_high=ez_high,
            stop_level=stop,
            targets=targets,
            confidence=round(confidence, 2),
            hard_guard_active=hard_guard,
            hard_guard_reason=guard_reason,
            capital_mode=cap["mode"],
            capital_mode_reason=cap["reason"],
            max_size_mult=cap["max_size_mult"],
            locked_until=cap["locked_until"],
            call_sub=call_sub,
            put_sub=put_sub,
            wait_until=wait_until,
            execution_mode=exec_mode,
            execution_mode_reason=exec_reason,
            wait_confidence=wait_conf,
            next_move_map=next_moves,
        )

    # ══════════════════════════════════════════════════════════
    # SCORING METHODS
    # ══════════════════════════════════════════════════════════

    def _score_structure(self, regime_name: str, vwap_pos: str, vwap_slope: float) -> tuple[int, int]:
        """Part 1 — Structure Score (max 30 each)."""
        cs = ps = 0
        r = regime_name.upper() if regime_name else ""

        # Regime
        if "TREND_UP" in r or r == "TREND_UP":
            cs += 12
        elif "TREND_DOWN" in r or r == "TREND_DOWN":
            ps += 12
        elif "RANGE" in r or "CHOP" in r:
            cs += 3; ps += 3
        elif "BREAKOUT" in r:
            cs += 6; ps += 6  # directional but unknown direction

        # Price vs VWAP
        if vwap_pos == "above":
            cs += 6
        elif vwap_pos == "below":
            ps += 6

        # VWAP slope
        if vwap_slope > 0.005:
            cs += 6
        elif vwap_slope < -0.005:
            ps += 6

        return cs, ps

    def _score_momentum(
        self, hh_count: int, ll_count: int,
        break_above: bool, break_below: bool,
        vol_regime_name: str, vol_ratio: float,
    ) -> tuple[int, int]:
        """Part 2 — Momentum / Flow (max 20 each)."""
        cs = ps = 0

        # Consecutive HH / LL
        if hh_count >= 2:
            cs += 6
        if ll_count >= 2:
            ps += 6

        # Break with volume
        if break_above:
            cs += 8
        if break_below:
            ps += 8

        # Volume regime
        if vol_regime_name == "HIGH_RISK":
            # HIGH volume is directional fuel — add to winning side
            if cs > ps:
                cs += 6
            elif ps > cs:
                ps += 6
            else:
                cs += 3; ps += 3
        elif vol_regime_name == "LOW":
            cs -= 6; ps -= 6

        return cs, ps

    def _score_extension(self, sigma: float, rsi: float) -> tuple[int, int]:
        """Part 3 — Extension Filter (max 15 / can go negative)."""
        cs = ps = 0

        # Price > 2σ above VWAP: punish call chasing, small put bonus
        if sigma > 2.0:
            cs -= 10; ps += 4
        elif sigma < -2.0:
            ps -= 10; cs += 4

        # RSI extremes
        if rsi > 75:
            cs -= 5; ps += 3
        elif rsi < 25:
            ps -= 5; cs += 3

        return cs, ps

    def _score_positioning(
        self, gex_regime: str, vwap_pos: str,
        nearest_sup: float, nearest_res: float,
        price: float, max_pain: float,
    ) -> tuple[int, int]:
        """Part 4 — Positioning Edge (max 15 each)."""
        cs = ps = 0

        # GEX + VWAP position
        if gex_regime == "positive":
            # Positive GEX: dealers hedge to support current direction
            if vwap_pos == "above":
                cs += 5
            elif vwap_pos == "below":
                ps += 5
        elif gex_regime == "negative":
            # Negative GEX: dealers amplify moves — directional fuel
            if vwap_pos == "above":
                cs += 5
            elif vwap_pos == "below":
                ps += 5

        # Near support → calls; near resistance → puts
        if nearest_sup > 0 and price > 0:
            sup_dist = (price - nearest_sup) / price
            if sup_dist < 0.003:  # within 0.3%
                cs += 6
        if nearest_res > 0 and price > 0:
            res_dist = (nearest_res - price) / price
            if res_dist < 0.003:
                ps += 6

        # Max pain magnet
        if max_pain > 0 and price > 0:
            if max_pain < price:
                ps += 4  # max pain below → gravity pulls down
            elif max_pain > price:
                cs += 4  # max pain above → gravity pulls up

        return cs, ps

    def _score_option_quality(
        self, spread_pct: float, iv_rank: float,
        delta: float, fv_verdict: str,
    ) -> tuple[int, int]:
        """Part 5 — Option Quality (max 10 each)."""
        cs = ps = 0
        bonus = 0

        # Spread
        if 0 < spread_pct < 6:
            bonus += 3
        elif spread_pct > 12:
            bonus -= 5

        # IV Rank
        if iv_rank < 20:
            bonus += 3
        elif fv_verdict == "EXPENSIVE":
            bonus -= 3

        # Delta
        abs_delta = abs(delta)
        if 0.40 <= abs_delta <= 0.60:
            bonus += 4
        elif 0 < abs_delta < 0.25:
            bonus -= 4

        cs += bonus; ps += bonus
        return cs, ps

    def _score_time_of_day(self, regime_name: str) -> tuple[int, int]:
        """Part 6 — Time of Day (max 10 each)."""
        cs = ps = 0
        now = datetime.utcnow()
        # Convert to ET (UTC-5 during EST, UTC-4 during EDT)
        # Approximate: market hours 14:30–21:00 UTC (9:30–16:00 ET)
        utc_h = now.hour + now.minute / 60.0

        # 9:30–10:30 ET = 14:30–15:30 UTC
        if 14.5 <= utc_h <= 15.5:
            cs += 5; ps += 5  # momentum allowed in open
        # 10:30–14:00 ET = 15:30–19:00 UTC
        elif 15.5 < utc_h <= 19.0:
            pass  # neutral
        # 14:00–15:30 ET = 19:00–20:30 UTC
        elif 19.0 < utc_h <= 20.5:
            r = regime_name.upper() if regime_name else ""
            if "BREAKOUT" not in r:
                cs -= 3; ps -= 3  # late day penalty without breakout
        # After 15:30 ET = 20:30+ UTC — very late, high bar
        elif utc_h > 20.5:
            cs -= 5; ps -= 5  # strong penalty for very late 0DTE

        return cs, ps

    # ══════════════════════════════════════════════════════════
    # HELPER METHODS
    # ══════════════════════════════════════════════════════════

    def _estimate_vwap_slope(self, regime) -> float:
        """Extract VWAP slope from regime detail string or return 0."""
        if not regime:
            return 0.0
        detail = getattr(regime, "detail", "") or ""
        # Format: "... slope=+0.0015 ..."
        import re
        m = re.search(r"slope=([+-]?[\d.]+)", detail)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                pass
        return 0.0

    def _count_hh_ll(self, df) -> tuple[int, int]:
        """Count consecutive higher-highs and lower-lows from recent bars."""
        if df is None or df.empty or len(df) < 3:
            return 0, 0
        try:
            highs = df["high"].values
            lows = df["low"].values
            n = len(highs)
            hh = 0
            for i in range(n - 1, max(n - 6, 0), -1):
                if highs[i] > highs[i - 1]:
                    hh += 1
                else:
                    break
            ll = 0
            for i in range(n - 1, max(n - 6, 0), -1):
                if lows[i] < lows[i - 1]:
                    ll += 1
                else:
                    break
            return hh, ll
        except Exception:
            return 0, 0

    def _detect_break(self, df, level: float, direction: str, vol_ratio: float) -> bool:
        """Detect if price broke above/below a level with volume."""
        if df is None or df.empty or level <= 0 or vol_ratio < 1.2:
            return False
        try:
            last = df.iloc[-1]
            prev = df.iloc[-2] if len(df) >= 2 else last
            if direction == "above":
                return float(last["close"]) > level and float(prev["close"]) <= level
            else:
                return float(last["close"]) < level and float(prev["close"]) >= level
        except Exception:
            return False

    def _compute_levels(
        self, decision: Decision, p: float, vwap: float, atr: float, sigma: float,
        nearest_sup: float, nearest_res: float,
    ) -> tuple[Optional[float], Optional[float], Optional[float], list[float]]:
        """Compute entry zone, stop, targets based on decision."""
        if decision == Decision.WAIT:
            return None, None, None, []

        if decision == Decision.BUY_CALLS:
            # Entry: pullback toward VWAP or support
            ez_high = round(vwap + atr * 0.2, 2) if vwap > 0 else round(p, 2)
            ez_low = round(vwap - atr * 0.3, 2) if vwap > 0 else round(p * 0.998, 2)
            if nearest_sup > 0 and nearest_sup > ez_low:
                ez_low = round(nearest_sup, 2)
            stop = round(ez_low - atr * 0.5, 2)
            t1 = round(p + atr * 1.0, 2)
            t2 = round(p + atr * 2.0, 2)
            if nearest_res > p:
                t1 = round(min(t1, nearest_res * 0.998), 2)
            return ez_low, ez_high, stop, [t1, t2]

        else:  # BUY_PUTS
            ez_low = round(vwap - atr * 0.2, 2) if vwap > 0 else round(p, 2)
            ez_high = round(vwap + atr * 0.3, 2) if vwap > 0 else round(p * 1.002, 2)
            if nearest_res > 0 and nearest_res < ez_high:
                ez_high = round(nearest_res, 2)
            stop = round(ez_high + atr * 0.5, 2)
            t1 = round(p - atr * 1.0, 2)
            t2 = round(p - atr * 2.0, 2)
            if nearest_sup > 0:
                t1 = round(max(t1, nearest_sup * 1.002), 2)
            return ez_low, ez_high, stop, [t1, t2]

    def _generate_trigger(
        self, decision: Decision, regime_name: str,
        vwap: float, vwap_pos: str, p: float,
        nearest_sup: float, nearest_res: float,
    ) -> str:
        r = regime_name.upper() if regime_name else ""
        vf = _fp(vwap) if vwap > 0 else "VWAP"

        if decision == Decision.BUY_CALLS:
            if "TREND_UP" in r:
                return f"Pullback to VWAP ({vf}) and bullish 1m close"
            if "BREAKOUT" in r:
                return f"1m/5m close above resistance {_fp(nearest_res)} + hold"
            return f"Price reclaims VWAP ({vf}) and holds 2×1m closes"

        if decision == Decision.BUY_PUTS:
            if "TREND_DOWN" in r:
                return f"Rally to VWAP ({vf}) and bearish 1m rejection"
            if "BREAKOUT" in r:
                return f"1m/5m close below support {_fp(nearest_sup)} + hold"
            return f"Price loses VWAP ({vf}) and fails 2×1m closes"

        # WAIT
        if "RANGE" in r or "CHOP" in r:
            return f"Only scalp at S/R extremes: {_fp(nearest_sup)} / {_fp(nearest_res)}"
        if "BREAKOUT" in r:
            return f"Wait for confirmed close above {_fp(nearest_res)} or below {_fp(nearest_sup)}"
        if abs((p - vwap) / (vwap or 1)) > 0.01 and vwap > 0:
            return f"Mean reversion to VWAP ({vf}) before new entries"
        return "Wait for regime to clarify and edge to develop"

    def _generate_invalidation(
        self, decision: Decision, vwap: float, p: float,
        nearest_sup: float, nearest_res: float,
    ) -> str:
        vf = _fp(vwap) if vwap > 0 else "VWAP"

        if decision == Decision.BUY_CALLS:
            return f"5m close below VWAP ({vf}) → calls invalid, reassess for puts"
        if decision == Decision.BUY_PUTS:
            return f"5m close above VWAP ({vf}) → puts invalid, reassess for calls"
        # WAIT
        if nearest_sup > 0 and nearest_res > 0:
            return f"Breakout above {_fp(nearest_res)} → calls | Break below {_fp(nearest_sup)} → puts"
        return "Watch for regime change or volume expansion"

    def _generate_because(
        self,
        call_score: int, put_score: int,
        call_sub: SubScores, put_sub: SubScores,
        regime_name: str, vwap_pos: str, vwap_slope: float,
        rsi: float, vol_regime_name: str,
        gex_regime: str, iv_rank: float, fv_verdict: str,
        sigma: float, vpa_bias: dict,
    ) -> list[str]:
        """Generate top 3 reason bullets, sorted by impact."""
        bullets = []

        # Structure reasons
        r = regime_name.replace("_", " ") if regime_name else "Unknown"
        if vwap_pos == "above":
            bullets.append((call_sub.structure, f"Above VWAP in {r} regime"))
        elif vwap_pos == "below":
            bullets.append((put_sub.structure, f"Below VWAP in {r} regime"))
        else:
            bullets.append((3, f"{r} regime — at VWAP"))

        if vwap_slope > 0.005:
            bullets.append((6, "VWAP slope trending up"))
        elif vwap_slope < -0.005:
            bullets.append((6, "VWAP slope trending down"))

        # Momentum
        if call_sub.momentum > 0 and call_sub.momentum > put_sub.momentum:
            bullets.append((call_sub.momentum, "Bullish momentum confirmed (HH pattern)"))
        elif put_sub.momentum > 0 and put_sub.momentum > call_sub.momentum:
            bullets.append((put_sub.momentum, "Bearish momentum confirmed (LL pattern)"))

        # Volume
        if vol_regime_name == "LOW":
            bullets.append((6, "Low volume — no conviction"))
        elif vol_regime_name == "HIGH_RISK":
            bullets.append((6, "High volume — strong directional fuel"))

        # Extension
        if abs(sigma) > 1.5:
            dir_str = "above" if sigma > 0 else "below"
            bullets.append((8, f"Extended {abs(sigma):.1f}σ {dir_str} VWAP — reversion risk"))

        # RSI
        if rsi > 70:
            bullets.append((5, f"RSI {rsi:.0f} — overbought"))
        elif rsi < 30:
            bullets.append((5, f"RSI {rsi:.0f} — oversold"))

        # Positioning
        if gex_regime == "positive":
            bullets.append((5, "GEX positive — dealers suppress moves"))
        elif gex_regime == "negative":
            bullets.append((5, "GEX negative — dealers amplify moves"))

        # Options quality
        if iv_rank < 20:
            bullets.append((3, f"IV Rank {iv_rank:.0f} — cheap premium"))
        elif fv_verdict == "EXPENSIVE":
            bullets.append((3, "Options marked EXPENSIVE"))

        # VPA
        vpa_dir = (vpa_bias or {}).get("bias", "neutral")
        _vpa_raw = (vpa_bias or {}).get("strength", 0)
        # strength may be string ("strong"/"moderate"/"weak") or float
        if isinstance(_vpa_raw, str):
            _str_map = {"strong": 0.9, "moderate": 0.6, "weak": 0.3}
            vpa_str = _str_map.get(_vpa_raw.lower(), 0.0)
        else:
            vpa_str = float(_vpa_raw or 0)
        if vpa_dir == "bullish" and vpa_str > 0.5:
            bullets.append((4, f"VPA bullish ({int(vpa_str*100)}% strength)"))
        elif vpa_dir == "bearish" and vpa_str > 0.5:
            bullets.append((4, f"VPA bearish ({int(vpa_str*100)}% strength)"))

        # Sort by impact (score contribution), take top 3
        bullets.sort(key=lambda x: x[0], reverse=True)
        return [b[1] for b in bullets[:3]]

    # ══════════════════════════════════════════════════════════
    # ACTION ENGINE METHODS
    # ══════════════════════════════════════════════════════════

    def _generate_wait_until(
        self, decision: Decision, regime_name: str,
        vwap: float, vwap_pos: str, p: float,
        nearest_sup: float, nearest_res: float, atr: float,
    ) -> list[dict]:
        """Generate conditional 'wait until' triggers — what flips WAIT → action."""
        triggers = []
        r = regime_name.upper() if regime_name else ""

        if decision != Decision.WAIT:
            # For active decisions, show what WOULD flip them
            return []

        # Primary: key levels
        if nearest_res > 0 and nearest_res > p:
            triggers.append({
                "direction": "above",
                "level": round(nearest_res, 2),
                "label": f"Break above {_fp(nearest_res)}",
                "action": "BUY_CALLS",
            })
        elif vwap > 0 and vwap > p:
            triggers.append({
                "direction": "above",
                "level": round(vwap, 2),
                "label": f"Reclaim VWAP {_fp(vwap)}",
                "action": "BUY_CALLS",
            })

        if nearest_sup > 0 and nearest_sup < p:
            triggers.append({
                "direction": "below",
                "level": round(nearest_sup, 2),
                "label": f"Break below {_fp(nearest_sup)}",
                "action": "BUY_PUTS",
            })
        elif vwap > 0 and vwap < p:
            triggers.append({
                "direction": "below",
                "level": round(vwap, 2),
                "label": f"Lose VWAP {_fp(vwap)}",
                "action": "BUY_PUTS",
            })

        # If RANGE_CHOP with no levels, use VWAP ± ATR as proxies
        if not triggers and "RANGE" in r:
            if vwap > 0 and atr > 0:
                triggers.append({
                    "direction": "above",
                    "level": round(vwap + atr * 1.2, 2),
                    "label": f"Break above {_fp(vwap + atr * 1.2)}",
                    "action": "BUY_CALLS",
                })
                triggers.append({
                    "direction": "below",
                    "level": round(vwap - atr * 1.2, 2),
                    "label": f"Break below {_fp(vwap - atr * 1.2)}",
                    "action": "BUY_PUTS",
                })

        return triggers

    def _determine_execution_mode(
        self, decision: Decision, regime_name: str,
        vol_regime_name: str, hard_guard: bool,
        sigma: float, capital_mode: str,
    ) -> tuple[str, str]:
        """Determine execution mode: SCALP_ONLY / BREAKOUT / FULL / STAND_ASIDE."""
        r = regime_name.upper() if regime_name else ""

        if hard_guard or capital_mode == "LOCKED":
            return "STAND_ASIDE", "Hard guard active — no trades"

        if decision == Decision.WAIT:
            if "RANGE" in r or "CHOP" in r:
                if vol_regime_name != "LOW":
                    return "SCALP_ONLY", "Range regime — only scalp extremes"
                return "STAND_ASIDE", "Low volume range — no edge"
            if abs(sigma) > 1.5:
                return "STAND_ASIDE", "Extended from VWAP — wait for reversion"
            return "STAND_ASIDE", "No clear setup — wait for trigger"

        # Active decision
        if "BREAKOUT" in r:
            return "BREAKOUT", "Breakout regime — ride momentum"
        if "TREND" in r:
            return "FULL", "Trend aligned — full conviction entry"

        return "SCALP_ONLY", "Mixed signals — take quick profits"

    def _classify_wait_confidence(
        self, decision: Decision, hard_guard: bool,
        call_score: int, put_score: int,
        regime_conf: float, vol_regime_name: str,
    ) -> str:
        """Classify how confident we are in the WAIT call."""
        if decision != Decision.WAIT:
            return ""

        top = max(call_score, put_score)
        gap = abs(call_score - put_score)

        # Strong WAIT: clear reasons to stay out
        if hard_guard:
            return "Strong"
        if top < 30 and vol_regime_name == "LOW":
            return "Strong"
        if top < 25:
            return "Strong"

        # Weak WAIT: close to threshold, could flip soon
        if top >= 50 and gap >= 10:
            return "Weak"
        if top >= 45 and regime_conf > 0.7:
            return "Weak"

        return "Moderate"

    def _generate_next_move_map(
        self, decision: Decision, regime_name: str,
        vwap: float, p: float,
        nearest_sup: float, nearest_res: float,
        atr: float, max_pain: float,
    ) -> list[dict]:
        """Generate 'if X → then Y' next move scenarios."""
        moves = []
        r = regime_name.upper() if regime_name else ""

        # Scenario 1: VWAP reclaim/lose
        if vwap > 0:
            if p < vwap:
                moves.append({
                    "scenario": f"Price reclaims VWAP ({_fp(vwap)})",
                    "outcome": f"Mean reversion rally toward {_fp(vwap + atr * 0.8)}",
                })
                moves.append({
                    "scenario": f"Price loses {_fp(nearest_sup) if nearest_sup > 0 else _fp(p - atr)}",
                    "outcome": f"Expansion down toward {_fp(p - atr * 1.5)}",
                })
            elif p > vwap:
                moves.append({
                    "scenario": f"Price holds above VWAP ({_fp(vwap)})",
                    "outcome": f"Continuation toward {_fp(nearest_res) if nearest_res > p else _fp(p + atr * 0.8)}",
                })
                moves.append({
                    "scenario": f"Price loses VWAP ({_fp(vwap)})",
                    "outcome": f"Pullback toward {_fp(nearest_sup) if nearest_sup > 0 else _fp(vwap - atr * 0.5)}",
                })
            else:
                moves.append({
                    "scenario": f"Breakout above {_fp(nearest_res) if nearest_res > 0 else _fp(p + atr)}",
                    "outcome": "Expansion up — momentum entry opportunity",
                })
                moves.append({
                    "scenario": f"Breakdown below {_fp(nearest_sup) if nearest_sup > 0 else _fp(p - atr)}",
                    "outcome": "Expansion down — put entry opportunity",
                })

        # Scenario 3: Max pain gravity
        if max_pain > 0 and abs(p - max_pain) / (p or 1) > 0.005:
            direction = "down" if max_pain < p else "up"
            moves.append({
                "scenario": f"No catalyst by 14:00 ET",
                "outcome": f"Drift {direction} toward max pain {_fp(max_pain)}",
            })

        return moves[:3]  # max 3 scenarios


# ── Helpers ───────────────────────────────────────────────────
def _fp(v) -> str:
    try:
        return f"{float(v):.2f}"
    except Exception:
        return "—"


# ── Singleton ─────────────────────────────────────────────────
decision_engine = DecisionEngine()
