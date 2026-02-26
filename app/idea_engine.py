"""
Idea Engine
===========
Transforms raw signal data into a structured daily briefing with:
  - Day-type classification (Trend / Chop / Reversal / Breakout)
  - Bias narrative (what the market is telegraphing)
  - Trade themes (macro story for the session)
  - Conditional IF/THEN plans (playbook-style setups)
  - Concrete trade ideas (entry / stop / target)
  - Positioning panel (call walls, put walls, gamma magnets)
  - Playbook mode filtering (0-DTE | Swing | Scalp)

Pure computation — no I/O.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Optional

import pandas as pd


# ── Enums ─────────────────────────────────────────────────────

class PlaybookMode(str, Enum):
    ALL   = "all"
    ZERO_DTE = "0dte"
    SWING = "swing"
    SCALP = "scalp"


class DayType(str, Enum):
    TREND_DAY      = "TREND_DAY"
    TWO_SIDED      = "TWO_SIDED"
    RANGE_CHOP     = "RANGE_CHOP"
    BREAKOUT_DAY   = "BREAKOUT_DAY"
    REVERSAL_DAY   = "REVERSAL_DAY"
    INSIDE_DAY     = "INSIDE_DAY"
    UNKNOWN        = "UNKNOWN"


class IdeaType(str, Enum):
    BREAKOUT    = "BREAKOUT"
    PULLBACK    = "PULLBACK"
    FADE        = "FADE"
    RANGE_PLAY  = "RANGE_PLAY"
    MOMENTUM    = "MOMENTUM"


# ── Data structures ───────────────────────────────────────────

@dataclass
class TradeTheme:
    """A macro-level narrative theme for the trading session."""
    title: str          # e.g. "Bull Continuation"
    description: str    # 1-2 sentence narrative
    direction: str      # "bullish" | "bearish" | "neutral"
    confidence: float   # 0.0-1.0
    emoji: str = ""     # visual marker


@dataclass
class BiasSummary:
    """Overall market bias with supporting evidence."""
    direction: str          # "bullish" | "bearish" | "neutral"
    strength: float         # 0.0-1.0
    headline: str           # e.g. "TREND_UP above VWAP, RSI 62, GEX positive"
    bullets: list[str] = field(default_factory=list)   # supporting points
    kill_switch: str = ""   # "If price drops below VWAP, bias flips"


@dataclass
class DayTypeProb:
    """Probability distribution over day types."""
    most_likely: DayType
    probabilities: dict[str, float] = field(default_factory=dict)
    description: str = ""
    setup_implications: str = ""    # how this affects which setups to take


@dataclass
class StrikeWall:
    """A key option wall from chain analysis."""
    strike: float
    wall_type: str          # "call_wall" | "put_wall"
    open_interest: int
    distance_pct: float
    label: str


@dataclass
class StrikeMagnet:
    """A gamma magnet strike."""
    strike: float
    net_gamma_exp: float
    distance_pct: float
    polarity: str           # "positive" | "negative"
    label: str


@dataclass
class PositioningData:
    """Options market positioning snapshot."""
    call_walls: list[StrikeWall] = field(default_factory=list)
    put_walls: list[StrikeWall] = field(default_factory=list)
    gamma_magnets: list[StrikeMagnet] = field(default_factory=list)
    max_pain: float = 0.0
    net_gex: float = 0.0
    gex_regime: str = "neutral"
    summary: str = ""       # e.g. "Price pinned between 620 call wall and 615 put floor"


@dataclass
class ConditionalPlan:
    """IF-THEN conditional trade plan."""
    condition: str              # "IF QQQ breaks above 621.50 with volume"
    action: str                 # "THEN buy calls"
    option_description: str     # "Look at 622–625 strikes"
    direction: str              # "CALL" | "PUT"
    trigger_level: float
    stop_description: str       # "Exit if reverses back below breakout level"
    targets: list[float] = field(default_factory=list)
    suitable_for: list[str] = field(default_factory=list)  # ["0dte", "scalp"]


@dataclass
class TradeIdea:
    """A fully-formed actionable trade idea."""
    idea_type: IdeaType
    direction: str              # "CALL" | "PUT"
    symbol: str
    headline: str               # "Breakout CALL — Trend continuation above 621.50"
    rationale: str              # 2-3 sentence explanation
    entry_level: float
    stop_level: float
    target_1: float
    target_2: float
    reward_risk: float
    option_hint: str            # "Look for 622C or 623C with delta 0.40-0.50"
    suitable_for: list[str] = field(default_factory=list)  # playbook modes
    confidence: float = 0.0
    score: int = 0              # 0–100 composite score
    score_breakdown: dict = field(default_factory=dict)  # 5 components
    ai_narrative: Optional[str] = None     # filled by LLM narrator if available
    warning: str = ""


@dataclass
class DailyBriefing:
    """
    The complete Idea Engine output for a single symbol / session.
    Fed into the /api/briefing/{symbol} endpoint.
    """
    symbol: str
    as_of: str
    underlying_price: float

    bias: BiasSummary
    day_type: DayTypeProb
    themes: list[TradeTheme] = field(default_factory=list)
    positioning: Optional[PositioningData] = None
    conditional_plans: list[ConditionalPlan] = field(default_factory=list)
    trade_ideas: list[TradeIdea] = field(default_factory=list)

    # Raw signal pass-throughs (used by LLM narrator)
    regime_name: str = ""
    regime_confidence: float = 0.0
    vwap_position: str = "at"
    rsi: float = 50.0
    atr: float = 0.0

    ai_briefing_narrative: Optional[str] = None   # top-level LLM narrative


# ── Input bundle ─────────────────────────────────────────────

@dataclass
class BriefingInput:
    """Everything the IdeaEngine needs — passed in from the endpoint."""
    symbol: str
    underlying_price: float
    df: pd.DataFrame                    # intraday OHLCV
    regime: object                      # RegimeResult (typed loosely to avoid circular)
    sr: object                          # SRResult
    chain_metrics: object               # ChainMetrics
    vpa_bias: dict                      # from VPAEngine.get_bias()
    active_alerts: list                 # list[ActiveAlert]
    expirations: list = field(default_factory=list)  # list[date]


# ── Engine ────────────────────────────────────────────────────

class IdeaEngine:
    """
    Generates a DailyBriefing from raw signal data.
    Pure computation — no I/O, no async.
    """

    def generate_briefing(self, inp: BriefingInput) -> DailyBriefing:
        """Main entry point — returns a full DailyBriefing."""
        try:
            bias      = self._build_bias_summary(inp)
            day_type  = self._classify_day_type(inp)
            themes    = self._classify_themes(inp, bias, day_type)
            positioning = self._build_positioning(inp)
            cond_plans  = self._build_conditional_plans(inp, bias, day_type)
            trade_ideas = self._generate_ideas(inp, bias, day_type)

            regime_name = ""
            regime_conf = 0.0
            vwap_pos    = "at"
            rsi_val     = 50.0
            atr_val     = 0.0
            if inp.regime:
                regime_name = getattr(inp.regime, "regime", "") 
                if hasattr(regime_name, "value"):
                    regime_name = regime_name.value
                regime_conf = getattr(inp.regime, "confidence", 0.0)
                vwap_pos    = getattr(inp.regime, "price_vs_vwap", "at")
                rsi_val     = getattr(inp.regime, "rsi_current", 50.0)
                atr_val     = getattr(inp.regime, "atr_current", 0.0)

            return DailyBriefing(
                symbol=inp.symbol,
                as_of=datetime.utcnow().isoformat(),
                underlying_price=inp.underlying_price,
                bias=bias,
                day_type=day_type,
                themes=themes,
                positioning=positioning,
                conditional_plans=cond_plans,
                trade_ideas=trade_ideas,
                regime_name=regime_name,
                regime_confidence=regime_conf,
                vwap_position=vwap_pos,
                rsi=rsi_val,
                atr=atr_val,
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            # Return a minimal briefing rather than crashing the endpoint
            return DailyBriefing(
                symbol=inp.symbol,
                as_of=datetime.utcnow().isoformat(),
                underlying_price=inp.underlying_price,
                bias=BiasSummary(direction="neutral", strength=0.5, headline="Data unavailable"),
                day_type=DayTypeProb(most_likely=DayType.UNKNOWN, description="Insufficient data"),
            )

    # ── Bias Summary ───────────────────────────────────────────

    def _build_bias_summary(self, inp: BriefingInput) -> BiasSummary:
        r = inp.regime
        cm = inp.chain_metrics
        vpa = inp.vpa_bias or {}

        bullets: list[str] = []
        direction = "neutral"
        strength  = 0.5

        if r:
            regime_val = getattr(r, "regime", None)
            regime_str = regime_val.value if hasattr(regime_val, "value") else str(regime_val)
            rsi        = getattr(r, "rsi_current", 50.0)
            pvwap      = getattr(r, "price_vs_vwap", "at")
            conf       = getattr(r, "confidence", 0.5)

            if regime_str in ("TREND_UP", "BREAKOUT_ATTEMPT"):
                direction = "bullish"
                strength  = min(1.0, conf + 0.1)
            elif regime_str in ("TREND_DOWN",):
                direction = "bearish"
                strength  = min(1.0, conf + 0.1)
            elif regime_str == "REVERSAL_EXHAUSTION":
                # Direction from VPA or default to neutral
                vpa_b = vpa.get("bias", "neutral")
                direction = "bullish" if "bullish" in vpa_b else ("bearish" if "bearish" in vpa_b else "neutral")
                strength  = 0.55
            else:
                direction = "neutral"
                strength  = 0.35

            bullets.append(f"Regime: {regime_str} ({int(conf*100)}% confidence)")
            bullets.append(f"Price {pvwap} VWAP — RSI {rsi:.0f}")

        # Chain overlay
        if cm:
            gex = getattr(cm, "gex_regime", "neutral")
            pcr = getattr(cm, "put_call_oi_ratio", 1.0)
            ivr = getattr(cm, "iv_rank", 50.0)

            if gex == "positive":
                bullets.append("GEX positive — dealers long gamma, expect mean-reversion")
            elif gex == "negative":
                bullets.append("GEX negative — dealers short gamma, moves may accelerate")

            if pcr > 1.4:
                bullets.append(f"Put/Call OI {pcr:.2f} — elevated hedging / bearish skew")
            elif pcr < 0.7:
                bullets.append(f"Put/Call OI {pcr:.2f} — bullish positioning")

            if ivr < 25:
                bullets.append(f"IV Rank {ivr:.0f} — cheap premium, favor buying options")
            elif ivr > 75:
                bullets.append(f"IV Rank {ivr:.0f} — elevated IV, consider spreads or selling")

        # VPA overlay
        vpa_bias_str = vpa.get("bias", "neutral") if vpa else "neutral"
        if vpa_bias_str not in ("neutral", ""):
            bullets.append(f"VPA bias: {vpa_bias_str}")
            if direction == "neutral":
                direction = "bullish" if "bullish" in vpa_bias_str else ("bearish" if "bearish" in vpa_bias_str else "neutral")

        # Kill switch
        kill = ""
        if inp.regime:
            pvwap = getattr(inp.regime, "price_vs_vwap", "at")
            vwap_curr = getattr(inp.regime, "vwap_current", 0.0)
            if direction == "bullish" and pvwap in ("above", "at"):
                kill = f"Bias flips bearish if price drops back below VWAP ({vwap_curr:.2f})"
            elif direction == "bearish" and pvwap in ("below", "at"):
                kill = f"Bias flips bullish if price reclaims VWAP ({vwap_curr:.2f})"

        dir_emoji_map = {"bullish": "▲", "bearish": "▼", "neutral": "↔"}
        regime_str_short = ""
        if inp.regime:
            rv = getattr(inp.regime, "regime", "")
            regime_str_short = rv.value if hasattr(rv, "value") else str(rv)

        headline = f"{dir_emoji_map.get(direction,'↔')} {direction.upper()} — {regime_str_short}"
        if inp.regime:
            rsi = getattr(inp.regime, "rsi_current", 50.0)
            headline += f", RSI {rsi:.0f}"
        if inp.chain_metrics:
            gex = getattr(inp.chain_metrics, "gex_regime", "neutral")
            headline += f", GEX {gex}"

        return BiasSummary(
            direction=direction,
            strength=round(strength, 2),
            headline=headline,
            bullets=bullets,
            kill_switch=kill,
        )

    # ── Day Type ──────────────────────────────────────────────

    def _classify_day_type(self, inp: BriefingInput) -> DayTypeProb:
        r    = inp.regime
        df   = inp.df
        probs: dict[str, float] = {dt.value: 0.05 for dt in DayType}

        if r is None or df is None or df.empty:
            return DayTypeProb(
                most_likely=DayType.UNKNOWN,
                probabilities=probs,
                description="Insufficient data to classify day type",
            )

        regime_val = getattr(r, "regime", None)
        regime_str = regime_val.value if hasattr(regime_val, "value") else str(regime_val)
        rsi        = getattr(r, "rsi_current", 50.0)
        atr        = getattr(r, "atr_current", 0.0)

        # Day range vs ATR
        day_range  = 0.0
        if len(df) >= 4:
            day_high = df["high"].max()
            day_low  = df["low"].min()
            day_range = day_high - day_low

        atr_expansion = (day_range / atr) if atr > 0 else 1.0

        # RSI extremes
        rsi_extended = rsi > 70 or rsi < 30

        # Regime-based probability assignment
        if regime_str == "TREND_UP":
            probs[DayType.TREND_DAY.value]    = 0.55
            probs[DayType.BREAKOUT_DAY.value] = 0.20
            probs[DayType.TWO_SIDED.value]    = 0.10
            probs[DayType.RANGE_CHOP.value]   = 0.05
        elif regime_str == "TREND_DOWN":
            probs[DayType.TREND_DAY.value]    = 0.55
            probs[DayType.REVERSAL_DAY.value] = 0.20
            probs[DayType.TWO_SIDED.value]    = 0.10
            probs[DayType.RANGE_CHOP.value]   = 0.05
        elif regime_str == "BREAKOUT_ATTEMPT":
            probs[DayType.BREAKOUT_DAY.value] = 0.50
            probs[DayType.TREND_DAY.value]    = 0.25
            probs[DayType.TWO_SIDED.value]    = 0.15
        elif regime_str == "REVERSAL_EXHAUSTION":
            probs[DayType.REVERSAL_DAY.value] = 0.50
            probs[DayType.TWO_SIDED.value]    = 0.25
            probs[DayType.TREND_DAY.value]    = 0.10
        elif regime_str == "RANGE_CHOP":
            probs[DayType.RANGE_CHOP.value]   = 0.55
            probs[DayType.INSIDE_DAY.value]   = 0.20
            probs[DayType.TWO_SIDED.value]    = 0.15

        # ATR expansion nudge
        if atr_expansion > 1.5:
            probs[DayType.TREND_DAY.value]    = min(probs[DayType.TREND_DAY.value] + 0.10, 1.0)
            probs[DayType.BREAKOUT_DAY.value] = min(probs[DayType.BREAKOUT_DAY.value] + 0.05, 1.0)
        elif atr_expansion < 0.6:
            probs[DayType.INSIDE_DAY.value]   = min(probs[DayType.INSIDE_DAY.value] + 0.20, 1.0)
            probs[DayType.RANGE_CHOP.value]   = min(probs[DayType.RANGE_CHOP.value] + 0.15, 1.0)

        # RSI extremes nudge reversal probability
        if rsi_extended:
            probs[DayType.REVERSAL_DAY.value] = min(probs[DayType.REVERSAL_DAY.value] + 0.15, 1.0)

        # Normalize
        total = sum(probs.values())
        if total > 0:
            probs = {k: round(v / total, 3) for k, v in probs.items()}

        most_likely_str = max(probs, key=lambda k: probs[k])
        most_likely = DayType(most_likely_str)

        desc_map = {
            DayType.TREND_DAY:    "Strong directional momentum — fade-the-counter setups risky",
            DayType.TWO_SIDED:    "Both bulls and bears active — look for range extremes",
            DayType.RANGE_CHOP:   "Low expansion, choppy — reduce size, tighten targets",
            DayType.BREAKOUT_DAY: "Price pressing key levels — breakout plays in focus",
            DayType.REVERSAL_DAY: "Extended move with exhaustion signals — reversal setups valid",
            DayType.INSIDE_DAY:   "Narrow range so far — wait for the expansion break",
            DayType.UNKNOWN:      "Insufficient data",
        }
        impl_map = {
            DayType.TREND_DAY:    "Favor pullback-to-VWAP entries in trend direction; avoid counters",
            DayType.TWO_SIDED:    "Range highs for puts, range lows for calls, keep stops tight",
            DayType.RANGE_CHOP:   "Scalp range extremes, book quickly, no overnight holds",
            DayType.BREAKOUT_DAY: "Buy the breakout + volume confirmation; watch for false breaks",
            DayType.REVERSAL_DAY: "Counter-trend entries off exhaustion candles with small size",
            DayType.INSIDE_DAY:   "Wait for opening range to expand before committing direction",
            DayType.UNKNOWN:      "No actionable implication",
        }

        return DayTypeProb(
            most_likely=most_likely,
            probabilities=probs,
            description=desc_map.get(most_likely, ""),
            setup_implications=impl_map.get(most_likely, ""),
        )

    # ── Themes ────────────────────────────────────────────────

    def _classify_themes(
        self,
        inp: BriefingInput,
        bias: BiasSummary,
        day_type: DayTypeProb,
    ) -> list[TradeTheme]:
        themes: list[TradeTheme] = []
        r  = inp.regime
        cm = inp.chain_metrics

        if r is None:
            return themes

        regime_val = getattr(r, "regime", None)
        regime_str = regime_val.value if hasattr(regime_val, "value") else str(regime_val)
        rsi        = getattr(r, "rsi_current", 50.0)
        pvwap      = getattr(r, "price_vs_vwap", "at")
        conf       = getattr(r, "confidence", 0.5)

        # Theme 1: Trend Continuation
        if bias.direction == "bullish" and regime_str in ("TREND_UP", "BREAKOUT_ATTEMPT") and pvwap == "above":
            themes.append(TradeTheme(
                title="Bull Continuation",
                description=(
                    f"Price above VWAP in {regime_str} regime ({int(conf*100)}% conf). "
                    "Pullbacks to VWAP are buying opportunities."
                ),
                direction="bullish",
                confidence=round(conf * 0.9, 2),
                emoji="🐂",
            ))
        elif bias.direction == "bearish" and regime_str in ("TREND_DOWN",) and pvwap == "below":
            themes.append(TradeTheme(
                title="Bear Continuation",
                description=(
                    f"Price below VWAP in TREND_DOWN regime ({int(conf*100)}% conf). "
                    "Bounces to VWAP are selling opportunities."
                ),
                direction="bearish",
                confidence=round(conf * 0.9, 2),
                emoji="🐻",
            ))

        # Theme 2: VWAP Reclaim / Rejection
        if pvwap == "above" and regime_str not in ("TREND_UP",):
            themes.append(TradeTheme(
                title="VWAP Reclaim Play",
                description="Price recently reclaimed VWAP — watch for confirmation hold to trigger calls.",
                direction="bullish",
                confidence=0.55,
                emoji="📈",
            ))
        elif pvwap == "below" and regime_str not in ("TREND_DOWN",):
            themes.append(TradeTheme(
                title="VWAP Rejection Play",
                description="Price rejected VWAP from below — failed reclaim often triggers put momentum.",
                direction="bearish",
                confidence=0.55,
                emoji="📉",
            ))

        # Theme 3: Breakout
        if regime_str == "BREAKOUT_ATTEMPT":
            themes.append(TradeTheme(
                title="Breakout Setup",
                description="Price pressing against key levels with expanding volume — breakout entries in focus.",
                direction=bias.direction,
                confidence=0.65,
                emoji="🚀",
            ))

        # Theme 4: Reversal
        if regime_str == "REVERSAL_EXHAUSTION":
            if rsi > 70:
                themes.append(TradeTheme(
                    title="Overbought Reversal",
                    description=f"RSI {rsi:.0f} in exhaustion regime — look for bearish reversal setups.",
                    direction="bearish",
                    confidence=0.60,
                    emoji="🔄",
                ))
            elif rsi < 30:
                themes.append(TradeTheme(
                    title="Oversold Bounce",
                    description=f"RSI {rsi:.0f} in exhaustion regime — look for bullish bounce setups.",
                    direction="bullish",
                    confidence=0.60,
                    emoji="🔄",
                ))

        # Theme 5: Choppy/Range
        if regime_str == "RANGE_CHOP" or day_type.most_likely == DayType.RANGE_CHOP:
            themes.append(TradeTheme(
                title="Range Scalp Day",
                description="Low ATR expansion and choppy price action — favor range extremes, quick in-and-out.",
                direction="neutral",
                confidence=0.50,
                emoji="↔️",
            ))

        # Theme 6: IV environment
        if cm:
            ivr = getattr(cm, "iv_rank", 50.0)
            if ivr < 20:
                themes.append(TradeTheme(
                    title="Cheap Premium — Buy Optionality",
                    description=f"IV Rank at {ivr:.0f} — premium is historically cheap. Favor debit strategies.",
                    direction="neutral",
                    confidence=0.70,
                    emoji="💰",
                ))
            elif ivr > 80:
                themes.append(TradeTheme(
                    title="Elevated IV — Spread or Sell",
                    description=f"IV Rank at {ivr:.0f} — premium is expensive. Buy spreads or sell premium.",
                    direction="neutral",
                    confidence=0.70,
                    emoji="⚠️",
                ))

        # Sort by confidence desc, cap at 4
        themes.sort(key=lambda t: t.confidence, reverse=True)
        return themes[:4]

    # ── Positioning ──────────────────────────────────────────

    def _build_positioning(self, inp: BriefingInput) -> Optional[PositioningData]:
        cm = inp.chain_metrics
        if cm is None:
            return None

        call_walls: list[StrikeWall] = []
        for w in getattr(cm, "top_call_walls", []):
            call_walls.append(StrikeWall(
                strike=w.get("strike", 0),
                wall_type="call_wall",
                open_interest=w.get("open_interest", 0),
                distance_pct=w.get("distance_pct", 0),
                label=w.get("label", ""),
            ))

        put_walls: list[StrikeWall] = []
        for w in getattr(cm, "top_put_walls", []):
            put_walls.append(StrikeWall(
                strike=w.get("strike", 0),
                wall_type="put_wall",
                open_interest=w.get("open_interest", 0),
                distance_pct=w.get("distance_pct", 0),
                label=w.get("label", ""),
            ))

        magnets: list[StrikeMagnet] = []
        for m in getattr(cm, "gamma_magnets", []):
            magnets.append(StrikeMagnet(
                strike=m.get("strike", 0),
                net_gamma_exp=m.get("net_gamma_exp", 0),
                distance_pct=m.get("distance_pct", 0),
                polarity=m.get("polarity", "neutral"),
                label=m.get("label", ""),
            ))

        max_pain   = round(getattr(cm, "max_pain", 0), 2)
        net_gex    = getattr(cm, "net_gex", 0)
        gex_regime = getattr(cm, "gex_regime", "neutral")

        # Build summary string
        parts: list[str] = []
        spot = inp.underlying_price
        if call_walls:
            cw = call_walls[0]
            parts.append(f"Top call wall at {cw.strike:.0f} ({cw.distance_pct:+.1f}%)")
        if put_walls:
            pw = put_walls[0]
            parts.append(f"Top put wall at {pw.strike:.0f} ({pw.distance_pct:+.1f}%)")
        if max_pain > 0:
            mp_dist = round((max_pain - spot) / spot * 100, 1) if spot > 0 else 0
            parts.append(f"Max pain {max_pain:.0f} ({mp_dist:+.1f}%)")
        summary = " · ".join(parts) if parts else "No significant walls detected"

        return PositioningData(
            call_walls=call_walls,
            put_walls=put_walls,
            gamma_magnets=magnets,
            max_pain=max_pain,
            net_gex=net_gex,
            gex_regime=gex_regime,
            summary=summary,
        )

    # ── Conditional Plans ─────────────────────────────────────

    def _build_conditional_plans(
        self,
        inp: BriefingInput,
        bias: BiasSummary,
        day_type: DayTypeProb,
    ) -> list[ConditionalPlan]:
        plans: list[ConditionalPlan] = []
        r   = inp.regime
        sr  = inp.sr
        cm  = inp.chain_metrics
        spot = inp.underlying_price
        sym  = inp.symbol

        if r is None or spot <= 0:
            return plans

        pvwap    = getattr(r, "price_vs_vwap", "at")
        vwap_val = getattr(r, "vwap_current", spot)
        atr      = getattr(r, "atr_current", spot * 0.003)
        regime_val = getattr(r, "regime", None)
        regime_str = regime_val.value if hasattr(regime_val, "value") else str(regime_val)

        # Nearest S/R
        ns_price  = getattr(sr, "nearest_support", None) if sr else None
        nr_price  = getattr(sr, "nearest_resistance", None) if sr else None

        # ── Plan 1: VWAP Reclaim (bullish) ────────────────
        if pvwap == "above":
            entry_level = round(vwap_val + atr * 0.2, 2)
            stop_level  = round(vwap_val - atr * 0.3, 2)
            t1          = round(entry_level + atr * 0.8, 2)
            t2          = round(entry_level + atr * 1.5, 2)
            plans.append(ConditionalPlan(
                condition=f"IF {sym} holds above VWAP ({vwap_val:.2f}) after pullback",
                action="THEN buy calls on the bounce",
                option_description=f"ATM to 1-strike OTM calls, look near {entry_level:.2f}",
                direction="CALL",
                trigger_level=entry_level,
                stop_description=f"Exit if price closes below VWAP ({vwap_val:.2f}) on 5m bar",
                targets=[t1, t2],
                suitable_for=["0dte", "scalp"],
            ))
        elif pvwap == "below":
            entry_level = round(vwap_val - atr * 0.2, 2)
            stop_level  = round(vwap_val + atr * 0.3, 2)
            t1          = round(entry_level - atr * 0.8, 2)
            t2          = round(entry_level - atr * 1.5, 2)
            plans.append(ConditionalPlan(
                condition=f"IF {sym} fails to reclaim VWAP ({vwap_val:.2f})",
                action="THEN buy puts on the rejection",
                option_description=f"ATM to 1-strike OTM puts, look near {entry_level:.2f}",
                direction="PUT",
                trigger_level=entry_level,
                stop_description=f"Exit if price reclaims VWAP ({vwap_val:.2f})",
                targets=[t1, t2],
                suitable_for=["0dte", "scalp"],
            ))

        # ── Plan 2: Breakout above resistance ───────────────
        if nr_price and nr_price > spot:
            dist = nr_price - spot
            plans.append(ConditionalPlan(
                condition=f"IF {sym} breaks above {nr_price:.2f} on volume",
                action="THEN buy calls — breakout continuation",
                option_description=f"Strike above {nr_price:.0f}; delta 0.40-0.55",
                direction="CALL",
                trigger_level=nr_price,
                stop_description=f"Exit if price falls back below {nr_price:.2f} (false break)",
                targets=[
                    round(nr_price + atr * 0.7, 2),
                    round(nr_price + atr * 1.4, 2),
                ],
                suitable_for=["0dte", "swing"],
            ))

        # ── Plan 3: Bounce off support (bullish bias) ───────
        if ns_price and ns_price < spot and bias.direction in ("bullish", "neutral"):
            plans.append(ConditionalPlan(
                condition=f"IF {sym} pulls back to {ns_price:.2f} and holds",
                action="THEN buy calls — support bounce",
                option_description=f"ATM calls with stop below {ns_price:.2f}",
                direction="CALL",
                trigger_level=ns_price,
                stop_description=f"Exit if price closes below {ns_price:.2f}",
                targets=[
                    round(ns_price + atr * 0.8, 2),
                    round(ns_price + atr * 1.5, 2),
                ],
                suitable_for=["scalp", "swing"],
            ))

        # ── Plan 4: Breakout below support (bearish bias) ───
        if ns_price and ns_price < spot and bias.direction in ("bearish",):
            plans.append(ConditionalPlan(
                condition=f"IF {sym} breaks below {ns_price:.2f} on volume",
                action="THEN buy puts — support breakdown",
                option_description=f"ATM to 1-strike OTM puts below {ns_price:.0f}",
                direction="PUT",
                trigger_level=ns_price,
                stop_description=f"Exit if price reclaims {ns_price:.2f}",
                targets=[
                    round(ns_price - atr * 0.8, 2),
                    round(ns_price - atr * 1.5, 2),
                ],
                suitable_for=["0dte", "swing"],
            ))

        # ── Plan 5: Gamma magnet gravitational pull ──────────
        if cm:
            magnets = getattr(cm, "gamma_magnets", [])
            if magnets:
                mag = magnets[0]
                mag_strike = mag.get("strike", 0)
                if abs(mag_strike - spot) / spot < 0.015:   # within 1.5%
                    polarity = mag.get("polarity", "neutral")
                    if polarity == "positive":
                        plans.append(ConditionalPlan(
                            condition=f"Gamma magnet at {mag_strike:.0f} — positive GEX will pin price",
                            action="Expect price to gravitate toward (and oscillate around) this level",
                            option_description="Strangle or range scalp around the magnet strike",
                            direction="CALL" if spot < mag_strike else "PUT",
                            trigger_level=mag_strike,
                            stop_description="Abandon if price moves >1 ATR away from magnet",
                            targets=[mag_strike],
                            suitable_for=["0dte", "scalp"],
                        ))

        return plans[:5]   # max 5 plans

    # ── Trade Ideas ──────────────────────────────────────────

    def _generate_ideas(
        self,
        inp: BriefingInput,
        bias: BiasSummary,
        day_type: DayTypeProb,
    ) -> list[TradeIdea]:
        ideas: list[TradeIdea] = []
        r    = inp.regime
        sr   = inp.sr
        cm   = inp.chain_metrics
        spot = inp.underlying_price
        sym  = inp.symbol

        if r is None or spot <= 0:
            return ideas

        pvwap      = getattr(r, "price_vs_vwap", "at")
        vwap_val   = getattr(r, "vwap_current", spot)
        atr        = getattr(r, "atr_current", spot * 0.003)
        rsi        = getattr(r, "rsi_current", 50.0)
        regime_val = getattr(r, "regime", None)
        regime_str = regime_val.value if hasattr(regime_val, "value") else str(regime_val)

        ns_price = getattr(sr, "nearest_support", None) if sr else None
        nr_price = getattr(sr, "nearest_resistance", None) if sr else None
        ivr      = getattr(cm, "iv_rank", 50.0) if cm else 50.0

        def _rr(entry, stop, t1):
            if abs(entry - stop) < 0.01:
                return 0.0
            return round(abs(t1 - entry) / abs(entry - stop), 2)

        def _option_hint(direction, entry, delta_range="0.40-0.55"):
            ct   = "Call" if direction == "CALL" else "Put"
            d    = "above" if direction == "CALL" else "below"
            near = round(entry / 1) * 1
            hint = f"Look for {sym} {ct} strike {d} {near:.0f}, delta {delta_range}"
            if ivr < 25:
                hint += ", buy outright (cheap IV)"
            elif ivr > 65:
                hint += ", consider debit spread (rich IV)"
            return hint

        # ── Idea 1: Breakout CALL ──────────────────────────
        if regime_str in ("TREND_UP", "BREAKOUT_ATTEMPT") and pvwap == "above":
            entry = round(spot + atr * 0.15, 2)
            stop  = round(vwap_val - atr * 0.1, 2)
            t1    = round(entry + atr * 0.75, 2)
            t2    = round(entry + atr * 1.5, 2)
            rr    = _rr(entry, stop, t1)
            conf  = min(0.85, getattr(r, "confidence", 0.6) + 0.05)
            if rr >= 1.5:
                ideas.append(TradeIdea(
                    idea_type=IdeaType.BREAKOUT,
                    direction="CALL",
                    symbol=sym,
                    headline=f"Breakout CALL — {regime_str} above VWAP",
                    rationale=(
                        f"{sym} is in {regime_str} with {int(conf*100)}% confidence, "
                        f"trading above VWAP ({vwap_val:.2f}). "
                        f"RSI at {rsi:.0f} supports momentum continuation. "
                        "Enter on any minor pullback with volume confirmation."
                    ),
                    entry_level=entry,
                    stop_level=stop,
                    target_1=t1,
                    target_2=t2,
                    reward_risk=rr,
                    option_hint=_option_hint("CALL", entry),
                    suitable_for=["0dte", "scalp"],
                    confidence=conf,
                    warning="" if rsi < 70 else f"RSI extended at {rsi:.0f} — reduce size",
                ))

        # ── Idea 2: Pullback CALL (touch VWAP in uptrend) ──
        if regime_str == "TREND_UP" and pvwap in ("above", "at") and rsi < 65:
            entry = round(vwap_val + atr * 0.05, 2)
            stop  = round(vwap_val - atr * 0.4, 2)
            t1    = round(entry + atr * 0.8, 2)
            t2    = round(entry + atr * 1.6, 2)
            rr    = _rr(entry, stop, t1)
            if rr >= 1.5:
                ideas.append(TradeIdea(
                    idea_type=IdeaType.PULLBACK,
                    direction="CALL",
                    symbol=sym,
                    headline=f"Pullback CALL — VWAP bounce in uptrend",
                    rationale=(
                        f"In TREND_UP regime, pullbacks to VWAP ({vwap_val:.2f}) are "
                        "buy opportunities. RSI not yet extended. "
                        "Wait for price to touch VWAP (or 1-bar close above) before entry."
                    ),
                    entry_level=entry,
                    stop_level=stop,
                    target_1=t1,
                    target_2=t2,
                    reward_risk=rr,
                    option_hint=_option_hint("CALL", entry, "0.35-0.50"),
                    suitable_for=["swing", "scalp"],
                    confidence=0.68,
                ))

        # ── Idea 3: Fade CALL (REVERSAL / overbought) ──────
        if regime_str in ("REVERSAL_EXHAUSTION", "TREND_DOWN") and pvwap == "below":
            entry = round(spot - atr * 0.15, 2)
            stop  = round(entry + atr * 0.5, 2)
            t1    = round(entry - atr * 0.8, 2)
            t2    = round(entry - atr * 1.5, 2)
            rr    = _rr(entry, stop, t1)
            conf  = min(0.80, getattr(r, "confidence", 0.5) + 0.05)
            if rr >= 1.5:
                ideas.append(TradeIdea(
                    idea_type=IdeaType.FADE,
                    direction="PUT",
                    symbol=sym,
                    headline=f"Fade PUT — {regime_str} below VWAP",
                    rationale=(
                        f"{sym} is in {regime_str}, trading below VWAP ({vwap_val:.2f}). "
                        "Look to fade any bounces that fail to reclaim VWAP."
                    ),
                    entry_level=entry,
                    stop_level=stop,
                    target_1=t1,
                    target_2=t2,
                    reward_risk=rr,
                    option_hint=_option_hint("PUT", entry),
                    suitable_for=["0dte", "swing"],
                    confidence=conf,
                    warning="" if rsi > 35 else f"RSI extended bearish at {rsi:.0f} — wait for bounce to fade",
                ))

        # ── Idea 4: Support Bounce (near SR) ───────────────
        if ns_price and abs(spot - ns_price) / spot < 0.01 and bias.direction != "bearish":
            entry = round(ns_price + atr * 0.1, 2)
            stop  = round(ns_price - atr * 0.35, 2)
            t1    = round(entry + atr * 0.7, 2)
            t2    = round(entry + atr * 1.4, 2)
            rr    = _rr(entry, stop, t1)
            if rr >= 1.3:
                ideas.append(TradeIdea(
                    idea_type=IdeaType.RANGE_PLAY,
                    direction="CALL",
                    symbol=sym,
                    headline=f"Support Bounce CALL — near {ns_price:.2f}",
                    rationale=(
                        f"Price is testing key support at {ns_price:.2f}. "
                        "A bullish reaction candle here creates a defined-risk call entry."
                    ),
                    entry_level=entry,
                    stop_level=stop,
                    target_1=t1,
                    target_2=t2,
                    reward_risk=rr,
                    option_hint=_option_hint("CALL", entry, "0.35-0.50"),
                    suitable_for=["swing", "scalp"],
                    confidence=0.60,
                ))

        # ── Idea 5: Resistance Fade (near SR) ──────────────
        if nr_price and abs(nr_price - spot) / spot < 0.01 and regime_str not in ("BREAKOUT_ATTEMPT",):
            entry = round(nr_price - atr * 0.1, 2)
            stop  = round(nr_price + atr * 0.35, 2)
            t1    = round(entry - atr * 0.7, 2)
            t2    = round(entry - atr * 1.4, 2)
            rr    = _rr(entry, stop, t1)
            if rr >= 1.3:
                ideas.append(TradeIdea(
                    idea_type=IdeaType.FADE,
                    direction="PUT",
                    symbol=sym,
                    headline=f"Resistance Fade PUT — near {nr_price:.2f}",
                    rationale=(
                        f"Price is approaching key resistance at {nr_price:.2f}. "
                        "If price fails here with a rejection candle, puts offer a clean short-term play."
                    ),
                    entry_level=entry,
                    stop_level=stop,
                    target_1=t1,
                    target_2=t2,
                    reward_risk=rr,
                    option_hint=_option_hint("PUT", entry, "0.35-0.50"),
                    suitable_for=["scalp", "0dte"],
                    confidence=0.58,
                ))

        # Score all ideas with the 5-component system
        ideas = [self._score_idea(idea, inp, bias) for idea in ideas]

        # Sort by score descending, then confidence
        ideas.sort(key=lambda i: (i.score, i.confidence), reverse=True)
        return ideas[:5]

    def _score_idea(
        self,
        idea: TradeIdea,
        inp: BriefingInput,
        bias: BiasSummary,
    ) -> TradeIdea:
        """Apply 5-component scoring to a trade idea."""
        r    = inp.regime
        cm   = inp.chain_metrics
        vpa  = inp.vpa_bias or {}

        regime_str = ""
        regime_conf = 0.5
        if r:
            rv = getattr(r, "regime", "")
            regime_str = rv.value if hasattr(rv, "value") else str(rv)
            regime_conf = float(getattr(r, "confidence", 0.5) or 0.5)

        # 1. Structure Score (0–30): R:R + proximity to S/R
        struct = 0
        rr = idea.reward_risk
        if rr >= 3.0:
            struct = 25
        elif rr >= 2.0:
            struct = 20
        elif rr >= 1.5:
            struct = 14
        elif rr >= 1.0:
            struct = 8
        # Bonus if near S/R (stop is tight)
        risk_amt = abs(idea.entry_level - idea.stop_level)
        if risk_amt > 0 and idea.entry_level > 0:
            risk_pct = risk_amt / idea.entry_level
            if risk_pct < 0.005:   # very tight stop = good structure
                struct = min(30, struct + 5)

        # 2. Regime Alignment (0–20): does idea match regime + bias direction?
        regime_align = 0
        idea_bullish = idea.direction == "CALL"
        regime_bullish = "UP" in regime_str or "BREAKOUT" in regime_str
        regime_bearish = "DOWN" in regime_str
        if idea_bullish and regime_bullish:
            regime_align = int(regime_conf * 20)
        elif not idea_bullish and regime_bearish:
            regime_align = int(regime_conf * 20)
        elif bias.direction == "bullish" and idea_bullish:
            regime_align = 10
        elif bias.direction == "bearish" and not idea_bullish:
            regime_align = 10
        # Penalty for counter-trend
        if idea_bullish and regime_bearish:
            regime_align = max(0, regime_align - 5)
        if not idea_bullish and regime_bullish:
            regime_align = max(0, regime_align - 5)

        # 3. Volume Confirmation (0–15)
        vol_conf = 0
        vol_regime = (inp.vpa_bias or {}).get("vol_regime", (getattr(inp, "vol_regime", None) or {}))
        vol_name = ""
        if isinstance(vol_regime, dict):
            vol_name = vol_regime.get("regime", "NORMAL")
        elif isinstance(vol_regime, str):
            vol_name = vol_regime
        # Fallback: check VPA bias alignment
        vpa_dir = vpa.get("bias", "neutral") or "neutral"
        if vol_name == "HIGH_RISK":
            vol_conf = 12  # high volume = conviction
        elif vol_name == "NORMAL":
            vol_conf = 8
        elif vol_name == "LOW":
            vol_conf = 3
        else:
            vol_conf = 8  # default
        # VPA bias alignment bonus
        if idea_bullish and ("bull" in vpa_dir):
            vol_conf = min(15, vol_conf + 3)
        elif not idea_bullish and ("bear" in vpa_dir):
            vol_conf = min(15, vol_conf + 3)

        # 4. Positioning Edge (0–15)
        pos_edge = 0
        if cm:
            gex = getattr(cm, "gex_regime", "neutral") or "neutral"
            ivr = float(getattr(cm, "iv_rank", 50) or 50)
            uoa = getattr(cm, "uoa_detected", False)
            # GEX alignment
            if gex == "positive" and idea.idea_type in (IdeaType.RANGE_PLAY, IdeaType.FADE):
                pos_edge += 5
            elif gex == "negative" and idea.idea_type in (IdeaType.BREAKOUT, IdeaType.MOMENTUM):
                pos_edge += 5
            # Cheap IV = better for buying
            if ivr < 30:
                pos_edge += 5
            elif ivr > 70:
                pos_edge += 2  # premium selling edge
            # UOA
            if uoa:
                pos_edge += 5
        pos_edge = min(15, pos_edge)

        # 5. Risk/Reward Score (0–20)
        rr_score = 0
        if rr >= 3.0:
            rr_score = 20
        elif rr >= 2.5:
            rr_score = 17
        elif rr >= 2.0:
            rr_score = 14
        elif rr >= 1.5:
            rr_score = 10
        elif rr >= 1.0:
            rr_score = 5

        total = struct + regime_align + vol_conf + pos_edge + rr_score
        total = max(0, min(100, total))

        idea.score = total
        idea.confidence = round(total / 100.0, 2)
        idea.score_breakdown = {
            "structure": struct,
            "regime_alignment": regime_align,
            "volume_confirmation": vol_conf,
            "positioning_edge": pos_edge,
            "risk_reward": rr_score,
        }
        return idea

    # ── Playbook filter ───────────────────────────────────────

    def filter_by_playbook(
        self,
        briefing: DailyBriefing,
        mode: PlaybookMode,
    ) -> DailyBriefing:
        """Return a copy of briefing with ideas/plans filtered to the given playbook mode."""
        if mode == PlaybookMode.ALL:
            return briefing

        mode_str = mode.value
        briefing.trade_ideas = [
            i for i in briefing.trade_ideas
            if not i.suitable_for or mode_str in i.suitable_for
        ]
        briefing.conditional_plans = [
            p for p in briefing.conditional_plans
            if not p.suitable_for or mode_str in p.suitable_for
        ]
        return briefing


# Singleton
idea_engine = IdeaEngine()
