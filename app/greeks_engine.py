"""
Greeks Signal Engine – Composite Options Signal Scoring
========================================================
Combines multiple independent edge sources into a single
composite score for options trading.

Signal Factors
──────────────
1. IV Rank / IV Percentile  (strongest single predictor)
2. Gamma Exposure (GEX)     (dealer positioning → regime)
3. Greeks Composite          (Γ/Θ ratio, Vega×IVR, Delta sweet-spot)
4. Unusual Options Activity  (smart money detection)
5. Put/Call OI Skew          (sentiment extreme → contrarian)
6. VPA Integration           (timing / confirmation from VPA engine)

Polygon.io provides: Greeks, IV, open_interest, volume per contract
via the Options Snapshot API.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import math


# ── Signal taxonomy ─────────────────────────────────────────

class CompositeSignal(Enum):
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    LEAN_BULLISH = "lean_bullish"
    NEUTRAL = "neutral"
    LEAN_BEARISH = "lean_bearish"
    SELL = "sell"
    STRONG_SELL = "strong_sell"


class TradeArchetype(Enum):
    MEAN_REVERSION_SELL = "mean_reversion_sell"       # IV high + positive GEX
    DIRECTIONAL_BREAKOUT = "directional_breakout"     # IV low + negative GEX + UOA
    PINNING_PLAY = "pinning_play"                     # Positive GEX + near max pain
    VOLATILITY_EXPANSION = "volatility_expansion"     # IV low + backwardation
    PREMIUM_HARVEST = "premium_harvest"               # IV high + theta dominant
    NO_EDGE = "no_edge"                               # No clear setup


@dataclass
class GreeksData:
    """Parsed Greeks from Polygon snapshot."""
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    iv: float = 0.0               # Implied volatility (decimal, e.g. 0.35 = 35%)
    open_interest: int = 0
    volume: int = 0
    underlying_price: float = 0.0
    break_even: float = 0.0
    last_price: float = 0.0


@dataclass
class ChainMetrics:
    """Aggregate metrics computed from the full options chain."""
    iv_rank: float = 0.0          # 0-100, where current IV sits in chain
    iv_percentile: float = 0.0    # 0-100, % of strikes with lower IV
    put_call_oi_ratio: float = 1.0
    put_call_volume_ratio: float = 1.0
    total_call_oi: int = 0
    total_put_oi: int = 0
    total_call_volume: int = 0
    total_put_volume: int = 0
    net_gex: float = 0.0          # Net Gamma Exposure
    gex_regime: str = "neutral"   # "positive" | "negative" | "neutral"
    max_pain: float = 0.0         # Strike with max pain
    uoa_detected: bool = False    # Unusual options activity
    uoa_details: list = field(default_factory=list)
    weighted_iv: float = 0.0      # OI-weighted average IV


@dataclass
class FactorScore:
    """Individual factor contribution to composite score."""
    name: str
    score: float          # -1.0 to +1.0 (bearish to bullish)
    confidence: float     # 0.0 to 1.0
    weight: float         # Factor weight in composite
    detail: str           # Human-readable explanation


@dataclass
class CompositeResult:
    """Final composite analysis result."""
    signal: CompositeSignal
    score: float                    # -1.0 to +1.0
    confidence: float               # 0.0 to 1.0
    trade_archetype: TradeArchetype
    archetype_description: str
    factors: list[FactorScore]
    greeks: GreeksData
    chain_metrics: ChainMetrics
    recommendation: str             # Actionable summary


# ── Engine ───────────────────────────────────────────────────

class GreeksSignalEngine:
    """
    Composite options signal engine that combines Greeks, IV surface,
    dealer positioning (GEX), and flow data into a unified score.
    """

    # Factor weights (sum to 1.0)
    WEIGHTS = {
        "iv_rank": 0.25,
        "gex": 0.20,
        "greeks_composite": 0.20,
        "uoa_flow": 0.15,
        "vpa_bias": 0.15,
        "pc_skew": 0.05,
    }

    # ── public API ───────────────────────────────────────────

    def analyze(
        self,
        contract_snapshot: dict,
        chain_data: list[dict],
        vpa_bias: dict | None = None,
        contract_type: str = "C",
    ) -> CompositeResult:
        """
        Run full composite analysis.

        Parameters
        ----------
        contract_snapshot : dict from PolygonClient.get_option_contract_snapshot()
        chain_data : list[dict] from PolygonClient.get_options_chain_snapshot()
        vpa_bias : dict with keys {bias, strength} from VPAEngine.get_bias()
        contract_type : "C" for call, "P" for put

        Returns
        -------
        CompositeResult with signal, score, factors, and recommendation.
        """
        greeks = self._parse_greeks(contract_snapshot)
        chain_metrics = self._compute_chain_metrics(chain_data, greeks.underlying_price)

        factors: list[FactorScore] = []

        # 1. IV Rank factor
        factors.append(self._score_iv_rank(chain_metrics, contract_type))

        # 2. GEX / Dealer Positioning factor
        factors.append(self._score_gex(chain_metrics))

        # 3. Greeks Composite factor
        factors.append(self._score_greeks_composite(greeks, chain_metrics, contract_type))

        # 4. Unusual Options Activity factor
        factors.append(self._score_uoa(chain_metrics))

        # 5. VPA Bias integration
        factors.append(self._score_vpa_bias(vpa_bias))

        # 6. Put/Call Skew factor
        factors.append(self._score_pc_skew(chain_metrics))

        # Compute composite score
        composite_score = sum(f.score * f.confidence * f.weight for f in factors)
        overall_confidence = self._compute_confidence(factors)

        # Determine signal
        signal = self._classify_signal(composite_score, overall_confidence)

        # Determine trade archetype
        archetype, arch_desc = self._determine_archetype(
            chain_metrics, greeks, composite_score, vpa_bias
        )

        # Generate recommendation
        recommendation = self._generate_recommendation(
            signal, archetype, greeks, chain_metrics, factors, contract_type
        )

        return CompositeResult(
            signal=signal,
            score=round(composite_score, 3),
            confidence=round(overall_confidence, 2),
            trade_archetype=archetype,
            archetype_description=arch_desc,
            factors=factors,
            greeks=greeks,
            chain_metrics=chain_metrics,
            recommendation=recommendation,
        )

    # ── Greeks parsing ───────────────────────────────────────

    def _parse_greeks(self, snapshot: dict) -> GreeksData:
        """Parse a contract snapshot into GreeksData."""
        if not snapshot:
            return GreeksData()

        greeks_raw = snapshot.get("greeks", {})
        return GreeksData(
            delta=float(greeks_raw.get("delta", 0) or 0),
            gamma=float(greeks_raw.get("gamma", 0) or 0),
            theta=float(greeks_raw.get("theta", 0) or 0),
            vega=float(greeks_raw.get("vega", 0) or 0),
            iv=float(snapshot.get("iv", 0) or 0),
            open_interest=int(snapshot.get("open_interest", 0) or 0),
            volume=int(snapshot.get("volume", 0) or 0),
            underlying_price=float(snapshot.get("underlying_price", 0) or 0),
            break_even=float(snapshot.get("break_even", 0) or 0),
            last_price=float(snapshot.get("last_price", 0) or 0),
        )

    # ── Chain metrics computation ────────────────────────────

    def _compute_chain_metrics(
        self, chain: list[dict], underlying_price: float
    ) -> ChainMetrics:
        """Compute aggregate metrics from all contracts in the chain."""
        if not chain:
            return ChainMetrics()

        metrics = ChainMetrics()

        # Separate calls and puts
        calls = [c for c in chain if c.get("contract_type", "").lower() == "call"]
        puts = [c for c in chain if c.get("contract_type", "").lower() == "put"]

        # ── Put/Call ratios ────────────────────────────
        metrics.total_call_oi = sum(c.get("open_interest", 0) or 0 for c in calls)
        metrics.total_put_oi = sum(c.get("open_interest", 0) or 0 for c in puts)
        metrics.total_call_volume = sum(c.get("volume", 0) or 0 for c in calls)
        metrics.total_put_volume = sum(c.get("volume", 0) or 0 for c in puts)

        if metrics.total_call_oi > 0:
            metrics.put_call_oi_ratio = metrics.total_put_oi / metrics.total_call_oi
        if metrics.total_call_volume > 0:
            metrics.put_call_volume_ratio = (
                metrics.total_put_volume / metrics.total_call_volume
            )

        # ── IV distribution (for IV Rank / Percentile) ─
        all_ivs = [
            c.get("iv", 0) for c in chain
            if c.get("iv") and c["iv"] > 0 and c.get("open_interest", 0) > 10
        ]

        if all_ivs:
            iv_sorted = sorted(all_ivs)
            iv_min = iv_sorted[0]
            iv_max = iv_sorted[-1]
            iv_median = iv_sorted[len(iv_sorted) // 2]

            # Use OI-weighted IV as "current IV" estimate
            total_oi_weight = 0
            weighted_iv_sum = 0.0
            for c in chain:
                oi = c.get("open_interest", 0) or 0
                iv = c.get("iv", 0) or 0
                if oi > 0 and iv > 0:
                    weighted_iv_sum += iv * oi
                    total_oi_weight += oi

            current_iv = weighted_iv_sum / total_oi_weight if total_oi_weight > 0 else iv_median
            metrics.weighted_iv = round(current_iv, 4)

            # IV Rank across the chain
            if iv_max > iv_min:
                metrics.iv_rank = round(
                    (current_iv - iv_min) / (iv_max - iv_min) * 100, 1
                )
            else:
                metrics.iv_rank = 50.0

            # IV Percentile
            below_count = sum(1 for iv in iv_sorted if iv < current_iv)
            metrics.iv_percentile = round(below_count / len(iv_sorted) * 100, 1)

        # ── GEX (Gamma Exposure) ───────────────────────
        # GEX = Σ (gamma × OI × 100 × spot²) for calls (positive)
        #     - Σ (gamma × OI × 100 × spot²) for puts (negative)
        spot = underlying_price or 1e-6
        net_gex = 0.0
        for c in chain:
            gamma = c.get("greeks", {}).get("gamma", 0) or 0
            oi = c.get("open_interest", 0) or 0
            ct = c.get("contract_type", "").lower()
            # Call gamma is positive GEX, put gamma is negative GEX
            if ct == "call":
                net_gex += gamma * oi * 100 * spot * 0.01
            elif ct == "put":
                net_gex -= gamma * oi * 100 * spot * 0.01

        metrics.net_gex = round(net_gex, 2)
        if net_gex > 1000:
            metrics.gex_regime = "positive"
        elif net_gex < -1000:
            metrics.gex_regime = "negative"
        else:
            metrics.gex_regime = "neutral"

        # ── Max Pain ───────────────────────────────────
        metrics.max_pain = self._calculate_max_pain(calls, puts)

        # ── Unusual Options Activity ───────────────────
        self._detect_uoa(chain, metrics)

        return metrics

    def _calculate_max_pain(
        self, calls: list[dict], puts: list[dict]
    ) -> float:
        """Calculate max pain strike (strike where total option value is minimized)."""
        strikes: set[float] = set()
        for c in calls + puts:
            s = c.get("strike", 0)
            if s > 0:
                strikes.add(s)

        if not strikes:
            return 0.0

        min_pain = float("inf")
        max_pain_strike = 0.0

        for test_price in sorted(strikes):
            total_pain = 0.0
            for c in calls:
                strike = c.get("strike", 0)
                oi = c.get("open_interest", 0) or 0
                if test_price > strike:
                    total_pain += (test_price - strike) * oi * 100
            for p in puts:
                strike = p.get("strike", 0)
                oi = p.get("open_interest", 0) or 0
                if test_price < strike:
                    total_pain += (strike - test_price) * oi * 100

            if total_pain < min_pain:
                min_pain = total_pain
                max_pain_strike = test_price

        return max_pain_strike

    def _detect_uoa(self, chain: list[dict], metrics: ChainMetrics) -> None:
        """Detect unusual options activity (Volume/OI > 3x, or large absolute volume)."""
        uoa_strikes: list[dict] = []

        for c in chain:
            vol = c.get("volume", 0) or 0
            oi = c.get("open_interest", 0) or 0
            strike = c.get("strike", 0)
            ct = c.get("contract_type", "")

            # Skip low-liquidity contracts
            if vol < 100:
                continue

            vol_oi_ratio = vol / max(oi, 1)

            # UOA criteria: Vol/OI > 3x, or absolute volume > 5000
            if vol_oi_ratio > 3.0 or vol > 5000:
                uoa_strikes.append({
                    "strike": strike,
                    "type": ct,
                    "volume": vol,
                    "open_interest": oi,
                    "vol_oi_ratio": round(vol_oi_ratio, 1),
                    "iv": c.get("iv", 0),
                })

        # Sort by vol/oi ratio descending, keep top 5
        uoa_strikes.sort(key=lambda x: x["vol_oi_ratio"], reverse=True)
        metrics.uoa_details = uoa_strikes[:5]
        metrics.uoa_detected = len(uoa_strikes) > 0

    # ── Individual factor scoring ────────────────────────────

    def _score_iv_rank(
        self, metrics: ChainMetrics, contract_type: str
    ) -> FactorScore:
        """
        IV Rank is the #1 predictor in options.
        High IV Rank → sell premium (mean reversion).
        Low IV Rank → buy premium (cheap options).
        """
        ivr = metrics.iv_rank

        if ivr >= 80:
            # Very high IV → bearish for option buyers, bullish for sellers
            score = -0.8 if contract_type == "C" else 0.3
            detail = f"IV Rank {ivr:.0f}% – extremely high, premium overpriced, favor selling"
            confidence = 0.85
        elif ivr >= 60:
            score = -0.4 if contract_type == "C" else 0.1
            detail = f"IV Rank {ivr:.0f}% – elevated, slight edge to sellers"
            confidence = 0.65
        elif ivr <= 20:
            # Very low IV → great for buying directional
            score = 0.7 if contract_type == "C" else -0.4
            detail = f"IV Rank {ivr:.0f}% – very low, options are cheap, favor buying"
            confidence = 0.80
        elif ivr <= 40:
            score = 0.3 if contract_type == "C" else -0.1
            detail = f"IV Rank {ivr:.0f}% – below average, slight edge to buyers"
            confidence = 0.55
        else:
            score = 0.0
            detail = f"IV Rank {ivr:.0f}% – neutral zone"
            confidence = 0.40

        return FactorScore(
            name="IV Rank",
            score=round(score, 2),
            confidence=confidence,
            weight=self.WEIGHTS["iv_rank"],
            detail=detail,
        )

    def _score_gex(self, metrics: ChainMetrics) -> FactorScore:
        """
        GEX regime determines market character:
        Positive GEX → mean-reverting, low vol, pinning
        Negative GEX → trending, high vol, breakouts
        """
        regime = metrics.gex_regime

        if regime == "positive":
            score = 0.0  # Neutral direction; but regime matters for archetype
            detail = (
                f"Positive GEX ({metrics.net_gex:,.0f}) – dealers long gamma, "
                f"expect mean-reversion & lower realized vol"
            )
            confidence = 0.70
        elif regime == "negative":
            score = 0.0
            detail = (
                f"Negative GEX ({metrics.net_gex:,.0f}) – dealers short gamma, "
                f"expect amplified moves & higher realized vol"
            )
            confidence = 0.70
        else:
            score = 0.0
            detail = f"Neutral GEX ({metrics.net_gex:,.0f}) – no strong dealer positioning"
            confidence = 0.30

        return FactorScore(
            name="GEX Regime",
            score=round(score, 2),
            confidence=confidence,
            weight=self.WEIGHTS["gex"],
            detail=detail,
        )

    def _score_greeks_composite(
        self, greeks: GreeksData, metrics: ChainMetrics, contract_type: str
    ) -> FactorScore:
        """
        Composite score from individual Greeks:
        - Gamma/Theta ratio (bang for buck)
        - Vega × IV Rank (vol edge)
        - Delta sweet spot (0.30-0.40 for high R:R)
        """
        score = 0.0
        details = []
        sub_confidence = 0.0
        n_factors = 0

        # ── Gamma/Theta ratio ────────────────────────
        theta_abs = abs(greeks.theta) if greeks.theta != 0 else 0.001
        gamma_theta = abs(greeks.gamma) / theta_abs if theta_abs > 0.0001 else 0

        if gamma_theta > 2.0:
            score += 0.3  # Good for buyers
            details.append(f"Γ/Θ={gamma_theta:.1f} (favorable for buyers)")
            sub_confidence += 0.7
        elif gamma_theta < 0.5:
            score -= 0.2  # Theta dominant – good for sellers
            details.append(f"Γ/Θ={gamma_theta:.1f} (theta dominant, favor selling)")
            sub_confidence += 0.6
        else:
            details.append(f"Γ/Θ={gamma_theta:.1f} (balanced)")
            sub_confidence += 0.4
        n_factors += 1

        # ── Vega × IV Rank ───────────────────────────
        vega_ivr = greeks.vega * (metrics.iv_rank / 100) if metrics.iv_rank > 0 else 0

        if metrics.iv_rank > 70 and greeks.vega > 0:
            # High IV + long vega = bad (IV likely to contract)
            score -= 0.3
            details.append(f"Vega×IVR={vega_ivr:.3f} – long vega in high IV, risk of IV crush")
            sub_confidence += 0.75
        elif metrics.iv_rank < 30 and greeks.vega > 0:
            # Low IV + long vega = good (cheap vol)
            score += 0.3
            details.append(f"Vega×IVR={vega_ivr:.3f} – long vega in low IV, vol expansion likely")
            sub_confidence += 0.70
        else:
            details.append(f"Vega×IVR={vega_ivr:.3f}")
            sub_confidence += 0.35
        n_factors += 1

        # ── Delta sweet spot ─────────────────────────
        delta_abs = abs(greeks.delta)

        if 0.25 <= delta_abs <= 0.45:
            score += 0.2
            details.append(f"Delta={greeks.delta:.2f} – sweet spot for R:R")
            sub_confidence += 0.6
        elif delta_abs > 0.80:
            score -= 0.1
            details.append(f"Delta={greeks.delta:.2f} – deep ITM, stock-like, low leverage")
            sub_confidence += 0.5
        elif delta_abs < 0.10:
            score -= 0.2
            details.append(f"Delta={greeks.delta:.2f} – far OTM, low probability")
            sub_confidence += 0.6
        else:
            details.append(f"Delta={greeks.delta:.2f}")
            sub_confidence += 0.4
        n_factors += 1

        # ── Theta as % of premium ────────────────────
        if greeks.last_price > 0:
            theta_pct = abs(greeks.theta) / greeks.last_price * 100
            if theta_pct > 3.0:
                score -= 0.15
                details.append(f"Theta/Premium={theta_pct:.1f}%/day – rapid decay")
                sub_confidence += 0.65
            elif theta_pct < 0.5:
                score += 0.1
                details.append(f"Theta/Premium={theta_pct:.1f}%/day – minimal decay")
                sub_confidence += 0.5
            else:
                details.append(f"Theta/Premium={theta_pct:.1f}%/day")
                sub_confidence += 0.35
            n_factors += 1

        avg_confidence = sub_confidence / max(n_factors, 1)

        return FactorScore(
            name="Greeks Composite",
            score=round(max(-1, min(1, score)), 2),
            confidence=round(avg_confidence, 2),
            weight=self.WEIGHTS["greeks_composite"],
            detail=" | ".join(details),
        )

    def _score_uoa(self, metrics: ChainMetrics) -> FactorScore:
        """
        Unusual Options Activity – smart money detection.
        Vol/OI > 3x on concentrated strikes signals conviction.
        """
        if not metrics.uoa_detected:
            return FactorScore(
                name="Flow / UOA",
                score=0.0,
                confidence=0.30,
                weight=self.WEIGHTS["uoa_flow"],
                detail="No unusual options activity detected",
            )

        # Determine directional bias from UOA
        call_uoa = sum(1 for u in metrics.uoa_details if u["type"].lower() == "call")
        put_uoa = sum(1 for u in metrics.uoa_details if u["type"].lower() == "put")
        total_uoa = len(metrics.uoa_details)

        max_ratio = max((u["vol_oi_ratio"] for u in metrics.uoa_details), default=0)

        if call_uoa > put_uoa:
            score = min(0.8, 0.3 + (max_ratio - 3) * 0.05)
            detail = (
                f"UOA detected: {call_uoa} call / {put_uoa} put sweeps, "
                f"max Vol/OI {max_ratio:.1f}x – bullish flow"
            )
        elif put_uoa > call_uoa:
            score = max(-0.8, -0.3 - (max_ratio - 3) * 0.05)
            detail = (
                f"UOA detected: {put_uoa} put / {call_uoa} call sweeps, "
                f"max Vol/OI {max_ratio:.1f}x – bearish flow"
            )
        else:
            score = 0.0
            detail = f"UOA detected: mixed call/put activity ({total_uoa} strikes)"

        confidence = min(0.85, 0.5 + total_uoa * 0.05)

        return FactorScore(
            name="Flow / UOA",
            score=round(score, 2),
            confidence=round(confidence, 2),
            weight=self.WEIGHTS["uoa_flow"],
            detail=detail,
        )

    def _score_vpa_bias(self, vpa_bias: dict | None) -> FactorScore:
        """Integrate VPA engine bias as a timing factor."""
        if not vpa_bias or vpa_bias.get("bias") == "neutral":
            return FactorScore(
                name="VPA Bias",
                score=0.0,
                confidence=0.30,
                weight=self.WEIGHTS["vpa_bias"],
                detail="VPA neutral – no strong price-volume signals",
            )

        bias = vpa_bias["bias"]
        strength = vpa_bias.get("strength", 0.5)
        reason = vpa_bias.get("reason", "")

        if bias == "bullish":
            score = min(0.9, 0.3 + strength * 0.6)
        elif bias == "bearish":
            score = max(-0.9, -0.3 - strength * 0.6)
        else:
            score = 0.0

        return FactorScore(
            name="VPA Bias",
            score=round(score, 2),
            confidence=round(min(0.85, 0.4 + strength * 0.4), 2),
            weight=self.WEIGHTS["vpa_bias"],
            detail=f"VPA {bias} (strength {strength:.0%}) – {reason}",
        )

    def _score_pc_skew(self, metrics: ChainMetrics) -> FactorScore:
        """
        Put/Call OI ratio extremes are contrarian signals.
        P/C > 1.5 → too much fear → contrarian bullish
        P/C < 0.5 → too much greed → contrarian bearish
        """
        pcr = metrics.put_call_oi_ratio

        if pcr > 1.5:
            score = 0.5  # Contrarian bullish
            detail = f"P/C OI ratio {pcr:.2f} – extreme put-heavy, contrarian bullish"
            confidence = 0.60
        elif pcr > 1.2:
            score = 0.2
            detail = f"P/C OI ratio {pcr:.2f} – moderately put-heavy"
            confidence = 0.45
        elif pcr < 0.5:
            score = -0.5  # Contrarian bearish
            detail = f"P/C OI ratio {pcr:.2f} – extreme call-heavy, contrarian bearish"
            confidence = 0.60
        elif pcr < 0.8:
            score = -0.2
            detail = f"P/C OI ratio {pcr:.2f} – moderately call-heavy"
            confidence = 0.45
        else:
            score = 0.0
            detail = f"P/C OI ratio {pcr:.2f} – balanced"
            confidence = 0.30

        return FactorScore(
            name="P/C Skew",
            score=round(score, 2),
            confidence=confidence,
            weight=self.WEIGHTS["pc_skew"],
            detail=detail,
        )

    # ── Composite computation ────────────────────────────────

    def _compute_confidence(self, factors: list[FactorScore]) -> float:
        """
        Overall confidence increases when multiple factors agree.
        Convergence bonus: 4+ factors in same direction → high confidence.
        """
        bullish_count = sum(1 for f in factors if f.score > 0.1 and f.confidence > 0.4)
        bearish_count = sum(1 for f in factors if f.score < -0.1 and f.confidence > 0.4)
        max_agreement = max(bullish_count, bearish_count)

        # Base: weighted average of individual confidences
        total_weight = sum(f.weight for f in factors)
        base_conf = (
            sum(f.confidence * f.weight for f in factors) / total_weight
            if total_weight > 0
            else 0.5
        )

        # Convergence bonus
        if max_agreement >= 5:
            bonus = 0.20
        elif max_agreement >= 4:
            bonus = 0.15
        elif max_agreement >= 3:
            bonus = 0.08
        else:
            bonus = 0.0

        return min(0.95, base_conf + bonus)

    def _classify_signal(self, score: float, confidence: float) -> CompositeSignal:
        """Map composite score to a signal classification."""
        effective = score * confidence

        if effective > 0.4:
            return CompositeSignal.STRONG_BUY
        elif effective > 0.2:
            return CompositeSignal.BUY
        elif effective > 0.05:
            return CompositeSignal.LEAN_BULLISH
        elif effective < -0.4:
            return CompositeSignal.STRONG_SELL
        elif effective < -0.2:
            return CompositeSignal.SELL
        elif effective < -0.05:
            return CompositeSignal.LEAN_BEARISH
        else:
            return CompositeSignal.NEUTRAL

    def _determine_archetype(
        self,
        metrics: ChainMetrics,
        greeks: GreeksData,
        composite_score: float,
        vpa_bias: dict | None,
    ) -> tuple[TradeArchetype, str]:
        """Determine the best trade archetype given the current setup."""
        ivr = metrics.iv_rank
        gex = metrics.gex_regime
        uoa = metrics.uoa_detected
        vpa = vpa_bias.get("bias", "neutral") if vpa_bias else "neutral"

        # Archetype 1: Mean-Reversion Sell (highest win rate ~80%)
        if ivr >= 70 and gex == "positive":
            return (
                TradeArchetype.MEAN_REVERSION_SELL,
                "IV is elevated with positive GEX (dealers dampen moves). "
                "High probability mean-reversion setup. "
                "Consider: Sell Iron Condor, Short Strangle, or Credit Spreads.",
            )

        # Archetype 2: Directional Breakout (highest R:R)
        if ivr <= 35 and gex == "negative" and (uoa or vpa != "neutral"):
            direction = "bullish" if composite_score > 0 else "bearish"
            return (
                TradeArchetype.DIRECTIONAL_BREAKOUT,
                f"Low IV + negative GEX + {'UOA flow' if uoa else 'VPA confirmation'}. "
                f"Breakout conditions ({direction}). "
                f"Consider: Buy 30-40 delta {'calls' if direction == 'bullish' else 'puts'}, 30-45 DTE.",
            )

        # Archetype 3: Pinning Play
        if gex == "positive" and metrics.max_pain > 0:
            distance_to_pain = abs(greeks.underlying_price - metrics.max_pain)
            pct_distance = distance_to_pain / greeks.underlying_price * 100 if greeks.underlying_price else 99
            if pct_distance < 2.0:
                return (
                    TradeArchetype.PINNING_PLAY,
                    f"Positive GEX with price near max-pain (${metrics.max_pain:.0f}, "
                    f"{pct_distance:.1f}% away). Expect pinning action. "
                    f"Consider: Sell ATM Butterfly or Short Straddle.",
                )

        # Archetype 4: Volatility Expansion
        if ivr <= 25:
            return (
                TradeArchetype.VOLATILITY_EXPANSION,
                "IV is very low – options are cheap. Volatility likely to expand. "
                "Consider: Long Straddle/Strangle, Calendar Spreads, or Directional debit spreads.",
            )

        # Archetype 5: Premium Harvest
        if ivr >= 65:
            return (
                TradeArchetype.PREMIUM_HARVEST,
                "IV is elevated – premium is rich. "
                "Consider: Credit Spreads, Covered Calls, Cash-Secured Puts.",
            )

        return (
            TradeArchetype.NO_EDGE,
            "No strong multi-factor convergence detected. "
            "Conditions are mixed – consider waiting for a clearer setup.",
        )

    def _generate_recommendation(
        self,
        signal: CompositeSignal,
        archetype: TradeArchetype,
        greeks: GreeksData,
        metrics: ChainMetrics,
        factors: list[FactorScore],
        contract_type: str,
    ) -> str:
        """Generate an actionable recommendation string."""
        # Count agreeing factors
        bullish = sum(1 for f in factors if f.score > 0.1 and f.confidence > 0.4)
        bearish = sum(1 for f in factors if f.score < -0.1 and f.confidence > 0.4)
        agreement = max(bullish, bearish)
        direction = "bullish" if bullish > bearish else "bearish" if bearish > bullish else "mixed"

        parts = [f"{agreement}/6 factors align {direction}."]

        # Key metrics summary
        parts.append(
            f"IV Rank: {metrics.iv_rank:.0f}% | "
            f"Delta: {greeks.delta:.2f} | "
            f"Γ/Θ: {abs(greeks.gamma) / max(abs(greeks.theta), 0.001):.1f} | "
            f"GEX: {metrics.gex_regime}"
        )

        if metrics.uoa_detected:
            parts.append(f"⚡ UOA detected on {len(metrics.uoa_details)} strikes")

        # Signal-specific advice
        if signal in (CompositeSignal.STRONG_BUY, CompositeSignal.BUY):
            parts.append(
                f"✅ Conditions favor {'long calls' if contract_type == 'C' else 'short puts'}."
            )
        elif signal in (CompositeSignal.STRONG_SELL, CompositeSignal.SELL):
            parts.append(
                f"🔻 Conditions favor {'short calls' if contract_type == 'C' else 'long puts'}."
            )
        else:
            parts.append("⏸ Mixed signals – consider waiting or reduce position size.")

        return " ".join(parts)


# Global engine instance
greeks_engine = GreeksSignalEngine()
