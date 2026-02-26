"""
Chain Analytics
================
Pure chain-level computation — no legacy composite scoring dependencies.

All functions operate on the raw options chain snapshot list returned
by PolygonClient.get_options_chain_snapshot().

Each chain item is a dict with at minimum:
    contract_type, strike, open_interest, volume, iv,
    greeks: {delta, gamma, theta, vega},
    last_price, bid, ask, underlying_price
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional


# ── Data classes ─────────────────────────────────────────────

@dataclass
class ChainMetrics:
    """Aggregate metrics derived from the full options chain."""
    # IV statistics
    iv_rank: float = 0.0            # 0-100: where current IV sits in chain
    iv_percentile: float = 0.0      # 0-100: % of strikes below current IV
    weighted_iv: float = 0.0        # OI-weighted average IV

    # Put / Call ratios
    put_call_oi_ratio: float = 1.0
    put_call_volume_ratio: float = 1.0
    total_call_oi: int = 0
    total_put_oi: int = 0
    total_call_volume: int = 0
    total_put_volume: int = 0

    # Dealer positioning
    net_gex: float = 0.0            # Net Gamma Exposure (dollars)
    gex_regime: str = "neutral"     # "positive" | "negative" | "neutral"

    # Key price levels from the chain
    max_pain: float = 0.0           # Strike with minimum total option value

    # Unusual activity
    uoa_detected: bool = False
    uoa_details: list = field(default_factory=list)  # top 5 unusual contracts

    # Call / put walls (highest OI strikes)
    call_wall: float = 0.0          # Highest OI call strike
    put_wall: float = 0.0           # Highest OI put strike

    # Top walls + gamma magnets (for Idea Engine)
    top_call_walls: list = field(default_factory=list)   # top-N call strikes by OI
    top_put_walls: list = field(default_factory=list)    # top-N put strikes by OI
    gamma_magnets: list = field(default_factory=list)    # strikes with highest combined gamma


@dataclass
class LiquidityScore:
    """Per-contract liquidity assessment."""
    ticker: str
    score: float            # 0.0-1.0
    spread_pct: float       # bid-ask as % of mid
    volume: int
    open_interest: int
    passes_gate: bool       # Meets minimum tradability threshold
    reason: str             # Why it passed/failed


# ── Main computation ─────────────────────────────────────────

def compute_chain_metrics(
    chain: list[dict],
    underlying_price: float = 0.0,
) -> ChainMetrics:
    """
    Compute aggregate metrics from all contracts in the options chain.

    Parameters
    ----------
    chain : list[dict] from PolygonClient.get_options_chain_snapshot()
    underlying_price : current underlying price (for GEX computation)

    Returns
    -------
    ChainMetrics
    """
    if not chain:
        return ChainMetrics()

    metrics = ChainMetrics()

    calls = [c for c in chain if c.get("contract_type", "").lower() == "call"]
    puts  = [c for c in chain if c.get("contract_type", "").lower() == "put"]

    # ── Put / Call counts ─────────────────────────────────
    metrics.total_call_oi     = sum(c.get("open_interest", 0) or 0 for c in calls)
    metrics.total_put_oi      = sum(c.get("open_interest", 0) or 0 for c in puts)
    metrics.total_call_volume = sum(c.get("volume", 0) or 0 for c in calls)
    metrics.total_put_volume  = sum(c.get("volume", 0) or 0 for c in puts)

    if metrics.total_call_oi > 0:
        metrics.put_call_oi_ratio = metrics.total_put_oi / metrics.total_call_oi
    if metrics.total_call_volume > 0:
        metrics.put_call_volume_ratio = metrics.total_put_volume / metrics.total_call_volume

    # ── IV distribution ────────────────────────────────────
    all_ivs = [
        c.get("iv", 0)
        for c in chain
        if c.get("iv") and c["iv"] > 0 and (c.get("open_interest", 0) or 0) > 10
    ]

    if all_ivs:
        iv_sorted = sorted(all_ivs)
        iv_min = iv_sorted[0]
        iv_max = iv_sorted[-1]
        iv_median = iv_sorted[len(iv_sorted) // 2]

        # OI-weighted current IV
        total_oi_w, weighted_iv_sum = 0, 0.0
        for c in chain:
            oi  = c.get("open_interest", 0) or 0
            iv  = c.get("iv", 0) or 0
            if oi > 0 and iv > 0:
                weighted_iv_sum += iv * oi
                total_oi_w += oi

        current_iv = weighted_iv_sum / total_oi_w if total_oi_w > 0 else iv_median
        metrics.weighted_iv = round(current_iv, 4)

        metrics.iv_rank = round(
            (current_iv - iv_min) / (iv_max - iv_min) * 100, 1
        ) if iv_max > iv_min else 50.0

        below = sum(1 for iv in iv_sorted if iv < current_iv)
        metrics.iv_percentile = round(below / len(iv_sorted) * 100, 1)

    # ── GEX ────────────────────────────────────────────────
    spot = underlying_price or 1e-6
    net_gex = 0.0
    for c in chain:
        gamma = (c.get("greeks") or {}).get("gamma", 0) or 0
        oi    = c.get("open_interest", 0) or 0
        ct    = c.get("contract_type", "").lower()
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

    # ── Max Pain ───────────────────────────────────────────
    metrics.max_pain = _calculate_max_pain(calls, puts)

    # ── Call / Put walls ───────────────────────────────────
    if calls:
        best_call = max(calls, key=lambda c: c.get("open_interest", 0) or 0)
        metrics.call_wall = float(best_call.get("strike", 0) or 0)
    if puts:
        best_put = max(puts, key=lambda c: c.get("open_interest", 0) or 0)
        metrics.put_wall = float(best_put.get("strike", 0) or 0)

    # ── Top walls + gamma magnets (Idea Engine) ────────────
    metrics.top_call_walls = find_top_call_walls(calls, underlying_price, top_n=3)
    metrics.top_put_walls  = find_top_put_walls(puts,  underlying_price, top_n=3)
    metrics.gamma_magnets  = find_gamma_magnets(chain, underlying_price, top_n=3)

    # ── UOA ────────────────────────────────────────────────
    _detect_uoa(chain, metrics)

    return metrics


# ── Per-contract helpers ─────────────────────────────────────

def compute_liquidity_score(contract: dict) -> LiquidityScore:
    """
    Assess how tradeable a contract is.
    Score 0-1; passes_gate=True if spread% ≤ 10% AND (volume ≥ 200 OR oi ≥ 500).

    Parameters
    ----------
    contract : dict — single options chain item from Polygon snapshot

    Returns
    -------
    LiquidityScore
    """
    ticker = contract.get("ticker", "")
    bid    = float(contract.get("bid", 0) or 0)
    ask    = float(contract.get("ask", 0) or 0)
    vol    = int(contract.get("volume", 0) or 0)
    oi     = int(contract.get("open_interest", 0) or 0)
    last   = float(contract.get("last_price", 0) or 0)

    mid = (bid + ask) / 2 if bid > 0 and ask > 0 else last
    spread = ask - bid if ask > bid else 0.0
    spread_pct = (spread / mid * 100) if mid > 0 else 999.0

    # Score components
    spread_score = max(0.0, 1.0 - spread_pct / 10.0)  # 0% → 1.0, 10% → 0.0
    volume_score = min(1.0, vol / 1000.0)              # 1000 vol → 1.0
    oi_score     = min(1.0, oi / 5000.0)               # 5000 OI → 1.0

    score = 0.4 * spread_score + 0.35 * volume_score + 0.25 * oi_score

    # Hard gates
    passes = spread_pct <= 10.0 and (vol >= 200 or oi >= 500)
    reason = (
        "OK" if passes
        else f"Rejected: spread={spread_pct:.1f}% vol={vol} oi={oi}"
    )

    return LiquidityScore(
        ticker=ticker,
        score=round(score, 3),
        spread_pct=round(spread_pct, 2),
        volume=vol,
        open_interest=oi,
        passes_gate=passes,
        reason=reason,
    )


def compute_spread_pct(contract: dict) -> float:
    """Return bid-ask spread as % of mid price. Returns 999.0 if no quotes."""
    bid  = float(contract.get("bid", 0) or 0)
    ask  = float(contract.get("ask", 0) or 0)
    last = float(contract.get("last_price", 0) or 0)
    mid = (bid + ask) / 2 if bid > 0 and ask > 0 else last
    if mid <= 0:
        return 999.0
    return (ask - bid) / mid * 100 if ask > bid else 0.0


def find_strikes_near_spot(
    chain: list[dict],
    spot: float,
    pct_range: float = 2.0,
) -> list[dict]:
    """
    Filter chain to contracts within ±pct_range% of spot.

    Parameters
    ----------
    pct_range : distance from spot as %, e.g. 2.0 = ±2%
    """
    if not chain or spot <= 0:
        return chain

    low  = spot * (1 - pct_range / 100)
    high = spot * (1 + pct_range / 100)
    return [c for c in chain if low <= (c.get("strike") or 0) <= high]


def detect_high_oi_strikes(
    chain: list[dict],
    spot: float,
    top_n: int = 5,
) -> list[dict]:
    """
    Find the top N strikes by OI near spot (±5%).
    Used by Setup 7 (Dealer Hedge Zone).

    Returns list of dicts with: strike, contract_type, open_interest, volume
    """
    nearby = find_strikes_near_spot(chain, spot, pct_range=5.0)
    sorted_by_oi = sorted(
        nearby,
        key=lambda c: c.get("open_interest", 0) or 0,
        reverse=True,
    )
    return [
        {
            "strike": c.get("strike"),
            "contract_type": c.get("contract_type"),
            "open_interest": c.get("open_interest", 0),
            "volume": c.get("volume", 0),
        }
        for c in sorted_by_oi[:top_n]
    ]


# ── Idea Engine helpers ──────────────────────────────────────

def find_top_call_walls(
    calls_or_chain: list[dict],
    spot: float,
    top_n: int = 3,
) -> list[dict]:
    """
    Return top_n call strikes by open interest that are above spot.
    Each dict: {strike, open_interest, volume, distance_pct}
    """
    calls = [
        c for c in calls_or_chain
        if c.get("contract_type", "").lower() == "call"
        and (c.get("strike", 0) or 0) > spot * 0.98
    ] or [
        c for c in calls_or_chain
        if c.get("contract_type") is None or c.get("contract_type", "").lower() == "call"
    ]
    # If input was already pre-filtered calls list treat all entries
    if not calls:
        calls = calls_or_chain

    sorted_calls = sorted(calls, key=lambda c: c.get("open_interest", 0) or 0, reverse=True)
    result = []
    for c in sorted_calls[:top_n]:
        strike = float(c.get("strike", 0) or 0)
        dist_pct = round((strike - spot) / spot * 100, 2) if spot > 0 else 0.0
        result.append({
            "strike": strike,
            "open_interest": int(c.get("open_interest", 0) or 0),
            "volume": int(c.get("volume", 0) or 0),
            "distance_pct": dist_pct,
            "label": f"CW {strike:.0f}",
        })
    return result


def find_top_put_walls(
    puts_or_chain: list[dict],
    spot: float,
    top_n: int = 3,
) -> list[dict]:
    """
    Return top_n put strikes by open interest that are below spot.
    Each dict: {strike, open_interest, volume, distance_pct}
    """
    puts = [
        c for c in puts_or_chain
        if c.get("contract_type", "").lower() == "put"
        and (c.get("strike", 0) or 0) < spot * 1.02
    ] or [c for c in puts_or_chain if c.get("contract_type", "").lower() == "put"]

    if not puts:
        puts = puts_or_chain

    sorted_puts = sorted(puts, key=lambda c: c.get("open_interest", 0) or 0, reverse=True)
    result = []
    for c in sorted_puts[:top_n]:
        strike = float(c.get("strike", 0) or 0)
        dist_pct = round((spot - strike) / spot * 100, 2) if spot > 0 else 0.0
        result.append({
            "strike": strike,
            "open_interest": int(c.get("open_interest", 0) or 0),
            "volume": int(c.get("volume", 0) or 0),
            "distance_pct": dist_pct,
            "label": f"PW {strike:.0f}",
        })
    return result


def find_gamma_magnets(
    chain: list[dict],
    spot: float,
    top_n: int = 3,
) -> list[dict]:
    """
    Find strikes with the highest combined (call + put) gamma exposure within ±5%.
    These are price levels where dealer hedging creates gravitational pull.
    Each dict: {strike, net_gamma_exp, distance_pct, label}
    """
    if not chain or spot <= 0:
        return []

    low  = spot * 0.95
    high = spot * 1.05

    gamma_by_strike: dict[float, float] = {}
    for c in chain:
        strike = float(c.get("strike", 0) or 0)
        if not (low <= strike <= high):
            continue
        gamma  = float((c.get("greeks") or {}).get("gamma", 0) or 0)
        oi     = int(c.get("open_interest", 0) or 0)
        ct     = c.get("contract_type", "").lower()
        gex    = gamma * oi * 100 * spot * 0.01
        if ct == "call":
            gamma_by_strike[strike] = gamma_by_strike.get(strike, 0.0) + gex
        elif ct == "put":
            gamma_by_strike[strike] = gamma_by_strike.get(strike, 0.0) - gex

    # Sort by absolute magnitude (strongest magnets first)
    sorted_strikes = sorted(gamma_by_strike.items(), key=lambda kv: abs(kv[1]), reverse=True)

    result = []
    for strike, net_gex in sorted_strikes[:top_n]:
        dist_pct = round((strike - spot) / spot * 100, 2)
        polarity = "+" if net_gex > 0 else "-"
        result.append({
            "strike": strike,
            "net_gamma_exp": round(net_gex, 0),
            "distance_pct": dist_pct,
            "polarity": "positive" if net_gex > 0 else "negative",
            "label": f"GM {strike:.0f} {polarity}",
        })
    return result


# ── Internal helpers ─────────────────────────────────────────

def _calculate_max_pain(calls: list[dict], puts: list[dict]) -> float:
    """Strike where total option value at expiration is minimized."""
    strikes: set[float] = set()
    for c in calls + puts:
        s = c.get("strike", 0)
        if s and s > 0:
            strikes.add(float(s))

    if not strikes:
        return 0.0

    min_pain       = float("inf")
    max_pain_strike = 0.0

    for test_price in sorted(strikes):
        total_pain = 0.0
        for c in calls:
            strike = c.get("strike", 0) or 0
            oi     = c.get("open_interest", 0) or 0
            if test_price > strike:
                total_pain += (test_price - strike) * oi * 100
        for p in puts:
            strike = p.get("strike", 0) or 0
            oi     = p.get("open_interest", 0) or 0
            if test_price < strike:
                total_pain += (strike - test_price) * oi * 100

        if total_pain < min_pain:
            min_pain         = total_pain
            max_pain_strike  = test_price

    return max_pain_strike


def _detect_uoa(chain: list[dict], metrics: ChainMetrics) -> None:
    """Detect unusual options activity and populate metrics.uoa_details."""
    uoa: list[dict] = []

    for c in chain:
        vol    = c.get("volume", 0) or 0
        oi     = c.get("open_interest", 0) or 0
        strike = c.get("strike", 0)
        ct     = c.get("contract_type", "")

        if vol < 100:
            continue

        vol_oi_ratio = vol / max(oi, 1)

        if vol_oi_ratio > 3.0 or vol > 5000:
            uoa.append({
                "strike": strike,
                "type": ct,
                "volume": vol,
                "open_interest": oi,
                "vol_oi_ratio": round(vol_oi_ratio, 1),
                "iv": c.get("iv", 0),
            })

    uoa.sort(key=lambda x: x["vol_oi_ratio"], reverse=True)
    metrics.uoa_details  = uoa[:5]
    metrics.uoa_detected = len(uoa) > 0
