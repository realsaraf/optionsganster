"""
Full-pipeline backtest — runs the complete OptionGangster engine stack
(Regime → S/R → VPA → Posture → Idea Engine) against minute-bar CSVs.

Usage:
    python backtest_full.py                    # backtest latest day
    python backtest_full.py 2026-02-24         # specific date
    python backtest_full.py 2026-02-20 2026-02-24  # date range
"""

import sys, os, math
from datetime import datetime, date, timedelta
from pathlib import Path
from dataclasses import dataclass, field

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from app.vpa_engine import VPAEngine
from app.regime_engine import RegimeEngine
from app.sr_engine import SREngine
from app.decision_engine import DecisionEngine
from app.idea_engine import IdeaEngine, BriefingInput
from app.chain_analytics import ChainMetrics
from app.risk_engine import RiskEngine

DATA_DIR = Path(__file__).parent / "data" / "qqq"

# ── Helpers ──────────────────────────────────────────────────

def load_day(date_str: str) -> pd.DataFrame:
    """Load a single day's QQQ bars from local CSV, filter to RTH."""
    path = DATA_DIR / f"{date_str}.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    for c in ["open", "close", "high", "low"]:
        df[c] = df[c].astype(float)
    df["volume"] = df["volume"].astype(int)
    df["datetime"] = pd.to_datetime(df["window_start"].astype(int), unit="ns").dt.strftime("%Y-%m-%d %H:%M:%S")
    df = df.sort_values("datetime").reset_index(drop=True)
    # Filter to RTH (09:30–16:00 ET = 14:30–21:00 UTC)
    df["_dt"] = pd.to_datetime(df["datetime"])
    df = df[(df["_dt"].dt.hour > 14) | ((df["_dt"].dt.hour == 14) & (df["_dt"].dt.minute >= 30))]
    df = df[df["_dt"].dt.hour < 21]
    df = df.drop(columns=["_dt"]).reset_index(drop=True)
    return df


def build_fake_daily(all_dates: list[str]) -> pd.DataFrame:
    """Build a pseudo-daily DataFrame from available CSV files for S/R engine."""
    rows = []
    for d in all_dates:
        df = load_day(d)
        if df.empty:
            continue
        rows.append({
            "date": d,
            "open": df["open"].iloc[0],
            "high": df["high"].max(),
            "low": df["low"].min(),
            "close": df["close"].iloc[-1],
            "volume": df["volume"].sum(),
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


@dataclass
class SimTrade:
    """A simulated trade from an Idea Engine entry."""
    time: str
    idea_type: str
    direction: str       # CALL / PUT
    headline: str
    score: int
    grade: str
    entry: float
    stop: float
    target_1: float
    target_2: float
    reward_risk: float
    exit_price: float = 0.0
    exit_time: str = ""
    exit_reason: str = ""
    pnl_pct: float = 0.0
    r_multiple: float = 0.0
    is_win: bool = False


def simulate_trade(trade: SimTrade, future_bars: pd.DataFrame) -> SimTrade:
    """Walk forward through future bars to see if stop/target gets hit."""
    if future_bars.empty:
        trade.exit_reason = "NO_DATA"
        return trade

    is_call = trade.direction == "CALL"
    risk = abs(trade.entry - trade.stop)
    if risk == 0:
        trade.exit_reason = "ZERO_RISK"
        return trade

    for _, bar in future_bars.iterrows():
        h, l, c = bar["high"], bar["low"], bar["close"]

        # Check stop first
        if is_call and l <= trade.stop:
            trade.exit_price = trade.stop
            trade.exit_time = bar["datetime"]
            trade.exit_reason = "STOPPED"
            break
        elif not is_call and h >= trade.stop:
            trade.exit_price = trade.stop
            trade.exit_time = bar["datetime"]
            trade.exit_reason = "STOPPED"
            break

        # Check target_1
        if is_call and h >= trade.target_1:
            trade.exit_price = trade.target_1
            trade.exit_time = bar["datetime"]
            trade.exit_reason = "HIT_T1"
            break
        elif not is_call and l <= trade.target_1:
            trade.exit_price = trade.target_1
            trade.exit_time = bar["datetime"]
            trade.exit_reason = "HIT_T1"
            break
    else:
        # End of day — close at last bar close
        trade.exit_price = future_bars["close"].iloc[-1]
        trade.exit_time = future_bars["datetime"].iloc[-1]
        trade.exit_reason = "EOD"

    # Compute P&L
    if is_call:
        trade.pnl_pct = (trade.exit_price - trade.entry) / trade.entry * 100
        trade.r_multiple = (trade.exit_price - trade.entry) / risk if risk else 0
    else:
        trade.pnl_pct = (trade.entry - trade.exit_price) / trade.entry * 100
        trade.r_multiple = (trade.entry - trade.exit_price) / risk if risk else 0

    trade.is_win = trade.r_multiple > 0
    return trade


# ── Main backtest ────────────────────────────────────────────

def backtest_day(date_str: str, daily_df: pd.DataFrame, verbose: bool = True) -> dict:
    """Run the full engine stack on a single day, simulating ideas as trades."""

    df = load_day(date_str)
    if df.empty or len(df) < 30:
        return {"date": date_str, "error": "Insufficient bars", "trades": []}

    # Create fresh engine instances per day
    regime_eng = RegimeEngine()
    sr_eng = SREngine()
    decision_eng = DecisionEngine()
    vpa_eng = VPAEngine()
    idea_eng = IdeaEngine()
    risk_eng = RiskEngine()

    spot_open = df["open"].iloc[0]
    spot_close = df["close"].iloc[-1]
    day_change_pct = (spot_close - spot_open) / spot_open * 100

    # ── Evaluate at checkpoints through the day ──────────────
    # We'll evaluate at 30-min windows: bar 30, 60, 90, 120, ...
    checkpoints = list(range(60, len(df), 60))  # every 60 bars (1 hr)
    if not checkpoints:
        checkpoints = [len(df) - 1]

    all_trades: list[SimTrade] = []
    posture_log = []
    idea_log = []
    seen_ideas = set()  # avoid duplicate entries

    for cp in checkpoints:
        window = df.iloc[:cp].copy()
        spot = window["close"].iloc[-1]
        cp_time = window["datetime"].iloc[-1]
        future = df.iloc[cp:].copy()

        # 1. Regime
        try:
            regime = regime_eng.classify(window, force_reclassify=True)
        except Exception as e:
            regime = None

        # 2. S/R (use daily bars)
        try:
            sr = sr_eng.analyze(daily_df, spot, intraday_bars=window)
        except Exception:
            sr = None

        # 3. VPA
        try:
            vpa_results = vpa_eng.analyze(window[["datetime", "open", "high", "low", "close", "volume"]].copy())
            vpa_bias = vpa_eng.get_bias(vpa_results)
            vol_regime = vpa_eng.get_volume_regime(window)
        except Exception:
            vpa_bias = {"bias": "neutral", "strength": 0, "reason": ""}
            vol_regime = {"regime": "NORMAL", "ratio": 1.0, "detail": ""}

        # 4. Decision
        try:
            decision = decision_eng.compute(
                regime=regime, sr=sr, chain_metrics=None,
                vpa_bias=vpa_bias, vol_regime=vol_regime,
                underlying_price=spot, df=window,
            )
            posture_log.append({
                "time": cp_time,
                "decision": decision.decision,
                "call_score": decision.call_score,
                "put_score": decision.put_score,
                "confidence": round(decision.confidence, 2),
                "capital_mode": decision.capital_mode,
            })
        except Exception as e:
            decision = None

        # 5. Idea Engine
        try:
            briefing_input = BriefingInput(
                symbol="QQQ",
                underlying_price=spot,
                df=window,
                regime=regime,
                sr=sr,
                chain_metrics=ChainMetrics(),
                vpa_bias=vpa_bias,
                active_alerts=[],
                expirations=[],
            )
            briefing = idea_eng.generate_briefing(briefing_input)
            ideas = briefing.trade_ideas
        except Exception as e:
            ideas = []

        # Record ideas and simulate top 2
        for idx, idea in enumerate(ideas[:2]):
            # Dedup: skip if we've already entered near same level
            key = f"{idea.direction}_{round(idea.entry_level, 1)}"
            if key in seen_ideas:
                continue
            seen_ideas.add(key)

            grade = "A" if idea.score >= 80 else "B" if idea.score >= 60 else "C" if idea.score >= 40 else "D"

            idea_log.append({
                "time": cp_time,
                "type": idea.idea_type.value,
                "direction": idea.direction,
                "headline": idea.headline,
                "score": idea.score,
                "grade": grade,
                "entry": idea.entry_level,
                "stop": idea.stop_level,
                "rr": round(idea.reward_risk, 1),
                "score_breakdown": idea.score_breakdown,
            })

            # Only trade A and B setups (score >= 60) or if top and score >= 50
            if idea.score < 50:
                continue

            trade = SimTrade(
                time=cp_time,
                idea_type=idea.idea_type.value,
                direction=idea.direction,
                headline=idea.headline,
                score=idea.score,
                grade=grade,
                entry=idea.entry_level,
                stop=idea.stop_level,
                target_1=idea.target_1,
                target_2=idea.target_2,
                reward_risk=round(idea.reward_risk, 2),
            )
            trade = simulate_trade(trade, future)
            all_trades.append(trade)

            # Update risk engine for capital mode
            risk_eng.record_outcome(trade.is_win)

    # ── Summary ──────────────────────────────────────────────
    wins = [t for t in all_trades if t.is_win]
    losses = [t for t in all_trades if not t.is_win]
    total_r = sum(t.r_multiple for t in all_trades)
    avg_r = total_r / len(all_trades) if all_trades else 0

    result = {
        "date": date_str,
        "day_change_pct": round(day_change_pct, 2),
        "open": round(spot_open, 2),
        "close": round(spot_close, 2),
        "bars": len(df),
        "checkpoints": len(checkpoints),
        "total_ideas": len(idea_log),
        "total_trades": len(all_trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(len(wins) / len(all_trades) * 100, 1) if all_trades else 0,
        "total_r": round(total_r, 2),
        "avg_r": round(avg_r, 2),
        "best_r": round(max((t.r_multiple for t in all_trades), default=0), 2),
        "worst_r": round(min((t.r_multiple for t in all_trades), default=0), 2),
        "trades": all_trades,
        "ideas": idea_log,
        "postures": posture_log,
    }

    if verbose:
        print(f"\n{'═'*70}")
        print(f"  {date_str}  |  QQQ {spot_open:.2f} → {spot_close:.2f}  ({day_change_pct:+.2f}%)")
        print(f"  {len(df)} bars  |  {len(checkpoints)} checkpoints")
        print(f"{'═'*70}")

        # Posture timeline
        print(f"\n  ── Posture Timeline ──")
        for p in posture_log:
            cm_str = f" [{p['capital_mode']}]" if p['capital_mode'] != 'NORMAL' else ''
            print(f"  {p['time'][-8:]}  {p['state']:<20s} {p['direction']:<8s} conf={p['confidence']:.0%}{cm_str}")
            print(f"           {p['headline']}")

        # Ideas generated
        print(f"\n  ── Ideas Generated ({len(idea_log)}) ──")
        for i in idea_log:
            bd = i.get('score_breakdown', {})
            bd_str = '  '.join(f"{k[:3]}={v}" for k, v in bd.items()) if bd else ''
            print(f"  {i['time'][-8:]}  [{i['grade']}{i['score']:>3}] {i['direction']} {i['type']:<12s} entry={i['entry']:.2f} stop={i['stop']:.2f} R:R={i['rr']}×")
            if bd_str:
                print(f"           {bd_str}")

        # Trade simulation results
        print(f"\n  ── Simulated Trades ({len(all_trades)}) ──")
        for t in all_trades:
            icon = "✅" if t.is_win else "❌"
            print(f"  {icon} {t.time[-8:]}→{t.exit_time[-8:] if t.exit_time else '???'}  "
                  f"[{t.grade}{t.score}] {t.direction} {t.idea_type:<12s}  "
                  f"entry={t.entry:.2f} exit={t.exit_price:.2f}  "
                  f"R={t.r_multiple:+.2f}  ({t.exit_reason})")

        # Summary box
        print(f"\n  {'─'*50}")
        wr = result['win_rate']
        wr_color = "🟢" if wr >= 55 else "🟡" if wr >= 40 else "🔴"
        r_color = "🟢" if avg_r > 0 else "🔴"
        print(f"  {wr_color} Win Rate: {wr:.0f}%  ({len(wins)}W / {len(losses)}L)")
        print(f"  {r_color} Total R: {total_r:+.2f}  |  Avg R: {avg_r:+.2f}")
        print(f"  Best: {result['best_r']:+.2f}R  |  Worst: {result['worst_r']:+.2f}R")

    return result


def main():
    dates = sorted(p.stem for p in DATA_DIR.glob("*.csv"))
    if not dates:
        print("No data found in data/qqq/. Run download_flatfiles.py first.")
        return

    # Parse args
    if len(sys.argv) >= 3:
        # Date range
        start, end = sys.argv[1], sys.argv[2]
        dates = [d for d in dates if start <= d <= end]
    elif len(sys.argv) == 2:
        # Single date
        dates = [sys.argv[1]]
    else:
        # Default: latest available
        dates = dates[-1:]

    if not dates:
        print("No matching dates found.")
        return

    # Build daily bars from all available files (for S/R)
    all_dates = sorted(p.stem for p in DATA_DIR.glob("*.csv"))
    daily_df = build_fake_daily(all_dates)

    print("=" * 70)
    print(f"  OptionGangster Full-Pipeline Backtest")
    print(f"  Dates: {dates[0]} to {dates[-1]}  ({len(dates)} days)")
    print(f"  Daily bars for S/R: {len(daily_df)} days")
    print("=" * 70)

    all_results = []
    grand_trades = []

    for d in dates:
        result = backtest_day(d, daily_df, verbose=True)
        all_results.append(result)
        grand_trades.extend(result.get("trades", []))

    # ── Grand summary ────────────────────────────────────────
    if len(dates) > 1:
        wins = sum(r["wins"] for r in all_results)
        losses = sum(r["losses"] for r in all_results)
        total_trades = sum(r["total_trades"] for r in all_results)
        total_r = sum(r["total_r"] for r in all_results)

        print(f"\n{'═'*70}")
        print(f"  GRAND SUMMARY  ({len(dates)} days)")
        print(f"{'═'*70}")
        print(f"  Total trades: {total_trades}")
        print(f"  Wins: {wins}  |  Losses: {losses}")
        if total_trades:
            print(f"  Win rate: {wins / total_trades * 100:.1f}%")
            print(f"  Total R: {total_r:+.2f}")
            print(f"  Avg R per trade: {total_r / total_trades:+.2f}")

        # By idea type
        from collections import Counter
        type_stats = {}
        for t in grand_trades:
            k = t.idea_type
            if k not in type_stats:
                type_stats[k] = {"wins": 0, "losses": 0, "total_r": 0}
            if t.is_win:
                type_stats[k]["wins"] += 1
            else:
                type_stats[k]["losses"] += 1
            type_stats[k]["total_r"] += t.r_multiple

        print(f"\n  By Setup Type:")
        for k, v in sorted(type_stats.items()):
            n = v["wins"] + v["losses"]
            wr = v["wins"] / n * 100 if n else 0
            print(f"    {k:<16s}  {n:>3} trades  WR={wr:5.1f}%  R={v['total_r']:+.2f}")

        # By grade
        grade_stats = {}
        for t in grand_trades:
            g = t.grade
            if g not in grade_stats:
                grade_stats[g] = {"wins": 0, "losses": 0, "total_r": 0}
            if t.is_win:
                grade_stats[g]["wins"] += 1
            else:
                grade_stats[g]["losses"] += 1
            grade_stats[g]["total_r"] += t.r_multiple

        print(f"\n  By Grade:")
        for g in ["A", "B", "C", "D"]:
            if g in grade_stats:
                v = grade_stats[g]
                n = v["wins"] + v["losses"]
                wr = v["wins"] / n * 100 if n else 0
                print(f"    Grade {g}:  {n:>3} trades  WR={wr:5.1f}%  R={v['total_r']:+.2f}")


if __name__ == "__main__":
    main()
