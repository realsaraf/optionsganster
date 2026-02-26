"""
ATM Options (5DTE) backtest runner.

Runs the same engine stack used by the backtest replay API to generate
trade ideas, then simulates trades using the frontend's ATM options logic.

Usage:
  python backtest_options_5dte.py
  python backtest_options_5dte.py --days 5 --step 5 --grid
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from app.indicators import vwap, vwap_bands, ema, rsi, atr_value
from app.vpa_engine import VPAEngine, VPASignal
from app.regime_engine import RegimeEngine
from app.sr_engine import SREngine
from app.idea_engine import IdeaEngine, BriefingInput, PlaybookMode
from app.chain_analytics import ChainMetrics

DATA_DIR = Path(__file__).parent / "data" / "qqq"


def _load_day(date_str: str) -> pd.DataFrame:
    path = DATA_DIR / f"{date_str}.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    for c in ["open", "close", "high", "low"]:
        df[c] = df[c].astype(float)
    df["volume"] = df["volume"].astype(int)
    df["datetime"] = pd.to_datetime(df["window_start"].astype(int), unit="ns").dt.strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    df = df.sort_values("datetime").reset_index(drop=True)
    dt_col = pd.to_datetime(df["datetime"])
    mask_start = (dt_col.dt.hour > 14) | ((dt_col.dt.hour == 14) & (dt_col.dt.minute >= 30))
    mask_end = dt_col.dt.hour < 21
    df = df[mask_start & mask_end]
    return df.reset_index(drop=True)


def _build_daily(all_dates: list[str]) -> pd.DataFrame:
    rows = []
    for d in all_dates:
        ddf = _load_day(d)
        if ddf.empty:
            continue
        rows.append(
            {
                "date": d,
                "open": ddf["open"].iloc[0],
                "high": ddf["high"].max(),
                "low": ddf["low"].min(),
                "close": ddf["close"].iloc[-1],
                "volume": int(ddf["volume"].sum()),
            }
        )
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _estimate_opt_price(s: float, k: float, opt_type: str, dte: float, day_frac: float, iv: float) -> float:
    t = max((dte - day_frac) / 365.0, 0.0001)
    r = 0.045
    sigma = iv or 0.20
    if s <= 0 or k <= 0 or sigma <= 0:
        return 0.01
    d1 = (math.log(s / k) + (r + 0.5 * sigma * sigma) * t) / (sigma * math.sqrt(t))
    d2 = d1 - sigma * math.sqrt(t)
    if opt_type == "call":
        return s * _norm_cdf(d1) - k * math.exp(-r * t) * _norm_cdf(d2)
    return k * math.exp(-r * t) * _norm_cdf(-d2) - s * _norm_cdf(-d1)


def _estimate_iv(snaps: list[dict]) -> float:
    prices = [s.get("underlying_price", 0) for s in snaps if s.get("underlying_price", 0) > 0]
    if len(prices) < 5:
        return 0.20
    hi, lo = max(prices), min(prices)
    mid = (hi + lo) / 2.0
    if mid <= 0:
        return 0.20
    daily_vol = (hi - lo) / mid
    ann_vol = daily_vol * math.sqrt(252)
    return max(0.10, min(ann_vol, 0.80))


def _time_to_min(t: str) -> int:
    if not t:
        return 0
    hhmm = t[-5:] if len(t) >= 5 else t
    try:
        hh, mm = hhmm.split(":")
        return int(hh) * 60 + int(mm)
    except Exception:
        return 0


def _minutes_between(t1: str, t2: str) -> int:
    return max(0, _time_to_min(t2) - _time_to_min(t1))


@dataclass
class StrategyConfig:
    name: str
    confidence_min: float = 0.45
    reg_score_min: int = 1
    rr_min: float = 1.2
    max_trades: int = 5
    cooldown_bars: int = 3
    trail_trigger: float = 0.6
    be_buffer: float = 0.05
    min_risk_atr_mult: float = 0.5
    entry_distance_mult: float = 2.0
    invert_direction: bool = False
    opt_rr_min: float | None = None
    premium_risk_pct: float | None = None
    dollars_per_trade: float | None = None
    opt_take_profit_pct: float | None = None
    opt_stop_loss_pct: float | None = None
    max_daily_losses: int | None = None
    max_daily_drawdown: float | None = None
    allow_choppy: bool = True
    allow_range_bound: bool = True
    entry_cutoff_utc_min: int | None = 19 * 60 + 30
    direction_mode: str = "both"  # both | call_only | put_only
    require_trend_regime: bool = False
    use_underlying_exits: bool = True
    sr_reversal_only: bool = False
    sr_proximity_pct: float = 0.15
    sr_min_strength: int = 60
    sr_wick_ratio_min: float = 0.35
    sr_reversal_boost: int = 0
    sr_reversal_penalty: int = 0


def _make_snapshots(date_str: str, step_bars: int, playbook: str) -> list[dict]:
    df = _load_day(date_str)
    if df.empty or len(df) < 30:
        return []

    all_csv_dates = sorted(p.stem for p in DATA_DIR.glob("*.csv"))
    daily_df = _build_daily([d for d in all_csv_dates if d <= date_str])

    re = RegimeEngine()
    sr = SREngine()
    ve = VPAEngine()
    ie = IdeaEngine()

    try:
        pb_mode = PlaybookMode(playbook.lower())
    except Exception:
        pb_mode = PlaybookMode.ALL

    warmup = min(60, len(df))
    checkpoints = list(range(warmup, len(df) + 1, step_bars))
    if not checkpoints or checkpoints[-1] < len(df):
        checkpoints.append(len(df))

    snaps: list[dict] = []

    for cp in checkpoints:
        window = df.iloc[:cp].copy()
        if window.empty:
            continue
        spot = float(window["close"].iloc[-1])
        cp_time = str(window["datetime"].iloc[-1])

        indicators = {}
        try:
            indicators["ema_9"] = [float(v) if not pd.isna(v) else 0.0 for v in ema(window["close"], 9)]
            indicators["ema_20"] = [float(v) if not pd.isna(v) else 0.0 for v in ema(window["close"], 20)]
        except Exception:
            indicators = {}

        regime_res = None
        if len(window) >= 10:
            try:
                regime_res = re.classify(window, force_reclassify=True)
            except Exception:
                regime_res = None

        sr_res = None
        if not daily_df.empty:
            try:
                sr_res = sr.analyze(daily_df, spot, intraday_bars=window)
            except Exception:
                sr_res = None

        vpa_bias = {}
        if len(window) >= 2:
            try:
                vpa_results = ve.analyze(
                    window[["datetime", "open", "high", "low", "close", "volume"]].copy()
                )
                vpa_bias = ve.get_bias(vpa_results) or {}
            except Exception:
                vpa_bias = {}

        ideas_out = []
        try:
            bi = BriefingInput(
                symbol="QQQ",
                underlying_price=spot,
                df=window,
                regime=regime_res,
                sr=sr_res,
                chain_metrics=ChainMetrics(),
                vpa_bias=vpa_bias,
                active_alerts=[],
                expirations=[],
            )
            briefing = ie.generate_briefing(bi)
            briefing = ie.filter_by_playbook(briefing, pb_mode)
            ideas_out = [
                {
                    "idea_type": i.idea_type.value,
                    "direction": i.direction,
                    "confidence": i.confidence,
                    "entry_level": i.entry_level,
                    "stop_level": i.stop_level,
                    "target_1": i.target_1,
                }
                for i in briefing.trade_ideas
            ]
        except Exception:
            ideas_out = []

        regime_out = {}
        if regime_res is not None:
            regime_out = {
                "regime": regime_res.regime.value,
                "price_vs_vwap": regime_res.price_vs_vwap,
                "rsi_current": regime_res.rsi_current,
                "atr_current": regime_res.atr_current,
            }

        snaps.append(
            {
                "time": cp_time,
                "underlying_price": spot,
                "trade_ideas": ideas_out,
                "regime": regime_out,
                "indicators": indicators,
                "key_levels": [
                    {
                        "price": float(l.price),
                        "kind": l.kind,
                        "strength": int(l.strength),
                    }
                    for l in (sr_res.levels[:30] if sr_res and sr_res.levels else [])
                ],
                "last_bar": {
                    "open": float(window["open"].iloc[-1]),
                    "high": float(window["high"].iloc[-1]),
                    "low": float(window["low"].iloc[-1]),
                    "close": float(window["close"].iloc[-1]),
                },
            }
        )

    return snaps


def _simulate_trades(snaps: list[dict], cfg: StrategyConfig, capital: float) -> list[dict]:
    trades = []
    pos = None
    cooldown = 0
    iv = _estimate_iv(snaps)
    day_pnl = 0.0
    day_losses = 0
    day_locked = False
    fixed_premium_per_trade = None
    if cfg.dollars_per_trade is not None:
        fixed_premium_per_trade = max(0.0, float(cfg.dollars_per_trade))
    elif cfg.premium_risk_pct is not None:
        fixed_premium_per_trade = max(0.0, capital * cfg.premium_risk_pct)

    for i, snap in enumerate(snaps):
        price = snap.get("underlying_price", 0) or 0
        time = snap.get("time", "")
        ideas = snap.get("trade_ideas", []) or []
        regime = snap.get("regime", {}) or {}
        indicators = snap.get("indicators", {}) or {}
        key_levels = snap.get("key_levels", []) or []
        last_bar = snap.get("last_bar") or {}

        # Exit logic
        if pos:
            exit_reason = None
            min_elapsed = _minutes_between(pos["entry_time"], time)
            current_opt_price = _estimate_opt_price(
                price,
                pos["strike"],
                "call" if pos["type"] == "CALLS" else "put",
                5,
                min_elapsed / (6.5 * 60),
                iv,
            )
            if current_opt_price < 0.01:
                current_opt_price = 0.01

            if cfg.opt_take_profit_pct is not None:
                if current_opt_price >= pos["opt_entry"] * (1.0 + cfg.opt_take_profit_pct):
                    exit_reason = "OPT_TP"
            if cfg.opt_stop_loss_pct is not None and not exit_reason:
                if current_opt_price <= pos["opt_entry"] * (1.0 - cfg.opt_stop_loss_pct):
                    exit_reason = "OPT_SL"

            if cfg.use_underlying_exits:
                if pos["type"] == "CALLS":
                    if not exit_reason and price >= pos["adj_target"]:
                        exit_reason = "TARGET"
                    elif not exit_reason and price <= pos["adj_stop"]:
                        exit_reason = "STOP"
                else:
                    if not exit_reason and price <= pos["adj_target"]:
                        exit_reason = "TARGET"
                    elif not exit_reason and price >= pos["adj_stop"]:
                        exit_reason = "STOP"

            if not exit_reason and pos.get("adj_target") and pos.get("entry"):
                dist = price - pos["entry"] if pos["type"] == "CALLS" else pos["entry"] - price
                full_dist = (
                    pos["adj_target"] - pos["entry"] if pos["type"] == "CALLS" else pos["entry"] - pos["adj_target"]
                )
                if full_dist > 0 and dist / full_dist >= cfg.trail_trigger:
                    be = pos["entry"] + (cfg.be_buffer if pos["type"] == "CALLS" else -cfg.be_buffer)
                    if pos["type"] == "CALLS":
                        pos["adj_stop"] = max(pos["adj_stop"], be)
                    else:
                        pos["adj_stop"] = min(pos["adj_stop"], be)

            reg_name = (regime.get("regime") or "").upper()
            if not exit_reason:
                if pos["type"] == "CALLS" and reg_name == "TREND_DOWN":
                    exit_reason = "REGIME"
                if pos["type"] == "PUTS" and reg_name == "TREND_UP":
                    exit_reason = "REGIME"

            if exit_reason:
                ul_pnl = price - pos["entry"] if pos["type"] == "CALLS" else pos["entry"] - price
                ul_pnl_pct = (ul_pnl / pos["entry"] * 100) if pos["entry"] else 0
                opt_entry = pos["opt_entry"]
                opt_exit = current_opt_price
                opt_pnl = pos["contracts"] * 100 * (opt_exit - opt_entry)
                opt_pnl_pct = (opt_exit - opt_entry) / opt_entry * 100 if opt_entry > 0 else 0
                day_pnl += opt_pnl
                if opt_pnl <= 0:
                    day_losses += 1
                    if cfg.max_daily_losses is not None and day_losses >= cfg.max_daily_losses:
                        day_locked = True
                if cfg.max_daily_drawdown is not None and day_pnl <= -abs(cfg.max_daily_drawdown):
                    day_locked = True

                trades.append(
                    {
                        "type": pos["type"],
                        "entry": pos["entry"],
                        "entry_time": pos["entry_time"],
                        "exit": price,
                        "exit_time": time,
                        "pnl": ul_pnl,
                        "pnl_pct": ul_pnl_pct,
                        "reason": exit_reason,
                        "idea_type": pos["idea_type"],
                        "opt_entry": opt_entry,
                        "opt_exit": opt_exit,
                        "opt_pnl": opt_pnl,
                        "opt_pnl_pct": opt_pnl_pct,
                        "contracts": pos["contracts"],
                        "strike": pos["strike"],
                    }
                )
                pos = None
                cooldown = cfg.cooldown_bars
                continue

        if day_locked:
            continue
        if cooldown > 0:
            cooldown -= 1
            continue
        if len(trades) >= cfg.max_trades:
            continue
        if pos:
            continue

        # Entry logic
        if not ideas:
            continue
        sorted_ideas = sorted(ideas, key=lambda a: a.get("confidence", 0) or 0, reverse=True)
        best = sorted_ideas[0]
        direction = best.get("direction") or ""
        conf = best.get("confidence") or 0
        if conf < cfg.confidence_min or direction not in ("CALL", "PUT") or not best.get("entry_level"):
            continue

        if cfg.invert_direction:
            direction = "PUT" if direction == "CALL" else "CALL"

        reg_name = (regime.get("regime") or "").upper()
        if cfg.direction_mode == "call_only" and direction != "CALL":
            continue
        if cfg.direction_mode == "put_only" and direction != "PUT":
            continue
        if cfg.require_trend_regime and reg_name not in ("TREND_UP", "TREND_DOWN"):
            continue
        if reg_name == "CHOPPY" and not cfg.allow_choppy:
            continue
        if reg_name == "RANGE_BOUND" and not cfg.allow_range_bound:
            continue

        if cfg.entry_cutoff_utc_min is not None:
            if _time_to_min(time[-5:]) > cfg.entry_cutoff_utc_min:
                continue

        sr_reversal_match = None
        if cfg.sr_reversal_only or cfg.sr_reversal_boost != 0 or cfg.sr_reversal_penalty != 0:
            o = float(last_bar.get("open", price))
            h = float(last_bar.get("high", price))
            l = float(last_bar.get("low", price))
            c = float(last_bar.get("close", price))
            bar_range = max(h - l, 1e-6)
            body = abs(c - o)
            lower_wick = max(0.0, min(o, c) - l)
            upper_wick = max(0.0, h - max(o, c))
            bullish_reject = (c >= o) and (lower_wick / bar_range >= cfg.sr_wick_ratio_min) and (body / bar_range <= 0.5)
            bearish_reject = (c <= o) and (upper_wick / bar_range >= cfg.sr_wick_ratio_min) and (body / bar_range <= 0.5)

            supports = [
                lv for lv in key_levels
                if (lv.get("kind") == "support") and int(lv.get("strength", 0)) >= cfg.sr_min_strength
            ]
            resistances = [
                lv for lv in key_levels
                if (lv.get("kind") == "resistance") and int(lv.get("strength", 0)) >= cfg.sr_min_strength
            ]
            near_support = False
            near_resistance = False
            if price > 0:
                if supports:
                    near_support = min(abs(price - float(lv["price"])) / price * 100 for lv in supports) <= cfg.sr_proximity_pct
                if resistances:
                    near_resistance = min(abs(price - float(lv["price"])) / price * 100 for lv in resistances) <= cfg.sr_proximity_pct

            if direction == "CALL":
                sr_reversal_match = near_support and bullish_reject
            else:
                sr_reversal_match = near_resistance and bearish_reject

            if cfg.sr_reversal_only and not sr_reversal_match:
                continue

        vwap_pos = (regime.get("price_vs_vwap") or "").lower()
        rsi_val = regime.get("rsi_current") or 50
        atr = regime.get("atr_current") or 0

        reg_score = 0
        if direction == "CALL":
            if "UP" in reg_name:
                reg_score += 2
            elif reg_name in ("RANGE_BOUND", "CHOPPY"):
                reg_score += 0
            elif "DOWN" in reg_name:
                reg_score -= 2
            if vwap_pos == "above":
                reg_score += 1
            elif vwap_pos == "below":
                reg_score -= 1
            if rsi_val > 70:
                reg_score -= 1
            if rsi_val < 40:
                reg_score -= 1
            if 40 <= rsi_val <= 60:
                reg_score += 1
        else:
            if "DOWN" in reg_name:
                reg_score += 2
            elif reg_name in ("RANGE_BOUND", "CHOPPY"):
                reg_score += 0
            elif "UP" in reg_name:
                reg_score -= 2
            if vwap_pos == "below":
                reg_score += 1
            elif vwap_pos == "above":
                reg_score -= 1
            if rsi_val < 30:
                reg_score -= 1
            if rsi_val > 60:
                reg_score -= 1
            if 40 <= rsi_val <= 60:
                reg_score += 1

        ema9 = indicators.get("ema_9")
        ema20 = indicators.get("ema_20")
        if isinstance(ema9, list) and isinstance(ema20, list) and ema9 and ema20:
            e9 = ema9[-1]
            e20 = ema20[-1]
            if direction == "CALL" and e9 > e20:
                reg_score += 1
            if direction == "CALL" and e9 < e20:
                reg_score -= 1
            if direction == "PUT" and e9 < e20:
                reg_score += 1
            if direction == "PUT" and e9 > e20:
                reg_score -= 1

        if sr_reversal_match is True:
            reg_score += cfg.sr_reversal_boost
        elif sr_reversal_match is False:
            reg_score -= cfg.sr_reversal_penalty

        if reg_score < cfg.reg_score_min:
            continue

        entry_lvl = best.get("entry_level") or 0
        stop_lvl = best.get("stop_level") or 0
        tgt1 = best.get("target_1") or 0

        if direction == "CALL":
            risk_amt = entry_lvl - stop_lvl
            reward_amt = tgt1 - entry_lvl
            if abs(price - entry_lvl) > risk_amt * cfg.entry_distance_mult and risk_amt > 0:
                continue
        else:
            risk_amt = stop_lvl - entry_lvl
            reward_amt = entry_lvl - tgt1
            if abs(price - entry_lvl) > risk_amt * cfg.entry_distance_mult and risk_amt > 0:
                continue

        min_risk = atr * cfg.min_risk_atr_mult if atr > 0 else price * 0.001
        if risk_amt < min_risk:
            rr = reward_amt / risk_amt if risk_amt > 0 else 1.5
            risk_amt = min_risk
            reward_amt = min_risk * rr

        if risk_amt <= 0 or reward_amt <= 0 or reward_amt / risk_amt < cfg.rr_min:
            continue

        adj_stop = price - risk_amt if direction == "CALL" else price + risk_amt
        adj_target = price + reward_amt if direction == "CALL" else price - reward_amt
        strike = round(price)
        opt_type = "call" if direction == "CALL" else "put"
        opt_price = _estimate_opt_price(price, strike, opt_type, 5, 0, iv)

        if cfg.opt_rr_min is not None:
            opt_stop = _estimate_opt_price(adj_stop, strike, opt_type, 5, 0, iv)
            opt_target = _estimate_opt_price(adj_target, strike, opt_type, 5, 0, iv)
            opt_risk = opt_price - opt_stop
            opt_reward = opt_target - opt_price
            if opt_risk <= 0 or opt_reward / opt_risk < cfg.opt_rr_min:
                continue

        premium_cap = capital
        if fixed_premium_per_trade is not None:
            premium_cap = fixed_premium_per_trade

        contracts = math.floor(premium_cap / (opt_price * 100))
        if contracts < 1:
            continue

        pos = {
            "type": "CALLS" if direction == "CALL" else "PUTS",
            "entry": price,
            "entry_time": time,
            "adj_stop": adj_stop,
            "adj_target": adj_target,
            "idea_type": best.get("idea_type") or "",
            "strike": strike,
            "opt_entry": opt_price,
            "contracts": contracts,
        }

    if pos and snaps:
        last = snaps[-1]
        price = last.get("underlying_price", pos["entry"])
        ul_pnl = price - pos["entry"] if pos["type"] == "CALLS" else pos["entry"] - price
        ul_pnl_pct = (ul_pnl / pos["entry"] * 100) if pos["entry"] else 0
        min_elapsed = _minutes_between(pos["entry_time"], last.get("time", ""))
        opt_entry = pos["opt_entry"]
        opt_exit = _estimate_opt_price(
            price,
            pos["strike"],
            "call" if pos["type"] == "CALLS" else "put",
            5,
            min_elapsed / (6.5 * 60),
            iv,
        )
        if opt_exit < 0.01:
            opt_exit = 0.01
        opt_pnl = pos["contracts"] * 100 * (opt_exit - opt_entry)
        opt_pnl_pct = (opt_exit - opt_entry) / opt_entry * 100 if opt_entry > 0 else 0

        trades.append(
            {
                "type": pos["type"],
                "entry": pos["entry"],
                "entry_time": pos["entry_time"],
                "exit": price,
                "exit_time": last.get("time", ""),
                "pnl": ul_pnl,
                "pnl_pct": ul_pnl_pct,
                "reason": "EOD",
                "idea_type": pos["idea_type"],
                "opt_entry": opt_entry,
                "opt_exit": opt_exit,
                "opt_pnl": opt_pnl,
                "opt_pnl_pct": opt_pnl_pct,
                "contracts": pos["contracts"],
                "strike": pos["strike"],
            }
        )
    return trades


def _summarize_day(date_str: str, trades: list[dict]) -> dict:
    total = sum(t.get("opt_pnl", 0) for t in trades)
    wins = sum(1 for t in trades if (t.get("opt_pnl", 0) > 0))
    losses = sum(1 for t in trades if (t.get("opt_pnl", 0) <= 0))
    reasons = {}
    reason_pnl = {}
    for t in trades:
        r = t.get("reason", "UNK")
        reasons[r] = reasons.get(r, 0) + 1
        reason_pnl[r] = reason_pnl.get(r, 0) + (t.get("opt_pnl", 0) or 0)
    return {
        "date": date_str,
        "trades": len(trades),
        "wins": wins,
        "losses": losses,
        "win_rate": (wins / len(trades) * 100) if trades else 0.0,
        "total_opt_pnl": total,
        "reasons": reasons,
        "reason_pnl": reason_pnl,
    }


def _print_summary(label: str, day_summaries: list[dict]) -> None:
    print(f"\n{label}")
    print("-" * len(label))
    total_pnl = sum(d["total_opt_pnl"] for d in day_summaries)
    total_trades = sum(d["trades"] for d in day_summaries)
    total_wins = sum(d["wins"] for d in day_summaries)
    total_losses = sum(d["losses"] for d in day_summaries)
    win_rate = (total_wins / total_trades * 100) if total_trades else 0.0
    print(f"Total PnL: ${total_pnl:,.2f} | Trades: {total_trades} | Win rate: {win_rate:.1f}%")
    for d in day_summaries:
        print(
            f"{d['date']}  PnL=${d['total_opt_pnl']:,.2f}  "
            f"Trades={d['trades']}  WR={d['win_rate']:.1f}%  "
            f"Reasons={d['reasons']}"
        )


def _run_dates(snaps_by_date: dict[str, list[dict]], cfg: StrategyConfig, capital: float) -> list[dict]:
    day_summaries = []
    for d, snaps in snaps_by_date.items():
        trades = _simulate_trades(snaps, cfg, capital=capital)
        day_summaries.append(_summarize_day(d, trades))
    return day_summaries


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=5, help="Number of most recent days to test")
    parser.add_argument("--step", type=int, default=5, help="Bars between checkpoints")
    parser.add_argument("--playbook", type=str, default="all", help="Playbook mode")
    parser.add_argument("--capital", type=float, default=2000.0, help="Capital per trade")
    parser.add_argument("--grid", action="store_true", help="Run a small grid search for better params")
    args = parser.parse_args()

    all_dates = sorted(p.stem for p in DATA_DIR.glob("*.csv"))
    if not all_dates:
        print("No data found in data/qqq/.")
        return 1
    dates = all_dates[-args.days :]

    snaps_by_date = {d: _make_snapshots(d, step_bars=args.step, playbook=args.playbook) for d in dates}

    base = StrategyConfig(name="baseline")
    base_summaries = _run_dates(snaps_by_date, base, args.capital)
    _print_summary("Baseline (frontend logic)", base_summaries)

    improved = StrategyConfig(
        name="aggressive_sr_reversal_v1",
        confidence_min=0.30,
        reg_score_min=1,
        rr_min=0.8,
        max_trades=5,
        cooldown_bars=2,
        trail_trigger=0.5,
        be_buffer=0.03,
        min_risk_atr_mult=0.2,
        entry_distance_mult=2.0,
        opt_rr_min=1.0,
        premium_risk_pct=0.20,
        dollars_per_trade=400.0,
        opt_take_profit_pct=None,
        opt_stop_loss_pct=0.12,
        max_daily_losses=2,
        max_daily_drawdown=1000,
        allow_choppy=True,
        allow_range_bound=True,
        entry_cutoff_utc_min=20 * 60,
        direction_mode="put_only",
        require_trend_regime=False,
        use_underlying_exits=True,
        sr_reversal_only=False,
        sr_proximity_pct=0.25,
        sr_min_strength=40,
        sr_wick_ratio_min=0.20,
        sr_reversal_boost=1,
        sr_reversal_penalty=1,
    )
    improved_summaries = _run_dates(snaps_by_date, improved, args.capital)
    _print_summary("Improved (option RR + risk cap)", improved_summaries)

    if not args.grid:
        return 0

    configs = []
    for conf in (0.45, 0.55, 0.65, 0.75):
        for reg_min in (1, 2, 3):
            for rr in (1.2, 1.5):
                for max_trades in (1, 2):
                    for inv in (False, True):
                        for opt_rr in (None, 1.1):
                            for premium_risk in (0.15, 0.25, 0.4):
                                for tp in (0.2, 0.3):
                                    for sl in (0.12, 0.18):
                                        for max_losses in (1, 2):
                                            for allow_choppy in (False, True):
                                                for mode in ("both", "call_only", "put_only"):
                                                    for trend_only in (False, True):
                                                        for ul_exits in (False, True):
                                                            name = (
                                                                f"c{conf}_rg{reg_min}_rr{rr}_mt{max_trades}_"
                                                                f"{'inv' if inv else 'norm'}_opr{opt_rr}_"
                                                                f"pr{premium_risk}_tp{tp}_sl{sl}_ml{max_losses}_"
                                                                f"{'ch' if allow_choppy else 'noch'}_{mode}_"
                                                                f"{'trend' if trend_only else 'any'}_"
                                                                f"{'ul' if ul_exits else 'opt'}"
                                                            )
                                                            configs.append(
                                                                StrategyConfig(
                                                                    name=name,
                                                                    confidence_min=conf,
                                                                    reg_score_min=reg_min,
                                                                    rr_min=rr,
                                                                    max_trades=max_trades,
                                                                    invert_direction=inv,
                                                                    opt_rr_min=opt_rr,
                                                                    premium_risk_pct=premium_risk,
                                                                    opt_take_profit_pct=tp,
                                                                    opt_stop_loss_pct=sl,
                                                                    max_daily_losses=max_losses,
                                                                    max_daily_drawdown=300,
                                                                    allow_choppy=allow_choppy,
                                                                    allow_range_bound=True,
                                                                    entry_cutoff_utc_min=19 * 60,
                                                                    direction_mode=mode,
                                                                    require_trend_regime=trend_only,
                                                                    use_underlying_exits=ul_exits,
                                                                )
                                                            )

    scored = []
    for cfg in configs:
        sums = _run_dates(snaps_by_date, cfg, args.capital)
        total_pnl = sum(d["total_opt_pnl"] for d in sums)
        total_trades = sum(d["trades"] for d in sums)
        win_rate = (
            sum(d["wins"] for d in sums) / total_trades * 100 if total_trades else 0.0
        )
        scored.append((total_pnl, win_rate, total_trades, cfg))

    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    print("\nTop configs (by total PnL):")
    shown = 0
    for total_pnl, win_rate, total_trades, cfg in scored:
        if total_trades <= 0:
            continue
        print(
            f"{cfg.name}  PnL=${total_pnl:,.2f}  Trades={total_trades}  WR={win_rate:.1f}%"
        )
        shown += 1
        if shown >= 5:
            break

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
