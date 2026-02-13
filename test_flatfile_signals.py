"""
Backtest VPA v3 trade-signal algorithm against locally-saved QQQ minute bars.

Reads from data/qqq/*.csv (downloaded once via download_flatfiles.py).
No S3 downloads needed – runs instantly.
"""
import sys
import os
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from app.vpa_engine import VPAEngine, VPASignal, TradeSignal

DATA_DIR = Path(__file__).parent / "data" / "qqq"


def load_day(date_str: str) -> pd.DataFrame:
    """Load a single day's QQQ bars from local CSV, filter to RTH."""
    path = DATA_DIR / f"{date_str}.csv"
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path)
    df["volume"] = df["volume"].astype(int)
    df["open"] = df["open"].astype(float)
    df["close"] = df["close"].astype(float)
    df["high"] = df["high"].astype(float)
    df["low"] = df["low"].astype(float)
    # nanosecond epoch → datetime
    df["datetime"] = pd.to_datetime(df["window_start"].astype(int), unit="ns").dt.strftime("%Y-%m-%d %H:%M:%S")
    df = df.sort_values("datetime").reset_index(drop=True)

    # Filter to RTH (09:30–16:00 ET = 14:30–21:00 UTC)
    df["_dt"] = pd.to_datetime(df["datetime"])
    df = df[(df["_dt"].dt.hour >= 14) | ((df["_dt"].dt.hour == 14) & (df["_dt"].dt.minute >= 30))]
    df = df[df["_dt"].dt.hour < 21]
    df = df.drop(columns=["_dt"]).reset_index(drop=True)
    return df


def backtest_day(df: pd.DataFrame, contract_type: str = "C") -> dict:
    """Run VPA + trade signals on one day. Returns stats dict."""
    engine = VPAEngine()
    vpa_df = df[["datetime", "open", "high", "low", "close", "volume"]].copy()
    results = engine.analyze(vpa_df)
    trades = engine.generate_trade_signals(results, contract_type=contract_type)

    buys = [t for t in trades if t.action == "BUY"]
    sells = [t for t in trades if t.action == "SELL"]

    total_pnl = 0.0
    wins = 0
    losses = 0
    for t in sells:
        try:
            pl_str = t.reason.split("P/L: ")[1].split(",")[0]
            pnl = float(pl_str)
            total_pnl += pnl
            if pnl > 0:
                wins += 1
            else:
                losses += 1
        except Exception:
            pass

    open_p = df["open"].iloc[0]
    close_p = df["close"].iloc[-1]
    day_change = (close_p - open_p) / open_p * 100

    return {
        "buys": len(buys),
        "sells": len(sells),
        "wins": wins,
        "losses": losses,
        "pnl": total_pnl,
        "day_change": day_change,
        "open": open_p,
        "close": close_p,
        "trades": trades,
    }


def main():
    dates = sorted(p.stem for p in DATA_DIR.glob("*.csv"))
    if not dates:
        print("No data found in data/qqq/. Run download_flatfiles.py first.")
        return

    print("=" * 70)
    print(f"  VPA v3 Backtest – {len(dates)} trading days ({dates[0]} to {dates[-1]})")
    print("=" * 70)

    # ── CALL backtest ────────────────────────────────────────
    print(f"\n{'─'*70}")
    print("  CALL MODE (bullish = BUY)")
    print(f"{'─'*70}")
    call_results = []
    for d in dates:
        df = load_day(d)
        if df.empty or len(df) < 20:
            continue
        stats = backtest_day(df, contract_type="C")
        call_results.append({"date": d, **stats})
        print(f"  {d}:  {stats['buys']}B/{stats['sells']}S  "
              f"W={stats['wins']} L={stats['losses']}  "
              f"P/L=${stats['pnl']:+.2f}  day={stats['day_change']:+.1f}%")

    total_call_pnl = sum(r["pnl"] for r in call_results)
    total_call_trades = sum(r["sells"] for r in call_results)
    total_call_wins = sum(r["wins"] for r in call_results)
    total_call_losses = sum(r["losses"] for r in call_results)

    print(f"\n  CALL SUMMARY ({len(call_results)} days)")
    print(f"  {'='*50}")
    print(f"  Total round-trips: {total_call_trades}")
    print(f"  Wins: {total_call_wins}  |  Losses: {total_call_losses}")
    if total_call_trades:
        print(f"  Win rate: {total_call_wins / total_call_trades * 100:.1f}%")
        print(f"  Total P/L: ${total_call_pnl:+.2f}")
        print(f"  Avg P/L per trade: ${total_call_pnl / total_call_trades:+.2f}")

    # ── PUT backtest ─────────────────────────────────────────
    print(f"\n{'─'*70}")
    print("  PUT MODE (bearish = BUY)")
    print(f"{'─'*70}")
    put_results = []
    for d in dates:
        df = load_day(d)
        if df.empty or len(df) < 20:
            continue
        stats = backtest_day(df, contract_type="P")
        put_results.append({"date": d, **stats})
        print(f"  {d}:  {stats['buys']}B/{stats['sells']}S  "
              f"W={stats['wins']} L={stats['losses']}  "
              f"P/L=${stats['pnl']:+.2f}  day={stats['day_change']:+.1f}%")

    total_put_pnl = sum(r["pnl"] for r in put_results)
    total_put_trades = sum(r["sells"] for r in put_results)
    total_put_wins = sum(r["wins"] for r in put_results)
    total_put_losses = sum(r["losses"] for r in put_results)

    print(f"\n  PUT SUMMARY ({len(put_results)} days)")
    print(f"  {'='*50}")
    print(f"  Total round-trips: {total_put_trades}")
    print(f"  Wins: {total_put_wins}  |  Losses: {total_put_losses}")
    if total_put_trades:
        print(f"  Win rate: {total_put_wins / total_put_trades * 100:.1f}%")
        print(f"  Total P/L: ${total_put_pnl:+.2f}")
        print(f"  Avg P/L per trade: ${total_put_pnl / total_put_trades:+.2f}")

    # ── Combined summary ─────────────────────────────────────
    combined_trades = total_call_trades + total_put_trades
    combined_pnl = total_call_pnl + total_put_pnl
    combined_wins = total_call_wins + total_put_wins
    combined_losses = total_call_losses + total_put_losses

    print(f"\n{'='*70}")
    print(f"  COMBINED SUMMARY")
    print(f"{'='*70}")
    print(f"  Days tested: {len(dates)}")
    print(f"  Total round-trips: {combined_trades}")
    print(f"  Wins: {combined_wins}  |  Losses: {combined_losses}")
    if combined_trades:
        print(f"  Win rate: {combined_wins / combined_trades * 100:.1f}%")
        print(f"  Total P/L: ${combined_pnl:+.2f}")
        print(f"  Avg P/L per trade: ${combined_pnl / combined_trades:+.2f}")
        print(f"  Avg trades per day: {combined_trades / len(dates):.1f}")


if __name__ == "__main__":
    main()
