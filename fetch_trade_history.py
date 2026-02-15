"""
Fetch historical market data for all traded options and underlying QQQ.
Saves CSVs into data/trades/analysis/
"""
import json
import asyncio
import httpx
import os
from datetime import datetime, date, timedelta
from collections import defaultdict
from zoneinfo import ZoneInfo
from pathlib import Path

# ── Config ──────────────────────────────────────────────────
from app.config import settings

API_KEY = settings.POLYGON_API_KEY
BASE_URL = "https://api.polygon.io"
_ET = ZoneInfo("America/New_York")

OUT_DIR = Path("data/trades/analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Rate-limit: polygon free/starter = 5 req/min, paid = higher
# We'll do conservative pacing
RATE_DELAY = 0.25  # seconds between API calls


async def get_json(client: httpx.AsyncClient, url: str, params: dict = None, retries=3) -> dict:
    for attempt in range(retries):
        resp = await client.get(url, params=params)
        if resp.status_code == 429:
            wait = 5 * (attempt + 1)
            print(f"  Rate limited, waiting {wait}s...")
            await asyncio.sleep(wait)
            continue
        resp.raise_for_status()
        return resp.json()
    resp = await client.get(url, params=params)
    resp.raise_for_status()
    return resp.json()


def parse_option_symbol(symbol_str: str):
    """
    Parse brokerage symbol like 'QQQ 02/13/2026 604.00 C'
    Returns (underlying, expiry_date, strike, right, polygon_ticker)
    """
    parts = symbol_str.split()
    underlying = parts[0]
    exp_str = parts[1]  # MM/DD/YYYY
    strike_str = parts[2]
    right = parts[3]  # C or P

    exp_date = datetime.strptime(exp_str, "%m/%d/%Y").date()
    strike = float(strike_str)

    # Build Polygon OCC ticker: O:QQQ260213C00604000
    exp_fmt = exp_date.strftime("%y%m%d")
    strike_occ = f"{int(strike * 1000):08d}"
    polygon_ticker = f"O:{underlying}{exp_fmt}{right}{strike_occ}"

    return underlying, exp_date, strike, right, polygon_ticker


async def fetch_aggs(client: httpx.AsyncClient, ticker: str, start: date, end: date,
                     timespan="minute", multiplier=5) -> list[dict]:
    """Fetch aggregate bars from Polygon."""
    url = f"{BASE_URL}/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{start}/{end}"
    params = {"limit": 50000, "sort": "asc"}
    try:
        data = await get_json(client, url, params)
        return data.get("results", [])
    except httpx.HTTPStatusError as e:
        print(f"  ERROR fetching {ticker}: {e.response.status_code} {e.response.text[:200]}")
        return []


async def fetch_daily_aggs(client: httpx.AsyncClient, ticker: str, start: date, end: date) -> list[dict]:
    """Fetch daily bars from Polygon."""
    url = f"{BASE_URL}/v2/aggs/ticker/{ticker}/range/1/day/{start}/{end}"
    params = {"limit": 50000, "sort": "asc"}
    try:
        data = await get_json(client, url, params)
        return data.get("results", [])
    except httpx.HTTPStatusError as e:
        print(f"  ERROR fetching daily {ticker}: {e.response.status_code} {e.response.text[:200]}")
        return []


def bars_to_csv(bars: list[dict], filepath: Path, daily=False):
    """Convert Polygon bars to CSV."""
    if not bars:
        print(f"  No data for {filepath.name}")
        return False

    lines = ["datetime,open,high,low,close,volume,vwap"]
    for bar in bars:
        if daily:
            dt_utc = datetime.fromtimestamp(bar["t"] / 1000, tz=ZoneInfo("UTC"))
            dt = dt_utc.strftime("%Y-%m-%d")
        else:
            dt_utc = datetime.fromtimestamp(bar["t"] / 1000, tz=ZoneInfo("UTC"))
            dt = dt_utc.astimezone(_ET).strftime("%Y-%m-%d %H:%M:%S")
        vwap = bar.get("vw", "")
        lines.append(f"{dt},{bar['o']},{bar['h']},{bar['l']},{bar['c']},{bar.get('v', 0)},{vwap}")

    filepath.write_text("\n".join(lines))
    print(f"  Saved {filepath.name} ({len(bars)} bars)")
    return True


async def main():
    # Load trades
    with open("data/trades/Cash_XXX523_Transactions_20260214-205503.json") as f:
        raw = json.load(f)

    txns = raw["BrokerageTransactions"]

    # Collect unique option contracts and their trade dates
    OPTION_ACTIONS = {"Buy to Open", "Sell to Close", "Expired", "Exchange or Exercise"}
    contracts = {}  # polygon_ticker -> {symbol, underlying, exp_date, strike, right, trade_dates}

    all_trade_dates = set()

    for t in txns:
        if t["Action"] not in OPTION_ACTIONS:
            continue

        symbol = t["Symbol"]
        parts = symbol.split()
        if len(parts) < 4:
            continue  # skip non-option symbols

        underlying, exp_date, strike, right, polygon_ticker = parse_option_symbol(symbol)

        date_str = t["Date"].split(" as of ")[0]
        try:
            trade_date = datetime.strptime(date_str, "%m/%d/%Y").date()
        except:
            continue

        if polygon_ticker not in contracts:
            contracts[polygon_ticker] = {
                "symbol": symbol,
                "underlying": underlying,
                "exp_date": exp_date,
                "strike": strike,
                "right": right,
                "trade_dates": set(),
            }
        contracts[polygon_ticker]["trade_dates"].add(trade_date)
        all_trade_dates.add(trade_date)

    print(f"Found {len(contracts)} unique option contracts")
    print(f"Trade dates span: {min(all_trade_dates)} to {max(all_trade_dates)}")
    print(f"Output directory: {OUT_DIR.absolute()}\n")

    async with httpx.AsyncClient(
        timeout=30.0,
        headers={"Authorization": f"Bearer {API_KEY}"},
    ) as client:

        # ── 1) Fetch option intraday data (5-min bars) ──────────
        print("=" * 70)
        print("FETCHING OPTION INTRADAY DATA (5-min bars)")
        print("=" * 70)

        options_dir = OUT_DIR / "options_intraday"
        options_dir.mkdir(exist_ok=True)

        for i, (ticker, info) in enumerate(sorted(contracts.items(), key=lambda x: min(x[1]["trade_dates"]))):
            trade_dates = sorted(info["trade_dates"])
            exp = info["exp_date"]

            # Fetch from earliest trade date through expiration
            start = min(trade_dates)
            end = min(exp, date(2026, 2, 14))  # don't go past today

            print(f"\n[{i+1}/{len(contracts)}] {ticker}")
            print(f"  Traded: {', '.join(d.strftime('%m/%d') for d in trade_dates)} | Exp: {exp}")

            bars = await fetch_aggs(client, ticker, start, end, timespan="minute", multiplier=5)
            await asyncio.sleep(RATE_DELAY)

            safe_name = ticker.replace(":", "_")
            bars_to_csv(bars, options_dir / f"{safe_name}_5min.csv")

        # ── 2) Fetch option daily data ──────────────────────────
        print("\n" + "=" * 70)
        print("FETCHING OPTION DAILY DATA")
        print("=" * 70)

        options_daily_dir = OUT_DIR / "options_daily"
        options_daily_dir.mkdir(exist_ok=True)

        for i, (ticker, info) in enumerate(sorted(contracts.items(), key=lambda x: min(x[1]["trade_dates"]))):
            trade_dates = sorted(info["trade_dates"])
            exp = info["exp_date"]

            # Fetch from a few days before first trade through expiration
            start = min(trade_dates) - timedelta(days=5)
            end = min(exp, date(2026, 2, 14))

            print(f"\n[{i+1}/{len(contracts)}] {ticker}")

            bars = await fetch_daily_aggs(client, ticker, start, end)
            await asyncio.sleep(RATE_DELAY)

            safe_name = ticker.replace(":", "_")
            bars_to_csv(bars, options_daily_dir / f"{safe_name}_daily.csv", daily=True)

        # ── 3) Fetch underlying QQQ data ────────────────────────
        print("\n" + "=" * 70)
        print("FETCHING QQQ UNDERLYING DATA")
        print("=" * 70)

        stock_dir = OUT_DIR / "underlying"
        stock_dir.mkdir(exist_ok=True)

        overall_start = min(all_trade_dates) - timedelta(days=5)
        overall_end = date(2026, 2, 14)

        # Daily bars for the full period
        print(f"\nQQQ daily bars: {overall_start} to {overall_end}")
        bars = await fetch_daily_aggs(client, "QQQ", overall_start, overall_end)
        await asyncio.sleep(RATE_DELAY)
        bars_to_csv(bars, stock_dir / "QQQ_daily.csv", daily=True)

        # Intraday 5-min bars for each trade date
        for td in sorted(all_trade_dates):
            print(f"\nQQQ 5-min bars for {td}")
            bars = await fetch_aggs(client, "QQQ", td, td, timespan="minute", multiplier=5)
            await asyncio.sleep(RATE_DELAY)
            bars_to_csv(bars, stock_dir / f"QQQ_5min_{td}.csv")

    print("\n" + "=" * 70)
    print("DONE! All data saved to:", OUT_DIR.absolute())
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
