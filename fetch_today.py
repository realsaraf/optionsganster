"""Quick script to fetch today's QQQ minute bars from Polygon and save as CSV."""
import httpx
import csv
from pathlib import Path
from app.config import settings

KEY = settings.POLYGON_API_KEY
DATE = "2026-02-25"
url = f"https://api.polygon.io/v2/aggs/ticker/QQQ/range/1/minute/{DATE}/{DATE}?adjusted=true&sort=asc&limit=50000&apiKey={KEY}"

r = httpx.get(url, timeout=30)
d = r.json()
print(f"Status: {d.get('status')}, count: {d.get('resultsCount', 0)}")

if d.get("results"):
    rows = d["results"]
    print(f"First bar t={rows[0]['t']}, o={rows[0]['o']}, c={rows[0]['c']}")
    print(f"Last bar  t={rows[-1]['t']}, o={rows[-1]['o']}, c={rows[-1]['c']}")

    outpath = Path("data/qqq") / f"{DATE}.csv"
    outpath.parent.mkdir(parents=True, exist_ok=True)
    with open(outpath, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ticker", "window_start", "open", "close", "high", "low", "volume", "transactions"])
        for bar in rows:
            # Polygon returns ms epoch, flat files use ns epoch
            w.writerow(["QQQ", bar["t"] * 1_000_000, bar["o"], bar["c"], bar["h"], bar["l"], bar["v"], bar.get("n", 0)])
    print(f"Saved {len(rows)} rows to {outpath}")
else:
    print("No results - market may not have opened yet or data not available")
    print(f"Response: {d}")
