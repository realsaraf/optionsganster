"""
One-time script to download all US stock/ETF tickers from Polygon
and save to data/tickers.json for instant autocomplete on startup.

Uses the polygon-api-client library which handles pagination automatically.
Requires a Polygon subscription (stocks + options real-time).

Re-run periodically (weekly/monthly) to pick up new listings / delistings.
"""
import json, os, sys
from polygon import RESTClient

# ── Load API key via app config ─────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from app.config import settings

API_KEY = settings.POLYGON_API_KEY
if not API_KEY:
    print("ERROR: No POLYGON_API_KEY in .env")
    sys.exit(1)

print(f"Using API key: {API_KEY[:6]}...{API_KEY[-4:]}")

OUT_PATH = os.path.join(os.path.dirname(__file__), "data", "tickers.json")
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

# ── Fetch using polygon client (auto-pagination) ───────────
client = RESTClient(api_key=API_KEY)

print("Fetching all active US stock & ETF tickers from Polygon...")
seen = set()
all_tickers = []
count = 0

for t in client.list_tickers(market="stocks", active=True, limit=1000, order="asc", sort="ticker"):
    count += 1
    sym = t.ticker
    ttype = getattr(t, "type", "")
    if ttype not in ("CS", "ETF", "ADRC"):
        continue
    if sym in seen:
        continue
    seen.add(sym)
    all_tickers.append({
        "s": sym,
        "n": getattr(t, "name", "") or "",
        "t": "E" if ttype == "ETF" else "S",
    })
    if len(all_tickers) % 1000 == 0:
        print(f"  ...{len(all_tickers)} tickers so far (scanned {count} raw)")

print(f"Scanned {count} raw tickers, kept {len(all_tickers)} (CS/ETF/ADRC)")

# ── Sort & save compact JSON ───────────────────────────────
all_tickers.sort(key=lambda t: t["s"])

with open(OUT_PATH, "w") as f:
    json.dump(all_tickers, f, separators=(",", ":"))

size_kb = os.path.getsize(OUT_PATH) / 1024
etfs = sum(1 for t in all_tickers if t["t"] == "E")
stocks = len(all_tickers) - etfs

# Sanity check
must_have = ["AAPL", "QQQ", "SPY", "TSLA", "COST", "NVDA", "MSFT", "AMZN", "META", "GOOGL"]
missing = [s for s in must_have if s not in seen]

print(f"\nSaved {len(all_tickers)} tickers ({etfs} ETFs, {stocks} stocks) -> {OUT_PATH} ({size_kb:.0f} KB)")
if missing:
    print(f"WARNING: Missing expected tickers: {missing}")
else:
    print("All expected major tickers present")
