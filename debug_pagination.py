"""Debug Polygon pagination to understand why tickers are missing."""
from app.config import settings
import httpx

c = httpx.Client(timeout=15, headers={"Authorization": f"Bearer {settings.POLYGON_API_KEY}"})

# Page 1
r = c.get("https://api.polygon.io/v3/reference/tickers", params={
    "market": "stocks", "active": "true", "limit": 1000, "order": "asc", "sort": "ticker"
})
d = r.json()
results = d.get("results", [])
print(f"Page 1: {len(results)} results")
print(f"  First: {results[0]['ticker']}  Last: {results[-1]['ticker']}")

# All types in page 1
from collections import Counter
types = Counter(r.get("type", "?") for r in results)
print(f"  Types: {dict(types)}")

next_url = d.get("next_url", "")
print(f"  next_url: {next_url[:150]}")

# Page 2
r2 = c.get(next_url)
d2 = r2.json()
results2 = d2.get("results", [])
print(f"\nPage 2: {len(results2)} results")
if results2:
    print(f"  First: {results2[0]['ticker']}  Last: {results2[-1]['ticker']}")
    types2 = Counter(r.get("type", "?") for r in results2)
    print(f"  Types: {dict(types2)}")

page1_syms = {r["ticker"] for r in results}
page2_syms = {r["ticker"] for r in results2}
overlap = page1_syms & page2_syms
new = page2_syms - page1_syms
print(f"  Overlap with page 1: {len(overlap)}")
print(f"  New tickers: {len(new)}")
if new:
    print(f"  Sample new: {sorted(new)[:15]}")

# Check: does page 1 cover past letter A?
last_a = max(t for t in page1_syms if t.startswith("A"))
first_b = min(t for t in page1_syms if t.startswith("B")) if any(t.startswith("B") for t in page1_syms) else "N/A"
print(f"\nPage 1 last A-ticker: {last_a}")  
print(f"Page 1 first B-ticker: {first_b}")

# Total unique CS+ETF in page 1
cs_etf = [r for r in results if r.get("type") in ("CS", "ETF", "ADRC")]
print(f"CS+ETF+ADRC in page 1: {len(cs_etf)}")
