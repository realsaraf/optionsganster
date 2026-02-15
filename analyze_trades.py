"""Analyze options trades from brokerage JSON export."""
import json
from datetime import datetime
from collections import defaultdict

with open("data/trades/Cash_XXX523_Transactions_20260214-205503.json") as f:
    data = json.load(f)

txns = data["BrokerageTransactions"]

# Filter to options-only actions
OPTION_ACTIONS = {"Buy to Open", "Sell to Close", "Expired", "Exchange or Exercise"}

# Group by symbol
trades_by_symbol = defaultdict(lambda: {"buys": [], "sells": [], "expired": False, "exercised": False})

for t in txns:
    action = t["Action"]
    symbol = t["Symbol"]
    
    # Skip non-options (stock buy/sell)
    if action not in OPTION_ACTIONS:
        continue
    
    # Parse price/amount
    def parse_money(s):
        if not s:
            return 0.0
        return float(s.replace("$", "").replace(",", "").replace("-", "")) * (-1 if s.startswith("-") else 1)
    
    qty = int(t["Quantity"].replace(",", ""))
    price = float(t["Price"].replace("$", "").replace(",", "")) if t["Price"] else 0.0
    fees = parse_money(t["Fees & Comm"]) if t["Fees & Comm"] else 0.0
    amount = parse_money(t["Amount"]) if t["Amount"] else 0.0
    
    # Parse the date (just the actual date part)
    date_str = t["Date"].split(" as of ")[0]
    try:
        date = datetime.strptime(date_str, "%m/%d/%Y")
    except:
        date = None
    
    entry = {"qty": abs(qty), "price": price, "fees": fees, "amount": amount, "date": date, "action": action}
    
    if action == "Buy to Open":
        trades_by_symbol[symbol]["buys"].append(entry)
    elif action == "Sell to Close":
        trades_by_symbol[symbol]["sells"].append(entry)
    elif action == "Expired":
        trades_by_symbol[symbol]["expired"] = True
    elif action == "Exchange or Exercise":
        trades_by_symbol[symbol]["exercised"] = True

# Build summary table
results = []
for symbol, data in trades_by_symbol.items():
    buys = data["buys"]
    sells = data["sells"]
    
    if not buys:
        continue  # skip expired-only with no buy record
    
    total_qty_bought = sum(b["qty"] for b in buys)
    total_cost = sum(b["qty"] * b["price"] * 100 for b in buys)  # options are per-contract (100 shares)
    total_buy_fees = sum(b["fees"] for b in buys)
    
    # Weighted avg buy price
    avg_buy = total_cost / (total_qty_bought * 100) if total_qty_bought else 0
    
    total_qty_sold = sum(s["qty"] for s in sells)
    total_proceeds = sum(s["qty"] * s["price"] * 100 for s in sells)
    total_sell_fees = sum(s["fees"] for s in sells)
    
    avg_sell = total_proceeds / (total_qty_sold * 100) if total_qty_sold else 0
    
    # Determine outcome
    if data["exercised"]:
        outcome = "EXERCISED"
    elif data["expired"] and not sells:
        outcome = "EXPIRED"
    elif sells:
        outcome = "CLOSED"
    else:
        outcome = "UNKNOWN"
    
    # Net P&L = proceeds - cost - all fees
    total_fees = total_buy_fees + total_sell_fees
    net_pnl = total_proceeds - total_cost - total_fees
    
    # Parse option details from symbol: "QQQ 02/13/2026 604.00 C"
    parts = symbol.split()
    underlying = parts[0]
    exp_date = parts[1] if len(parts) > 1 else ""
    strike = parts[2] if len(parts) > 2 else ""
    opt_type = parts[3] if len(parts) > 3 else ""
    opt_type_full = "CALL" if opt_type == "C" else "PUT" if opt_type == "P" else opt_type
    
    # Earliest buy date for sorting
    earliest_date = min(b["date"] for b in buys if b["date"]) if buys else None
    
    results.append({
        "date": earliest_date,
        "symbol": symbol,
        "type": opt_type_full,
        "strike": strike,
        "expiry": exp_date,
        "qty": total_qty_bought,
        "avg_buy": avg_buy,
        "avg_sell": avg_sell,
        "total_cost": total_cost,
        "total_proceeds": total_proceeds,
        "total_fees": total_fees,
        "net_pnl": net_pnl,
        "outcome": outcome,
    })

# Sort by date ascending
results.sort(key=lambda x: x["date"] if x["date"] else datetime.min)

# Print table
print(f"\n{'='*160}")
print(f"  OPTIONS TRADE ANALYSIS  |  01/02/2026 - 02/13/2026")
print(f"{'='*160}")
print(f"{'Date':<12} {'Symbol':<30} {'Type':<5} {'Strike':>7} {'Qty':>5} {'Avg Buy':>9} {'Avg Sell':>9} {'Total Cost':>12} {'Proceeds':>12} {'Fees':>9} {'Net P&L':>12} {'Outcome':<10}")
print(f"{'-'*160}")

total_pnl = 0
total_fees_all = 0
wins = 0
losses = 0

for r in results:
    date_str = r["date"].strftime("%m/%d/%Y") if r["date"] else "N/A"
    pnl_str = f"${r['net_pnl']:>+,.2f}"
    marker = "WIN" if r["net_pnl"] > 0 else "LOSS" if r["net_pnl"] < 0 else "EVEN"
    
    if r["net_pnl"] > 0:
        wins += 1
    elif r["net_pnl"] < 0:
        losses += 1
    
    total_pnl += r["net_pnl"]
    total_fees_all += r["total_fees"]
    
    print(f"{date_str:<12} {r['symbol']:<30} {r['type']:<5} {r['strike']:>7} {r['qty']:>5} ${r['avg_buy']:>7.3f} ${r['avg_sell']:>7.3f} ${r['total_cost']:>10,.2f} ${r['total_proceeds']:>10,.2f} ${r['total_fees']:>7,.2f} {pnl_str:>12} {r['outcome']:<10}")

print(f"{'-'*160}")
print(f"\n{'='*80}")
print(f"  SUMMARY")
print(f"{'='*80}")
print(f"  Total Trades:     {len(results)}")
print(f"  Wins:             {wins}  ({wins/(wins+losses)*100:.1f}%)" if (wins+losses) > 0 else "")
print(f"  Losses:           {losses}  ({losses/(wins+losses)*100:.1f}%)" if (wins+losses) > 0 else "")
print(f"  Total Fees:       ${total_fees_all:>,.2f}")
print(f"  Total Net P&L:    ${total_pnl:>+,.2f}")
print(f"{'='*80}")
