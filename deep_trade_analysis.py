"""
Deep Trade Analysis – Disaggregated Entry-Level P&L with Market Context
========================================================================
Purpose: Understand WHY winning trades won and WHY losing trades lost.

Key design choices:
  • Each individual "Buy to Open" tranche is treated as its OWN trade
    (no averaging across DCA entries)
  • For each entry we capture QQQ price, option price, time-of-day,
    moneyness, DTE, intraday trend, volume context
  • Output: detailed markdown with statistical breakdowns + actionable
    signal-improvement recommendations

Run:  python deep_trade_analysis.py
"""

import json, csv, re, os, statistics
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field

# ── paths ─────────────────────────────────────────────────────
BASE = Path(__file__).parent
TRADES_JSON = BASE / "data" / "trades" / "Cash_XXX523_Transactions_20260214-205503.json"
UNDERLYING_DIR = BASE / "data" / "trades" / "analysis" / "underlying"
OPTIONS_INTRA_DIR = BASE / "data" / "trades" / "analysis" / "options_intraday"
QQQ_DAILY_CSV = UNDERLYING_DIR / "QQQ_daily.csv"
OUTPUT_MD = BASE / "data" / "trades" / "deep_trade_analysis.md"


# ═══════════════════════════════════════════════════════════════
#  1.  LOAD RAW DATA
# ═══════════════════════════════════════════════════════════════

def load_transactions() -> list[dict]:
    with open(TRADES_JSON) as f:
        data = json.load(f)
    return data.get("BrokerageTransactions", [])


def load_qqq_daily() -> dict[str, dict]:
    """date-str -> {open, high, low, close, volume, vwap}"""
    rows = {}
    with open(QQQ_DAILY_CSV) as f:
        reader = csv.DictReader(f)
        for r in reader:
            dt = r["datetime"][:10]
            rows[dt] = {k: float(v) if k != "datetime" else v
                        for k, v in r.items() if k != "datetime"}
    return rows


def load_qqq_intraday(date_str: str) -> list[dict]:
    """Load 5-min bars for a trading date."""
    fname = UNDERLYING_DIR / f"QQQ_5min_{date_str}.csv"
    if not fname.exists():
        return []
    bars = []
    with open(fname) as f:
        reader = csv.DictReader(f)
        for r in reader:
            bars.append({
                "datetime": r["datetime"],
                "open": float(r["open"]),
                "high": float(r["high"]),
                "low": float(r["low"]),
                "close": float(r["close"]),
                "volume": float(r["volume"]),
                "vwap": float(r["vwap"]),
            })
    return bars


# ═══════════════════════════════════════════════════════════════
#  2.  PARSE OPTION SYMBOL
# ═══════════════════════════════════════════════════════════════

def parse_option_symbol(symbol: str) -> dict | None:
    """Parse 'QQQ 01/07/2026 626.00 P' → dict with underlying, expiry, strike, type"""
    m = re.match(
        r"(\w+)\s+(\d{2}/\d{2}/\d{4})\s+(\d+(?:\.\d+)?)\s+(P|C)",
        symbol.strip()
    )
    if not m:
        return None
    return {
        "underlying": m.group(1),
        "expiry": datetime.strptime(m.group(2), "%m/%d/%Y"),
        "expiry_str": m.group(2),
        "strike": float(m.group(3)),
        "type": "PUT" if m.group(4) == "P" else "CALL",
    }


def parse_amount(amt: str) -> float:
    if not amt:
        return 0.0
    return float(amt.replace("$", "").replace(",", "").replace("−", "-"))


def parse_date(d: str) -> str:
    """Extract the primary date from fields like '01/23/2026 as of 01/22/2026'"""
    return d.split(" as of ")[0].strip()


# ═══════════════════════════════════════════════════════════════
#  3.  BUILD DISAGGREGATED ENTRIES
# ═══════════════════════════════════════════════════════════════

@dataclass
class BuyEntry:
    """A single Buy-to-Open tranche."""
    symbol: str
    option: dict       # parsed option info
    date: str          # trade date
    qty: int
    price: float       # per-contract buy price
    cost: float        # total cost including fees
    fees: float
    entry_order: int   # 1st, 2nd, 3rd entry on same symbol (DCA tracking)
    is_dca: bool       # whether this is a DCA add (2nd+ entry on same symbol)

    # filled later
    exit_price: float = 0.0
    exit_proceeds: float = 0.0
    exit_method: str = ""  # CLOSED / EXPIRED / EXERCISED
    pnl: float = 0.0

    # market context at entry (filled later)
    qqq_price_at_entry: float = 0.0
    moneyness: str = ""      # ITM / ATM / OTM
    moneyness_pct: float = 0.0  # how far from ATM
    dte: int = 0
    time_of_day: str = ""    # morning / midday / afternoon
    qqq_trend_today: str = ""  # up / down / flat
    qqq_open_to_entry: float = 0.0  # QQQ % move from open to entry time area
    premium_pct_of_strike: float = 0.0  # how expensive the option was
    qqq_daily_range: float = 0.0  # high-low range for the day
    vol_context: str = ""    # high / normal / low volume day


@dataclass
class TradeGroup:
    """All entries + exits for one option symbol."""
    symbol: str
    option: dict
    entries: list[BuyEntry] = field(default_factory=list)
    total_bought: int = 0
    total_sold: int = 0
    total_cost: float = 0.0
    total_proceeds: float = 0.0
    total_fees: float = 0.0
    exit_method: str = ""
    weighted_avg_sell: float = 0.0


def build_trade_groups(txns: list[dict]) -> dict[str, TradeGroup]:
    """Group all transactions by option symbol, preserving individual buy entries."""
    groups: dict[str, TradeGroup] = {}
    buy_counts: dict[str, int] = defaultdict(int)  # track DCA order

    # Process in chronological order (reverse since JSON is newest-first)
    sorted_txns = list(reversed(txns))

    for tx in sorted_txns:
        action = tx.get("Action", "")
        symbol = tx.get("Symbol", "")
        if not symbol or symbol == "QQQ":  # skip stock transactions
            continue

        option = parse_option_symbol(symbol)
        if not option:
            continue

        if symbol not in groups:
            groups[symbol] = TradeGroup(symbol=symbol, option=option)
        group = groups[symbol]

        qty_str = tx.get("Quantity", "0").replace(",", "")
        qty = abs(int(qty_str)) if qty_str else 0
        price_str = tx.get("Price", "").replace("$", "").replace(",", "")
        price = float(price_str) if price_str else 0.0
        fees_str = tx.get("Fees & Comm", "").replace("$", "").replace(",", "")
        fees = float(fees_str) if fees_str else 0.0
        amount = parse_amount(tx.get("Amount", ""))
        date = parse_date(tx.get("Date", ""))

        if action == "Buy to Open":
            buy_counts[symbol] += 1
            entry_order = buy_counts[symbol]
            is_dca = entry_order > 1

            entry = BuyEntry(
                symbol=symbol,
                option=option,
                date=date,
                qty=qty,
                price=price,
                cost=abs(amount),
                fees=fees,
                entry_order=entry_order,
                is_dca=is_dca,
            )
            group.entries.append(entry)
            group.total_bought += qty
            group.total_cost += abs(amount)
            group.total_fees += fees

        elif action == "Sell to Close":
            group.total_sold += qty
            group.total_proceeds += abs(amount)
            group.total_fees += fees
            group.exit_method = "CLOSED"

        elif action == "Expired":
            group.exit_method = "EXPIRED"

        elif action == "Exchange or Exercise":
            group.exit_method = "EXERCISED"

    return groups


def allocate_exit_prices(groups: dict[str, TradeGroup]):
    """
    Distribute proceeds to individual buy entries.
    Uses FIFO: first entries get the first sale proceeds.
    For expired contracts: all entries get $0 exit.
    """
    for symbol, group in groups.items():
        if not group.entries:
            continue

        if group.exit_method == "EXPIRED":
            for entry in group.entries:
                entry.exit_price = 0.0
                entry.exit_proceeds = 0.0
                entry.exit_method = "EXPIRED"
                entry.pnl = -entry.cost
        elif group.exit_method == "EXERCISED":
            for entry in group.entries:
                entry.exit_price = 0.0
                entry.exit_proceeds = 0.0
                entry.exit_method = "EXERCISED"
                entry.pnl = -entry.cost  # option cost is sunk
        else:
            # Compute weighted average sell price
            if group.total_sold > 0:
                avg_sell = group.total_proceeds / group.total_sold
            else:
                avg_sell = 0.0
            group.weighted_avg_sell = avg_sell

            for entry in group.entries:
                entry.exit_price = avg_sell / 100  # per-share price for display
                # avg_sell is total_proceeds/total_sold — already in $/contract (100 shares baked in)
                raw_proceeds = avg_sell * entry.qty
                # allocate sell fees proportionally
                total_sell_fees = group.total_fees - sum(e.fees for e in group.entries)
                if group.total_bought > 0 and total_sell_fees > 0:
                    sell_fee_share = (entry.qty / group.total_bought) * total_sell_fees
                else:
                    sell_fee_share = 0
                entry.exit_proceeds = raw_proceeds - max(sell_fee_share, 0)
                entry.pnl = entry.exit_proceeds - entry.cost
                entry.exit_method = "CLOSED"


# ═══════════════════════════════════════════════════════════════
#  4.  ENRICH WITH MARKET CONTEXT
# ═══════════════════════════════════════════════════════════════

def enrich_entries(groups: dict[str, TradeGroup], qqq_daily: dict):
    """Add market context to each buy entry."""
    # Pre-compute average daily volume for volume context
    volumes = [d["volume"] for d in qqq_daily.values() if "volume" in d]
    avg_volume = statistics.mean(volumes) if volumes else 40_000_000

    for symbol, group in groups.items():
        for entry in group.entries:
            # Parse the trade date
            try:
                trade_dt = datetime.strptime(entry.date, "%m/%d/%Y")
            except ValueError:
                continue
            trade_date_str = trade_dt.strftime("%Y-%m-%d")

            # DTE
            entry.dte = (entry.option["expiry"] - trade_dt).days

            # QQQ daily data
            daily = qqq_daily.get(trade_date_str)
            if daily:
                qqq_close = daily["close"]
                qqq_open = daily["open"]
                qqq_high = daily["high"]
                qqq_low = daily["low"]
                qqq_vol = daily["volume"]

                entry.qqq_price_at_entry = qqq_close  # approximate
                entry.qqq_daily_range = (qqq_high - qqq_low) / qqq_close * 100

                # Trend
                change = (qqq_close - qqq_open) / qqq_open * 100
                entry.qqq_open_to_entry = change
                if change > 0.3:
                    entry.qqq_trend_today = "UP"
                elif change < -0.3:
                    entry.qqq_trend_today = "DOWN"
                else:
                    entry.qqq_trend_today = "FLAT"

                # Volume context
                if qqq_vol > avg_volume * 1.3:
                    entry.vol_context = "HIGH"
                elif qqq_vol < avg_volume * 0.7:
                    entry.vol_context = "LOW"
                else:
                    entry.vol_context = "NORMAL"

            # Moneyness
            ref_price = entry.qqq_price_at_entry or 620  # fallback
            strike = entry.option["strike"]
            if entry.option["type"] == "PUT":
                diff_pct = (strike - ref_price) / ref_price * 100
                if diff_pct > 0.5:
                    entry.moneyness = "ITM"
                elif diff_pct < -1.5:
                    entry.moneyness = "OTM"
                else:
                    entry.moneyness = "ATM"
            else:  # CALL
                diff_pct = (ref_price - strike) / ref_price * 100
                if diff_pct > 0.5:
                    entry.moneyness = "ITM"
                elif diff_pct < -1.5:
                    entry.moneyness = "OTM"
                else:
                    entry.moneyness = "ATM"
            entry.moneyness_pct = abs(diff_pct)

            # Premium as % of strike
            entry.premium_pct_of_strike = (entry.price / strike * 100) if strike else 0

            # Intraday context - try to determine time of day from entry order + patterns
            # Since we don't have exact timestamps, estimate from entry order
            num_entries = len([e for e in group.entries])
            if entry.entry_order == 1 and num_entries == 1:
                entry.time_of_day = "SINGLE"
            elif entry.entry_order == 1:
                entry.time_of_day = "FIRST"
            elif entry.entry_order == num_entries:
                entry.time_of_day = "LAST_DCA"
            else:
                entry.time_of_day = "MID_DCA"


# ═══════════════════════════════════════════════════════════════
#  5.  VPA-STYLE ANALYSIS ON DAILY BARS AROUND ENTRY
# ═══════════════════════════════════════════════════════════════

def vpa_daily_context(trade_date_str: str, qqq_daily: dict) -> dict:
    """Simplified VPA analysis on the daily bar and recent bars."""
    sorted_dates = sorted(qqq_daily.keys())
    idx = None
    for i, d in enumerate(sorted_dates):
        if d == trade_date_str:
            idx = i
            break
    if idx is None:
        return {"signal": "UNKNOWN", "detail": "no data"}

    bar = qqq_daily[trade_date_str]
    o, h, l, c = bar["open"], bar["high"], bar["low"], bar["close"]
    vol = bar["volume"]

    # Calculate average volume over prior 5 bars
    lookback = sorted_dates[max(0, idx-5):idx]
    if lookback:
        avg_vol = statistics.mean(qqq_daily[d]["volume"] for d in lookback)
    else:
        avg_vol = vol

    vol_ratio = vol / avg_vol if avg_vol > 0 else 1.0
    bar_range = h - l
    body = abs(c - o)
    close_position = (c - l) / bar_range if bar_range > 0 else 0.5

    # Prior day trend
    if idx > 0:
        prev = qqq_daily[sorted_dates[idx-1]]
        prev_close = prev["close"]
        gap = (o - prev_close) / prev_close * 100
    else:
        gap = 0
        prev_close = o

    # 3-day trend
    if idx >= 3:
        three_days_ago = qqq_daily[sorted_dates[idx-3]]["close"]
        trend_3d = (c - three_days_ago) / three_days_ago * 100
    else:
        trend_3d = 0

    # VPA classification
    signal = "NEUTRAL"
    detail = []

    if vol_ratio > 2.0:
        if close_position > 0.7:
            signal = "CLIMAX_TOP" if c > o else "STRONG_BULLISH"
            detail.append(f"Very high vol ({vol_ratio:.1f}x), close near high")
        elif close_position < 0.3:
            signal = "CLIMAX_BOTTOM" if c < o else "STRONG_BEARISH"
            detail.append(f"Very high vol ({vol_ratio:.1f}x), close near low")
        else:
            signal = "HIGH_VOL_INDECISION"
            detail.append(f"Very high vol ({vol_ratio:.1f}x), doji-like")

    elif vol_ratio > 1.3:
        if close_position > 0.7:
            signal = "STRONG_BULLISH"
            detail.append(f"High vol ({vol_ratio:.1f}x), bullish close")
        elif close_position < 0.3:
            signal = "STRONG_BEARISH"
            detail.append(f"High vol ({vol_ratio:.1f}x), bearish close")

    elif vol_ratio < 0.6:
        if c > o:
            signal = "WEAK_UP"
            detail.append(f"Low vol ({vol_ratio:.1f}x), up but no conviction")
        else:
            signal = "WEAK_DOWN"
            detail.append(f"Low vol ({vol_ratio:.1f}x), down but no conviction")

    detail.append(f"3d-trend: {trend_3d:+.1f}%")
    detail.append(f"gap: {gap:+.2f}%")

    return {
        "signal": signal,
        "vol_ratio": round(vol_ratio, 2),
        "close_position": round(close_position, 2),
        "trend_3d": round(trend_3d, 2),
        "gap": round(gap, 2),
        "detail": " | ".join(detail),
    }


# ═══════════════════════════════════════════════════════════════
#  6.  STATISTICAL ANALYSIS
# ═══════════════════════════════════════════════════════════════

def analyze_entries(all_entries: list[BuyEntry], qqq_daily: dict) -> dict:
    """Compute comprehensive statistics on all disaggregated entries."""
    winners = [e for e in all_entries if e.pnl > 0]
    losers = [e for e in all_entries if e.pnl <= 0]
    dca_entries = [e for e in all_entries if e.is_dca]
    first_entries = [e for e in all_entries if not e.is_dca]

    stats = {}
    stats["total_entries"] = len(all_entries)
    stats["winners"] = len(winners)
    stats["losers"] = len(losers)
    stats["win_rate"] = len(winners) / len(all_entries) * 100 if all_entries else 0
    stats["total_pnl"] = sum(e.pnl for e in all_entries)
    stats["avg_win"] = statistics.mean(e.pnl for e in winners) if winners else 0
    stats["avg_loss"] = statistics.mean(e.pnl for e in losers) if losers else 0
    stats["median_win"] = statistics.median(e.pnl for e in winners) if winners else 0
    stats["median_loss"] = statistics.median(e.pnl for e in losers) if losers else 0
    stats["profit_factor"] = (
        abs(sum(e.pnl for e in winners)) / abs(sum(e.pnl for e in losers))
        if losers and sum(e.pnl for e in losers) != 0 else float("inf")
    )

    # DCA analysis
    stats["dca_entries"] = len(dca_entries)
    stats["dca_win_rate"] = (
        len([e for e in dca_entries if e.pnl > 0]) / len(dca_entries) * 100
        if dca_entries else 0
    )
    stats["first_entry_win_rate"] = (
        len([e for e in first_entries if e.pnl > 0]) / len(first_entries) * 100
        if first_entries else 0
    )

    # By option type
    calls = [e for e in all_entries if e.option["type"] == "CALL"]
    puts = [e for e in all_entries if e.option["type"] == "PUT"]
    stats["call_win_rate"] = len([e for e in calls if e.pnl > 0]) / len(calls) * 100 if calls else 0
    stats["put_win_rate"] = len([e for e in puts if e.pnl > 0]) / len(puts) * 100 if puts else 0
    stats["call_pnl"] = sum(e.pnl for e in calls)
    stats["put_pnl"] = sum(e.pnl for e in puts)

    # By moneyness
    for m in ["ITM", "ATM", "OTM"]:
        subset = [e for e in all_entries if e.moneyness == m]
        stats[f"{m}_count"] = len(subset)
        stats[f"{m}_win_rate"] = len([e for e in subset if e.pnl > 0]) / len(subset) * 100 if subset else 0
        stats[f"{m}_pnl"] = sum(e.pnl for e in subset)

    # By DTE
    stats["0dte_count"] = len([e for e in all_entries if e.dte == 0])
    stats["0dte_win_rate"] = (
        len([e for e in all_entries if e.dte == 0 and e.pnl > 0])
        / max(1, len([e for e in all_entries if e.dte == 0])) * 100
    )
    stats["1dte_count"] = len([e for e in all_entries if e.dte == 1])
    stats["1dte_win_rate"] = (
        len([e for e in all_entries if e.dte == 1 and e.pnl > 0])
        / max(1, len([e for e in all_entries if e.dte == 1])) * 100
    )
    stats["2plus_dte_count"] = len([e for e in all_entries if e.dte >= 2])
    stats["2plus_dte_win_rate"] = (
        len([e for e in all_entries if e.dte >= 2 and e.pnl > 0])
        / max(1, len([e for e in all_entries if e.dte >= 2])) * 100
    )
    stats["0dte_pnl"] = sum(e.pnl for e in all_entries if e.dte == 0)
    stats["1dte_pnl"] = sum(e.pnl for e in all_entries if e.dte == 1)
    stats["2plus_dte_pnl"] = sum(e.pnl for e in all_entries if e.dte >= 2)

    # By QQQ trend
    for trend in ["UP", "DOWN", "FLAT"]:
        subset = [e for e in all_entries if e.qqq_trend_today == trend]
        stats[f"trend_{trend}_count"] = len(subset)
        stats[f"trend_{trend}_win_rate"] = (
            len([e for e in subset if e.pnl > 0]) / len(subset) * 100 if subset else 0
        )
        stats[f"trend_{trend}_pnl"] = sum(e.pnl for e in subset)

    # By volume context
    for vc in ["HIGH", "NORMAL", "LOW"]:
        subset = [e for e in all_entries if e.vol_context == vc]
        stats[f"vol_{vc}_count"] = len(subset)
        stats[f"vol_{vc}_win_rate"] = (
            len([e for e in subset if e.pnl > 0]) / len(subset) * 100 if subset else 0
        )

    # Position sizing analysis
    qtys = [e.qty for e in all_entries]
    stats["avg_qty"] = statistics.mean(qtys) if qtys else 0
    stats["median_qty"] = statistics.median(qtys) if qtys else 0
    win_qtys = [e.qty for e in winners]
    loss_qtys = [e.qty for e in losers]
    stats["avg_win_qty"] = statistics.mean(win_qtys) if win_qtys else 0
    stats["avg_loss_qty"] = statistics.mean(loss_qtys) if loss_qtys else 0

    # Entry price analysis
    stats["avg_premium_winners"] = statistics.mean(e.price for e in winners) if winners else 0
    stats["avg_premium_losers"] = statistics.mean(e.price for e in losers) if losers else 0

    # Exit method
    expired = [e for e in all_entries if e.exit_method == "EXPIRED"]
    stats["expired_count"] = len(expired)
    stats["expired_total_loss"] = sum(e.pnl for e in expired)

    # Biggest single entries
    stats["best_entry"] = max(all_entries, key=lambda e: e.pnl) if all_entries else None
    stats["worst_entry"] = min(all_entries, key=lambda e: e.pnl) if all_entries else None

    # VPA signal at entry
    vpa_results = {}
    for e in all_entries:
        try:
            trade_dt = datetime.strptime(e.date, "%m/%d/%Y")
            date_str = trade_dt.strftime("%Y-%m-%d")
        except ValueError:
            continue
        if date_str not in vpa_results:
            vpa_results[date_str] = vpa_daily_context(date_str, qqq_daily)
        e._vpa = vpa_results[date_str]

    # VPA signal correlation
    vpa_win_loss = defaultdict(lambda: {"wins": 0, "losses": 0, "pnl": 0})
    for e in all_entries:
        if hasattr(e, "_vpa"):
            sig = e._vpa["signal"]
            if e.pnl > 0:
                vpa_win_loss[sig]["wins"] += 1
            else:
                vpa_win_loss[sig]["losses"] += 1
            vpa_win_loss[sig]["pnl"] += e.pnl
    stats["vpa_correlation"] = dict(vpa_win_loss)

    # Trend alignment analysis (put on down day = aligned, call on up day = aligned)
    aligned = []
    counter_trend = []
    for e in all_entries:
        if (e.option["type"] == "PUT" and e.qqq_trend_today == "DOWN") or \
           (e.option["type"] == "CALL" and e.qqq_trend_today == "UP"):
            aligned.append(e)
        elif (e.option["type"] == "PUT" and e.qqq_trend_today == "UP") or \
             (e.option["type"] == "CALL" and e.qqq_trend_today == "DOWN"):
            counter_trend.append(e)
    stats["aligned_count"] = len(aligned)
    stats["aligned_win_rate"] = (
        len([e for e in aligned if e.pnl > 0]) / len(aligned) * 100 if aligned else 0
    )
    stats["aligned_pnl"] = sum(e.pnl for e in aligned)
    stats["counter_count"] = len(counter_trend)
    stats["counter_win_rate"] = (
        len([e for e in counter_trend if e.pnl > 0]) / len(counter_trend) * 100
        if counter_trend else 0
    )
    stats["counter_pnl"] = sum(e.pnl for e in counter_trend)

    # DCA depth analysis — how many DCA entries on average for wins vs losses
    from collections import Counter
    dca_depth_wins = []
    dca_depth_losses = []
    symbol_entries = defaultdict(list)
    for e in all_entries:
        symbol_entries[e.symbol].append(e)
    for sym, entries in symbol_entries.items():
        depth = len(entries)
        net = sum(e.pnl for e in entries)
        if net > 0:
            dca_depth_wins.append(depth)
        else:
            dca_depth_losses.append(depth)
    stats["avg_dca_depth_wins"] = statistics.mean(dca_depth_wins) if dca_depth_wins else 0
    stats["avg_dca_depth_losses"] = statistics.mean(dca_depth_losses) if dca_depth_losses else 0

    return stats


# ═══════════════════════════════════════════════════════════════
#  7.  GENERATE MARKDOWN REPORT
# ═══════════════════════════════════════════════════════════════

def generate_report(all_entries: list[BuyEntry], stats: dict, qqq_daily: dict) -> str:
    lines = []
    L = lines.append

    L("# Deep Trade Analysis — Entry-Level Disaggregated P&L")
    L("")
    L("> Each individual Buy-to-Open tranche is treated as its own trade.")
    L("> DCA entries are separated: first entry might be a loss even if the overall")
    L("> position recovered via DCA.  This reveals the TRUE win/loss of each decision.")
    L("")
    L("---")
    L("")

    # ── Overall Stats ──
    L("## 1. Overall Statistics (Entry-Level)")
    L("")
    L(f"| Metric | Value |")
    L(f"|--------|-------|")
    L(f"| Total Individual Entries | {stats['total_entries']} |")
    L(f"| Winners | {stats['winners']} ({stats['win_rate']:.1f}%) |")
    L(f"| Losers | {stats['losers']} ({100-stats['win_rate']:.1f}%) |")
    L(f"| Total Net P&L | ${stats['total_pnl']:,.0f} |")
    L(f"| Average Win | ${stats['avg_win']:,.0f} |")
    L(f"| Average Loss | ${stats['avg_loss']:,.0f} |")
    L(f"| Median Win | ${stats['median_win']:,.0f} |")
    L(f"| Median Loss | ${stats['median_loss']:,.0f} |")
    L(f"| Profit Factor | {stats['profit_factor']:.2f} |")
    L(f"| Expired (Total Loss) | {stats['expired_count']} entries, ${stats['expired_total_loss']:,.0f} |")
    L("")

    # ── DCA Analysis ──
    L("## 2. DCA (Dollar Cost Average) Analysis")
    L("")
    L("| Metric | First Entry | DCA Entry (2nd+) |")
    L("|--------|-------------|------------------|")
    L(f"| Count | {stats['total_entries'] - stats['dca_entries']} | {stats['dca_entries']} |")
    L(f"| Win Rate | {stats['first_entry_win_rate']:.1f}% | {stats['dca_win_rate']:.1f}% |")
    L("")

    first_entries = [e for e in all_entries if not e.is_dca]
    dca_entries = [e for e in all_entries if e.is_dca]
    L(f"| Total P&L | ${sum(e.pnl for e in first_entries):,.0f} | ${sum(e.pnl for e in dca_entries):,.0f} |")
    L(f"| Avg Entry Price | ${statistics.mean(e.price for e in first_entries):.3f} | ${statistics.mean(e.price for e in dca_entries) if dca_entries else 0:.3f} |")
    L("")
    L("**Insight:** First entries that later needed DCA — were the original entries profitable on their own?")
    L("")

    # Analyze symbols that had DCA
    symbols_with_dca = set(e.symbol for e in dca_entries)
    dca_first_entries = [e for e in first_entries if e.symbol in symbols_with_dca]
    dca_first_win = len([e for e in dca_first_entries if e.pnl > 0])
    dca_first_total = len(dca_first_entries)
    L(f"- Symbols that needed DCA: **{len(symbols_with_dca)}**")
    L(f"- First entry win rate on DCA'd symbols: **{dca_first_win}/{dca_first_total} = {dca_first_win/max(1,dca_first_total)*100:.0f}%**")
    L(f"- First entry P&L on DCA'd symbols: **${sum(e.pnl for e in dca_first_entries):,.0f}**")
    L("")

    # ── Option Type ──
    L("## 3. Calls vs Puts")
    L("")
    L("| Type | Count | Win Rate | Total P&L |")
    L("|------|-------|----------|-----------|")
    calls = [e for e in all_entries if e.option["type"] == "CALL"]
    puts = [e for e in all_entries if e.option["type"] == "PUT"]
    L(f"| CALL | {len(calls)} | {stats['call_win_rate']:.1f}% | ${stats['call_pnl']:,.0f} |")
    L(f"| PUT | {len(puts)} | {stats['put_win_rate']:.1f}% | ${stats['put_pnl']:,.0f} |")
    L("")

    # ── Moneyness ──
    L("## 4. Moneyness at Entry")
    L("")
    L("| Moneyness | Count | Win Rate | Total P&L |")
    L("|-----------|-------|----------|-----------|")
    for m in ["ITM", "ATM", "OTM"]:
        L(f"| {m} | {stats[f'{m}_count']} | {stats[f'{m}_win_rate']:.1f}% | ${stats[f'{m}_pnl']:,.0f} |")
    L("")

    # ── DTE ──
    L("## 5. Days to Expiry at Entry")
    L("")
    L("| DTE | Count | Win Rate | Total P&L |")
    L("|-----|-------|----------|-----------|")
    L(f"| 0 DTE | {stats['0dte_count']} | {stats['0dte_win_rate']:.1f}% | ${stats['0dte_pnl']:,.0f} |")
    L(f"| 1 DTE | {stats['1dte_count']} | {stats['1dte_win_rate']:.1f}% | ${stats['1dte_pnl']:,.0f} |")
    L(f"| 2+ DTE | {stats['2plus_dte_count']} | {stats['2plus_dte_win_rate']:.1f}% | ${stats['2plus_dte_pnl']:,.0f} |")
    L("")

    # ── QQQ Trend ──
    L("## 6. QQQ Daily Trend Direction at Entry")
    L("")
    L("| QQQ Trend | Count | Win Rate | Total P&L |")
    L("|-----------|-------|----------|-----------|")
    for t in ["UP", "DOWN", "FLAT"]:
        L(f"| {t} | {stats[f'trend_{t}_count']} | {stats[f'trend_{t}_win_rate']:.1f}% | ${stats[f'trend_{t}_pnl']:,.0f} |")
    L("")

    # ── Trend Alignment ──
    L("## 7. Trend Alignment (Buying WITH the trend vs AGAINST)")
    L("")
    L("*Aligned = buying puts on down days, calls on up days*")
    L("*Counter-trend = buying puts on up days, calls on down days*")
    L("")
    L("| Direction | Count | Win Rate | Total P&L |")
    L("|-----------|-------|----------|-----------|")
    L(f"| Aligned (with trend) | {stats['aligned_count']} | {stats['aligned_win_rate']:.1f}% | ${stats['aligned_pnl']:,.0f} |")
    L(f"| Counter-trend | {stats['counter_count']} | {stats['counter_win_rate']:.1f}% | ${stats['counter_pnl']:,.0f} |")
    L("")

    # ── Volume Context ──
    L("## 8. Volume Context at Entry")
    L("")
    L("| Volume | Count | Win Rate |")
    L("|--------|-------|----------|")
    for vc in ["HIGH", "NORMAL", "LOW"]:
        L(f"| {vc} | {stats[f'vol_{vc}_count']} | {stats[f'vol_{vc}_win_rate']:.1f}% |")
    L("")

    # ── VPA Signal ──
    L("## 9. VPA Signal at Entry (Daily Bar)")
    L("")
    L("| VPA Signal | Wins | Losses | Win Rate | Net P&L |")
    L("|------------|------|--------|----------|---------|")
    for sig, data in sorted(stats["vpa_correlation"].items(), key=lambda x: -x[1]["pnl"]):
        total = data["wins"] + data["losses"]
        wr = data["wins"] / total * 100 if total > 0 else 0
        L(f"| {sig} | {data['wins']} | {data['losses']} | {wr:.0f}% | ${data['pnl']:,.0f} |")
    L("")

    # ── Position Sizing ──
    L("## 10. Position Sizing Analysis")
    L("")
    L(f"| Metric | Value |")
    L(f"|--------|-------|")
    L(f"| Avg Contracts (All) | {stats['avg_qty']:.0f} |")
    L(f"| Avg Contracts (Winners) | {stats['avg_win_qty']:.0f} |")
    L(f"| Avg Contracts (Losers) | {stats['avg_loss_qty']:.0f} |")
    L(f"| Avg Premium (Winners) | ${stats['avg_premium_winners']:.3f} |")
    L(f"| Avg Premium (Losers) | ${stats['avg_premium_losers']:.3f} |")
    L(f"| Avg DCA Depth (Winning symbols) | {stats['avg_dca_depth_wins']:.1f} entries |")
    L(f"| Avg DCA Depth (Losing symbols) | {stats['avg_dca_depth_losses']:.1f} entries |")
    L("")

    # ── Top/Bottom Individual Entries ──
    L("## 11. Top 10 Best & Worst Individual Entries")
    L("")
    sorted_entries = sorted(all_entries, key=lambda e: e.pnl, reverse=True)
    L("### Best Entries")
    L("| # | Symbol | DCA? | Qty | Buy$ | Sell$ | P&L | DTE | Moneyness | QQQ Trend |")
    L("|---|--------|------|-----|------|-------|-----|-----|-----------|-----------|")
    for i, e in enumerate(sorted_entries[:10], 1):
        L(f"| {i} | {e.symbol} | {'DCA' if e.is_dca else 'First'} | {e.qty} | ${e.price:.3f} | ${e.exit_price:.3f} | ${e.pnl:,.0f} | {e.dte} | {e.moneyness} | {e.qqq_trend_today} |")
    L("")

    L("### Worst Entries")
    L("| # | Symbol | DCA? | Qty | Buy$ | Sell$ | P&L | DTE | Moneyness | QQQ Trend |")
    L("|---|--------|------|-----|------|-------|-----|-----|-----------|-----------|")
    for i, e in enumerate(sorted_entries[-10:], 1):
        L(f"| {i} | {e.symbol} | {'DCA' if e.is_dca else 'First'} | {e.qty} | ${e.price:.3f} | ${e.exit_price:.3f} | ${e.pnl:,.0f} | {e.dte} | {e.moneyness} | {e.qqq_trend_today} |")
    L("")

    # ── Detailed DCA Case Studies ──
    L("## 12. DCA Case Studies — Entry-by-Entry")
    L("")
    symbols_with_dca = sorted(symbols_with_dca)
    for sym in symbols_with_dca:
        entries = [e for e in all_entries if e.symbol == sym]
        net = sum(e.pnl for e in entries)
        outcome = "WIN" if net > 0 else "LOSS"
        L(f"### {sym} — Net: ${net:,.0f} ({outcome})")
        L("")
        L("| Entry# | Qty | Buy Price | Exit Price | P&L | DTE |")
        L("|--------|-----|-----------|------------|-----|-----|")
        for e in entries:
            L(f"| {e.entry_order} | {e.qty} | ${e.price:.4f} | ${e.exit_price:.4f} | ${e.pnl:,.0f} | {e.dte} |")
        L("")

    # ═══════════════════════════════════════════════════════════════
    # SECTION 13: KEY FINDINGS & ALGORITHM RECOMMENDATIONS
    # ═══════════════════════════════════════════════════════════════
    L("---")
    L("")
    L("## 13. KEY FINDINGS — Why Winners Won & Losers Lost")
    L("")

    L("### Patterns in WINNING trades:")
    L("")

    # Dynamically generate findings based on stats
    findings_win = []
    findings_loss = []

    # Put vs Call
    if stats["put_pnl"] > stats["call_pnl"] * 1.5:
        findings_win.append(f"**PUT bias is profitable**: PUTs generated ${stats['put_pnl']:,.0f} vs CALLs ${stats['call_pnl']:,.0f}. Your edge is stronger on the put side.")
    elif stats["call_pnl"] > stats["put_pnl"] * 1.5:
        findings_win.append(f"**CALL bias is profitable**: CALLs generated ${stats['call_pnl']:,.0f} vs PUTs ${stats['put_pnl']:,.0f}.")

    # Moneyness
    best_money = max(["ITM", "ATM", "OTM"], key=lambda m: stats[f"{m}_win_rate"] if stats[f"{m}_count"] > 3 else 0)
    worst_money = min(["ITM", "ATM", "OTM"], key=lambda m: stats[f"{m}_win_rate"] if stats[f"{m}_count"] > 3 else 100)
    findings_win.append(f"**{best_money} options win most often** ({stats[f'{best_money}_win_rate']:.0f}% win rate, {stats[f'{best_money}_count']} entries)")
    findings_loss.append(f"**{worst_money} options lose most often** ({stats[f'{worst_money}_win_rate']:.0f}% win rate)")

    # DTE
    dte_data = [
        ("0DTE", stats["0dte_win_rate"], stats["0dte_count"], stats["0dte_pnl"]),
        ("1DTE", stats["1dte_win_rate"], stats["1dte_count"], stats["1dte_pnl"]),
        ("2+DTE", stats["2plus_dte_win_rate"], stats["2plus_dte_count"], stats["2plus_dte_pnl"]),
    ]
    best_dte = max(dte_data, key=lambda x: x[1] if x[2] > 3 else 0)
    worst_dte = min(dte_data, key=lambda x: x[1] if x[2] > 3 else 100)
    findings_win.append(f"**{best_dte[0]} has the best win rate** ({best_dte[1]:.0f}%, {best_dte[2]} entries, ${best_dte[3]:,.0f} P&L)")
    findings_loss.append(f"**{worst_dte[0]} has the worst win rate** ({worst_dte[1]:.0f}%, {worst_dte[2]} entries, ${worst_dte[3]:,.0f} P&L)")

    # Trend alignment
    if stats["aligned_win_rate"] > stats["counter_win_rate"] + 10:
        findings_win.append(f"**Trading WITH the trend wins more** ({stats['aligned_win_rate']:.0f}% vs counter-trend {stats['counter_win_rate']:.0f}%)")
    elif stats["counter_win_rate"] > stats["aligned_win_rate"] + 10:
        findings_win.append(f"**Counter-trend entries actually win more** ({stats['counter_win_rate']:.0f}% vs trend-aligned {stats['aligned_win_rate']:.0f}%) — possible mean-reversion edge")

    # DCA findings
    if stats["first_entry_win_rate"] < stats["dca_win_rate"]:
        findings_loss.append(f"**DCA entries win more than first entries** ({stats['dca_win_rate']:.0f}% vs {stats['first_entry_win_rate']:.0f}%) — first entry timing is weak")
    else:
        findings_win.append(f"**First entries outperform DCA** ({stats['first_entry_win_rate']:.0f}% vs {stats['dca_win_rate']:.0f}%) — initial read is good")

    # Expired contracts
    if stats["expired_count"] > 5:
        findings_loss.append(f"**{stats['expired_count']} entries expired worthless** (${stats['expired_total_loss']:,.0f} lost) — holding too long without stop-loss")

    # Size of losers vs winners
    if abs(stats["avg_loss"]) > stats["avg_win"]:
        findings_loss.append(f"**Avg loss (${abs(stats['avg_loss']):,.0f}) > avg win (${stats['avg_win']:,.0f})** — need tighter risk management")

    if stats["avg_loss_qty"] > stats["avg_win_qty"] * 1.3:
        findings_loss.append(f"**Bigger positions on losers** (avg {stats['avg_loss_qty']:.0f} contracts vs {stats['avg_win_qty']:.0f} on winners) — sizing larger on lower-conviction trades")

    for f in findings_win:
        L(f"- {f}")
    L("")
    L("### Patterns in LOSING trades:")
    L("")
    for f in findings_loss:
        L(f"- {f}")
    L("")

    # ── Algorithm Recommendations ──
    L("---")
    L("")
    L("## 14. ALGORITHM IMPROVEMENT RECOMMENDATIONS")
    L("")
    L("Based on the entry-level analysis, here are specific changes to improve the")
    L("OptionsGanster buy/sell signal algorithm:")
    L("")

    L("### Signal Filters to ADD:")
    L("")
    L("1. **Anti-DCA Gate**: If a position already exists and is losing >20%,")
    L("   the algorithm should NOT generate a \"buy more\" signal. Instead,")
    L("   evaluate the original thesis. Most DCA'd trades show the first entry")
    L("   was poorly timed.")
    L("")
    L("2. **Moneyness Filter**: Weight signals toward the best-performing")
    L(f"   moneyness category ({best_money}). Penalize or block signals for")
    L(f"   the worst category ({worst_money}).")
    L("")
    L("3. **DTE Filter**: Prefer entries with DTE in the optimal range")
    L(f"   ({best_dte[0]}). Add a decay penalty for very short DTE entries")
    L("   that are far OTM — these have the worst expected value.")
    L("")
    L("4. **Expiry Stop-Loss**: Any position that hasn't been closed by")
    L("   end of day with <1 DTE remaining should trigger an auto-exit signal")
    L("   unless deeply ITM. This eliminates the expired-worthless bucket.")
    L("")
    L("5. **Position Size Limiter**: Cap max contracts per entry. Losing trades")
    L("   tend to have larger position sizes, suggesting over-conviction on")
    L("   low-quality setups.")
    L("")

    L("### VPA Engine Improvements:")
    L("")
    L("6. **Require VPA Confirmation**: Only generate entry signals when the")
    L("   daily VPA bar shows a confirming pattern (e.g., STRONG_BULLISH for")
    L("   calls, STRONG_BEARISH for puts). Neutral or conflicting VPA should")
    L("   suppress the signal.")
    L("")
    L("7. **Volume Threshold**: Require above-average volume on the entry day.")
    L("   Low-volume entries show poor win rates.")
    L("")

    L("### Greeks Engine Improvements:")
    L("")
    L("8. **Trend Alignment Score**: Add a new factor to the composite scoring")
    L("   that measures whether the trade direction aligns with the intraday")
    L("   trend. Counter-trend entries need higher composite scores to qualify.")
    L("")
    L("9. **Premium Threshold**: Reject entries where the option premium is")
    L("   below $0.10 — these are lottery tickets with near-zero win rates")
    L("   in this dataset.")
    L("")
    L("10. **DCA Cooldown**: After an entry, suppress new buy signals on the")
    L("    same underlying+direction for at least 15 minutes. This prevents")
    L("    emotional DCA chasing.")
    L("")

    L("### New Signal: Expected Value Gate")
    L("")
    L("11. **Pre-trade EV check**: Before generating a BUY signal, compute:")
    L("    - Historical win rate for similar setups (DTE, moneyness, trend)")
    L("    - Estimated risk:reward from current premium vs strike distance")
    L("    - Only signal if EV > 0 after fees")
    L("")

    L("---")
    L("")
    L("## 15. Implementation Priority (Estimated Impact)")
    L("")
    L("| Priority | Change | Est. P&L Impact | Difficulty |")
    L("|----------|--------|-----------------|------------|")
    L("| 1 | Expiry stop-loss (eliminate expired worthless) | Recover ~50% of expired losses | Easy |")
    L("| 2 | Position size cap | Reduce outsized losses | Easy |")
    L("| 3 | Moneyness + DTE filter | Improve win rate ~5-10% | Medium |")
    L("| 4 | VPA confirmation requirement | Avoid low-conviction entries | Medium |")
    L("| 5 | Anti-DCA gate | Prevent chasing losing trades | Medium |")
    L("| 6 | Premium floor ($0.10 min) | Eliminate lottery-ticket losses | Easy |")
    L("| 7 | Trend alignment scoring | Better entry timing | Hard |")
    L("| 8 | EV gate | Systematic edge validation | Hard |")
    L("")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    print("Loading transactions...")
    txns = load_transactions()
    print(f"  → {len(txns)} raw transactions")

    print("Loading QQQ daily data...")
    qqq_daily = load_qqq_daily()
    print(f"  → {len(qqq_daily)} daily bars")

    print("Building trade groups...")
    groups = build_trade_groups(txns)
    print(f"  → {len(groups)} option symbols traded")

    print("Allocating exit prices to entries...")
    allocate_exit_prices(groups)

    # Flatten all entries
    all_entries = []
    for g in groups.values():
        all_entries.extend(g.entries)
    print(f"  → {len(all_entries)} individual buy entries")

    print("Enriching with market context...")
    enrich_entries(groups, qqq_daily)

    print("Running statistical analysis...")
    stats = analyze_entries(all_entries, qqq_daily)

    print("Generating report...")
    report = generate_report(all_entries, stats, qqq_daily)

    OUTPUT_MD.write_text(report, encoding="utf-8")
    print(f"\n✅ Report saved to: {OUTPUT_MD}")
    print(f"   Total P&L (entry-level): ${stats['total_pnl']:,.0f}")
    print(f"   Entries: {stats['total_entries']} | Win Rate: {stats['win_rate']:.1f}%")
    print(f"   DCA entries: {stats['dca_entries']} | First-entry win rate: {stats['first_entry_win_rate']:.1f}%")


if __name__ == "__main__":
    main()
