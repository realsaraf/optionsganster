# Deep Trade Analysis — Entry-Level Disaggregated P&L

> Each individual Buy-to-Open tranche is treated as its own trade.
> DCA entries are separated: first entry might be a loss even if the overall
> position recovered via DCA.  This reveals the TRUE win/loss of each decision.

---

## 1. Overall Statistics (Entry-Level)

| Metric | Value |
|--------|-------|
| Total Individual Entries | 120 |
| Winners | 59 (49.2%) |
| Losers | 61 (50.8%) |
| Total Net P&L | $13,194 |
| Average Win | $968 |
| Average Loss | $-720 |
| Median Win | $720 |
| Median Loss | $-360 |
| Profit Factor | 1.30 |
| Expired (Total Loss) | 22 entries, $-18,975 |

## 2. DCA (Dollar Cost Average) Analysis

| Metric | First Entry | DCA Entry (2nd+) |
|--------|-------------|------------------|
| Count | 54 | 66 |
| Win Rate | 51.9% | 47.0% |

| Total P&L | $1,220 | $11,974 |
| Avg Entry Price | $0.978 | $0.575 |

**Insight:** First entries that later needed DCA — were the original entries profitable on their own?

- Symbols that needed DCA: **29**
- First entry win rate on DCA'd symbols: **13/29 = 45%**
- First entry P&L on DCA'd symbols: **$2,290**

## 3. Calls vs Puts

| Type | Count | Win Rate | Total P&L |
|------|-------|----------|-----------|
| CALL | 58 | 60.3% | $482 |
| PUT | 62 | 38.7% | $12,712 |

## 4. Moneyness at Entry

| Moneyness | Count | Win Rate | Total P&L |
|-----------|-------|----------|-----------|
| ITM | 14 | 92.9% | $10,869 |
| ATM | 102 | 45.1% | $4,831 |
| OTM | 4 | 0.0% | $-2,506 |

## 5. Days to Expiry at Entry

| DTE | Count | Win Rate | Total P&L |
|-----|-------|----------|-----------|
| 0 DTE | 108 | 49.1% | $8,119 |
| 1 DTE | 3 | 33.3% | $1,681 |
| 2+ DTE | 9 | 55.6% | $3,394 |

## 6. QQQ Daily Trend Direction at Entry

| QQQ Trend | Count | Win Rate | Total P&L |
|-----------|-------|----------|-----------|
| UP | 40 | 55.0% | $6,239 |
| DOWN | 54 | 46.3% | $-651 |
| FLAT | 26 | 46.2% | $7,607 |

## 7. Trend Alignment (Buying WITH the trend vs AGAINST)

*Aligned = buying puts on down days, calls on up days*
*Counter-trend = buying puts on up days, calls on down days*

| Direction | Count | Win Rate | Total P&L |
|-----------|-------|----------|-----------|
| Aligned (with trend) | 43 | 76.7% | $22,904 |
| Counter-trend | 51 | 27.5% | $-17,317 |

## 8. Volume Context at Entry

| Volume | Count | Win Rate |
|--------|-------|----------|
| HIGH | 17 | 23.5% |
| NORMAL | 89 | 55.1% |
| LOW | 14 | 42.9% |

## 9. VPA Signal at Entry (Daily Bar)

| VPA Signal | Wins | Losses | Win Rate | Net P&L |
|------------|------|--------|----------|---------|
| NEUTRAL | 52 | 47 | 53% | $11,940 |
| STRONG_BEARISH | 7 | 12 | 37% | $1,614 |
| WEAK_UP | 0 | 2 | 0% | $-360 |

## 10. Position Sizing Analysis

| Metric | Value |
|--------|-------|
| Avg Contracts (All) | 19 |
| Avg Contracts (Winners) | 18 |
| Avg Contracts (Losers) | 21 |
| Avg Premium (Winners) | $0.963 |
| Avg Premium (Losers) | $0.557 |
| Avg DCA Depth (Winning symbols) | 2.1 entries |
| Avg DCA Depth (Losing symbols) | 2.4 entries |

## 11. Top 10 Best & Worst Individual Entries

### Best Entries
| # | Symbol | DCA? | Qty | Buy$ | Sell$ | P&L | DTE | Moneyness | QQQ Trend |
|---|--------|------|-----|------|-------|-----|-----|-----------|-----------|
| 1 | QQQ 01/16/2026 618.00 P | DCA | 19 | $2.150 | $4.403 | $4,256 | 2 | ATM | DOWN |
| 2 | QQQ 01/05/2026 615.00 P | First | 20 | $1.528 | $3.581 | $4,079 | 3 | ATM | DOWN |
| 3 | QQQ 01/07/2026 626.00 P | First | 85 | $0.450 | $0.879 | $3,531 | 0 | ATM | FLAT |
| 4 | QQQ 01/14/2026 629.00 C | First | 40 | $1.900 | $2.663 | $3,000 | 1 | ATM | FLAT |
| 5 | QQQ 01/05/2026 620.00 P | DCA | 25 | $0.650 | $1.848 | $2,963 | 0 | ATM | FLAT |
| 6 | QQQ 01/05/2026 620.00 P | First | 22 | $0.790 | $1.848 | $2,299 | 0 | ATM | FLAT |
| 7 | QQQ 01/15/2026 627.00 C | DCA | 50 | $0.600 | $1.043 | $2,151 | 0 | ATM | DOWN |
| 8 | QQQ 01/09/2026 622.00 C | DCA | 20 | $0.720 | $1.622 | $1,778 | 0 | ITM | UP |
| 9 | QQQ 01/07/2026 626.00 P | DCA | 35 | $0.360 | $0.879 | $1,769 | 0 | ATM | FLAT |
| 10 | QQQ 01/30/2026 626.00 P | DCA | 15 | $1.610 | $2.781 | $1,737 | 0 | ITM | DOWN |

### Worst Entries
| # | Symbol | DCA? | Qty | Buy$ | Sell$ | P&L | DTE | Moneyness | QQQ Trend |
|---|--------|------|-----|------|-------|-----|-----|-----------|-----------|
| 1 | QQQ 02/03/2026 622.00 C | First | 10 | $1.140 | $0.000 | $-1,147 | 0 | ATM | DOWN |
| 2 | QQQ 01/16/2026 628.00 C | DCA | 100 | $0.140 | $0.033 | $-1,199 | 0 | ATM | DOWN |
| 3 | QQQ 02/06/2026 600.00 P | First | 15 | $1.610 | $0.780 | $-1,265 | 4 | OTM | UP |
| 4 | QQQ 01/20/2026 608.00 P | DCA | 30 | $0.520 | $0.000 | $-1,580 | 0 | ATM | DOWN |
| 5 | QQQ 01/16/2026 628.00 C | DCA | 100 | $0.220 | $0.033 | $-1,999 | 0 | ATM | DOWN |
| 6 | QQQ 01/20/2026 608.00 P | First | 20 | $1.020 | $0.000 | $-2,053 | 0 | ATM | DOWN |
| 7 | QQQ 02/02/2026 624.00 P | First | 22 | $1.080 | $0.000 | $-2,391 | 0 | ATM | UP |
| 8 | QQQ 01/22/2026 618.00 P | First | 20 | $1.475 | $0.000 | $-2,963 | 0 | ATM | FLAT |
| 9 | QQQ 01/20/2026 626.00 C | First | 100 | $1.310 | $0.893 | $-4,299 | 4 | ATM | DOWN |
| 10 | QQQ 01/16/2026 623.00 C | First | 100 | $0.630 | $0.000 | $-6,366 | 0 | ATM | DOWN |

## 12. DCA Case Studies — Entry-by-Entry

### QQQ 01/02/2026 620.00 P — Net: $1,612 (WIN)

| Entry# | Qty | Buy Price | Exit Price | P&L | DTE |
|--------|-----|-----------|------------|-----|-----|
| 1 | 10 | $1.2200 | $2.5634 | $1,330 | 0 |
| 2 | 2 | $1.1400 | $2.5634 | $282 | 0 |

### QQQ 01/05/2026 617.00 P — Net: $-3,834 (LOSS)

| Entry# | Qty | Buy Price | Exit Price | P&L | DTE |
|--------|-----|-----------|------------|-----|-----|
| 1 | 10 | $1.0900 | $0.1659 | $-937 | 0 |
| 2 | 20 | $0.6600 | $0.1659 | $-1,015 | 0 |
| 3 | 20 | $0.5500 | $0.1659 | $-795 | 0 |
| 4 | 50 | $0.3700 | $0.1659 | $-1,087 | 0 |

### QQQ 01/05/2026 620.00 P — Net: $5,263 (WIN)

| Entry# | Qty | Buy Price | Exit Price | P&L | DTE |
|--------|-----|-----------|------------|-----|-----|
| 1 | 22 | $0.7900 | $1.8485 | $2,299 | 0 |
| 2 | 25 | $0.6500 | $1.8485 | $2,963 | 0 |

### QQQ 01/06/2026 622.00 C — Net: $3,144 (WIN)

| Entry# | Qty | Buy Price | Exit Price | P&L | DTE |
|--------|-----|-----------|------------|-----|-----|
| 1 | 10 | $0.7200 | $0.7706 | $37 | 0 |
| 2 | 10 | $0.8900 | $0.7706 | $-133 | 0 |
| 3 | 10 | $0.7300 | $0.7706 | $27 | 0 |
| 4 | 20 | $0.5200 | $0.7706 | $475 | 0 |
| 5 | 20 | $0.3700 | $0.7706 | $775 | 0 |
| 6 | 30 | $0.2700 | $0.7706 | $1,462 | 0 |
| 7 | 7 | $0.1300 | $0.7706 | $439 | 0 |
| 8 | 1 | $0.1400 | $0.7706 | $62 | 0 |

### QQQ 01/07/2026 623.00 P — Net: $-3,223 (LOSS)

| Entry# | Qty | Buy Price | Exit Price | P&L | DTE |
|--------|-----|-----------|------------|-----|-----|
| 1 | 10 | $0.7700 | $0.0859 | $-697 | 0 |
| 2 | 10 | $0.7500 | $0.0859 | $-677 | 0 |
| 3 | 10 | $0.4700 | $0.0859 | $-397 | 0 |
| 4 | 10 | $0.3200 | $0.0859 | $-247 | 0 |
| 5 | 30 | $0.2100 | $0.0859 | $-412 | 0 |
| 6 | 30 | $0.2200 | $0.0859 | $-442 | 0 |
| 7 | 40 | $0.1600 | $0.0859 | $-350 | 0 |

### QQQ 01/07/2026 626.00 P — Net: $5,299 (WIN)

| Entry# | Qty | Buy Price | Exit Price | P&L | DTE |
|--------|-----|-----------|------------|-----|-----|
| 1 | 85 | $0.4500 | $0.8786 | $3,531 | 0 |
| 2 | 35 | $0.3600 | $0.8786 | $1,769 | 0 |

### QQQ 01/08/2026 618.00 P — Net: $-38 (LOSS)

| Entry# | Qty | Buy Price | Exit Price | P&L | DTE |
|--------|-----|-----------|------------|-----|-----|
| 1 | 3 | $0.1100 | $0.0000 | $-35 | 0 |
| 2 | 1 | $0.0200 | $0.0000 | $-3 | 0 |

### QQQ 01/08/2026 621.00 C — Net: $1,765 (WIN)

| Entry# | Qty | Buy Price | Exit Price | P&L | DTE |
|--------|-----|-----------|------------|-----|-----|
| 1 | 20 | $0.8000 | $1.0349 | $443 | 0 |
| 2 | 10 | $0.7700 | $1.0349 | $252 | 0 |
| 3 | 20 | $0.5400 | $1.0349 | $963 | 0 |
| 4 | 2 | $0.4900 | $1.0349 | $106 | 0 |

### QQQ 01/09/2026 622.00 C — Net: $4,041 (WIN)

| Entry# | Qty | Buy Price | Exit Price | P&L | DTE |
|--------|-----|-----------|------------|-----|-----|
| 1 | 10 | $1.3400 | $1.6221 | $269 | 0 |
| 2 | 10 | $1.7700 | $1.6221 | $-161 | 0 |
| 3 | 10 | $1.2700 | $1.6221 | $339 | 0 |
| 4 | 10 | $1.0500 | $1.6221 | $559 | 0 |
| 5 | 20 | $0.9800 | $1.6221 | $1,258 | 0 |
| 6 | 20 | $0.7200 | $1.6221 | $1,778 | 0 |

### QQQ 01/12/2026 625.00 P — Net: $1,121 (WIN)

| Entry# | Qty | Buy Price | Exit Price | P&L | DTE |
|--------|-----|-----------|------------|-----|-----|
| 1 | 10 | $0.9400 | $0.9034 | $-50 | 0 |
| 2 | 90 | $0.7600 | $0.9034 | $1,171 | 0 |

### QQQ 01/12/2026 627.00 C — Net: $1,033 (WIN)

| Entry# | Qty | Buy Price | Exit Price | P&L | DTE |
|--------|-----|-----------|------------|-----|-----|
| 1 | 20 | $0.6290 | $0.6451 | $6 | 0 |
| 2 | 20 | $0.6300 | $0.6451 | $4 | 0 |
| 3 | 20 | $0.4600 | $0.6451 | $344 | 0 |
| 4 | 25 | $0.3600 | $0.6451 | $680 | 0 |

### QQQ 01/12/2026 627.00 P — Net: $-100 (LOSS)

| Entry# | Qty | Buy Price | Exit Price | P&L | DTE |
|--------|-----|-----------|------------|-----|-----|
| 1 | 1 | $0.2200 | $0.0000 | $-23 | 0 |
| 2 | 1 | $0.2000 | $0.0000 | $-21 | 0 |
| 3 | 2 | $0.2100 | $0.0000 | $-43 | 0 |
| 4 | 1 | $0.1000 | $0.0000 | $-11 | 0 |
| 5 | 1 | $0.0200 | $0.0000 | $-3 | 0 |

### QQQ 01/14/2026 622.00 C — Net: $487 (WIN)

| Entry# | Qty | Buy Price | Exit Price | P&L | DTE |
|--------|-----|-----------|------------|-----|-----|
| 1 | 20 | $0.8300 | $0.7867 | $-113 | 0 |
| 2 | 9 | $0.1800 | $0.7867 | $534 | 0 |
| 3 | 1 | $0.1100 | $0.7867 | $66 | 0 |

### QQQ 01/15/2026 625.00 P — Net: $2,571 (WIN)

| Entry# | Qty | Buy Price | Exit Price | P&L | DTE |
|--------|-----|-----------|------------|-----|-----|
| 1 | 10 | $1.0000 | $1.5034 | $490 | 0 |
| 2 | 20 | $0.8900 | $1.5034 | $1,200 | 0 |
| 3 | 20 | $1.0500 | $1.5034 | $880 | 0 |

### QQQ 01/15/2026 627.00 C — Net: $3,531 (WIN)

| Entry# | Qty | Buy Price | Exit Price | P&L | DTE |
|--------|-----|-----------|------------|-----|-----|
| 1 | 30 | $0.7900 | $1.0434 | $720 | 0 |
| 2 | 20 | $0.7000 | $1.0434 | $660 | 0 |
| 3 | 50 | $0.6000 | $1.0434 | $2,151 | 0 |

### QQQ 01/15/2026 629.00 C — Net: $-234 (LOSS)

| Entry# | Qty | Buy Price | Exit Price | P&L | DTE |
|--------|-----|-----------|------------|-----|-----|
| 1 | 20 | $0.1200 | $0.0234 | $-220 | 0 |
| 2 | 2 | $0.0800 | $0.0234 | $-14 | 0 |

### QQQ 01/16/2026 618.00 P — Net: $4,446 (WIN)

| Entry# | Qty | Buy Price | Exit Price | P&L | DTE |
|--------|-----|-----------|------------|-----|-----|
| 1 | 1 | $2.4900 | $4.4034 | $190 | 2 |
| 2 | 19 | $2.1500 | $4.4034 | $4,256 | 2 |

### QQQ 01/16/2026 628.00 C — Net: $-8,705 (LOSS)

| Entry# | Qty | Buy Price | Exit Price | P&L | DTE |
|--------|-----|-----------|------------|-----|-----|
| 1 | 20 | $0.5100 | $0.0334 | $-980 | 0 |
| 2 | 20 | $0.5100 | $0.0334 | $-980 | 0 |
| 3 | 30 | $0.3900 | $0.0334 | $-1,110 | 0 |
| 4 | 30 | $0.3000 | $0.0334 | $-840 | 0 |
| 5 | 100 | $0.2200 | $0.0334 | $-1,999 | 0 |
| 6 | 100 | $0.1400 | $0.0334 | $-1,199 | 0 |
| 7 | 100 | $0.1000 | $0.0334 | $-799 | 0 |
| 8 | 100 | $0.1000 | $0.0334 | $-799 | 0 |

### QQQ 01/20/2026 601.00 P — Net: $-88 (LOSS)

| Entry# | Qty | Buy Price | Exit Price | P&L | DTE |
|--------|-----|-----------|------------|-----|-----|
| 1 | 9 | $0.1300 | $0.0534 | $-81 | 0 |
| 2 | 1 | $0.1100 | $0.0534 | $-7 | 0 |

### QQQ 01/20/2026 608.00 P — Net: $-3,633 (LOSS)

| Entry# | Qty | Buy Price | Exit Price | P&L | DTE |
|--------|-----|-----------|------------|-----|-----|
| 1 | 20 | $1.0200 | $0.0000 | $-2,053 | 0 |
| 2 | 30 | $0.5200 | $0.0000 | $-1,580 | 0 |

### QQQ 01/21/2026 600.00 P — Net: $-1,320 (LOSS)

| Entry# | Qty | Buy Price | Exit Price | P&L | DTE |
|--------|-----|-----------|------------|-----|-----|
| 1 | 10 | $0.9400 | $0.5334 | $-420 | 1 |
| 2 | 20 | $0.9700 | $0.5334 | $-900 | 1 |

### QQQ 01/22/2026 620.00 P — Net: $1,344 (WIN)

| Entry# | Qty | Buy Price | Exit Price | P&L | DTE |
|--------|-----|-----------|------------|-----|-----|
| 1 | 10 | $1.3600 | $2.4684 | $1,095 | 0 |
| 2 | 2 | $1.2100 | $2.4684 | $249 | 0 |

### QQQ 01/23/2026 622.00 P — Net: $-300 (LOSS)

| Entry# | Qty | Buy Price | Exit Price | P&L | DTE |
|--------|-----|-----------|------------|-----|-----|
| 1 | 10 | $0.2600 | $0.0000 | $-267 | 0 |
| 2 | 2 | $0.1600 | $0.0000 | $-33 | 0 |

### QQQ 01/30/2026 626.00 P — Net: $2,525 (WIN)

| Entry# | Qty | Buy Price | Exit Price | P&L | DTE |
|--------|-----|-----------|------------|-----|-----|
| 1 | 10 | $1.9800 | $2.7814 | $788 | 0 |
| 2 | 15 | $1.6100 | $2.7814 | $1,737 | 0 |

### QQQ 02/03/2026 622.00 C — Net: $-1,445 (LOSS)

| Entry# | Qty | Buy Price | Exit Price | P&L | DTE |
|--------|-----|-----------|------------|-----|-----|
| 1 | 10 | $1.1400 | $0.0000 | $-1,147 | 0 |
| 2 | 6 | $0.4900 | $0.0000 | $-298 | 0 |

### QQQ 02/03/2026 625.00 C — Net: $-1,057 (LOSS)

| Entry# | Qty | Buy Price | Exit Price | P&L | DTE |
|--------|-----|-----------|------------|-----|-----|
| 1 | 30 | $0.3300 | $0.0000 | $-1,010 | 0 |
| 2 | 2 | $0.1600 | $0.0000 | $-33 | 0 |
| 3 | 1 | $0.0800 | $0.0000 | $-9 | 0 |
| 4 | 2 | $0.0200 | $0.0000 | $-5 | 0 |

### QQQ 02/06/2026 600.00 P — Net: $-1,953 (LOSS)

| Entry# | Qty | Buy Price | Exit Price | P&L | DTE |
|--------|-----|-----------|------------|-----|-----|
| 1 | 15 | $1.6100 | $0.7798 | $-1,265 | 4 |
| 2 | 15 | $1.2200 | $0.7798 | $-680 | 4 |
| 3 | 1 | $0.8400 | $0.7798 | $-7 | 4 |

### QQQ 02/13/2026 595.00 P — Net: $-440 (LOSS)

| Entry# | Qty | Buy Price | Exit Price | P&L | DTE |
|--------|-----|-----------|------------|-----|-----|
| 1 | 10 | $1.2700 | $0.9234 | $-360 | 0 |
| 2 | 5 | $1.0700 | $0.9234 | $-80 | 0 |

### QQQ 02/13/2026 604.00 C — Net: $2,450 (WIN)

| Entry# | Qty | Buy Price | Exit Price | P&L | DTE |
|--------|-----|-----------|------------|-----|-----|
| 1 | 10 | $1.4700 | $2.2317 | $748 | 0 |
| 2 | 10 | $1.4400 | $2.2317 | $778 | 0 |
| 3 | 10 | $1.2950 | $2.2317 | $923 | 0 |

---

## 13. KEY FINDINGS — Why Winners Won & Losers Lost

### Patterns in WINNING trades:

- **PUT bias is profitable**: PUTs generated $12,712 vs CALLs $482. Your edge is stronger on the put side.
- **ITM options win most often** (93% win rate, 14 entries)
- **2+DTE has the best win rate** (56%, 9 entries, $3,394 P&L)
- **Trading WITH the trend wins more** (77% vs counter-trend 27%)
- **First entries outperform DCA** (52% vs 47%) — initial read is good

### Patterns in LOSING trades:

- **OTM options lose most often** (0% win rate)
- **0DTE has the worst win rate** (49%, 108 entries, $8,119 P&L)
- **22 entries expired worthless** ($-18,975 lost) — holding too long without stop-loss

---

## 14. ALGORITHM IMPROVEMENT RECOMMENDATIONS

Based on the entry-level analysis, here are specific changes to improve the
OptionsGanster buy/sell signal algorithm:

### Signal Filters to ADD:

1. **Anti-DCA Gate**: If a position already exists and is losing >20%,
   the algorithm should NOT generate a "buy more" signal. Instead,
   evaluate the original thesis. Most DCA'd trades show the first entry
   was poorly timed.

2. **Moneyness Filter**: Weight signals toward the best-performing
   moneyness category (ITM). Penalize or block signals for
   the worst category (OTM).

3. **DTE Filter**: Prefer entries with DTE in the optimal range
   (2+DTE). Add a decay penalty for very short DTE entries
   that are far OTM — these have the worst expected value.

4. **Expiry Stop-Loss**: Any position that hasn't been closed by
   end of day with <1 DTE remaining should trigger an auto-exit signal
   unless deeply ITM. This eliminates the expired-worthless bucket.

5. **Position Size Limiter**: Cap max contracts per entry. Losing trades
   tend to have larger position sizes, suggesting over-conviction on
   low-quality setups.

### VPA Engine Improvements:

6. **Require VPA Confirmation**: Only generate entry signals when the
   daily VPA bar shows a confirming pattern (e.g., STRONG_BULLISH for
   calls, STRONG_BEARISH for puts). Neutral or conflicting VPA should
   suppress the signal.

7. **Volume Threshold**: Require above-average volume on the entry day.
   Low-volume entries show poor win rates.

### Greeks Engine Improvements:

8. **Trend Alignment Score**: Add a new factor to the composite scoring
   that measures whether the trade direction aligns with the intraday
   trend. Counter-trend entries need higher composite scores to qualify.

9. **Premium Threshold**: Reject entries where the option premium is
   below $0.10 — these are lottery tickets with near-zero win rates
   in this dataset.

10. **DCA Cooldown**: After an entry, suppress new buy signals on the
    same underlying+direction for at least 15 minutes. This prevents
    emotional DCA chasing.

### New Signal: Expected Value Gate

11. **Pre-trade EV check**: Before generating a BUY signal, compute:
    - Historical win rate for similar setups (DTE, moneyness, trend)
    - Estimated risk:reward from current premium vs strike distance
    - Only signal if EV > 0 after fees

---

## 15. Implementation Priority (Estimated Impact)

| Priority | Change | Est. P&L Impact | Difficulty |
|----------|--------|-----------------|------------|
| 1 | Expiry stop-loss (eliminate expired worthless) | Recover ~50% of expired losses | Easy |
| 2 | Position size cap | Reduce outsized losses | Easy |
| 3 | Moneyness + DTE filter | Improve win rate ~5-10% | Medium |
| 4 | VPA confirmation requirement | Avoid low-conviction entries | Medium |
| 5 | Anti-DCA gate | Prevent chasing losing trades | Medium |
| 6 | Premium floor ($0.10 min) | Eliminate lottery-ticket losses | Easy |
| 7 | Trend alignment scoring | Better entry timing | Hard |
| 8 | EV gate | Systematic edge validation | Hard |
