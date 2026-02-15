# Algorithm Improvement Plan — OptionsGanster

## Based on Deep Trade Analysis of 120 Individual Buy Entries (Jan-Feb 2026)

---

## Critical Findings Summary

| Finding | Data | Impact |
|---------|------|--------|
| **Trend alignment is the #1 edge** | WITH trend: 77% WR, +$22,904 / AGAINST trend: 27% WR, -$17,317 | MASSIVE |
| **ITM options dominate** | 93% WR on ITM entries vs 0% on OTM | HIGH |
| **PUTs >>> CALLs** | PUTs: +$12,712 / CALLs: +$482 | HIGH |
| **Expired worthless = $19K lost** | 22 entries expired → -$18,975 | HIGH |
| **Counter-trend calls are toxic** | Buying calls on DOWN days is the #1 loss pattern | HIGH |
| **DCA rarely helps** | DCA WR 47% vs First entry 52% | MEDIUM |
| **High volume days lose** | 23.5% WR on high-vol days | MEDIUM |

---

## Phase 1: Quick Wins (1-2 days each)

### 1.1 Trend Alignment Gate → `greeks_engine.py`

**Location:** Add new factor to `GreeksSignalEngine` or modify `_generate_recommendation()`

**Logic:**
```python
def _score_trend_alignment(self, trade_direction: str, qqq_trend: str) -> float:
    """
    HARD GATE: If direction opposes the daily trend, penalize heavily.
    Data: With-trend = 77% WR (+$22,904), Counter-trend = 27% WR (-$17,317)
    """
    if trade_direction == "bullish" and qqq_trend == "DOWN":
        return -0.8  # STRONG suppress
    if trade_direction == "bearish" and qqq_trend == "UP":
        return -0.8  # STRONG suppress
    if trade_direction == "bullish" and qqq_trend == "UP":
        return +0.5  # BOOST
    if trade_direction == "bearish" and qqq_trend == "DOWN":
        return +0.5  # BOOST
    return 0.0  # FLAT day — no bonus/penalty
```

**Where to add:** New 7th factor in `WEIGHTS` dict (redistribute weights):
```python
WEIGHTS = {
    "iv_rank": 0.20,       # was 0.25
    "gex_regime": 0.18,    # was 0.20
    "greeks_composite": 0.18, # was 0.20
    "uoa_flow": 0.12,      # was 0.15
    "vpa_bias": 0.12,      # was 0.15
    "pc_skew": 0.05,       # same
    "trend_alignment": 0.15, # NEW — 3rd highest weight
}
```

**Expected Impact:** Eliminates ~50% of counter-trend losses → save ~$8,600

---

### 1.2 Expiry Stop-Loss Signal → `greeks_engine.py` or `live_feed.py`

**Logic:** If DTE = 0 and position is not ITM, generate EXIT signal

```python
def check_expiry_exit(self, dte: int, moneyness: str, current_pnl_pct: float) -> bool:
    """
    Auto-exit signal for expiring positions.
    Data: 22 entries expired worthless = -$18,975
    Recovering even 50% = +$9,487
    """
    if dte == 0 and moneyness != "ITM":
        return True  # SIGNAL: EXIT NOW
    if dte == 0 and current_pnl_pct < -50:
        return True  # Deep loss on expiry day
    return False
```

**Expected Impact:** Recover ~$9,500 (50% of expired losses)

---

### 1.3 Premium Floor Filter → `greeks_engine.py`

**Logic:** Suppress BUY signals where premium < $0.15

```python
# In _generate_recommendation() or signal gating:
if option_premium < 0.15:
    return "SUPPRESS — lottery ticket premium, near-zero EV"
```

**Data:** Entries at sub-$0.15 premiums (like QQQ 628C at $0.10) almost always expire worthless.

**Expected Impact:** Prevent ~$2,000 in lottery losses

---

## Phase 2: Medium Effort (3-5 days each)

### 2.1 Moneyness Scoring → `greeks_engine.py`

**Modify `_score_greeks_composite()`** to weight based on moneyness data:

```python
# Current: delta sweet spot is 0.25-0.45
# New: Prefer higher delta (ATM-to-ITM) based on 93% ITM win rate

if 0.40 <= delta_abs <= 0.65:      # ATM-to-slightly-ITM
    score += 0.3                    # was 0.2
    detail = "Delta in optimal range for this strategy"
elif delta_abs > 0.65:              # ITM
    score += 0.2                    # was -0.1 (REVERSED — ITM is GOOD)
    detail = "ITM — high probability"
elif delta_abs < 0.15:              # Far OTM
    score -= 0.4                    # was -0.2 (STRONGER penalty)
    detail = "Far OTM — 0% historical win rate"
```

**Expected Impact:** Shift entries toward ITM/ATM → +5-10% win rate

---

### 2.2 Anti-DCA Gate → new module or `live_feed.py`

**Logic:** Track open positions. If same underlying+direction already has an entry that is losing >20%, suppress new BUY signals.

```python
class PositionTracker:
    def __init__(self):
        self.open_positions = {}  # symbol -> {entry_price, qty, timestamp}
    
    def should_allow_entry(self, symbol: str, direction: str, current_price: float) -> bool:
        key = f"{symbol}_{direction}"
        if key in self.open_positions:
            pos = self.open_positions[key]
            pnl_pct = (current_price - pos["entry_price"]) / pos["entry_price"]
            if pnl_pct < -0.20:
                return False  # BLOCK — existing position already losing >20%
        return True
```

**Data:** DCA win rate (47%) < First entry (52%). DCA'd symbols' first entry WR = 45%.  
The initial thesis was often wrong — throwing more money at it rarely helps.

**Expected Impact:** Prevent $3,000-5,000 in DCA chasing losses

---

### 2.3 VPA Confirmation Requirement → `vpa_engine.py` / `greeks_engine.py`

**Strengthen `_score_vpa_bias()`:**

```python
# Current: VPA neutral → score=0.0 with weight=0.15
# New: VPA neutral or conflicting → NEGATIVE score (actively suppress)

if not vpa_bias or vpa_bias.get("bias") == "neutral":
    return FactorScore(
        name="VPA Bias",
        score=-0.2,          # was 0.0 — now actively penalizes
        confidence=0.50,     # was 0.30
        weight=self.WEIGHTS["vpa_bias"],
        detail="VPA neutral — no price-volume confirmation, CAUTION",
    )
```

Most entries (99/120) happened on "NEUTRAL" VPA days. The engine needs to be more selective.

**Expected Impact:** Filter out low-conviction entries → +3-5% win rate

---

## Phase 3: Structural Changes (1-2 weeks)

### 3.1 Volume Regime Detector → `vpa_engine.py`

**Problem:** 23.5% win rate on HIGH volume days (vs 55% on NORMAL days).  
High volume = choppy, institutional activity = harder to trade.

```python
def get_volume_regime(self, bars: list) -> str:
    """Classify current volume regime."""
    avg_vol = mean(b.volume for b in bars[-10:])
    current = bars[-1].volume
    ratio = current / avg_vol
    
    if ratio > 1.5:
        return "HIGH_RISK"   # Signal: reduce position size or skip
    elif ratio < 0.6:
        return "LOW"         # Signal: caution, no conviction
    return "NORMAL"          # Signal: proceed normally
```

### 3.2 Expected Value Gate → new module `ev_engine.py`

**Pre-trade EV calculation based on historical lookup:**

```python
def compute_ev(self, setup: dict) -> float:
    """
    Estimate expected value based on similar historical setups.
    
    Inputs: moneyness, dte, trend_alignment, vpa_signal, premium
    
    Lookup in historical database of 120 entries:
    - Win rate for this combination
    - Average win/loss for this combination
    - EV = (win_rate × avg_win) + ((1-win_rate) × avg_loss) - fees
    """
    # Simplified version:
    base_wr = 0.49  # overall
    
    # Adjustments from data:
    if setup["trend_aligned"]:
        base_wr += 0.28  # 77% - 49%
    if setup["moneyness"] == "ITM":
        base_wr += 0.15
    elif setup["moneyness"] == "OTM":
        base_wr -= 0.49  # never won
    if setup["premium"] < 0.15:
        base_wr -= 0.30
    
    avg_win = 968    # from data
    avg_loss = -720  # from data
    ev = (base_wr * avg_win) + ((1 - base_wr) * avg_loss)
    
    return ev  # Only signal if EV > 0
```

---

## Implementation Order (by $$ impact)

| # | Change | Est. $ Impact | Effort | Files to Modify |
|---|--------|---------------|--------|-----------------|
| 1 | Trend alignment gate | +$8,600 | 2 hrs | `greeks_engine.py` |
| 2 | Expiry stop-loss | +$9,500 | 2 hrs | `live_feed.py` or `greeks_engine.py` |
| 3 | Premium floor ($0.15) | +$2,000 | 30 min | `greeks_engine.py` |
| 4 | Moneyness (prefer ITM/ATM) | +$3,000 | 2 hrs | `greeks_engine.py` |
| 5 | Anti-DCA gate | +$4,000 | 4 hrs | new `position_tracker.py` |
| 6 | VPA confirmation required | +$2,000 | 2 hrs | `greeks_engine.py` |
| 7 | Volume regime detector | +$1,500 | 3 hrs | `vpa_engine.py` |
| 8 | EV gate | +$3,000 | 1 day | new `ev_engine.py` |

**Total estimated improvement: +$33,600 over the same trading period**
(From $13,194 → ~$46,800, or ~3.5x improvement)

---

## Quick Reference: The "DO" and "DON'T" Rules

### DO ✅
- Buy PUTs on DOWN days (aligned) → 77% WR
- Buy CALLs on UP days (aligned) → 77% WR  
- Prefer ITM/ATM strikes → 93% / 45% WR
- Trade on NORMAL volume days → 55% WR
- Hold 2+ DTE when possible → 56% WR
- Exit before expiry if OTM → avoid $19K in expired losses

### DON'T ❌
- Buy CALLs on DOWN days → only 27% WR, lost $17K
- Buy OTM options → 0% WR in your data
- DCA into a losing position → 47% WR, original thesis was wrong
- Buy sub-$0.15 premium options → almost always expire worthless
- Hold through expiry without stop → 22 entries expired = -$19K
- Trade with large size on HIGH volume days → 23.5% WR
