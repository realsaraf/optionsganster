"""
Pytest tests for the VPA Engine v2
Uses synthetic OHLCV data – no API calls needed.
"""
import pandas as pd
import numpy as np
import pytest

from app.vpa_engine import VPAEngine, VPASignal, VPAThresholds


# ── helpers ──────────────────────────────────────────────────

def make_df(bars: list[dict]) -> pd.DataFrame:
    """Build a DataFrame from a list of bar dicts."""
    defaults = {"datetime": "2026-01-01 09:30:00"}
    rows = []
    for i, bar in enumerate(bars):
        row = {**defaults, **bar}
        row["datetime"] = f"2026-01-01 09:{30 + i}:00"
        rows.append(row)
    return pd.DataFrame(rows)


def neutral_bar(volume: int = 1000) -> dict:
    """A flat, neutral bar used as filler / history."""
    return {"open": 10.0, "high": 10.1, "low": 9.9, "close": 10.0, "volume": volume}


def make_history(n: int = 12, volume: int = 1000) -> list[dict]:
    """Generate n neutral bars to seed the rolling average."""
    return [neutral_bar(volume) for _ in range(n)]


# ── fixtures ─────────────────────────────────────────────────

@pytest.fixture
def engine() -> VPAEngine:
    return VPAEngine(thresholds=VPAThresholds(volume_avg_period=10))


# ── basic tests ──────────────────────────────────────────────

class TestBasicSignals:
    def test_too_few_bars_returns_empty(self, engine: VPAEngine):
        df = make_df([neutral_bar(), neutral_bar()])
        assert engine.analyze(df) == []

    def test_neutral_bars_produce_neutral_signal(self, engine: VPAEngine):
        df = make_df(make_history(12))
        results = engine.analyze(df)
        assert len(results) > 0
        assert all(r.signal == VPASignal.NEUTRAL for r in results)

    def test_strong_bullish(self, engine: VPAEngine):
        """High volume + up bar + close near high → STRONG_BULLISH."""
        bars = make_history(12, volume=1000)
        bars.append({"open": 10.0, "high": 11.0, "low": 9.9, "close": 10.9, "volume": 3000})
        df = make_df(bars)
        results = engine.analyze(df)
        last = results[-1]
        assert last.signal == VPASignal.STRONG_BULLISH
        assert last.confidence >= 0.7

    def test_strong_bearish(self, engine: VPAEngine):
        """High volume + down bar + close near low → STRONG_BEARISH."""
        bars = make_history(12, volume=1000)
        bars.append({"open": 10.0, "high": 10.1, "low": 9.0, "close": 9.1, "volume": 3000})
        df = make_df(bars)
        results = engine.analyze(df)
        last = results[-1]
        assert last.signal == VPASignal.STRONG_BEARISH

    def test_climax_top(self, engine: VPAEngine):
        """Very high volume + up bar + close near high → CLIMAX_TOP."""
        bars = make_history(12, volume=1000)
        bars.append({"open": 10.0, "high": 11.0, "low": 9.9, "close": 10.9, "volume": 5000})
        df = make_df(bars)
        results = engine.analyze(df)
        last = results[-1]
        assert last.signal == VPASignal.CLIMAX_TOP
        assert last.confidence >= 0.8

    def test_climax_bottom(self, engine: VPAEngine):
        """Very high volume + down bar + close near low → CLIMAX_BOTTOM."""
        bars = make_history(12, volume=1000)
        bars.append({"open": 10.0, "high": 10.1, "low": 9.0, "close": 9.1, "volume": 5000})
        df = make_df(bars)
        results = engine.analyze(df)
        last = results[-1]
        assert last.signal == VPASignal.CLIMAX_BOTTOM

    def test_weak_up(self, engine: VPAEngine):
        """Low volume + up bar → WEAK_UP."""
        bars = make_history(12, volume=1000)
        bars.append({"open": 10.0, "high": 10.5, "low": 9.9, "close": 10.3, "volume": 400})
        df = make_df(bars)
        results = engine.analyze(df)
        last = results[-1]
        assert last.signal == VPASignal.WEAK_UP

    def test_weak_down(self, engine: VPAEngine):
        """Low volume + down bar → WEAK_DOWN."""
        bars = make_history(12, volume=1000)
        bars.append({"open": 10.0, "high": 10.1, "low": 9.5, "close": 9.7, "volume": 400})
        df = make_df(bars)
        results = engine.analyze(df)
        last = results[-1]
        assert last.signal == VPASignal.WEAK_DOWN

    def test_distribution(self, engine: VPAEngine):
        """High volume + up bar but close near low → DISTRIBUTION."""
        bars = make_history(12, volume=1000)
        bars.append({"open": 10.0, "high": 11.0, "low": 9.9, "close": 10.1, "volume": 3000})
        df = make_df(bars)
        results = engine.analyze(df)
        last = results[-1]
        assert last.signal == VPASignal.DISTRIBUTION

    def test_accumulation(self, engine: VPAEngine):
        """High volume + down bar but close near high → ACCUMULATION."""
        bars = make_history(12, volume=1000)
        bars.append({"open": 10.0, "high": 10.1, "low": 9.0, "close": 9.9, "volume": 3000})
        df = make_df(bars)
        results = engine.analyze(df)
        last = results[-1]
        assert last.signal == VPASignal.ACCUMULATION


# ── pin bar tests ────────────────────────────────────────────

class TestPinBars:
    def test_bullish_pin_bar(self, engine: VPAEngine):
        """Long lower wick + close near high → PIN_BAR_BULL."""
        bars = make_history(12, volume=1000)
        # body = 0.3, lower wick = 0.8, upper wick = 0  →  lower ratio ≈ 2.67
        bars.append({"open": 9.9, "high": 10.2, "low": 9.1, "close": 10.2, "volume": 1000})
        df = make_df(bars)
        results = engine.analyze(df)
        last = results[-1]
        assert last.signal == VPASignal.PIN_BAR_BULL

    def test_bearish_pin_bar(self, engine: VPAEngine):
        """Long upper wick + close near low → PIN_BAR_BEAR."""
        bars = make_history(12, volume=1000)
        # body = 0.3, upper wick = 0.8, lower wick = 0 → upper ratio ≈ 2.67
        bars.append({"open": 10.1, "high": 10.9, "low": 9.8, "close": 9.8, "volume": 1000})
        df = make_df(bars)
        results = engine.analyze(df)
        last = results[-1]
        assert last.signal == VPASignal.PIN_BAR_BEAR


# ── multi-bar pattern tests ─────────────────────────────────

class TestMultiBarPatterns:
    def test_no_demand(self, engine: VPAEngine):
        """Strong bullish → weak up  →  upgraded to NO_DEMAND."""
        bars = make_history(12, volume=1000)
        # Strong bullish bar
        bars.append({"open": 10.0, "high": 11.0, "low": 9.9, "close": 10.9, "volume": 3000})
        # Weak up bar
        bars.append({"open": 10.9, "high": 11.2, "low": 10.8, "close": 11.1, "volume": 400})
        df = make_df(bars)
        results = engine.analyze(df)
        last = results[-1]
        assert last.signal == VPASignal.NO_DEMAND

    def test_no_supply(self, engine: VPAEngine):
        """Strong bearish → weak down  →  upgraded to NO_SUPPLY."""
        bars = make_history(12, volume=1000)
        # Strong bearish bar
        bars.append({"open": 10.0, "high": 10.1, "low": 9.0, "close": 9.1, "volume": 3000})
        # Weak down bar
        bars.append({"open": 9.1, "high": 9.2, "low": 8.8, "close": 8.9, "volume": 400})
        df = make_df(bars)
        results = engine.analyze(df)
        last = results[-1]
        assert last.signal == VPASignal.NO_SUPPLY

    def test_confirmed_reversal_up(self, engine: VPAEngine):
        """Climax bottom → strong bullish  →  CONFIRMED_REVERSAL_UP."""
        bars = make_history(12, volume=1000)
        # Climax bottom
        bars.append({"open": 10.0, "high": 10.1, "low": 9.0, "close": 9.1, "volume": 5000})
        # Strong bullish follow-through
        bars.append({"open": 9.1, "high": 10.2, "low": 9.0, "close": 10.1, "volume": 2500})
        df = make_df(bars)
        results = engine.analyze(df)
        last = results[-1]
        assert last.signal == VPASignal.CONFIRMED_REVERSAL_UP
        assert last.confidence >= 0.85

    def test_confirmed_reversal_down(self, engine: VPAEngine):
        """Climax top → strong bearish  →  CONFIRMED_REVERSAL_DOWN."""
        bars = make_history(12, volume=1000)
        # Climax top
        bars.append({"open": 10.0, "high": 11.0, "low": 9.9, "close": 10.9, "volume": 5000})
        # Strong bearish follow-through
        bars.append({"open": 10.9, "high": 11.0, "low": 9.8, "close": 9.9, "volume": 2500})
        df = make_df(bars)
        results = engine.analyze(df)
        last = results[-1]
        assert last.signal == VPASignal.CONFIRMED_REVERSAL_DOWN
        assert last.confidence >= 0.85


# ── bias tests ───────────────────────────────────────────────

class TestBias:
    def test_empty_results(self, engine: VPAEngine):
        bias = engine.get_bias([])
        assert bias["bias"] == "neutral"

    def test_all_neutral_returns_neutral(self, engine: VPAEngine):
        df = make_df(make_history(15))
        results = engine.analyze(df)
        bias = engine.get_bias(results)
        assert bias["bias"] == "neutral"

    def test_bullish_bias(self, engine: VPAEngine):
        """Several bullish signals → bias should be bullish."""
        bars = make_history(12, volume=1000)
        for _ in range(5):
            bars.append({"open": 10.0, "high": 11.0, "low": 9.9, "close": 10.9, "volume": 3000})
        df = make_df(bars)
        results = engine.analyze(df)
        bias = engine.get_bias(results)
        assert bias["bias"] == "bullish"
        assert bias["strength"] > 0

    def test_bearish_bias(self, engine: VPAEngine):
        """Several bearish signals → bias should be bearish."""
        bars = make_history(12, volume=1000)
        for _ in range(5):
            bars.append({"open": 10.0, "high": 10.1, "low": 9.0, "close": 9.1, "volume": 3000})
        df = make_df(bars)
        results = engine.analyze(df)
        bias = engine.get_bias(results)
        assert bias["bias"] == "bearish"
        assert bias["strength"] > 0

    def test_decay_gives_more_weight_to_recent(self, engine: VPAEngine):
        """
        Old bearish signals + recent bullish → bias should still be bullish
        because decay down-weights older bars.
        """
        from app.vpa_engine import VPAResult

        results = []
        # 3 old bearish signals
        for i in range(3):
            results.append(VPAResult(
                signal=VPASignal.STRONG_BEARISH, confidence=0.75,
                description="test", bar_index=i,
                datetime=f"2026-01-01 09:{30 + i}:00",
                price=10.0, volume=2000, volume_ratio=2.0,
            ))
        # 5 recent bullish signals
        for i in range(5):
            results.append(VPAResult(
                signal=VPASignal.STRONG_BULLISH, confidence=0.75,
                description="test", bar_index=3 + i,
                datetime=f"2026-01-01 09:{33 + i}:00",
                price=10.0, volume=2000, volume_ratio=2.0,
            ))

        bias = engine.get_bias(results)
        assert bias["bias"] == "bullish"
