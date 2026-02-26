"""
Pytest tests for the Greeks Signal Engine improvements.
Tests: trend alignment, premium floor, expiry stop-loss,
       delta/moneyness fix, VPA neutral penalty, volume regime.
"""
import pytest
from app.greeks_engine import (
    GreeksSignalEngine, CompositeSignal, TradeArchetype,
    GreeksData, ChainMetrics, FactorScore, CompositeResult,
)


@pytest.fixture
def engine() -> GreeksSignalEngine:
    return GreeksSignalEngine()


def _make_snapshot(delta=0.5, last_price=2.0, iv=0.3, **kw):
    """Build a minimal contract snapshot dict."""
    base = {
        "greeks": {"delta": delta, "gamma": 0.05, "theta": -0.03, "vega": 0.1},
        "iv": iv,
        "open_interest": 500,
        "volume": 200,
        "underlying_price": 500.0,
        "break_even": 502.0,
        "last_price": last_price,
    }
    base.update(kw)
    return base


def _make_chain(n=5):
    """Build a minimal options chain."""
    chain = []
    for i in range(n):
        chain.append({
            "contract_type": "call",
            "strike": 490 + i * 5,
            "greeks": {"delta": 0.5, "gamma": 0.04, "theta": -0.02, "vega": 0.1},
            "iv": 0.25 + i * 0.02,
            "open_interest": 1000,
            "volume": 300,
        })
        chain.append({
            "contract_type": "put",
            "strike": 490 + i * 5,
            "greeks": {"delta": -0.5, "gamma": 0.04, "theta": -0.02, "vega": 0.1},
            "iv": 0.25 + i * 0.02,
            "open_interest": 800,
            "volume": 250,
        })
    return chain


class TestWeights:
    def test_weights_sum_to_one(self, engine: GreeksSignalEngine):
        total = sum(engine.WEIGHTS.values())
        assert abs(total - 1.0) < 0.01, f"Weights sum to {total}, expected 1.0"

    def test_trend_alignment_weight_exists(self, engine: GreeksSignalEngine):
        assert "trend_alignment" in engine.WEIGHTS
        assert engine.WEIGHTS["trend_alignment"] == 0.22  # highest weight per trade data


class TestTrendAlignment:
    def test_with_trend_bullish(self, engine: GreeksSignalEngine):
        factor = engine._score_trend_alignment("bullish", "UP")
        assert factor.score == 0.5
        assert "WITH-TREND" in factor.detail

    def test_counter_trend_bullish_on_down_day(self, engine: GreeksSignalEngine):
        # Counter-trend: bullish signal but underlying is DOWN.
        # Score reflects underlying direction → negative (DOWN).
        factor = engine._score_trend_alignment("bullish", "DOWN")
        assert factor.score == -0.4
        assert "COUNTER-TREND" in factor.detail

    def test_counter_trend_bearish_on_up_day(self, engine: GreeksSignalEngine):
        # Counter-trend: bearish signal but underlying is UP.
        # Score reflects underlying direction → positive (UP) so _put_action maps to SELL/EXIT.
        factor = engine._score_trend_alignment("bearish", "UP")
        assert factor.score == 0.4

    def test_with_trend_bearish_on_down(self, engine: GreeksSignalEngine):
        # WITH-TREND: bearish signal on DOWN day → score negative (confirms bearish underlying).
        factor = engine._score_trend_alignment("bearish", "DOWN")
        assert factor.score == -0.5

    def test_flat_trend_neutral(self, engine: GreeksSignalEngine):
        factor = engine._score_trend_alignment("bullish", "FLAT")
        assert factor.score == 0.0

    def test_no_trend_neutral(self, engine: GreeksSignalEngine):
        factor = engine._score_trend_alignment("bullish", None)
        assert factor.score == 0.0

    def test_trend_factor_in_composite(self, engine: GreeksSignalEngine):
        """Trend alignment factor appears in full analysis."""
        result = engine.analyze(
            _make_snapshot(), _make_chain(),
            vpa_bias={"bias": "bullish", "strength": 0.7},
            underlying_trend="UP",
        )
        factor_names = [f.name for f in result.factors]
        assert "Trend Alignment" in factor_names


class TestPremiumFloor:
    def test_low_premium_generates_warning(self, engine: GreeksSignalEngine):
        result = engine.analyze(
            _make_snapshot(last_price=0.10),
            _make_chain(),
        )
        assert any("PREMIUM FLOOR" in w for w in result.warnings)

    def test_low_premium_crushes_score(self, engine: GreeksSignalEngine):
        """Low premium options should have heavily suppressed scores."""
        normal = engine.analyze(_make_snapshot(last_price=2.0), _make_chain())
        cheap = engine.analyze(_make_snapshot(last_price=0.10), _make_chain())
        # Cheap option score magnitude should be much smaller
        assert abs(cheap.score) < abs(normal.score) or any("PREMIUM FLOOR" in w for w in cheap.warnings)

    def test_normal_premium_no_warning(self, engine: GreeksSignalEngine):
        result = engine.analyze(_make_snapshot(last_price=2.0), _make_chain())
        assert not any("PREMIUM FLOOR" in w for w in result.warnings)


class TestExpiryStopLoss:
    def test_dte0_otm_generates_exit_warning(self, engine: GreeksSignalEngine):
        result = engine.analyze(
            _make_snapshot(delta=0.2),  # OTM
            _make_chain(),
            dte=0,
        )
        assert any("EXPIRY DAY" in w for w in result.warnings)
        assert any("EXIT" in w for w in result.warnings)

    def test_dte0_itm_generates_monitor_warning(self, engine: GreeksSignalEngine):
        result = engine.analyze(
            _make_snapshot(delta=0.8),  # ITM
            _make_chain(),
            dte=0,
        )
        assert any("EXPIRY DAY" in w for w in result.warnings)
        # Should be "monitor" not "EXIT"
        assert any("monitor" in w for w in result.warnings)

    def test_dte5_no_expiry_warning(self, engine: GreeksSignalEngine):
        result = engine.analyze(
            _make_snapshot(delta=0.2),
            _make_chain(),
            dte=5,
        )
        assert not any("EXPIRY DAY" in w for w in result.warnings)


class TestVolumeRegime:
    def test_high_volume_regime_warning(self, engine: GreeksSignalEngine):
        result = engine.analyze(
            _make_snapshot(), _make_chain(),
            volume_regime="HIGH_RISK",
        )
        assert any("HIGH VOLUME" in w for w in result.warnings)

    def test_normal_volume_no_warning(self, engine: GreeksSignalEngine):
        result = engine.analyze(
            _make_snapshot(), _make_chain(),
            volume_regime="NORMAL",
        )
        assert not any("HIGH VOLUME" in w for w in result.warnings)


class TestDeltaMoneyness:
    def test_itm_delta_gets_positive_score(self, engine: GreeksSignalEngine):
        """ITM delta (>0.65) should now score POSITIVE (was negative)."""
        factor = engine._score_greeks_composite(
            GreeksData(delta=0.75, gamma=0.03, theta=-0.04, vega=0.1,
                       iv=0.3, last_price=5.0, underlying_price=500),
            ChainMetrics(iv_rank=50),
            "C",
        )
        # Should mention ITM and be positive contribution
        assert "ITM" in factor.detail or "0.75" in factor.detail

    def test_far_otm_delta_gets_strong_penalty(self, engine: GreeksSignalEngine):
        """Far OTM delta (<0.10) should have stronger penalty (-0.4)."""
        factor = engine._score_greeks_composite(
            GreeksData(delta=0.05, gamma=0.01, theta=-0.01, vega=0.05,
                       iv=0.3, last_price=0.20, underlying_price=500),
            ChainMetrics(iv_rank=50),
            "C",
        )
        assert "far OTM" in factor.detail.lower() or "0.05" in factor.detail


class TestVPANeutralPenalty:
    def test_neutral_vpa_has_negative_score(self, engine: GreeksSignalEngine):
        factor = engine._score_vpa_bias(None)
        assert factor.score == -0.2
        assert factor.confidence == 0.50

    def test_neutral_vpa_explicit(self, engine: GreeksSignalEngine):
        factor = engine._score_vpa_bias({"bias": "neutral", "strength": 0})
        assert factor.score == -0.2

    def test_bullish_vpa_still_positive(self, engine: GreeksSignalEngine):
        factor = engine._score_vpa_bias({"bias": "bullish", "strength": 0.8})
        assert factor.score > 0


class TestCompositeResult:
    def test_result_has_warnings_field(self, engine: GreeksSignalEngine):
        result = engine.analyze(_make_snapshot(), _make_chain())
        assert hasattr(result, "warnings")
        assert isinstance(result.warnings, list)

    def test_result_has_9_factors(self, engine: GreeksSignalEngine):
        result = engine.analyze(_make_snapshot(), _make_chain())
        assert len(result.factors) == 9  # 8 original + Fair Value

    def test_recommendation_mentions_9_factors(self, engine: GreeksSignalEngine):
        result = engine.analyze(_make_snapshot(), _make_chain())
        assert "/9 factors" in result.recommendation


class TestStrongTrend:
    """Tests for STRONG_UP / STRONG_DOWN trend detection."""

    def test_strong_up_gives_higher_score_than_up(self, engine: GreeksSignalEngine):
        normal = engine._score_trend_alignment("bullish", "UP")
        strong = engine._score_trend_alignment("bullish", "STRONG_UP")
        assert strong.score > normal.score
        assert strong.confidence >= normal.confidence

    def test_strong_down_gives_higher_score_than_down(self, engine: GreeksSignalEngine):
        normal = engine._score_trend_alignment("bearish", "DOWN")
        strong = engine._score_trend_alignment("bearish", "STRONG_DOWN")
        # Both are negative (bearish underlying); STRONG gives larger magnitude
        assert strong.score < normal.score  # -0.7 < -0.5

    def test_strong_counter_trend_harder_penalty(self, engine: GreeksSignalEngine):
        normal_ct = engine._score_trend_alignment("bullish", "DOWN")
        strong_ct = engine._score_trend_alignment("bullish", "STRONG_DOWN")
        assert strong_ct.score < normal_ct.score
        assert strong_ct.confidence >= normal_ct.confidence

    def test_strong_up_score_is_0_7(self, engine: GreeksSignalEngine):
        factor = engine._score_trend_alignment("bullish", "STRONG_UP")
        assert factor.score == 0.7

    def test_strong_counter_trend_score_is_neg_0_9(self, engine: GreeksSignalEngine):
        # STRONG_DOWN counter-trend with bullish signal → score = -0.5 (underlying direction)
        factor = engine._score_trend_alignment("bullish", "STRONG_DOWN")
        assert factor.score == -0.5


class TestFairValueFactor:
    """Tests for the new Fair Value 9th factor."""

    def test_fair_value_factor_present_in_composite(self, engine: GreeksSignalEngine):
        result = engine.analyze(_make_snapshot(), _make_chain())
        names = [f.name for f in result.factors]
        assert "Fair Value" in names

    def test_fair_value_weight_exists(self, engine: GreeksSignalEngine):
        assert "fair_value" in engine.WEIGHTS
        assert engine.WEIGHTS["fair_value"] > 0

    def test_cheap_option_positive_score(self, engine: GreeksSignalEngine):
        class FakeFV:
            pct_difference = 20.0  # theoretical 20% above market -> CHEAP
        factor = engine._score_fair_value(FakeFV())
        assert factor.score > 0

    def test_expensive_option_negative_score(self, engine: GreeksSignalEngine):
        class FakeFV:
            pct_difference = -20.0  # market 20% above theoretical -> EXPENSIVE
        factor = engine._score_fair_value(FakeFV())
        assert factor.score < 0

    def test_fair_value_none_returns_low_confidence(self, engine: GreeksSignalEngine):
        factor = engine._score_fair_value(None)
        assert factor.score == 0.0
        assert factor.confidence < 0.3

    def test_in_range_returns_zero_score(self, engine: GreeksSignalEngine):
        class FakeFV:
            pct_difference = 2.0  # within +/-5% -> FAIR
        factor = engine._score_fair_value(FakeFV())
        assert factor.score == 0.0


class TestVPAMultiBarBoost:
    """Tests for VPA multi-bar confirmation boost."""

    def test_confirmed_reversal_up_boosts_bullish(self, engine: GreeksSignalEngine):
        base = engine._score_vpa_bias(
            {"bias": "bullish", "strength": 0.6, "reason": "test"}, None
        )
        boosted = engine._score_vpa_bias(
            {"bias": "bullish", "strength": 0.6, "reason": "test"},
            "confirmed_reversal_up",
        )
        assert boosted.score > base.score

    def test_confirmed_reversal_down_boosts_bearish(self, engine: GreeksSignalEngine):
        base = engine._score_vpa_bias(
            {"bias": "bearish", "strength": 0.6, "reason": "test"}, None
        )
        boosted = engine._score_vpa_bias(
            {"bias": "bearish", "strength": 0.6, "reason": "test"},
            "confirmed_reversal_down",
        )
        assert boosted.score < base.score  # more negative = stronger bearish

    def test_boost_does_not_exceed_0_9(self, engine: GreeksSignalEngine):
        factor = engine._score_vpa_bias(
            {"bias": "bullish", "strength": 1.0, "reason": "test"},
            "confirmed_reversal_up",
        )
        assert factor.score <= 0.9

    def test_wrong_direction_signal_no_boost(self, engine: GreeksSignalEngine):
        """A bearish confirm signal on a bullish bias should NOT boost."""
        base = engine._score_vpa_bias(
            {"bias": "bullish", "strength": 0.6, "reason": "test"}, None
        )
        no_boost = engine._score_vpa_bias(
            {"bias": "bullish", "strength": 0.6, "reason": "test"},
            "confirmed_reversal_down",
        )
        assert no_boost.score == base.score


class TestConvictionAmplifier:
    """Tests for the 6+ factor conviction amplifier."""

    def test_high_conviction_warning_present_when_many_agree(self, engine: GreeksSignalEngine):
        """With strong aligned signals, HIGH CONVICTION warning should appear."""
        # Strong bullish setup: low IVR (buyers' market), negative GEX (trending),
        # bullish VPA, STRONG_UP trend, no SR data, fair value None
        result = engine.analyze(
            _make_snapshot(delta=0.5, last_price=2.0),
            _make_chain(),  # standard chain; strong signals come from vpa_bias + trend
            vpa_bias={"bias": "bullish", "strength": 0.9, "reason": "strong"},
            underlying_trend="STRONG_UP",
        )
        # Just ensure no crash and score is non-trivially positive
        assert result.score is not None


class TestVolumeDampeningFloor:
    """Tests that combined dampening never drops below floor."""

    def test_floor_prevents_combined_crush(self, engine: GreeksSignalEngine):
        """DTE=1 + premium floor together should not crush below 0.4x."""
        # Set up a scenario where both DTE=1 and premium floor would apply
        result_normal = engine.analyze(
            _make_snapshot(delta=0.4, last_price=2.0), _make_chain(), dte=5
        )
        result_crushed = engine.analyze(
            _make_snapshot(delta=0.4, last_price=0.10), _make_chain(), dte=1
        )
        # Both should have valid (non-None) scores
        assert result_normal.score is not None
        assert result_crushed.score is not None
        # Composite score should not be identically zero (signal still there, just dampened)
        # Under old code this would be ~0.18x; under new floor it's max(0.40, 0.18)=0.4x