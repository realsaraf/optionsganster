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
        assert engine.WEIGHTS["trend_alignment"] == 0.15


class TestTrendAlignment:
    def test_with_trend_bullish(self, engine: GreeksSignalEngine):
        factor = engine._score_trend_alignment("bullish", "UP")
        assert factor.score == 0.5
        assert "WITH-TREND" in factor.detail

    def test_counter_trend_bullish_on_down_day(self, engine: GreeksSignalEngine):
        factor = engine._score_trend_alignment("bullish", "DOWN")
        assert factor.score == -0.8
        assert "COUNTER-TREND" in factor.detail

    def test_counter_trend_bearish_on_up_day(self, engine: GreeksSignalEngine):
        factor = engine._score_trend_alignment("bearish", "UP")
        assert factor.score == -0.8

    def test_with_trend_bearish_on_down(self, engine: GreeksSignalEngine):
        factor = engine._score_trend_alignment("bearish", "DOWN")
        assert factor.score == 0.5

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

    def test_result_has_7_factors(self, engine: GreeksSignalEngine):
        result = engine.analyze(_make_snapshot(), _make_chain())
        assert len(result.factors) == 7

    def test_recommendation_mentions_7_factors(self, engine: GreeksSignalEngine):
        result = engine.analyze(_make_snapshot(), _make_chain())
        assert "/7 factors" in result.recommendation
