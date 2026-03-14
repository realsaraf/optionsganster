from datetime import datetime

from app.alert_manager import AlertManager, AlertState
from app.edge_scorer import EdgeResult
from app.setup_engine import SetupAlert


def _make_edge_result(symbol: str = "QQQ") -> EdgeResult:
    setup = SetupAlert(
        name="VWAP Reclaim",
        direction="CALL",
        symbol=symbol,
        trigger_price=500.0,
        entry_condition="1m close above VWAP",
        stop_price=498.0,
        target_1=503.0,
        target_2=505.0,
        time_stop_minutes=15,
        regime="TREND_UP",
        reasons=["Trend aligned", "Above VWAP"],
        confidence=0.82,
    )
    return EdgeResult(
        edge_score=74,
        tier="A",
        setup=setup,
        pick=None,
        structure_score=24,
        regime_score=16,
        momentum_score=10,
        options_score=14,
        risk_score=10,
        detail="test",
    )


def test_alert_manager_propagates_symbol_and_entry_condition():
    manager = AlertManager()
    alerts = manager.process_tick(
        [_make_edge_result()],
        current_price=500.5,
        current_time=datetime(2026, 3, 14, 14, 30),
    )

    assert len(alerts) == 1
    alert = alerts[0]
    assert alert.symbol == "QQQ"
    assert alert.entry_condition == "1m close above VWAP"


def test_alert_manager_activation_and_resolution_use_entry_premium():
    manager = AlertManager()
    alert = manager.process_tick(
        [_make_edge_result("SPY")],
        current_price=500.5,
        current_time=datetime(2026, 3, 14, 14, 30),
    )[0]

    activated = manager.mark_activated(alert.id, 1.25)
    resolved = manager.resolve_alert(alert.id, AlertState.HIT_T1, 1.75)

    assert activated is not None
    assert resolved is not None
    assert resolved.symbol == "SPY"
    assert resolved.entry_premium == 1.25
    assert resolved.exit_premium == 1.75
    assert resolved.pnl_pct == 40.0


def test_alert_manager_allows_same_setup_on_different_symbols():
    manager = AlertManager()

    qqq = manager.process_tick(
        [_make_edge_result("QQQ")],
        current_price=500.5,
        current_time=datetime(2026, 3, 14, 14, 30),
        symbol="QQQ",
    )
    spy = manager.process_tick(
        [_make_edge_result("SPY")],
        current_price=500.5,
        current_time=datetime(2026, 3, 14, 14, 31),
        symbol="SPY",
    )

    assert len(qqq) == 1
    assert len(spy) == 1
    assert {alert.symbol for alert in manager.get_active_alerts()} == {"QQQ", "SPY"}


def test_alert_manager_resolves_only_matching_symbol():
    manager = AlertManager()
    qqq = manager.process_tick(
        [_make_edge_result("QQQ")],
        current_price=500.5,
        current_time=datetime(2026, 3, 14, 14, 30),
        symbol="QQQ",
    )[0]
    spy = manager.process_tick(
        [_make_edge_result("SPY")],
        current_price=500.5,
        current_time=datetime(2026, 3, 14, 14, 31),
        symbol="SPY",
    )[0]

    manager.process_tick(
        [],
        current_price=503.5,
        current_time=datetime(2026, 3, 14, 14, 35),
        symbol="QQQ",
    )

    active_symbols = {alert.symbol for alert in manager.get_active_alerts()}
    all_alerts = {alert.id: alert for alert in manager.get_all_alerts()}

    assert qqq.id in all_alerts
    assert all_alerts[qqq.id].state == AlertState.HIT_T1
    assert spy.id in all_alerts
    assert spy.symbol in active_symbols
