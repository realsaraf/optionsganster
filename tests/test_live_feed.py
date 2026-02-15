"""
Backtest / integration tests for the live feed pipeline.

Proves that CSV historical data, when converted to Polygon wire format,
flows through the exact same processing pipeline as live WebSocket data
and produces identical bar/signal output.

Key verifications:
  1. CSV → Polygon msg_dict produces the same format as real Polygon messages
  2. msg_dict → bar conversion is the single authoritative path for BOTH live & mock
  3. _handle_aggregate (the live pipeline) correctly processes CSV-sourced data
  4. All 3 message types (EquityAgg, LaunchpadValue, FMV) produce valid OHLCV bars
  5. Signal detection fires on historical data with expected patterns
  6. Subscriber queues receive the correct payloads
"""

import asyncio
import csv
import json
import os
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

# ── Import the live feed internals ────────────────────────────
from app.live_feed import (
    _csv_row_to_msg_dict,
    _msg_dict_to_bar,
    _load_csv_bars,
    _base_symbol,
    _detect_bar_signals,
    _scale_to_option,
    build_mock_sow,
    LiveFeedManager,
)

_ET = ZoneInfo("America/New_York")
_DATA_DIR = Path(__file__).parent.parent / "data"


# ─────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────

@pytest.fixture
def sample_csv_rows():
    """Load first 20 rows from a real CSV file."""
    csv_file = _DATA_DIR / "qqq" / "2026-01-02.csv"
    if not csv_file.exists():
        pytest.skip("No CSV test data available")
    rows = []
    with open(csv_file, newline="") as fh:
        reader = csv.DictReader(fh)
        for i, row in enumerate(reader):
            rows.append(row)
            if i >= 19:
                break
    return rows


@pytest.fixture
def sample_polygon_msg():
    """A realistic Polygon A.* aggregate message as would arrive from raw WS."""
    return {
        "ev": "A",
        "sym": "O:QQQ260213C00601000",
        "v": 150,
        "av": 12500,
        "op": 5.20,
        "vw": 5.35,
        "o": 5.30,
        "c": 5.40,
        "h": 5.45,
        "l": 5.25,
        "a": 3,
        "s": 1739451600000,   # 2025-02-13 10:00:00 EST in ms
        "e": 1739451601000,
    }


@pytest.fixture
def lfm():
    """A fresh LiveFeedManager instance for testing _handle_aggregate."""
    return LiveFeedManager()


# ─────────────────────────────────────────────────────────────
# 1. CSV → Polygon msg_dict conversion
# ─────────────────────────────────────────────────────────────

class TestCsvToMsgDict:
    """Verify CSV rows produce valid Polygon-format messages."""

    def test_basic_conversion(self, sample_csv_rows):
        """CSV row converts to msg_dict with all required Polygon fields."""
        row = sample_csv_rows[0]
        msg = _csv_row_to_msg_dict(row)

        # All Polygon aggregate fields must be present
        required_keys = {"ev", "sym", "v", "av", "op", "vw", "o", "c", "h", "l", "a", "s", "e"}
        assert required_keys.issubset(msg.keys()), f"Missing keys: {required_keys - msg.keys()}"

        # Event type
        assert msg["ev"] == "A"

        # Symbol from CSV
        assert msg["sym"] == row["ticker"]

        # Numeric values match CSV
        assert msg["o"] == float(row["open"])
        assert msg["c"] == float(row["close"])
        assert msg["h"] == float(row["high"])
        assert msg["l"] == float(row["low"])
        assert msg["v"] == int(row["volume"])

    def test_timestamp_conversion(self, sample_csv_rows):
        """CSV window_start (nanoseconds) → Polygon s field (milliseconds)."""
        row = sample_csv_rows[0]
        msg = _csv_row_to_msg_dict(row)

        window_start_ns = int(row["window_start"])
        expected_ms = window_start_ns // 1_000_000

        assert msg["s"] == expected_ms
        assert msg["e"] == expected_ms + 60_000  # 1-minute bar

        # Verify the timestamp is reasonable (year 2026)
        dt = datetime.fromtimestamp(msg["s"] / 1000, tz=ZoneInfo("UTC"))
        assert dt.year == 2026

    def test_ticker_override(self, sample_csv_rows):
        """When ticker is provided, it overrides the CSV ticker field."""
        row = sample_csv_rows[0]
        msg = _csv_row_to_msg_dict(row, ticker="O:QQQ260213C00601000")
        assert msg["sym"] == "O:QQQ260213C00601000"

    def test_vwap_approximation(self, sample_csv_rows):
        """vwap is approximated as (high + low) / 2 from CSV data."""
        row = sample_csv_rows[0]
        msg = _csv_row_to_msg_dict(row)
        expected_vw = round((float(row["high"]) + float(row["low"])) / 2, 4)
        assert msg["vw"] == expected_vw

    def test_all_rows_convert(self, sample_csv_rows):
        """Every CSV row successfully converts to a valid msg_dict."""
        for i, row in enumerate(sample_csv_rows):
            msg = _csv_row_to_msg_dict(row)
            assert msg["ev"] == "A", f"Row {i}: bad ev"
            assert msg["v"] > 0 or msg["v"] == 0, f"Row {i}: negative volume"
            assert msg["s"] > 0, f"Row {i}: bad timestamp"
            assert msg["h"] >= msg["l"], f"Row {i}: high < low"

    def test_matches_live_polygon_format(self, sample_csv_rows, sample_polygon_msg):
        """CSV msg_dict has the exact same keys as a real Polygon message."""
        csv_msg = _csv_row_to_msg_dict(sample_csv_rows[0])
        live_keys = set(sample_polygon_msg.keys())
        csv_keys = set(csv_msg.keys())
        assert csv_keys == live_keys, f"Key mismatch: CSV has {csv_keys - live_keys}, missing {live_keys - csv_keys}"

    def test_transactions_mapped_to_a(self, sample_csv_rows):
        """CSV 'transactions' column maps to Polygon 'a' (average_size) field."""
        row = sample_csv_rows[0]
        msg = _csv_row_to_msg_dict(row)
        assert msg["a"] == int(row["transactions"])


# ─────────────────────────────────────────────────────────────
# 2. msg_dict → bar conversion
# ─────────────────────────────────────────────────────────────

class TestMsgDictToBar:
    """Verify the shared bar conversion produces correct output."""

    def test_bar_format(self, sample_polygon_msg):
        """Bar dict has all required fields."""
        bar = _msg_dict_to_bar(sample_polygon_msg)
        required = {"datetime", "open", "high", "low", "close", "volume", "vwap", "accumulated_volume"}
        assert required.issubset(bar.keys())

    def test_ohlcv_values(self, sample_polygon_msg):
        """OHLCV values are correctly mapped from msg_dict."""
        bar = _msg_dict_to_bar(sample_polygon_msg)
        assert bar["open"] == sample_polygon_msg["o"]
        assert bar["high"] == sample_polygon_msg["h"]
        assert bar["low"] == sample_polygon_msg["l"]
        assert bar["close"] == sample_polygon_msg["c"]
        assert bar["volume"] == sample_polygon_msg["v"]
        assert bar["vwap"] == sample_polygon_msg["vw"]
        assert bar["accumulated_volume"] == sample_polygon_msg["av"]

    def test_timestamp_to_eastern(self, sample_polygon_msg):
        """Polygon ms timestamp is correctly converted to Eastern time string."""
        bar = _msg_dict_to_bar(sample_polygon_msg)
        # Verify datetime is a valid string
        dt = datetime.strptime(bar["datetime"], "%Y-%m-%d %H:%M:%S")
        assert dt.year >= 2025

    def test_zero_timestamp_fallback(self):
        """When s=0, bar uses current time as fallback."""
        msg = {"ev": "A", "sym": "TEST", "o": 1, "c": 2, "h": 3, "l": 0.5, "v": 100, "s": 0}
        bar = _msg_dict_to_bar(msg)
        dt = datetime.strptime(bar["datetime"], "%Y-%m-%d %H:%M:%S")
        # Should be approximately now (within last minute)
        now = datetime.now()
        assert abs((now - dt).total_seconds()) < 120

    def test_csv_roundtrip(self, sample_csv_rows):
        """CSV → msg_dict → bar produces valid OHLCV bar."""
        row = sample_csv_rows[0]
        msg = _csv_row_to_msg_dict(row)
        bar = _msg_dict_to_bar(msg)

        assert bar["open"] == float(row["open"])
        assert bar["close"] == float(row["close"])
        assert bar["high"] == float(row["high"])
        assert bar["low"] == float(row["low"])
        assert bar["volume"] == int(row["volume"])
        assert "datetime" in bar
        assert bar["datetime"] != ""

    def test_csv_bar_matches_live_bar_format(self, sample_csv_rows, sample_polygon_msg):
        """A bar from CSV data has the exact same keys as a bar from live data."""
        csv_msg = _csv_row_to_msg_dict(sample_csv_rows[0])
        csv_bar = _msg_dict_to_bar(csv_msg)
        live_bar = _msg_dict_to_bar(sample_polygon_msg)
        assert set(csv_bar.keys()) == set(live_bar.keys())


# ─────────────────────────────────────────────────────────────
# 3. Simulated message types (EquityAgg, LaunchpadValue, FMV)
# ─────────────────────────────────────────────────────────────

class TestMessageTypes:
    """Verify all 3 Polygon message types produce valid bars when processed."""

    def test_equity_agg_full_ohlcv(self):
        """EquityAgg (full OHLCV) → msg_dict → valid bar."""
        # This is what the raw WS or Massive EquityAgg handler produces
        msg = {
            "ev": "A", "sym": "O:SPY260115C00600000",
            "v": 500, "av": 25000, "op": 10.0,
            "vw": 10.50, "o": 10.20, "c": 10.60,
            "h": 10.80, "l": 10.10, "a": 5,
            "s": 1739451600000, "e": 1739451601000,
        }
        bar = _msg_dict_to_bar(msg)
        assert bar["open"] == 10.20
        assert bar["close"] == 10.60
        assert bar["high"] == 10.80
        assert bar["low"] == 10.10
        assert bar["volume"] == 500
        assert bar["vwap"] == 10.50

    def test_launchpad_value_synthetic_ohlcv(self):
        """LaunchpadValue (single .value) → synthetic OHLCV msg_dict → valid bar.

        Simulates what the _MassiveUpstreamWS processor creates from
        LaunchpadValue messages (which only have .value, no .open/.close).
        """
        # The processor aggregates LV ticks into per-second candles:
        val = 5.35
        sec = 1739451600  # seconds
        msg = {
            "ev": "A", "sym": "O:QQQ260213C00601000",
            "o": val, "h": val + 0.05, "l": val - 0.02, "c": val + 0.03,
            "v": 1, "vw": val,
            "s": sec * 1000, "e": sec * 1000,
            "av": 0, "op": 0, "a": 0,
        }
        bar = _msg_dict_to_bar(msg)
        assert bar["open"] == val
        assert bar["high"] == val + 0.05
        assert bar["low"] == val - 0.02
        assert bar["close"] == val + 0.03
        assert bar["volume"] == 1  # LV always volume=1

    def test_fmv_synthetic_ohlcv(self):
        """FairMarketValue (single .fmv) → synthetic OHLCV msg_dict → valid bar.

        Same processing path as LaunchpadValue but sourced from FMV data.
        """
        fmv_val = 8.50
        sec = 1739451600
        msg = {
            "ev": "A", "sym": "O:AAPL260115C00200000",
            "o": fmv_val, "h": fmv_val, "l": fmv_val, "c": fmv_val,
            "v": 1, "vw": fmv_val,
            "s": sec * 1000, "e": sec * 1000,
            "av": 0, "op": 0, "a": 0,
        }
        bar = _msg_dict_to_bar(msg)
        assert bar["open"] == fmv_val
        assert bar["close"] == fmv_val
        assert bar["volume"] == 1

    def test_csv_as_equity_agg(self, sample_csv_rows):
        """CSV row → msg_dict matches EquityAgg format (full OHLCV with volume)."""
        row = sample_csv_rows[0]
        msg = _csv_row_to_msg_dict(row)

        # Should look like EquityAgg (has real volume, real OHLCV)
        assert msg["v"] > 0, "CSV rows should have real volume (unlike LV/FMV v=1)"
        assert msg["o"] > 0
        assert msg["h"] >= msg["l"]
        assert msg["s"] > 0

        bar = _msg_dict_to_bar(msg)
        assert bar["volume"] == int(row["volume"])


# ─────────────────────────────────────────────────────────────
# 4. _handle_aggregate pipeline (end-to-end)
# ─────────────────────────────────────────────────────────────

class TestHandleAggregate:
    """Test the full _handle_aggregate pipeline with CSV data."""

    @pytest.mark.asyncio
    async def test_csv_through_pipeline(self, sample_csv_rows, lfm):
        """CSV data flows through _handle_aggregate and reaches subscriber queues."""
        ticker = "QQQ"
        queue = asyncio.Queue(maxsize=100)
        lfm._subscribers[ticker].add(queue)

        # Feed first 10 CSV rows through the pipeline
        for row in sample_csv_rows[:10]:
            msg = _csv_row_to_msg_dict(row, ticker=ticker)
            await lfm._handle_aggregate(msg)

        # Verify bars arrived in subscriber queue
        assert not queue.empty(), "No bars reached the subscriber queue!"

        received = []
        while not queue.empty():
            payload = json.loads(queue.get_nowait())
            received.append(payload)

        assert len(received) == 10

        # Verify payload format matches what the frontend expects
        for payload in received:
            assert payload["type"] == "bar"
            assert payload["ticker"] == ticker
            assert "bar" in payload
            assert "signals" in payload

            bar = payload["bar"]
            assert "datetime" in bar
            assert "open" in bar
            assert "high" in bar
            assert "low" in bar
            assert "close" in bar
            assert "volume" in bar
            assert "vwap" in bar

    @pytest.mark.asyncio
    async def test_bar_history_accumulates(self, sample_csv_rows, lfm):
        """_handle_aggregate accumulates bar history for signal detection."""
        ticker = "QQQ"
        lfm._subscribers[ticker].add(asyncio.Queue(maxsize=100))

        for row in sample_csv_rows[:15]:
            msg = _csv_row_to_msg_dict(row, ticker=ticker)
            await lfm._handle_aggregate(msg)

        assert ticker in lfm._bar_history
        assert len(lfm._bar_history[ticker]) == 15

    @pytest.mark.asyncio
    async def test_option_ticker_through_pipeline(self, sample_csv_rows, lfm):
        """Option tickers (O:...) flow through correctly."""
        ticker = "O:QQQ260213C00601000"
        queue = asyncio.Queue(maxsize=100)
        lfm._subscribers[ticker].add(queue)

        msg = _csv_row_to_msg_dict(sample_csv_rows[0], ticker=ticker)
        await lfm._handle_aggregate(msg)

        payload = json.loads(queue.get_nowait())
        assert payload["ticker"] == ticker
        assert payload["bar"]["open"] == float(sample_csv_rows[0]["open"])

    @pytest.mark.asyncio
    async def test_multiple_subscribers_receive_bars(self, sample_csv_rows, lfm):
        """Multiple subscriber queues all receive the same bar."""
        ticker = "QQQ"
        q1 = asyncio.Queue(maxsize=100)
        q2 = asyncio.Queue(maxsize=100)
        q3 = asyncio.Queue(maxsize=100)
        lfm._subscribers[ticker].update({q1, q2, q3})

        msg = _csv_row_to_msg_dict(sample_csv_rows[0], ticker=ticker)
        await lfm._handle_aggregate(msg)

        for q in [q1, q2, q3]:
            assert not q.empty()
            payload = json.loads(q.get_nowait())
            assert payload["type"] == "bar"
            assert payload["ticker"] == ticker

    @pytest.mark.asyncio
    async def test_no_subscribers_no_crash(self, sample_csv_rows, lfm):
        """Pipeline handles zero subscribers without error."""
        msg = _csv_row_to_msg_dict(sample_csv_rows[0], ticker="QQQ")
        await lfm._handle_aggregate(msg)  # should not raise

    @pytest.mark.asyncio
    async def test_empty_sym_ignored(self, lfm):
        """Messages with empty sym are silently dropped."""
        msg = {"ev": "A", "sym": "", "o": 1, "c": 2, "h": 3, "l": 0.5, "v": 100, "s": 1000000}
        await lfm._handle_aggregate(msg)  # should not raise


# ─────────────────────────────────────────────────────────────
# 5. Signal detection on historical data
# ─────────────────────────────────────────────────────────────

class TestSignalDetection:
    """Verify signal detection works on CSV-sourced bars."""

    def test_needs_history(self):
        """No signals when history is too short."""
        bar = {"open": 10, "high": 11, "low": 9, "close": 10.5, "volume": 100,
               "datetime": "2026-01-02 09:30:00"}
        signals = _detect_bar_signals([bar] * 3, bar)
        assert signals == []

    def test_high_volume_bullish(self):
        """High volume green bar triggers strong_bullish signal."""
        # Build history with average volume ~100
        history = []
        for i in range(20):
            history.append({
                "open": 10 + i * 0.01, "high": 10.5 + i * 0.01,
                "low": 9.5 + i * 0.01, "close": 10.2 + i * 0.01,
                "volume": 100, "datetime": f"2026-01-02 09:{30 + i}:00",
            })

        # New bar with 3x volume, solid green body
        bar = {
            "open": 10.0, "high": 11.0, "low": 10.0, "close": 10.9,
            "volume": 300, "datetime": "2026-01-02 09:55:00",
        }
        signals = _detect_bar_signals(history, bar)
        signal_names = [s["signal"] for s in signals]
        assert "strong_bullish" in signal_names

    def test_high_volume_bearish(self):
        """High volume red bar triggers strong_bearish signal."""
        history = []
        for i in range(20):
            history.append({
                "open": 10, "high": 10.5, "low": 9.5, "close": 10.2,
                "volume": 100, "datetime": f"2026-01-02 09:{30 + i}:00",
            })

        bar = {
            "open": 10.9, "high": 11.0, "low": 10.0, "close": 10.1,
            "volume": 300, "datetime": "2026-01-02 09:55:00",
        }
        signals = _detect_bar_signals(history, bar)
        signal_names = [s["signal"] for s in signals]
        assert "strong_bearish" in signal_names

    def test_low_volume_weak_signal(self):
        """Very low volume triggers weak_up or weak_down."""
        history = [{
            "open": 10, "high": 10.5, "low": 9.5, "close": 10.2,
            "volume": 1000, "datetime": f"2026-01-02 09:{30+i}:00",
        } for i in range(20)]

        bar = {
            "open": 10.0, "high": 10.3, "low": 9.9, "close": 10.2,
            "volume": 100,  # 0.1x average
            "datetime": "2026-01-02 09:55:00",
        }
        signals = _detect_bar_signals(history, bar)
        signal_names = [s["signal"] for s in signals]
        assert "weak_up" in signal_names or "weak_down" in signal_names

    @pytest.mark.asyncio
    async def test_signals_in_pipeline_output(self, lfm):
        """Signals appear in the pipeline output payload."""
        ticker = "QQQ"
        queue = asyncio.Queue(maxsize=200)
        lfm._subscribers[ticker].add(queue)

        # Feed enough bars to build history, then a high-volume bar
        base_ts = 1739451600000  # some base ms timestamp
        for i in range(20):
            msg = {
                "ev": "A", "sym": ticker,
                "o": 600 + i * 0.1, "c": 600.2 + i * 0.1,
                "h": 600.5 + i * 0.1, "l": 599.5 + i * 0.1,
                "v": 100, "vw": 600, "av": 0,
                "s": base_ts + i * 60000, "e": base_ts + i * 60000 + 1000,
            }
            await lfm._handle_aggregate(msg)

        # Now send a high-volume bar
        climax_msg = {
            "ev": "A", "sym": ticker,
            "o": 600, "c": 601.8, "h": 602, "l": 600,
            "v": 500,  # 5x average
            "vw": 601, "av": 0,
            "s": base_ts + 20 * 60000, "e": base_ts + 20 * 60000 + 1000,
        }
        await lfm._handle_aggregate(climax_msg)

        # Drain queue and check last message
        last_payload = None
        while not queue.empty():
            last_payload = json.loads(queue.get_nowait())

        assert last_payload is not None
        assert len(last_payload["signals"]) > 0, "Expected signals on 5x volume bar"
        assert last_payload["signals"][0]["signal"] == "strong_bullish"


# ─────────────────────────────────────────────────────────────
# 6. build_mock_sow consistency
# ─────────────────────────────────────────────────────────────

class TestBuildMockSow:
    """Verify SOW bars use the same conversion path."""

    def test_sow_bar_format(self):
        """SOW bars have the same fields as live bars."""
        bars, idx = build_mock_sow("QQQ", interval_minutes=5, num_candles=10)
        if not bars:
            pytest.skip("No CSV data for SOW test")

        live_bar_keys = {"datetime", "open", "high", "low", "close", "volume", "vwap", "accumulated_volume"}
        for bar in bars:
            assert live_bar_keys.issubset(bar.keys()), f"Missing: {live_bar_keys - bar.keys()}"

    def test_sow_ohlcv_valid(self):
        """SOW bars have valid OHLCV values (high >= low, etc.)."""
        bars, _ = build_mock_sow("QQQ", interval_minutes=1, num_candles=20)
        if not bars:
            pytest.skip("No CSV data")

        for bar in bars:
            assert bar["high"] >= bar["low"], f"high < low: {bar}"
            assert bar["volume"] > 0
            assert bar["open"] > 0
            assert bar["close"] > 0

    def test_sow_option_scaling(self):
        """Option tickers get price scaling applied."""
        stock_bars, _ = build_mock_sow("QQQ", interval_minutes=5, num_candles=5)
        option_bars, _ = build_mock_sow("O:QQQ260213C00601000", interval_minutes=5, num_candles=5)
        if not stock_bars or not option_bars:
            pytest.skip("No CSV data")

        # Option prices should be ~$5 range (scaled), not ~$600 range (raw QQQ)
        avg_option_price = sum(b["close"] for b in option_bars) / len(option_bars)
        avg_stock_price = sum(b["close"] for b in stock_bars) / len(stock_bars)

        assert avg_option_price < 50, f"Option price too high: {avg_option_price} (should be ~$5 scaled)"
        assert avg_stock_price > 100, f"Stock price too low: {avg_stock_price} (should be ~$600 QQQ)"

    def test_sow_returns_start_index(self):
        """SOW returns correct start_index for continuing playback."""
        bars, start_idx = build_mock_sow("QQQ", interval_minutes=5, num_candles=10)
        if not bars:
            pytest.skip("No CSV data")
        # start_idx should equal num_candles * rows_per_candle
        assert start_idx == 10 * 5  # 10 candles × 5 rows each


# ─────────────────────────────────────────────────────────────
# 7. CSV ↔ Live format parity (the critical backtest)
# ─────────────────────────────────────────────────────────────

class TestFormatParity:
    """
    The CRITICAL test: prove that CSV data, when converted through
    _csv_row_to_msg_dict → _msg_dict_to_bar, produces bars identical
    to what the live pipeline would create from Polygon messages.
    """

    @pytest.mark.asyncio
    async def test_csv_bar_matches_live_bar_keys(self, sample_csv_rows, lfm):
        """Bar from CSV pipeline has same keys as bar from live pipeline."""
        # CSV path
        csv_msg = _csv_row_to_msg_dict(sample_csv_rows[0], ticker="QQQ")
        csv_bar = _msg_dict_to_bar(csv_msg)

        # Simulated live path (same function!)
        live_msg = {
            "ev": "A", "sym": "QQQ",
            "o": 619.0, "c": 620.59, "h": 620.69, "l": 619.0,
            "v": 24104, "vw": 619.85, "av": 50000,
            "s": 1739451600000, "e": 1739451601000,
        }
        live_bar = _msg_dict_to_bar(live_msg)

        assert set(csv_bar.keys()) == set(live_bar.keys()), "Bar format mismatch!"

    @pytest.mark.asyncio
    async def test_same_conversion_function(self, sample_csv_rows, lfm):
        """Both CSV and live paths use the exact same _msg_dict_to_bar function."""
        ticker = "QQQ"
        queue = asyncio.Queue(maxsize=100)
        lfm._subscribers[ticker].add(queue)

        # Feed CSV row through the LIVE pipeline (_handle_aggregate)
        csv_msg = _csv_row_to_msg_dict(sample_csv_rows[0], ticker=ticker)
        await lfm._handle_aggregate(csv_msg)

        # Get the bar from the pipeline output
        pipeline_payload = json.loads(queue.get_nowait())
        pipeline_bar = pipeline_payload["bar"]

        # Also create bar directly via shared function
        direct_bar = _msg_dict_to_bar(csv_msg)

        # They MUST be identical — same function, same input
        assert pipeline_bar == direct_bar, (
            f"Pipeline bar != direct bar!\n"
            f"Pipeline: {pipeline_bar}\n"
            f"Direct:   {direct_bar}"
        )

    def test_csv_timestamps_preserved(self, sample_csv_rows):
        """CSV timestamps (from window_start) are preserved through conversion."""
        row = sample_csv_rows[0]
        msg = _csv_row_to_msg_dict(row)
        bar = _msg_dict_to_bar(msg)

        # The bar datetime should correspond to the CSV window_start
        ws_ns = int(row["window_start"])
        expected_utc = datetime.fromtimestamp(ws_ns / 1e9, tz=ZoneInfo("UTC"))
        expected_et = expected_utc.astimezone(_ET).replace(tzinfo=None)
        expected_str = expected_et.strftime("%Y-%m-%d %H:%M:%S")

        assert bar["datetime"] == expected_str, (
            f"Timestamp mismatch: got '{bar['datetime']}', expected '{expected_str}'"
        )

    @pytest.mark.asyncio
    async def test_full_day_replay(self, lfm):
        """Replay an entire day of CSV data through the live pipeline."""
        csv_file = _DATA_DIR / "qqq" / "2026-01-02.csv"
        if not csv_file.exists():
            pytest.skip("No CSV data")

        ticker = "QQQ"
        queue = asyncio.Queue(maxsize=5000)
        lfm._subscribers[ticker].add(queue)

        rows = []
        with open(csv_file, newline="") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)

        # Feed all rows through the live pipeline
        for row in rows:
            msg = _csv_row_to_msg_dict(row, ticker=ticker)
            await lfm._handle_aggregate(msg)

        # Verify all bars arrived
        received_count = 0
        while not queue.empty():
            payload = json.loads(queue.get_nowait())
            assert payload["type"] == "bar"
            assert payload["ticker"] == ticker
            bar = payload["bar"]
            assert bar["high"] >= bar["low"]
            assert bar["volume"] >= 0
            received_count += 1

        assert received_count == len(rows), (
            f"Expected {len(rows)} bars, got {received_count}"
        )

    @pytest.mark.asyncio
    async def test_multi_day_replay_with_signals(self, lfm):
        """Replay multiple days and verify signals are generated."""
        qqq_dir = _DATA_DIR / "qqq"
        if not qqq_dir.is_dir():
            pytest.skip("No CSV data directory")

        csv_files = sorted(f for f in os.listdir(qqq_dir) if f.endswith(".csv"))[:3]
        if not csv_files:
            pytest.skip("No CSV files")

        ticker = "QQQ"
        queue = asyncio.Queue(maxsize=10000)
        lfm._subscribers[ticker].add(queue)

        total_rows = 0
        for fname in csv_files:
            with open(qqq_dir / fname, newline="") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    msg = _csv_row_to_msg_dict(row, ticker=ticker)
                    await lfm._handle_aggregate(msg)
                    total_rows += 1

        # Count bars and signals
        bars_received = 0
        signals_received = 0
        while not queue.empty():
            payload = json.loads(queue.get_nowait())
            bars_received += 1
            signals_received += len(payload.get("signals", []))

        assert bars_received == total_rows
        # With 3 days of data we should get at least some signals
        assert signals_received > 0, (
            f"No signals generated from {total_rows} bars across {len(csv_files)} days!"
        )


# ─────────────────────────────────────────────────────────────
# 8. Helper function tests
# ─────────────────────────────────────────────────────────────

class TestHelpers:
    """Test utility functions used in the pipeline."""

    def test_base_symbol_stock(self):
        assert _base_symbol("QQQ") == "qqq"
        assert _base_symbol("SPY") == "spy"

    def test_base_symbol_option(self):
        assert _base_symbol("O:QQQ260213C00601000") == "qqq"
        assert _base_symbol("O:SPY260115P00500000") == "spy"

    def test_scale_to_option(self):
        """Option scaling converts stock percentage moves to option price."""
        base = 600.0
        # At base price, option should be ~$5
        assert abs(_scale_to_option(600.0, 600.0) - 5.0) < 0.01
        # 1% up in stock → 10% up in option
        scaled = _scale_to_option(606.0, 600.0)  # +1%
        assert scaled > 5.0  # option should be up
        assert abs(scaled - 5.5) < 0.01  # ~10% of $5 = $5.50

    def test_load_csv_bars(self):
        """CSV loader returns list of dicts with correct keys."""
        bars = _load_csv_bars("qqq")
        if not bars:
            pytest.skip("No CSV data")
        expected_keys = {"ticker", "volume", "open", "close", "high", "low", "window_start", "transactions"}
        assert expected_keys.issubset(bars[0].keys()), f"Missing: {expected_keys - bars[0].keys()}"
        assert len(bars) > 100, "Expected at least 100 bars from QQQ data"
