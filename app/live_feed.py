"""
Live WebSocket Feed Manager
============================
Manages TWO upstream WebSockets:
  1. Options  – raw websockets to wss://delayed.polygon.io/options (for O:* tickers)
  2. Stocks   – massive Python client to delayed.massive.com/stocks (for stock tickers)

Both connections are lazy – they only start when a client subscribes to a
ticker of that market.  Subscriptions are reference-counted with a 5-second
grace period on last unsubscribe.

Usage from FastAPI:
  feed_manager = LiveFeedManager()
  await feed_manager.start()          # call in lifespan startup
  await feed_manager.stop()           # call in lifespan shutdown
  await feed_manager.subscribe(ticker, queue)
  await feed_manager.unsubscribe(ticker, queue)
"""

import asyncio
import csv
import json
import os
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import websockets
import websockets.exceptions

from massive import WebSocketClient
from massive.websocket.models import Feed, Market

from app.config import settings

_ET = ZoneInfo("America/New_York")

def _log(msg: str):
    """Print with [LiveFeed] prefix so it's visible in uvicorn output."""
    print(f"[LiveFeed] {msg}", flush=True)


# ── Polygon / Massive WebSocket URLs ────────────────────────
OPTIONS_WS_DELAYED = "wss://delayed.polygon.io/options"
OPTIONS_WS_REALTIME = "wss://socket.polygon.io/options"

# Grace period before upstream unsubscribe (seconds)
UNSUBSCRIBE_GRACE = 5


def _is_option_ticker(ticker: str) -> bool:
    """Return True if the ticker is an options OCC symbol (starts with 'O:')."""
    return ticker.startswith("O:")


def _market_label(ticker: str) -> str:
    return "options" if _is_option_ticker(ticker) else "stocks"


# ─────────────────────────────────────────────────────────────
# Upstream connection – one per market
# ─────────────────────────────────────────────────────────────
class _UpstreamWS:
    """Encapsulates a single upstream Polygon WebSocket connection."""

    def __init__(self, label: str, url: str, on_message):
        self.label = label          # "options" or "stocks"
        self.url = url
        self._on_message = on_message  # async callback(raw_msg)
        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self.authenticated = False
        self._auth_failed_permanent = False  # True if plan doesn't cover this market
        self._task: Optional[asyncio.Task] = None
        self._running = False
        self._reconnect_delay = 1
        # tickers actively subscribed upstream for this market
        self._active_tickers: set[str] = set()

    async def ensure_started(self):
        """Start the connection loop if not already running."""
        if self._auth_failed_permanent:
            _log(f"[{self.label}] Skipping – plan doesn't include {self.label} websocket")
            return
        self._running = True
        if self._task is None or self._task.done():
            _log(f"[{self.label}] Starting upstream WebSocket…")
            self._auth_event = asyncio.Event()
            self._task = asyncio.create_task(self._connection_loop())

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        if self.ws:
            await self.ws.close()
        self.ws = None
        self.authenticated = False
        _log(f"[{self.label}] Upstream stopped")

    # ── subscribe / unsubscribe on the wire ──────────────────

    async def subscribe(self, ticker: str):
        self._active_tickers.add(ticker)
        if self.ws and self.authenticated:
            msg = json.dumps({"action": "subscribe", "params": f"A.{ticker}"})
            try:
                await self.ws.send(msg)
                _log(f"[{self.label}] Upstream subscribe: A.{ticker}")
            except Exception as e:
                _log(f"[{self.label}] Failed to subscribe {ticker}: {e}")
        else:
            _log(f"[{self.label}] ⚠ queued subscribe for {ticker} (not yet connected)")

    async def unsubscribe(self, ticker: str):
        self._active_tickers.discard(ticker)
        if self.ws and self.authenticated:
            msg = json.dumps({"action": "unsubscribe", "params": f"A.{ticker}"})
            try:
                await self.ws.send(msg)
                _log(f"[{self.label}] Upstream unsubscribe: A.{ticker}")
            except Exception as e:
                _log(f"[{self.label}] Failed to unsubscribe {ticker}: {e}")

    # ── internal connection loop ─────────────────────────────

    async def _connection_loop(self):
        while self._running and not self._auth_failed_permanent:
            # Idle check – no active tickers → sleep
            if not self._active_tickers:
                await asyncio.sleep(1)
                continue

            try:
                _log(f"[{self.label}] Connecting to {self.url}")
                async with websockets.connect(self.url, ping_interval=20, ping_timeout=10) as ws:
                    self.ws = ws
                    self._reconnect_delay = 1

                    await self._authenticate(ws)

                    # Wait for auth response before subscribing
                    # The _handle_raw will set the auth_event
                    try:
                        await asyncio.wait_for(self._auth_event.wait(), timeout=5.0)
                    except asyncio.TimeoutError:
                        _log(f"[{self.label}] Auth response timeout")

                    if self._auth_failed_permanent:
                        _log(f"[{self.label}] Plan doesn't support this market – stopping")
                        break

                    if self.authenticated:
                        await self._resubscribe_all(ws)

                    async for raw_msg in ws:
                        await self._handle_raw(raw_msg)

            except (websockets.exceptions.ConnectionClosed,
                    websockets.exceptions.ConnectionClosedError,
                    ConnectionRefusedError, OSError) as e:
                _log(f"[{self.label}] Disconnected: {e}")
            except asyncio.CancelledError:
                break
            except Exception as e:
                _log(f"[{self.label}] Error: {e}")

            self.ws = None
            self.authenticated = False

            if self._running and not self._auth_failed_permanent:
                _log(f"[{self.label}] Reconnecting in {self._reconnect_delay}s…")
                await asyncio.sleep(self._reconnect_delay)
                self._reconnect_delay = min(self._reconnect_delay * 2, 30)
                self._auth_event = asyncio.Event()  # reset for next attempt

    async def _authenticate(self, ws):
        api_key = settings.POLYGON_API_KEY
        masked = api_key[:4] + '…' + api_key[-4:] if len(api_key) > 8 else '***'
        _log(f"[{self.label}] Sending auth (key: {masked})")
        await ws.send(json.dumps({"action": "auth", "params": api_key}))

    async def _resubscribe_all(self, ws):
        if self._active_tickers:
            params = ",".join(f"A.{t}" for t in self._active_tickers)
            await ws.send(json.dumps({"action": "subscribe", "params": params}))
            _log(f"[{self.label}] Re-subscribed {len(self._active_tickers)} tickers")

    async def _handle_raw(self, raw_msg: str):
        try:
            messages = json.loads(raw_msg)
            if not isinstance(messages, list):
                messages = [messages]

            for msg in messages:
                ev = msg.get("ev")

                if ev == "status":
                    status = msg.get("status", "")
                    message = msg.get("message", "")
                    if status == "auth_success":
                        self.authenticated = True
                        self._auth_event.set()
                        _log(f"[{self.label}] ✅ AUTHENTICATED")
                    elif status == "auth_failed":
                        # Check if this is a permanent plan issue
                        if "plan" in message.lower() or "upgrade" in message.lower():
                            self._auth_failed_permanent = True
                            _log(f"[{self.label}] ❌ AUTH FAILED (plan): {message}")
                        else:
                            _log(f"[{self.label}] ❌ AUTH FAILED: {message}")
                        self._auth_event.set()
                    elif status == "success":
                        _log(f"[{self.label}] ✅ {message}")
                    elif status == "connected":
                        _log(f"[{self.label}] Connected to Polygon")
                    else:
                        _log(f"[{self.label}] status: {status} – {message}")

                elif ev in ("A", "AM"):
                    await self._on_message(msg)

                else:
                    _log(f"[{self.label}] Unknown event: {ev} → {json.dumps(msg)[:200]}")

        except json.JSONDecodeError:
            _log(f"[{self.label}] Non-JSON: {raw_msg[:200]}")



# ─────────────────────────────────────────────────────────────
# Massive-based upstream for stocks
# ─────────────────────────────────────────────────────────────
class _MassiveUpstreamWS:
    """Encapsulates an upstream connection via the massive Python client."""

    def __init__(self, label: str, on_message):
        self.label = label
        self._on_message = on_message   # async callback(msg_dict)
        self.authenticated = False
        self._auth_failed_permanent = False
        self._task: Optional[asyncio.Task] = None
        self._running = False
        self._active_tickers: set[str] = set()
        self._client: Optional[WebSocketClient] = None
        # Expose ws-like attribute for get_status compat
        self.ws = None

    async def ensure_started(self):
        """Start the massive WS loop if not already running."""
        if self._auth_failed_permanent:
            _log(f"[{self.label}] Skipping – auth previously failed")
            return
        self._running = True
        if self._task is None or self._task.done():
            _log(f"[{self.label}] Starting upstream via massive client…")
            self._task = asyncio.create_task(self._connection_loop())

    async def stop(self):
        self._running = False
        if self._client:
            try:
                self._client.close()
            except Exception:
                pass
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._client = None
        self.ws = None
        self.authenticated = False
        _log(f"[{self.label}] Upstream stopped")

    async def subscribe(self, ticker: str):
        self._active_tickers.add(ticker)
        if self._client and self.authenticated:
            try:
                self._client.subscribe(f"A.{ticker}")
                _log(f"[{self.label}] Upstream subscribe: A.{ticker}")
            except Exception as e:
                _log(f"[{self.label}] Failed to subscribe {ticker}: {e}")
        else:
            _log(f"[{self.label}] ⚠ queued subscribe for {ticker} (not yet connected)")

    async def unsubscribe(self, ticker: str):
        self._active_tickers.discard(ticker)
        if self._client and self.authenticated:
            try:
                self._client.unsubscribe(f"A.{ticker}")
                _log(f"[{self.label}] Upstream unsubscribe: A.{ticker}")
            except Exception as e:
                _log(f"[{self.label}] Failed to unsubscribe {ticker}: {e}")

    async def _connection_loop(self):
        """Run the massive WebSocketClient.connect() loop."""
        while self._running and not self._auth_failed_permanent:
            if not self._active_tickers:
                await asyncio.sleep(1)
                continue

            try:
                api_key = settings.POLYGON_API_KEY
                masked = api_key[:4] + '…' + api_key[-4:] if len(api_key) > 8 else '***'
                _log(f"[{self.label}] Connecting via massive (key: {masked}, feed=Delayed, market=Stocks)")

                # Build initial subscriptions from active tickers
                initial_subs = [f"A.{t}" for t in self._active_tickers]

                self._client = WebSocketClient(
                    api_key=api_key,
                    feed=Feed.Delayed,
                    market=Market.Stocks,
                    subscriptions=initial_subs,
                    max_reconnects=5,
                )

                self.authenticated = True
                self.ws = True   # truthy sentinel for get_status
                _log(f"[{self.label}] ✅ CONNECTED (massive) – subscribed {', '.join(self._active_tickers)}")

                async def _processor(msgs):
                    for m in msgs:
                        # Convert EquityAgg to raw dict matching Polygon format
                        msg_dict = {
                            "ev": str(m.event_type) if m.event_type else "A",
                            "sym": m.symbol or "",
                            "v": m.volume or 0,
                            "av": m.accumulated_volume or 0,
                            "op": m.official_open_price or 0,
                            "vw": m.aggregate_vwap or m.vwap or 0,
                            "o": m.open or 0,
                            "c": m.close or 0,
                            "h": m.high or 0,
                            "l": m.low or 0,
                            "a": m.average_size or 0,
                            "s": m.start_timestamp or 0,
                            "e": m.end_timestamp or 0,
                        }
                        await self._on_message(msg_dict)

                await self._client.connect(_processor)

            except Exception as e:
                err_msg = str(e).lower()
                if "auth" in err_msg or "plan" in err_msg or "upgrade" in err_msg:
                    self._auth_failed_permanent = True
                    _log(f"[{self.label}] ❌ AUTH FAILED (permanent): {e}")
                    break
                _log(f"[{self.label}] Error: {e}")
            finally:
                self.ws = None
                self.authenticated = False
                self._client = None

            if self._running and not self._auth_failed_permanent:
                _log(f"[{self.label}] Reconnecting in 5s…")
                await asyncio.sleep(5)


# ─────────────────────────────────────────────────────────────
# Per-client mock CSV playback
# ─────────────────────────────────────────────────────────────
_DATA_DIR = Path(__file__).parent.parent / "data"


def _base_symbol(ticker: str) -> str:
    """Extract underlying symbol: 'QQQ' → 'qqq', 'O:QQQ260115C00620000' → 'qqq'."""
    import re
    if ticker.startswith("O:"):
        m = re.match(r'^([A-Z]+)', ticker[2:])
        return m.group(1).lower() if m else ticker[2:].lower()
    return ticker.lower()


def _load_csv_bars(symbol_lower: str) -> list[dict]:
    """Load all CSV files for a symbol from data/<symbol>/, sorted by date.
    Falls back to 'qqq' data if no data exists for the requested symbol."""
    symbol_dir = _DATA_DIR / symbol_lower
    if not symbol_dir.is_dir():
        # Fallback to QQQ data (the only guaranteed mock dataset)
        fallback = _DATA_DIR / "qqq"
        if symbol_lower != "qqq" and fallback.is_dir():
            _log(f"[MockPlayback] No CSV data for '{symbol_lower}', falling back to 'qqq'")
            symbol_dir = fallback
        else:
            _log(f"[MockPlayback] No CSV data directory for '{symbol_lower}' at {symbol_dir}")
            return []

    files = sorted(f for f in os.listdir(symbol_dir) if f.endswith(".csv"))
    bars: list[dict] = []
    for fname in files:
        with open(symbol_dir / fname, newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                bars.append(row)

    _log(f"[MockPlayback] Loaded {len(bars)} bars from {len(files)} CSV files for '{symbol_lower}'")
    return bars


def _detect_bar_signals(history: list[dict], bar: dict) -> list[dict]:
    """
    Lightweight VPA signal detection on the latest bar.
    Uses the provided history for volume-ratio calculation.
    """
    if len(history) < 5:
        return []

    recent_vols = [b["volume"] for b in history[-20:] if b["volume"] > 0]
    if not recent_vols:
        return []
    avg_vol = sum(recent_vols) / len(recent_vols)
    if avg_vol == 0:
        return []

    vol_ratio = bar["volume"] / avg_vol
    price_change = bar["close"] - bar["open"]
    bar_range = bar["high"] - bar["low"]

    signals = []

    if vol_ratio >= 2.0 and bar_range > 0:
        body_ratio = abs(price_change) / bar_range if bar_range > 0 else 0

        if price_change > 0 and body_ratio > 0.6:
            signals.append({
                "signal": "strong_bullish",
                "confidence": min(vol_ratio / 5.0, 1.0),
                "description": f"Strong bullish: vol {vol_ratio:.1f}x avg, solid green bar",
                "datetime": bar["datetime"],
                "price": bar["close"],
                "volume": bar["volume"],
                "volume_ratio": round(vol_ratio, 2),
            })
        elif price_change < 0 and body_ratio > 0.6:
            signals.append({
                "signal": "strong_bearish",
                "confidence": min(vol_ratio / 5.0, 1.0),
                "description": f"Strong bearish: vol {vol_ratio:.1f}x avg, solid red bar",
                "datetime": bar["datetime"],
                "price": bar["close"],
                "volume": bar["volume"],
                "volume_ratio": round(vol_ratio, 2),
            })

        if vol_ratio >= 3.0:
            upper_wick = bar["high"] - max(bar["open"], bar["close"])
            lower_wick = min(bar["open"], bar["close"]) - bar["low"]

            if upper_wick > bar_range * 0.5 and price_change < 0:
                signals.append({
                    "signal": "climax_top",
                    "confidence": min(vol_ratio / 6.0, 1.0),
                    "description": f"Climax top: vol {vol_ratio:.1f}x avg, rejection wick",
                    "datetime": bar["datetime"],
                    "price": bar["close"],
                    "volume": bar["volume"],
                    "volume_ratio": round(vol_ratio, 2),
                })
            elif lower_wick > bar_range * 0.5 and price_change > 0:
                signals.append({
                    "signal": "climax_bottom",
                    "confidence": min(vol_ratio / 6.0, 1.0),
                    "description": f"Climax bottom: vol {vol_ratio:.1f}x avg, bounce wick",
                    "datetime": bar["datetime"],
                    "price": bar["close"],
                    "volume": bar["volume"],
                    "volume_ratio": round(vol_ratio, 2),
                })

    elif vol_ratio < 0.5 and abs(price_change) > 0:
        if price_change > 0:
            signals.append({
                "signal": "weak_up",
                "confidence": 0.3,
                "description": f"Weak up move: vol only {vol_ratio:.1f}x avg",
                "datetime": bar["datetime"],
                "price": bar["close"],
                "volume": bar["volume"],
                "volume_ratio": round(vol_ratio, 2),
            })
        else:
            signals.append({
                "signal": "weak_down",
                "confidence": 0.3,
                "description": f"Weak down move: vol only {vol_ratio:.1f}x avg",
                "datetime": bar["datetime"],
                "price": bar["close"],
                "volume": bar["volume"],
                "volume_ratio": round(vol_ratio, 2),
            })

    return signals


def _floor_dt(dt: datetime, interval_sec: int) -> datetime:
    """Floor a datetime to the nearest interval boundary."""
    ts = int(dt.timestamp())
    floored = ts - (ts % interval_sec)
    return datetime.fromtimestamp(floored)


# ── Option-like price scaling ──
_OPTION_BASE_PRICE = 5.0     # Simulated option centre price
_OPTION_LEVERAGE   = 10      # Amplify stock % moves to look like option moves

def _scale_to_option(raw_price: float, base_stock_price: float) -> float:
    """Convert a stock price into a simulated option price.

    Maps the stock's percentage change from its baseline into a leveraged
    change centred around _OPTION_BASE_PRICE.  A 0.1 % move in the stock
    becomes a 1 % move in the simulated option (at 10x leverage).
    """
    pct = (raw_price / base_stock_price) - 1.0
    return round(_OPTION_BASE_PRICE * (1 + pct * _OPTION_LEVERAGE), 4)


def build_mock_sow(ticker: str, interval_minutes: int = 5, num_candles: int = 50) -> tuple[list[dict], int]:
    """
    Build a snapshot-of-world from CSV data: *num_candles* pre-built historical
    candles whose timestamps count backwards from *now*, so the chart already has
    price history when mock live mode starts.

    Returns (sow_bars, start_index) where start_index is the CSV row offset
    at which mock_csv_playback should continue after the SOW data.
    """
    base_sym = _base_symbol(ticker)
    raw_bars = _load_csv_bars(base_sym)
    if not raw_bars:
        return [], 0

    is_option = ticker.startswith("O:")
    base_stock_price = float(raw_bars[0]["close"]) if is_option and float(raw_bars[0]["close"]) > 0 else 0

    interval_sec = max(interval_minutes, 1) * 60

    # Determine how many raw rows go into each aggregated candle.
    # Each raw row is 1-minute data, so rows_per_candle = interval_minutes.
    rows_per_candle = max(interval_minutes, 1)

    # We need enough raw rows: num_candles * rows_per_candle
    rows_needed = num_candles * rows_per_candle
    if rows_needed > len(raw_bars):
        rows_needed = len(raw_bars)
        num_candles = rows_needed // rows_per_candle

    if num_candles == 0:
        return [], 0

    # Slice the raw data (take the first rows_needed rows)
    raw_slice = raw_bars[:rows_needed]

    # Build aggregated candles with timestamps counting BACK from (now - 1 interval)
    now_utc = datetime.now(tz=ZoneInfo("UTC"))
    now_et = now_utc.astimezone(_ET).replace(tzinfo=None)
    # The most recent SOW candle ends one interval before "now" (so live bars start at "now")
    latest_candle_end = _floor_dt(now_et, interval_sec)

    sow_bars: list[dict] = []
    for ci in range(num_candles):
        chunk_start = ci * rows_per_candle
        chunk = raw_slice[chunk_start:chunk_start + rows_per_candle]

        o_vals = [float(r["open"]) for r in chunk]
        c_vals = [float(r["close"]) for r in chunk]
        h_vals = [float(r["high"]) for r in chunk]
        l_vals = [float(r["low"]) for r in chunk]
        v_vals = [int(r["volume"]) for r in chunk]

        o = o_vals[0]
        c = c_vals[-1]
        h = max(h_vals)
        lo = min(l_vals)
        v = sum(v_vals)

        if is_option and base_stock_price > 0:
            o  = _scale_to_option(o, base_stock_price)
            c  = _scale_to_option(c, base_stock_price)
            h  = _scale_to_option(h, base_stock_price)
            lo = _scale_to_option(lo, base_stock_price)

        # Timestamp: count back from latest_candle_end
        candle_offset = num_candles - 1 - ci  # 0 = most recent, num_candles-1 = oldest
        candle_dt = latest_candle_end - timedelta(seconds=candle_offset * interval_sec)
        dt_str = candle_dt.strftime("%Y-%m-%d %H:%M:%S")

        sow_bars.append({
            "datetime": dt_str,
            "open": o,
            "high": h,
            "low": lo,
            "close": c,
            "volume": v,
            "vwap": round((h + lo) / 2, 4),
            "accumulated_volume": 0,
        })

    _log(f"[MockSOW] Built {len(sow_bars)} candles for {ticker} (interval={interval_minutes}m)")
    return sow_bars, rows_needed  # start_index for continuing playback


async def mock_csv_playback(ticker: str, queue: asyncio.Queue, interval_minutes: int = 5,
                            start_index: int = 0):
    """
    Per-client CSV playback coroutine.
    Loads historical CSV data for the ticker's underlying symbol and pushes
    aggregated candles into the provided asyncio.Queue, formatted identically
    to real-feed bar payloads.

    Ticks arrive every 1s but are aggregated into the selected interval
    (e.g. 1m, 5m, 15m).  Each tick updates the current candle's OHLCV.
    A new candle starts when the interval window advances.

    *start_index* lets the playback resume after SOW rows that were already
    consumed by build_mock_sow().
    """
    base_sym = _base_symbol(ticker)
    bars = _load_csv_bars(base_sym)
    if not bars:
        return

    interval_sec = max(interval_minutes, 1) * 60
    _log(f"[MockPlayback] interval={interval_minutes}m ({interval_sec}s) for {ticker}, start_index={start_index}")

    is_option = ticker.startswith("O:")
    base_stock_price = float(bars[0]["close"]) if is_option and float(bars[0]["close"]) > 0 else 0

    history: list[dict] = []
    idx = start_index % len(bars)  # Resume from where SOW left off

    # Current candle accumulator
    candle_dt: str | None = None     # floored datetime string for current candle
    candle_open = 0.0
    candle_high = 0.0
    candle_low = 0.0
    candle_close = 0.0
    candle_volume = 0

    try:
        while True:
            row = bars[idx]

            o = float(row["open"])
            c = float(row["close"])
            h = float(row["high"])
            lo = float(row["low"])
            v = int(row["volume"])

            if is_option and base_stock_price > 0:
                o  = _scale_to_option(o, base_stock_price)
                c  = _scale_to_option(c, base_stock_price)
                h  = _scale_to_option(h, base_stock_price)
                lo = _scale_to_option(lo, base_stock_price)

            # Floor current time to the interval boundary
            now_utc = datetime.now(tz=ZoneInfo("UTC"))
            now_et = now_utc.astimezone(_ET).replace(tzinfo=None)
            floored = _floor_dt(now_et, interval_sec)
            floored_str = floored.strftime("%Y-%m-%d %H:%M:%S")

            if floored_str != candle_dt:
                # New candle window — start fresh
                candle_dt = floored_str
                candle_open = o
                candle_high = h
                candle_low = lo
                candle_close = c
                candle_volume = v
            else:
                # Same candle window — aggregate
                candle_high = max(candle_high, h)
                candle_low = min(candle_low, lo)
                candle_close = c
                candle_volume += v

            bar = {
                "datetime": candle_dt,
                "open": candle_open,
                "high": candle_high,
                "low": candle_low,
                "close": candle_close,
                "volume": candle_volume,
                "vwap": round((candle_high + candle_low) / 2, 4),
                "accumulated_volume": 0,
            }

            # Only add to history when a new candle starts (for signal detection)
            if len(history) == 0 or history[-1]["datetime"] != candle_dt:
                history.append(bar)
                if len(history) > 100:
                    history = history[-100:]
            else:
                history[-1] = bar  # update current candle in history

            signals = _detect_bar_signals(history, bar)

            payload = json.dumps({
                "type": "bar",
                "ticker": ticker,
                "bar": bar,
                "signals": signals,
            })

            try:
                queue.put_nowait(payload)
            except asyncio.QueueFull:
                try:
                    queue.get_nowait()
                    queue.put_nowait(payload)
                except Exception:
                    pass

            idx = (idx + 1) % len(bars)
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        _log(f"[MockPlayback] Stopped playback for {ticker}")
        return
    except Exception as e:
        _log(f"[MockPlayback] ERROR for {ticker}: {e}")
        import traceback
        traceback.print_exc()


# ─────────────────────────────────────────────────────────────
# LiveFeedManager – public API
# ─────────────────────────────────────────────────────────────
class LiveFeedManager:
    """
    Manages TWO upstream Polygon/Massive WebSockets (stocks + options)
    and downstream client fan-out with ref-counted subscriptions.
    """

    def __init__(self):
        self._running = False

        # Two upstream connections (lazy – created on first subscribe)
        self._options_upstream = _UpstreamWS(
            "options", OPTIONS_WS_DELAYED, self._handle_aggregate
        )
        self._stocks_upstream = _MassiveUpstreamWS(
            "stocks", self._handle_aggregate
        )

        # downstream clients: ticker → set of asyncio.Queue
        self._subscribers: dict[str, set[asyncio.Queue]] = defaultdict(set)

        # grace-period timers: ticker → asyncio.Task
        self._unsub_timers: dict[str, asyncio.Task] = {}

        # accumulated bars per ticker (keep last 100)
        self._bar_history: dict[str, list[dict]] = defaultdict(list)

    def _upstream_for(self, ticker: str):
        return self._options_upstream if _is_option_ticker(ticker) else self._stocks_upstream

    # ── Lifecycle ────────────────────────────────────────────

    async def start(self):
        if self._running:
            return
        self._running = True
        _log("LiveFeedManager ready (dual WS: options + stocks, lazy connect)")

    async def stop(self):
        self._running = False
        await self._options_upstream.stop()
        await self._stocks_upstream.stop()
        for t in self._unsub_timers.values():
            t.cancel()
        self._unsub_timers.clear()
        _log("LiveFeedManager stopped")

    # ── Aggregate handler (called from BOTH upstream connections) ──

    async def _handle_aggregate(self, msg: dict):
        """Process an aggregate bar from either upstream and fan out."""
        sym = msg.get("sym", "")
        if not sym:
            return

        market = _market_label(sym)

        # Log first bar per ticker, then every 10th
        hist_len = len(self._bar_history.get(sym, []))
        if hist_len == 0 or hist_len % 10 == 0:
            _log(f"[{market}] bar #{hist_len+1} {sym}: c={msg.get('c')} v={msg.get('v')} subs={len(self._subscribers.get(sym, set()))}")

        # Convert Polygon agg → our bar format
        start_ms = msg.get("s", 0)
        dt_utc = datetime.fromtimestamp(start_ms / 1000, tz=ZoneInfo("UTC"))
        dt_et = dt_utc.astimezone(_ET).replace(tzinfo=None)

        bar = {
            "datetime": dt_et.strftime("%Y-%m-%d %H:%M:%S"),
            "open": msg.get("o", 0),
            "high": msg.get("h", 0),
            "low": msg.get("l", 0),
            "close": msg.get("c", 0),
            "volume": msg.get("v", 0),
            "vwap": msg.get("vw", 0),
            "accumulated_volume": msg.get("av", 0),
        }

        # Accumulate for VPA analysis
        history = self._bar_history[sym]
        history.append(bar)
        if len(history) > 100:
            self._bar_history[sym] = history[-100:]

        # Run lightweight signal detection on latest bar
        signals = self._detect_signals(sym, bar)

        # Build the message to send downstream
        payload = json.dumps({
            "type": "bar",
            "ticker": sym,
            "bar": bar,
            "signals": signals,
        })

        # Fan out to all subscribers for this ticker
        subscribers = self._subscribers.get(sym, set())
        dead_queues = []
        for q in subscribers:
            try:
                q.put_nowait(payload)
            except asyncio.QueueFull:
                # Drop message if client is too slow
                try:
                    q.get_nowait()  # drop oldest
                    q.put_nowait(payload)
                except Exception:
                    dead_queues.append(q)

        # Clean up dead queues
        for q in dead_queues:
            subscribers.discard(q)

    def _detect_signals(self, sym: str, bar: dict) -> list[dict]:
        """
        Lightweight VPA signal detection on the latest bar.
        Delegates to module-level _detect_bar_signals.
        """
        history = self._bar_history.get(sym, [])
        return _detect_bar_signals(history, bar)

    # ── Client subscription management ───────────────────────

    async def subscribe(self, ticker: str, queue: asyncio.Queue) -> list[dict]:
        """
        Subscribe a client queue to a ticker (option or stock).
        Returns the current bar history (SOW) for initial chart draw.
        """
        # Cancel any pending unsubscribe timer
        timer = self._unsub_timers.pop(ticker, None)
        if timer:
            timer.cancel()
            _log(f"Cancelled unsub timer for {ticker}")

        self._subscribers[ticker].add(queue)
        count = len(self._subscribers[ticker])
        _log(f"Subscribe {ticker} ({_market_label(ticker)}) – now {count} client(s)")

        # First subscriber → ensure the correct upstream is running, then subscribe
        if count == 1:
            upstream = self._upstream_for(ticker)
            if upstream._auth_failed_permanent:
                _log(f"Skipping upstream subscribe for {ticker} – plan doesn't support {upstream.label} WS")
            else:
                # Add ticker BEFORE starting connection loop so the idle-check
                # doesn't block the initial connection attempt.
                upstream._active_tickers.add(ticker)
                await upstream.ensure_started()
                # Wait for auth (up to 5s)
                for _ in range(50):
                    if upstream.authenticated or upstream._auth_failed_permanent:
                        break
                    await asyncio.sleep(0.1)
                if upstream.authenticated:
                    # subscribe() re-adds to _active_tickers (set, no-op)
                    # and sends the WS subscribe command
                    await upstream.subscribe(ticker)

        return list(self._bar_history.get(ticker, []))

    async def unsubscribe(self, ticker: str, queue: asyncio.Queue):
        """
        Unsubscribe a client queue from a ticker.
        Starts a 5-second grace timer; if no new subscribers, unsubscribes upstream.
        """
        subs = self._subscribers.get(ticker, set())
        subs.discard(queue)
        count = len(subs)
        _log(f"Unsubscribe {ticker} ({_market_label(ticker)}) – now {count} client(s)")

        if count == 0:
            self._unsub_timers[ticker] = asyncio.create_task(
                self._grace_unsubscribe(ticker)
            )

    async def _grace_unsubscribe(self, ticker: str):
        """Wait UNSUBSCRIBE_GRACE seconds, then unsubscribe upstream if still 0 clients."""
        try:
            await asyncio.sleep(UNSUBSCRIBE_GRACE)
            if len(self._subscribers.get(ticker, set())) == 0:
                await self._upstream_for(ticker).unsubscribe(ticker)
                self._bar_history.pop(ticker, None)
                self._subscribers.pop(ticker, None)
                _log(f"Grace period expired – unsubscribed {ticker} upstream")
        except asyncio.CancelledError:
            pass

    # ── Status ───────────────────────────────────────────────

    def get_status(self) -> dict:
        """Return current feed manager status."""
        opts = self._options_upstream
        stks = self._stocks_upstream
        return {
            "options_connected": opts.ws is not None and opts.authenticated,
            "options_plan_ok": not opts._auth_failed_permanent,
            "stocks_connected": stks.ws is not None and stks.authenticated,
            "stocks_plan_ok": not stks._auth_failed_permanent,
            "subscriptions": {
                ticker: len(subs) for ticker, subs in self._subscribers.items() if subs
            },
            "bar_history_sizes": {
                ticker: len(bars) for ticker, bars in self._bar_history.items()
            },
        }


# Singleton instance
live_feed_manager = LiveFeedManager()
