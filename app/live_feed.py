"""
Live WebSocket Feed Manager
============================
Manages TWO upstream WebSockets:
  1. Options  – raw websockets to wss://socket.polygon.io/options (realtime, for O:* tickers)
  2. Stocks   – massive Python client via RealTime feed (for stock tickers)

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

# Backoff when Polygon says "max_connections" (seconds)
# Polygon needs time to release the old socket on their side
MAX_CONN_BACKOFF = 30


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
        self._conn_limit_hit = False  # True when Polygon says max_connections
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
                        self._conn_limit_hit = False  # clear on successful auth
                        await self._resubscribe_all(ws)

                    async for raw_msg in ws:
                        await self._handle_raw(raw_msg)

            except (websockets.exceptions.ConnectionClosed,
                    websockets.exceptions.ConnectionClosedError,
                    ConnectionRefusedError, OSError) as e:
                _log(f"[{self.label}] Disconnected: {e}")
                # Check if this was a policy violation (often follows max_connections)
                if "1008" in str(e) or "policy" in str(e).lower():
                    self._conn_limit_hit = True
                    self._reconnect_delay = max(self._reconnect_delay, MAX_CONN_BACKOFF)
            except asyncio.CancelledError:
                break
            except Exception as e:
                _log(f"[{self.label}] Error: {e}")

            # Ensure the WS is fully closed before attempting reconnect
            if self.ws:
                try:
                    await self.ws.close()
                except Exception:
                    pass
            self.ws = None
            self.authenticated = False

            if self._running and not self._auth_failed_permanent:
                _log(f"[{self.label}] Reconnecting in {self._reconnect_delay}s…")
                await asyncio.sleep(self._reconnect_delay)
                self._reconnect_delay = min(self._reconnect_delay * 2, 60)
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
                    elif status == "max_connections":
                        # Polygon says we have too many WS connections.
                        # Force a long backoff so the stale socket can expire.
                        self._conn_limit_hit = True
                        _log(f"[{self.label}] ⚠ MAX CONNECTIONS – will back off {MAX_CONN_BACKOFF}s: {message}")
                        self._reconnect_delay = MAX_CONN_BACKOFF
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

    def __init__(self, label: str, on_message, feed: Feed = Feed.Delayed, market: Market = Market.Stocks):
        self.label = label
        self._on_message = on_message   # async callback(msg_dict)
        self._feed = feed
        self._market = market
        self._is_launchpad = (feed == Feed.Launchpad)
        self.authenticated = False
        self._auth_failed_permanent = False
        self._conn_limit_hit = False  # True when connection limit exceeded
        self._task: Optional[asyncio.Task] = None
        self._running = False
        self._active_tickers: set[str] = set()
        self._client: Optional[WebSocketClient] = None
        # Expose ws-like attribute for get_status compat
        self.ws = None
        # Track bars received (for RealTime→Delayed fallback)
        self._bar_count = 0
        # Per-ticker aggregation for LaunchpadValue ticks (no OHLCV)
        # Builds per-second OHLCV candles from single-value FMV ticks.
        self._lv_candles: dict[str, dict] = {}  # {sym: {sec, o, h, l, c}}

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
                await self._client.close()
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
                if self._is_launchpad:
                    self._client.subscribe(f"LV.{ticker}")
                    _log(f"[{self.label}] Upstream subscribe: A.{ticker} + LV.{ticker}")
                else:
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
                if self._is_launchpad:
                    self._client.unsubscribe(f"LV.{ticker}")
                    _log(f"[{self.label}] Upstream unsubscribe: A.{ticker} + LV.{ticker}")
                else:
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
                _log(f"[{self.label}] Connecting via massive (key: {masked}, feed={self._feed.name}, market={self._market.name})")

                # Build initial subscriptions from active tickers
                # Always subscribe to A.* (per-second aggregates)
                initial_subs = [f"A.{t}" for t in self._active_tickers]
                # On Launchpad feed, ALSO subscribe to LV.* (LaunchpadValue)
                # because Launchpad may only serve FMV data via LV.* channel
                if self._is_launchpad:
                    initial_subs += [f"LV.{t}" for t in self._active_tickers]
                    _log(f"[{self.label}] Launchpad mode: subscribing to both A.* and LV.* channels")

                self._client = WebSocketClient(
                    api_key=api_key,
                    feed=self._feed,
                    market=self._market,
                    subscriptions=initial_subs,
                    max_reconnects=5,
                )

                self.authenticated = True
                self._conn_limit_hit = False  # clear on successful connect
                self.ws = True   # truthy sentinel for get_status
                _log(f"[{self.label}] ✅ CONNECTED (massive) – subscribed {', '.join(self._active_tickers)}")

                # ── Message processor ──
                # Handles EquityAgg (full OHLCV), LaunchpadValue (FMV),
                # and FairMarketValue with per-message error handling.
                _lv_candles = self._lv_candles  # local ref for closure

                async def _processor(msgs):
                    for m in msgs:
                        try:
                            msg_dict = None

                            # ── LaunchpadValue: only has .value (Fair Market Value) ──
                            if hasattr(m, 'value') and m.value is not None and not hasattr(m, 'open'):
                                val = float(m.value)
                                sym = getattr(m, 'symbol', '') or ''
                                ts = getattr(m, 'timestamp', 0) or 0
                                # Normalize timestamp to milliseconds
                                if ts > 1e16:      # nanoseconds
                                    ts = int(ts // 1_000_000)
                                elif ts > 1e13:    # microseconds
                                    ts = int(ts // 1_000)
                                else:
                                    ts = int(ts)

                                # Aggregate into per-second OHLCV candle
                                sec = ts // 1000
                                candle = _lv_candles.get(sym)
                                if candle is None or candle["sec"] != sec:
                                    _lv_candles[sym] = {"sec": sec, "o": val, "h": val, "l": val, "c": val}
                                else:
                                    candle["h"] = max(candle["h"], val)
                                    candle["l"] = min(candle["l"], val)
                                    candle["c"] = val

                                c = _lv_candles[sym]
                                msg_dict = {
                                    "ev": "A", "sym": sym,
                                    "o": c["o"], "h": c["h"], "l": c["l"], "c": c["c"],
                                    "v": 1, "vw": val,
                                    "s": sec * 1000, "e": sec * 1000,
                                    "av": 0, "op": 0, "a": 0,
                                }

                            # ── FairMarketValue: only has .fmv ──
                            elif hasattr(m, 'fmv') and m.fmv is not None:
                                val = float(m.fmv)
                                sym = getattr(m, 'ticker', '') or getattr(m, 'symbol', '') or ''
                                ts = getattr(m, 'timestamp', 0) or 0
                                if ts > 1e16:
                                    ts = int(ts // 1_000_000)
                                elif ts > 1e13:
                                    ts = int(ts // 1_000)
                                else:
                                    ts = int(ts)

                                sec = ts // 1000
                                candle = _lv_candles.get(sym)
                                if candle is None or candle["sec"] != sec:
                                    _lv_candles[sym] = {"sec": sec, "o": val, "h": val, "l": val, "c": val}
                                else:
                                    candle["h"] = max(candle["h"], val)
                                    candle["l"] = min(candle["l"], val)
                                    candle["c"] = val

                                c = _lv_candles[sym]
                                msg_dict = {
                                    "ev": "A", "sym": sym,
                                    "o": c["o"], "h": c["h"], "l": c["l"], "c": c["c"],
                                    "v": 1, "vw": val,
                                    "s": sec * 1000, "e": sec * 1000,
                                    "av": 0, "op": 0, "a": 0,
                                }

                            # ── EquityAgg: full OHLCV aggregate data ──
                            else:
                                msg_dict = {
                                    "ev": str(getattr(m, 'event_type', 'A') or 'A'),
                                    "sym": getattr(m, 'symbol', '') or '',
                                    "v": getattr(m, 'volume', 0) or 0,
                                    "av": getattr(m, 'accumulated_volume', 0) or 0,
                                    "op": getattr(m, 'official_open_price', 0) or 0,
                                    "vw": getattr(m, 'aggregate_vwap', None) or getattr(m, 'vwap', 0) or 0,
                                    "o": getattr(m, 'open', 0) or 0,
                                    "c": getattr(m, 'close', 0) or 0,
                                    "h": getattr(m, 'high', 0) or 0,
                                    "l": getattr(m, 'low', 0) or 0,
                                    "a": getattr(m, 'average_size', 0) or 0,
                                    "s": getattr(m, 'start_timestamp', 0) or 0,
                                    "e": getattr(m, 'end_timestamp', 0) or 0,
                                }

                            if msg_dict:
                                self._bar_count += 1
                                await self._on_message(msg_dict)

                        except Exception as proc_err:
                            _log(f"[{self.label}] Processor error for {type(m).__name__}: {proc_err}")

                await self._client.connect(_processor)

            except Exception as e:
                err_msg = str(e).lower()
                if "auth" in err_msg or "plan" in err_msg or "upgrade" in err_msg:
                    self._auth_failed_permanent = True
                    _log(f"[{self.label}] ❌ AUTH FAILED (permanent): {e}")
                    break
                # Detect connection-limit / policy-violation errors
                if "max_connection" in err_msg or "1008" in err_msg or "policy" in err_msg:
                    self._conn_limit_hit = True
                    backoff = MAX_CONN_BACKOFF
                    _log(f"[{self.label}] ⚠ Connection limit error – backing off {backoff}s: {e}")
                else:
                    backoff = 5
                    _log(f"[{self.label}] Error: {e}")
            finally:
                if self._client:
                    try:
                        await self._client.close()
                    except Exception:
                        pass
                self.ws = None
                self.authenticated = False
                self._client = None

            if self._running and not self._auth_failed_permanent:
                _log(f"[{self.label}] Reconnecting in {backoff}s…")
                await asyncio.sleep(backoff)


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


def _csv_row_to_msg_dict(row: dict, ticker: str | None = None) -> dict:
    """Convert a CSV row to Polygon-compatible aggregate message dict.

    CSV columns: ticker, volume, open, close, high, low, window_start, transactions
    Returns the same dict format as Polygon WebSocket A.* messages:
      {ev, sym, v, av, op, vw, o, c, h, l, a, s, e}

    This ensures CSV mock data flows through the same processing
    pipeline as live WebSocket data from Polygon.
    """
    ws_ns = int(row["window_start"])    # nanoseconds in CSV
    start_ms = ws_ns // 1_000_000       # Polygon uses milliseconds

    o = float(row["open"])
    c = float(row["close"])
    h = float(row["high"])
    l = float(row["low"])
    v = int(row["volume"])

    return {
        "ev": "A",
        "sym": ticker or row.get("ticker", ""),
        "v": v,
        "av": 0,
        "op": 0,
        "vw": round((h + l) / 2, 4),
        "o": o,
        "c": c,
        "h": h,
        "l": l,
        "a": int(row.get("transactions", 0)),
        "s": start_ms,
        "e": start_ms + 60_000,  # 1-minute bar
    }


def _msg_dict_to_bar(msg: dict) -> dict:
    """Convert a Polygon-format msg_dict to our internal bar format.

    This is the single authoritative conversion used by both
    live feed (_handle_aggregate) and mock CSV playback.
    Ensures bars always have the same shape regardless of source.
    """
    start_ms = msg.get("s", 0) or 0
    if start_ms == 0:
        # No timestamp provided – use current time as fallback
        start_ms = int(datetime.now(tz=ZoneInfo("UTC")).timestamp() * 1000)
    dt_utc = datetime.fromtimestamp(start_ms / 1000, tz=ZoneInfo("UTC"))
    dt_et = dt_utc.astimezone(_ET).replace(tzinfo=None)

    return {
        "datetime": dt_et.strftime("%Y-%m-%d %H:%M:%S"),
        "open": msg.get("o", 0),
        "high": msg.get("h", 0),
        "low": msg.get("l", 0),
        "close": msg.get("c", 0),
        "volume": msg.get("v", 0),
        "vwap": msg.get("vw", 0),
        "accumulated_volume": msg.get("av", 0),
    }


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

        # Convert CSV rows to Polygon-format msg_dicts for format consistency
        msgs = [_csv_row_to_msg_dict(r) for r in chunk]
        o_vals = [m["o"] for m in msgs]
        c_vals = [m["c"] for m in msgs]
        h_vals = [m["h"] for m in msgs]
        l_vals = [m["l"] for m in msgs]
        v_vals = [m["v"] for m in msgs]

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
                            start_index: int = 0,
                            sow_bars: list[dict] | None = None):
    """
    Per-client CSV playback coroutine — *time-lapse* replay.

    Replays historical CSV data at accelerated speed:
    • Each CSV row = 1 minute of real market data
    • ``rows_per_candle`` consecutive rows (e.g. 5 for 5-min interval)
      are aggregated into one candle
    • Within each candle the chart updates every second as individual
      rows are folded in (mimics a live intra-candle update)
    • A **new candle** starts every ``rows_per_candle`` seconds of
      wall-clock time, giving realistic volume and rapid signal flow

    Previous design used ``floor(now, interval)`` for candle boundaries,
    which meant a 5-minute interval needed 5 minutes of real waiting and
    crammed hundreds of CSV rows into a single candle — breaking volume
    ratios and suppressing all signal detection.

    *sow_bars* should be the list returned by ``build_mock_sow()``
    so the playback history is pre-seeded and signal detection works
    from the very first bar.
    """
    base_sym = _base_symbol(ticker)
    bars = _load_csv_bars(base_sym)
    if not bars:
        return

    rows_per_candle = max(interval_minutes, 1)
    _log(f"[MockPlayback] interval={interval_minutes}m, rows_per_candle={rows_per_candle} "
         f"for {ticker}, start_index={start_index}")

    is_option = ticker.startswith("O:")
    base_stock_price = (float(bars[0]["close"])
                        if is_option and float(bars[0]["close"]) > 0 else 0)

    # ── Seed history from SOW bars so signals fire immediately ──
    history: list[dict] = list(sow_bars) if sow_bars else []
    if len(history) > 100:
        history = history[-100:]

    idx = start_index % len(bars)

    # Running candle timestamp — advances with every completed candle.
    # Start just after the last SOW candle.
    now_utc = datetime.now(tz=ZoneInfo("UTC"))
    now_et = now_utc.astimezone(_ET).replace(tzinfo=None)
    interval_sec = max(interval_minutes, 1) * 60
    candle_base_dt = _floor_dt(now_et, interval_sec)
    candle_counter = 0  # how many candles we've started

    try:
        while True:
            # ── Build one candle from rows_per_candle CSV rows ──
            candle_dt = candle_base_dt + timedelta(seconds=candle_counter * interval_sec)
            candle_dt_str = candle_dt.strftime("%Y-%m-%d %H:%M:%S")

            candle_open = candle_high = candle_low = candle_close = 0.0
            candle_volume = 0

            for tick_i in range(rows_per_candle):
                row = bars[idx]
                msg = _csv_row_to_msg_dict(row, ticker)
                o, c, h, lo, v = msg["o"], msg["c"], msg["h"], msg["l"], msg["v"]

                if is_option and base_stock_price > 0:
                    o  = _scale_to_option(o, base_stock_price)
                    c  = _scale_to_option(c, base_stock_price)
                    h  = _scale_to_option(h, base_stock_price)
                    lo = _scale_to_option(lo, base_stock_price)

                if tick_i == 0:
                    candle_open = o
                    candle_high = h
                    candle_low = lo
                else:
                    candle_high = max(candle_high, h)
                    candle_low = min(candle_low, lo)
                candle_close = c
                candle_volume += v

                bar = {
                    "datetime": candle_dt_str,
                    "open": candle_open,
                    "high": candle_high,
                    "low": candle_low,
                    "close": candle_close,
                    "volume": candle_volume,
                    "vwap": round((candle_high + candle_low) / 2, 4),
                    "accumulated_volume": 0,
                }

                # Keep history in sync — add new, or update current candle
                if len(history) == 0 or history[-1]["datetime"] != candle_dt_str:
                    history.append(bar)
                    if len(history) > 100:
                        history = history[-100:]
                else:
                    history[-1] = bar

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
                await asyncio.sleep(1)   # 1 second per CSV row

            candle_counter += 1

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
        # Options: try raw WebSocket first (gives proper OHLCV aggregates),
        # with Massive Launchpad as fallback (gives FMV values)
        self._options_raw = _UpstreamWS(
            "options-raw", OPTIONS_WS_REALTIME, self._handle_aggregate,
        )
        self._options_launchpad = _MassiveUpstreamWS(
            "options-lp", self._handle_aggregate,
            feed=Feed.Launchpad, market=Market.Options,
        )
        # Track which options upstream is active
        self._options_upstream_active = None  # will be set on first subscribe

        # Stocks: use Massive client with RealTime feed (auto-falls back to Delayed)
        self._stocks_upstream = _MassiveUpstreamWS(
            "stocks", self._handle_aggregate,
            feed=Feed.RealTime, market=Market.Stocks,
        )
        self._stocks_fallback_task: Optional[asyncio.Task] = None

        # downstream clients: ticker → set of asyncio.Queue
        self._subscribers: dict[str, set[asyncio.Queue]] = defaultdict(set)

        # grace-period timers: ticker → asyncio.Task
        self._unsub_timers: dict[str, asyncio.Task] = {}

        # accumulated bars per ticker (keep last 100)
        self._bar_history: dict[str, list[dict]] = defaultdict(list)

    def _upstream_for(self, ticker: str):
        if _is_option_ticker(ticker):
            return self._options_upstream_active or self._options_raw
        return self._stocks_upstream

    # ── Lifecycle ────────────────────────────────────────────

    async def start(self):
        if self._running:
            return
        self._running = True
        _log("LiveFeedManager ready (dual WS: options + stocks, lazy connect)")

    async def stop(self):
        self._running = False
        if self._stocks_fallback_task:
            self._stocks_fallback_task.cancel()
        await self._options_raw.stop()
        await self._options_launchpad.stop()
        await self._stocks_upstream.stop()
        for t in self._unsub_timers.values():
            t.cancel()
        self._unsub_timers.clear()
        _log("LiveFeedManager stopped")

    async def _stocks_realtime_fallback(self):
        """If RealTime stocks upstream delivers no bars within 30s, fall back to Delayed."""
        try:
            await asyncio.sleep(30)
            if self._stocks_upstream._bar_count > 0:
                return  # RealTime is working fine
            if self._stocks_upstream._feed != Feed.RealTime:
                return  # already switched
            _log("[stocks] ⚠ No bars received after 30s on RealTime – falling back to Delayed")
            # Capture active tickers before stopping
            active = set(self._stocks_upstream._active_tickers)
            await self._stocks_upstream.stop()
            # Create a new Delayed upstream
            self._stocks_upstream = _MassiveUpstreamWS(
                "stocks", self._handle_aggregate,
                feed=Feed.Delayed, market=Market.Stocks,
            )
            # Re-subscribe to all previously active tickers
            for t in active:
                self._stocks_upstream._active_tickers.add(t)
            await self._stocks_upstream.ensure_started()
            # Wait for auth
            for _ in range(50):
                if self._stocks_upstream.authenticated or self._stocks_upstream._auth_failed_permanent:
                    break
                await asyncio.sleep(0.1)
            if self._stocks_upstream.authenticated:
                for t in active:
                    await self._stocks_upstream.subscribe(t)
                _log("[stocks] ✅ Switched to Delayed feed successfully")
            else:
                _log("[stocks] ❌ Delayed feed also failed to authenticate")
        except asyncio.CancelledError:
            pass
        except Exception as e:
            _log(f"[stocks] Fallback error: {e}")

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

        # Convert Polygon agg → our bar format (shared conversion function)
        bar = _msg_dict_to_bar(msg)

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

    async def subscribe(self, ticker: str, queue: asyncio.Queue) -> tuple[list[dict], str | None]:
        """
        Subscribe a client queue to a ticker (option or stock).
        Returns (bar_history, error_string_or_None).
        error is non-None when upstream hit connection limit or auth failed.

        For option tickers, tries the raw WebSocket (delayed OHLCV) first.
        If that auth-fails, falls back to Massive Launchpad (FMV values).
        """
        # Cancel any pending unsubscribe timer
        timer = self._unsub_timers.pop(ticker, None)
        if timer:
            timer.cancel()
            _log(f"Cancelled unsub timer for {ticker}")

        # Determine the upstream to use
        if _is_option_ticker(ticker):
            upstream = await self._resolve_options_upstream(ticker)
            if upstream is None:
                return [], "auth_failed"
        else:
            upstream = self._stocks_upstream

        # Check if upstream is in connection-limit backoff BEFORE subscribing
        if upstream._conn_limit_hit:
            _log(f"Subscribe REJECTED for {ticker} – upstream {upstream.label} in connection-limit backoff")
            return [], "conn_limit"

        if upstream._auth_failed_permanent:
            _log(f"Subscribe REJECTED for {ticker} – plan doesn't support {upstream.label} WS")
            return [], "auth_failed"

        self._subscribers[ticker].add(queue)
        count = len(self._subscribers[ticker])
        _log(f"Subscribe {ticker} ({_market_label(ticker)}) – now {count} client(s)")

        # First subscriber → ensure the correct upstream is running, then subscribe
        if count == 1:
            # Add ticker BEFORE starting connection loop so the idle-check
            # doesn't block the initial connection attempt.
            upstream._active_tickers.add(ticker)
            await upstream.ensure_started()
            # Wait for auth (up to 5s)
            for _ in range(50):
                if upstream.authenticated or upstream._auth_failed_permanent:
                    break
                if upstream._conn_limit_hit:
                    # Connection limit hit while we were waiting
                    self._subscribers[ticker].discard(queue)
                    upstream._active_tickers.discard(ticker)
                    _log(f"Subscribe REJECTED for {ticker} – connection limit hit during connect")
                    return [], "conn_limit"
                await asyncio.sleep(0.1)
            if upstream.authenticated:
                # subscribe() re-adds to _active_tickers (set, no-op)
                # and sends the WS subscribe command
                await upstream.subscribe(ticker)
                # Start RealTime→Delayed fallback timer for stocks
                if upstream is self._stocks_upstream and upstream._feed == Feed.RealTime:
                    if self._stocks_fallback_task is None or self._stocks_fallback_task.done():
                        self._stocks_fallback_task = asyncio.create_task(self._stocks_realtime_fallback())
            elif upstream._auth_failed_permanent:
                # If the raw websocket auth failed, try fallback to Launchpad
                if upstream is self._options_raw and not self._options_launchpad._auth_failed_permanent:
                    _log(f"Raw options WS auth failed, falling back to Launchpad for {ticker}")
                    upstream._active_tickers.discard(ticker)
                    self._options_upstream_active = self._options_launchpad
                    upstream = self._options_launchpad
                    upstream._active_tickers.add(ticker)
                    await upstream.ensure_started()
                    for _ in range(50):
                        if upstream.authenticated or upstream._auth_failed_permanent:
                            break
                        await asyncio.sleep(0.1)
                    if upstream.authenticated:
                        await upstream.subscribe(ticker)
                    elif upstream._auth_failed_permanent:
                        self._subscribers[ticker].discard(queue)
                        return [], "auth_failed"
                else:
                    self._subscribers[ticker].discard(queue)
                    return [], "auth_failed"

        return list(self._bar_history.get(ticker, [])), None

    async def _resolve_options_upstream(self, ticker: str):
        """Determine which options upstream to use.
        Tries raw WebSocket first, falls back to Launchpad if auth fails."""
        if self._options_upstream_active is not None:
            return self._options_upstream_active

        # Try raw WebSocket first (gives proper OHLCV aggregates)
        raw = self._options_raw
        if not raw._auth_failed_permanent:
            self._options_upstream_active = raw
            _log(f"Using raw options WebSocket (realtime) for {ticker}")
            return raw

        # Raw WS not available → try Launchpad
        lp = self._options_launchpad
        if not lp._auth_failed_permanent:
            self._options_upstream_active = lp
            _log(f"Using Launchpad options feed for {ticker}")
            return lp

        _log(f"No options upstream available for {ticker}")
        return None

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
        opts = self._options_upstream_active or self._options_raw
        stks = self._stocks_upstream
        opts_type = "raw" if opts is self._options_raw else "launchpad" if opts is self._options_launchpad else "none"
        return {
            "options_connected": opts.ws is not None and opts.authenticated,
            "options_plan_ok": not opts._auth_failed_permanent,
            "options_upstream_type": opts_type,
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
