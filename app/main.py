"""
OptionsGangster – VPA Options Analysis Tool
Main FastAPI Application

Changes from v1:
  - Async endpoints that properly await the async PolygonClient
  - FastAPI lifespan for clean startup/shutdown of httpx client
  - Dependency-injection ready (get_polygon, get_vpa)
  - Removed duplicate __main__ block (use run.py)
"""
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta
from typing import Optional
import hashlib, json, logging, math, random, re, time
import asyncio

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response, Cookie, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware

import httpx
import pandas as pd

from app.config import settings
from app.polygon_client import PolygonClient, polygon_client
from app.vpa_engine import VPAEngine, VPASignal, VPAResult, vpa_engine
# ── New signal engine stack ─────────────────────────────────
from app.regime_engine import regime_engine, RegimeEngine, MarketRegime
from app.setup_engine import setup_engine, SetupEngine, SetupAlert
from app.option_picker import option_picker, OptionPicker, OptionPick
from app.edge_scorer import edge_scorer, EdgeScorer, EdgeResult
from app.risk_engine import risk_engine, RiskEngine, TradePlan
from app.alert_manager import alert_manager, AlertManager, AlertState, ActiveAlert
from app.chain_analytics import compute_chain_metrics, ChainMetrics
from app.indicators import (
    atr_value, ema, ema_value, rsi, rsi_value,
    vwap, vwap_bands, vwap_value, vwap_band_values, rolling_rv,
)
from app.live_feed import LiveFeedManager, live_feed_manager, mock_csv_playback, build_mock_sow
from app.fair_value_engine import FairValueEngine, fair_value_engine
from app.sr_engine import SREngine, sr_engine, SRResult
from app.data_layer import DataLayer
from app.idea_engine import idea_engine, IdeaEngine, BriefingInput, PlaybookMode
from app.decision_engine import decision_engine, DecisionEngine, TradeDecision
from app.performance_tracker import perf_tracker
from app.alert_store import alert_store
from app.auth_store import auth_store
from app.firebase_auth import FirebaseAuthError, firebase_enabled, get_firebase_web_config, verify_firebase_token
from app.scanner_engine import scanner_engine
from app.subscription_scanner import SubscriptionScanner
from app.ticker_service import ticker_service
from app.user_alert_store import DEFAULT_ALERT_TYPES, user_alert_store
from app.mongo import close_db
import app.llm_narrator as llm_narrator

logger = logging.getLogger("optionsganster")

# ── Auth config ─────────────────────────────────────────────
# Legacy seed users migrated into Mongo-backed auth on startup.
_USERS: dict[str, dict] = {
    "realsaraf@gmail.com": {
        "password_hash": hashlib.sha256("saraf1237".encode()).hexdigest(),
        "role": "admin",
        "display_name": "realsaraf",
    }
}
_PROTECTED_ACCOUNT_EMAILS = {email.lower() for email, record in _USERS.items() if record.get("role") == "admin"}


# ── Ideas Hub (user-submitted ideas + WS broadcast) ─────────
from datetime import timezone

class IdeasHub:
    """Manages user-submitted trade ideas for the day, with WebSocket fan-out."""

    def __init__(self):
        self._ideas: list[dict] = []      # today's ideas
        self._date: str = ""              # current date key
        self._clients: dict[int, asyncio.Queue] = {}  # ws clients
        self._next_id = 0
        self._lock = asyncio.Lock()

    def _ensure_today(self):
        today = date.today().isoformat()
        if self._date != today:
            self._ideas.clear()
            self._date = today

    async def submit(self, idea: dict) -> dict:
        """Submit a new idea. Returns the full idea dict with id/timestamp."""
        self._ensure_today()
        idea["id"] = len(self._ideas) + 1
        idea["timestamp"] = datetime.now(timezone.utc).isoformat()
        self._ideas.append(idea)
        # Broadcast to all connected WS clients
        msg = json.dumps({"type": "new_idea", "idea": idea})
        dead = []
        for cid, q in self._clients.items():
            try:
                q.put_nowait(msg)
            except asyncio.QueueFull:
                dead.append(cid)
        for cid in dead:
            self._clients.pop(cid, None)
        return idea

    def get_ideas(self) -> list[dict]:
        self._ensure_today()
        return list(self._ideas)

    async def register(self) -> tuple[int, asyncio.Queue]:
        async with self._lock:
            cid = self._next_id
            self._next_id += 1
            q: asyncio.Queue = asyncio.Queue(maxsize=50)
            self._clients[cid] = q
            return cid, q

    async def unregister(self, cid: int):
        async with self._lock:
            self._clients.pop(cid, None)


_ideas_hub = IdeasHub()



# ── Watchlist Price Hub ─────────────────────────────────────
# Shared across all connected watchlist clients.  Fetches prices
# once per second for the UNION of all clients' symbols, then
# fans out filtered results to each client's queue.

class WatchlistHub:
    """
    Server-side hub that de-duplicates watchlist price API calls across
    all connected users.  Only makes Polygon API calls while at least
    one client is connected, and only for the union of requested symbols.
    Each client receives only the symbols it asked for.
    """

    def __init__(self):
        self._lock = asyncio.Lock()
        # client_id → {"queue": asyncio.Queue, "symbols": set[str], "mock": bool}
        self._clients: dict[int, dict] = {}
        self._next_id = 0
        self._poll_task: asyncio.Task | None = None

    async def register(self, symbols: set[str], mock: bool) -> tuple[int, asyncio.Queue]:
        """Register a new client. Returns (client_id, queue)."""
        async with self._lock:
            cid = self._next_id
            self._next_id += 1
            q: asyncio.Queue = asyncio.Queue(maxsize=50)
            self._clients[cid] = {"queue": q, "symbols": symbols, "mock": mock}
            print(f"[WatchlistHub] Client {cid} registered: {len(symbols)} symbols, mock={mock}")
            # Start poll loop if first client
            if len(self._clients) == 1:
                self._poll_task = asyncio.create_task(self._poll_loop())
                print("[WatchlistHub] Poll loop started")
            return cid, q

    async def update_symbols(self, cid: int, symbols: set[str], mock: bool):
        """Update which symbols a client wants."""
        async with self._lock:
            if cid in self._clients:
                self._clients[cid]["symbols"] = symbols
                self._clients[cid]["mock"] = mock

    async def unregister(self, cid: int):
        """Remove a client. Stops the poll loop when no clients remain."""
        async with self._lock:
            self._clients.pop(cid, None)
            print(f"[WatchlistHub] Client {cid} unregistered, {len(self._clients)} remaining")
            if not self._clients and self._poll_task:
                self._poll_task.cancel()
                self._poll_task = None
                print("[WatchlistHub] Poll loop stopped (no clients)")

    async def _poll_loop(self):
        """Fetch prices once per second and fan out to clients."""
        try:
            while True:
                await self._fetch_and_fanout()
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            return

    async def _fetch_and_fanout(self):
        """Single fetch for the union of all symbols, then filter per client."""
        async with self._lock:
            if not self._clients:
                return
            # Snapshot client info (avoid holding lock during API call)
            clients_snapshot = {
                cid: {"queue": info["queue"], "symbols": set(info["symbols"]), "mock": info["mock"]}
                for cid, info in self._clients.items()
            }

        # Compute union of all requested symbols
        all_symbols: set[str] = set()
        for info in clients_snapshot.values():
            all_symbols |= info["symbols"]

        if not all_symbols:
            return

        symbol_list = sorted(all_symbols)

        # Fetch prices – always use real API (mock mode only affects WS option feed)
        real_prices: dict[str, dict] = {}

        try:
            price_lookup = await polygon_client.get_snapshot_prices(symbol_list)
            if not price_lookup:
                price_lookup = await polygon_client.get_prev_close_prices(symbol_list)
            for sym in symbol_list:
                snap = price_lookup.get(sym, {})
                last_price = float(snap.get("lastPrice", 0))
                prev_close = float(snap.get("prevClose", 0))
                if last_price == 0 and prev_close > 0:
                    last_price = prev_close
                real_prices[sym] = {
                    "symbol": sym,
                    "price": last_price,
                    "change": round(float(snap.get("todaysChange", 0)), 2),
                    "changePct": round(float(snap.get("todaysChangePerc", 0)), 2),
                    "high": float(snap.get("dayHigh", 0)),
                    "low": float(snap.get("dayLow", 0)),
                    "volume": int(snap.get("dayVolume", 0)),
                }
        except Exception as e:
            print(f"[WatchlistHub] Price fetch error: {e}")

        # Fan out filtered results to each client
        dead_clients = []
        for cid, info in clients_snapshot.items():
            filtered = [real_prices[s] for s in info["symbols"] if s in real_prices]
            if not filtered:
                continue
            payload = json.dumps({"type": "prices", "prices": filtered})
            try:
                info["queue"].put_nowait(payload)
            except asyncio.QueueFull:
                try:
                    info["queue"].get_nowait()
                    info["queue"].put_nowait(payload)
                except Exception:
                    dead_clients.append(cid)

        # Clean up dead clients
        if dead_clients:
            async with self._lock:
                for cid in dead_clients:
                    self._clients.pop(cid, None)


# Singleton
_watchlist_hub = WatchlistHub()


# ── Lifespan (startup / shutdown) ───────────────────────────

@asynccontextmanager
async def lifespan(application: FastAPI):
    # startup – start the live WebSocket feed manager
    await live_feed_manager.start()
    # Pre-warm Polygon caches in background so first page load is fast
    asyncio.create_task(polygon_client.warm_cache(["QQQ"]))
    # Load ticker directory from local JSON file (instant)
    ticker_service.load()
    await auth_store.ensure_seed_users(_USERS)
    await subscription_scanner.start()
    yield
    # shutdown – stop live feed + close the shared httpx client
    await subscription_scanner.stop()
    await live_feed_manager.stop()
    await polygon_client.close()
    await close_db()


app = FastAPI(
    title="OptionsGangster",
    description="Volume Price Analysis for Options Trading",
    version="2.0.0",
    lifespan=lifespan,
)

alert_manager.register_listener(alert_store.handle_alert_event)
alert_manager.register_listener(user_alert_store.handle_alert_event)


# ── Auth middleware ─────────────────────────────────────────
PUBLIC_PATHS = {"/login", "/api/login", "/api/auth/config", "/api/auth/google", "/api/logout"}

class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if path in PUBLIC_PATHS or path.startswith("/static"):
            return await call_next(request)
        token = request.cookies.get("session")
        session_user = await auth_store.get_session(token) if token else None
        if not session_user:
            if path.startswith("/api/") or path.startswith("/ws/"):
                return Response(status_code=401, content="Unauthorized")
            return RedirectResponse("/login", status_code=302)
        # Attach user info to request state for downstream use
        request.state.user = session_user
        return await call_next(request)

app.add_middleware(AuthMiddleware)


async def _require_websocket_user(ws: WebSocket) -> Optional[dict]:
    """Resolve the current user from the session cookie for websocket endpoints."""
    token = ws.cookies.get("session")
    user = await auth_store.get_session(token) if token else None
    if not user:
        await ws.close(code=4401)
        return None
    return user


# ── Dependency helpers ──────────────────────────────────────

def get_polygon() -> PolygonClient:
    return polygon_client


def get_vpa() -> VPAEngine:
    return vpa_engine


# DataLayer singleton (wraps polygon_client with daily-keyed caches)
_data_layer = DataLayer(polygon_client)


def get_data_layer() -> DataLayer:
    return _data_layer


def get_sr_engine() -> SREngine:
    return sr_engine


# ── Response models ─────────────────────────────────────────

class OHLCVBar(BaseModel):
    datetime: str
    open: float
    high: float
    low: float
    close: float
    volume: int


class VPASignalResponse(BaseModel):
    signal: str
    confidence: float
    description: str
    datetime: str
    price: float
    volume: int
    volume_ratio: float


class GreeksResponse(BaseModel):
    delta: float
    gamma: float
    theta: float
    vega: float
    iv: float
    open_interest: int
    volume: int
    underlying_price: float
    break_even: float
    last_price: float


class ChainMetricsResponse(BaseModel):
    iv_rank: float
    iv_percentile: float
    put_call_oi_ratio: float
    put_call_volume_ratio: float
    total_call_oi: int
    total_put_oi: int
    total_call_volume: int
    total_put_volume: int
    net_gex: float
    gex_regime: str
    max_pain: float
    uoa_detected: bool
    uoa_details: list
    weighted_iv: float


class FactorScoreResponse(BaseModel):
    name: str
    score: float
    confidence: float
    weight: float
    detail: str


class CompositeSignalResponse(BaseModel):
    signal: str
    action: str          # BUY / SELL / HOLD  (clear actionable label)
    score: float
    confidence: float
    trade_archetype: str
    archetype_description: str
    factors: list[FactorScoreResponse]
    greeks: GreeksResponse
    chain_metrics: ChainMetricsResponse
    recommendation: str
    warnings: list[str] = []       # Risk warnings (trend, expiry, premium floor)


class FairValueResponse(BaseModel):
    underlying_price: float
    strike: float
    time_to_expiry: float
    risk_free_rate: float
    historical_vol: float
    dividend_yield: float
    contract_type: str
    theoretical_price: float
    d1: float
    d2: float
    market_price: float
    market_iv: float
    price_difference: float
    pct_difference: float
    is_cheap: bool
    is_expensive: bool
    verdict: str
    detail: str
    bid: float
    ask: float
    spread: float
    spread_pct: float
    hv_vs_iv: float


class SRLevelResponse(BaseModel):
    price: float
    kind: str          # "support" | "resistance"
    source: str        # "swing" | "fib_ret" | "fib_ext" | "poc" | "round"
    strength: float
    label: str


class SRResponse(BaseModel):
    levels: list[SRLevelResponse]
    poc: Optional[float] = None
    vah: Optional[float] = None
    val: Optional[float] = None
    fib_high: Optional[float] = None
    fib_low: Optional[float] = None
    nearest_support: Optional[float] = None
    nearest_resistance: Optional[float] = None
    proximity_score: float = 0.0
    proximity_detail: str = ""


class AnalysisResponse(BaseModel):
    symbol: str
    expiration: str
    strike: float
    right: str
    interval: int
    bars: list[OHLCVBar]
    signals: list[VPASignalResponse]
    bias: dict
    composite: Optional[CompositeSignalResponse] = None
    underlying_bars: list[OHLCVBar] = []
    volume_regime: Optional[dict] = None
    fair_value: Optional[FairValueResponse] = None
    support_resistance: Optional[SRResponse] = None


# ── New signal engine response models ────────────────────────

class RegimeResponse(BaseModel):
    regime: str
    confidence: float
    detail: str
    vwap_current: float = 0.0
    atr_current: float = 0.0
    rsi_current: float = 0.0
    price_vs_vwap: str = "at"  # "above" | "below" | "at"


class AlertOptionResponse(BaseModel):
    ticker: str
    strike: float
    expiration: str
    option_type: str
    delta: float
    spread_pct: float
    iv: float
    entry_premium_est: float
    dte: int
    notes: str = ""


class AlertResponse(BaseModel):
    id: str
    state: str
    created_at: str
    expires_at: str
    setup_name: str
    direction: str
    edge_score: int
    tier: str
    regime: str
    trigger_price: float
    entry_condition: str
    stop_price: float
    target_1: float
    target_2: Optional[float] = None
    reward_risk_ratio: float
    time_stop_minutes: int
    kill_switch_conditions: list[str] = []
    reasons: list[str] = []
    option: Optional[AlertOptionResponse] = None
    activated_premium: Optional[float] = None


class AlertHistoryResponse(BaseModel):
    id: str
    symbol: str
    state: str
    detected_at: str
    activated_at: Optional[str] = None
    resolved_at: Optional[str] = None
    expires_at: Optional[str] = None
    setup_name: str
    entry_condition: str = ""
    direction: str
    edge_score: int
    tier: str
    regime: str
    trigger_price: float
    stop_price: float
    target_1: float
    target_2: Optional[float] = None
    reward_risk_ratio: float = 0.0
    reasons: list[str] = []
    option: Optional[AlertOptionResponse] = None
    entry_premium: float = 0.0
    exit_premium: float = 0.0
    pnl_pct: float = 0.0


class IndicatorsResponse(BaseModel):
    vwap: list[float] = []
    vwap_upper: list[float] = []
    vwap_lower: list[float] = []
    ema_9: list[float] = []
    ema_20: list[float] = []
    rsi: list[float] = []
    atr: float = 0.0


class ChainSnapshotResponse(BaseModel):
    iv_rank: float = 0.0
    iv_percentile: float = 0.0
    put_call_oi_ratio: float = 1.0
    put_call_volume_ratio: float = 1.0
    net_gex: float = 0.0
    gex_regime: str = "neutral"
    max_pain: float = 0.0
    uoa_detected: bool = False
    uoa_details: list = []
    call_wall: float = 0.0
    put_wall: float = 0.0
    weighted_iv: float = 0.0


class SignalResponse(BaseModel):
    symbol: str
    as_of: str
    asset_class: str = "equity"   # "equity" | "futures"
    proxy_ticker: str = ""        # ETF proxy used for data (futures only)
    bars: list[OHLCVBar] = []
    regime: Optional[RegimeResponse] = None
    active_alerts: list[AlertResponse] = []
    key_levels: list[SRLevelResponse] = []
    indicators: Optional[IndicatorsResponse] = None
    chain_snapshot: Optional[ChainSnapshotResponse] = None
    vpa: Optional[dict] = None
    fair_value: Optional[FairValueResponse] = None
    volume_regime: Optional[dict] = None
    proximity_score: float = 0.0
    proximity_detail: str = ""
    prev_close: float = 0.0       # Yesterday's close for day-change calc


def _build_alert_option_response(alert: ActiveAlert) -> Optional[AlertOptionResponse]:
    opt = alert.option
    if not opt:
        return None
    return AlertOptionResponse(
        ticker=opt.ticker,
        strike=opt.strike,
        expiration=str(opt.expiration) if opt.expiration else "",
        option_type=opt.option_type,
        delta=round(opt.delta, 3),
        spread_pct=round(opt.spread_pct, 2),
        iv=round(opt.iv, 4),
        entry_premium_est=round(opt.entry_premium_est, 2),
        dte=opt.dte,
        notes=opt.notes,
    )


def _build_alert_response(alert: ActiveAlert) -> AlertResponse:
    p = alert.plan
    return AlertResponse(
        id=alert.id,
        state=alert.state.value,
        created_at=alert.detected_at,
        expires_at=alert.expires_at or "",
        setup_name=alert.setup_name,
        direction=alert.direction,
        edge_score=alert.edge_score,
        tier=alert.tier,
        regime=alert.regime,
        trigger_price=p.entry_price,
        entry_condition=alert.entry_condition,
        stop_price=p.stop_price,
        target_1=p.target_1,
        target_2=p.target_2,
        reward_risk_ratio=round(p.reward_risk_ratio, 2),
        time_stop_minutes=p.time_stop_minutes,
        kill_switch_conditions=p.kill_switch_conditions,
        reasons=alert.reasons,
        option=_build_alert_option_response(alert),
        activated_premium=alert.entry_premium or None,
    )


def _build_alert_history_response(alert: ActiveAlert) -> AlertHistoryResponse:
    p = alert.plan
    return AlertHistoryResponse(
        id=alert.id,
        symbol=alert.symbol,
        state=alert.state.value,
        detected_at=alert.detected_at,
        activated_at=alert.activated_at,
        resolved_at=alert.resolved_at,
        expires_at=alert.expires_at,
        setup_name=alert.setup_name,
        entry_condition=alert.entry_condition,
        direction=alert.direction,
        edge_score=alert.edge_score,
        tier=alert.tier,
        regime=alert.regime,
        trigger_price=p.entry_price,
        stop_price=p.stop_price,
        target_1=p.target_1,
        target_2=p.target_2,
        reward_risk_ratio=round(p.reward_risk_ratio, 2),
        reasons=alert.reasons,
        option=_build_alert_option_response(alert),
        entry_premium=alert.entry_premium,
        exit_premium=alert.exit_premium,
        pnl_pct=alert.pnl_pct,
    )


def _build_alert_history_response_from_doc(doc: dict) -> AlertHistoryResponse:
    opt = doc.get("option") or None
    opt_out = None
    if opt:
        opt_out = AlertOptionResponse(
            ticker=opt.get("ticker", ""),
            strike=float(opt.get("strike", 0.0) or 0.0),
            expiration=str(opt.get("expiration", "") or ""),
            option_type=opt.get("option_type", ""),
            delta=float(opt.get("delta", 0.0) or 0.0),
            spread_pct=float(opt.get("spread_pct", 0.0) or 0.0),
            iv=float(opt.get("iv", 0.0) or 0.0),
            entry_premium_est=float(opt.get("entry_premium_est", 0.0) or 0.0),
            dte=int(opt.get("dte", 0) or 0),
            notes=opt.get("notes", "") or "",
        )

    return AlertHistoryResponse(
        id=doc.get("alert_id", doc.get("id", "")),
        symbol=doc.get("symbol", ""),
        state=doc.get("state", ""),
        detected_at=doc.get("detected_at", ""),
        activated_at=doc.get("activated_at"),
        resolved_at=doc.get("resolved_at"),
        expires_at=doc.get("expires_at"),
        setup_name=doc.get("setup_name", ""),
        entry_condition=doc.get("entry_condition", ""),
        direction=doc.get("direction", ""),
        edge_score=int(doc.get("edge_score", 0) or 0),
        tier=doc.get("tier", ""),
        regime=doc.get("regime", ""),
        trigger_price=float(doc.get("trigger_price", doc.get("entry_price", 0.0)) or 0.0),
        stop_price=float(doc.get("stop_price", 0.0) or 0.0),
        target_1=float(doc.get("target_1", 0.0) or 0.0),
        target_2=doc.get("target_2"),
        reward_risk_ratio=float(doc.get("reward_risk_ratio", 0.0) or 0.0),
        reasons=doc.get("reasons", []) or [],
        option=opt_out,
        entry_premium=float(doc.get("entry_premium", 0.0) or 0.0),
        exit_premium=float(doc.get("exit_premium", 0.0) or 0.0),
        pnl_pct=float(doc.get("pnl_pct", 0.0) or 0.0),
    )


# ── Legacy models (kept while old endpoints coexist) ──────────

class ExpirationResponse(BaseModel):
    symbol: str
    expirations: list[str]


class StrikeResponse(BaseModel):
    symbol: str
    expiration: str
    strikes: list[float]


# ── Auth endpoints ──────────────────────────────────────────

LOGIN_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>OptionsGangster – AI-Powered Options Signals</title>
<meta name="description" content="AI-driven options signals that cut through the noise. One actionable verdict — BUY, SELL, or HOLD — with a confidence score. Free to use.">
<meta property="og:type" content="website">
<meta property="og:url" content="https://optionsgangster.com">
<meta property="og:title" content="OptionsGangster – AI-Powered Options Signals">
<meta property="og:description" content="Stop guessing. Our AI analyzes multiple market dimensions in real time and delivers one clear verdict with a confidence score. Free.">
<meta property="og:image" content="https://optionsgangster.com/static/og-image.png">
<meta property="og:image:width" content="1200">
<meta property="og:image:height" content="630">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:title" content="OptionsGangster – AI-Powered Options Signals">
<meta name="twitter:description" content="Stop guessing. Our AI analyzes multiple market dimensions in real time and delivers one clear verdict with a confidence score.">
<meta name="twitter:image" content="https://optionsgangster.com/static/og-image.png">
<link rel="icon" type="image/png" sizes="32x32" href="/static/favicon-32.png">
<link rel="icon" type="image/x-icon" href="/static/favicon.ico">
<link rel="apple-touch-icon" sizes="180x180" href="/static/favicon-180.png">
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{min-height:100vh;background:#0a0a1a;font-family:'Segoe UI',system-ui,-apple-system,sans-serif;color:#e0e0e0;overflow-x:hidden}

/* ── NAV ── */
nav{position:fixed;top:0;left:0;right:0;z-index:100;padding:16px 32px;display:flex;align-items:center;justify-content:space-between;background:rgba(10,10,26,.85);backdrop-filter:blur(12px);border-bottom:1px solid rgba(255,255,255,.06)}
.nav-logo{font-size:1.3rem;font-weight:800;letter-spacing:-.5px}
.nav-logo span{color:#ef5350}
.nav-links{display:flex;gap:24px;align-items:center}
.nav-links a{color:#9ca3af;text-decoration:none;font-size:.9rem;transition:color .2s}
.nav-links a:hover{color:#fff}
.nav-cta{background:#3b82f6;color:#fff !important;padding:8px 20px;border-radius:6px;font-weight:600;font-size:.9rem;border:none;cursor:pointer;transition:all .2s}
.nav-cta:hover{background:#2563eb;transform:translateY(-1px)}

/* ── HERO ── */
.hero{min-height:100vh;display:flex;align-items:center;justify-content:center;text-align:center;padding:120px 24px 80px;position:relative}
.hero::before{content:'';position:absolute;top:0;left:50%;transform:translateX(-50%);width:800px;height:800px;background:radial-gradient(circle,rgba(59,130,246,.08) 0%,transparent 70%);pointer-events:none}
.hero-content{max-width:720px;position:relative;z-index:1}
.hero-badge{display:inline-block;padding:6px 16px;border-radius:20px;background:rgba(59,130,246,.12);border:1px solid rgba(59,130,246,.25);color:#60a5fa;font-size:.8rem;font-weight:600;letter-spacing:.5px;text-transform:uppercase;margin-bottom:24px}
.hero h1{font-size:3.5rem;font-weight:800;line-height:1.1;margin-bottom:20px;letter-spacing:-1px}
.hero h1 .accent{background:linear-gradient(135deg,#3b82f6,#8b5cf6);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.hero p{font-size:1.15rem;color:#9ca3af;line-height:1.7;margin-bottom:40px;max-width:560px;margin-left:auto;margin-right:auto}
.hero-actions{display:flex;gap:16px;justify-content:center;flex-wrap:wrap}
.btn-primary{padding:14px 32px;border-radius:8px;background:#3b82f6;color:#fff;font-size:1rem;font-weight:700;border:none;cursor:pointer;transition:all .25s;letter-spacing:.3px}
.btn-primary:hover{background:#2563eb;transform:translateY(-2px);box-shadow:0 8px 24px rgba(59,130,246,.3)}
.btn-secondary{padding:14px 32px;border-radius:8px;background:transparent;color:#d1d4dc;font-size:1rem;font-weight:600;border:1px solid #374151;cursor:pointer;transition:all .25s}
.btn-secondary:hover{border-color:#6b7280;background:rgba(255,255,255,.03)}

/* ── STATS ── */
.stats{display:flex;gap:48px;justify-content:center;margin-top:64px;flex-wrap:wrap}
.stat{text-align:center}
.stat-value{font-size:2rem;font-weight:800;color:#fff}
.stat-value .green{color:#26a69a}
.stat-value .blue{color:#3b82f6}
.stat-label{font-size:.8rem;color:#6b7280;margin-top:4px;text-transform:uppercase;letter-spacing:.5px}

/* ── FEATURES ── */
.features{padding:80px 24px;max-width:1100px;margin:0 auto}
.section-label{text-align:center;color:#3b82f6;font-size:.8rem;font-weight:700;letter-spacing:1px;text-transform:uppercase;margin-bottom:12px}
.section-title{text-align:center;font-size:2.2rem;font-weight:800;margin-bottom:48px;letter-spacing:-.5px}
.features-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:24px}
.feature-card{background:#111827;border:1px solid #1e293b;border-radius:12px;padding:32px;transition:all .25s}
.feature-card:hover{border-color:#374151;transform:translateY(-4px);box-shadow:0 12px 36px rgba(0,0,0,.3)}
.feature-icon{width:48px;height:48px;border-radius:10px;display:flex;align-items:center;justify-content:center;font-size:1.4rem;margin-bottom:16px}
.fi-blue{background:rgba(59,130,246,.12);color:#60a5fa}
.fi-green{background:rgba(38,166,154,.12);color:#26a69a}
.fi-purple{background:rgba(139,92,246,.12);color:#a78bfa}
.fi-red{background:rgba(239,83,80,.12);color:#ef5350}
.fi-yellow{background:rgba(250,204,21,.12);color:#facc15}
.fi-cyan{background:rgba(34,211,238,.12);color:#22d3ee}
.feature-card h3{font-size:1.1rem;font-weight:700;margin-bottom:8px}
.feature-card p{font-size:.9rem;color:#9ca3af;line-height:1.6}

/* ── HOW IT WORKS ── */
.how{padding:80px 24px;background:#0d0d20}
.how-grid{max-width:900px;margin:0 auto;display:grid;grid-template-columns:repeat(3,1fr);gap:32px}
.how-step{text-align:center;padding:24px}
.step-num{width:40px;height:40px;border-radius:50%;background:rgba(59,130,246,.15);color:#60a5fa;display:flex;align-items:center;justify-content:center;font-weight:800;font-size:1.1rem;margin:0 auto 16px}
.how-step h3{font-size:1rem;font-weight:700;margin-bottom:8px}
.how-step p{font-size:.85rem;color:#9ca3af;line-height:1.5}

/* ── CTA ── */
.cta{padding:80px 24px;text-align:center}
.cta-box{max-width:640px;margin:0 auto;background:linear-gradient(135deg,#111827,#0f172a);border:1px solid #1e293b;border-radius:16px;padding:48px 40px}
.cta-box h2{font-size:1.8rem;font-weight:800;margin-bottom:12px}
.cta-box p{color:#9ca3af;margin-bottom:28px;font-size:1rem}

/* ── FOOTER ── */
footer{padding:32px 24px;text-align:center;color:#4b5563;font-size:.8rem;border-top:1px solid #1e293b}

/* ── MODAL ── */
.modal-overlay{display:none;position:fixed;inset:0;z-index:200;background:rgba(0,0,0,.65);backdrop-filter:blur(6px);align-items:center;justify-content:center}
.modal-overlay.open{display:flex}
.modal{background:#111827;border:1px solid #1e293b;border-radius:14px;padding:40px;width:400px;max-width:90vw;box-shadow:0 24px 64px rgba(0,0,0,.6);position:relative;animation:modalIn .25s ease}
@keyframes modalIn{from{opacity:0;transform:scale(.95) translateY(10px)}to{opacity:1;transform:scale(1) translateY(0)}}
.modal-close{position:absolute;top:14px;right:16px;background:none;border:none;color:#6b7280;font-size:1.3rem;cursor:pointer;transition:color .2s;line-height:1}
.modal-close:hover{color:#fff}
.modal h2{font-size:1.4rem;font-weight:800;margin-bottom:4px}
.modal .sub{color:#6b7280;font-size:.85rem;margin-bottom:24px}
.field{margin-bottom:16px}
.field label{display:block;font-size:.8rem;color:#9ca3af;margin-bottom:4px;font-weight:500}
.field input{width:100%;padding:11px 14px;border-radius:8px;border:1px solid #374151;background:#1a1a2e;color:#e0e0e0;font-size:.95rem;outline:none;transition:border-color .2s}
.field input:focus{border-color:#3b82f6}
.modal .btn-primary{width:100%;margin-top:20px}
.err{color:#ef4444;font-size:.85rem;text-align:center;margin-top:12px;min-height:20px}

/* ── RESPONSIVE ── */
@media(max-width:768px){
  nav{padding:12px 16px}
  .nav-links a:not(.nav-cta){display:none}
  .hero h1{font-size:2.2rem}
  .hero p{font-size:1rem}
  .stats{gap:24px}
  .stat-value{font-size:1.5rem}
  .features-grid{grid-template-columns:1fr}
  .how-grid{grid-template-columns:1fr;gap:16px}
  .cta-box{padding:32px 20px}
}
</style>
</head>
<body>

<!-- NAV -->
<nav>
  <div class="nav-logo">Options<span>Gangster</span></div>
  <div class="nav-links">
    <a href="#features">Features</a>
    <a href="#how">How It Works</a>
    <button class="nav-cta" onclick="openLogin()">Sign In</button>
  </div>
</nav>

<!-- HERO -->
<section class="hero">
  <div class="hero-content">
    <div class="hero-badge">AI-Powered Options Intelligence</div>
    <h1>Stop Guessing.<br><span class="accent">Start Trading Smarter.</span></h1>
    <p>Our proprietary AI analyzes multiple market dimensions in real time — volatility, flow, structure, momentum — and distills it into one actionable BUY / SELL / HOLD verdict you can trust.</p>
    <div class="hero-actions">
      <button class="btn-primary" onclick="openLogin()">Get Started Free</button>
      <button class="btn-secondary" onclick="document.getElementById('features').scrollIntoView({behavior:'smooth'})">See Features</button>
    </div>
    <div class="stats">
      <div class="stat"><div class="stat-value"><span class="green">AI</span></div><div class="stat-label">Driven</div></div>
      <div class="stat"><div class="stat-value"><span class="blue">Live</span></div><div class="stat-label">Signals</div></div>
      <div class="stat"><div class="stat-value"><span class="green">$0</span></div><div class="stat-label">Monthly</div></div>
    </div>
  </div>
</section>

<!-- FEATURES -->
<section class="features" id="features">
  <div class="section-label">Features</div>
  <h2 class="section-title">Your Unfair Advantage in Options</h2>
  <div class="features-grid">
    <div class="feature-card">
      <div class="feature-icon fi-blue">&#x1F9E0;</div>
      <h3>Multi-Dimensional AI</h3>
      <p>Our engine doesn't look at one thing — it evaluates multiple market dimensions simultaneously to find signals humans miss.</p>
    </div>
    <div class="feature-card">
      <div class="feature-icon fi-green">&#9889;</div>
      <h3>Real-Time Verdicts</h3>
      <p>No lagging indicators. The AI processes live market data and delivers an actionable BUY, SELL, or HOLD signal in seconds.</p>
    </div>
    <div class="feature-card">
      <div class="feature-icon fi-purple">&#x1F3AF;</div>
      <h3>Confidence Scoring</h3>
      <p>Every signal comes with a confidence percentage so you know when conditions are strong — and when to sit on your hands.</p>
    </div>
    <div class="feature-card">
      <div class="feature-icon fi-red">&#x1F4CA;</div>
      <h3>Visual Chart Intel</h3>
      <p>Signals aren't just text — they're plotted right on the chart with smart markers. See exactly where the AI sees opportunity.</p>
    </div>
    <div class="feature-card">
      <div class="feature-icon fi-yellow">&#x1F50D;</div>
      <h3>Smart Money Detection</h3>
      <p>The AI detects unusual flow and institutional footprints that retail traders typically miss — before the move happens.</p>
    </div>
    <div class="feature-card">
      <div class="feature-icon fi-cyan">&#x1F4E1;</div>
      <h3>Live Auto-Refresh</h3>
      <p>Enable LIVE mode and the AI re-scans every 10 seconds. Stay on top of rapidly changing conditions without lifting a finger.</p>
    </div>
  </div>
</section>

<!-- HOW IT WORKS -->
<section class="how" id="how">
  <div class="section-label">How It Works</div>
  <h2 class="section-title">Three Steps to Better Trades</h2>
  <div class="how-grid">
    <div class="how-step">
      <div class="step-num">1</div>
      <h3>Pick a Contract</h3>
      <p>Choose any ticker, expiration, and strike. Calls or puts — the AI handles both.</p>
    </div>
    <div class="how-step">
      <div class="step-num">2</div>
      <h3>Hit Analyze</h3>
      <p>The AI scans the contract across multiple dimensions in seconds and delivers a clear signal on the chart.</p>
    </div>
    <div class="how-step">
      <div class="step-num">3</div>
      <h3>Trade with Confidence</h3>
      <p>Get a clear BUY, SELL, or HOLD verdict with a confidence score. Go LIVE for continuous updates.</p>
    </div>
  </div>
</section>

<!-- CTA -->
<section class="cta">
  <div class="cta-box">
    <h2>Ready to Level Up Your Options Game?</h2>
    <p>Join now. No credit card required. Full access to the AI signal engine.</p>
    <button class="btn-primary" onclick="openLogin()">Sign In &amp; Start Analyzing</button>
  </div>
</section>

<footer>&copy; 2026 OptionsGangster. Built for traders who want an edge.</footer>

<!-- LOGIN MODAL -->
<div class="modal-overlay" id="modal">
  <div class="modal">
    <button class="modal-close" onclick="closeLogin()">&times;</button>
    <h2>Welcome Back</h2>
    <p class="sub">Sign in to access your dashboard</p>
    <form id="lf">
      <div class="field"><label>Email</label><input type="email" id="em" required autofocus></div>
      <div class="field"><label>Password</label><input type="password" id="pw" required></div>
      <button class="btn-primary" type="submit">Sign In</button>
      <p class="err" id="err"></p>
    </form>
  </div>
</div>

<script>
function openLogin(){document.getElementById('modal').classList.add('open');document.getElementById('em').focus()}
function closeLogin(){document.getElementById('modal').classList.remove('open');document.getElementById('err').textContent=''}
document.getElementById('modal').addEventListener('click',e=>{if(e.target===e.currentTarget)closeLogin()});
document.addEventListener('keydown',e=>{if(e.key==='Escape')closeLogin()});
document.getElementById('lf').addEventListener('submit',async e=>{
  e.preventDefault();
  const r=await fetch('/api/login',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({email:document.getElementById('em').value,
      password:document.getElementById('pw').value})});
  if(r.ok){window.location.href='/'}else{
    const d=await r.json();document.getElementById('err').textContent=d.detail||'Login failed'}
});
</script>
</body>
</html>
"""

@app.get("/login", response_class=HTMLResponse)
async def login_page():
    return FileResponse(
        "app/static/login.html",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


class GoogleLoginRequest(BaseModel):
    id_token: str


class AuthorizedUserCreateRequest(BaseModel):
    email: str
    display_name: str = ""
    role: str = "general"


class AuthorizedUserUpdateRequest(BaseModel):
    display_name: Optional[str] = None
    role: Optional[str] = None
    is_authorized: Optional[bool] = None


class SymbolAlertSettingsRequest(BaseModel):
    symbol: str
    alert_types: dict[str, bool] = Field(default_factory=dict)


class AlertSettingsUpdateRequest(BaseModel):
    symbols: list[str] = Field(default_factory=list)
    alert_mode: str = "shared"
    shared_alert_types: dict[str, bool] = Field(default_factory=dict)
    symbol_settings: list[SymbolAlertSettingsRequest] = Field(default_factory=list)


class AlertNotificationReadRequest(BaseModel):
    notification_keys: list[str] = Field(default_factory=list)

@app.post("/api/login")
async def login_removed():
    raise HTTPException(status_code=410, detail="Password login has been removed. Use Google sign-in.")


@app.get("/api/auth/config")
async def get_auth_config():
    return {
        "enabled": firebase_enabled(),
        "firebase": get_firebase_web_config(),
    }


@app.post("/api/auth/google")
async def google_login(body: GoogleLoginRequest):
    try:
        claims = verify_firebase_token(body.id_token)
    except FirebaseAuthError as exc:
        raise HTTPException(status_code=401, detail=str(exc))

    email = claims.get("email", "").lower()
    user_record = await auth_store.get_authorized_user(email)
    if not user_record:
        raise HTTPException(status_code=403, detail="This Google account is not authorized for this app")

    user = await auth_store.record_google_login(claims, user_record)
    token = await auth_store.create_session(user)
    resp = Response(
        content=json.dumps({"ok": True, "role": user.get("role", "general"), "display_name": user.get("display_name", email)}),
        media_type="application/json",
    )
    resp.set_cookie(key="session", value=token, httponly=True, max_age=86400 * 7, samesite="lax")
    return resp


@app.get("/api/logout")
async def logout(request: Request):
    token = request.cookies.get("session")
    if token:
        await auth_store.delete_session(token)
    resp = RedirectResponse("/login", status_code=302)
    resp.delete_cookie("session")
    return resp


@app.get("/api/me")
async def get_me(request: Request):
    """Return the current user's profile (role, display_name)."""
    user = getattr(request.state, "user", None)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    email = user["email"]
    return {
        "email": email,
        "role": user["role"],
        "display_name": user["display_name"],
        "can_delete_account": email.lower() not in _PROTECTED_ACCOUNT_EMAILS,
    }


def _require_admin(request: Request) -> dict:
    user = getattr(request.state, "user", None)
    if not user or user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return user


def _require_user(request: Request) -> dict:
    user = getattr(request.state, "user", None)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user


def _isoformat_fields(payload: dict, fields: tuple[str, ...]) -> dict:
    cloned = dict(payload)
    for field in fields:
        if hasattr(cloned.get(field), "isoformat"):
            cloned[field] = cloned[field].isoformat()
    return cloned


async def _build_alert_settings_payload(email: str) -> dict:
    settings_doc = await user_alert_store.get_settings(email)
    notifications = await user_alert_store.list_notifications(email, limit=25)
    unread_count = await user_alert_store.unread_count(email)
    settings_doc = _isoformat_fields(settings_doc, ("created_at", "updated_at"))
    notifications = [
        _isoformat_fields(doc, ("created_at", "read_at"))
        for doc in notifications
    ]
    return {
        "symbols": settings_doc.get("symbols", []),
        "alert_mode": settings_doc.get("alert_mode", "shared"),
        "shared_alert_types": settings_doc.get("shared_alert_types", dict(DEFAULT_ALERT_TYPES)),
        "symbol_settings": settings_doc.get("symbol_settings", []),
        "alert_types": settings_doc.get("shared_alert_types", dict(DEFAULT_ALERT_TYPES)),
        "notifications": notifications,
        "unread_count": unread_count,
        "scan_interval_seconds": 60,
    }


@app.get("/api/admin/users")
async def list_authorized_users(request: Request):
    _require_admin(request)
    users = await auth_store.list_users()
    for user in users:
        user["is_active"] = bool(user.get("is_authorized", True)) and not bool(user.get("is_deleted", False))
        user["is_protected"] = user.get("email", "").lower() in _PROTECTED_ACCOUNT_EMAILS
        if hasattr(user.get("created_at"), "isoformat"):
            user["created_at"] = user["created_at"].isoformat()
        if hasattr(user.get("updated_at"), "isoformat"):
            user["updated_at"] = user["updated_at"].isoformat()
        if hasattr(user.get("last_login_at"), "isoformat"):
            user["last_login_at"] = user["last_login_at"].isoformat()
    return users


@app.post("/api/admin/users")
async def create_authorized_user(request: Request, body: AuthorizedUserCreateRequest):
    _require_admin(request)
    try:
        return await auth_store.create_user(body.email, body.display_name, body.role)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.patch("/api/admin/users/{email:path}")
async def update_authorized_user(email: str, request: Request, body: AuthorizedUserUpdateRequest):
    _require_admin(request)
    if body.is_authorized is False and email.lower() in _PROTECTED_ACCOUNT_EMAILS:
        raise HTTPException(status_code=403, detail="Protected admin account cannot be deactivated")
    user = await auth_store.update_user(
        email,
        display_name=body.display_name,
        role=body.role,
        is_authorized=body.is_authorized,
    )
    if not user:
        raise HTTPException(status_code=404, detail=f"User not found: {email}")
    if body.is_authorized is False:
        await auth_store.delete_sessions_for_email(email)
    user["is_active"] = bool(user.get("is_authorized", True)) and not bool(user.get("is_deleted", False))
    user["is_protected"] = user.get("email", "").lower() in _PROTECTED_ACCOUNT_EMAILS
    for field in ("created_at", "updated_at", "last_login_at"):
        if hasattr(user.get(field), "isoformat"):
            user[field] = user[field].isoformat()
    return user


@app.delete("/api/admin/users/{email:path}")
async def remove_authorized_user(email: str, request: Request):
    actor = _require_admin(request)
    normalized = email.lower()
    if normalized in _PROTECTED_ACCOUNT_EMAILS:
        raise HTTPException(status_code=403, detail="Protected admin account cannot be removed")
    user = await auth_store.soft_delete_user(normalized, deleted_by=actor["email"])
    if not user:
        raise HTTPException(status_code=404, detail=f"User not found: {email}")
    return {"ok": True, "email": normalized}


@app.get("/api/settings/alerts")
async def get_alert_settings(request: Request):
    user = _require_user(request)
    return await _build_alert_settings_payload(user["email"])


@app.put("/api/settings/alerts")
async def update_alert_settings(request: Request, body: AlertSettingsUpdateRequest):
    user = _require_user(request)
    normalized_symbols: list[str] = []
    for symbol in body.symbols:
        data_symbol, _, _ = _normalize_symbol(symbol)
        normalized_symbols.append(data_symbol)
    normalized_symbol_settings: list[dict] = []
    for item in body.symbol_settings:
        data_symbol, _, _ = _normalize_symbol(item.symbol)
        normalized_symbol_settings.append({"symbol": data_symbol, "alert_types": item.alert_types})
    await user_alert_store.update_settings(
        user["email"],
        normalized_symbols,
        body.alert_mode,
        body.shared_alert_types,
        normalized_symbol_settings,
    )
    return await _build_alert_settings_payload(user["email"])


@app.delete("/api/me")
async def delete_my_account(request: Request):
    user = _require_user(request)
    email = user["email"].lower()
    if email in _PROTECTED_ACCOUNT_EMAILS:
        raise HTTPException(status_code=403, detail="Protected admin account cannot be deleted")
    deleted = await auth_store.soft_delete_user(email, deleted_by=email)
    if not deleted:
        raise HTTPException(status_code=404, detail="User not found")
    resp = Response(content=json.dumps({"ok": True}), media_type="application/json")
    resp.delete_cookie("session")
    return resp


@app.post("/api/settings/alert-notifications/read")
async def mark_alert_notifications_read(request: Request, body: AlertNotificationReadRequest):
    user = _require_user(request)
    modified = await user_alert_store.mark_notifications_read(
        user["email"],
        body.notification_keys or None,
    )
    return {
        "ok": True,
        "modified": modified,
        "unread_count": await user_alert_store.unread_count(user["email"]),
    }


# ── Endpoints ───────────────────────────────────────────────

@app.get("/")
async def root():
    """Serve the main UI."""
    return FileResponse(
        "app/static/index.html",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


@app.get("/api/expirations/{symbol}")
async def get_expirations(
    symbol: str,
    poly: PolygonClient = Depends(get_polygon),
) -> ExpirationResponse:
    """Get available expiration dates for a symbol."""
    try:
        expirations = await poly.get_expirations(symbol.upper())
        return ExpirationResponse(
            symbol=symbol.upper(),
            expirations=[exp.strftime("%Y-%m-%d") for exp in expirations],
        )
    except httpx.HTTPStatusError as e:
        code = e.response.status_code if e.response is not None else 500
        raise HTTPException(status_code=code, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/strikes/{symbol}/{expiration}")
async def get_strikes(
    symbol: str,
    expiration: str,
    poly: PolygonClient = Depends(get_polygon),
) -> StrikeResponse:
    """Get available strikes for a symbol and expiration."""
    try:
        exp_date = datetime.strptime(expiration, "%Y-%m-%d").date()
        strikes = await poly.get_strikes(symbol.upper(), exp_date)
        return StrikeResponse(
            symbol=symbol.upper(),
            expiration=expiration,
            strikes=strikes,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/underlying/{symbol}")
async def get_underlying_price(
    symbol: str,
    poly: PolygonClient = Depends(get_polygon),
):
    """Get current price of underlying for ATM strike selection."""
    try:
        price = await poly.get_underlying_price(symbol.upper())
        return {"symbol": symbol.upper(), "price": price}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/option/bars")
async def get_option_bars_endpoint(
    ticker: str = Query(..., description="OCC option ticker e.g. O:QQQ260225C00610000"),
    interval: int = Query(5, description="Bar interval in minutes"),
    poly: PolygonClient = Depends(get_polygon),
):
    """Fetch historical OHLCV bars for an option contract + its underlying stock."""
    try:
        raw = ticker.replace("O:", "")
        m = re.match(r'^([A-Z]+)(\d{6})([CP])(\d{8})$', raw)
        if not m:
            raise HTTPException(status_code=400, detail=f"Invalid OCC ticker: {ticker}")
        sym = m.group(1)
        dt_str = m.group(2)
        right = m.group(3)
        strike = int(m.group(4)) / 1000
        expiration = date(int('20' + dt_str[:2]), int(dt_str[2:4]), int(dt_str[4:6]))

        end_dt = date.today()
        start_dt = end_dt - timedelta(days=5)

        try:
            opt_df, stock_df = await asyncio.gather(
                poly.get_option_ohlcv(sym, expiration, strike, right, start_dt, end_dt, interval),
                poly.get_stock_ohlcv(sym, start_date=start_dt, end_date=end_dt, interval_min=interval),
            )
        except Exception:
            opt_df = pd.DataFrame()
            stock_df = pd.DataFrame()

        def df_to_bars(df):
            if df.empty:
                return []
            bars = []
            for _, row in df.iterrows():
                dt = row['datetime']
                bars.append({
                    'datetime': dt.strftime('%Y-%m-%d %H:%M:%S') if hasattr(dt, 'strftime') else str(dt),
                    'open': round(float(row['open']), 4),
                    'high': round(float(row['high']), 4),
                    'low': round(float(row['low']), 4),
                    'close': round(float(row['close']), 4),
                    'volume': int(row.get('volume', 0) or 0),
                })
            return bars

        return {
            'option_bars': df_to_bars(opt_df),
            'stock_bars': df_to_bars(stock_df),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Futures symbol normalisation ──────────────────────────────
# Maps TradingView-style /XX shorthand → closest ETF proxy on current plan.
# (Polygon Starter/Stocks plan does NOT cover I: index/futures tickers.)
# The ETF proxy is used for OHLCV data; asset_class stays "futures" so the
# UI suppresses chain/options features and shows a proxy badge.
_FUTURES_MAP: dict[str, str] = {
    "/ES":  "SPY",   # E-mini S&P 500   → SPDR S&P 500 ETF
    "/NQ":  "QQQ",   # E-mini Nasdaq 100→ Invesco QQQ
    "/YM":  "DIA",   # E-mini Dow Jones → SPDR Dow Jones ETF
    "/RTY": "IWM",   # E-mini Russell   → iShares Russell 2000
    "/CL":  "USO",   # Crude Oil WTI    → United States Oil Fund
    "/GC":  "GLD",   # Gold             → SPDR Gold Shares
    "/SI":  "SLV",   # Silver           → iShares Silver Trust
    "/ZN":  "TLT",   # 10-Yr T-Note     → iShares 20+ Year Treasury
    "/ZB":  "TLT",   # 30-Yr T-Bond     → iShares 20+ Year Treasury
    "/NG":  "UNG",   # Natural Gas      → United States Natural Gas
    "/6E":  "FXE",   # Euro FX          → Invesco CurrencyShares Euro
    "/6J":  "FXY",   # Japanese Yen     → Invesco CurrencyShares Yen
    "/VX":  "VXX",   # VIX Futures      → iPath Series B VIX ETN
    "/HG":  "CPER",  # Copper           → United States Copper Index
    "/ZC":  "CORN",  # Corn             → Teucrium Corn Fund
    "/ZS":  "SOYB",  # Soybeans         → Teucrium Soybean Fund
}
_FUTURES_INFO: dict[str, dict] = {
    "/ES":  {"name": "E-mini S&P 500 (SPY proxy)",    "type": "Futures"},
    "/NQ":  {"name": "E-mini Nasdaq 100 (QQQ proxy)", "type": "Futures"},
    "/YM":  {"name": "E-mini Dow Jones (DIA proxy)",  "type": "Futures"},
    "/RTY": {"name": "E-mini Russell 2000 (IWM proxy)","type": "Futures"},
    "/CL":  {"name": "Crude Oil WTI (USO proxy)",     "type": "Futures"},
    "/GC":  {"name": "Gold (GLD proxy)",              "type": "Futures"},
    "/SI":  {"name": "Silver (SLV proxy)",            "type": "Futures"},
    "/ZN":  {"name": "10-Year T-Note (TLT proxy)",    "type": "Futures"},
    "/ZB":  {"name": "30-Year T-Bond (TLT proxy)",    "type": "Futures"},
    "/NG":  {"name": "Natural Gas (UNG proxy)",       "type": "Futures"},
    "/6E":  {"name": "Euro FX (FXE proxy)",           "type": "Futures"},
    "/6J":  {"name": "Japanese Yen (FXY proxy)",      "type": "Futures"},
    "/VX":  {"name": "VIX Futures (VXX proxy)",       "type": "Futures"},
    "/HG":  {"name": "Copper (CPER proxy)",           "type": "Futures"},
    "/ZC":  {"name": "Corn (CORN proxy)",             "type": "Futures"},
    "/ZS":  {"name": "Soybeans (SOYB proxy)",         "type": "Futures"},
}


def _normalize_symbol(sym: str) -> tuple[str, str, str]:
    """
    Normalize raw symbol to (data_ticker, asset_class, proxy_ticker).
    /NQ  -> ("QQQ", "futures", "QQQ")   – use QQQ OHLCV, UI shows futures mode + proxy badge
    QQQ  -> ("QQQ", "equity",  "")
    """
    s = sym.strip().upper()
    if s in _FUTURES_MAP:
        proxy = _FUTURES_MAP[s]
        return proxy, "futures", proxy
    # Already in Polygon continuous format (I:XX1!) – no plan coverage, fall through as equity
    if re.match(r"^I:[A-Z0-9]+1!$", s):
        return s, "futures", s
    return s, "equity", ""


async def _run_alert_pipeline_for_symbol(symbol: str, interval: int = 5) -> None:
    """Run the existing setup→edge→alert pipeline for a subscribed symbol."""
    sym_raw = symbol.upper()
    sym, asset_class, _ = _normalize_symbol(sym_raw)
    end_dt = date.today()
    start_dt = end_dt - timedelta(days=7 if asset_class == "futures" else 5)

    intraday_df = pd.DataFrame()
    try:
        intraday_df = await polygon_client.get_stock_ohlcv(
            sym,
            start_date=start_dt,
            end_date=end_dt,
            interval_min=interval,
        )
    except Exception:
        pass

    if intraday_df.empty:
        try:
            intraday_df = await _data_layer.get_intraday(sym, end_dt, interval_min=interval)
        except Exception:
            return

    if intraday_df.empty:
        return

    underlying_price = float(intraday_df.iloc[-1]["close"])
    if underlying_price <= 0:
        return

    regime_result = None
    if len(intraday_df) >= 5:
        try:
            regime_result = regime_engine.classify(intraday_df, symbol=sym)
        except Exception:
            pass
    if regime_result is None:
        return

    try:
        daily_bars_df = await _data_layer.get_daily(sym)
    except Exception:
        return
    if daily_bars_df.empty:
        return

    try:
        sr_result_obj = sr_engine.analyze(
            daily_df=daily_bars_df,
            intraday_df=intraday_df,
            current_price=underlying_price,
        )
    except Exception:
        return

    expirations: list[date] = []
    chain_data: list[dict] = []
    try:
        expirations = await polygon_client.get_expirations(sym)
        if expirations:
            chain_data = await polygon_client.get_options_chain_snapshot(sym, expirations[0]) or []
    except Exception:
        chain_data = []

    rv = rolling_rv(intraday_df)
    setups = setup_engine.detect_all(
        df=intraday_df,
        regime=regime_result,
        sr=sr_result_obj,
        chain=chain_data,
        symbol=sym,
    )

    edge_results: list[EdgeResult] = []
    for setup in setups:
        pick = option_picker.pick(
            chain=chain_data,
            direction=setup.direction,
            spot=underlying_price,
            edge_score=60,
            expiration_date=expirations[0] if expirations else None,
        )
        scored = edge_scorer.score(
            setup=setup,
            regime=regime_result,
            option=pick,
            df=intraday_df,
            rv=float(rv.iloc[-1]) if len(rv) > 0 else 0.0,
        )
        if scored.tier != "NO_EDGE":
            edge_results.append(scored)

    current_time = datetime.utcnow()
    alert_manager.process_tick(
        edge_results=edge_results,
        current_price=underlying_price,
        current_time=current_time,
        symbol=sym,
    )
    alert_manager.process_tick(
        edge_results=[],
        current_price=underlying_price,
        current_time=current_time,
        symbol=sym,
    )


subscription_scanner = SubscriptionScanner(
    get_symbols=user_alert_store.get_all_subscribed_symbols,
    scan_symbol=_run_alert_pipeline_for_symbol,
    interval_seconds=60,
    max_concurrency=3,
)


@app.get("/api/signals/{symbol:path}")
async def get_signals(
    symbol: str,
    interval: int = Query(1, description="Bar interval in minutes (1, 5, 15)"),
    nocache: bool = Query(False, description="Bypass caches for latest data"),
    poly: PolygonClient = Depends(get_polygon),
    engine: VPAEngine = Depends(get_vpa),
    dl: DataLayer = Depends(get_data_layer),
    sr_eng: SREngine = Depends(get_sr_engine),
) -> SignalResponse:
    """
    New unified signal endpoint.
    Returns regime, active alerts (with entry/stop/target), key S/R levels,
    indicator series, and chain snapshot for the given underlying symbol.
    """
    sym_raw = symbol.upper()
    sym, asset_class, proxy_ticker = _normalize_symbol(sym_raw)
    now_ts = datetime.utcnow().isoformat()

    try:
        end_dt = date.today()
        # Futures trade ~23h/day – fetch a wider window so "today" includes overnight session
        start_dt = end_dt - timedelta(days=7 if asset_class == "futures" else 5)

        if nocache:
            poly.clear_stock_ohlcv_cache(
                symbol=sym, start_date=start_dt, end_date=end_dt, interval_min=interval,
            )

        # ── Parallel fetches ─────────────────────────────────
        intraday_df, daily_bars_df, intraday_5m_df = await asyncio.gather(
            poly.get_stock_ohlcv(sym, start_date=start_dt, end_date=end_dt, interval_min=interval),
            dl.get_daily_bars(sym, num_days=90),
            dl.get_intraday_bars(sym, lookback_days=20, interval_min=5),
        )

        # Underlying price
        underlying_price = 0.0
        if not intraday_df.empty:
            underlying_price = float(intraday_df.iloc[-1]["close"])
        if underlying_price <= 0:
            try:
                underlying_price = await poly.get_underlying_price(sym)
            except Exception:
                pass

        # Premarket bars
        try:
            premarket_df = await poly.get_premarket_bars(sym, end_dt)
        except Exception:
            premarket_df = pd.DataFrame()

        # Nearest expiration chain (equity only – futures have no options chain here)
        chain_data: list[dict] = []
        chain_metrics_obj: ChainMetrics | None = None
        expirations: list = []
        if asset_class == "equity":
            try:
                expirations = await poly.get_expirations(sym)
                if expirations:
                    nearest_exp = expirations[0]
                    chain_data = await poly.get_options_chain_snapshot(sym, nearest_exp)
                    if chain_data and underlying_price > 0:
                        chain_metrics_obj = compute_chain_metrics(chain_data, underlying_price)
            except Exception as ce:
                print(f"[Signals] Chain fetch non-fatal: {ce}")

        # ── Regime ───────────────────────────────────────────
        regime_result = None
        if not intraday_df.empty and len(intraday_df) >= 10:
            try:
                regime_result = regime_engine.classify(intraday_df, symbol=sym)
            except Exception as re_err:
                print(f"[Signals] Regime error: {re_err}")

        # ── S/R ──────────────────────────────────────────────
        sr_result_obj: SRResult | None = None
        if underlying_price > 0 and not daily_bars_df.empty:
            try:
                sr_result_obj = sr_eng.analyze(
                    daily_bars=daily_bars_df,
                    underlying_price=underlying_price,
                    intraday_bars=intraday_5m_df,
                    premarket_bars=premarket_df,
                )
            except Exception as sr_err:
                print(f"[Signals] SR error: {sr_err}")

        # ── VPA ──────────────────────────────────────────────
        vpa_bias = None
        vpa_signals_out: list[dict] = []
        if not intraday_df.empty:
            try:
                vpa_results = engine.analyze(intraday_df)
                vpa_bias = engine.get_bias(vpa_results)
                vpa_signals_out = [
                    dict(
                        signal=r.signal.value,
                        confidence=r.confidence,
                        description=r.description,
                        datetime=r.datetime,
                    )
                    for r in vpa_results
                    if r.signal != VPASignal.NEUTRAL
                ]
            except Exception:
                pass

        # ── Fair value (equity only) ─────────────────────────
        fv_response = None
        if asset_class == "equity":
            try:
                daily_closes_fv = await poly.get_stock_daily_ohlcv(sym, num_days=45)
                if daily_closes_fv and underlying_price > 0 and chain_data:
                    exp_date_fv = expirations[0] if expirations else None
                if exp_date_fv:
                    # Use nearest ATM call for IV reference
                    atm_c = min(
                        (c for c in chain_data if c.get("contract_type", "").lower() == "call"),
                        key=lambda c: abs(c["strike"] - underlying_price),
                        default=None,
                    )
                    if atm_c:
                        fv_result = fair_value_engine.analyze(
                            underlying_price=underlying_price,
                            strike=float(atm_c["strike"]),
                            expiration=exp_date_fv,
                            contract_type="C",
                            daily_closes=daily_closes_fv,
                            market_bid=float(atm_c.get("bid", 0) or 0),
                            market_ask=float(atm_c.get("ask", 0) or 0),
                            market_iv=float(atm_c.get("iv", 0) or 0),
                        )
                        fv_response = FairValueResponse(
                            underlying_price=fv_result.underlying_price,
                            strike=fv_result.strike,
                            time_to_expiry=fv_result.time_to_expiry,
                            risk_free_rate=fv_result.risk_free_rate,
                            historical_vol=fv_result.historical_vol,
                            dividend_yield=fv_result.dividend_yield,
                            contract_type=fv_result.contract_type,
                            theoretical_price=fv_result.theoretical_price,
                            d1=fv_result.d1,
                            d2=fv_result.d2,
                            market_price=fv_result.market_price,
                            market_iv=fv_result.market_iv,
                            price_difference=fv_result.price_difference,
                            pct_difference=fv_result.pct_difference,
                            is_cheap=fv_result.is_cheap,
                            is_expensive=fv_result.is_expensive,
                            verdict=fv_result.verdict,
                            detail=fv_result.detail,
                            bid=fv_result.bid,
                            ask=fv_result.ask,
                            spread=fv_result.spread,
                            spread_pct=fv_result.spread_pct,
                            hv_vs_iv=fv_result.hv_vs_iv,
                        )
            except Exception as fv_err:
                print(f"[Signals] Fair value non-fatal: {fv_err}")

        # ── Setup Detection → Edge Score → Risk Plan → Alerts ─
        if (
            regime_result is not None
            and sr_result_obj is not None
            and not intraday_df.empty
            and underlying_price > 0
        ):
            try:
                rv = rolling_rv(intraday_df)
                setups = setup_engine.detect_all(
                    df=intraday_df,
                    regime=regime_result,
                    sr=sr_result_obj,
                    chain=chain_data,
                    symbol=sym,
                )

                edge_results: list[EdgeResult] = []
                for setup in setups:
                    # Pick best contract
                    pick = option_picker.pick(
                        chain=chain_data,
                        direction=setup.direction,
                        spot=underlying_price,
                        edge_score=60,   # preliminary — will update after scoring
                        expiration_date=expirations[0] if expirations else None,
                    )
                    scored = edge_scorer.score(
                        setup=setup,
                        regime=regime_result,
                        option=pick,
                        df=intraday_df,
                        rv=float(rv.iloc[-1]) if len(rv) > 0 else 0.0,
                    )
                    if scored.tier != "NO_EDGE":
                        edge_results.append(scored)

                # Feed into alert manager
                current_time = datetime.utcnow()
                new_alerts = alert_manager.process_tick(
                    edge_results=edge_results,
                    current_price=underlying_price,
                    current_time=current_time,
                    symbol=sym,
                )
                # Evaluate active alerts for kill-switch conditions
                alert_manager.process_tick(
                    edge_results=[],
                    current_price=underlying_price,
                    current_time=current_time,
                    symbol=sym,
                )
            except Exception as pipe_err:
                import traceback
                print(f"[Signals] Pipeline error: {pipe_err}")
                traceback.print_exc()

        # ── Build indicator series ──────────────────────────
        indicators_response: IndicatorsResponse | None = None
        if not intraday_df.empty and len(intraday_df) >= 5:
            try:
                _vwap_s = vwap(intraday_df)
                _vwap_u, _vwap_l = vwap_bands(intraday_df, 1.0)
                _ema9 = ema(intraday_df["close"], 9)
                _ema20 = ema(intraday_df["close"], 20)
                _rsi_s = rsi(intraday_df)
                _atr = atr_value(intraday_df)

                def _to_floats(s: pd.Series) -> list[float]:
                    return [round(float(v), 4) if not pd.isna(v) else 0.0 for v in s]

                indicators_response = IndicatorsResponse(
                    vwap=_to_floats(_vwap_s),
                    vwap_upper=_to_floats(_vwap_u),
                    vwap_lower=_to_floats(_vwap_l),
                    ema_9=_to_floats(_ema9),
                    ema_20=_to_floats(_ema20),
                    rsi=_to_floats(_rsi_s),
                    atr=round(_atr, 4),
                )
            except Exception as ind_err:
                print(f"[Signals] Indicators error: {ind_err}")

        # ── Build response ────────────────────────────────────
        bars_out = [
            OHLCVBar(
                datetime=str(row["datetime"]),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=int(row["volume"]),
            )
            for _, row in intraday_df.iterrows()
        ] if not intraday_df.empty else []

        regime_out: RegimeResponse | None = None
        if regime_result:
            regime_out = RegimeResponse(
                regime=regime_result.regime.value,
                confidence=round(regime_result.confidence, 3),
                detail=regime_result.detail,
                vwap_current=round(regime_result.vwap_current, 4),
                atr_current=round(regime_result.atr_current, 4),
                rsi_current=round(regime_result.rsi_current, 2),
                price_vs_vwap=regime_result.price_vs_vwap,
            )

        key_levels_out: list[SRLevelResponse] = []
        if sr_result_obj:
            key_levels_out = [
                SRLevelResponse(
                    price=l.price, kind=l.kind, source=l.source,
                    strength=l.strength, label=l.label,
                )
                for l in sr_result_obj.levels[:30]
            ]

        chain_snap_out: ChainSnapshotResponse | None = None
        if chain_metrics_obj:
            chain_snap_out = ChainSnapshotResponse(
                iv_rank=round(chain_metrics_obj.iv_rank, 2),
                iv_percentile=round(chain_metrics_obj.iv_percentile, 2),
                put_call_oi_ratio=round(chain_metrics_obj.put_call_oi_ratio, 3),
                put_call_volume_ratio=round(chain_metrics_obj.put_call_volume_ratio, 3),
                net_gex=round(chain_metrics_obj.net_gex, 0),
                gex_regime=chain_metrics_obj.gex_regime,
                max_pain=round(chain_metrics_obj.max_pain, 2),
                uoa_detected=chain_metrics_obj.uoa_detected,
                uoa_details=getattr(chain_metrics_obj, 'uoa_details', [])[:5],
                call_wall=round(chain_metrics_obj.call_wall, 2),
                put_wall=round(chain_metrics_obj.put_wall, 2),
                weighted_iv=round(chain_metrics_obj.weighted_iv, 4),
            )

        # Collect active alerts
        active_alerts_out = [_build_alert_response(alert) for alert in alert_manager.get_active_alerts()]

        # ── Volume regime ────────────────────────────────────
        vol_regime_out: dict | None = None
        if not intraday_df.empty:
            try:
                vol_regime_out = engine.get_volume_regime(intraday_df)
            except Exception:
                pass

        # ── Proximity ────────────────────────────────────────
        prox_score_out = 0.0
        prox_detail_out = ""
        if sr_result_obj:
            prox_score_out = sr_result_obj.proximity_score
            prox_detail_out = sr_result_obj.proximity_detail

        # ── Previous close (yesterday) ────────────────────
        prev_close_val = 0.0
        if not daily_bars_df.empty:
            today_str = date.today().isoformat()
            # daily_bars_df may have "date" or "datetime" column
            date_col = "datetime" if "datetime" in daily_bars_df.columns else "date"
            prev_days = daily_bars_df[
                daily_bars_df[date_col].astype(str).str[:10] < today_str
            ]
            if not prev_days.empty:
                prev_close_val = float(prev_days.iloc[-1]["close"])

        return SignalResponse(
            symbol=sym_raw,
            as_of=now_ts,
            asset_class=asset_class,
            proxy_ticker=proxy_ticker,
            bars=bars_out,
            regime=regime_out,
            active_alerts=active_alerts_out,
            key_levels=key_levels_out,
            indicators=indicators_response,
            chain_snapshot=chain_snap_out,
            vpa={"bias": vpa_bias, "signals": vpa_signals_out} if vpa_bias else None,
            fair_value=fv_response,
            volume_regime=vol_regime_out,
            proximity_score=round(prox_score_out, 3),
            proximity_detail=prox_detail_out,
            prev_close=prev_close_val,
        )

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/alerts/{alert_id}/activate")
async def activate_alert(alert_id: str, premium: float = Query(...)):
    """Mark an alert as activated (entered) at a given premium."""
    alert = alert_manager.mark_activated(alert_id, premium)
    if not alert:
        raise HTTPException(status_code=404, detail=f"Alert not found: {alert_id}")
    return {"ok": True}


@app.post("/api/alerts/{alert_id}/resolve")
async def resolve_alert(
    alert_id: str,
    state: str = Query(..., description="STOPPED | HIT_T1 | HIT_T2 | TIMED_OUT"),
    exit_premium: float = Query(0.0),
):
    """Resolve an active alert (stopped, target hit, etc.)."""
    try:
        alert_state = AlertState(state.upper())
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Unknown state: {state}")

    # Look up alert before resolving for perf tracking
    alert_obj = alert_manager.get_active_alert(alert_id)
    if not alert_obj:
        raise HTTPException(status_code=404, detail=f"Alert not found: {alert_id}")

    entry_prem = alert_obj.entry_premium
    plan = alert_obj.plan

    resolved_alert = alert_manager.resolve_alert(alert_id, alert_state, exit_premium)
    if not resolved_alert:
        raise HTTPException(status_code=404, detail=f"Alert not found: {alert_id}")

    # Record win/loss in risk engine for capital mode
    is_win = alert_state in (AlertState.HIT_T1, AlertState.HIT_T2)
    risk_engine.record_outcome(is_win)

    # Log trade to MongoDB (fire-and-forget)
    try:
        pnl = exit_premium - entry_prem if entry_prem else 0.0
        asyncio.create_task(perf_tracker.log_trade(
            symbol=resolved_alert.symbol or "QQQ",
            direction=resolved_alert.direction,
            setup_type=resolved_alert.setup_name,
            entry_price=entry_prem,
            exit_price=exit_premium,
            stop_price=plan.stop_price if plan else 0.0,
            target_price=plan.target_1 if plan else 0.0,
            pnl=pnl,
            alert_id=alert_id,
        ))
    except Exception:
        pass  # don't block resolve on logging failures

    return {"ok": True}


@app.post("/api/alerts/reset")
async def reset_alerts():
    """Clear all alerts and reset the alert manager (admin use)."""
    alert_manager.reset()
    regime_engine.reset()
    return {"ok": True}


@app.get("/api/alerts/history", response_model=list[AlertHistoryResponse])
async def get_alert_history(
    symbol: str = Query(..., description="Underlying symbol"),
    alert_date: Optional[str] = Query(None),
):
    """Return actionable alerts for one symbol and trading day."""
    if not alert_date:
        alert_date = date.today().isoformat()
    try:
        day = date.fromisoformat(alert_date)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid alert_date: {alert_date}")

    docs = await alert_store.get_day_history(symbol, day)
    if docs:
        return [_build_alert_history_response_from_doc(doc) for doc in docs]

    history = []
    symbol_upper = symbol.upper()
    day_str = day.isoformat()
    for alert in alert_manager.get_all_alerts():
        if (alert.symbol or "").upper() != symbol_upper:
            continue
        if not (alert.detected_at or "").startswith(day_str):
            continue
        history.append(_build_alert_history_response(alert))

    history.sort(key=lambda item: item.detected_at)
    return history


# ── Performance & Scanner endpoints ──────────────────────────

@app.get("/api/performance/stats")
async def get_performance_stats(
    symbol: str = Query("", description="Filter by symbol (optional)"),
    days: int = Query(30, ge=1, le=365),
):
    """Return trade performance stats from MongoDB."""
    try:
        stats = await perf_tracker.get_stats(symbol=symbol, days=days)
        by_setup = await perf_tracker.get_stats_by_setup(days=days)
        stats["by_setup"] = by_setup
        return stats
    except Exception as e:
        logger.warning("Performance stats error: %s", e)
        return {"total_trades": 0, "wins": 0, "losses": 0, "win_rate": 0,
                "avg_r": 0, "total_pnl": 0, "expectancy": 0, "best_r": 0,
                "worst_r": 0, "avg_score": 0, "recent_trades": [], "by_setup": []}

@app.get("/api/performance/session")
async def get_session_stats():
    """Return today's session stats from risk_engine (in-memory)."""
    return risk_engine.get_session_stats()

@app.get("/api/tickers")
async def search_tickers(
    q: str = Query("", description="Search query"),
    limit: int = Query(20, ge=1, le=50),
):
    """Autocomplete ticker search. Prepends futures when query starts with '/'."""
    q_clean = q.strip()
    results: list[dict] = []
    if q_clean.startswith("/"):
        q_up = q_clean.upper()
        futures_hits = [
            {"symbol": sym, "name": info["name"], "type": info["type"]}
            for sym, info in _FUTURES_INFO.items()
            if sym.startswith(q_up) or q_up == "/"
        ]
        results.extend(futures_hits)
    equity_hits = ticker_service.search(q_clean, limit=limit)
    results.extend(equity_hits)
    return results[:limit]


@app.get("/api/scanner")
async def run_scanner(
    symbols: str = Query("QQQ,SPY,NVDA,TSLA,AMD,META", description="Comma-separated symbols"),
):
    """On-demand multi-symbol scan."""
    sym_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    try:
        results = await scanner_engine.scan(
            symbols=sym_list,
            polygon_client=polygon_client,
            data_layer=None,
        )
        return [r.to_dict() for r in results]
    except Exception as e:
        logger.warning("Scanner error: %s", e)
        return []


# ── User-submitted Ideas endpoints ───────────────────────────

class IdeaSubmitRequest(BaseModel):
    symbol: str
    expiration: str
    strike: float
    right: str          # "C" or "P"
    price: float        # contract price at time of submission
    action: str         # e.g. "BUY", "SELL"
    note: str = ""      # optional note from the user


@app.post("/api/ideas/submit")
async def submit_idea(body: IdeaSubmitRequest, request: Request):
    """Submit a trade idea. Admin-only."""
    user = getattr(request.state, "user", None)
    if not user or user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Only admins can post ideas")

    dte = 0
    try:
        exp_date = datetime.strptime(body.expiration, "%Y-%m-%d").date()
        dte = (exp_date - date.today()).days
    except Exception:
        pass

    exp_short = ""
    try:
        exp_short = datetime.strptime(body.expiration, "%Y-%m-%d").strftime("%b %d")
    except Exception:
        exp_short = body.expiration

    idea = {
        "symbol": body.symbol.upper(),
        "expiration": body.expiration,
        "exp_label": f"{exp_short} ({dte}d)",
        "strike": body.strike,
        "right": body.right.upper(),
        "type_label": "CALL" if body.right.upper() == "C" else "PUT",
        "price": round(body.price, 2),
        "action": body.action.upper(),
        "note": body.note,
        "posted_by": user["display_name"],
    }
    result = await _ideas_hub.submit(idea)
    return {"ok": True, "idea": result}


@app.get("/api/ideas")
async def get_user_ideas():
    """Get today's user-submitted ideas."""
    ideas = _ideas_hub.get_ideas()
    return {"ideas": ideas}


# ── Ideas WebSocket (broadcasts new ideas to all clients) ────

@app.websocket("/ws/ideas")
async def websocket_ideas(ws: WebSocket):
    user = await _require_websocket_user(ws)
    if not user:
        return
    await ws.accept()
    cid, queue = await _ideas_hub.register()

    async def _sender():
        try:
            while True:
                msg = await queue.get()
                await ws.send_text(msg)
        except (WebSocketDisconnect, Exception):
            pass

    sender_task = asyncio.create_task(_sender())
    try:
        # Keep alive; client doesn't send messages
        while True:
            await ws.receive_text()
    except (WebSocketDisconnect, Exception):
        pass
    finally:
        sender_task.cancel()
        await _ideas_hub.unregister(cid)


# ── OLD algorithmic ideas (kept for reference but endpoint replaced above) ──
import time as _time

_ideas_cache: dict[str, dict] = {}   # sym → {"ideas": [...], "ts": float}
_IDEAS_TTL = 300  # 5 minutes


@app.get("/api/ideas/scan")
async def get_trade_ideas_scan(
    symbols: str = Query("QQQ,SPY,IWM,AAPL,MSFT,NVDA,TSLA,AMD,AMZN,META,GOOGL"),
    refresh: bool = Query(False, description="Force bypass cache"),
    poly: PolygonClient = Depends(get_polygon),
    engine: VPAEngine = Depends(get_vpa),
):
    """
    Scan watchlist symbols for trade ideas using the new signal engine.
    Uses VPA bias + chain metrics + liquidity gates to surface actionable setups.
    Results are cached per-symbol for 5 minutes.
    """
    symbol_list = [s.strip().upper() for s in symbols.split(",")]
    ideas: list[dict] = []
    now = _time.time()

    symbols_to_scan: list[str] = []
    for sym in symbol_list:
        cached = _ideas_cache.get(sym)
        if not refresh and cached and (now - cached["ts"]) < _IDEAS_TTL:
            ideas.extend(cached["ideas"])
        else:
            symbols_to_scan.append(sym)

    _sym_ideas: dict[str, list[dict]] = {sym: [] for sym in symbols_to_scan}

    async def scan_symbol(sym: str):
        try:
            expirations = await poly.get_expirations(sym)
            if not expirations:
                return

            price_data = await poly.get_snapshot_prices([sym])
            snap_data = price_data.get(sym, {})
            underlying_price = float(snap_data.get("lastPrice", 0) or snap_data.get("prevClose", 0))
            if underlying_price == 0:
                return

            # VPA direction bias
            vpa_direction: str | None = None
            try:
                stock_df = await poly.get_stock_ohlcv(
                    symbol=sym, start_date=date.today(), end_date=date.today(), interval_min=5,
                )
                if not stock_df.empty and len(stock_df) >= 2:
                    vpa_results = engine.analyze(stock_df)
                    bias = engine.get_bias(vpa_results)
                    b = bias.get("bias", "neutral") if bias else "neutral"
                    if b in ("bullish", "strong_bullish"):
                        vpa_direction = "CALL"
                    elif b in ("bearish", "strong_bearish"):
                        vpa_direction = "PUT"
            except Exception:
                pass

            if not vpa_direction:
                return

            exp = expirations[0]
            dte = (exp - date.today()).days
            if dte < 0:
                return

            chain = await poly.get_options_chain_snapshot(sym, exp)
            if not chain:
                return

            ct_filter = "call" if vpa_direction == "CALL" else "put"
            candidates = [
                c for c in chain
                if c.get("contract_type", "").lower() == ct_filter
                and abs(c.get("strike", 0) - underlying_price) / underlying_price < 0.03
                and (c.get("volume", 0) or 0) >= 50
                and (c.get("open_interest", 0) or 0) >= 100
            ]
            if not candidates:
                return

            best = min(candidates, key=lambda c: abs(c.get("strike", 0) - underlying_price))
            last_price = best.get("last_price", 0) or 0
            if last_price < 0.10:
                return

            delta = abs(float((best.get("greeks") or {}).get("delta", 0) or 0))
            exp_str = exp.strftime("%Y-%m-%d")
            exp_short = exp.strftime("%b %d")
            action = "BUY" if vpa_direction == "CALL" else "BUY PUT"
            idea_dict = {
                "symbol": sym,
                "expiration": exp_str,
                "exp_label": f"{exp_short} ({dte}d)",
                "strike": best["strike"],
                "right": "C" if vpa_direction == "CALL" else "P",
                "type_label": vpa_direction,
                "price": round(last_price, 2),
                "signal": "call_bias" if vpa_direction == "CALL" else "put_bias",
                "action": action,
                "score": round(delta, 3),
                "confidence": round(delta, 3),
                "archetype": "momentum",
                "recommendation": f"VPA {vpa_direction} bias. Delta {delta:.2f}, {dte}d exp.",
            }
            ideas.append(idea_dict)
            _sym_ideas[sym].append(idea_dict)

        except Exception as e:
            print(f"[IDEAS] Error scanning {sym}: {e}")

    if symbols_to_scan:
        sem = asyncio.Semaphore(3)

        async def throttled_scan(sym):
            async with sem:
                await scan_symbol(sym)

        await asyncio.gather(*(throttled_scan(sym) for sym in symbols_to_scan))

        for sym in symbols_to_scan:
            _ideas_cache[sym] = {"ideas": _sym_ideas[sym], "ts": _time.time()}

    ideas.sort(key=lambda x: abs(x["score"]) * x["confidence"], reverse=True)
    return {"ideas": ideas[:20]}


@app.get("/api/watchlist/prices")
async def get_watchlist_prices(
    symbols: str = Query("SPY,QQQ,IWM,AAPL,MSFT,NVDA,TSLA,AMD"),
    mock: bool = Query(False, description="Ignored – always uses real API"),
    poly: PolygonClient = Depends(get_polygon),
):
    """Get real-time/delayed prices for watchlist symbols using snapshot API.
    Note: mock parameter is accepted but ignored – stock prices always come from real API."""
    try:
        symbol_list = [s.strip().upper() for s in symbols.split(",")]

        # Always use real API for stock prices (even in mock mode)
        # Use snapshot API for real-time/15-min delayed prices
        # Falls back to previous-day close if snapshot returns 403 (plan limitation)
        price_lookup = await poly.get_snapshot_prices(symbol_list)
        if not price_lookup:
            price_lookup = await poly.get_prev_close_prices(symbol_list)

        results = []
        for sym in symbol_list:
            snapshot = price_lookup.get(sym, {})
            last_price = float(snapshot.get("lastPrice", 0))
            prev_close = float(snapshot.get("prevClose", 0))
            # Fallback to previous close when no trade in current session
            if last_price == 0 and prev_close > 0:
                last_price = prev_close
            change = float(snapshot.get("todaysChange", 0))
            change_pct = float(snapshot.get("todaysChangePerc", 0))
            high = float(snapshot.get("dayHigh", 0))
            low = float(snapshot.get("dayLow", 0))
            volume = int(snapshot.get("dayVolume", 0))

            results.append(
                {
                    "symbol": sym,
                    "price": last_price,
                    "change": round(change, 2),
                    "changePct": round(change_pct, 2),
                    "high": high,
                    "low": low,
                    "volume": volume,
                }
            )

        return {"prices": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── WebSocket price ticker ──────────────────────────────────

@app.websocket("/ws/prices")
async def websocket_prices(ws: WebSocket):
    """
    Lightweight price-tick WebSocket for the dashboard.

    Client → server:
      {"action": "sub",   "tickers": ["QQQ", "O:QQQ260225C00610000"]}
      {"action": "unsub", "tickers": ["O:QQQ260225C00610000"]}

    Server → client (for every completed 1-minute bar from live_feed):
      {"type": "tick", "ticker": "QQQ",   "price": 512.34, "prev": 512.10, "change": 0.24}
      {"type": "tick", "ticker": "O:...", "price": 3.45,   "prev": 3.30,   "change": 0.15}
    """
    user = await _require_websocket_user(ws)
    if not user:
        return
    await ws.accept()
    queue: asyncio.Queue = asyncio.Queue(maxsize=300)
    subscribed: set[str] = set()
    last_prices: dict[str, float] = {}

    async def _sender():
        try:
            while True:
                raw = await queue.get()
                try:
                    msg = json.loads(raw)
                    if msg.get("type") not in ("bar", "analysis"):
                        continue
                    ticker = msg.get("ticker", "")
                    if not ticker:
                        continue
                    # bar messages carry the close price
                    bar = msg.get("bar")
                    if bar:
                        price = float(bar.get("close", 0) or 0)
                    else:
                        # analysis messages carry last_price
                        price = float(msg.get("last_price", 0) or 0)
                    if not price:
                        continue
                    prev = last_prices.get(ticker, price)
                    last_prices[ticker] = price
                    await ws.send_text(json.dumps({
                        "type": "tick",
                        "ticker": ticker,
                        "price": round(price, 4),
                        "prev": round(prev, 4),
                        "change": round(price - prev, 4),
                    }))
                except Exception:
                    pass
        except Exception:
            pass

    sender_task = asyncio.create_task(_sender())
    try:
        while True:
            raw = await ws.receive_text()
            try:
                cmd = json.loads(raw)
            except Exception:
                continue
            action = cmd.get("action", "")
            tickers = [str(t) for t in cmd.get("tickers", []) if t]
            if action == "sub":
                for t in tickers:
                    if t not in subscribed:
                        try:
                            await live_feed_manager.subscribe(t, queue)
                        except Exception as e:
                            await ws.send_text(json.dumps({
                                "type": "error", "ticker": t, "message": str(e)
                            }))
                            continue
                        subscribed.add(t)
                await ws.send_text(json.dumps({
                    "type": "subbed", "tickers": list(subscribed)
                }))
            elif action == "unsub":
                for t in tickers:
                    if t in subscribed:
                        try:
                            await live_feed_manager.unsubscribe(t, queue)
                        except Exception:
                            pass
                        subscribed.discard(t)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[PriceWS] Error: {e}")
    finally:
        sender_task.cancel()
        for t in list(subscribed):
            try:
                await live_feed_manager.unsubscribe(t, queue)
            except Exception:
                pass


# ── WebSocket live feed ─────────────────────────────────────

@app.websocket("/ws/live")
async def websocket_live(ws: WebSocket):
    """
    WebSocket endpoint for live option feed.

    Client sends JSON messages:
      {"action": "subscribe", "ticker": "O:SPY251219C00650000"}
      {"action": "unsubscribe", "ticker": "O:SPY251219C00650000"}

    Server pushes:
      {"type": "bar", "ticker": "...", "bar": {...}, "signals": [...]}
      {"type": "sow", "ticker": "...", "bars": [...]}   (snapshot on subscribe)
      {"type": "analysis", ...}   (periodic full re-analysis for option tickers)
      {"type": "status", "connected": true, "subscriptions": {...}}
    """
    user = await _require_websocket_user(ws)
    if not user:
        return
    await ws.accept()

    # Per-client message queue
    queue: asyncio.Queue = asyncio.Queue(maxsize=500)
    subscribed_tickers: set[str] = set()
    mock_tasks: dict[str, asyncio.Task] = {}   # ticker → mock playback task
    analysis_task: asyncio.Task | None = None   # periodic analysis for option ticker

    def _parse_occ_ticker(occ: str):
        """Parse O:SPY260212C00650000 → (symbol, expiration, strike, right)."""
        try:
            raw = occ.replace("O:", "")
            # Find where the date starts (6 digits after symbol)
            import re
            m = re.match(r'^([A-Z]+)(\d{6})([CP])(\d{8})$', raw)
            if not m:
                return None
            sym = m.group(1)
            dt_str = m.group(2)  # YYMMDD
            right = m.group(3)
            strike = int(m.group(4)) / 1000
            exp = f"20{dt_str[:2]}-{dt_str[2:4]}-{dt_str[4:6]}"
            return sym, exp, strike, right
        except Exception:
            return None

    async def _analysis_loop(ticker: str, q: asyncio.Queue):
        """Periodically run full VPA + greeks analysis and push to client."""
        parsed = _parse_occ_ticker(ticker)
        if not parsed:
            return
        symbol, expiration, strike, right = parsed
        exp_date = datetime.strptime(expiration, "%Y-%m-%d").date()
        end_dt = date.today()
        start_dt = end_dt - timedelta(days=5)

        await asyncio.sleep(3)  # let WS settle before first analysis

        while True:
            try:
                # Clear caches for fresh data
                polygon_client.clear_ohlcv_cache(
                    symbol=symbol, expiration=exp_date,
                    strike=strike, right=right,
                    start_date=start_dt, end_date=end_dt, interval_min=5,
                )

                # Fetch option OHLCV
                df = await polygon_client.get_option_ohlcv(
                    symbol=symbol, expiration=exp_date,
                    strike=strike, right=right,
                    start_date=start_dt, end_date=end_dt, interval_min=5,
                )
                if df.empty:
                    await asyncio.sleep(10)
                    continue

                # Full VPA analysis
                vpa_results = vpa_engine.analyze(df)
                bias = vpa_engine.get_bias(vpa_results)
                last_vpa_signal_str = vpa_results[-1].signal.value if vpa_results else None

                # ── Underlying trend + volume regime for new engine params
                underlying_trend = None
                vol_regime = None
                try:
                    stock_df = await polygon_client.get_stock_ohlcv(
                        symbol=symbol, start_date=start_dt,
                        end_date=end_dt, interval_min=5,
                    )
                    if not stock_df.empty and len(stock_df) >= 2:
                        day_open = stock_df.iloc[0]["open"]
                        day_close = stock_df.iloc[-1]["close"]
                        pct = (day_close - day_open) / day_open * 100 if day_open else 0
                        if pct > 0.50:    underlying_trend = "STRONG_UP"
                        elif pct > 0.15:  underlying_trend = "UP"
                        elif pct < -0.50: underlying_trend = "STRONG_DOWN"
                        elif pct < -0.15: underlying_trend = "DOWN"
                        else:             underlying_trend = "FLAT"
                        vol_regime_info = vpa_engine.get_volume_regime(stock_df)
                        vol_regime = vol_regime_info.get("regime")
                except Exception:
                    pass

                dte = (exp_date - date.today()).days

                all_signals = [
                    dict(
                        signal=r.signal.value,
                        confidence=r.confidence,
                        description=r.description,
                        datetime=r.datetime,
                        price=r.price,
                        volume=r.volume,
                        volume_ratio=round(r.volume_ratio, 2),
                    )
                    for r in vpa_results
                    if r.signal != VPASignal.NEUTRAL
                ]

                payload: dict = {
                    "type": "analysis",
                    "ticker": ticker,
                    "bias": bias,
                    "all_signals": all_signals,
                    "total_bars": len(df),
                    "last_price": float(df.iloc[-1]["close"]),
                }

                # ── New engine: regime → setupdetect → edge → alerts ─
                try:
                    if hasattr(polygon_client, '_contract_snapshot_cache'):
                        snap_key = ("snapshot", symbol, exp_date, strike, right)
                        polygon_client._contract_snapshot_cache.pop(snap_key, None)

                    _und_bars, _daily_bars, _chain_snap_data, _daily_fv = await asyncio.gather(
                        polygon_client.get_stock_ohlcv(
                            symbol=symbol, start_date=start_dt, end_date=end_dt, interval_min=5,
                        ),
                        _data_layer.get_daily_bars(symbol, num_days=90),
                        polygon_client.get_options_chain_snapshot(symbol, exp_date),
                        polygon_client.get_stock_daily_ohlcv(symbol, num_days=45),
                    )

                    _und_price = float(_und_bars.iloc[-1]["close"]) if not _und_bars.empty else 0.0

                    # Regime
                    _regime = None
                    if not _und_bars.empty and len(_und_bars) >= 10:
                        try:
                            _regime = regime_engine.classify(_und_bars, symbol=symbol)
                        except Exception:
                            pass

                    # S/R
                    _sr = None
                    if _und_price > 0 and not _daily_bars.empty:
                        try:
                            _intra5 = await _data_layer.get_intraday_bars(symbol, lookback_days=20, interval_min=5)
                            _sr = sr_engine.analyze(
                                daily_bars=_daily_bars,
                                underlying_price=_und_price,
                                intraday_bars=_intra5,
                            )
                        except Exception:
                            pass

                    # Chain metrics
                    _chain_m = None
                    if _chain_snap_data and _und_price > 0:
                        try:
                            _chain_m = compute_chain_metrics(_chain_snap_data, _und_price)
                        except Exception:
                            pass

                    # Setups → edge → alerts
                    if _regime and _sr and not _und_bars.empty:
                        try:
                            _rv = rolling_rv(_und_bars)
                            _setups = setup_engine.detect_all(
                                df=_und_bars, regime=_regime, sr=_sr,
                                chain=_chain_snap_data or [], symbol=symbol,
                            )
                            _edge_results = []
                            _exps = await polygon_client.get_expirations(symbol)
                            for _s in _setups:
                                _pick = option_picker.pick(
                                    chain=_chain_snap_data or [],
                                    direction=_s.direction,
                                    spot=_und_price,
                                    edge_score=60,
                                    expiration_date=_exps[0] if _exps else None,
                                )
                                _scored = edge_scorer.score(
                                    setup=_s, regime=_regime, option=_pick,
                                    df=_und_bars,
                                    rv=float(_rv.iloc[-1]) if len(_rv) > 0 else 0.0,
                                )
                                if _scored.tier != "NO_EDGE":
                                    _edge_results.append(_scored)
                            _now = datetime.utcnow()
                            alert_manager.process_tick(
                                edge_results=_edge_results,
                                current_price=_und_price,
                                current_time=_now,
                                symbol=symbol,
                            )
                        except Exception as _pe:
                            print(f"[WS-Analysis] Pipeline error: {_pe}")

                    # Fair value on ATM call for reference
                    fv_result = None
                    if _daily_fv and _und_price > 0 and _chain_snap_data:
                        try:
                            _atm_c = min(
                                (c for c in _chain_snap_data if c.get("contract_type","").lower()=="call"),
                                key=lambda c: abs(c["strike"] - _und_price), default=None,
                            )
                            if _atm_c:
                                fv_result = fair_value_engine.analyze(
                                    underlying_price=_und_price,
                                    strike=float(_atm_c["strike"]),
                                    expiration=exp_date,
                                    contract_type="C",
                                    daily_closes=_daily_fv,
                                    market_bid=float(_atm_c.get("bid", 0) or 0),
                                    market_ask=float(_atm_c.get("ask", 0) or 0),
                                    market_iv=float(_atm_c.get("iv", 0) or 0),
                                )
                        except Exception:
                            pass

                    # Enrich payload
                    if _regime:
                        payload["regime"] = {
                            "regime": _regime.regime.value,
                            "confidence": round(_regime.confidence, 3),
                            "detail": _regime.detail,
                            "vwap_current": round(_regime.vwap_current, 4),
                            "rsi": round(_regime.rsi_current, 2),
                        }
                    if _chain_m:
                        payload["chain_snapshot"] = {
                            "iv_rank": round(_chain_m.iv_rank, 2),
                            "put_call_oi_ratio": round(_chain_m.put_call_oi_ratio, 3),
                            "gex_regime": _chain_m.gex_regime,
                            "max_pain": round(_chain_m.max_pain, 2),
                            "uoa_detected": _chain_m.uoa_detected,
                        }
                    payload["active_alerts"] = [
                        a.to_dict() for a in alert_manager.get_active_alerts()
                    ]
                    if fv_result:
                        payload["fair_value"] = {
                            "verdict": fv_result.verdict,
                            "theoretical_price": fv_result.theoretical_price,
                            "historical_vol": fv_result.historical_vol,
                            "market_iv": fv_result.market_iv,
                        }

                except Exception as comp_err:
                    print(f"[WS-Analysis] Engine error (non-fatal): {comp_err}")

                # Push to client queue (drop if full)
                try:
                    q.put_nowait(json.dumps(payload))
                except asyncio.QueueFull:
                    pass

            except asyncio.CancelledError:
                return
            except Exception as e:
                print(f"[WS-Analysis] Error: {e}")

            await asyncio.sleep(10)

    async def _sender():
        """Forward messages from the queue to the WebSocket client."""
        try:
            while True:
                msg = await queue.get()
                await ws.send_text(msg)
        except WebSocketDisconnect:
            pass
        except Exception as e:
            print(f"[WS-Sender] Error: {e}")
            import traceback
            traceback.print_exc()

    sender_task = asyncio.create_task(_sender())

    try:
        # Send initial status
        status = live_feed_manager.get_status()
        await ws.send_text(json.dumps({"type": "status", **status}))

        # Read client commands
        while True:
            raw = await ws.receive_text()
            try:
                cmd = json.loads(raw)
            except json.JSONDecodeError:
                await ws.send_text(json.dumps({"type": "error", "message": "Invalid JSON"}))
                continue

            action = cmd.get("action", "")
            ticker = cmd.get("ticker", "")

            if action == "subscribe" and ticker:
                is_mock = cmd.get("mock", False)

                if is_mock:
                    # Per-session mock: CSV playback directly into client queue
                    interval_min = cmd.get("interval", 5)
                    if ticker not in mock_tasks:
                        # Build SOW (historical candles) from CSV data
                        sow_bars, start_idx = build_mock_sow(
                            ticker, interval_minutes=interval_min, num_candles=50
                        )

                        # Build mock metadata so frontend can display correct info
                        mock_meta = {"mock": True}
                        if sow_bars:
                            mock_meta["first_dt"] = sow_bars[0]["datetime"]
                            mock_meta["last_dt"] = sow_bars[-1]["datetime"]
                            mock_meta["last_price"] = sow_bars[-1]["close"]
                            mock_meta["first_price"] = sow_bars[0]["open"]
                            mock_meta["interval"] = interval_min

                        # Send SOW so the chart has data immediately
                        if sow_bars:
                            await ws.send_text(json.dumps({
                                "type": "sow",
                                "ticker": ticker,
                                "bars": sow_bars,
                                **mock_meta,
                            }))

                        task = asyncio.create_task(
                            mock_csv_playback(ticker, queue, interval_minutes=interval_min,
                                              start_index=start_idx,
                                              sow_bars=sow_bars)
                        )
                        mock_tasks[ticker] = task
                        subscribed_tickers.add(ticker)

                    await ws.send_text(json.dumps({
                        "type": "subscribed",
                        "ticker": ticker,
                        "message": f"Subscribed to {ticker} (mock)",
                    }))
                else:
                    # Real mode – subscribe via LiveFeedManager
                    sow_bars, sub_error = await live_feed_manager.subscribe(ticker, queue)

                    if sub_error:
                        # Connection limit hit or auth failed – notify client
                        error_messages = {
                            "conn_limit": "Polygon connection limit reached. The server is backing off and will retry automatically. Please try again in ~30 seconds.",
                            "auth_failed": "Polygon authentication failed. Your plan may not support live WebSocket streaming.",
                        }
                        await ws.send_text(json.dumps({
                            "type": "error",
                            "code": sub_error,
                            "message": error_messages.get(sub_error, f"Subscribe failed: {sub_error}"),
                        }))
                        continue

                    subscribed_tickers.add(ticker)

                    # Send snapshot of accumulated bars
                    await ws.send_text(json.dumps({
                        "type": "sow",
                        "ticker": ticker,
                        "bars": sow_bars,
                    }))

                    await ws.send_text(json.dumps({
                        "type": "subscribed",
                        "ticker": ticker,
                        "message": f"Subscribed to {ticker}",
                    }))

                    # Send updated status (upstream may have connected during subscribe)
                    status = live_feed_manager.get_status()
                    await ws.send_text(json.dumps({"type": "status", **status}))

                    # Start periodic analysis for option tickers
                    if ticker.startswith("O:") and analysis_task is None:
                        analysis_task = asyncio.create_task(
                            _analysis_loop(ticker, queue)
                        )

            elif action == "unsubscribe" and ticker:
                if ticker in mock_tasks:
                    # Cancel mock playback task
                    mock_tasks[ticker].cancel()
                    del mock_tasks[ticker]
                    subscribed_tickers.discard(ticker)

                    await ws.send_text(json.dumps({
                        "type": "unsubscribed",
                        "ticker": ticker,
                    }))

                elif ticker in subscribed_tickers:
                    await live_feed_manager.unsubscribe(ticker, queue)
                    subscribed_tickers.discard(ticker)

                    # Stop analysis task if unsubscribing the option ticker
                    if ticker.startswith("O:") and analysis_task:
                        analysis_task.cancel()
                        analysis_task = None

                    await ws.send_text(json.dumps({
                        "type": "unsubscribed",
                        "ticker": ticker,
                    }))

            elif action == "status":
                status = live_feed_manager.get_status()
                await ws.send_text(json.dumps({"type": "status", **status}))

            else:
                await ws.send_text(json.dumps({
                    "type": "error",
                    "message": f"Unknown action: {action}",
                }))

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[WS] WebSocket error: {e}")
    finally:
        # Clean up: cancel analysis task and unsubscribe from all tickers
        if analysis_task:
            analysis_task.cancel()
        sender_task.cancel()
        # Cancel all mock playback tasks
        for task in mock_tasks.values():
            task.cancel()
        # Unsubscribe real tickers from live feed manager
        for ticker in subscribed_tickers:
            if ticker not in mock_tasks:
                await live_feed_manager.unsubscribe(ticker, queue)
        mock_tasks.clear()
        print(f"[WS] Client disconnected, cleaned up {len(subscribed_tickers)} subscription(s)")


@app.get("/api/feed/status")
async def feed_status():
    """Get the current status of the live feed manager."""
    return live_feed_manager.get_status()


# ── Watchlist WebSocket feed ────────────────────────────────

@app.websocket("/ws/watchlist")
async def websocket_watchlist(ws: WebSocket):
    """
    Dedicated WebSocket for watchlist price streaming.

    Client sends:
      {"symbols": ["SPY","QQQ","AAPL"], "mock": false}
      (can be re-sent to update symbols or mock state)

    Server pushes (every ~1 sec):
      {"type": "prices", "prices": [{symbol, price, change, ...}, ...]}
      (filtered to only the symbols THIS client requested)
    """
    user = await _require_websocket_user(ws)
    if not user:
        return
    await ws.accept()
    cid: int | None = None

    async def _sender(q: asyncio.Queue):
        try:
            while True:
                msg = await q.get()
                await ws.send_text(msg)
        except (WebSocketDisconnect, Exception):
            pass

    sender_task: asyncio.Task | None = None

    try:
        while True:
            raw = await ws.receive_text()
            try:
                cmd = json.loads(raw)
            except json.JSONDecodeError:
                await ws.send_text(json.dumps({"type": "error", "message": "Invalid JSON"}))
                continue

            symbols_raw = cmd.get("symbols", [])
            mock = cmd.get("mock", False)
            symbols = {s.strip().upper() for s in symbols_raw if s.strip()}

            if not symbols:
                await ws.send_text(json.dumps({"type": "error", "message": "No symbols provided"}))
                continue

            if cid is None:
                # First message → register
                cid, queue = await _watchlist_hub.register(symbols, mock)
                sender_task = asyncio.create_task(_sender(queue))
                await ws.send_text(json.dumps({
                    "type": "subscribed",
                    "symbols": sorted(symbols),
                    "mock": mock,
                }))
            else:
                # Subsequent messages → update symbols / mock flag
                await _watchlist_hub.update_symbols(cid, symbols, mock)
                await ws.send_text(json.dumps({
                    "type": "subscribed",
                    "symbols": sorted(symbols),
                    "mock": mock,
                }))

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[WS-Watchlist] Error: {e}")
    finally:
        if sender_task:
            sender_task.cancel()
        if cid is not None:
            await _watchlist_hub.unregister(cid)
        print(f"[WS-Watchlist] Client disconnected")


# ── Briefing endpoint (Idea Engine) ────────────────────────

class StrikeWallResponse(BaseModel):
    strike: float
    wall_type: str
    open_interest: int
    distance_pct: float
    label: str

class StrikeMagnetResponse(BaseModel):
    strike: float
    net_gamma_exp: float
    distance_pct: float
    polarity: str
    label: str

class PositioningResponse(BaseModel):
    call_walls: list[StrikeWallResponse] = []
    put_walls: list[StrikeWallResponse] = []
    gamma_magnets: list[StrikeMagnetResponse] = []
    max_pain: float = 0.0
    net_gex: float = 0.0
    gex_regime: str = "neutral"
    summary: str = ""

class TradeThemeResponse(BaseModel):
    title: str
    description: str
    direction: str
    confidence: float
    emoji: str = ""

class BiasSummaryResponse(BaseModel):
    direction: str
    strength: float
    headline: str
    bullets: list[str] = []
    kill_switch: str = ""

class DayTypeProbResponse(BaseModel):
    most_likely: str
    probabilities: dict[str, float] = {}
    description: str = ""
    setup_implications: str = ""

class ConditionalPlanResponse(BaseModel):
    condition: str
    action: str
    option_description: str
    direction: str
    trigger_level: float
    stop_description: str
    targets: list[float] = []
    suitable_for: list[str] = []

class TradeIdeaResponse(BaseModel):
    idea_type: str
    direction: str
    symbol: str
    headline: str
    rationale: str
    entry_level: float
    stop_level: float
    target_1: float
    target_2: float
    reward_risk: float
    option_hint: str
    suitable_for: list[str] = []
    confidence: float = 0.0
    score: int = 0
    score_breakdown: dict = {}
    ai_narrative: Optional[str] = None
    warning: str = ""
    trade_state: str = "PENDING"       # PENDING / ACTIVE / INVALIDATED
    trade_state_reason: str = ""

class DecisionResponse(BaseModel):
    decision: str          # BUY_CALLS / BUY_PUTS / WAIT
    call_score: int = 0
    put_score: int = 0
    wait_reason: str = ""
    trigger: str = ""
    invalidation: str = ""
    because: list[str] = []
    entry_zone_low: Optional[float] = None
    entry_zone_high: Optional[float] = None
    stop_level: Optional[float] = None
    targets: list[float] = []
    confidence: float = 0.0
    hard_guard_active: bool = False
    hard_guard_reason: str = ""
    capital_mode: str = "NORMAL"
    capital_mode_reason: str = ""
    max_size_mult: float = 1.0
    locked_until: Optional[str] = None
    call_sub: dict = {}
    put_sub: dict = {}
    # Action Engine fields
    wait_until: list[dict] = []
    execution_mode: str = "STAND_ASIDE"
    execution_mode_reason: str = ""
    wait_confidence: str = ""
    next_move_map: list[dict] = []


class PerformanceStatsResponse(BaseModel):
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    avg_r: float = 0.0
    total_pnl: float = 0.0
    expectancy: float = 0.0
    best_r: float = 0.0
    worst_r: float = 0.0
    avg_score: int = 0
    recent_trades: list[dict] = []
    by_setup: list[dict] = []

class ScanResultResponse(BaseModel):
    symbol: str
    score: int = 0
    clean_trend: int = 0
    compression: int = 0
    breakout_prox: int = 0
    positioning_edge: int = 0
    regime: str = ""
    spot: float = 0.0
    atr: float = 0.0
    note: str = ""


class BriefingResponse(BaseModel):
    symbol: str
    as_of: str
    underlying_price: float
    bias: BiasSummaryResponse
    day_type: DayTypeProbResponse
    themes: list[TradeThemeResponse] = []
    positioning: Optional[PositioningResponse] = None
    conditional_plans: list[ConditionalPlanResponse] = []
    trade_ideas: list[TradeIdeaResponse] = []
    regime_name: str = ""
    regime_confidence: float = 0.0
    vwap_position: str = "at"
    rsi: float = 50.0
    atr: float = 0.0
    posture: Optional[DecisionResponse] = None
    ai_briefing_narrative: Optional[str] = None
    llm_available: bool = False


# ── Helpers for DecisionResponse + TradeState ────────────────

def _build_decision_response(dr) -> Optional[DecisionResponse]:
    """Build DecisionResponse from a TradeDecision dataclass."""
    if dr is None:
        return None
    return DecisionResponse(
        decision=dr.decision,
        call_score=dr.call_score,
        put_score=dr.put_score,
        wait_reason=dr.wait_reason,
        trigger=dr.trigger,
        invalidation=dr.invalidation,
        because=dr.because,
        entry_zone_low=dr.entry_zone_low,
        entry_zone_high=dr.entry_zone_high,
        stop_level=dr.stop_level,
        targets=dr.targets,
        confidence=dr.confidence,
        hard_guard_active=dr.hard_guard_active,
        hard_guard_reason=dr.hard_guard_reason,
        capital_mode=dr.capital_mode,
        capital_mode_reason=dr.capital_mode_reason,
        max_size_mult=dr.max_size_mult,
        locked_until=dr.locked_until,
        call_sub=dr.call_sub.__dict__,
        put_sub=dr.put_sub.__dict__,
        wait_until=dr.wait_until,
        execution_mode=dr.execution_mode,
        execution_mode_reason=dr.execution_mode_reason,
        wait_confidence=dr.wait_confidence,
        next_move_map=dr.next_move_map,
    )


def _compute_trade_state(idea, underlying_price: float) -> tuple[str, str]:
    """Determine trade state: PENDING / ACTIVE / INVALIDATED."""
    if underlying_price <= 0:
        return "PENDING", "Waiting for price data"

    entry = float(idea.entry_level or 0)
    stop = float(idea.stop_level or 0)
    direction = (idea.direction or "").upper()

    if entry <= 0:
        return "PENDING", "No entry level defined"

    if direction == "CALL":
        # CALL is invalidated if price broke below stop
        if stop > 0 and underlying_price < stop:
            return "INVALIDATED", f"Price {underlying_price:.2f} below stop {stop:.2f}"
        # ACTIVE if price is at or above entry
        if underlying_price >= entry * 0.998:
            return "ACTIVE", f"Price at entry zone ({entry:.2f})"
        return "PENDING", f"Waiting for price to reach {entry:.2f}"
    else:
        # PUT is invalidated if price broke above stop
        if stop > 0 and underlying_price > stop:
            return "INVALIDATED", f"Price {underlying_price:.2f} above stop {stop:.2f}"
        # ACTIVE if price is at or below entry
        if underlying_price <= entry * 1.002:
            return "ACTIVE", f"Price at entry zone ({entry:.2f})"
        return "PENDING", f"Waiting for price to reach {entry:.2f}"


@app.get("/api/briefing/{sym:path}", response_model=BriefingResponse)
async def get_briefing(
    sym: str,
    playbook: str = Query("all", description="all | 0dte | swing | scalp"),
    interval: int = Query(5, ge=1, le=60),
    poly: PolygonClient = Depends(get_polygon),
    dl: DataLayer = Depends(get_data_layer),
):
    """
    Idea Engine briefing for a symbol.
    Returns bias, day-type, trade themes, IF/THEN plans, trade ideas, and positioning.
    Optionally filtered by playbook mode.
    """
    try:
        sym_raw = sym.upper()
        sym, asset_class, proxy_ticker = _normalize_symbol(sym_raw)

        # ── Fetch underlying data (reuse same pipeline as /api/signals) ──
        today = date.today()
        intraday_df = pd.DataFrame()
        try:
            intraday_df = await poly.get_stock_ohlcv(
                symbol=sym,
                start_date=today,
                end_date=today,
                interval_min=interval,
            )
        except Exception:
            pass

        if intraday_df.empty:
            try:
                intraday_df = await dl.get_intraday(sym, today, interval_min=interval)
            except Exception:
                pass

        underlying_price = 0.0
        if not intraday_df.empty:
            underlying_price = float(intraday_df["close"].iloc[-1])
        else:
            try:
                price_data = await poly.get_snapshot_prices([sym])
                snap = price_data.get(sym, {})
                underlying_price = float(snap.get("lastPrice", 0) or snap.get("prevClose", 0))
            except Exception:
                pass

        # ── Regime ──────────────────────────────────────────────
        regime_res = None
        if not intraday_df.empty and len(intraday_df) >= 5:
            try:
                regime_res = regime_engine.classify(intraday_df, symbol=sym)
            except Exception:
                pass

        # ── SR ──────────────────────────────────────────────────
        sr_res = None
        try:
            daily_df = await dl.get_daily(sym)
            if not daily_df.empty:
                sr_res = sr_engine.analyze(
                    daily_df=daily_df,
                    intraday_df=intraday_df if not intraday_df.empty else None,
                    current_price=underlying_price,
                )
        except Exception:
            pass

        # ── Chain metrics ────────────────────────────────────────
        chain_metrics_res = None
        chain_data_raw = None
        try:
            expirations = await poly.get_expirations(sym)
            if expirations:
                chain_data_raw = await poly.get_options_chain_snapshot(sym, expirations[0])
                if chain_data_raw:
                    chain_metrics_res = compute_chain_metrics(chain_data_raw, underlying_price)
        except Exception:
            pass

        # ── ATM option quality for Decision Engine ───────────────
        _atm_spread_pct = 0.0
        _atm_delta = 0.0
        try:
            if chain_data_raw and underlying_price > 0:
                _atm_calls = [c for c in chain_data_raw if c.get("contract_type", "").lower() == "call"]
                if _atm_calls:
                    _atm = min(_atm_calls, key=lambda c: abs(c["strike"] - underlying_price))
                    _bid = float(_atm.get("bid", 0) or 0)
                    _ask = float(_atm.get("ask", 0) or 0)
                    _mid = (_bid + _ask) / 2
                    if _mid > 0:
                        _atm_spread_pct = (_ask - _bid) / _mid * 100
                    _greeks = _atm.get("greeks", {}) or {}
                    _atm_delta = float(_greeks.get("delta", 0) or 0)
        except Exception:
            pass

        # ── VPA ─────────────────────────────────────────────────
        vpa_bias_dict: dict = {}
        vpa_vol_regime: dict = {}
        try:
            if not intraday_df.empty and len(intraday_df) >= 2:
                vpa_results = vpa_engine.analyze(intraday_df)
                vpa_bias_dict = vpa_engine.get_bias(vpa_results) or {}
                vpa_vol_regime = vpa_engine.get_volume_regime(intraday_df) or {}
        except Exception:
            pass

        # ── Active alerts ────────────────────────────────────────
        active_alerts_list = alert_manager.get_active_alerts()

        # ── Run Idea Engine ───────────────────────────────────────
        briefing_input = BriefingInput(
            symbol=sym,
            underlying_price=underlying_price,
            df=intraday_df,
            regime=regime_res,
            sr=sr_res,
            chain_metrics=chain_metrics_res,
            vpa_bias=vpa_bias_dict,
            active_alerts=active_alerts_list,
        )
        briefing = idea_engine.generate_briefing(briefing_input)

        # ── Playbook filter ───────────────────────────────────────
        try:
            pb_mode = PlaybookMode(playbook.lower())
        except ValueError:
            pb_mode = PlaybookMode.ALL
        briefing = idea_engine.filter_by_playbook(briefing, pb_mode)

        # ── Decision Engine ────────────────────────────────────────
        decision_result: Optional[TradeDecision] = None
        try:
            decision_result = decision_engine.compute(
                regime=regime_res,
                sr=sr_res,
                chain_metrics=chain_metrics_res,
                vpa_bias=vpa_bias_dict,
                vol_regime=vpa_vol_regime,
                underlying_price=underlying_price,
                df=intraday_df,
                spread_pct=_atm_spread_pct,
                delta=_atm_delta,
            )
        except Exception as _pe:
            pass

        # ── Optional LLM narration ────────────────────────────────
        if llm_narrator.is_available():
            try:
                briefing.ai_briefing_narrative = llm_narrator.narrate_briefing(briefing)
                for idea in briefing.trade_ideas:
                    idea.ai_narrative = llm_narrator.narrate_idea(idea)
            except Exception:
                pass

        # ── Serialise to response model ───────────────────────────
        def _pos_resp(pos):
            if pos is None:
                return None
            return PositioningResponse(
                call_walls=[StrikeWallResponse(**{k: getattr(w, k) for k in ('strike','wall_type','open_interest','distance_pct','label')}) for w in pos.call_walls],
                put_walls=[StrikeWallResponse(**{k: getattr(w, k) for k in ('strike','wall_type','open_interest','distance_pct','label')}) for w in pos.put_walls],
                gamma_magnets=[StrikeMagnetResponse(**{k: getattr(m, k) for k in ('strike','net_gamma_exp','distance_pct','polarity','label')}) for m in pos.gamma_magnets],
                max_pain=pos.max_pain,
                net_gex=pos.net_gex,
                gex_regime=pos.gex_regime,
                summary=pos.summary,
            )

        return BriefingResponse(
            symbol=briefing.symbol,
            as_of=briefing.as_of,
            underlying_price=briefing.underlying_price,
            bias=BiasSummaryResponse(
                direction=briefing.bias.direction,
                strength=briefing.bias.strength,
                headline=briefing.bias.headline,
                bullets=briefing.bias.bullets,
                kill_switch=briefing.bias.kill_switch,
            ),
            day_type=DayTypeProbResponse(
                most_likely=briefing.day_type.most_likely.value,
                probabilities=briefing.day_type.probabilities,
                description=briefing.day_type.description,
                setup_implications=briefing.day_type.setup_implications,
            ),
            themes=[TradeThemeResponse(
                title=t.title, description=t.description,
                direction=t.direction, confidence=t.confidence, emoji=t.emoji,
            ) for t in briefing.themes],
            positioning=_pos_resp(briefing.positioning),
            conditional_plans=[ConditionalPlanResponse(
                condition=p.condition, action=p.action,
                option_description=p.option_description, direction=p.direction,
                trigger_level=p.trigger_level, stop_description=p.stop_description,
                targets=p.targets, suitable_for=p.suitable_for,
            ) for p in briefing.conditional_plans],
            trade_ideas=[TradeIdeaResponse(
                idea_type=i.idea_type.value, direction=i.direction, symbol=i.symbol,
                headline=i.headline, rationale=i.rationale,
                entry_level=i.entry_level, stop_level=i.stop_level,
                target_1=i.target_1, target_2=i.target_2,
                reward_risk=i.reward_risk, option_hint=i.option_hint,
                suitable_for=i.suitable_for, confidence=i.confidence,
                score=i.score, score_breakdown=i.score_breakdown,
                ai_narrative=i.ai_narrative, warning=i.warning,
                trade_state=_compute_trade_state(i, underlying_price)[0],
                trade_state_reason=_compute_trade_state(i, underlying_price)[1],
            ) for i in briefing.trade_ideas],
            regime_name=briefing.regime_name,
            regime_confidence=briefing.regime_confidence,
            vwap_position=briefing.vwap_position,
            rsi=briefing.rsi,
            atr=briefing.atr,
            posture=_build_decision_response(decision_result),
            ai_briefing_narrative=briefing.ai_briefing_narrative,
            llm_available=llm_narrator.is_available(),
        )

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ── Live Feed WebSocket — replaces REST polling in live mode ─────

@app.websocket("/ws/feed/{sym}")
async def websocket_feed(ws: WebSocket, sym: str):
    """
    Market data push WebSocket.  No more REST polling.

    Server pushes on connect and every `interval` seconds:
      {"type":"pulse",    "as_of":…, "underlying_price":…,
                          "regime":…, "active_alerts":[…],
                          "chain_snapshot":…, "volume_regime":…,
                          "proximity_score":…, "proximity_detail":…,
                          "posture":…}
      {"type":"briefing", …BriefingResponse fields… (every 2nd cycle)}
      {"type":"error",    "detail":"…"}

    Client → server:
      {"action":"refresh"}                  — force immediate push
      {"action":"set_interval","interval":N} — set cadence in seconds (≥10)
    """
    user = await _require_websocket_user(ws)
    if not user:
        return
    await ws.accept()
    sym = sym.upper()
    refresh_event = asyncio.Event()
    running = True
    interval_sec = 30
    cycle_count = 0

    # ── Receive loop ──────────────────────────────────────────
    async def _recv():
        nonlocal interval_sec, running
        try:
            while True:
                raw = await ws.receive_text()
                try:
                    msg = json.loads(raw)
                    act = msg.get("action")
                    if act == "refresh":
                        refresh_event.set()
                    elif act == "set_interval":
                        interval_sec = max(10, int(msg.get("interval", 30)))
                except Exception:
                    pass
        except Exception:
            running = False

    recv_task = asyncio.create_task(_recv())

    # ── Pulse: regime + alerts + chain + posture (fast) ───────
    async def _push_pulse():
        try:
            today = date.today()
            intraday_df = pd.DataFrame()
            try:
                intraday_df = await polygon_client.get_stock_ohlcv(
                    sym, start_date=today, end_date=today, interval_min=5,
                )
            except Exception:
                pass

            underlying_price = 0.0
            if not intraday_df.empty:
                underlying_price = float(intraday_df.iloc[-1]["close"])

            # Regime
            regime_res = None
            if not intraday_df.empty and len(intraday_df) >= 5:
                try:
                    regime_res = regime_engine.classify(intraday_df, symbol=sym)
                except Exception:
                    pass

            # VPA / volume regime
            vpa_bias_p: dict = {}
            vol_regime_p: dict = {}
            if not intraday_df.empty and len(intraday_df) >= 2:
                try:
                    _vpa = vpa_engine.analyze(intraday_df)
                    vpa_bias_p = vpa_engine.get_bias(_vpa) or {}
                    vol_regime_p = vpa_engine.get_volume_regime(intraday_df) or {}
                except Exception:
                    pass

            # SR
            sr_p = None
            try:
                daily_df = await _data_layer.get_daily(sym)
                if not daily_df.empty and underlying_price > 0:
                    sr_p = sr_engine.analyze(
                        daily_df=daily_df,
                        intraday_df=intraday_df if not intraday_df.empty else None,
                        current_price=underlying_price,
                    )
            except Exception:
                pass

            # Chain
            chain_snap_p = None
            chain_metrics_p = None
            _cd_p_raw = None
            try:
                exps_p = await polygon_client.get_expirations(sym)
                if exps_p:
                    _cd_p_raw = await polygon_client.get_options_chain_snapshot(sym, exps_p[0])
                    if _cd_p_raw and underlying_price > 0:
                        chain_metrics_p = compute_chain_metrics(_cd_p_raw, underlying_price)
                        chain_snap_p = ChainSnapshotResponse(
                            iv_rank=round(chain_metrics_p.iv_rank, 2),
                            iv_percentile=round(chain_metrics_p.iv_percentile, 2),
                            put_call_oi_ratio=round(chain_metrics_p.put_call_oi_ratio, 3),
                            put_call_volume_ratio=round(chain_metrics_p.put_call_volume_ratio, 3),
                            net_gex=round(chain_metrics_p.net_gex, 0),
                            gex_regime=chain_metrics_p.gex_regime,
                            max_pain=round(chain_metrics_p.max_pain, 2),
                            uoa_detected=chain_metrics_p.uoa_detected,
                            uoa_details=getattr(chain_metrics_p, "uoa_details", [])[:5],
                            call_wall=round(chain_metrics_p.call_wall, 2),
                            put_wall=round(chain_metrics_p.put_wall, 2),
                            weighted_iv=round(chain_metrics_p.weighted_iv, 4),
                        )
            except Exception:
                pass

            # ATM option quality for Decision
            _ws_spread = 0.0
            _ws_delta = 0.0
            try:
                if _cd_p_raw and underlying_price > 0:
                    _ws_calls = [c for c in _cd_p_raw if c.get("contract_type", "").lower() == "call"]
                    if _ws_calls:
                        _ws_atm = min(_ws_calls, key=lambda c: abs(c["strike"] - underlying_price))
                        _wb = float(_ws_atm.get("bid", 0) or 0)
                        _wa = float(_ws_atm.get("ask", 0) or 0)
                        _wm = (_wb + _wa) / 2
                        if _wm > 0:
                            _ws_spread = (_wa - _wb) / _wm * 100
                        _wg = _ws_atm.get("greeks", {}) or {}
                        _ws_delta = float(_wg.get("delta", 0) or 0)
            except Exception:
                pass

            # Decision
            decision_p = None
            try:
                dt = decision_engine.compute(
                    regime=regime_res, sr=sr_p, chain_metrics=chain_metrics_p,
                    vpa_bias=vpa_bias_p, vol_regime=vol_regime_p,
                    underlying_price=underlying_price, df=intraday_df,
                    spread_pct=_ws_spread, delta=_ws_delta,
                )
                _dr = _build_decision_response(dt)
                decision_p = _dr.dict() if _dr else None
            except Exception:
                pass

            # ── Setup Detection → Alerts (same pipeline as /api/signals) ──
            try:
                if (
                    regime_res is not None
                    and sr_p is not None
                    and not intraday_df.empty
                    and underlying_price > 0
                ):
                    _rv_p = rolling_rv(intraday_df)
                    _setups_p = setup_engine.detect_all(
                        df=intraday_df, regime=regime_res, sr=sr_p,
                        chain=_cd_p_raw or [], symbol=sym,
                    )
                    _edge_p: list[EdgeResult] = []
                    _exps_for_pick = None
                    try:
                        _exps_for_pick_list = await polygon_client.get_expirations(sym)
                        if _exps_for_pick_list:
                            _exps_for_pick = _exps_for_pick_list[0]
                    except Exception:
                        pass
                    for _sp in _setups_p:
                        _pick_p = option_picker.pick(
                            chain=_cd_p_raw or [], direction=_sp.direction,
                            spot=underlying_price, edge_score=60,
                            expiration_date=_exps_for_pick,
                        )
                        _scored_p = edge_scorer.score(
                            setup=_sp, regime=regime_res, option=_pick_p,
                            df=intraday_df,
                            rv=float(_rv_p.iloc[-1]) if len(_rv_p) > 0 else 0.0,
                        )
                        if _scored_p.tier != "NO_EDGE":
                            _edge_p.append(_scored_p)
                    alert_manager.process_tick(
                        edge_results=_edge_p,
                        current_price=underlying_price,
                        current_time=datetime.utcnow(),
                        symbol=sym,
                    )
            except Exception as _alert_err:
                print(f"[Feed] Alert pipeline error: {_alert_err}")

            # Active alerts (in-memory singleton)
            def _alert_dict(alert):
                return _build_alert_response(alert).dict()

            regime_out = None
            if regime_res:
                regime_out = RegimeResponse(
                    regime=regime_res.regime.value, confidence=round(regime_res.confidence, 3),
                    detail=regime_res.detail, vwap_current=round(regime_res.vwap_current, 4),
                    atr_current=round(regime_res.atr_current, 4),
                    rsi_current=round(regime_res.rsi_current, 2),
                    price_vs_vwap=regime_res.price_vs_vwap,
                ).dict()

            prox_score = 0.0
            prox_detail = ""
            if sr_p:
                prox_score = round(sr_p.proximity_score, 3)
                prox_detail = sr_p.proximity_detail or ""

            await ws.send_text(json.dumps({
                "type": "pulse",
                "as_of": datetime.utcnow().isoformat(),
                "underlying_price": underlying_price,
                "regime": regime_out,
                "active_alerts": [_alert_dict(a) for a in alert_manager.get_active_alerts()],
                "chain_snapshot": chain_snap_p.dict() if chain_snap_p else None,
                "volume_regime": vol_regime_p,
                "proximity_score": prox_score,
                "proximity_detail": prox_detail,
                "posture": decision_p,
            }, default=str))

        except Exception as e:
            try:
                await ws.send_text(json.dumps({"type": "error", "detail": f"pulse: {e}"}))
            except Exception:
                pass

    # ── Briefing: ideas + themes + posture (every 2nd cycle) ─
    async def _push_briefing():
        try:
            today = date.today()
            intraday_df = pd.DataFrame()
            try:
                intraday_df = await polygon_client.get_stock_ohlcv(
                    sym, start_date=today, end_date=today, interval_min=5,
                )
            except Exception:
                pass

            underlying_price = 0.0
            if not intraday_df.empty:
                underlying_price = float(intraday_df.iloc[-1]["close"])
            else:
                try:
                    snap = await polygon_client.get_snapshot_prices([sym])
                    s = snap.get(sym, {})
                    underlying_price = float(s.get("lastPrice", 0) or s.get("prevClose", 0))
                except Exception:
                    pass

            regime_res = None
            if not intraday_df.empty and len(intraday_df) >= 5:
                try:
                    regime_res = regime_engine.classify(intraday_df, symbol=sym)
                except Exception:
                    pass

            sr_res = None
            try:
                daily_df = await _data_layer.get_daily(sym)
                if not daily_df.empty:
                    sr_res = sr_engine.analyze(
                        daily_df=daily_df,
                        intraday_df=intraday_df if not intraday_df.empty else None,
                        current_price=underlying_price,
                    )
            except Exception:
                pass

            chain_metrics_res = None
            _cd_b_raw = None
            try:
                exps = await polygon_client.get_expirations(sym)
                if exps:
                    _cd_b_raw = await polygon_client.get_options_chain_snapshot(sym, exps[0])
                    if _cd_b_raw:
                        chain_metrics_res = compute_chain_metrics(_cd_b_raw, underlying_price)
            except Exception:
                pass

            # ATM option quality for Decision
            _b_spread = 0.0
            _b_delta = 0.0
            try:
                if _cd_b_raw and underlying_price > 0:
                    _b_calls = [c for c in _cd_b_raw if c.get("contract_type", "").lower() == "call"]
                    if _b_calls:
                        _b_atm = min(_b_calls, key=lambda c: abs(c["strike"] - underlying_price))
                        _bb = float(_b_atm.get("bid", 0) or 0)
                        _ba = float(_b_atm.get("ask", 0) or 0)
                        _bm = (_bb + _ba) / 2
                        if _bm > 0:
                            _b_spread = (_ba - _bb) / _bm * 100
                        _bg = _b_atm.get("greeks", {}) or {}
                        _b_delta = float(_bg.get("delta", 0) or 0)
            except Exception:
                pass

            vpa_bias_dict: dict = {}
            vpa_vol_regime: dict = {}
            try:
                if not intraday_df.empty and len(intraday_df) >= 2:
                    vpa_res = vpa_engine.analyze(intraday_df)
                    vpa_bias_dict = vpa_engine.get_bias(vpa_res) or {}
                    vpa_vol_regime = vpa_engine.get_volume_regime(intraday_df) or {}
            except Exception:
                pass

            briefing_input = BriefingInput(
                symbol=sym, underlying_price=underlying_price, df=intraday_df,
                regime=regime_res, sr=sr_res, chain_metrics=chain_metrics_res,
                vpa_bias=vpa_bias_dict, active_alerts=alert_manager.get_active_alerts(),
            )
            briefing = idea_engine.generate_briefing(briefing_input)
            briefing = idea_engine.filter_by_playbook(briefing, PlaybookMode.ALL)

            decision_result = None
            try:
                decision_result = decision_engine.compute(
                    regime=regime_res, sr=sr_res, chain_metrics=chain_metrics_res,
                    vpa_bias=vpa_bias_dict, vol_regime=vpa_vol_regime,
                    underlying_price=underlying_price, df=intraday_df,
                    spread_pct=_b_spread, delta=_b_delta,
                )
            except Exception:
                pass

            def _pos_resp_b(pos):
                if pos is None:
                    return None
                return PositioningResponse(
                    call_walls=[StrikeWallResponse(**{k: getattr(w, k) for k in ('strike','wall_type','open_interest','distance_pct','label')}) for w in pos.call_walls],
                    put_walls=[StrikeWallResponse(**{k: getattr(w, k) for k in ('strike','wall_type','open_interest','distance_pct','label')}) for w in pos.put_walls],
                    gamma_magnets=[StrikeMagnetResponse(**{k: getattr(m, k) for k in ('strike','net_gamma_exp','distance_pct','polarity','label')}) for m in pos.gamma_magnets],
                    max_pain=pos.max_pain, net_gex=pos.net_gex,
                    gex_regime=pos.gex_regime, summary=pos.summary,
                )

            briefing_resp = BriefingResponse(
                symbol=briefing.symbol, as_of=briefing.as_of,
                underlying_price=briefing.underlying_price,
                bias=BiasSummaryResponse(
                    direction=briefing.bias.direction, strength=briefing.bias.strength,
                    headline=briefing.bias.headline, bullets=briefing.bias.bullets,
                    kill_switch=briefing.bias.kill_switch,
                ),
                day_type=DayTypeProbResponse(
                    most_likely=briefing.day_type.most_likely.value,
                    probabilities=briefing.day_type.probabilities,
                    description=briefing.day_type.description,
                    setup_implications=briefing.day_type.setup_implications,
                ),
                themes=[TradeThemeResponse(
                    title=t.title, description=t.description,
                    direction=t.direction, confidence=t.confidence, emoji=t.emoji,
                ) for t in briefing.themes],
                positioning=_pos_resp_b(briefing.positioning),
                conditional_plans=[ConditionalPlanResponse(
                    condition=p.condition, action=p.action,
                    option_description=p.option_description, direction=p.direction,
                    trigger_level=p.trigger_level, stop_description=p.stop_description,
                    targets=p.targets, suitable_for=p.suitable_for,
                ) for p in briefing.conditional_plans],
                trade_ideas=[TradeIdeaResponse(
                    idea_type=i.idea_type.value, direction=i.direction, symbol=i.symbol,
                    headline=i.headline, rationale=i.rationale,
                    entry_level=i.entry_level, stop_level=i.stop_level,
                    target_1=i.target_1, target_2=i.target_2,
                    reward_risk=i.reward_risk, option_hint=i.option_hint,
                    suitable_for=i.suitable_for, confidence=i.confidence,
                    score=i.score, score_breakdown=i.score_breakdown,
                    ai_narrative=i.ai_narrative, warning=i.warning,
                    trade_state=_compute_trade_state(i, underlying_price)[0],
                    trade_state_reason=_compute_trade_state(i, underlying_price)[1],
                ) for i in briefing.trade_ideas],
                regime_name=briefing.regime_name, regime_confidence=briefing.regime_confidence,
                vwap_position=briefing.vwap_position, rsi=briefing.rsi, atr=briefing.atr,
                posture=_build_decision_response(decision_result),
                llm_available=llm_narrator.is_available(),
            )
            payload = json.loads(briefing_resp.json())
            payload["type"] = "briefing"
            await ws.send_text(json.dumps(payload, default=str))

        except Exception as e:
            try:
                await ws.send_text(json.dumps({"type": "error", "detail": f"briefing: {e}"}))
            except Exception:
                pass

    # ── Main loop ─────────────────────────────────────────────
    try:
        while running:
            await _push_pulse()
            if cycle_count % 2 == 0:   # briefing every 2nd cycle (60s by default)
                await _push_briefing()
            cycle_count += 1
            refresh_event.clear()
            try:
                await asyncio.wait_for(refresh_event.wait(), timeout=float(interval_sec))
            except asyncio.TimeoutError:
                pass
    except Exception:
        pass
    finally:
        running = False
        recv_task.cancel()


# ── Static files (must be last so it doesn't shadow API routes) ─

# ── Backtest Replay endpoint ─────────────────────────────────

from pathlib import Path as _Path

_BACKTEST_DATA_DIR = _Path(__file__).parent.parent / "data" / "qqq"


def _bt_load_day(date_str: str) -> pd.DataFrame:
    """Load a single day's QQQ bars from local CSV, filter to RTH."""
    path = _BACKTEST_DATA_DIR / f"{date_str}.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    for c in ["open", "close", "high", "low"]:
        df[c] = df[c].astype(float)
    df["volume"] = df["volume"].astype(int)
    df["datetime"] = pd.to_datetime(df["window_start"].astype(int), unit="ns").dt.strftime("%Y-%m-%d %H:%M:%S")
    df = df.sort_values("datetime").reset_index(drop=True)
    dt_col = pd.to_datetime(df["datetime"])
    mask_start = (dt_col.dt.hour > 14) | ((dt_col.dt.hour == 14) & (dt_col.dt.minute >= 30))
    mask_end = dt_col.dt.hour < 21
    df = df[mask_start & mask_end]
    return df.reset_index(drop=True)


def _bt_build_daily(all_dates: list[str]) -> pd.DataFrame:
    """Build pseudo-daily OHLCV from minute CSVs for S/R engine."""
    rows = []
    for d in all_dates:
        ddf = _bt_load_day(d)
        if ddf.empty:
            continue
        rows.append({
            "date": d, "open": ddf["open"].iloc[0], "high": ddf["high"].max(),
            "low": ddf["low"].min(), "close": ddf["close"].iloc[-1],
            "volume": int(ddf["volume"].sum()),
        })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


@app.get("/api/backtest/dates")
async def backtest_dates():
    """Return list of available backtest dates."""
    dates = sorted(p.stem for p in _BACKTEST_DATA_DIR.glob("*.csv"))
    return {"dates": dates}


@app.get("/api/backtest/replay/{sym}")
async def backtest_replay(
    sym: str,
    bt_date: str = Query(..., description="YYYY-MM-DD"),
    step_bars: int = Query(10, ge=5, le=120, description="Bars between checkpoints"),
    playbook: str = Query("all"),
):
    """
    Backtest-replay endpoint.
    Runs the full engine stack at each checkpoint and returns an array of snapshots
    that the frontend can step through like a VCR.
    Each snapshot mirrors the shape of the live /api/signals + /api/briefing responses.
    """
    sym = sym.upper()
    all_csv_dates = sorted(p.stem for p in _BACKTEST_DATA_DIR.glob("*.csv"))
    if bt_date not in all_csv_dates:
        raise HTTPException(status_code=404, detail=f"No data for {bt_date}")

    df = _bt_load_day(bt_date)
    if df.empty or len(df) < 30:
        raise HTTPException(status_code=400, detail="Insufficient bars for that date")

    daily_df = _bt_build_daily([d for d in all_csv_dates if d <= bt_date])

    # Fresh engines per replay
    _re = RegimeEngine()
    _sr = SREngine()
    _de = DecisionEngine()
    _ve = VPAEngine()
    _ie = IdeaEngine()

    try:
        pb_mode = PlaybookMode(playbook.lower())
    except ValueError:
        pb_mode = PlaybookMode.ALL

    # Warmup: always start at 60 bars so engines have enough data
    warmup = min(60, len(df))
    checkpoints = list(range(warmup, len(df) + 1, step_bars))
    if not checkpoints or checkpoints[-1] < len(df):
        checkpoints.append(len(df))

    snapshots = []

    for cp in checkpoints:
        window = df.iloc[:cp].copy()
        spot = float(window["close"].iloc[-1])
        cp_time = str(window["datetime"].iloc[-1])

        # ── Bars ──
        bars_out = [
            {"datetime": str(r["datetime"]), "open": float(r["open"]),
             "high": float(r["high"]), "low": float(r["low"]),
             "close": float(r["close"]), "volume": int(r["volume"])}
            for _, r in window.iterrows()
        ]

        # ── Indicators ──
        ind_out = None
        if len(window) >= 5:
            try:
                _v = vwap(window); _vu, _vl = vwap_bands(window, 1.0)
                _e9 = ema(window["close"], 9); _e20 = ema(window["close"], 20)
                _rs = rsi(window); _at = atr_value(window)
                def _fl(s): return [round(float(v), 4) if not pd.isna(v) else 0.0 for v in s]
                ind_out = {"vwap": _fl(_v), "vwap_upper": _fl(_vu), "vwap_lower": _fl(_vl),
                           "ema_9": _fl(_e9), "ema_20": _fl(_e20), "rsi": _fl(_rs), "atr": round(_at, 4)}
            except Exception:
                pass

        # ── Regime ──
        regime_out = None
        regime_res = None
        if len(window) >= 10:
            try:
                regime_res = _re.classify(window, force_reclassify=True)
                regime_out = {
                    "regime": regime_res.regime.value,
                    "confidence": round(regime_res.confidence, 3),
                    "detail": regime_res.detail,
                    "vwap_current": round(regime_res.vwap_current, 4),
                    "atr_current": round(regime_res.atr_current, 4),
                    "rsi_current": round(regime_res.rsi_current, 2),
                    "price_vs_vwap": regime_res.price_vs_vwap,
                }
            except Exception:
                pass

        # ── S/R ──
        sr_res = None
        levels_out = []
        if not daily_df.empty:
            try:
                sr_res = _sr.analyze(daily_df, spot, intraday_bars=window)
                levels_out = [
                    {"price": l.price, "kind": l.kind, "source": l.source,
                     "strength": l.strength, "label": l.label}
                    for l in sr_res.levels[:30]
                ]
            except Exception:
                pass

        # ── VPA ──
        vpa_out = None
        vpa_bias = {}
        vol_regime = {}
        if len(window) >= 2:
            try:
                vpa_results = _ve.analyze(window[["datetime", "open", "high", "low", "close", "volume"]].copy())
                vpa_bias = _ve.get_bias(vpa_results) or {}
                vol_regime = _ve.get_volume_regime(window) or {}
                vpa_sigs = [
                    {"signal": r.signal.value, "confidence": r.confidence,
                     "description": r.description, "datetime": r.datetime}
                    for r in vpa_results if r.signal != VPASignal.NEUTRAL
                ]
                vpa_out = {"bias": vpa_bias, "signals": vpa_sigs[-7:]}
            except Exception:
                pass

        # ── Volume regime ──
        vol_out = None
        if vol_regime:
            ratio_val = vol_regime.get("ratio", 1.0)
            if ratio_val < 0.5:
                vr_label = "LOW"
            elif ratio_val > 2.0:
                vr_label = "HIGH"
            else:
                vr_label = "NORMAL"
            vol_out = {"regime": vr_label, "ratio": round(ratio_val, 2),
                       "detail": vol_regime.get("detail", "")}

        # ── S/R proximity ──
        prox_score = 0.0
        prox_detail = ""
        if sr_res and spot > 0:
            try:
                nearest = min(sr_res.levels, key=lambda l: abs(l.price - spot)) if sr_res.levels else None
                if nearest:
                    dist_pct = (spot - nearest.price) / spot * 100
                    prox_score = round(dist_pct, 3)
                    proximity_label = "Near" if abs(dist_pct) < 0.2 else "Moderate" if abs(dist_pct) < 0.5 else "Far"
                    prox_detail = f"At {nearest.kind} ${nearest.price:.2f} ({abs(dist_pct):.1f}% away, strength {nearest.strength:.0f}%) → {'bounce' if nearest.kind == 'support' else 'rejection'} expected"
            except Exception:
                pass

        # ── Decision ──
        posture_out = None
        decision_res = None
        try:
            decision_res = _de.compute(
                regime=regime_res, sr=sr_res, chain_metrics=None,
                vpa_bias=vpa_bias, vol_regime=vol_regime,
                underlying_price=spot, df=window,
            )
            posture_out = {
                "decision": decision_res.decision,
                "call_score": decision_res.call_score,
                "put_score": decision_res.put_score,
                "wait_reason": decision_res.wait_reason,
                "trigger": decision_res.trigger,
                "invalidation": decision_res.invalidation,
                "because": decision_res.because,
                "entry_zone_low": decision_res.entry_zone_low,
                "entry_zone_high": decision_res.entry_zone_high,
                "stop_level": decision_res.stop_level,
                "targets": decision_res.targets,
                "confidence": decision_res.confidence,
                "hard_guard_active": decision_res.hard_guard_active,
                "hard_guard_reason": decision_res.hard_guard_reason,
                "capital_mode": decision_res.capital_mode,
                "capital_mode_reason": decision_res.capital_mode_reason,
                "max_size_mult": decision_res.max_size_mult,
                "locked_until": decision_res.locked_until,
                "call_sub": decision_res.call_sub.__dict__,
                "put_sub": decision_res.put_sub.__dict__,
            }
        except Exception:
            pass

        # ── Ideas via idea engine ──
        ideas_out = []
        briefing_data = {}
        try:
            bi = BriefingInput(
                symbol=sym, underlying_price=spot, df=window,
                regime=regime_res, sr=sr_res, chain_metrics=ChainMetrics(),
                vpa_bias=vpa_bias, active_alerts=[], expirations=[],
            )
            briefing = _ie.generate_briefing(bi)
            briefing = _ie.filter_by_playbook(briefing, pb_mode)

            ideas_out = [
                {"idea_type": i.idea_type.value, "direction": i.direction,
                 "symbol": i.symbol, "headline": i.headline, "rationale": i.rationale,
                 "entry_level": i.entry_level, "stop_level": i.stop_level,
                 "target_1": i.target_1, "target_2": i.target_2,
                 "reward_risk": i.reward_risk, "option_hint": i.option_hint,
                 "suitable_for": i.suitable_for, "confidence": i.confidence,
                 "score": i.score, "score_breakdown": i.score_breakdown,
                 "warning": i.warning}
                for i in briefing.trade_ideas
            ]

            briefing_data = {
                "bias": {"direction": briefing.bias.direction,
                         "strength": briefing.bias.strength,
                         "headline": briefing.bias.headline,
                         "bullets": briefing.bias.bullets,
                         "kill_switch": briefing.bias.kill_switch} if briefing.bias else None,
                "day_type": {"most_likely": briefing.day_type.most_likely.value,
                             "probabilities": briefing.day_type.probabilities,
                             "description": briefing.day_type.description,
                             "setup_implications": briefing.day_type.setup_implications} if briefing.day_type else None,
                "themes": [{"title": t.title, "description": t.description,
                            "direction": t.direction, "confidence": t.confidence,
                            "emoji": t.emoji} for t in briefing.themes],
                "conditional_plans": [{"condition": p.condition, "action": p.action,
                                       "option_description": p.option_description,
                                       "direction": p.direction,
                                       "trigger_level": p.trigger_level,
                                       "stop_description": p.stop_description,
                                       "targets": p.targets,
                                       "suitable_for": p.suitable_for}
                                      for p in briefing.conditional_plans],
            }
        except Exception:
            pass

        snapshot = {
            "checkpoint": cp,
            "time": cp_time,
            "underlying_price": spot,
            "bars": bars_out,
            "regime": regime_out,
            "key_levels": levels_out,
            "indicators": ind_out,
            "vpa": vpa_out,
            "volume_regime": vol_out,
            "proximity_score": prox_score,
            "proximity_detail": prox_detail,
            "posture": posture_out,
            "trade_ideas": ideas_out,
            "briefing": briefing_data,
        }
        snapshots.append(snapshot)

    # Summary
    return {
        "symbol": sym,
        "date": bt_date,
        "total_bars": len(df),
        "step_bars": step_bars,
        "num_snapshots": len(snapshots),
        "open": float(df["open"].iloc[0]),
        "close": float(df["close"].iloc[-1]),
        "day_change_pct": round((float(df["close"].iloc[-1]) - float(df["open"].iloc[0])) / float(df["open"].iloc[0]) * 100, 2),
        "snapshots": snapshots,
    }


app.mount("/static", StaticFiles(directory="app/static"), name="static")
