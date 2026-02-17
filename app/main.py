"""
OptionsGanster – VPA Options Analysis Tool
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
import hashlib, json, logging, math, random, secrets, time
import asyncio

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response, Cookie, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware

import httpx
import pandas as pd

from app.config import settings
from app.polygon_client import PolygonClient, polygon_client
from app.vpa_engine import VPAEngine, VPASignal, VPAResult, vpa_engine
from app.greeks_engine import GreeksSignalEngine, greeks_engine
from app.live_feed import LiveFeedManager, live_feed_manager, mock_csv_playback, build_mock_sow
from app.fair_value_engine import FairValueEngine, fair_value_engine
from app.sr_engine import SREngine, sr_engine, SRResult
from app.data_layer import DataLayer

logger = logging.getLogger("optionsganster")

# ── Auth config ─────────────────────────────────────────────
# Multi-user store: email → {password_hash, role, display_name}
_USERS: dict[str, dict] = {
    "realsaraf@gmail.com": {
        "password_hash": hashlib.sha256("saraf1237".encode()).hexdigest(),
        "role": "admin",
        "display_name": "realsaraf",
    },
    "user@og.com": {
        "password_hash": hashlib.sha256("og1236".encode()).hexdigest(),
        "role": "general",
        "display_name": "OG Trader",
    },    
    "kutubtalukder@gmail.com": {
        "password_hash": hashlib.sha256("kt1236".encode()).hexdigest(),
        "role": "general",
        "display_name": "KT Trader",
    }
}
_sessions: dict[str, dict] = {}  # token → {email, role, display_name}


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
    yield
    # shutdown – stop live feed + close the shared httpx client
    await live_feed_manager.stop()
    await polygon_client.close()


app = FastAPI(
    title="OptionsGanster",
    description="Volume Price Analysis for Options Trading",
    version="2.0.0",
    lifespan=lifespan,
)


# ── Auth middleware ─────────────────────────────────────────
PUBLIC_PATHS = {"/login", "/api/login"}

class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if path in PUBLIC_PATHS or path.startswith("/static"):
            return await call_next(request)
        token = request.cookies.get("session")
        if not token or token not in _sessions:
            if path.startswith("/api/") or path.startswith("/ws/"):
                return Response(status_code=401, content="Unauthorized")
            return RedirectResponse("/login", status_code=302)
        # Attach user info to request state for downstream use
        request.state.user = _sessions[token]
        return await call_next(request)

app.add_middleware(AuthMiddleware)


# ── Dependency helpers ──────────────────────────────────────

def get_polygon() -> PolygonClient:
    return polygon_client


def get_vpa() -> VPAEngine:
    return vpa_engine


def get_greeks_engine() -> GreeksSignalEngine:
    return greeks_engine


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
<title>OptionsGanster – AI-Powered Options Signals</title>
<meta name="description" content="AI-driven options signals that cut through the noise. One actionable verdict — BUY, SELL, or HOLD — with a confidence score. Free to use.">
<meta property="og:type" content="website">
<meta property="og:url" content="https://optionsganster.com">
<meta property="og:title" content="OptionsGanster – AI-Powered Options Signals">
<meta property="og:description" content="Stop guessing. Our AI analyzes multiple market dimensions in real time and delivers one clear verdict with a confidence score. Free.">
<meta property="og:image" content="https://optionsganster.com/static/og-image.png">
<meta property="og:image:width" content="1200">
<meta property="og:image:height" content="630">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:title" content="OptionsGanster – AI-Powered Options Signals">
<meta name="twitter:description" content="Stop guessing. Our AI analyzes multiple market dimensions in real time and delivers one clear verdict with a confidence score.">
<meta name="twitter:image" content="https://optionsganster.com/static/og-image.png">
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
  <div class="nav-logo">Options<span>Ganster</span></div>
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

<footer>&copy; 2026 OptionsGanster. Built for traders who want an edge.</footer>

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
    return HTMLResponse(LOGIN_HTML)


class LoginRequest(BaseModel):
    email: str
    password: str

@app.post("/api/login")
async def login(body: LoginRequest):
    pw_hash = hashlib.sha256(body.password.encode()).hexdigest()
    user_record = _USERS.get(body.email)
    if not user_record or pw_hash != user_record["password_hash"]:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    token = secrets.token_hex(32)
    _sessions[token] = {
        "email": body.email,
        "role": user_record["role"],
        "display_name": user_record["display_name"],
    }
    resp = Response(
        content=json.dumps({"ok": True, "role": user_record["role"], "display_name": user_record["display_name"]}),
        media_type="application/json",
    )
    resp.set_cookie(key="session", value=token, httponly=True, max_age=86400 * 7, samesite="lax")
    return resp


@app.get("/api/logout")
async def logout(request: Request):
    token = request.cookies.get("session")
    if token:
        _sessions.pop(token, None)
    resp = RedirectResponse("/login", status_code=302)
    resp.delete_cookie("session")
    return resp


@app.get("/api/me")
async def get_me(request: Request):
    """Return the current user's profile (role, display_name)."""
    user = getattr(request.state, "user", None)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return {"email": user["email"], "role": user["role"], "display_name": user["display_name"]}


# ── Endpoints ───────────────────────────────────────────────

@app.get("/")
async def root():
    """Serve the main UI."""
    return FileResponse(
        "app/static/index_watchlist.html",
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


@app.get("/api/analyze/live")
async def analyze_live(
    symbol: str = Query(..., description="Underlying symbol"),
    expiration: str = Query(..., description="Expiration date (YYYY-MM-DD)"),
    strike: float = Query(..., description="Strike price"),
    right: str = Query(..., description="C for Call, P for Put"),
    after: Optional[str] = Query(None, description="Return only bars with datetime > this value"),
    mock: bool = Query(False, description="Use mock data for testing"),
    poly: PolygonClient = Depends(get_polygon),
    engine: VPAEngine = Depends(get_vpa),
):
    """Live mode endpoint – returns NEW bars & signals since *after*.
    Full composite / greeks / bias refresh is now pushed by the WS
    analysis loop (see /ws/live _analysis_loop).
    Note: mock parameter is accepted but ignored – always uses real API data.
    Mock mode only affects the WebSocket live feed (CSV playback as proxy)."""

    # ── Fetch OHLCV ──────────────────────────────────────────
    exp_date = datetime.strptime(expiration, "%Y-%m-%d").date()
    end_dt = date.today()
    start_dt = end_dt - timedelta(days=5)

    # Always clear cache for live
    poly.clear_ohlcv_cache(
        symbol=symbol.upper(), expiration=exp_date,
        strike=strike, right=right.upper(),
        start_date=start_dt, end_date=end_dt, interval_min=5,
    )
    poly.clear_stock_ohlcv_cache(
        symbol=symbol.upper(), start_date=start_dt,
        end_date=end_dt, interval_min=5,
    )

    try:
        df = await poly.get_option_ohlcv(
            symbol=symbol.upper(), expiration=exp_date,
            strike=strike, right=right.upper(),
            start_date=start_dt, end_date=end_dt, interval_min=5,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if df.empty:
        return {"bars": [], "signals": []}

    # Run VPA on the FULL history
    all_vpa = engine.analyze(df)

    # ── Build delta bars (after filter) ──────────────────────
    delta_df = df
    if after:
        delta_df = df[df["datetime"].astype(str) > after]

    bars = [
        dict(
            datetime=str(row["datetime"]),
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=int(row["volume"]),
        )
        for _, row in delta_df.iterrows()
    ]

    signals = [
        dict(
            signal=r.signal.value,
            confidence=r.confidence,
            description=r.description,
            datetime=r.datetime,
            price=r.price,
            volume=r.volume,
            volume_ratio=round(r.volume_ratio, 2),
        )
        for r in all_vpa
        if r.signal != VPASignal.NEUTRAL
    ]

    # Filter signals to only those in the new bars time range
    if after:
        signals = [s for s in signals if s["datetime"] > after]

    return {"bars": bars, "signals": signals}


@app.get("/api/analyze")
async def analyze_option(
    symbol: str = Query(..., description="Underlying symbol (e.g., QQQ, SPY)"),
    expiration: str = Query(..., description="Expiration date (YYYY-MM-DD)"),
    strike: float = Query(..., description="Strike price"),
    right: str = Query(..., description="C for Call, P for Put"),
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    interval: int = Query(1, description="Interval in minutes (1, 5, 15, etc.)"),
    nocache: bool = Query(False, description="Bypass OHLCV cache for live updates"),
    mock: bool = Query(False, description="Use mock data for testing"),
    poly: PolygonClient = Depends(get_polygon),
    engine: VPAEngine = Depends(get_vpa),
    gengine: GreeksSignalEngine = Depends(get_greeks_engine),
    dl: DataLayer = Depends(get_data_layer),
    sr_eng: SREngine = Depends(get_sr_engine),
) -> AnalysisResponse:
    """Fetch options data and run VPA analysis.
    Note: mock parameter is accepted but ignored – analysis always uses real API data.
    Mock mode only affects the WebSocket live feed (CSV playback as proxy)."""
    try:
        exp_date = datetime.strptime(expiration, "%Y-%m-%d").date()
        end_dt = (
            datetime.strptime(end_date, "%Y-%m-%d").date()
            if end_date
            else date.today()
        )
        start_dt = (
            datetime.strptime(start_date, "%Y-%m-%d").date()
            if start_date
            else end_dt - timedelta(days=5)
        )

        # Clear cache if nocache=true (for live mode)
        if nocache:
            poly.clear_ohlcv_cache(
                symbol=symbol.upper(),
                expiration=exp_date,
                strike=strike,
                right=right.upper(),
                start_date=start_dt,
                end_date=end_dt,
                interval_min=interval,
            )
            poly.clear_stock_ohlcv_cache(
                symbol=symbol.upper(),
                start_date=start_dt,
                end_date=end_dt,
                interval_min=interval,
            )

        # Fetch OHLCV data (async + cached) — options + underlying in parallel
        import asyncio as _aio

        async def _safe_stock_fetch():
            try:
                return await poly.get_stock_ohlcv(
                    symbol=symbol.upper(),
                    start_date=start_dt,
                    end_date=end_dt,
                    interval_min=interval,
                )
            except Exception as exc:
                print(f"[ANALYZE] Stock fetch failed: {exc}")
                import pandas as _pd
                return _pd.DataFrame(
                    columns=["datetime", "open", "high", "low", "close", "volume"]
                )

        df, stock_df = await _aio.gather(
            poly.get_option_ohlcv(
                symbol=symbol.upper(),
                expiration=exp_date,
                strike=strike,
                right=right.upper(),
                start_date=start_dt,
                end_date=end_dt,
                interval_min=interval,
            ),
            _safe_stock_fetch(),
        )

        if df.empty:
            raise HTTPException(
                status_code=404, detail="No data found for this contract"
            )

        # Run VPA analysis (CPU-bound, but fast enough for single requests)
        vpa_results = engine.analyze(df)
        bias = engine.get_bias(vpa_results)

        # ── Compute underlying trend from stock bars ─────
        underlying_trend = None
        if not stock_df.empty and len(stock_df) >= 2:
            day_open = stock_df.iloc[0]["open"]
            day_close = stock_df.iloc[-1]["close"]
            pct_change = (day_close - day_open) / day_open * 100 if day_open else 0
            if pct_change > 0.15:
                underlying_trend = "UP"
            elif pct_change < -0.15:
                underlying_trend = "DOWN"
            else:
                underlying_trend = "FLAT"

        # ── Compute DTE ──────────────────────────────────
        dte = (exp_date - date.today()).days if exp_date else None

        # ── Volume regime from VPA engine ────────────────
        vol_regime_info = None
        if not stock_df.empty:
            vol_regime_info = engine.get_volume_regime(stock_df)

        # ── S/R, Fibonacci, Volume POC analysis ────────────
        sr_result_obj: SRResult | None = None
        sr_response: SRResponse | None = None
        try:
            # Get underlying price for SR analysis
            sr_underlying_price = 0.0
            if not stock_df.empty:
                sr_underlying_price = float(stock_df.iloc[-1]["close"])
            if sr_underlying_price <= 0:
                try:
                    sr_underlying_price = await poly.get_underlying_price(symbol.upper())
                except Exception:
                    pass

            if sr_underlying_price > 0:
                # Fetch daily bars (90 days) + intraday bars (20 days, 5-min)
                daily_bars_df, intraday_bars_df = await _aio.gather(
                    dl.get_daily_bars(symbol.upper(), num_days=90),
                    dl.get_intraday_bars(symbol.upper(), lookback_days=20, interval_min=5),
                )

                sr_result_obj = sr_eng.analyze(
                    daily_bars=daily_bars_df,
                    underlying_price=sr_underlying_price,
                    intraday_bars=intraday_bars_df,
                )

                # Cache for potential reuse
                dl.set_cached_sr(symbol.upper(), {
                    "poc": sr_result_obj.poc,
                    "vah": sr_result_obj.vah,
                    "val": sr_result_obj.val,
                })

                # Build response model
                sr_response = SRResponse(
                    levels=[
                        SRLevelResponse(
                            price=l.price,
                            kind=l.kind,
                            source=l.source,
                            strength=l.strength,
                            label=l.label,
                        )
                        for l in sr_result_obj.levels[:25]  # Cap at 25 levels
                    ],
                    poc=sr_result_obj.poc,
                    vah=sr_result_obj.vah,
                    val=sr_result_obj.val,
                    fib_high=sr_result_obj.fib_high,
                    fib_low=sr_result_obj.fib_low,
                    nearest_support=sr_result_obj.nearest_support,
                    nearest_resistance=sr_result_obj.nearest_resistance,
                    proximity_score=sr_result_obj.proximity_score,
                    proximity_detail=sr_result_obj.proximity_detail,
                )
        except Exception as sr_err:
            print(f"S/R analysis error (non-fatal): {sr_err}")

        # ── Composite Greeks analysis ────────────────────
        composite_response = None
        contract_snap = None
        try:
            import asyncio
            contract_snap, chain_snap = await asyncio.gather(
                poly.get_option_contract_snapshot(
                    symbol.upper(), exp_date, strike, right.upper()
                ),
                poly.get_options_chain_snapshot(
                    symbol.upper(), exp_date
                ),
            )
            if contract_snap:
                comp_result = gengine.analyze(
                    contract_snapshot=contract_snap,
                    chain_data=chain_snap,
                    vpa_bias=bias,
                    contract_type=right.upper(),
                    underlying_trend=underlying_trend,
                    dte=dte,
                    volume_regime=vol_regime_info["regime"] if vol_regime_info else None,
                    sr_result=sr_result_obj,
                )
                # Map composite signal → actionable BUY / SELL / HOLD
                # For calls: bullish underlying = BUY (the call)
                # For puts:  bearish underlying = BUY (the put), so invert
                _call_action = {
                    "strong_buy": "STRONG BUY",
                    "buy": "BUY",
                    "lean_bullish": "BUY",
                    "neutral": "HOLD",
                    "lean_bearish": "SELL",
                    "sell": "SELL",
                    "strong_sell": "STRONG SELL",
                }
                _put_action = {
                    "strong_buy": "STRONG SELL",
                    "buy": "SELL",
                    "lean_bullish": "SELL",
                    "neutral": "HOLD",
                    "lean_bearish": "BUY",
                    "sell": "BUY",
                    "strong_sell": "STRONG BUY",
                }
                _action_map = _put_action if right.upper() == "P" else _call_action
                composite_response = CompositeSignalResponse(
                    signal=comp_result.signal.value,
                    action=_action_map.get(comp_result.signal.value, "HOLD"),
                    score=comp_result.score,
                    confidence=comp_result.confidence,
                    trade_archetype=comp_result.trade_archetype.value,
                    archetype_description=comp_result.archetype_description,
                    factors=[
                        FactorScoreResponse(
                            name=f.name,
                            score=f.score,
                            confidence=f.confidence,
                            weight=f.weight,
                            detail=f.detail,
                        )
                        for f in comp_result.factors
                    ],
                    greeks=GreeksResponse(
                        delta=comp_result.greeks.delta,
                        gamma=comp_result.greeks.gamma,
                        theta=comp_result.greeks.theta,
                        vega=comp_result.greeks.vega,
                        iv=comp_result.greeks.iv,
                        open_interest=comp_result.greeks.open_interest,
                        volume=comp_result.greeks.volume,
                        underlying_price=comp_result.greeks.underlying_price,
                        break_even=comp_result.greeks.break_even,
                        last_price=comp_result.greeks.last_price,
                    ),
                    chain_metrics=ChainMetricsResponse(
                        iv_rank=comp_result.chain_metrics.iv_rank,
                        iv_percentile=comp_result.chain_metrics.iv_percentile,
                        put_call_oi_ratio=comp_result.chain_metrics.put_call_oi_ratio,
                        put_call_volume_ratio=comp_result.chain_metrics.put_call_volume_ratio,
                        total_call_oi=comp_result.chain_metrics.total_call_oi,
                        total_put_oi=comp_result.chain_metrics.total_put_oi,
                        total_call_volume=comp_result.chain_metrics.total_call_volume,
                        total_put_volume=comp_result.chain_metrics.total_put_volume,
                        net_gex=comp_result.chain_metrics.net_gex,
                        gex_regime=comp_result.chain_metrics.gex_regime,
                        max_pain=comp_result.chain_metrics.max_pain,
                        uoa_detected=comp_result.chain_metrics.uoa_detected,
                        uoa_details=comp_result.chain_metrics.uoa_details,
                        weighted_iv=comp_result.chain_metrics.weighted_iv,
                    ),
                    recommendation=comp_result.recommendation,
                    warnings=comp_result.warnings,
                )
        except Exception as comp_err:
            print(f"Composite analysis error (non-fatal): {comp_err}")

        # ── Fair Value (Black-Scholes) ───────────────────
        fair_value_response = None
        try:
            # Fetch daily closes for historical vol (30-day window needs ~45 daily prices)
            daily_closes = await poly.get_stock_daily_ohlcv(symbol.upper(), num_days=45)

            # Get bid/ask from contract snapshot (already fetched above)
            fv_bid = 0.0
            fv_ask = 0.0
            fv_iv = 0.0
            fv_underlying = 0.0
            if contract_snap:
                fv_bid = float(contract_snap.get("bid", 0) or 0)
                fv_ask = float(contract_snap.get("ask", 0) or 0)
                fv_iv = float(contract_snap.get("iv", 0) or 0)
                fv_underlying = float(contract_snap.get("underlying_price", 0) or 0)
            # Fallback underlying price from price cache
            if fv_underlying <= 0:
                try:
                    fv_underlying = await poly.get_underlying_price(symbol.upper())
                except Exception:
                    pass

            if daily_closes and fv_underlying > 0:
                fv_result = fair_value_engine.analyze(
                    underlying_price=fv_underlying,
                    strike=strike,
                    expiration=exp_date,
                    contract_type=right.upper(),
                    daily_closes=daily_closes,
                    market_bid=fv_bid,
                    market_ask=fv_ask,
                    market_iv=fv_iv,
                )
                fair_value_response = FairValueResponse(
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
            print(f"Fair value analysis error (non-fatal): {fv_err}")

        bars = [
            OHLCVBar(
                datetime=str(row["datetime"]),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=int(row["volume"]),
            )
            for _, row in df.iterrows()
        ]

        signals = [
            VPASignalResponse(
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

        # Build underlying bars
        underlying_bars = [
            OHLCVBar(
                datetime=str(row["datetime"]),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=int(row["volume"]),
            )
            for _, row in stock_df.iterrows()
        ] if not stock_df.empty else []

        return AnalysisResponse(
            symbol=symbol.upper(),
            expiration=expiration,
            strike=strike,
            right=right.upper(),
            interval=interval,
            bars=bars,
            signals=signals,
            bias=bias,
            composite=composite_response,
            underlying_bars=underlying_bars,
            volume_regime=vol_regime_info,
            fair_value=fair_value_response,
            support_resistance=sr_response,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
    gengine: GreeksSignalEngine = Depends(get_greeks_engine),
    engine: VPAEngine = Depends(get_vpa),
):
    """
    Scan watchlist symbols for trade ideas.
    Results are cached per-symbol for 5 minutes.
    For each symbol, check nearest expiration chain and find setups
    where the composite signal is BUY / STRONG_BUY / SELL / STRONG_SELL.
    Returns top ideas sorted by |score| × confidence.
    """
    import asyncio

    symbol_list = [s.strip().upper() for s in symbols.split(",")]
    ideas: list[dict] = []
    now = _time.time()

    # Separate symbols into cached vs need-scan
    symbols_to_scan: list[str] = []
    for sym in symbol_list:
        cached = _ideas_cache.get(sym)
        if not refresh and cached and (now - cached["ts"]) < _IDEAS_TTL:
            ideas.extend(cached["ideas"])
        else:
            symbols_to_scan.append(sym)

    # Per-symbol results collector for cache storage
    _sym_ideas: dict[str, list[dict]] = {sym: [] for sym in symbols_to_scan}

    async def scan_symbol(sym: str):
        try:
            # Get expirations — use first 2 nearest
            expirations = await poly.get_expirations(sym)
            if not expirations:
                return

            # Get underlying price for ATM calculation
            price_data = await poly.get_snapshot_prices([sym])
            snap_data = price_data.get(sym, {})
            underlying_price = float(snap_data.get("lastPrice", 0) or snap_data.get("prevClose", 0))
            if underlying_price == 0:
                return

            # ── Fetch stock bars for VPA bias + underlying trend ──
            import pandas as pd
            vpa_bias = None
            underlying_trend = None
            vol_regime = None
            try:
                stock_df = await poly.get_stock_ohlcv(
                    symbol=sym,
                    start_date=date.today(),
                    end_date=date.today(),
                    interval_min=5,
                )
                if not stock_df.empty and len(stock_df) >= 2:
                    # VPA bias
                    vpa_results = engine.analyze(stock_df)
                    vpa_bias = engine.get_bias(vpa_results)

                    # Underlying trend from open→close
                    day_open = stock_df.iloc[0]["open"]
                    day_close = stock_df.iloc[-1]["close"]
                    pct_change = (day_close - day_open) / day_open * 100 if day_open else 0
                    if pct_change > 0.15:
                        underlying_trend = "UP"
                    elif pct_change < -0.15:
                        underlying_trend = "DOWN"
                    else:
                        underlying_trend = "FLAT"

                    # Volume regime
                    vol_regime_info = engine.get_volume_regime(stock_df)
                    vol_regime = vol_regime_info.get("regime") if vol_regime_info else None
            except Exception as vpa_err:
                print(f"[IDEAS] VPA/trend fetch failed for {sym}: {vpa_err}")

            for exp in expirations[:2]:  # Check nearest 2 expirations
                dte = (exp - date.today()).days
                if dte < 0:
                    continue

                # Get chain snapshot
                chain = await poly.get_options_chain_snapshot(sym, exp)
                if not chain:
                    continue

                # Find ATM strikes (2 closest to underlying price for calls + puts)
                strikes_seen = sorted(set(c["strike"] for c in chain if c["strike"] > 0))
                if not strikes_seen:
                    continue

                # Find the ATM strike
                atm_strike = min(strikes_seen, key=lambda s: abs(s - underlying_price))
                atm_idx = strikes_seen.index(atm_strike)

                # Check ATM and 1 strike OTM for both calls and puts
                check_strikes = set()
                check_strikes.add(atm_strike)
                if atm_idx + 1 < len(strikes_seen):
                    check_strikes.add(strikes_seen[atm_idx + 1])  # 1 OTM call
                if atm_idx - 1 >= 0:
                    check_strikes.add(strikes_seen[atm_idx - 1])  # 1 OTM put

                for strike_val in check_strikes:
                    for ct in ["call", "put"]:
                        # Find the contract in chain
                        contract = next(
                            (c for c in chain
                             if c["strike"] == strike_val
                             and c["contract_type"].lower() == ct),
                            None,
                        )
                        if not contract:
                            continue

                        # Skip illiquid contracts
                        if (contract.get("volume", 0) or 0) < 10 and (contract.get("open_interest", 0) or 0) < 50:
                            continue

                        # Skip too-cheap contracts
                        last_price = contract.get("last_price", 0) or 0
                        if last_price < 0.10:
                            continue

                        # Build snapshot dict for greeks engine
                        contract_snap = {
                            "greeks": contract.get("greeks", {}),
                            "iv": contract.get("iv", 0),
                            "open_interest": contract.get("open_interest", 0),
                            "volume": contract.get("volume", 0),
                            "underlying_price": underlying_price,
                            "break_even": 0,
                            "last_price": last_price,
                        }

                        right = "C" if ct == "call" else "P"
                        try:
                            result = gengine.analyze(
                                contract_snapshot=contract_snap,
                                chain_data=chain,
                                vpa_bias=vpa_bias,
                                contract_type=right,
                                underlying_trend=underlying_trend,
                                dte=dte,
                                volume_regime=vol_regime,
                            )
                        except Exception:
                            continue

                        # Only keep actionable signals (skip neutral)
                        sig = result.signal.value
                        if sig == "neutral":
                            continue

                        # Map signal to action
                        _call_action = {
                            "strong_buy": "STRONG BUY", "buy": "BUY",
                            "lean_bullish": "LEAN BUY",
                            "strong_sell": "STRONG SELL", "sell": "SELL",
                            "lean_bearish": "LEAN SELL",
                        }
                        _put_action = {
                            "strong_buy": "STRONG SELL", "buy": "SELL",
                            "lean_bullish": "LEAN SELL",
                            "strong_sell": "STRONG BUY", "sell": "BUY",
                            "lean_bearish": "LEAN BUY",
                        }
                        action_map = _put_action if right == "P" else _call_action
                        action = action_map.get(sig, "HOLD")

                        exp_str = exp.strftime("%Y-%m-%d")
                        # Friendly exp label
                        exp_short = exp.strftime("%b %d")

                        idea_dict = {
                            "symbol": sym,
                            "expiration": exp_str,
                            "exp_label": f"{exp_short} ({dte}d)",
                            "strike": strike_val,
                            "right": right,
                            "type_label": "CALL" if right == "C" else "PUT",
                            "price": round(last_price, 2),
                            "signal": sig,
                            "action": action,
                            "score": result.score,
                            "confidence": result.confidence,
                            "archetype": result.trade_archetype.value,
                            "recommendation": result.recommendation[:120],
                        }
                        ideas.append(idea_dict)
                        _sym_ideas[sym].append(idea_dict)

        except Exception as e:
            print(f"[IDEAS] Error scanning {sym}: {e}")

    # Scan uncached symbols in parallel (with concurrency limit)
    if symbols_to_scan:
        sem = asyncio.Semaphore(3)  # Max 3 concurrent to avoid rate limits

        async def throttled_scan(sym):
            async with sem:
                await scan_symbol(sym)

        await asyncio.gather(*(throttled_scan(sym) for sym in symbols_to_scan))

        # Store results in per-symbol cache
        for sym in symbols_to_scan:
            _ideas_cache[sym] = {"ideas": _sym_ideas[sym], "ts": _time.time()}

    # Sort by |score| × confidence descending → best ideas first
    ideas.sort(key=lambda x: abs(x["score"]) * x["confidence"], reverse=True)

    # Return top 20 ideas
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
                        underlying_trend = "UP" if pct > 0.15 else "DOWN" if pct < -0.15 else "FLAT"
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

                # Composite (greeks + chain)
                try:
                    # Clear snapshot cache for fresh greeks/price data
                    if hasattr(polygon_client, '_contract_snapshot_cache'):
                        snap_key = ("snapshot", symbol, exp_date, strike, right)
                        polygon_client._contract_snapshot_cache.pop(snap_key, None)

                    contract_snap, chain_snap = await asyncio.gather(
                        polygon_client.get_option_contract_snapshot(
                            symbol, exp_date, strike, right
                        ),
                        polygon_client.get_options_chain_snapshot(
                            symbol, exp_date
                        ),
                    )
                    if contract_snap:
                        comp = greeks_engine.analyze(
                            contract_snapshot=contract_snap,
                            chain_data=chain_snap,
                            vpa_bias=bias,
                            contract_type=right,
                            underlying_trend=underlying_trend,
                            dte=dte,
                            volume_regime=vol_regime,
                        )
                        _call_action = {
                            "strong_buy": "STRONG BUY", "buy": "BUY",
                            "lean_bullish": "BUY", "neutral": "HOLD",
                            "lean_bearish": "SELL", "sell": "SELL",
                            "strong_sell": "STRONG SELL",
                        }
                        _put_action = {
                            "strong_buy": "STRONG SELL", "buy": "SELL",
                            "lean_bullish": "SELL", "neutral": "HOLD",
                            "lean_bearish": "BUY", "sell": "BUY",
                            "strong_sell": "STRONG BUY",
                        }
                        _action_map = _put_action if right == "P" else _call_action
                        payload["composite"] = {
                            "signal": comp.signal.value,
                            "action": _action_map.get(comp.signal.value, "HOLD"),
                            "score": comp.score,
                            "confidence": comp.confidence,
                            "trade_archetype": comp.trade_archetype.value,
                            "archetype_description": comp.archetype_description,
                            "factors": [
                                {"name": f.name, "score": f.score,
                                 "confidence": f.confidence, "weight": f.weight,
                                 "detail": f.detail}
                                for f in comp.factors
                            ],
                            "greeks": {
                                "delta": comp.greeks.delta,
                                "gamma": comp.greeks.gamma,
                                "theta": comp.greeks.theta,
                                "vega": comp.greeks.vega,
                                "iv": comp.greeks.iv,
                                "open_interest": comp.greeks.open_interest,
                                "volume": comp.greeks.volume,
                                "underlying_price": comp.greeks.underlying_price,
                                "break_even": comp.greeks.break_even,
                                "last_price": comp.greeks.last_price,
                            },
                            "chain_metrics": {
                                "iv_rank": comp.chain_metrics.iv_rank,
                                "iv_percentile": comp.chain_metrics.iv_percentile,
                                "put_call_oi_ratio": comp.chain_metrics.put_call_oi_ratio,
                                "put_call_volume_ratio": comp.chain_metrics.put_call_volume_ratio,
                                "total_call_oi": comp.chain_metrics.total_call_oi,
                                "total_put_oi": comp.chain_metrics.total_put_oi,
                                "total_call_volume": comp.chain_metrics.total_call_volume,
                                "total_put_volume": comp.chain_metrics.total_put_volume,
                                "net_gex": comp.chain_metrics.net_gex,
                                "gex_regime": comp.chain_metrics.gex_regime,
                                "max_pain": comp.chain_metrics.max_pain,
                                "uoa_detected": comp.chain_metrics.uoa_detected,
                                "uoa_details": comp.chain_metrics.uoa_details,
                                "weighted_iv": comp.chain_metrics.weighted_iv,
                            },
                            "recommendation": comp.recommendation,
                            "warnings": comp.warnings,
                        }
                except Exception as comp_err:
                    print(f"[WS-Analysis] Composite error (non-fatal): {comp_err}")

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


# ── Static files (must be last so it doesn't shadow API routes) ─
app.mount("/static", StaticFiles(directory="app/static"), name="static")
