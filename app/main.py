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
import hashlib, math, random, secrets, time

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response, Cookie
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

# ── Auth config ─────────────────────────────────────────────
AUTH_EMAIL = "realsaraf@gmail.com"
AUTH_PASS_HASH = hashlib.sha256("saraf1236".encode()).hexdigest()
_sessions: set[str] = set()  # in-memory session tokens


# ── Mock data generator for live-mode testing ───────────────
class MockDataGenerator:
    """Generates fake OHLCV bars that tick every second for testing live mode."""

    def __init__(self):
        self._streams: dict[str, dict] = {}  # keyed by contract signature

    def _key(self, symbol: str, expiration: str, strike: float, right: str) -> str:
        return f"{symbol}|{expiration}|{strike}|{right}"

    def get_bars(self, symbol: str, expiration: str, strike: float, right: str,
                 after: Optional[str] = None) -> tuple[list[dict], list[dict]]:
        """Return (bars, signals).  If *after* is set, only return bars newer than it."""
        key = self._key(symbol, expiration, strike, right)
        now = datetime.utcnow()

        if key not in self._streams:
            # Seed with 30 historical bars, each 1 second apart
            base_price = round(random.uniform(2.0, 10.0), 2)
            bars: list[dict] = []
            start = now - timedelta(seconds=30)
            price = base_price
            for i in range(30):
                dt = start + timedelta(seconds=i)
                change = round(random.gauss(0, 0.05), 2)
                o = round(price, 2)
                c = round(price + change, 2)
                h = round(max(o, c) + abs(random.gauss(0, 0.02)), 2)
                l = round(min(o, c) - abs(random.gauss(0, 0.02)), 2)
                vol = random.randint(50, 500)
                bars.append(dict(datetime=dt.strftime("%Y-%m-%d %H:%M:%S"),
                                 open=o, high=h, low=l, close=c, volume=vol))
                price = c
            self._streams[key] = dict(bars=bars, last_price=price)
        else:
            stream = self._streams[key]
            # Append a new bar for the current second
            last_bar_dt = datetime.strptime(stream["bars"][-1]["datetime"], "%Y-%m-%d %H:%M:%S")
            seconds_since = max(1, int((now - last_bar_dt).total_seconds()))
            price = stream["last_price"]
            for s in range(seconds_since):
                dt = last_bar_dt + timedelta(seconds=s + 1)
                change = round(random.gauss(0, 0.05), 2)
                o = round(price, 2)
                c = round(price + change, 2)
                h = round(max(o, c) + abs(random.gauss(0, 0.02)), 2)
                l = round(min(o, c) - abs(random.gauss(0, 0.02)), 2)
                vol = random.randint(50, 500)
                stream["bars"].append(dict(datetime=dt.strftime("%Y-%m-%d %H:%M:%S"),
                                           open=o, high=h, low=l, close=c, volume=vol))
                price = c
            stream["last_price"] = price

        all_bars = self._streams[key]["bars"]

        # Filter by *after* to return only new bars
        if after:
            all_bars = [b for b in all_bars if b["datetime"] > after]

        # Generate mock signals for any new bar with |change| > 0.06
        signals: list[dict] = []
        for b in all_bars:
            change = b["close"] - b["open"]
            vol_ratio = round(random.uniform(0.5, 3.0), 2)
            if abs(change) > 0.06:
                sig_type = "strong_bullish" if change > 0 else "strong_bearish"
                signals.append(dict(
                    signal=sig_type,
                    confidence=round(min(abs(change) * 5, 1.0), 2),
                    description=f"Mock {sig_type.replace('_', ' ')} signal",
                    datetime=b["datetime"],
                    price=b["close"],
                    volume=b["volume"],
                    volume_ratio=vol_ratio,
                ))

        return all_bars, signals


_mock_gen = MockDataGenerator()


# ── Mock watchlist price generator ──────────────────────────
_mock_stock_prices: dict[str, float] = {}

def _get_mock_watchlist_prices(symbols: list[str]) -> list[dict]:
    """Return fake stock prices that drift randomly each call."""
    BASE_PRICES = {
        "SPY": 600, "QQQ": 520, "IWM": 225, "AAPL": 245,
        "MSFT": 425, "NVDA": 135, "TSLA": 350, "AMD": 120,
        "AMZN": 230, "META": 680, "GOOGL": 195,
    }
    results = []
    for sym in symbols:
        if sym not in _mock_stock_prices:
            _mock_stock_prices[sym] = BASE_PRICES.get(sym, round(random.uniform(50, 500), 2))
        # Random walk
        _mock_stock_prices[sym] = round(
            _mock_stock_prices[sym] * (1 + random.gauss(0, 0.001)), 2
        )
        price = _mock_stock_prices[sym]
        change = round(random.uniform(-2, 2), 2)
        results.append({
            "symbol": sym,
            "price": price,
            "change": change,
            "changePct": round(change / price * 100, 2),
            "high": round(price + abs(random.gauss(0, 1)), 2),
            "low": round(price - abs(random.gauss(0, 1)), 2),
            "volume": random.randint(1_000_000, 80_000_000),
        })
    return results


# ── Lifespan (startup / shutdown) ───────────────────────────

@asynccontextmanager
async def lifespan(application: FastAPI):
    # startup – nothing extra needed, lazy client init
    yield
    # shutdown – close the shared httpx client
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
            if path.startswith("/api/"):
                return Response(status_code=401, content="Unauthorized")
            return RedirectResponse("/login", status_code=302)
        return await call_next(request)

app.add_middleware(AuthMiddleware)


# ── Dependency helpers ──────────────────────────────────────

def get_polygon() -> PolygonClient:
    return polygon_client


def get_vpa() -> VPAEngine:
    return vpa_engine


def get_greeks_engine() -> GreeksSignalEngine:
    return greeks_engine


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
      <div class="stat"><div class="stat-value"><span class="green">$300</span></div><div class="stat-label">Monthly</div></div>
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
    if body.email != AUTH_EMAIL or pw_hash != AUTH_PASS_HASH:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    token = secrets.token_hex(32)
    _sessions.add(token)
    resp = Response(content='{"ok":true}', media_type="application/json")
    resp.set_cookie(key="session", value=token, httponly=True, max_age=86400 * 7, samesite="lax")
    return resp


@app.get("/api/logout")
async def logout(request: Request):
    token = request.cookies.get("session")
    if token:
        _sessions.discard(token)
    resp = RedirectResponse("/login", status_code=302)
    resp.delete_cookie("session")
    return resp


# ── Endpoints ───────────────────────────────────────────────

@app.get("/")
async def root():
    """Serve the main UI."""
    return FileResponse("app/static/index_watchlist.html")


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
):
    """Live mode endpoint – returns only NEW bars & signals since *after*."""

    if mock:
        bars, signals = _mock_gen.get_bars(symbol, expiration, strike, right, after=after)
        return {
            "bars": bars,
            "signals": signals,
            "is_mock": True,
        }

    # Real live – delegate to full analyze with nocache, then filter
    # We import the full endpoint logic inline so we can filter the result
    from app.polygon_client import polygon_client as poly
    from app.vpa_engine import vpa_engine as engine

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

    # Filter to only bars after the given timestamp
    if after:
        df = df[df["datetime"].astype(str) > after]

    if df.empty:
        return {"bars": [], "signals": []}

    vpa_results = engine.analyze(df)

    bars = [
        dict(
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
) -> AnalysisResponse:
    """Fetch options data and run VPA analysis."""
    try:
        # ── Mock mode: return fake data immediately ───────────
        if mock:
            mock_bars, mock_signals = _mock_gen.get_bars(symbol, expiration, strike, right)
            last_price = mock_bars[-1]["close"] if mock_bars else 5.0

            # Mock bias based on recent price action
            if len(mock_bars) >= 2:
                direction = mock_bars[-1]["close"] - mock_bars[0]["open"]
                bias_label = "bullish" if direction > 0 else "bearish" if direction < 0 else "neutral"
            else:
                bias_label = "neutral"

            # Mock underlying bars (parallel to option bars)
            mock_underlying = []
            underlying_price = _mock_stock_prices.get(symbol.upper(), 500.0)
            for b in mock_bars:
                chg = round(random.gauss(0, 0.3), 2)
                o = round(underlying_price, 2)
                c = round(underlying_price + chg, 2)
                mock_underlying.append(dict(
                    datetime=b["datetime"],
                    open=o, high=round(max(o, c) + 0.1, 2),
                    low=round(min(o, c) - 0.1, 2), close=c,
                    volume=random.randint(100000, 2000000),
                ))
                underlying_price = c

            # Mock composite signal
            score = round(random.uniform(-0.5, 0.5), 3)
            action_map = {True: "BUY", False: "SELL"} if abs(score) > 0.1 else {True: "HOLD", False: "HOLD"}
            action = action_map.get(score > 0, "HOLD")
            if abs(score) > 0.3:
                action = f"STRONG {action}"
            mock_composite = CompositeSignalResponse(
                signal="lean_bullish" if score > 0 else "lean_bearish",
                action=action,
                score=score,
                confidence=round(random.uniform(0.4, 0.9), 2),
                trade_archetype="momentum_play",
                archetype_description="Mock archetype – testing live mode",
                factors=[
                    FactorScoreResponse(name="VPA", score=round(random.uniform(-0.3, 0.3), 2),
                                        confidence=0.7, weight=0.25, detail="Mock VPA factor"),
                    FactorScoreResponse(name="Greeks", score=round(random.uniform(-0.2, 0.2), 2),
                                        confidence=0.6, weight=0.25, detail="Mock Greeks factor"),
                    FactorScoreResponse(name="Flow", score=round(random.uniform(-0.2, 0.2), 2),
                                        confidence=0.65, weight=0.25, detail="Mock Flow factor"),
                    FactorScoreResponse(name="Structure", score=round(random.uniform(-0.2, 0.2), 2),
                                        confidence=0.55, weight=0.25, detail="Mock Structure factor"),
                ],
                greeks=GreeksResponse(
                    delta=round(random.uniform(0.2, 0.8), 4),
                    gamma=round(random.uniform(0.01, 0.05), 4),
                    theta=round(random.uniform(-0.1, -0.01), 4),
                    vega=round(random.uniform(0.05, 0.3), 4),
                    iv=round(random.uniform(0.2, 0.6), 4),
                    open_interest=random.randint(500, 50000),
                    volume=random.randint(100, 10000),
                    underlying_price=round(underlying_price, 2),
                    break_even=round(strike + last_price, 2),
                    last_price=round(last_price, 2),
                ),
                chain_metrics=ChainMetricsResponse(
                    iv_rank=round(random.uniform(20, 80), 1),
                    iv_percentile=round(random.uniform(20, 80), 1),
                    put_call_oi_ratio=round(random.uniform(0.5, 2.0), 2),
                    put_call_volume_ratio=round(random.uniform(0.5, 2.0), 2),
                    total_call_oi=random.randint(100000, 500000),
                    total_put_oi=random.randint(100000, 500000),
                    total_call_volume=random.randint(50000, 200000),
                    total_put_volume=random.randint(50000, 200000),
                    net_gex=round(random.uniform(-1e9, 1e9), 0),
                    gex_regime="positive" if random.random() > 0.5 else "negative",
                    max_pain=round(strike + random.uniform(-10, 10), 0),
                    uoa_detected=random.random() > 0.7,
                    uoa_details=[],
                    weighted_iv=round(random.uniform(0.2, 0.5), 4),
                ),
                recommendation=f"Mock {action} recommendation – confidence {round(random.uniform(40, 90))}%",
            )

            return AnalysisResponse(
                symbol=symbol.upper(),
                expiration=expiration,
                strike=strike,
                right=right.upper(),
                interval=interval,
                bars=[OHLCVBar(**b) for b in mock_bars],
                signals=[VPASignalResponse(**s) for s in mock_signals],
                bias={"bias": bias_label, "strength": round(random.uniform(0.3, 0.9), 2),
                      "reason": f"Mock {bias_label} bias"},
                composite=mock_composite,
                underlying_bars=[OHLCVBar(**u) for u in mock_underlying],
            )

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

        # ── Composite Greeks analysis ────────────────────
        composite_response = None
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
                )
        except Exception as comp_err:
            print(f"Composite analysis error (non-fatal): {comp_err}")

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
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/watchlist/prices")
async def get_watchlist_prices(
    symbols: str = Query("SPY,QQQ,IWM,AAPL,MSFT,NVDA,TSLA,AMD"),
    mock: bool = Query(False, description="Use mock data for testing"),
    poly: PolygonClient = Depends(get_polygon),
):
    """Get real-time/delayed prices for watchlist symbols using snapshot API."""
    try:
        symbol_list = [s.strip().upper() for s in symbols.split(",")]

        if mock:
            return {"prices": _get_mock_watchlist_prices(symbol_list)}

        # Use snapshot API for real-time/15-min delayed prices
        price_lookup = await poly.get_snapshot_prices(symbol_list)

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


# ── Static files (must be last so it doesn't shadow API routes) ─
app.mount("/static", StaticFiles(directory="app/static"), name="static")
