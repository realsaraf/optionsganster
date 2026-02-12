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
import hashlib, secrets

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response, Cookie
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware

import httpx

from app.config import settings
from app.polygon_client import PolygonClient, polygon_client
from app.vpa_engine import VPAEngine, VPASignal, vpa_engine
from app.greeks_engine import GreeksSignalEngine, greeks_engine

# ── Auth config ─────────────────────────────────────────────
AUTH_EMAIL = "realsaraf@gmail.com"
AUTH_PASS_HASH = hashlib.sha256("saraf1236".encode()).hexdigest()
_sessions: set[str] = set()  # in-memory session tokens


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
<title>OptionsGanster – Login</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{height:100vh;display:flex;align-items:center;justify-content:center;
  background:#0a0a1a;font-family:'Segoe UI',sans-serif;color:#e0e0e0}
.card{background:#111827;border:1px solid #1e293b;border-radius:12px;padding:40px;
  width:380px;box-shadow:0 8px 32px rgba(0,0,0,.5)}
h1{text-align:center;margin-bottom:8px;font-size:1.5rem}
.sub{text-align:center;color:#6b7280;font-size:.85rem;margin-bottom:28px}
.field{margin-bottom:16px}
label{display:block;font-size:.8rem;color:#9ca3af;margin-bottom:4px}
input{width:100%;padding:10px 12px;border-radius:6px;border:1px solid #374151;
  background:#1a1a2e;color:#e0e0e0;font-size:.95rem;outline:none}
input:focus{border-color:#3b82f6}
btn,.btn{display:block;width:100%;padding:12px;border:none;border-radius:6px;
  background:#3b82f6;color:#fff;font-size:1rem;cursor:pointer;margin-top:20px;font-weight:600}
.btn:hover{background:#2563eb}
.err{color:#ef4444;font-size:.85rem;text-align:center;margin-top:12px;min-height:20px}
</style>
</head>
<body>
<div class="card">
  <h1>OptionsGanster</h1>
  <p class="sub">Volume Price Analysis for Options</p>
  <form id="lf">
    <div class="field"><label>Email</label><input type="email" id="em" required autofocus></div>
    <div class="field"><label>Password</label><input type="password" id="pw" required></div>
    <button class="btn" type="submit">Sign In</button>
    <p class="err" id="err"></p>
  </form>
</div>
<script>
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
    poly: PolygonClient = Depends(get_polygon),
    engine: VPAEngine = Depends(get_vpa),
    gengine: GreeksSignalEngine = Depends(get_greeks_engine),
) -> AnalysisResponse:
    """Fetch options data and run VPA analysis."""
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

        # Fetch OHLCV data (async + cached)
        df = await poly.get_option_ohlcv(
            symbol=symbol.upper(),
            expiration=exp_date,
            strike=strike,
            right=right.upper(),
            start_date=start_dt,
            end_date=end_dt,
            interval_min=interval,
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
                _action_map = _put_action if right.upper() == "PUT" else _call_action
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
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/watchlist/prices")
async def get_watchlist_prices(
    symbols: str = Query("SPY,QQQ,IWM,AAPL,MSFT,NVDA,TSLA,AMD"),
    poly: PolygonClient = Depends(get_polygon),
):
    """Get real-time/delayed prices for watchlist symbols using snapshot API."""
    try:
        symbol_list = [s.strip().upper() for s in symbols.split(",")]

        # Use snapshot API for real-time/15-min delayed prices
        price_lookup = await poly.get_snapshot_prices(symbol_list)

        results = []
        for sym in symbol_list:
            snapshot = price_lookup.get(sym, {})
            last_price = float(snapshot.get("lastPrice", 0))
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
