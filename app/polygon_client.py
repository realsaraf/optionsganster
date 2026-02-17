"""
Polygon/Massive API Client - Fetches historical options OHLCV data
Fully async with TTL caching and pagination support.
Docs: https://polygon.io/docs/options
"""
from datetime import datetime, date
from typing import Optional
from zoneinfo import ZoneInfo
import asyncio
import logging
import httpx
import pandas as pd
from cachetools import TTLCache

logger = logging.getLogger("optionsganster")

_ET = ZoneInfo("America/New_York")

from app.config import settings


class PolygonClient:
    """
    Async Polygon.io / Massive.com Options Data Client
    with TTL caching and automatic pagination.
    """

    BASE_URL = "https://api.polygon.io"

    def __init__(self):
        self._client: Optional[httpx.AsyncClient] = None

        # Caches keyed by request signature
        self._expirations_cache: TTLCache = TTLCache(
            maxsize=128, ttl=settings.CACHE_TTL_EXPIRATIONS
        )
        self._strikes_cache: TTLCache = TTLCache(
            maxsize=256, ttl=settings.CACHE_TTL_STRIKES
        )
        self._price_cache: TTLCache = TTLCache(
            maxsize=128, ttl=settings.CACHE_TTL_PRICES
        )
        self._ohlcv_cache: TTLCache = TTLCache(
            maxsize=64, ttl=settings.CACHE_TTL_OHLCV
        )

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=30.0,
                headers={"Authorization": f"Bearer {settings.POLYGON_API_KEY}"},
            )
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    # ── helpers ──────────────────────────────────────────────

    @staticmethod
    def _format_option_ticker(
        symbol: str, expiration: date, strike: float, right: str
    ) -> str:
        """
        Format option ticker in OCC format.
        Example: O:QQQ250117C00525000
        """
        exp_str = expiration.strftime("%y%m%d")
        strike_str = f"{int(strike * 1000):08d}"
        return f"O:{symbol.upper()}{exp_str}{right.upper()}{strike_str}"

    async def _get_json(
        self, url: str, params: dict | None = None, *, _retries: int = 3
    ) -> dict:
        """
        Make an authenticated GET request and return JSON.
        Retries on 429 with short backoff.
        """
        for attempt in range(_retries):
            resp = await self.client.get(url, params=params)
            if resp.status_code == 429:
                wait = 2 * (attempt + 1)  # 2s, 4s, 6s
                await asyncio.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        # Final attempt – let it raise
        resp = await self.client.get(url, params=params)
        resp.raise_for_status()
        return resp.json()

    async def _get_all_pages(self, url: str, params: dict) -> list[dict]:
        """
        Follow Polygon pagination – keeps fetching `next_url` until exhausted.
        Returns the combined `results` list.
        """
        all_results: list[dict] = []
        while url:
            data = await self._get_json(url, params)
            all_results.extend(data.get("results", []))
            next_url = data.get("next_url")
            if next_url:
                # next_url already includes query params
                url = next_url
                params = {}  # params are baked into next_url
            else:
                break
        return all_results

    # ── OHLCV ────────────────────────────────────────────────

    async def get_option_ohlcv(
        self,
        symbol: str,
        expiration: date,
        strike: float,
        right: str,
        start_date: date,
        end_date: date,
        interval_min: int = 1,
    ) -> pd.DataFrame:
        """Fetch OHLCV data for an option contract (cached)."""
        cache_key = (symbol, expiration, strike, right, start_date, end_date, interval_min)
        if cache_key in self._ohlcv_cache:
            return self._ohlcv_cache[cache_key]

        ticker = self._format_option_ticker(symbol, expiration, strike, right)

        url = (
            f"{self.BASE_URL}/v2/aggs/ticker/{ticker}"
            f"/range/{interval_min}/minute/{start_date}/{end_date}"
        )
        params = {"limit": 50000, "sort": "asc"}

        try:
            data = await self._get_json(url, params)
            bars = data.get("results", [])

            if not bars:
                return pd.DataFrame(
                    columns=["datetime", "open", "high", "low", "close", "volume"]
                )

            records = []
            for bar in bars:
                # Convert UTC millis → Eastern Time (naive) for chart display
                dt_utc = datetime.fromtimestamp(bar["t"] / 1000, tz=ZoneInfo("UTC"))
                dt = dt_utc.astimezone(_ET).replace(tzinfo=None)
                records.append(
                    {
                        "datetime": dt,
                        "open": bar["o"],
                        "high": bar["h"],
                        "low": bar["l"],
                        "close": bar["c"],
                        "volume": bar.get("v", 0),
                    }
                )

            df = pd.DataFrame(records).sort_values("datetime").reset_index(drop=True)
            self._ohlcv_cache[cache_key] = df
            return df

        except httpx.HTTPStatusError as e:
            code = e.response.status_code
            if code in (401, 403):
                raise PermissionError(
                    "Invalid API key or insufficient permissions. "
                    "Options data requires a paid Polygon/Massive subscription."
                )
            raise Exception(f"Polygon API error ({code}): {e.response.text}")

    # ── Underlying (stock) OHLCV ─────────────────────────────

    _stock_ohlcv_cache: dict = {}

    async def get_stock_ohlcv(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        interval_min: int = 5,
    ) -> pd.DataFrame:
        """Fetch OHLCV bars for the underlying stock/ETF (cached)."""
        cache_key = (symbol, start_date, end_date, interval_min)
        if cache_key in self._stock_ohlcv_cache:
            return self._stock_ohlcv_cache[cache_key]

        url = (
            f"{self.BASE_URL}/v2/aggs/ticker/{symbol}"
            f"/range/{interval_min}/minute/{start_date}/{end_date}"
        )
        params = {"limit": 50000, "sort": "asc"}

        try:
            print(f"[STOCK] Fetching {url}")
            data = await self._get_json(url, params)
            bars = data.get("results", [])
            print(f"[STOCK] Got {len(bars)} bars for {symbol}")

            if not bars:
                empty_df = pd.DataFrame(
                    columns=["datetime", "open", "high", "low", "close", "volume"]
                )
                self._stock_ohlcv_cache[cache_key] = empty_df
                return empty_df

            records = []
            for bar in bars:
                # Convert UTC millis → Eastern Time (naive) for chart display
                dt_utc = datetime.fromtimestamp(bar["t"] / 1000, tz=ZoneInfo("UTC"))
                dt = dt_utc.astimezone(_ET).replace(tzinfo=None)
                records.append(
                    {
                        "datetime": dt,
                        "open": bar["o"],
                        "high": bar["h"],
                        "low": bar["l"],
                        "close": bar["c"],
                        "volume": bar.get("v", 0),
                    }
                )

            df = pd.DataFrame(records).sort_values("datetime").reset_index(drop=True)
            self._stock_ohlcv_cache[cache_key] = df
            return df

        except Exception as e:
            print(f"[STOCK] Error fetching {symbol}: {e}")
            return pd.DataFrame(
                columns=["datetime", "open", "high", "low", "close", "volume"]
            )

    def clear_stock_ohlcv_cache(self, **kwargs):
        """Clear the stock OHLCV cache (for live mode)."""
        if kwargs:
            key = (kwargs.get("symbol"), kwargs.get("start_date"),
                   kwargs.get("end_date"), kwargs.get("interval_min"))
            self._stock_ohlcv_cache.pop(key, None)
        else:
            self._stock_ohlcv_cache.clear()

    # ── Stock daily bars (for historical volatility) ─────────

    _stock_daily_cache: dict = {}

    async def get_stock_daily_ohlcv(
        self,
        symbol: str,
        num_days: int = 60,
    ) -> list[float]:
        """
        Fetch daily closing prices for the underlying stock/ETF.
        Returns a list of close prices (oldest first) for HV calculation.
        Uses Polygon aggregates with 1-day interval.
        """
        cache_key = (symbol.upper(), num_days)
        if cache_key in self._stock_daily_cache:
            return self._stock_daily_cache[cache_key]

        end_dt = date.today()
        # Fetch extra calendar days to account for weekends/holidays
        from datetime import timedelta
        start_dt = end_dt - timedelta(days=int(num_days * 1.6))

        url = (
            f"{self.BASE_URL}/v2/aggs/ticker/{symbol.upper()}"
            f"/range/1/day/{start_dt}/{end_dt}"
        )
        params = {"limit": 5000, "sort": "asc"}

        try:
            data = await self._get_json(url, params)
            bars = data.get("results", [])
            closes = [float(bar["c"]) for bar in bars if "c" in bar]
            # Keep only the most recent num_days+1 closes
            closes = closes[-(num_days + 1):]
            self._stock_daily_cache[cache_key] = closes
            return closes
        except Exception as e:
            print(f"[DailyOHLCV] Error fetching daily bars for {symbol}: {e}")
            return []

    # ── Stock daily bars (full OHLCV for S/R engine) ────────

    _stock_daily_bars_cache: dict = {}

    async def get_stock_daily_bars(
        self,
        symbol: str,
        num_days: int = 90,
    ) -> pd.DataFrame:
        """
        Fetch daily OHLCV bars (full candles) for S/R, Fibonacci, and Volume Profile.
        Returns DataFrame with columns: date, open, high, low, close, volume.
        Separate from get_stock_daily_ohlcv() which returns closes only.
        """
        cache_key = (symbol.upper(), num_days)
        if cache_key in self._stock_daily_bars_cache:
            return self._stock_daily_bars_cache[cache_key]

        end_dt = date.today()
        from datetime import timedelta
        start_dt = end_dt - timedelta(days=int(num_days * 1.6))

        url = (
            f"{self.BASE_URL}/v2/aggs/ticker/{symbol.upper()}"
            f"/range/1/day/{start_dt}/{end_dt}"
        )
        params = {"limit": 5000, "sort": "asc"}

        try:
            data = await self._get_json(url, params)
            bars = data.get("results", [])

            if not bars:
                empty = pd.DataFrame(
                    columns=["date", "open", "high", "low", "close", "volume"]
                )
                self._stock_daily_bars_cache[cache_key] = empty
                return empty

            records = []
            for bar in bars:
                dt_utc = datetime.fromtimestamp(bar["t"] / 1000, tz=ZoneInfo("UTC"))
                dt_et = dt_utc.astimezone(_ET).date()
                records.append({
                    "date": dt_et,
                    "open": float(bar["o"]),
                    "high": float(bar["h"]),
                    "low": float(bar["l"]),
                    "close": float(bar["c"]),
                    "volume": int(bar.get("v", 0)),
                })

            df = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
            # Keep only the most recent num_days bars
            df = df.tail(num_days).reset_index(drop=True)
            self._stock_daily_bars_cache[cache_key] = df
            return df

        except Exception as e:
            print(f"[DailyBars] Error fetching daily bars for {symbol}: {e}")
            return pd.DataFrame(
                columns=["date", "open", "high", "low", "close", "volume"]
            )

    # ── Expirations ──────────────────────────────────────────

    async def get_expirations(self, symbol: str) -> list[date]:
        """Get available expiration dates for a symbol (cached).
        
        Filters to future dates only and sorts by expiration_date 
        to minimize pages needed from Polygon.
        """
        cache_key = symbol.upper()
        if cache_key in self._expirations_cache:
            return self._expirations_cache[cache_key]

        today = date.today()
        url = f"{self.BASE_URL}/v3/reference/options/contracts"
        params = {
            "underlying_ticker": cache_key,
            "expiration_date.gte": today.strftime("%Y-%m-%d"),
            "contract_type": "call",   # only calls – halves pages, same unique dates
            "sort": "expiration_date",
            "order": "asc",
            "limit": 1000,
        }

        expirations: set[date] = set()
        # Paginate but stop early once we have 30+ unique dates
        next_url: str | None = url
        next_params: dict | None = params
        while next_url:
            data = await self._get_json(next_url, next_params)
            for contract in data.get("results", []):
                exp_str = contract.get("expiration_date")
                if exp_str:
                    expirations.add(datetime.strptime(exp_str, "%Y-%m-%d").date())
            # If we have enough unique expirations, stop paginating
            if len(expirations) >= 30:
                break
            raw_next = data.get("next_url")
            if raw_next:
                next_url = raw_next
                next_params = None  # params baked into next_url
            else:
                break

        sorted_exps = sorted(expirations)[:30]
        self._expirations_cache[cache_key] = sorted_exps
        return sorted_exps

    # ── Strikes ──────────────────────────────────────────────

    async def get_strikes(self, symbol: str, expiration: date) -> list[float]:
        """Get available strikes for a symbol and expiration (cached, paginated)."""
        cache_key = (symbol.upper(), expiration)
        if cache_key in self._strikes_cache:
            return self._strikes_cache[cache_key]

        url = f"{self.BASE_URL}/v3/reference/options/contracts"
        params = {
            "underlying_ticker": symbol.upper(),
            "expiration_date": expiration.strftime("%Y-%m-%d"),
            "contract_type": "call",   # only calls – halves pages, same unique strikes
            "limit": 1000,
        }

        results = await self._get_all_pages(url, params)

        strikes: set[float] = set()
        for contract in results:
            strike = contract.get("strike_price")
            if strike:
                strikes.add(float(strike))

        sorted_strikes = sorted(strikes)
        self._strikes_cache[cache_key] = sorted_strikes
        return sorted_strikes

    # ── Cache pre-warming ────────────────────────────────────

    async def warm_cache(self, symbols: list[str] | None = None):
        """Pre-warm expirations + first-expiration strikes for given symbols.
        Called at server startup so the first page load is fast.
        """
        symbols = symbols or ["QQQ"]
        for sym in symbols:
            try:
                print(f"[CacheWarm] Pre-warming expirations for {sym}…")
                exps = await self.get_expirations(sym)
                if exps:
                    print(f"[CacheWarm] {sym}: {len(exps)} expirations, warming strikes for {exps[0]}…")
                    await self.get_strikes(sym, exps[0])
                    # Also warm the underlying price
                    await self.get_underlying_price(sym)
                print(f"[CacheWarm] {sym} done.")
            except Exception as e:
                print(f"[CacheWarm] Failed for {sym}: {e}")

    # ── Underlying price ─────────────────────────────────────

    async def get_underlying_price(self, symbol: str) -> float:
        """Get current/last close price of underlying (cached)."""
        cache_key = symbol.upper()
        if cache_key in self._price_cache:
            return self._price_cache[cache_key]

        url = f"{self.BASE_URL}/v2/aggs/ticker/{cache_key}/prev"
        try:
            data = await self._get_json(url)
            results = data.get("results", [])
            price = float(results[0].get("c", 0)) if results else 0.0
            self._price_cache[cache_key] = price
            return price
        except Exception:
            return 0.0

    # ── Grouped daily (watchlist) ────────────────────────────

    async def get_grouped_daily(self, target_date: date) -> dict[str, dict]:
        """
        Fetch grouped daily bars for ALL US stocks (one API call).
        Returns a dict keyed by ticker symbol.
        """
        url = (
            f"{self.BASE_URL}/v2/aggs/grouped/locale/us/market/stocks"
            f"/{target_date.strftime('%Y-%m-%d')}"
        )
        data = await self._get_json(url)
        return {r.get("T", ""): r for r in data.get("results", [])}

    # ── Snapshot prices (real-time/delayed) ──────────────────

    async def get_snapshot_prices(self, symbols: list[str]) -> dict[str, dict]:
        """
        Fetch snapshot prices for multiple symbols (real-time or 15-min delayed).
        Returns near-real-time prices including last trade, day OHLV, prev close.
        Uses a 10-second TTL cache for efficiency.
        """
        # Create cache key from sorted symbols
        cache_key = tuple(sorted(s.upper() for s in symbols))
        
        # Check snapshot cache (separate from price_cache, uses 10s TTL)
        snapshot_cache_ttl = 10
        if not hasattr(self, '_snapshot_cache'):
            self._snapshot_cache = TTLCache(maxsize=64, ttl=snapshot_cache_ttl)
        
        if cache_key in self._snapshot_cache:
            return self._snapshot_cache[cache_key]

        # Build comma-separated ticker list
        ticker_str = ",".join(s.upper() for s in symbols)
        
        url = f"{self.BASE_URL}/v2/snapshot/locale/us/markets/stocks/tickers"
        params = {"tickers": ticker_str}
        
        try:
            data = await self._get_json(url, params)
            tickers = data.get("tickers", [])
            
            # Build result dict keyed by symbol
            result = {}
            for ticker_data in tickers:
                symbol = ticker_data.get("ticker", "")
                if not symbol:
                    continue
                    
                last_trade = ticker_data.get("lastTrade", {})
                day_data = ticker_data.get("day", {})
                prev_day = ticker_data.get("prevDay", {})
                
                result[symbol] = {
                    "lastPrice": last_trade.get("p", 0.0),
                    "todaysChange": ticker_data.get("todaysChange", 0.0),
                    "todaysChangePerc": ticker_data.get("todaysChangePerc", 0.0),
                    "dayOpen": day_data.get("o", 0.0),
                    "dayHigh": day_data.get("h", 0.0),
                    "dayLow": day_data.get("l", 0.0),
                    "dayVolume": day_data.get("v", 0),
                    "prevClose": prev_day.get("c", 0.0),
                }
            
            self._snapshot_cache[cache_key] = result
            return result
            
        except Exception as e:
            # Log error and fallback to empty dict
            print(f"Error fetching snapshot prices for {ticker_str}: {e}")
            return {}

    async def get_prev_close_prices(self, symbols: list[str]) -> dict[str, dict]:
        """
        Fetch previous-day close for multiple symbols using grouped daily bars.
        Uses a SINGLE API call (get_grouped_daily) instead of N individual calls
        to avoid rate-limit (429) errors on basic plans.
        Cached for 60 seconds.
        """
        from datetime import timedelta

        if not hasattr(self, '_prev_grouped_cache'):
            self._prev_grouped_cache = TTLCache(maxsize=1, ttl=60)

        cache_key = "grouped_daily"
        all_bars = self._prev_grouped_cache.get(cache_key)

        if all_bars is None:
            # Try today first, then walk back up to 5 calendar days
            # (covers weekends / holidays)
            today = date.today()
            for offset in range(0, 6):
                target = today - timedelta(days=offset)
                try:
                    all_bars = await self.get_grouped_daily(target)
                    if all_bars:
                        self._prev_grouped_cache[cache_key] = all_bars
                        print(f"[PrevClose] Loaded grouped daily for {target} "
                              f"({len(all_bars)} tickers)")
                        break
                except Exception as e:
                    print(f"[PrevClose] Error fetching grouped daily for {target}: {e}")
            else:
                all_bars = {}

        result = {}
        for sym in symbols:
            s = sym.upper()
            bar = all_bars.get(s, {})
            if bar:
                result[s] = {
                    "lastPrice": float(bar.get("c", 0)),
                    "prevClose": float(bar.get("c", 0)),
                    "dayOpen": float(bar.get("o", 0)),
                    "dayHigh": float(bar.get("h", 0)),
                    "dayLow": float(bar.get("l", 0)),
                    "dayVolume": int(bar.get("v", 0)),
                    "todaysChange": 0.0,
                    "todaysChangePerc": 0.0,
                }

        return result

    # ── Options Snapshot (Greeks, IV, OI) ───────────────────

    async def get_option_contract_snapshot(
        self,
        symbol: str,
        expiration: date,
        strike: float,
        right: str,
    ) -> dict:
        """
        Fetch snapshot for a single option contract.
        Returns Greeks (delta, gamma, theta, vega), IV, open_interest, volume, etc.
        Polygon endpoint: GET /v3/snapshot/options/{underlyingAsset}/{optionContract}
        """
        cache_key = ("snapshot", symbol, expiration, strike, right)
        if not hasattr(self, "_contract_snapshot_cache"):
            self._contract_snapshot_cache = TTLCache(maxsize=256, ttl=15)
        if cache_key in self._contract_snapshot_cache:
            return self._contract_snapshot_cache[cache_key]

        ticker = self._format_option_ticker(symbol, expiration, strike, right)

        url = f"{self.BASE_URL}/v3/snapshot/options/{symbol.upper()}/{ticker}"
        try:
            data = await self._get_json(url)
            result = data.get("results", {})

            # Polygon snapshot underlying_asset only has ticker, not price.
            # Get underlying price from our cache or prev-close API.
            underlying_price = 0.0
            cached_price = self._price_cache.get(symbol.upper())
            if cached_price:
                underlying_price = cached_price
            else:
                try:
                    underlying_price = await self.get_underlying_price(symbol)
                except Exception:
                    pass

            day = result.get("day", {})
            last_quote = result.get("last_quote", {})
            parsed = {
                "greeks": result.get("greeks", {}),
                "iv": result.get("implied_volatility", 0),
                "open_interest": result.get("open_interest", 0),
                "volume": day.get("volume", 0),
                "last_price": day.get("close", 0) or day.get("vwap", 0) or 0,
                "break_even": result.get("break_even_price", 0) or 0,
                "underlying_price": underlying_price,
                "change_to_break_even": 0,
                "bid": last_quote.get("bid", 0) or day.get("close", 0) or 0,
                "ask": last_quote.get("ask", 0) or day.get("close", 0) or 0,
            }
            self._contract_snapshot_cache[cache_key] = parsed
            return parsed
        except Exception as e:
            print(f"Error fetching contract snapshot: {e}")
            return {}

    async def get_options_chain_snapshot(
        self,
        symbol: str,
        expiration: date | None = None,
        page_size: int = 250,
        max_contracts: int = 5000,
    ) -> list[dict]:
        """
        Fetch snapshot for ALL option contracts of an underlying (for a given expiration).
        Paginates through all pages to get the complete chain.
        Used for: IV Rank calculation, GEX, Put/Call ratio, Max Pain, UOA detection.
        Polygon endpoint: GET /v3/snapshot/options/{underlyingAsset}
        """
        cache_key = ("chain", symbol, expiration)
        if not hasattr(self, "_chain_snapshot_cache"):
            self._chain_snapshot_cache = TTLCache(maxsize=32, ttl=30)
        if cache_key in self._chain_snapshot_cache:
            return self._chain_snapshot_cache[cache_key]

        url = f"{self.BASE_URL}/v3/snapshot/options/{symbol.upper()}"
        params: dict = {"limit": page_size, "order": "asc", "sort": "strike_price"}
        if expiration:
            params["expiration_date"] = expiration.strftime("%Y-%m-%d")

        try:
            all_results: list[dict] = []
            current_url = url
            current_params: dict | None = params
            page_count = 0

            while current_url and len(all_results) < max_contracts:
                data = await self._get_json(current_url, current_params)
                results_batch = data.get("results", [])
                if not results_batch:
                    break
                for r in results_batch:
                    all_results.append({
                        "ticker": r.get("details", {}).get("ticker", ""),
                        "strike": r.get("details", {}).get("strike_price", 0),
                        "expiration": r.get("details", {}).get("expiration_date", ""),
                        "contract_type": r.get("details", {}).get("contract_type", ""),
                        "greeks": r.get("greeks", {}),
                        "iv": r.get("implied_volatility", 0),
                        "open_interest": r.get("open_interest", 0),
                        "volume": r.get("day", {}).get("volume", 0),
                        "last_price": r.get("day", {}).get("close", 0),
                    })
                page_count += 1
                next_url = data.get("next_url")
                if next_url:
                    current_url = next_url
                    current_params = None  # next_url includes params
                else:
                    break

            print(f"Chain snapshot: {len(all_results)} contracts in {page_count} pages for {symbol} exp={expiration}")
            self._chain_snapshot_cache[cache_key] = all_results
            return all_results
        except Exception as e:
            print(f"Error fetching chain snapshot: {e}")
            return []

    # ── Cache management ─────────────────────────────────────

    def clear_ohlcv_cache(
        self,
        symbol: str,
        expiration: date,
        strike: float,
        right: str,
        start_date: date,
        end_date: date,
        interval_min: int,
    ) -> None:
        """Clear a specific entry from the OHLCV cache."""
        cache_key = (symbol, expiration, strike, right, start_date, end_date, interval_min)
        self._ohlcv_cache.pop(cache_key, None)


# Singleton – lifecycle managed via FastAPI lifespan
polygon_client = PolygonClient()
