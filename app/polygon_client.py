"""
Polygon/Massive API Client - Fetches historical options OHLCV data
Fully async with TTL caching and pagination support.
Docs: https://polygon.io/docs/options
"""
from datetime import datetime, date
from typing import Optional
import asyncio
import httpx
import pandas as pd
from cachetools import TTLCache

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
                dt = datetime.fromtimestamp(bar["t"] / 1000)
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
            parsed = {
                "greeks": result.get("greeks", {}),
                "iv": result.get("implied_volatility", 0),
                "open_interest": result.get("open_interest", 0),
                "volume": day.get("volume", 0),
                "last_price": day.get("close", 0) or day.get("vwap", 0) or 0,
                "break_even": result.get("break_even_price", 0) or 0,
                "underlying_price": underlying_price,
                "change_to_break_even": 0,
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
