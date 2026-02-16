"""
Fair Value Engine – Black-Scholes Theoretical Pricing
=====================================================
Calculates theoretical option fair value using the Black-Scholes model
and compares it to the market price (bid/ask midpoint) for cheapness detection.

Inputs sourced from Polygon.io snapshot + historical stock bars.
"""
from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional
import math

# Try scipy first (precise), fall back to a polynomial approximation
try:
    from scipy.stats import norm as _norm
    _ncdf = _norm.cdf
except ImportError:
    def _ncdf(x: float) -> float:
        """Abramowitz & Stegun approximation (26.2.17) of the cumulative normal CDF.
        Accurate to ~1e-7."""
        if x > 6.0:
            return 1.0
        if x < -6.0:
            return 0.0
        b1 = 0.319381530
        b2 = -0.356563782
        b3 = 1.781477937
        b4 = -1.821255978
        b5 = 1.330274429
        p = 0.2316419
        x_abs = abs(x)
        t = 1.0 / (1.0 + p * x_abs)
        n = (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-x_abs * x_abs / 2.0)
        approx = n * (b1 * t + b2 * t**2 + b3 * t**3 + b4 * t**4 + b5 * t**5)
        if x >= 0:
            return 1.0 - approx
        else:
            return approx


@dataclass
class FairValueResult:
    """Result of Black-Scholes fair value calculation."""
    # Inputs
    underlying_price: float       # S
    strike: float                # K
    time_to_expiry: float        # T (years)
    risk_free_rate: float        # r
    historical_vol: float        # σ (realized HV, annualized)
    dividend_yield: float        # q
    contract_type: str           # "C" or "P"

    # Black-Scholes outputs
    theoretical_price: float     # BS model price
    d1: float
    d2: float

    # Market comparison
    market_price: float          # bid/ask midpoint
    market_iv: float             # implied volatility from Polygon
    price_difference: float      # theoretical - market
    pct_difference: float        # (theoretical - market) / market * 100
    is_cheap: bool               # theoretical > market by threshold
    is_expensive: bool           # market > theoretical by threshold
    verdict: str                 # "CHEAP", "EXPENSIVE", "FAIR"
    detail: str                  # Human-readable explanation

    # Confidence / quality
    bid: float
    ask: float
    spread: float                # ask - bid
    spread_pct: float            # spread / midpoint * 100
    hv_vs_iv: float              # HV - IV difference (positive = vol underpriced)


class FairValueEngine:
    """
    Computes Black-Scholes theoretical price and compares to market.
    Uses realized historical volatility (not market IV) as the vol input
    to detect absolute mispricing.
    """

    # Default risk-free rate (annualized, continuously compounded)
    DEFAULT_RISK_FREE_RATE = 0.045   # ~4.5% as of 2026
    # Cheapness threshold: theoretical must exceed market by this % to flag
    CHEAP_THRESHOLD_PCT = 5.0
    EXPENSIVE_THRESHOLD_PCT = 5.0

    def compute_historical_vol(
        self,
        daily_closes: list[float],
        window: int = 30,
    ) -> float:
        """
        Compute annualized historical volatility from daily closing prices.

        Parameters
        ----------
        daily_closes : list of daily close prices (most recent last)
        window : number of trading days to use (default 30)

        Returns
        -------
        Annualized volatility as a decimal (e.g., 0.25 for 25%)
        """
        if len(daily_closes) < 2:
            return 0.0

        # Use up to `window` most recent closes
        closes = daily_closes[-(window + 1):]  # need N+1 prices for N returns
        if len(closes) < 2:
            return 0.0

        # Log returns
        log_returns = []
        for i in range(1, len(closes)):
            if closes[i - 1] <= 0 or closes[i] <= 0:
                continue
            log_returns.append(math.log(closes[i] / closes[i - 1]))

        if len(log_returns) < 2:
            return 0.0

        # Standard deviation of log returns
        n = len(log_returns)
        mean = sum(log_returns) / n
        variance = sum((r - mean) ** 2 for r in log_returns) / (n - 1)
        daily_vol = math.sqrt(variance)

        # Annualize: multiply by sqrt(252) trading days
        return daily_vol * math.sqrt(252)

    def black_scholes(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0,
        contract_type: str = "C",
    ) -> tuple[float, float, float]:
        """
        Compute Black-Scholes theoretical price.

        Parameters
        ----------
        S : underlying price
        K : strike price
        T : time to expiry in years
        r : risk-free rate (decimal)
        sigma : volatility (decimal)
        q : dividend yield (decimal)
        contract_type : "C" for call, "P" for put

        Returns
        -------
        (theoretical_price, d1, d2)
        """
        # Edge cases
        if T <= 0:
            if contract_type == "C":
                return (max(S - K, 0.0), 0.0, 0.0)
            else:
                return (max(K - S, 0.0), 0.0, 0.0)

        if sigma <= 0.001:
            # Zero vol: intrinsic value discounted
            if contract_type == "C":
                val = max(S * math.exp(-q * T) - K * math.exp(-r * T), 0.0)
            else:
                val = max(K * math.exp(-r * T) - S * math.exp(-q * T), 0.0)
            return (val, 0.0, 0.0)

        if S <= 0 or K <= 0:
            return (0.0, 0.0, 0.0)

        sqrt_T = math.sqrt(T)
        sigma_sqrt_T = sigma * sqrt_T

        d1 = (math.log(S / K) + (r - q + sigma * sigma / 2.0) * T) / sigma_sqrt_T
        d2 = d1 - sigma_sqrt_T

        if contract_type == "C":
            price = (
                S * math.exp(-q * T) * _ncdf(d1)
                - K * math.exp(-r * T) * _ncdf(d2)
            )
        else:
            price = (
                K * math.exp(-r * T) * _ncdf(-d2)
                - S * math.exp(-q * T) * _ncdf(-d1)
            )

        return (max(price, 0.0), d1, d2)

    def analyze(
        self,
        underlying_price: float,
        strike: float,
        expiration: date,
        contract_type: str,
        daily_closes: list[float],
        market_bid: float = 0.0,
        market_ask: float = 0.0,
        market_iv: float = 0.0,
        risk_free_rate: Optional[float] = None,
        dividend_yield: float = 0.0,
        hv_window: int = 30,
    ) -> FairValueResult:
        """
        Full fair value analysis for an option contract.

        Parameters
        ----------
        underlying_price : current underlying stock/ETF price
        strike : option strike
        expiration : option expiry date
        contract_type : "C" or "P"
        daily_closes : list of daily closing prices for HV calc
        market_bid : option bid price
        market_ask : option ask price
        market_iv : implied vol from market (decimal)
        risk_free_rate : override default rate
        dividend_yield : annualized dividend yield
        hv_window : window for historical vol (trading days)
        """
        r = risk_free_rate if risk_free_rate is not None else self.DEFAULT_RISK_FREE_RATE
        today = date.today()

        # Time to expiry in years
        days_to_expiry = (expiration - today).days
        T = max(days_to_expiry, 0) / 365.25

        # Historical volatility
        hv = self.compute_historical_vol(daily_closes, window=hv_window)
        # If HV calc fails, fall back to market IV + 5% (conservative)
        if hv < 0.01 and market_iv > 0:
            hv = market_iv * 1.05

        # Black-Scholes price
        theo_price, d1, d2 = self.black_scholes(
            S=underlying_price,
            K=strike,
            T=T,
            r=r,
            sigma=hv,
            q=dividend_yield,
            contract_type=contract_type,
        )

        # Market midpoint
        if market_bid > 0 and market_ask > 0:
            market_price = (market_bid + market_ask) / 2.0
        elif market_ask > 0:
            market_price = market_ask
        elif market_bid > 0:
            market_price = market_bid
        else:
            market_price = 0.0

        spread = market_ask - market_bid if market_ask > 0 and market_bid > 0 else 0.0
        spread_pct = (spread / market_price * 100) if market_price > 0 else 0.0

        # Difference
        price_diff = theo_price - market_price
        pct_diff = (price_diff / market_price * 100) if market_price > 0 else 0.0

        # Cheapness flags
        is_cheap = pct_diff > self.CHEAP_THRESHOLD_PCT
        is_expensive = pct_diff < -self.EXPENSIVE_THRESHOLD_PCT

        # HV vs IV comparison
        hv_vs_iv = hv - market_iv if market_iv > 0 else 0.0

        # Verdict
        if market_price <= 0:
            verdict = "NO MARKET"
            detail = "No market price available for comparison."
        elif is_cheap:
            verdict = "CHEAP"
            detail = (
                f"Theoretical (${theo_price:.2f}) exceeds market (${market_price:.2f}) "
                f"by {pct_diff:+.1f}%. Option appears undervalued using {hv*100:.1f}% HV."
            )
        elif is_expensive:
            verdict = "EXPENSIVE"
            detail = (
                f"Market (${market_price:.2f}) exceeds theoretical (${theo_price:.2f}) "
                f"by {abs(pct_diff):.1f}%. Option appears overvalued using {hv*100:.1f}% HV."
            )
        else:
            verdict = "FAIR"
            detail = (
                f"Theoretical (${theo_price:.2f}) is within {abs(pct_diff):.1f}% of "
                f"market (${market_price:.2f}). Fairly priced."
            )

        # Add IV comparison note
        if market_iv > 0:
            if hv_vs_iv > 0.03:
                detail += f" HV ({hv*100:.1f}%) > IV ({market_iv*100:.1f}%) — vol may be underpriced."
            elif hv_vs_iv < -0.03:
                detail += f" IV ({market_iv*100:.1f}%) > HV ({hv*100:.1f}%) — vol may be overpriced."

        return FairValueResult(
            underlying_price=round(underlying_price, 2),
            strike=round(strike, 2),
            time_to_expiry=round(T, 6),
            risk_free_rate=round(r, 4),
            historical_vol=round(hv, 4),
            dividend_yield=round(dividend_yield, 4),
            contract_type=contract_type,
            theoretical_price=round(theo_price, 4),
            d1=round(d1, 4),
            d2=round(d2, 4),
            market_price=round(market_price, 4),
            market_iv=round(market_iv, 4),
            price_difference=round(price_diff, 4),
            pct_difference=round(pct_diff, 2),
            is_cheap=is_cheap,
            is_expensive=is_expensive,
            verdict=verdict,
            detail=detail,
            bid=round(market_bid, 4),
            ask=round(market_ask, 4),
            spread=round(spread, 4),
            spread_pct=round(spread_pct, 2),
            hv_vs_iv=round(hv_vs_iv, 4),
        )


# Singleton
fair_value_engine = FairValueEngine()
