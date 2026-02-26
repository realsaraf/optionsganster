"""
Technical Indicators
=====================
Pure functions operating on OHLCV DataFrames.
All functions expect columns: datetime, open, high, low, close, volume.
No I/O, no API calls — feed them DataFrames.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd


# ── ATR ──────────────────────────────────────────────────────

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range."""
    high = df["high"]
    low = df["low"]
    close_prev = df["close"].shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - close_prev).abs(),
            (low - close_prev).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return tr.ewm(alpha=1 / period, adjust=False).mean()


def atr_value(df: pd.DataFrame, period: int = 14) -> float:
    """Return latest ATR value (scalar), or 0.0 if insufficient data."""
    if len(df) < period + 1:
        return 0.0
    series = atr(df, period)
    val = series.iloc[-1]
    return float(val) if not math.isnan(val) else 0.0


# ── EMA ──────────────────────────────────────────────────────

def ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=span, adjust=False).mean()


def ema_value(series: pd.Series, span: int) -> float:
    """Return latest EMA value, or the last raw value if insufficient."""
    result = ema(series, span)
    return float(result.iloc[-1]) if len(result) > 0 else 0.0


# ── RSI ──────────────────────────────────────────────────────

def rsi(series_or_df: pd.DataFrame | pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index (0-100)."""
    if isinstance(series_or_df, pd.DataFrame):
        closes = series_or_df["close"]
    else:
        closes = series_or_df

    delta = closes.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, 1e-9)
    return 100 - (100 / (1 + rs))


def rsi_value(df: pd.DataFrame, period: int = 14) -> float:
    """Return latest RSI value, or 50.0 if insufficient data."""
    if len(df) < period + 1:
        return 50.0
    series = rsi(df, period)
    val = series.iloc[-1]
    return float(val) if not math.isnan(val) else 50.0


# ── VWAP ─────────────────────────────────────────────────────

def vwap(df: pd.DataFrame) -> pd.Series:
    """
    Volume-Weighted Average Price with intraday day-boundary reset.
    Expects a 'datetime' column (naive, Eastern time).
    """
    if df.empty:
        return pd.Series(dtype=float)

    df = df.copy()
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    tpv = typical_price * df["volume"]

    # Group by calendar date to reset at each session open
    if "datetime" in df.columns:
        df["_date"] = pd.to_datetime(df["datetime"]).dt.date
    else:
        df["_date"] = 0  # single-day fallback

    cum_tpv = tpv.groupby(df["_date"]).cumsum()
    cum_vol = df["volume"].groupby(df["_date"]).cumsum()

    return cum_tpv / cum_vol.replace(0, np.nan)


def vwap_value(df: pd.DataFrame) -> float:
    """Return latest VWAP value."""
    if df.empty:
        return 0.0
    series = vwap(df)
    val = series.iloc[-1]
    return float(val) if not math.isnan(val) else 0.0


def vwap_bands(
    df: pd.DataFrame, num_stdev: float = 1.0
) -> tuple[pd.Series, pd.Series]:
    """
    VWAP ± N standard deviations.
    Returns (upper_band, lower_band).
    """
    if df.empty:
        empty = pd.Series(dtype=float)
        return empty, empty

    df = df.copy()
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    tpv = typical_price * df["volume"]
    tp2v = (typical_price ** 2) * df["volume"]

    if "datetime" in df.columns:
        df["_date"] = pd.to_datetime(df["datetime"]).dt.date
    else:
        df["_date"] = 0

    cum_tpv = tpv.groupby(df["_date"]).cumsum()
    cum_vol = df["volume"].groupby(df["_date"]).cumsum()
    cum_tp2v = tp2v.groupby(df["_date"]).cumsum()

    vwap_series = cum_tpv / cum_vol.replace(0, np.nan)
    variance = (cum_tp2v / cum_vol.replace(0, np.nan)) - vwap_series ** 2
    stdev = variance.clip(lower=0).apply(np.sqrt)

    upper = vwap_series + num_stdev * stdev
    lower = vwap_series - num_stdev * stdev
    return upper, lower


def vwap_band_values(df: pd.DataFrame, num_stdev: float = 1.0) -> tuple[float, float]:
    """Return latest (upper, lower) VWAP band scalar values."""
    upper, lower = vwap_bands(df, num_stdev)
    if upper.empty:
        return 0.0, 0.0
    u = float(upper.iloc[-1]) if not math.isnan(upper.iloc[-1]) else 0.0
    l = float(lower.iloc[-1]) if not math.isnan(lower.iloc[-1]) else 0.0
    return u, l


# ── Rolling Realized Volatility ──────────────────────────────

def rolling_rv(df: pd.DataFrame, window: int = 30) -> pd.Series:
    """
    Intraday rolling realized volatility (annualized).
    Uses log returns of close prices.
    """
    log_returns = np.log(df["close"] / df["close"].shift(1))
    rv = log_returns.rolling(window).std()
    # Annualize: assume 252 trading days × 390 one-minute bars
    bars_per_day = 390
    return rv * math.sqrt(bars_per_day * 252)


def rolling_rv_value(df: pd.DataFrame, window: int = 30) -> float:
    """Return latest rolling RV value (annualized), or 0.0."""
    if len(df) < window + 1:
        return 0.0
    series = rolling_rv(df, window)
    val = series.iloc[-1]
    return float(val) if not math.isnan(val) else 0.0


# ── Opening Range ─────────────────────────────────────────────

def opening_range(
    df: pd.DataFrame, minutes: int = 15
) -> tuple[float, float]:
    """
    Compute the Opening Range High / Low from the first N minutes of the session.
    Expects 'datetime' column. Returns (orh, orl) — (0.0, 0.0) if no data.
    """
    if df.empty or "datetime" not in df.columns:
        return 0.0, 0.0

    df = df.copy()
    df["_dt"] = pd.to_datetime(df["datetime"])

    # First bar date
    first_date = df["_dt"].dt.date.iloc[0]
    today_bars = df[df["_dt"].dt.date == first_date].sort_values("_dt")

    if today_bars.empty:
        return 0.0, 0.0

    session_open = today_bars["_dt"].iloc[0]
    cutoff = session_open + pd.Timedelta(minutes=minutes)
    or_bars = today_bars[today_bars["_dt"] < cutoff]

    if or_bars.empty:
        return 0.0, 0.0

    return float(or_bars["high"].max()), float(or_bars["low"].min())


def opening_range_today(
    df: pd.DataFrame, minutes: int = 15
) -> tuple[float, float]:
    """
    Opening range for TODAY specifically (last unique date in df).
    Returns (orh, orl).
    """
    if df.empty or "datetime" not in df.columns:
        return 0.0, 0.0

    df = df.copy()
    df["_dt"] = pd.to_datetime(df["datetime"])
    today = df["_dt"].dt.date.max()
    today_bars = df[df["_dt"].dt.date == today].sort_values("_dt")

    if today_bars.empty:
        return 0.0, 0.0

    session_open = today_bars["_dt"].iloc[0]
    cutoff = session_open + pd.Timedelta(minutes=minutes)
    or_bars = today_bars[today_bars["_dt"] < cutoff]

    if or_bars.empty:
        return 0.0, 0.0

    return float(or_bars["high"].max()), float(or_bars["low"].min())


# ── Implied Move ─────────────────────────────────────────────

def implied_move(spot: float, iv: float, dte_years: float) -> float:
    """
    Expected 1-sigma move in dollar terms.
    Formula: spot × iv × sqrt(dte_years)
    """
    if spot <= 0 or iv <= 0 or dte_years < 0:
        return 0.0
    return spot * iv * math.sqrt(dte_years)


# ── Slope ────────────────────────────────────────────────────

def slope(series: pd.Series, window: int = 15) -> float:
    """
    Linear regression slope of the last `window` values.
    Returns slope per bar (change per bar, not annualized).
    Returns 0.0 if insufficient data.
    """
    if len(series) < window:
        return 0.0

    y = series.iloc[-window:].values.astype(float)
    x = np.arange(window, dtype=float)

    # Remove NaN
    mask = ~np.isnan(y)
    if mask.sum() < 3:
        return 0.0

    x_clean = x[mask]
    y_clean = y[mask]

    # Linear regression via normal equations
    n = len(x_clean)
    sx = x_clean.sum()
    sy = y_clean.sum()
    sxx = (x_clean ** 2).sum()
    sxy = (x_clean * y_clean).sum()
    denom = n * sxx - sx ** 2
    if abs(denom) < 1e-9:
        return 0.0

    return float((n * sxy - sx * sy) / denom)


def vwap_slope(df: pd.DataFrame, window: int = 15) -> float:
    """Slope of VWAP over last `window` bars."""
    v = vwap(df)
    return slope(v, window)


# ── Higher Highs / Higher Lows ────────────────────────────────

def higher_highs_lows(
    df: pd.DataFrame, window: int = 15
) -> tuple[int, int]:
    """
    Count consecutive Higher Highs and Higher Lows in the last `window` bars.
    Returns (hh_count, hl_count).
    """
    if len(df) < 2:
        return 0, 0

    recent = df.tail(window)
    highs = recent["high"].values
    lows = recent["low"].values

    hh_count = 0
    hl_count = 0

    for i in range(1, len(highs)):
        if highs[i] > highs[i - 1]:
            hh_count += 1
        if lows[i] > lows[i - 1]:
            hl_count += 1

    return hh_count, hl_count


def lower_lows_highs(
    df: pd.DataFrame, window: int = 15
) -> tuple[int, int]:
    """Count consecutive Lower Lows and Lower Highs."""
    if len(df) < 2:
        return 0, 0

    recent = df.tail(window)
    highs = recent["high"].values
    lows = recent["low"].values

    ll_count = 0
    lh_count = 0

    for i in range(1, len(highs)):
        if lows[i] < lows[i - 1]:
            ll_count += 1
        if highs[i] < highs[i - 1]:
            lh_count += 1

    return ll_count, lh_count


# ── Volume Spike Detection ────────────────────────────────────

def volume_ratio(df: pd.DataFrame, period: int = 20) -> float:
    """
    Latest bar's volume relative to rolling average.
    Returns ratio (>1.5 = elevated, >2.5 = spike).
    """
    if len(df) < 2:
        return 1.0
    avg = df["volume"].iloc[-period - 1 : -1].mean()
    if avg <= 0:
        return 1.0
    return float(df["volume"].iloc[-1]) / avg


def is_volume_spike(df: pd.DataFrame, threshold: float = 2.0, period: int = 20) -> bool:
    """True if latest bar volume is >= threshold × rolling average."""
    return volume_ratio(df, period) >= threshold


# ── Candle Pattern Helpers ────────────────────────────────────

def body_size(bar: pd.Series) -> float:
    """Absolute candle body size."""
    return abs(bar["close"] - bar["open"])


def upper_wick(bar: pd.Series) -> float:
    """Upper wick size."""
    return bar["high"] - max(bar["open"], bar["close"])


def lower_wick(bar: pd.Series) -> float:
    """Lower wick size."""
    return min(bar["open"], bar["close"]) - bar["low"]


def is_bullish_engulf(df: pd.DataFrame) -> bool:
    """Two-bar bullish engulfing pattern (last two bars)."""
    if len(df) < 2:
        return False
    prev, curr = df.iloc[-2], df.iloc[-1]
    return (
        prev["close"] < prev["open"]  # prev bearish
        and curr["close"] > curr["open"]  # curr bullish
        and curr["open"] < prev["close"]
        and curr["close"] > prev["open"]
    )


def is_bearish_engulf(df: pd.DataFrame) -> bool:
    """Two-bar bearish engulfing pattern."""
    if len(df) < 2:
        return False
    prev, curr = df.iloc[-2], df.iloc[-1]
    return (
        prev["close"] > prev["open"]  # prev bullish
        and curr["close"] < curr["open"]  # curr bearish
        and curr["open"] > prev["close"]
        and curr["close"] < prev["open"]
    )


def is_rejection_wick_up(bar: pd.Series, min_wick_ratio: float = 2.0) -> bool:
    """Upper wick is min_wick_ratio × body — bearish rejection."""
    b = body_size(bar)
    if b < 1e-6:
        return False
    return upper_wick(bar) >= min_wick_ratio * b


def is_rejection_wick_down(bar: pd.Series, min_wick_ratio: float = 2.0) -> bool:
    """Lower wick is min_wick_ratio × body — bullish rejection."""
    b = body_size(bar)
    if b < 1e-6:
        return False
    return lower_wick(bar) >= min_wick_ratio * b
