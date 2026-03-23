import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from threading import Lock


SPOT_TICKERS = {
    "gold": "GC=F",
    "silver": "SI=F",
}

FX_TICKER = "GBPUSD=X"
FX_INR_TICKER = "USDINR=X"
TROY_OZ_PER_KG = 32.1507

DEFAULT_ETC_TICKERS = {
    "SGLN.L": "iShares Physical Gold",
    "SSLN.L": "iShares Physical Silver",
    "SGLS.L": "iShares Physical Gold (GBP Hedged)",
    "SLVP.L": "iShares Physical Silver (GBP Hedged)",
}

# Shared history cache for both dashboard and simulator
_HISTORY_CACHE: dict[tuple[str, str, str, str | None], pd.DataFrame] = {}
_HISTORY_CACHE_LOCK = Lock()


def _cache_bucket(ttl_seconds: int | None) -> str | None:
    if not ttl_seconds or ttl_seconds <= 0:
        return None
    return str(int(datetime.utcnow().timestamp() // ttl_seconds))


def _download_history(ticker: str, start_key: str, interval: str, end_key: str | None = None) -> pd.DataFrame:
    """Download history from Yahoo Finance."""
    kwargs = {
        "start": start_key,
        "interval": interval,
        "progress": False,
    }
    if end_key:
        kwargs["end"] = end_key
    df = yf.download(ticker, **kwargs)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index)
    return df


def fetch_history_cached(ticker: str, start: datetime, interval: str, ttl_seconds: int | None = None) -> pd.DataFrame:
    """Fetch history with caching. Used by both dashboard and simulator. Non-blocking read."""
    start_key = start.date().isoformat()
    key = (ticker, start_key, interval, _cache_bucket(ttl_seconds))
    
    # Fast path: check cache without blocking
    with _HISTORY_CACHE_LOCK:
        cached = _HISTORY_CACHE.get(key)
        if cached is not None and not cached.empty:
            return cached.copy()
    
    # Download outside lock (this is the slow part)
    df = _download_history(ticker, start_key, interval)
    
    # Store result
    with _HISTORY_CACHE_LOCK:
        _HISTORY_CACHE[key] = df.copy()
    
    return df


def clear_history_cache():
    """Clear the shared history cache."""
    with _HISTORY_CACHE_LOCK:
        _HISTORY_CACHE.clear()


def _safe_float(v, default=np.nan) -> float:
    try:
        if v is None:
            return default
        f = float(v)
        return f
    except Exception:
        return default


def _compute_change_pct(ticker: str, info, last_price: float) -> float:
    prev_close = _safe_float(info.get("regularMarketPreviousClose", np.nan), np.nan)
    if np.isnan(prev_close):
        prev_close = _safe_float(info.get("previousClose", np.nan), np.nan)
    if not np.isnan(prev_close) and prev_close != 0 and not np.isnan(last_price):
        return (last_price - prev_close) / prev_close * 100

    change_pct = _safe_float(info.get("regularMarketChangePercent", np.nan), np.nan)
    if not np.isnan(change_pct):
        # Some providers return decimal form (0.0123) instead of percent (1.23)
        return change_pct * 100 if abs(change_pct) <= 1 else change_pct

    try:
        h = yf.Ticker(ticker).history(period="7d", interval="1d")
        if len(h) >= 2:
            prev = float(h["Close"].iloc[-2])
            curr = float(h["Close"].iloc[-1])
            if prev != 0:
                return (curr - prev) / prev * 100
    except Exception:
        pass
    return 0.0


def fetch_spot_prices():
    """Fetch current spot prices for gold and silver, plus GBP/USD rate."""
    result = {}
    for metal, ticker in SPOT_TICKERS.items():
        try:
            info = yf.Ticker(ticker).fast_info
            last_price = _safe_float(info.get("lastPrice", np.nan), np.nan)
            result[metal] = {
                "price_usd": last_price,
                "change_pct": _compute_change_pct(ticker, info, last_price),
            }
        except Exception:
            result[metal] = {"price_usd": np.nan, "change_pct": 0.0}

    try:
        gbp_usd = yf.Ticker(FX_TICKER).fast_info["lastPrice"]
    except Exception:
        gbp_usd = 1.35

    try:
        usd_inr = yf.Ticker(FX_INR_TICKER).fast_info["lastPrice"]
    except Exception:
        usd_inr = 83.0

    result["gbp_usd"] = gbp_usd
    result["usd_inr"] = usd_inr
    for metal in ("gold", "silver"):
        usd = result[metal]["price_usd"]
        result[metal]["price_gbp"] = usd / gbp_usd if not np.isnan(usd) else np.nan
        result[metal]["price_inr_per_kg"] = usd * TROY_OZ_PER_KG * usd_inr if not np.isnan(usd) else np.nan

    if not np.isnan(result["gold"]["price_usd"]) and not np.isnan(result["silver"]["price_usd"]):
        result["ratio"] = result["gold"]["price_usd"] / result["silver"]["price_usd"]
    else:
        result["ratio"] = np.nan

    return result


def fetch_etc_prices(tickers: list[str]):
    """Fetch current prices for user-selected ETC tickers."""
    result = {}
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).fast_info
            last_price = _safe_float(info.get("lastPrice", np.nan), np.nan)
            result[ticker] = {
                "price": last_price,
                "currency": info.get("currency", "GBp"),
                "change_pct": _compute_change_pct(ticker, info, last_price),
            }
        except Exception:
            result[ticker] = {"price": np.nan, "currency": "GBp", "change_pct": 0.0}
    return result


def fetch_historical(ticker: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    """Fetch OHLCV historical data for a given ticker. Uses shared cache."""
    start = datetime.utcnow() - timedelta(days={
        "2y": 730, "3mo": 90, "1mo": 30, "1w": 7
    }.get(period, 730))
    
    # Check cache first
    start_key = start.date().isoformat()
    key = (ticker, start_key, interval)
    
    with _HISTORY_CACHE_LOCK:
        cached = _HISTORY_CACHE.get(key)
        if cached is not None and not cached.empty:
            return cached.copy()
    
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index = pd.to_datetime(df.index)
        
        with _HISTORY_CACHE_LOCK:
            _HISTORY_CACHE[key] = df.copy()
        
        return df
    except Exception:
        return pd.DataFrame()


def fetch_multi_timeframe_data(ticker: str):
    """Fetch historical data across 4 timeframes for Fibonacci + technicals. Uses shared cache."""
    now = datetime.utcnow()
    intraday_start = now - timedelta(days=59)
    short_start = now - timedelta(days=90)
    med_start = now - timedelta(days=730)
    long_start = now - timedelta(days=3650)
    
    return {
        "intraday_term": fetch_history_cached(ticker, intraday_start, "5m", ttl_seconds=300),
        "short_term": fetch_history_cached(ticker, short_start, "1h", ttl_seconds=300),
        "medium_term": fetch_history_cached(ticker, med_start, "1d", ttl_seconds=300),
        "long_term": fetch_history_cached(ticker, long_start, "1mo", ttl_seconds=300),
    }


def compute_tracking_difference(spot_series: pd.Series, etc_series: pd.Series) -> pd.Series:
    """Compute percentage tracking difference between spot and ETC over aligned dates."""
    aligned = pd.concat([spot_series, etc_series], axis=1, join="inner")
    aligned.columns = ["spot", "etc"]
    if aligned.empty:
        return pd.Series(dtype=float)
    spot_norm = aligned["spot"] / aligned["spot"].iloc[0]
    etc_norm = aligned["etc"] / aligned["etc"].iloc[0]
    return (etc_norm - spot_norm) / spot_norm * 100
