import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


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


def _safe_float(v, default=np.nan) -> float:
    try:
        if v is None:
            return default
        f = float(v)
        return f
    except Exception:
        return default


def _compute_change_pct(ticker: str, info, last_price: float) -> float:
    change_pct = _safe_float(info.get("regularMarketChangePercent", np.nan), np.nan)
    if not np.isnan(change_pct):
        return change_pct

    prev_close = _safe_float(info.get("regularMarketPreviousClose", np.nan), np.nan)
    if np.isnan(prev_close):
        prev_close = _safe_float(info.get("previousClose", np.nan), np.nan)
    if not np.isnan(prev_close) and prev_close != 0 and not np.isnan(last_price):
        return (last_price - prev_close) / prev_close * 100

    try:
        h = yf.Ticker(ticker).history(period="2d", interval="1d")
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
    """Fetch OHLCV historical data for a given ticker."""
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index = pd.to_datetime(df.index)
        return df
    except Exception:
        return pd.DataFrame()


def fetch_multi_timeframe_data(ticker: str):
    """Fetch historical data at three timeframes for Fibonacci + technicals."""
    long_term = fetch_historical(ticker, period="2y", interval="1d")
    medium_term = fetch_historical(ticker, period="3mo", interval="1d")
    short_term = fetch_historical(ticker, period="1mo", interval="1h")
    return {
        "long_term": long_term,
        "medium_term": medium_term,
        "short_term": short_term,
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
