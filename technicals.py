import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
from scipy.stats import linregress


# ---------------------------------------------------------------------------
# R2a: Multi-Timeframe Fibonacci Levels
# ---------------------------------------------------------------------------

FIBONACCI_RATIOS = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]


def fibonacci_levels(df: pd.DataFrame) -> dict:
    """Calculate Fibonacci retracement levels from high/low of the given dataframe."""
    if df.empty or "High" not in df.columns or "Low" not in df.columns:
        return {}
    high = df["High"].max()
    low = df["Low"].min()
    diff = high - low
    levels = {}
    for ratio in FIBONACCI_RATIOS:
        levels[f"{ratio*100:.1f}%"] = high - diff * ratio
    levels["swing_high"] = high
    levels["swing_low"] = low
    return levels


def multi_timeframe_fibonacci(data_dict: dict) -> dict:
    """Compute Fibonacci levels for long-term, medium-term, short-term data."""
    result = {}
    for tf_name, df in data_dict.items():
        result[tf_name] = fibonacci_levels(df)
    return result


# ---------------------------------------------------------------------------
# R2b: VOBHS -- Volatility Oscillator Black Hole Sign
# ---------------------------------------------------------------------------

def volatility_oscillator(df: pd.DataFrame, length: int = 100) -> pd.DataFrame:
    """Volatility Oscillator: spike = close - open, with upper/lower std bands."""
    spike = df["Close"] - df["Open"]
    upper = spike.rolling(length).std()
    lower = -upper
    vo = pd.DataFrame({"spike": spike, "upper": upper, "lower": lower}, index=df.index)
    vo["signal"] = np.where(spike > upper, 1, np.where(spike < lower, -1, 0))
    return vo


def _wma(series: pd.Series, length: int) -> pd.Series:
    """Weighted Moving Average."""
    weights = np.arange(1, length + 1, dtype=float)
    return series.rolling(length).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)


def hull_moving_average(series: pd.Series, length: int = 9) -> pd.Series:
    """Hull Moving Average: WMA(2*WMA(n/2) - WMA(n), sqrt(n))."""
    half_len = max(int(length / 2), 1)
    sqrt_len = max(int(np.sqrt(length)), 1)
    wma_half = _wma(series, half_len)
    wma_full = _wma(series, length)
    hull_raw = 2 * wma_half - wma_full
    return _wma(hull_raw, sqrt_len)


def _highpass_filter(series: pd.Series, period: int = 48) -> pd.Series:
    """Ehlers 2-pole high-pass filter."""
    alpha = (0.707 * 2 * np.pi) / period
    a1 = np.exp(-alpha)
    b1 = 2 * a1 * np.cos(alpha)
    c2 = b1
    c3 = -(a1 * a1)
    c1 = (1 + c2 - c3) / 4.0
    vals = series.values.astype(float)
    hp = np.zeros_like(vals)
    for i in range(2, len(vals)):
        if np.isnan(vals[i]) or np.isnan(vals[i - 1]) or np.isnan(vals[i - 2]):
            hp[i] = 0.0
        else:
            hp[i] = c1 * (vals[i] - 2 * vals[i - 1] + vals[i - 2]) + c2 * hp[i - 1] + c3 * hp[i - 2]
    return pd.Series(hp, index=series.index)


def _super_smoother(series: pd.Series, period: int = 10) -> pd.Series:
    """Ehlers 2-pole Super Smoother filter."""
    alpha = (1.414 * np.pi) / period
    a1 = np.exp(-alpha)
    b1 = 2 * a1 * np.cos(alpha)
    c2 = b1
    c3 = -(a1 * a1)
    c1 = 1 - c2 - c3
    vals = series.values.astype(float)
    ss = np.zeros_like(vals)
    for i in range(2, len(vals)):
        if np.isnan(vals[i]) or np.isnan(vals[i - 1]):
            ss[i] = 0.0
        else:
            ss[i] = c1 * (vals[i] + vals[i - 1]) / 2.0 + c2 * ss[i - 1] + c3 * ss[i - 2]
    return pd.Series(ss, index=series.index)


def boom_hunter_pro(df: pd.DataFrame, lp_period: int = 6, hp_period: int = 48,
                    sq_period: int = 5) -> pd.DataFrame:
    """
    Boom Hunter Pro (BHS) -- simplified from Veryfid's PineScript.
    Uses Ehlers' filters on close to produce trigger & quotient lines.
    """
    close = df["Close"]
    hp = _highpass_filter(close, hp_period)
    filt = _super_smoother(hp, lp_period)

    peak = filt.rolling(sq_period).max()
    valley = filt.rolling(sq_period).min()
    mid_range = (peak + valley) / 2.0

    trigger = _super_smoother(filt, lp_period)
    quotient = mid_range

    boom = pd.DataFrame({"trigger": trigger, "quotient": quotient}, index=df.index)
    boom["signal"] = np.where(
        (trigger > quotient) & (trigger.shift(1) <= quotient.shift(1)), 1,
        np.where(
            (trigger < quotient) & (trigger.shift(1) >= quotient.shift(1)), -1, 0
        )
    )
    return boom


def modified_atr(df: pd.DataFrame, period: int = 14, multiplier: float = 1.5) -> pd.DataFrame:
    """Modified ATR for dynamic stop-loss distance."""
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    return pd.DataFrame({
        "atr": atr,
        "stop_long": close - atr * multiplier,
        "stop_short": close + atr * multiplier,
    }, index=df.index)


def vobhs_composite(df: pd.DataFrame) -> dict:
    """Run all VOBHS components and return results dict."""
    vo = volatility_oscillator(df)
    hma = hull_moving_average(df["Close"], length=9)
    boom = boom_hunter_pro(df)
    atr = modified_atr(df)

    hma_signal = np.where(df["Close"] > hma, 1, np.where(df["Close"] < hma, -1, 0))
    hma_signal = pd.Series(hma_signal, index=df.index)

    return {
        "volatility_oscillator": vo,
        "hull_ma": hma,
        "hma_signal": hma_signal,
        "boom_hunter": boom,
        "modified_atr": atr,
    }


# ---------------------------------------------------------------------------
# R2c: Triangle Pattern Detection + Bollinger Squeeze
# ---------------------------------------------------------------------------

def detect_triangles(df: pd.DataFrame, order: int = 5, lookback: int = 60) -> dict:
    """
    Detect triangle patterns using local extrema and linear regression.
    Returns trendline slopes and whether a triangle (converging lines) exists.
    """
    if len(df) < lookback:
        return {"pattern": "insufficient_data"}

    recent = df.tail(lookback)
    highs = recent["High"].values
    lows = recent["Low"].values
    x = np.arange(len(recent))

    local_max_idx = argrelextrema(highs, np.greater, order=order)[0]
    local_min_idx = argrelextrema(lows, np.less, order=order)[0]

    if len(local_max_idx) < 2 or len(local_min_idx) < 2:
        return {"pattern": "no_pattern"}

    res_slope, res_intercept, _, _, _ = linregress(local_max_idx, highs[local_max_idx])
    sup_slope, sup_intercept, _, _, _ = linregress(local_min_idx, lows[local_min_idx])

    resistance_line = res_slope * x + res_intercept
    support_line = sup_slope * x + sup_intercept

    converging = (res_slope < 0 and sup_slope > 0)
    ascending = (abs(res_slope) < 0.05 * np.std(highs) and sup_slope > 0)
    descending = (res_slope < 0 and abs(sup_slope) < 0.05 * np.std(lows))

    last_close = recent["Close"].iloc[-1]
    breakout_up = last_close > resistance_line[-1]
    breakout_down = last_close < support_line[-1]

    if converging:
        pattern = "symmetrical_triangle"
    elif ascending:
        pattern = "ascending_triangle"
    elif descending:
        pattern = "descending_triangle"
    else:
        pattern = "no_pattern"

    return {
        "pattern": pattern,
        "resistance_slope": res_slope,
        "support_slope": sup_slope,
        "resistance_line": resistance_line,
        "support_line": support_line,
        "breakout_up": breakout_up,
        "breakout_down": breakout_down,
        "dates": recent.index,
    }


def bollinger_squeeze(df: pd.DataFrame, bb_period: int = 20, bb_std: float = 2.0,
                      kc_period: int = 20, kc_mult: float = 1.5) -> pd.DataFrame:
    """
    Bollinger Band squeeze detection.
    Squeeze = Bollinger Bands inside Keltner Channels (narrowing volatility).
    """
    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    bb_mid = close.rolling(bb_period).mean()
    bb_std_val = close.rolling(bb_period).std()
    bb_upper = bb_mid + bb_std * bb_std_val
    bb_lower = bb_mid - bb_std * bb_std_val
    bb_width = (bb_upper - bb_lower) / bb_mid

    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    kc_mid = close.rolling(kc_period).mean()
    kc_atr = tr.rolling(kc_period).mean()
    kc_upper = kc_mid + kc_mult * kc_atr
    kc_lower = kc_mid - kc_mult * kc_atr

    squeeze_on = (bb_lower > kc_lower) & (bb_upper < kc_upper)
    squeeze_off = (bb_lower < kc_lower) | (bb_upper > kc_upper)

    return pd.DataFrame({
        "bb_mid": bb_mid,
        "bb_upper": bb_upper, "bb_lower": bb_lower, "bb_width": bb_width,
        "kc_upper": kc_upper, "kc_lower": kc_lower,
        "squeeze_on": squeeze_on, "squeeze_off": squeeze_off,
    }, index=df.index)


# ---------------------------------------------------------------------------
# R2d: Momentum Bars
# ---------------------------------------------------------------------------

def momentum_bars(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Rate of Change momentum oscillator with directional coloring."""
    close = df["Close"]
    roc = ((close - close.shift(period)) / close.shift(period)) * 100
    roc_prev = roc.shift(1)
    direction = np.where(
        (roc > 0) & (roc > roc_prev), 2,       # strong bullish
        np.where(roc > 0, 1,                     # mild bullish
        np.where((roc < 0) & (roc < roc_prev), -2,  # strong bearish
        np.where(roc < 0, -1, 0)))               # mild bearish
    )
    return pd.DataFrame({
        "roc": roc,
        "direction": direction,
    }, index=df.index)


# ---------------------------------------------------------------------------
# R2e: Whale Accumulation Detection (volume-based)
# ---------------------------------------------------------------------------

def whale_volume_detection(df: pd.DataFrame, avg_period: int = 20,
                           threshold: float = 2.0) -> pd.DataFrame:
    """Flag days with abnormally high volume (>threshold x average)."""
    if "Volume" not in df.columns or df["Volume"].sum() == 0:
        return pd.DataFrame({"volume_ratio": pd.Series(dtype=float),
                             "whale_flag": pd.Series(dtype=bool)}, index=df.index)
    vol = df["Volume"]
    avg_vol = vol.rolling(avg_period).mean()
    ratio = vol / avg_vol
    whale = ratio > threshold
    return pd.DataFrame({
        "volume": vol,
        "avg_volume": avg_vol,
        "volume_ratio": ratio,
        "whale_flag": whale,
    }, index=df.index)


# ---------------------------------------------------------------------------
# R3: RSI and SMA
# ---------------------------------------------------------------------------

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple Moving Average."""
    return series.rolling(period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=period, adjust=False).mean()


def sma_crossover(series: pd.Series, fast: int = 20, slow: int = 50) -> pd.DataFrame:
    """Detect SMA crossovers."""
    sma_fast = sma(series, fast)
    sma_slow = sma(series, slow)
    cross_up = (sma_fast > sma_slow) & (sma_fast.shift(1) <= sma_slow.shift(1))
    cross_down = (sma_fast < sma_slow) & (sma_fast.shift(1) >= sma_slow.shift(1))
    return pd.DataFrame({
        f"sma_{fast}": sma_fast,
        f"sma_{slow}": sma_slow,
        "cross_up": cross_up,
        "cross_down": cross_down,
        "fast_above_slow": sma_fast > sma_slow,
    }, index=series.index)


def ema_crossover(series: pd.Series, fast: int = 9, slow: int = 21) -> pd.DataFrame:
    """Detect EMA crossovers for short-term trend signals."""
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    cross_up = (ema_fast > ema_slow) & (ema_fast.shift(1) <= ema_slow.shift(1))
    cross_down = (ema_fast < ema_slow) & (ema_fast.shift(1) >= ema_slow.shift(1))
    return pd.DataFrame({
        f"ema_{fast}": ema_fast,
        f"ema_{slow}": ema_slow,
        "cross_up": cross_up,
        "cross_down": cross_down,
        "fast_above_slow": ema_fast > ema_slow,
    }, index=series.index)
