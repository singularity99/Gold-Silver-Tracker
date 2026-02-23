import json
import numpy as np
import pandas as pd
from streamlit_lightweight_charts import renderLightweightCharts
from technicals import (
    vobhs_composite, rsi, sma_crossover, ema_crossover, momentum_bars,
    bollinger_squeeze, whale_volume_detection, fibonacci_levels,
)

COLOR_BULL = "rgba(38,166,154,0.9)"
COLOR_BEAR = "rgba(239,83,80,0.9)"

_DARK_LAYOUT = {
    "background": {"type": "solid", "color": "#131722"},
    "textColor": "#d1d4dc",
}

_DARK_GRID = {
    "vertLines": {"color": "rgba(42, 46, 57, 0.3)"},
    "horzLines": {"color": "rgba(42, 46, 57, 0.6)"},
}


def _timescale_preset(timeframe_label: str, is_intraday: bool) -> dict:
    label = (timeframe_label or "").lower()
    if "4 week" in label:
        return {
            "borderColor": "rgba(197, 203, 206, 0.4)",
            "timeVisible": True,
            "secondsVisible": False,
            "rightOffset": 2,
            "barSpacing": 8,
            "minBarSpacing": 4,
            "rightBarStaysOnScroll": True,
        }
    if "3 month" in label:
        return {
            "borderColor": "rgba(197, 203, 206, 0.4)",
            "timeVisible": False,
            "secondsVisible": False,
            "rightOffset": 3,
            "barSpacing": 6,
            "minBarSpacing": 3,
            "rightBarStaysOnScroll": True,
        }
    return {
        "borderColor": "rgba(197, 203, 206, 0.4)",
        "timeVisible": is_intraday,
        "secondsVisible": False,
        "rightOffset": 4,
        "barSpacing": 5,
        "minBarSpacing": 2,
        "rightBarStaysOnScroll": True,
    }


def _df_to_candles(df: pd.DataFrame) -> list[dict]:
    """Convert OHLCV dataframe to lightweight-charts candlestick format."""
    out = df[["Open", "High", "Low", "Close"]].copy()
    out.columns = ["open", "high", "low", "close"]
    out["time"] = df.index.strftime("%Y-%m-%d")
    if _has_intraday_index(df):
        out["time"] = (df.index.astype(np.int64) // 10**9).astype(int)
    return json.loads(out.to_json(orient="records"))


def _series_to_line(series: pd.Series, df: pd.DataFrame) -> list[dict]:
    """Convert a pandas Series to lightweight-charts line format."""
    out = pd.DataFrame({"value": series}, index=df.index)
    out["time"] = df.index.strftime("%Y-%m-%d")
    if _has_intraday_index(df):
        out["time"] = (df.index.astype(np.int64) // 10**9).astype(int)
    out = out.dropna(subset=["value"])
    return json.loads(out.to_json(orient="records"))


def _series_to_histogram(series: pd.Series, df: pd.DataFrame,
                         color_series: pd.Series = None) -> list[dict]:
    """Convert a pandas Series to histogram format with optional per-bar color."""
    out = pd.DataFrame({"value": series}, index=df.index)
    out["time"] = df.index.strftime("%Y-%m-%d")
    if _has_intraday_index(df):
        out["time"] = (df.index.astype(np.int64) // 10**9).astype(int)
    if color_series is not None:
        out["color"] = color_series.values
    out = out.dropna(subset=["value"])
    return json.loads(out.to_json(orient="records"))


def _has_intraday_index(df: pd.DataFrame) -> bool:
    if len(df) > 1 and hasattr(df.index, "dtype") and pd.api.types.is_datetime64_any_dtype(df.index):
        diff = (df.index[1] - df.index[0]).total_seconds()
        return diff < 86400
    return False


def _fib_markers(fib_levels: dict, df: pd.DataFrame) -> list[dict]:
    """Create horizontal line data for Fibonacci levels (as line series)."""
    lines = []
    for ratio_key, level in fib_levels.items():
        if ratio_key in ("swing_high", "swing_low"):
            continue
        time_data = _series_to_line(
            pd.Series(level, index=df.index), df
        )
        lines.append((ratio_key, level, time_data))
    return lines


def render_metal_chart(df: pd.DataFrame, fib_levels: dict, metal: str,
                       timeframe_label: str, ema_fast: int = 9,
                       ema_slow: int = 21, sma_fast: int = 20,
                       sma_slow: int = 50, rsi_period: int = 14,
                       chart_key: str = "chart"):
    """
    Render a multi-pane TradingView-style chart with:
    Pane 1: Candlestick + EMA + HMA + SMA + Fibonacci lines
    Pane 2: Volume (with whale detection coloring)
    Pane 3: RSI
    Pane 4: Volatility Oscillator
    Pane 5: Boom Hunter Pro
    Pane 6: Momentum Bars
    """
    if df.empty or len(df) < 20:
        return

    is_intraday = _has_intraday_index(df)

    # Compute indicators
    vobhs = vobhs_composite(df) if len(df) > 100 else None
    rsi_vals = rsi(df["Close"], period=rsi_period)
    ema_data = ema_crossover(df["Close"], fast=ema_fast, slow=ema_slow)
    sma_data = sma_crossover(df["Close"], fast=sma_fast, slow=sma_slow)
    mom = momentum_bars(df)
    whale = whale_volume_detection(df)

    candles = _df_to_candles(df)

    # --- Pane 1: Price + overlays ---
    price_series = [
        {
            "type": "Candlestick",
            "data": candles,
            "options": {
                "upColor": COLOR_BULL, "downColor": COLOR_BEAR,
                "borderVisible": False,
                "wickUpColor": COLOR_BULL, "wickDownColor": COLOR_BEAR,
            },
        },
    ]

    # EMA overlays (short-term trend)
    price_series.append({
        "type": "Line",
        "data": _series_to_line(ema_data[f"ema_{ema_fast}"], df),
        "options": {"color": "rgba(255,255,0,0.8)", "lineWidth": 1},
    })
    price_series.append({
        "type": "Line",
        "data": _series_to_line(ema_data[f"ema_{ema_slow}"], df),
        "options": {"color": "rgba(255,200,0,0.8)", "lineWidth": 1},
    })

    # SMA overlays
    price_series.append({
        "type": "Line",
        "data": _series_to_line(sma_data[f"sma_{sma_fast}"], df),
        "options": {"color": "rgba(255,165,0,0.6)", "lineWidth": 1},
    })
    price_series.append({
        "type": "Line",
        "data": _series_to_line(sma_data[f"sma_{sma_slow}"], df),
        "options": {"color": "rgba(0,255,255,0.8)", "lineWidth": 1},
    })

    # HMA overlay
    if vobhs:
        price_series.append({
            "type": "Line",
            "data": _series_to_line(vobhs["hull_ma"], df),
            "options": {"color": "rgba(186,85,211,0.9)", "lineWidth": 2},
        })

    # Fibonacci lines as overlays
    for ratio_key, level, line_data in _fib_markers(fib_levels, df):
        price_series.append({
            "type": "Line",
            "data": line_data,
            "options": {
                "color": "rgba(150,150,150,0.5)", "lineWidth": 1,
                "lineStyle": 2,
                "crosshairMarkerVisible": False,
                "lastValueVisible": True,
                "priceLineVisible": False,
            },
        })

    # Bollinger Bands
    bsq = bollinger_squeeze(df)
    if not bsq.empty:
        price_series.append({
            "type": "Line",
            "data": _series_to_line(bsq["bb_upper"], df),
            "options": {"color": "rgba(100,149,237,0.4)", "lineWidth": 1, "lineStyle": 2},
        })
        price_series.append({
            "type": "Line",
            "data": _series_to_line(bsq["bb_lower"], df),
            "options": {"color": "rgba(100,149,237,0.4)", "lineWidth": 1, "lineStyle": 2},
        })

    scale_preset = _timescale_preset(timeframe_label, is_intraday)

    price_chart = {
        "height": 450,
        "layout": _DARK_LAYOUT,
        "grid": _DARK_GRID,
        "crosshair": {"mode": 1},
        "timeScale": scale_preset,
        "watermark": {
            "visible": True, "fontSize": 36,
            "horzAlign": "center", "vertAlign": "center",
            "color": "rgba(171, 71, 188, 0.15)",
            "text": f"{metal.upper()} - {timeframe_label}",
        },
    }

    # --- Pane 2: Volume ---
    vol_colors = np.where(df["Close"] >= df["Open"], COLOR_BULL, COLOR_BEAR)
    if not whale.empty and "whale_flag" in whale.columns:
        vol_colors = np.where(whale["whale_flag"], "rgba(255,215,0,0.9)", vol_colors)
    volume_data = _series_to_histogram(df["Volume"], df, pd.Series(vol_colors, index=df.index))

    volume_chart = {
        "height": 120,
        "layout": {"background": {"type": "solid", "color": "transparent"}, "textColor": "#d1d4dc"},
        "grid": {"vertLines": {"color": "rgba(42,46,57,0)"}, "horzLines": {"color": "rgba(42,46,57,0.3)"}},
        "timeScale": {"visible": False},
        "watermark": {"visible": True, "fontSize": 16, "horzAlign": "left", "vertAlign": "top",
                       "color": "rgba(171,71,188,0.5)", "text": "Volume (gold = whale)"},
    }

    # --- Pane 3: RSI ---
    rsi_data = _series_to_line(rsi_vals, df)
    rsi_70 = _series_to_line(pd.Series(70.0, index=df.index), df)
    rsi_30 = _series_to_line(pd.Series(30.0, index=df.index), df)

    rsi_chart = {
        "height": 150,
        "layout": _DARK_LAYOUT,
        "grid": _DARK_GRID,
        "timeScale": {"visible": False},
        "watermark": {"visible": True, "fontSize": 16, "horzAlign": "left", "vertAlign": "top",
                       "color": "rgba(171,71,188,0.5)", "text": "RSI (14)"},
    }

    rsi_series = [
        {"type": "Line", "data": rsi_data, "options": {"color": "rgba(33,150,243,1)", "lineWidth": 2}},
        {"type": "Line", "data": rsi_70, "options": {"color": "rgba(239,83,80,0.5)", "lineWidth": 1, "lineStyle": 2}},
        {"type": "Line", "data": rsi_30, "options": {"color": "rgba(38,166,154,0.5)", "lineWidth": 1, "lineStyle": 2}},
    ]

    charts_config = [
        {"chart": price_chart, "series": price_series},
        {"chart": volume_chart, "series": [{"type": "Histogram", "data": volume_data,
                                             "options": {"priceFormat": {"type": "volume"}, "priceScaleId": ""},
                                             "priceScale": {"scaleMargins": {"top": 0, "bottom": 0}, "alignLabels": False}}]},
        {"chart": rsi_chart, "series": rsi_series},
    ]

    # --- Pane 4: Volatility Oscillator (if enough data) ---
    if vobhs:
        vo = vobhs["volatility_oscillator"]
        vo_colors = pd.Series(
            np.where(vo["spike"] > vo["upper"], COLOR_BULL,
                     np.where(vo["spike"] < vo["lower"], COLOR_BEAR, "rgba(150,150,150,0.5)")),
            index=df.index,
        )
        vo_data = _series_to_histogram(vo["spike"], df, vo_colors)
        vo_upper = _series_to_line(vo["upper"], df)
        vo_lower = _series_to_line(vo["lower"], df)

        vo_chart = {
            "height": 150,
            "layout": _DARK_LAYOUT,
            "grid": _DARK_GRID,
            "timeScale": {"visible": False},
            "watermark": {"visible": True, "fontSize": 16, "horzAlign": "left", "vertAlign": "top",
                           "color": "rgba(171,71,188,0.5)", "text": "Volatility Oscillator"},
        }
        charts_config.append({
            "chart": vo_chart,
            "series": [
                {"type": "Histogram", "data": vo_data, "options": {}},
                {"type": "Line", "data": vo_upper, "options": {"color": "rgba(38,166,154,0.6)", "lineWidth": 1, "lineStyle": 2}},
                {"type": "Line", "data": vo_lower, "options": {"color": "rgba(239,83,80,0.6)", "lineWidth": 1, "lineStyle": 2}},
            ],
        })

        # --- Pane 5: Boom Hunter Pro ---
        boom = vobhs["boom_hunter"]
        trigger_data = _series_to_line(boom["trigger"], df)
        quotient_data = _series_to_line(boom["quotient"], df)

        boom_chart = {
            "height": 150,
            "layout": _DARK_LAYOUT,
            "grid": _DARK_GRID,
            "timeScale": {"visible": False},
            "watermark": {"visible": True, "fontSize": 16, "horzAlign": "left", "vertAlign": "top",
                           "color": "rgba(171,71,188,0.5)", "text": "Boom Hunter Pro (BHS)"},
        }
        charts_config.append({
            "chart": boom_chart,
            "series": [
                {"type": "Line", "data": trigger_data, "options": {"color": "rgba(255,255,255,0.9)", "lineWidth": 2}},
                {"type": "Line", "data": quotient_data, "options": {"color": "rgba(239,83,80,0.9)", "lineWidth": 2}},
            ],
        })

    # --- Pane 6: Momentum Bars ---
    if not mom.empty:
        mom_colors = pd.Series(
            np.where(mom["direction"] == 2, "rgba(0,128,0,0.9)",
                     np.where(mom["direction"] == 1, "rgba(144,238,144,0.8)",
                              np.where(mom["direction"] == -1, "rgba(250,128,114,0.8)",
                                       np.where(mom["direction"] == -2, "rgba(139,0,0,0.9)", "rgba(150,150,150,0.4)")))),
            index=df.index,
        )
        mom_data = _series_to_histogram(mom["roc"], df, mom_colors)

        mom_chart = {
            "height": 130,
            "layout": _DARK_LAYOUT,
            "grid": _DARK_GRID,
            "timeScale": {"visible": True, **scale_preset},
            "watermark": {"visible": True, "fontSize": 16, "horzAlign": "left", "vertAlign": "top",
                           "color": "rgba(171,71,188,0.5)", "text": "Momentum (ROC)"},
        }
        charts_config.append({
            "chart": mom_chart,
            "series": [{"type": "Histogram", "data": mom_data, "options": {}}],
        })

    renderLightweightCharts(charts_config, chart_key)
