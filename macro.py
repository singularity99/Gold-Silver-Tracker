"""
Zeburg (TM) Macro Navigation Framework - Crossover-Based Implementation

This module implements crossover-based scoring per the Zeburg methodology:
- Signals come from indicators CROSSING their equilibrium/trajectory, not absolute levels
- Tracks both crossed and pending indicators
- Hierarchical phase classification: Leading -> Imminent -> Coincident -> Lagging
"""

import numpy as np
import pandas as pd
import yfinance as yf

FRED_BASE = "https://fred.stlouisfed.org/graph/fredgraph.csv?id="

FRED_SERIES = {
    "yield_spread_10y2y": "T10Y2Y",
    "yield_spread_10y3m": "T10Y3M",
    "building_permits": "PERMIT",
    "housing_starts": "HOUST",
    "factory_orders": "AMTMNO",
    "consumer_sentiment": "UMCSENT",
    "credit_spread": "BAA10YM",
    "financial_stress": "STLFSI4",
    "payrolls": "PAYEMS",
    "industrial_prod": "INDPRO",
    "initial_claims": "ICSA",
    "unemployment": "UNRATE",
    "fed_funds": "FEDFUNDS",
    "cpi": "CPIAUCSL",
}


def _fetch_fred_series(series_id: str) -> pd.Series:
    try:
        df = pd.read_csv(FRED_BASE + series_id)
        date_col = "DATE" if "DATE" in df.columns else ("observation_date" if "observation_date" in df.columns else None)
        if date_col is None or series_id not in df.columns:
            return pd.Series(dtype=float)
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df[series_id] = pd.to_numeric(df[series_id], errors="coerce")
        s = df.dropna(subset=[date_col]).set_index(date_col)[series_id].dropna()
        return s.sort_index()
    except Exception:
        return pd.Series(dtype=float)


def fetch_macro_series_bundle() -> dict[str, pd.Series]:
    return {k: _fetch_fred_series(v) for k, v in FRED_SERIES.items()}


def _fetch_yahoo_close_bundle(tickers: list[str], period: str = "3y") -> dict[str, pd.Series]:
    try:
        df = yf.download(tickers=tickers, period=period, interval="1d", progress=False, auto_adjust=False)
        out = {}
        if isinstance(df.columns, pd.MultiIndex):
            for tk in tickers:
                if ("Close", tk) in df.columns:
                    out[tk] = pd.to_numeric(df[("Close", tk)], errors="coerce").dropna()
                elif (tk, "Close") in df.columns:
                    out[tk] = pd.to_numeric(df[(tk, "Close")], errors="coerce").dropna()
                else:
                    out[tk] = pd.Series(dtype=float)
        else:
            if "Close" in df.columns and len(tickers) == 1:
                out[tickers[0]] = pd.to_numeric(df["Close"], errors="coerce").dropna()
            else:
                out = {tk: pd.Series(dtype=float) for tk in tickers}
        return out
    except Exception:
        return {tk: pd.Series(dtype=float) for tk in tickers}


def _slice_asof(s: pd.Series, asof=None) -> pd.Series:
    if s is None or s.empty:
        return pd.Series(dtype=float)
    out = s.dropna()
    if asof is not None:
        out = out[out.index <= pd.Timestamp(asof)]
    return out


def _latest(s: pd.Series, asof=None) -> float:
    v = _slice_asof(s, asof)
    return float(v.iloc[-1]) if not v.empty else np.nan


def _pct_change(s: pd.Series, periods: int, asof=None) -> float:
    v = _slice_asof(s, asof)
    if len(v) <= periods:
        return np.nan
    prev = float(v.iloc[-(periods + 1)])
    now = float(v.iloc[-1])
    if prev == 0:
        return np.nan
    return (now / prev - 1.0) * 100


def _mean_tail(s: pd.Series, tail: int, asof=None) -> float:
    v = _slice_asof(s, asof)
    if v.empty:
        return np.nan
    return float(v.tail(tail).mean())


def _classify(score: float, buy: float, strong_buy: float, sell: float, strong_sell: float) -> str:
    if score >= strong_buy:
        return "Strong Buy"
    if score >= buy:
        return "Buy"
    if score <= strong_sell:
        return "Strong Sell"
    if score <= sell:
        return "Sell / Take Profit"
    return "Neutral"


# ── Crossover Detection Functions ──────────────────────────────────────────────

def _detect_zero_cross(series: pd.Series, lookback: int = 4) -> dict:
    """
    Detect crossover of zero line (e.g., yield curve inversion).
    Returns: {crossed: bool, direction: 'above'/'below'/None, current: float, prev: float}
    """
    if series is None or len(series) < lookback + 1:
        return {"crossed": False, "direction": None, "current": np.nan, "prev": np.nan}
    
    recent = series.tail(lookback + 1)
    current = float(recent.iloc[-1])
    prev = float(recent.iloc[-2])
    
    crossed_below = prev >= 0 and current < 0
    crossed_above = prev <= 0 and current > 0
    
    if crossed_below:
        return {"crossed": True, "direction": "below", "current": current, "prev": prev}
    elif crossed_above:
        return {"crossed": True, "direction": "above", "current": current, "prev": prev}
    else:
        return {"crossed": False, "direction": None, "current": current, "prev": prev}


def _detect_ma_cross(series: pd.Series, window: int = 26, asof=None) -> dict:
    """
    Detect crossover of moving average (equilibrium).
    Returns: {crossed: bool, direction: 'above'/'below'/None, current: float, ma: float, distance: float}
    """
    v = _slice_asof(series, asof)
    if v is None or len(v) < window + 1:
        return {"crossed": False, "direction": None, "current": np.nan, "ma": np.nan, "distance": np.nan}
    
    ma = v.rolling(window).mean()
    current = float(v.iloc[-1])
    prev = float(v.iloc[-2])
    current_ma = float(ma.iloc[-1])
    prev_ma = float(ma.iloc[-2])
    
    crossed_below = prev >= prev_ma and current < current_ma
    crossed_above = prev <= prev_ma and current > current_ma
    
    distance = (current - current_ma) / current_ma * 100 if current_ma != 0 else 0
    
    if crossed_below:
        return {"crossed": True, "direction": "below", "current": current, "ma": current_ma, "distance": distance}
    elif crossed_above:
        return {"crossed": True, "direction": "above", "current": current, "ma": current_ma, "distance": distance}
    else:
        return {"crossed": False, "direction": None, "current": current, "ma": current_ma, "distance": distance}


def _detect_trend_cross(series: pd.Series, short_window: int = 13, long_window: int = 52, asof=None) -> dict:
    """
    Detect when short-term average crosses long-term average.
    Used for claims acceleration (13w vs 52w) and similar indicators.
    """
    v = _slice_asof(series, asof)
    if v is None or len(v) < long_window + 1:
        return {"crossed": False, "direction": None, "ratio": np.nan}
    
    short_ma = v.rolling(short_window).mean()
    long_ma = v.rolling(long_window).mean()
    
    current_short = float(short_ma.iloc[-1])
    current_long = float(long_ma.iloc[-1])
    prev_short = float(short_ma.iloc[-2])
    prev_long = float(long_ma.iloc[-2])
    
    ratio = current_short / current_long if current_long != 0 else np.nan
    
    crossed_above = prev_short <= prev_long and current_short > current_long
    crossed_below = prev_short >= prev_long and current_short < current_long
    
    if crossed_above:
        return {"crossed": True, "direction": "above", "ratio": ratio}
    elif crossed_below:
        return {"crossed": True, "direction": "below", "ratio": ratio}
    else:
        return {"crossed": False, "direction": None, "ratio": ratio}


def _detect_sahm_trigger(unrate_series: pd.Series, asof=None) -> dict:
    """
    Sahm Rule: recession trigger when 3-month average unemployment rate 
    is at least 0.5 percentage points above its 12-month low.
    """
    v = _slice_asof(unrate_series, asof)
    if v is None or len(v) < 12:
        return {"crossed": False, "triggered": False, "value": np.nan}
    
    three_month_avg = float(v.tail(3).mean())
    twelve_month_low = float(v.tail(12).min())
    sahm_value = three_month_avg - twelve_month_low
    
    # Check if just crossed the 0.5 threshold
    prev_three_month = float(v.iloc[-4:-1].mean()) if len(v) >= 4 else three_month_avg
    prev_twelve_low = float(v.iloc[-13:-1].min()) if len(v) >= 13 else twelve_month_low
    prev_sahm = prev_three_month - prev_twelve_low
    
    crossed = prev_sahm < 0.5 and sahm_value >= 0.5
    triggered = sahm_value >= 0.5
    
    return {
        "crossed": crossed,
        "triggered": triggered,
        "value": sahm_value,
        "threshold": 0.5
    }


# ── Indicator Vote with Crossover Logic ───────────────────────────────────────

def _indicator_vote(name: str, cross_info: dict, indicator_type: str) -> dict:
    """
    Build indicator vote based on crossover detection.
    Returns: {name, vote: int, status: str, detail: str, crossed: bool, direction: str}
    """
    crossed = cross_info.get("crossed", False)
    direction = cross_info.get("direction")
    
    if not cross_info or np.isnan(cross_info.get("current", np.nan)):
        return {
            "name": name,
            "vote": 0,
            "status": "No data",
            "detail": "Data unavailable",
            "crossed": False,
            "direction": None
        }
    
    # Determine vote based on crossover direction
    vote = 0
    status = "Waiting"
    detail = ""
    
    if crossed:
        status = "Crossed"
        if direction == "below":
            # Crossing below equilibrium is typically bearish for macro
            if indicator_type in ("leading", "coincident"):
                vote = -1  # Bearish signal
                detail = f"Crossed below equilibrium ({cross_info.get('prev', 0):.2f} -> {cross_info.get('current', 0):.2f})"
            else:  # imminent
                vote = -1  # Bearish trigger
                detail = f"Trigger crossed ({cross_info.get('value', 0):.2f})"
        elif direction == "above":
            # Crossing above equilibrium is typically bullish
            if indicator_type in ("leading", "coincident"):
                vote = 1  # Bullish signal
                detail = f"Crossed above equilibrium ({cross_info.get('prev', 0):.2f} -> {cross_info.get('current', 0):.2f})"
            else:  # imminent
                vote = 1  # Bullish trigger (clearing)
                detail = f"Trigger cleared ({cross_info.get('value', 0):.2f})"
    else:
        # Waiting for crossover - show current position relative to threshold
        current = cross_info.get("current", np.nan)
        ma = cross_info.get("ma", np.nan)
        distance = cross_info.get("distance", np.nan)
        ratio = cross_info.get("ratio", np.nan)
        
        if not np.isnan(current):
            if indicator_type == "yield":
                if current < 0:
                    status = "Waiting (inverted)"
                    detail = f"Yield curve inverted at {current:.2f}% - waiting for re-steepening"
                else:
                    status = "Waiting (normal)"
                    detail = f"Yield curve normal at {current:.2f}% - watching for inversion"
            elif indicator_type == "sahm":
                triggered = cross_info.get("triggered", False)
                value = cross_info.get("value", 0)
                if triggered:
                    status = "Triggered"
                    vote = -1
                    detail = f"Sahm rule triggered at {value:.2f}pp (threshold 0.5pp)"
                else:
                    status = "Waiting"
                    detail = f"Sahm value {value:.2f}pp (threshold 0.5pp)"
            elif not np.isnan(distance):
                if distance < 0:
                    status = "Waiting (below MA)"
                    detail = f"{distance:.1f}% below equilibrium"
                else:
                    status = "Waiting (above MA)"
                    detail = f"{distance:.1f}% above equilibrium"
            elif not np.isnan(ratio):
                if ratio > 1:
                    status = "Waiting (above trend)"
                    detail = f"Ratio {ratio:.2f} (above long-term average)"
                else:
                    status = "Waiting (below trend)"
                    detail = f"Ratio {ratio:.2f} (below long-term average)"
            else:
                status = "Waiting"
                detail = f"Current: {current:.2f}"
    
    return {
        "name": name,
        "vote": vote,
        "status": status,
        "detail": detail,
        "crossed": crossed,
        "direction": direction
    }


def _build_macro_response(
    phase: str,
    confidence: float,
    leading_score: float,
    coincident_score: float,
    imminent_score: float,
    leading_indicators: list,
    coincident_indicators: list,
    imminent_indicators: list,
    metrics: dict,
    source: str,
) -> dict:
    phase_bias = {
        "Expansion": {"gold": -0.04, "silver": 0.04},
        "Slowdown": {"gold": 0.03, "silver": -0.01},
        "Contraction": {"gold": 0.08, "silver": -0.06},
        "Recovery": {"gold": 0.00, "silver": 0.05},
    }
    adaptive_thresholds = {
        "gold": {
            "Expansion": {"buy": 0.22, "strong_buy": 0.42, "sell": -0.22, "strong_sell": -0.42},
            "Slowdown": {"buy": 0.18, "strong_buy": 0.38, "sell": -0.22, "strong_sell": -0.42},
            "Contraction": {"buy": 0.15, "strong_buy": 0.35, "sell": -0.25, "strong_sell": -0.45},
            "Recovery": {"buy": 0.20, "strong_buy": 0.40, "sell": -0.20, "strong_sell": -0.40},
        },
        "silver": {
            "Expansion": {"buy": 0.18, "strong_buy": 0.38, "sell": -0.22, "strong_sell": -0.42},
            "Slowdown": {"buy": 0.24, "strong_buy": 0.44, "sell": -0.18, "strong_sell": -0.38},
            "Contraction": {"buy": 0.28, "strong_buy": 0.48, "sell": -0.15, "strong_sell": -0.35},
            "Recovery": {"buy": 0.20, "strong_buy": 0.40, "sell": -0.20, "strong_sell": -0.40},
        },
    }
    
    # Extract simple votes dict for backward compatibility
    leading_votes = {ind["name"]: ind["vote"] for ind in leading_indicators}
    coincident_votes = {ind["name"]: ind["vote"] for ind in coincident_indicators}
    imminent_votes = {ind["name"]: ind["vote"] for ind in imminent_indicators}
    
    return {
        "phase": phase,
        "confidence": confidence,
        "leading_score": round(leading_score, 2),
        "coincident_score": round(coincident_score, 2),
        "imminent_score": round(imminent_score, 2),
        "leading_votes": leading_votes,
        "coincident_votes": coincident_votes,
        "imminent_votes": imminent_votes,
        "leading_indicators": leading_indicators,
        "coincident_indicators": coincident_indicators,
        "imminent_indicators": imminent_indicators,
        "source": source,
        "metrics": metrics,
        "phase_bias": phase_bias,
        "adaptive_thresholds": adaptive_thresholds,
    }


def _phase_from_scores(leading_score: float, coincident_score: float, imminent_score: float) -> str:
    if coincident_score <= -0.5 or (coincident_score < 0 and imminent_score <= -0.5):
        return "Contraction"
    if coincident_score >= 0.25 and leading_score >= 0:
        return "Expansion"
    if coincident_score < 0 and leading_score >= 0:
        return "Recovery"
    if leading_score < 0:
        return "Slowdown"
    return "Recovery"


def _get_yahoo_proxy_state(asof=None) -> dict:
    """Fallback using Yahoo proxies when FRED unavailable."""
    tickers = ["^TNX", "^IRX", "HG=F", "GC=F", "XHB", "ITB", "XLY", "XLP", "XLI", "XLU", "IWM", "^GSPC", "^VIX", "HYG", "LQD"]
    b = _fetch_yahoo_close_bundle(tickers)

    spread_series = _slice_asof(b.get("^TNX", pd.Series(dtype=float)), asof) - _slice_asof(b.get("^IRX", pd.Series(dtype=float)), asof)
    housing_series = _slice_asof(b.get("XHB", pd.Series(dtype=float)), asof)
    permits_series = _slice_asof(b.get("ITB", pd.Series(dtype=float)), asof)
    
    # Leading indicators with crossover detection
    leading_indicators = []
    
    # 1. Yield curve - zero crossover
    spread_cross = _detect_zero_cross(spread_series)
    leading_indicators.append(_indicator_vote(
        "Yield curve (10Y-2Y proxy)", 
        spread_cross, 
        "yield"
    ))
    
    # 2. Housing - MA crossover
    housing_cross = _detect_ma_cross(housing_series, window=126)  # ~6 months
    leading_indicators.append(_indicator_vote(
        "Housing starts proxy (XHB)", 
        housing_cross, 
        "leading"
    ))
    
    # 3. Permits - MA crossover
    permits_cross = _detect_ma_cross(permits_series, window=126)
    leading_indicators.append(_indicator_vote(
        "Building permits proxy (ITB)", 
        permits_cross, 
        "leading"
    ))
    
    # Calculate scores
    leading_votes = [ind["vote"] for ind in leading_indicators]
    leading_score = float(np.mean(leading_votes)) if leading_votes else 0.0
    
    # Simplified coincident/imminent for proxy mode
    vix = _latest(b.get("^VIX", pd.Series(dtype=float)), asof)
    coincident_indicators = [{
        "name": "Risk appetite proxy",
        "vote": -1 if vix >= 25 else (1 if vix <= 14 else 0),
        "status": "Triggered" if vix >= 25 else "Waiting",
        "detail": f"VIX at {vix:.1f}",
        "crossed": vix >= 25,
        "direction": "above" if vix >= 25 else None
    }]
    
    imminent_indicators = [{
        "name": "VIX spike risk",
        "vote": -1 if vix >= 30 else 0,
        "status": "Triggered" if vix >= 30 else "Waiting",
        "detail": f"VIX at {vix:.1f} (threshold 30)",
        "crossed": vix >= 30,
        "direction": "above" if vix >= 30 else None
    }]
    
    coincident_score = float(np.mean([ind["vote"] for ind in coincident_indicators]))
    imminent_score = float(np.mean([ind["vote"] for ind in imminent_indicators]))
    
    phase = _phase_from_scores(leading_score, coincident_score, imminent_score)
    confidence = round(min(1.0, (abs(leading_score) + abs(coincident_score) + abs(imminent_score)) / 2.5), 2)

    metrics = {
        "yield_spread_10y2y": spread_cross.get("current", np.nan),
        "yield_spread_10y3m": spread_cross.get("current", np.nan),
        "vix": vix,
    }
    
    return _build_macro_response(
        phase, confidence, leading_score, coincident_score, imminent_score,
        leading_indicators, coincident_indicators, imminent_indicators,
        metrics, "yahoo_proxy"
    )


def get_macro_framework_state(asof=None, series_bundle: dict[str, pd.Series] | None = None) -> dict:
    """
    Get macro state using crossover-based methodology.
    
    Returns dict with:
    - phase: current macro phase
    - leading_indicators: list of indicator dicts with crossover status
    - coincident_indicators: list of indicator dicts with crossover status  
    - imminent_indicators: list of indicator dicts with crossover status
    """
    b = series_bundle or fetch_macro_series_bundle()
    if not any((s is not None and len(s) > 0) for s in b.values()):
        return _get_yahoo_proxy_state(asof)

    # ── Leading Indicators (Crossover-Based) ────────────────────────────────────
    leading_indicators = []
    
    # 1. Yield curve 10Y-3M - zero crossover (inversion detection)
    spread_series = _slice_asof(b.get("yield_spread_10y3m", pd.Series(dtype=float)), asof)
    spread_cross = _detect_zero_cross(spread_series)
    leading_indicators.append(_indicator_vote(
        "Yield curve (10Y-3M)", 
        spread_cross, 
        "yield"
    ))
    
    # 2. Building permits - MA crossover (26 weeks ~ 6 months)
    permits_series = _slice_asof(b.get("building_permits", pd.Series(dtype=float)), asof)
    permits_cross = _detect_ma_cross(permits_series, window=26)
    leading_indicators.append(_indicator_vote(
        "Building permits", 
        permits_cross, 
        "leading"
    ))
    
    # 3. Housing starts - MA crossover
    housing_series = _slice_asof(b.get("housing_starts", pd.Series(dtype=float)), asof)
    housing_cross = _detect_ma_cross(housing_series, window=26)
    leading_indicators.append(_indicator_vote(
        "Housing starts", 
        housing_cross, 
        "leading"
    ))
    
    # 4. Consumer sentiment - MA crossover (12 months)
    sentiment_series = _slice_asof(b.get("consumer_sentiment", pd.Series(dtype=float)), asof)
    sentiment_cross = _detect_ma_cross(sentiment_series, window=12)
    leading_indicators.append(_indicator_vote(
        "Consumer sentiment", 
        sentiment_cross, 
        "leading"
    ))
    
    # 5. Credit spreads - MA crossover (higher = worse)
    credit_series = _slice_asof(b.get("credit_spread", pd.Series(dtype=float)), asof)
    credit_cross = _detect_ma_cross(credit_series, window=26)
    # Invert logic: crossing above MA is bearish for credit
    if credit_cross.get("direction") == "above":
        credit_cross["direction"] = "below"  # Treat as bearish signal
    elif credit_cross.get("direction") == "below":
        credit_cross["direction"] = "above"  # Treat as bullish signal
    leading_indicators.append(_indicator_vote(
        "Credit spreads (BAA-10Y)", 
        credit_cross, 
        "leading"
    ))
    
    # 6. St. Louis Financial Stress Index
    fsi_series = _slice_asof(b.get("financial_stress", pd.Series(dtype=float)), asof)
    fsi_cross = _detect_ma_cross(fsi_series, window=26)
    # Higher FSI = more stress = bearish
    if fsi_cross.get("direction") == "above":
        fsi_cross["direction"] = "below"
    elif fsi_cross.get("direction") == "below":
        fsi_cross["direction"] = "above"
    leading_indicators.append(_indicator_vote(
        "St. Louis FSI", 
        fsi_cross, 
        "leading"
    ))

    # ── Coincident Indicators ──────────────────────────────────────────────────
    coincident_indicators = []
    
    # 1. Payrolls - MA crossover
    payroll_series = _slice_asof(b.get("payrolls", pd.Series(dtype=float)), asof)
    payroll_cross = _detect_ma_cross(payroll_series, window=26)
    coincident_indicators.append(_indicator_vote(
        "Nonfarm payrolls", 
        payroll_cross, 
        "coincident"
    ))
    
    # 2. Industrial production - MA crossover
    indpro_series = _slice_asof(b.get("industrial_prod", pd.Series(dtype=float)), asof)
    indpro_cross = _detect_ma_cross(indpro_series, window=26)
    coincident_indicators.append(_indicator_vote(
        "Industrial production", 
        indpro_cross, 
        "coincident"
    ))

    # ── Imminent Recession Indicators ──────────────────────────────────────────
    imminent_indicators = []
    
    # 1. Initial claims - 13w vs 52w crossover (acceleration)
    claims_series = _slice_asof(b.get("initial_claims", pd.Series(dtype=float)), asof)
    claims_cross = _detect_trend_cross(claims_series, short_window=13, long_window=52)
    # Add threshold check: ratio > 1.08 is significant
    ratio = claims_cross.get("ratio", np.nan)
    if not np.isnan(ratio):
        claims_cross["current"] = ratio
        if ratio >= 1.08:
            claims_cross["crossed"] = True
            claims_cross["direction"] = "above"
            claims_cross["detail_extension"] = f"Claims accelerating (ratio {ratio:.2f})"
    imminent_indicators.append(_indicator_vote(
        "Initial claims acceleration", 
        claims_cross, 
        "imminent"
    ))
    
    # 2. Sahm Rule trigger
    unrate_series = _slice_asof(b.get("unemployment", pd.Series(dtype=float)), asof)
    sahm_cross = _detect_sahm_trigger(unrate_series)
    imminent_indicators.append(_indicator_vote(
        "Sahm rule trigger", 
        sahm_cross, 
        "sahm"
    ))

    # ── Calculate Scores ───────────────────────────────────────────────────────
    leading_votes = [ind["vote"] for ind in leading_indicators if ind["status"] != "No data"]
    coincident_votes = [ind["vote"] for ind in coincident_indicators if ind["status"] != "No data"]
    imminent_votes = [ind["vote"] for ind in imminent_indicators if ind["status"] != "No data"]
    
    leading_score = float(np.mean(leading_votes)) if leading_votes else 0.0
    coincident_score = float(np.mean(coincident_votes)) if coincident_votes else 0.0
    imminent_score = float(np.mean(imminent_votes)) if imminent_votes else 0.0
    
    phase = _phase_from_scores(leading_score, coincident_score, imminent_score)
    confidence = round(min(1.0, (abs(leading_score) + abs(coincident_score) + abs(imminent_score)) / 2.5), 2)

    # Build metrics
    metrics = {
        "yield_spread_10y2y": _latest(b.get("yield_spread_10y2y", pd.Series(dtype=float)), asof),
        "yield_spread_10y3m": spread_cross.get("current", np.nan),
        "building_permits_cross": permits_cross.get("distance", np.nan),
        "housing_cross": housing_cross.get("distance", np.nan),
        "claims_ratio": ratio,
        "sahm_value": sahm_cross.get("value", np.nan),
        "credit_spread_level": _latest(b.get("credit_spread", pd.Series(dtype=float)), asof),
        "financial_stress_level": _latest(b.get("financial_stress", pd.Series(dtype=float)), asof),
        "payroll_6m_change": _pct_change(b.get("payrolls", pd.Series(dtype=float)), 6, asof),
        "indpro_6m_change": _pct_change(b.get("industrial_prod", pd.Series(dtype=float)), 6, asof),
        "unemployment": _latest(b.get("unemployment", pd.Series(dtype=float)), asof),
        "cpi_yoy": _pct_change(b.get("cpi", pd.Series(dtype=float)), 12, asof),
        "fed_funds": _latest(b.get("fed_funds", pd.Series(dtype=float)), asof),
    }
    
    return _build_macro_response(
        phase, confidence, leading_score, coincident_score, imminent_score,
        leading_indicators, coincident_indicators, imminent_indicators,
        metrics, "fred"
    )


def apply_macro_overlay(score: dict, macro_state: dict, metal: str) -> dict:
    if not score or not macro_state:
        return score
    phase = macro_state.get("phase")
    if phase not in {"Expansion", "Slowdown", "Contraction", "Recovery"}:
        return score

    metal_key = "gold" if str(metal).lower().startswith("g") else "silver"
    bias = float(macro_state.get("phase_bias", {}).get(phase, {}).get(metal_key, 0.0))
    thresholds = (
        macro_state.get("adaptive_thresholds", {})
        .get(metal_key, {})
        .get(phase, {"buy": 0.2, "strong_buy": 0.4, "sell": -0.2, "strong_sell": -0.4})
    )

    raw_score = float(score.get("composite_score", 0.0))
    adjusted_score = max(-1.0, min(1.0, raw_score + bias))
    adjusted_signal = _classify(
        adjusted_score,
        buy=float(thresholds["buy"]),
        strong_buy=float(thresholds["strong_buy"]),
        sell=float(thresholds["sell"]),
        strong_sell=float(thresholds["strong_sell"]),
    )

    out = dict(score)
    out["raw_composite_score"] = raw_score
    out["raw_signal"] = score.get("signal", "Neutral")
    out["composite_score"] = round(adjusted_score, 2)
    out["signal"] = adjusted_signal
    out["macro_overlay"] = {
        "phase": phase,
        "confidence": macro_state.get("confidence", 0.0),
        "bias": round(bias, 2),
        "thresholds": thresholds,
    }
    return out
