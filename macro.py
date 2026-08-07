"""
Zeberg Macro Navigation Framework (TM) - composite-index implementation.

Rebuilt to follow the architecture described in Zeberg, "Macro Navigation
Framework", SSRN working paper 6064235:

- Indicators are grouped into leading / coincident / lagging composites and
  standardized, rather than voted on individually.
- Each composite is evaluated against a long-term growth equilibrium (zero in
  standardized-growth space), not against its own trailing moving average.
- Phase is a persistent state advanced by composite crossings, in sequence:
  leading rolls over first, coincident confirms, lagging adjusts last.
- Imminent Recession Indicators are phase-conditional accelerators, active only
  once leading has rolled over, and signal through clustering rather than
  individually.

Section 9.1 of the paper withholds the weighting schemes, transformations, and
smoothing parameters as proprietary. The choices below are therefore ours and
are marked OURS; the architecture above is the paper's.
"""

import numpy as np
import pandas as pd
import yfinance as yf

FRED_BASE = "https://fred.stlouisfed.org/graph/fredgraph.csv?id="

FRED_SERIES = {
    # Leading - sensitive to financing costs and credit conditions
    "yield_spread_10y3m": "T10Y3M",
    "building_permits": "PERMIT",
    "housing_starts": "HOUST",
    "new_orders": "NEWORDER",
    "consumer_sentiment": "UMCSENT",
    "credit_spread": "BAA10YM",
    "financial_stress": "STLFSI4",
    # Coincident - the economy itself (the NBER four)
    "payrolls": "PAYEMS",
    "industrial_prod": "INDPRO",
    "real_personal_income": "W875RX1",
    "real_sales": "CMRMTSPL",
    # Lagging - adjust only after the real economy has turned
    "cpi": "CPIAUCSL",
    "fed_funds": "FEDFUNDS",
    # Imminent-recession inputs
    "tbill_3m": "TB3MS",
    "initial_claims": "ICSA",
    "unemployment": "UNRATE",
    # Retained for the metrics panel
    "yield_spread_10y2y": "T10Y2Y",
    "factory_orders": "AMTMNO",
}

# OURS: composite construction parameters. The paper withholds its own
# (section 9.1), so these were chosen by sweeping smoothing and dead-band
# against NBER dating back to 1960.
ZSCORE_MIN_PERIODS = 60   # 5y burn-in before a standardized value is trusted
COMPOSITE_SMOOTH = 3      # months of smoothing on each composite
COINCIDENT_ANNUAL = 6     # months over which coincident growth is annualized
PHASE_BAND = 0.25         # dead zone around equilibrium, in each index's units
CONFIDENCE_SCALE = 1.5    # leading z at which confidence saturates
CONFIDENCE_SCALE_COI = 3.0  # coincident growth (pp) at which confidence saturates


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


def _fetch_yahoo_close_bundle(tickers: list[str], period: str = "10y") -> dict[str, pd.Series]:
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


# ── Composite construction ────────────────────────────────────────────────────

def _to_monthly(s: pd.Series) -> pd.Series:
    """Collapse any frequency onto a month-start grid."""
    if s is None or s.empty:
        return pd.Series(dtype=float)
    return s.resample("MS").mean().dropna()


def _momentum(s: pd.Series, kind: str) -> pd.Series:
    """Express a series as the momentum measure its tier is evaluated on.

    "yoy"        - 12-month percent change, for activity levels that trend.
    "annualized" - COINCIDENT_ANNUAL-month change at an annual rate. Turns far
                   closer to the cycle peak than a 12-month change, which is
                   what lets the coincident crossing land near recession onset.
    "level"      - the series as published, for spreads and stress indices that
                   are already expressed relative to their own scale.
    """
    m = _to_monthly(s)
    if m.empty:
        return m
    if kind == "yoy":
        return (m.pct_change(12) * 100).dropna()
    if kind == "annualized":
        n = COINCIDENT_ANNUAL
        return (((m / m.shift(n)) ** (12.0 / n) - 1.0) * 100).dropna()
    return m


def _growth(bundle: dict, key: str) -> pd.Series:
    """Coincident components stay in growth units - see _build_tiers."""
    return _momentum(bundle.get(key, pd.Series(dtype=float)), "annualized")


def _zscore_expanding(s: pd.Series, min_periods: int = ZSCORE_MIN_PERIODS) -> pd.Series:
    """Standardize causally.

    Uses only history available at each point, so a backtest cannot borrow the
    mean and variance of its own future.
    """
    if s is None or s.empty:
        return pd.Series(dtype=float)
    mean = s.expanding(min_periods=min_periods).mean()
    std = s.expanding(min_periods=min_periods).std()
    return ((s - mean) / std.replace(0.0, np.nan)).dropna()


def _standardized(bundle: dict, key: str, kind: str, invert: bool = False) -> pd.Series:
    z = _zscore_expanding(_momentum(bundle.get(key, pd.Series(dtype=float)), kind))
    return -z if invert else z


def _composite(parts: dict[str, pd.Series], smooth: int = COMPOSITE_SMOOTH) -> pd.Series:
    """Average the standardized components onto one index.

    Equal weights - the paper does not disclose its weighting scheme, and equal
    weighting is the choice that adds no unstated assumptions of our own.
    """
    live = {k: v for k, v in parts.items() if v is not None and not v.empty}
    if not live:
        return pd.Series(dtype=float)
    df = pd.concat(live.values(), axis=1)
    c = df.mean(axis=1, skipna=True)
    if smooth > 1:
        c = c.rolling(smooth, min_periods=1).mean()
    return c.dropna()


def _build_tiers(b: dict) -> dict[str, dict[str, pd.Series]]:
    """Components of each tier, keyed by display name.

    The two tiers are measured against different equilibria, because the paper
    defines their crossings differently. Leading indicators mark Slowdown when
    momentum falls below its long-term trend, so they are standardized and read
    against zero. Coincident indicators mark Contraction when activity "falls
    below the long-term growth path and turns negative", so they stay in growth
    units and are read against zero growth. Standardizing the coincident tier
    instead would put its equilibrium at the historical mean, which is crossed
    roughly half the time - the arbitrary statistical threshold the paper
    explicitly rejects.
    """
    leading = {
        "Yield curve (10Y-3M)": _standardized(b, "yield_spread_10y3m", "level"),
        "Building permits": _standardized(b, "building_permits", "yoy"),
        "Housing starts": _standardized(b, "housing_starts", "yoy"),
        "Core capital goods orders": _standardized(b, "new_orders", "yoy"),
        "Consumer sentiment": _standardized(b, "consumer_sentiment", "yoy"),
        "Credit spreads (BAA-10Y)": _standardized(b, "credit_spread", "level", invert=True),
        "St. Louis FSI": _standardized(b, "financial_stress", "level", invert=True),
    }
    coincident = {
        "Nonfarm payrolls": _growth(b, "payrolls"),
        "Industrial production": _growth(b, "industrial_prod"),
        "Real personal income": _growth(b, "real_personal_income"),
        "Real manufacturing & trade sales": _growth(b, "real_sales"),
    }
    lagging = {
        "CPI inflation": _standardized(b, "cpi", "yoy"),
        "Fed funds rate": _standardized(b, "fed_funds", "level"),
    }
    return {"leading": leading, "coincident": coincident, "lagging": lagging}


# ── Phase state machine ───────────────────────────────────────────────────────

def _seed_phase(lei: float, coi: float) -> str:
    if coi < 0:
        return "Contraction"
    return "Slowdown" if lei < 0 else "Expansion"


def _trend_growth(coi: pd.Series) -> pd.Series:
    """The long-term growth rate, estimated causally from history to date."""
    return coi.expanding(min_periods=ZSCORE_MIN_PERIODS).mean()


def _run_phase_machine(lei: pd.Series, coi: pd.Series, band: float = PHASE_BAND) -> pd.Series:
    """Advance the phase along the paper's sequence.

    The two coincident thresholds are the paper's, not a simplification of it.
    Expansion begins when the economy accelerates above its long-term growth
    rate; contraction begins when activity falls below the growth path and
    turns negative. Those are different lines, so entering Expansion is tested
    against trend growth while entering Contraction is tested against zero.

    Expansion   -> Slowdown    leading falls below trend
    Slowdown    -> Contraction coincident growth turns negative
    Slowdown    -> Expansion   leading recovers and growth is above trend
    Contraction -> Recovery    leading turns up from depressed levels
    Recovery    -> Expansion   growth accelerates back above trend
    Recovery    -> Contraction leading rolls back over

    OURS: `band` is a symmetric dead zone around each line, in that index's own
    units, so a composite grazing the line does not flip the regime repeatedly.
    """
    idx = lei.index.union(coi.index).sort_values()
    l_s = lei.reindex(idx).ffill()
    c_s = coi.reindex(idx).ffill()
    t_s = _trend_growth(c_s)

    phases, state = [], None
    for t in idx:
        l, c, trend = l_s.loc[t], c_s.loc[t], t_s.loc[t]
        if pd.isna(l) or pd.isna(c):
            phases.append("No signal")
            continue
        # Before enough history accumulates to estimate trend growth, fall back
        # to the zero line so the machine still advances.
        trend = 0.0 if pd.isna(trend) else float(trend)
        accelerating = c > trend + band

        if state is None:
            state = _seed_phase(l, c)
        elif state == "Expansion":
            if l < -band:
                state = "Slowdown"
        elif state == "Slowdown":
            if c < -band:
                state = "Contraction"
            elif l > band and accelerating:
                state = "Expansion"
        elif state == "Contraction":
            if l > band:
                state = "Recovery"
        elif state == "Recovery":
            if l < -band:
                state = "Contraction"
            elif accelerating:
                state = "Expansion"
        phases.append(state)
    return pd.Series(phases, index=idx, dtype=object)


# ── Imminent Recession Indicators (phase-conditional) ─────────────────────────

def _imminent_components(b: dict, asof=None) -> list[dict]:
    """The three accelerators the paper names.

    Evaluated only in Slowdown, and meaningful through clustering rather than
    individually.
    """
    rows = []

    # Rapid decline in short rates as markets price easing.
    tb = _to_monthly(_slice_asof(b.get("tbill_3m", pd.Series(dtype=float)), asof))
    drop = float(tb.iloc[-1] - tb.iloc[-4]) if len(tb) >= 4 else np.nan
    rows.append({
        "name": "Short-rate collapse",
        "fired": bool(not np.isnan(drop) and drop <= -0.50),
        "detail": "Data unavailable" if np.isnan(drop) else f"3M bill {drop:+.2f}pp over 3 months (fires <= -0.50)",
    })

    # Re-steepening of a previously inverted curve.
    sp = _to_monthly(_slice_asof(b.get("yield_spread_10y3m", pd.Series(dtype=float)), asof))
    if len(sp) >= 24:
        window = sp.tail(24)
        was_inverted = bool((window < 0).any())
        trough = float(window.min())
        steepening = float(sp.iloc[-1]) - trough
        rows.append({
            "name": "Curve re-steepening",
            "fired": bool(was_inverted and steepening >= 0.50),
            "detail": (f"{steepening:+.2f}pp off a {trough:.2f} trough"
                       if was_inverted else "No inversion in prior 24 months"),
        })
    else:
        rows.append({"name": "Curve re-steepening", "fired": False, "detail": "Data unavailable"})

    # Abrupt increase in initial claims.
    cl = _slice_asof(b.get("initial_claims", pd.Series(dtype=float)), asof)
    if len(cl) >= 52:
        ratio = float(cl.tail(13).mean() / cl.tail(52).mean())
        rows.append({
            "name": "Initial claims acceleration",
            "fired": bool(ratio >= 1.08),
            "detail": f"13w/52w ratio {ratio:.2f} (fires >= 1.08)",
        })
    else:
        rows.append({"name": "Initial claims acceleration", "fired": False, "detail": "Data unavailable"})

    return rows


def _sahm_value(unrate: pd.Series, asof=None) -> float:
    v = _slice_asof(unrate, asof)
    if len(v) < 12:
        return np.nan
    return float(v.tail(3).mean() - v.tail(12).min())


# ── Presentation ──────────────────────────────────────────────────────────────

def _tier_rows(components: dict[str, pd.Series], asof=None, unit: str = "z") -> list[dict]:
    """One row per component: its position relative to the tier's equilibrium."""
    rows = []
    for name, series in components.items():
        v = _slice_asof(series, asof)
        if v.empty:
            rows.append({"name": name, "vote": 0, "status": "No data",
                         "detail": "Data unavailable", "value": np.nan})
            continue
        cur = float(v.iloc[-1])
        prev = float(v.iloc[-2]) if len(v) > 1 else cur
        crossed = (prev <= 0 < cur) or (prev >= 0 > cur)
        above = cur > 0
        status = ("Crossed above" if above else "Crossed below") if crossed else \
                 ("Above equilibrium" if above else "Below equilibrium")
        if unit == "pp":
            detail = f"{cur:+.2f}% annualized vs 0% equilibrium"
        else:
            detail = f"z {cur:+.2f} vs long-term trend"
        if crossed:
            detail += f" (from {prev:+.2f})"
        rows.append({"name": name, "vote": 1 if above else -1,
                     "status": status, "detail": detail, "value": cur})
    return rows


def _confidence(phase: str, lei: float, coi: float) -> float:
    """Distance of the composite that governs this phase from its equilibrium."""
    if phase == "Contraction":
        driver, scale = coi, CONFIDENCE_SCALE_COI
    else:
        driver, scale = lei, CONFIDENCE_SCALE
    if driver is None or np.isnan(driver):
        return 0.0
    return round(min(1.0, abs(float(driver)) / scale), 2)


PHASE_BIAS = {
    "Expansion": {"gold": -0.04, "silver": 0.04},
    "Slowdown": {"gold": 0.03, "silver": -0.01},
    "Contraction": {"gold": 0.08, "silver": -0.06},
    "Recovery": {"gold": 0.00, "silver": 0.05},
}

ADAPTIVE_THRESHOLDS = {
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


def _build_macro_response(phase, confidence, leading_score, coincident_score, lagging_score,
                          imminent_score, leading_indicators, coincident_indicators,
                          lagging_indicators, imminent_indicators, metrics, source,
                          phase_history=None) -> dict:
    return {
        "phase": phase,
        "confidence": confidence,
        "leading_score": round(float(leading_score), 2),
        "coincident_score": round(float(coincident_score), 2),
        "lagging_score": round(float(lagging_score), 2),
        "imminent_score": round(float(imminent_score), 2),
        "leading_votes": {i["name"]: i["vote"] for i in leading_indicators},
        "coincident_votes": {i["name"]: i["vote"] for i in coincident_indicators},
        "imminent_votes": {i["name"]: i["vote"] for i in imminent_indicators},
        "leading_indicators": leading_indicators,
        "coincident_indicators": coincident_indicators,
        "lagging_indicators": lagging_indicators,
        "imminent_indicators": imminent_indicators,
        "source": source,
        "metrics": metrics,
        "phase_bias": PHASE_BIAS,
        "adaptive_thresholds": ADAPTIVE_THRESHOLDS,
        "phase_history": phase_history,
    }


# ── Yahoo fallback ────────────────────────────────────────────────────────────

def _get_yahoo_proxy_state(asof=None) -> dict:
    """Degraded path for when FRED is unreachable.

    Same state machine, ETF proxies in place of the official series.
    """
    tickers = ["^TNX", "^IRX", "XHB", "ITB", "XLI", "IWM", "^GSPC", "HYG", "LQD"]
    b = _fetch_yahoo_close_bundle(tickers)

    spread = _slice_asof(b.get("^TNX", pd.Series(dtype=float)), asof) - _slice_asof(b.get("^IRX", pd.Series(dtype=float)), asof)
    proxy = {
        "yield_spread_10y3m": spread.dropna(),
        "building_permits": _slice_asof(b.get("ITB", pd.Series(dtype=float)), asof),
        "housing_starts": _slice_asof(b.get("XHB", pd.Series(dtype=float)), asof),
        "industrial_prod": _slice_asof(b.get("XLI", pd.Series(dtype=float)), asof),
        "real_sales": _slice_asof(b.get("IWM", pd.Series(dtype=float)), asof),
    }
    leading = {
        "Yield curve proxy (10Y-3M)": _standardized(proxy, "yield_spread_10y3m", "level"),
        "Homebuilders (ITB)": _standardized(proxy, "building_permits", "yoy"),
        "Housing proxy (XHB)": _standardized(proxy, "housing_starts", "yoy"),
    }
    coincident = {
        "Industrials (XLI)": _standardized(proxy, "industrial_prod", "yoy"),
        "Small caps (IWM)": _standardized(proxy, "real_sales", "yoy"),
    }

    lei, coi = _composite(leading), _composite(coincident)
    if lei.empty or coi.empty:
        return _build_macro_response("No signal", 0.0, 0.0, 0.0, 0.0, 0.0, [], [], [], [],
                                     {}, "unavailable")

    hist = _run_phase_machine(lei, coi)
    phase = str(hist.iloc[-1])
    l_now, c_now = float(lei.iloc[-1]), float(coi.iloc[-1])
    return _build_macro_response(
        phase, _confidence(phase, l_now, c_now), l_now, c_now, 0.0, 0.0,
        _tier_rows(leading, asof), _tier_rows(coincident, asof), [], [],
        {"leading_composite": l_now, "coincident_composite": c_now},
        "yahoo_proxy", hist,
    )


# ── Public entry point ────────────────────────────────────────────────────────

def get_macro_framework_state(asof=None, series_bundle: dict[str, pd.Series] | None = None) -> dict:
    """Classify the business-cycle phase per the Zeberg framework."""
    b = series_bundle or fetch_macro_series_bundle()
    if not any((s is not None and len(s) > 0) for s in b.values()):
        return _get_yahoo_proxy_state(asof)

    sliced = {k: _slice_asof(v, asof) for k, v in b.items()}
    tiers = _build_tiers(sliced)
    lei = _composite(tiers["leading"])
    coi = _composite(tiers["coincident"])
    lag = _composite(tiers["lagging"])

    if lei.empty or coi.empty:
        return _get_yahoo_proxy_state(asof)

    history = _run_phase_machine(lei, coi)
    phase = str(history.iloc[-1])
    l_now = float(lei.iloc[-1])
    c_now = float(coi.iloc[-1])
    g_now = float(lag.iloc[-1]) if not lag.empty else 0.0

    # Imminent indicators are accelerators within Slowdown, not an always-on
    # tier. Outside that phase the paper does not evaluate them.
    imminent_rows = []
    imminent_score = 0.0
    if phase == "Slowdown":
        comps = _imminent_components(sliced, asof)
        fired = sum(1 for c in comps if c["fired"])
        clustered = fired >= 2  # clustering, not isolated signals
        for c in comps:
            imminent_rows.append({
                "name": c["name"],
                "vote": -1 if c["fired"] else 0,
                "status": "Firing" if c["fired"] else ("No data" if c["detail"] == "Data unavailable" else "Quiet"),
                "detail": c["detail"],
            })
        imminent_rows.append({
            "name": "Cluster assessment",
            "vote": -1 if clustered else 0,
            "status": "Recession imminent" if clustered else "Not clustered",
            "detail": f"{fired}/3 accelerators firing (clusters at 2)",
        })
        imminent_score = -float(fired) / 3.0
    else:
        imminent_rows.append({
            "name": "Imminent indicators",
            "vote": 0,
            "status": "Not evaluated",
            "detail": f"Phase-conditional: active only in Slowdown (now {phase})",
        })

    metrics = {
        "leading_composite": round(l_now, 3),
        "coincident_composite": round(c_now, 3),
        "lagging_composite": round(g_now, 3),
        "yield_spread_10y3m": _latest(b.get("yield_spread_10y3m"), asof),
        "yield_spread_10y2y": _latest(b.get("yield_spread_10y2y"), asof),
        "sahm_value": _sahm_value(b.get("unemployment", pd.Series(dtype=float)), asof),
        "unemployment": _latest(b.get("unemployment"), asof),
        "credit_spread_level": _latest(b.get("credit_spread"), asof),
        "financial_stress_level": _latest(b.get("financial_stress"), asof),
        "payroll_6m_change": _pct_change(b.get("payrolls", pd.Series(dtype=float)), 6, asof),
        "indpro_6m_change": _pct_change(b.get("industrial_prod", pd.Series(dtype=float)), 6, asof),
        "cpi_yoy": _pct_change(b.get("cpi", pd.Series(dtype=float)), 12, asof),
        "fed_funds": _latest(b.get("fed_funds"), asof),
    }

    return _build_macro_response(
        phase, _confidence(phase, l_now, c_now), l_now, c_now, g_now, imminent_score,
        _tier_rows(tiers["leading"], asof), _tier_rows(tiers["coincident"], asof, unit="pp"),
        _tier_rows(tiers["lagging"], asof), imminent_rows,
        metrics, "fred", history,
    )


def apply_macro_overlay(score: dict, macro_state: dict, metal: str) -> dict:
    if not score or not macro_state:
        return score
    phase = macro_state.get("phase")
    if phase not in PHASE_BIAS:
        return score

    metal_key = "gold" if str(metal).lower().startswith("g") else "silver"
    bias = float(PHASE_BIAS.get(phase, {}).get(metal_key, 0.0))
    thresholds = (
        ADAPTIVE_THRESHOLDS.get(metal_key, {})
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
