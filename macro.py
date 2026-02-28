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


def _build_macro_response(
    phase: str,
    confidence: float,
    leading_score: float,
    coincident_score: float,
    imminent_score: float,
    leading_votes: dict,
    coincident_votes: dict,
    imminent_votes: dict,
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
    return {
        "phase": phase,
        "confidence": confidence,
        "leading_score": round(leading_score, 2),
        "coincident_score": round(coincident_score, 2),
        "imminent_score": round(imminent_score, 2),
        "leading_votes": leading_votes,
        "coincident_votes": coincident_votes,
        "imminent_votes": imminent_votes,
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
    tickers = ["^TNX", "^IRX", "HG=F", "GC=F", "XHB", "ITB", "XLY", "XLP", "XLI", "XLU", "IWM", "^GSPC", "^VIX", "HYG", "LQD"]
    b = _fetch_yahoo_close_bundle(tickers)

    spread = _latest(b.get("^TNX", pd.Series(dtype=float)), asof) - _latest(b.get("^IRX", pd.Series(dtype=float)), asof)
    cgr = _slice_asof(b.get("HG=F", pd.Series(dtype=float)), asof) / _slice_asof(b.get("GC=F", pd.Series(dtype=float)), asof)
    cgr_3m = _pct_change(cgr, 63, asof)
    housing_6m = _pct_change(b.get("XHB", pd.Series(dtype=float)), 126, asof)
    permits_6m = _pct_change(b.get("ITB", pd.Series(dtype=float)), 126, asof)
    xly_xlp = _slice_asof(b.get("XLY", pd.Series(dtype=float)), asof) / _slice_asof(b.get("XLP", pd.Series(dtype=float)), asof)
    sentiment_3m = _pct_change(xly_xlp, 63, asof)

    xli_xlu = _slice_asof(b.get("XLI", pd.Series(dtype=float)), asof) / _slice_asof(b.get("XLU", pd.Series(dtype=float)), asof)
    iwm_spx = _slice_asof(b.get("IWM", pd.Series(dtype=float)), asof) / _slice_asof(b.get("^GSPC", pd.Series(dtype=float)), asof)
    xli_xlu_3m = _pct_change(xli_xlu, 63, asof)
    iwm_spx_3m = _pct_change(iwm_spx, 63, asof)

    vix = _latest(b.get("^VIX", pd.Series(dtype=float)), asof)
    hyg_lqd = _slice_asof(b.get("HYG", pd.Series(dtype=float)), asof) / _slice_asof(b.get("LQD", pd.Series(dtype=float)), asof)
    hyg_lqd_3m = _pct_change(hyg_lqd, 63, asof)
    hyg_lqd_1m = _pct_change(hyg_lqd, 21, asof)

    leading_votes = {
        "Yield curve proxy (^TNX-^IRX)": 1 if spread > 0.2 else (-1 if spread < -0.2 else 0),
        "Building permits proxy ITB (6m)": 1 if permits_6m > 3.0 else (-1 if permits_6m < -3.0 else 0),
        "Housing starts proxy XHB (6m)": 1 if housing_6m > 3.0 else (-1 if housing_6m < -3.0 else 0),
        "Consumer sentiment proxy XLY/XLP (3m)": 1 if sentiment_3m > 1.5 else (-1 if sentiment_3m < -1.5 else 0),
        "Credit spread proxy HYG/LQD (3m)": 1 if hyg_lqd_3m > 1.0 else (-1 if hyg_lqd_3m < -1.0 else 0),
        "Financial stress proxy VIX": 1 if vix <= 16 else (-1 if vix >= 25 else 0),
    }
    coincident_votes = {
        "XLI/XLU ratio (3m)": 1 if xli_xlu_3m > 1.5 else (-1 if xli_xlu_3m < -1.5 else 0),
        "IWM/SPX ratio (3m)": 1 if iwm_spx_3m > 1.0 else (-1 if iwm_spx_3m < -1.0 else 0),
    }
    imminent_votes = {
        "VIX spike risk": -1 if vix >= 30 else (1 if vix <= 14 else 0),
        "Credit shock proxy HYG/LQD (1m)": 1 if hyg_lqd_1m > 1.5 else (-1 if hyg_lqd_1m < -1.5 else 0),
    }

    leading_score = float(np.mean(list(leading_votes.values()))) if leading_votes else 0.0
    coincident_score = float(np.mean(list(coincident_votes.values()))) if coincident_votes else 0.0
    imminent_score = float(np.mean(list(imminent_votes.values()))) if imminent_votes else 0.0
    phase = _phase_from_scores(leading_score, coincident_score, imminent_score)
    confidence = round(min(1.0, (abs(leading_score) + abs(coincident_score) + abs(imminent_score)) / 2.5), 2)

    metrics = {
        "yield_spread_10y2y": spread,
        "yield_spread_10y3m": spread,
        "building_permits_6m_change": permits_6m,
        "housing_6m_change": housing_6m,
        "factory_orders_6m_change": np.nan,
        "consumer_sentiment_6m_change": sentiment_3m,
        "credit_spread_3m_change": hyg_lqd_3m,
        "financial_stress_level": vix,
        "payroll_6m_change": np.nan,
        "indpro_6m_change": np.nan,
        "claims_13w_52w_ratio": np.nan,
        "sahm_like": np.nan,
        "cpi_yoy": np.nan,
        "fed_funds": np.nan,
        "vix": vix,
        "copper_gold_3m": cgr_3m,
        "xly_xlp_3m": sentiment_3m,
        "xli_xlu_3m": xli_xlu_3m,
        "iwm_spx_3m": iwm_spx_3m,
        "hyg_lqd_3m": hyg_lqd_3m,
    }
    return _build_macro_response(
        phase,
        confidence,
        leading_score,
        coincident_score,
        imminent_score,
        leading_votes,
        coincident_votes,
        imminent_votes,
        metrics,
        "yahoo_proxy",
    )


def get_macro_framework_state(asof=None, series_bundle: dict[str, pd.Series] | None = None) -> dict:
    b = series_bundle or fetch_macro_series_bundle()
    if not any((s is not None and len(s) > 0) for s in b.values()):
        return _get_yahoo_proxy_state(asof)

    spread = _latest(b.get("yield_spread_10y2y", pd.Series(dtype=float)), asof)
    spread_10y3m = _latest(b.get("yield_spread_10y3m", pd.Series(dtype=float)), asof)
    permits_chg_6m = _pct_change(b.get("building_permits", pd.Series(dtype=float)), 6, asof)
    housing_chg_6m = _pct_change(b.get("housing_starts", pd.Series(dtype=float)), 6, asof)
    orders_chg_6m = _pct_change(b.get("factory_orders", pd.Series(dtype=float)), 6, asof)
    sentiment_chg_6m = _pct_change(b.get("consumer_sentiment", pd.Series(dtype=float)), 6, asof)
    credit_spread = _latest(b.get("credit_spread", pd.Series(dtype=float)), asof)
    credit_spread_3m = _pct_change(b.get("credit_spread", pd.Series(dtype=float)), 3, asof)
    stl_fsi = _latest(b.get("financial_stress", pd.Series(dtype=float)), asof)

    payroll_chg_6m = _pct_change(b.get("payrolls", pd.Series(dtype=float)), 6, asof)
    indpro_chg_6m = _pct_change(b.get("industrial_prod", pd.Series(dtype=float)), 6, asof)

    claims_13w = _mean_tail(b.get("initial_claims", pd.Series(dtype=float)), 13, asof)
    claims_52w = _mean_tail(b.get("initial_claims", pd.Series(dtype=float)), 52, asof)
    claims_ratio = claims_13w / claims_52w if claims_52w and claims_52w == claims_52w else np.nan

    unrate = _slice_asof(b.get("unemployment", pd.Series(dtype=float)), asof)
    unrate_3m = float(unrate.tail(3).mean()) if len(unrate) >= 3 else np.nan
    unrate_12m_min = float(unrate.tail(12).min()) if len(unrate) >= 12 else np.nan
    sahm = (unrate_3m - unrate_12m_min) if (unrate_3m == unrate_3m and unrate_12m_min == unrate_12m_min) else np.nan

    cpi_yoy = _pct_change(b.get("cpi", pd.Series(dtype=float)), 12, asof)
    fed_funds = _latest(b.get("fed_funds", pd.Series(dtype=float)), asof)

    def v_leading_spread(x):
        if np.isnan(x):
            return 0
        if x > 0.1:
            return 1
        if x < -0.1:
            return -1
        return 0

    def v_leading_permits(x):
        if np.isnan(x):
            return 0
        if x > 2.0:
            return 1
        if x < -5.0:
            return -1
        return 0

    def v_leading_housing(x):
        if np.isnan(x):
            return 0
        if x > 2.0:
            return 1
        if x < -5.0:
            return -1
        return 0

    def v_leading_orders(x):
        if np.isnan(x):
            return 0
        if x > 1.0:
            return 1
        if x < -1.0:
            return -1
        return 0

    def v_leading_sentiment(x):
        if np.isnan(x):
            return 0
        if x > 1.0:
            return 1
        if x < -5.0:
            return -1
        return 0

    def v_leading_credit(x):
        if np.isnan(x):
            return 0
        if x <= -0.15:
            return 1
        if x >= 0.15:
            return -1
        return 0

    def v_leading_stress(x):
        if np.isnan(x):
            return 0
        if x <= 0.0:
            return 1
        if x >= 0.5:
            return -1
        return 0

    def v_coincident_payrolls(x):
        if np.isnan(x):
            return 0
        if x > 0.8:
            return 1
        if x < 0.0:
            return -1
        return 0

    def v_coincident_indpro(x):
        if np.isnan(x):
            return 0
        if x > 1.0:
            return 1
        if x < -1.0:
            return -1
        return 0

    def v_imminent_claims(x):
        if np.isnan(x):
            return 0
        if x >= 1.08:
            return -1
        if x <= 0.95:
            return 1
        return 0

    def v_imminent_sahm(x):
        if np.isnan(x):
            return 0
        if x >= 0.5:
            return -1
        if x <= 0.2:
            return 1
        return 0

    leading_votes = {
        "Yield curve (10Y-3M)": v_leading_spread(spread_10y3m),
        "Building permits (6m)": v_leading_permits(permits_chg_6m),
        "Housing starts (6m)": v_leading_housing(housing_chg_6m),
        "Consumer sentiment (6m)": v_leading_sentiment(sentiment_chg_6m),
        "Credit spread (3m change)": v_leading_credit(credit_spread_3m),
        "St. Louis FSI": v_leading_stress(stl_fsi),
    }
    coincident_votes = {
        "Payrolls (6m)": v_coincident_payrolls(payroll_chg_6m),
        "Industrial production (6m)": v_coincident_indpro(indpro_chg_6m),
    }
    imminent_votes = {
        "Initial claims stress": v_imminent_claims(claims_ratio),
        "Sahm-style trigger": v_imminent_sahm(sahm),
    }

    leading_score = float(np.mean(list(leading_votes.values()))) if leading_votes else 0.0
    coincident_score = float(np.mean(list(coincident_votes.values()))) if coincident_votes else 0.0
    imminent_score = float(np.mean(list(imminent_votes.values()))) if imminent_votes else 0.0

    phase = _phase_from_scores(leading_score, coincident_score, imminent_score)

    confidence = round(min(1.0, (abs(leading_score) + abs(coincident_score) + abs(imminent_score)) / 2.5), 2)

    return _build_macro_response(
        phase,
        confidence,
        leading_score,
        coincident_score,
        imminent_score,
        leading_votes,
        coincident_votes,
        imminent_votes,
        {
            "yield_spread_10y2y": spread,
            "yield_spread_10y3m": spread_10y3m,
            "building_permits_6m_change": permits_chg_6m,
            "housing_6m_change": housing_chg_6m,
            "factory_orders_6m_change": orders_chg_6m,
            "consumer_sentiment_6m_change": sentiment_chg_6m,
            "credit_spread_level": credit_spread,
            "credit_spread_3m_change": credit_spread_3m,
            "financial_stress_level": stl_fsi,
            "payroll_6m_change": payroll_chg_6m,
            "indpro_6m_change": indpro_chg_6m,
            "claims_13w_52w_ratio": claims_ratio,
            "sahm_like": sahm,
            "cpi_yoy": cpi_yoy,
            "fed_funds": fed_funds,
        },
        "fred",
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
