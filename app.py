import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date

from data import (
    fetch_spot_prices, fetch_etc_prices, fetch_historical,
    fetch_multi_timeframe_data, compute_tracking_difference, DEFAULT_ETC_TICKERS,
    SPOT_TICKERS,
)
from technicals import (
    multi_timeframe_fibonacci, vobhs_composite, detect_triangles,
    bollinger_squeeze, momentum_bars, whale_volume_detection,
    rsi, sma_crossover, fibonacci_levels,
)
from signals import score_metal, allocation_recommendation, INDICATORS, _rescale_indicators, DEFAULT_TF_WEIGHTS
from charts import render_metal_chart
from portfolio import (
    get_portfolio, set_total_pot, add_purchase, delete_purchase, get_summary,
    export_portfolio_json, import_portfolio_json, _github_config,
    get_config, set_config,
)
from news import fetch_catalyst_news
from analysis import generate_summary, build_context_prompt
import chat
from macro import get_macro_framework_state, apply_macro_overlay
from style import (
    GLOBAL_CSS, GOLD, SILVER, GREEN, RED, AMBER, TEXT_MUTED,
    ticker_strip_html, signal_card_html, etc_tile_html,
    news_card_html, port_stat_html, render_component,
    analysis_card_html,
)
from simulator import run_simulations, clear_simulator_caches

st.set_page_config(
    page_title="Gold & Silver Strategy Monitor",
    page_icon="🥇",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject global CSS
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

# ========================= SIDEBAR CONFIG =========================
st.sidebar.title("Settings")

available_etcs = list(DEFAULT_ETC_TICKERS.keys())
WEIGHT_PRESETS = {
    "Short-bias": {"Short": 80, "Medium": 15, "Long": 5},
    "Balanced": {"Short": 34, "Medium": 33, "Long": 33},
    "Long-bias": {"Short": 20, "Medium": 30, "Long": 50},
}


def _normalise_weights(weights: dict | None) -> dict:
    short = int((weights or {}).get("Short", DEFAULT_TF_WEIGHTS["Short"]))
    medium = int((weights or {}).get("Medium", DEFAULT_TF_WEIGHTS["Medium"]))
    long = int((weights or {}).get("Long", DEFAULT_TF_WEIGHTS["Long"]))
    short, medium, long = max(0, short), max(0, medium), max(0, long)
    total = short + medium + long
    if total == 100:
        return {"Short": short, "Medium": medium, "Long": long}
    if total <= 0:
        return DEFAULT_TF_WEIGHTS.copy()
    scaled = {
        "Short": int(round(short * 100 / total)),
        "Medium": int(round(medium * 100 / total)),
    }
    scaled["Long"] = 100 - scaled["Short"] - scaled["Medium"]
    return scaled


def _normalise_selected_etcs(selected: list[str] | None) -> list[str]:
    valid = [tk for tk in (selected or []) if tk in available_etcs]
    return valid if valid else available_etcs[:2]


def _normalise_config(config: dict | None) -> dict:
    cfg = config or {}
    fib_tolerance = float(cfg.get("fib_tolerance", 2.0))
    fib_tolerance = min(5.0, max(0.5, round(fib_tolerance * 2) / 2))
    return {
        "selected_etcs": _normalise_selected_etcs(cfg.get("selected_etcs")),
        "fib_tolerance": fib_tolerance,
        "gs_ratio_threshold": float(cfg.get("gs_ratio_threshold", 63.0)),
        "rsi_period": max(2, int(cfg.get("rsi_period", 14))),
        "ema_fast": max(1, int(cfg.get("ema_fast", 9))),
        "ema_slow": max(1, int(cfg.get("ema_slow", 21))),
        "sma_fast": max(1, int(cfg.get("sma_fast", 20))),
        "sma_slow": max(1, int(cfg.get("sma_slow", 50))),
        "whale_vol_threshold": max(0.1, float(cfg.get("whale_vol_threshold", 2.0))),
        "macro_overlay_enabled": bool(cfg.get("macro_overlay_enabled", True)),
        "tf_weights": _normalise_weights(cfg.get("tf_weights", DEFAULT_TF_WEIGHTS)),
        "profiles": cfg.get("profiles", {}) if isinstance(cfg.get("profiles", {}), dict) else {},
    }


def _version_epoch(version: str) -> float:
    if not version:
        return float("-inf")
    try:
        return datetime.fromisoformat(version.replace("Z", "+00:00")).timestamp()
    except Exception:
        return float("-inf")


def _set_state_from_config(config: dict):
    cfg = _normalise_config(config)
    st.session_state["selected_etcs"] = cfg["selected_etcs"]
    st.session_state["fib_tolerance"] = cfg["fib_tolerance"]
    st.session_state["gs_ratio_threshold"] = cfg["gs_ratio_threshold"]
    st.session_state["rsi_period"] = cfg["rsi_period"]
    st.session_state["ema_fast"] = cfg["ema_fast"]
    st.session_state["ema_slow"] = cfg["ema_slow"]
    st.session_state["sma_fast"] = cfg["sma_fast"]
    st.session_state["sma_slow"] = cfg["sma_slow"]
    st.session_state["whale_vol_threshold"] = cfg["whale_vol_threshold"]
    st.session_state["macro_overlay_enabled"] = cfg["macro_overlay_enabled"]
    st.session_state["profiles"] = cfg.get("profiles", {})

    w = cfg["tf_weights"]
    st.session_state["w_short"] = w["Short"]
    st.session_state["w_medium"] = w["Medium"]
    st.session_state["w_long"] = w["Long"]


def _state_weights() -> dict:
    return {
        "Short": int(st.session_state["w_short"]),
        "Medium": int(st.session_state["w_medium"]),
        "Long": int(st.session_state["w_long"]),
    }


def _state_config() -> dict:
    return _normalise_config({
        "selected_etcs": st.session_state.get("selected_etcs", []),
        "fib_tolerance": st.session_state.get("fib_tolerance", 2.0),
        "gs_ratio_threshold": st.session_state.get("gs_ratio_threshold", 63.0),
        "rsi_period": st.session_state.get("rsi_period", 14),
        "ema_fast": st.session_state.get("ema_fast", 9),
        "ema_slow": st.session_state.get("ema_slow", 21),
        "sma_fast": st.session_state.get("sma_fast", 20),
        "sma_slow": st.session_state.get("sma_slow", 50),
        "whale_vol_threshold": st.session_state.get("whale_vol_threshold", 2.0),
        "macro_overlay_enabled": st.session_state.get("macro_overlay_enabled", True),
        "tf_weights": _state_weights() if all(k in st.session_state for k in ("w_short", "w_medium", "w_long")) else DEFAULT_TF_WEIGHTS,
        "profiles": st.session_state.get("profiles", {}),
    })


def _persist_config():
    cfg = _state_config()
    updated_at = set_config(cfg)
    _set_state_from_config(cfg)
    st.session_state["_config_version"] = updated_at


def _sync_from_shared_store(force: bool = False):
    shared_cfg_raw, shared_version = get_config()
    shared_cfg = _normalise_config(shared_cfg_raw)
    shared_version = shared_version or ""
    local_version = st.session_state.get("_config_version", "")
    if force or not local_version or _version_epoch(shared_version) > _version_epoch(local_version):
        _set_state_from_config(shared_cfg)
        st.session_state["_config_version"] = shared_version


@st.fragment(run_every="10s")
def _watch_shared_config_updates():
    shared_cfg_raw, shared_version = get_config()
    shared_version = shared_version or ""
    local_version = st.session_state.get("_config_version", "")
    if shared_version and _version_epoch(shared_version) > _version_epoch(local_version):
        _set_state_from_config(_normalise_config(shared_cfg_raw))
        st.session_state["_config_version"] = shared_version
        st.rerun()


def _rebalance_keys(keys: list[str], changed_key: str):
    others = [k for k in keys if k != changed_key]
    new_val = st.session_state[changed_key]
    remaining = 100 - new_val
    other_sum = sum(st.session_state[k] for k in others)
    if other_sum > 0:
        for k in others:
            st.session_state[k] = max(0, round(st.session_state[k] / other_sum * remaining))
        diff = 100 - new_val - sum(st.session_state[k] for k in others)
        if diff != 0:
            st.session_state[others[0]] += diff
    else:
        for i, k in enumerate(others):
            st.session_state[k] = remaining // len(others) + (1 if i < remaining % len(others) else 0)


def _rebalance(changed_key):
    _rebalance_keys(["w_short", "w_medium", "w_long"], changed_key)
    weights = _normalise_weights(_state_weights())
    st.session_state["w_short"] = weights["Short"]
    st.session_state["w_medium"] = weights["Medium"]
    st.session_state["w_long"] = weights["Long"]
    _persist_config()


def _apply_weight_preset(name: str):
    preset = WEIGHT_PRESETS.get(name)
    if not preset:
        return
    st.session_state["w_short"] = preset["Short"]
    st.session_state["w_medium"] = preset["Medium"]
    st.session_state["w_long"] = preset["Long"]
    _persist_config()


def _active_profile_payload() -> dict:
    cfg = _state_config()
    return {k: v for k, v in cfg.items() if k != "profiles"}


def _save_profile():
    name = st.session_state.get("profile_name_input", "").strip()
    if not name:
        return
    profiles = dict(st.session_state.get("profiles", {}))
    profiles[name] = _active_profile_payload()
    st.session_state["profiles"] = profiles
    set_config({"profiles": profiles})


def _apply_selected_profile():
    name = st.session_state.get("profile_select", "")
    profile = st.session_state.get("profiles", {}).get(name)
    if not isinstance(profile, dict):
        return
    merged = dict(profile)
    merged["profiles"] = st.session_state.get("profiles", {})
    _set_state_from_config(merged)
    _persist_config()


def _delete_selected_profile():
    name = st.session_state.get("profile_select", "")
    profiles = dict(st.session_state.get("profiles", {}))
    if name in profiles:
        profiles.pop(name)
        st.session_state["profiles"] = profiles
        set_config({"profiles": profiles})


_sync_from_shared_store(force=False)
_watch_shared_config_updates()

selected_etcs = st.sidebar.multiselect(
    "Track ETCs (Hargreaves Lansdown)",
    available_etcs,
    key="selected_etcs",
    on_change=_persist_config,
)

st.sidebar.subheader("Signal Parameters")
fib_tolerance = st.sidebar.slider(
    "Fibonacci proximity tolerance (%)", 0.5, 5.0, 2.0, 0.5,
    key="fib_tolerance", on_change=_persist_config,
)
gs_ratio_threshold = st.sidebar.number_input(
    "G/S Ratio threshold (favour silver above)",
    step=0.5,
    key="gs_ratio_threshold",
    on_change=_persist_config,
)
rsi_period = st.sidebar.number_input("RSI period", step=1, key="rsi_period", on_change=_persist_config)
ema_fast = st.sidebar.number_input("Fast EMA", step=1, key="ema_fast", on_change=_persist_config)
ema_slow = st.sidebar.number_input("Slow EMA", step=1, key="ema_slow", on_change=_persist_config)
sma_fast = st.sidebar.number_input("Fast SMA", step=5, key="sma_fast", on_change=_persist_config)
sma_slow = st.sidebar.number_input("Slow SMA", step=5, key="sma_slow", on_change=_persist_config)
whale_vol_threshold = st.sidebar.number_input(
    "Whale volume threshold (x avg)",
    step=0.5,
    key="whale_vol_threshold",
    on_change=_persist_config,
)

st.sidebar.subheader("Macro Overlay")
st.sidebar.checkbox(
    "Enable macro regime overlay",
    key="macro_overlay_enabled",
    on_change=_persist_config,
    help="Zeburg (TM)-inspired regime filter adjusts final signal thresholds and small phase bias.",
)

st.sidebar.subheader("Timeframe Weights")
st.sidebar.slider("Short-term %", 0, 100, key="w_short", on_change=_rebalance, args=("w_short",))
st.sidebar.slider("Medium-term %", 0, 100, key="w_medium", on_change=_rebalance, args=("w_medium",))
st.sidebar.slider("Long-term %", 0, 100, key="w_long", on_change=_rebalance, args=("w_long",))
st.sidebar.caption("Weight presets")
preset_cols = st.sidebar.columns(3)
for i, preset_name in enumerate(WEIGHT_PRESETS.keys()):
    preset_cols[i].button(
        preset_name,
        key=f"preset_{preset_name}",
        on_click=_apply_weight_preset,
        args=(preset_name,),
    )

st.sidebar.subheader("Saved Profiles")
profile_names = sorted(st.session_state.get("profiles", {}).keys())
st.sidebar.selectbox("Choose profile", [""] + profile_names, key="profile_select")
profile_cols = st.sidebar.columns(2)
profile_cols[0].button("Apply Profile", on_click=_apply_selected_profile)
profile_cols[1].button("Delete Profile", on_click=_delete_selected_profile)
st.sidebar.text_input("Save current settings as", key="profile_name_input")
st.sidebar.button("Save Profile", on_click=_save_profile)

st.sidebar.button("Pull latest shared settings", on_click=_sync_from_shared_store, kwargs={"force": True})

tf_weight_config = _normalise_weights(_state_weights())

st.sidebar.subheader("Portfolio")


def _save_total_pot():
    raw_amount = st.session_state.get("total_pot_cfg")
    if raw_amount is None:
        raw_amount = get_portfolio().get("total_pot", 2_000_000.0)
        st.session_state["total_pot_cfg"] = float(raw_amount)
    amount = float(raw_amount)
    set_total_pot(amount)
    st.session_state["_last_total_pot_synced"] = amount


portfolio_data = get_portfolio()
shared_total_pot = float(portfolio_data.get("total_pot", 2_000_000.0))
if "total_pot_cfg" not in st.session_state or st.session_state.get("_last_total_pot_synced") != shared_total_pot:
    st.session_state["total_pot_cfg"] = shared_total_pot
    st.session_state["_last_total_pot_synced"] = shared_total_pot
st.sidebar.number_input("Total investment pot (GBP)", step=10_000.0, key="total_pot_cfg", on_change=_save_total_pot)


@st.cache_data(show_spinner=False, ttl=3600)
def _macro_state_cached():
    return get_macro_framework_state()


@st.cache_data(show_spinner=False, ttl=300)
def _fetch_spot_cached():
    return fetch_spot_prices()


@st.cache_data(show_spinner=False, ttl=300)
def _fetch_etc_cached(tickers: tuple[str, ...]):
    return fetch_etc_prices(list(tickers)) if tickers else {}


def _get_spot_and_etc(selected):
    spot_local = _fetch_spot_cached()
    etc_local = _fetch_etc_cached(tuple(selected)) if selected else {}
    return spot_local, etc_local


# Defer fetching until after sidebar selections are set
spot, etc_prices = _get_spot_and_etc(selected_etcs)

# ========================= PRICE TICKER STRIP =========================
ticker_html = ticker_strip_html(
    gold_usd=spot["gold"]["price_usd"],
    gold_chg=spot["gold"]["change_pct"],
    silver_usd=spot["silver"]["price_usd"],
    silver_chg=spot["silver"]["change_pct"],
    gs_ratio=spot["ratio"],
    gbp_usd=spot["gbp_usd"],
    gold_gbp=spot["gold"]["price_gbp"],
    silver_gbp=spot["silver"]["price_gbp"],
    gold_inr=spot["gold"].get("price_inr_per_kg"),
    silver_inr=spot["silver"].get("price_inr_per_kg"),
    usd_inr=spot.get("usd_inr"),
)
st.html(render_component(ticker_html))

# ========================= ETC PRICE TILES =========================
if selected_etcs:
    etc_html = '<div class="etc-grid">'
    for ticker in selected_etcs:
        ep = etc_prices.get(ticker, {})
        price = ep.get("price", np.nan)
        curr = ep.get("currency", "GBp")
        if not np.isnan(price):
            price_str = f"\u00a3{price / 100:,.2f}" if curr == "GBp" else f"\u00a3{price:,.2f}"
        else:
            price_str = "N/A"
        etc_html += etc_tile_html(ticker, price_str, ep.get("change_pct", 0))
    etc_html += '</div>'
    st.html(render_component(etc_html))

# ========================= MAIN TABS =========================
tab_dashboard, tab_charts, tab_portfolio, tab_news, tab_simulator = st.tabs([
    "Dashboard", "Charts", "Portfolio", "News", "Simulator"
])

# ─────────────────────── DASHBOARD TAB ───────────────────────
with tab_dashboard:
    # Compute signals for both metals
    with st.spinner("Analysing markets..."):
        macro_state = _macro_state_cached()
        gold_tf = fetch_multi_timeframe_data("GC=F")
        silver_tf = fetch_multi_timeframe_data("SI=F")
        gold_fib = multi_timeframe_fibonacci(gold_tf)
        silver_fib = multi_timeframe_fibonacci(silver_tf)
        gold_daily = gold_tf["long_term"]
        silver_daily = silver_tf["long_term"]

        gold_score = score_metal(gold_daily, gold_fib, spot["gold"]["price_usd"], tf_weight_config) if not gold_daily.empty else None
        silver_score = score_metal(silver_daily, silver_fib, spot["silver"]["price_usd"], tf_weight_config) if not silver_daily.empty else None

        if st.session_state.get("macro_overlay_enabled", True):
            if gold_score:
                gold_score = apply_macro_overlay(gold_score, macro_state, "gold")
            if silver_score:
                silver_score = apply_macro_overlay(silver_score, macro_state, "silver")

    with st.expander("Macro Business Indicators (Zeburg (TM) Lens)", expanded=False):
        phase = macro_state.get("phase", "Unknown")
        conf = macro_state.get("confidence", 0.0)
        source = macro_state.get("source", "unknown")
        score_cols = st.columns(4)
        score_cols[0].metric("Regime", phase)
        score_cols[1].metric("Confidence", f"{conf:.0%}")
        score_cols[2].metric("Leading Score", f"{macro_state.get('leading_score', 0):+.2f}")
        score_cols[3].metric("Coincident Score", f"{macro_state.get('coincident_score', 0):+.2f}")
        st.caption(f"Source: {source} | Imminent Score: {macro_state.get('imminent_score', 0):+.2f}")
        if not st.session_state.get("macro_overlay_enabled", True):
            st.caption("Macro overlay is disabled in Settings. This section is informational only.")

        def _votes_df(votes: dict) -> pd.DataFrame:
            rows = []
            for k, v in votes.items():
                rows.append({
                    "Indicator": k,
                    "Vote": "Bullish" if v > 0 else ("Bearish" if v < 0 else "Neutral"),
                    "Score": int(v),
                })
            return pd.DataFrame(rows)

        vcols = st.columns(3)
        vcols[0].markdown("**Leading**")
        vcols[0].dataframe(_votes_df(macro_state.get("leading_votes", {})), use_container_width=True, hide_index=True)
        vcols[1].markdown("**Coincident**")
        vcols[1].dataframe(_votes_df(macro_state.get("coincident_votes", {})), use_container_width=True, hide_index=True)
        vcols[2].markdown("**Imminent Recession**")
        vcols[2].dataframe(_votes_df(macro_state.get("imminent_votes", {})), use_container_width=True, hide_index=True)

        st.markdown("**Macro metrics detail**")
        m = macro_state.get("metrics", {})
        metrics_df = pd.DataFrame([
            {"Metric": "Yield spread (10Y-2Y)", "Value": m.get("yield_spread_10y2y", np.nan)},
            {"Metric": "Yield spread (10Y-3M)", "Value": m.get("yield_spread_10y3m", np.nan)},
            {"Metric": "Building permits (6m %)", "Value": m.get("building_permits_6m_change", np.nan)},
            {"Metric": "Housing (6m %)", "Value": m.get("housing_6m_change", np.nan)},
            {"Metric": "Factory orders (6m %)", "Value": m.get("factory_orders_6m_change", np.nan)},
            {"Metric": "Consumer sentiment (6m %)", "Value": m.get("consumer_sentiment_6m_change", np.nan)},
            {"Metric": "Credit spread level", "Value": m.get("credit_spread_level", np.nan)},
            {"Metric": "Credit spread (3m %)", "Value": m.get("credit_spread_3m_change", np.nan)},
            {"Metric": "St. Louis FSI", "Value": m.get("financial_stress_level", np.nan)},
            {"Metric": "Payrolls (6m %)", "Value": m.get("payroll_6m_change", np.nan)},
            {"Metric": "Industrial Production (6m %)", "Value": m.get("indpro_6m_change", np.nan)},
            {"Metric": "Claims stress (13w/52w)", "Value": m.get("claims_13w_52w_ratio", np.nan)},
            {"Metric": "Sahm-like trigger", "Value": m.get("sahm_like", np.nan)},
            {"Metric": "Fed Funds", "Value": m.get("fed_funds", np.nan)},
            {"Metric": "CPI YoY", "Value": m.get("cpi_yoy", np.nan)},
            {"Metric": "VIX", "Value": m.get("vix", np.nan)},
            {"Metric": "Copper/Gold (3m %)", "Value": m.get("copper_gold_3m", np.nan)},
            {"Metric": "XLI/XLU (3m %)", "Value": m.get("xli_xlu_3m", np.nan)},
            {"Metric": "IWM/SPX (3m %)", "Value": m.get("iwm_spx_3m", np.nan)},
            {"Metric": "HYG/LQD (3m %)", "Value": m.get("hyg_lqd_3m", np.nan)},
        ])
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    alerts = []
    for metal_name, score_obj in (("gold", gold_score), ("silver", silver_score)):
        if not score_obj:
            continue
        signal = score_obj.get("signal", "Neutral")
        state_key = f"signal_regime_{metal_name}"
        prev_signal = st.session_state.get(state_key)
        if prev_signal is not None and prev_signal != signal:
            alerts.append(f"**{metal_name.title()}**: {prev_signal} → {signal}")
        st.session_state[state_key] = signal
    if alerts:
        st.warning("Signal regime change detected\n\n" + "\n".join(f"- {x}" for x in alerts))

    # ── Signal Cards ──
    cards_html = '<div class="cards-row">'
    if gold_score:
        cards_html += '<div>' + signal_card_html("gold", gold_score, spot["gold"]["price_usd"]) + '</div>'
    if silver_score:
        cards_html += '<div>' + signal_card_html("silver", silver_score, spot["silver"]["price_usd"]) + '</div>'
    cards_html += '</div>'
    st.html(render_component(cards_html))

    # ── Scoring Guide ──
    with st.expander("Scoring Guide"):
        st.markdown(
            "**Score Ranges**\n\n"
            "| Score Range | Signal | Strength |\n"
            "|---|---|---|\n"
            "| **+0.40 to +1.00** | Strong Buy | Strong |\n"
            "| **+0.20 to +0.39** | Buy | Moderate |\n"
            "| **-0.19 to +0.19** | Neutral | Weak |\n"
            "| **-0.20 to -0.39** | Sell / Take Profit | Moderate |\n"
            "| **-0.40 to -1.00** | Strong Sell | Strong |\n"
            "\n"
            "Each of the 14 indicators votes **+1** (bullish), **0** (neutral), or **-1** (bearish), "
            "multiplied by its weight. The composite score is the weighted sum divided by 100. "
            "Sub-scores per timeframe are normalised to the same -1.0 to +1.0 range.\n\n"
            "**Macro overlay (optional, Zeburg (TM)-inspired):**\n"
            "- Regime classification: Expansion / Slowdown / Contraction / Recovery.\n"
            "- Regime acts as a **modifier** (small score bias + adaptive thresholds), not a replacement for technical votes.\n"
            "- Lower-layer signals (market/technicals) do not override higher-layer regime context.\n\n"
        )
        # Dynamic indicator table based on current weight config
        rescaled = _rescale_indicators(tf_weight_config)
        indicator_desc = {
            "Fib Long-term (2yr)": "Major structural support/resistance",
            "Fib Medium-term (3mo)": "3M structure: 50% hold/break with 2-bar confirmation",
            "Fib Short-term (2-3wk)": "Entry timing levels",
            "Volatility Oscillator": "Price acceleration breakouts",
            "Boom Hunter Pro (BHS)": "COG-based trend/reversal (Ehlers)",
            "EMA Crossover (9/21)": "Short-term trend direction",
            "Hull Moving Average": "Medium trend filter (distance + slope)",
            "Modified ATR": "Volatility regime (narrow=conviction)",
            "Triangle Pattern": "Pattern breakout with 2-bar confirmation",
            "Bollinger Squeeze": "Post-squeeze direction filter (with RSI)",
            "Momentum Bars (ROC)": "Rate of change direction/strength",
            "Whale Volume": "Institutional accumulation (volume spikes)",
            "RSI (14)": "Momentum regime with 52/48 hysteresis",
            "SMA Crossover (20/50)": "Structural trend direction",
        }
        table = f"**{len(rescaled)} Indicators by Timeframe** (weights reflect current config)\n\n"
        table += "| # | Indicator | Timeframe | Base | Scaled | What it measures |\n"
        table += "|---|-----------|-----------|------|--------|------------------|\n"
        total_scaled = 0
        for i, (name, tf, weight, _) in enumerate(rescaled, 1):
            base = next(w for n, _, w, _ in INDICATORS if n == name)
            total_scaled += weight
            table += f"| {i} | {name} | {tf} | {base} | {int(round(weight))} | {indicator_desc.get(name, '')} |\n"
        table += f"| | **Total** | | **100** | **{total_scaled:.0f}** | |\n"
        table += (
            f"\n**Timeframe weights**: Short {tf_weight_config['Short']}% "
            f"| Medium {tf_weight_config['Medium']}% | Long {tf_weight_config['Long']}%\n\n"
            "**Correlation groups**: EMA/HMA/SMA (Trend MA), RSI/Momentum (Momentum), VO/Bollinger (Volatility). "
            "Conflicts are flagged when correlated indicators disagree."
        )
        st.markdown(table)

    # ── Indicator Breakdown Tables ──
    for metal_name, sc in [("Gold", gold_score), ("Silver", silver_score)]:
        if not sc:
            continue
        with st.expander(f"{metal_name} Indicator Breakdown"):
            table_df = pd.DataFrame(sc["indicator_table"])
            table_df["Conflict"] = table_df["Conflict"].map({True: "\u26a0\ufe0f", False: ""})
            if "Weight" in table_df.columns:
                table_df["Weight"] = pd.to_numeric(table_df["Weight"], errors="coerce").round(0).astype(int)
            if "Weighted Score" in table_df.columns:
                table_df["Weighted Score"] = pd.to_numeric(table_df["Weighted Score"], errors="coerce").round(0).astype(int)
            def _color_vote(val):
                if val == "Bullish": return "color: #26A69A"
                elif val == "Bearish": return "color: #EF5350"
                return "color: #6B7280"
            def _color_wscore(val):
                if val > 0: return "color: #26A69A"
                elif val < 0: return "color: #EF5350"
                return "color: #6B7280"
            def _color_direction(val):
                if "Improving" in str(val): return "color: #26A69A"
                elif "Deteriorating" in str(val): return "color: #EF5350"
                return "color: #6B7280"
            cols = ["Indicator", "Timeframe", "Vote", "Direction", "Weight", "Weighted Score", "Conflict", "Correlation Group", "Detail"]
            cols = [c for c in cols if c in table_df.columns]
            table_df = table_df[cols]
            # Build HTML table with center alignment and compact styling
            html = '<table style="width:100%;border-collapse:collapse;text-align:center;font-size:0.8rem;">'
            html += '<thead><tr>'
            for i, c in enumerate(cols):
                align = "left" if i == 0 or i == len(cols) - 1 else "center"
                html += f'<th style="padding:4px 6px;border-bottom:1px solid #2D3139;text-align:{align};color:#9CA3AF;font-size:0.8rem;font-weight:500;white-space:nowrap;">{c}</th>'
            html += '</tr></thead><tbody>'
            for _, row in table_df.iterrows():
                html += '<tr>'
                for i, c in enumerate(cols):
                    val = row[c]
                    align = "left" if i == 0 or i == len(cols) - 1 else "center"
                    width_style = "min-width:180px;white-space:nowrap;" if i == 0 else ""
                    style = f"padding:3px 6px;border-bottom:1px solid #2D3139;text-align:{align};font-size:0.75rem;{width_style}"
                    if c == "Vote":
                        color = "#26A69A" if val == "Bullish" else ("#EF5350" if val == "Bearish" else "#6B7280")
                        style += f"color:{color};font-weight:500;"
                    elif c == "Weighted Score":
                        color = "#26A69A" if val > 0 else ("#EF5350" if val < 0 else "#6B7280")
                        style += f"color:{color};"
                    elif c == "Direction":
                        if "Improving" in str(val):
                            style += "color:#26A69A;"
                        elif "Deteriorating" in str(val):
                            style += "color:#EF5350;"
                    html += f'<td style="{style}">{val}</td>'
                html += '</tr>'
            html += '</tbody></table>'
            st.html(html)
            if sc["conflicts"]:
                for c in sc["conflicts"]:
                    st.caption(
                        f"\u26a0\ufe0f **{c['group']}**: {c['ind_a']} ({c['vote_a']}) vs "
                        f"{c['ind_b']} ({c['vote_b']}) -- {c['explanation']}"
                    )

    # ── Market Analysis + Chat ──
    st.subheader("Market Analysis")
    for metal_name, sc, fib_data, price_key in [
        ("gold", gold_score, gold_fib, "gold"),
        ("silver", silver_score, silver_fib, "silver"),
    ]:
        if not sc:
            continue
        summary_text = generate_summary(metal_name, sc, spot[price_key]["price_usd"])
        with st.container(border=True):
            st.html(render_component(analysis_card_html(metal_name, summary_text)))
            context = build_context_prompt(metal_name, sc, spot[price_key]["price_usd"], fib_data)
            if chat.is_chat_available():
                chat_key = f"chat_history_{metal_name}"
                if chat_key not in st.session_state:
                    st.session_state[chat_key] = []
                for msg in st.session_state[chat_key]:
                    with st.chat_message(msg["role"]):
                        st.markdown(msg["content"])
                if prompt := st.chat_input(f"Ask about {metal_name.title()} signals...", key=f"chat_input_{metal_name}"):
                    st.session_state[chat_key].append({"role": "user", "content": prompt})
                    with st.chat_message("user"):
                        st.markdown(prompt)
                    with st.chat_message("assistant"):
                        with st.spinner("Thinking..."):
                            answer = chat.ask(prompt, context, st.session_state[chat_key])
                        st.markdown(answer)
                    st.session_state[chat_key].append({"role": "assistant", "content": answer})
            else:
                st.caption(f"Add GROQ_API_KEY to enable AI chat for {metal_name.title()} signals.")

    # ── Allocation Recommendation ──
    if gold_score and silver_score:
        st.subheader("Allocation Recommendation")
        alloc = allocation_recommendation(gold_score, silver_score, spot["ratio"])
        alloc_html = '<div class="etc-grid">'
        alloc_html += port_stat_html("Gold Allocation", f"{alloc['gold_pct']}%")
        alloc_html += port_stat_html("Silver Allocation", f"{alloc['silver_pct']}%")
        summary = get_summary(etc_prices)
        remaining = summary["remaining"]
        alloc_html += port_stat_html("Remaining to Deploy", f"\u00a3{remaining:,.0f}")
        alloc_html += '</div>'
        st.html(render_component(alloc_html))

        suggested_silver = remaining * alloc["silver_pct"] / 100
        suggested_gold = remaining * alloc["gold_pct"] / 100
        st.write(f"**Suggested next deployment**: ~\u00a3{suggested_silver:,.0f} Silver + ~\u00a3{suggested_gold:,.0f} Gold (split over 2-3 days)")
        for r in alloc["reasoning"]:
            st.write(f"- {r}")

# ─────────────────────── CHARTS TAB ───────────────────────
with tab_charts:
    chart_metal = st.radio("Metal", ["Gold", "Silver"], horizontal=True, key="chart_metal_select")
    spot_ticker = "GC=F" if chart_metal == "Gold" else "SI=F"
    metal_key = "gold" if chart_metal == "Gold" else "silver"

    with st.spinner(f"Loading {chart_metal} charts..."):
        tf_data = fetch_multi_timeframe_data(spot_ticker)
        fib = multi_timeframe_fibonacci(tf_data)

    tf_options = {"2 Years": "long_term", "3 Months": "medium_term", "4 Weeks": "short_term"}
    selected_tf = st.radio("Timeframe", list(tf_options.keys()), horizontal=True, key="chart_tf_select")
    tf_key = tf_options[selected_tf]
    chart_df = tf_data[tf_key]

    if not chart_df.empty and len(chart_df) > 20:
        chart_fib = fib.get(tf_key, {})
        render_metal_chart(
            chart_df, chart_fib, metal_key, selected_tf,
            ema_fast=ema_fast, ema_slow=ema_slow,
            sma_fast=sma_fast, sma_slow=sma_slow, rsi_period=rsi_period,
            chart_key=f"tv_{metal_key}_{tf_key}",
        )
        tri = detect_triangles(chart_df)
        if tri["pattern"] not in ("no_pattern", "insufficient_data"):
            st.info(f"Triangle pattern detected: **{tri['pattern']}** | Breakout Up: {tri['breakout_up']} | Breakout Down: {tri['breakout_down']}")
    else:
        st.warning(f"Insufficient data for {selected_tf} chart")

    # Tracking difference
    if selected_etcs:
        with st.expander("Tracking Difference (Spot vs ETC)"):
            for ticker in selected_etcs:
                mk = "gold" if "SGL" in ticker.upper() else ("silver" if "SSL" in ticker.upper() or "SLV" in ticker.upper() else
                      ("gold" if "G" in ticker.upper() and "S" not in ticker.upper()[:4] else "silver"))
                spot_tk = SPOT_TICKERS[mk]
                spot_hist = fetch_historical(spot_tk, period="3mo")
                etc_hist = fetch_historical(ticker, period="3mo")
                if not spot_hist.empty and not etc_hist.empty:
                    td = compute_tracking_difference(spot_hist["Close"], etc_hist["Close"])
                    if not td.empty:
                        st.write(f"**{ticker}** vs {spot_tk} (3-month): latest diff = {td.iloc[-1]:.2f}%")

# ─────────────────────── PORTFOLIO TAB ───────────────────────
with tab_portfolio:
    summary = get_summary(etc_prices)

    # Stats row
    stats_html = '<div class="etc-grid">'
    stats_html += port_stat_html("Total Pot", f"\u00a3{summary['total_pot']:,.0f}")
    stats_html += port_stat_html("Deployed", f"\u00a3{summary['total_deployed']:,.0f}")
    stats_html += port_stat_html("Remaining", f"\u00a3{summary['remaining']:,.0f}")
    stats_html += port_stat_html("Current Value", f"\u00a3{summary['total_current_value']:,.0f}")
    pnl_col = GREEN if summary['total_pnl'] >= 0 else RED
    stats_html += f"""
    <div class="port-stat">
        <div class="port-stat-label">Total P&L</div>
        <div class="port-stat-val" style="color:{pnl_col}">\u00a3{summary['total_pnl']:,.0f}
            <span style="font-size:0.8rem"> ({summary['total_pnl_pct']:+.1f}%)</span>
        </div>
    </div>"""
    stats_html += '</div>'
    st.html(render_component(stats_html))

    if summary["deployment_pct"] > 0:
        st.progress(min(summary["deployment_pct"] / 100, 1.0), text=f"{summary['deployment_pct']:.0f}% deployed")

    # Positions
    if summary["positions"]:
        st.subheader("Open Positions")
        pos_df = pd.DataFrame(summary["positions"])
        st.dataframe(pos_df.style.format({
            "cost_gbp": "\u00a3{:,.0f}", "current_value_gbp": "\u00a3{:,.0f}",
            "pnl_gbp": "\u00a3{:,.0f}", "pnl_pct": "{:+.1f}%", "quantity": "{:,.2f}",
        }), use_container_width=True)

    # Purchase log
    if summary["purchases"]:
        with st.expander("Purchase Log"):
            st.dataframe(pd.DataFrame(summary["purchases"]), use_container_width=True)

    # Persistence status
    gh_cfg = _github_config()
    if gh_cfg:
        st.caption(f"Portfolio persisted to GitHub: {gh_cfg['repo']}")
    else:
        st.caption("Portfolio stored locally. Set GITHUB_TOKEN + GITHUB_REPO for cloud persistence.")

    # Backup / Restore
    with st.expander("Backup & Restore"):
        st.download_button("Download Portfolio JSON", data=export_portfolio_json(),
                           file_name="portfolio_data.json", mime="application/json")
        uploaded = st.file_uploader("Restore from JSON backup", type=["json"])
        if uploaded:
            import_portfolio_json(uploaded.read().decode("utf-8"))
            st.success("Portfolio restored.")
            st.rerun()

    # Add purchase
    with st.expander("Log New Purchase"):
        with st.form("add_purchase"):
            p_metal = st.selectbox("Metal", ["Gold", "Silver"])
            p_ticker = st.selectbox("ETC Ticker", selected_etcs if selected_etcs else list(DEFAULT_ETC_TICKERS.keys()))
            p_amount = st.number_input("Amount (GBP)", min_value=0.0, step=1000.0)
            p_price = st.number_input("Price per unit", min_value=0.0, step=0.01)
            p_qty = st.number_input("Quantity", min_value=0.0, step=1.0)
            p_notes = st.text_input("Notes")
            submitted = st.form_submit_button("Add Purchase")
            if submitted and p_amount > 0:
                add_purchase(p_metal, p_ticker, p_amount, p_price, p_qty, p_notes)
                st.success("Purchase logged.")
                st.rerun()

# ─────────────────────── NEWS TAB ───────────────────────
with tab_news:
    with st.spinner("Fetching headlines..."):
        articles = fetch_catalyst_news()

    if articles:
        for art in articles:
            published = art.get("published", "")
            title = art["title"]
            source = title.rsplit(" - ", 1)[-1] if " - " in title else art["source"]
            clean_title = title.rsplit(" - ", 1)[0] if " - " in title else title
            summary_text = art.get("summary_text", "")
            read_link = art.get("real_link", art["link"])
            st.html(render_component(
                news_card_html(published, source, clean_title, summary_text, read_link)
            ))
    else:
        st.write("No catalyst headlines found right now.")

# ─────────────────────── SIMULATOR TAB ───────────────────────
with tab_simulator:
    st.subheader("Historical Signal Simulator")
    with st.form("sim_form"):
        sim_start = st.date_input("Backtest start date", value=date(2026, 1, 1))
        strategy = st.selectbox(
            "Strategy",
            options=[
                ("baseline", "Baseline (per-signal targets)"),
                ("agree", "Short+Medium agree only"),
                ("hysteresis", "Composite hysteresis bands"),
                ("banded", "Score-banded sizing"),
                ("confirm", "2-bar confirmation for entries"),
                ("cooldown", "Cooldown after flips"),
                ("time_filter", "Time-in-market filter (Short & Medium)"),
                ("decay", "Position decay on weakening short score"),
            ],
            format_func=lambda x: x[1],
        )[0]
        with st.expander("What do these strategies mean?", expanded=False):
            st.markdown("""
            - **Baseline (per-signal targets):** Uses the raw signal mapping (Strong Buy 60%, Buy 30%, Sell/Strong Sell 30%, Neutral 0%) at each checkpoint.
            - **Short+Medium agree only:** Requires both Short and Medium scores to align (>= +0.2 to enter; >= +0.4 for Strong). If they don’t agree, target is 0% (flat). Reduces whipsaw.
            - **Composite hysteresis bands:** Uses composite score with bands to avoid flip-flop. Enter above +0.2 (Strong above +0.4), exit below -0.1. Holds when inside the band.
            - **Score-banded sizing:** Scales size with composite: 0→0%, 0.4→60% (linear in between). No fixed steps.
            - **2-bar confirmation:** Only enters after 2 consecutive Buy/Strong Buy readings; exits immediately on Neutral/Sell.
            - **Cooldown after flips:** On Sell/Neutral sets a 3-bar cooldown; bullish entries during cooldown require composite ≥0.35.
            - **Time-in-market filter:** Needs Short >0 for 3 bars and Medium >0 for 2 bars to allow entry; otherwise stay flat.
            - **Position decay:** If holding and Short score weakens vs prior bar, trims target by 5% (but stays long if signal is still bullish).
            """)
        run_sim = st.form_submit_button("Run backtest", use_container_width=True)
        run_all = st.form_submit_button("Run all strategies", use_container_width=True)

    if run_sim:
        with st.spinner("Running backtests (morning vs end-of-day)..."):
            sim_results = run_simulations(
                start_date=datetime.combine(sim_start, datetime.min.time()),
                tf_weights=tf_weight_config,
                strategy=strategy,
            )
        st.session_state["sim_results"] = sim_results

    if run_all:
        all_results = {}
        strat_options = ["baseline", "agree", "hysteresis", "banded", "confirm", "cooldown", "time_filter", "decay"]
        with st.spinner("Running all strategies across all scenarios..."):
            start_dt = datetime.combine(sim_start, datetime.min.time())
            prog = st.progress(0.0, text="Starting strategy batch...")
            for idx, strat in enumerate(strat_options, start=1):
                try:
                    all_results[strat] = run_simulations(start_dt, tf_weight_config, strat)
                except Exception as e:
                    all_results[strat] = {"error": str(e)}
                prog.progress(idx / len(strat_options), text=f"Running {strat} ({idx}/{len(strat_options)})")
            prog.empty()
        st.session_state["all_sim_results"] = all_results

    sim_results = st.session_state.get("sim_results")
    all_sim_results = st.session_state.get("all_sim_results")
    if sim_results:
        good_sim_results = {k: v for k, v in sim_results.items() if isinstance(v, dict) and "metrics" in v}
        bad_sim_results = {k: v for k, v in sim_results.items() if not (isinstance(v, dict) and "metrics" in v)}
        if good_sim_results:
            metrics_cols = st.columns(len(good_sim_results))
        else:
            metrics_cols = []
        for col, (name, res) in zip(metrics_cols, good_sim_results.items()):
            m = res["metrics"]
            col.markdown(f"**{name.replace('_', ' ').title()}**")
            col.metric("Final Equity (GBP)", f"£{m['final_equity_gbp']:,.0f}", delta=f"£{m['pnl_gbp_abs']:,.0f} ({m['pnl_gbp_pct']*100:.1f}%)")
            col.metric("Final Equity (USD)", f"${m['final_equity_usd']:,.0f}", delta=f"${m['pnl_usd_abs']:,.0f} ({m['pnl_usd_pct']*100:.1f}%)")
            col.caption(f"CAGR {m['CAGR']*100:.1f}% | Sharpe {m['Sharpe']:.2f} | Max DD {m['Max Drawdown']*100:.1f}%")

        if good_sim_results:
            eq_df = pd.DataFrame({f"{name} (GBP)": res["equity"]["equity_gbp"] for name, res in good_sim_results.items()})
            st.line_chart(eq_df, height=260)
            end_caption = " | ".join([
                f"{name.replace('_',' ').title()}: £{res['metrics']['final_equity_gbp']:,.0f} ({res['metrics']['pnl_gbp_pct']*100:.1f}%)"
                for name, res in good_sim_results.items()
            ])
            st.caption(f"Ending equity & P&L vs £{next(iter(good_sim_results.values()))['metrics']['initial_cash']:,.0f}: {end_caption}")

            sel = st.selectbox("View trades for", list(good_sim_results.keys()))
            trades = good_sim_results[sel]["trades"]
            if not trades.empty:
                trades_display = trades.copy()
                trades_display["timestamp"] = trades_display["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
                st.dataframe(
                    trades_display[[
                        "timestamp", "metal", "signal", "target_weight", "units_delta", "price_gbp",
                        "notional_gbp", "commission_gbp", "equity_gbp_after", "equity_usd_after", "pnl_gbp_pct"
                    ]],
                    use_container_width=True, hide_index=True,
                )
            else:
                st.write("No trades generated for this window.")

        if bad_sim_results:
            st.warning("Some scenarios failed: " + ", ".join(
                f"{k}: {v.get('error', 'missing metrics') if isinstance(v, dict) else 'invalid result'}"
                for k, v in bad_sim_results.items()
            ))

    if all_sim_results:
        st.subheader("All strategies summary (final P&L)")
        rows = []
        err_rows = []
        for strat, scenarios in all_sim_results.items():
            if isinstance(scenarios, dict) and set(scenarios.keys()) == {"error"}:
                err_rows.append(f"{strat}: {scenarios.get('error','unknown error')}")
                continue
            if not isinstance(scenarios, dict):
                err_rows.append(f"{strat}: unexpected result type")
                continue
            for scen, res in scenarios.items():
                if not isinstance(res, dict):
                    err_rows.append(f"{strat}/{scen}: unexpected type")
                    continue
                if "error" in res:
                    err_rows.append(f"{strat}/{scen}: {res['error']}")
                    continue
                if "metrics" not in res:
                    err_rows.append(f"{strat}/{scen}: missing metrics")
                    continue
                m = res["metrics"]
                rows.append({
                    "Strategy": strat,
                    "Scenario": scen,
                    "Final Equity (GBP)": f"£{m.get('final_equity_gbp', 0):,.0f}",
                    "P&L GBP": f"£{m.get('pnl_gbp_abs', 0):,.0f}",
                    "P&L GBP %": f"{m.get('pnl_gbp_pct', 0)*100:,.1f}%",
                    "Final Equity (USD)": f"${m.get('final_equity_usd', 0):,.0f}",
                    "P&L USD": f"${m.get('pnl_usd_abs', 0):,.0f}",
                    "P&L USD %": f"{m.get('pnl_usd_pct', 0)*100:,.1f}%",
                })
        summary_df = pd.DataFrame(rows)
        if summary_df.empty:
            st.info("No successful strategy results to display.")
        else:
            table_height = min(900, max(400, 35 * (len(summary_df) + 1)))
            st.dataframe(
                summary_df.sort_values(["Strategy", "Scenario"]).reset_index(drop=True),
                use_container_width=True,
                height=table_height,
            )
        if err_rows:
            st.warning("Some strategies failed:\n" + "\n".join(err_rows))

# ========================= FOOTER =========================
st.divider()
col_refresh, col_info = st.columns([1, 3])
with col_refresh:
    if st.button("Refresh All Data"):
        st.cache_data.clear()
        clear_simulator_caches()
        st.rerun()
with col_info:
    st.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')} | Data: Yahoo Finance | Not financial advice")
