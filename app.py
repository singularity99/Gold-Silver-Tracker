import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

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
from signals import score_metal, allocation_recommendation
from charts import render_metal_chart
from portfolio import (
    get_portfolio, set_total_pot, add_purchase, delete_purchase, get_summary,
    export_portfolio_json, import_portfolio_json, _github_config,
)
from news import fetch_catalyst_news

st.set_page_config(page_title="Gold & Silver Strategy Monitor", layout="wide")
st.title("Gold & Silver Strategy Monitor")
st.caption("Dual-price tracking: Spot (for signals) + ETC (for trading on Hargreaves Lansdown)")

# ========================= SIDEBAR CONFIG =========================
st.sidebar.header("Configuration")

st.sidebar.subheader("ETC Tickers (HL Trading)")
default_tickers = list(DEFAULT_ETC_TICKERS.keys())
selected_etcs = st.sidebar.multiselect(
    "Select ETCs to track",
    options=default_tickers,
    default=default_tickers[:2],
    format_func=lambda x: f"{x} ({DEFAULT_ETC_TICKERS[x]})",
)

st.sidebar.subheader("Signal Parameters")
fib_tolerance = st.sidebar.slider("Fibonacci proximity tolerance (%)", 0.5, 5.0, 2.0, 0.5)
gs_ratio_threshold = st.sidebar.number_input("G/S Ratio threshold (favour silver above)", value=63.0, step=0.5)
vol_osc_length = st.sidebar.number_input("Volatility Oscillator length", value=100, step=10)
rsi_period = st.sidebar.number_input("RSI period", value=14, step=1)
ema_fast = st.sidebar.number_input("Fast EMA", value=9, step=1)
ema_slow = st.sidebar.number_input("Slow EMA", value=21, step=1)
sma_fast = st.sidebar.number_input("Fast SMA", value=20, step=5)
sma_slow = st.sidebar.number_input("Slow SMA", value=50, step=5)
whale_vol_threshold = st.sidebar.number_input("Whale volume threshold (x avg)", value=2.0, step=0.5)

st.sidebar.subheader("Portfolio")
portfolio_data = get_portfolio()
new_pot = st.sidebar.number_input("Total investment pot (GBP)", value=portfolio_data["total_pot"], step=10_000)
if new_pot != portfolio_data["total_pot"]:
    set_total_pot(new_pot)

# ========================= DATA LOADING =========================
with st.spinner("Fetching live prices..."):
    spot = fetch_spot_prices()
    etc_prices = fetch_etc_prices(selected_etcs) if selected_etcs else {}

# ========================= PRICE DISPLAY =========================
st.header("Live Prices")

price_cols = st.columns(4)
gold_usd = spot["gold"]["price_usd"]
silver_usd = spot["silver"]["price_usd"]

price_cols[0].metric("Gold Spot (USD)", f"${gold_usd:,.0f}" if not np.isnan(gold_usd) else "N/A",
                     f"{spot['gold']['change_pct']:+.1f}%" if not np.isnan(gold_usd) else None)
price_cols[1].metric("Gold Spot (GBP)", f"\u00a3{spot['gold']['price_gbp']:,.0f}" if not np.isnan(spot['gold']['price_gbp']) else "N/A")
price_cols[2].metric("Silver Spot (USD)", f"${silver_usd:.2f}" if not np.isnan(silver_usd) else "N/A",
                     f"{spot['silver']['change_pct']:+.1f}%" if not np.isnan(silver_usd) else None)
price_cols[3].metric("Silver Spot (GBP)", f"\u00a3{spot['silver']['price_gbp']:.2f}" if not np.isnan(spot['silver']['price_gbp']) else "N/A")

ratio_col, fx_col = st.columns(2)
ratio_col.metric("Gold/Silver Ratio", f"{spot['ratio']:.1f}:1" if not np.isnan(spot['ratio']) else "N/A")
fx_col.metric("GBP/USD Rate", f"{spot['gbp_usd']:.4f}")

# --- ETC Prices + Tracking Difference ---
if selected_etcs:
    st.subheader("ETC Prices (Hargreaves Lansdown)")
    etc_cols = st.columns(len(selected_etcs))
    for i, ticker in enumerate(selected_etcs):
        ep = etc_prices.get(ticker, {})
        price = ep.get("price", np.nan)
        curr = ep.get("currency", "GBp")
        if not np.isnan(price):
            if curr == "GBp":
                label = f"\u00a3{price / 100:,.2f}"
            else:
                label = f"\u00a3{price:,.2f}"
        else:
            label = "N/A"
        etc_cols[i].metric(f"{ticker}", label, f"{ep.get('change_pct', 0):+.1f}%")

    with st.expander("Tracking Difference (Spot vs ETC)"):
        for ticker in selected_etcs:
            metal_key = "gold" if "G" in ticker.upper() and "S" not in ticker.upper()[:4] else "silver"
            if "SGL" in ticker.upper():
                metal_key = "gold"
            elif "SSL" in ticker.upper() or "SLV" in ticker.upper():
                metal_key = "silver"
            spot_ticker = SPOT_TICKERS[metal_key]
            spot_hist = fetch_historical(spot_ticker, period="3mo")
            etc_hist = fetch_historical(ticker, period="3mo")
            if not spot_hist.empty and not etc_hist.empty:
                td = compute_tracking_difference(spot_hist["Close"], etc_hist["Close"])
                if not td.empty:
                    st.write(f"**{ticker}** vs {spot_ticker} (3-month): latest diff = {td.iloc[-1]:.2f}%")

# ========================= TECHNICAL ANALYSIS & SIGNALS =========================
st.header("Technical Analysis & Signals")

tab_gold, tab_silver = st.tabs(["Gold", "Silver"])

for tab, metal, spot_ticker in [(tab_gold, "gold", "GC=F"), (tab_silver, "silver", "SI=F")]:
    with tab:
        with st.spinner(f"Analysing {metal}..."):
            tf_data = fetch_multi_timeframe_data(spot_ticker)
            fib = multi_timeframe_fibonacci(tf_data)
            df_daily = tf_data["long_term"]

            if df_daily.empty:
                st.error(f"Could not fetch data for {metal}")
                continue

            current = spot[metal]["price_usd"]
            score = score_metal(df_daily, fib, current)

        # --- Signal Banner ---
        sig = score["signal"]
        comp = score["composite_score"]
        banner = f"**{sig}** | Score: {comp:+.2f} | {score['bullish_votes']}B / {score['bearish_votes']}S / {score['neutral_votes']}N out of {score['total_indicators']}"
        if "Strong Buy" in sig:
            st.success(banner)
        elif "Buy" in sig:
            st.info(banner)
        elif "Sell" in sig or "Strong Sell" in sig:
            st.error(banner)
        else:
            st.warning(banner)

        # --- Timeframe Sub-Scores ---
        tf_scores = score["timeframe_scores"]
        tf_weights = score["timeframe_weights"]
        tf_cols = st.columns(3)
        for i, (tf, label) in enumerate([("Short", "Short-term (days-weeks)"), ("Medium", "Medium-term (weeks-months)"), ("Long", "Long-term (months-years)")]):
            val = round(tf_scores[tf], 2)
            pct = tf_weights[tf]
            color = "green" if val > 0.1 else ("red" if val < -0.1 else "gray")
            if val >= 0.40:
                delta_label = "Strong Buy"
            elif val >= 0.20:
                delta_label = "Buy"
            elif val <= -0.40:
                delta_label = "Strong Sell"
            elif val <= -0.20:
                delta_label = "Sell"
            else:
                delta_label = "Neutral"
            tf_cols[i].metric(f"{label} ({pct}% weight)", f"{val:+.2f}", delta_label)

        # --- Score Legend ---
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
                "**14 Indicators by Timeframe**\n\n"
                "| # | Indicator | Timeframe | Weight | What it measures |\n"
                "|---|-----------|-----------|--------|------------------|\n"
                "| 1 | Fib Long-term (2yr) | Long | 5 | Major structural support/resistance |\n"
                "| 2 | Fib Medium-term (3mo) | Medium | 9 | Current trend support/resistance |\n"
                "| 3 | Fib Short-term (2-3wk) | Short | 7 | Entry timing levels |\n"
                "| 4 | Volatility Oscillator | Short | 8 | Price acceleration breakouts |\n"
                "| 5 | Boom Hunter Pro (BHS) | Short | 11 | COG-based trend/reversal (Ehlers) |\n"
                "| 6 | EMA Crossover (9/21) | Short | 7 | Short-term trend direction |\n"
                "| 7 | Hull Moving Average | Medium | 7 | Medium-term trend (low-lag) |\n"
                "| 8 | Modified ATR | Long | 4 | Volatility regime (narrow=conviction) |\n"
                "| 9 | Triangle Pattern | Medium | 6 | Chart pattern breakout detection |\n"
                "| 10 | Bollinger Squeeze | Medium | 6 | Volatility compression/expansion |\n"
                "| 11 | Momentum Bars (ROC) | Short | 8 | Rate of change direction/strength |\n"
                "| 12 | Whale Volume | Short | 7 | Institutional accumulation (volume spikes) |\n"
                "| 13 | RSI (14) | Medium | 9 | Overbought/oversold conditions |\n"
                "| 14 | SMA Crossover (20/50) | Long | 6 | Structural trend direction |\n"
                "| | **Total** | | **100** | |\n"
                "\n"
                "**Timeframe weights**: Short 48% | Medium 37% | Long 15%\n\n"
                "**Correlation groups**: EMA/HMA/SMA (Trend MA), RSI/Momentum (Momentum), VO/Bollinger (Volatility). "
                "Conflicts are flagged when correlated indicators disagree."
            )

        # --- Correlation Conflicts ---
        if score["conflicts"]:
            for conflict in score["conflicts"]:
                st.warning(f"Conflict: {conflict}")

        # --- Weighted Indicator Breakdown Table ---
        with st.expander("Indicator Breakdown (weighted)"):
            table_df = pd.DataFrame(score["indicator_table"])
            def _color_vote(val):
                if val == "Bullish":
                    return "color: green"
                elif val == "Bearish":
                    return "color: red"
                return "color: gray"
            def _color_wscore(val):
                if val > 0:
                    return "color: green"
                elif val < 0:
                    return "color: red"
                return "color: gray"
            def _color_direction(val):
                if "Improving" in str(val):
                    return "color: green"
                elif "Deteriorating" in str(val):
                    return "color: red"
                return "color: gray"
            styled = (table_df.style
                      .applymap(_color_vote, subset=["Vote"])
                      .applymap(_color_wscore, subset=["Weighted Score"])
                      .applymap(_color_direction, subset=["Direction"]))
            st.dataframe(styled, use_container_width=True, hide_index=True)

        # --- Fibonacci Levels ---
        with st.expander("Fibonacci Levels"):
            for tf_name, levels in fib.items():
                st.write(f"**{tf_name.replace('_', ' ').title()}**")
                for ratio_key, value in levels.items():
                    if ratio_key in ("swing_high", "swing_low"):
                        continue
                    st.write(f"  {ratio_key}: ${value:,.0f}")

        # --- Timeframe Switcher + Charts ---
        st.subheader(f"{metal.title()} Charts")

        tf_options = {"2 Years": "long_term", "3 Months": "medium_term", "4 Weeks": "short_term"}
        selected_tf = st.radio("Timeframe", list(tf_options.keys()), horizontal=True, key=f"tf_{metal}")
        tf_key = tf_options[selected_tf]
        chart_df = tf_data[tf_key]

        if not chart_df.empty and len(chart_df) > 20:
            chart_fib = fib.get(tf_key, {})
            render_metal_chart(
                chart_df, chart_fib, metal, selected_tf,
                ema_fast=ema_fast, ema_slow=ema_slow,
                sma_fast=sma_fast, sma_slow=sma_slow, rsi_period=rsi_period,
                chart_key=f"tv_{metal}_{tf_key}",
            )

            tri = detect_triangles(chart_df)
            if tri["pattern"] not in ("no_pattern", "insufficient_data"):
                st.info(f"Triangle pattern detected: **{tri['pattern']}** | Breakout Up: {tri['breakout_up']} | Breakout Down: {tri['breakout_down']}")
        else:
            st.warning(f"Insufficient data for {selected_tf} chart")

# ========================= ALLOCATION RECOMMENDATION =========================
st.header("Allocation Recommendation")

with st.spinner("Computing allocation..."):
    gold_tf = fetch_multi_timeframe_data("GC=F")
    silver_tf = fetch_multi_timeframe_data("SI=F")
    gold_fib = multi_timeframe_fibonacci(gold_tf)
    silver_fib = multi_timeframe_fibonacci(silver_tf)
    gold_daily = gold_tf["long_term"]
    silver_daily = silver_tf["long_term"]

    if not gold_daily.empty and not silver_daily.empty:
        gold_score = score_metal(gold_daily, gold_fib, spot["gold"]["price_usd"])
        silver_score = score_metal(silver_daily, silver_fib, spot["silver"]["price_usd"])
        alloc = allocation_recommendation(gold_score, silver_score, spot["ratio"])

        alloc_cols = st.columns(3)
        alloc_cols[0].metric("Gold Allocation", f"{alloc['gold_pct']}%")
        alloc_cols[1].metric("Silver Allocation", f"{alloc['silver_pct']}%")

        summary = get_summary(etc_prices)
        remaining = summary["remaining"]
        suggested_silver = remaining * alloc["silver_pct"] / 100
        suggested_gold = remaining * alloc["gold_pct"] / 100
        alloc_cols[2].metric("Remaining to Deploy", f"\u00a3{remaining:,.0f}")

        st.write(f"**Suggested next deployment**: ~\u00a3{suggested_silver:,.0f} into Silver ETC + ~\u00a3{suggested_gold:,.0f} into Gold ETC (split over 2-3 days)")
        for r in alloc["reasoning"]:
            st.write(f"- {r}")

# ========================= PORTFOLIO TRACKER =========================
st.header("Portfolio Tracker")

summary = get_summary(etc_prices)

port_cols = st.columns(5)
port_cols[0].metric("Total Pot", f"\u00a3{summary['total_pot']:,.0f}")
port_cols[1].metric("Deployed", f"\u00a3{summary['total_deployed']:,.0f}")
port_cols[2].metric("Remaining", f"\u00a3{summary['remaining']:,.0f}")
port_cols[3].metric("Current Value", f"\u00a3{summary['total_current_value']:,.0f}")
pnl_delta = f"{summary['total_pnl_pct']:+.1f}%" if summary['total_deployed'] > 0 else None
port_cols[4].metric("Total P&L", f"\u00a3{summary['total_pnl']:,.0f}", pnl_delta)

if summary["deployment_pct"] > 0:
    st.progress(min(summary["deployment_pct"] / 100, 1.0), text=f"{summary['deployment_pct']:.0f}% deployed")

# Position details
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
        log_df = pd.DataFrame(summary["purchases"])
        st.dataframe(log_df, use_container_width=True)

# Persistence status
gh_cfg = _github_config()
if gh_cfg:
    st.caption(f"Portfolio data persisted to GitHub repo: {gh_cfg['repo']}")
else:
    st.caption("Portfolio data stored locally. Set GITHUB_TOKEN + GITHUB_REPO in Streamlit secrets for cloud persistence.")

# Backup / Restore
with st.expander("Backup & Restore Portfolio"):
    st.download_button(
        "Download Portfolio JSON",
        data=export_portfolio_json(),
        file_name="portfolio_data.json",
        mime="application/json",
    )
    uploaded = st.file_uploader("Restore from JSON backup", type=["json"])
    if uploaded:
        import_portfolio_json(uploaded.read().decode("utf-8"))
        st.success("Portfolio restored from backup.")
        st.rerun()

# Add new purchase
with st.expander("Log New Purchase"):
    with st.form("add_purchase"):
        p_metal = st.selectbox("Metal", ["Gold", "Silver"])
        p_ticker = st.selectbox("ETC Ticker", selected_etcs if selected_etcs else list(DEFAULT_ETC_TICKERS.keys()))
        p_amount = st.number_input("Amount (GBP)", min_value=0.0, step=1000.0)
        p_price = st.number_input("Price per unit (pence for GBp, pounds for GBP)", min_value=0.0, step=0.01)
        p_qty = st.number_input("Quantity (units/shares)", min_value=0.0, step=1.0)
        p_notes = st.text_input("Notes")
        submitted = st.form_submit_button("Add Purchase")
        if submitted and p_amount > 0:
            add_purchase(p_metal, p_ticker, p_amount, p_price, p_qty, p_notes)
            st.success("Purchase logged.")
            st.rerun()

# ========================= NEWS FEED =========================
st.header("Catalyst News Feed")

with st.spinner("Fetching headlines and article summaries..."):
    articles = fetch_catalyst_news()

if articles:
    for art in articles:
        published = art.get("published", "")
        # Extract publisher from title suffix (e.g. "Headline - CNBC")
        title = art["title"]
        source = title.rsplit(" - ", 1)[-1] if " - " in title else art["source"]
        clean_title = title.rsplit(" - ", 1)[0] if " - " in title else title
        summary_text = art.get("summary_text", "")
        read_link = art.get("real_link", art["link"])
        st.markdown(f"**{published}** | {source} | \"{clean_title}\"")
        if summary_text:
            display = summary_text[:500] + "..." if len(summary_text) > 500 else summary_text
            st.caption(display)
        else:
            st.caption("(Summary not available)")
        st.markdown(f"[Read full article]({read_link})")
        st.divider()
else:
    st.write("No catalyst headlines found right now.")

# ========================= REFRESH =========================
st.divider()
if st.button("Refresh All Data"):
    st.cache_data.clear()
    st.rerun()

st.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')} | Data: Yahoo Finance + Kitco/Reuters RSS | Not financial advice")
