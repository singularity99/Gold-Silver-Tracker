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
from analysis import generate_summary, build_context_prompt
import chat
from style import (
    GLOBAL_CSS, GOLD, SILVER, GREEN, RED, AMBER, TEXT_MUTED,
    ticker_strip_html, signal_card_html, etc_tile_html,
    news_card_html, port_stat_html, render_component,
    analysis_card_html,
)

st.set_page_config(
    page_title="Gold & Silver Strategy Monitor",
    page_icon="🥇",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Inject global CSS
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

# ========================= SIDEBAR CONFIG =========================
st.sidebar.title("Settings")

available_etcs = list(DEFAULT_ETC_TICKERS.keys())
selected_etcs = st.sidebar.multiselect("Track ETCs (Hargreaves Lansdown)", available_etcs, default=available_etcs[:2])

st.sidebar.subheader("Signal Parameters")
fib_tolerance = st.sidebar.slider("Fibonacci proximity tolerance (%)", 0.5, 5.0, 2.0, 0.5)
gs_ratio_threshold = st.sidebar.number_input("G/S Ratio threshold (favour silver above)", value=63.0, step=0.5)
rsi_period = st.sidebar.number_input("RSI period", value=14, step=1)
ema_fast = st.sidebar.number_input("Fast EMA", value=9, step=1)
ema_slow = st.sidebar.number_input("Slow EMA", value=21, step=1)
sma_fast = st.sidebar.number_input("Fast SMA", value=20, step=5)
sma_slow = st.sidebar.number_input("Slow SMA", value=50, step=5)
whale_vol_threshold = st.sidebar.number_input("Whale volume threshold (x avg)", value=2.0, step=0.5)

st.sidebar.subheader("Timeframe Weights")
# Auto-adjusting sliders: changing one redistributes the others proportionally
for k in ("w_short", "w_medium", "w_long"):
    if k not in st.session_state:
        st.session_state[k] = {"w_short": 48, "w_medium": 37, "w_long": 15}[k]

def _rebalance(changed_key):
    """When one slider changes, proportionally adjust the other two to keep sum = 100."""
    keys = ["w_short", "w_medium", "w_long"]
    others = [k for k in keys if k != changed_key]
    new_val = st.session_state[changed_key]
    remaining = 100 - new_val
    other_sum = sum(st.session_state[k] for k in others)
    if other_sum > 0:
        for k in others:
            st.session_state[k] = max(0, round(st.session_state[k] / other_sum * remaining))
        # Fix rounding to exactly 100
        diff = 100 - new_val - sum(st.session_state[k] for k in others)
        if diff != 0:
            st.session_state[others[0]] += diff
    else:
        for i, k in enumerate(others):
            st.session_state[k] = remaining // len(others) + (1 if i < remaining % len(others) else 0)

st.sidebar.slider("Short-term %", 0, 100, key="w_short", on_change=_rebalance, args=("w_short",))
st.sidebar.slider("Medium-term %", 0, 100, key="w_medium", on_change=_rebalance, args=("w_medium",))
st.sidebar.slider("Long-term %", 0, 100, key="w_long", on_change=_rebalance, args=("w_long",))
tf_weight_config = {"Short": st.session_state.w_short, "Medium": st.session_state.w_medium, "Long": st.session_state.w_long}

st.sidebar.subheader("Portfolio")
portfolio_data = get_portfolio()
new_pot = st.sidebar.number_input("Total investment pot (GBP)", value=portfolio_data["total_pot"], step=10_000)
if new_pot != portfolio_data["total_pot"]:
    set_total_pot(new_pot)

# ========================= DATA FETCH =========================
with st.spinner("Loading market data..."):
    spot = fetch_spot_prices()
    etc_prices = fetch_etc_prices(selected_etcs) if selected_etcs else {}

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
tab_dashboard, tab_charts, tab_portfolio, tab_news = st.tabs([
    "Dashboard", "Charts", "Portfolio", "News"
])

# ─────────────────────── DASHBOARD TAB ───────────────────────
with tab_dashboard:
    # Compute signals for both metals
    with st.spinner("Analysing markets..."):
        gold_tf = fetch_multi_timeframe_data("GC=F")
        silver_tf = fetch_multi_timeframe_data("SI=F")
        gold_fib = multi_timeframe_fibonacci(gold_tf)
        silver_fib = multi_timeframe_fibonacci(silver_tf)
        gold_daily = gold_tf["long_term"]
        silver_daily = silver_tf["long_term"]

        gold_score = score_metal(gold_daily, gold_fib, spot["gold"]["price_usd"], tf_weight_config) if not gold_daily.empty else None
        silver_score = score_metal(silver_daily, silver_fib, spot["silver"]["price_usd"], tf_weight_config) if not silver_daily.empty else None

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

    # ── Conflicts (shown in indicator breakdown) ──

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
            # Chat inside the same container
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

    # ── Indicator Breakdown Tables ──
    for metal_name, sc in [("Gold", gold_score), ("Silver", silver_score)]:
        if not sc:
            continue
        with st.expander(f"{metal_name} Indicator Breakdown"):
            table_df = pd.DataFrame(sc["indicator_table"])
            # Map conflict boolean to visual marker
            table_df["Conflict"] = table_df["Conflict"].map({True: "\u26a0\ufe0f", False: ""})
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
            # Reorder columns to put Conflict early
            cols = ["Indicator", "Timeframe", "Vote", "Direction", "Weight", "Weighted Score", "Conflict", "Correlation Group", "Detail"]
            cols = [c for c in cols if c in table_df.columns]
            table_df = table_df[cols]
            styled = (table_df.style
                      .applymap(_color_vote, subset=["Vote"])
                      .applymap(_color_wscore, subset=["Weighted Score"])
                      .applymap(_color_direction, subset=["Direction"]))
            st.dataframe(styled, use_container_width=True, hide_index=True)
            # Show conflict explanations below the table
            if sc["conflicts"]:
                for c in sc["conflicts"]:
                    st.caption(
                        f"\u26a0\ufe0f **{c['group']}**: {c['ind_a']} ({c['vote_a']}) vs "
                        f"{c['ind_b']} ({c['vote_b']}) -- {c['explanation']}"
                    )

    # ── Fibonacci Levels ──
    for metal_name, fib_data in [("Gold", gold_fib), ("Silver", silver_fib)]:
        with st.expander(f"{metal_name} Fibonacci Levels"):
            for tf_name, levels in fib_data.items():
                st.write(f"**{tf_name.replace('_', ' ').title()}**")
                for ratio_key, value in levels.items():
                    if ratio_key in ("swing_high", "swing_low"):
                        continue
                    st.write(f"  {ratio_key}: ${value:,.0f}")

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

# ========================= FOOTER =========================
st.divider()
col_refresh, col_info = st.columns([1, 3])
with col_refresh:
    if st.button("Refresh All Data"):
        st.cache_data.clear()
        st.rerun()
with col_info:
    st.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')} | Data: Yahoo Finance | Not financial advice")
