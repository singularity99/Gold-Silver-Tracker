"""Professional dark trading terminal CSS and HTML templates."""

# ── Colour Palette ──────────────────────────────────────────────────
GOLD = "#D4A843"
SILVER = "#C0C0C0"
BG_DARK = "#0E1117"
BG_CARD = "#1A1D23"
BG_CARD_HOVER = "#22262E"
BORDER = "#2D3139"
GREEN = "#26A69A"
RED = "#EF5350"
AMBER = "#FFB300"
TEXT_PRIMARY = "#FAFAFA"
TEXT_SECONDARY = "#9CA3AF"
TEXT_MUTED = "#6B7280"

# ── Global CSS ──────────────────────────────────────────────────────
GLOBAL_CSS = f"""
<style>
/* ─── Base overrides ─── */
.stApp {{
    background-color: {BG_DARK};
}}
section[data-testid="stSidebar"] {{
    background-color: {BG_CARD};
    border-right: 1px solid {BORDER};
}}
h1, h2, h3 {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}}

/* ─── Price Ticker Strip ─── */
.ticker-strip {{
    display: flex;
    gap: 0;
    background: linear-gradient(180deg, #1E2128 0%, {BG_CARD} 100%);
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 0;
    margin-bottom: 1rem;
    overflow-x: auto;
    -webkit-overflow-scrolling: touch;
}}
.ticker-item {{
    flex: 1;
    min-width: 140px;
    padding: 10px 16px;
    border-right: 1px solid {BORDER};
    text-align: center;
}}
.ticker-item:last-child {{
    border-right: none;
}}
.ticker-label {{
    font-size: 0.7rem;
    color: {TEXT_SECONDARY};
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 2px;
}}
.ticker-price {{
    font-size: 1.3rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', 'SF Mono', monospace;
    color: {TEXT_PRIMARY};
}}
.ticker-change {{
    font-size: 0.75rem;
    font-family: 'JetBrains Mono', monospace;
    margin-top: 1px;
}}
.ticker-change.up {{ color: {GREEN}; }}
.ticker-change.down {{ color: {RED}; }}
.ticker-change.neutral {{ color: {TEXT_MUTED}; }}
.ticker-gold .ticker-price {{ color: {GOLD}; }}
.ticker-silver .ticker-price {{ color: {SILVER}; }}

/* ─── Signal Score Cards ─── */
.signal-card {{
    background: {BG_CARD};
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 20px;
    margin-bottom: 1rem;
    position: relative;
    overflow: hidden;
}}
.signal-card::before {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
}}
.signal-card.strong-buy::before {{ background: {GREEN}; }}
.signal-card.buy::before {{ background: {GREEN}; opacity: 0.6; }}
.signal-card.neutral::before {{ background: {AMBER}; }}
.signal-card.sell::before {{ background: {RED}; opacity: 0.6; }}
.signal-card.strong-sell::before {{ background: {RED}; }}

.signal-header {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
}}
.signal-metal {{
    font-size: 1.1rem;
    font-weight: 600;
    color: {TEXT_PRIMARY};
}}
.signal-label {{
    font-size: 0.8rem;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 4px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}}
.signal-label.strong-buy {{ background: rgba(38,166,154,0.2); color: {GREEN}; }}
.signal-label.buy {{ background: rgba(38,166,154,0.15); color: {GREEN}; }}
.signal-label.neutral {{ background: rgba(255,179,0,0.15); color: {AMBER}; }}
.signal-label.sell {{ background: rgba(239,83,80,0.15); color: {RED}; }}
.signal-label.strong-sell {{ background: rgba(239,83,80,0.2); color: {RED}; }}

.score-big {{
    font-size: 2.8rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    line-height: 1;
    margin-bottom: 12px;
}}
.score-big.positive {{ color: {GREEN}; }}
.score-big.negative {{ color: {RED}; }}
.score-big.flat {{ color: {AMBER}; }}

.sub-scores {{
    display: flex;
    gap: 8px;
    margin-bottom: 10px;
}}
.sub-score {{
    flex: 1;
    background: rgba(255,255,255,0.03);
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 8px;
    text-align: center;
}}
.sub-score-label {{
    font-size: 0.65rem;
    color: {TEXT_MUTED};
    text-transform: uppercase;
    letter-spacing: 0.3px;
}}
.sub-score-val {{
    font-size: 1.1rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    margin: 2px 0;
}}
.sub-score-signal {{
    font-size: 0.6rem;
    font-weight: 600;
    text-transform: uppercase;
}}

.warning-badge {{
    display: inline-block;
    background: rgba(239,83,80,0.15);
    color: {RED};
    font-size: 0.7rem;
    padding: 2px 8px;
    border-radius: 3px;
    margin-top: 4px;
}}
.improving-badge {{
    display: inline-block;
    background: rgba(38,166,154,0.15);
    color: {GREEN};
    font-size: 0.7rem;
    padding: 2px 8px;
    border-radius: 3px;
    margin-top: 4px;
}}

/* ─── ETC Price Tiles ─── */
.etc-grid {{
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    margin-bottom: 1rem;
}}
.etc-tile {{
    flex: 1;
    min-width: 120px;
    background: {BG_CARD};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 10px 12px;
    text-align: center;
}}
.etc-ticker {{
    font-size: 0.7rem;
    color: {TEXT_SECONDARY};
    font-weight: 600;
}}
.etc-price {{
    font-size: 1.1rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    color: {TEXT_PRIMARY};
    margin: 2px 0;
}}
.etc-change {{
    font-size: 0.7rem;
    font-family: 'JetBrains Mono', monospace;
}}

/* ─── News Cards ─── */
.news-card {{
    background: {BG_CARD};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 14px 16px;
    margin-bottom: 10px;
}}
.news-meta {{
    font-size: 0.7rem;
    color: {TEXT_MUTED};
    margin-bottom: 4px;
}}
.news-title {{
    font-size: 0.9rem;
    font-weight: 600;
    color: {TEXT_PRIMARY};
    margin-bottom: 6px;
}}
.news-summary {{
    font-size: 0.8rem;
    color: {TEXT_SECONDARY};
    line-height: 1.4;
    margin-bottom: 6px;
}}
.news-link {{
    font-size: 0.75rem;
    color: {GOLD};
}}
.news-link:hover {{
    color: #E8C14A;
}}

/* ─── Portfolio Cards ─── */
.port-stat {{
    background: {BG_CARD};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 12px 16px;
    text-align: center;
}}
.port-stat-label {{
    font-size: 0.65rem;
    color: {TEXT_MUTED};
    text-transform: uppercase;
    letter-spacing: 0.3px;
}}
.port-stat-val {{
    font-size: 1.4rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    color: {TEXT_PRIMARY};
    margin: 2px 0;
}}

/* ─── Compact Metrics Override ─── */
[data-testid="stMetricValue"] {{
    font-family: 'JetBrains Mono', 'SF Mono', monospace !important;
}}
[data-testid="stMetricLabel"] {{
    font-size: 0.75rem !important;
}}

/* ─── Tab Styling ─── */
.stTabs [data-baseweb="tab-list"] {{
    gap: 0;
    background: {BG_CARD};
    border-radius: 6px;
    padding: 2px;
    border: 1px solid {BORDER};
    overflow-x: auto;
    overflow-y: hidden;
    white-space: nowrap;
}}
.stTabs [data-baseweb="tab"] {{
    border-radius: 4px;
    padding: 8px 20px;
    font-weight: 600;
    font-size: 0.85rem;
    flex: 0 0 auto;
}}
.stTabs [aria-selected="true"] {{
    background: rgba(212,168,67,0.15) !important;
    color: {GOLD} !important;
}}

/* ─── Expander styling ─── */
.streamlit-expanderHeader {{
    background: {BG_CARD} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 6px !important;
}}

/* ─── Mobile Responsive ─── */
@media (max-width: 768px) {{
    .block-container {{
        padding-left: 0.75rem !important;
        padding-right: 0.75rem !important;
    }}
    .stTabs [data-baseweb="tab"] {{
        padding: 8px 12px;
        font-size: 0.8rem;
    }}
    .signal-card {{
        padding: 14px;
    }}
    .signal-header {{
        flex-wrap: wrap;
        gap: 8px;
        align-items: flex-start;
    }}
    .signal-metal {{
        font-size: 1rem;
    }}
    .signal-label {{
        font-size: 0.72rem;
        padding: 2px 8px;
    }}
    .score-big {{
        font-size: 2rem;
    }}
    .sub-scores {{
        flex-direction: column;
    }}
    .etc-grid {{
        flex-direction: column;
    }}
    .news-card {{
        padding: 10px 12px;
    }}
}}
</style>
"""


# ── Component CSS (for st.html iframe rendering) ───────────────────
COMPONENT_CSS = f"""
<style>
body {{ margin: 0; padding: 0; background: transparent; font-family: 'Inter', -apple-system, sans-serif; color: {TEXT_PRIMARY}; }}
.ticker-desktop {{ display:block; }}
.ticker-mobile {{ display:none; }}
.ticker-tbl {{ width:100%; border-collapse:collapse; background:linear-gradient(180deg,#1E2128 0%,{BG_CARD} 100%); border:1px solid {BORDER}; border-radius:6px; overflow:hidden; }}
.ticker-tbl td {{ padding:8px 14px; text-align:center; vertical-align:middle; border-right:1px solid {BORDER}; border-bottom:1px solid {BORDER}; }}
.ticker-tbl tr:last-child td {{ border-bottom:none; }}
.ticker-tbl td:last-child {{ border-right:none; }}
.ticker-tbl td.metal-label {{ font-size:1.4rem; font-weight:800; letter-spacing:1px; width:80px; }}
.ticker-tbl td.metal-label.gold {{ color:{GOLD}; }}
.ticker-tbl td.metal-label.silver {{ color:{SILVER}; }}
.ticker-tbl td.divider {{ border-right:2px solid {BORDER}; }}
.ticker-label {{ font-size:0.65rem; color:{TEXT_SECONDARY}; text-transform:uppercase; letter-spacing:0.5px; margin-bottom:1px; }}
.ticker-price {{ font-size:1.2rem; font-weight:700; font-family:'JetBrains Mono',monospace; color:{TEXT_PRIMARY}; }}
.ticker-change {{ font-size:0.7rem; font-family:'JetBrains Mono',monospace; margin-top:1px; }}
.ticker-change.up {{ color:{GREEN}; }} .ticker-change.down {{ color:{RED}; }} .ticker-change.neutral {{ color:{TEXT_MUTED}; }}
.ticker-gold .ticker-price {{ color:{GOLD}; }} .ticker-silver .ticker-price {{ color:{SILVER}; }}

.ticker-mobile-card {{ background:linear-gradient(180deg,#1E2128 0%,{BG_CARD} 100%); border:1px solid {BORDER}; border-radius:8px; padding:10px; }}
.ticker-mobile-head {{ font-size:0.95rem; font-weight:700; letter-spacing:0.7px; margin-bottom:8px; }}
.ticker-mobile-head.gold {{ color:{GOLD}; }}
.ticker-mobile-head.silver {{ color:{SILVER}; }}
.ticker-mobile-head.fx {{ color:{TEXT_SECONDARY}; }}
.ticker-mobile-grid {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:8px; }}
.ticker-mobile-cell {{ background:rgba(255,255,255,0.03); border:1px solid {BORDER}; border-radius:6px; padding:8px; text-align:center; }}
.ticker-mobile-cell .ticker-price {{ font-size:1rem; }}
@media (max-width: 768px) {{
    .ticker-desktop {{ display:none; }}
    .ticker-mobile {{ display:grid; gap:8px; margin-bottom:0.5rem; }}
}}

.signal-card {{ background:{BG_CARD}; border:1px solid {BORDER}; border-radius:8px; padding:20px; position:relative; }}
.signal-card::before {{ content:''; position:absolute; top:0; left:0; right:0; height:3px; }}
.signal-card.strong-buy::before {{ background:{GREEN}; }}
.signal-card.buy::before {{ background:{GREEN}; opacity:0.6; }}
.signal-card.neutral::before {{ background:{AMBER}; }}
.signal-card.sell::before {{ background:{RED}; opacity:0.6; }}
.signal-card.strong-sell::before {{ background:{RED}; }}
.signal-header {{ display:flex; justify-content:space-between; align-items:center; margin-bottom:12px; }}
.signal-metal {{ font-size:1.1rem; font-weight:600; }}
.signal-label {{ font-size:0.8rem; font-weight:600; padding:3px 10px; border-radius:4px; text-transform:uppercase; letter-spacing:0.5px; }}
.signal-label.strong-buy {{ background:rgba(38,166,154,0.2); color:{GREEN}; }}
.signal-label.buy {{ background:rgba(38,166,154,0.15); color:{GREEN}; }}
.signal-label.neutral {{ background:rgba(255,179,0,0.15); color:{AMBER}; }}
.signal-label.sell {{ background:rgba(239,83,80,0.15); color:{RED}; }}
.signal-label.strong-sell {{ background:rgba(239,83,80,0.2); color:{RED}; }}
.score-big {{ font-size:2.8rem; font-weight:700; font-family:'JetBrains Mono',monospace; line-height:1; margin-bottom:12px; }}
.score-big.positive {{ color:{GREEN}; }} .score-big.negative {{ color:{RED}; }} .score-big.flat {{ color:{AMBER}; }}
.sub-scores {{ display:flex; gap:8px; margin-bottom:10px; }}
.sub-score {{ flex:1; background:rgba(255,255,255,0.03); border:1px solid {BORDER}; border-radius:6px; padding:8px; text-align:center; }}
.sub-score-label {{ font-size:0.65rem; color:{TEXT_MUTED}; text-transform:uppercase; letter-spacing:0.3px; }}
.sub-score-val {{ font-size:1.1rem; font-weight:700; font-family:'JetBrains Mono',monospace; margin:2px 0; }}
.sub-score-signal {{ font-size:0.6rem; font-weight:600; text-transform:uppercase; }}
.badge-wrap {{ display:inline-block; cursor:pointer; }}
.warning-badge {{ display:inline-block; background:rgba(239,83,80,0.15); color:{RED}; font-size:0.7rem; padding:2px 8px; border-radius:3px; margin-top:4px; }}
.improving-badge {{ display:inline-block; background:rgba(38,166,154,0.15); color:{GREEN}; font-size:0.7rem; padding:2px 8px; border-radius:3px; margin-top:4px; }}
.badge-panel {{ max-height:0; overflow:hidden; transition:max-height 0.3s ease; background:rgba(30,33,40,0.95); border-radius:6px; font-size:0.75rem; line-height:1.5; color:{TEXT_SECONDARY}; }}
.badge-wrap:hover .badge-panel {{ max-height:500px; padding:10px 12px; margin-top:6px; border:1px solid {BORDER}; }}
.conflict-badge {{ display:inline-block; background:rgba(255,179,0,0.15); color:{AMBER}; font-size:0.7rem; padding:2px 8px; border-radius:3px; margin-top:4px; }}

.etc-grid {{ display:flex; gap:8px; flex-wrap:wrap; }}
.etc-tile {{ flex:1; min-width:120px; background:{BG_CARD}; border:1px solid {BORDER}; border-radius:6px; padding:10px 12px; text-align:center; }}
.etc-ticker {{ font-size:0.7rem; color:{TEXT_SECONDARY}; font-weight:600; }}
.etc-price {{ font-size:1.1rem; font-weight:700; font-family:'JetBrains Mono',monospace; color:{TEXT_PRIMARY}; margin:2px 0; }}
.etc-change {{ font-size:0.7rem; font-family:'JetBrains Mono',monospace; }}

.news-card {{ background:{BG_CARD}; border:1px solid {BORDER}; border-radius:6px; padding:14px 16px; margin-bottom:10px; }}
.news-meta {{ font-size:0.7rem; color:{TEXT_MUTED}; margin-bottom:4px; }}
.news-title {{ font-size:0.9rem; font-weight:600; color:{TEXT_PRIMARY}; margin-bottom:6px; }}
.news-summary {{ font-size:0.8rem; color:{TEXT_SECONDARY}; line-height:1.4; margin-bottom:6px; }}
.news-link {{ font-size:0.75rem; color:{GOLD}; text-decoration:none; }}

.port-stat {{ flex:1; min-width:120px; background:{BG_CARD}; border:1px solid {BORDER}; border-radius:6px; padding:12px 16px; text-align:center; }}
.port-stat-label {{ font-size:0.65rem; color:{TEXT_MUTED}; text-transform:uppercase; letter-spacing:0.3px; }}
.port-stat-val {{ font-size:1.4rem; font-weight:700; font-family:'JetBrains Mono',monospace; color:{TEXT_PRIMARY}; margin:2px 0; }}

.cards-row {{ display:flex; gap:12px; flex-wrap:wrap; }}
.cards-row > div {{ flex:1; min-width:300px; }}

@media (max-width: 768px) {{
    .signal-card {{ padding:14px; }}
    .signal-header {{ flex-wrap:wrap; gap:8px; align-items:flex-start; }}
    .signal-metal {{ font-size:1rem; }}
    .signal-label {{ font-size:0.72rem; padding:2px 8px; }}
    .score-big {{ font-size:2rem; }}
    .sub-scores {{ flex-direction:column; }}
    .etc-grid, .cards-row {{ flex-direction:column; }}
    .cards-row > div {{ min-width:0; }}
    .badge-wrap {{ display:block; margin-top:4px; }}
    .badge-wrap:hover .badge-panel, .badge-wrap:active .badge-panel {{ max-height:600px; padding:10px 12px; margin-top:6px; border:1px solid {BORDER}; }}
}}
</style>
"""


def render_component(html_content: str, height: int = None) -> str:
    """Wrap HTML content with component CSS for st.html rendering."""
    return COMPONENT_CSS + html_content


# ── HTML Template Helpers ───────────────────────────────────────────

def ticker_strip_html(gold_usd, gold_chg, silver_usd, silver_chg,
                      gs_ratio, gbp_usd, gold_gbp=None, silver_gbp=None,
                      gold_inr=None, silver_inr=None, usd_inr=None) -> str:
    def _chg_class(val):
        if val > 0: return "up"
        if val < 0: return "down"
        return "neutral"

    def _inr_fmt(val):
        if val and val == val:
            if val >= 1e7:
                return f"\u20b9{val / 1e7:.2f} Cr/kg"
            elif val >= 1e5:
                return f"\u20b9{val / 1e5:.2f} L/kg"
            else:
                return f"\u20b9{val:,.0f}/kg"
        return "N/A"

    def _td(cls, label, price, change="", chg_cls="neutral"):
        return f"""<td class="{cls}">
            <div class="ticker-label">{label}</div>
            <div class="ticker-price">{price}</div>
            <div class="ticker-change {chg_cls}">{change}</div>
        </td>"""

    def _mobile_cell(label, price, change="", chg_cls="neutral"):
        chg = change if change else "&nbsp;"
        return f"""<div class="ticker-mobile-cell">
            <div class="ticker-label">{label}</div>
            <div class="ticker-price">{price}</div>
            <div class="ticker-change {chg_cls}">{chg}</div>
        </div>"""

    gold_chg_str = f"{gold_chg:+.1f}%" if gold_chg == gold_chg else ""
    gold_chg_cls = _chg_class(gold_chg)
    silver_chg_str = f"{silver_chg:+.1f}%" if silver_chg == silver_chg else ""
    silver_chg_cls = _chg_class(silver_chg)
    usd_inr_val = usd_inr if usd_inr and usd_inr == usd_inr else None
    gbp_inr_val = (gbp_usd * usd_inr) if usd_inr_val and gbp_usd == gbp_usd else None

    # Desktop layout: single table keeps row alignment exact
    desktop_html = '<table class="ticker-tbl">'
    # Row 1: Gold + G/S Ratio + GBP/USD
    desktop_html += '<tr>'
    desktop_html += '<td class="metal-label gold">GOLD</td>'
    desktop_html += _td("ticker-gold", "USD/toz", f"${gold_usd:,.0f}" if gold_usd == gold_usd else "N/A", gold_chg_str, gold_chg_cls)
    desktop_html += _td("ticker-gold", "GBP/toz", f"\u00a3{gold_gbp:,.0f}" if gold_gbp and gold_gbp == gold_gbp else "N/A", gold_chg_str, gold_chg_cls)
    desktop_html += _td("ticker-gold divider", "INR/kg", _inr_fmt(gold_inr) if gold_inr and gold_inr == gold_inr else "N/A", gold_chg_str, gold_chg_cls)
    desktop_html += _td("", "G/S Ratio", f"{gs_ratio:.1f}:1" if gs_ratio == gs_ratio else "N/A", "&nbsp;")
    desktop_html += _td("", "GBP/USD", f"{gbp_usd:.4f}" if gbp_usd == gbp_usd else "N/A", "&nbsp;")
    desktop_html += '</tr>'
    # Row 2: Silver + USD/INR + GBP/INR
    desktop_html += '<tr>'
    desktop_html += '<td class="metal-label silver">SILVER</td>'
    desktop_html += _td("ticker-silver", "USD/toz", f"${silver_usd:.2f}" if silver_usd == silver_usd else "N/A", silver_chg_str, silver_chg_cls)
    desktop_html += _td("ticker-silver", "GBP/toz", f"\u00a3{silver_gbp:.2f}" if silver_gbp and silver_gbp == silver_gbp else "N/A", silver_chg_str, silver_chg_cls)
    desktop_html += _td("ticker-silver divider", "INR/kg", _inr_fmt(silver_inr) if silver_inr and silver_inr == silver_inr else "N/A", silver_chg_str, silver_chg_cls)
    desktop_html += _td("", "USD/INR", f"\u20b9{usd_inr_val:.2f}" if usd_inr_val else "N/A", "&nbsp;")
    desktop_html += _td("", "GBP/INR", f"\u20b9{gbp_inr_val:.2f}" if gbp_inr_val else "N/A", "&nbsp;")
    desktop_html += '</tr>'
    desktop_html += '</table>'

    # Mobile layout: stacked cards with larger tap-friendly cells
    mobile_html = '<div class="ticker-mobile">'
    mobile_html += '<div class="ticker-mobile-card">'
    mobile_html += '<div class="ticker-mobile-head gold">GOLD</div>'
    mobile_html += '<div class="ticker-mobile-grid">'
    mobile_html += _mobile_cell("USD/toz", f"${gold_usd:,.0f}" if gold_usd == gold_usd else "N/A", gold_chg_str, gold_chg_cls)
    mobile_html += _mobile_cell("GBP/toz", f"\u00a3{gold_gbp:,.0f}" if gold_gbp and gold_gbp == gold_gbp else "N/A", gold_chg_str, gold_chg_cls)
    mobile_html += _mobile_cell("INR/kg", _inr_fmt(gold_inr) if gold_inr and gold_inr == gold_inr else "N/A", gold_chg_str, gold_chg_cls)
    mobile_html += _mobile_cell("G/S Ratio", f"{gs_ratio:.1f}:1" if gs_ratio == gs_ratio else "N/A")
    mobile_html += '</div></div>'

    mobile_html += '<div class="ticker-mobile-card">'
    mobile_html += '<div class="ticker-mobile-head silver">SILVER</div>'
    mobile_html += '<div class="ticker-mobile-grid">'
    mobile_html += _mobile_cell("USD/toz", f"${silver_usd:.2f}" if silver_usd == silver_usd else "N/A", silver_chg_str, silver_chg_cls)
    mobile_html += _mobile_cell("GBP/toz", f"\u00a3{silver_gbp:.2f}" if silver_gbp and silver_gbp == silver_gbp else "N/A", silver_chg_str, silver_chg_cls)
    mobile_html += _mobile_cell("INR/kg", _inr_fmt(silver_inr) if silver_inr and silver_inr == silver_inr else "N/A", silver_chg_str, silver_chg_cls)
    mobile_html += _mobile_cell("USD/INR", f"\u20b9{usd_inr_val:.2f}" if usd_inr_val else "N/A")
    mobile_html += '</div></div>'

    mobile_html += '<div class="ticker-mobile-card">'
    mobile_html += '<div class="ticker-mobile-head fx">FX</div>'
    mobile_html += '<div class="ticker-mobile-grid">'
    mobile_html += _mobile_cell("GBP/USD", f"{gbp_usd:.4f}" if gbp_usd == gbp_usd else "N/A")
    mobile_html += _mobile_cell("GBP/INR", f"\u20b9{gbp_inr_val:.2f}" if gbp_inr_val else "N/A")
    mobile_html += '</div></div>'
    mobile_html += '</div>'

    return f'<div class="ticker-desktop">{desktop_html}</div>{mobile_html}'


def signal_card_html(metal, score, price_usd) -> str:
    comp = score["composite_score"]
    sig = score["signal"]
    sig_cls = sig.lower().replace(" / take profit", "").replace(" ", "-")
    score_cls = "positive" if comp > 0.05 else ("negative" if comp < -0.05 else "flat")

    tf_scores = score["timeframe_scores"]
    rows = score.get("indicator_table", [])
    det_count = sum(1 for r in rows if "Deteriorating" in r.get("Direction", ""))
    imp_count = sum(1 for r in rows if "Improving" in r.get("Direction", ""))

    metal_color = GOLD if metal == "gold" else SILVER

    def _sub(tf, label, weight):
        val = round(tf_scores[tf], 2)
        if val >= 0.40: slabel, scol = "Strong Buy", GREEN
        elif val >= 0.20: slabel, scol = "Buy", GREEN
        elif val <= -0.40: slabel, scol = "Strong Sell", RED
        elif val <= -0.20: slabel, scol = "Sell", RED
        else: slabel, scol = "Neutral", AMBER
        vcol = GREEN if val > 0.05 else (RED if val < -0.05 else AMBER)
        return f"""
        <div class="sub-score">
            <div class="sub-score-label">{label} ({weight}%)</div>
            <div class="sub-score-val" style="color:{vcol}">{val:+.2f}</div>
            <div class="sub-score-signal" style="color:{scol}">{slabel}</div>
        </div>"""

    det_rows = [r for r in rows if "Deteriorating" in r.get("Direction", "")]
    imp_rows = [r for r in rows if "Improving" in r.get("Direction", "")]

    def _tooltip_lines(indicator_rows, color):
        lines = ""
        for r in indicator_rows:
            vote = r.get("Vote", "Neutral")
            vcol = GREEN if vote == "Bullish" else (RED if vote == "Bearish" else AMBER)
            name = r.get("Indicator", "")
            tf = r.get("Timeframe", "")
            detail = r.get("Detail", "")[:80]
            lines += f'<div style="margin-bottom:4px;"><span style="color:{vcol};font-weight:600;">{vote}</span> <span style="color:{TEXT_PRIMARY}">{name}</span> <span style="color:{TEXT_MUTED}">({tf})</span><br/><span style="color:{TEXT_MUTED}">{detail}</span></div>'
        return lines

    badges = ""
    if det_count > 0:
        det_tooltip = _tooltip_lines(det_rows, RED)
        badges += f'<div class="badge-wrap"><span class="warning-badge">{det_count} deteriorating</span><div class="badge-panel">{det_tooltip}</div></div> '
    if imp_count > 0:
        imp_tooltip = _tooltip_lines(imp_rows, GREEN)
        badges += f'<div class="badge-wrap"><span class="improving-badge">{imp_count} improving</span><div class="badge-panel">{imp_tooltip}</div></div>'

    # Conflict badge
    conflicts = score.get("conflicts", [])
    if conflicts:
        conflict_tooltip = ""
        for c in conflicts:
            conflict_tooltip += (
                f'<div style="margin-bottom:4px;">'
                f'<span style="color:{GREEN if c["vote_a"] == "bullish" else RED}">{c["ind_a"]}</span> vs '
                f'<span style="color:{GREEN if c["vote_b"] == "bullish" else RED}">{c["ind_b"]}</span></div>'
            )
        badges += f' <div class="badge-wrap"><span class="conflict-badge">{len(conflicts)} conflicting</span><div class="badge-panel">{conflict_tooltip}</div></div>'

    overlay_note = ""
    if score.get("macro_overlay"):
        mo = score["macro_overlay"]
        phase = mo.get("phase", "Unknown")
        bias = mo.get("bias", 0.0)
        raw_score = score.get("raw_composite_score", comp)
        overlay_note = (
            f'<div style="font-size:0.7rem;color:{TEXT_MUTED};margin-top:6px;">'
            f'Macro overlay: {phase} (bias {bias:+.2f}) | Raw {raw_score:+.2f}'
            f'</div>'
        )

    return f"""
    <div class="signal-card {sig_cls}">
        <div class="signal-header">
            <span class="signal-metal" style="color:{metal_color}">{metal.upper()} &mdash; ${price_usd:,.0f}</span>
            <span class="signal-label {sig_cls}">{sig}</span>
        </div>
        <div style="font-size:0.7rem;color:{TEXT_MUTED};margin-bottom:2px;">WEIGHTED COMPOSITE SCORE</div>
        <div class="score-big {score_cls}">{comp:+.2f}</div>
        <div class="sub-scores">
            {_sub("Short", "Short", score["timeframe_weights"]["Short"])}
            {_sub("Medium", "Medium", score["timeframe_weights"]["Medium"])}
            {_sub("Long", "Long", score["timeframe_weights"]["Long"])}
        </div>
        {overlay_note}
        <div>{badges}</div>
    </div>"""


def etc_tile_html(ticker, price_str, change_pct) -> str:
    chg_col = GREEN if change_pct > 0 else (RED if change_pct < 0 else TEXT_MUTED)
    return f"""
    <div class="etc-tile">
        <div class="etc-ticker">{ticker}</div>
        <div class="etc-price">{price_str}</div>
        <div class="etc-change" style="color:{chg_col}">{change_pct:+.1f}%</div>
    </div>"""


def analysis_card_html(metal: str, summary_text: str) -> str:
    """Render market analysis summary in a styled card."""
    metal_color = GOLD if metal.lower() == "gold" else SILVER
    # Convert markdown-like formatting to HTML
    import re
    html = summary_text
    # Bold
    html = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html)
    # Line breaks
    html = html.replace('\n\n', '</p><p>').replace('\n- ', f'<br/><span style="color:{TEXT_MUTED};">&bull;</span> ')
    # Wrap sections
    sections = html.split('</p><p>')
    out = ""
    for i, section in enumerate(sections):
        section = section.strip()
        if not section:
            continue
        if section.startswith('<strong>Action'):
            out += f'<div style="margin-top:10px;padding:10px;background:rgba(255,255,255,0.03);border-left:3px solid {metal_color};border-radius:0 4px 4px 0;">{section}</div>'
        elif 'Convergence' in section or 'Recovery' in section:
            out += f'<div style="margin-top:8px;padding:8px 10px;background:rgba(255,179,0,0.08);border-radius:4px;font-size:0.85rem;">{section}</div>'
        elif 'Active conflicts' in section:
            out += f'<div style="margin-top:8px;padding:8px 10px;background:rgba(239,83,80,0.08);border-radius:4px;font-size:0.85rem;">{section}</div>'
        elif i == 0:
            out += f'<div style="font-size:1.05rem;margin-bottom:10px;color:{TEXT_PRIMARY};">{section}</div>'
        else:
            out += f'<div style="font-size:0.85rem;color:{TEXT_SECONDARY};margin-bottom:6px;line-height:1.5;">{section}</div>'

    return f"""
    <div style="padding:4px 0;border-top:3px solid {metal_color};margin-bottom:0;">
        {out}
    </div>"""


def news_card_html(published, source, title, summary_text, link) -> str:
    summary_html = f'<div class="news-summary">{summary_text[:400]}{"..." if len(summary_text)>400 else ""}</div>' if summary_text else ""
    return f"""
    <div class="news-card">
        <div class="news-meta">{published} &bull; {source}</div>
        <div class="news-title">&ldquo;{title}&rdquo;</div>
        {summary_html}
        <a class="news-link" href="{link}" target="_blank">Read full article &rarr;</a>
    </div>"""


def port_stat_html(label, value) -> str:
    return f"""
    <div class="port-stat">
        <div class="port-stat-label">{label}</div>
        <div class="port-stat-val">{value}</div>
    </div>"""
