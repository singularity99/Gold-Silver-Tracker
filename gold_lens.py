"""
Gold Macro Lens - the macro backdrop for gold specifically.

The Zeberg framework in macro.py answers "where are we in the business
cycle". It carries no gold-specific input. This module answers a narrower
question: is the macro backdrop working for or against gold right now.

Four series, because gold's price is mechanically a function of them:

    DFII10   10-yr real yield   - what cash pays you to not hold gold
    T10YIE   10-yr breakeven    - the inflation gold is meant to hedge
    DXY      dollar index       - the currency gold is priced in
    WALCL    Fed balance sheet  - QT drains liquidity, QE adds it

The primary read is directional. A quadrant on the 3-month change in the
real yield against the 3-month change in the breakeven places the backdrop
in one of four regimes; the driver rows beneath give the levels behind it.

This module reports. It makes no buy or sell inference, and deliberately
publishes no price target - see CAVEAT below for why that restraint is not
just conservatism.
"""

import numpy as np
import pandas as pd
import yfinance as yf

from macro import _fetch_fred_series, _slice_asof, _to_monthly

# Bump when the lens changes how a reading is computed, so a deploy
# invalidates the cached state (see FRAMEWORK_VERSION in macro.py).
LENS_VERSION = "2026.10.08-goldlens-2"

MOMENTUM_MONTHS = 3     # window for the quadrant's change measure
TRAIL_MONTHS = 15       # how much history the quadrant trail shows

# A 3-month change smaller than this reads as flat. Per series, because the
# units differ: 0.1pp is a real move in a yield, 0.1 index points is noise in
# the dollar, and $50bn is the smallest balance-sheet shift worth a label.
NEUTRAL_BAND = {
    "real_yield": 0.10,     # percentage points
    "breakeven": 0.10,      # percentage points
    "dollar": 1.00,         # index points
    "balance_sheet": 0.05,  # $tn
}

FRED_SERIES = {
    "real_yield": "DFII10",
    "breakeven": "T10YIE",
    "balance_sheet": "WALCL",
}
DXY_TICKER = "DX-Y.NYB"

# Thresholds carried over from the plan being operationalised. They are
# display context, not triggers: nothing here acts on them.
BANDS = {
    "real_yield": [2.0, 2.5, 3.0],
    "breakeven": [2.0, 2.5],
    "dollar": [97.0, 100.0],
    "balance_sheet": [],
}


def _fetch_dxy() -> pd.Series:
    try:
        h = yf.Ticker(DXY_TICKER).history(period="10y")["Close"].dropna()
        if h.empty:
            return pd.Series(dtype=float)
        h.index = pd.to_datetime(h.index).tz_localize(None)
        return h
    except Exception:
        return pd.Series(dtype=float)


def fetch_gold_lens_bundle() -> dict[str, pd.Series]:
    b = {k: _fetch_fred_series(sid) for k, sid in FRED_SERIES.items()}
    if not b["balance_sheet"].empty:
        b["balance_sheet"] = b["balance_sheet"] / 1e6   # millions -> $tn
    b["dollar"] = _fetch_dxy()
    return b


# ── Driver rows ───────────────────────────────────────────────────────────────

def _change(s: pd.Series, days: int) -> float:
    if s is None or s.empty:
        return np.nan
    cut = s.index[-1] - pd.Timedelta(days=days)
    prior = s[s.index <= cut]
    if prior.empty:
        return np.nan
    return float(s.iloc[-1]) - float(prior.iloc[-1])


def _percentile_5y(s: pd.Series) -> tuple[float, float, float]:
    """Where the latest value sits in its own 5-year distribution."""
    if s is None or s.empty:
        return np.nan, np.nan, np.nan
    w = s[s.index >= s.index[-1] - pd.Timedelta(days=1825)]
    if w.empty:
        return np.nan, np.nan, np.nan
    last = float(s.iloc[-1])
    return float((w < last).mean() * 100), float(w.min()), float(w.max())


def _spark(s: pd.Series, years: int = 3, n: int = 30) -> list[float]:
    if s is None or s.empty:
        return []
    w = s[s.index >= s.index[-1] - pd.Timedelta(days=365 * years)].dropna()
    if w.empty:
        return []
    idx = np.unique(np.linspace(0, len(w) - 1, min(n, len(w))).astype(int))
    return [round(float(v), 4) for v in w.iloc[idx]]


def _band_text(key: str, stance_note: str | None = None) -> str:
    if stance_note:
        return stance_note
    bands = BANDS.get(key, [])
    return "bands " + " / ".join(f"{b:g}" for b in bands) if bands else ""


def _driver_row(key, series, name, role, dp, unit, higher_helps_gold, band_note=None) -> dict:
    s = _slice_asof(series)
    if s.empty:
        return {"key": key, "name": name, "role": role, "value": "—", "unit": unit,
                "raw": np.nan, "chg3m": np.nan, "chg12m": np.nan, "pct5y": np.nan,
                "min5y": np.nan, "max5y": np.nan, "chip": "No data", "stance": "none",
                "band": "", "spark": [], "asof": ""}

    raw = float(s.iloc[-1])
    chg3m = _change(s, 91)
    chg12m = _change(s, 365)
    pct, lo, hi = _percentile_5y(s)

    # Direction, not level, decides the chip. The level correlation between
    # gold and these series is unreliable (see CAVEAT); the change
    # correlation is the part that holds up.
    if np.isnan(chg3m) or abs(chg3m) < NEUTRAL_BAND.get(key, 0.10):
        chip, stance = "Neutral", "neutral"
    else:
        helping = (chg3m > 0) if higher_helps_gold else (chg3m < 0)
        chip, stance = ("Tailwind", "good") if helping else ("Headwind", "bad")

    return {
        "key": key, "name": name, "role": role,
        "value": f"{raw:,.{dp}f}{unit}", "unit": unit, "raw": round(raw, 4),
        "chg3m": None if np.isnan(chg3m) else round(chg3m, 3),
        "chg12m": None if np.isnan(chg12m) else round(chg12m, 3),
        "pct5y": None if np.isnan(pct) else round(pct, 1),
        "min5y": None if np.isnan(lo) else round(lo, 3),
        "max5y": None if np.isnan(hi) else round(hi, 3),
        "chip": chip, "stance": stance,
        "band": _band_text(key, band_note),
        "spark": _spark(s),
        "asof": str(s.index[-1].date()),
        "dp": dp,
    }


# ── Quadrant ──────────────────────────────────────────────────────────────────

# Short enough for a st.metric value, which truncates around 16 characters.
# The chart quadrants carry the fuller labels.
QUADRANTS = {
    # (real yield falling?, breakeven rising?) -> name, stance
    (True, True): ("Debasement", "supportive"),
    (True, False): ("Disinflation", "supportive"),
    (False, True): ("Reflation", "hostile"),
    (False, False): ("Real-rate shock", "hostile"),
}


def _quadrant_trail(real: pd.Series, bei: pd.Series) -> list[dict]:
    """Month-end 3-month changes, as the path through the quadrant."""
    r = _to_monthly(real)
    b = _to_monthly(bei)
    if r.empty or b.empty:
        return []
    df = pd.concat([r.rename("r"), b.rename("b")], axis=1).dropna()
    df["dr"] = df["r"].diff(MOMENTUM_MONTHS)
    df["db"] = df["b"].diff(MOMENTUM_MONTHS)
    t = df.dropna().tail(TRAIL_MONTHS)
    return [{"m": d.strftime("%b %y"), "dr": round(float(x.dr), 3), "db": round(float(x.db), 3)}
            for d, x in t.iterrows()]


def _quadrant(real: pd.Series, bei: pd.Series) -> dict:
    trail = _quadrant_trail(real, bei)
    if not trail:
        return {"name": "No data", "stance": "none", "dr": None, "db": None,
                "real_share": None, "trail": [], "note": "Insufficient history."}

    now = trail[-1]
    dr, db = now["dr"], now["db"]
    name, stance = QUADRANTS[(dr < 0, db > 0)]

    # How much of the nominal move is real. Reported rather than used to
    # rename the quadrant: the quadrant is defined by sign, and overriding
    # that with a judgement is exactly what this lens is meant to avoid.
    nominal = abs(dr) + abs(db)
    real_share = (abs(dr) / nominal) if nominal > 1e-9 else None

    direction = "falling" if dr < 0 else "rising"
    hedge = "bid" if db > 0 else "unwanted"
    note = (f"Real yields {direction} {abs(dr):+.2f}pp over {MOMENTUM_MONTHS}m "
            f"against breakevens {db:+.2f}pp — inflation hedge {hedge}.")
    if real_share is not None:
        note += f" {real_share * 100:.0f}% of the nominal move is real."

    return {"name": name, "stance": stance, "dr": dr, "db": db,
            "real_share": None if real_share is None else round(real_share, 3),
            "trail": trail, "note": note}


# ── The caveat, computed rather than asserted ─────────────────────────────────

def _diagnostics(real: pd.Series, gold: pd.Series) -> dict:
    """Does the textbook real-yield/gold relationship actually hold?

    It does not, in levels. Central-bank buying has driven gold and real
    yields up together. The inverse relation survives only in changes, which
    is why the lens reads direction and refuses to imply a price.
    """
    if real is None or real.empty or gold is None or gold.empty:
        return {}
    j = pd.concat([real.rename("r"), gold.rename("g")], axis=1).dropna()
    j = j[j.index >= j.index[-1] - pd.Timedelta(days=1825)]
    if len(j) < 60:
        return {}
    m = j.resample("ME").last().dropna()
    return {
        "corr_levels": round(float(j["r"].corr(j["g"])), 3),
        "corr_monthly_chg": round(float(m["r"].diff().corr(m["g"].diff())), 3),
        "n": int(len(j)),
    }


def _fetch_gold() -> pd.Series:
    """Gold itself, only ever used to test whether this lens earns its place."""
    try:
        h = yf.Ticker("GC=F").history(period="5y")["Close"].dropna()
        if h.empty:
            return pd.Series(dtype=float)
        h.index = pd.to_datetime(h.index).tz_localize(None)
        return h
    except Exception:
        return pd.Series(dtype=float)


def get_gold_lens_state(bundle: dict[str, pd.Series] | None = None,
                        gold: pd.Series | None = None) -> dict:
    """Current gold macro backdrop. Observational only."""
    b = bundle or fetch_gold_lens_bundle()
    if gold is None:
        gold = _fetch_gold()

    real = _slice_asof(b.get("real_yield", pd.Series(dtype=float)))
    bei = _slice_asof(b.get("breakeven", pd.Series(dtype=float)))
    wal = _slice_asof(b.get("balance_sheet", pd.Series(dtype=float)))
    dxy = _slice_asof(b.get("dollar", pd.Series(dtype=float)))

    # QT / QE stance from the 6-month direction of the balance sheet.
    chg6m = _change(wal, 182)
    if np.isnan(chg6m):
        stance_note = ""
    elif chg6m > 0.05:
        stance_note = f"QT ended · {chg6m * 1000:+,.0f}bn / 6m"
    elif chg6m < -0.05:
        stance_note = f"QT running · {chg6m * 1000:+,.0f}bn / 6m"
    else:
        stance_note = f"flat · {chg6m * 1000:+,.0f}bn / 6m"

    drivers = [
        _driver_row("real_yield", real, "10-yr real yield",
                    "What cash pays you to not hold gold", 2, "%", higher_helps_gold=False),
        _driver_row("breakeven", bei, "10-yr breakeven",
                    "The inflation gold is meant to hedge", 2, "%", higher_helps_gold=True),
        _driver_row("dollar", dxy, "Dollar index (DXY)",
                    "The currency gold is priced in", 2, "", higher_helps_gold=False),
        _driver_row("balance_sheet", wal, "Fed balance sheet",
                    "QT drains liquidity · QE adds it", 3, "tn",
                    higher_helps_gold=True, band_note=stance_note),
    ]

    quad = _quadrant(real, bei)
    headwinds = sum(1 for d in drivers if d["stance"] == "bad")
    tailwinds = sum(1 for d in drivers if d["stance"] == "good")

    return {
        "version": LENS_VERSION,
        "quadrant": quad,
        "drivers": drivers,
        "headwinds": headwinds,
        "tailwinds": tailwinds,
        "diagnostics": _diagnostics(real, gold if gold is not None else pd.Series(dtype=float)),
        "source": "fred+yahoo",
        "asof": max([d["asof"] for d in drivers if d["asof"]], default=""),
    }
