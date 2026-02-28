import pandas as pd
import numpy as np
from technicals import (
    multi_timeframe_fibonacci, vobhs_composite, detect_triangles,
    bollinger_squeeze, momentum_bars, whale_volume_detection,
    rsi, sma_crossover, ema_crossover, FIBONACCI_RATIOS,
)

SIGNAL_STRONG_BUY = "Strong Buy"
SIGNAL_BUY = "Buy"
SIGNAL_NEUTRAL = "Neutral"
SIGNAL_SELL = "Sell / Take Profit"
SIGNAL_STRONG_SELL = "Strong Sell"

STRENGTH_STRONG = "Strong"
STRENGTH_MODERATE = "Moderate"
STRENGTH_WEAK = "Weak"

# Indicator definitions: (name, timeframe, base_weight, correlation_group)
# Base weights sum to 100. Actual weights rescaled by configurable timeframe percentages.
INDICATORS = [
    ("Fib Long-term (2yr)",    "Long",   5,  "Price Levels"),
    ("Fib Medium-term (3mo)",  "Medium", 9.25,  "Price Levels"),
    ("Fib Short-term (2-3wk)", "Short",  7,  "Price Levels"),
    ("Volatility Oscillator",  "Short",  8,  "Volatility"),
    ("Boom Hunter Pro (BHS)",  "Short",  11, "Trend (COG)"),
    ("EMA Crossover (9/21)",   "Short",  7,  "Trend (MA)"),
    ("Hull Moving Average",    "Medium", 11.10,  "Trend (MA)"),
    ("Modified ATR",           "Long",   4,  "Volatility"),
    ("Triangle Pattern",       "Medium", 5.55,  "Pattern"),
    ("Bollinger Squeeze",      "Medium", 3.70,  "Volatility"),
    ("Momentum Bars (ROC)",    "Short",  8,  "Momentum"),
    ("Whale Volume",           "Short",  7,  "Volume"),
    ("RSI (14)",               "Medium", 7.40,  "Momentum"),
    ("SMA Crossover (20/50)",  "Long",   6,  "Trend (MA)"),
]

# Correlated pairs -- used for conflict detection
CORRELATED_PAIRS = [
    ("EMA Crossover (9/21)", "Hull Moving Average", "Trend (MA) short vs medium"),
    ("Hull Moving Average", "SMA Crossover (20/50)", "Trend (MA) medium vs long"),
    ("EMA Crossover (9/21)", "SMA Crossover (20/50)", "Trend (MA) short vs long"),
    ("RSI (14)", "Momentum Bars (ROC)", "Momentum"),
    ("Volatility Oscillator", "Bollinger Squeeze", "Volatility"),
]


def _price_near_fib_support(price: float, fib_levels: dict, tolerance_pct: float = 2.0) -> bool:
    if not fib_levels:
        return False
    for ratio_key in ["78.6%", "61.8%", "50.0%"]:
        level = fib_levels.get(ratio_key)
        if level and abs(price - level) / level * 100 <= tolerance_pct:
            return True
    return False


def _price_near_fib_resistance(price: float, fib_levels: dict, tolerance_pct: float = 2.0) -> bool:
    if not fib_levels:
        return False
    for ratio_key in ["23.6%", "0.0%"]:
        level = fib_levels.get(ratio_key)
        if level and abs(price - level) / level * 100 <= tolerance_pct:
            return True
    return False


def _fib_vote(price: float, fib_levels: dict) -> tuple[int, str]:
    if _price_near_fib_support(price, fib_levels):
        return 1, "Near support (bullish)"
    elif _price_near_fib_resistance(price, fib_levels):
        return -1, "Near resistance (bearish)"
    return 0, "Between levels"


def _medium_fib_vote(price: float, prev_price: float | None, fib_levels: dict) -> tuple[int, str]:
    if not fib_levels:
        return 0, "No medium-term fib levels"
    lvl_50 = fib_levels.get("50.0%")
    lvl_38 = fib_levels.get("38.2%")
    lvl_62 = fib_levels.get("61.8%")
    if lvl_50 is None or lvl_38 is None or lvl_62 is None:
        return _fib_vote(price, fib_levels)

    confirmation_buffer = 0.002  # 0.2%
    zone_low, zone_high = min(lvl_38, lvl_62), max(lvl_38, lvl_62)
    in_congestion = zone_low <= price <= zone_high

    above_now = price > lvl_50 * (1 + confirmation_buffer)
    below_now = price < lvl_50 * (1 - confirmation_buffer)
    above_prev = prev_price is not None and prev_price > lvl_50 * (1 + confirmation_buffer)
    below_prev = prev_price is not None and prev_price < lvl_50 * (1 - confirmation_buffer)

    if above_now and above_prev:
        return 1, "Above 50% retracement with 2-bar hold (bullish)"
    if below_now and below_prev:
        return -1, "Below 50% retracement with 2-bar hold (bearish)"
    if in_congestion:
        return 0, "Between 38.2% and 61.8% retracement (congestion)"
    return 0, "Around 50% retracement, awaiting confirmation"


def _compute_raw_values(df: pd.DataFrame) -> dict:
    """Extract the raw numeric value for each indicator at the last bar and a lookback bar."""
    vals = {}
    if len(df) < 10:
        return vals
    close = df["Close"]

    # Momentum ROC
    mom = momentum_bars(df)
    if not mom.empty and not np.isnan(mom["roc"].iloc[-1]):
        vals["Momentum Bars (ROC)"] = (mom["roc"].iloc[-1], mom["roc"].iloc[-6] if len(mom) > 6 else mom["roc"].iloc[0])

    # RSI
    rsi_series = rsi(close)
    if not rsi_series.empty and not np.isnan(rsi_series.iloc[-1]):
        vals["RSI (14)"] = (rsi_series.iloc[-1], rsi_series.iloc[-6] if len(rsi_series) > 6 else rsi_series.iloc[0])

    # EMA gap (fast - slow)
    ema_data = ema_crossover(close)
    if not ema_data.empty:
        gap_now = ema_data["ema_9"].iloc[-1] - ema_data["ema_21"].iloc[-1]
        gap_prev = ema_data["ema_9"].iloc[-6] - ema_data["ema_21"].iloc[-6] if len(ema_data) > 6 else gap_now
        vals["EMA Crossover (9/21)"] = (gap_now, gap_prev)

    # SMA gap
    sma_data = sma_crossover(close)
    if not sma_data.empty:
        gap_now = sma_data["sma_20"].iloc[-1] - sma_data["sma_50"].iloc[-1]
        gap_prev = sma_data["sma_20"].iloc[-6] - sma_data["sma_50"].iloc[-6] if len(sma_data) > 6 else gap_now
        vals["SMA Crossover (20/50)"] = (gap_now, gap_prev)

    # Whale volume ratio
    whale = whale_volume_detection(df)
    if not whale.empty and "volume_ratio" in whale.columns:
        r_now = whale["volume_ratio"].iloc[-1]
        r_prev = whale["volume_ratio"].iloc[-6] if len(whale) > 6 else r_now
        if not np.isnan(r_now):
            vals["Whale Volume"] = (r_now, r_prev if not np.isnan(r_prev) else r_now)

    # VOBHS components
    if len(df) > 100:
        vobhs = vobhs_composite(df)
        vo = vobhs["volatility_oscillator"]["spike"]
        vals["Volatility Oscillator"] = (vo.iloc[-1], vo.iloc[-6] if len(vo) > 6 else vo.iloc[0])

        boom = vobhs["boom_hunter"]
        trig_now = boom["trigger"].iloc[-1] - boom["quotient"].iloc[-1]
        trig_prev = (boom["trigger"].iloc[-6] - boom["quotient"].iloc[-6]) if len(boom) > 6 else trig_now
        vals["Boom Hunter Pro (BHS)"] = (trig_now, trig_prev)

        hma_now = df["Close"].iloc[-1] - vobhs["hull_ma"].iloc[-1]
        hma_prev = (df["Close"].iloc[-6] - vobhs["hull_ma"].iloc[-6]) if len(vobhs["hull_ma"]) > 6 else hma_now
        vals["Hull Moving Average"] = (hma_now, hma_prev)

        atr = vobhs["modified_atr"]["atr"]
        vals["Modified ATR"] = (atr.iloc[-1], atr.iloc[-6] if len(atr) > 6 else atr.iloc[0])

    # Bollinger
    bsq = bollinger_squeeze(df)
    if not bsq.empty and len(bsq) > 6:
        vals["Bollinger Squeeze"] = (bsq["bb_width"].iloc[-1], bsq["bb_width"].iloc[-6])

    return vals


def _direction_label(name: str, now: float, prev: float) -> str:
    """Return an arrow + label showing direction of the raw value."""
    if np.isnan(now) or np.isnan(prev):
        return "-"
    diff = now - prev
    threshold = abs(prev) * 0.02 if prev != 0 else 0.01
    if abs(diff) < threshold:
        return "\u2194 Stable"
    # For ATR and Bollinger width, rising = more volatile = deteriorating
    inverted = name in ("Modified ATR", "Bollinger Squeeze")
    if diff > 0:
        return "\u2198 Deteriorating" if inverted else "\u2197 Improving"
    else:
        return "\u2197 Improving" if inverted else "\u2198 Deteriorating"


# Default timeframe weight targets (percentage of total)
DEFAULT_TF_WEIGHTS = {"Short": 48, "Medium": 37, "Long": 15}


def _rescale_indicators(tf_weights: dict) -> list:
    """Rescale indicator weights so each timeframe sums to its target percentage."""
    # Sum of base weights per timeframe
    base_sums = {"Short": 0, "Medium": 0, "Long": 0}
    for _, tf, w, _ in INDICATORS:
        base_sums[tf] += w
    rescaled = []
    for name, tf, w, cg in INDICATORS:
        if base_sums[tf] > 0:
            new_w = w / base_sums[tf] * tf_weights[tf]
        else:
            new_w = 0
        rescaled.append((name, tf, new_w, cg))
    return rescaled


def score_metal(df: pd.DataFrame, fib_data: dict, current_price: float,
                tf_weights: dict = None) -> dict:
    """
    Compute weighted composite signal score for a single metal.
    All 14 indicators always vote. Weighted by importance and trading horizon.
    tf_weights: optional dict {"Short": %, "Medium": %, "Long": %} summing to 100.
    """
    if tf_weights is None:
        tf_weights = DEFAULT_TF_WEIGHTS
    indicators = _rescale_indicators(tf_weights)

    votes = {}  # indicator_name -> (vote, detail_text)
    raw_values = _compute_raw_values(df)

    # --- Fibonacci (3 timeframes) ---
    tf_map = {"long_term": "Fib Long-term (2yr)",
              "medium_term": "Fib Medium-term (3mo)",
              "short_term": "Fib Short-term (2-3wk)"}
    prev_close = float(df["Close"].iloc[-2]) if len(df) > 1 else None
    for tf_key, ind_name in tf_map.items():
        fib_levels = fib_data.get(tf_key, {})
        if tf_key == "medium_term":
            vote, detail = _medium_fib_vote(current_price, prev_close, fib_levels)
        else:
            vote, detail = _fib_vote(current_price, fib_levels)
        votes[ind_name] = (vote, detail)

    # --- VOBHS components ---
    if len(df) > 100:
        vobhs = vobhs_composite(df)

        # Volatility Oscillator
        vo_last = int(vobhs["volatility_oscillator"]["signal"].iloc[-1])
        votes["Volatility Oscillator"] = (vo_last,
            {1: "Bullish spike above upper band", -1: "Bearish spike below lower band", 0: "Within bands (neutral)"}.get(vo_last, "Neutral"))

        # Boom Hunter Pro
        boom = vobhs["boom_hunter"]
        boom_cross = int(boom["signal"].iloc[-1])
        if boom_cross != 0:
            votes["Boom Hunter Pro (BHS)"] = (boom_cross,
                "Buy crossover (trigger crossed above quotient)" if boom_cross == 1 else "Sell crossover (trigger crossed below quotient)")
        else:
            boom_trend = 1 if boom["trigger"].iloc[-1] > boom["quotient"].iloc[-1] else -1
            votes["Boom Hunter Pro (BHS)"] = (boom_trend,
                "Trigger above quotient (bullish trend)" if boom_trend == 1 else "Trigger below quotient (bearish trend)")

        # Hull Moving Average (medium-term trend filter with slope + distance buffer)
        hma_series = vobhs["hull_ma"]
        hma_now = float(hma_series.iloc[-1]) if len(hma_series) else np.nan
        hma_prev5 = float(hma_series.iloc[-6]) if len(hma_series) > 5 else np.nan
        close_now = float(df["Close"].iloc[-1])
        if np.isnan(hma_now) or hma_now == 0 or np.isnan(hma_prev5) or hma_prev5 == 0:
            hma_vote = 0
            hma_detail = "Insufficient HMA data"
        else:
            gap_pct = (close_now - hma_now) / hma_now * 100
            slope_pct = (hma_now - hma_prev5) / hma_prev5 * 100
            if abs(gap_pct) <= 0.5 or abs(slope_pct) <= 0.1:
                hma_vote = 0
                hma_detail = f"Within HMA neutrality band (gap {gap_pct:+.2f}%, slope {slope_pct:+.2f}%)"
            elif gap_pct > 0.5 and slope_pct > 0.1:
                hma_vote = 1
                hma_detail = f"Above HMA with positive slope (gap {gap_pct:+.2f}%, slope {slope_pct:+.2f}%)"
            elif gap_pct < -0.5 and slope_pct < -0.1:
                hma_vote = -1
                hma_detail = f"Below HMA with negative slope (gap {gap_pct:+.2f}%, slope {slope_pct:+.2f}%)"
            else:
                hma_vote = 0
                hma_detail = f"Price/slope mismatch vs HMA (gap {gap_pct:+.2f}%, slope {slope_pct:+.2f}%)"
        votes["Hull Moving Average"] = (hma_vote, hma_detail)

        # Modified ATR -- now votes based on ATR trend (narrowing = bullish conviction, widening = caution)
        atr_series = vobhs["modified_atr"]["atr"]
        atr_current = atr_series.iloc[-1]
        atr_avg = atr_series.rolling(20).mean().iloc[-1]
        if not np.isnan(atr_current) and not np.isnan(atr_avg):
            if atr_current < atr_avg * 0.8:
                atr_vote = 1
                atr_detail = f"ATR narrowing (${atr_current:,.0f} < avg ${atr_avg:,.0f}) -- low volatility, trend conviction"
            elif atr_current > atr_avg * 1.2:
                atr_vote = -1
                atr_detail = f"ATR widening (${atr_current:,.0f} > avg ${atr_avg:,.0f}) -- high volatility, caution"
            else:
                atr_vote = 0
                atr_detail = f"ATR normal (${atr_current:,.0f} ~ avg ${atr_avg:,.0f})"
        else:
            atr_vote = 0
            atr_detail = "Insufficient ATR data"
        stop_long = vobhs["modified_atr"]["stop_long"].iloc[-1]
        stop_short = vobhs["modified_atr"]["stop_short"].iloc[-1]
        atr_detail += f" | Stops: long ${stop_long:,.0f} / short ${stop_short:,.0f}"
        votes["Modified ATR"] = (atr_vote, atr_detail)
    else:
        for name in ["Volatility Oscillator", "Boom Hunter Pro (BHS)", "Hull Moving Average", "Modified ATR"]:
            votes[name] = (0, "Insufficient data (<100 bars)")

    # --- Triangle Pattern ---
    tri = detect_triangles(df)
    if tri["pattern"] not in ("no_pattern", "insufficient_data"):
        if len(df) > 1 and "resistance_line" in tri and "support_line" in tri:
            breakout_buffer = 0.0025  # 0.25%
            c_now = float(df["Close"].iloc[-1])
            c_prev = float(df["Close"].iloc[-2])
            r_now = float(tri["resistance_line"][-1])
            r_prev = float(tri["resistance_line"][-2])
            s_now = float(tri["support_line"][-1])
            s_prev = float(tri["support_line"][-2])
            bull_confirm = c_now > r_now * (1 + breakout_buffer) and c_prev > r_prev * (1 + breakout_buffer)
            bear_confirm = c_now < s_now * (1 - breakout_buffer) and c_prev < s_prev * (1 - breakout_buffer)
        else:
            bull_confirm = False
            bear_confirm = False

        if bull_confirm:
            votes["Triangle Pattern"] = (1, f"{tri['pattern']} -- 2-bar bullish breakout confirmation")
        elif bear_confirm:
            votes["Triangle Pattern"] = (-1, f"{tri['pattern']} -- 2-bar bearish breakout confirmation")
        else:
            votes["Triangle Pattern"] = (0, f"{tri['pattern']} -- forming / no confirmed breakout")
    else:
        votes["Triangle Pattern"] = (0, "No triangle pattern detected")

    # --- Bollinger Squeeze -- now votes ---
    bsq = bollinger_squeeze(df)
    if not bsq.empty:
        squeeze_on = bool(bsq["squeeze_on"].iloc[-1])
        squeeze_recent = bool((bsq["squeeze_on"].tail(4)).any())
        bb_width = float(bsq["bb_width"].iloc[-1])
        bb_mid = float(bsq["bb_mid"].iloc[-1]) if "bb_mid" in bsq.columns and not np.isnan(bsq["bb_mid"].iloc[-1]) else np.nan
        close_now = float(df["Close"].iloc[-1])
        rsi_val = float(rsi(df["Close"]).iloc[-1]) if len(df) > 0 else np.nan
        if squeeze_on:
            votes["Bollinger Squeeze"] = (0, f"Squeeze ON (bandwidth {bb_width:.4f}) -- compression regime")
        elif squeeze_recent and not np.isnan(bb_mid) and not np.isnan(rsi_val):
            if close_now > bb_mid and rsi_val >= 52:
                votes["Bollinger Squeeze"] = (1, f"Post-squeeze release with RSI {rsi_val:.0f}>52 (bullish)")
            elif close_now < bb_mid and rsi_val <= 48:
                votes["Bollinger Squeeze"] = (-1, f"Post-squeeze release with RSI {rsi_val:.0f}<48 (bearish)")
            else:
                votes["Bollinger Squeeze"] = (0, f"Post-squeeze release but direction not confirmed (RSI {rsi_val:.0f})")
        else:
            votes["Bollinger Squeeze"] = (0, f"No active post-squeeze directional regime ({bb_width:.4f})")
    else:
        votes["Bollinger Squeeze"] = (0, "Insufficient data")

    # --- Momentum Bars ---
    mom = momentum_bars(df)
    if not mom.empty:
        mom_dir = int(mom["direction"].iloc[-1])
        roc_val = mom["roc"].iloc[-1]
        label = {2: "Strong bullish", 1: "Mild bullish", -1: "Mild bearish", -2: "Strong bearish", 0: "Flat"}.get(mom_dir, "Flat")
        if mom_dir >= 1:
            votes["Momentum Bars (ROC)"] = (1, f"{label} (ROC: {roc_val:+.1f}%)")
        elif mom_dir <= -1:
            votes["Momentum Bars (ROC)"] = (-1, f"{label} (ROC: {roc_val:+.1f}%)")
        else:
            votes["Momentum Bars (ROC)"] = (0, f"{label} (ROC: {roc_val:+.1f}%)")
    else:
        votes["Momentum Bars (ROC)"] = (0, "Insufficient data")

    # --- Whale Volume -- always votes ---
    whale = whale_volume_detection(df)
    if not whale.empty and "whale_flag" in whale.columns:
        ratio_val = whale["volume_ratio"].iloc[-1]
        if not np.isnan(ratio_val):
            if whale["whale_flag"].iloc[-1]:
                votes["Whale Volume"] = (1, f"HIGH VOLUME ({ratio_val:.1f}x avg) -- institutional accumulation")
            elif ratio_val < 0.5:
                votes["Whale Volume"] = (-1, f"Low volume ({ratio_val:.1f}x avg) -- weak conviction")
            else:
                votes["Whale Volume"] = (0, f"Normal volume ({ratio_val:.1f}x avg)")
        else:
            votes["Whale Volume"] = (0, "No volume data available")
    else:
        votes["Whale Volume"] = (0, "No volume data available")

    # --- RSI (medium momentum regime with hysteresis) ---
    rsi_val = rsi(df["Close"]).iloc[-1]
    if not np.isnan(rsi_val):
        if rsi_val >= 52:
            votes["RSI (14)"] = (1, f"{rsi_val:.0f} -- above 52 momentum regime (bullish)")
        elif rsi_val <= 48:
            votes["RSI (14)"] = (-1, f"{rsi_val:.0f} -- below 48 momentum regime (bearish)")
        else:
            votes["RSI (14)"] = (0, f"{rsi_val:.0f} -- 48-52 neutral hysteresis zone")
    else:
        votes["RSI (14)"] = (0, "Insufficient data")

    # --- EMA Crossover (9/21) -- short-term trend ---
    ema_data = ema_crossover(df["Close"])
    if not ema_data.empty:
        if ema_data["cross_up"].iloc[-1]:
            votes["EMA Crossover (9/21)"] = (1, "9 EMA just crossed above 21 EMA (fresh bullish crossover)")
        elif ema_data["cross_down"].iloc[-1]:
            votes["EMA Crossover (9/21)"] = (-1, "9 EMA just crossed below 21 EMA (fresh bearish crossover)")
        elif ema_data["fast_above_slow"].iloc[-1]:
            votes["EMA Crossover (9/21)"] = (1, "9 EMA above 21 EMA (short-term uptrend)")
        else:
            votes["EMA Crossover (9/21)"] = (-1, "9 EMA below 21 EMA (short-term downtrend)")
    else:
        votes["EMA Crossover (9/21)"] = (0, "Insufficient data")

    # --- SMA Crossover (20/50) -- long-term trend ---
    sma_data = sma_crossover(df["Close"])
    if not sma_data.empty:
        if sma_data["fast_above_slow"].iloc[-1]:
            votes["SMA Crossover (20/50)"] = (1, "20 SMA above 50 SMA (bullish)")
        else:
            votes["SMA Crossover (20/50)"] = (-1, "20 SMA below 50 SMA (bearish)")
    else:
        votes["SMA Crossover (20/50)"] = (0, "Insufficient data")

    # --- Build weighted score ---
    indicator_rows = []
    weighted_sum = 0.0
    tf_sums = {"Short": 0.0, "Medium": 0.0, "Long": 0.0}
    tf_weight_sums = {"Short": 0.0, "Medium": 0.0, "Long": 0.0}

    for name, timeframe, weight, corr_group in indicators:
        vote, detail = votes.get(name, (0, "N/A"))
        w_score = vote * weight
        weighted_sum += w_score
        tf_sums[timeframe] += w_score
        tf_weight_sums[timeframe] += weight
        # Direction from raw values
        if name in raw_values:
            now_val, prev_val = raw_values[name]
            direction = _direction_label(name, now_val, prev_val)
        else:
            direction = "-"
        indicator_rows.append({
            "Indicator": name,
            "Timeframe": timeframe,
            "Vote": {1: "Bullish", -1: "Bearish", 0: "Neutral"}.get(vote, "Neutral"),
            "Direction": direction,
            "Weight": weight,
            "Weighted Score": w_score,
            "Correlation Group": corr_group,
            "Detail": detail,
        })

    composite = weighted_sum / 100.0

    # Per-timeframe sub-scores (normalized to -1..+1)
    tf_normalized = {}
    for tf in ("Short", "Medium", "Long"):
        if tf_weight_sums[tf] > 0:
            tf_normalized[tf] = tf_sums[tf] / tf_weight_sums[tf]
        else:
            tf_normalized[tf] = 0.0

    # Conflict detection: flag when correlated indicators disagree
    CONFLICT_EXPLANATIONS = {
        "Trend (MA) short vs medium": "Short-term trend has weakened but the medium-term swing is intact. Common during healthy pullbacks -- wait for the faster MA to re-align before acting.",
        "Trend (MA) medium vs long": "Medium-term momentum diverges from the long-term structure. If medium is bearish, the broader trend may be turning -- watch for follow-through.",
        "Trend (MA) short vs long": "Short-term and long-term trends disagree. Short-term noise vs structural direction -- prioritise the longer timeframe unless short-term persists.",
        "Momentum": "RSI and ROC disagree. One sees overbought/oversold while the other sees trend strength. Check which is leading -- RSI often leads at extremes.",
        "Volatility": "Volatility Oscillator and Bollinger Squeeze disagree. Compression may be building before a breakout that the oscillator hasn't caught yet.",
    }
    conflicts = []
    conflicting_indicators = set()
    for ind_a, ind_b, group in CORRELATED_PAIRS:
        vote_a = votes.get(ind_a, (0, ""))[0]
        vote_b = votes.get(ind_b, (0, ""))[0]
        if vote_a != 0 and vote_b != 0 and vote_a != vote_b:
            explanation = CONFLICT_EXPLANATIONS.get(group, "")
            conflicting_indicators.add(ind_a)
            conflicting_indicators.add(ind_b)
            conflicts.append({
                "group": group,
                "ind_a": ind_a,
                "vote_a": "bullish" if vote_a > 0 else "bearish",
                "ind_b": ind_b,
                "vote_b": "bullish" if vote_b > 0 else "bearish",
                "explanation": explanation,
            })
    # Tag indicator rows with conflict info
    for row in indicator_rows:
        if row["Indicator"] in conflicting_indicators:
            row["Conflict"] = True
        else:
            row["Conflict"] = False

    # Round to 2dp so classification matches the displayed value
    composite = round(composite, 2)

    # Signal classification
    if composite >= 0.40:
        signal = SIGNAL_STRONG_BUY
        strength = STRENGTH_STRONG
    elif composite >= 0.20:
        signal = SIGNAL_BUY
        strength = STRENGTH_MODERATE
    elif composite <= -0.40:
        signal = SIGNAL_STRONG_SELL
        strength = STRENGTH_STRONG
    elif composite <= -0.20:
        signal = SIGNAL_SELL
        strength = STRENGTH_MODERATE
    else:
        signal = SIGNAL_NEUTRAL
        strength = STRENGTH_WEAK

    bullish = sum(1 for name, _, _, _ in indicators if votes.get(name, (0,))[0] > 0)
    bearish = sum(1 for name, _, _, _ in indicators if votes.get(name, (0,))[0] < 0)
    neutral = sum(1 for name, _, _, _ in indicators if votes.get(name, (0,))[0] == 0)

    return {
        "signal": signal,
        "strength": strength,
        "composite_score": composite,
        "bullish_votes": bullish,
        "bearish_votes": bearish,
        "neutral_votes": neutral,
        "total_indicators": len(indicators),
        "indicator_table": indicator_rows,
        "timeframe_scores": tf_normalized,
        "timeframe_weights": tf_weights,
        "conflicts": conflicts,
    }


def allocation_recommendation(gold_score: dict, silver_score: dict, gs_ratio: float) -> dict:
    """Recommend gold vs. silver allocation based on signals and G/S ratio."""
    silver_pct = 55
    if gs_ratio > 63:
        silver_pct += 10
    if silver_score["bullish_votes"] > gold_score["bullish_votes"]:
        silver_pct += 5
    elif gold_score["bullish_votes"] > silver_score["bullish_votes"]:
        silver_pct -= 10

    silver_pct = max(30, min(70, silver_pct))
    gold_pct = 100 - silver_pct

    reasoning = []
    if gs_ratio > 63:
        reasoning.append(f"G/S ratio {gs_ratio:.1f} > 63 -- favour silver (catch-up potential)")
    if silver_score["bullish_votes"] > gold_score["bullish_votes"]:
        reasoning.append("Silver showing more bullish indicators than gold")
    elif gold_score["bullish_votes"] > silver_score["bullish_votes"]:
        reasoning.append("Gold showing more bullish indicators than silver")

    return {
        "gold_pct": gold_pct,
        "silver_pct": silver_pct,
        "reasoning": reasoning,
    }
