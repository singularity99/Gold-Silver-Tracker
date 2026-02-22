import pandas as pd
import numpy as np
from technicals import (
    multi_timeframe_fibonacci, vobhs_composite, detect_triangles,
    bollinger_squeeze, momentum_bars, whale_volume_detection,
    rsi, sma_crossover, FIBONACCI_RATIOS,
)


SIGNAL_STRONG_BUY = "Strong Buy"
SIGNAL_BUY = "Buy"
SIGNAL_NEUTRAL = "Neutral"
SIGNAL_SELL = "Sell / Take Profit"
SIGNAL_STRONG_SELL = "Strong Sell"

STRENGTH_STRONG = "Strong"
STRENGTH_MODERATE = "Moderate"
STRENGTH_WEAK = "Weak"


def _price_near_fib_support(price: float, fib_levels: dict, tolerance_pct: float = 2.0) -> bool:
    """Check if price is within tolerance of any Fibonacci support level (lower half)."""
    if not fib_levels:
        return False
    support_ratios = ["78.6%", "61.8%", "50.0%"]
    for ratio_key in support_ratios:
        level = fib_levels.get(ratio_key)
        if level and abs(price - level) / level * 100 <= tolerance_pct:
            return True
    return False


def _price_near_fib_resistance(price: float, fib_levels: dict, tolerance_pct: float = 2.0) -> bool:
    """Check if price is within tolerance of any Fibonacci resistance level (upper half)."""
    if not fib_levels:
        return False
    resist_ratios = ["23.6%", "0.0%"]
    for ratio_key in resist_ratios:
        level = fib_levels.get(ratio_key)
        if level and abs(price - level) / level * 100 <= tolerance_pct:
            return True
    return False


def score_metal(df: pd.DataFrame, fib_data: dict, current_price: float) -> dict:
    """
    Compute composite signal score for a single metal.
    Returns dict with signal, strength, votes, and component details.
    """
    votes = []
    details = {}

    # --- Fibonacci proximity ---
    for tf_name, fib_levels in fib_data.items():
        near_support = _price_near_fib_support(current_price, fib_levels)
        near_resistance = _price_near_fib_resistance(current_price, fib_levels)
        if near_support:
            votes.append(1)
            details[f"fib_{tf_name}"] = "Near support (bullish)"
        elif near_resistance:
            votes.append(-1)
            details[f"fib_{tf_name}"] = "Near resistance (bearish)"
        else:
            votes.append(0)
            details[f"fib_{tf_name}"] = "Between levels"

    # --- VOBHS ---
    if len(df) > 100:
        vobhs = vobhs_composite(df)

        vo_last = vobhs["volatility_oscillator"]["signal"].iloc[-1]
        votes.append(int(vo_last))
        details["volatility_osc"] = {1: "Bullish spike", -1: "Bearish spike", 0: "Neutral"}.get(vo_last, "Neutral")

        boom_last = vobhs["boom_hunter"]["signal"].iloc[-1]
        boom_trend = 1 if vobhs["boom_hunter"]["trigger"].iloc[-1] > vobhs["boom_hunter"]["quotient"].iloc[-1] else -1
        if boom_last != 0:
            votes.append(int(boom_last))
            details["boom_hunter"] = "Buy crossover" if boom_last == 1 else "Sell crossover"
        else:
            votes.append(boom_trend)
            details["boom_hunter"] = "Trigger above quotient (bullish)" if boom_trend == 1 else "Trigger below quotient (bearish)"

        hma_last = int(vobhs["hma_signal"].iloc[-1])
        votes.append(hma_last)
        details["hull_ma"] = {1: "Price above HMA (bullish)", -1: "Price below HMA (bearish)", 0: "Neutral"}.get(hma_last, "Neutral")

        atr_data = vobhs["modified_atr"]
        details["stop_long"] = f"${atr_data['stop_long'].iloc[-1]:,.0f}"
        details["stop_short"] = f"${atr_data['stop_short'].iloc[-1]:,.0f}"
    else:
        details["vobhs"] = "Insufficient data"

    # --- Triangle patterns ---
    tri = detect_triangles(df)
    if tri["pattern"] != "no_pattern" and tri["pattern"] != "insufficient_data":
        if tri.get("breakout_up"):
            votes.append(1)
            details["triangle"] = f"{tri['pattern']} -- breakout UP"
        elif tri.get("breakout_down"):
            votes.append(-1)
            details["triangle"] = f"{tri['pattern']} -- breakout DOWN"
        else:
            votes.append(0)
            details["triangle"] = f"{tri['pattern']} -- consolidating"
    else:
        details["triangle"] = "No triangle pattern"

    # --- Bollinger Squeeze ---
    bsq = bollinger_squeeze(df)
    if not bsq.empty:
        if bsq["squeeze_on"].iloc[-1]:
            details["bollinger_squeeze"] = "Squeeze ON (consolidation, expect breakout)"
        else:
            details["bollinger_squeeze"] = "Squeeze OFF (trending)"

    # --- Momentum ---
    mom = momentum_bars(df)
    if not mom.empty:
        mom_dir = int(mom["direction"].iloc[-1])
        if mom_dir >= 1:
            votes.append(1)
        elif mom_dir <= -1:
            votes.append(-1)
        else:
            votes.append(0)
        details["momentum"] = {2: "Strong bullish", 1: "Mild bullish", -1: "Mild bearish", -2: "Strong bearish", 0: "Flat"}.get(mom_dir, "Flat")

    # --- Whale volume ---
    whale = whale_volume_detection(df)
    if not whale.empty and "whale_flag" in whale.columns:
        if whale["whale_flag"].iloc[-1]:
            votes.append(1)
            details["whale_volume"] = f"HIGH VOLUME ({whale['volume_ratio'].iloc[-1]:.1f}x avg)"
        else:
            details["whale_volume"] = f"Normal ({whale['volume_ratio'].iloc[-1]:.1f}x avg)" if not np.isnan(whale["volume_ratio"].iloc[-1]) else "No volume data"

    # --- RSI ---
    rsi_val = rsi(df["Close"]).iloc[-1]
    if rsi_val < 30:
        votes.append(1)
        details["rsi"] = f"{rsi_val:.0f} (oversold -- bullish)"
    elif rsi_val > 70:
        votes.append(-1)
        details["rsi"] = f"{rsi_val:.0f} (overbought -- bearish)"
    else:
        votes.append(0)
        details["rsi"] = f"{rsi_val:.0f} (neutral)"

    # --- SMA Crossover ---
    sma_data = sma_crossover(df["Close"])
    if sma_data["fast_above_slow"].iloc[-1]:
        votes.append(1)
        details["sma_cross"] = "20 SMA above 50 SMA (bullish)"
    else:
        votes.append(-1)
        details["sma_cross"] = "20 SMA below 50 SMA (bearish)"

    # --- Composite scoring ---
    bullish = sum(1 for v in votes if v > 0)
    bearish = sum(1 for v in votes if v < 0)
    total = len(votes)

    if bullish >= 5 and any(_price_near_fib_support(current_price, fl) for fl in fib_data.values()):
        signal = SIGNAL_STRONG_BUY
        strength = STRENGTH_STRONG
    elif bullish >= 4:
        signal = SIGNAL_STRONG_BUY
        strength = STRENGTH_STRONG
    elif bullish >= 3:
        signal = SIGNAL_BUY
        strength = STRENGTH_MODERATE
    elif bearish >= 4:
        signal = SIGNAL_STRONG_SELL
        strength = STRENGTH_STRONG
    elif bearish >= 3:
        signal = SIGNAL_SELL
        strength = STRENGTH_MODERATE
    else:
        signal = SIGNAL_NEUTRAL
        strength = STRENGTH_WEAK

    return {
        "signal": signal,
        "strength": strength,
        "bullish_votes": bullish,
        "bearish_votes": bearish,
        "total_indicators": total,
        "details": details,
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
