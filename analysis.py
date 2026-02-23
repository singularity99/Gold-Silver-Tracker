"""Rules-engine market analysis summary -- no API key required."""


def _signal_label(score: float) -> str:
    if score >= 0.40:
        return "Strong Buy"
    elif score >= 0.20:
        return "Buy"
    elif score <= -0.40:
        return "Strong Sell"
    elif score <= -0.20:
        return "Sell"
    return "Neutral"


def _tf_label(tf: str) -> str:
    return {"Short": "Short-term (days-weeks)", "Medium": "Medium-term (weeks-months)",
            "Long": "Long-term (months-years)"}.get(tf, tf)


def generate_summary(metal: str, score: dict, price_usd: float) -> str:
    """Generate plain English market analysis from scoring data."""
    composite = score["composite_score"]
    signal = score["signal"]
    tf_scores = score["timeframe_scores"]
    tf_weights = score["timeframe_weights"]
    conflicts = score["conflicts"]
    rows = score["indicator_table"]

    # Count deteriorating / improving indicators
    deteriorating = [r for r in rows if "Deteriorating" in r.get("Direction", "")]
    improving = [r for r in rows if "Improving" in r.get("Direction", "")]
    bullish_det = [r for r in deteriorating if r["Vote"] == "Bullish"]
    bearish_imp = [r for r in improving if r["Vote"] == "Bearish"]

    # Headline
    det_warning = ""
    if len(bullish_det) >= 2:
        det_warning = " -- but deteriorating"
    elif len(bearish_imp) >= 2:
        det_warning = " -- but recovering"
    headline = f"**{metal.title()}: {signal} ({composite:+.2f}){det_warning}**\n\n"

    # Per-timeframe narratives
    tf_parts = []
    for tf in ("Short", "Medium", "Long"):
        val = round(tf_scores[tf], 2)
        pct = tf_weights[tf]
        label = _signal_label(val)
        tf_rows = [r for r in rows if r["Timeframe"] == tf]

        # Key drivers
        drivers = []
        for r in tf_rows:
            if r["Vote"] != "Neutral":
                direction = r.get("Direction", "")
                dir_note = ""
                if "Deteriorating" in direction:
                    dir_note = " (deteriorating)"
                elif "Improving" in direction:
                    dir_note = " (improving)"
                vote = "bullish" if r["Vote"] == "Bullish" else "bearish"
                drivers.append(f"{r['Indicator']} {vote}{dir_note}")

        # Count deteriorating in this timeframe
        tf_det = [r for r in tf_rows if "Deteriorating" in r.get("Direction", "")]

        narrative = f"**{_tf_label(tf)}** ({pct}% weight): {label} ({val:+.2f})"
        if drivers:
            narrative += f" -- {', '.join(drivers)}."
        else:
            narrative += " -- all neutral, no strong signals."

        if tf_det:
            names = [r["Indicator"].split("(")[0].strip() for r in tf_det]
            narrative += f" Warning: {', '.join(names)} deteriorating."

        tf_parts.append(narrative)

    # Convergence analysis
    convergence = ""
    short_val = round(tf_scores["Short"], 2)
    med_val = round(tf_scores["Medium"], 2)
    long_val = round(tf_scores["Long"], 2)

    if len(bullish_det) >= 2 and short_val > med_val + 0.3:
        convergence = (
            f"\n\n**Convergence warning**: Short-term ({short_val:+.2f}) is significantly "
            f"above medium-term ({med_val:+.2f}) but {len(bullish_det)} short-term indicators "
            f"are deteriorating ({', '.join(r['Indicator'] for r in bullish_det)}). "
            f"Short-term score is likely to converge toward medium-term as these indicators flip."
        )
    elif len(bearish_imp) >= 2 and short_val < med_val - 0.3:
        convergence = (
            f"\n\n**Recovery building**: Short-term ({short_val:+.2f}) is below medium-term "
            f"({med_val:+.2f}) but {len(bearish_imp)} indicators are improving. "
            f"Score may recover as these indicators flip bullish."
        )

    # Conflicts
    conflict_text = ""
    if conflicts:
        conflict_text = "\n\n**Active conflicts**:\n"
        for c in conflicts:
            conflict_text += f"- {c}\n"

    # Actionable conclusion
    action = "\n\n**Action**: "
    if composite >= 0.40 and len(bullish_det) < 2:
        action += "Strong bullish alignment across timeframes. Consider deploying a tranche."
    elif composite >= 0.20 and len(bullish_det) >= 2:
        action += ("Bullish but fragile -- multiple indicators deteriorating. "
                   "Wait for direction arrows to stabilise before deploying.")
    elif composite >= 0.40 and len(bullish_det) >= 2:
        action += ("Score reads Strong Buy but several indicators are deteriorating. "
                   "This reading is likely to weaken. Hold off on new deployment "
                   "until direction confirms, or deploy a smaller tranche.")
    elif -0.19 <= composite <= 0.19:
        action += "Neutral -- no clear edge. Wait for a directional signal before acting."
    elif composite <= -0.20:
        action += ("Bearish. Avoid deploying new capital. If already positioned, "
                   "monitor stop-loss levels from Modified ATR.")
    else:
        action += "Mildly bullish. Consider a partial tranche if direction arrows are stable."

    return headline + "\n\n".join(tf_parts) + convergence + conflict_text + action


def build_context_prompt(metal: str, score: dict, price_usd: float,
                         fib_data: dict = None) -> str:
    """Build a context string for the LLM chat, containing full current state."""
    lines = [
        f"## Current {metal.title()} Market State",
        f"Price: ${price_usd:,.2f}",
        f"Signal: {score['signal']} | Composite Score: {score['composite_score']:+.2f}",
        f"Votes: {score['bullish_votes']}B / {score['bearish_votes']}S / {score['neutral_votes']}N",
        "",
        "### Timeframe Sub-Scores",
    ]
    for tf in ("Short", "Medium", "Long"):
        val = round(score["timeframe_scores"][tf], 2)
        pct = score["timeframe_weights"][tf]
        lines.append(f"- {_tf_label(tf)} ({pct}% weight): {val:+.2f} = {_signal_label(val)}")

    lines.append("\n### Indicator Breakdown")
    lines.append("| Indicator | Timeframe | Vote | Direction | Weight | Detail |")
    lines.append("|---|---|---|---|---|---|")
    for r in score["indicator_table"]:
        lines.append(
            f"| {r['Indicator']} | {r['Timeframe']} | {r['Vote']} | "
            f"{r.get('Direction', '-')} | {r['Weight']} | {r['Detail']} |"
        )

    if score["conflicts"]:
        lines.append("\n### Active Conflicts")
        for c in score["conflicts"]:
            lines.append(f"- {c}")

    if fib_data:
        lines.append("\n### Fibonacci Levels")
        for tf_name, levels in fib_data.items():
            lines.append(f"**{tf_name.replace('_', ' ').title()}**")
            for k, v in levels.items():
                if k not in ("swing_high", "swing_low"):
                    lines.append(f"  {k}: ${v:,.0f}")

    return "\n".join(lines)
