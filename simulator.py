import numpy as np
import pandas as pd
import yfinance as yf
from functools import lru_cache
from threading import Lock
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from technicals import multi_timeframe_fibonacci
from signals import (
    score_metal,
    DEFAULT_TF_WEIGHTS,
    SIGNAL_STRONG_BUY,
    SIGNAL_BUY,
    SIGNAL_NEUTRAL,
    SIGNAL_SELL,
    SIGNAL_STRONG_SELL,
)

GBPUSD_TICKER = "GBPUSD=X"
START_DEFAULT = datetime(2026, 1, 1)
TRADE_SCENARIOS = {
    "morning": {"hour": 10},  # 09–10 London bar
    "end_of_day": {"hour": 17},  # 16–17 London bar
    "intraday_short": {"hour": "all", "tf_weights": {"Short": 100, "Medium": 0, "Long": 0}},
}
TARGET_ALLOC = {
    SIGNAL_STRONG_BUY: 0.60,
    SIGNAL_BUY: 0.30,
    SIGNAL_NEUTRAL: 0.0,
    SIGNAL_SELL: 0.30,
    SIGNAL_STRONG_SELL: 0.30,
}


_HISTORY_CACHE: dict[tuple[str, str, str], pd.DataFrame] = {}
_HISTORY_CACHE_LOCK = Lock()


def _download_history(ticker: str, start_key: str, interval: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start_key, interval=interval, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index)
    return df


def _expected_min_rows(start: datetime, interval: str) -> int:
    days = max((datetime.utcnow().date() - start.date()).days, 1)
    if interval == "1h":
        return max(24, int(days * 3))
    if interval == "1d":
        return max(20, int(days * 0.15))
    return 1


def _is_sufficient(df: pd.DataFrame, start: datetime, interval: str) -> bool:
    if df is None or df.empty:
        return False
    return len(df) >= _expected_min_rows(start, interval)


def _fetch_history(ticker: str, start: datetime, interval: str) -> pd.DataFrame:
    start_key = start.date().isoformat()
    key = (ticker, start_key, interval)
    with _HISTORY_CACHE_LOCK:
        cached = _HISTORY_CACHE.get(key)
        if cached is not None and _is_sufficient(cached, start, interval):
            return cached.copy()

        df = _download_history(ticker, start_key, interval)
        if not _is_sufficient(df, start, interval):
            df = _download_history(ticker, start_key, interval)
        _HISTORY_CACHE[key] = df.copy()
        return df.copy()


def _to_london(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df.tz_convert(ZoneInfo("Europe/London"))


def _latest_before(df: pd.DataFrame, ts) -> pd.Series | None:
    eligible = df[df.index <= ts]
    if eligible.empty:
        return None
    return eligible.iloc[-1]


def _atr(df: pd.DataFrame, period: int = 14) -> float:
    if df.empty:
        return np.nan
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean().iloc[-1] if len(tr) >= period else np.nan


def _adx(df: pd.DataFrame, period: int = 14) -> float:
    if len(df) < period + 1:
        return np.nan
    high = df["High"].values
    low = df["Low"].values
    close = df["Close"].values
    plus_dm = np.maximum(high[1:] - high[:-1], 0)
    minus_dm = np.maximum(low[:-1] - low[1:], 0)
    plus_dm[plus_dm < minus_dm] = 0
    minus_dm[minus_dm < plus_dm] = 0
    tr = np.maximum(high[1:], close[:-1]) - np.minimum(low[1:], close[:-1])
    atr = pd.Series(tr).rolling(period).mean().values
    if len(atr) < period:
        return np.nan
    atr = atr[-1]
    if atr == 0 or np.isnan(atr):
        return np.nan
    plus_di = 100 * (pd.Series(plus_dm).rolling(period).mean().iloc[-1] / atr)
    minus_di = 100 * (pd.Series(minus_dm).rolling(period).mean().iloc[-1] / atr)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di) if (plus_di + minus_di) != 0 else np.nan
    return dx


def _drawdown(series: pd.Series) -> tuple[float, float]:
    if series.empty:
        return 0.0, 0.0
    cum_max = series.cummax()
    dd = (series - cum_max) / cum_max
    max_dd = dd.min()
    end_idx = dd.idxmin()
    start_idx = series.loc[:end_idx].idxmax() if not series.loc[:end_idx].empty else series.index[0]
    return float(max_dd), float(series.index.get_loc(end_idx) - series.index.get_loc(start_idx))


def _perf_stats(equity: pd.Series) -> dict:
    if equity.empty or len(equity) < 2:
        return {"CAGR": 0.0, "Volatility": 0.0, "Sharpe": 0.0, "Max Drawdown": 0.0}
    daily = equity.resample("1D").last().dropna()
    rets = daily.pct_change().dropna()
    if daily.empty or rets.empty:
        return {"CAGR": 0.0, "Volatility": 0.0, "Sharpe": 0.0, "Max Drawdown": 0.0}
    total_return = daily.iloc[-1] / daily.iloc[0]
    days = (daily.index[-1] - daily.index[0]).days or 1
    cagr = total_return ** (365 / days) - 1
    vol = rets.std() * np.sqrt(252)
    sharpe = cagr / vol if vol > 0 else 0.0
    mdd, _ = _drawdown(daily)
    return {
        "CAGR": float(cagr),
        "Volatility": float(vol),
        "Sharpe": float(sharpe),
        "Max Drawdown": float(mdd),
    }


def _signal_to_weight(signal: str) -> float:
    return TARGET_ALLOC.get(signal, 0.0)


def _compute_signal(metal: str, daily_df: pd.DataFrame, hourly_df: pd.DataFrame, as_of_ts, tf_weights: dict) -> tuple[str, dict]:
    long_df = daily_df[daily_df.index.date <= as_of_ts.date()]
    medium_df = long_df.tail(90)
    short_df = hourly_df[(hourly_df.index <= as_of_ts) & (hourly_df.index >= as_of_ts - timedelta(days=30))]
    fib = multi_timeframe_fibonacci({
        "long_term": long_df,
        "medium_term": medium_df,
        "short_term": short_df,
    })
    prev_bar = _latest_before(hourly_df, as_of_ts)
    if prev_bar is None or long_df.empty:
        return SIGNAL_NEUTRAL, {}
    price_now = float(prev_bar["Close"])
    score = score_metal(long_df, fib, price_now, tf_weights=tf_weights)
    return score.get("signal", SIGNAL_NEUTRAL), score


def _target_baseline(signal: str) -> float:
    return _signal_to_weight(signal)


def _target_agree(score: dict) -> float:
    tf = score.get("timeframe_scores", {})
    s_short = tf.get("Short", 0.0)
    s_med = tf.get("Medium", 0.0)
    if s_short >= 0.4 and s_med >= 0.4:
        return TARGET_ALLOC[SIGNAL_STRONG_BUY]
    if s_short >= 0.2 and s_med >= 0.2:
        return TARGET_ALLOC[SIGNAL_BUY]
    if s_short <= -0.2 and s_med <= -0.2:
        return TARGET_ALLOC[SIGNAL_SELL]
    return 0.0


def _target_hysteresis(score: dict, last_target: float) -> float:
    comp = score.get("composite_score", 0.0)
    enter_hi = 0.2
    strong_hi = 0.4
    exit_lo = -0.1
    if last_target > 0:  # currently invested
        if comp < exit_lo:
            return 0.0
        if comp >= strong_hi:
            return TARGET_ALLOC[SIGNAL_STRONG_BUY]
        if comp >= enter_hi:
            return TARGET_ALLOC[SIGNAL_BUY]
        return last_target  # hold
    else:  # flat
        if comp >= strong_hi:
            return TARGET_ALLOC[SIGNAL_STRONG_BUY]
        if comp >= enter_hi:
            return TARGET_ALLOC[SIGNAL_BUY]
        return 0.0


def _target_banded(score: dict) -> float:
    comp = score.get("composite_score", 0.0)
    if comp <= 0:
        return 0.0
    if comp >= 0.4:
        return TARGET_ALLOC[SIGNAL_STRONG_BUY]
    return TARGET_ALLOC[SIGNAL_STRONG_BUY] * (comp / 0.4)


def _rebalance(positions: dict, cash_gbp: float, targets: dict, prices_gbp: dict, commission: float) -> tuple[dict, float, list]:
    trades = []
    equity = cash_gbp + sum(positions[m] * prices_gbp[m] for m in positions)
    total_target = sum(targets.values())
    if total_target > 1:
        targets = {k: v / total_target for k, v in targets.items()}
    for metal, target_w in targets.items():
        price = prices_gbp[metal]
        if np.isnan(price) or price <= 0:
            continue
        current_units = positions.get(metal, 0.0)
        target_notional = equity * target_w
        desired_units = target_notional / price
        delta_units = desired_units - current_units
        if abs(delta_units) < 1e-9:
            continue
        trade_notional_gbp = delta_units * price
        cost = commission if abs(delta_units) > 0 else 0.0
        if trade_notional_gbp > 0:
            max_affordable_units = max(0.0, (cash_gbp - cost) / price)
            delta_units = min(delta_units, max_affordable_units)
            trade_notional_gbp = delta_units * price
        cash_gbp -= trade_notional_gbp + cost
        positions[metal] = current_units + delta_units
        trades.append({
            "metal": metal,
            "target_weight": target_w,
            "units_delta": delta_units,
            "notional_gbp": trade_notional_gbp,
            "commission_gbp": cost,
        })
    return positions, cash_gbp, trades


def _simulate_with_cache(start: datetime, initial_cash: float, tf_weights: dict,
                        commission_gbp: float, trade_scenarios: dict, strategy: str) -> dict:
    trade_scenarios = trade_scenarios or TRADE_SCENARIOS
    start_buffer = start - timedelta(days=400)

    gold_daily = _to_london(_fetch_history("GC=F", start_buffer, "1d"))
    silver_daily = _to_london(_fetch_history("SI=F", start_buffer, "1d"))
    gold_hourly = _to_london(_fetch_history("GC=F", start, "1h"))
    silver_hourly = _to_london(_fetch_history("SI=F", start, "1h"))
    fx_hourly = _to_london(_fetch_history(GBPUSD_TICKER, start, "1h"))

    results = {}
    scenario_states = {scenario: {
        "gold": {"last_target": 0.0, "cooldown": 0, "bull_count": 0, "consec_short": 0, "consec_med": 0, "prev_short": None},
        "silver": {"last_target": 0.0, "cooldown": 0, "bull_count": 0, "consec_short": 0, "consec_med": 0, "prev_short": None},
    } for scenario in trade_scenarios}

    for scenario, cfg in trade_scenarios.items():
        hour = cfg.get("hour", "all")
        scenario_tf_weights = cfg.get("tf_weights") or tf_weights
        positions = {"gold": 0.0, "silver": 0.0}
        cash_gbp = initial_cash
        equity_records = []
        trade_records = []
        hourly_index = gold_hourly.index.intersection(silver_hourly.index).intersection(fx_hourly.index)
        hourly_index = hourly_index.sort_values()
        last_prices_gbp = {"gold": np.nan, "silver": np.nan}
        for ts in hourly_index:
            fx_row = fx_hourly.loc[ts]
            gbp_usd = float(fx_row["Close"])
            gold_row = gold_hourly.loc[ts]
            silver_row = silver_hourly.loc[ts]
            prices_gbp = {
                "gold": float(gold_row["Close"]) / gbp_usd if gbp_usd else np.nan,
                "silver": float(silver_row["Close"]) / gbp_usd if gbp_usd else np.nan,
            }
            last_prices_gbp = prices_gbp
            equity = cash_gbp + sum(positions[m] * prices_gbp[m] for m in positions)
            equity_records.append((ts, equity, equity * gbp_usd))

            if hour != "all" and ts.hour != hour:
                continue
            prev_idx = hourly_index[hourly_index.get_loc(ts) - 1] if hourly_index.get_loc(ts) > 0 else None
            if prev_idx is None:
                continue
            gold_sig, gold_score = _compute_signal("gold", gold_daily, gold_hourly, prev_idx, scenario_tf_weights)
            silver_sig, silver_score = _compute_signal("silver", silver_daily, silver_hourly, prev_idx, scenario_tf_weights)

            def _pick_target(metal_signal: str, metal_score: dict, metal: str) -> float:
                state = scenario_states[scenario][metal]
                short_score = metal_score.get("timeframe_scores", {}).get("Short", 0.0)
                med_score = metal_score.get("timeframe_scores", {}).get("Medium", 0.0)
                comp = metal_score.get("composite_score", 0.0)

                # Update counters
                state["consec_short"] = state["consec_short"] + 1 if short_score > 0 else 0
                state["consec_med"] = state["consec_med"] + 1 if med_score > 0 else 0
                state["bull_count"] = state["bull_count"] + 1 if metal_signal in (SIGNAL_BUY, SIGNAL_STRONG_BUY) else 0
                if state["cooldown"] > 0:
                    state["cooldown"] -= 1

                def _apply_strategy() -> float:
                    if strategy == "agree":
                        return _target_agree(metal_score)
                    if strategy == "hysteresis":
                        return _target_hysteresis(metal_score, state["last_target"])
                    if strategy == "banded":
                        return _target_banded(metal_score)
                    if strategy == "confirm":
                        if state["bull_count"] >= 2:
                            return _target_baseline(metal_signal)
                        return 0.0
                    if strategy == "cooldown":
                        if metal_signal in (SIGNAL_SELL, SIGNAL_STRONG_SELL, SIGNAL_NEUTRAL):
                            state["cooldown"] = 3
                        if metal_signal in (SIGNAL_BUY, SIGNAL_STRONG_BUY):
                            if state["cooldown"] > 0 and comp < 0.35 and state["last_target"] == 0:
                                return 0.0
                        return _target_baseline(metal_signal)
                    if strategy == "time_filter":
                        if state["consec_short"] >= 3 and state["consec_med"] >= 2:
                            return _target_baseline(metal_signal)
                        return 0.0
                    if strategy == "decay":
                        base = _target_baseline(metal_signal)
                        prev_s = state.get("prev_short")
                        if base > 0 and prev_s is not None and short_score < prev_s and short_score > 0:
                            base = max(0.0, base - 0.05)
                        return base
                    return _target_baseline(metal_signal)

                tgt = _apply_strategy()
                tgt = max(0.0, min(1.0, tgt))
                state["last_target"] = tgt
                state["prev_short"] = short_score
                return tgt

            targets = {
                "gold": _pick_target(gold_sig, gold_score, "gold"),
                "silver": _pick_target(silver_sig, silver_score, "silver"),
            }
            positions, cash_gbp, trades = _rebalance(positions, cash_gbp, targets, prices_gbp, commission_gbp)
            for tr in trades:
                tr.update({
                    "timestamp": ts,
                    "price_gbp": prices_gbp[tr["metal"]],
                    "gbp_usd": gbp_usd,
                    "signal": gold_sig if tr["metal"] == "gold" else silver_sig,
                    "equity_gbp_after": cash_gbp + sum(positions[m] * prices_gbp[m] for m in positions),
                    "equity_usd_after": (cash_gbp + sum(positions[m] * prices_gbp[m] for m in positions)) * gbp_usd,
                })
                tr["pnl_gbp_abs"] = tr["equity_gbp_after"] - initial_cash
                tr["pnl_gbp_pct"] = tr["pnl_gbp_abs"] / initial_cash if initial_cash else 0.0
                trade_records.append(tr)

        if len(equity_records) == 0:
            results[scenario] = {"error": "no data for scenario"}
            continue

        eq_df = pd.DataFrame(equity_records, columns=["ts", "equity_gbp", "equity_usd"])
        eq_df = eq_df.set_index("ts")
        trades_df = pd.DataFrame(trade_records)
        if eq_df.empty:
            results[scenario] = {"error": "no equity series"}
            continue
        metrics = _perf_stats(eq_df["equity_gbp"])
        final_eq_gbp = eq_df["equity_gbp"].iloc[-1]
        final_eq_usd = eq_df["equity_usd"].iloc[-1]
        pnl_gbp_abs = final_eq_gbp - initial_cash
        base_usd = eq_df["equity_usd"].iloc[0] if eq_df["equity_gbp"].iloc[0] else initial_cash
        pnl_usd_abs = final_eq_usd - base_usd
        metrics.update({
            "final_equity_gbp": float(final_eq_gbp),
            "final_equity_usd": float(final_eq_usd),
            "pnl_gbp_abs": float(pnl_gbp_abs),
            "pnl_gbp_pct": float(pnl_gbp_abs / initial_cash) if initial_cash else 0.0,
            "pnl_usd_abs": float(pnl_usd_abs),
            "pnl_usd_pct": float(pnl_usd_abs / (initial_cash if initial_cash else 1)),
            "initial_cash": float(initial_cash),
        })
        results[scenario] = {
            "equity": eq_df,
            "trades": trades_df,
            "metrics": metrics,
            "final_positions": {"positions": positions, "cash_gbp": cash_gbp, "last_prices_gbp": last_prices_gbp},
        }
    return results


@lru_cache(maxsize=64)
def simulate_cached(start_iso: str, initial_cash: float, tf_weights_key: tuple, strategy: str) -> dict:
    tf_dict = {"Short": tf_weights_key[0], "Medium": tf_weights_key[1], "Long": tf_weights_key[2]}
    return _simulate_with_cache(
        start=datetime.fromisoformat(start_iso),
        initial_cash=initial_cash,
        tf_weights=tf_dict,
        commission_gbp=10.0,
        trade_scenarios=TRADE_SCENARIOS,
        strategy=strategy,
    )


def simulate(start: datetime = START_DEFAULT, initial_cash: float = 2_000_000.0,
             tf_weights: dict | None = None, commission_gbp: float = 10.0,
             trade_scenarios: dict = None, strategy: str = "baseline") -> dict:
    tf_weights = tf_weights or DEFAULT_TF_WEIGHTS
    tf_key = (tf_weights.get("Short", 48), tf_weights.get("Medium", 37), tf_weights.get("Long", 15))
    if trade_scenarios is None and commission_gbp == 10.0:
        return simulate_cached(start.isoformat(), initial_cash, tf_key, strategy)
    return _simulate_with_cache(start, initial_cash, tf_weights, commission_gbp, trade_scenarios or TRADE_SCENARIOS, strategy)


def run_simulations(start_date: datetime = START_DEFAULT, tf_weights: dict | None = None, strategy: str = "baseline") -> dict:
    return simulate(start=start_date, tf_weights=tf_weights or DEFAULT_TF_WEIGHTS, strategy=strategy)


def clear_simulator_caches() -> None:
    with _HISTORY_CACHE_LOCK:
        _HISTORY_CACHE.clear()
    simulate_cached.cache_clear()
