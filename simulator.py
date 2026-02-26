import numpy as np
import pandas as pd
import yfinance as yf
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
    "morning": {"hour": 10, "tf_weights": DEFAULT_TF_WEIGHTS},  # 09–10 London bar
    "end_of_day": {"hour": 17, "tf_weights": DEFAULT_TF_WEIGHTS},  # 16–17 London bar
    "intraday_short": {"hour": "all", "tf_weights": {"Short": 100, "Medium": 0, "Long": 0}},
}
TARGET_ALLOC = {
    SIGNAL_STRONG_BUY: 0.60,
    SIGNAL_BUY: 0.30,
    SIGNAL_NEUTRAL: 0.0,
    SIGNAL_SELL: 0.30,
    SIGNAL_STRONG_SELL: 0.30,
}

DEFAULT_OVERLAYS = {
    "regime_filter": False,
    "low_vol_thresh": 0.03,  # 3% daily vol
    "high_vol_thresh": 0.06,  # 6% daily vol (breakout regime)
    "volume_filter": False,
    "vol_percentile": 40,  # require current hourly vol >= 40th percentile of lookback
    "adx_filter": False,
    "adx_threshold": 20,
    "atr_overlays": False,
    "atr_stop_mult": 2.0,
    "atr_tp_mult": 3.0,
}


def _fetch_history(ticker: str, start: datetime, interval: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, interval=interval, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index)
    return df


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


def simulate(start: datetime = START_DEFAULT, initial_cash: float = 2_000_000.0,
             tf_weights: dict | None = None, commission_gbp: float = 10.0,
             trade_scenarios: dict = None, strategy: str = "baseline",
             overlays: dict | None = None) -> dict:
    tf_weights = tf_weights or DEFAULT_TF_WEIGHTS
    trade_scenarios = trade_scenarios or TRADE_SCENARIOS
    overlays = {**DEFAULT_OVERLAYS, **(overlays or {})}
    start_buffer = start - timedelta(days=400)

    gold_daily = _to_london(_fetch_history("GC=F", start_buffer, "1d"))
    silver_daily = _to_london(_fetch_history("SI=F", start_buffer, "1d"))
    gold_hourly = _to_london(_fetch_history("GC=F", start, "1h"))
    silver_hourly = _to_london(_fetch_history("SI=F", start, "1h"))
    fx_hourly = _to_london(_fetch_history(GBPUSD_TICKER, start, "1h"))

    results = {}
    scenario_states = {scenario: {"gold": {"last_target": 0.0, "entry_price": None, "entry_atr": None},
                                   "silver": {"last_target": 0.0, "entry_price": None, "entry_atr": None}}
                       for scenario in trade_scenarios}

    for scenario, cfg in trade_scenarios.items():
        hour = cfg.get("hour", "all")
        scenario_tf_weights = cfg.get("tf_weights", tf_weights)
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

            def _gates(metal: str, metal_score: dict, current_price: float) -> bool:
                # Regime filter: daily realized vol 20d
                if overlays.get("regime_filter", False):
                    daily_df = gold_daily if metal == "gold" else silver_daily
                    if len(daily_df) >= 25:
                        ret = daily_df["Close"].pct_change().dropna()
                        rv = ret.rolling(20).std().iloc[-1] if len(ret) >= 20 else np.nan
                        if not np.isnan(rv):
                            if not (rv <= overlays["low_vol_thresh"] or rv >= overlays["high_vol_thresh"]):
                                return False

                # Volume filter (intraday): require current hourly vol above percentile
                if overlays.get("volume_filter", False):
                    hourly_df = gold_hourly if metal == "gold" else silver_hourly
                    lookback = hourly_df[hourly_df.index >= ts - timedelta(days=30)]
                    if not lookback.empty and not np.isnan(hourly_df.loc[ts, "Volume"]):
                        pct = np.nanpercentile(lookback["Volume"].values, overlays.get("vol_percentile", 40))
                        if hourly_df.loc[ts, "Volume"] < pct:
                            return False

                # ADX filter on short window
                if overlays.get("adx_filter", False):
                    short_df = (gold_hourly if metal == "gold" else silver_hourly)
                    adx_val = _adx(short_df[short_df.index <= ts].tail(200))
                    if not np.isnan(adx_val) and adx_val < overlays.get("adx_threshold", 20):
                        return False

                # ATR stop/take-profit handled separately
                return True

            def _pick_target(metal_signal: str, metal_score: dict, metal: str) -> float:
                if strategy == "agree":
                    return _target_agree(metal_score)
                if strategy == "hysteresis":
                    last_t = scenario_states[scenario][metal]["last_target"]
                    tgt = _target_hysteresis(metal_score, last_t)
                    scenario_states[scenario][metal]["last_target"] = tgt
                    return tgt
                return _target_baseline(metal_signal)

            targets = {}
            for metal, sig, score in (("gold", gold_sig, gold_score), ("silver", silver_sig, silver_score)):
                price_now = prices_gbp[metal]
                if overlays.get("atr_overlays", False):
                    short_df = (gold_hourly if metal == "gold" else silver_hourly)
                    atr_val = _atr(short_df[short_df.index <= ts].tail(200))
                    st_m = scenario_states[scenario][metal]
                    if st_m["entry_price"] and not np.isnan(atr_val):
                        stop = st_m["entry_price"] - overlays.get("atr_stop_mult", 2.0) * atr_val
                        tp = st_m["entry_price"] + overlays.get("atr_tp_mult", 3.0) * atr_val
                        if price_now <= stop or price_now >= tp:
                            targets[metal] = 0.0
                            continue
                if _gates(metal, score, price_now):
                    tgt = _pick_target(sig, score, metal)
                else:
                    tgt = (positions[metal] * price_now) / equity if equity > 0 else 0.0  # hold
                targets[metal] = tgt
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

            # Track entries for ATR overlay
            if overlays.get("atr_overlays", False):
                for tr in trades:
                    metal = tr["metal"]
                    st_m = scenario_states[scenario][metal]
                    short_df = (gold_hourly if metal == "gold" else silver_hourly)
                    atr_val = _atr(short_df[short_df.index <= ts].tail(200))
                    if tr["units_delta"] > 0:  # buy adds/opens
                        st_m["entry_price"] = tr["price_gbp"]
                        st_m["entry_atr"] = atr_val
                    if positions[metal] <= 0:
                        st_m["entry_price"] = None
                        st_m["entry_atr"] = None

        eq_df = pd.DataFrame(equity_records, columns=["ts", "equity_gbp", "equity_usd"])
        eq_df = eq_df.set_index("ts")
        trades_df = pd.DataFrame(trade_records)
        metrics = _perf_stats(eq_df["equity_gbp"])
        final_eq_gbp = eq_df["equity_gbp"].iloc[-1] if not eq_df.empty else initial_cash
        final_eq_usd = eq_df["equity_usd"].iloc[-1] if not eq_df.empty else initial_cash
        pnl_gbp_abs = final_eq_gbp - initial_cash
        pnl_usd_abs = final_eq_usd - (initial_cash * (eq_df["equity_usd"].iloc[0] / eq_df["equity_gbp"].iloc[0]) if not eq_df.empty and eq_df["equity_gbp"].iloc[0] else initial_cash)
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


def run_simulations(start_date: datetime = START_DEFAULT, tf_weights: dict | None = None, strategy: str = "baseline", overlays: dict | None = None) -> dict:
    return simulate(start=start_date, tf_weights=tf_weights or DEFAULT_TF_WEIGHTS, strategy=strategy, overlays=overlays)
