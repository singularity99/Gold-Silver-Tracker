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
TRADE_WINDOWS = {
    "morning": 10,  # 09–10 London bar close
    "end_of_day": 17,  # 16–17 London bar close
}
TARGET_ALLOC = {
    SIGNAL_STRONG_BUY: 0.60,
    SIGNAL_BUY: 0.30,
    SIGNAL_NEUTRAL: 0.0,
    SIGNAL_SELL: 0.30,
    SIGNAL_STRONG_SELL: 0.30,
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
             trade_hours: dict = None) -> dict:
    tf_weights = tf_weights or DEFAULT_TF_WEIGHTS
    trade_hours = trade_hours or TRADE_WINDOWS
    start_buffer = start - timedelta(days=400)

    gold_daily = _to_london(_fetch_history("GC=F", start_buffer, "1d"))
    silver_daily = _to_london(_fetch_history("SI=F", start_buffer, "1d"))
    gold_hourly = _to_london(_fetch_history("GC=F", start, "1h"))
    silver_hourly = _to_london(_fetch_history("SI=F", start, "1h"))
    fx_hourly = _to_london(_fetch_history(GBPUSD_TICKER, start, "1h"))

    results = {}
    for scenario, hour in trade_hours.items():
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

            if ts.hour != hour:
                continue
            prev_idx = hourly_index[hourly_index.get_loc(ts) - 1] if hourly_index.get_loc(ts) > 0 else None
            if prev_idx is None:
                continue
            gold_sig, _ = _compute_signal("gold", gold_daily, gold_hourly, prev_idx, tf_weights)
            silver_sig, _ = _compute_signal("silver", silver_daily, silver_hourly, prev_idx, tf_weights)
            targets = {
                "gold": _signal_to_weight(gold_sig),
                "silver": _signal_to_weight(silver_sig),
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


def run_simulations(start_date: datetime = START_DEFAULT, tf_weights: dict | None = None) -> dict:
    return simulate(start=start_date, tf_weights=tf_weights or DEFAULT_TF_WEIGHTS)
