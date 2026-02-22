# Gold & Silver Strategy Monitor

A Streamlit dashboard for monitoring gold and silver prices and generating trading signals aligned to a structured precious metals investment strategy.

## Features

- **Dual-price tracking**: Spot prices (for technical analysis signals) + ETC prices (for actual Hargreaves Lansdown trading)
- **Dynamic support/resistance**: Fibonacci levels at 3 timeframes (2-year, 3-month, 2-3 week)
- **VOBHS indicators**: Volatility Oscillator, Boom Hunter Pro (BHS), Hull Moving Average, Modified ATR
- **Technical analysis**: RSI, SMA crossovers, Bollinger squeeze, momentum bars, triangle pattern detection
- **Whale accumulation detection**: Volume spike flagging on futures data
- **Composite signal engine**: Multi-indicator scoring (Strong Buy / Buy / Neutral / Sell)
- **GBP 2M portfolio tracker**: Purchase logging, live P&L, deployment progress, allocation recommendations
- **Catalyst news feed**: Filtered RSS headlines for Fed, Iran, tariffs, Asia demand, central bank buying
- **Configurable parameters**: All thresholds adjustable via sidebar
- **ETC tracking difference**: Spot vs ETC divergence monitoring

## Supported ETCs

- SGLN.L (iShares Physical Gold)
- SSLN.L (iShares Physical Silver)
- SGLS.L (iShares Physical Gold GBP Hedged)
- SLVP.L (iShares Physical Silver GBP Hedged)

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Project Structure

```
app.py          - Streamlit dashboard
data.py         - Price fetching (yfinance), FX conversion, historical data
technicals.py   - All technical indicators (Fibonacci, VOBHS, RSI, SMA, triangles, Bollinger, momentum, whale detection)
signals.py      - Composite signal scoring engine
portfolio.py    - GBP portfolio tracker with JSON persistence
news.py         - RSS catalyst news feed
```

## Disclaimer

This is not financial advice. Precious metals carry significant risk. Consult a qualified advisor.
