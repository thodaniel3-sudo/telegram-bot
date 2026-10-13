# V100 Telegram Bot

A Telegram bot that analyzes the **Deriv Volatility 100 (1s) Index**
(symbol `1HZ100V`) and provides a 15-minute directional forecast
(UPWARD BIAS / DOWNWARD BIAS / NEUTRAL), along with tracked prediction
performance.

## ⚠️ Disclaimer

This bot produces **statistical market estimates**, not guarantees.
It does **not** place trades. It does **not** request trading credentials.
Past behavior of synthetic indices does not guarantee future results.
Do not use it as the sole basis for any financial decision.

## Features

- Live connection to Deriv's public WebSocket API (no account needed)
- Technical indicators: EMA 9, EMA 21, RSI 14, recent volatility, momentum
- Directional forecast over a configurable horizon (default 15 minutes)
- SQLite-based prediction tracking and outcome evaluation
- Telegram commands: `/start`, `/help`, `/analyze`, `/status`,
  `/accuracy`, `/history`, `/settings`
- Optional machine-learning model (planned, not yet enabled)

## Project Structure

- `main.py` — Entry point. Starts the Telegram bot.
- `config.py` — Loads secrets and settings from `.env`.
- `market_data.py` — Connects to Deriv, fetches candles.
- `indicators.py` — Calculates EMA, RSI, volatility, etc.
- `predictor.py` — Combines indicators into UP / DOWN / NEUTRAL.
- `database.py` — SQLite: saves predictions, evaluates them.
- `backtest.py` — Runs historical tests on the strategy.
- `test_indicators.py` — Basic automated tests.

## Setup (Windows)

1. Install Python 3.12 from https://python.org (check "Add to PATH").
2. Install VS Code and the Python extension.
3. Open this folder in VS Code.
4. Create a virtual environment:
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1