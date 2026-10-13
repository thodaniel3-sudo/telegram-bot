"""
indicators.py
-------------
Compute technical indicators on a DataFrame of candles.

Public function
---------------
add_indicators(df) -> pandas.DataFrame
    Return a copy of df with indicator columns appended.

Every indicator at row i uses only rows 0..i. No look-ahead.

Indicators provided
-------------------
ema_9          Exponential Moving Average, span 9
ema_21         Exponential Moving Average, span 21
ema_diff_pct   (ema_9 - ema_21) / ema_21 * 100
rsi_14         Relative Strength Index, period 14 (0..100)
volatility     Standard deviation of last 20 returns (%)
momentum_5     Percent change of close over the last 5 candles
trend_score    Fraction of last 10 candles that closed up, in [0, 1]
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────
# Individual indicator implementations
# ─────────────────────────────────────────────────────────────
def _ema(series: pd.Series, span: int) -> pd.Series:
    """
    Exponential Moving Average.

    The first value equals the first data point (pandas behaviour).
    Later values weight recent data more heavily via the smoothing
    factor alpha = 2 / (span + 1).
    """
    return series.ewm(span=span, adjust=False).mean()


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Relative Strength Index.

    RSI = 100 - 100 / (1 + average_gain / average_loss)

    Values near 100 mean the market has been rising almost
    every candle. Values near 0 mean it has been falling almost
    every candle. 50 is neutral.
    """
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    # Wilder's smoothing = exponential moving average with alpha = 1/period.
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

    # Avoid division by zero: replace 0 avg_loss with a tiny number.
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # If avg_loss is 0 and avg_gain > 0, RSI should be 100.
    rsi = rsi.where(avg_loss != 0.0, 100.0)
    # If both are 0 (flat market), RSI is undefined — set to 50.
    rsi = rsi.where(~((avg_gain == 0.0) & (avg_loss == 0.0)), 50.0)

    return rsi


def _rolling_volatility(series: pd.Series, window: int = 20) -> pd.Series:
    """
    Realized volatility.

    Percentage log-returns over the last `window` candles, then take
    the standard deviation. Returned as a percentage.
    """
    log_returns = np.log(series / series.shift(1))
    return log_returns.rolling(window=window, min_periods=2).std() * 100.0


def _momentum(series: pd.Series, lookback: int = 5) -> pd.Series:
    """Percent change of the close over the last `lookback` candles."""
    return series.pct_change(periods=lookback) * 100.0


def _trend_score(df: pd.DataFrame, window: int = 10) -> pd.Series:
    """
    Fraction of the last `window` candles that closed above their open.

    Returns a value in [0, 1]. 1.0 means every recent candle was up,
    0.0 means every recent candle was down.
    """
    up = (df["close"] > df["open"]).astype(float)
    return up.rolling(window=window, min_periods=1).mean()


# ─────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy of df with indicator columns appended.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain at least the columns: open, high, low, close.
        Usually produced by market_data.fetch_candles().

    Returns
    -------
    pandas.DataFrame
        The input rows plus columns:
        ema_9, ema_21, ema_diff_pct, rsi_14,
        volatility, momentum_5, trend_score
    """
    required = {"open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Candle DataFrame is missing columns: {sorted(missing)}")

    if len(df) < 30:
        raise ValueError(
            f"Need at least 30 candles to compute indicators; got {len(df)}."
        )

    out = df.copy()

    close = out["close"]

    out["ema_9"] = _ema(close, span=9)
    out["ema_21"] = _ema(close, span=21)
    out["ema_diff_pct"] = (out["ema_9"] - out["ema_21"]) / out["ema_21"] * 100.0

    out["rsi_14"] = _rsi(close, period=14)

    out["volatility"] = _rolling_volatility(close, window=20)

    out["momentum_5"] = _momentum(close, lookback=5)

    out["trend_score"] = _trend_score(out, window=10)

    return out


# ─────────────────────────────────────────────────────────────
# Quick self-test
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import market_data

    print("=" * 70)
    print("indicators.py self-test")
    print("=" * 70)

    print("\n[1] Downloading 120 candles from Deriv...")
    df = market_data.fetch_candles(count=120)
    print(f"    Received {len(df)} candles.")

    print("\n[2] Computing indicators...")
    enriched = add_indicators(df)
    print(f"    Output columns: {list(enriched.columns)}")

    print("\n[3] Last 5 rows (rounded for readability):\n")
    cols_to_show = [
        "datetime",
        "close",
        "ema_9",
        "ema_21",
        "ema_diff_pct",
        "rsi_14",
        "volatility",
        "momentum_5",
        "trend_score",
    ]
    printable = enriched[cols_to_show].tail(5).copy()
    for col in printable.columns:
        if col != "datetime":
            printable[col] = printable[col].round(4)
    print(printable.to_string(index=False))

    print("\n[4] Sanity checks on the most recent row:")
    last = enriched.iloc[-1]

    ok = True

    if not (0.0 <= last["rsi_14"] <= 100.0):
        print(f"    FAIL: rsi_14 out of range: {last['rsi_14']}")
        ok = False
    else:
        print(f"    OK: rsi_14 = {last['rsi_14']:.2f}  (in [0, 100])")

    if not (0.0 <= last["trend_score"] <= 1.0):
        print(f"    FAIL: trend_score out of range: {last['trend_score']}")
        ok = False
    else:
        print(f"    OK: trend_score = {last['trend_score']:.2f}  (in [0, 1])")

    if not (last["volatility"] >= 0.0):
        print(f"    FAIL: volatility is negative: {last['volatility']}")
        ok = False
    else:
        print(f"    OK: volatility = {last['volatility']:.4f} %")

    if ok:
        print("\nAll checks passed.")
    else:
        print("\nSome checks failed.")
        raise SystemExit(1)