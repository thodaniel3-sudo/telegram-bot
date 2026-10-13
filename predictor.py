"""
predictor.py
------------
Turn indicator values into a directional forecast.

Public API
----------
Forecast                        dataclass holding the result.
predict(df) -> Forecast         Analyze the most recent row of df.

The returned Forecast has
    direction   : "UPWARD BIAS" | "DOWNWARD BIAS" | "NEUTRAL"
    score       : int  (net agreement count; NOT a probability)
    reasons     : list[str]  human-readable explanations
    snapshot    : dict  of the latest indicator values

The predictor uses simple rules. It does not use machine learning.
Every rule is transparent and easy to inspect or adjust.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


# ─────────────────────────────────────────────────────────────
# Tunable thresholds
# ─────────────────────────────────────────────────────────────
# Thresholds are intentionally small because Volatility 100 (1s)
# moves in small increments. All values are in percent.
EMA_DIFF_THRESHOLD_PCT = 0.02    # |ema_9 - ema_21| as % of price
RSI_BULLISH = 60.0
RSI_BEARISH = 40.0
MOMENTUM_BULLISH_PCT = 0.05
MOMENTUM_BEARISH_PCT = -0.05
TREND_BULLISH = 0.60
TREND_BEARISH = 0.40

# Net score required to make a directional call.
DECISION_THRESHOLD = 2


# ─────────────────────────────────────────────────────────────
# Data container
# ─────────────────────────────────────────────────────────────
@dataclass
class Forecast:
    direction: str
    score: int
    reasons: list[str] = field(default_factory=list)
    snapshot: dict[str, Any] = field(default_factory=dict)

    @property
    def is_directional(self) -> bool:
        return self.direction in ("UPWARD BIAS", "DOWNWARD BIAS")


# ─────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────
def predict(df: pd.DataFrame) -> Forecast:
    """
    Analyze the most recent candle and return a Forecast.

    Parameters
    ----------
    df : pandas.DataFrame
        Output of indicators.add_indicators(). Must contain at least:
        close, ema_9, ema_21, ema_diff_pct, rsi_14,
        volatility, momentum_5, trend_score
    """
    required = {
        "close",
        "ema_9",
        "ema_21",
        "ema_diff_pct",
        "rsi_14",
        "volatility",
        "momentum_5",
        "trend_score",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"DataFrame is missing indicator columns: {sorted(missing)}. "
            "Did you forget to call indicators.add_indicators()?"
        )

    last = df.iloc[-1]

    # Read each indicator into a plain Python value.
    close = float(last["close"])
    ema_9 = float(last["ema_9"])
    ema_21 = float(last["ema_21"])
    ema_diff_pct = float(last["ema_diff_pct"])
    rsi = float(last["rsi_14"])
    vol = float(last["volatility"])
    momentum = float(last["momentum_5"])
    trend = float(last["trend_score"])

    snapshot = {
        "close": close,
        "ema_9": ema_9,
        "ema_21": ema_21,
        "ema_diff_pct": ema_diff_pct,
        "rsi_14": rsi,
        "volatility": vol,
        "momentum_5": momentum,
        "trend_score": trend,
    }

    score = 0
    reasons: list[str] = []

    # ── Rule 1: EMA trend ──────────────────────────────────
    if ema_diff_pct > EMA_DIFF_THRESHOLD_PCT:
        score += 1
        reasons.append(
            f"EMA 9 is above EMA 21 by {ema_diff_pct:+.3f}% — short-term trend is upward."
        )
    elif ema_diff_pct < -EMA_DIFF_THRESHOLD_PCT:
        score -= 1
        reasons.append(
            f"EMA 9 is below EMA 21 by {ema_diff_pct:+.3f}% — short-term trend is downward."
        )
    else:
        reasons.append(
            f"EMA 9 and EMA 21 are essentially equal ({ema_diff_pct:+.3f}%) — no trend signal."
        )

    # ── Rule 2: RSI ────────────────────────────────────────
    if rsi >= RSI_BULLISH:
        score += 1
        reasons.append(
            f"RSI 14 is {rsi:.1f} (above {RSI_BULLISH:.0f}) — momentum is strong."
        )
    elif rsi <= RSI_BEARISH:
        score -= 1
        reasons.append(
            f"RSI 14 is {rsi:.1f} (below {RSI_BEARISH:.0f}) — momentum is weak."
        )
    else:
        reasons.append(
            f"RSI 14 is {rsi:.1f} — neutral zone."
        )

    # ── Rule 3: Short-term momentum ────────────────────────
    if momentum >= MOMENTUM_BULLISH_PCT:
        score += 1
        reasons.append(
            f"5-candle momentum is {momentum:+.3f}% — recent candles rising."
        )
    elif momentum <= MOMENTUM_BEARISH_PCT:
        score -= 1
        reasons.append(
            f"5-candle momentum is {momentum:+.3f}% — recent candles falling."
        )
    else:
        reasons.append(
            f"5-candle momentum is {momentum:+.3f}% — flat recent movement."
        )

    # ── Rule 4: Trend consistency ──────────────────────────
    if trend >= TREND_BULLISH:
        score += 1
        reasons.append(
            f"{trend * 100:.0f}% of the last 10 candles closed up — bullish consistency."
        )
    elif trend <= TREND_BEARISH:
        score -= 1
        reasons.append(
            f"{(1 - trend) * 100:.0f}% of the last 10 candles closed down — bearish consistency."
        )
    else:
        reasons.append(
            f"{trend * 100:.0f}% of the last 10 candles closed up — mixed candle directions."
        )

    # ── Add volatility as a neutral context line ──────────
    reasons.append(
        f"Recent realized volatility is {vol:.4f}% — "
        + (
            "market is calm."
            if vol < 0.15
            else "market is active."
            if vol < 0.30
            else "market is unusually volatile."
        )
    )

    # ── Final decision ─────────────────────────────────────
    if score >= DECISION_THRESHOLD:
        direction = "UPWARD BIAS"
    elif score <= -DECISION_THRESHOLD:
        direction = "DOWNWARD BIAS"
    else:
        direction = "NEUTRAL"

    return Forecast(
        direction=direction,
        score=score,
        reasons=reasons,
        snapshot=snapshot,
    )


# ─────────────────────────────────────────────────────────────
# Quick self-test
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import market_data
    import indicators

    print("=" * 70)
    print("predictor.py self-test")
    print("=" * 70)

    print("\n[1] Downloading 200 candles from Deriv...")
    raw = market_data.fetch_candles(count=200)
    print(f"    Received {len(raw)} candles.")

    print("\n[2] Computing indicators...")
    enriched = indicators.add_indicators(raw)

    print("\n[3] Generating forecast...")
    forecast = predict(enriched)

    print("\n" + "-" * 70)
    print(f"Direction : {forecast.direction}")
    print(f"Net score : {forecast.score:+d}   (rule-based agreement, NOT a probability)")
    print("-" * 70)
    print("\nReasons:")
    for r in forecast.reasons:
        print(f"  • {r}")
    print("\nSnapshot:")
    for key, value in forecast.snapshot.items():
        print(f"  {key:<14}: {value}")
    print("\nDone.")