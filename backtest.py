"""
backtest.py
-----------
Walk-forward backtest of the rule-based predictor.

Downloads historical candles, walks forward in time, generates a
forecast at each step using only past data, and evaluates whether
the forecast was correct after the forecast horizon has passed.

Run with:
    python backtest.py

Output:
    - total forecasts generated
    - evaluated forecasts
    - correct / incorrect / neutral
    - directional accuracy (%)
    - NEUTRAL rate (%)
    - breakdown by direction (UPWARD / DOWNWARD)
    - a plain-text English summary that is safe to share

Design
------
- No look-ahead bias: at candle i, the feature window uses [0..i].
- The evaluation target is close[i + horizon_candles].
- The evaluation rule matches database.evaluate_prediction():
    * |move| < MIN_MOVE_PCT      -> FLAT     (directional forecast = incorrect)
    * move > 0                    -> UP
    * move < 0                    -> DOWN
    * NEUTRAL forecasts are never counted as correct or incorrect.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import config
import indicators
import market_data
import predictor


# ─────────────────────────────────────────────────────────────
# Configuration for this backtest
# ─────────────────────────────────────────────────────────────
# How many candles to download. More = more reliable but slower.
# Deriv allows up to 5000 per request. Start with 3000.
CANDLE_COUNT = 3000

# Minimum warm-up candles before we start forecasting.
# indicators.add_indicators() needs 30; we use 50 for safety.
WARMUP = 50

# The evaluation rule uses config.MIN_MOVE_PCT.


# ─────────────────────────────────────────────────────────────
# Results container
# ─────────────────────────────────────────────────────────────
@dataclass
class BacktestResult:
    total: int = 0
    evaluated: int = 0
    correct: int = 0
    incorrect: int = 0
    neutral: int = 0
    correct_up: int = 0
    total_up: int = 0
    correct_down: int = 0
    total_down: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def directional_accuracy_pct(self) -> float | None:
        if self.evaluated == 0:
            return None
        return round(100.0 * self.correct / self.evaluated, 2)

    @property
    def neutral_rate_pct(self) -> float | None:
        if self.total == 0:
            return None
        return round(100.0 * self.neutral / self.total, 2)


# ─────────────────────────────────────────────────────────────
# Core loop
# ─────────────────────────────────────────────────────────────
def _evaluate_direction(
    start_price: float, end_price: float, direction: str
) -> tuple[str, int | None]:
    """
    Reproduce the evaluation rule used by database.evaluate_prediction().
    Returns (actual_direction, is_correct).
    """
    move_pct = (end_price - start_price) / start_price * 100.0

    if abs(move_pct) < config.MIN_MOVE_PCT:
        actual = "FLAT"
    elif move_pct > 0:
        actual = "UP"
    else:
        actual = "DOWN"

    if direction == "NEUTRAL":
        return actual, None
    if direction == "UPWARD BIAS":
        return actual, 1 if actual == "UP" else 0
    if direction == "DOWNWARD BIAS":
        return actual, 1 if actual == "DOWN" else 0
    raise ValueError(f"Unknown direction: {direction!r}")


def run_backtest(candle_count: int = CANDLE_COUNT) -> BacktestResult:
    """
    Download candles, walk forward, and return a BacktestResult.
    """
    horizon_candles = max(
        1, config.FORECAST_HORIZON_MINUTES * 60 // config.CANDLE_GRANULARITY
    )
    print(f"Candle granularity : {config.CANDLE_GRANULARITY}s")
    print(f"Forecast horizon   : {config.FORECAST_HORIZON_MINUTES} min "
          f"= {horizon_candles} candles")
    print(f"Minimum move       : {config.MIN_MOVE_PCT}%")
    print(f"Warm-up candles    : {WARMUP}")
    print(f"Requested candles  : {candle_count}")

    print("\nDownloading candles from Deriv...")
    raw = market_data.fetch_candles(count=candle_count)
    print(f"Received {len(raw)} candles.")

    if len(raw) < WARMUP + horizon_candles + 10:
        raise RuntimeError(
            f"Not enough candles. Need at least "
            f"{WARMUP + horizon_candles + 10}, got {len(raw)}."
        )

    print("\nComputing indicators over the entire history...")
    enriched = indicators.add_indicators(raw)

    result = BacktestResult()

    print("\nWalking forward through history...")
    last_start = len(enriched) - horizon_candles

    for i in range(WARMUP, last_start):
        # Slice: all rows up to and including i. This is what the
        # live bot would see at that moment in time.
        window = enriched.iloc[: i + 1]

        try:
            forecast = predictor.predict(window)
        except Exception as exc:  # noqa: BLE001
            result.errors.append(f"row {i}: {type(exc).__name__}: {exc}")
            continue

        result.total += 1

        if forecast.direction == "NEUTRAL":
            result.neutral += 1
            continue

        start_price = float(window.iloc[-1]["close"])
        end_price = float(enriched.iloc[i + horizon_candles]["close"])

        actual, is_correct = _evaluate_direction(
            start_price, end_price, forecast.direction
        )

        result.evaluated += 1
        if forecast.direction == "UPWARD BIAS":
            result.total_up += 1
            if is_correct == 1:
                result.correct_up += 1
        elif forecast.direction == "DOWNWARD BIAS":
            result.total_down += 1
            if is_correct == 1:
                result.correct_down += 1

        if is_correct == 1:
            result.correct += 1
        elif is_correct == 0:
            result.incorrect += 1

    return result


# ─────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────
def print_report(result: BacktestResult) -> None:
    print("\n" + "=" * 64)
    print("BACKTEST REPORT")
    print("=" * 64)

    print(f"\nTotal forecasts generated : {result.total}")
    print(f"NEUTRAL forecasts         : {result.neutral}")
    print(f"Directional forecasts     : {result.evaluated}")
    print(f"  Correct                 : {result.correct}")
    print(f"  Incorrect               : {result.incorrect}")

    acc = result.directional_accuracy_pct
    if acc is None:
        print("\nDirectional accuracy      : Not enough data")
    else:
        print(f"\nDirectional accuracy      : {acc}%")

    neutral_rate = result.neutral_rate_pct
    if neutral_rate is not None:
        print(f"NEUTRAL rate              : {neutral_rate}%")

    print("\nBreakdown by direction:")
    if result.total_up > 0:
        up_acc = round(100.0 * result.correct_up / result.total_up, 2)
        print(f"  UPWARD BIAS   : {up_acc}% "
              f"({result.correct_up}/{result.total_up})")
    else:
        print("  UPWARD BIAS   : no directional forecasts")

    if result.total_down > 0:
        down_acc = round(100.0 * result.correct_down / result.total_down, 2)
        print(f"  DOWNWARD BIAS : {down_acc}% "
              f"({result.correct_down}/{result.total_down})")
    else:
        print("  DOWNWARD BIAS : no directional forecasts")

    # Honest reliability warning based on sample size.
    n = result.evaluated
    print("\nReliability:")
    if n < 30:
        print("  ⚠️  Sample too small to be meaningful.")
    elif n < 100:
        print("  ⚠️  Modest sample — treat with caution.")
    elif n < 500:
        print("  Fair sample size.")
    else:
        print("  Reasonable sample size.")

    print("\nInterpretation:")
    if acc is None:
        print("  Not enough directional forecasts to judge.")
    elif acc < 50:
        print("  Worse than random. The rules do not work as written.")
    elif acc < 55:
        print("  Around random. The rules may have no real edge at this horizon.")
    elif acc < 65:
        print("  Modest edge — could be genuine, could be noise. "
              "Gather more data before trusting it.")
    elif acc < 80:
        print("  Strong edge in this sample. Verify on a different period "
              "before trusting it.")
    else:
        print("  Suspiciously high. Double-check for data leakage.")

    if result.errors:
        print(f"\nErrors during backtest: {len(result.errors)}")
        for e in result.errors[:5]:
            print(f"  {e}")
        if len(result.errors) > 5:
            print(f"  ... and {len(result.errors) - 5} more.")


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    result = run_backtest()
    print_report(result)