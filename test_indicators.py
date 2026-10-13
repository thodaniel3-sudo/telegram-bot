"""
test_indicators.py
------------------
Automated tests for indicators.py, predictor.py, and database.py.

Run with:
    pytest test_indicators.py -v

Every test uses synthetic data, not live market data, so results
are deterministic and can be checked at any time.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

import database
import indicators
import predictor


# ─────────────────────────────────────────────────────────────
# Synthetic candle factory
# ─────────────────────────────────────────────────────────────
def make_candles(closes: list[float]) -> pd.DataFrame:
    """
    Build a minimal candle DataFrame from a list of close prices.

    Each candle's open equals the previous close (or the close itself
    for the very first one). High is close + 0.1, low is close - 0.1.
    Epochs are 60 seconds apart ending at "now".
    """
    n = len(closes)
    now = datetime.now(timezone.utc).replace(microsecond=0)
    epochs = [
        int((now - timedelta(minutes=(n - 1 - i))).timestamp())
        for i in range(n)
    ]

    opens = [closes[0]] + closes[:-1]
    highs = [c + 0.1 for c in closes]
    lows = [c - 0.1 for c in closes]

    return pd.DataFrame(
        {
            "epoch": epochs,
            "datetime": pd.to_datetime(epochs, unit="s", utc=True),
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
        }
    )


def rising_series(n: int = 60, start: float = 100.0, step: float = 0.5) -> list[float]:
    return [start + step * i for i in range(n)]


def falling_series(n: int = 60, start: float = 100.0, step: float = 0.5) -> list[float]:
    return [start - step * i for i in range(n)]


def flat_series(n: int = 60, value: float = 100.0) -> list[float]:
    return [value] * n


# ─────────────────────────────────────────────────────────────
# Tests: indicators.add_indicators
# ─────────────────────────────────────────────────────────────
def test_add_indicators_returns_expected_columns():
    df = make_candles(rising_series(60))
    enriched = indicators.add_indicators(df)
    expected = {
        "ema_9",
        "ema_21",
        "ema_diff_pct",
        "rsi_14",
        "volatility",
        "momentum_5",
        "trend_score",
    }
    assert expected.issubset(set(enriched.columns))


def test_add_indicators_rejects_too_few_rows():
    df = make_candles(rising_series(10))
    with pytest.raises(ValueError):
        indicators.add_indicators(df)


def test_add_indicators_rejects_missing_columns():
    df = pd.DataFrame({"close": [1, 2, 3]})
    with pytest.raises(ValueError):
        indicators.add_indicators(df)


def test_rsi_is_100_on_pure_uptrend():
    """If every candle is higher than the last, RSI must be at 100."""
    df = make_candles(rising_series(50))
    enriched = indicators.add_indicators(df)
    last_rsi = enriched["rsi_14"].iloc[-1]
    assert last_rsi == pytest.approx(100.0, abs=0.01)


def test_rsi_is_0_on_pure_downtrend():
    """If every candle is lower than the last, RSI must be near 0."""
    df = make_candles(falling_series(50))
    enriched = indicators.add_indicators(df)
    last_rsi = enriched["rsi_14"].iloc[-1]
    assert last_rsi == pytest.approx(0.0, abs=0.01)


def test_rsi_is_50_on_flat_market():
    """A market with no changes produces RSI = 50 (our choice for undefined)."""
    df = make_candles(flat_series(50))
    enriched = indicators.add_indicators(df)
    last_rsi = enriched["rsi_14"].iloc[-1]
    assert last_rsi == pytest.approx(50.0, abs=0.01)


def test_ema_9_above_ema_21_in_uptrend():
    df = make_candles(rising_series(60))
    enriched = indicators.add_indicators(df)
    last = enriched.iloc[-1]
    assert last["ema_9"] > last["ema_21"]
    assert last["ema_diff_pct"] > 0


def test_ema_9_below_ema_21_in_downtrend():
    df = make_candles(falling_series(60))
    enriched = indicators.add_indicators(df)
    last = enriched.iloc[-1]
    assert last["ema_9"] < last["ema_21"]
    assert last["ema_diff_pct"] < 0


def test_volatility_is_zero_in_flat_market():
    df = make_candles(flat_series(60))
    enriched = indicators.add_indicators(df)
    # Skip the first 19 rows where the rolling window has not filled.
    tail = enriched["volatility"].iloc[20:]
    assert (tail < 1e-9).all()


def test_trend_score_matches_candle_direction():
    """In a pure uptrend, every candle closes above its open -> 1.0."""
    df = make_candles(rising_series(30))
    enriched = indicators.add_indicators(df)
    assert enriched["trend_score"].iloc[-1] == pytest.approx(1.0, abs=0.01)

    df = make_candles(falling_series(30))
    enriched = indicators.add_indicators(df)
    assert enriched["trend_score"].iloc[-1] == pytest.approx(0.0, abs=0.01)


# ─────────────────────────────────────────────────────────────
# Tests: predictor.predict
# ─────────────────────────────────────────────────────────────
def test_predictor_returns_upward_bias_on_uptrend():
    df = make_candles(rising_series(60, step=1.0))
    enriched = indicators.add_indicators(df)
    forecast = predictor.predict(enriched)
    assert forecast.direction == "UPWARD BIAS"
    assert forecast.score > 0
    assert len(forecast.reasons) >= 4


def test_predictor_returns_downward_bias_on_downtrend():
    df = make_candles(falling_series(60, step=1.0))
    enriched = indicators.add_indicators(df)
    forecast = predictor.predict(enriched)
    assert forecast.direction == "DOWNWARD BIAS"
    assert forecast.score < 0


def test_predictor_returns_neutral_on_flat_market():
    df = make_candles(flat_series(60))
    enriched = indicators.add_indicators(df)
    forecast = predictor.predict(enriched)
    assert forecast.direction == "NEUTRAL"


def test_predictor_rejects_missing_indicator_columns():
    df = make_candles(rising_series(40))  # no indicators added
    with pytest.raises(ValueError):
        predictor.predict(df)


def test_predictor_snapshot_has_all_expected_keys():
    df = make_candles(rising_series(60))
    enriched = indicators.add_indicators(df)
    forecast = predictor.predict(enriched)
    for key in (
        "close",
        "ema_9",
        "ema_21",
        "ema_diff_pct",
        "rsi_14",
        "volatility",
        "momentum_5",
        "trend_score",
    ):
        assert key in forecast.snapshot


# ─────────────────────────────────────────────────────────────
# Tests: database.evaluate_prediction rules
# ─────────────────────────────────────────────────────────────
@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    """Redirect config.DATABASE_PATH to a temporary file for each test."""
    db_path = tmp_path / "test_predictions.db"
    monkeypatch.setattr(database.config, "DATABASE_PATH", db_path)
    database.init_db()
    yield db_path


def test_evaluate_upward_bias_up_is_correct(temp_db):
    pid = database.record_prediction(
        symbol="TEST",
        start_price=100.0,
        direction="UPWARD BIAS",
        score=2,
        horizon_minutes=0,
        snapshot={"rsi_14": 55.0},
    )
    result = database.evaluate_prediction(pid, end_price=100.5)
    assert result["end_direction"] == "UP"
    assert result["is_correct"] == 1


def test_evaluate_upward_bias_down_is_incorrect(temp_db):
    pid = database.record_prediction(
        symbol="TEST",
        start_price=100.0,
        direction="UPWARD BIAS",
        score=2,
        horizon_minutes=0,
        snapshot={"rsi_14": 55.0},
    )
    result = database.evaluate_prediction(pid, end_price=99.5)
    assert result["end_direction"] == "DOWN"
    assert result["is_correct"] == 0


def test_evaluate_downward_bias_down_is_correct(temp_db):
    pid = database.record_prediction(
        symbol="TEST",
        start_price=100.0,
        direction="DOWNWARD BIAS",
        score=-2,
        horizon_minutes=0,
        snapshot={"rsi_14": 45.0},
    )
    result = database.evaluate_prediction(pid, end_price=99.0)
    assert result["end_direction"] == "DOWN"
    assert result["is_correct"] == 1


def test_evaluate_flat_move_is_incorrect_for_directional(temp_db):
    """A move smaller than MIN_MOVE_PCT counts as FLAT and fails a directional forecast."""
    pid = database.record_prediction(
        symbol="TEST",
        start_price=100.0,
        direction="UPWARD BIAS",
        score=2,
        horizon_minutes=0,
        snapshot={"rsi_14": 55.0},
    )
    # 0.001% move — well below MIN_MOVE_PCT of 0.01%.
    result = database.evaluate_prediction(pid, end_price=100.001)
    assert result["end_direction"] == "FLAT"
    assert result["is_correct"] == 0


def test_evaluate_neutral_is_not_counted(temp_db):
    pid = database.record_prediction(
        symbol="TEST",
        start_price=100.0,
        direction="NEUTRAL",
        score=0,
        horizon_minutes=0,
        snapshot={"rsi_14": 50.0},
    )
    result = database.evaluate_prediction(pid, end_price=105.0)
    # NEUTRAL is never judged right or wrong.
    assert result["is_correct"] is None
    assert result["end_direction"] == "UP"


def test_double_evaluation_is_idempotent(temp_db):
    """Evaluating the same prediction twice must not change the result."""
    pid = database.record_prediction(
        symbol="TEST",
        start_price=100.0,
        direction="UPWARD BIAS",
        score=2,
        horizon_minutes=0,
        snapshot={"rsi_14": 55.0},
    )
    first = database.evaluate_prediction(pid, end_price=101.0)
    second = database.evaluate_prediction(pid, end_price=50.0)  # absurd
    assert first["is_correct"] == second["is_correct"] == 1
    assert second["end_price"] == 101.0  # unchanged


# ─────────────────────────────────────────────────────────────
# Tests: accuracy statistics
# ─────────────────────────────────────────────────────────────
def test_accuracy_stats_counts_correctly(temp_db):
    # Two correct, one incorrect, one neutral.
    pid1 = database.record_prediction(
        symbol="TEST", start_price=100.0, direction="UPWARD BIAS",
        score=2, horizon_minutes=0, snapshot={},
    )
    database.evaluate_prediction(pid1, 101.0)  # correct

    pid2 = database.record_prediction(
        symbol="TEST", start_price=100.0, direction="DOWNWARD BIAS",
        score=-2, horizon_minutes=0, snapshot={},
    )
    database.evaluate_prediction(pid2, 99.0)   # correct

    pid3 = database.record_prediction(
        symbol="TEST", start_price=100.0, direction="UPWARD BIAS",
        score=2, horizon_minutes=0, snapshot={},
    )
    database.evaluate_prediction(pid3, 99.0)   # incorrect

    pid4 = database.record_prediction(
        symbol="TEST", start_price=100.0, direction="NEUTRAL",
        score=0, horizon_minutes=0, snapshot={},
    )
    database.evaluate_prediction(pid4, 105.0)  # not counted

    stats = database.get_accuracy_stats()
    assert stats["total_predictions"] == 4
    assert stats["evaluated_predictions"] == 3
    assert stats["correct_predictions"] == 2
    assert stats["incorrect_predictions"] == 1
    assert stats["neutral_predictions"] == 1
    # 2 / 3 = 66.67%
    assert stats["directional_accuracy_pct"] == pytest.approx(66.67, abs=0.01)