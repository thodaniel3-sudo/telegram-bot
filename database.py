"""
database.py
-----------
SQLite storage for predictions.

The database lives at config.DATABASE_PATH (data/predictions.db).

Public API
----------
init_db()                                 create table if missing
record_prediction(...) -> int             insert forecast, return ID
get_pending_predictions() -> list[dict]   forecasts due but not evaluated
evaluate_prediction(id, end_price)        mark forecast as evaluated
get_recent_predictions(n) -> list[dict]   latest N forecasts
get_accuracy_stats() -> dict              aggregate performance metrics
subscribe(chat_id, username)              register chat for auto-forecasts
unsubscribe(chat_id) -> bool              stop auto-forecasts
is_subscribed(chat_id) -> bool            check subscription status
get_subscribers() -> list[int]            all subscribed chat IDs

All timestamps are stored as UTC ISO strings ("YYYY-MM-DDTHH:MM:SS+00:00").
This makes the database portable across time zones.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import datetime, timedelta, timezone
from typing import Any

import config


# ─────────────────────────────────────────────────────────────
# Schema
# ─────────────────────────────────────────────────────────────
_SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol             TEXT    NOT NULL,
    created_at         TEXT    NOT NULL,
    target_at          TEXT    NOT NULL,
    start_price        REAL    NOT NULL,
    direction          TEXT    NOT NULL,
    score              INTEGER NOT NULL,
    horizon_minutes    INTEGER NOT NULL,
    rsi_14             REAL,
    ema_9              REAL,
    ema_21             REAL,
    volatility         REAL,
    momentum_5         REAL,
    trend_score        REAL,
    model_version      TEXT    NOT NULL,
    end_price          REAL,
    end_direction      TEXT,
    is_correct         INTEGER,
    evaluated_at       TEXT
);

CREATE INDEX IF NOT EXISTS idx_predictions_target_at
    ON predictions (target_at);

CREATE INDEX IF NOT EXISTS idx_predictions_is_correct
    ON predictions (is_correct);

CREATE TABLE IF NOT EXISTS subscriptions (
    chat_id       INTEGER PRIMARY KEY,
    username      TEXT,
    subscribed_at TEXT NOT NULL
);
"""


# ─────────────────────────────────────────────────────────────
# Connection helper
# ─────────────────────────────────────────────────────────────
def _connect() -> sqlite3.Connection:
    """Open a connection with row access by column name."""
    conn = sqlite3.connect(config.DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create the tables and indexes if they do not already exist."""
    with closing(_connect()) as conn:
        conn.executescript(_SCHEMA)
        conn.commit()


# ─────────────────────────────────────────────────────────────
# Time helpers
# ─────────────────────────────────────────────────────────────
def _utc_now_iso() -> str:
    """Return the current UTC time as an ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _plus_minutes_iso(minutes: int) -> str:
    """Return UTC now + N minutes, ISO 8601."""
    return (
        datetime.now(timezone.utc) + timedelta(minutes=minutes)
    ).isoformat(timespec="seconds")


# ─────────────────────────────────────────────────────────────
# Insert
# ─────────────────────────────────────────────────────────────
def record_prediction(
    *,
    symbol: str,
    start_price: float,
    direction: str,
    score: int,
    horizon_minutes: int,
    snapshot: dict[str, Any],
    model_version: str = "rules_v1",
) -> int:
    """
    Insert a new forecast and return its row id.

    `snapshot` is the dict produced by predictor.Forecast.snapshot.
    Missing keys are stored as NULL.
    """
    created_at = _utc_now_iso()
    target_at = _plus_minutes_iso(horizon_minutes)

    with closing(_connect()) as conn:
        cur = conn.execute(
            """
            INSERT INTO predictions (
                symbol, created_at, target_at,
                start_price, direction, score, horizon_minutes,
                rsi_14, ema_9, ema_21, volatility, momentum_5, trend_score,
                model_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                symbol,
                created_at,
                target_at,
                float(start_price),
                direction,
                int(score),
                int(horizon_minutes),
                snapshot.get("rsi_14"),
                snapshot.get("ema_9"),
                snapshot.get("ema_21"),
                snapshot.get("volatility"),
                snapshot.get("momentum_5"),
                snapshot.get("trend_score"),
                model_version,
            ),
        )
        conn.commit()
        return int(cur.lastrowid)


# ─────────────────────────────────────────────────────────────
# Query
# ─────────────────────────────────────────────────────────────
def get_pending_predictions() -> list[dict[str, Any]]:
    """
    Return forecasts whose target_at time has passed but which have
    not yet been evaluated (is_correct IS NULL).
    """
    now = _utc_now_iso()
    with closing(_connect()) as conn:
        rows = conn.execute(
            """
            SELECT * FROM predictions
            WHERE is_correct IS NULL
              AND target_at <= ?
            ORDER BY target_at ASC
            """,
            (now,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_recent_predictions(limit: int = 10) -> list[dict[str, Any]]:
    """Return the most recent `limit` forecasts (newest first)."""
    with closing(_connect()) as conn:
        rows = conn.execute(
            """
            SELECT * FROM predictions
            ORDER BY id DESC
            LIMIT ?
            """,
            (int(limit),),
        ).fetchall()
    return [dict(r) for r in rows]


def get_prediction(prediction_id: int) -> dict[str, Any] | None:
    """Return a single prediction by ID, or None if not found."""
    with closing(_connect()) as conn:
        row = conn.execute(
            "SELECT * FROM predictions WHERE id = ?", (prediction_id,)
        ).fetchone()
    return dict(row) if row else None


# ─────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────
def evaluate_prediction(
    prediction_id: int,
    end_price: float,
) -> dict[str, Any]:
    """
    Mark a forecast as evaluated.

    Rules
    -----
    - If direction was NEUTRAL, is_correct stays NULL (never counted).
    - Otherwise, compute the percentage move from start_price to end_price.
      * If |move| < config.MIN_MOVE_PCT  -> end_direction = "FLAT"
        and is_correct = 0 (directional forecast failed).
      * If move > 0                      -> end_direction = "UP"
      * If move < 0                      -> end_direction = "DOWN"

      A directional forecast (UPWARD BIAS or DOWNWARD BIAS) is correct
      if its direction matches end_direction. Otherwise is_correct = 0.
    """
    pred = get_prediction(prediction_id)
    if pred is None:
        raise ValueError(f"No prediction with id {prediction_id}")

    if pred["is_correct"] is not None:
        # Already evaluated — do nothing.
        return pred

    start_price = float(pred["start_price"])
    end_price = float(end_price)
    direction = pred["direction"]

    if start_price <= 0:
        raise ValueError(f"Invalid start_price in prediction {prediction_id}")

    move_pct = (end_price - start_price) / start_price * 100.0

    if abs(move_pct) < config.MIN_MOVE_PCT:
        end_direction = "FLAT"
    elif move_pct > 0:
        end_direction = "UP"
    else:
        end_direction = "DOWN"

    # NEUTRAL forecasts are not judged as right or wrong.
    if direction == "NEUTRAL":
        is_correct = None
    elif direction == "UPWARD BIAS":
        is_correct = 1 if end_direction == "UP" else 0
    elif direction == "DOWNWARD BIAS":
        is_correct = 1 if end_direction == "DOWN" else 0
    else:
        raise ValueError(f"Unknown direction stored in DB: {direction!r}")

    evaluated_at = _utc_now_iso()

    with closing(_connect()) as conn:
        conn.execute(
            """
            UPDATE predictions
               SET end_price = ?,
                   end_direction = ?,
                   is_correct = ?,
                   evaluated_at = ?
             WHERE id = ?
            """,
            (
                end_price,
                end_direction,
                is_correct,
                evaluated_at,
                prediction_id,
            ),
        )
        conn.commit()

    return get_prediction(prediction_id)  # type: ignore[return-value]


# ─────────────────────────────────────────────────────────────
# Aggregate statistics
# ─────────────────────────────────────────────────────────────
def get_accuracy_stats() -> dict[str, Any]:
    """
    Return a dict with the current performance metrics.

    All fields are counts or derived percentages. Nothing is fabricated.
    """
    with closing(_connect()) as conn:
        total = conn.execute(
            "SELECT COUNT(*) FROM predictions"
        ).fetchone()[0]

        evaluated = conn.execute(
            "SELECT COUNT(*) FROM predictions WHERE is_correct IS NOT NULL"
        ).fetchone()[0]

        correct = conn.execute(
            "SELECT COUNT(*) FROM predictions WHERE is_correct = 1"
        ).fetchone()[0]

        incorrect = conn.execute(
            "SELECT COUNT(*) FROM predictions WHERE is_correct = 0"
        ).fetchone()[0]

        neutral = conn.execute(
            "SELECT COUNT(*) FROM predictions WHERE direction = 'NEUTRAL'"
        ).fetchone()[0]

        by_direction_rows = conn.execute(
            """
            SELECT direction,
                   COUNT(*)                                        AS total,
                   SUM(CASE WHEN is_correct = 1 THEN 1 ELSE 0 END) AS correct
            FROM predictions
            WHERE is_correct IS NOT NULL
            GROUP BY direction
            """
        ).fetchall()

    by_direction = {
        row["direction"]: {
            "evaluated": int(row["total"]),
            "correct": int(row["correct"]),
            "accuracy_pct": (
                round(100.0 * row["correct"] / row["total"], 2)
                if row["total"] > 0
                else None
            ),
        }
        for row in by_direction_rows
    }

    directional_accuracy_pct = (
        round(100.0 * correct / evaluated, 2) if evaluated > 0 else None
    )
    neutral_rate_pct = (
        round(100.0 * neutral / total, 2) if total > 0 else None
    )

    return {
        "total_predictions": int(total),
        "evaluated_predictions": int(evaluated),
        "correct_predictions": int(correct),
        "incorrect_predictions": int(incorrect),
        "neutral_predictions": int(neutral),
        "directional_accuracy_pct": directional_accuracy_pct,
        "neutral_rate_pct": neutral_rate_pct,
        "by_direction": by_direction,
    }


# ─────────────────────────────────────────────────────────────
# Subscriptions (for automatic forecasts)
# ─────────────────────────────────────────────────────────────
def subscribe(chat_id: int, username: str | None = None) -> None:
    """Register a chat to receive automatic forecasts. Idempotent."""
    with closing(_connect()) as conn:
        conn.execute(
            """
            INSERT INTO subscriptions (chat_id, username, subscribed_at)
            VALUES (?, ?, ?)
            ON CONFLICT(chat_id) DO NOTHING
            """,
            (int(chat_id), username or "", _utc_now_iso()),
        )
        conn.commit()


def unsubscribe(chat_id: int) -> bool:
    """Remove a chat from automatic forecasts. Returns True if it existed."""
    with closing(_connect()) as conn:
        cur = conn.execute(
            "DELETE FROM subscriptions WHERE chat_id = ?", (int(chat_id),)
        )
        conn.commit()
        return cur.rowcount > 0


def is_subscribed(chat_id: int) -> bool:
    """Return True if the chat is subscribed."""
    with closing(_connect()) as conn:
        row = conn.execute(
            "SELECT 1 FROM subscriptions WHERE chat_id = ?", (int(chat_id),)
        ).fetchone()
    return row is not None


def get_subscribers() -> list[int]:
    """Return all subscribed chat IDs."""
    with closing(_connect()) as conn:
        rows = conn.execute("SELECT chat_id FROM subscriptions").fetchall()
    return [int(r["chat_id"]) for r in rows]


# ─────────────────────────────────────────────────────────────
# Quick self-test
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 70)
    print("database.py self-test")
    print("=" * 70)

    init_db()
    print(f"\nDatabase at: {config.DATABASE_PATH}")
    print("Tables 'predictions' and 'subscriptions' created (if missing).")

    # Insert a fake test prediction — clearly marked so we can remove it.
    test_snapshot = {
        "rsi_14": 50.0,
        "ema_9": 100.0,
        "ema_21": 100.1,
        "volatility": 0.1,
        "momentum_5": 0.02,
        "trend_score": 0.6,
    }
    pid = record_prediction(
        symbol="TEST_SYMBOL",
        start_price=100.0,
        direction="UPWARD BIAS",
        score=2,
        horizon_minutes=0,          # evaluate immediately
        snapshot=test_snapshot,
        model_version="selftest_v1",
    )
    print(f"\nInserted test prediction id={pid}")

    pending = get_pending_predictions()
    print(f"Pending predictions : {len(pending)}")

    # Evaluate as if the price went up 0.05%.
    evaluated = evaluate_prediction(pid, end_price=100.05)
    print(
        f"Evaluated id={pid} -> "
        f"end_direction={evaluated['end_direction']} "
        f"is_correct={evaluated['is_correct']}"
    )

    stats = get_accuracy_stats()
    print("\nAccuracy stats:")
    for key, value in stats.items():
        if key != "by_direction":
            print(f"  {key:<28}: {value}")

    print("\nRecent predictions:")
    for row in get_recent_predictions(3):
        print(
            f"  id={row['id']:<3} "
            f"symbol={row['symbol']:<12} "
            f"direction={row['direction']:<14} "
            f"is_correct={row['is_correct']}"
        )

    # Subscriptions test
    print("\nSubscriptions test:")
    subscribe(chat_id=999999999, username="selftest_user")
    print(f"  is_subscribed(999999999): {is_subscribed(999999999)}")
    print(f"  get_subscribers()       : {get_subscribers()}")
    print(f"  unsubscribe(999999999)  : {unsubscribe(999999999)}")
    print(f"  is_subscribed after unsub: {is_subscribed(999999999)}")

    # Clean up the test row so it does not pollute real statistics.
    with closing(_connect()) as conn:
        conn.execute("DELETE FROM predictions WHERE model_version = 'selftest_v1'")
        conn.execute("DELETE FROM subscriptions WHERE chat_id = 999999999")
        conn.commit()

    print("\nTest rows removed.")
    print("All checks passed.")