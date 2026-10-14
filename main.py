"""
main.py
-------
The V100 Telegram bot.

Run in polling mode (local development):
    python main.py

Webhook mode (Railway deployment) is handled by app.py.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
)

import config
import database
import indicators
import market_data
import predictor

# West Africa Time (UTC+1, no daylight saving).
WAT = timezone(timedelta(hours=1))


# ─────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler(config.LOG_FILE_PATH, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("v100-bot")

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("telegram").setLevel(logging.WARNING)


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def _is_authorized(update: Update) -> bool:
    if config.TELEGRAM_ALLOWED_USER_ID is None:
        return True
    user = update.effective_user
    return user is not None and user.id == config.TELEGRAM_ALLOWED_USER_ID


def _now_wat() -> datetime:
    return datetime.now(WAT).replace(microsecond=0)


def _fmt_both(dt: datetime) -> str:
    """Show WAT first, then UTC in parentheses."""
    dt_wat = dt.astimezone(WAT)
    dt_utc = dt.astimezone(timezone.utc)
    return f"{dt_wat.strftime('%H:%M')} WAT  ({dt_utc.strftime('%H:%M')} UTC)"


async def _require_authorization(update: Update) -> bool:
    if _is_authorized(update):
        return True
    await update.message.reply_text("This bot is restricted to its owner.")
    return False


def _confidence_label(score: int) -> str:
    magnitude = abs(score)
    if magnitude >= 4:
        return "Strong"
    if magnitude >= 3:
        return "Moderate"
    if magnitude >= 2:
        return "Weak"
    return "None"


def _build_forecast_text(forecast, last_close: float, prediction_id: int) -> str:
    """Build the standard 'Programmed for Analysis' message."""
    stats = database.get_accuracy_stats()

    now_wat = _now_wat()
    target_wat = now_wat + timedelta(minutes=config.FORECAST_HORIZON_MINUTES)

    direction_display = forecast.direction.replace(" BIAS", "")
    confidence = _confidence_label(forecast.score)

    if stats["evaluated_predictions"] == 0:
        accuracy_line = "No evaluated forecasts yet"
        sample_line = "(sample = 0)"
    else:
        acc = stats["directional_accuracy_pct"]
        accuracy_line = f"{acc}%"
        n = stats["evaluated_predictions"]
        if n < 30:
            sample_line = f"(n = {n} evaluated — small sample)"
        else:
            sample_line = f"(n = {n} evaluated forecasts)"

    neutral_line = (
        f"{stats['neutral_rate_pct']}%"
        if stats["neutral_rate_pct"] is not None
        else "Not available"
    )

    return (
        "*Programmed for Analysis.*\n"
        f"*Predict Time:* {config.FORECAST_HORIZON_MINUTES} minutes\n"
        f"*Time In:*  {_fmt_both(now_wat)}\n"
        f"*Time Out:* {_fmt_both(target_wat)}\n\n"
        f"*Direction:* {direction_display}\n"
        "*Signal:* Rule-based (EMA + RSI + Momentum)\n"
        f"*Confidence:* {confidence}  (rule agreement {forecast.score:+d})\n"
        f"*Price at analysis:* `{last_close:.4f}`\n"
        f"*Prediction ID:* `{prediction_id}`\n\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "*MEASURED HISTORICAL ACCURACY*\n"
        f"Directional: *{accuracy_line}*\n"
        f"{sample_line}\n"
        f"Neutral rate: {neutral_line}\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        "*Not investment advice.*\n"
        "No trade should be placed based on this message. "
        "The accuracy figure above is measured on historical data "
        "and does not guarantee future results."
    )


def _run_analysis_and_record() -> tuple[str, int]:
    """Fetch data, compute indicators, predict, and record. Returns (text, id)."""
    raw = market_data.fetch_candles(count=200)
    enriched = indicators.add_indicators(raw)
    forecast = predictor.predict(enriched)
    last = enriched.iloc[-1]

    prediction_id = database.record_prediction(
        symbol=config.MARKET_SYMBOL,
        start_price=float(last["close"]),
        direction=forecast.direction,
        score=forecast.score,
        horizon_minutes=config.FORECAST_HORIZON_MINUTES,
        snapshot=forecast.snapshot,
        model_version="rules_v1",
    )

    text = _build_forecast_text(forecast, float(last["close"]), prediction_id)
    return text, prediction_id


# ─────────────────────────────────────────────────────────────
# Command handlers
# ─────────────────────────────────────────────────────────────
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _require_authorization(update):
        return

    text = (
        "*V100 Analyst* — Volatility 100 (1s) analysis\n\n"
        "This bot produces a 15-minute directional analysis of the "
        "Deriv Volatility 100 (1s) index, using a transparent rule-based "
        "engine (EMA + RSI + momentum + trend).\n\n"
        "Use /analyze to run a fresh forecast.\n"
        "Use /subscribe to receive forecasts automatically every 15 minutes.\n"
        "Use /help to see all commands.\n\n"
        "_The bot reports its real measured accuracy on every forecast. "
        "It does not place trades. It is not investment advice._"
    )
    await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _require_authorization(update):
        return

    text = (
        "*Commands*\n\n"
        "/analyze — Fresh 15-minute directional analysis\n"
        "/subscribe — Receive forecasts automatically every 15 minutes\n"
        "/unsubscribe — Stop receiving automatic forecasts\n"
        "/status — Bot and market-data status\n"
        "/accuracy — Measured prediction performance\n"
        "/history — Recent forecasts and outcomes\n"
        "/settings — Current configuration values\n"
        "/evaluate — Manually evaluate due forecasts\n"
        "/help — This message"
    )
    await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)


async def cmd_analyze(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _require_authorization(update):
        return

    msg = await update.message.reply_text("Fetching market data…")
    try:
        text, _ = _run_analysis_and_record()
        await msg.edit_text(text, parse_mode=ParseMode.MARKDOWN)
    except Exception as exc:  # noqa: BLE001
        log.exception("Error in /analyze")
        await msg.edit_text(
            f"*Analysis failed.*\n\nReason: `{type(exc).__name__}: {exc}`"
        )


async def cmd_subscribe(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _require_authorization(update):
        return

    chat = update.effective_chat
    user = update.effective_user
    if chat is None:
        return

    database.subscribe(chat.id, user.username if user else None)

    text = (
        "*Subscribed.*\n\n"
        "You will receive an automatic forecast every "
        f"*{config.FORECAST_HORIZON_MINUTES} minutes*.\n\n"
        "The forecasts use the same honest format as `/analyze`, "
        "including the real measured accuracy.\n\n"
        "Use `/unsubscribe` at any time to stop."
    )
    await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)


async def cmd_unsubscribe(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _require_authorization(update):
        return

    chat = update.effective_chat
    if chat is None:
        return

    removed = database.unsubscribe(chat.id)
    if removed:
        text = "*Unsubscribed.* You will no longer receive automatic forecasts."
    else:
        text = "You were not subscribed."
    await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _require_authorization(update):
        return

    msg = await update.message.reply_text("Checking status…")

    try:
        ok = market_data.verify_symbol()
        symbol_status = "OK" if ok else "Symbol not active"
    except Exception as exc:  # noqa: BLE001
        symbol_status = f"ERROR: {type(exc).__name__}"

    stats = database.get_accuracy_stats()
    subscribers = database.get_subscribers()

    text = (
        "*BOT STATUS*\n\n"
        f"Symbol: `{config.MARKET_SYMBOL}`\n"
        f"Symbol status: `{symbol_status}`\n"
        f"Candle size: `{config.CANDLE_GRANULARITY}s`\n"
        f"Forecast horizon: `{config.FORECAST_HORIZON_MINUTES} minutes`\n"
        f"Total predictions: `{stats['total_predictions']}`\n"
        f"Evaluated: `{stats['evaluated_predictions']}`\n"
        f"Auto-subscribers: `{len(subscribers)}`\n"
        f"Check time: `{_fmt_both(_now_wat())}`"
    )
    await msg.edit_text(text, parse_mode=ParseMode.MARKDOWN)


async def cmd_accuracy(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _require_authorization(update):
        return

    stats = database.get_accuracy_stats()

    if stats["evaluated_predictions"] == 0:
        await update.message.reply_text(
            "*ACCURACY REPORT*\n\nNo evaluated forecasts yet.",
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    n = stats["evaluated_predictions"]
    if n < 30:
        reliability = "⚠ Sample too small to be meaningful."
    elif n < 100:
        reliability = "⚠ Modest sample — treat with caution."
    else:
        reliability = "Fair sample size."

    text = (
        "*ACCURACY REPORT*\n\n"
        f"Total forecasts: `{stats['total_predictions']}`\n"
        f"Evaluated: `{stats['evaluated_predictions']}`\n"
        f"Correct: `{stats['correct_predictions']}`\n"
        f"Incorrect: `{stats['incorrect_predictions']}`\n"
        f"NEUTRAL (excluded): `{stats['neutral_predictions']}`\n\n"
        f"*Directional accuracy:* `{stats['directional_accuracy_pct']}%`\n"
        f"*NEUTRAL rate:* `{stats['neutral_rate_pct']}%`\n\n"
        f"{reliability}"
    )
    await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)


async def cmd_history(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _require_authorization(update):
        return

    rows = database.get_recent_predictions(5)
    if not rows:
        await update.message.reply_text("No predictions yet.")
        return

    lines = ["*RECENT FORECASTS*", ""]
    for row in rows:
        try:
            created_dt = datetime.fromisoformat(row["created_at"])
            if created_dt.tzinfo is None:
                created_dt = created_dt.replace(tzinfo=timezone.utc)
            created_display = _fmt_both(created_dt)
        except Exception:
            created_display = row["created_at"][:16].replace("T", " ")

        if row["is_correct"] is None:
            outcome = "pending"
        elif row["is_correct"] == 1:
            outcome = "correct"
        else:
            outcome = "incorrect"

        lines.append(
            f"`#{row['id']}` {created_display} | *{row['direction']}* | {outcome}"
        )

    await update.message.reply_text(
        "\n".join(lines), parse_mode=ParseMode.MARKDOWN
    )


async def cmd_settings(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _require_authorization(update):
        return

    text = (
        "*CURRENT SETTINGS*\n\n"
        f"Symbol: `{config.MARKET_SYMBOL}`\n"
        f"Candle granularity: `{config.CANDLE_GRANULARITY}s`\n"
        f"Forecast horizon: `{config.FORECAST_HORIZON_MINUTES} minutes`\n"
        f"Minimum move for evaluation: `{config.MIN_MOVE_PCT}%`\n"
        f"Display timezone: `WAT and UTC side-by-side`"
    )
    await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)


async def cmd_evaluate(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _require_authorization(update):
        return

    pending = database.get_pending_predictions()
    if not pending:
        await update.message.reply_text(
            "No forecasts due for evaluation yet."
        )
        return

    msg = await update.message.reply_text(
        f"Evaluating {len(pending)} forecast(s)…"
    )

    evaluated = 0
    details = []
    for row in pending:
        try:
            end_price = market_data.fetch_latest_price(symbol=row["symbol"])
            result = database.evaluate_prediction(row["id"], end_price)
            evaluated += 1
            outcome = (
                "correct" if result["is_correct"] == 1
                else "incorrect" if result["is_correct"] == 0
                else "not counted"
            )
            details.append(f"`#{row['id']}` {outcome}")
        except Exception:  # noqa: BLE001
            details.append(f"`#{row['id']}` FAILED")

    await msg.edit_text(
        f"*EVALUATION REPORT*\n\nEvaluated: `{evaluated}`\n\n"
        + "\n".join(details),
        parse_mode=ParseMode.MARKDOWN,
    )


# ─────────────────────────────────────────────────────────────
# Background tasks
# ─────────────────────────────────────────────────────────────
async def periodic_evaluation(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Auto-evaluate due predictions. Runs every minute."""
    try:
        pending = database.get_pending_predictions()
        for row in pending:
            try:
                end_price = market_data.fetch_latest_price(symbol=row["symbol"])
                database.evaluate_prediction(row["id"], end_price)
                log.info("Auto-evaluated prediction #%s", row["id"])
            except Exception as exc:  # noqa: BLE001
                log.warning("Auto-evaluation of #%s failed: %s", row["id"], exc)
    except Exception:  # noqa: BLE001
        log.exception("Periodic evaluation task failed")


async def periodic_forecast(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send automatic forecasts to all subscribers. Runs every 15 minutes."""
    try:
        subscribers = database.get_subscribers()
        if not subscribers:
            return

        text, prediction_id = _run_analysis_and_record()
        log.info("Sending auto-forecast #%s to %d subscriber(s)",
                 prediction_id, len(subscribers))

        for chat_id in subscribers:
            try:
                await context.bot.send_message(
                    chat_id=chat_id,
                    text=text,
                    parse_mode=ParseMode.MARKDOWN,
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("Failed to send to chat %s: %s", chat_id, exc)
    except Exception:  # noqa: BLE001
        log.exception("Periodic forecast task failed")


# ─────────────────────────────────────────────────────────────
# Shared handlers (used by both polling and webhook modes)
# ─────────────────────────────────────────────────────────────
def get_handlers() -> dict[str, CommandHandler]:
    """Return the command handlers. Used by main.py and app.py."""
    return {
        "start": CommandHandler("start", cmd_start),
        "help": CommandHandler("help", cmd_help),
        "analyze": CommandHandler("analyze", cmd_analyze),
        "subscribe": CommandHandler("subscribe", cmd_subscribe),
        "unsubscribe": CommandHandler("unsubscribe", cmd_unsubscribe),
        "status": CommandHandler("status", cmd_status),
        "accuracy": CommandHandler("accuracy", cmd_accuracy),
        "history": CommandHandler("history", cmd_history),
        "settings": CommandHandler("settings", cmd_settings),
        "evaluate": CommandHandler("evaluate", cmd_evaluate),
    }


# ─────────────────────────────────────────────────────────────
# Startup (polling mode, for local use)
# ─────────────────────────────────────────────────────────────
def main() -> None:
    log.info("Starting V100 Analyst bot (polling mode)")
    log.info("Configuration:\n%s", config.summary())

    database.init_db()

    application = (
        Application.builder()
        .token(config.TELEGRAM_BOT_TOKEN)
        .build()
    )

    for name, handler in get_handlers().items():
        application.add_handler(handler)

    if application.job_queue is not None:
        application.job_queue.run_repeating(
            periodic_evaluation, interval=60, first=30
        )
        application.job_queue.run_repeating(
            periodic_forecast,
            interval=config.FORECAST_HORIZON_MINUTES * 60,
            first=config.FORECAST_HORIZON_MINUTES * 60,
        )
    else:
        log.warning("JobQueue not available.")

    log.info("Bot is now polling Telegram. Press Ctrl+C to stop.")
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log.info("Bot stopped by user.")