"""
app.py
------
Render deployment entry point.

Runs the Telegram bot in webhook mode (required for Render free tier)
and provides a /health endpoint for Render's health checks.

Local development uses `python main.py` (polling mode) instead.
"""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager

import httpx
import uvicorn
from fastapi import FastAPI, Request, Response
from telegram import Update
from telegram.ext import Application

import config
import database
import main as bot_main


logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
log = logging.getLogger("v100-app")

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("telegram").setLevel(logging.WARNING)


bot_app: Application | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global bot_app

    log.info("Starting V100 Analyst in WEBHOOK mode")
    database.init_db()

    bot_app = (
        Application.builder()
        .token(config.TELEGRAM_BOT_TOKEN)
        .build()
    )

    for _name, handler in bot_main.get_handlers().items():
        bot_app.add_handler(handler)

    await bot_app.initialize()

    render_url = os.getenv("RENDER_EXTERNAL_URL", "").rstrip("/")
    if not render_url:
        log.error("RENDER_EXTERNAL_URL is not set.")
        raise RuntimeError("RENDER_EXTERNAL_URL is required.")

    webhook_url = f"{render_url}/webhook"
    await bot_app.bot.set_webhook(
        url=webhook_url,
        allowed_updates=Update.ALL_TYPES,
    )
    log.info("Webhook registered at %s", webhook_url)

    # Background tasks
    asyncio.create_task(_periodic_evaluation())
    asyncio.create_task(_periodic_forecast())
    asyncio.create_task(_self_ping(render_url))

    yield

    log.info("Shutting down")
    try:
        await bot_app.bot.delete_webhook()
    except Exception:
        pass
    await bot_app.shutdown()


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def root() -> dict:
    return {
        "service": "v100-analyst-bot",
        "status": "running",
        "endpoints": ["/health", "/webhook"],
        "note": "This is a Telegram bot. Use @v100_analyst_bot to interact.",
    }


@app.api_route("/health", methods=["GET", "HEAD"])
async def health() -> dict:
    return {"status": "ok"}


@app.post("/webhook")
async def webhook(request: Request) -> Response:
    if bot_app is None:
        return Response(status_code=503)
    data = await request.json()
    update = Update.de_json(data, bot_app.bot)
    await bot_app.process_update(update)
    return Response(status_code=200)


async def _periodic_evaluation() -> None:
    while True:
        try:
            await asyncio.sleep(60)
            await bot_main.periodic_evaluation(None)  # type: ignore[arg-type]
        except Exception:
            log.exception("Periodic evaluation failed")


async def _periodic_forecast() -> None:
    interval = config.FORECAST_HORIZON_MINUTES * 60
    while True:
        try:
            await asyncio.sleep(interval)
            if bot_app is not None:
                await bot_main.periodic_forecast(
                    type("Ctx", (), {"bot": bot_app.bot})  # minimal context
                )
        except Exception:
            log.exception("Periodic forecast failed")


async def _self_ping(base_url: str) -> None:
    await asyncio.sleep(30)
    while True:
        try:
            async with httpx.AsyncClient() as client:
                await client.get(f"{base_url}/health", timeout=10)
        except Exception as exc:
            log.warning("Self-ping failed: %s", exc)
        await asyncio.sleep(240)


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")