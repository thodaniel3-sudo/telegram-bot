"""
config.py
---------
Central configuration for the V100 Telegram Bot.

Loads secrets and settings from the .env file, validates them, and
exposes them as simple Python objects that the rest of the project
can import.

Every other file in this project should read settings like this:

    import config
    token = config.TELEGRAM_BOT_TOKEN
    symbol = config.MARKET_SYMBOL

Do not read environment variables directly anywhere else.
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv


# ─────────────────────────────────────────────────────────────
# Locate the project root and load .env
# ─────────────────────────────────────────────────────────────
# BASE_DIR is the folder that contains this file (the project root).
BASE_DIR = Path(__file__).resolve().parent

# Path to the .env file.
ENV_FILE = BASE_DIR / ".env"

# load_dotenv reads key=value lines from the file and puts them
# into os.environ. If the file is missing, it does nothing silently.
# We check for that below and raise a clear error.
load_dotenv(dotenv_path=ENV_FILE)


# ─────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────
def _get_required(name: str) -> str:
    """
    Return the value of a required environment variable.
    Exit with a clear message if it is missing or empty.
    """
    value = os.getenv(name)
    if value is None or value.strip() == "":
        print(f"\n[CONFIG ERROR] Required setting '{name}' is missing from .env")
        print(f"Open the file: {ENV_FILE}")
        print(f"and add a line like:  {name}=your_value_here\n")
        sys.exit(1)
    return value.strip()


def _get_optional(name: str, default: str) -> str:
    """Return an optional environment variable, or a default value."""
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return value.strip()


def _get_int(name: str, default: int) -> int:
    """Read an integer setting. Fall back to default if missing or invalid."""
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw.strip())
    except ValueError:
        print(f"[CONFIG WARNING] Setting '{name}' is not a valid integer: {raw!r}")
        print(f"                 Using default: {default}")
        return default


def _get_float(name: str, default: float) -> float:
    """Read a float setting. Fall back to default if missing or invalid."""
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw.strip())
    except ValueError:
        print(f"[CONFIG WARNING] Setting '{name}' is not a valid number: {raw!r}")
        print(f"                 Using default: {default}")
        return default


# ─────────────────────────────────────────────────────────────
# Required settings (bot will not start without these)
# ─────────────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN: str = _get_required("TELEGRAM_BOT_TOKEN")


# ─────────────────────────────────────────────────────────────
# Optional settings (sensible defaults apply)
# ─────────────────────────────────────────────────────────────
# Empty string means "no restriction". Only set this if you want
# the bot to respond to a single Telegram account.
_allowed_raw = _get_optional("TELEGRAM_ALLOWED_USER_ID", "")
TELEGRAM_ALLOWED_USER_ID: int | None = (
    int(_allowed_raw) if _allowed_raw.isdigit() else None
)

# The Deriv market symbol. 1HZ100V = Volatility 100 (1s) Index.
MARKET_SYMBOL: str = _get_optional("MARKET_SYMBOL", "1HZ100V")

# Candle granularity in seconds. 60 = one-minute candles.
CANDLE_GRANULARITY: int = _get_int("CANDLE_GRANULARITY", 60)

# Forecast horizon in minutes.
FORECAST_HORIZON_MINUTES: int = _get_int("FORECAST_HORIZON_MINUTES", 15)

# Minimum absolute move (in percent) required to count as directional.
MIN_MOVE_PCT: float = _get_float("MIN_MOVE_PCT", 0.01)

# Logging level: DEBUG, INFO, WARNING, ERROR
LOG_LEVEL: str = _get_optional("LOG_LEVEL", "INFO").upper()


# ─────────────────────────────────────────────────────────────
# Derived / fixed paths used across the project
# ─────────────────────────────────────────────────────────────
DATA_DIR: Path = BASE_DIR / "data"
LOGS_DIR: Path = BASE_DIR / "logs"
MODELS_DIR: Path = BASE_DIR / "models"

DATABASE_PATH: Path = DATA_DIR / "predictions.db"
LOG_FILE_PATH: Path = LOGS_DIR / "bot.log"

# Deriv public WebSocket endpoint (no authentication required
# for public market data).
DERIV_WS_URL: str = "wss://api.derivws.com/trading/v1/options/ws/public"


# ─────────────────────────────────────────────────────────────
# Ensure required folders exist
# ─────────────────────────────────────────────────────────────
for folder in (DATA_DIR, LOGS_DIR, MODELS_DIR):
    folder.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────
# Friendly summary (safe to print — contains no secret)
# ─────────────────────────────────────────────────────────────
def summary() -> str:
    """Return a short human-readable description of the configuration."""
    token_masked = (
        TELEGRAM_BOT_TOKEN[:8] + "..." + TELEGRAM_BOT_TOKEN[-4:]
        if len(TELEGRAM_BOT_TOKEN) > 12
        else "<invalid>"
    )
    return (
        "Configuration loaded:\n"
        f"  Token              : {token_masked}\n"
        f"  Allowed user ID    : {TELEGRAM_ALLOWED_USER_ID or 'anyone'}\n"
        f"  Symbol             : {MARKET_SYMBOL}\n"
        f"  Candle granularity : {CANDLE_GRANULARITY} s\n"
        f"  Forecast horizon   : {FORECAST_HORIZON_MINUTES} min\n"
        f"  Minimum move       : {MIN_MOVE_PCT} %\n"
        f"  Log level          : {LOG_LEVEL}\n"
        f"  Database path      : {DATABASE_PATH}\n"
        f"  Log file path      : {LOG_FILE_PATH}\n"
    )


if __name__ == "__main__":
    # Running "python config.py" prints the current configuration.
    # This is useful for testing that the .env file is loaded correctly.
    print(summary())