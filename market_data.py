"""
market_data.py
--------------
Fetch market data from Deriv's public WebSocket API.

Public functions
----------------
fetch_candles(symbol, count, granularity) -> pandas.DataFrame
    Download the most recent completed candles for a given symbol.

fetch_latest_price(symbol) -> float
    Return the current live spot price.

verify_symbol(symbol) -> bool
    Return True if the symbol is currently active on Deriv.

Design notes
------------
- All functions are synchronous from the caller's point of view.
  Internally they use asyncio to talk to the WebSocket.
- No authentication is required: this endpoint serves public data.
- Every returned candle has already been completed by Deriv
  (no partial/in-progress candles), so indicators are safe to compute.
"""

from __future__ import annotations

import asyncio
import json
from typing import Optional

import pandas as pd
import websockets

import config


# ─────────────────────────────────────────────────────────────
# Low-level helpers (private to this module)
# ─────────────────────────────────────────────────────────────
_TIMEOUT_SECONDS = 15
_MAX_RETRIES = 3


async def _send_and_receive(request: dict) -> dict:
    """
    Open a WebSocket to Deriv, send one JSON request, receive one
    JSON response, close the connection, return the parsed response.

    Raises
    ------
    RuntimeError
        If the connection fails, times out, or Deriv returns an error.
    """
    last_exception: Optional[Exception] = None

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            async with websockets.connect(
                config.DERIV_WS_URL,
                open_timeout=_TIMEOUT_SECONDS,
                close_timeout=_TIMEOUT_SECONDS,
            ) as ws:
                await ws.send(json.dumps(request))

                # Wait for the first message (Deriv replies immediately).
                raw = await asyncio.wait_for(
                    ws.recv(), timeout=_TIMEOUT_SECONDS
                )

            response = json.loads(raw)

            if "error" in response:
                message = response["error"].get("message", "unknown error")
                raise RuntimeError(f"Deriv API error: {message}")

            return response

        except Exception as exc:  # noqa: BLE001 -- we want to retry on any error
            last_exception = exc
            if attempt < _MAX_RETRIES:
                # Brief pause before retrying.
                await asyncio.sleep(1.0 * attempt)

    raise RuntimeError(
        f"Failed to reach Deriv API after {_MAX_RETRIES} attempts: "
        f"{last_exception}"
    )


def _run(coro):
    """Run an async coroutine from synchronous code."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No running loop — normal case for standalone scripts.
        return asyncio.run(coro)

    # If we are already inside a loop (e.g. inside the Telegram bot),
    # we need to run the coroutine in a separate thread so asyncio
    # does not complain about nested loops.
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


# ─────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────
def fetch_candles(
    symbol: str | None = None,
    count: int = 200,
    granularity: int | None = None,
) -> pd.DataFrame:
    """
    Fetch the most recent completed candles.

    Parameters
    ----------
    symbol : str, optional
        Market symbol. Defaults to config.MARKET_SYMBOL.
    count : int, optional
        Number of candles to fetch. Maximum is 5000.
    granularity : int, optional
        Candle duration in seconds (60, 120, 300, 600, 900, 1800,
        3600, 7200, 14400, 28800, 86400). Defaults to
        config.CANDLE_GRANULARITY.

    Returns
    -------
    pandas.DataFrame
        Columns: epoch (int, UTC seconds), datetime (UTC),
        open, high, low, close (float).
        The most recent row is the latest completed candle.
    """
    symbol = symbol or config.MARKET_SYMBOL
    granularity = granularity or config.CANDLE_GRANULARITY

    if not 1 <= count <= 5000:
        raise ValueError(f"count must be 1..5000, got {count}")

    request = {
        "ticks_history": symbol,
        "adjust_start_time": 1,
        "count": count,
        "end": "latest",
        "start": 1,
        "style": "candles",
        "granularity": granularity,
    }

    response = _run(_send_and_receive(request))

    candles = response.get("candles")
    if not candles:
        raise RuntimeError(
            f"Deriv returned no candles for symbol '{symbol}'. "
            "The symbol may be invalid or inactive."
        )

    df = pd.DataFrame(candles)

    # Convert types: prices are strings in the JSON, epochs are int.
    for col in ("open", "high", "low", "close"):
        df[col] = df[col].astype(float)

    df["epoch"] = df["epoch"].astype(int)
    df["datetime"] = pd.to_datetime(df["epoch"], unit="s", utc=True)

    # Reorder columns for readability.
    df = df[["epoch", "datetime", "open", "high", "low", "close"]]

    return df


def fetch_latest_price(symbol: str | None = None) -> float:
    """
    Return the current live spot price for the given symbol.
    """
    symbol = symbol or config.MARKET_SYMBOL

    request = {"ticks": symbol}

    response = _run(_send_and_receive(request))

    tick = response.get("tick")
    if not tick or "quote" not in tick:
        raise RuntimeError(
            f"Deriv did not return a live price for symbol '{symbol}'."
        )

    return float(tick["quote"])


def verify_symbol(symbol: str | None = None) -> bool:
    """
    Return True if the symbol is currently active on Deriv.
    """
    symbol = symbol or config.MARKET_SYMBOL

    request = {"active_symbols": "brief"}

    try:
        response = _run(_send_and_receive(request))
    except RuntimeError:
        return False

    symbols = response.get("active_symbols", [])
    for entry in symbols:
        if (
            entry.get("underlying_symbol") == symbol
            or entry.get("symbol") == symbol
        ):
            return True
    return False


# ─────────────────────────────────────────────────────────────
# Quick self-test
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("market_data.py self-test")
    print("=" * 60)

    print(f"\nSymbol in config : {config.MARKET_SYMBOL}")

    print("\n[1] Verifying symbol is active on Deriv...")
    ok = verify_symbol()
    print(f"    Active? {ok}")

    if not ok:
        print("    Stopping: symbol is not active.")
        raise SystemExit(1)

    print(f"\n[2] Fetching the last 5 candles (granularity "
          f"{config.CANDLE_GRANULARITY}s)...")
    df = fetch_candles(count=5)
    print(f"    Received {len(df)} candles.\n")
    print(df.to_string(index=False))

    print("\n[3] Fetching current live price...")
    price = fetch_latest_price()
    print(f"    Live price: {price:.4f}")

    print("\nAll checks passed.")