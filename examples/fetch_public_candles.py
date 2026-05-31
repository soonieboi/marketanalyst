#!/usr/bin/env python3
"""
Fetch recent public OHLCV candles and save them in Kronos-compatible CSV files.

This uses Yahoo Finance's public chart endpoint. It is suitable for quick
experiments, but not a production market-data feed.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


DEFAULT_SYMBOLS = ["NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA"]
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


def fetch_chart(symbol: str, interval: str, data_range: str) -> list[dict[str, object]]:
    params = urlencode(
        {
            "range": data_range,
            "interval": interval,
            "includePrePost": "false",
            "events": "div,splits",
        }
    )
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?{params}"
    request = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 Kronos local data fetcher",
            "Accept": "application/json",
        },
    )

    with urlopen(request, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))

    chart = payload.get("chart", {})
    error = chart.get("error")
    if error:
        raise RuntimeError(f"{symbol}: Yahoo returned error: {error}")

    results = chart.get("result") or []
    if not results:
        raise RuntimeError(f"{symbol}: no chart data returned")

    result = results[0]
    timestamps = result.get("timestamp") or []
    quote = ((result.get("indicators") or {}).get("quote") or [{}])[0]

    rows = []
    for idx, ts in enumerate(timestamps):
        try:
            open_ = quote["open"][idx]
            high = quote["high"][idx]
            low = quote["low"][idx]
            close = quote["close"][idx]
            volume = quote["volume"][idx]
        except (KeyError, IndexError):
            continue

        if None in (open_, high, low, close):
            continue

        volume = 0 if volume is None else volume
        close = float(close)
        volume = int(volume)
        rows.append(
            {
                "timestamps": datetime.fromtimestamp(ts, timezone.utc).isoformat(),
                "open": float(open_),
                "high": float(high),
                "low": float(low),
                "close": close,
                "volume": volume,
                "amount": close * volume,
            }
        )

    if not rows:
        raise RuntimeError(f"{symbol}: no usable OHLCV rows returned")

    return rows


def write_csv(symbol: str, rows: list[dict[str, object]], interval: str) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / f"{symbol}_{interval}.csv"
    fieldnames = ["timestamps", "open", "high", "low", "close", "volume", "amount"]

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch public candles for Kronos.")
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=DEFAULT_SYMBOLS,
        help="Ticker symbols to fetch. Defaults to NVDA and Mag 7.",
    )
    parser.add_argument("--interval", default="5m", help="Candle interval, e.g. 1m, 5m, 15m, 1h, 1d.")
    parser.add_argument("--range", default="5d", dest="data_range", help="Yahoo range, e.g. 1d, 5d, 1mo.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    failures = 0

    for symbol in args.symbols:
        symbol = symbol.upper()
        try:
            rows = fetch_chart(symbol, args.interval, args.data_range)
            path = write_csv(symbol, rows, args.interval)
            print(f"{symbol}: wrote {len(rows)} rows to {path}")
        except (HTTPError, URLError, TimeoutError, RuntimeError) as exc:
            failures += 1
            print(f"{symbol}: failed: {exc}", file=sys.stderr)

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
