#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$ROOT_DIR/logs"
LOG_FILE="$LOG_DIR/daily_data_refresh.log"

mkdir -p "$LOG_DIR"

{
  echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] starting daily data refresh"
  cd "$ROOT_DIR"

  echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] refreshing 5-minute candles"
  python3 examples/fetch_public_candles.py \
    --symbols NVDA AAPL MSFT AMZN GOOGL META TSLA \
    --interval 5m \
    --range 1mo

  echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] refreshing 1-hour candles"
  python3 examples/fetch_public_candles.py \
    --symbols NVDA AAPL MSFT AMZN GOOGL META TSLA \
    --interval 1h \
    --range 1y

  echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] refreshing 1-day candles"
  python3 examples/fetch_public_candles.py \
    --symbols NVDA AAPL MSFT AMZN GOOGL META TSLA \
    --interval 1d \
    --range 5y

  echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] daily data refresh completed"
} >> "$LOG_FILE" 2>&1
