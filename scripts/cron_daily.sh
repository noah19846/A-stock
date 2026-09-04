#!/usr/bin/env bash
# Cron: 交易日才跑正式日更（默认 14:30）
# 默认发信：服务器 crontab 设 CRON_MAIL=1
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export TZ="${TZ:-Asia/Shanghai}"
mkdir -p "$ROOT/logs"

LOG="$ROOT/logs/cron_daily_$(date +%Y%m%d).log"
exec >>"$LOG" 2>&1

echo "==== $(date '+%F %T %Z') cron_daily start ===="

if ! "$ROOT/.venv/bin/python" "$ROOT/scripts/is_trade_day.py"; then
  echo "skip: not a trade day"
  echo "==== $(date '+%F %T %Z') cron_daily end (skipped) ===="
  exit 0
fi

ARGS=()
if [[ "${CRON_MAIL:-0}" == "1" ]]; then
  ARGS+=(--mail)
fi

echo "run: make daily ARGS='${ARGS[*]:-}'"
make -C "$ROOT" daily ARGS="${ARGS[*]:-}"
echo "==== $(date '+%F %T %Z') cron_daily end (ok) ===="
