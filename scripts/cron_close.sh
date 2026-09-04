#!/usr/bin/env bash
# Cron: 交易日才跑收盘（默认 15:10）
# 默认发信：服务器 crontab 设 CRON_MAIL=1（观察簿邮件）
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export TZ="${TZ:-Asia/Shanghai}"
mkdir -p "$ROOT/logs"

LOG="$ROOT/logs/cron_close_$(date +%Y%m%d).log"
exec >>"$LOG" 2>&1

echo "==== $(date '+%F %T %Z') cron_close start ===="

if ! "$ROOT/.venv/bin/python" "$ROOT/scripts/is_trade_day.py"; then
  echo "skip: not a trade day"
  echo "==== $(date '+%F %T %Z') cron_close end (skipped) ===="
  exit 0
fi

MAIL_ARGS=()
if [[ "${CRON_MAIL:-0}" == "1" ]]; then
  MAIL_ARGS+=(--mail)
fi

echo "run: make close ARGS='${MAIL_ARGS[*]:-}'"
make -C "$ROOT" close ARGS="${MAIL_ARGS[*]:-}"
echo "==== $(date '+%F %T %Z') cron_close end (ok) ===="
