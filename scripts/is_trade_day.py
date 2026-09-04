#!/usr/bin/env python3
"""Exit 0 if today (Asia/Shanghai) is an A-share trade day, else 1."""

from __future__ import annotations

from datetime import date

import akshare as ak


def main() -> int:
    today = date.today().isoformat()
    td = ak.tool_trade_date_hist_sina()
    days = set(td["trade_date"].astype(str).str.slice(0, 10))
    if today in days:
        print(f"trade_day={today}")
        return 0
    print(f"non_trade_day={today}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
