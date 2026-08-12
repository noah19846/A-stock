"""
实盘交易台账（trades/live_trades.csv）

用法：
  .venv/bin/python trades_log.py list
  .venv/bin/python trades_log.py open
  .venv/bin/python trades_log.py buy --code 603507 --shares 200 --fill 23.16 \\
      --strategy short/default --signal-date 2026-08-03
  .venv/bin/python trades_log.py sell --id T20260803-001 --price 23.82 --pnl 118.52 --reason 清仓

只记买入价/卖出价与实际盈利；手续费 = 卖出金额 - 买入金额 - 盈利（可负）。

默认离场（未显式传 --stop/--target/--hold 时）：
  short/*  → 止损-3% / 止盈+15% / 最多8日
  long|rally → 读 data/rally_exit.json（默认 -10% / +25% / 40日，并在 notes 记 D15峰值<8%清仓）
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
LEDGER = ROOT / "trades" / "live_trades.csv"
RALLY_EXIT = ROOT / "data" / "rally_exit.json"

COLS = [
    "trade_id",
    "status",
    "code",
    "name",
    "side",
    "shares",
    "fill_price",
    "amount",
    "fees",
    "entry_date",
    "signal_date",
    "strategy",
    "stop_price",
    "target_price",
    "hold_days_max",
    "exit_date",
    "exit_price",
    "exit_reason",
    "pnl",
    "pnl_pct",
    "notes",
]


def log(msg: str) -> None:
    print(msg, flush=True)


def load() -> pd.DataFrame:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    if not LEDGER.exists():
        df = pd.DataFrame(columns=COLS)
        df.to_csv(LEDGER, index=False, encoding="utf-8-sig")
        return df
    df = pd.read_csv(LEDGER, dtype={"code": str, "trade_id": str})
    if "code" in df.columns:
        df["code"] = df["code"].astype(str).str.zfill(6)
    # 兼容旧列：含费成本价已废弃
    if "cost_price" in df.columns:
        df = df.drop(columns=["cost_price"])
    df = df.fillna("")
    return df


def save(df: pd.DataFrame) -> None:
    out = df.copy()
    if "cost_price" in out.columns:
        out = out.drop(columns=["cost_price"])
    for c in COLS:
        if c not in out.columns:
            out[c] = ""
    out[COLS].to_csv(LEDGER, index=False, encoding="utf-8-sig")


def name_of(code: str) -> str:
    code = str(code).zfill(6)
    path = ROOT / "data" / "stock_list.csv"
    if path.exists():
        try:
            sl = pd.read_csv(path, dtype=str)
            hit = sl[sl["股票代码"].astype(str).str.zfill(6) == code]
            if not hit.empty:
                return str(hit.iloc[0]["股票名称"])
        except Exception:
            pass
    return ""


def next_id(entry_date: str) -> str:
    df = load()
    day = entry_date.replace("-", "")
    prefix = f"T{day}-"
    n = 1
    if not df.empty and "trade_id" in df.columns:
        same = df["trade_id"].astype(str).str.startswith(prefix)
        if same.any():
            seqs = []
            for tid in df.loc[same, "trade_id"]:
                try:
                    seqs.append(int(str(tid).split("-")[-1]))
                except Exception:
                    pass
            if seqs:
                n = max(seqs) + 1
    return f"{prefix}{n:03d}"


def cmd_list(status: str = "") -> None:
    df = load()
    if df.empty:
        log("(空台账)")
        return
    if status:
        df = df[df["status"] == status]
    if df.empty:
        log(f"(无 status={status})")
        return
    cols = [
        c
        for c in [
            "trade_id",
            "status",
            "code",
            "name",
            "shares",
            "fill_price",
            "exit_price",
            "fees",
            "entry_date",
            "exit_date",
            "strategy",
            "stop_price",
            "target_price",
            "pnl",
            "pnl_pct",
            "notes",
        ]
        if c in df.columns
    ]
    log(df[cols].to_string(index=False))
    log(f"\n→ {LEDGER}")


def _strategy_kind(strategy: str) -> str:
    s = (strategy or "").lower()
    if "short" in s:
        return "short"
    if "long" in s or "rally" in s:
        return "long"
    return ""


def load_rally_exit() -> dict:
    if RALLY_EXIT.exists():
        return json.loads(RALLY_EXIT.read_text(encoding="utf-8"))
    return {
        "stop": 0.10,
        "target": 0.25,
        "hold_days_max": 40,
        "early_check_day": 15,
        "early_min_mfe": 0.08,
        "desc": "硬止损-10%；止盈+25%；最多40日；D15峰值<8%清仓",
    }


def cmd_buy(args: argparse.Namespace) -> None:
    code = str(args.code).zfill(6)
    shares = int(args.shares)
    fill = float(args.fill)
    entry_date = args.date or datetime.now().strftime("%Y-%m-%d")
    signal_date = args.signal_date or entry_date
    amount = round(fill * shares, 2)
    stop = args.stop
    target = args.target
    hold = args.hold
    notes = args.notes or ""
    kind = _strategy_kind(args.strategy or "")

    if kind == "short":
        if stop is None:
            stop = round(fill * (1 - 0.03), 2)
        if target is None:
            target = round(fill * (1 + 0.15), 2)
        if hold is None:
            hold = 8
    elif kind == "long":
        rex = load_rally_exit()
        if stop is None:
            stop = round(fill * (1 - float(rex.get("stop", 0.10))), 2)
        if target is None:
            target = round(fill * (1 + float(rex.get("target", 0.25))), 2)
        if hold is None:
            hold = int(rex.get("hold_days_max", 40))
        early = (
            f"D{int(rex.get('early_check_day', 15))}峰值<"
            f"{float(rex.get('early_min_mfe', 0.08))*100:.0f}%清仓"
        )
        if early not in notes:
            notes = f"{notes} | {early}".strip(" |") if notes else early

    row = {
        "trade_id": next_id(entry_date),
        "status": "open",
        "code": code,
        "name": args.name or name_of(code),
        "side": "buy",
        "shares": shares,
        "fill_price": fill,
        "amount": amount,
        "fees": "",
        "entry_date": entry_date,
        "signal_date": signal_date,
        "strategy": args.strategy or "",
        "stop_price": stop if stop is not None else "",
        "target_price": target if target is not None else "",
        "hold_days_max": "" if hold is None else int(hold),
        "exit_date": "",
        "exit_price": "",
        "exit_reason": "",
        "pnl": "",
        "pnl_pct": "",
        "notes": notes,
    }
    df = load()
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    save(df)
    log(
        f"已记账 {row['trade_id']}  {code} {row['name']}  {shares}股  "
        f"买入价 {fill}  止损 {row['stop_price']}  止盈 {row['target_price']}"
        f"  持有上限 {row['hold_days_max']}"
        + (f"  notes={notes}" if notes else "")
    )
    log(f"→ {LEDGER}")


def cmd_sell(args: argparse.Namespace) -> None:
    df = load()
    tid = str(args.id)
    hit = df.index[df["trade_id"].astype(str) == tid]
    if len(hit) == 0:
        raise SystemExit(f"找不到 trade_id={tid}")
    i = int(hit[0])
    if str(df.at[i, "status"]) != "open":
        raise SystemExit(f"{tid} 状态不是 open：{df.at[i, 'status']}")
    exit_price = float(args.price)
    pnl = round(float(args.pnl), 2)
    shares = int(df.at[i, "shares"])
    fill = float(df.at[i, "fill_price"])
    buy_amt = round(fill * shares, 2)
    sell_amt = round(exit_price * shares, 2)
    # 手续费 = 卖出金额 - 买入金额 - 盈利
    fees = round(sell_amt - buy_amt - pnl, 2)
    pnl_pct = round(pnl / buy_amt * 100, 3) if buy_amt else 0.0
    exit_date = args.date or datetime.now().strftime("%Y-%m-%d")
    df.at[i, "status"] = "closed"
    df.at[i, "amount"] = buy_amt
    df.at[i, "fees"] = fees
    df.at[i, "exit_date"] = exit_date
    df.at[i, "exit_price"] = exit_price
    df.at[i, "exit_reason"] = args.reason or ""
    df.at[i, "pnl"] = pnl
    df.at[i, "pnl_pct"] = pnl_pct
    if args.notes:
        prev = str(df.at[i, "notes"] or "")
        df.at[i, "notes"] = (prev + " | " if prev else "") + args.notes
    save(df)
    log(
        f"已平仓 {tid}  卖出价 {exit_price}  PnL {pnl:+.2f} ({pnl_pct:+.3f}%)  "
        f"手续费 {fees:.2f}  原因={args.reason or '-'}"
    )
    log(f"→ {LEDGER}")


def main() -> None:
    parser = argparse.ArgumentParser(description="实盘交易台账")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_l = sub.add_parser("list", help="列出记录")
    p_l.add_argument("--status", default="", help="open / closed")

    sub.add_parser("open", help="只看持仓中")

    p_b = sub.add_parser("buy", help="记一笔买入")
    p_b.add_argument("--code", required=True)
    p_b.add_argument("--shares", type=int, required=True)
    p_b.add_argument("--fill", type=float, required=True, help="买入成交价")
    p_b.add_argument("--name", default="")
    p_b.add_argument("--date", default="", help="买入日 YYYY-MM-DD")
    p_b.add_argument("--signal-date", default="")
    p_b.add_argument("--strategy", default="short/default")
    p_b.add_argument("--stop", type=float, default=None)
    p_b.add_argument("--target", type=float, default=None)
    p_b.add_argument("--hold", type=int, default=None)
    p_b.add_argument("--notes", default="")

    p_s = sub.add_parser("sell", help="平仓")
    p_s.add_argument("--id", required=True, help="trade_id")
    p_s.add_argument("--price", type=float, required=True, help="卖出成交价")
    p_s.add_argument("--pnl", type=float, required=True, help="实际盈利（可为负）")
    p_s.add_argument("--date", default="")
    p_s.add_argument("--reason", default="")
    p_s.add_argument("--notes", default="")

    args = parser.parse_args()
    if args.cmd == "list":
        cmd_list(args.status)
    elif args.cmd == "open":
        cmd_list("open")
    elif args.cmd == "buy":
        cmd_buy(args)
    elif args.cmd == "sell":
        cmd_sell(args)


if __name__ == "__main__":
    main()
