"""涨停后放量洗盘、缩量回踩支撑的独立研究策略。"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import daily_cache

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "limit_wash_retest_signals.csv"
PREFIX = ("600", "601", "603", "605", "000", "001", "002")

LIMIT_RET = 0.095
WASH_RET_MAX = -0.01
WASH_VOL_MIN = 0.80
WASH_VOL_MAX = 1.50
SHRINK_VOL_MAX = 0.85
SUPPORT_BREAK = 0.02
BUY_GAP_MAX = 0.03
SIDEWAYS_WINDOW = 60
SIDEWAYS_RANGE_MAX = 0.25
FIRST_BOARD_LOOKBACK = 20
HOLD_MAX = 8
STOP = 0.03
TARGET = 0.08
COST = 0.0015


def name_map() -> dict[str, str]:
    p = ROOT / "data" / "stock_list.csv"
    d = pd.read_csv(p, dtype=str)
    return dict(zip(d.iloc[:, 0].str.zfill(6), d.iloc[:, 1]))


def find_signals(df: pd.DataFrame, code: str, name: str) -> list[dict]:
    if df is None or len(df) < 100:
        return []
    px = df.px.to_numpy(float)
    op = df.op.to_numpy(float)
    hi = df.hi.to_numpy(float)
    lo = df.lo.to_numpy(float)
    vol = df.vol.to_numpy(float)
    ret = df.ret.to_numpy(float)
    dates = pd.to_datetime(df["日期"])
    out = []
    i = max(SIDEWAYS_WINDOW, FIRST_BOARD_LOOKBACK)
    while i + 2 < len(df):
        if not np.isfinite(ret[i]) or ret[i] < LIMIT_RET:
            i += 1
            continue
        # 涨停前先横盘一段时间，且过去一段时间没有出现过涨停，避免把连板或高位二板混进来。
        sideways = px[i - SIDEWAYS_WINDOW : i]
        sideways_range = np.max(sideways) / np.min(sideways) - 1 if np.min(sideways) > 0 else np.inf
        prior_limit = np.nanmax(ret[i - FIRST_BOARD_LOOKBACK : i]) >= LIMIT_RET
        if sideways_range > SIDEWAYS_RANGE_MAX or prior_limit:
            i += 1
            continue
        if i + 1 >= len(df) or px[i + 1] >= px[i] or ret[i + 1] > WASH_RET_MAX:
            i += 1
            continue
        if vol[i] <= 0 or not (WASH_VOL_MIN <= vol[i + 1] / vol[i] <= WASH_VOL_MAX):
            i += 1
            continue

        wash_close = px[i + 1]
        wash_vol = vol[i + 1]
        # 支撑固定为放量洗盘日低点，新低不能被重新定义成支撑。
        support = lo[i + 1]
        declines = 0
        buy_i = None
        for j in range(i + 2, min(i + 6, len(df))):
            declines += int(px[j] < px[j - 1])
            recent_vol = vol[i + 2 : j + 1]
            shrink = len(recent_vol) >= 2 and np.mean(recent_vol) / wash_vol <= SHRINK_VOL_MAX
            not_broken = lo[j] >= support * (1.0 - SUPPORT_BREAK)
            # 买点必须仍在支撑上方，且不能离支撑太远；否则属于追涨或已经破位。
            still_near = support <= px[j] <= support * (1.0 + BUY_GAP_MAX)
            if declines >= 2 and shrink and not_broken and still_near:
                buy_i = j
                break
            if px[j] < support * (1.0 - SUPPORT_BREAK):
                break
        if buy_i is not None:
            out.append(
                {
                    "code": code,
                    "name": name,
                    "limit_date": str(dates.iloc[i].date()),
                    "wash_date": str(dates.iloc[i + 1].date()),
                    "buy_date": str(dates.iloc[buy_i].date()),
                    "limit_ret%": round(ret[i] * 100, 2),
                    "sideways_range%": round(sideways_range * 100, 2),
                    "wash_ret%": round(ret[i + 1] * 100, 2),
                    "wash_vol_ratio": round(vol[i + 1] / vol[i], 2),
                    "buy_price": round(px[buy_i], 4),
                    "support": round(support, 4),
                    "buy_support_gap%": round((px[buy_i] / support - 1) * 100, 2),
                }
            )
            i = buy_i
        i += 1
    return out


def scan(asof: str | None = None) -> pd.DataFrame:
    daily_cache.preload()
    names = name_map()
    rows = []
    for code in daily_cache.cached_codes():
        if not code.startswith(PREFIX) or "ST" in names.get(code, "").upper():
            continue
        d = daily_cache.get(code, asof=asof)
        rows.extend(find_signals(d, code, names.get(code, "")))
    return pd.DataFrame(rows)


def backtest(start: str = "2024-01-01", end: str | None = None) -> pd.DataFrame:
    daily_cache.preload()
    names = name_map()
    trades = []
    for code in daily_cache.cached_codes():
        if not code.startswith(PREFIX) or "ST" in names.get(code, "").upper():
            continue
        d = daily_cache.get(code, asof=end)
        signals = find_signals(d, code, names.get(code, ""))
        if d is None:
            continue
        dates = pd.to_datetime(d["日期"])
        px, hi, lo = d.px.to_numpy(float), d.hi.to_numpy(float), d.lo.to_numpy(float)
        for s in signals:
            if s["buy_date"] < start:
                continue
            bi = int(np.flatnonzero(dates == pd.Timestamp(s["buy_date"]))[0])
            # 信号在收盘后产生，按下一交易日收盘价入场，避免使用当日未知的收盘成交。
            entry_i = bi + 1
            if entry_i >= len(d):
                continue
            entry = px[entry_i]
            exit_i = min(entry_i + HOLD_MAX - 1, len(d) - 1)
            exit_price = px[exit_i]
            reason = "到期"
            exit_date = dates.iloc[exit_i]
            for j in range(entry_i, exit_i + 1):
                if lo[j] <= entry * (1 - STOP):
                    exit_price, reason = entry * (1 - STOP), "止损"
                    exit_date = dates.iloc[j]
                    break
                if hi[j] >= entry * (1 + TARGET):
                    exit_price, reason = entry * (1 + TARGET), "止盈"
                    exit_date = dates.iloc[j]
                    break
            trades.append(
                {
                    **s,
                    "entry_date": str(dates.iloc[entry_i].date()),
                    "entry_price": round(entry, 4),
                    "exit_date": str(exit_date.date()),
                    "exit_reason": reason,
                    "ret%": round((exit_price / entry - 1 - COST) * 100, 3),
                }
            )
    return pd.DataFrame(trades)


def forward_stats(start: str = "2024-01-01", end: str | None = None) -> pd.DataFrame:
    """按买点当日收盘价买入，只观察后续固定交易日表现，不模拟交易退出。"""
    daily_cache.preload()
    names = name_map()
    rows = []
    horizons = (1, 3, 5, 8)
    for code in daily_cache.cached_codes():
        if not code.startswith(PREFIX) or "ST" in names.get(code, "").upper():
            continue
        d = daily_cache.get(code, asof=end)
        if d is None:
            continue
        signals = find_signals(d, code, names.get(code, ""))
        dates = pd.to_datetime(d["日期"])
        px = d.px.to_numpy(float)
        for s in signals:
            if s["buy_date"] < start:
                continue
            bi = int(np.flatnonzero(dates == pd.Timestamp(s["buy_date"]))[0])
            row = {**s, "entry_price_close": round(px[bi], 4)}
            for h in horizons:
                fi = bi + h
                row[f"fwd_{h}d%"] = round((px[fi] / px[bi] - 1) * 100, 3) if fi < len(px) else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", action="store_true")
    ap.add_argument("--backtest", action="store_true")
    ap.add_argument("--forward", action="store_true", help="只统计买点当日收盘买入后的固定周期表现")
    ap.add_argument("--asof", default="")
    ap.add_argument("--start", default="2024-01-01")
    args = ap.parse_args()
    if args.backtest:
        out = backtest(args.start, args.asof or None)
        print(f"交易数 {len(out)}")
        if not out.empty:
            print(f"胜率 {(out['ret%'] > 0).mean() * 100:.1f}%  平均 {out['ret%'].mean():+.2f}%  中位数 {out['ret%'].median():+.2f}%")
            print(out.groupby("exit_reason")["ret%"].agg(["count", "mean"]).to_string())
            print(out.head(30).to_string(index=False))
        return
    if args.forward:
        out = forward_stats(args.start, args.asof or None)
        print(f"信号数 {len(out)}")
        if not out.empty:
            for h in (1, 3, 5, 8):
                col = f"fwd_{h}d%"
                q = out[col].dropna()
                print(f"{h}日后：上涨比例 {(q > 0).mean() * 100:.1f}%  平均 {q.mean():+.2f}%  中位数 {q.median():+.2f}%  样本 {len(q)}")
            print(out.to_string(index=False))
        return
    out = scan(args.asof or None)
    out.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"候选 {len(out)} 条 → {OUT}")
    if not out.empty:
        print(out.to_string(index=False))


if __name__ == "__main__":
    main()
