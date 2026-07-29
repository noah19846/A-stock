"""
短线爆发信号回测

信号日 T（收盘后判定，规则同「可短打」）：
  hard_score >= 0.85 且 soft_score >= min_score
  + 流动性/过热硬过滤

成交：
  默认次日开盘买入；对照：次日收盘买入

出场（相对买入价，持有最多 hold_n 日）：
  - 盘中最高触及 +target → 按目标价出（乐观可成交假设）
  - 盘中最低触及 -stop → 按止损价出
  - 同日既触目标又触止损 → 按「先止损」保守处理
  - 否则持有期满按收盘出

成本：双边合计 cost_bps（默认 15bp = 0.15%）

输出：
  data/short_burst_backtest_trades.csv
  data/short_burst_backtest_summary.txt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_short_burst_features import features_at, load_daily, load_maps, score_row

ROOT = Path(__file__).resolve().parent
DAILY = ROOT / "data" / "daily_raw"
BANDS = ROOT / "data" / "short_burst_feature_bands.json"
OUT_TRADES = ROOT / "data" / "short_burst_backtest_trades.csv"
OUT_SUMMARY = ROOT / "data" / "short_burst_backtest_summary.txt"

LOOKBACK = 520
HOLD_N = 5
TARGET = 0.08
STOP = 0.03
COST_BPS = 15.0  # 往返


def load_bands() -> dict:
    return json.loads(BANDS.read_text(encoding="utf-8"))


def is_signal(feat: dict, soft: float, hard: float, bands: dict) -> bool:
    min_score = float(bands.get("min_score", 0.72))
    if hard < 0.85 or soft < min_score:
        return False
    if feat["流通市值亿"] < 40 or feat["流通市值亿"] > 800:
        return False
    if feat["换手率%"] > 12 or feat["换手率%"] < 0.8:
        return False
    if feat["前5日涨幅%"] > 8 or feat["距20日高点%"] > -2:
        return False
    if feat.get("前1日涨幅%", 0) > 3:
        return False
    return True


def simulate_trade(
    df: pd.DataFrame,
    signal_i: int,
    entry_mode: str,
    hold_n: int,
    target: float,
    stop: float,
    cost: float,
) -> dict | None:
    """signal_i 为信号日；买入在 signal_i+1。"""
    buy_i = signal_i + 1
    if buy_i >= len(df):
        return None
    if entry_mode == "open":
        entry = float(df["op"].iloc[buy_i])
    else:
        entry = float(df["px"].iloc[buy_i])
    if not np.isfinite(entry) or entry <= 0:
        return None

    # 涨停买不进：开盘相对昨收涨幅 >= 9.5% 则放弃
    prev = float(df["px"].iloc[signal_i])
    if entry_mode == "open" and prev > 0 and (entry / prev - 1.0) >= 0.095:
        return {
            "skipped": True,
            "reason": "次日高开近涨停，放弃",
            "signal_i": signal_i,
            "buy_i": buy_i,
        }

    hi = df["hi"].values
    lo = df["lo"].values
    px = df["px"].values
    dates = df["日期"]

    exit_i = None
    exit_px = None
    reason = None
    last = min(buy_i + hold_n - 1, len(df) - 1)

    for j in range(buy_i, last + 1):
        day_hi = float(hi[j])
        day_lo = float(lo[j])
        hit_tp = day_hi >= entry * (1.0 + target)
        hit_sl = day_lo <= entry * (1.0 - stop)
        if hit_tp and hit_sl:
            exit_i = j
            exit_px = entry * (1.0 - stop)
            reason = "同日止损优先"
            break
        if hit_sl:
            exit_i = j
            exit_px = entry * (1.0 - stop)
            reason = "止损"
            break
        if hit_tp:
            exit_i = j
            exit_px = entry * (1.0 + target)
            reason = "止盈"
            break

    if exit_i is None:
        exit_i = last
        exit_px = float(px[exit_i])
        reason = "到期"

    raw_ret = exit_px / entry - 1.0
    net_ret = raw_ret - cost
    hold_days = int(exit_i - buy_i + 1)

    # 持有期最大不利（相对买入价）
    mfe = float(hi[buy_i : exit_i + 1].max() / entry - 1.0)
    mae = float(lo[buy_i : exit_i + 1].min() / entry - 1.0)

    return {
        "skipped": False,
        "signal_date": str(dates.iloc[signal_i].date()),
        "entry_date": str(dates.iloc[buy_i].date()),
        "exit_date": str(dates.iloc[exit_i].date()),
        "entry": round(entry, 4),
        "exit": round(exit_px, 4),
        "hold_days": hold_days,
        "exit_reason": reason,
        "raw_ret%": round(raw_ret * 100.0, 3),
        "net_ret%": round(net_ret * 100.0, 3),
        "mfe%": round(mfe * 100.0, 3),
        "mae%": round(mae * 100.0, 3),
        "buy_i": buy_i,
        "exit_i": exit_i,
    }


def backtest_stock(
    code: str,
    df: pd.DataFrame,
    bands: dict,
    entry_mode: str,
    hold_n: int,
    target: float,
    stop: float,
    cost: float,
) -> list[dict]:
    start = max(60, len(df) - LOOKBACK)
    end = len(df) - hold_n - 2
    if end <= start:
        return []

    trades = []
    i = start
    while i <= end:
        feat = features_at(df, i)
        if feat is None:
            i += 1
            continue
        soft, hard, _ = score_row(feat, bands)
        if not is_signal(feat, soft, hard, bands):
            i += 1
            continue
        tr = simulate_trade(df, i, entry_mode, hold_n, target, stop, cost)
        if tr is None:
            i += 1
            continue
        if tr.get("skipped"):
            i += 1
            continue
        tr["code"] = code
        tr["soft"] = round(soft, 3)
        tr["hard"] = round(hard, 3)
        trades.append(tr)
        # 非重叠：出场后再找信号
        i = int(tr["exit_i"]) + 1
    return trades


def summarize(trades: pd.DataFrame, label: str) -> str:
    if trades.empty:
        return f"[{label}] 无成交"
    r = trades["net_ret%"]
    win = r > 0
    reasons = trades["exit_reason"].value_counts().to_dict()
    # 简易资金曲线：等权每笔满仓接力（非同时多票）
    equity = (1.0 + r / 100.0).cumprod()
    peak = equity.cummax()
    dd = (equity / peak - 1.0).min() * 100.0
    # 按年
    trades = trades.copy()
    trades["year"] = pd.to_datetime(trades["entry_date"]).dt.year
    by_year = trades.groupby("year").agg(
        n=("net_ret%", "count"),
        winrate=("net_ret%", lambda s: float((s > 0).mean() * 100)),
        avg=("net_ret%", "mean"),
        med=("net_ret%", "median"),
        sum=("net_ret%", "sum"),
    )

    lines = [
        f"=== {label} ===",
        f"成交笔数: {len(trades)}",
        f"覆盖股票: {trades['code'].nunique()}",
        f"胜率: {win.mean()*100:.1f}%",
        f"平均净收益: {r.mean():.3f}%",
        f"中位净收益: {r.median():.3f}%",
        f"盈亏比(均盈/|均亏|): "
        f"{(r[win].mean() / abs(r[~win].mean())) if (~win).any() and win.any() else float('nan'):.2f}",
        f"平均持有天数: {trades['hold_days'].mean():.2f}",
        f"止盈占比: {reasons.get('止盈', 0)/len(trades)*100:.1f}%",
        f"止损占比: {(reasons.get('止损', 0)+reasons.get('同日止损优先', 0))/len(trades)*100:.1f}%",
        f"到期占比: {reasons.get('到期', 0)/len(trades)*100:.1f}%",
        f"单笔接力权益总回报: {(equity.iloc[-1]-1)*100:.1f}%",
        f"单笔接力最大回撤: {dd:.1f}%",
        f"期望值(每笔): {r.mean():.3f}%",
        "",
        "出场原因:",
        str(pd.Series(reasons)),
        "",
        "分年:",
        by_year.round(2).to_string(),
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="短线爆发回测")
    parser.add_argument("--entry", choices=["open", "close"], default="open")
    parser.add_argument("--hold", type=int, default=HOLD_N)
    parser.add_argument("--target", type=float, default=TARGET)
    parser.add_argument("--stop", type=float, default=STOP)
    parser.add_argument("--cost-bps", type=float, default=COST_BPS)
    args = parser.parse_args()

    if not BANDS.exists():
        raise SystemExit(f"缺少 {BANDS}")

    bands = load_bands()
    name_map, ind_map = load_maps()
    cost = args.cost_bps / 10000.0

    all_trades: list[dict] = []
    files = sorted(p for p in DAILY.glob("*.csv") if p.name[0].isdigit())
    n = 0
    for path in files:
        code = path.stem.zfill(6)
        if not code.startswith(("600", "601", "603", "605", "000", "001", "002")):
            continue
        if "ST" in name_map.get(code, "").upper():
            continue
        df = load_daily(code)
        if df is None:
            continue
        n += 1
        trades = backtest_stock(
            code, df, bands, args.entry, args.hold, args.target, args.stop, cost
        )
        for t in trades:
            t["name"] = name_map.get(code, "")
            t["industry"] = ind_map.get(code, "")
            all_trades.append(t)
        if n % 400 == 0:
            print(f"…已扫 {n} 只，累计成交 {len(all_trades)}", flush=True)

    out = pd.DataFrame(all_trades)
    if not out.empty:
        out = out.sort_values("entry_date").reset_index(drop=True)
        # 去掉内部索引列
        drop_cols = [c for c in ("buy_i", "exit_i", "skipped") if c in out.columns]
        out = out.drop(columns=drop_cols, errors="ignore")

    OUT_TRADES.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_TRADES, index=False, encoding="utf-8-sig")

    label = (
        f"entry={args.entry}, hold={args.hold}, "
        f"tp=+{args.target*100:.0f}%, sl=-{args.stop*100:.0f}%, cost={args.cost_bps:.0f}bp"
    )
    text = summarize(out, label)
    # 额外：若同时多票，按日信号数
    if not out.empty:
        daily_n = out.groupby("entry_date").size()
        text += (
            f"\n\n同日新开仓笔数 中位/P90: {daily_n.median():.0f} / {daily_n.quantile(0.9):.0f}"
            f"\n说明: 「单笔接力权益」是假设每次只做一笔串起来；实盘若并行多票，回撤与复合会不同。"
            f"\n注意: 规则区间来自含同期样本的特征挖掘，存在一定过拟合，分年表现请重点看。"
        )
    OUT_SUMMARY.write_text(text, encoding="utf-8")
    print(text, flush=True)
    print(f"\n→ {OUT_TRADES}\n→ {OUT_SUMMARY}", flush=True)


if __name__ == "__main__":
    main()
