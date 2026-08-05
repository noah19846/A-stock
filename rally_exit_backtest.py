"""
长线 rally Entry 信号的出场规则扫描

口径：
  - 信号：近 setup_lookback 日曾 Setup，当日 Entry，且非过热
  - 成交：T+1 开盘；次日高开近涨停则放弃
  - 成本：双边合计 cost_bps（默认 15bp）
  - 同日既触止盈又触止损 → 先止损
  - 同票入场集锁定 max_hold 日，保证多套出场规则共用同一批入场

用法：
  .venv/bin/python rally_exit_backtest.py
  .venv/bin/python rally_exit_backtest.py --quick
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from rally_buy_screener import ENTRY, RISK, SETUP, load_maps

ROOT = Path(__file__).resolve().parent
OUT_SUMMARY = ROOT / "data" / "rally_exit_backtest_summary.txt"
OUT_COMPARE = ROOT / "data" / "rally_exit_backtest_compare.csv"
OUT_TRADES_DIR = ROOT / "data"
OUT_JSON = ROOT / "data" / "rally_exit_backtest_pick.json"

LOOKBACK = 520
COST_BPS = 15.0
MIN_TRADES = 80
HOLD_LOCK = 80  # 入场后锁定交易日，保证各规则同入场集


@dataclass(frozen=True)
class ExitRule:
    id: str
    family: str  # fixed | ma | hybrid
    hold_n: int
    stop: float | None = None
    target: float | None = None
    ma_period: int | None = None  # 10/20/60
    ma_below_days: int | None = None  # 连续收盘在均线下 N 日
    reclaim_days: int | None = None  # 破均线后 N 日内站不回则清
    protect_at: float | None = None
    protect_floor: float = 0.0
    trail: float | None = None

    def label(self) -> str:
        parts = [self.family, f"H{self.hold_n}"]
        if self.stop is not None:
            parts.append(f"SL{int(round(self.stop * 100))}")
        if self.target is not None:
            parts.append(f"TP{int(round(self.target * 100))}")
        if self.ma_period is not None:
            bd = self.ma_below_days or 1
            parts.append(f"MA{self.ma_period}x{bd}")
        if self.reclaim_days is not None:
            parts.append(f"R{self.reclaim_days}")
        if self.protect_at is not None:
            parts.append(f"P{int(round(self.protect_at * 100))}")
            if self.trail is not None:
                parts.append(f"T{int(round(self.trail * 100))}")
            elif self.protect_floor == 0:
                parts.append("BE")
        return "_".join(parts)


def build_rules(*, quick: bool = False) -> list[ExitRule]:
    rules: list[ExitRule] = []
    holds = [40, 60] if quick else [40, 60, 80]
    stops = [0.08, 0.10] if quick else [0.06, 0.08, 0.10]
    targets = [0.20, 0.30] if quick else [0.15, 0.20, 0.25, 0.30]

    for h in holds:
        for sl in stops:
            for tp in targets:
                rules.append(
                    ExitRule(
                        id=f"fixed_sl{int(sl*100)}_tp{int(tp*100)}_h{h}",
                        family="fixed",
                        hold_n=h,
                        stop=sl,
                        target=tp,
                    )
                )

    ma_hold = 60 if quick else 80
    for period, days in [(20, 1), (20, 2), (20, 3), (10, 1), (60, 1)]:
        rules.append(
            ExitRule(
                id=f"ma{period}_x{days}_h{ma_hold}",
                family="ma",
                hold_n=ma_hold,
                ma_period=period,
                ma_below_days=days,
            )
        )
    for reclaim in ([3, 5] if not quick else [5]):
        rules.append(
            ExitRule(
                id=f"ma20_reclaim{reclaim}_h{ma_hold}",
                family="ma",
                hold_n=ma_hold,
                ma_period=20,
                reclaim_days=reclaim,
            )
        )

    # 混合：硬止损 + 保本/跟踪 + 连续破 MA20 + 时间上限
    hybrids = [
        (0.08, 0.12, 0.0, None, 3, 60),
        (0.08, 0.15, 0.0, None, 3, 60),
        (0.10, 0.12, 0.0, None, 3, 60),
        (0.08, 0.12, 0.0, 0.08, 3, 60),
        (0.08, 0.15, 0.0, 0.10, 3, 60),
        (0.08, 0.12, 0.0, None, 2, 60),
        (0.08, 0.12, 0.0, None, 3, 80),
        (0.10, 0.15, 0.0, 0.10, 3, 80),
    ]
    if quick:
        hybrids = hybrids[:4]
    for sl, pat, floor, trail, k, h in hybrids:
        tid = f"hy_sl{int(sl*100)}_p{int(pat*100)}"
        tid += f"_t{int(trail*100)}" if trail else "_be"
        tid += f"_ma20x{k}_h{h}"
        rules.append(
            ExitRule(
                id=tid,
                family="hybrid",
                hold_n=h,
                stop=sl,
                ma_period=20,
                ma_below_days=k,
                protect_at=pat,
                protect_floor=floor,
                trail=trail,
            )
        )
    return rules


def _roll_mean(a: np.ndarray, n: int) -> np.ndarray:
    return pd.Series(a).rolling(n, min_periods=n).mean().to_numpy(dtype=float)


def _roll_max(a: np.ndarray, n: int) -> np.ndarray:
    return pd.Series(a).rolling(n, min_periods=n).max().to_numpy(dtype=float)


def precompute_signal_masks(df: pd.DataFrame) -> dict[str, np.ndarray]:
    px = df["px"].to_numpy(dtype=float)
    amt = df["amt"].to_numpy(dtype=float)
    turn = df["turn"].to_numpy(dtype=float)
    mv = df["mv"].to_numpy(dtype=float)
    ma10 = df["ma10"].to_numpy(dtype=float)
    ma20 = df["ma20"].to_numpy(dtype=float)
    ma60 = df["ma60"].to_numpy(dtype=float)
    n = len(px)

    px_ma20 = px / ma20
    ma10_gt = ma10 > ma20
    above20 = px > ma20
    amt5 = _roll_mean(amt, 5)
    amt20 = _roll_mean(amt, 20)
    amt_ratio = amt5 / amt20
    hi60 = _roll_max(px, 60)
    dist60 = (px / hi60 - 1.0) * 100.0

    ret5 = np.full(n, np.nan)
    ret20 = np.full(n, np.nan)
    ret5[5:] = (px[5:] / px[:-5] - 1.0) * 100.0
    ret20[20:] = (px[20:] / px[:-20] - 1.0) * 100.0

    setup = (
        (px_ma20 >= SETUP["px_ma20_lo"])
        & (px_ma20 <= SETUP["px_ma20_hi"])
        & (dist60 >= SETUP["dist60_lo"])
        & (dist60 <= SETUP["dist60_hi"])
        & (ret20 >= SETUP["ret20_lo"])
        & (ret20 <= SETUP["ret20_hi"])
        & np.isfinite(amt_ratio)
        & (amt_ratio <= SETUP["amt_ratio_hi"])
        & (mv >= SETUP["mv_lo"])
        & (mv <= SETUP["mv_hi"])
        & (turn <= SETUP["turn_hi"])
    )

    entry = (
        above20
        & ma10_gt
        & np.isfinite(amt_ratio)
        & (amt_ratio >= ENTRY["amt_ratio_lo"])
        & (amt_ratio <= ENTRY["amt_ratio_hi"])
        & (ret5 >= ENTRY["ret5_lo"])
        & (dist60 <= ENTRY["not_extended_dist60"])
        & (ret20 <= ENTRY["max_ret20_chase"])
        & (px_ma20 <= ENTRY["px_ma20_hi"])
    )

    hot = (ret20 >= RISK["too_hot_ret20"]) | (dist60 > 0)

    lookback = int(ENTRY["setup_lookback"])
    # 近 lookback 日（不含今日，且从 i-2 起）曾 Setup
    setup_recent = np.zeros(n, dtype=bool)
    for lag in range(2, lookback + 1):
        shifted = np.zeros(n, dtype=bool)
        if lag < n:
            shifted[lag:] = setup[: n - lag]
        setup_recent |= shifted

    signal = entry & setup_recent & ~hot & np.isfinite(ma20) & np.isfinite(ma60)

    return {
        "signal": signal,
        "op": df["op"].to_numpy(dtype=float),
        "hi": df["hi"].to_numpy(dtype=float),
        "lo": df["lo"].to_numpy(dtype=float),
        "px": px,
        "ma10": ma10,
        "ma20": ma20,
        "ma60": ma60,
        "dates": df["日期"],
        "score_proxy": (-dist60) + (amt_ratio.clip(0, 2) * 5),  # 粗排序
    }


def _ma_arr(arrs: dict, period: int) -> np.ndarray:
    if period == 10:
        return arrs["ma10"]
    if period == 60:
        return arrs["ma60"]
    return arrs["ma20"]


def simulate_trade(
    arrs: dict,
    signal_i: int,
    rule: ExitRule,
    cost: float,
) -> dict | None:
    op, hi, lo, px = arrs["op"], arrs["hi"], arrs["lo"], arrs["px"]
    dates = arrs["dates"]
    buy_i = signal_i + 1
    if buy_i >= len(px):
        return None
    entry = float(op[buy_i])
    if not np.isfinite(entry) or entry <= 0:
        return None

    prev = float(px[signal_i])
    if prev > 0 and (entry / prev - 1.0) >= 0.095:
        return {
            "skipped": True,
            "reason": "次日高开近涨停，放弃",
            "signal_i": signal_i,
            "buy_i": buy_i,
            "exit_i": buy_i,
        }

    last = min(buy_i + rule.hold_n - 1, len(px) - 1)
    hard_sl = entry * (1.0 - rule.stop) if rule.stop is not None else None
    tp = entry * (1.0 + rule.target) if rule.target is not None else None
    sl = hard_sl
    peak_hi = entry
    armed = False
    below_streak = 0
    broke = False
    days_since_break = 0

    ma = _ma_arr(arrs, rule.ma_period) if rule.ma_period else None

    exit_i = None
    exit_px = None
    reason = None

    for j in range(buy_i, last + 1):
        day_hi = float(hi[j])
        day_lo = float(lo[j])
        day_px = float(px[j])

        if np.isfinite(day_hi) and day_hi > peak_hi:
            peak_hi = day_hi
        peak_ret = peak_hi / entry - 1.0
        if rule.protect_at is not None and peak_ret >= float(rule.protect_at):
            armed = True
            floor_px = entry * (1.0 + float(rule.protect_floor))
            if sl is None or floor_px > sl:
                sl = floor_px
            if rule.trail is not None:
                trail_px = peak_hi * (1.0 - float(rule.trail))
                if sl is None or trail_px > sl:
                    sl = trail_px

        hit_tp = tp is not None and day_hi >= tp
        hit_sl = sl is not None and day_lo <= sl
        if hit_tp and hit_sl:
            exit_i, exit_px, reason = j, sl, "同日止损优先"
            break
        if hit_sl:
            if armed and hard_sl is not None and sl is not None and sl > hard_sl + 1e-12:
                reason = (
                    "跟踪止损"
                    if (rule.trail is not None and sl > entry * (1.0 + rule.protect_floor) + 1e-12)
                    else "保本止损"
                )
            else:
                reason = "止损"
            exit_i, exit_px = j, sl
            break
        if hit_tp:
            exit_i, exit_px, reason = j, tp, "止盈"
            break

        # 均线离场（收盘确认）
        if ma is not None and np.isfinite(day_px) and np.isfinite(float(ma[j])):
            ma_j = float(ma[j])
            if rule.reclaim_days is not None:
                if day_px < ma_j:
                    if not broke:
                        broke = True
                        days_since_break = 1
                    else:
                        days_since_break += 1
                    if days_since_break >= int(rule.reclaim_days):
                        exit_i, exit_px, reason = j, day_px, "破均线未收回"
                        break
                else:
                    broke = False
                    days_since_break = 0
            elif rule.ma_below_days is not None:
                if day_px < ma_j:
                    below_streak += 1
                else:
                    below_streak = 0
                if below_streak >= int(rule.ma_below_days):
                    exit_i, exit_px, reason = j, day_px, f"连破MA{rule.ma_period}"
                    break

    if exit_i is None:
        exit_i = last
        exit_px = float(px[exit_i])
        reason = "到期"

    raw_ret = exit_px / entry - 1.0
    net_ret = raw_ret - cost
    hold_days = int(exit_i - buy_i + 1)
    mfe = float(hi[buy_i : exit_i + 1].max() / entry - 1.0)
    mae = float(lo[buy_i : exit_i + 1].min() / entry - 1.0)

    return {
        "skipped": False,
        "signal_date": str(dates.iloc[signal_i].date()),
        "entry_date": str(dates.iloc[buy_i].date()),
        "exit_date": str(dates.iloc[exit_i].date()),
        "entry": round(entry, 4),
        "exit": round(float(exit_px), 4),
        "hold_days": hold_days,
        "exit_reason": reason,
        "raw_ret%": round(raw_ret * 100.0, 3),
        "net_ret%": round(net_ret * 100.0, 3),
        "mfe%": round(mfe * 100.0, 3),
        "mae%": round(mae * 100.0, 3),
        "buy_i": buy_i,
        "exit_i": exit_i,
        "rule_id": rule.id,
        "armed": armed,
    }


def collect_entries(
    codes_dfs: dict[str, pd.DataFrame],
) -> tuple[list[dict], dict[str, dict]]:
    """同票锁定 HOLD_LOCK 日，得到跨规则共用的入场列表。"""
    entries: list[dict] = []
    arrays: dict[str, dict] = {}
    for code, df in codes_dfs.items():
        start = max(60, len(df) - LOOKBACK)
        end = len(df) - HOLD_LOCK - 2
        if end <= start:
            continue
        arrs = precompute_signal_masks(df)
        arrays[code] = arrs
        sig = arrs["signal"]
        i = start
        while i <= end:
            if not bool(sig[i]):
                i += 1
                continue
            buy_i = i + 1
            entries.append(
                {
                    "code": code,
                    "signal_i": i,
                    "signal_date": str(arrs["dates"].iloc[i].date()),
                    "score": float(arrs["score_proxy"][i])
                    if np.isfinite(arrs["score_proxy"][i])
                    else 0.0,
                }
            )
            i = buy_i + HOLD_LOCK
    return entries, arrays


def run_rules(
    entries: list[dict],
    arrays: dict[str, dict],
    rules: list[ExitRule],
    cost: float,
) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {r.id: [] for r in rules}
    for e in entries:
        code = e["code"]
        arrs = arrays[code]
        i = int(e["signal_i"])
        for rule in rules:
            tr = simulate_trade(arrs, i, rule, cost)
            if tr is None or tr.get("skipped"):
                continue
            tr["code"] = code
            tr["score"] = round(float(e["score"]), 3)
            out[rule.id].append(tr)
    return out


def summarize_rule(trades: list[dict], rule: ExitRule) -> dict:
    if not trades:
        return {
            "rule_id": rule.id,
            "family": rule.family,
            "label": rule.label(),
            "n": 0,
            "winrate": np.nan,
            "mean_net": np.nan,
            "median_net": np.nan,
            "payoff": np.nan,
            "avg_hold": np.nan,
            "mae_p50": np.nan,
            "reasons": {},
            "score": -1e9,
            "desc": rule_desc(rule),
        }
    df = pd.DataFrame(trades)
    r = df["net_ret%"]
    win = r > 0
    winrate = float(win.mean() * 100)
    mean_net = float(r.mean())
    median_net = float(r.median())
    if win.any() and (~win).any():
        payoff = float(r[win].mean() / abs(r[~win].mean()))
    else:
        payoff = float("nan")
    reasons = df["exit_reason"].value_counts().to_dict()
    return {
        "rule_id": rule.id,
        "family": rule.family,
        "label": rule.label(),
        "n": int(len(df)),
        "winrate": round(winrate, 2),
        "mean_net": round(mean_net, 3),
        "median_net": round(median_net, 3),
        "payoff": round(payoff, 3) if np.isfinite(payoff) else None,
        "avg_hold": round(float(df["hold_days"].mean()), 2),
        "mae_p50": round(float(df["mae%"].median()), 3),
        "reasons": {str(k): int(v) for k, v in reasons.items()},
        "score": 0.0,
        "desc": rule_desc(rule),
        "hold_ok": 20 <= float(df["hold_days"].mean()) <= 60,
    }


def rule_desc(rule: ExitRule) -> str:
    bits: list[str] = []
    if rule.stop is not None:
        bits.append(f"硬止损 -{rule.stop*100:.0f}%")
    if rule.target is not None:
        bits.append(f"止盈 +{rule.target*100:.0f}%")
    if rule.protect_at is not None:
        if rule.trail is not None:
            bits.append(
                f"浮盈≥{rule.protect_at*100:.0f}%后跟踪回撤{rule.trail*100:.0f}%"
            )
        else:
            bits.append(f"浮盈≥{rule.protect_at*100:.0f}%抬到保本")
    if rule.ma_period is not None and rule.reclaim_days is not None:
        bits.append(f"破MA{rule.ma_period}后{rule.reclaim_days}日未收回则清")
    elif rule.ma_period is not None and rule.ma_below_days is not None:
        bits.append(f"连续{rule.ma_below_days}日收盘破MA{rule.ma_period}清仓")
    bits.append(f"最多持有{rule.hold_n}个交易日")
    return "；".join(bits)


def score_rules(rows: list[dict]) -> list[dict]:
    eligible = [r for r in rows if r["n"] >= MIN_TRADES and np.isfinite(r["mean_net"])]
    if len(eligible) < 2:
        for r in rows:
            r["score"] = r.get("mean_net") or -1e9
        return sorted(rows, key=lambda x: x["score"], reverse=True)

    means = np.array([r["mean_net"] for r in eligible], dtype=float)
    wins = np.array([r["winrate"] for r in eligible], dtype=float)

    def z(a: np.ndarray) -> np.ndarray:
        s = a.std()
        if s < 1e-12:
            return np.zeros_like(a)
        return (a - a.mean()) / s

    scores = 0.55 * z(means) + 0.45 * z(wins)
    by_id = {r["rule_id"]: float(s) for r, s in zip(eligible, scores)}
    for r in rows:
        r["score"] = round(by_id.get(r["rule_id"], -1e9), 4)

    def sort_key(r: dict):
        hold_pen = 0 if r.get("hold_ok") else 1
        return (-r["score"], hold_pen, -(r["median_net"] or -999), -r["n"])

    return sorted(rows, key=sort_key)


def write_outputs(
    ranked: list[dict],
    bucket: dict[str, list[dict]],
    rules_by_id: dict[str, ExitRule],
    name_map: dict,
    ind_map: dict,
    meta: dict,
) -> dict:
    OUT_TRADES_DIR.mkdir(parents=True, exist_ok=True)
    cmp = pd.DataFrame(
        [
            {
                "rank": i + 1,
                "rule_id": r["rule_id"],
                "family": r["family"],
                "label": r["label"],
                "n": r["n"],
                "winrate": r["winrate"],
                "mean_net": r["mean_net"],
                "median_net": r["median_net"],
                "payoff": r["payoff"],
                "avg_hold": r["avg_hold"],
                "mae_p50": r["mae_p50"],
                "score": r["score"],
                "desc": r["desc"],
            }
            for i, r in enumerate(ranked)
            if r["n"] > 0
        ]
    )
    cmp.to_csv(OUT_COMPARE, index=False, encoding="utf-8-sig")

    pick = next((r for r in ranked if r["n"] >= MIN_TRADES), ranked[0] if ranked else None)
    if pick is None:
        raise SystemExit("无可用规则结果")

    # 推荐规则成交明细
    rid = pick["rule_id"]
    trades = pd.DataFrame(bucket.get(rid, []))
    if not trades.empty:
        trades["name"] = trades["code"].map(lambda c: name_map.get(c, ""))
        trades["industry"] = trades["code"].map(lambda c: ind_map.get(c, ""))
        trades = trades.sort_values("entry_date").reset_index(drop=True)
        drop_cols = [c for c in ("buy_i", "exit_i", "skipped", "armed") if c in trades.columns]
        trades = trades.drop(columns=drop_cols, errors="ignore")
    trades_path = OUT_TRADES_DIR / f"rally_exit_backtest_trades_{rid}.csv"
    trades.to_csv(trades_path, index=False, encoding="utf-8-sig")

    # Top5 也落盘简表
    top5 = [r for r in ranked if r["n"] >= MIN_TRADES][:5]
    lines = [
        "=== 长线 Entry 出场规则扫描 ===",
        f"入场锁定: {HOLD_LOCK}日 | 成本: {meta['cost_bps']:.0f}bp | 最低样本: {MIN_TRADES}",
        f"信号数(入场): {meta['n_entries']} | 股票池: {meta['n_codes']} | 规则数: {meta['n_rules']}",
        f"扫描耗时: {meta['elapsed_s']:.1f}s",
        "",
        "【推荐】",
        f"  {pick['rule_id']}",
        f"  {pick['desc']}",
        f"  n={pick['n']}  胜率={pick['winrate']:.1f}%  期望={pick['mean_net']:.3f}%  "
        f"中位={pick['median_net']:.3f}%  持有={pick['avg_hold']:.1f}日  score={pick['score']:.3f}",
        f"  出场原因: {pick['reasons']}",
        "",
        "【Top5】",
    ]
    for i, r in enumerate(top5, 1):
        lines.append(
            f"{i}. {r['rule_id']}  wr={r['winrate']:.1f}%  E={r['mean_net']:.3f}%  "
            f"med={r['median_net']:.3f}%  hold={r['avg_hold']:.1f}d  n={r['n']}  "
            f"score={r['score']:.3f}"
        )
        lines.append(f"   {r['desc']}")

    lines.append("")
    lines.append("【分家族最佳】")
    for fam in ("hybrid", "fixed", "ma"):
        best = next((r for r in ranked if r["family"] == fam and r["n"] >= MIN_TRADES), None)
        if best:
            lines.append(
                f"  {fam}: {best['rule_id']}  wr={best['winrate']:.1f}%  "
                f"E={best['mean_net']:.3f}%  score={best['score']:.3f}"
            )

    text = "\n".join(lines)
    OUT_SUMMARY.write_text(text, encoding="utf-8")
    print(text, flush=True)
    print(f"\n→ {OUT_COMPARE}\n→ {trades_path}\n→ {OUT_SUMMARY}", flush=True)

    pick_payload = {
        "pick": pick,
        "top5": top5,
        "meta": meta,
        "family_best": {
            fam: next(
                (r for r in ranked if r["family"] == fam and r["n"] >= MIN_TRADES),
                None,
            )
            for fam in ("hybrid", "fixed", "ma")
        },
        "scatter": [
            {
                "rule_id": r["rule_id"],
                "family": r["family"],
                "winrate": r["winrate"],
                "mean_net": r["mean_net"],
                "n": r["n"],
                "avg_hold": r["avg_hold"],
                "score": r["score"],
            }
            for r in ranked
            if r["n"] >= MIN_TRADES
        ],
    }
    # JSON-safe
    def scrub(o):
        if isinstance(o, dict):
            return {k: scrub(v) for k, v in o.items()}
        if isinstance(o, list):
            return [scrub(x) for x in o]
        if isinstance(o, (np.floating, float)):
            if not np.isfinite(o):
                return None
            return float(o)
        if isinstance(o, (np.integer, int)):
            return int(o)
        return o

    OUT_JSON.write_text(json.dumps(scrub(pick_payload), ensure_ascii=False, indent=2), encoding="utf-8")
    return pick_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="长线出场规则扫描")
    parser.add_argument("--quick", action="store_true", help="缩小网格，便于冒烟")
    parser.add_argument("--cost-bps", type=float, default=COST_BPS)
    args = parser.parse_args()

    rules = build_rules(quick=args.quick)
    rules_by_id = {r.id: r for r in rules}
    cost = args.cost_bps / 10000.0
    name_map, ind_map = load_maps()

    import daily_cache

    t0 = time.time()
    daily_cache.preload(force=False, prefer_process=True)
    print(f"[cache] 预载完成 {time.time()-t0:.1f}s", flush=True)

    codes = [
        c
        for c in daily_cache.cached_codes()
        if c.startswith(("600", "601", "603", "605", "000", "001", "002"))
        and "ST" not in name_map.get(c, "").upper()
    ]
    dfs = {c: daily_cache.get(c) for c in codes}
    dfs = {c: df for c, df in dfs.items() if df is not None}

    t1 = time.time()
    entries, arrays = collect_entries(dfs)
    print(f"入场信号 {len(entries)} 笔（{len(dfs)} 只，锁定{HOLD_LOCK}日）", flush=True)
    bucket = run_rules(entries, arrays, rules, cost)
    print(f"规则结算完成 {len(rules)} 套，耗时 {time.time()-t1:.1f}s", flush=True)

    rows = [summarize_rule(bucket[r.id], r) for r in rules]
    ranked = score_rules(rows)
    meta = {
        "n_entries": len(entries),
        "n_codes": len(dfs),
        "n_rules": len(rules),
        "cost_bps": args.cost_bps,
        "hold_lock": HOLD_LOCK,
        "min_trades": MIN_TRADES,
        "elapsed_s": time.time() - t0,
        "quick": args.quick,
    }
    write_outputs(ranked, bucket, rules_by_id, name_map, ind_map, meta)


if __name__ == "__main__":
    main()
