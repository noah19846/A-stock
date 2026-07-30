"""
更严入场网格：固定出场 +15%/-3%/hold8，搜索 top_k × soft/hard × 额外过滤。

输出：
  data/strict_entry_grid.csv
  data/strict_entry_grid_top.txt
"""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_short_burst_features import load_maps, score_row
from short_burst_backtest import (
    LOOKBACK,
    feat_at_table,
    is_signal,
    load_bands,
    precompute_feature_table,
    simulate_trade_arr,
)

ROOT = Path(__file__).resolve().parent
OUT_CSV = ROOT / "data" / "strict_entry_grid.csv"
OUT_TXT = ROOT / "data" / "strict_entry_grid_top.txt"
BASE_BANDS = ROOT / "data" / "short_burst_feature_bands_top3.json"

HOLD = 8
TARGET = 0.15
STOP = 0.03
COST = 0.0015
HARD_W = 1.2

# 额外硬过滤（在打分之后、TopK 之前）
EXTRA = {
    "none": {},
    "dist4": {"距20日高点%_hi": -4.0},
    "dist5": {"距20日高点%_hi": -5.0},
    "ma5_100": {"收盘/MA5_hi": 1.00},
    "ma5_995": {"收盘/MA5_hi": 0.995},
    "ret1_0": {"前1日涨幅%_hi": 0.0},
    "amt15": {"额能比5_20_hi": 1.5},
    "cold": {"距20日高点%_hi": -4.0, "收盘/MA5_hi": 1.00, "前1日涨幅%_hi": 0.0},
}


def pass_extra(feat: dict, extra: dict) -> bool:
    if not extra:
        return True
    if "距20日高点%_hi" in extra and feat["距20日高点%"] > extra["距20日高点%_hi"]:
        return False
    if "收盘/MA5_hi" in extra and feat["收盘/MA5"] > extra["收盘/MA5_hi"]:
        return False
    if "前1日涨幅%_hi" in extra and feat.get("前1日涨幅%", 0) > extra["前1日涨幅%_hi"]:
        return False
    if "额能比5_20_hi" in extra and feat["额能比5_20"] > extra["额能比5_20_hi"]:
        return False
    return True


def collect_signals(dfs: dict[str, pd.DataFrame], bands: dict) -> tuple[pd.DataFrame, dict]:
    """以最松门槛 soft>=0.8 hard>=0.95 收集候选，附带特征供事后过滤。"""
    loose = copy.deepcopy(bands)
    loose["signal_soft"] = 0.80
    loose["signal_hard"] = 0.95

    rows: list[dict] = []
    arrays: dict[str, dict] = {}
    for code, df in dfs.items():
        start = max(60, len(df) - LOOKBACK)
        end = len(df) - HOLD - 2
        if end <= start:
            continue
        table = precompute_feature_table(df)
        op = df["op"].to_numpy(dtype=float)
        hi = df["hi"].to_numpy(dtype=float)
        lo = df["lo"].to_numpy(dtype=float)
        px = df["px"].to_numpy(dtype=float)
        ma5 = df["ma5"].to_numpy(dtype=float)
        ma10 = df["ma10"].to_numpy(dtype=float)
        ma20 = df["ma20"].to_numpy(dtype=float)
        dates = df["日期"]
        arrays[code] = {"op": op, "hi": hi, "lo": lo, "px": px, "dates": dates}

        for i in range(start, end + 1):
            feat = feat_at_table(table, i, float(ma5[i]), float(ma10[i]), float(ma20[i]))
            if feat is None:
                continue
            soft, hard, _ = score_row(feat, bands)
            if not is_signal(feat, soft, hard, loose):
                continue
            rows.append(
                {
                    "code": code,
                    "i": i,
                    "signal_date": str(dates.iloc[i].date()),
                    "soft": float(soft),
                    "hard": float(hard),
                    "rank": float(hard) * HARD_W + float(soft),
                    "距20日高点%": float(feat["距20日高点%"]),
                    "收盘/MA5": float(feat["收盘/MA5"]),
                    "前1日涨幅%": float(feat.get("前1日涨幅%", np.nan)),
                    "额能比5_20": float(feat["额能比5_20"]),
                }
            )
    return pd.DataFrame(rows), arrays


def run_config(
    sigs: pd.DataFrame,
    arrays: dict,
    soft_min: float,
    hard_min: float,
    top_k: int,
    extra: dict,
) -> dict:
    if sigs.empty:
        return _empty_stats(soft_min, hard_min, top_k, extra)
    m = (sigs["soft"] >= soft_min) & (sigs["hard"] >= hard_min)
    sub = sigs.loc[m].copy()
    if sub.empty:
        return _empty_stats(soft_min, hard_min, top_k, extra)

    if extra:
        mask = [
            pass_extra(
                {
                    "距20日高点%": r["距20日高点%"],
                    "收盘/MA5": r["收盘/MA5"],
                    "前1日涨幅%": r["前1日涨幅%"],
                    "额能比5_20": r["额能比5_20"],
                },
                extra,
            )
            for _, r in sub.iterrows()
        ]
        sub = sub.iloc[[i for i, k in enumerate(mask) if k]]
    if sub.empty:
        return _empty_stats(soft_min, hard_min, top_k, extra)

    picked = (
        sub.sort_values(["signal_date", "rank"], ascending=[True, False])
        .groupby("signal_date", group_keys=False)
        .head(top_k)
    )

    trades = []
    next_free = {c: 0 for c in arrays}
    for _, row in picked.iterrows():
        code = row["code"]
        i = int(row["i"])
        if code not in arrays or i < next_free[code]:
            continue
        arr = arrays[code]
        tr = simulate_trade_arr(
            arr["op"],
            arr["hi"],
            arr["lo"],
            arr["px"],
            arr["dates"],
            i,
            "open",
            HOLD,
            TARGET,
            STOP,
            COST,
        )
        if tr is None or tr.get("skipped"):
            if tr is not None:
                next_free[code] = max(next_free[code], int(tr["exit_i"]) + 1)
            continue
        trades.append(tr)
        next_free[code] = int(tr["exit_i"]) + 1

    return stats_from_trades(trades, soft_min, hard_min, top_k, extra)


def _empty_stats(soft_min, hard_min, top_k, extra) -> dict:
    return {
        "soft": soft_min,
        "hard": hard_min,
        "top_k": top_k,
        "extra": _extra_name(extra),
        "n": 0,
        "exp%": np.nan,
        "win%": np.nan,
        "med%": np.nan,
        "mfe5%": np.nan,
        "mfe_med": np.nan,
        "sl%": np.nan,
        "avg_win": np.nan,
        "avg_loss": np.nan,
    }


def _extra_name(extra: dict) -> str:
    for k, v in EXTRA.items():
        if v == extra:
            return k
    return "custom"


def stats_from_trades(trades: list[dict], soft_min, hard_min, top_k, extra) -> dict:
    if not trades:
        return _empty_stats(soft_min, hard_min, top_k, extra)
    t = pd.DataFrame(trades)
    r = t["net_ret%"]
    win = r > 0
    return {
        "soft": soft_min,
        "hard": hard_min,
        "top_k": top_k,
        "extra": _extra_name(extra),
        "n": int(len(t)),
        "exp%": round(float(r.mean()), 3),
        "win%": round(float(win.mean() * 100), 1),
        "med%": round(float(r.median()), 3),
        "mfe5%": round(float((t["mfe%"] >= 5).mean() * 100), 1),
        "mfe_med": round(float(t["mfe%"].median()), 2),
        "sl%": round(float((t["exit_reason"] == "止损").mean() * 100), 1),
        "avg_win": round(float(r[win].mean()), 2) if win.any() else np.nan,
        "avg_loss": round(float(r[~win].mean()), 2) if (~win).any() else np.nan,
    }


def main() -> None:
    import daily_cache

    bands = load_bands(BASE_BANDS)
    name_map, _ = load_maps()

    t0 = time.time()
    daily_cache.preload(force=False, prefer_process=True)
    print(f"[cache] {time.time() - t0:.1f}s", flush=True)

    codes = [
        c
        for c in daily_cache.cached_codes()
        if c.startswith(("600", "601", "603", "605", "000", "001", "002"))
        and "ST" not in name_map.get(c, "").upper()
    ]
    dfs = {c: daily_cache.get(c) for c in codes}
    dfs = {c: df for c, df in dfs.items() if df is not None}
    print(f"universe {len(dfs)}", flush=True)

    t1 = time.time()
    sigs, arrays = collect_signals(dfs, bands)
    print(f"loose signals {len(sigs)} in {time.time() - t1:.1f}s", flush=True)

    softs = [0.80, 0.85, 0.90, 0.95]
    hards = [0.95, 1.0]
    topks = [1, 2, 3]

    rows = []
    ncfg = len(softs) * len(hards) * len(topks) * len(EXTRA)
    done = 0
    t2 = time.time()
    for soft in softs:
        for hard in hards:
            for top_k in topks:
                for ename, extra in EXTRA.items():
                    st = run_config(sigs, arrays, soft, hard, top_k, extra)
                    rows.append(st)
                    done += 1
                    if done % 40 == 0:
                        print(f"  grid {done}/{ncfg} …", flush=True)

    grid = pd.DataFrame(rows)
    grid.to_csv(OUT_CSV, index=False)
    print(f"grid done {time.time() - t2:.1f}s → {OUT_CSV}", flush=True)

    # 筛选与排序
    ok = grid[grid["n"] >= 80].copy()
    ok = ok.sort_values(["exp%", "mfe5%", "n"], ascending=[False, False, False])

    baseline = grid[
        (grid["soft"] == 0.8)
        & (grid["hard"] == 0.95)
        & (grid["top_k"] == 3)
        & (grid["extra"] == "none")
    ]
    lines = [
        "=== 更严入场网格（出场固定 +15/-3/hold8）===",
        f"loose候选信号: {len(sigs)}",
        f"配置数: {len(grid)}  其中 n>=80: {len(ok)}",
        "",
        "基线 Top3 soft0.8/hard0.95/extra=none:",
        baseline.to_string(index=False) if len(baseline) else "(missing)",
        "",
        "Top15 by exp% (n>=80):",
        ok.head(15).to_string(index=False),
        "",
        "Top1 备选（top_k=1, n>=80）:",
        ok[ok["top_k"] == 1].head(8).to_string(index=False)
        if (ok["top_k"] == 1).any()
        else "(none)",
        "",
        "期望>=0.5 且 n>=80:",
        ok[ok["exp%"] >= 0.5].head(12).to_string(index=False)
        if (ok["exp%"] >= 0.5).any()
        else "(none)",
    ]
    text = "\n".join(lines)
    OUT_TXT.write_text(text, encoding="utf-8")
    print(text, flush=True)
    print(f"→ {OUT_TXT}", flush=True)


if __name__ == "__main__":
    main()
