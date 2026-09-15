"""横盘首板后的放量洗盘、缩量回踩策略。

这是原「无量首板」tab 的替代策略：只输出当天刚形成回踩买点的股票，
并把首板前横盘箱体下沿作为止损参考，不把历史信号混入当日列表。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import daily_cache
from analyze_short_burst_features import load_maps

ROOT = Path(__file__).resolve().parent
OUT_SCAN = ROOT / "data" / "box_retest_signals.csv"

LIMIT_RET = 0.095
SIDEWAYS_WINDOW = 30
SIDEWAYS_RANGE_MAX = 0.25
FIRST_BOARD_LOOKBACK = 20
WASH_RET_MAX = -0.01
WASH_VOL_MIN = 0.80
WASH_VOL_MAX = 1.50
SHRINK_VOL_MAX = 0.85
SUPPORT_BREAK = 0.02
BUY_GAP_MAX = 0.03
MAINBOARD_PREFIX = ("600", "601", "603", "605", "000", "001", "002")


def _find_latest(df: pd.DataFrame) -> dict | None:
    if df is None or len(df) < SIDEWAYS_WINDOW + 8:
        return None
    px = df["px"].to_numpy(float)
    hi = df["hi"].to_numpy(float)
    lo = df["lo"].to_numpy(float)
    vol = df["vol"].to_numpy(float)
    ret = df["ret"].to_numpy(float)
    dates = pd.to_datetime(df["日期"])
    latest = len(df) - 1
    # 买点必须落在最新交易日，避免把过去历史信号继续展示为今日信号。
    for i in range(max(SIDEWAYS_WINDOW, FIRST_BOARD_LOOKBACK), latest - 1):
        if not np.isfinite(ret[i]) or ret[i] < LIMIT_RET:
            continue
        sideways = px[i - SIDEWAYS_WINDOW : i]
        if np.min(sideways) <= 0:
            continue
        sideways_range = np.max(sideways) / np.min(sideways) - 1
        prior_limit = np.nanmax(ret[i - FIRST_BOARD_LOOKBACK : i]) >= LIMIT_RET
        if sideways_range > SIDEWAYS_RANGE_MAX or prior_limit:
            continue
        wash = i + 1
        if px[wash] >= px[i] or ret[wash] > WASH_RET_MAX or vol[i] <= 0:
            continue
        wash_ratio = vol[wash] / vol[i]
        if not (WASH_VOL_MIN <= wash_ratio <= WASH_VOL_MAX):
            continue
        support = lo[wash]
        declines = 0
        for j in range(i + 2, min(i + 6, len(df))):
            declines += int(px[j] < px[j - 1])
            recent_vol = vol[i + 2 : j + 1]
            shrink = len(recent_vol) >= 2 and np.mean(recent_vol) / vol[wash] <= SHRINK_VOL_MAX
            not_broken = lo[j] >= support * (1 - SUPPORT_BREAK)
            near_support = support <= px[j] <= support * (1 + BUY_GAP_MAX)
            if declines >= 2 and shrink and not_broken and near_support:
                if j != latest:
                    continue
                box_bottom = float(np.min(lo[i - SIDEWAYS_WINDOW : i]))
                box_top = float(np.max(hi[i - SIDEWAYS_WINDOW : i]))
                return {
                    "limit_i": i,
                    "buy_i": j,
                    "limit_date": str(dates.iloc[i].date()),
                    "wash_date": str(dates.iloc[wash].date()),
                    "buy_date": str(dates.iloc[j].date()),
                    "limit_ret": float(ret[i]),
                    "wash_ret": float(ret[wash]),
                    "wash_vol_ratio": float(wash_ratio),
                    "buy_price": float(px[j]),
                    "support": float(support),
                    "box_bottom": box_bottom,
                    "box_top": box_top,
                    "sideways_range": sideways_range,
                }
            if lo[j] < support * (1 - SUPPORT_BREAK):
                break
    return None


def _row(code: str, name: str, industry: str, df: pd.DataFrame, s: dict) -> dict:
    buy = s["buy_price"]
    box_bottom = s["box_bottom"]
    stop = box_bottom * (1 - SUPPORT_BREAK)
    score = 100 - s["sideways_range"] * 100 - abs(s["wash_vol_ratio"] - 1) * 20
    return {
        "code": code,
        "name": name,
        "industry": industry,
        "asof": s["buy_date"],
        "stage": "可买入",
        "can_buy": True,
        "score": round(max(0.0, score), 1),
        "when": f"尾盘买入 {buy:.2f}；收盘跌破箱体止损位 {stop:.2f} 止损",
        "reasons": (
            f"横盘{SIDEWAYS_WINDOW}日振幅 {s['sideways_range'] * 100:.1f}%；"
            f"首板 {s['limit_date']}；洗盘量比 {s['wash_vol_ratio']:.2f}；"
            f"回踩支撑 {s['support']:.2f}"
        ),
        "买入价": round(buy, 2),
        "首板日期": s["limit_date"],
        "洗盘日期": s["wash_date"],
        "横盘振幅%": round(s["sideways_range"] * 100, 2),
        "洗盘涨跌%": round(s["wash_ret"] * 100, 2),
        "洗盘量比": round(s["wash_vol_ratio"], 2),
        "回踩支撑": round(s["support"], 2),
        "箱体底": round(box_bottom, 2),
        "箱体顶": round(s["box_top"], 2),
        "箱体止损位": round(stop, 2),
        "止损参考": round(stop, 2),
        "strategy_ids": "box_retest",
        "strategy_tags": "涨停箱体回踩",
    }


def scan(asof: str | None = None) -> pd.DataFrame:
    daily_cache.preload()
    names, industries = load_maps()
    rows = []
    for code in daily_cache.cached_codes():
        if not code.startswith(MAINBOARD_PREFIX):
            continue
        name = names.get(code, "")
        if "ST" in name.upper() or name.startswith("*"):
            continue
        df = daily_cache.get(code, asof=asof)
        if df is None or df.empty:
            continue
        signal = _find_latest(df)
        if signal is not None:
            rows.append(_row(code, name, industries.get(code, ""), df, signal))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="横盘首板后的箱体回踩")
    ap.add_argument("--scan", action="store_true")
    ap.add_argument("--asof", default=None)
    args = ap.parse_args()
    out = scan(args.asof)
    OUT_SCAN.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_SCAN, index=False, encoding="utf-8-sig")
    print(f"涨停箱体回踩 可买入 {len(out)} → {OUT_SCAN}")
    if not out.empty:
        print(out[["code", "name", "买入价", "箱体底", "箱体止损位", "洗盘量比"]].to_string(index=False))
