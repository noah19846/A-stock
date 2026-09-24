"""Wyckoff 日线阶段研究筛选器：吸筹、震荡、第一次拉伸。

实验策略：接入信号池「威科夫 · 实验」tab，不并入 default 可交易逻辑。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import daily_cache
from analyze_short_burst_features import load_maps

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "wyckoff_signals.csv"
PREFIX = ("600", "601", "603", "605", "000", "001", "002")
STAGE_ORDER = {"第一次拉伸": 0, "吸筹": 1, "震荡": 2}


def _base(df: pd.DataFrame, code: str, name: str, industry: str, asof: str) -> dict | None:
    if df is None or len(df) < 140:
        return None
    px = df.px.to_numpy(float)
    hi = df.hi.to_numpy(float)
    lo = df.lo.to_numpy(float)
    vol = df.vol.to_numpy(float)
    ret = df.ret.to_numpy(float)
    i = len(df) - 1
    if not np.isfinite(px[i]) or px[i] <= 0:
        return None
    hi60, lo60 = np.max(hi[i - 60 : i]), np.min(lo[i - 60 : i])
    hi30, lo30 = np.max(hi[i - 30 : i]), np.min(lo[i - 30 : i])
    hi10, lo10 = np.max(hi[i - 10 : i]), np.min(lo[i - 10 : i])
    vol10 = np.mean(vol[i - 10 : i])
    vol20 = np.mean(vol[i - 30 : i - 10])
    range60 = hi60 / lo60 - 1
    range30 = hi30 / lo30 - 1
    range10 = hi10 / lo10 - 1
    r20 = px[i] / px[i - 20] - 1
    r10 = px[i] / px[i - 10] - 1
    r60 = px[i] / px[i - 60] - 1
    r120 = px[i] / px[i - 120] - 1
    ma60 = np.mean(px[i - 60 : i])
    vratio = vol10 / vol20 if vol20 > 0 else np.nan
    position30 = (px[i] - lo30) / (hi30 - lo30) if hi30 > lo30 else 0.5
    # 下跌日成交量没有持续放大，符合吸筹/震荡中的抛压衰竭特征。
    down = ret[i - 10 : i] < 0
    up = ret[i - 10 : i] > 0
    down_vol = np.mean(vol[i - 10 : i][down]) if down.any() else 0
    up_vol = np.mean(vol[i - 10 : i][up]) if up.any() else np.inf
    down_up_vol = down_vol / up_vol if up_vol > 0 else np.nan
    prev_hi30 = np.max(hi[i - 30 : i])
    breakout = px[i] / prev_hi30 - 1
    close_pos = (px[i] - lo[i]) / (hi[i] - lo[i]) if hi[i] > lo[i] else 0.5
    # 更长周期的下降趋势已经结束：不再持续创新低，且价格回到60日均线附近。
    trend_ok = (
        r60 >= -0.15
        and r120 >= -0.25
        and px[i] >= ma60 * 0.95
        and np.min(lo[i - 10 : i]) >= np.min(lo[i - 40 : i - 10]) * 0.97
    )
    # Spring：先刺破旧箱体下沿，随后收回；Test：收回后缩量回踩但不再破位。
    spring_box_lo = np.min(lo[i - 40 : i - 10])
    spring_idx = [k for k in range(i - 10, i) if lo[k] < spring_box_lo * 0.98 and px[k] >= spring_box_lo]
    spring = bool(spring_idx)
    spring_k = spring_idx[-1] if spring else None
    test = bool(
        spring
        and i - spring_k >= 2
        and px[i] <= spring_box_lo * 1.06
        and np.mean(vol[i - 3 : i]) <= vol[spring_k] * 0.85
        and lo[i] >= spring_box_lo * 0.98
    )
    # 先判定突破，且用独立评分确认第一次拉伸，避免普通放量反弹混入。
    markup_score = 0.0
    markup_score += min(35.0, max(0.0, breakout) * 700)
    markup_score += min(25.0, max(0.0, vratio - 1.0) * 35)
    markup_score += 20.0 if close_pos >= 0.65 else 0.0
    markup_score += 20.0 if trend_ok else 0.0
    if range30 <= 0.25 and breakout >= 0.01 and 0.02 <= ret[i] < 0.095 and vratio >= 1.20 and markup_score >= 70:
        stage = "第一次拉伸"
        score = markup_score
    elif trend_ok and range60 <= 0.35 and range30 <= 0.22 and vratio <= 0.85 and position30 <= 0.65 and down_up_vol <= 1.0 and (spring or test):
        stage = "吸筹"
        score = 65 + (12 if spring else 0) + (12 if test else 0) + max(0, 1.0 - down_up_vol) * 10
    elif trend_ok and range30 <= 0.15 and range10 <= 0.10 and abs(r20) <= 0.10 and vratio <= 1.00:
        stage = "震荡"
        score = 70 + max(0, 0.15 - range30) * 100 + max(0, 1.0 - vratio) * 10
    else:
        return None
    reasons = (
        f"{stage}：30日振幅{range30 * 100:.1f}%、量比{vratio:.2f}、"
        f"箱体位置{position30 * 100:.0f}%、Spring={spring}、Test={test}"
    )
    return {
        "code": code,
        "name": name,
        "industry": industry,
        "asof": asof,
        "stage": stage,
        "can_buy": False,
        "score": round(float(score), 1),
        "when": f"实验观察：{reasons}",
        "收盘": round(float(df["收盘"].iloc[i]), 2),
        "今日涨跌%": round(ret[i] * 100, 2),
        "60日振幅%": round(range60 * 100, 2),
        "30日振幅%": round(range30 * 100, 2),
        "10日振幅%": round(range10 * 100, 2),
        "20日涨跌%": round(r20 * 100, 2),
        "60日涨跌%": round(r60 * 100, 2),
        "120日涨跌%": round(r120 * 100, 2),
        "趋势结束": trend_ok,
        "10日量/前20日量": round(vratio, 2),
        "30日箱体位置%": round(position30 * 100, 1),
        "下跌日量/上涨日量": round(down_up_vol, 2),
        "突破箱顶%": round(breakout * 100, 2),
        "第一次拉伸分": round(markup_score, 1),
        "Spring": spring,
        "Test": test,
        "Spring箱底": round(float(spring_box_lo), 2),
        "止损参考": round(lo30 * 0.98, 2),
        "reasons": reasons,
        "strategy_ids": "wyckoff",
        "strategy_tags": f"威科夫-{stage}",
    }


def scan(asof: str | None = None) -> pd.DataFrame:
    daily_cache.preload()
    names, industries = load_maps()
    rows = []
    for code in daily_cache.cached_codes():
        if not code.startswith(PREFIX):
            continue
        name = names.get(code, "")
        if "ST" in name.upper() or name.startswith("*"):
            continue
        df = daily_cache.get(code, asof=asof)
        if df is None or df.empty:
            continue
        row = _base(df, code, name, industries.get(code, ""), str(df["日期"].iloc[-1].date()))
        if row:
            rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out["_o"] = out["stage"].map(STAGE_ORDER).fillna(9)
        out = out.sort_values(["_o", "score"], ascending=[True, False]).drop(
            columns=["_o"]
        ).reset_index(drop=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--asof", default=None)
    args = ap.parse_args()
    out = scan(args.asof)
    out.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"威科夫候选 {len(out)} → {OUT}")
    if not out.empty:
        print(out[["code", "name", "stage", "score", "今日涨跌%", "30日振幅%", "10日量/前20日量", "止损参考"]].to_string(index=False))
