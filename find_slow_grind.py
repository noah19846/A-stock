"""
筛选「连续小幅上涨很久」的慢牛/蚂蚁搬家形态（后复权）。

判定要点：
- 评估最近 40/60/80/100/120 个交易日，取评分最高窗口
- 涨幅主要由小阳（日涨 0~3%）贡献，而非少数大阳
- 上涨日占比高、回撤可控、价格路径较平滑（对数价线性 R²）
- 允许穿插微跌，统计「绵延段」最长长度

输出：data/slow_grind_up.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DAILY = ROOT / "data" / "daily_raw"
LIST = ROOT / "data" / "stock_list.csv"
INDUSTRY = ROOT / "data" / "stock_industry.csv"
OUT = ROOT / "data" / "slow_grind_up.csv"

WINDOWS = (40, 60, 80, 100, 120)
SMALL_LO, SMALL_HI = 0.0, 0.03
TINY_DN = -0.01
MIN_CUM = 0.08
MIN_SMALL_RATIO = 0.32
MIN_UP_RATIO = 0.50
MAX_DD = 0.18
MIN_R2 = 0.45
MIN_SMALL_CONTRIB = 0.50
MIN_GRIND_STREAK = 6


def load_maps() -> tuple[dict[str, str], dict[str, str]]:
    name_map: dict[str, str] = {}
    if LIST.exists():
        nm = pd.read_csv(LIST, dtype=str)
        c = "股票代码" if "股票代码" in nm.columns else nm.columns[0]
        n = "股票名称" if "股票名称" in nm.columns else nm.columns[1]
        nm[c] = nm[c].astype(str).str.zfill(6)
        name_map = dict(zip(nm[c], nm[n].astype(str)))
    ind_map: dict[str, str] = {}
    if INDUSTRY.exists():
        ind = pd.read_csv(INDUSTRY, dtype=str)
        ind["股票代码"] = ind["股票代码"].astype(str).str.zfill(6)
        ind_map = dict(zip(ind["股票代码"], ind["行业板块"].astype(str)))
    return name_map, ind_map


def longest(mask: np.ndarray) -> int:
    best = cur = 0
    for v in mask:
        if v:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def score_window(px: np.ndarray, dates: np.ndarray, amt: np.ndarray) -> dict | None:
    ret = np.diff(px) / px[:-1]
    cum = px[-1] / px[0] - 1.0
    if not np.isfinite(cum) or cum < MIN_CUM:
        return None
    up_ratio = float((ret > 0).mean())
    if up_ratio < MIN_UP_RATIO:
        return None
    peak = np.maximum.accumulate(px)
    dd = float(((px - peak) / peak).min())
    if dd < -MAX_DD:
        return None
    small = (ret > SMALL_LO) & (ret <= SMALL_HI)
    grind = small | ((ret >= TINY_DN) & (ret <= 0))
    small_ratio = float(small.mean())
    if small_ratio < MIN_SMALL_RATIO:
        return None
    pos_sum = float(ret[ret > 0].sum())
    small_sum = float(ret[small].sum())
    contrib = small_sum / pos_sum if pos_sum > 1e-9 else 0.0
    if contrib < MIN_SMALL_CONTRIB:
        return None
    big_up_ratio = float((ret > 0.05).mean())
    if big_up_ratio > 0.10:
        return None
    grind_streak = longest(grind)
    pure_streak = longest(small)
    if grind_streak < MIN_GRIND_STREAK:
        return None
    pos = ret[ret > 0]
    avg_up = float(pos.mean()) if len(pos) else 0.0
    med_up = float(np.median(pos)) if len(pos) else 0.0
    if med_up > 0.025:
        return None
    big_dn = float((ret < -0.03).mean())
    x = np.arange(len(px), dtype=float)
    y = np.log(np.clip(px, 1e-9, None))
    coef = np.polyfit(x, y, 1)
    yhat = np.polyval(coef, x)
    ss_res = ((y - yhat) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    if r2 < MIN_R2 or coef[0] <= 0:
        return None
    daily_slope = float(np.exp(coef[0]) - 1)
    score = (
        grind_streak * 1.2
        + pure_streak * 2.0
        + small_ratio * 50
        + contrib * 40
        + r2 * 40
        + min(cum, 0.35) * 55
        + up_ratio * 12
        - big_up_ratio * 80
        - big_dn * 40
        - max(0.0, cum - 0.50) * 35
        - abs(med_up - 0.008) * 120
        - max(0.0, -dd - 0.10) * 60
    )
    return {
        "区间起点": str(dates[0])[:10],
        "区间终点": str(dates[-1])[:10],
        "窗口交易日": int(len(px)),
        "累计涨幅%": round(cum * 100, 2),
        "上涨日占比%": round(up_ratio * 100, 1),
        "小阳日占比%": round(small_ratio * 100, 1),
        "小阳贡献占比%": round(contrib * 100, 1),
        "纯小阳最长连涨": int(pure_streak),
        "绵延最长段": int(grind_streak),
        "正收益日均涨%": round(avg_up * 100, 2),
        "正收益日中位涨%": round(med_up * 100, 2),
        "窗口最大单日涨%": round(float(ret.max()) * 100, 2),
        "窗口最大回撤%": round(dd * 100, 2),
        "大涨日占比%": round(big_up_ratio * 100, 1),
        "趋势R2": round(r2, 3),
        "日均涨幅%": round(daily_slope * 100, 3),
        "日均成交额亿": round(float(np.nanmean(amt)) / 1e8, 2),
        "评分": round(score, 2),
    }


def run() -> pd.DataFrame:
    name_map, ind_map = load_maps()
    rows: list[dict] = []
    files = sorted(p for p in DAILY.glob("*.csv") if p.name[0].isdigit())
    for path in files:
        code = path.stem.zfill(6)
        if not code.startswith(("600", "601", "603", "605", "000", "001", "002")):
            continue
        name = name_map.get(code, "")
        if "ST" in name.upper():
            continue
        try:
            df = pd.read_csv(path, usecols=["日期", "收盘", "后复权因子", "成交额"])
        except Exception:
            continue
        if len(df) < max(WINDOWS) + 2:
            continue
        df = df.sort_values("日期")
        px_all = (df["收盘"] * df["后复权因子"]).astype(float).values
        dates_all = df["日期"].astype(str).values
        amt_all = df["成交额"].astype(float).values
        best = None
        for w in WINDOWS:
            m = score_window(px_all[-w:], dates_all[-w:], amt_all[-w:])
            if m is None:
                continue
            if best is None or m["评分"] > best["评分"]:
                best = m
        if best is None:
            continue
        rows.append(
            {
                "股票代码": code,
                "股票名称": name,
                "行业板块": ind_map.get(code, ""),
                **best,
            }
        )
    res = pd.DataFrame(rows)
    if res.empty:
        return res
    return res.sort_values(
        ["评分", "小阳贡献占比%", "趋势R2"], ascending=False
    ).reset_index(drop=True)


def main() -> None:
    res = run()
    print(f"命中 {len(res)} 只")
    if res.empty:
        return
    OUT.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"已保存 {OUT}")
    cols = [
        "股票代码",
        "股票名称",
        "行业板块",
        "窗口交易日",
        "累计涨幅%",
        "小阳日占比%",
        "小阳贡献占比%",
        "纯小阳最长连涨",
        "绵延最长段",
        "正收益日中位涨%",
        "窗口最大单日涨%",
        "窗口最大回撤%",
        "趋势R2",
        "评分",
    ]
    print(res[cols].to_string(index=False))


if __name__ == "__main__":
    main()
