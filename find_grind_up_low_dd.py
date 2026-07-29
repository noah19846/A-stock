"""
筛选「持续上涨、回撤不大」：允许大阳甚至涨停，不要求小阳。

核心：窗口内累计上涨明显、最大回撤可控、趋势向上且较平滑。
输出：data/grind_up_low_dd.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DAILY = ROOT / "data" / "daily_raw"
LIST = ROOT / "data" / "stock_list.csv"
INDUSTRY = ROOT / "data" / "stock_industry.csv"
OUT = ROOT / "data" / "grind_up_low_dd.csv"

WINDOWS = (40, 60, 80, 100, 120)
MIN_CUM = 0.10          # 窗口累计至少 +10%
MAX_DD = 0.12           # 窗口最大回撤不超过 12%
MIN_UP_RATIO = 0.48     # 上涨日占比
MIN_R2 = 0.45           # 对数价线性趋势
MIN_STREAK = 5          # 上涨日（ret>0）最长连涨
TINY_DN = -0.015        # 绵延段：涨 或 微跌


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
    if len(ret) < 10:
        return None
    cum = float(px[-1] / px[0] - 1.0)
    if not np.isfinite(cum) or cum < MIN_CUM:
        return None

    peak = np.maximum.accumulate(px)
    dd = float(((px - peak) / peak).min())
    if dd < -MAX_DD:
        return None

    up = ret > 0
    up_ratio = float(up.mean())
    if up_ratio < MIN_UP_RATIO:
        return None

    up_streak = longest(up)
    grind = up | ((ret >= TINY_DN) & (ret <= 0))
    grind_streak = longest(grind)
    if up_streak < MIN_STREAK and grind_streak < MIN_STREAK + 2:
        return None

    # 大阴不宜多（<-4%）
    big_dn_ratio = float((ret < -0.04).mean())
    if big_dn_ratio > 0.08:
        return None

    x = np.arange(len(px), dtype=float)
    y = np.log(np.clip(px, 1e-9, None))
    coef = np.polyfit(x, y, 1)
    if coef[0] <= 0:
        return None
    yhat = np.polyval(coef, x)
    ss_res = ((y - yhat) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    if r2 < MIN_R2:
        return None

    pos = ret[ret > 0]
    avg_up = float(pos.mean()) if len(pos) else 0.0
    med_up = float(np.median(pos)) if len(pos) else 0.0
    max_up = float(ret.max())
    limit_up_days = int((ret >= 0.095).sum())  # 约涨停
    big_up_days = int((ret > 0.05).sum())
    daily_slope = float(np.exp(coef[0]) - 1)

    # 回撤越小、涨得越久/越稳越好；允许涨停加一点分
    score = (
        min(cum, 0.60) * 80
        + (MAX_DD + dd) * 200          # dd 负值，回撤越小分越高
        + r2 * 50
        + up_ratio * 25
        + up_streak * 2.0
        + grind_streak * 1.0
        + min(limit_up_days, 5) * 1.5
        - big_dn_ratio * 80
        - max(0.0, -dd - 0.06) * 100
    )

    return {
        "区间起点": str(dates[0])[:10],
        "区间终点": str(dates[-1])[:10],
        "窗口交易日": int(len(px)),
        "累计涨幅%": round(cum * 100, 2),
        "上涨日占比%": round(up_ratio * 100, 1),
        "上涨最长连涨": int(up_streak),
        "绵延最长段": int(grind_streak),
        "正收益日均涨%": round(avg_up * 100, 2),
        "正收益日中位涨%": round(med_up * 100, 2),
        "窗口最大单日涨%": round(max_up * 100, 2),
        "窗口最大回撤%": round(dd * 100, 2),
        "大涨日数>5%": big_up_days,
        "近涨停日数": limit_up_days,
        "大阴日占比%": round(big_dn_ratio * 100, 1),
        "趋势R2": round(r2, 3),
        "日均涨幅%": round(daily_slope * 100, 3),
        "日均成交额亿": round(float(np.nanmean(amt)) / 1e8, 2),
        "评分": round(score, 2),
    }


def run() -> pd.DataFrame:
    name_map, ind_map = load_maps()
    rows: list[dict] = []
    for path in sorted(p for p in DAILY.glob("*.csv") if p.name[0].isdigit()):
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
        ["评分", "累计涨幅%", "趋势R2"], ascending=False
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
        "上涨日占比%",
        "上涨最长连涨",
        "绵延最长段",
        "窗口最大单日涨%",
        "近涨停日数",
        "窗口最大回撤%",
        "趋势R2",
        "日均成交额亿",
        "评分",
    ]
    print(res[cols].head(40).to_string(index=False))


if __name__ == "__main__":
    main()
