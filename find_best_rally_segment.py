"""
找近一段时间内「历史上最优上涨子区间」：
允许大阳/涨停，要求子区间累计涨幅高、最大回撤可控、趋势向上。

不是「今天仍在涨」，而是「曾经出现过类似天创时尚 4/7~6/16 那种主升浪」。

输出：data/best_rally_segment.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DAILY = ROOT / "data" / "daily_raw"
LIST = ROOT / "data" / "stock_list.csv"
INDUSTRY = ROOT / "data" / "stock_industry.csv"
OUT = ROOT / "data" / "best_rally_segment.csv"

LOOKBACK = 180          # 只在最近 N 个交易日里找子区间
MIN_LEN, MAX_LEN = 30, 80
STEP = 2                # 起点步长（加速）
LEN_STEP = 5            # 窗口长度步长

MIN_CUM = 0.30          # 子区间至少 +30%
MAX_DD = 0.12           # 子区间最大回撤 ≤12%
MIN_UP_RATIO = 0.52
MIN_R2 = 0.55
MIN_STREAK = 5
TINY_DN = -0.015
MAX_BIG_DN_RATIO = 0.08


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


def eval_segment(px: np.ndarray, dates: np.ndarray, amt: np.ndarray) -> dict | None:
    if len(px) < MIN_LEN:
        return None
    ret = np.diff(px) / px[:-1]
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

    big_dn_ratio = float((ret < -0.04).mean())
    if big_dn_ratio > MAX_BIG_DN_RATIO:
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

    max_up = float(ret.max())
    limit_up_days = int((ret >= 0.095).sum())
    big_up_days = int((ret > 0.05).sum())
    daily_slope = float(np.exp(coef[0]) - 1)
    # 终点是否接近区间高点（主升浪未深度回撤结束）
    end_from_high = float(px[-1] / px.max() - 1.0)

    score = (
        min(cum, 2.0) * 40
        + (MAX_DD + dd) * 180
        + r2 * 45
        + up_ratio * 20
        + up_streak * 1.5
        + grind_streak * 0.8
        + min(limit_up_days, 8) * 1.2
        + max(0.0, end_from_high + 0.05) * 30  # 收在高位附近加分
        - big_dn_ratio * 60
        - max(0.0, -end_from_high - 0.08) * 40  # 段末已大回撤减分
    )

    return {
        "子区间起点": str(dates[0])[:10],
        "子区间终点": str(dates[-1])[:10],
        "子区间交易日": int(len(px)),
        "累计涨幅%": round(cum * 100, 2),
        "上涨日占比%": round(up_ratio * 100, 1),
        "上涨最长连涨": int(up_streak),
        "绵延最长段": int(grind_streak),
        "窗口最大单日涨%": round(max_up * 100, 2),
        "窗口最大回撤%": round(dd * 100, 2),
        "大涨日数>5%": big_up_days,
        "近涨停日数": limit_up_days,
        "大阴日占比%": round(big_dn_ratio * 100, 1),
        "趋势R2": round(r2, 3),
        "日均涨幅%": round(daily_slope * 100, 3),
        "段末距最高%": round(end_from_high * 100, 2),
        "日均成交额亿": round(float(np.nanmean(amt)) / 1e8, 2),
        "评分": round(score, 2),
    }


def best_segment(px: np.ndarray, dates: np.ndarray, amt: np.ndarray) -> dict | None:
    """在序列上扫描所有子区间，返回评分最高且通过阈值的一段。"""
    n = len(px)
    best: dict | None = None
    for length in range(MIN_LEN, min(MAX_LEN, n) + 1, LEN_STEP):
        for i in range(0, n - length + 1, STEP):
            j = i + length
            m = eval_segment(px[i:j], dates[i:j], amt[i:j])
            if m is None:
                continue
            if best is None or m["评分"] > best["评分"]:
                best = m
    return best


def run() -> pd.DataFrame:
    name_map, ind_map = load_maps()
    rows: list[dict] = []
    files = sorted(p for p in DAILY.glob("*.csv") if p.name[0].isdigit())
    total = len(files)
    for k, path in enumerate(files, 1):
        if k % 500 == 0:
            print(f"  …{k}/{total}")
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
        if len(df) < MIN_LEN + 5:
            continue
        df = df.sort_values("日期")
        df = df.tail(LOOKBACK)
        px = (df["收盘"] * df["后复权因子"]).astype(float).values
        dates = df["日期"].astype(str).values
        amt = df["成交额"].astype(float).values
        best = best_segment(px, dates, amt)
        if best is None:
            continue
        # 距今多少交易日（用全文件最后一天）
        last = str(df["日期"].iloc[-1])[:10]
        rows.append(
            {
                "股票代码": code,
                "股票名称": name,
                "行业板块": ind_map.get(code, ""),
                "数据截止": last,
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
    print(
        f"扫描近 {LOOKBACK} 日，子区间长度 {MIN_LEN}~{MAX_LEN}，"
        f"累计≥{MIN_CUM:.0%} 且回撤≤{MAX_DD:.0%} …"
    )
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
        "子区间起点",
        "子区间终点",
        "子区间交易日",
        "累计涨幅%",
        "窗口最大回撤%",
        "上涨日占比%",
        "近涨停日数",
        "趋势R2",
        "段末距最高%",
        "评分",
    ]
    print(res[cols].head(40).to_string(index=False))
    # 天创时尚是否命中
    hit = res[res["股票代码"] == "603608"]
    if len(hit):
        print("\n天创时尚 603608 已命中：")
        print(hit[cols].to_string(index=False))
    else:
        print("\n注意：天创时尚 603608 仍未命中，可再放宽参数")


if __name__ == "__main__":
    main()
