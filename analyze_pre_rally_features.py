"""
从「历史最优上涨子区间」的起点往前挖涨前共性，
再扫描当下接近这些共性的股票，供观察/埋伏。

正样本：best_rally_segment.csv 中质量较好的主升浪
对照：同股票随机历史日（非主升前）、以及全市场最新截面

输出：
  data/pre_rally_feature_compare.csv
  data/pre_rally_watch.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DAILY = ROOT / "data" / "daily_raw"
LIST = ROOT / "data" / "stock_list.csv"
INDUSTRY = ROOT / "data" / "stock_industry.csv"
RALLY_CSV = ROOT / "data" / "best_rally_segment.csv"
OUT_CMP = ROOT / "data" / "pre_rally_feature_compare.csv"
OUT_WATCH = ROOT / "data" / "pre_rally_watch.csv"

# 正样本质量门槛（更接近天创那种主升）
Q_CUM = 50.0
Q_DD = -10.0
Q_R2 = 0.70
PRE_WIN = 40  # 涨前观察窗口（交易日）


FEATURE_COLS = [
    "前20日涨幅%",
    "前40日涨幅%",
    "前60日涨幅%",
    "前40日波动%",
    "前40日最大回撤%",
    "距60日高点%",
    "距120日高点%",
    "收盘/MA20",
    "MA20/MA60",
    "量能比5_20",
    "额能比5_20",
    "换手率%",
    "前40日上涨占比%",
    "前40日振幅中位%",
    "安静日占比%",
    "距上次大涨>5%天数",
    "流通市值亿",
]


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


def load_daily(code: str) -> pd.DataFrame | None:
    path = DAILY / f"{code}.csv"
    if not path.exists():
        return None
    try:
        df = pd.read_csv(
            path,
            usecols=["日期", "收盘", "最高", "最低", "成交量", "成交额", "流通股本", "换手率", "后复权因子"],
        )
    except Exception:
        return None
    if len(df) < PRE_WIN + 80:
        return None
    df = df.sort_values("日期").reset_index(drop=True)
    df["日期"] = pd.to_datetime(df["日期"])
    f = df["后复权因子"].astype(float)
    df["px"] = df["收盘"].astype(float) * f
    df["hi"] = df["最高"].astype(float) * f
    df["lo"] = df["最低"].astype(float) * f
    df["vol"] = df["成交量"].astype(float)
    df["amt"] = df["成交额"].astype(float)
    df["float_shares"] = df["流通股本"].astype(float)
    df["turn"] = df["换手率"].astype(float) * 100.0  # -> %
    df["ret"] = df["px"].pct_change()
    df["ma20"] = df["px"].rolling(20).mean()
    df["ma60"] = df["px"].rolling(60).mean()
    return df


def features_at(df: pd.DataFrame, i: int) -> dict | None:
    """在索引 i 这一天收盘后的特征（即次日若启动，则这是涨前最后一天）。"""
    if i < 60 or i >= len(df):
        return None
    w = PRE_WIN
    if i < w:
        return None
    px = df["px"].values
    ret = df["ret"].values
    vol = df["vol"].values
    amt = df["amt"].values
    hi = df["hi"].values
    lo = df["lo"].values

    def safe_ret(n: int) -> float:
        if i - n < 0 or px[i - n] <= 0:
            return np.nan
        return float(px[i] / px[i - n] - 1.0) * 100.0

    seg_ret = ret[i - w + 1 : i + 1]
    seg_ret = seg_ret[np.isfinite(seg_ret)]
    if len(seg_ret) < w // 2:
        return None

    # 回撤（前40日）
    seg_px = px[i - w : i + 1]
    peak = np.maximum.accumulate(seg_px)
    dd = float(((seg_px - peak) / peak).min()) * 100.0

    # 距高点
    def dist_high(n: int) -> float:
        sl = px[max(0, i - n + 1) : i + 1]
        if len(sl) == 0 or sl.max() <= 0:
            return np.nan
        return float(px[i] / sl.max() - 1.0) * 100.0

    ma20 = float(df["ma20"].iloc[i]) if np.isfinite(df["ma20"].iloc[i]) else np.nan
    ma60 = float(df["ma60"].iloc[i]) if np.isfinite(df["ma60"].iloc[i]) else np.nan
    close = float(px[i])

    vol5 = float(np.nanmean(vol[i - 4 : i + 1]))
    vol20 = float(np.nanmean(vol[i - 19 : i + 1]))
    amt5 = float(np.nanmean(amt[i - 4 : i + 1]))
    amt20 = float(np.nanmean(amt[i - 19 : i + 1]))

    amp = (hi[i - w + 1 : i + 1] - lo[i - w + 1 : i + 1]) / np.clip(
        px[i - w + 1 : i + 1], 1e-9, None
    )
    quiet = float(np.mean(np.abs(seg_ret) < 0.02)) * 100.0

    # 距上次单日涨>5%
    big = np.where(ret[: i + 1] > 0.05)[0]
    days_since_big = float(i - big[-1]) if len(big) else 999.0

    mv = float(df["float_shares"].iloc[i] * df["收盘"].iloc[i] / 1e8)

    return {
        "前20日涨幅%": safe_ret(20),
        "前40日涨幅%": safe_ret(40),
        "前60日涨幅%": safe_ret(60),
        "前40日波动%": float(np.nanstd(seg_ret) * 100.0),
        "前40日最大回撤%": dd,
        "距60日高点%": dist_high(60),
        "距120日高点%": dist_high(120),
        "收盘/MA20": close / ma20 if ma20 and np.isfinite(ma20) else np.nan,
        "MA20/MA60": ma20 / ma60 if ma20 and ma60 and np.isfinite(ma60) else np.nan,
        "量能比5_20": vol5 / vol20 if vol20 > 0 else np.nan,
        "额能比5_20": amt5 / amt20 if amt20 > 0 else np.nan,
        "换手率%": float(df["turn"].iloc[i]),
        "前40日上涨占比%": float(np.mean(seg_ret > 0) * 100.0),
        "前40日振幅中位%": float(np.nanmedian(amp) * 100.0),
        "安静日占比%": quiet,
        "距上次大涨>5%天数": days_since_big,
        "流通市值亿": mv,
    }


def rally_quality(df: pd.DataFrame) -> pd.DataFrame:
    q = df[
        (df["累计涨幅%"] >= Q_CUM)
        & (df["窗口最大回撤%"] >= Q_DD)
        & (df["趋势R2"] >= Q_R2)
    ].copy()
    return q


def collect_positive(rallies: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in rallies.iterrows():
        code = str(r["股票代码"]).zfill(6)
        start = pd.Timestamp(r["子区间起点"])
        df = load_daily(code)
        if df is None:
            continue
        # 涨前最后一天 = 子区间起点的前一交易日
        idx = df.index[df["日期"] < start]
        if len(idx) == 0:
            continue
        i = int(idx[-1])
        feat = features_at(df, i)
        if feat is None:
            continue
        feat["股票代码"] = code
        feat["标签"] = "涨前"
        feat["锚点日期"] = str(df["日期"].iloc[i].date())
        feat["随后累计%"] = float(r["累计涨幅%"])
        rows.append(feat)
    return pd.DataFrame(rows)


def collect_controls(rallies: pd.DataFrame, per_stock: int = 3, seed: int = 42) -> pd.DataFrame:
    """同股票随机日做对照（避开主升起点前后20日）。"""
    rng = np.random.default_rng(seed)
    # 主升禁区
    ban: dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]] = {}
    for _, r in rallies.iterrows():
        code = str(r["股票代码"]).zfill(6)
        s = pd.Timestamp(r["子区间起点"])
        e = pd.Timestamp(r["子区间终点"])
        ban.setdefault(code, []).append((s - pd.Timedelta(days=40), e + pd.Timedelta(days=10)))

    rows = []
    codes = sorted(ban.keys())
    for code in codes:
        df = load_daily(code)
        if df is None or len(df) < 120:
            continue
        zones = ban[code]
        candidates = []
        for i in range(60, len(df) - 5):
            d = df["日期"].iloc[i]
            if any(a <= d <= b for a, b in zones):
                continue
            candidates.append(i)
        if len(candidates) < per_stock:
            continue
        pick = rng.choice(candidates, size=per_stock, replace=False)
        for i in pick:
            feat = features_at(df, int(i))
            if feat is None:
                continue
            feat["股票代码"] = code
            feat["标签"] = "对照"
            feat["锚点日期"] = str(df["日期"].iloc[int(i)].date())
            feat["随后累计%"] = np.nan
            rows.append(feat)
    return pd.DataFrame(rows)


def compare(pos: pd.DataFrame, ctrl: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in FEATURE_COLS:
        a = pos[col].astype(float).dropna()
        b = ctrl[col].astype(float).dropna()
        if len(a) < 20 or len(b) < 20:
            continue
        # Cliff's delta-ish: (P(a>b) - P(a<b))
        # simpler: median diff + mean diff
        med_a, med_b = float(a.median()), float(b.median())
        mean_a, mean_b = float(a.mean()), float(b.mean())
        # separation: standardized median gap
        pooled = float(np.sqrt(0.5 * (a.var() + b.var()))) or 1.0
        sep = (med_a - med_b) / pooled
        rows.append(
            {
                "特征": col,
                "涨前中位数": round(med_a, 3),
                "对照中位数": round(med_b, 3),
                "中位差": round(med_a - med_b, 3),
                "涨前均值": round(mean_a, 3),
                "对照均值": round(mean_b, 3),
                "分离度": round(sep, 3),
                "涨前样本": len(a),
                "对照样本": len(b),
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values("分离度", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)


def profile_from_pos(pos: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """返回每个特征的 (p25, p75) 作为「涨前画像」区间。"""
    prof = {}
    for col in FEATURE_COLS:
        a = pos[col].astype(float).dropna()
        if len(a) < 20:
            continue
        prof[col] = (float(a.quantile(0.25)), float(a.quantile(0.75)))
    return prof


def score_match(feat: dict, prof: dict[str, tuple[float, float]], cmp: pd.DataFrame) -> float:
    """落在涨前四分位区间内得分；方向与分离度一致时额外加权。"""
    # top separating features
    top = cmp.head(10)["特征"].tolist()
    score = 0.0
    wsum = 0.0
    for _, row in cmp.iterrows():
        col = row["特征"]
        if col not in feat or col not in prof:
            continue
        v = feat[col]
        if v is None or not np.isfinite(v):
            continue
        lo, hi = prof[col]
        weight = min(abs(float(row["分离度"])), 2.0) + 0.3
        if col in top:
            weight *= 1.4
        inside = lo <= v <= hi
        # 也允许略宽：扩展到 p15-p85 近似用区间放宽 25%
        span = hi - lo
        soft_lo, soft_hi = lo - 0.25 * span, hi + 0.25 * span
        if inside:
            hit = 1.0
        elif soft_lo <= v <= soft_hi:
            hit = 0.5
        else:
            hit = 0.0
        score += hit * weight
        wsum += weight
    return score / wsum if wsum else 0.0


def scan_today(prof: dict, cmp: pd.DataFrame, name_map: dict, ind_map: dict) -> pd.DataFrame:
    rows = []
    files = sorted(p for p in DAILY.glob("*.csv") if p.name[0].isdigit())
    for path in files:
        code = path.stem.zfill(6)
        if not code.startswith(("600", "601", "603", "605", "000", "001", "002")):
            continue
        name = name_map.get(code, "")
        if "ST" in name.upper():
            continue
        df = load_daily(code)
        if df is None:
            continue
        i = len(df) - 1
        feat = features_at(df, i)
        if feat is None:
            continue
        # 基础流动性
        if not np.isfinite(feat.get("流通市值亿", np.nan)):
            continue
        if feat["流通市值亿"] < 30 or feat["流通市值亿"] > 1500:
            continue
        # 已在大幅主升中的先降权：距60日高点太近且前20日涨太多
        if feat["前20日涨幅%"] > 25:
            continue
        s = score_match(feat, prof, cmp)
        if s < 0.45:
            continue
        rows.append(
            {
                "股票代码": code,
                "股票名称": name,
                "行业板块": ind_map.get(code, ""),
                "匹配分": round(s, 3),
                "锚点日期": str(df["日期"].iloc[i].date()),
                **{k: (None if not np.isfinite(feat[k]) else round(float(feat[k]), 3)) for k in FEATURE_COLS},
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("匹配分", ascending=False).reset_index(drop=True)


def main() -> None:
    if not RALLY_CSV.exists():
        raise SystemExit(f"缺少 {RALLY_CSV}，请先跑 find_best_rally_segment.py")

    name_map, ind_map = load_maps()
    rallies = pd.read_csv(RALLY_CSV, dtype={"股票代码": str})
    rallies["股票代码"] = rallies["股票代码"].astype(str).str.zfill(6)
    q = rally_quality(rallies)
    print(f"主升样本（质量过滤后）: {len(q)} / {len(rallies)}")

    pos = collect_positive(q)
    print(f"涨前特征样本: {len(pos)}")
    ctrl = collect_controls(q, per_stock=3)
    print(f"对照特征样本: {len(ctrl)}")

    cmp = compare(pos, ctrl)
    OUT_CMP.parent.mkdir(parents=True, exist_ok=True)
    cmp.to_csv(OUT_CMP, index=False, encoding="utf-8-sig")
    print(f"\n特征对比已保存 {OUT_CMP}")
    print(cmp.head(12).to_string(index=False))

    prof = profile_from_pos(pos)
    print("\n涨前画像（四分位区间）要点：")
    for col in cmp.head(8)["特征"]:
        lo, hi = prof[col]
        print(f"  {col}: [{lo:.3f}, {hi:.3f}]")

    watch = scan_today(prof, cmp, name_map, ind_map)
    watch.to_csv(OUT_WATCH, index=False, encoding="utf-8-sig")
    print(f"\n当下候选 {len(watch)} 只 → {OUT_WATCH}")
    cols = [
        "股票代码",
        "股票名称",
        "行业板块",
        "匹配分",
        "前20日涨幅%",
        "距60日高点%",
        "前40日波动%",
        "量能比5_20",
        "换手率%",
        "流通市值亿",
    ]
    print(watch[cols].head(30).to_string(index=False))


if __name__ == "__main__":
    main()
