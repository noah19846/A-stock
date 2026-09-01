"""
对比短线爆发正样本 vs 同票随机对照日的买点特征，
导出分位区间规则，并扫描今日接近画像的股票。

输入：data/short_burst_entries.csv
输出：
  data/short_burst_feature_compare.csv
  data/short_burst_feature_bands.json
  data/short_burst_watch.csv
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DAILY = ROOT / "data" / "daily_raw"
LIST = ROOT / "data" / "stock_list.csv"
INDUSTRY = ROOT / "data" / "stock_industry.csv"
ENTRIES = ROOT / "data" / "short_burst_entries.csv"
OUT_CMP = ROOT / "data" / "short_burst_feature_compare.csv"
OUT_BANDS = ROOT / "data" / "short_burst_feature_bands.json"
OUT_WATCH = ROOT / "data" / "short_burst_watch.csv"

_DAILY_CACHE: dict[str, pd.DataFrame | None] = {}

# 流通市值默认门槛（亿元）；策略 JSON 里可用 "mv_yi": [lo, hi] 覆盖
DEFAULT_MV_YI = (40.0, 800.0)


def mv_bounds(bands: dict | None = None) -> tuple[float, float]:
    """读取流通市值门槛。优先 bands['mv_yi']=[lo,hi]。"""
    raw = None
    if bands:
        raw = bands.get("mv_yi") or bands.get("mv")
    if raw is not None and len(raw) >= 2:
        return float(raw[0]), float(raw[1])
    return DEFAULT_MV_YI


FEATURE_COLS = [
    "前1日涨幅%",
    "前3日涨幅%",
    "前5日涨幅%",
    "前10日涨幅%",
    "前20日涨幅%",
    "收盘/MA5",
    "收盘/MA10",
    "收盘/MA20",
    "MA5/MA10",
    "MA10/MA20",
    "距20日高点%",
    "距60日高点%",
    "距60日低点%",
    "当日振幅%",
    "额能比1_5",
    "额能比5_20",
    "量能比5_20",
    "换手率%",
    "流通市值亿",
    "前10日上涨占比%",
    "前10日波动%",
    "安静日占比10%",
    "距上次大涨>5%天数",
]


@lru_cache(maxsize=1)
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
    """兼容旧调用；底层走共享 daily_cache。"""
    import daily_cache

    code = str(code).zfill(6)
    if code in _DAILY_CACHE:
        return _DAILY_CACHE[code]
    df = daily_cache.get(code)
    _DAILY_CACHE[code] = df
    return df


def features_at(df: pd.DataFrame, i: int) -> dict | None:
    if i < 60 or i >= len(df):
        return None
    px = df["px"].values
    hi = df["hi"].values
    lo = df["lo"].values
    vol = df["vol"].values
    amt = df["amt"].values
    ret = df["ret"].values
    close = float(px[i])
    ma5 = float(df["ma5"].iloc[i])
    ma10 = float(df["ma10"].iloc[i])
    ma20 = float(df["ma20"].iloc[i])
    if not all(np.isfinite([close, ma5, ma10, ma20])) or min(ma5, ma10, ma20) <= 0:
        return None

    def safe_ret(n: int) -> float:
        if i - n < 0 or px[i - n] <= 0:
            return np.nan
        return float(px[i] / px[i - n] - 1.0) * 100.0

    def dist_high(n: int) -> float:
        sl = px[max(0, i - n + 1) : i + 1]
        return float(close / sl.max() - 1.0) * 100.0

    def dist_low(n: int) -> float:
        sl = lo[max(0, i - n + 1) : i + 1]
        lo_n = float(np.nanmin(sl))
        if not np.isfinite(lo_n) or lo_n <= 0:
            return np.nan
        return float(close / lo_n - 1.0) * 100.0

    amt1 = float(amt[i])
    amt5 = float(np.nanmean(amt[i - 4 : i + 1]))
    amt20 = float(np.nanmean(amt[i - 19 : i + 1]))
    vol5 = float(np.nanmean(vol[i - 4 : i + 1]))
    vol20 = float(np.nanmean(vol[i - 19 : i + 1]))
    seg = ret[i - 9 : i + 1]
    seg = seg[np.isfinite(seg)]
    amp = float((hi[i] - lo[i]) / close * 100.0) if close else np.nan
    quiet = float(np.mean(np.abs(seg) < 0.015) * 100.0) if len(seg) else np.nan
    big = np.where(ret[: i + 1] > 0.05)[0]
    days_since_big = float(i - big[-1]) if len(big) else 999.0

    return {
        "前1日涨幅%": safe_ret(1),
        "前3日涨幅%": safe_ret(3),
        "前5日涨幅%": safe_ret(5),
        "前10日涨幅%": safe_ret(10),
        "前20日涨幅%": safe_ret(20),
        "收盘/MA5": close / ma5,
        "收盘/MA10": close / ma10,
        "收盘/MA20": close / ma20,
        "MA5/MA10": ma5 / ma10,
        "MA10/MA20": ma10 / ma20,
        "距20日高点%": dist_high(20),
        "距60日高点%": dist_high(60),
        "距60日低点%": dist_low(60),
        "当日振幅%": amp,
        "额能比1_5": amt1 / amt5 if amt5 > 0 else np.nan,
        "额能比5_20": amt5 / amt20 if amt20 > 0 else np.nan,
        "量能比5_20": vol5 / vol20 if vol20 > 0 else np.nan,
        "换手率%": float(df["turn"].iloc[i]),
        "流通市值亿": float(df["mv"].iloc[i]),
        "前10日上涨占比%": float(np.mean(seg > 0) * 100.0) if len(seg) else np.nan,
        "前10日波动%": float(np.nanstd(seg) * 100.0) if len(seg) else np.nan,
        "安静日占比10%": quiet,
        "距上次大涨>5%天数": days_since_big,
    }


def collect_positives(entries: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for code, g in entries.groupby("股票代码"):
        code = str(code).zfill(6)
        df = load_daily(code)
        if df is None:
            continue
        date_to_i = {str(d.date()): i for i, d in enumerate(df["日期"])}
        for _, r in g.iterrows():
            i = date_to_i.get(str(r["买入日期"]))
            if i is None:
                continue
            feat = features_at(df, int(i))
            if feat is None:
                continue
            feat["股票代码"] = code
            feat["标签"] = "爆发买点"
            feat["锚点日期"] = str(r["买入日期"])
            feat["首次摸高日"] = r.get("首次摸高日", np.nan)
            feat["最大回撤%"] = r.get("最大回撤%", np.nan)
            feat["最大涨幅%"] = r.get("最大涨幅%", np.nan)
            rows.append(feat)
    return pd.DataFrame(rows)


def collect_controls(entries: pd.DataFrame, per_stock: int = 4, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ban: dict[str, set[str]] = {}
    for _, r in entries.iterrows():
        code = str(r["股票代码"]).zfill(6)
        ban.setdefault(code, set()).add(str(r["买入日期"]))

    rows = []
    for code in sorted(ban.keys()):
        df = load_daily(code)
        if df is None or len(df) < 120:
            continue
        banned = ban[code]
        # 也避开正样本前后 5 日
        ban_idx: set[int] = set()
        date_to_i = {str(d.date()): i for i, d in enumerate(df["日期"])}
        for d in banned:
            if d in date_to_i:
                i0 = date_to_i[d]
                for j in range(max(60, i0 - 5), min(len(df) - 6, i0 + 6)):
                    ban_idx.add(j)
        candidates = [i for i in range(60, len(df) - 6) if i not in ban_idx]
        if len(candidates) < per_stock:
            continue
        for i in rng.choice(candidates, size=per_stock, replace=False):
            feat = features_at(df, int(i))
            if feat is None:
                continue
            feat["股票代码"] = code
            feat["标签"] = "对照"
            feat["锚点日期"] = str(df["日期"].iloc[int(i)].date())
            rows.append(feat)
    return pd.DataFrame(rows)


def compare(pos: pd.DataFrame, ctrl: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in FEATURE_COLS:
        a = pos[col].astype(float).dropna()
        b = ctrl[col].astype(float).dropna()
        if len(a) < 30 or len(b) < 30:
            continue
        med_a, med_b = float(a.median()), float(b.median())
        pooled = float(np.sqrt(0.5 * (a.var() + b.var()))) or 1.0
        sep = (med_a - med_b) / pooled
        rows.append(
            {
                "特征": col,
                "爆发中位数": round(med_a, 3),
                "对照中位数": round(med_b, 3),
                "中位差": round(med_a - med_b, 3),
                "爆发P25": round(float(a.quantile(0.25)), 3),
                "爆发P75": round(float(a.quantile(0.75)), 3),
                "分离度": round(sep, 3),
                "爆发样本": len(a),
                "对照样本": len(b),
            }
        )
    return pd.DataFrame(rows).sort_values("分离度", key=lambda s: s.abs(), ascending=False).reset_index(
        drop=True
    )


def build_bands(cmp: pd.DataFrame, top_k: int = 10) -> dict:
    """用分离度最高的特征建收紧区间（正样本 P35~P65）。"""
    rules = []
    for _, r in cmp.head(top_k).iterrows():
        col = r["特征"]
        # 从 compare 重算更紧分位需要原序列；这里用 P25/P75 内缩
        lo = float(r["爆发P25"])
        hi = float(r["爆发P75"])
        span = hi - lo
        if span == 0:
            span = abs(hi) * 0.05 + 0.01
        # 内缩到约 P35~P65
        lo2 = lo + 0.2 * span
        hi2 = hi - 0.2 * span
        w = min(1.8, 0.9 + abs(float(r["分离度"])) * 0.6)
        rules.append(
            {
                "feature": col,
                "lo": round(lo2, 4),
                "hi": round(hi2, 4),
                "weight": round(w, 3),
                "sep": float(r["分离度"]),
                "median": float(r["爆发中位数"]),
            }
        )
    # 硬门槛：历史买点最稳的几条（不可只用宽松打分）
    hard = [
        {"feature": "前1日涨幅%", "lo": -5.0, "hi": 0.3, "weight": 2.0},
        {"feature": "收盘/MA5", "lo": 0.96, "hi": 1.02, "weight": 1.8},
        {"feature": "距20日高点%", "lo": -14.0, "hi": -2.5, "weight": 1.6},
        {"feature": "额能比5_20", "lo": 0.90, "hi": 1.70, "weight": 1.4},
        {"feature": "当日振幅%", "lo": 2.8, "hi": 8.0, "weight": 1.2},
        {"feature": "前5日涨幅%", "lo": -8.0, "hi": 6.0, "weight": 1.2},
    ]
    return {"rules": rules, "hard": hard, "min_score": 0.72, "min_hard": 0.75}


def score_row(feat: dict, bands: dict) -> tuple[float, float, list[str]]:
    """返回 (soft_score, hard_score, reasons)。"""
    reasons: list[str] = []

    def _score(rules: list[dict], prefix: str) -> float:
        checks = []
        for rule in rules:
            col = rule["feature"]
            v = feat.get(col)
            if v is None or not np.isfinite(float(v)):
                continue
            v = float(v)
            ok = rule["lo"] <= v <= rule["hi"]
            checks.append((ok, float(rule.get("weight", 1.0))))
            mark = "✓" if ok else "✗"
            reasons.append(f"{prefix}{mark} {col}={v:.3f} ∈[{rule['lo']:.3f},{rule['hi']:.3f}]")
        if not checks:
            return 0.0
        wsum = sum(w for _, w in checks)
        return sum((1.0 if ok else 0.0) * w for ok, w in checks) / wsum

    soft = _score(bands.get("rules", []), "")
    hard = _score(bands.get("hard", []), "[硬]")
    return soft, hard, reasons


def scan_today(bands: dict, name_map: dict, ind_map: dict, min_score: float = 0.72) -> pd.DataFrame:
    rows = []
    min_hard = float(bands.get("min_hard", 0.75))
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
        mv_lo, mv_hi = mv_bounds(bands)
        if feat["流通市值亿"] < mv_lo or feat["流通市值亿"] > mv_hi:
            continue
        if feat["换手率%"] > 12 or feat["换手率%"] < 0.8:
            continue
        if feat["前5日涨幅%"] > 8 or feat["距20日高点%"] > -2:
            continue
        soft, hard, reasons = score_row(feat, bands)
        if soft < min_score and hard < min_hard:
            continue
        if hard >= 0.85 and soft >= min_score:
            stage = "可短打"
        elif hard >= min_hard or soft >= 0.8:
            stage = "观察"
        else:
            continue
        rows.append(
            {
                "code": code,
                "name": name,
                "industry": ind_map.get(code, ""),
                "asof": str(df["日期"].iloc[i].date()),
                "stage": stage,
                "score": round(soft, 3),
                "hard_score": round(hard, 3),
                "reasons": "；".join(reasons),
                **{k: round(float(feat[k]), 4) if np.isfinite(feat[k]) else np.nan for k in FEATURE_COLS},
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    order = {"可短打": 0, "观察": 1}
    out["_o"] = out["stage"].map(order)
    return out.sort_values(["_o", "hard_score", "score"], ascending=[True, False, False]).drop(
        columns=["_o"]
    ).reset_index(drop=True)


def main() -> None:
    if not ENTRIES.exists():
        raise SystemExit(f"缺少 {ENTRIES}，请先运行 find_short_burst_entries.py")

    entries = pd.read_csv(ENTRIES, dtype={"股票代码": str})
    entries["股票代码"] = entries["股票代码"].astype(str).str.zfill(6)
    print(f"正样本条目(原始): {len(entries)}")
    # 质量切片：浅回撤（允许几乎不破买入价）、排除极端妖涨，避免特征被一字板带偏
    q = entries[
        (entries["最大回撤%"] >= -3.0)
        & (entries["最大回撤%"] <= 0.5)
        & (entries["最大涨幅%"] >= 8.0)
        & (entries["最大涨幅%"] <= 28.0)
        & (entries["首次摸高日"] <= 5)
        & (entries["额20日均亿"] >= 0.4)
    ].copy()
    print(f"正样本条目(质量切片): {len(q)}")
    # 每只股票最多保留 8 条，防头部行业刷屏
    q = (
        q.sort_values(["首次摸高日", "最大回撤%"], ascending=[True, False])
        .groupby("股票代码", as_index=False)
        .head(3)
    )
    # 随机抽上限，加速；固定种子可复现
    if len(q) > 5000:
        q = q.sample(n=5000, random_state=42)
    print(f"正样本条目(建模用): {len(q)}", flush=True)
    entries = q

    pos = collect_positives(entries)
    print(f"特征化正样本: {len(pos)}", flush=True)
    ctrl = collect_controls(entries, per_stock=2)
    print(f"对照样本: {len(ctrl)}", flush=True)

    cmp = compare(pos, ctrl)
    cmp.to_csv(OUT_CMP, index=False, encoding="utf-8-sig")
    print("\n特征分离度 Top15:")
    print(cmp.head(15).to_string(index=False))

    bands = build_bands(cmp, top_k=12)
    OUT_BANDS.write_text(json.dumps(bands, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n规则 → {OUT_BANDS}")

    name_map, ind_map = load_maps()
    watch = scan_today(bands, name_map, ind_map, min_score=float(bands["min_score"]))
    watch.to_csv(OUT_WATCH, index=False, encoding="utf-8-sig")
    print(f"今日匹配: {len(watch)} → {OUT_WATCH}")
    if not watch.empty:
        cols = ["code", "name", "industry", "stage", "score", "收盘/MA5", "前5日涨幅%", "额能比5_20", "距20日高点%"]
        cols = [c for c in cols if c in watch.columns]
        print(watch[cols].head(30).to_string(index=False))


if __name__ == "__main__":
    main()
