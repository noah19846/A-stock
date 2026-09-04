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
    """名称优先 stock_list；缺失时用 stock_industry 的股票名称兜底。"""
    name_map: dict[str, str] = {}
    ind_map: dict[str, str] = {}

    def _code_col(df: pd.DataFrame) -> str:
        return "股票代码" if "股票代码" in df.columns else df.columns[0]

    def _name_col(df: pd.DataFrame) -> str | None:
        if "股票名称" in df.columns:
            return "股票名称"
        return df.columns[1] if len(df.columns) >= 2 else None

    def _fill_names(df: pd.DataFrame, *, override: bool) -> None:
        c, n = _code_col(df), _name_col(df)
        if n is None:
            return
        df = df.copy()
        df[c] = df[c].astype(str).str.zfill(6)
        for code, name in zip(df[c], df[n].astype(str)):
            name = (name or "").strip()
            if not name or name.lower() == "nan":
                continue
            if override or code not in name_map:
                name_map[code] = name

    if INDUSTRY.exists():
        ind = pd.read_csv(INDUSTRY, dtype=str, encoding="utf-8-sig")
        c = _code_col(ind)
        ind[c] = ind[c].astype(str).str.zfill(6)
        if "行业板块" in ind.columns:
            ind_map = dict(zip(ind[c], ind["行业板块"].astype(str)))
        _fill_names(ind, override=False)
    if LIST.exists():
        nm = pd.read_csv(LIST, dtype=str, encoding="utf-8-sig")
        _fill_names(nm, override=True)
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


def _finite_feat(feat: dict, key: str) -> float | None:
    v = feat.get(key)
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def is_impulse_leftover(feat: dict, bands: dict) -> bool:
    """急涨后浅回：距60日低已高，且近20日涨幅仍大（纽威/盈峰这类）。首打一律观察。"""
    cfg = bands.get("impulse_demote") or {}
    if not cfg:
        return False
    d_min = cfg.get("dist60_low_pct")
    r_min = cfg.get("ret20_pct")
    if d_min is None or r_min is None:
        return False
    d60 = _finite_feat(feat, "距60日低点%")
    r20 = _finite_feat(feat, "前20日涨幅%")
    if d60 is None or r20 is None:
        return False
    return d60 > float(d_min) and r20 > float(r_min)


def is_impulse_digesting(feat: dict, bands: dict) -> bool:
    """急涨后正在消化：距60低仍高、前20日已降到 demote 以下但仍偏热。需更深回撤才允许首打。"""
    if is_impulse_leftover(feat, bands):
        return False
    cfg = bands.get("impulse_demote") or {}
    if not cfg:
        return False
    d_min = cfg.get("dist60_low_pct")
    digest_r = cfg.get("digest_ret20_pct", 10.0)
    if d_min is None:
        return False
    d60 = _finite_feat(feat, "距60日低点%")
    r20 = _finite_feat(feat, "前20日涨幅%")
    if d60 is None or r20 is None:
        return False
    return d60 > float(d_min) and r20 > float(digest_r)


def passes_impulse_reentry(feat: dict, bands: dict) -> bool:
    """厚实再入：相对20日高更深回撤，且收盘回到 MA20 附近。"""
    re = bands.get("impulse_reentry") or {}
    if not re:
        return True
    d20_need = re.get("dist20_high_pct")
    px_need = re.get("px_ma20")
    ok = True
    if d20_need is not None:
        d20 = _finite_feat(feat, "距20日高点%")
        if d20 is None or d20 > float(d20_need):
            ok = False
    if px_need is not None:
        px = _finite_feat(feat, "收盘/MA20")
        if px is None or px > float(px_need):
            ok = False
    return ok


def impulse_blocks_trade(feat: dict, bands: dict) -> bool:
    """急涨浅回链路：leftover 一律挡；digesting 未过厚实再入也挡。"""
    if is_impulse_leftover(feat, bands):
        return True
    if is_impulse_digesting(feat, bands) and not passes_impulse_reentry(feat, bands):
        return True
    return False


def impulse_watch_when(feat: dict, bands: dict) -> str:
    """观察理由 + 观察簿独立升级清单（不要求再过 default 可短打）。"""
    cfg = bands.get("impulse_demote") or {}
    promo = ((bands.get("watch_promote") or {}).get("impulse") or {})
    d_min = float(cfg.get("dist60_low_pct", 30))
    r_hot = float(cfg.get("ret20_pct", 15))
    d20_need = float(promo.get("dist20_high_pct", -8))
    d20_floor = float(promo.get("dist20_high_floor", -18))
    px_need = float(promo.get("px_ma20_hi", 1.03))
    px_lo = float(promo.get("px_ma20_lo", 0.94))

    d60 = _finite_feat(feat, "距60日低点%") or 0.0
    r20 = _finite_feat(feat, "前20日涨幅%") or 0.0
    d20 = _finite_feat(feat, "距20日高点%")
    px = _finite_feat(feat, "收盘/MA20")
    d20_s = f"{d20:.1f}%" if d20 is not None else "—"
    px_s = f"{px:.3f}" if px is not None else "—"

    phase = (
        f"观察理由：急涨浅回未消化（距60低+{d60:.0f}%>{d_min:.0f}%，前20日+{r20:.0f}%，"
        f"现距20高{d20_s}），首打盈亏比差。"
    )
    return (
        f"{phase} 观察簿升级（独立于default）："
        f"①距20日高点∈[{d20_floor:.0f}%,{d20_need:.0f}%]（现{d20_s}）；"
        f"②收盘/MA20∈[{px_lo:.2f},{px_need:.2f}]（现{px_s}）；"
        f"③止跌（收阳或收盘/MA5≥0.985）；④未贴回20高、流动性仍可。"
        f"不要求再过短线硬条件/画像。"
    )


def watch_promote_status(
    feat: dict,
    bands: dict,
    opened_reason: str | None,
    *,
    hard_score: float | None = None,
) -> tuple[bool, str, list[str]]:
    """观察簿专用升级，不要求 default 可短打。

    返回 (ready, note, gaps)。
    """
    reason = opened_reason or ""
    promo_root = bands.get("watch_promote") or {}
    gaps: list[str] = []

    def _hot() -> bool:
        return (
            (feat.get("前5日涨幅%") or 0) > 8
            or (feat.get("距20日高点%") or -99) > -2
            or (feat.get("前1日涨幅%") or 0) > 3
        )

    mv_lo, mv_hi = mv_bounds(bands)
    mv = _finite_feat(feat, "流通市值亿")
    turn = _finite_feat(feat, "换手率%")
    if mv is None or mv < mv_lo or mv > mv_hi:
        gaps.append("市值不在进出区间")
    if turn is None or turn > 12 or turn < 0.8:
        gaps.append("换手不适合短线")
    if _hot():
        gaps.append("已偏强/贴20高")

    if reason.startswith("impulse"):
        cfg = promo_root.get("impulse") or {}
        d20_need = float(cfg.get("dist20_high_pct", -8))
        d20_floor = float(cfg.get("dist20_high_floor", -18))
        px_hi = float(cfg.get("px_ma20_hi", 1.03))
        px_lo = float(cfg.get("px_ma20_lo", 0.94))
        ma5_lo = float(cfg.get("px_ma5_lo", 0.985))
        d20 = _finite_feat(feat, "距20日高点%")
        px = _finite_feat(feat, "收盘/MA20")
        ma5 = _finite_feat(feat, "收盘/MA5")
        ret1 = _finite_feat(feat, "今日涨跌%")
        if d20 is None or d20 > d20_need or d20 < d20_floor:
            gaps.append(f"距20高需∈[{d20_floor:.0f},{d20_need:.0f}]%（现{d20 if d20 is not None else '—'}）")
        if px is None or px > px_hi or px < px_lo:
            gaps.append(f"收盘/MA20需∈[{px_lo:.2f},{px_hi:.2f}]（现{px if px is not None else '—'}）")
        if cfg.get("need_stabilize", True):
            ok_stab = (ret1 is not None and ret1 >= 0) or (ma5 is not None and ma5 >= ma5_lo)
            if not ok_stab:
                gaps.append(f"未止跌（需收阳或收盘/MA5≥{ma5_lo}）")
        if gaps:
            return False, "观察簿未升级：" + "；".join(gaps), gaps
        return (
            True,
            (
                f"观察簿厚实再入（非default）：距20高{d20:.1f}%且收盘/MA20={px:.3f}，"
                f"已止跌。计划同短线：轻仓、+15%/-3%、最多8日。"
            ),
            [],
        )

    if reason == "near_setup":
        cfg = promo_root.get("near_setup") or {}
        ma5_lo = float(cfg.get("px_ma5_lo", 0.975))
        ma5_hi = float(cfg.get("px_ma5_hi", 1.008))
        prev1_hi = float(cfg.get("prev1_hi", 0.0))
        min_hard = float(cfg.get("min_hard", 0.80))
        ma5 = _finite_feat(feat, "收盘/MA5")
        prev1 = _finite_feat(feat, "前1日涨幅%")
        # hard_score 不在 feat 里，调用方会再拦一道；这里只看结构
        if ma5 is None or ma5 < ma5_lo or ma5 > ma5_hi:
            gaps.append(f"收盘/MA5需∈[{ma5_lo:.3f},{ma5_hi:.3f}]（现{ma5 if ma5 is not None else '—'}）")
        if prev1 is None or prev1 > prev1_hi:
            gaps.append(f"需先有阴线（前1日涨幅≤{prev1_hi:.1f}%，现{prev1 if prev1 is not None else '—'}）")
        if hard_score is not None and hard_score < min_hard:
            gaps.append(f"硬条件仍偏弱（{hard_score:.2f}<{min_hard:.2f}）")
        if gaps:
            return False, "观察簿未升级：" + "；".join(gaps), gaps
        return True, "观察簿确认再入（非default）：阴线后收盘贴回MA5。", []

    return False, "无独立升级规则（非急涨浅回/近画像观察）", gaps


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
