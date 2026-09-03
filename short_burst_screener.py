"""
短线「浅回撤→几天内+8%」买点筛选器

依赖：
  data/short_burst_feature_bands.json  （由 analyze_short_burst_features.py 生成）

用法：
  .venv/bin/python short_burst_screener.py 601696
  .venv/bin/python short_burst_screener.py --scan
  .venv/bin/python short_burst_screener.py --scan --entry-only
  .venv/bin/python short_burst_screener.py --scan --bands data/short_burst_feature_bands_strict.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_short_burst_features import (
    FEATURE_COLS,
    features_at,
    impulse_blocks_trade,
    impulse_watch_when,
    is_impulse_digesting,
    is_impulse_leftover,
    load_maps,
    mv_bounds,
    score_row,
)

ROOT = Path(__file__).resolve().parent
BANDS = ROOT / "data" / "short_burst_feature_bands.json"
OUT_SCAN = ROOT / "data" / "short_burst_signals.csv"
OUT_NOW = ROOT / "data" / "short_burst_now.csv"


@dataclass
class Verdict:
    code: str
    name: str
    industry: str
    asof: str
    stage: str
    can_trade: bool
    score: float
    hard_score: float
    when: str
    reasons: list[str]
    metrics: dict

    def to_row(self) -> dict:
        d = asdict(self)
        d["reasons"] = "；".join(self.reasons)
        m = d.pop("metrics")
        d.update(m)
        return d


def load_bands(path: Path | None = None) -> dict:
    p = path or BANDS
    if not p.exists():
        raise SystemExit(f"缺少 {p}，请先运行 analyze_short_burst_features.py 或指定 --bands")
    return json.loads(p.read_text(encoding="utf-8"))


def evaluate(
    code: str,
    bands: dict | None = None,
    name_map=None,
    ind_map=None,
    asof: str | None = None,
) -> Verdict:
    code = str(code).zfill(6)
    if bands is None:
        bands = load_bands()
    if name_map is None or ind_map is None:
        name_map, ind_map = load_maps()
    name = name_map.get(code, "")
    industry = ind_map.get(code, "")

    import daily_cache

    df = daily_cache.get(code, asof=asof)
    if df is None:
        return Verdict(code, name, industry, "", "不关注", False, 0.0, 0.0, "无日线", ["缺少数据"], {})

    i = len(df) - 1
    asof_s = str(df["日期"].iloc[i].date())
    feat = features_at(df, i)
    if feat is None:
        return Verdict(code, name, industry, asof_s, "不关注", False, 0.0, 0.0, "指标不足", [], {})

    return verdict_from_feat(code, name, industry, asof_s, df, i, feat, bands)


def verdict_from_feat(
    code: str,
    name: str,
    industry: str,
    asof_s: str,
    df: pd.DataFrame,
    i: int,
    feat: dict,
    bands: dict,
) -> Verdict:
    soft, hard, reasons = score_row(feat, bands)
    min_score = float(bands.get("min_score", 0.72))
    min_hard = float(bands.get("min_hard", 0.75))
    buy_hard = float(bands.get("signal_hard", 0.95))
    buy_soft = float(bands.get("signal_soft", max(min_score, 0.80)))
    watch_hard = float(bands.get("watch_hard", 0.85))
    watch_soft = float(bands.get("watch_soft", max(min_hard, 0.75)))

    hot = feat["前5日涨幅%"] > 8 or feat["距20日高点%"] > -2 or feat.get("前1日涨幅%", 0) > 3
    mv_lo, mv_hi = mv_bounds(bands)
    bad_liq = (
        feat["流通市值亿"] < mv_lo
        or feat["流通市值亿"] > mv_hi
        or feat["换手率%"] > 12
    )

    if hot:
        # 偏热不再进池（避免 signals 被「已偏强」刷成千级）
        stage = "已偏强"
        can = False
        when = "近端偏热或贴20高，短线首打盈亏比差"
    elif bad_liq:
        stage = "不关注"
        can = False
        when = "市值/换手不适合短线进出"
    elif hard >= buy_hard and soft >= buy_soft:
        stage = "可短打"
        can = True
        ex = bands.get("exit") or {}
        tp = float(ex.get("target", 0.15)) * 100
        sl = float(ex.get("stop", 0.03)) * 100
        hold = int(ex.get("hold", 8))
        when = (
            f"硬条件+画像都过。计划：轻仓短打；目标约+{tp:.0f}%减仓；"
            f"盘中相对成本回撤破约-{sl:.0f}%离场；满{hold}日未达目标评估离场。"
        )
        if impulse_blocks_trade(feat, bands):
            stage = "观察"
            can = False
            when = impulse_watch_when(feat, bands)
            reasons.append("急涨浅回链路：未满足厚实再入，暂不首打")
    elif hard >= watch_hard and soft >= watch_soft:
        # 观察必须硬条件与画像同时接近，禁止「只过一边」灌水
        stage = "观察"
        can = False
        when = (
            "接近短线爆发画像，入观察簿：阴线后收盘贴回MA5且硬条件不太差则升级"
            "（不要求再过 default 可短打）"
        )
    else:
        stage = "不关注"
        can = False
        when = "不像历史浅回撤快涨买点"

    metrics = {k: round(float(feat[k]), 4) if np.isfinite(feat[k]) else np.nan for k in FEATURE_COLS}
    metrics["收盘"] = round(float(df["收盘"].iloc[i]), 3)
    metrics["今日涨跌%"] = (
        round(float(df["ret"].iloc[i]) * 100, 3) if np.isfinite(df["ret"].iloc[i]) else np.nan
    )
    warn_thr = float(bands.get("warn_dist60_low_pct", 20.0))
    d60l = feat.get("距60日低点%")
    metrics["离底过远提醒"] = bool(
        d60l is not None
        and np.isfinite(float(d60l))
        and float(d60l) > warn_thr
    )
    metrics["急涨浅回提醒"] = bool(
        is_impulse_leftover(feat, bands) or is_impulse_digesting(feat, bands)
    )

    return Verdict(
        code=code,
        name=name,
        industry=industry,
        asof=asof_s,
        stage=stage,
        can_trade=can,
        score=round(soft, 3),
        hard_score=round(hard, 3),
        when=when,
        reasons=reasons,
        metrics=metrics,
    )


def rank_score(hard: float, soft: float, bands: dict) -> float:
    w = float(bands.get("rank_hard_weight", 1.2))
    return float(hard) * w + float(soft)


def apply_daily_top_k(df: pd.DataFrame, bands: dict) -> pd.DataFrame:
    """可短打按日排名，仅保留 TopK 为可短打，其余降为观察。"""
    k = int(bands.get("daily_top_k") or 0)
    if k <= 0 or df.empty or "stage" not in df.columns:
        return df
    out = df.copy()
    buy = out["stage"] == "可短打"
    if buy.sum() == 0:
        return out
    hard = out["hard_score"] if "hard_score" in out.columns else out.get("hard", 0)
    soft = out["score"] if "score" in out.columns else out.get("soft", 0)
    out["_rank"] = [
        rank_score(h, s, bands) for h, s in zip(hard.fillna(0), soft.fillna(0))
    ]
    # 单日截面（scan 本身就是 asof 一日）；若有 asof 列则按日
    if "asof" in out.columns:
        keep_idx = set()
        for _, g in out[buy].groupby("asof", sort=False):
            keep_idx.update(g.nlargest(k, "_rank").index.tolist())
    else:
        keep_idx = set(out.loc[buy].nlargest(k, "_rank").index.tolist())

    demote = buy & ~out.index.isin(keep_idx)
    if demote.any():
        out.loc[demote, "stage"] = "观察"
        out.loc[demote, "can_trade"] = False
        out.loc[demote, "when"] = (
            f"画像够格但当日未进 Top{k}（按硬分×{bands.get('rank_hard_weight', 1.2)}+画像分排序），不做首打"
        )
    out = out.drop(columns=["_rank"], errors="ignore")
    order = {"可短打": 0, "观察": 1}
    out["_o"] = out["stage"].map(order).fillna(9)
    return (
        out.sort_values(["_o", "hard_score", "score"], ascending=[True, False, False])
        .drop(columns=["_o"])
        .reset_index(drop=True)
    )


def scan(
    entry_only: bool = False,
    asof: str | None = None,
    bands: dict | None = None,
) -> pd.DataFrame:
    if bands is None:
        bands = load_bands()
    out = scan_multi([("default", bands)], asof=asof, entry_only=entry_only)
    return out.get("default", pd.DataFrame())


def scan_multi(
    strategies: list[tuple[str, dict]],
    asof: str | None = None,
    entry_only: bool = False,
) -> dict[str, pd.DataFrame]:
    """一次算特征，对多套 bands 分别打分。strategies=[(id, bands), ...]。"""
    import daily_cache

    if not strategies:
        return {}
    name_map, ind_map = load_maps()
    rows_by: dict[str, list[dict]] = {sid: [] for sid, _ in strategies}
    codes = daily_cache.cached_codes()
    if not codes:
        codes = [
            p.stem.zfill(6)
            for p in sorted(Path(ROOT / "data" / "daily_raw").glob("*.csv"))
            if p.name[0].isdigit()
            and p.stem.zfill(6).startswith(("600", "601", "603", "605", "000", "001", "002"))
        ]
    prefixes = ("600", "601", "603", "605", "000", "001", "002")
    for code in codes:
        if not code.startswith(prefixes):
            continue
        if "ST" in name_map.get(code, "").upper():
            continue
        df = daily_cache.get(code, asof=asof)
        if df is None:
            continue
        i = len(df) - 1
        asof_s = str(df["日期"].iloc[i].date())
        feat = features_at(df, i)
        if feat is None:
            continue
        name = name_map.get(code, "")
        industry = ind_map.get(code, "")
        for sid, bands in strategies:
            v = verdict_from_feat(code, name, industry, asof_s, df, i, feat, bands)
            if v.stage not in ("可短打", "观察"):
                continue
            if entry_only and not v.can_trade:
                continue
            rows_by[sid].append(v.to_row())

    out: dict[str, pd.DataFrame] = {}
    order = {"可短打": 0, "观察": 1}
    for sid, bands in strategies:
        df = pd.DataFrame(rows_by[sid])
        if df.empty:
            out[sid] = df
            continue
        df["_o"] = df["stage"].map(order).fillna(9)
        df = df.sort_values(
            ["_o", "hard_score", "score"], ascending=[True, False, False]
        ).drop(columns=["_o"]).reset_index(drop=True)
        df = apply_daily_top_k(df, bands)
        if entry_only and not df.empty and "can_trade" in df.columns:
            df = df[df["can_trade"] == True].reset_index(drop=True)  # noqa: E712
        out[sid] = df
    return out


def print_verdict(v: Verdict) -> None:
    print(f"{v.code} {v.name}  {v.industry}")
    print(f"日期 {v.asof}")
    print(f"阶段：{v.stage}    可短打：{'是' if v.can_trade else '否'}")
    print(f"画像分 {v.score:.3f}   硬条件分 {v.hard_score:.3f}")
    print(f"怎么做：{v.when}")
    print("明细：")
    for r in v.reasons:
        print(" ", r)
    print("关键指标：")
    for k in [
        "收盘",
        "今日涨跌%",
        "前1日涨幅%",
        "收盘/MA5",
        "前3日涨幅%",
        "前5日涨幅%",
        "距20日高点%",
        "当日振幅%",
        "额能比5_20",
        "换手率%",
    ]:
        print(f"  {k}: {v.metrics.get(k)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="短线浅回撤快涨筛选器")
    parser.add_argument("code", nargs="?", help="股票代码")
    parser.add_argument("--scan", action="store_true")
    parser.add_argument("--entry-only", action="store_true")
    parser.add_argument(
        "--bands",
        type=str,
        default=str(BANDS),
        help="特征区间 JSON（默认 data/short_burst_feature_bands.json）",
    )
    args = parser.parse_args()
    bands = load_bands(Path(args.bands))

    if args.code:
        print_verdict(evaluate(args.code, bands=bands))
        return

    if args.scan:
        print(f"扫描中… bands={args.bands}", flush=True)
        df = scan(entry_only=args.entry_only, bands=bands)
        OUT_SCAN.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUT_SCAN, index=False, encoding="utf-8-sig")
        now = df[df["can_trade"] == True].copy() if "can_trade" in df.columns else df.iloc[0:0]  # noqa: E712
        now.to_csv(OUT_NOW, index=False, encoding="utf-8-sig")
        print(f"共 {len(df)} 条 → {OUT_SCAN}", flush=True)
        print(f"可短打 {len(now)} → {OUT_NOW}", flush=True)
        cols = [
            "code",
            "name",
            "industry",
            "stage",
            "score",
            "hard_score",
            "收盘/MA5",
            "前1日涨幅%",
            "前5日涨幅%",
            "距20日高点%",
            "额能比5_20",
        ]
        cols = [c for c in cols if c in df.columns]
        print(df[cols].head(40).to_string(index=False))
        return

    parser.print_help()


if __name__ == "__main__":
    main()
