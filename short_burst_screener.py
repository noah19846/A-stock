"""
短线「浅回撤→几天内+8%」买点筛选器

依赖：
  data/short_burst_feature_bands.json  （由 analyze_short_burst_features.py 生成）

用法：
  .venv/bin/python short_burst_screener.py 601696
  .venv/bin/python short_burst_screener.py --scan
  .venv/bin/python short_burst_screener.py --scan --entry-only
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
    load_daily,
    load_maps,
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


def load_bands() -> dict:
    if not BANDS.exists():
        raise SystemExit(f"缺少 {BANDS}，请先运行 analyze_short_burst_features.py")
    return json.loads(BANDS.read_text(encoding="utf-8"))


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

    soft, hard, reasons = score_row(feat, bands)
    min_score = float(bands.get("min_score", 0.72))
    min_hard = float(bands.get("min_hard", 0.75))

    hot = feat["前5日涨幅%"] > 8 or feat["距20日高点%"] > -2 or feat.get("前1日涨幅%", 0) > 3
    bad_liq = feat["流通市值亿"] < 40 or feat["流通市值亿"] > 800 or feat["换手率%"] > 12

    if hot:
        # 偏热不再进池（避免 signals 被「已偏强」刷成千级）
        stage = "已偏强"
        can = False
        when = "近端偏热或贴20高，短线首打盈亏比差"
    elif bad_liq:
        stage = "不关注"
        can = False
        when = "市值/换手不适合短线进出"
    elif hard >= 0.95 and soft >= max(min_score, 0.80):
        stage = "可短打"
        can = True
        when = (
            "硬条件+画像都过。计划：轻仓短打；目标约+8%减仓；"
            "盘中相对成本回撤破约-3%离场；满5日未达目标评估离场。"
        )
    elif hard >= 0.85 and soft >= max(min_hard, 0.75):
        # 观察必须硬条件与画像同时接近，禁止「只过一边」灌水
        stage = "观察"
        can = False
        when = "接近短线爆发画像，等阴线企稳后的量能确认或收盘更贴MA5"
    else:
        stage = "不关注"
        can = False
        when = "不像历史浅回撤快涨买点"

    metrics = {k: round(float(feat[k]), 4) if np.isfinite(feat[k]) else np.nan for k in FEATURE_COLS}
    metrics["收盘"] = round(float(df["收盘"].iloc[i]), 3)
    metrics["今日涨跌%"] = (
        round(float(df["ret"].iloc[i]) * 100, 3) if np.isfinite(df["ret"].iloc[i]) else np.nan
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


def scan(entry_only: bool = False, asof: str | None = None) -> pd.DataFrame:
    import daily_cache

    bands = load_bands()
    name_map, ind_map = load_maps()
    rows = []
    codes = daily_cache.cached_codes()
    if not codes:
        codes = [
            p.stem.zfill(6)
            for p in sorted(Path(ROOT / "data" / "daily_raw").glob("*.csv"))
            if p.name[0].isdigit()
            and p.stem.zfill(6).startswith(("600", "601", "603", "605", "000", "001", "002"))
        ]
    for code in codes:
        if not code.startswith(("600", "601", "603", "605", "000", "001", "002")):
            continue
        if "ST" in name_map.get(code, "").upper():
            continue
        v = evaluate(code, bands, name_map, ind_map, asof=asof)
        if v.stage not in ("可短打", "观察"):
            continue
        if entry_only and not v.can_trade:
            continue
        rows.append(v.to_row())
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    order = {"可短打": 0, "观察": 1}
    out["_o"] = out["stage"].map(order).fillna(9)
    return out.sort_values(["_o", "hard_score", "score"], ascending=[True, False, False]).drop(
        columns=["_o"]
    ).reset_index(drop=True)


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
    args = parser.parse_args()

    if args.code:
        print_verdict(evaluate(args.code))
        return

    if args.scan:
        print("扫描中…", flush=True)
        df = scan(entry_only=args.entry_only)
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
