"""
底部启动（三连小阳 + 底部横盘，严格相似）：

形态：
  · 三日小阳，日涨约 0.2%～3.5%；每日量比 1.05～2.0；相对三日前量 ≥1.15×
  · 前 22 日横盘：振幅 ≤16%、|净涨跌| ≤10%、距 60 日低 ≤8%

交易：
  · 信号日尾盘买入
  · 止损 −3%（盘中最低触及）；另标注横盘箱体底作结构止损参考
  · 不设止盈；满 5 个交易日收盘卖

用法：
  .venv/bin/python bottom_base_screener.py 603693
  .venv/bin/python bottom_base_screener.py --scan --asof 2026-09-10
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_short_burst_features import load_maps
from three_yang_screener import (
    BASE_ABS_RET_MAX,
    BASE_DAYS,
    BASE_DIST60_LOW_MAX,
    BASE_RANGE_MAX,
    MAINBOARD_PREFIX,
    VOL_GROWTH_MIN,
    match_at,
)

ROOT = Path(__file__).resolve().parent
OUT_SCAN = ROOT / "data" / "bottom_base_signals.csv"

STOP_PCT = 0.03
HOLD_DAYS = 5
MV_LO, MV_HI = 20.0, 800.0


@dataclass
class Verdict:
    code: str
    name: str
    industry: str
    asof: str
    stage: str
    can_buy: bool
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


def _blank(code, name, industry, asof, why, metrics=None) -> Verdict:
    return Verdict(
        code, name, industry, asof, "不关注", False, 0.0, 0.0, why, [why], metrics or {}
    )


def evaluate(
    code: str,
    name_map=None,
    ind_map=None,
    asof: str | None = None,
) -> Verdict:
    code = str(code).zfill(6)
    if name_map is None or ind_map is None:
        name_map, ind_map = load_maps()
    name = name_map.get(code, "")
    industry = ind_map.get(code, "")

    import daily_cache

    df = daily_cache.get(code, asof=asof)
    if df is None or len(df) < 90:
        return _blank(code, name, industry, "", "无足够日线")
    asof_s = str(df["日期"].iloc[-1].date())
    if "ST" in name.upper() or name.startswith("*"):
        return _blank(code, name, industry, asof_s, "ST 跳过")

    mv = float(df["mv"].iloc[-1]) if "mv" in df.columns else np.nan
    if np.isfinite(mv) and (mv < MV_LO or mv > MV_HI):
        return _blank(code, name, industry, asof_s, f"市值不在 {MV_LO:.0f}–{MV_HI:.0f} 亿")

    px = df["px"].to_numpy(dtype=np.float64)
    hi = df["hi"].to_numpy(dtype=np.float64)
    lo = df["lo"].to_numpy(dtype=np.float64)
    op = df["op"].to_numpy(dtype=np.float64)
    vol = df["vol"].to_numpy(dtype=np.float64)
    ret = df["ret"].to_numpy(dtype=np.float64)
    i = len(df) - 1

    hit = match_at(px, op, vol, ret, i, hi=hi, lo=lo, require_base=True)
    if hit is None:
        return _blank(code, name, industry, asof_s, "未满足三连小阳+底部横盘")

    # 箱体：横盘段 + 三连阳期间的最低/横盘最高（图表用前复权价）
    base_end = i - 3
    base_a = base_end - BASE_DAYS + 1
    box_lo_hfq = float(np.nanmin(lo[base_a : i + 1]))
    box_hi_hfq = float(np.nanmax(hi[base_a : base_end + 1]))
    fac = df["后复权因子"].to_numpy(dtype=np.float64)
    fac_last = float(fac[i]) if np.isfinite(fac[i]) and fac[i] > 0 else 1.0
    box_bottom = round(box_lo_hfq / fac_last, 4)
    box_top = round(box_hi_hfq / fac_last, 4)
    box_stop = round(box_bottom * 0.998, 4)

    close_raw = float(df["收盘"].iloc[i])
    stop_px = round(close_raw * (1.0 - STOP_PCT), 2)
    r1 = float(ret[i]) if np.isfinite(ret[i]) else 0.0

    score = 70.0
    if hit.dist60_low is not None:
        score += min(
            20.0,
            max(0.0, (BASE_DIST60_LOW_MAX - hit.dist60_low) / BASE_DIST60_LOW_MAX) * 20,
        )
    if hit.base_range is not None:
        score += min(
            15.0,
            max(0.0, (BASE_RANGE_MAX - hit.base_range) / BASE_RANGE_MAX) * 15,
        )
    score += min(10.0, hit.cum3 / 0.06 * 10)
    score += min(5.0, max(0.0, (hit.vol_growth - VOL_GROWTH_MIN) / 1.5) * 5)

    reasons = [
        f"三日涨幅 {hit.ret1 * 100:.2f}/{hit.ret2 * 100:.2f}/{hit.ret3 * 100:.2f}%",
        f"三日量比 {hit.vr1:.2f}/{hit.vr2:.2f}/{hit.vr3:.2f}",
        f"量相对三日前 {hit.vol_growth:.2f}×",
        f"箱体底 {box_bottom}（跌破可作结构止损参考）",
    ]
    if hit.base_range is not None:
        reasons.append(
            f"前{BASE_DAYS}日振幅 {hit.base_range * 100:.1f}% "
            f"（≤{BASE_RANGE_MAX * 100:.0f}%）"
        )
    if hit.base_ret is not None:
        reasons.append(
            f"横盘净涨跌 {hit.base_ret * 100:.1f}%（| |≤{BASE_ABS_RET_MAX * 100:.0f}%）"
        )
    if hit.dist60_low is not None:
        reasons.append(
            f"距60日低 {hit.dist60_low * 100:.1f}%（≤{BASE_DIST60_LOW_MAX * 100:.0f}%）"
        )

    when = (
        f"建议：三连小阳底部启动，当日尾盘买；"
        f"固定止损 −{STOP_PCT * 100:.0f}%（约 {stop_px}）；"
        f"箱体底 {box_bottom}（结构止损参考，约 {box_stop}）；"
        f"不设止盈；满 {HOLD_DAYS} 个交易日收盘卖"
    )

    metrics = {
        "收盘": round(close_raw, 2),
        "今日涨跌%": round(r1 * 100, 2),
        "三日累计%": round(hit.cum3 * 100, 2),
        "三日涨幅%": f"{hit.ret1 * 100:.2f}/{hit.ret2 * 100:.2f}/{hit.ret3 * 100:.2f}",
        "三日量比": f"{hit.vr1:.2f}/{hit.vr2:.2f}/{hit.vr3:.2f}",
        "量增长": round(hit.vol_growth, 2),
        "横盘振幅%": None
        if hit.base_range is None
        else round(hit.base_range * 100, 1),
        "横盘涨跌%": None if hit.base_ret is None else round(hit.base_ret * 100, 1),
        "距60日低%": None
        if hit.dist60_low is None
        else round(hit.dist60_low * 100, 1),
        "流通市值亿": None if not np.isfinite(mv) else round(mv, 1),
        "箱体底": box_bottom,
        "箱体顶": box_top,
        "止损参考": stop_px,
        "strategy_ids": "bottom_base",
        "strategy_tags": "底部启动",
    }

    return Verdict(
        code,
        name,
        industry,
        asof_s,
        "可买入",
        True,
        round(float(score), 1),
        1.0,
        when,
        reasons,
        metrics,
    )


def scan(asof: str | None = None, watch_top_k: int = 8) -> pd.DataFrame:
    import daily_cache

    name_map, ind_map = load_maps()
    rows = []
    for code in daily_cache.cached_codes():
        if not code.startswith(MAINBOARD_PREFIX):
            continue
        if "ST" in name_map.get(code, "").upper():
            continue
        v = evaluate(code, name_map, ind_map, asof=asof)
        if v.stage in ("可买入", "观察"):
            rows.append(v.to_row())
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    order = {"可买入": 0, "观察": 1}
    out["_o"] = out["stage"].map(order).fillna(9)
    out = out.sort_values(["_o", "score"], ascending=[True, False])
    out = out.drop(columns=["_o"]).reset_index(drop=True)
    if watch_top_k > 0 and "观察" in set(out["stage"]):
        buy = out[out["stage"] == "可买入"]
        watch = out[out["stage"] == "观察"].head(watch_top_k)
        out = pd.concat([buy, watch], ignore_index=True)
    return out


def print_verdict(v: Verdict) -> None:
    print(f"{v.code} {v.name}  {v.industry}")
    print(f"日期 {v.asof}")
    print(f"阶段：{v.stage}    {'可买' if v.can_buy else '观察'}")
    print(f"综合分 {v.score:.1f}")
    print(f"怎么做：{v.when}")
    print("明细：")
    for r in v.reasons:
        print(" ", r)
    print("关键指标：")
    for k in [
        "收盘",
        "今日涨跌%",
        "三日累计%",
        "三日涨幅%",
        "三日量比",
        "横盘振幅%",
        "距60日低%",
        "箱体底",
        "箱体顶",
        "止损参考",
        "流通市值亿",
    ]:
        print(f"  {k}: {v.metrics.get(k)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="底部启动：三连小阳+底部横盘（严格）")
    parser.add_argument("code", nargs="?", help="单票评估")
    parser.add_argument("--scan", action="store_true")
    parser.add_argument("--asof", default=None)
    args = parser.parse_args()

    import daily_cache

    daily_cache.preload(force=False)
    if args.code and not args.scan:
        v = evaluate(args.code, asof=args.asof)
        print_verdict(v)
        return

    out = scan(asof=args.asof)
    OUT_SCAN.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_SCAN, index=False, encoding="utf-8-sig")
    n_buy = int((out["stage"] == "可买入").sum()) if not out.empty else 0
    print(f"底部启动 可买入 {n_buy} → {OUT_SCAN}")
    if not out.empty:
        cols = [
            c
            for c in [
                "code",
                "name",
                "stage",
                "score",
                "今日涨跌%",
                "三日累计%",
                "横盘振幅%",
                "距60日低%",
            ]
            if c in out.columns
        ]
        print(out[cols].to_string(index=False))


if __name__ == "__main__":
    main()
