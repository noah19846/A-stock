"""
低位无量首板：专门抓「次日续板」而不是底部启动。

硬规则：
  1. 当日涨停（主板 ≥9.5%）
  2. 真首板：昨日未涨停，且近 10 日仅今日一板
  3. 无量：量 / 20 日均量 < 0.95（可买入必须）；换手缩量只能进观察
  4. 低位：距 60 日高点 ≤ −15%（−15%～−8% 进观察）
  5. 近 20 日涨幅 ≤ 20%，收盘 / MA20 ≤ 1.15

日更会写入信号池「🟥 无量首板」tab。

用法：
  .venv/bin/python quiet_limit_screener.py 002953
  .venv/bin/python quiet_limit_screener.py --scan --asof 2026-08-17
  .venv/bin/python run_daily_pool.py --board-only
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_short_burst_features import load_maps
from volume_buy_screener import MAINBOARD_PREFIX, rolling_mean

ROOT = Path(__file__).resolve().parent
OUT_SCAN = ROOT / "data" / "quiet_limit_signals.csv"

LIMIT = 0.095
VOL_QUIET = 0.95
TURN_QUIET = 0.85
DIST60_BUY = -0.15
DIST60_WATCH = -0.08
R20_MAX = 0.20
PX_MA20_MAX = 1.15
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
    if df is None or len(df) < 80:
        return _blank(code, name, industry, "", "无足够日线")
    asof_s = str(df["日期"].iloc[-1].date())
    if "ST" in name.upper() or name.startswith("*"):
        return _blank(code, name, industry, asof_s, "ST 跳过")

    px = df["px"].to_numpy(dtype=np.float64)
    hi = df["hi"].to_numpy(dtype=np.float64)
    vol = df["vol"].to_numpy(dtype=np.float64)
    turn = df["turn"].to_numpy(dtype=np.float64)
    ret = df["ret"].to_numpy(dtype=np.float64)
    mv = float(df["mv"].iloc[-1])
    if not np.isfinite(mv) or mv < MV_LO or mv > MV_HI:
        return _blank(code, name, industry, asof_s, "市值不在 20–800 亿")
    if vol[-1] <= 0:
        return _blank(code, name, industry, asof_s, "停牌或无量")

    r1 = float(ret[-1]) if np.isfinite(ret[-1]) else 0.0
    if r1 < LIMIT:
        return _blank(code, name, industry, asof_s, "当日未涨停")

    prev_limit = bool(np.isfinite(ret[-2]) and ret[-2] >= LIMIT)
    n_limit10 = int(np.nansum(ret[-11:-1] >= LIMIT)) if len(ret) >= 11 else 99
    first = (not prev_limit) and n_limit10 == 0
    if not first:
        return _blank(
            code,
            name,
            industry,
            asof_s,
            "不是首板（昨日已涨停或近10日已有板）",
        )

    vol_ma20 = rolling_mean(vol, 20)
    v20 = vol_ma20[-1]
    vol_ratio = vol[-1] / v20 if np.isfinite(v20) and v20 > 0 else np.nan
    turn1 = float(turn[-1]) if np.isfinite(turn[-1]) else np.nan
    turn20 = float(np.nanmean(turn[-21:-1]))
    turn_ratio = (
        turn1 / turn20 if np.isfinite(turn1) and np.isfinite(turn20) and turn20 > 0 else np.nan
    )
    quiet = (np.isfinite(vol_ratio) and vol_ratio < VOL_QUIET) or (
        np.isfinite(turn_ratio) and turn_ratio < TURN_QUIET
    )
    if not quiet:
        return _blank(
            code,
            name,
            industry,
            asof_s,
            f"放量板（量比 {vol_ratio:.2f}），不是无量首板",
            {"量能比1_20": round(float(vol_ratio), 2) if np.isfinite(vol_ratio) else None},
        )

    hi60 = float(np.max(hi[-60:]))
    dist60h = px[-1] / hi60 - 1.0
    if dist60h > DIST60_WATCH:
        return _blank(
            code,
            name,
            industry,
            asof_s,
            f"高位板（距60日高点 {dist60h * 100:.1f}%），续板差",
        )

    r5 = float(px[-1] / px[-6] - 1.0) if px[-6] > 0 else np.nan
    r20 = float(px[-1] / px[-21] - 1.0) if px[-21] > 0 else np.nan
    ma20 = float(df["ma20"].iloc[-1]) if "ma20" in df.columns else np.nan
    px_ma20 = px[-1] / ma20 if np.isfinite(ma20) and ma20 > 0 else np.nan

    metrics = {
        "收盘": round(float(df["收盘"].iloc[-1]), 2),
        "今日涨跌%": round(r1 * 100, 2),
        "前5日涨幅%": round(r5 * 100, 2) if np.isfinite(r5) else None,
        "前20日涨幅%": round(r20 * 100, 2) if np.isfinite(r20) else None,
        "距60日高点%": round(dist60h * 100, 1),
        "量能比1_20": round(float(vol_ratio), 2) if np.isfinite(vol_ratio) else None,
        "换手率%": round(turn1, 2) if np.isfinite(turn1) else None,
        "换手比20": round(float(turn_ratio), 2) if np.isfinite(turn_ratio) else None,
        "收盘/MA20": round(float(px_ma20), 3) if np.isfinite(px_ma20) else None,
        "流通市值亿": round(mv, 1),
        "近10日板数": n_limit10,
        "strategy_ids": "quiet_first",
        "strategy_tags": "无量首板",
    }

    low = dist60h <= DIST60_BUY
    not_hot20 = (not np.isfinite(r20)) or r20 <= R20_MAX
    not_stretched = (not np.isfinite(px_ma20)) or px_ma20 <= PX_MA20_MAX
    very_quiet = np.isfinite(vol_ratio) and vol_ratio < VOL_QUIET

    reasons = [
        f"首板涨停 {r1 * 100:.2f}%",
        f"量/20日均={vol_ratio:.2f}" if np.isfinite(vol_ratio) else "量比缺失",
        f"距60日高点 {dist60h * 100:.1f}%",
    ]
    if np.isfinite(turn_ratio):
        reasons.append(f"换手/20日换手={turn_ratio:.2f}")

    score = 40.0
    score += max(0.0, VOL_QUIET - (vol_ratio if np.isfinite(vol_ratio) else VOL_QUIET)) * 80
    score += min(25.0, max(0.0, -dist60h - 0.15) * 80)
    if very_quiet:
        score += 8
    if not_hot20:
        score += 6
    if not_stretched:
        score += 4

    if low and not_hot20 and not_stretched and very_quiet:
        stage = "可买入"
        can_buy = True
        when = (
            "建议：低位无量首板，看次日缩量续板；"
            "次日高开可跟，放量长阴或开盘弱于 −2% 放弃"
        )
    elif low or dist60h <= DIST60_WATCH:
        stage = "观察"
        can_buy = False
        why_bits = []
        if not very_quiet:
            why_bits.append("量能未明显缩（主要靠换手）")
        if not low:
            why_bits.append("位置偏高（未到 −15%）")
        if not not_hot20:
            why_bits.append("近20日已大涨")
        if not not_stretched:
            why_bits.append("远离 MA20")
        when = "等待量价确认：" + ("；".join(why_bits) if why_bits else "条件略松")
        reasons.extend(why_bits)
    else:
        return _blank(code, name, industry, asof_s, "首板但位置/热度不够干净", metrics)

    return Verdict(
        code,
        name,
        industry,
        asof_s,
        stage,
        can_buy,
        round(float(score), 1),
        1.0 if can_buy else 0.5,
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
    print(f"阶段：{v.stage}    {'续板候选' if v.can_buy else '观察'}")
    print(f"综合分 {v.score:.1f}")
    print(f"怎么做：{v.when}")
    print("明细：")
    for r in v.reasons:
        print(" ", r)
    print("关键指标：")
    for k in [
        "收盘",
        "今日涨跌%",
        "前20日涨幅%",
        "距60日高点%",
        "量能比1_20",
        "换手率%",
        "换手比20",
        "收盘/MA20",
        "流通市值亿",
    ]:
        print(f"  {k}: {v.metrics.get(k)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="低位无量首板（次日续板）")
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
    n_watch = int((out["stage"] == "观察").sum()) if not out.empty else 0
    print(f"无量首板 可买入 {n_buy} / 观察 {n_watch} → {OUT_SCAN}")
    if not out.empty:
        cols = [
            c
            for c in [
                "code",
                "name",
                "stage",
                "score",
                "今日涨跌%",
                "量能比1_20",
                "距60日高点%",
                "前20日涨幅%",
            ]
            if c in out.columns
        ]
        print(out[cols].to_string(index=False))


if __name__ == "__main__":
    main()
