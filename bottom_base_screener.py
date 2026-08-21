"""
底部横盘启动：深跌后缩量箱体 → 温和放量突破 → 吃小目标（默认 +3%）就走。

硬规则（可买入 = 回测「宽松」档，约 710 笔 / +3% 胜率 ~62%）：
  1. 距 60 日高点 ≤ −20%
  2. 启动前 15 日振幅 ≤ 14%，涨幅 ∈ [−15%, +8%]，最大连阳 ≤ 4
  3. 横盘均量 ≤ 20 日均量 × 1.0
  4. 启动日阳线涨幅 1.2%～8%，非涨停；量/20 均 ∈ [1.05, 2.5]
  5. 收盘突破前 15 日高点；贴近 20 日低（启动前 ≤ +10%）
  6. 收盘/MA20 ∈ [0.90, 1.08]；MA20 近 5 日斜率 ≤ 3%
  7. 主板非 ST，流通市值 25–700 亿

更严子集（距60高 ≤ −25% 且振幅 ≤ 12%）只加分，仍算可买入。

离场提示：止盈 +3%；跌破箱体底（前 15 日+启动日最低 ×0.998）止损；最多 5 日。

用法：
  .venv/bin/python bottom_base_screener.py 000912
  .venv/bin/python bottom_base_screener.py --scan --asof 2026-08-20
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
OUT_SCAN = ROOT / "data" / "bottom_base_signals.csv"

LIMIT = 0.095
MV_LO, MV_HI = 25.0, 700.0
# 可买入 = 宽松档（回测主样本）
DIST60_BUY = -0.20
RNG15_BUY = 0.14
# 更严子集仅用于加分
DIST60_TIGHT = -0.25
RNG15_TIGHT = 0.12
TP = 0.03
BOX_LOOKBACK = 15
R1_LO, R1_HI = 0.012, 0.08
VR_LO, VR_HI = 1.05, 2.5
PX_MA20_LO, PX_MA20_HI = 0.90, 1.08
NEAR_LOW_MAX = 0.10
R15_LO, R15_HI = -0.15, 0.08
STREAK_MAX = 4
VBASE_MAX = 1.0


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


def _max_up_streak(rets: np.ndarray) -> int:
    best = cur = 0
    for r in rets:
        if np.isfinite(r) and r > 0:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return int(best)


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
    if df is None or len(df) < 120:
        return _blank(code, name, industry, "", "无足够日线")
    asof_s = str(df["日期"].iloc[-1].date())
    if "ST" in name.upper() or name.startswith("*"):
        return _blank(code, name, industry, asof_s, "ST 跳过")

    px = df["px"].to_numpy(dtype=np.float64)
    hi = df["hi"].to_numpy(dtype=np.float64)
    lo = df["lo"].to_numpy(dtype=np.float64)
    op = df["op"].to_numpy(dtype=np.float64)
    vol = df["vol"].to_numpy(dtype=np.float64)
    ret = df["ret"].to_numpy(dtype=np.float64)
    fac = df["后复权因子"].to_numpy(dtype=np.float64)
    mv = float(df["mv"].iloc[-1])
    if not np.isfinite(mv) or mv < MV_LO or mv > MV_HI:
        return _blank(code, name, industry, asof_s, "市值不在 25–700 亿")
    if vol[-1] <= 0:
        return _blank(code, name, industry, asof_s, "停牌或无量")

    i = len(df) - 1
    if i < 120:
        return _blank(code, name, industry, asof_s, "日线太短")

    r1 = float(ret[i]) if np.isfinite(ret[i]) else 0.0
    if r1 >= LIMIT:
        return _blank(code, name, industry, asof_s, "当日涨停，不做尾盘追板")
    if not (R1_LO <= r1 <= R1_HI):
        return _blank(code, name, industry, asof_s, "启动日涨幅不在 1.2%～8%")
    if px[i] <= op[i]:
        return _blank(code, name, industry, asof_s, "启动日非阳线")

    hi60 = float(np.max(hi[i - 59 : i + 1]))
    dist60h = px[i] / hi60 - 1.0
    if dist60h > DIST60_BUY:
        return _blank(
            code,
            name,
            industry,
            asof_s,
            f"位置不够低（距60高 {dist60h * 100:.1f}%）",
        )

    lb = BOX_LOOKBACK
    box_lo_hfq = float(np.min(lo[i - lb : i + 1]))
    box_hi_hfq = float(np.max(hi[i - lb : i]))
    rng15 = box_hi_hfq / float(np.min(lo[i - lb : i])) - 1.0 if i >= lb else np.nan
    if not np.isfinite(rng15) or rng15 > RNG15_BUY:
        return _blank(
            code,
            name,
            industry,
            asof_s,
            f"前{lb}日振幅过大（{rng15 * 100:.1f}%）" if np.isfinite(rng15) else "振幅缺失",
        )

    r15 = float(px[i - 1] / px[i - 1 - lb] - 1.0) if px[i - 1 - lb] > 0 else np.nan
    if not np.isfinite(r15) or r15 > R15_HI or r15 < R15_LO:
        return _blank(code, name, industry, asof_s, "横盘期涨跌过大，不像箱体")

    streak = _max_up_streak(ret[i - lb : i])
    if streak > STREAK_MAX:
        return _blank(code, name, industry, asof_s, f"横盘期连阳偏多（{streak}）")

    vol_ma20 = rolling_mean(vol, 20)
    v20_prev = vol_ma20[i - 1]
    v_base = float(np.nanmean(vol[i - lb : i]))
    if not (np.isfinite(v20_prev) and v20_prev > 0 and v_base / v20_prev <= VBASE_MAX):
        return _blank(code, name, industry, asof_s, "横盘期未明显缩量")

    vr = vol[i] / v20_prev if v20_prev > 0 else np.nan
    if not (np.isfinite(vr) and VR_LO <= vr <= VR_HI):
        return _blank(
            code,
            name,
            industry,
            asof_s,
            f"启动量能不合适（量比 {vr:.2f}）" if np.isfinite(vr) else "量比缺失",
        )

    lo20 = float(np.min(lo[i - 19 : i]))
    near_low = px[i - 1] / lo20 - 1.0 if lo20 > 0 else np.nan
    if not np.isfinite(near_low) or near_low > NEAR_LOW_MAX:
        return _blank(code, name, industry, asof_s, "启动前离20日低偏远")

    if px[i] < box_hi_hfq * 0.995:
        return _blank(code, name, industry, asof_s, "未突破横盘高点")

    ma20 = float(df["ma20"].iloc[i]) if "ma20" in df.columns else float(rolling_mean(px, 20)[i])
    px_ma20 = px[i] / ma20 if np.isfinite(ma20) and ma20 > 0 else np.nan
    if not np.isfinite(px_ma20) or px_ma20 > PX_MA20_HI or px_ma20 < PX_MA20_LO:
        return _blank(code, name, industry, asof_s, "相对 MA20 过远/过弱")

    ma20_prev5 = float(df["ma20"].iloc[i - 5]) if i >= 5 else np.nan
    ma20_slope = ma20 / ma20_prev5 - 1.0 if np.isfinite(ma20_prev5) and ma20_prev5 > 0 else np.nan
    if np.isfinite(ma20_slope) and ma20_slope > 0.03:
        return _blank(code, name, industry, asof_s, "MA20 已陡升，偏中段")

    # 图表为前复权：px/fac[-1] 与 load_qfq_bars 同尺度
    fac_last = float(fac[i]) if np.isfinite(fac[i]) and fac[i] > 0 else 1.0
    box_bottom = round(box_lo_hfq / fac_last, 4)
    box_top = round(box_hi_hfq / fac_last, 4)
    close_raw = float(df["收盘"].iloc[i])
    target = round(close_raw * (1.0 + TP), 2)
    stop_hint = round(box_bottom * 0.998, 4)

    # 更严子集：只加分，仍全部可买入
    tight = dist60h <= DIST60_TIGHT and rng15 <= RNG15_TIGHT

    reasons = [
        f"启动阳线 {r1 * 100:.2f}%",
        f"距60日高点 {dist60h * 100:.1f}%",
        f"前{lb}日振幅 {rng15 * 100:.1f}%",
        f"量/20日均={vr:.2f}",
        f"箱体底 {box_bottom}",
    ]
    if tight:
        reasons.append("更严子集（距60高≤−25% 且振幅≤12%）")
    score = 40.0
    score += min(25.0, max(0.0, -dist60h - 0.20) * 50)
    score += min(15.0, max(0.0, RNG15_BUY - rng15) * 80)
    score += min(10.0, max(0.0, vr - 1.0) * 12)
    if tight:
        score += 12

    stage = "可买入"
    can_buy = True
    when = (
        f"建议：底部横盘启动，尾盘/次日弱开可跟；"
        f"止盈 +{TP * 100:.0f}%（约 {target}）；"
        f"跌破箱体底 {stop_hint} 离场；最多 5 个交易日"
    )

    metrics = {
        "收盘": round(close_raw, 2),
        "今日涨跌%": round(r1 * 100, 2),
        "距60日高点%": round(dist60h * 100, 1),
        "前15日振幅%": round(rng15 * 100, 1),
        "前15日涨幅%": round(r15 * 100, 1) if np.isfinite(r15) else None,
        "量能比1_20": round(float(vr), 2),
        "收盘/MA20": round(float(px_ma20), 3),
        "流通市值亿": round(mv, 1),
        "箱体底": box_bottom,
        "箱体顶": box_top,
        "止盈价": target,
        "止损参考": stop_hint,
        "strategy_ids": "bottom_base",
        "strategy_tags": "底部启动",
    }

    return Verdict(
        code,
        name,
        industry,
        asof_s,
        stage,
        can_buy,
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
        "距60日高点%",
        "前15日振幅%",
        "量能比1_20",
        "收盘/MA20",
        "箱体底",
        "箱体顶",
        "止盈价",
        "止损参考",
        "流通市值亿",
    ]:
        print(f"  {k}: {v.metrics.get(k)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="底部横盘启动（+3% 小目标）")
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
    print(f"底部启动 可买入 {n_buy} / 观察 {n_watch} → {OUT_SCAN}")
    if not out.empty:
        cols = [
            c
            for c in [
                "code",
                "name",
                "stage",
                "score",
                "今日涨跌%",
                "距60日高点%",
                "前15日振幅%",
                "量能比1_20",
                "箱体底",
                "止盈价",
            ]
            if c in out.columns
        ]
        print(out[cols].to_string(index=False))


if __name__ == "__main__":
    main()
