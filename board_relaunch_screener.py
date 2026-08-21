"""
板后重启：涨停 → 下跌回撤 → 横盘 → 再启动（可成交买点）。

入选列表（与大盘无关，过关就进池）：
  · 结构底座 + 启动日 K 线强（上影 ≤25%、收盘位 ≥70%）
  · 量比 1.2～2.0、启动涨 1.5%～7% 等（见下）

是否推荐买入：
  · 大盘偏强（主板等权日涨 ≥ +0.3%）→ 阶段「可买入」
  · 大盘偏弱 → 仍留在列表，阶段「观察」，理由写明「不推荐买入：大盘偏弱」

结构底座：
  1. 此前 8～35 日内有涨停
  2. 板后 15 日内最大回撤 ≥ 15%
  3. 谷底后横盘 8～20 日，振幅 ≤ 14%，期间不再涨停
  4. 启动日阳线突破箱顶；量/20 均 ∈ [1.2, 2.0]
  5. 主板非 ST，流通市值 25～700 亿

离场（回测默认）：
  · 止盈：相对买入价 +3%
  · 止损：跌破启动日最低价 × 0.998
  · 时间：最多持有 5 个交易日，先到先出

用法：
  .venv/bin/python board_relaunch_screener.py --scan
  .venv/bin/python board_relaunch_screener.py --market --asof 2026-08-21
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from functools import lru_cache

import numpy as np
import pandas as pd

from analyze_short_burst_features import load_maps
from volume_buy_screener import MAINBOARD_PREFIX, rolling_mean

ROOT = __import__("pathlib").Path(__file__).resolve().parent
OUT_SCAN = ROOT / "data" / "board_relaunch_signals.csv"

LIMIT = 0.095
MV_LO, MV_HI = 25.0, 700.0
TP = 0.03

DD_WIN = 15
DD_MIN = 0.15
CONSOL_LO, CONSOL_HI = 8, 20
RNG_MAX = 0.14
GAP_MIN = 8
GAP_MAX = 35
R1_LO, R1_HI = 0.015, 0.07
VR_LO, VR_HI = 1.2, 2.0
UPPER_MAX = 0.25
CLOSE_POS_MIN = 0.70
MKT_STRONG = 0.003  # +0.3%


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


@lru_cache(maxsize=8)
def market_ew_return(asof: str) -> float | None:
    """主板等权日涨跌（相对昨收）。asof=YYYY-MM-DD。"""
    import daily_cache

    asof_ts = pd.Timestamp(asof)
    rets: list[float] = []
    for code in daily_cache.cached_codes():
        if not code.startswith(MAINBOARD_PREFIX):
            continue
        df = daily_cache.get(code, asof=asof)
        if df is None or len(df) < 2:
            continue
        last = pd.Timestamp(df["日期"].iloc[-1])
        if last.normalize() != asof_ts.normalize():
            continue
        r = float(df["ret"].iloc[-1]) if "ret" in df.columns else np.nan
        if not np.isfinite(r):
            px = df["px"].to_numpy(dtype=np.float64)
            if len(px) < 2 or px[-2] <= 0:
                continue
            r = float(px[-1] / px[-2] - 1.0)
        if np.isfinite(r):
            rets.append(r)
    if len(rets) < 200:
        return None
    return float(np.mean(rets))


def market_status(asof: str) -> dict:
    r = market_ew_return(asof)
    if r is None:
        return {
            "asof": asof,
            "ew_ret": None,
            "ew_ret_pct": None,
            "strong": False,
            "threshold_pct": MKT_STRONG * 100,
            "note": "样本不足，无法判定大盘",
        }
    strong = r >= MKT_STRONG
    return {
        "asof": asof,
        "ew_ret": r,
        "ew_ret_pct": round(r * 100, 3),
        "strong": strong,
        "threshold_pct": MKT_STRONG * 100,
        "note": (
            f"主板等权 {r * 100:+.3f}% ≥ +{MKT_STRONG * 100:.1f}% → 偏强"
            if strong
            else f"主板等权 {r * 100:+.3f}% < +{MKT_STRONG * 100:.1f}% → 未偏强"
        ),
    }


def evaluate(
    code: str,
    name_map=None,
    ind_map=None,
    asof: str | None = None,
    mkt_strong: bool | None = None,
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

    if mkt_strong is None:
        st = market_status(asof_s)
        mkt_strong = bool(st["strong"])
        mkt_pct = st["ew_ret_pct"]
    else:
        st = market_status(asof_s)
        mkt_pct = st["ew_ret_pct"]

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
    r1 = float(ret[i]) if np.isfinite(ret[i]) else 0.0
    if r1 >= LIMIT:
        return _blank(code, name, industry, asof_s, "启动日涨停，不做追板")
    if not (R1_LO <= r1 <= R1_HI):
        return _blank(code, name, industry, asof_s, "启动日涨幅不在 1.5%～7%")
    if px[i] <= op[i]:
        return _blank(code, name, industry, asof_s, "启动日非阳线")

    rng1 = float(hi[i] - lo[i])
    if rng1 <= 0:
        return _blank(code, name, industry, asof_s, "启动日无振幅")
    upper = float(hi[i] - max(px[i], op[i])) / rng1
    close_pos = float(px[i] - lo[i]) / rng1
    candle_ok = upper <= UPPER_MAX and close_pos >= CLOSE_POS_MIN
    if not candle_ok:
        return _blank(
            code,
            name,
            industry,
            asof_s,
            f"启动K线不够强（上影{upper * 100:.0f}% 收盘位{close_pos * 100:.0f}%）",
        )

    vol_ma20 = rolling_mean(vol, 20)
    v20 = vol_ma20[i - 1] if i >= 1 else np.nan
    if not (np.isfinite(v20) and v20 > 0):
        return _blank(code, name, industry, asof_s, "均量缺失")
    vr = vol[i] / v20
    if not (VR_LO <= vr <= VR_HI):
        return _blank(code, name, industry, asof_s, f"量比不合适（{vr:.2f}）")

    # 向前找涨停锚点
    best = None
    for L in range(i - GAP_MIN, max(i - GAP_MAX - 1, 60), -1):
        if L < 60:
            break
        if not (np.isfinite(ret[L]) and ret[L] >= LIMIT):
            continue
        end_dd = min(L + DD_WIN, i - 1)
        if end_dd <= L + 2:
            continue
        trough = L + 1 + int(np.argmin(lo[L + 1 : end_dd + 1]))
        if trough >= i:
            continue
        dd = 1.0 - lo[trough] / px[L]
        if dd < DD_MIN:
            continue
        consol_len = i - trough
        if not (CONSOL_LO <= consol_len <= CONSOL_HI):
            continue
        box_start, box_end = trough, i - 1
        box_rets = ret[box_start : box_end + 1]
        if np.any(np.isfinite(box_rets) & (box_rets >= LIMIT)):
            continue
        box_hi = float(np.max(hi[box_start : box_end + 1]))
        box_lo = float(np.min(lo[box_start : box_end + 1]))
        if box_lo <= 0:
            continue
        rng = box_hi / box_lo - 1.0
        if rng > RNG_MAX:
            continue
        if px[box_end] > px[L] * 0.97:
            continue
        if px[i] < box_hi * 0.995:
            continue
        gap = i - L
        if gap < GAP_MIN:
            continue
        cand = dict(
            L=L,
            trough=trough,
            dd=dd,
            consol=consol_len,
            rng=rng,
            box_hi=box_hi,
            box_lo=box_lo,
            gap=gap,
        )
        # 取最近的合格涨停
        best = cand
        break

    if best is None:
        return _blank(code, name, industry, asof_s, "无合格「板后跌→横盘→启动」结构")

    fac_last = float(fac[i]) if np.isfinite(fac[i]) and fac[i] > 0 else 1.0
    box_bottom = round(best["box_lo"] / fac_last, 4)
    box_top = round(best["box_hi"] / fac_last, 4)
    close_raw = float(df["收盘"].iloc[i])
    target = round(close_raw * (1.0 + TP), 2)
    stop_hint = round(float(lo[i]) / fac_last * 0.998, 4)

    reasons = [
        f"距涨停 {best['gap']} 日",
        f"板后回撤 {best['dd'] * 100:.1f}%",
        f"横盘 {best['consol']} 日振幅 {best['rng'] * 100:.1f}%",
        f"启动 {r1 * 100:.2f}% 量比 {vr:.2f}",
        f"上影 {upper * 100:.0f}% 收盘位 {close_pos * 100:.0f}%",
        f"大盘等权 {mkt_pct:+.3f}%" if mkt_pct is not None else "大盘未知",
    ]
    score = 40.0
    score += min(20.0, (best["dd"] - 0.15) * 80)
    score += min(15.0, (CLOSE_POS_MIN and close_pos - 0.7) * 40)
    score += min(10.0, (UPPER_MAX - upper) * 40)
    score += min(10.0, max(0.0, 2.0 - abs(vr - 1.5)) * 5)
    if mkt_strong:
        score += 12

    metrics = {
        "收盘": round(close_raw, 2),
        "今日涨跌%": round(r1 * 100, 2),
        "量能比1_20": round(float(vr), 2),
        "上影占比%": round(upper * 100, 1),
        "收盘位置%": round(close_pos * 100, 1),
        "板后回撤%": round(best["dd"] * 100, 1),
        "横盘天数": int(best["consol"]),
        "横盘振幅%": round(best["rng"] * 100, 1),
        "距涨停日": int(best["gap"]),
        "大盘等权%": mkt_pct,
        "流通市值亿": round(mv, 1),
        "箱体底": box_bottom,
        "箱体顶": box_top,
        "止盈价": target,
        "止损参考": stop_hint,
        "strategy_ids": "board_relaunch",
        "strategy_tags": "板后重启",
    }

    exit_tip = (
        f"止盈 +{TP * 100:.0f}%（约 {target}）；"
        f"跌破启动日最低×0.998（约 {stop_hint}）止损；"
        f"最多持有 5 个交易日"
    )

    if mkt_strong:
        stage = "可买入"
        can_buy = True
        when = (
            f"建议：尾盘可跟，或次日开盘涨幅≤+1.5%再跟；{exit_tip}"
            f"（大盘等权 {mkt_pct:+.2f}% 偏强）"
        )
        hard = 1.0
    else:
        stage = "观察"
        can_buy = False
        mkt_txt = (
            f"{mkt_pct:+.2f}%" if mkt_pct is not None else "未知"
        )
        when = (
            f"不推荐买入：大盘偏弱（主板等权 {mkt_txt}"
            f" < +{MKT_STRONG * 100:.1f}%）；"
            f"形态已入选，等大盘转强或放弃；{exit_tip}"
        )
        reasons.append(f"不推荐买入：大盘偏弱（等权 {mkt_txt}）")
        hard = 0.5

    return Verdict(
        code,
        name,
        industry,
        asof_s,
        stage,
        can_buy,
        round(float(score), 1),
        hard,
        when,
        reasons,
        metrics,
    )


def scan(asof: str | None = None) -> pd.DataFrame:
    import daily_cache

    name_map, ind_map = load_maps()
    if asof is None:
        # 用任一只有数据的最新日
        sample = next(iter(daily_cache.cached_codes()), None)
        if sample is None:
            return pd.DataFrame()
        df0 = daily_cache.get(sample)
        asof = str(df0["日期"].iloc[-1].date()) if df0 is not None else None
    assert asof
    st = market_status(asof)
    mkt_strong = bool(st["strong"])
    print(f"[大盘] {st['note']}", flush=True)

    rows = []
    for code in daily_cache.cached_codes():
        if not code.startswith(MAINBOARD_PREFIX):
            continue
        if "ST" in name_map.get(code, "").upper():
            continue
        v = evaluate(code, name_map, ind_map, asof=asof, mkt_strong=mkt_strong)
        if v.stage in ("可买入", "观察"):
            rows.append(v.to_row())
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    order = {"可买入": 0, "观察": 1}
    out["_o"] = out["stage"].map(order).fillna(9)
    out = out.sort_values(["_o", "score"], ascending=[True, False])
    return out.drop(columns=["_o"]).reset_index(drop=True)


def print_verdict(v: Verdict) -> None:
    print(f"{v.code} {v.name}  {v.industry}")
    print(f"日期 {v.asof}")
    print(f"阶段：{v.stage}")
    print(f"综合分 {v.score:.1f}")
    print(f"怎么做：{v.when}")
    for r in v.reasons:
        print(" ", r)


def main() -> None:
    parser = argparse.ArgumentParser(description="板后重启（上影小+收强 × 大盘偏强）")
    parser.add_argument("code", nargs="?", help="单票评估")
    parser.add_argument("--scan", action="store_true")
    parser.add_argument("--market", action="store_true", help="只打印大盘等权判定")
    parser.add_argument("--asof", default=None)
    args = parser.parse_args()

    import daily_cache

    daily_cache.preload(force=False)

    asof = args.asof
    if asof is None and (args.market or args.code):
        sample = next(
            (c for c in daily_cache.cached_codes() if c.startswith("600")),
            None,
        )
        df0 = daily_cache.get(sample) if sample else None
        asof = str(df0["日期"].iloc[-1].date()) if df0 is not None else None

    if args.market:
        st = market_status(asof or "")
        print(st["note"])
        print(st)
        return

    if args.code and not args.scan:
        v = evaluate(args.code, asof=asof)
        print_verdict(v)
        return

    out = scan(asof=asof)
    OUT_SCAN.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_SCAN, index=False, encoding="utf-8-sig")
    n_buy = int((out["stage"] == "可买入").sum()) if not out.empty else 0
    n_watch = int((out["stage"] == "观察").sum()) if not out.empty else 0
    print(f"板后重启 可买入 {n_buy} / 观察 {n_watch} → {OUT_SCAN}")
    if not out.empty:
        cols = [
            c
            for c in [
                "code",
                "name",
                "stage",
                "score",
                "今日涨跌%",
                "上影占比%",
                "收盘位置%",
                "板后回撤%",
                "大盘等权%",
                "止盈价",
            ]
            if c in out.columns
        ]
        print(out[cols].to_string(index=False))


if __name__ == "__main__":
    main()
