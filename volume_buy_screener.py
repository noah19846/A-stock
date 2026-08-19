"""
量价买点筛选：把成交量口诀落成日线硬规则（独立于长/短/宝藏）。

形态：
  地量价稳 / 低位缩量双谷 / 缩量后放量启动 /
  盘整后连续放量小涨 / 无量涨停 / 低位放量涨停

用法：
  .venv/bin/python volume_buy_screener.py 601882
  .venv/bin/python volume_buy_screener.py --scan
  .venv/bin/python volume_buy_screener.py --scan --asof 2026-08-18
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_short_burst_features import load_maps

ROOT = Path(__file__).resolve().parent
OUT_SCAN = ROOT / "data" / "volume_buy_signals.csv"
MAINBOARD_PREFIX = ("600", "601", "603", "605", "000", "001", "002")

PATTERN_TITLE = {
    "dry_stable": "地量价稳",
    "double_trough": "缩量双谷",
    "expand_after_dry": "缩量后启动",
    "consol_vol_up": "盘整放量",
    "quiet_limit_up": "无量涨停",
    "low_limit_up": "低位放量涨停",
}


def rolling_mean(x: np.ndarray, w: int) -> np.ndarray:
    n = len(x)
    out = np.full(n, np.nan)
    if n < w:
        return out
    c = np.cumsum(x, dtype=np.float64)
    out[w - 1] = c[w - 1] / w
    if n > w:
        out[w:] = (c[w:] - c[:-w]) / w
    return out


def local_minima(x: np.ndarray, order: int = 3) -> list[int]:
    idx: list[int] = []
    n = len(x)
    for i in range(order, n - order):
        w = x[i - order : i + order + 1]
        if np.any(~np.isfinite(w)):
            continue
        if x[i] == np.min(w) and x[i] < x[i - 1] and x[i] <= x[i + 1]:
            idx.append(i)
    return idx


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


def detect_patterns(df: pd.DataFrame) -> tuple[list[dict], dict] | None:
    n = len(df)
    if n < 80:
        return None
    px = df["px"].to_numpy(dtype=np.float64)
    hi = df["hi"].to_numpy(dtype=np.float64)
    lo = df["lo"].to_numpy(dtype=np.float64)
    vol = df["vol"].to_numpy(dtype=np.float64)
    turn = df["turn"].to_numpy(dtype=np.float64)
    ret = df["ret"].to_numpy(dtype=np.float64)
    mv = float(df["mv"].iloc[-1])
    if not np.isfinite(mv) or mv < 20 or mv > 800:
        return None
    if vol[-1] <= 0 or not np.isfinite(vol[-1]):
        return None

    vol_ma20 = rolling_mean(vol, 20)
    vol_ma10 = rolling_mean(vol, 10)
    v20 = vol_ma20[-1]
    if not np.isfinite(v20) or v20 <= 0:
        return None

    r1 = float(ret[-1]) if np.isfinite(ret[-1]) else 0.0
    r3 = float(px[-1] / px[-4] - 1.0) if px[-4] > 0 else np.nan
    r5 = float(px[-1] / px[-6] - 1.0) if px[-6] > 0 else np.nan
    r20 = float(px[-1] / px[-21] - 1.0) if px[-21] > 0 else np.nan
    hi60 = float(np.max(hi[-60:]))
    lo60 = float(np.min(lo[-60:]))
    dist60h = px[-1] / hi60 - 1.0
    dist60l = px[-1] / lo60 - 1.0
    rng30 = (float(np.max(hi[-30:])) - float(np.min(lo[-30:]))) / float(
        np.mean(px[-30:])
    )
    vol_ratio = vol[-1] / v20
    vol5 = float(np.mean(vol[-5:]))
    vol60 = float(np.mean(vol[-60:]))
    vol5_60 = vol5 / vol60 if vol60 > 0 else np.nan
    vol_pct = float(np.mean(vol[-60:] <= vol[-1]))
    turn1 = float(turn[-1]) if np.isfinite(turn[-1]) else np.nan
    turn20 = float(np.nanmean(turn[-21:-1]))

    # 高价区巨量滞涨 / 高位长阴 → 不当买点
    if dist60h > -0.06 and vol_ratio > 2.0 and abs(r1) < 0.02:
        return [], {}
    if dist60h > -0.08 and r1 < -0.05 and vol_ratio > 1.5:
        return [], {}

    patterns: list[dict] = []

    no_new_low = float(np.min(lo[-2:])) > float(np.min(lo[-22:-2])) * 1.002
    dry = np.isfinite(vol5_60) and vol5_60 < 0.55 and vol_pct <= 0.22
    decline_slow = (np.isfinite(r5) and np.isfinite(r20) and r5 > r20) or (
        np.isfinite(r5) and abs(r5) < 0.03
    )
    lowish = dist60h <= -0.10 and dist60l <= 0.18
    if dry and no_new_low and decline_slow and lowish and r1 > -0.035:
        score = (0.55 - vol5_60) * 80 + (0.22 - vol_pct) * 60
        score += max(0.0, 0.18 - dist60l) * 80
        if np.isfinite(r5) and -0.02 <= r5 <= 0.04:
            score += 12
        patterns.append(
            {
                "id": "dry_stable",
                "score": float(score),
                "why": "近5日量能处于地量区，近2日未再创新低，跌势趋缓",
            }
        )

    dry_window = float(np.min(vol[-18:-5])) / vol60 if vol60 > 0 else np.nan
    expand = (
        float(np.mean(vol[-3:])) / float(np.mean(vol[-23:-3]))
        if np.mean(vol[-23:-3]) > 0
        else np.nan
    )
    ma10_now, ma10_prev = vol_ma10[-1], vol_ma10[-8]
    ma_rising = (
        np.isfinite(ma10_now)
        and np.isfinite(ma10_prev)
        and ma10_now > ma10_prev * 1.08
    )
    ma_was_flat = False
    if np.all(np.isfinite(vol_ma10[-18:-8])):
        sl = vol_ma10[-18:-8]
        ma_was_flat = (np.std(sl) / np.mean(sl)) < 0.18 if np.mean(sl) > 0 else False
    slight_up = np.isfinite(r3) and 0.005 <= r3 <= 0.07
    expand_ok = (
        np.isfinite(dry_window)
        and dry_window < 0.50
        and np.isfinite(expand)
        and expand >= 1.35
        and slight_up
        and dist60h <= -0.08
        and r1 > -0.02
        and vol_ratio >= 1.15
        and (not np.isfinite(turn1) or turn1 <= 15)
        and (not np.isfinite(r5) or -8 <= r5 * 100 <= 15)
    )
    if expand_ok:
        score = (0.50 - dry_window) * 70 + (expand - 1.35) * 40
        score += (0.07 - abs(r3 - 0.03)) * 80
        if ma_rising:
            score += 10
        if ma_was_flat:
            score += 12
        patterns.append(
            {
                "id": "expand_after_dry",
                "score": float(score),
                "why": "地量后近3日量能抬升、股价小幅上扬",
            }
        )

    consec = 0
    for k in range(1, 4):
        vk = (
            vol[-k] / vol_ma20[-k]
            if np.isfinite(vol_ma20[-k]) and vol_ma20[-k] > 0
            else 0
        )
        rk = ret[-k] if np.isfinite(ret[-k]) else 0
        if vk >= 1.55 and 0.004 <= rk <= 0.045:
            consec += 1
        else:
            break
    if consec >= 2 and rng30 <= 0.22 and dist60h <= -0.04 and r1 < 0.07:
        score = (0.22 - rng30) * 120 + consec * 18 + min(vol_ratio, 3) * 8
        patterns.append(
            {
                "id": "consol_vol_up",
                "score": float(score),
                "why": f"近30日振幅{rng30 * 100:.1f}%，连续{consec}日放量且股价小幅上扬",
            }
        )

    limit = r1 >= 0.095
    prev_limit = bool(np.isfinite(ret[-2]) and ret[-2] >= 0.095)
    quiet_limit = limit and (
        (np.isfinite(turn20) and turn20 > 0 and turn1 < 0.75 * turn20)
        or vol_ratio < 0.95
    )
    if quiet_limit and not prev_limit and dist60h <= 0.0:
        score = 55 + (0.95 - min(vol_ratio, 0.95)) * 40
        if dist60h <= -0.10:
            score += 10
        patterns.append(
            {
                "id": "quiet_limit_up",
                "score": float(score),
                "why": f"首个涨停且量能偏弱（量/20日均={vol_ratio:.2f}）",
            }
        )

    if limit and dist60h <= -0.15 and vol_ratio >= 1.70 and not prev_limit:
        score = 50 + min(vol_ratio, 4) * 8 + min(-dist60h, 0.4) * 40
        patterns.append(
            {
                "id": "low_limit_up",
                "score": float(score),
                "why": f"距60日高点{dist60h * 100:.1f}%，放量涨停",
            }
        )

    vma = rolling_mean(vol, 5)
    mins = [i for i in local_minima(vma[-80:], order=4) if np.isfinite(vma[-80:][i])]
    if len(mins) >= 2:
        t1, t2 = mins[-2], mins[-1]
        gap = t2 - t1
        base = n - 80
        i1, i2 = base + t1, base + t2
        v1, v2 = float(vma[i1]), float(vma[i2])
        p1, p2 = float(px[i1]), float(px[i2])
        if (
            gap >= 8
            and v1 > 0
            and v2 < v1 * 0.88
            and abs(p2 / p1 - 1.0) <= 0.09
            and dist60h <= -0.10
            and np.isfinite(r3)
            and 0.0 <= r3 <= 0.08
            and vol_ratio < 1.8
            and i2 >= n - 12
        ):
            score = (1.0 - v2 / v1) * 80 + (0.09 - abs(p2 / p1 - 1)) * 100 + 10
            patterns.append(
                {
                    "id": "double_trough",
                    "score": float(score),
                    "why": f"第二量谷为第一谷的{v2 / v1 * 100:.0f}%，价位接近，近端小幅回升",
                }
            )

    metrics = {
        "收盘": round(float(df["收盘"].iloc[-1]), 2),
        "今日涨跌%": round(r1 * 100, 2),
        "前5日涨幅%": round(r5 * 100, 2) if np.isfinite(r5) else None,
        "前20日涨幅%": round(r20 * 100, 2) if np.isfinite(r20) else None,
        "距60日高点%": round(dist60h * 100, 1),
        "距60日低点%": round(dist60l * 100, 1),
        "量能比1_20": round(float(vol_ratio), 2),
        "量能比5_60": round(float(vol5_60), 2) if np.isfinite(vol5_60) else None,
        "当日量分位%": round(vol_pct * 100, 1),
        "换手率%": round(turn1, 2) if np.isfinite(turn1) else None,
        "近30日振幅%": round(rng30 * 100, 1),
        "流通市值亿": round(mv, 1),
    }
    return patterns, metrics


def classify(patterns: list[dict], metrics: dict) -> tuple[str, bool]:
    ids = {p["id"] for p in patterns}
    n = len(patterns)
    vol5_60 = metrics.get("量能比5_60")
    dist60h = metrics.get("距60日高点%")
    buy_ids = {
        "expand_after_dry",
        "consol_vol_up",
        "quiet_limit_up",
        "low_limit_up",
    }
    if n >= 2 or (ids & buy_ids):
        return "可买入", True
    if (
        "dry_stable" in ids
        and vol5_60 is not None
        and vol5_60 <= 0.40
        and dist60h is not None
        and dist60h <= -15
    ):
        return "观察", False
    return "不关注", False


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
        return Verdict(
            code, name, industry, "", "不关注", False, 0.0, 0.0, "无足够日线", ["缺少数据"], {}
        )
    asof_s = str(df["日期"].iloc[-1].date())
    if "ST" in name.upper() or name.startswith("*"):
        return Verdict(
            code, name, industry, asof_s, "不关注", False, 0.0, 0.0, "ST 跳过", [], {}
        )

    detected = detect_patterns(df)
    if detected is None:
        return Verdict(
            code, name, industry, asof_s, "不关注", False, 0.0, 0.0, "指标不足", [], {}
        )
    patterns, metrics = detected
    if not patterns:
        return Verdict(
            code, name, industry, asof_s, "不关注", False, 0.0, 0.0, "无量价买点", [], metrics
        )

    patterns.sort(key=lambda x: -x["score"])
    stage, can_buy = classify(patterns, metrics)
    if stage == "不关注":
        return Verdict(
            code, name, industry, asof_s, "不关注", False, 0.0, 0.0, "形态不够干净", [], metrics
        )

    total = sum(p["score"] for p in patterns) + (len(patterns) - 1) * 8
    titles = [PATTERN_TITLE[p["id"]] for p in patterns]
    why = "；".join(p["why"] for p in patterns)
    extra = {
        "strategy_ids": "|".join(p["id"] for p in patterns),
        "strategy_tags": "|".join(titles),
        "n_pat": len(patterns),
    }
    metrics = {**metrics, **extra}
    when = ("建议：" if can_buy else "等待量价确认：") + why
    return Verdict(
        code,
        name,
        industry,
        asof_s,
        stage,
        can_buy,
        round(float(total), 1),
        float(len(patterns)),
        when,
        [PATTERN_TITLE[p["id"]] + "：" + p["why"] for p in patterns],
        metrics,
    )


def scan(asof: str | None = None, watch_top_k: int = 12) -> pd.DataFrame:
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
    print(f"综合分 {v.score:.1f}   形态数 {v.hard_score:.0f}")
    print(f"怎么做：{v.when}")
    print("明细：")
    for r in v.reasons:
        print(" ", r)
    print("关键指标：")
    for k in [
        "收盘",
        "今日涨跌%",
        "前5日涨幅%",
        "距60日高点%",
        "量能比1_20",
        "量能比5_60",
        "换手率%",
        "流通市值亿",
        "strategy_tags",
    ]:
        print(f"  {k}: {v.metrics.get(k)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="量价买点筛选")
    parser.add_argument("code", nargs="?", help="单票评估")
    parser.add_argument("--scan", action="store_true")
    parser.add_argument("--asof", default=None)
    args = parser.parse_args()

    if args.code and not args.scan:
        import daily_cache

        daily_cache.preload(force=False)
        v = evaluate(args.code, asof=args.asof)
        print_verdict(v)
        return

    import daily_cache

    daily_cache.preload(force=False)
    out = scan(asof=args.asof)
    OUT_SCAN.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_SCAN, index=False, encoding="utf-8-sig")
    n_buy = int((out["stage"] == "可买入").sum()) if not out.empty else 0
    n_watch = int((out["stage"] == "观察").sum()) if not out.empty else 0
    print(f"量价 可买入 {n_buy} / 观察 {n_watch} → {OUT_SCAN}")
    if not out.empty:
        cols = [
            c
            for c in ["code", "name", "stage", "strategy_tags", "score", "今日涨跌%", "距60日高点%"]
            if c in out.columns
        ]
        print(out[cols].to_string(index=False))


if __name__ == "__main__":
    main()
