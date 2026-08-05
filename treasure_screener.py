"""
宝藏观察池：阶段底 → 数月内可能翻倍的技术面严选（独立于长短线）

只做日线技术过滤；不做消息/基本面。默认只输出「宝藏观察」，入场极严。

规则：data/treasure_bands.json

用法：
  .venv/bin/python treasure_screener.py 002583
  .venv/bin/python treasure_screener.py --scan
  .venv/bin/python treasure_screener.py --scan --asof 2026-03-27
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
BANDS_PATH = ROOT / "data" / "treasure_bands.json"
LIST = ROOT / "data" / "stock_list.csv"
INDUSTRY = ROOT / "data" / "stock_industry.csv"
OUT_SCAN = ROOT / "data" / "treasure_signals.csv"


def load_bands(path: Path | None = None) -> dict:
    p = path or BANDS_PATH
    if not p.exists():
        raise SystemExit(f"缺少 {p}")
    return json.loads(p.read_text(encoding="utf-8"))


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


def metrics_at(df: pd.DataFrame, i: int | None = None) -> dict | None:
    if i is None:
        i = len(df) - 1
    if i < 60:
        return None
    px = df["px"].to_numpy(dtype=float)
    amt = df["amt"].to_numpy(dtype=float)
    turn = df["turn"].to_numpy(dtype=float)
    mv = df["mv"].to_numpy(dtype=float)
    close = float(px[i])
    ma5 = float(df["ma5"].iloc[i])
    ma10 = float(df["ma10"].iloc[i])
    ma20 = float(df["ma20"].iloc[i])
    ma60 = float(df["ma60"].iloc[i])
    if not all(np.isfinite([close, ma5, ma10, ma20, ma60])) or ma20 <= 0 or ma60 <= 0:
        return None

    def ret_n(n: int) -> float:
        return float(px[i] / px[i - n] - 1.0) * 100.0

    amt5 = float(np.nanmean(amt[i - 4 : i + 1]))
    amt20 = float(np.nanmean(amt[i - 19 : i + 1]))
    hi60 = float(np.nanmax(px[i - 59 : i + 1]))
    hi120 = float(np.nanmax(px[max(0, i - 119) : i + 1]))
    low40 = float(np.nanmin(px[i - 39 : i + 1]))
    rets40 = np.diff(px[i - 40 : i + 1]) / px[i - 40 : i]
    vol40 = float(np.std(rets40) * 100) if len(rets40) > 5 else np.nan
    quiet40 = float(np.mean(np.abs(rets40) < 0.01) * 100) if len(rets40) else np.nan
    turn_i = float(turn[i])
    # daily_cache turn 多为小数占比
    if np.isfinite(turn_i) and turn_i < 0.5:
        turn_i *= 100.0

    return {
        "收盘": round(float(df["收盘"].iloc[i]), 3),
        "今日涨跌%": round(float(df["ret"].iloc[i]) * 100, 3)
        if np.isfinite(df["ret"].iloc[i])
        else np.nan,
        "px_ma5": round(close / ma5, 4),
        "px_ma20": round(close / ma20, 4),
        "px_ma60": round(close / ma60, 4),
        "ma10_gt_ma20": bool(ma10 > ma20),
        "above_ma20": bool(close > ma20),
        "距60日高点%": round((close / hi60 - 1.0) * 100.0, 3),
        "距120日高点%": round((close / hi120 - 1.0) * 100.0, 3),
        "距40日低点%": round((close / low40 - 1.0) * 100.0, 3),
        "前5日涨幅%": round(ret_n(5), 3),
        "前20日涨幅%": round(ret_n(20), 3),
        "前40日涨幅%": round(ret_n(40), 3),
        "前60日涨幅%": round(ret_n(60), 3),
        "额能比5_20": round(amt5 / amt20, 4) if amt20 > 0 else np.nan,
        "换手率%": round(turn_i, 3),
        "流通市值亿": round(float(mv[i]), 3),
        "前40日波动%": round(vol40, 3) if np.isfinite(vol40) else np.nan,
        "安静日占比40%": round(quiet40, 3) if np.isfinite(quiet40) else np.nan,
    }


def _in_range(x: float, lo: float, hi: float) -> bool:
    return np.isfinite(x) and lo <= x <= hi


def score_hard(m: dict, hard: dict) -> tuple[float, list[str], list[str]]:
    """返回 (通过率0~1, 通过明细, 未过明细)。"""
    ok_msgs: list[str] = []
    bad_msgs: list[str] = []
    checks: list[bool] = []

    def add(cond: bool, msg: str) -> None:
        checks.append(cond)
        (ok_msgs if cond else bad_msgs).append(("✓ " if cond else "✗ ") + msg)

    px20 = hard["px_ma20"]
    add(not bool(m["above_ma20"]), "收盘未站上 MA20（宝藏底，不是长线 Entry）")
    if hard.get("ma10_gt_ma20") is False:
        add(not bool(m["ma10_gt_ma20"]), "MA10 尚未上穿 MA20（趋势未提前转多）")
    if "px_ma5" in hard:
        px5 = hard["px_ma5"]
        add(
            _in_range(m["px_ma5"], float(px5[0]), float(px5[1])),
            f"收盘/MA5={m['px_ma5']:.3f} ∈[{px5[0]},{px5[1]}]",
        )
    add(
        _in_range(m["px_ma20"], float(px20[0]), float(px20[1])),
        f"收盘/MA20={m['px_ma20']:.3f} ∈[{px20[0]},{px20[1]}]",
    )

    d60 = hard["dist60_hi_pct"]
    add(
        _in_range(m["距60日高点%"], float(d60[0]), float(d60[1])),
        f"距60高={m['距60日高点%']:.1f}% ∈[{d60[0]},{d60[1]}]",
    )
    r20 = hard["ret20_pct"]
    add(
        _in_range(m["前20日涨幅%"], float(r20[0]), float(r20[1])),
        f"前20日={m['前20日涨幅%']:.1f}% ∈[{r20[0]},{r20[1]}]",
    )
    r5 = hard["ret5_pct"]
    add(
        _in_range(m["前5日涨幅%"], float(r5[0]), float(r5[1])),
        f"前5日={m['前5日涨幅%']:.1f}% ∈[{r5[0]},{r5[1]}]",
    )
    vol = hard["vol40_pct"]
    add(
        _in_range(m["前40日波动%"], float(vol[0]), float(vol[1])),
        f"40日波动={m['前40日波动%']:.2f}% ∈[{vol[0]},{vol[1]}]",
    )
    add(
        np.isfinite(m["安静日占比40%"])
        and m["安静日占比40%"] <= float(hard["quiet40_pct_hi"]),
        f"安静日占比={m['安静日占比40%']:.1f}% ≤{hard['quiet40_pct_hi']}",
    )
    ar = hard["amt_ratio_5_20"]
    add(
        _in_range(m["额能比5_20"], float(ar[0]), float(ar[1])),
        f"额能比={m['额能比5_20']:.2f} ∈[{ar[0]},{ar[1]}]",
    )
    mv = hard["mv_yi"]
    add(
        _in_range(m["流通市值亿"], float(mv[0]), float(mv[1])),
        f"流通市值={m['流通市值亿']:.0f}亿 ∈[{mv[0]},{mv[1]}]",
    )
    turn = hard["turnover_pct"]
    add(
        _in_range(m["换手率%"], float(turn[0]), float(turn[1])),
        f"换手={m['换手率%']:.2f}% ∈[{turn[0]},{turn[1]}]",
    )

    near = float(hard.get("near_low40_pct_hi", 3.0))
    add(
        np.isfinite(m["距40日低点%"]) and m["距40日低点%"] <= near,
        f"距40日低={m['距40日低点%']:.2f}% ≤{near}%（阶段底附近）",
    )

    # 可选：距120高不要贴新高
    if "dist120_hi_pct" in hard:
        d120 = hard["dist120_hi_pct"]
        add(
            _in_range(m["距120日高点%"], float(d120[0]), float(d120[1])),
            f"距120高={m['距120日高点%']:.1f}% ∈[{d120[0]},{d120[1]}]",
        )

    rate = float(sum(1 for c in checks if c) / max(len(checks), 1))
    return rate, ok_msgs, bad_msgs


def soft_rank(m: dict) -> float:
    """同池内排序：更深、更噪、更贴低点略加分（非硬门槛）。"""
    s = 0.0
    if np.isfinite(m["距60日高点%"]):
        s += min(12.0, max(0.0, -m["距60日高点%"] - 10.0)) * 0.08
    if np.isfinite(m["距40日低点%"]):
        s += max(0.0, 3.0 - m["距40日低点%"]) * 0.15
    if np.isfinite(m["前40日波动%"]):
        s += min(2.0, max(0.0, m["前40日波动%"] - 1.7)) * 0.2
    if np.isfinite(m["安静日占比40%"]):
        s += max(0.0, 45.0 - m["安静日占比40%"]) * 0.02
    if np.isfinite(m["流通市值亿"]) and m["流通市值亿"] < 80:
        s += 0.3
    return float(s)


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
    if df is None or len(df) < 80:
        return Verdict(
            code, name, industry, "", "不关注", False, 0.0, 0.0, "无足够日线", ["缺少数据"], {}
        )

    i = len(df) - 1
    asof_s = str(df["日期"].iloc[i].date())
    m = metrics_at(df, i)
    if m is None:
        return Verdict(
            code, name, industry, asof_s, "不关注", False, 0.0, 0.0, "指标不足", [], {}
        )

    hard = bands.get("hard") or {}
    hard_rate, ok_msgs, bad_msgs = score_hard(m, hard)
    soft = soft_rank(m)
    all_hard = len(bad_msgs) == 0

    # 过热：近端已大涨 / 贴近新高 → 不是宝藏底
    if m["距60日高点%"] > -5 or m["前20日涨幅%"] > 8 or m["above_ma20"]:
        stage = "已离底"
        when = "已离开阶段底画像（偏强或站上均线），转去长/短线框架看，不进宝藏观察池"
        reasons = bad_msgs + ok_msgs
        return Verdict(
            code,
            name,
            industry,
            asof_s,
            stage,
            False,
            round(soft, 3),
            round(hard_rate, 3),
            when,
            reasons,
            m,
        )

    if all_hard:
        stage = "宝藏观察"
        when = (
            "技术面阶段底严选已齐：深回撤、贴40日低、仍在MA20下、波动偏噪、量能未失控。"
            "仅观察池——等你自己的消息/主题确认后再考虑介入；"
            "介入后按中长线趋势持有（非短线几天结算），突破失效再议。"
        )
        reasons = ["【硬条件全部满足】"] + ok_msgs
        can_buy = False  # 观察池，不自动给可买
    else:
        stage = "不关注"
        when = "未满足宝藏硬条件"
        reasons = bad_msgs + ok_msgs
        can_buy = False

    return Verdict(
        code,
        name,
        industry,
        asof_s,
        stage,
        can_buy,
        round(soft + hard_rate, 3),
        round(hard_rate, 3),
        when,
        reasons,
        m,
    )


def scan(
    *,
    asof: str | None = None,
    bands_path: Path | None = None,
    include_left: bool = False,
) -> pd.DataFrame:
    """默认只输出宝藏观察。"""
    import daily_cache

    bands = load_bands(bands_path)
    name_map, ind_map = load_maps()
    rows = []
    codes = daily_cache.cached_codes()
    for code in codes:
        if not code.startswith(("600", "601", "603", "605", "000", "001", "002")):
            continue
        if "ST" in name_map.get(code, "").upper():
            continue
        v = evaluate(code, bands, name_map, ind_map, asof=asof)
        if v.stage == "宝藏观察" or (include_left and v.stage == "已离底"):
            rows.append(v.to_row())
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    order = {"宝藏观察": 0, "已离底": 1}
    out["_o"] = out["stage"].map(order).fillna(9)
    out = out.sort_values(["_o", "score", "hard_score"], ascending=[True, False, False])
    out = out.drop(columns=["_o"]).reset_index(drop=True)
    top_k = int(bands.get("daily_top_k") or 0)
    if top_k > 0 and "宝藏观察" in set(out["stage"]):
        watch = out[out["stage"] == "宝藏观察"]
        other = out[out["stage"] != "宝藏观察"]
        watch = watch.head(top_k)
        out = pd.concat([watch, other], ignore_index=True)
    return out


def print_verdict(v: Verdict) -> None:
    print(f"{v.code} {v.name}  {v.industry}")
    print(f"日期 {v.asof}")
    print(f"阶段：{v.stage}    观察池（不自动给买入）")
    print(f"综合分 {v.score:.3f}   硬条件分 {v.hard_score:.3f}")
    print(f"怎么做：{v.when}")
    print("明细：")
    for r in v.reasons:
        print(" ", r)
    print("关键指标：")
    for k in [
        "收盘",
        "今日涨跌%",
        "px_ma20",
        "距40日低点%",
        "距60日高点%",
        "前5日涨幅%",
        "前20日涨幅%",
        "前40日波动%",
        "安静日占比40%",
        "额能比5_20",
        "换手率%",
        "流通市值亿",
    ]:
        print(f"  {k}: {v.metrics.get(k)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="宝藏观察池（技术面）")
    parser.add_argument("code", nargs="?", help="单票评估")
    parser.add_argument("--scan", action="store_true")
    parser.add_argument("--asof", default=None)
    parser.add_argument("--bands", default=str(BANDS_PATH))
    parser.add_argument("--include-left", action="store_true", help="附带已离底对照")
    args = parser.parse_args()

    bands = load_bands(Path(args.bands))
    if args.code and not args.scan:
        v = evaluate(args.code, bands=bands, asof=args.asof)
        print_verdict(v)
        return

    import daily_cache

    daily_cache.preload(force=False)
    out = scan(asof=args.asof, bands_path=Path(args.bands), include_left=args.include_left)
    out.to_csv(OUT_SCAN, index=False, encoding="utf-8-sig")
    n = len(out)
    print(f"宝藏观察 {n} 只 → {OUT_SCAN}")
    if n:
        cols = [
            c
            for c in [
                "code",
                "name",
                "industry",
                "stage",
                "score",
                "距60日高点%",
                "距40日低点%",
                "px_ma20",
                "流通市值亿",
            ]
            if c in out.columns
        ]
        print(out[cols].head(30).to_string(index=False))


if __name__ == "__main__":
    main()
