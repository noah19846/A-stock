"""
主升浪「埋伏→买入」筛选器

基于历史最优上涨子区间的涨前共性，把单票日线判定为：
  - 不关注
  - 观察/埋伏（Setup）
  - 可买入（Entry 触发）
  - 已偏强（错过舒适买点，只适合追高纪律外）
  - 持仓风控提示（若已买入）

默认可买入离场纪律见 data/rally_exit.json：
  硬止损-10% / 止盈+25% / 最多40日；第15日峰值浮盈仍<8%则收盘清仓。

用法：
  .venv/bin/python rally_buy_screener.py 603608
  .venv/bin/python rally_buy_screener.py --scan
  .venv/bin/python rally_buy_screener.py --scan --entry-only
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DAILY = ROOT / "data" / "daily_raw"
LIST = ROOT / "data" / "stock_list.csv"
INDUSTRY = ROOT / "data" / "stock_industry.csv"
OUT_SCAN = ROOT / "data" / "rally_buy_signals.csv"
SETUP_CFG_PATH = ROOT / "data" / "rally_buy_setup.json"

# ---------- Setup（涨前画像，可观察/埋伏）----------
# 区间贴近历史主升启动前的中位附近；全部硬满足才算 Setup（不再用软分凑 0.75）
SETUP = {
    "px_ma20_lo": 0.93,
    "px_ma20_hi": 0.995,  # 仍在 MA20 下方/贴着
    "dist60_lo": -20.0,
    "dist60_hi": -12.0,  # 距60日高点回撤
    "ret20_lo": -12.0,
    "ret20_hi": -1.0,
    "amt_ratio_hi": 0.88,  # 明显缩量/平量
    "mv_lo": 50.0,
    "mv_hi": 550.0,
    "turn_hi": 7.0,
}


def load_setup_cfg(path: Path | None = None) -> dict:
    """从 data/rally_buy_setup.json 覆盖 SETUP（目前主要用于市值门槛）。"""
    p = path or SETUP_CFG_PATH
    if not p.exists():
        return SETUP
    raw = json.loads(p.read_text(encoding="utf-8"))
    for k in ("mv_lo", "mv_hi", "px_ma20_lo", "px_ma20_hi", "dist60_lo", "dist60_hi",
              "ret20_lo", "ret20_hi", "amt_ratio_hi", "turn_hi"):
        if k in raw and raw[k] is not None:
            SETUP[k] = float(raw[k])
    return SETUP


load_setup_cfg()

# ---------- Entry（买入触发）----------
ENTRY = {
    "need_above_ma20": True,
    "prefer_ma10_ma20": True,  # 硬要求 MA10>MA20
    "amt_ratio_lo": 0.95,
    "amt_ratio_hi": 1.35,  # 量能恢复但未失控
    "ret5_lo": 0.0,
    "not_extended_dist60": -5.0,
    "max_ret20_chase": 12.0,
    "px_ma20_hi": 1.04,  # 刚站上，不追太远
    "setup_lookback": 11,  # 近 N 日曾完整满足 Setup（历史间隔中位约 11）
}

# ---------- 风险（持仓/放弃）----------
RISK = {
    "stop_below_ma20_days": 3,
    "too_hot_ret20": 35.0,
}

# ---------- 离场纪律（data/rally_exit.json；折中：D15峰值<8%清仓）----------
EXIT_CFG_PATH = ROOT / "data" / "rally_exit.json"
EXIT = {
    "stop": 0.10,
    "target": 0.25,
    "hold_days_max": 40,
    "early_check_day": 15,
    "early_min_mfe": 0.08,
    "desc": (
        "硬止损-10%；止盈+25%；最多40个交易日；"
        "第15个交易日若峰值浮盈仍<8%则收盘清仓"
    ),
}


def load_exit_cfg(path: Path | None = None) -> dict:
    p = path or EXIT_CFG_PATH
    cfg = dict(EXIT)
    if p.exists():
        import json

        raw = json.loads(p.read_text(encoding="utf-8"))
        for k in (
            "stop",
            "target",
            "hold_days_max",
            "early_check_day",
            "early_min_mfe",
            "desc",
        ):
            if k in raw:
                cfg[k] = raw[k]
    return cfg


def exit_advice_text(cfg: dict | None = None) -> str:
    c = cfg or load_exit_cfg()
    if c.get("desc"):
        return str(c["desc"])
    return (
        f"硬止损-{float(c['stop'])*100:.0f}%；"
        f"止盈+{float(c['target'])*100:.0f}%；"
        f"最多{int(c['hold_days_max'])}个交易日；"
        f"第{int(c['early_check_day'])}个交易日若峰值浮盈仍"
        f"<{float(c['early_min_mfe'])*100:.0f}%则收盘清仓"
    )


@dataclass
class Verdict:
    code: str
    name: str
    industry: str
    asof: str
    stage: str  # 不关注/观察埋伏/可买入/已偏强/风险
    can_buy: bool
    when: str
    score_setup: float
    score_entry: float
    reasons: list[str]
    metrics: dict

    def to_row(self) -> dict:
        d = asdict(self)
        d["reasons"] = "；".join(self.reasons)
        # flatten metrics
        m = d.pop("metrics")
        d.update(m)
        return d


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


def load_ohlcv(code: str, asof: str | None = None) -> pd.DataFrame | None:
    import daily_cache

    return daily_cache.get(str(code).zfill(6), asof=asof)


def metrics_at(df: pd.DataFrame, i: int | None = None) -> dict | None:
    if i is None:
        i = len(df) - 1
    if i < 60:
        return None
    px = df["px"].values
    amt = df["amt"].values
    close = float(px[i])
    ma10 = float(df["ma10"].iloc[i])
    ma20 = float(df["ma20"].iloc[i])
    ma60 = float(df["ma60"].iloc[i])
    if not all(np.isfinite([close, ma10, ma20, ma60])):
        return None

    def ret_n(n: int) -> float:
        return float(px[i] / px[i - n] - 1.0) * 100.0

    amt5 = float(np.nanmean(amt[i - 4 : i + 1]))
    amt20 = float(np.nanmean(amt[i - 19 : i + 1]))
    hi60 = float(np.nanmax(px[i - 59 : i + 1]))
    seg = df["ret"].iloc[i - 39 : i + 1].astype(float).dropna()
    vol40 = float(seg.std() * 100) if len(seg) else np.nan

    return {
        "收盘": round(float(df["收盘"].iloc[i]), 3),
        "收盘_hfq": round(close, 3),
        "px_ma20": round(close / ma20, 4),
        "px_ma10": round(close / ma10, 4),
        "ma10_ma20": round(ma10 / ma20, 4),
        "ma20_ma60": round(ma20 / ma60, 4),
        "above_ma20": bool(close > ma20),
        "ma10_gt_ma20": bool(ma10 > ma20),
        "距60日高点%": round((close / hi60 - 1.0) * 100.0, 3),
        "前5日涨幅%": round(ret_n(5), 3),
        "前20日涨幅%": round(ret_n(20), 3),
        "前40日涨幅%": round(ret_n(40), 3),
        "额能比5_20": round(amt5 / amt20, 4) if amt20 > 0 else np.nan,
        "换手率%": round(float(df["turn"].iloc[i]), 3),
        "流通市值亿": round(float(df["mv"].iloc[i]), 3),
        "前40日波动%": round(vol40, 3) if np.isfinite(vol40) else np.nan,
        "今日涨跌%": round(float(df["ret"].iloc[i]) * 100, 3)
        if np.isfinite(df["ret"].iloc[i])
        else np.nan,
    }


def setup_score(m: dict) -> tuple[float, list[str]]:
    """0~1 加权通过率（仅供展示/排序）；阶段判定用 setup_pass 硬条件。"""
    reasons: list[str] = []
    checks = []

    def add(ok: bool, msg: str, w: float = 1.0):
        checks.append((ok, w))
        if ok:
            reasons.append("✓ " + msg)
        else:
            reasons.append("✗ " + msg)

    add(
        SETUP["px_ma20_lo"] <= m["px_ma20"] <= SETUP["px_ma20_hi"],
        f"收盘/MA20={m['px_ma20']:.3f} 在[{SETUP['px_ma20_lo']},{SETUP['px_ma20_hi']}]",
        1.4,
    )
    add(
        SETUP["dist60_lo"] <= m["距60日高点%"] <= SETUP["dist60_hi"],
        f"距60高={m['距60日高点%']:.1f}% 在[{SETUP['dist60_lo']},{SETUP['dist60_hi']}]",
        1.3,
    )
    add(
        SETUP["ret20_lo"] <= m["前20日涨幅%"] <= SETUP["ret20_hi"],
        f"前20日={m['前20日涨幅%']:.1f}% 在[{SETUP['ret20_lo']},{SETUP['ret20_hi']}]",
        1.1,
    )
    add(
        np.isfinite(m["额能比5_20"]) and m["额能比5_20"] <= SETUP["amt_ratio_hi"],
        f"额能比={m['额能比5_20']:.2f} ≤{SETUP['amt_ratio_hi']}（缩量）",
        1.0,
    )
    add(
        SETUP["mv_lo"] <= m["流通市值亿"] <= SETUP["mv_hi"],
        f"流通市值={m['流通市值亿']:.0f}亿 在[{SETUP['mv_lo']},{SETUP['mv_hi']}]",
        0.8,
    )
    add(
        m["换手率%"] <= SETUP["turn_hi"],
        f"换手={m['换手率%']:.2f}% ≤{SETUP['turn_hi']}",
        0.7,
    )

    wsum = sum(w for _, w in checks) or 1.0
    score = sum((1.0 if ok else 0.0) * w for ok, w in checks) / wsum
    return score, reasons


def setup_pass(m: dict) -> bool:
    """观察/埋伏：Setup 区间必须全部满足（硬条件）。"""
    return (
        SETUP["px_ma20_lo"] <= m["px_ma20"] <= SETUP["px_ma20_hi"]
        and SETUP["dist60_lo"] <= m["距60日高点%"] <= SETUP["dist60_hi"]
        and SETUP["ret20_lo"] <= m["前20日涨幅%"] <= SETUP["ret20_hi"]
        and np.isfinite(m["额能比5_20"])
        and m["额能比5_20"] <= SETUP["amt_ratio_hi"]
        and SETUP["mv_lo"] <= m["流通市值亿"] <= SETUP["mv_hi"]
        and m["换手率%"] <= SETUP["turn_hi"]
    )


def entry_score(m: dict) -> tuple[float, list[str]]:
    reasons: list[str] = []
    checks = []

    def add(ok: bool, msg: str, w: float = 1.0):
        checks.append((ok, w))
        reasons.append(("✓ " if ok else "✗ ") + msg)

    add(m["above_ma20"], "收盘站上 MA20（历史主升中位约第4日出现）", 1.6)
    add(m["ma10_gt_ma20"], "MA10 > MA20（趋势初步转多）", 1.2)
    add(
        np.isfinite(m["额能比5_20"]) and m["额能比5_20"] >= ENTRY["amt_ratio_lo"],
        f"额能比={m['额能比5_20']:.2f} ≥{ENTRY['amt_ratio_lo']}（量能恢复）",
        1.0,
    )
    add(
        np.isfinite(m["额能比5_20"]) and m["额能比5_20"] <= ENTRY["amt_ratio_hi"],
        f"额能比={m['额能比5_20']:.2f} ≤{ENTRY['amt_ratio_hi']}（未失控放量）",
        0.8,
    )
    add(m["前5日涨幅%"] >= ENTRY["ret5_lo"], f"近5日涨幅={m['前5日涨幅%']:.1f}% ≥0", 0.9)
    add(
        m["距60日高点%"] <= ENTRY["not_extended_dist60"],
        f"距60高={m['距60日高点%']:.1f}% ≤{ENTRY['not_extended_dist60']}%",
        1.0,
    )
    add(
        m["前20日涨幅%"] <= ENTRY["max_ret20_chase"],
        f"前20日={m['前20日涨幅%']:.1f}% ≤{ENTRY['max_ret20_chase']}%",
        1.1,
    )
    add(
        m["px_ma20"] <= ENTRY["px_ma20_hi"],
        f"收盘/MA20={m['px_ma20']:.3f} ≤{ENTRY['px_ma20_hi']}（刚突破）",
        1.0,
    )

    wsum = sum(w for _, w in checks) or 1.0
    score = sum((1.0 if ok else 0.0) * w for ok, w in checks) / wsum
    return score, reasons


def entry_pass(m: dict) -> bool:
    """买入触发：Entry 硬条件全部满足。"""
    return (
        bool(m["above_ma20"])
        and bool(m["ma10_gt_ma20"])
        and np.isfinite(m["额能比5_20"])
        and ENTRY["amt_ratio_lo"] <= m["额能比5_20"] <= ENTRY["amt_ratio_hi"]
        and m["前5日涨幅%"] >= ENTRY["ret5_lo"]
        and m["距60日高点%"] <= ENTRY["not_extended_dist60"]
        and m["前20日涨幅%"] <= ENTRY["max_ret20_chase"]
        and m["px_ma20"] <= ENTRY["px_ma20_hi"]
    )


def had_setup_recently(df: pd.DataFrame, lookback: int | None = None) -> bool:
    """近 lookback 日（不含今日）是否出现过完整 Setup——解决「Setup 在下、Entry 在上」的时序矛盾。"""
    if lookback is None:
        lookback = int(ENTRY["setup_lookback"])
    n = len(df)
    for j in range(2, lookback + 1):
        i = n - j
        if i < 60:
            break
        m = metrics_at(df, i)
        if m is not None and setup_pass(m):
            return True
    return False


def evaluate(
    code: str,
    name_map: dict | None = None,
    ind_map: dict | None = None,
    asof: str | None = None,
) -> Verdict:
    code = str(code).zfill(6)
    if name_map is None or ind_map is None:
        name_map, ind_map = load_maps()
    name = name_map.get(code, "")
    industry = ind_map.get(code, "")
    df = load_ohlcv(code, asof=asof)
    if df is None:
        return Verdict(
            code,
            name,
            industry,
            "",
            "不关注",
            False,
            "无足够日线数据",
            0.0,
            0.0,
            ["缺少日线或长度不足"],
            {},
        )

    asof_s = str(df["日期"].iloc[-1].date())
    m = metrics_at(df)
    if m is None:
        return Verdict(
            code, name, industry, asof_s, "不关注", False, "指标不足", 0.0, 0.0, ["均线未就绪"], {}
        )

    s_setup, r_setup = setup_score(m)
    s_entry, r_entry = entry_score(m)
    is_setup = setup_pass(m)
    is_entry = entry_pass(m)

    # 过热
    if m["前20日涨幅%"] >= RISK["too_hot_ret20"] or m["距60日高点%"] > 0:
        stage = "已偏强"
        can_buy = False
        when = "已离开舒适买点：涨幅/新高过大，只适合有纪律的趋势跟踪，不适合再当埋伏首买"
        reasons = r_entry + r_setup
    elif is_setup:
        # 今日仍在 Setup 区 → 只观察，等站上 MA20
        stage = "观察埋伏"
        can_buy = False
        missing = [x for x in r_entry if x.startswith("✗")]
        when = (
            "先观察/轻仓埋伏池，不急着买。"
            "等待买入触发："
            + ("；".join(missing[:3]) if missing else "收盘站上MA20且额能比回升")
        )
        reasons = ["【Setup 硬条件已齐】"] + r_setup + ["【Entry 未齐】"] + r_entry
    elif is_entry and had_setup_recently(df):
        stage = "可买入"
        can_buy = True
        when = (
            "近几日曾满足涨前 Setup，今日 Entry 硬条件齐：站上MA20、均线转多、量能恢复且未过热。"
            f"建议：分批；离场：{exit_advice_text()}。"
        )
        reasons = ["【Setup 近几日曾满足】"] + r_setup + ["【Entry 硬条件已齐】"] + r_entry
    elif is_entry:
        stage = "已偏强"
        can_buy = False
        when = "有突破形态但近几日未见完整缩量回撤 Setup，追价盈亏比一般"
        reasons = r_entry + r_setup
    else:
        stage = "不关注"
        can_buy = False
        when = "既非涨前画像，也未形成舒适买点"
        reasons = r_setup + r_entry

    return Verdict(
        code=code,
        name=name,
        industry=industry,
        asof=asof_s,
        stage=stage,
        can_buy=can_buy,
        when=when,
        score_setup=round(s_setup, 3),
        score_entry=round(s_entry, 3),
        reasons=reasons,
        metrics=m,
    )


def scan(
    entry_only: bool = False,
    min_setup: float = 0.0,
    asof: str | None = None,
    include_hot: bool = False,
) -> pd.DataFrame:
    """默认只输出 观察埋伏 / 可买入。include_hot=True 时附带已偏强对照。"""
    import daily_cache

    name_map, ind_map = load_maps()
    rows = []
    codes = daily_cache.cached_codes()
    if not codes:
        codes = [
            p.stem.zfill(6)
            for p in sorted(DAILY.glob("*.csv"))
            if p.name[0].isdigit()
            and p.stem.zfill(6).startswith(("600", "601", "603", "605", "000", "001", "002"))
        ]
    for code in codes:
        if not code.startswith(("600", "601", "603", "605", "000", "001", "002")):
            continue
        if "ST" in name_map.get(code, "").upper():
            continue
        v = evaluate(code, name_map, ind_map, asof=asof)
        if v.stage == "不关注":
            continue
        if v.stage == "已偏强" and not include_hot:
            continue
        if entry_only and not v.can_buy:
            continue
        if min_setup > 0 and v.score_setup < min_setup and not v.can_buy and v.stage != "已偏强":
            continue
        rows.append(v.to_row())
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    order = {"可买入": 0, "观察埋伏": 1, "已偏强": 2, "风险": 3, "不关注": 4}
    out["_ord"] = out["stage"].map(order).fillna(9)
    out = out.sort_values(["_ord", "score_entry", "score_setup"], ascending=[True, False, False])
    return out.drop(columns=["_ord"]).reset_index(drop=True)


def print_verdict(v: Verdict) -> None:
    print(f"{v.code} {v.name}  {v.industry}")
    print(f"日期 {v.asof}")
    print(f"阶段：{v.stage}    能否买：{'是' if v.can_buy else '否'}")
    print(f"Setup分 {v.score_setup:.3f}   Entry分 {v.score_entry:.3f}")
    print(f"何时买/怎么做：{v.when}")
    print("明细：")
    for r in v.reasons:
        print(" ", r)
    print("关键指标：")
    for k in [
        "收盘",
        "今日涨跌%",
        "px_ma20",
        "ma10_gt_ma20",
        "距60日高点%",
        "前5日涨幅%",
        "前20日涨幅%",
        "额能比5_20",
        "换手率%",
        "流通市值亿",
    ]:
        print(f"  {k}: {v.metrics.get(k)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="主升浪埋伏/买入筛选器")
    parser.add_argument("code", nargs="?", help="股票代码，如 603608")
    parser.add_argument("--scan", action="store_true", help="扫描全市场")
    parser.add_argument("--entry-only", action="store_true", help="只输出可买入")
    args = parser.parse_args()

    if args.code:
        v = evaluate(args.code)
        print_verdict(v)
        return

    if args.scan:
        print("扫描中…")
        df = scan(entry_only=args.entry_only)
        OUT_SCAN.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUT_SCAN, index=False, encoding="utf-8-sig")
        print(f"共 {len(df)} 条 → {OUT_SCAN}")
        cols = [
            "code",
            "name",
            "industry",
            "stage",
            "can_buy",
            "score_setup",
            "score_entry",
            "when",
            "收盘",
            "px_ma20",
            "距60日高点%",
            "前20日涨幅%",
            "额能比5_20",
        ]
        cols = [c for c in cols if c in df.columns]
        print(df[cols].head(40).to_string(index=False))
        return

    parser.print_help()


if __name__ == "__main__":
    main()
