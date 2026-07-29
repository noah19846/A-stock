"""
挖掘「短线浅回撤后快速爆发」买点：

以收盘价买入日 T 为锚点，未来 N 个交易日内：
  - 最低价相对买入价回撤 ≥ -max_dd（默认 -3%）
  - 最高价相对买入价涨幅 ≥ +min_gain（默认 +8%）
  - 首次摸到 +min_gain 的天数尽量短（记录）

输出：
  data/short_burst_entries.csv   全部正样本（已做非重叠去重）
  data/short_burst_entry_stats.txt 汇总
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DAILY = ROOT / "data" / "daily_raw"
LIST = ROOT / "data" / "stock_list.csv"
INDUSTRY = ROOT / "data" / "stock_industry.csv"
OUT = ROOT / "data" / "short_burst_entries.csv"
OUT_STATS = ROOT / "data" / "short_burst_entry_stats.txt"

DEFAULT_N = 5
DEFAULT_MAX_DD = 0.03  # 允许最大不利 3%
DEFAULT_MIN_GAIN = 0.08
LOOKBACK_DAYS = 520  # ~2 年交易日
MIN_AMT20 = 3.0e7  # 20 日均成交额下限（元）
MIN_MV = 30.0  # 流通市值亿
MAX_MV = 1000.0


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


def load_daily(code: str) -> pd.DataFrame | None:
    path = DAILY / f"{code}.csv"
    if not path.exists():
        return None
    try:
        df = pd.read_csv(
            path,
            usecols=["日期", "收盘", "最高", "最低", "成交额", "流通股本", "换手率", "后复权因子"],
        )
    except Exception:
        return None
    if len(df) < 80:
        return None
    df = df.sort_values("日期").reset_index(drop=True)
    df["日期"] = pd.to_datetime(df["日期"])
    f = df["后复权因子"].astype(float)
    df["px"] = df["收盘"].astype(float) * f
    df["hi"] = df["最高"].astype(float) * f
    df["lo"] = df["最低"].astype(float) * f
    df["amt"] = df["成交额"].astype(float)
    df["turn"] = df["换手率"].astype(float) * 100.0
    df["mv"] = df["流通股本"].astype(float) * df["收盘"].astype(float) / 1e8
    df["ret"] = df["px"].pct_change()
    return df


def scan_stock(
    code: str,
    df: pd.DataFrame,
    n: int,
    max_dd: float,
    min_gain: float,
) -> list[dict]:
    """返回非重叠正样本：命中后跳到首次摸高日之后再找。"""
    px = df["px"].values
    hi = df["hi"].values
    lo = df["lo"].values
    amt = df["amt"].values
    mv = df["mv"].values
    turn = df["turn"].values
    ret = df["ret"].values
    dates = df["日期"].values

    start_i = max(60, len(df) - LOOKBACK_DAYS)
    end_i = len(df) - n - 1
    if end_i <= start_i:
        return []

    hits: list[dict] = []
    i = start_i
    while i <= end_i:
        entry = float(px[i])
        if entry <= 0 or not np.isfinite(entry):
            i += 1
            continue

        amt20 = float(np.nanmean(amt[max(0, i - 19) : i + 1]))
        if amt20 < MIN_AMT20:
            i += 1
            continue
        mvi = float(mv[i])
        if not (MIN_MV <= mvi <= MAX_MV):
            i += 1
            continue

        # 当天已大涨的买点质量差，跳过（避免涨停追价标签）
        if np.isfinite(ret[i]) and ret[i] >= 0.07:
            i += 1
            continue

        fut_hi = hi[i + 1 : i + n + 1]
        fut_lo = lo[i + 1 : i + n + 1]
        if len(fut_hi) < n:
            break

        dd = float(fut_lo.min() / entry - 1.0)
        gain = float(fut_hi.max() / entry - 1.0)
        if dd < -max_dd or gain < min_gain:
            i += 1
            continue

        # 首次摸到 min_gain 的交易日序号（1..n）
        first_hit = None
        for k in range(n):
            if fut_hi[k] / entry - 1.0 >= min_gain:
                first_hit = k + 1
                break
        if first_hit is None:
            i += 1
            continue

        # 首次跌破 -max_dd 的天数（应不存在或晚于摸高；仅记录）
        first_stop = None
        for k in range(n):
            if fut_lo[k] / entry - 1.0 < -max_dd:
                first_stop = k + 1
                break

        close_n = float(px[i + n] / entry - 1.0)
        hits.append(
            {
                "股票代码": code,
                "买入日期": str(pd.Timestamp(dates[i]).date()),
                "前瞻日数N": n,
                "最大回撤%": round(dd * 100.0, 3),
                "最大涨幅%": round(gain * 100.0, 3),
                "首次摸高日": first_hit,
                "首次破止损日": first_stop if first_stop is not None else "",
                f"持有{n}日收盘涨%": round(close_n * 100.0, 3),
                "当日涨跌%": round(float(ret[i]) * 100.0, 3) if np.isfinite(ret[i]) else np.nan,
                "换手率%": round(float(turn[i]), 3),
                "流通市值亿": round(mvi, 2),
                "额20日均亿": round(amt20 / 1e8, 3),
                "_idx": i,
            }
        )
        # 非重叠：跳到摸高日之后
        i = i + first_hit + 1
    return hits


def main() -> None:
    parser = argparse.ArgumentParser(description="短线浅回撤快速爆发买点挖掘")
    parser.add_argument("--n", type=int, default=DEFAULT_N, help="前瞻交易日数")
    parser.add_argument("--max-dd", type=float, default=DEFAULT_MAX_DD, help="最大不利回撤（正数，如 0.03）")
    parser.add_argument("--min-gain", type=float, default=DEFAULT_MIN_GAIN, help="目标涨幅（如 0.08）")
    args = parser.parse_args()

    name_map, ind_map = load_maps()
    rows: list[dict] = []
    files = sorted(p for p in DAILY.glob("*.csv") if p.name[0].isdigit())
    n_files = 0
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
        n_files += 1
        hits = scan_stock(code, df, args.n, args.max_dd, args.min_gain)
        for h in hits:
            h["股票名称"] = name
            h["行业板块"] = ind_map.get(code, "")
            del h["_idx"]
            rows.append(h)

    out = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if out.empty:
        print("无正样本，放宽参数再试")
        out.to_csv(OUT, index=False, encoding="utf-8-sig")
        return

    out = out.sort_values(["首次摸高日", "最大回撤%", "最大涨幅%"], ascending=[True, False, False])
    out.to_csv(OUT, index=False, encoding="utf-8-sig")

    lines = [
        f"参数: N={args.n}, max_dd=-{args.max_dd*100:.1f}%, min_gain=+{args.min_gain*100:.1f}%",
        f"扫描股票数: {n_files}",
        f"正样本数(非重叠): {len(out)}",
        f"覆盖股票数: {out['股票代码'].nunique()}",
        f"首次摸高日 中位/均值: {out['首次摸高日'].median():.1f} / {out['首次摸高日'].mean():.2f}",
        f"最大回撤% 中位: {out['最大回撤%'].median():.2f}",
        f"最大涨幅% 中位: {out['最大涨幅%'].median():.2f}",
        f"持有{args.n}日收盘涨% 中位: {out[f'持有{args.n}日收盘涨%'].median():.2f}",
        "",
        "按行业 Top10:",
        out["行业板块"].value_counts().head(10).to_string(),
        "",
        "样例(最快摸高):",
        out.head(15)[
            ["股票代码", "股票名称", "买入日期", "首次摸高日", "最大回撤%", "最大涨幅%", "行业板块"]
        ].to_string(index=False),
    ]
    text = "\n".join(lines)
    OUT_STATS.write_text(text, encoding="utf-8")
    print(text)
    print(f"\n→ {OUT}")


if __name__ == "__main__":
    main()
