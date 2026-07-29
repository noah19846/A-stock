"""
选股（先落地日线层）：读取 data/daily_raw 本地日线，
用后复权价算均线/涨幅，做流动性 + 趋势过滤。

板块主线、财务过滤后续再接。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
DAILY_DIR = ROOT / "data" / "daily_raw"
LIST_CSV = ROOT / "data" / "stock_list.csv"
INDUSTRY_CSV = ROOT / "data" / "stock_industry.csv"
POOL_DIR = ROOT / "watch_pool"

# 成交额单位：元；阈值 5 亿元
amount_min = 5e8
# 换手率：本地为小数，指标里转成百分数后与 3~12 比较
turn_min, turn_max = 3.0, 12.0
# 流通市值：亿元
mv_min, mv_max = 50.0, 800.0


def get_all_daily_kline(daily_dir: Path = DAILY_DIR) -> pd.DataFrame:
    """加载本地主板日线（不复权 OHLCV + 后复权因子）。"""
    frames = []
    files = sorted(
        p
        for p in daily_dir.glob("*.csv")
        if p.name[0].isdigit() and not p.name.startswith("_")
    )
    for path in files:
        try:
            df = pd.read_csv(
                path,
                dtype={"股票代码": str},
                usecols=[
                    "股票代码",
                    "日期",
                    "开盘",
                    "最高",
                    "最低",
                    "收盘",
                    "成交量",
                    "成交额",
                    "流通股本",
                    "换手率",
                    "后复权因子",
                ],
            )
        except Exception as e:
            print(f"跳过 {path.name}: {e}")
            continue
        if df.empty:
            continue
        df["股票代码"] = df["股票代码"].astype(str).str.zfill(6)
        frames.append(df)

    if not frames:
        raise FileNotFoundError(f"未找到日线 CSV：{daily_dir}")

    out = pd.concat(frames, ignore_index=True)
    out["日期"] = pd.to_datetime(out["日期"], errors="coerce")
    bad = out["日期"].isna().sum()
    if bad:
        print(f"丢弃无效日期行：{bad}")
        out = out.dropna(subset=["日期"])
    out = out.sort_values(["股票代码", "日期"]).reset_index(drop=True)
    print(f"日线加载完成：{out['股票代码'].nunique()} 支，{len(out)} 行")
    return out


def ensure_stock_list(force: bool = False) -> Path:
    """本地缓存全量 A 股代码-名称；不存在时从 akshare 拉取。"""
    if LIST_CSV.exists() and not force:
        return LIST_CSV

    import akshare as ak

    print("拉取全量股票列表...")
    df = ak.stock_info_a_code_name()
    df = df.rename(columns={"code": "股票代码", "name": "股票名称"})
    df["股票代码"] = df["股票代码"].astype(str).str.zfill(6)
    LIST_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(LIST_CSV, index=False, encoding="utf-8-sig")
    print(f"股票列表已保存：{LIST_CSV} 共 {len(df)} 支")
    return LIST_CSV


def load_stock_name_map(force_refresh: bool = False) -> dict[str, str]:
    """返回 {股票代码: 股票名称}，供选股结果 map。"""
    path = ensure_stock_list(force=force_refresh)
    df = pd.read_csv(path, dtype={"股票代码": str})
    df["股票代码"] = df["股票代码"].astype(str).str.zfill(6)
    return dict(zip(df["股票代码"], df["股票名称"].astype(str)))


def load_industry_map() -> dict[str, str]:
    """返回 {股票代码: 行业板块}。需先跑 fetch_board.py。"""
    if not INDUSTRY_CSV.exists():
        print(f"警告：缺少 {INDUSTRY_CSV}，请先运行 .venv/bin/python fetch_board.py")
        return {}
    df = pd.read_csv(INDUSTRY_CSV, dtype={"股票代码": str})
    df["股票代码"] = df["股票代码"].astype(str).str.zfill(6)
    return dict(zip(df["股票代码"], df["行业板块"].astype(str)))


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """后复权价做均线/涨幅；成交额保持元；换手率转百分数。"""
    df = df.copy()
    df["收盘_hfq"] = df["收盘"] * df["后复权因子"]
    df["换手率_pct"] = df["换手率"] * 100.0

    g = df.groupby("股票代码", group_keys=False)

    df["ma10"] = g["收盘_hfq"].transform(lambda x: x.rolling(10).mean())
    df["ma20"] = g["收盘_hfq"].transform(lambda x: x.rolling(20).mean())
    df["ma60"] = g["收盘_hfq"].transform(lambda x: x.rolling(60).mean())

    df["20日均成交额"] = g["成交额"].transform(lambda x: x.rolling(20).mean())
    df["20日均换手率"] = g["换手率_pct"].transform(lambda x: x.rolling(20).mean())

    df["5日均量"] = g["成交量"].transform(lambda x: x.rolling(5).mean())
    df["3日均量"] = g["成交量"].transform(lambda x: x.rolling(3).mean())
    df["缩量条件"] = df["3日均量"] < df["5日均量"] * 0.7

    # 流通市值（亿元）= 流通股本 * 不复权收盘 / 1e8
    df["流通市值"] = df["流通股本"] * df["收盘"] / 1e8

    prev60 = g["收盘_hfq"].shift(60)
    df["近60日涨幅"] = (df["收盘_hfq"] / prev60 - 1.0) * 100.0
    return df


def select_by_daily(df_daily: pd.DataFrame) -> pd.DataFrame:
    """流动性 + 趋势过滤，取每只股票最新交易日一行。"""
    df = add_indicators(df_daily)

    # 只在最新交易日上筛选
    latest = df["日期"].max()
    snap = df[df["日期"] == latest].copy()
    print(f"截面日期：{latest.date()}  候选 {len(snap)} 支")

    # 本地库已排除 ST；占位字段便于以后接 get_all_stock
    snap["ST标识"] = False

    cond_liquid = (
        (snap["20日均成交额"] >= amount_min)
        & (snap["流通市值"] >= mv_min)
        & (snap["流通市值"] <= mv_max)
        & (snap["20日均换手率"] >= turn_min)
        & (snap["20日均换手率"] <= turn_max)
        & (~snap["ST标识"])
    )

    cond_trend = (
        (snap["收盘_hfq"] > snap["ma20"])
        & (snap["ma10"] > snap["ma20"])
        & (snap["ma20"] > snap["ma60"])
        & (snap["近60日涨幅"] < 35)
        # & (snap["缩量条件"])  # 暂关：今日流动性池内几乎无人满足
    )

    liquid = snap[cond_liquid]
    print(f"流动性通过：{len(liquid)}")
    if len(liquid):
        print(
            "  趋势分项："
            f"收盘>ma20={(liquid['收盘_hfq'] > liquid['ma20']).sum()} "
            f"ma10>ma20={(liquid['ma10'] > liquid['ma20']).sum()} "
            f"ma20>ma60={(liquid['ma20'] > liquid['ma60']).sum()} "
            f"60日涨幅<35%={(liquid['近60日涨幅'] < 35).sum()} "
            f"缩量={liquid['缩量条件'].sum()}"
        )
        no_shrink = (
            (liquid["收盘_hfq"] > liquid["ma20"])
            & (liquid["ma10"] > liquid["ma20"])
            & (liquid["ma20"] > liquid["ma60"])
            & (liquid["近60日涨幅"] < 35)
        )
        print(f"  去掉缩量后趋势通过：{no_shrink.sum()}")

    picked = snap[cond_liquid & cond_trend].copy()
    print(f"最终通过：{len(picked)}")

    name_map = load_stock_name_map()
    industry_map = load_industry_map()
    picked["股票名称"] = picked["股票代码"].map(name_map).fillna("")
    picked["行业板块"] = picked["股票代码"].map(industry_map).fillna("")
    missing_name = picked.loc[picked["股票名称"] == "", "股票代码"].tolist()
    missing_ind = picked.loc[picked["行业板块"] == "", "股票代码"].tolist()
    if missing_name:
        print(f"警告：{len(missing_name)} 支未匹配到名称：{missing_name[:10]}")
    if missing_ind:
        print(f"警告：{len(missing_ind)} 支未匹配到行业：{missing_ind[:10]}")

    cols = [
        "股票代码",
        "股票名称",
        "行业板块",
        "日期",
        "收盘",
        "收盘_hfq",
        "ma10",
        "ma20",
        "ma60",
        "20日均成交额",
        "20日均换手率",
        "流通市值",
        "近60日涨幅",
    ]
    return picked[cols].sort_values("近60日涨幅").reset_index(drop=True)


def day_dir(as_of) -> Path:
    """按截面日期归档：watch_pool/YYYY-MM-DD/"""
    d = pd.Timestamp(as_of).strftime("%Y-%m-%d")
    return POOL_DIR / d


def save_watch_pool(watch_pool: pd.DataFrame, as_of) -> Path:
    """写入 watch_pool/YYYY-MM-DD/pool.csv（当日汇总）。"""
    out_dir = day_dir(as_of)
    out_dir.mkdir(parents=True, exist_ok=True)
    # 清掉历史遗留的单票 csv
    for old in out_dir.glob("*.csv"):
        if old.name != "pool.csv":
            old.unlink()
    out_csv = out_dir / "pool.csv"
    watch_pool.to_csv(out_csv, index=False, encoding="utf-8-sig")
    return out_dir


def main() -> None:
    df_daily = get_all_daily_kline()
    watch_pool = select_by_daily(df_daily)

    print("\n今日自动选股观察池（日线层）：")
    print(watch_pool.to_string(index=False))

    if watch_pool.empty:
        print("\n今日无入选，未写文件")
        return

    as_of = watch_pool["日期"].iloc[0]
    out_dir = save_watch_pool(watch_pool, as_of)
    print(f"\n共 {len(watch_pool)} 只，已保存 {out_dir / 'pool.csv'}")

    try:
        from plot_watch_pool import run as plot_run

        html_path = plot_run(date=pd.Timestamp(as_of).strftime("%Y-%m-%d"))
        print(f"蜡烛图 HTML：{html_path}")
    except Exception as e:
        print(f"生成蜡烛图失败（可稍后手动跑 plot_watch_pool.py）：{e}")


if __name__ == "__main__":
    main()

