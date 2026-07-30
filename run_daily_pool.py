"""
一键日更：更新 K 线 → 中长线埋伏/买入 → 短线可操作 → 写入 signal_pool

目录结构：
  signal_pool/
    YYYY-MM-DD/
      meta.json
      index.html          # Tab：中长线 / 短线；导航标注可买
      long/
        signals.csv
        now.csv           # 仅可买入
      short/
        signals.csv
        now.csv           # 仅可短打

用法：
  .venv/bin/python run_daily_pool.py
  .venv/bin/python run_daily_pool.py --preview
  .venv/bin/python run_daily_pool.py --skip-update
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
POOL_ROOT = ROOT / "signal_pool"
PYTHON = sys.executable

def log(msg: str) -> None:
    print(msg, flush=True)


def clear_feature_cache() -> None:
    try:
        import analyze_short_burst_features as asbf

        asbf._DAILY_CACHE.clear()
    except Exception:
        pass
    try:
        import daily_cache

        daily_cache.clear()
    except Exception:
        pass


def preload_daily(asof: str | None = None) -> None:
    """日更开头并行预载主板日线，长/短线共用。"""
    import daily_cache

    t0 = time.time()
    n = daily_cache.preload(force=True)
    log(f"[cache] 并行预载日线 {n} 只，耗时 {time.time() - t0:.1f}s")
    # 同步到短线旧缓存别名，避免其它脚本再读盘
    try:
        import analyze_short_burst_features as asbf

        asbf._DAILY_CACHE.clear()
        for code in daily_cache.cached_codes():
            asbf._DAILY_CACHE[code] = daily_cache.get(code)
    except Exception:
        pass
    _ = asof  # 预载全量近端；筛选时再按 asof 切片


def resolve_asof() -> str:
    from update_daily import latest_trade_date

    try:
        return latest_trade_date()
    except Exception:
        sample = next((ROOT / "data" / "daily_raw").glob("600*.csv"), None)
        if sample is None:
            return datetime.now().strftime("%Y-%m-%d")
        df = pd.read_csv(sample, usecols=["日期"])
        return str(pd.to_datetime(df["日期"]).max().date())


def run_update(force: bool = True) -> None:
    cmd = [PYTHON, str(ROOT / "update_daily.py")]
    if force:
        cmd.append("--force")
    log(f"[1/4] 更新日线: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def stocks_from_df(
    df: pd.DataFrame,
    days: int = 180,
    asof: str | None = None,
) -> list[dict]:
    """构建 HTML 列表：K 线静态数据嵌在条目里，页面只在点击时渲染图表。"""
    from plot_watch_pool import day_return_pct, load_qfq_bars

    if df.empty:
        return []
    cutoff = pd.Timestamp(asof) if asof else None
    stocks: list[dict] = []
    for _, row in df.iterrows():
        code = str(row["code"]).zfill(6)
        loaded = load_qfq_bars(code, days=days)
        if not loaded:
            continue
        bars, spot = loaded
        if cutoff is not None:
            bars = [b for b in bars if pd.Timestamp(b["date"]) <= cutoff]
            if len(bars) < 30:
                continue
            last = bars[-1]
            spot = {
                "turnover_pct": spot.get("turnover_pct"),
                "amount_yi": round(last["amount"] / 1e8, 4) if last.get("amount") else spot.get("amount_yi"),
                "volume_yi": round(last["volume"] / 1e8, 4) if last.get("volume") else spot.get("volume_yi"),
                "float_yi": spot.get("float_yi"),
            }
        ret1d = day_return_pct(bars)
        advice = str(row.get("stage", "") or "")
        can_buy = advice in ("可买入", "可短打") or bool(row.get("can_buy") or row.get("can_trade"))
        when_raw = str(row.get("when", "") or "").strip()
        when_tip = when_raw
        if advice in ("可买入", "可短打") and "建议：" in when_raw:
            when_tip = when_raw.split("建议：", 1)[1].strip()
        elif advice in ("观察埋伏", "观察") and "等待买入触发：" in when_raw:
            when_tip = when_raw.split("等待买入触发：", 1)[1].strip()
        elif advice in ("观察埋伏", "观察") and "等待" in when_raw:
            # 短线等其它表述：尽量保留后半段
            when_tip = when_raw
        item: dict = {
            "code": code,
            "name": str(row.get("name", "") or code),
            "industry": str(row.get("industry", "") or ""),
            "advice": advice,
            "canBuy": can_buy,
            "when": when_tip,
            "close": float(bars[-1]["close"]) if bars else None,
            "ret1d": None if ret1d is None else round(ret1d, 2),
            "ret60": None,
            "turnover": spot["turnover_pct"],
            "amountYi": spot["amount_yi"],
            "volumeYi": spot["volume_yi"],
            "floatYi": spot["float_yi"],
            "bars": bars,
        }
        if "score_entry" in row and pd.notna(row["score_entry"]):
            item["scoreEntry"] = float(row["score_entry"])
        if "score_setup" in row and pd.notna(row["score_setup"]):
            item["scoreSetup"] = float(row["score_setup"])
        if "score" in row and pd.notna(row["score"]):
            item["score"] = float(row["score"])
        if "hard_score" in row and pd.notna(row["hard_score"]):
            item["hardScore"] = float(row["hard_score"])
        stocks.append(item)
    return stocks


def run_long(day_dir: Path, asof: str | None = None) -> tuple[dict, list[dict]]:
    log("[2/4] 中长线筛选（rally_buy）…")
    import rally_buy_screener as rbs

    t0 = time.time()
    df = rbs.scan(entry_only=False, asof=asof)

    out_dir = day_dir / "long"
    out_dir.mkdir(parents=True, exist_ok=True)
    old_html = out_dir / "index.html"
    if old_html.exists():
        old_html.unlink()

    if not df.empty:
        buys = df[df["stage"] == "可买入"].copy()
        df_out = df.copy()
        buys.to_csv(out_dir / "now.csv", index=False, encoding="utf-8-sig")
    else:
        df_out = df
        pd.DataFrame().to_csv(out_dir / "now.csv", index=False, encoding="utf-8-sig")

    df_out.to_csv(out_dir / "signals.csv", index=False, encoding="utf-8-sig")
    stocks = stocks_from_df(df_out, asof=asof)
    n_buy = int((df_out["stage"] == "可买入").sum()) if not df_out.empty else 0
    n_watch = int((df_out["stage"] == "观察埋伏").sum()) if not df_out.empty else 0
    n_hot = int((df_out["stage"] == "已偏强").sum()) if not df_out.empty else 0
    log(
        f"  long: 共 {len(df_out)} 条（可买入 {n_buy} / 观察埋伏 {n_watch}"
        + (f" / 已偏强 {n_hot}" if n_hot else "")
        + f"）→ {out_dir / 'signals.csv'}；HTML 列表 {len(stocks)} 只"
        f"（筛选用时 {time.time() - t0:.1f}s）"
    )
    return {
        "rows": len(df_out),
        "buy": n_buy,
        "watch": n_watch,
        "hot": n_hot,
        "charts": len(stocks),
    }, stocks


def run_short(
    day_dir: Path,
    asof: str | None = None,
    bands_path: Path | None = None,
) -> tuple[dict, list[dict]]:
    log("[3/4] 短线筛选（short_burst）…")
    bands = bands_path or (ROOT / "data" / "short_burst_feature_bands.json")
    out_dir = day_dir / "short"
    out_dir.mkdir(parents=True, exist_ok=True)
    old_html = out_dir / "index.html"
    if old_html.exists():
        old_html.unlink()

    if not bands.exists():
        log(f"  缺少 {bands.name}，跳过短线")
        pd.DataFrame().to_csv(out_dir / "signals.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame().to_csv(out_dir / "now.csv", index=False, encoding="utf-8-sig")
        return {"rows": 0, "buy": 0, "watch": 0, "charts": 0, "skipped": True}, []

    import short_burst_screener as sbs

    t0 = time.time()
    band_cfg = sbs.load_bands(bands)
    log(f"  bands={bands.name} top_k={band_cfg.get('daily_top_k', 0)}")
    df = sbs.scan(entry_only=False, asof=asof, bands=band_cfg)
    if not df.empty:
        now = df[df["stage"] == "可短打"].copy()
        df_out = df.copy()
        now.to_csv(out_dir / "now.csv", index=False, encoding="utf-8-sig")
    else:
        df_out = df
        pd.DataFrame().to_csv(out_dir / "now.csv", index=False, encoding="utf-8-sig")

    df_out.to_csv(out_dir / "signals.csv", index=False, encoding="utf-8-sig")
    stocks = stocks_from_df(df_out, asof=asof)
    n_buy = int((df_out["stage"] == "可短打").sum()) if not df_out.empty else 0
    n_watch = int((df_out["stage"] == "观察").sum()) if not df_out.empty else 0
    log(
        f"  short: 共 {len(df_out)} 条（可短打 {n_buy} / 观察 {n_watch}）"
        f" → {out_dir / 'signals.csv'}；HTML 列表 {len(stocks)} 只"
        f"（筛选用时 {time.time() - t0:.1f}s）"
    )
    return {
        "rows": len(df_out),
        "buy": n_buy,
        "watch": n_watch,
        "charts": len(stocks),
        "bands": str(bands),
    }, stocks


def write_meta(day_dir: Path, asof: str, mode: str, long_stat: dict, short_stat: dict) -> None:
    meta = {
        "date": asof,
        "mode": mode,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "pool_root": str(POOL_ROOT),
        "index_html": str(day_dir / "index.html"),
        "long": long_stat,
        "short": short_stat,
        "workflow": {
            "preview": "收盘前用实时价预筛，供次日早盘参考",
            "final": "收盘后强刷正式截面，覆盖同日目录",
            "note": "preview 与 final 名单可能不一致，以 final 为准归档",
        },
    }
    path = day_dir / "meta.json"
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"[4/4] meta → {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="日更 + 长/短线信号池 → signal_pool")
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--skip-update", action="store_true")
    parser.add_argument("--long-only", action="store_true")
    parser.add_argument("--short-only", action="store_true")
    parser.add_argument("--date", default="")
    parser.add_argument(
        "--short-bands",
        default="",
        help="短线 bands JSON，如 data/short_burst_feature_bands_top3.json",
    )
    args = parser.parse_args()

    t0 = time.time()
    mode = "preview" if args.preview else "final"

    if not args.skip_update:
        run_update(force=True)
    else:
        log("[1/4] 跳过日线更新")

    clear_feature_cache()
    asof = args.date.strip() or resolve_asof()
    day_dir = POOL_ROOT / asof
    day_dir.mkdir(parents=True, exist_ok=True)
    log(f"输出目录: {day_dir}  mode={mode}")

    # 长/短线共用：只并行读盘一遍
    preload_daily(asof=args.date.strip() or None)

    do_long = not args.short_only
    do_short = not args.long_only

    long_stat: dict = {}
    short_stat: dict = {}
    long_stocks: list[dict] = []
    short_stocks: list[dict] = []

    # 若 --date 指定历史日，筛选按该日收盘截面（截断日线），避免用到之后的数据
    asof_for_scan = args.date.strip() or None
    short_bands = Path(args.short_bands) if args.short_bands.strip() else None

    if do_long:
        long_stat, long_stocks = run_long(day_dir, asof=asof_for_scan)
    else:
        log("[2/4] 跳过中长线")
    if do_short:
        short_stat, short_stocks = run_short(
            day_dir, asof=asof_for_scan, bands_path=short_bands
        )
    else:
        log("[3/4] 跳过短线")

    from plot_watch_pool import build_html

    out_html = day_dir / "index.html"
    build_html(
        asof,
        [],
        out_html,
        days=180,
        panels={"long": long_stocks, "short": short_stocks},
    )
    log(f"  合并图: long={len(long_stocks)} short={len(short_stocks)} → {out_html}")

    write_meta(day_dir, asof, mode, long_stat, short_stat)
    log(f"完成，耗时 {time.time() - t0:.0f}s")
    log(f"打开: {out_html}")


if __name__ == "__main__":
    main()
