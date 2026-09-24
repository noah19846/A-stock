"""
一键日更：更新 K 线 → 筛选 → 写入 signal_pool / hot_sectors

目录结构：
  signal_pool/YYYY-MM-DD/
    meta.json / index.html / long/ / short/ / watch/ / scalp/ / …
  hot_sectors/YYYY-MM-DD/
    meta.json / index.html / roles.json / …

模式：
  默认（正式）：刷盘中价，完整生成 HTML（下单看这份）
  --close：收盘重拉 K 线；不生成 HTML；短线只筛「观察」增量入观察簿

用法：
  .venv/bin/python run_daily_pool.py
  .venv/bin/python run_daily_pool.py --mail          # 正式 + QQ 邮件摘要
  .venv/bin/python run_daily_pool.py --close
  .venv/bin/python run_daily_pool.py --close --mail  # 收盘 + 观察簿邮件
  .venv/bin/python run_daily_pool.py --rebuild-html
  .venv/bin/python run_daily_pool.py --from-intraday-snapshot --from-date … --to-date …
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
HOT_ROOT = ROOT / "hot_sectors"
PYTHON = sys.executable

def log(msg: str) -> None:
    print(msg, flush=True)


def log_signal_table(df: pd.DataFrame, *, title: str = "") -> None:
    """条数>0 时打印代码/名称/阶段/今日交易简况。"""
    if df is None or df.empty:
        return
    out = df.copy()
    if "code" in out.columns:
        out["code"] = out["code"].astype(str).str.zfill(6)
    cols_prefer = [
        ("code", "代码"),
        ("name", "名称"),
        ("stage", "阶段"),
        ("今日涨跌%", "涨跌%"),
        ("收盘", "收盘"),
        ("换手率%", "换手%"),
        ("strategy_tags", "策略"),
    ]
    use: list[tuple[str, str]] = []
    for c, label in cols_prefer:
        if c in out.columns:
            use.append((c, label))
    if not use:
        return
    rows = []
    for _, r in out.iterrows():
        cells = []
        for c, _ in use:
            v = r.get(c, "")
            if c == "今日涨跌%" and v == v and v != "":
                try:
                    cells.append(f"{float(v):+.2f}")
                except Exception:
                    cells.append(str(v))
            elif c in ("收盘", "换手率%") and v == v and v != "":
                try:
                    cells.append(f"{float(v):.2f}")
                except Exception:
                    cells.append(str(v))
            else:
                cells.append("" if v != v or v is None else str(v))
        rows.append(cells)
    widths = [len(lab) for _, lab in use]
    for cells in rows:
        for i, cell in enumerate(cells):
            widths[i] = max(widths[i], len(cell))
    if title:
        log(f"  --- {title} ---")
    header = "  ".join(lab.ljust(widths[i]) for i, (_, lab) in enumerate(use))
    log(f"  {header}")
    log(f"  {'  '.join('-' * w for w in widths)}")
    for cells in rows:
        line = "  ".join(cells[i].ljust(widths[i]) for i in range(len(use)))
        log(f"  {line}")


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
    log(f"[cache] 并行预载日线 {n} 只（{daily_cache.source()}），耗时 {time.time() - t0:.1f}s")
    _ = asof


def resolve_asof() -> str:
    """以本地日线最新日期为准（SQLite / CSV）；不依赖交易所日历接口。"""
    db = ROOT / "data" / "db" / "daily.db"
    if db.exists() and db.stat().st_size > 0:
        try:
            import daily_db

            mx = str(daily_db.stats().get("max_date") or "").strip()
            if mx:
                return mx[:10]
        except Exception:
            pass
    sample = next((ROOT / "data" / "daily_raw").glob("600*.csv"), None)
    if sample is None:
        return datetime.now().strftime("%Y-%m-%d")
    df = pd.read_csv(sample, usecols=["日期"])
    return str(pd.to_datetime(df["日期"]).max().date())


def bar_quality_for_mode(mode: str) -> str:
    """日线质量档：正式跑盘中价=intraday，收盘跑=close。"""
    return "close" if mode == "close" else "intraday"


def normalize_bar_quality(quality: str) -> str:
    q = str(quality or "").strip()
    if q in ("preview", "intraday"):
        return "intraday"
    if q in ("final", "close"):
        return "close"
    return q


def run_update(force: bool = False, *, quality: str = "close") -> None:
    cmd = [PYTHON, str(ROOT / "update_daily.py")]
    if force:
        cmd.append("--force")
    db = ROOT / "data" / "db" / "daily.db"
    if db.exists() and db.stat().st_size > 0:
        cmd.append("--sqlite")
    log(f"[1/4] 更新日线: {' '.join(cmd)}  quality={quality}")
    subprocess.run(cmd, cwd=str(ROOT), check=True)
    # 成功后记下「哪一档」写入了当日，避免把盘中价当成收盘价
    asof = None
    try:
        from update_daily import latest_trade_date

        asof = latest_trade_date()
    except Exception:
        asof = datetime.now().strftime("%Y-%m-%d")
    write_last_update(asof, quality)
    if quality == "intraday":
        capture_intraday_snapshot(asof)


def capture_intraday_snapshot(trade_date: str) -> dict:
    """把当日盘中写入的 daily_bars 另存一份，供以后新策略回放。"""
    import daily_db

    info = daily_db.save_intraday_snapshot(
        trade_date,
        snapshot_at=datetime.now().isoformat(timespec="seconds"),
        source="intraday",
    )
    if info.get("ok"):
        log(
            f"  盘中截面已存: {info['trade_date']} @ {info['snapshot_at']}  "
            f"n={info['n_bars']}"
        )
    else:
        log(f"  盘中截面未存: {info.get('error') or info}")
    return info


UPDATE_META = ROOT / "data" / "db" / "last_update.json"


def read_last_update() -> dict:
    if not UPDATE_META.exists():
        return {}
    try:
        return json.loads(UPDATE_META.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_last_update(trade_date: str, quality: str) -> None:
    UPDATE_META.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "trade_date": str(trade_date)[:10],
        "quality": normalize_bar_quality(quality),  # intraday | close
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    UPDATE_META.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"  日线质量标记: {payload['trade_date']} / {quality}")


def should_skip_daily_update(mode: str) -> bool:
    """是否跳过日更。

    仅看「日期」不够：正式盘中价日期已是当日但并非收盘准确价。
    规则：
      - official：不自动跳过（每次都刷盘中价）
      - close：仅当标记为同日且 quality=close 时跳过（同日盘中价不能顶替）
    """
    info = read_last_update()
    if not info:
        return False
    if mode != "close":
        return False
    if normalize_bar_quality(str(info.get("quality") or "")) != "close":
        return False
    mx = str(info.get("trade_date") or "")[:10]
    if not mx:
        return False
    try:
        from update_daily import latest_trade_date

        return mx == latest_trade_date()
    except Exception:
        return mx >= datetime.now().strftime("%Y-%m-%d")


def load_hot_industries(asof: str, mode: str = "official") -> dict[str, str]:
    """当日热门板块 industry → 档位（趋势热 / 超短反抽）。"""
    path = HOT_ROOT / asof / "roles.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    out: dict[str, str] = {}
    for key, tier in (("trend", "趋势热"), ("burst", "超短反抽")):
        for pack in data.get(key) or []:
            if not isinstance(pack, dict):
                continue
            sector = pack.get("sector")
            name = ""
            if isinstance(sector, dict):
                name = str(sector.get("industry") or "")
            elif isinstance(sector, str):
                name = sector
            if name:
                out[name] = tier
    return out


def hot_map_for_day_dir(day_dir: Path, asof: str | None) -> dict[str, str]:
    if not asof:
        return {}
    return load_hot_industries(asof, "official")


def _dist60_low_pct_from_bars(bars: list[dict], *, min_bars: int = 60) -> float | None:
    """收盘相对近 min_bars 根 K 线最低价的涨幅 %。"""
    if len(bars) < min_bars:
        return None
    seg = bars[-min_bars:]
    lo = min(float(b["low"]) for b in seg)
    close = float(bars[-1]["close"])
    if lo <= 0:
        return None
    return round((close / lo - 1.0) * 100.0, 2)


def _ret20_pct_from_bars(bars: list[dict], *, n: int = 20) -> float | None:
    if len(bars) < n + 1:
        return None
    prev = float(bars[-(n + 1)]["close"])
    close = float(bars[-1]["close"])
    if prev <= 0:
        return None
    return round((close / prev - 1.0) * 100.0, 2)


WARN_DIST60_LOW_PCT = 20.0
IMPULSE_DIST60_LOW_PCT = 30.0
IMPULSE_RET20_PCT = 15.0


def load_watch_book_stocks(
    asof: str, day_dir: Path, *, refresh: bool = True
) -> list[dict]:
    """观察簿在册票 → HTML 列表（独立 tab，不并入短线）。"""
    import watch_book as wb

    if refresh:
        df = wb.open_as_signal_df()
        wb.write_watch_panel_csv(day_dir, df)
    else:
        df = _read_pool_csv(day_dir / "watch" / "signals.csv")
        if df.empty:
            # 旧目录可能只有 watch_book.csv
            alt = day_dir / "watch_book.csv"
            if alt.exists():
                raw = _read_pool_csv(alt)
                if not raw.empty and "stage" not in raw.columns:
                    # open.csv 形态 → 转 signals
                    df = wb.open_as_signal_df()
                    if not df.empty:
                        wb.write_watch_panel_csv(day_dir, df)
                else:
                    df = raw
    if df is None or df.empty:
        return []
    return stocks_from_df(
        df, asof=asof, hot_industries=hot_map_for_day_dir(day_dir, asof)
    )


def stocks_from_df(
    df: pd.DataFrame,
    days: int = 180,
    asof: str | None = None,
    hot_industries: dict[str, str] | None = None,
) -> list[dict]:
    """构建 HTML 列表：K 线静态数据嵌在条目里，页面只在点击时渲染图表。"""
    from plot_watch_pool import day_return_pct, load_qfq_bars

    if df.empty:
        return []
    cutoff = pd.Timestamp(asof) if asof else None
    hot = hot_industries or {}
    from analyze_short_burst_features import load_maps

    name_map, _ = load_maps()
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
        elif advice in ("观察埋伏", "观察", "宝藏观察") and "等待买入触发：" in when_raw:
            when_tip = when_raw.split("等待买入触发：", 1)[1].strip()
        elif advice in ("观察埋伏", "观察", "宝藏观察") and "等待量价确认：" in when_raw:
            when_tip = when_raw.split("等待量价确认：", 1)[1].strip()
        elif advice in ("观察埋伏", "观察", "宝藏观察") and ("等待" in when_raw or "观察" in when_raw):
            when_tip = when_raw
        raw_name = row.get("name", "")
        if raw_name is None or (isinstance(raw_name, float) and raw_name != raw_name):
            raw_name = ""
        else:
            raw_name = str(raw_name).strip()
        if not raw_name or raw_name.lower() == "nan" or raw_name == code:
            raw_name = ""
        stock_name = raw_name or name_map.get(code) or code
        item: dict = {
            "code": code,
            "name": stock_name,
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
        industry = item["industry"]
        if industry and industry in hot:
            item["hotSector"] = True
            item["hotTier"] = hot[industry]
        if "score_entry" in row and pd.notna(row["score_entry"]):
            item["scoreEntry"] = float(row["score_entry"])
        if "score_setup" in row and pd.notna(row["score_setup"]):
            item["scoreSetup"] = float(row["score_setup"])
        if "score" in row and pd.notna(row["score"]):
            item["score"] = float(row["score"])
        if "hard_score" in row and pd.notna(row["hard_score"]):
            item["hardScore"] = float(row["hard_score"])
        if "距60日低点%" in row.index and pd.notna(row.get("距60日低点%")):
            item["dist60LowPct"] = round(float(row["距60日低点%"]), 2)
        if "前20日涨幅%" in row.index and pd.notna(row.get("前20日涨幅%")):
            item["ret20Pct"] = round(float(row["前20日涨幅%"]), 2)
        short_like = advice in ("可短打", "观察")
        if item.get("dist60LowPct") is None and short_like:
            from_bars = _dist60_low_pct_from_bars(bars)
            if from_bars is not None:
                item["dist60LowPct"] = from_bars
        if item.get("ret20Pct") is None and short_like:
            r20 = _ret20_pct_from_bars(bars)
            if r20 is not None:
                item["ret20Pct"] = r20
        if short_like:
            warn = row.get("离底过远提醒")
            if warn in (True, 1, "True", "1", "true") or (
                item.get("dist60LowPct") is not None
                and item["dist60LowPct"] > WARN_DIST60_LOW_PCT
            ):
                item["warnFarFromLow"] = True
            impulse = row.get("急涨浅回提醒")
            if impulse in (True, 1, "True", "1", "true") or (
                item.get("dist60LowPct") is not None
                and item.get("ret20Pct") is not None
                and item["dist60LowPct"] > IMPULSE_DIST60_LOW_PCT
                and item["ret20Pct"] > IMPULSE_RET20_PCT
            ):
                item["warnImpulse"] = True
        if "箱体底" in row and pd.notna(row["箱体底"]):
            item["boxBottom"] = float(row["箱体底"])
        if "箱体顶" in row and pd.notna(row["箱体顶"]):
            item["boxTop"] = float(row["箱体顶"])
        if "止盈价" in row and pd.notna(row["止盈价"]):
            item["takeProfit"] = float(row["止盈价"])
        if "止损参考" in row and pd.notna(row["止损参考"]):
            item["stopRef"] = float(row["止损参考"])
        # 多策略标签（短线合并池）
        tags = row.get("strategy_tags")
        ids = row.get("strategy_ids")
        if isinstance(tags, str) and tags.strip():
            item["strategyTags"] = [t for t in tags.split("|") if t]
        elif isinstance(tags, (list, tuple)):
            item["strategyTags"] = list(tags)
        if isinstance(ids, str) and ids.strip():
            item["strategyIds"] = [t for t in ids.split("|") if t]
        elif isinstance(ids, (list, tuple)):
            item["strategyIds"] = list(ids)
        stocks.append(item)
    return stocks


def _read_pool_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, dtype={"code": str})
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    if df.empty or "code" not in df.columns:
        return pd.DataFrame()
    df["code"] = df["code"].astype(str).str.zfill(6)
    return df


def list_pool_html_jobs() -> list[tuple[str, str, Path]]:
    """已有信号池 HTML：(asof, mode, day_dir)。"""
    jobs: list[tuple[str, str, Path]] = []
    if not POOL_ROOT.exists():
        return jobs
    for day in sorted(p for p in POOL_ROOT.iterdir() if p.is_dir()):
        asof = day.name
        if len(asof) != 10 or asof[4] != "-":
            continue
        if (day / "index.html").exists():
            jobs.append((asof, "live", day))
    return jobs


def rebuild_html_from_csvs(day_dir: Path, asof: str) -> None:
    """用已落盘的 signals.csv 重刷 K 线 HTML（不重跑筛选）。"""
    from plot_watch_pool import build_html

    def panel(name: str) -> list[dict]:
        return stocks_from_df(
            _read_pool_csv(day_dir / name / "signals.csv"),
            asof=asof,
            hot_industries=hot_map_for_day_dir(day_dir, asof),
        )

    long_stocks = panel("long")
    short_stocks = panel("short")
    watch_stocks = load_watch_book_stocks(asof, day_dir, refresh=False)
    scalp_stocks = panel("scalp")
    treasure_stocks = panel("treasure")
    board_stocks = panel("board")
    base_stocks = panel("base")
    relaunch_stocks = panel("relaunch")
    wyckoff_stocks = panel("wyckoff")
    out_html = day_dir / "index.html"
    build_html(
        asof,
        [],
        out_html,
        days=180,
        panels={
            "long": long_stocks,
            "short": short_stocks,
            "watch": watch_stocks,
            "scalp": scalp_stocks,
            "treasure": treasure_stocks,
            "board": board_stocks,
            "base": base_stocks,
            "relaunch": relaunch_stocks,
            "wyckoff": wyckoff_stocks,
        },
    )
    log(
        f"  重刷 {asof}: "
        f"long={len(long_stocks)} short={len(short_stocks)} watch={len(watch_stocks)} "
        f"scalp={len(scalp_stocks)} treasure={len(treasure_stocks)} "
        f"board={len(board_stocks)} base={len(base_stocks)} "
        f"relaunch={len(relaunch_stocks)} wyckoff={len(wyckoff_stocks)} → {out_html}"
    )


def rebuild_all_pool_html() -> int:
    jobs = list_pool_html_jobs()
    if not jobs:
        log("没有可重刷的 signal_pool HTML")
        return 0
    latest = max(asof for asof, _, _ in jobs)
    log(f"预载日线 asof={latest} …")
    preload_daily(asof=latest)
    log(f"重刷 K 线 HTML {len(jobs)} 份")
    for asof, mode, day_dir in jobs:
        rebuild_html_from_csvs(day_dir, asof)
    return len(jobs)


def load_short_registry() -> dict:
    path = ROOT / "data" / "short_strategies.json"
    return json.loads(path.read_text(encoding="utf-8"))


def daily_short_strategy_ids() -> list[str]:
    """每日 HTML「短线」tab 合并的策略（default/strict/r3）。"""
    reg = load_short_registry()
    ids = reg.get("daily_html") or ["default", "strict", "r3"]
    return [str(x) for x in ids]


def daily_scalp_strategy_ids() -> list[str]:
    """每日 HTML 独立「Scalp」tab 的策略。"""
    reg = load_short_registry()
    ids = reg.get("daily_scalp_tabs") or ["scalp"]
    return [str(x) for x in ids]


def strategy_by_id(sid: str) -> dict:
    reg = load_short_registry()
    by_id = {s["id"]: s for s in reg.get("strategies", [])}
    if sid not in by_id:
        raise SystemExit(f"未知短线策略 id: {sid}")
    return by_id[sid]


def run_treasure(day_dir: Path, asof: str | None = None) -> tuple[dict, list[dict]]:
    log("[2b/4] 宝藏观察池（treasure）…")
    import treasure_screener as tbs

    t0 = time.time()
    df = tbs.scan(asof=asof)

    out_dir = day_dir / "treasure"
    out_dir.mkdir(parents=True, exist_ok=True)
    if not df.empty:
        df_out = df.copy()
    else:
        df_out = df
    df_out.to_csv(out_dir / "signals.csv", index=False, encoding="utf-8-sig")
    stocks = stocks_from_df(
        df_out, asof=asof, hot_industries=hot_map_for_day_dir(day_dir, asof)
    )
    n_watch = int((df_out["stage"] == "宝藏观察").sum()) if not df_out.empty else 0
    log(
        f"  treasure: 共 {len(df_out)} 条（宝藏观察 {n_watch}）"
        f"→ {out_dir / 'signals.csv'}；HTML 列表 {len(stocks)} 只"
        f"（筛选用时 {time.time() - t0:.1f}s）"
    )
    if len(df_out) > 0:
        log_signal_table(df_out, title="treasure 明细")
    return {
        "rows": len(df_out),
        "watch": n_watch,
        "buy": 0,
        "charts": len(stocks),
    }, stocks


def run_board(day_dir: Path, asof: str | None = None) -> tuple[dict, list[dict]]:
    log("[2d/4] 涨停箱体回踩（board）…")
    import box_retest_screener as qls

    t0 = time.time()
    df = qls.scan(asof=asof)

    out_dir = day_dir / "board"
    out_dir.mkdir(parents=True, exist_ok=True)
    df_out = df.copy() if not df.empty else df
    if not df_out.empty:
        buys = df_out[df_out["stage"] == "可买入"].copy()
        buys.to_csv(out_dir / "now.csv", index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame().to_csv(out_dir / "now.csv", index=False, encoding="utf-8-sig")
    df_out.to_csv(out_dir / "signals.csv", index=False, encoding="utf-8-sig")
    stocks = stocks_from_df(
        df_out, asof=asof, hot_industries=hot_map_for_day_dir(day_dir, asof)
    )
    n_buy = int((df_out["stage"] == "可买入").sum()) if not df_out.empty else 0
    n_watch = int((df_out["stage"] == "观察").sum()) if not df_out.empty else 0
    log(
        f"  board: 共 {len(df_out)} 条（箱体回踩候选 {n_buy} / 观察 {n_watch}）"
        f"→ {out_dir / 'signals.csv'}；HTML 列表 {len(stocks)} 只"
        f"（筛选用时 {time.time() - t0:.1f}s）"
    )
    if len(df_out) > 0:
        log_signal_table(df_out, title="box_retest 明细")
    return {
        "rows": len(df_out),
        "buy": n_buy,
        "watch": n_watch,
        "charts": len(stocks),
    }, stocks


def run_base(day_dir: Path, asof: str | None = None) -> tuple[dict, list[dict]]:
    log("[2e/4] 底部启动（三连小阳+横盘）…")
    import bottom_base_screener as bbs

    t0 = time.time()
    df = bbs.scan(asof=asof)

    out_dir = day_dir / "base"
    out_dir.mkdir(parents=True, exist_ok=True)
    df_out = df.copy() if not df.empty else df
    if not df_out.empty:
        buys = df_out[df_out["stage"] == "可买入"].copy()
        buys.to_csv(out_dir / "now.csv", index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame().to_csv(out_dir / "now.csv", index=False, encoding="utf-8-sig")
    df_out.to_csv(out_dir / "signals.csv", index=False, encoding="utf-8-sig")
    stocks = stocks_from_df(
        df_out, asof=asof, hot_industries=hot_map_for_day_dir(day_dir, asof)
    )
    n_buy = int((df_out["stage"] == "可买入").sum()) if not df_out.empty else 0
    n_watch = int((df_out["stage"] == "观察").sum()) if not df_out.empty else 0
    log(
        f"  base: 共 {len(df_out)} 条（可买入 {n_buy} / 观察 {n_watch}）"
        f"→ {out_dir / 'signals.csv'}；HTML 列表 {len(stocks)} 只"
        f"（筛选用时 {time.time() - t0:.1f}s）"
    )
    if len(df_out) > 0:
        log_signal_table(df_out, title="base 明细")
    return {
        "rows": len(df_out),
        "buy": n_buy,
        "watch": n_watch,
        "charts": len(stocks),
    }, stocks


def run_relaunch(day_dir: Path, asof: str | None = None) -> tuple[dict, list[dict]]:
    log("[2f/4] 板后重启（relaunch）…")
    import board_relaunch_screener as brs

    t0 = time.time()
    df = brs.scan(asof=asof)

    out_dir = day_dir / "relaunch"
    out_dir.mkdir(parents=True, exist_ok=True)
    df_out = df.copy() if not df.empty else df
    if not df_out.empty:
        buys = df_out[df_out["stage"] == "可买入"].copy()
        buys.to_csv(out_dir / "now.csv", index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame().to_csv(out_dir / "now.csv", index=False, encoding="utf-8-sig")
    df_out.to_csv(out_dir / "signals.csv", index=False, encoding="utf-8-sig")
    stocks = stocks_from_df(
        df_out, asof=asof, hot_industries=hot_map_for_day_dir(day_dir, asof)
    )
    n_buy = int((df_out["stage"] == "可买入").sum()) if not df_out.empty else 0
    n_watch = int((df_out["stage"] == "观察").sum()) if not df_out.empty else 0
    mkt = {}
    try:
        mkt = brs.market_status(asof or (stocks[0]["bars"][-1]["date"] if stocks else ""))
    except Exception:
        mkt = {}
    log(
        f"  relaunch: 共 {len(df_out)} 条（可买入 {n_buy} / 观察 {n_watch}）"
        f"→ {out_dir / 'signals.csv'}；HTML 列表 {len(stocks)} 只"
        f"（筛选用时 {time.time() - t0:.1f}s）"
    )
    if mkt.get("note"):
        log(f"  relaunch 大盘: {mkt['note']}")
    if len(df_out) > 0:
        log_signal_table(df_out, title="relaunch 明细")
    return {
        "rows": len(df_out),
        "buy": n_buy,
        "watch": n_watch,
        "charts": len(stocks),
        "market": mkt,
    }, stocks


def run_wyckoff(day_dir: Path, asof: str | None = None) -> tuple[dict, list[dict]]:
    log("[2g/4] 威科夫 · 实验（wyckoff）…")
    import wyckoff_screener as wks

    t0 = time.time()
    df = wks.scan(asof=asof)

    out_dir = day_dir / "wyckoff"
    out_dir.mkdir(parents=True, exist_ok=True)
    df_out = df.copy() if not df.empty else df
    # 实验 tab：不做「可买入」落盘，全部进 signals 供观察
    pd.DataFrame().to_csv(out_dir / "now.csv", index=False, encoding="utf-8-sig")
    df_out.to_csv(out_dir / "signals.csv", index=False, encoding="utf-8-sig")
    stocks = stocks_from_df(
        df_out, asof=asof, hot_industries=hot_map_for_day_dir(day_dir, asof)
    )
    n_markup = (
        int((df_out["stage"] == "第一次拉伸").sum()) if not df_out.empty else 0
    )
    n_accum = int((df_out["stage"] == "吸筹").sum()) if not df_out.empty else 0
    n_range = int((df_out["stage"] == "震荡").sum()) if not df_out.empty else 0
    log(
        f"  wyckoff: 共 {len(df_out)} 条"
        f"（拉伸 {n_markup} / 吸筹 {n_accum} / 震荡 {n_range}）"
        f"→ {out_dir / 'signals.csv'}；HTML 列表 {len(stocks)} 只"
        f"（筛选用时 {time.time() - t0:.1f}s）"
    )
    if len(df_out) > 0:
        log_signal_table(df_out, title="wyckoff 明细")
    return {
        "rows": len(df_out),
        "buy": 0,
        "watch": len(df_out),
        "markup": n_markup,
        "accum": n_accum,
        "range": n_range,
        "charts": len(stocks),
        "experimental": True,
    }, stocks


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
    stocks = stocks_from_df(
        df_out, asof=asof, hot_industries=hot_map_for_day_dir(day_dir, asof)
    )
    n_buy = int((df_out["stage"] == "可买入").sum()) if not df_out.empty else 0
    n_watch = int((df_out["stage"] == "观察埋伏").sum()) if not df_out.empty else 0
    n_hot = int((df_out["stage"] == "已偏强").sum()) if not df_out.empty else 0
    log(
        f"  long: 共 {len(df_out)} 条（可买入 {n_buy} / 观察埋伏 {n_watch}"
        + (f" / 已偏强 {n_hot}" if n_hot else "")
        + f"）→ {out_dir / 'signals.csv'}；HTML 列表 {len(stocks)} 只"
        f"（筛选用时 {time.time() - t0:.1f}s）"
    )
    if len(df_out) > 0:
        log_signal_table(df_out, title="long 明细")
    return {
        "rows": len(df_out),
        "buy": n_buy,
        "watch": n_watch,
        "hot": n_hot,
        "charts": len(stocks),
    }, stocks


def _scan_one_short(
    sid: str,
    asof: str | None,
    bands_path: Path | None = None,
) -> tuple[dict, pd.DataFrame]:
    """单策略短线扫描，返回策略 meta 与结果表。"""
    import short_burst_screener as sbs

    meta = strategy_by_id(sid) if bands_path is None else {
        "id": sid,
        "title": sid,
        "bands": str(bands_path),
    }
    bands = Path(meta["bands"]) if bands_path is None else bands_path
    if not bands.is_absolute():
        bands = ROOT / bands
    if not bands.exists():
        log(f"  [{sid}] 缺少 {bands.name}，跳过")
        return meta, pd.DataFrame()

    band_cfg = sbs.load_bands(bands)
    log(f"  [{sid}/{meta.get('title', sid)}] bands={bands.name} top_k={band_cfg.get('daily_top_k', 0)}")
    df = sbs.scan(entry_only=False, asof=asof, bands=band_cfg)
    if df.empty:
        return meta, df
    out = df.copy()
    out["strategy_id"] = sid
    out["strategy_title"] = str(meta.get("title", sid))
    return meta, out


def _strategy_strictness(sid: str) -> int:
    """越大越严：r3 > strict > default。"""
    return {"r3": 3, "strict": 2, "default": 1}.get(str(sid), 0)


def _merge_short_frames(frames: list[tuple[dict, pd.DataFrame]]) -> pd.DataFrame:
    """按 code 合并多策略结果，附 strategy_ids / strategy_tags。

    标签规则：某策略对该股为「可短打」才打该策略标签；
    若某股仅出现在某策略的「观察」且未被其它策略可短打覆盖，则仍列出并打该策略标签。

    排序：可短打优先，再按命中策略严格度（r3 > strict > default），再 hard/soft。
    """
    by_code: dict[str, dict] = {}
    stage_rank = {"可短打": 0, "观察": 1, "已偏强": 2, "不关注": 3}
    # sid -> title，便于重排标签
    title_by_sid: dict[str, str] = {}

    for meta, df in frames:
        if df is None or df.empty:
            continue
        sid = str(meta.get("id", ""))
        title = str(meta.get("title", sid))
        if sid:
            title_by_sid[sid] = title
        for _, row in df.iterrows():
            code = str(row["code"]).zfill(6)
            stage = str(row.get("stage", "") or "")
            if stage not in ("可短打", "观察"):
                continue
            cur = by_code.get(code)
            when_piece = f"[{title}] {row.get('when', '')}".strip()
            tag_ok = stage == "可短打"

            if cur is None:
                item = row.to_dict()
                item["code"] = code
                item["strategy_ids"] = [sid]
                item["strategy_tags"] = [title]
                item["_whens"] = [when_piece]
                item["_best_stage"] = stage
                item["_has_buy_tag"] = tag_ok
                by_code[code] = item
                continue

            # 可短打才追加策略标签；避免 TopK 落选的「观察」误标成 Strict/R3
            if tag_ok and sid not in cur["strategy_ids"]:
                cur["strategy_ids"].append(sid)
                cur["strategy_tags"].append(title)
                cur["_whens"].append(when_piece)
                cur["_has_buy_tag"] = True
            elif (not cur.get("_has_buy_tag")) and sid not in cur["strategy_ids"]:
                # 尚无任何可短打标签时，保留观察策略标签
                cur["strategy_ids"].append(sid)
                cur["strategy_tags"].append(title)
                cur["_whens"].append(when_piece)

            if stage_rank.get(stage, 9) < stage_rank.get(cur["_best_stage"], 9):
                keep_ids = cur["strategy_ids"]
                keep_tags = cur["strategy_tags"]
                keep_whens = cur["_whens"]
                keep_buy = cur.get("_has_buy_tag", False)
                for k, v in row.items():
                    if k in ("strategy_id", "strategy_title"):
                        continue
                    cur[k] = v
                cur["code"] = code
                cur["_best_stage"] = stage
                cur["strategy_ids"] = keep_ids
                cur["strategy_tags"] = keep_tags
                cur["_whens"] = keep_whens
                cur["_has_buy_tag"] = keep_buy or tag_ok
            elif stage == cur["_best_stage"]:
                hard = float(row.get("hard_score") or 0)
                soft = float(row.get("score") or 0)
                if hard > float(cur.get("hard_score") or 0) or (
                    hard == float(cur.get("hard_score") or 0)
                    and soft > float(cur.get("score") or 0)
                ):
                    keep_ids = cur["strategy_ids"]
                    keep_tags = cur["strategy_tags"]
                    keep_whens = cur["_whens"]
                    keep_buy = cur.get("_has_buy_tag", False)
                    for k, v in row.items():
                        if k in ("strategy_id", "strategy_title"):
                            continue
                        cur[k] = v
                    cur["code"] = code
                    cur["strategy_ids"] = keep_ids
                    cur["strategy_tags"] = keep_tags
                    cur["_whens"] = keep_whens
                    cur["_has_buy_tag"] = keep_buy or tag_ok

    if not by_code:
        return pd.DataFrame()

    rows = []
    for item in by_code.values():
        item["stage"] = item.pop("_best_stage")
        item["when"] = " | ".join(item.pop("_whens"))
        item.pop("_has_buy_tag", None)
        # 标签按严格度降序：Strict_r3 | Strict | 默认
        ids = list(item["strategy_ids"])
        ids_sorted = sorted(ids, key=_strategy_strictness, reverse=True)
        item["strategy_ids"] = "|".join(ids_sorted)
        item["strategy_tags"] = "|".join(
            title_by_sid.get(s, s) for s in ids_sorted
        )
        item["_strict"] = max((_strategy_strictness(s) for s in ids_sorted), default=0)
        item["can_trade"] = item["stage"] == "可短打"
        rows.append(item)

    out = pd.DataFrame(rows)
    order = {"可短打": 0, "观察": 1}
    out["_o"] = out["stage"].map(order).fillna(9)
    sort_cols = ["_o", "_strict"]
    ascending = [True, False]
    if "hard_score" in out.columns:
        sort_cols.append("hard_score")
        ascending.append(False)
    if "score" in out.columns:
        sort_cols.append("score")
        ascending.append(False)
    out = out.sort_values(sort_cols, ascending=ascending).drop(
        columns=["_o", "_strict"]
    )
    return out.reset_index(drop=True)


def run_short(
    day_dir: Path,
    asof: str | None = None,
    bands_path: Path | None = None,
    strategy_ids: list[str] | None = None,
    scalp_strategy_ids: list[str] | None = None,
) -> tuple[dict, list[dict], dict, list[dict]]:
    """短线筛选。

    默认合并 daily_html（default/strict/r3）进「短线」tab；
    若传入 scalp_strategy_ids（如 ["scalp"]），同一次扫描另出「Scalp」tab。
    返回 (short_stat, short_stocks, scalp_stat, scalp_stocks)。
    """
    import short_burst_screener as sbs

    out_dir = day_dir / "short"
    out_dir.mkdir(parents=True, exist_ok=True)
    old_html = out_dir / "index.html"
    if old_html.exists():
        old_html.unlink()

    t0 = time.time()
    scalp_ids = [str(x) for x in (scalp_strategy_ids or [])]

    def _load_specs(id_list: list[str]) -> list[tuple[str, dict, dict]]:
        specs: list[tuple[str, dict, dict]] = []
        for sid in id_list:
            meta = strategy_by_id(sid)
            bands = Path(meta["bands"])
            if not bands.is_absolute():
                bands = ROOT / bands
            if not bands.exists():
                log(f"  [{sid}] 缺少 {bands.name}，跳过")
                continue
            band_cfg = sbs.load_bands(bands)
            log(
                f"  [{sid}/{meta.get('title', sid)}] bands={bands.name} "
                f"top_k={band_cfg.get('daily_top_k', 0)}"
            )
            specs.append((sid, meta, band_cfg))
        return specs

    # 解析策略列表
    if bands_path is not None and strategy_ids is None:
        ids = ["default"]
        log("[3/4] 短线筛选（单 bands）…")
        strategy_specs: list[tuple[str, dict, dict]] = [
            (
                "default",
                {"id": "default", "title": "默认", "bands": str(bands_path)},
                sbs.load_bands(bands_path if bands_path.is_absolute() else ROOT / bands_path),
            )
        ]
        scalp_specs: list[tuple[str, dict, dict]] = []
    else:
        ids = list(strategy_ids) if strategy_ids is not None else daily_short_strategy_ids()
        scan_note = list(ids) + [s for s in scalp_ids if s not in ids]
        log(f"[3/4] 短线筛选（多策略一次扫描: {', '.join(scan_note) or '(空)'}）…")
        strategy_specs = _load_specs(ids)
        scalp_specs = _load_specs([s for s in scalp_ids if s not in ids])

    all_specs = strategy_specs + scalp_specs
    scanned = sbs.scan_multi(
        [(sid, bands) for sid, _, bands in all_specs],
        asof=asof,
        entry_only=False,
    )

    def _frames_from_specs(
        specs: list[tuple[str, dict, dict]],
    ) -> list[tuple[dict, pd.DataFrame]]:
        out: list[tuple[dict, pd.DataFrame]] = []
        for sid, meta, _bands in specs:
            df = scanned.get(sid, pd.DataFrame())
            if not df.empty:
                row = df.copy()
                row["strategy_id"] = sid
                row["strategy_title"] = str(meta.get("title", sid))
                out.append((meta, row))
            else:
                out.append((meta, df))
        return out

    frames = _frames_from_specs(strategy_specs)
    scalp_frames = _frames_from_specs(scalp_specs)

    def _write_strategy_frames(
        base: Path,
        frame_list: list[tuple[dict, pd.DataFrame]],
        *,
        log_prefix: str,
    ) -> dict[str, dict]:
        base.mkdir(parents=True, exist_ok=True)
        per: dict[str, dict] = {}
        for meta, df in frame_list:
            sid = str(meta.get("id", "unknown"))
            sub = base / sid
            sub.mkdir(parents=True, exist_ok=True)
            if df.empty:
                pd.DataFrame().to_csv(sub / "signals.csv", index=False, encoding="utf-8-sig")
                pd.DataFrame().to_csv(sub / "now.csv", index=False, encoding="utf-8-sig")
                per[sid] = {"rows": 0, "buy": 0, "watch": 0}
                continue
            now = df[df["stage"] == "可短打"].copy()
            df.to_csv(sub / "signals.csv", index=False, encoding="utf-8-sig")
            now.to_csv(sub / "now.csv", index=False, encoding="utf-8-sig")
            n_buy = int((df["stage"] == "可短打").sum())
            n_watch = int((df["stage"] == "观察").sum())
            per[sid] = {"rows": len(df), "buy": n_buy, "watch": n_watch}
            log(f"    {sid}: 可短打 {n_buy} / 观察 {n_watch}")
            if len(df) > 0:
                title = f"{log_prefix}/{sid}"
                if meta.get("title"):
                    title = f"{log_prefix}/{sid}({meta.get('title')})"
                log_signal_table(df, title=title)
        return per

    per_stat = _write_strategy_frames(out_dir, frames, log_prefix="short")

    df_out = _merge_short_frames(frames)
    if not df_out.empty:
        now = df_out[df_out["stage"] == "可短打"].copy()
        now.to_csv(out_dir / "now.csv", index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame().to_csv(out_dir / "now.csv", index=False, encoding="utf-8-sig")
    df_out.to_csv(out_dir / "signals.csv", index=False, encoding="utf-8-sig")

    stocks = stocks_from_df(
        df_out, asof=asof, hot_industries=hot_map_for_day_dir(day_dir, asof)
    )
    n_buy = int((df_out["stage"] == "可短打").sum()) if not df_out.empty else 0
    n_watch = int((df_out["stage"] == "观察").sum()) if not df_out.empty else 0
    log(
        f"  short 合并: 共 {len(df_out)} 条（可短打 {n_buy} / 观察 {n_watch}）"
        f" → {out_dir / 'signals.csv'}；HTML 列表 {len(stocks)} 只"
        f"（筛选用时 {time.time() - t0:.1f}s）"
    )
    if len(df_out) > 0:
        log_signal_table(df_out, title="short 合并明细")

    # Scalp 独立目录 + tab
    scalp_dir = day_dir / "scalp"
    scalp_stocks: list[dict] = []
    scalp_per: dict[str, dict] = {}
    if scalp_frames:
        scalp_per = _write_strategy_frames(scalp_dir, scalp_frames, log_prefix="scalp")
        scalp_df = _merge_short_frames(scalp_frames)
        if not scalp_df.empty:
            scalp_df[scalp_df["stage"] == "可短打"].to_csv(
                scalp_dir / "now.csv", index=False, encoding="utf-8-sig"
            )
        else:
            pd.DataFrame().to_csv(scalp_dir / "now.csv", index=False, encoding="utf-8-sig")
        scalp_df.to_csv(scalp_dir / "signals.csv", index=False, encoding="utf-8-sig")
        scalp_stocks = stocks_from_df(
            scalp_df, asof=asof, hot_industries=hot_map_for_day_dir(day_dir, asof)
        )
        sb = int((scalp_df["stage"] == "可短打").sum()) if not scalp_df.empty else 0
        sw = int((scalp_df["stage"] == "观察").sum()) if not scalp_df.empty else 0
        log(
            f"  scalp tab: 共 {len(scalp_df)} 条（可短打 {sb} / 观察 {sw}）"
            f" → {scalp_dir / 'signals.csv'}；HTML 列表 {len(scalp_stocks)} 只"
        )
        if len(scalp_df) > 0:
            log_signal_table(scalp_df, title="scalp 明细")
    else:
        scalp_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame().to_csv(scalp_dir / "signals.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame().to_csv(scalp_dir / "now.csv", index=False, encoding="utf-8-sig")

    short_stat = {
        "rows": len(df_out),
        "buy": n_buy,
        "watch": n_watch,
        "charts": len(stocks),
        "strategies": per_stat,
        "strategy_ids": [m.get("id") for m, _ in frames],
    }
    scalp_stat = {
        "rows": 0,
        "buy": 0,
        "watch": 0,
        "charts": len(scalp_stocks),
        "strategies": scalp_per,
        "strategy_ids": [m.get("id") for m, _ in scalp_frames],
    }
    if scalp_frames:
        sdf = _merge_short_frames(scalp_frames)
        scalp_stat["rows"] = len(sdf)
        scalp_stat["buy"] = int((sdf["stage"] == "可短打").sum()) if not sdf.empty else 0
        scalp_stat["watch"] = int((sdf["stage"] == "观察").sum()) if not sdf.empty else 0
        scalp_stat["charts"] = len(scalp_stocks)

    return short_stat, stocks, scalp_stat, scalp_stocks


def short_has_strict_or_r3_buy(short_stat: dict) -> bool:
    """当日 strict / r3 是否筛出至少一只可短打。"""
    strategies = short_stat.get("strategies") or {}
    for sid in ("strict", "r3"):
        st = strategies.get(sid) or {}
        if int(st.get("buy") or 0) > 0:
            return True
    return False


def list_trade_dates(end: str, start: str | None = None) -> list[str]:
    """从本地日线推断交易日列表（升序）。"""
    dates = None
    db = ROOT / "data" / "db" / "daily.db"
    if db.exists() and db.stat().st_size > 0:
        try:
            import daily_db

            conn = daily_db.connect()
            try:
                # 任取一只主板代码的日期序列
                row = conn.execute(
                    "SELECT code FROM daily_bars WHERE code LIKE '600%' LIMIT 1"
                ).fetchone()
                if row:
                    ser = pd.read_sql_query(
                        "SELECT trade_date FROM daily_bars WHERE code=? ORDER BY trade_date",
                        conn,
                        params=(row[0],),
                    )["trade_date"]
                    dates = pd.to_datetime(ser).dt.strftime("%Y-%m-%d")
            finally:
                conn.close()
        except Exception:
            dates = None
    if dates is None:
        sample = next((ROOT / "data" / "daily_raw").glob("600*.csv"), None)
        if sample is None:
            return []
        df = pd.read_csv(sample, usecols=["日期"])
        dates = pd.to_datetime(df["日期"]).dt.strftime("%Y-%m-%d")
    dates = dates[(dates <= end)]
    if start:
        dates = dates[dates >= start]
    return sorted(dates.unique().tolist())


def prev_trade_dates_before(before: str, n: int = 120) -> list[str]:
    """before 之前的交易日，从近到远。"""
    all_d = list_trade_dates(before)
    earlier = [d for d in all_d if d < before]
    earlier.reverse()
    return earlier[:n]


def write_meta(
    day_dir: Path,
    asof: str,
    mode: str,
    long_stat: dict,
    short_stat: dict,
    treasure_stat: dict | None = None,
    scalp_stat: dict | None = None,
    board_stat: dict | None = None,
    base_stat: dict | None = None,
    relaunch_stat: dict | None = None,
    wyckoff_stat: dict | None = None,
    watch_stat: dict | None = None,
) -> None:
    meta = {
        "date": asof,
        "mode": mode,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "pool_root": str(POOL_ROOT),
        "day_dir": str(day_dir),
        "index_html": str(day_dir / "index.html"),
        "long": long_stat,
        "short": short_stat,
        "watch": watch_stat or {},
        "scalp": scalp_stat or {},
        "treasure": treasure_stat or {},
        "board": board_stat or {},
        "base": base_stat or {},
        "relaunch": relaunch_stat or {},
        "wyckoff": wyckoff_stat or {},
        "workflow": {
            "official": "正式：刷盘中价 quality=intraday；完整 HTML 写入 signal_pool/日期/（交易入口）",
            "close": "收盘：重拉 quality=close；不生成 HTML；短线筛观察并增量入观察簿（不覆盖已在册）",
            "note": "不能用「库最大日期=今日」判断准确——正式盘中价也会写成今日",
            "replay": "新策略补历史：--from-intraday-snapshot --from-date … --to-date …",
        },
    }
    try:
        import daily_db

        snap = daily_db.get_intraday_snapshot(asof)
        if snap:
            meta["intraday_snapshot"] = snap
    except Exception:
        pass
    path = day_dir / "meta.json"
    prev: dict = {}
    if path.exists():
        try:
            prev = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            prev = {}
    for key in (
        "long",
        "short",
        "watch",
        "scalp",
        "treasure",
        "board",
        "base",
        "relaunch",
        "wyckoff",
    ):
        if not meta.get(key) and prev.get(key):
            meta[key] = prev[key]
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"[4/4] meta → {path}")


def pool_day_dir(asof: str, mode: str = "official") -> Path:
    """信号池目录：signal_pool/日期/。"""
    return POOL_ROOT / asof


def run_hot_sectors(asof: str, *, mode: str = "official") -> dict:
    """生成 hot_sectors/YYYY-MM-DD/ 角色 HTML（与信号池一样不再分子目录）。"""
    import gen_hot_sector_html as ghs

    log("[3b/4] 热门板块角色（趋势热 + 超短反抽）…")
    t0 = time.time()
    # 日线已由 run_one_day / 区间预载完成，这里直接用缓存
    meta = ghs.generate(asof=asof, mode="official", log_fn=log)
    log(
        f"  hot: 趋势{meta.get('n_trend', 0)} / 超短{meta.get('n_burst', 0)} / "
        f"可买类{meta.get('n_buyish', 0)} → {meta.get('index_html')} "
        f"（{time.time() - t0:.1f}s）"
    )
    return meta


def run_one_day(
    asof: str,
    *,
    mode: str = "official",
    do_long: bool = True,
    do_short: bool = True,
    do_treasure: bool = True,
    do_hot: bool = True,
    do_board: bool = True,
    do_base: bool = True,
    do_relaunch: bool = True,
    do_wyckoff: bool = True,
    short_bands: Path | None = None,
    short_strategy_ids: list[str] | None = None,
    scalp_strategy_ids: list[str] | None = None,
    preload: bool = True,
    update_watch_book: bool = True,
    send_mail: bool = False,
) -> tuple[dict, dict, dict, dict, dict, dict, dict, dict, dict]:
    """生成单日 signal_pool（+可选 hot_sectors）。

    official：完整筛选并写入 signal_pool/日期/ HTML（交易入口）。
    close：不生成 HTML，仅短线筛「观察」并增量入观察簿。
    """
    empty = ({}, {}, {}, {}, {}, {}, {}, {}, {})

    if mode == "close":
        log(f"收盘 {asof}：不生成信号池 HTML（交易看正式写入的日期目录）")
        if preload:
            preload_daily(asof=asof)
        if not do_short:
            log("  跳过短线，无观察可入簿")
            return empty
        import tempfile

        import watch_book as wb

        with tempfile.TemporaryDirectory(prefix="close_watch_") as tmp:
            tmp_dir = Path(tmp)
            log("[收盘] 短线扫描（仅用于观察簿增量入簿，不落交易目录）…")
            _short_stat, short_stocks, _scalp_stat, _scalp_stocks = run_short(
                tmp_dir,
                asof=asof,
                bands_path=short_bands,
                strategy_ids=short_strategy_ids,
                scalp_strategy_ids=[],  # 收盘不需要 scalp 输出
            )
            n_watch = sum(
                1
                for s in short_stocks
                if str(s.get("advice") or "") == "观察"
            )
            log(f"  当日短线观察候选 {n_watch} 只")
            if update_watch_book:
                wb.enroll_new_only(asof, short_stocks=short_stocks)
            else:
                log("  跳过观察簿（--skip-watch-book）")
        live = pool_day_dir(asof)
        if send_mail:
            import pool_mail

            watch_stocks = load_watch_book_stocks(asof, live, refresh=True)
            log(f"[收盘] 发送观察簿邮件（{len(watch_stocks)} 只）…")
            pool_mail.send_watch_mail(asof, watch_stocks)
        if (live / "index.html").exists():
            log(f"  交易 HTML：{live / 'index.html'}")
        return empty

    day_dir = pool_day_dir(asof, mode)
    day_dir.mkdir(parents=True, exist_ok=True)
    log(f"输出目录: {day_dir}  mode={mode}")

    if preload:
        preload_daily(asof=asof)

    long_stat: dict = {}
    short_stat: dict = {}
    treasure_stat: dict = {}
    scalp_stat: dict = {}
    hot_stat: dict = {}
    board_stat: dict = {}
    base_stat: dict = {}
    relaunch_stat: dict = {}
    wyckoff_stat: dict = {}
    long_stocks: list[dict] = []
    short_stocks: list[dict] = []
    treasure_stocks: list[dict] = []
    scalp_stocks: list[dict] = []
    board_stocks: list[dict] = []
    base_stocks: list[dict] = []
    relaunch_stocks: list[dict] = []
    wyckoff_stocks: list[dict] = []

    def existing(name: str) -> list[dict]:
        return stocks_from_df(
            _read_pool_csv(day_dir / name / "signals.csv"),
            asof=asof,
            hot_industries=hot_map_for_day_dir(day_dir, asof),
        )

    # 先写热门 roles.json，信号池 HTML 才能打「热门」标
    if do_hot:
        hot_stat = run_hot_sectors(asof, mode=mode)
    else:
        log("[3b/4] 跳过热门板块")

    if do_long:
        long_stat, long_stocks = run_long(day_dir, asof=asof)
    else:
        log("[2/4] 跳过中长线")
        long_stocks = existing("long")
    if do_treasure:
        treasure_stat, treasure_stocks = run_treasure(day_dir, asof=asof)
    else:
        log("[2b/4] 跳过宝藏观察")
        treasure_stocks = existing("treasure")
    if do_board:
        board_stat, board_stocks = run_board(day_dir, asof=asof)
    else:
        log("[2d/4] 跳过涨停箱体回踩")
        board_stocks = existing("board")
    if do_base:
        base_stat, base_stocks = run_base(day_dir, asof=asof)
    else:
        log("[2e/4] 跳过底部启动")
        base_stocks = existing("base")
    if do_relaunch:
        relaunch_stat, relaunch_stocks = run_relaunch(day_dir, asof=asof)
    else:
        log("[2f/4] 跳过板后重启")
        relaunch_stocks = existing("relaunch")
    if do_wyckoff:
        wyckoff_stat, wyckoff_stocks = run_wyckoff(day_dir, asof=asof)
    else:
        log("[2g/4] 跳过威科夫实验")
        wyckoff_stocks = existing("wyckoff")
    watch_stocks: list[dict] = []
    watch_stat: dict = {}
    if do_short:
        short_stat, short_stocks, scalp_stat, scalp_stocks = run_short(
            day_dir,
            asof=asof,
            bands_path=short_bands,
            strategy_ids=short_strategy_ids,
            scalp_strategy_ids=scalp_strategy_ids,
        )
        if update_watch_book:
            import watch_book as wb

            log("[3c/4] 短线观察簿复检…")
            book = wb.review_day(asof, short_stocks=short_stocks, day_dir=day_dir)
            n_up = len(book.get("promoted") or [])
            if n_up:
                log(f"  观察簿升级 {n_up} 只（仅记入观察簿状态，不并入短线 tab）")
        watch_stocks = load_watch_book_stocks(asof, day_dir, refresh=True)
        watch_stat = {
            "rows": len(watch_stocks),
            "watch": len(watch_stocks),
            "charts": len(watch_stocks),
        }
        log(f"  观察簿 tab: {len(watch_stocks)} 只 → {day_dir / 'watch' / 'signals.csv'}")
    else:
        log("[3/4] 跳过短线")
        short_stocks = existing("short")
        scalp_stocks = existing("scalp")
        watch_stocks = load_watch_book_stocks(asof, day_dir, refresh=False)
        watch_stat = {"rows": len(watch_stocks), "watch": len(watch_stocks)}

    if (
        do_long
        or do_short
        or do_treasure
        or do_board
        or do_base
        or do_relaunch
        or do_wyckoff
    ):
        from plot_watch_pool import build_html

        panels = {
            "long": long_stocks,
            "short": short_stocks,
            "watch": watch_stocks,
            "scalp": scalp_stocks,
            "treasure": treasure_stocks,
            "board": board_stocks,
            "base": base_stocks,
            "relaunch": relaunch_stocks,
            "wyckoff": wyckoff_stocks,
        }
        out_html = day_dir / "index.html"
        build_html(
            asof,
            [],
            out_html,
            days=180,
            panels=panels,
        )
        log(
            f"  合并图: long={len(long_stocks)} short={len(short_stocks)} "
            f"watch={len(watch_stocks)} scalp={len(scalp_stocks)} "
            f"treasure={len(treasure_stocks)} board={len(board_stocks)} "
            f"base={len(base_stocks)} relaunch={len(relaunch_stocks)} "
            f"wyckoff={len(wyckoff_stocks)} → {out_html}"
        )
        write_meta(
            day_dir,
            asof,
            mode,
            long_stat,
            short_stat,
            treasure_stat,
            scalp_stat=scalp_stat,
            board_stat=board_stat,
            base_stat=base_stat,
            relaunch_stat=relaunch_stat,
            wyckoff_stat=wyckoff_stat,
            watch_stat=watch_stat,
        )
        if send_mail:
            import pool_mail

            log("[4b/4] 发送邮件摘要（无 K 线）…")
            pool_mail.send_pool_mail(asof, panels)

    return (
        long_stat,
        short_stat,
        treasure_stat,
        scalp_stat,
        hot_stat,
        board_stat,
        base_stat,
        relaunch_stat,
        wyckoff_stat,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="日更 + 长/短线信号池 → signal_pool")
    parser.add_argument(
        "--close",
        action="store_true",
        help="收盘：重拉 K 线 + 观察增量入簿，不生成交易 HTML",
    )
    parser.add_argument(
        "--mail",
        action="store_true",
        help="跑完后发 QQ 邮件（默认不发）：正式=信号池摘要，收盘=观察簿；需 .env",
    )
    parser.add_argument("--skip-update", action="store_true")
    parser.add_argument(
        "--rebuild-html",
        action="store_true",
        help="用已有 signals.csv 重刷全部 signal_pool K 线 HTML（不更新日线、不重筛）",
    )
    parser.add_argument(
        "--force-update",
        action="store_true",
        help="强制重拉当日日线并按当前 mode（official/close）重写质量标记",
    )
    parser.add_argument("--long-only", action="store_true")
    parser.add_argument("--short-only", action="store_true")
    parser.add_argument("--treasure-only", action="store_true")
    parser.add_argument("--board-only", action="store_true", help="只跑涨停箱体回踩（其它 tab 用已有 CSV）")
    parser.add_argument("--base-only", action="store_true", help="只跑底部横盘启动（其它 tab 用已有 CSV）")
    parser.add_argument("--relaunch-only", action="store_true", help="只跑板后重启（其它 tab 用已有 CSV）")
    parser.add_argument("--wyckoff-only", action="store_true", help="只跑威科夫实验 tab（其它 tab 用已有 CSV）")
    parser.add_argument("--hot-only", action="store_true", help="只跑热门板块角色 HTML")
    parser.add_argument("--skip-treasure", action="store_true", help="不跑宝藏观察池")
    parser.add_argument("--skip-board", action="store_true", help="不跑涨停箱体回踩")
    parser.add_argument("--skip-base", action="store_true", help="不跑底部启动")
    parser.add_argument("--skip-relaunch", action="store_true", help="不跑板后重启")
    parser.add_argument("--skip-wyckoff", action="store_true", help="不跑威科夫实验")
    parser.add_argument("--skip-hot", action="store_true", help="不跑热门板块")
    parser.add_argument(
        "--skip-watch-book",
        action="store_true",
        help="不更新短线观察簿（区间回放默认也不更新，可用 --watch-book 强制）",
    )
    parser.add_argument(
        "--watch-book",
        action="store_true",
        help="强制更新观察簿（含 --from-date/--from-intraday-snapshot 回放）",
    )
    parser.add_argument(
        "--from-intraday-snapshot",
        action="store_true",
        help="用已存盘中截面回放筛选（正式模式；自动跳过日更；无截面的交易日跳过）",
    )
    parser.add_argument("--date", default="")
    parser.add_argument(
        "--from-date",
        default="",
        help="区间起始日 YYYY-MM-DD（与 --to-date 合用，跳过日线更新）",
    )
    parser.add_argument(
        "--to-date",
        default="",
        help="区间结束日 YYYY-MM-DD",
    )
    parser.add_argument(
        "--until-strict-r3",
        action="store_true",
        help="若 --from/--to 区间内 strict/r3 均无「可短打」，则继续往前生成，直到筛出一只为止",
    )
    parser.add_argument(
        "--short-bands",
        default="",
        help="短线单 bands JSON（覆盖多策略合并）",
    )
    parser.add_argument(
        "--short-strategy",
        default="",
        choices=["", "default", "strict", "r3", "scalp"],
        help="仅跑单策略：default | strict | r3 | scalp（覆盖多策略合并）",
    )
    args = parser.parse_args()

    t0 = time.time()
    if args.rebuild_html:
        n = rebuild_all_pool_html()
        log(f"完成，重刷 {n} 份 HTML，耗时 {time.time() - t0:.0f}s")
        return

    mode = "close" if args.close else "official"
    use_snap = bool(args.from_intraday_snapshot)
    if use_snap and mode != "official":
        raise SystemExit("--from-intraday-snapshot 不能与 --close 同时使用")

    do_long = True
    do_short = True
    do_treasure = not args.skip_treasure
    do_board = not args.skip_board
    do_base = not args.skip_base
    do_relaunch = not args.skip_relaunch
    do_wyckoff = not args.skip_wyckoff
    do_hot = not args.skip_hot
    if args.hot_only:
        do_long = False
        do_short = False
        do_treasure = False
        do_board = False
        do_base = False
        do_relaunch = False
        do_wyckoff = False
        do_hot = True
    elif args.wyckoff_only:
        do_long = False
        do_short = False
        do_treasure = False
        do_board = False
        do_base = False
        do_relaunch = False
        do_wyckoff = True
        do_hot = False
    elif args.relaunch_only:
        do_long = False
        do_short = False
        do_treasure = False
        do_board = False
        do_base = False
        do_relaunch = True
        do_wyckoff = False
        do_hot = False
    elif args.base_only:
        do_long = False
        do_short = False
        do_treasure = False
        do_board = False
        do_base = True
        do_relaunch = False
        do_wyckoff = False
        do_hot = False
    elif args.board_only:
        do_long = False
        do_short = False
        do_treasure = False
        do_board = True
        do_base = False
        do_relaunch = False
        do_wyckoff = False
        do_hot = False
    elif args.treasure_only:
        do_long = False
        do_short = False
        do_treasure = True
        do_board = False
        do_base = False
        do_relaunch = False
        do_wyckoff = False
        do_hot = False
    elif args.long_only:
        do_short = False
        do_board = False
        do_base = False
        do_relaunch = False
        do_wyckoff = False
        do_hot = False
    elif args.short_only:
        do_long = False
        do_treasure = False
        do_board = False
        do_base = False
        do_relaunch = False
        do_wyckoff = False
        do_hot = False

    short_bands = Path(args.short_bands) if args.short_bands.strip() else None
    short_ids: list[str] | None = None
    scalp_ids: list[str] | None = None
    if args.short_strategy:
        if args.short_strategy == "scalp":
            short_ids = []
            scalp_ids = ["scalp"]
            log("仅 Scalp tab")
        else:
            short_ids = [args.short_strategy]
            scalp_ids = []
            log(f"短线单策略 {args.short_strategy}")
    elif short_bands is None and do_short:
        short_ids = daily_short_strategy_ids()
        scalp_ids = daily_scalp_strategy_ids()
        log(f"短线多策略合并: {', '.join(short_ids)}；Scalp tab: {', '.join(scalp_ids)}")
    elif do_short:
        scalp_ids = []

    from_d = args.from_date.strip()
    to_d = args.to_date.strip()
    # 区间/截面回放默认不写观察簿，避免历史日把「在册」弄脏；单日正式/收盘默认写
    update_watch_book = bool(args.watch_book) or (
        not args.skip_watch_book and not use_snap and not (from_d or to_d)
    )
    # 区间回放默认不发邮件，避免连发多封；单日 + --mail 才发
    send_mail = bool(args.mail) and not (from_d or to_d)
    if args.mail and (from_d or to_d):
        log("提示：区间回放不发邮件（仅单日 + --mail）")

    def run_day(d: str, *, preload: bool) -> tuple:
        import daily_cache

        if use_snap:
            info = daily_cache.set_intraday_overlay(d)
            if not info:
                msg = (
                    f"无盘中截面 {d}（查看：.venv/bin/python daily_db.py snapshots）"
                )
                if preload:
                    raise SystemExit(msg)
                log(f"  跳过 {d}：{msg}")
                return ({}, {}, {}, {}, {}, {}, {}, {}, {})
            log(
                f"  回放截面 {info['trade_date']} @ {info['snapshot_at']}  "
                f"n={info.get('n_loaded', info.get('n_bars'))}"
            )
        try:
            return run_one_day(
                d,
                mode=mode,
                do_long=do_long,
                do_short=do_short,
                do_treasure=do_treasure,
                do_hot=do_hot,
                do_board=do_board,
                do_base=do_base,
                do_relaunch=do_relaunch,
                do_wyckoff=do_wyckoff,
                short_bands=short_bands,
                short_strategy_ids=short_ids,
                scalp_strategy_ids=scalp_ids,
                preload=preload,
                update_watch_book=update_watch_book,
                send_mail=send_mail,
            )
        finally:
            if use_snap:
                daily_cache.set_intraday_overlay(None)

    if from_d or to_d:
        if not (from_d and to_d):
            raise SystemExit("--from-date 与 --to-date 需同时指定")
        if use_snap or not args.skip_update:
            log(
                "[1/4] 区间回填模式，自动跳过日线更新"
                + ("（回放盘中截面）" if use_snap else "（可用 --skip-update 显式）")
            )
        else:
            log("[1/4] 跳过日线更新")
        clear_feature_cache()
        preload_daily(asof=to_d)

        dates = list_trade_dates(to_d, start=from_d)
        if not dates:
            raise SystemExit(f"区间 {from_d}..{to_d} 无交易日")
        log(f"区间交易日 {len(dates)} 个: {dates[0]} → {dates[-1]}")

        hit = False
        for d in dates:
            log(f"\n===== {d} =====")
            _, short_stat, _, _, _, _, _, _, _ = run_day(d, preload=False)
            if do_short and short_has_strict_or_r3_buy(short_stat):
                hit = True
                log(f"  ✓ {d} strict/r3 有可短打")

        if args.until_strict_r3 and do_short and not hit:
            log("\n区间内无 strict/r3 可短打，往前继续生成…")
            for d in prev_trade_dates_before(from_d, n=180):
                log(f"\n===== 回溯 {d} =====")
                _, short_stat, _, _, _, _, _, _, _ = run_day(d, preload=False)
                if short_has_strict_or_r3_buy(short_stat):
                    log(f"  ✓ 回溯命中 {d}（strict/r3 有可短打），停止")
                    hit = True
                    break
            if not hit:
                log("警告：回溯 180 个交易日仍无 strict/r3 可短打")

        log(f"完成，耗时 {time.time() - t0:.0f}s")
        return

    if use_snap:
        args.skip_update = True
        log("[1/4] 回放盘中截面，跳过日线更新")
    elif not args.skip_update:
        quality = bar_quality_for_mode(mode)
        if args.force_update:
            run_update(force=True, quality=quality)
        elif should_skip_daily_update(mode):
            info = read_last_update()
            log(
                f"[1/4] 跳过日线更新：已有同日收盘"
                f"（{info.get('trade_date')} @ {info.get('updated_at')}；"
                f"正式盘中不会跳过，可用 --force-update 强刷）"
            )
        else:
            # 正式 / 未做过收盘：必须 force，才能覆盖盘中写入的当日 bar
            run_update(force=True, quality=quality)
    else:
        log("[1/4] 跳过日线更新")

    clear_feature_cache()
    asof = args.date.strip() or resolve_asof()
    _, _, _, _, hot_stat, _, _, _ = run_day(asof, preload=True)
    log(f"完成，耗时 {time.time() - t0:.0f}s")
    if mode == "official" and (
        do_long
        or do_short
        or do_treasure
        or do_board
        or do_base
        or do_relaunch
        or do_wyckoff
    ):
        log(f"打开信号池: {pool_day_dir(asof, mode) / 'index.html'}")
    elif mode == "close":
        live = pool_day_dir(asof)
        if (live / "index.html").exists():
            log(f"交易 HTML: {live / 'index.html'}")
    if do_hot and mode == "official" and hot_stat.get("index_html"):
        log(f"打开热门板块: {hot_stat['index_html']}")


if __name__ == "__main__":
    main()
