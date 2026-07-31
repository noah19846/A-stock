"""
一键日更：更新 K 线 → 中长线埋伏/买入 → 短线可操作 → 写入 signal_pool

目录结构：
  signal_pool/
    YYYY-MM-DD/
      meta.json / index.html / long/ / short/   # final（收盘后）
      preview/
        meta.json / index.html / long/ / short/ # 盘中 14:30 预览，不被 final 覆盖

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
    from update_daily import latest_trade_date

    try:
        return latest_trade_date()
    except Exception:
        sample = next((ROOT / "data" / "daily_raw").glob("600*.csv"), None)
        if sample is None:
            return datetime.now().strftime("%Y-%m-%d")
        df = pd.read_csv(sample, usecols=["日期"])
        return str(pd.to_datetime(df["日期"]).max().date())


def run_update(force: bool = False, *, quality: str = "final") -> None:
    cmd = [PYTHON, str(ROOT / "update_daily.py")]
    if force:
        cmd.append("--force")
    db = ROOT / "data" / "db" / "daily.db"
    if db.exists() and db.stat().st_size > 0:
        cmd.append("--sqlite")
    log(f"[1/4] 更新日线: {' '.join(cmd)}  quality={quality}")
    subprocess.run(cmd, cwd=str(ROOT), check=True)
    # 成功后记下「哪一档」写入了当日，避免把 preview 当成收盘正式数据
    asof = None
    try:
        from update_daily import latest_trade_date

        asof = latest_trade_date()
    except Exception:
        asof = datetime.now().strftime("%Y-%m-%d")
    write_last_update(asof, quality)


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
        "quality": quality,  # preview | final
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    UPDATE_META.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"  日线质量标记: {payload['trade_date']} / {quality}")


def should_skip_daily_update(mode: str) -> bool:
    """是否跳过日更。

    仅看「日期」不够：14:30 preview 用的是盘中截面，日期已是当日但并非收盘准确价。
    规则：
      - preview：不自动跳过（每次预览都刷盘中价）
      - final：仅当标记为同日且 quality=final 时跳过（同日 preview 不能顶替）
    """
    info = read_last_update()
    if not info:
        return False
    if mode != "final":
        return False
    if str(info.get("quality") or "") != "final":
        return False
    mx = str(info.get("trade_date") or "")[:10]
    if not mx:
        return False
    try:
        from update_daily import latest_trade_date

        return mx == latest_trade_date()
    except Exception:
        return mx >= datetime.now().strftime("%Y-%m-%d")


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


def load_short_registry() -> dict:
    path = ROOT / "data" / "short_strategies.json"
    return json.loads(path.read_text(encoding="utf-8"))


def daily_short_strategy_ids() -> list[str]:
    """每日 HTML 展示的短线策略（默认去掉 scalp）。"""
    reg = load_short_registry()
    ids = reg.get("daily_html") or ["default", "strict", "r3"]
    return [str(x) for x in ids if str(x) != "scalp"]


def strategy_by_id(sid: str) -> dict:
    reg = load_short_registry()
    by_id = {s["id"]: s for s in reg.get("strategies", [])}
    if sid not in by_id:
        raise SystemExit(f"未知短线策略 id: {sid}")
    return by_id[sid]


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


def _merge_short_frames(frames: list[tuple[dict, pd.DataFrame]]) -> pd.DataFrame:
    """按 code 合并多策略结果，附 strategy_ids / strategy_tags。

    标签规则：某策略对该股为「可短打」才打该策略标签；
    若某股仅出现在某策略的「观察」且未被其它策略可短打覆盖，则仍列出并打该策略标签。
    """
    by_code: dict[str, dict] = {}
    stage_rank = {"可短打": 0, "观察": 1, "已偏强": 2, "不关注": 3}

    for meta, df in frames:
        if df is None or df.empty:
            continue
        sid = str(meta.get("id", ""))
        title = str(meta.get("title", sid))
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
        item["strategy_ids"] = "|".join(item["strategy_ids"])
        item["strategy_tags"] = "|".join(item["strategy_tags"])
        item["can_trade"] = item["stage"] == "可短打"
        rows.append(item)

    out = pd.DataFrame(rows)
    order = {"可短打": 0, "观察": 1}
    out["_o"] = out["stage"].map(order).fillna(9)
    if "hard_score" in out.columns and "score" in out.columns:
        out = out.sort_values(
            ["_o", "hard_score", "score"], ascending=[True, False, False]
        ).drop(columns=["_o"])
    else:
        out = out.sort_values(["_o"]).drop(columns=["_o"])
    return out.reset_index(drop=True)


def run_short(
    day_dir: Path,
    asof: str | None = None,
    bands_path: Path | None = None,
    strategy_ids: list[str] | None = None,
) -> tuple[dict, list[dict]]:
    """短线筛选。默认合并 daily_html 三套（default/strict/r3）；单策略时用 strategy_ids 或 bands_path。"""
    import short_burst_screener as sbs

    out_dir = day_dir / "short"
    out_dir.mkdir(parents=True, exist_ok=True)
    old_html = out_dir / "index.html"
    if old_html.exists():
        old_html.unlink()

    t0 = time.time()
    # 解析策略列表
    if bands_path is not None and strategy_ids is None:
        ids = ["default"]
        log("[3/4] 短线筛选（单 bands）…")
        strategy_specs: list[tuple[str, dict, dict]] = []  # sid, meta, bands
        band_cfg = sbs.load_bands(bands_path if bands_path.is_absolute() else ROOT / bands_path)
        strategy_specs.append(
            ("default", {"id": "default", "title": "默认", "bands": str(bands_path)}, band_cfg)
        )
    else:
        ids = strategy_ids or daily_short_strategy_ids()
        log(f"[3/4] 短线筛选（多策略一次扫描: {', '.join(ids)}）…")
        strategy_specs = []
        for sid in ids:
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
            strategy_specs.append((sid, meta, band_cfg))

    scanned = sbs.scan_multi(
        [(sid, bands) for sid, _, bands in strategy_specs],
        asof=asof,
        entry_only=False,
    )
    frames: list[tuple[dict, pd.DataFrame]] = []
    for sid, meta, _bands in strategy_specs:
        df = scanned.get(sid, pd.DataFrame())
        if not df.empty:
            out = df.copy()
            out["strategy_id"] = sid
            out["strategy_title"] = str(meta.get("title", sid))
            frames.append((meta, out))
        else:
            frames.append((meta, df))

    # 每策略落盘
    per_stat: dict[str, dict] = {}
    for meta, df in frames:
        sid = str(meta.get("id", "unknown"))
        sub = out_dir / sid
        sub.mkdir(parents=True, exist_ok=True)
        if df.empty:
            pd.DataFrame().to_csv(sub / "signals.csv", index=False, encoding="utf-8-sig")
            pd.DataFrame().to_csv(sub / "now.csv", index=False, encoding="utf-8-sig")
            per_stat[sid] = {"rows": 0, "buy": 0, "watch": 0}
            continue
        now = df[df["stage"] == "可短打"].copy()
        df.to_csv(sub / "signals.csv", index=False, encoding="utf-8-sig")
        now.to_csv(sub / "now.csv", index=False, encoding="utf-8-sig")
        n_buy = int((df["stage"] == "可短打").sum())
        n_watch = int((df["stage"] == "观察").sum())
        per_stat[sid] = {"rows": len(df), "buy": n_buy, "watch": n_watch}
        log(f"    {sid}: 可短打 {n_buy} / 观察 {n_watch}")
        if len(df) > 0:
            title = f"short/{sid}"
            if meta.get("title"):
                title = f"short/{sid}({meta.get('title')})"
            log_signal_table(df, title=title)

    df_out = _merge_short_frames(frames)
    if not df_out.empty:
        now = df_out[df_out["stage"] == "可短打"].copy()
        now.to_csv(out_dir / "now.csv", index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame().to_csv(out_dir / "now.csv", index=False, encoding="utf-8-sig")
    df_out.to_csv(out_dir / "signals.csv", index=False, encoding="utf-8-sig")

    stocks = stocks_from_df(df_out, asof=asof)
    n_buy = int((df_out["stage"] == "可短打").sum()) if not df_out.empty else 0
    n_watch = int((df_out["stage"] == "观察").sum()) if not df_out.empty else 0
    log(
        f"  short 合并: 共 {len(df_out)} 条（可短打 {n_buy} / 观察 {n_watch}）"
        f" → {out_dir / 'signals.csv'}；HTML 列表 {len(stocks)} 只"
        f"（筛选用时 {time.time() - t0:.1f}s）"
    )
    if len(df_out) > 0:
        log_signal_table(df_out, title="short 合并明细")
    return {
        "rows": len(df_out),
        "buy": n_buy,
        "watch": n_watch,
        "charts": len(stocks),
        "strategies": per_stat,
        "strategy_ids": [m.get("id") for m, _ in frames],
    }, stocks


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


def write_meta(day_dir: Path, asof: str, mode: str, long_stat: dict, short_stat: dict) -> None:
    meta = {
        "date": asof,
        "mode": mode,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "pool_root": str(POOL_ROOT),
        "day_dir": str(day_dir),
        "index_html": str(day_dir / "index.html"),
        "long": long_stat,
        "short": short_stat,
        "workflow": {
            "preview": "收盘前预筛：强制拉盘中截面写入库，标记 quality=preview；HTML 在 preview/",
            "final": "收盘后正式：强制重拉覆盖当日 bar，标记 quality=final；仅当已有同日 final 才跳过日更",
            "note": "不能用「库最大日期=今日」判断准确——preview 也会写成今日",
        },
    }
    path = day_dir / "meta.json"
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"[4/4] meta → {path}")


def pool_day_dir(asof: str, mode: str = "final") -> Path:
    """final → signal_pool/日期/；preview → signal_pool/日期/preview/。"""
    base = POOL_ROOT / asof
    if mode == "preview":
        return base / "preview"
    return base


def run_one_day(
    asof: str,
    *,
    mode: str = "final",
    do_long: bool = True,
    do_short: bool = True,
    short_bands: Path | None = None,
    short_strategy_ids: list[str] | None = None,
    preload: bool = True,
) -> tuple[dict, dict]:
    """生成单日 signal_pool。返回 (long_stat, short_stat)。"""
    day_dir = pool_day_dir(asof, mode)
    day_dir.mkdir(parents=True, exist_ok=True)
    log(f"输出目录: {day_dir}  mode={mode}")

    if preload:
        preload_daily(asof=asof)

    long_stat: dict = {}
    short_stat: dict = {}
    long_stocks: list[dict] = []
    short_stocks: list[dict] = []

    if do_long:
        long_stat, long_stocks = run_long(day_dir, asof=asof)
    else:
        log("[2/4] 跳过中长线")
    if do_short:
        short_stat, short_stocks = run_short(
            day_dir,
            asof=asof,
            bands_path=short_bands,
            strategy_ids=short_strategy_ids,
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
    return long_stat, short_stat


def main() -> None:
    parser = argparse.ArgumentParser(description="日更 + 长/短线信号池 → signal_pool")
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--skip-update", action="store_true")
    parser.add_argument(
        "--force-update",
        action="store_true",
        help="强制重拉当日日线并按当前 mode（preview/final）重写质量标记",
    )
    parser.add_argument("--long-only", action="store_true")
    parser.add_argument("--short-only", action="store_true")
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
        choices=["", "default", "scalp", "strict", "r3"],
        help="仅跑单策略：default | scalp | strict | r3（覆盖多策略合并）",
    )
    args = parser.parse_args()

    t0 = time.time()
    mode = "preview" if args.preview else "final"

    do_long = not args.short_only
    do_short = not args.long_only

    short_bands = Path(args.short_bands) if args.short_bands.strip() else None
    short_ids: list[str] | None = None
    if args.short_strategy:
        short_ids = [args.short_strategy]
        log(f"短线单策略 {args.short_strategy}")
    elif short_bands is None:
        short_ids = daily_short_strategy_ids()
        log(f"短线多策略合并: {', '.join(short_ids)}")

    from_d = args.from_date.strip()
    to_d = args.to_date.strip()
    if from_d or to_d:
        if not (from_d and to_d):
            raise SystemExit("--from-date 与 --to-date 需同时指定")
        if not args.skip_update:
            log("[1/4] 区间回填模式，自动跳过日线更新（可用 --skip-update 显式）")
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
            _, short_stat = run_one_day(
                d,
                mode=mode,
                do_long=do_long,
                do_short=do_short,
                short_bands=short_bands,
                short_strategy_ids=short_ids,
                preload=False,
            )
            if do_short and short_has_strict_or_r3_buy(short_stat):
                hit = True
                log(f"  ✓ {d} strict/r3 有可短打")

        if args.until_strict_r3 and do_short and not hit:
            log("\n区间内无 strict/r3 可短打，往前继续生成…")
            for d in prev_trade_dates_before(from_d, n=180):
                log(f"\n===== 回溯 {d} =====")
                _, short_stat = run_one_day(
                    d,
                    mode=mode,
                    do_long=do_long,
                    do_short=do_short,
                    short_bands=short_bands,
                    short_strategy_ids=short_ids,
                    preload=False,
                )
                if short_has_strict_or_r3_buy(short_stat):
                    log(f"  ✓ 回溯命中 {d}（strict/r3 有可短打），停止")
                    hit = True
                    break
            if not hit:
                log("警告：回溯 180 个交易日仍无 strict/r3 可短打")

        log(f"完成，耗时 {time.time() - t0:.0f}s")
        return

    if not args.skip_update:
        if args.force_update:
            # 强制重拉并按当前 mode 记质量档
            run_update(force=True, quality=mode)
        elif should_skip_daily_update(mode):
            info = read_last_update()
            log(
                f"[1/4] 跳过日线更新：已有同日 final"
                f"（{info.get('trade_date')} @ {info.get('updated_at')}；"
                f"preview 不会跳过，可用 --force-update 强刷）"
            )
        else:
            # preview / 未做过 final：必须 force，才能覆盖盘中写入的当日 bar
            run_update(force=True, quality=mode)
    else:
        log("[1/4] 跳过日线更新")

    clear_feature_cache()
    asof = args.date.strip() or resolve_asof()
    run_one_day(
        asof,
        mode=mode,
        do_long=do_long,
        do_short=do_short,
        short_bands=short_bands,
        short_strategy_ids=short_ids,
        preload=True,
    )
    log(f"完成，耗时 {time.time() - t0:.0f}s")
    log(f"打开: {pool_day_dir(asof, mode) / 'index.html'}")


if __name__ == "__main__":
    main()
