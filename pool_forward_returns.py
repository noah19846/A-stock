"""
信号池 / 热门推荐 → 次日起逐日涨跌幅（HTML）

每个选股日七张独立表（页内 tab）：
  1) 短线可短打
  2) 待观察（专门观察簿，按选股日在册）
  3) 长线可买入
  4) 涨停箱体回踩候选
  5) 底部启动可买入
  6) 板后重启可买入
  7) 热门可买 / 轻仓可买

选出日 T 的名单，打印 T 之后每个交易日的涨跌幅，直到日线最新。
默认从最早有信号池的日期起，按选股日分段生成一张 HTML。
数据来自 signal_pool/日期/。

用法：
  .venv/bin/python pool_forward_returns.py
  .venv/bin/python pool_forward_returns.py 7-24
  .venv/bin/python pool_forward_returns.py 2026-08-13 --only
  .venv/bin/python pool_forward_returns.py 8-13 --stdout
"""

from __future__ import annotations

import argparse
import html
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import daily_cache

ROOT = Path(__file__).resolve().parent
POOL_ROOT = ROOT / "signal_pool"
HOT_ROOT = ROOT / "hot_sectors"
OUT_HTML = ROOT / "forward_returns" / "index.html"

HOT_BUY_ACTIONS = {"可买", "轻仓可买"}


def log(msg: str) -> None:
    print(msg, flush=True)


def resolve_pick_date(raw: str) -> str:
    """7-24 / 07-24 / 2026-07-24 / 20260724 → YYYY-MM-DD。"""
    s = str(raw).strip().replace("/", "-")
    if re.fullmatch(r"\d{8}", s):
        return f"{s[:4]}-{s[4:6]}-{s[6:8]}"
    if re.fullmatch(r"\d{4}-\d{1,2}-\d{1,2}", s):
        y, m, d = s.split("-")
        return f"{int(y):04d}-{int(m):02d}-{int(d):02d}"
    if re.fullmatch(r"\d{1,2}-\d{1,2}", s):
        m, d = s.split("-")
        years = sorted(
            {
                p.name[:4]
                for p in POOL_ROOT.iterdir()
                if p.is_dir() and re.fullmatch(r"\d{4}-\d{2}-\d{2}", p.name)
            },
            reverse=True,
        )
        year = years[0] if years else str(datetime.now().year)
        return f"{year}-{int(m):02d}-{int(d):02d}"
    raise SystemExit(f"无法解析日期: {raw}")


def list_pool_dates() -> list[str]:
    return sorted(
        p.name
        for p in POOL_ROOT.iterdir()
        if p.is_dir() and re.fullmatch(r"\d{4}-\d{2}-\d{2}", p.name)
    )


def pool_day_dir(pick_date: str) -> Path | None:
    """读 signal_pool/日期/。"""
    base = POOL_ROOT / pick_date
    return base if base.is_dir() else None


def hot_day_dir(pick_date: str) -> Path | None:
    """读 hot_sectors/日期/。"""
    base = HOT_ROOT / pick_date
    return base if base.is_dir() else None


def _read_signal_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, dtype={"code": str})
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    if df.empty or "code" not in df.columns:
        return pd.DataFrame()
    df["code"] = df["code"].astype(str).str.zfill(6)
    if "name" not in df.columns:
        df["name"] = ""
    if "stage" not in df.columns:
        df["stage"] = ""
    return df


def _watch_open_asof(pick_date: str) -> pd.DataFrame:
    """观察簿在选股日仍在册的票（含当日新入、当日尚未关闭）。"""
    try:
        import daily_db
    except Exception:
        return pd.DataFrame()
    if not daily_db.db_exists():
        return pd.DataFrame()
    conn = daily_db.connect()
    try:
        rows = conn.execute(
            """
            SELECT code, name, opened_date, opened_reason, last_note, status
            FROM watch_book
            WHERE opened_date <= ?
              AND (status = 'open'
                   OR closed_date IS NULL
                   OR closed_date > ?)
            """,
            (pick_date, pick_date),
        ).fetchall()
    except Exception:
        return pd.DataFrame()
    finally:
        conn.close()
    out = []
    for r in rows:
        code = str(r["code"] or "").zfill(6)
        if not code or code == "000000":
            continue
        reason = str(r["opened_reason"] or "")
        note = str(r["last_note"] or "").strip()
        detail = note or reason or "观察簿"
        out.append(
            {
                "code": code,
                "name": str(r["name"] or ""),
                "source": "待观察",
                "src_detail": detail,
            }
        )
    return pd.DataFrame(out)


def load_watch_names(pick_date: str) -> pd.DataFrame:
    """专门待观察池：当日 watch/signals.csv，没有则按观察簿还原当日在册。"""
    parts: list[pd.DataFrame] = []
    d = pool_day_dir(pick_date)
    if d is not None:
        wdf = _read_signal_csv(d / "watch" / "signals.csv")
        if not wdf.empty:
            w = wdf.copy()
            w["source"] = "待观察"
            if "when" in w.columns:
                w["src_detail"] = w["when"].astype(str)
            elif "opened_reason" in w.columns:
                w["src_detail"] = w["opened_reason"].astype(str)
            else:
                w["src_detail"] = (
                    w["stage"].astype(str) if "stage" in w.columns else "观察"
                )
            parts.append(w[["code", "name", "source", "src_detail"]])
    if not parts:
        book = _watch_open_asof(pick_date)
        if not book.empty:
            parts.append(book)
    return _dedupe_picks(parts, {"待观察": 0})


def load_short_buys(pick_date: str) -> pd.DataFrame:
    d = pool_day_dir(pick_date)
    if d is None:
        return pd.DataFrame()
    df = _read_signal_csv(d / "short" / "signals.csv")
    if df.empty:
        return df
    if "can_trade" in df.columns:
        out = df[df["can_trade"].astype(str).str.lower().isin(["true", "1"])]
    else:
        out = df[df["stage"].astype(str) == "可短打"]
    out = out.copy()
    out["source"] = "短线"
    out["src_detail"] = out["stage"].astype(str)
    return out[["code", "name", "source", "src_detail"]]


def load_long_buys(pick_date: str) -> pd.DataFrame:
    d = pool_day_dir(pick_date)
    if d is None:
        return pd.DataFrame()
    df = _read_signal_csv(d / "long" / "signals.csv")
    if df.empty:
        return df
    if "can_buy" in df.columns:
        out = df[df["can_buy"].astype(str).str.lower().isin(["true", "1"])]
    else:
        out = df[df["stage"].astype(str) == "可买入"]
    out = out.copy()
    out["source"] = "长线"
    out["src_detail"] = out["stage"].astype(str)
    return out[["code", "name", "source", "src_detail"]]


def load_hot_buys(pick_date: str) -> pd.DataFrame:
    d = hot_day_dir(pick_date)
    if d is None:
        return pd.DataFrame()
    path = d / "roles.json"
    if not path.exists():
        # final 目录有时 roles 在父级
        alt = HOT_ROOT / pick_date / "roles.json"
        path = alt if alt.exists() else path
    if not path.exists():
        return pd.DataFrame()
    import json

    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("actions_buyish") or []
    out_rows = []
    for r in rows:
        action = str(r.get("action") or "")
        if action not in HOT_BUY_ACTIONS:
            continue
        code = str(r.get("code") or "").zfill(6)
        if not code or code == "000000":
            continue
        detail = action
        role = str(r.get("role") or "")
        tier = str(r.get("tier") or "")
        if role or tier:
            detail = f"{action}/{role or '-'}·{tier or '-'}"
        out_rows.append(
            {
                "code": code,
                "name": str(r.get("name") or ""),
                "source": "热门",
                "src_detail": detail,
            }
        )
    return pd.DataFrame(out_rows)


def _dedupe_picks(parts: list[pd.DataFrame], rank: dict[str, int]) -> pd.DataFrame:
    parts = [p for p in parts if p is not None and not p.empty]
    if not parts:
        return pd.DataFrame(columns=["code", "name", "sources", "details"])
    all_df = pd.concat(parts, ignore_index=True)
    rows = []
    for code, g in all_df.groupby("code", sort=False):
        name = next((str(x) for x in g["name"] if str(x).strip()), "")
        sources = list(dict.fromkeys(g["source"].tolist()))
        details = list(dict.fromkeys(g["src_detail"].astype(str).tolist()))
        rows.append(
            {
                "code": str(code).zfill(6),
                "name": name,
                "sources": "+".join(sources),
                "details": " | ".join(details),
            }
        )
    out = pd.DataFrame(rows)

    def sort_key(s: str) -> tuple:
        bits = s.split("+")
        return (min(rank.get(p, 9) for p in bits), s)

    out["_k"] = out["sources"].map(sort_key)
    return out.sort_values(["_k", "code"]).drop(columns=["_k"]).reset_index(drop=True)


def load_board_buys(pick_date: str) -> pd.DataFrame:
    d = pool_day_dir(pick_date)
    if d is None:
        return pd.DataFrame()
    df = _read_signal_csv(d / "board" / "signals.csv")
    if df.empty:
        return df
    if "can_buy" in df.columns:
        out = df[df["can_buy"].astype(str).str.lower().isin(["true", "1"])]
    else:
        out = df[df["stage"].astype(str) == "可买入"]
    out = out.copy()
    out["source"] = "涨停箱体回踩"
    tags = out["strategy_tags"] if "strategy_tags" in out.columns else out["stage"]
    out["src_detail"] = tags.astype(str)
    return out[["code", "name", "source", "src_detail"]]


def load_base_buys(pick_date: str) -> pd.DataFrame:
    d = pool_day_dir(pick_date)
    if d is None:
        return pd.DataFrame()
    df = _read_signal_csv(d / "base" / "signals.csv")
    if df.empty:
        return df
    if "can_buy" in df.columns:
        out = df[df["can_buy"].astype(str).str.lower().isin(["true", "1"])]
    else:
        out = df[df["stage"].astype(str) == "可买入"]
    out = out.copy()
    out["source"] = "底部启动"
    tags = out["strategy_tags"] if "strategy_tags" in out.columns else out["stage"]
    out["src_detail"] = tags.astype(str)
    return out[["code", "name", "source", "src_detail"]]


def load_relaunch_buys(pick_date: str) -> pd.DataFrame:
    d = pool_day_dir(pick_date)
    if d is None:
        return pd.DataFrame()
    df = _read_signal_csv(d / "relaunch" / "signals.csv")
    if df.empty:
        return df
    if "can_buy" in df.columns:
        out = df[df["can_buy"].astype(str).str.lower().isin(["true", "1"])]
    else:
        out = df[df["stage"].astype(str) == "可买入"]
    out = out.copy()
    out["source"] = "板后重启"
    tags = out["strategy_tags"] if "strategy_tags" in out.columns else out["stage"]
    out["src_detail"] = tags.astype(str)
    return out[["code", "name", "source", "src_detail"]]


def as_picks(raw: pd.DataFrame, source: str) -> pd.DataFrame:
    """单来源名单 → compute_forward 用的 code/name/sources/details。"""
    return _dedupe_picks([raw], {source: 0})


def load_hot_picks(pick_date: str) -> pd.DataFrame:
    """hot_sectors：可买 / 轻仓可买。"""
    raw = load_hot_buys(pick_date)
    if raw.empty:
        return pd.DataFrame(columns=["code", "name", "sources", "details"])
    return _dedupe_picks([raw], {"热门": 0})


def forward_dates(df: pd.DataFrame, pick_date: str) -> list[pd.Timestamp]:
    cutoff = pd.Timestamp(pick_date)
    dates = pd.to_datetime(df["日期"])
    return [d for d in dates if d > cutoff]


def fmt_pct(x: float) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    return f"{x * 100:+.2f}"


def fmt_md(d: pd.Timestamp | str) -> str:
    if isinstance(d, str):
        d = pd.Timestamp(d)
    return f"{d.month:02d}-{d.day:02d}"


def compute_forward(
    picks: pd.DataFrame, pick_date: str, asof: str | None
) -> tuple[list[pd.Timestamp], list[dict], dict[pd.Timestamp, list[float]]]:
    """返回 (日期列, 行数据, 每日收益列表)。"""
    cols: list[pd.Timestamp] = []
    rows_out: list[dict] = []
    day_sums: dict[pd.Timestamp, list[float]] = {}

    if picks.empty:
        return cols, rows_out, day_sums

    all_dates: set[pd.Timestamp] = set()
    series: dict[str, dict[pd.Timestamp, float]] = {}
    for code in picks["code"]:
        df = daily_cache.get(code, asof=asof)
        if df is None or df.empty:
            series[code] = {}
            continue
        m: dict[pd.Timestamp, float] = {}
        dates = pd.to_datetime(df["日期"])
        rets = df["ret"].to_numpy(dtype=float)
        for d in forward_dates(df, pick_date):
            idxs = np.where(dates == d)[0]
            if len(idxs) == 0:
                continue
            r = float(rets[int(idxs[0])])
            m[d] = r
            all_dates.add(d)
        series[code] = m

    cols = sorted(all_dates)
    day_sums = {d: [] for d in cols}

    for _, p in picks.iterrows():
        code = p["code"]
        m = series.get(code, {})
        cum = 1.0
        ok = bool(m) and all(d in m and np.isfinite(m[d]) for d in cols)
        day_rets = []
        for d in cols:
            r = m.get(d, np.nan)
            day_rets.append(r)
            if np.isfinite(r):
                day_sums[d].append(r)
                cum *= 1.0 + r
            else:
                ok = False
        rows_out.append(
            {
                "code": code,
                "name": p.get("name") or "",
                "sources": p.get("sources") or "",
                "details": p.get("details") or "",
                "rets": day_rets,
                "cum": (cum - 1.0) if ok and cols else np.nan,
                "missing": False,  # 无次日涨跌仍列出当日选股
            }
        )
    return cols, rows_out, day_sums


def eq_cum_of(cols: list[pd.Timestamp], day_sums: dict) -> float:
    if not cols:
        return np.nan
    cum = 1.0
    for d in cols:
        xs = day_sums[d]
        if not xs:
            return np.nan
        cum *= 1.0 + float(np.mean(xs))
    return cum - 1.0


def make_block(
    key: str,
    title: str,
    picks: pd.DataFrame,
    pick_date: str,
    asof: str | None,
    meta: str,
) -> dict:
    cols, rows, day_sums = compute_forward(picks, pick_date, asof)
    return {
        "key": key,
        "title": title,
        "picks": picks,
        "cols": cols,
        "rows": rows,
        "day_sums": day_sums,
        "meta": meta,
        "eq_cum": eq_cum_of(cols, day_sums),
    }


def print_block(pick_date: str, block: dict) -> None:
    cols = block["cols"]
    rows = block["rows"]
    day_sums = block["day_sums"]
    log(
        f"选股日 {pick_date}  [{block['title']}]  {block['meta']}  → "
        + (
            f"次日起 {fmt_md(cols[0])} … {fmt_md(cols[-1])}（{len(cols)} 日）"
            if cols
            else "尚无次日涨跌"
        )
    )
    if not rows:
        log("")
        return
    head = f"{'代码':<6} {'名称':<8} {'来源':<10}"
    if cols:
        for d in cols:
            head += f" {fmt_md(d):>7}"
        head += f" {'累计':>7}"
    log(head)
    log("-" * len(head))
    for r in rows:
        line = f"{r['code']:<6} {str(r['name'])[:8]:<8} {str(r['sources'])[:10]:<10}"
        if cols:
            for v in r["rets"]:
                line += f" {fmt_pct(v):>7}" if np.isfinite(v) else f" {'—':>7}"
            line += f" {fmt_pct(r['cum']):>7}"
        log(line)
    if cols:
        avg = f"{'等权':<6} {'日均':<8} {'':<10}"
        cum = 1.0
        for d in cols:
            xs = day_sums[d]
            v = float(np.mean(xs)) if xs else np.nan
            avg += f" {fmt_pct(v):>7}"
            if np.isfinite(v):
                cum *= 1.0 + v
        avg += f" {fmt_pct(cum - 1.0):>7}"
        log(avg)
    log("")


def print_section(sec: dict) -> None:
    for block in sec["blocks"]:
        print_block(sec["pick_date"], block)


def _pct_class(x: float) -> str:
    if not np.isfinite(x):
        return "na"
    if x > 0:
        return "up"
    if x < 0:
        return "down"
    return "flat"


def _pct_cell(x: float) -> str:
    cls = _pct_class(x)
    text = fmt_pct(x) if np.isfinite(x) else "—"
    if np.isfinite(x):
        text = f"{x * 100:+.2f}"
    return f'<td class="num {cls}">{html.escape(text)}</td>'


def xueqiu_url(code: str) -> str:
    c = str(code).zfill(6)
    prefix = "SH" if c.startswith(("5", "6", "9")) else "SZ"
    return f"https://xueqiu.com/S/{prefix}{c}"


def name_link(code: str, name: str) -> str:
    label = name or code
    return (
        f'<a class="xq" href="{html.escape(xueqiu_url(code))}" '
        f'target="_blank" rel="noopener noreferrer" '
        f'title="雪球打开 {html.escape(code)} {html.escape(label)}">'
        f"{html.escape(label)}</a>"
    )


def block_codes(block: dict) -> set[str]:
    return {r["code"] for r in block.get("rows") or []}


def render_table_block(block: dict) -> str:
    cols: list[pd.Timestamp] = block["cols"]
    rows: list[dict] = block["rows"]
    day_sums = block["day_sums"]
    meta = block["meta"]

    if not rows:
        return (
            f'<div class="block">'
            f'<p class="muted">{html.escape(meta)} · 无推荐</p></div>'
        )

    date_heads = ""
    if cols:
        date_heads = (
            "".join(f"<th>{html.escape(fmt_md(d))}</th>" for d in cols)
            + "<th>累计</th>"
        )
    thead = f"<tr><th>代码</th><th>名称</th><th>说明</th>{date_heads}</tr>"
    tbody = []
    for r in rows:
        code = r["code"]
        cells = [
            f'<td class="code">{html.escape(code)}</td>',
            f"<td>{name_link(code, str(r['name']))}</td>",
            f'<td class="src" title="{html.escape(str(r["details"]))}">'
            f'{html.escape(str(r["details"] or r["sources"]))}</td>',
        ]
        if cols:
            for v in r["rets"]:
                cells.append(_pct_cell(v))
            cells.append(_pct_cell(r["cum"]))
        tbody.append("<tr>" + "".join(cells) + "</tr>")

    if cols:
        eq_cells = ['<td class="code">等权</td>', "<td>日均</td>", "<td></td>"]
        cum = 1.0
        for d in cols:
            xs = day_sums[d]
            v = float(np.mean(xs)) if xs else np.nan
            eq_cells.append(_pct_cell(v))
            if np.isfinite(v):
                cum *= 1.0 + v
        eq_cells.append(_pct_cell(cum - 1.0))
        tbody.append('<tr class="eq">' + "".join(eq_cells) + "</tr>")
        sub = (
            f"{html.escape(meta)} · 次日 {html.escape(fmt_md(cols[0]))}…"
            f"{html.escape(fmt_md(cols[-1]))} ({len(cols)}日)"
        )
    else:
        sub = f"{html.escape(meta)} · 尚无次日涨跌"

    return (
        f'<div class="block">'
        f'<p class="muted">{sub}</p>'
        f'<div class="wrap"><table>'
        f"<thead>{thead}</thead><tbody>{''.join(tbody)}</tbody>"
        f"</table></div></div>"
    )


def render_panel(sections: list[dict], *, id_prefix: str = "day") -> tuple[str, str]:
    """返回 (侧栏导航 HTML, 主区 HTML)。"""
    nav: list[str] = []
    body: list[str] = []
    for sec in sections:
        pick_date = sec["pick_date"]
        anchor = f"{id_prefix}-{pick_date}"
        blocks = sec.get("blocks") or []
        counts = {b["key"]: len(b["rows"]) for b in blocks}
        n_s = counts.get("short", 0)
        n_w = counts.get("watch", 0)
        n_l = counts.get("long", 0)
        n_b = counts.get("board", 0)
        n_base = counts.get("base", 0)
        n_r = counts.get("relaunch", 0)
        n_h = counts.get("hot", 0)
        nav.append(
            f'<a href="#{anchor}">{html.escape(pick_date)} '
            f'<span class="muted">短{n_s} 观{n_w} 长{n_l} 板{n_b} 底{n_base} 启{n_r} 热{n_h}</span></a>'
        )

        tab_btns = []
        panes = []
        for i, b in enumerate(blocks):
            key = b["key"]
            n = len(b["rows"])
            cls = "active" if i == 0 else ""
            tab_btns.append(
                f'<button type="button" class="{cls}" data-src="{html.escape(key)}">'
                f'{html.escape(b["title"])} <span class="muted">{n}</span></button>'
            )
            pane_cls = "src-pane active" if i == 0 else "src-pane"
            panes.append(
                f'<div class="{pane_cls}" data-src="{html.escape(key)}">'
                f"{render_table_block(b)}</div>"
            )
        if not panes:
            inner = '<p class="muted">无推荐或尚无次日涨跌数据</p>'
        else:
            inner = (
                f'<div class="src-tabs" role="tablist">{"".join(tab_btns)}</div>'
                f'<div class="src-panes">{"".join(panes)}</div>'
            )
        body.append(
            f'<section id="{anchor}" class="day">'
            f"<h2>{html.escape(pick_date)}</h2>"
            f"{inner}</section>"
        )
    return "".join(nav), "".join(body)


def render_html(
    sections: list[dict],
    *,
    start: str,
    end: str,
    generated_at: str,
) -> str:
    nav, body = render_panel(sections, id_prefix="day")

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>选股次日涨跌幅 {html.escape(start)} → {html.escape(end)}</title>
<style>
:root {{
  --bg: #f0f2f5;
  --card: #fff;
  --text: #1a1d21;
  --muted: #5c6570;
  --line: #d8dee6;
  --up: #c62828;
  --down: #2e7d32;
  --nav-w: 220px;
  --accent: #1a56db;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  font-family: "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", sans-serif;
  background: var(--bg);
  color: var(--text);
  line-height: 1.45;
}}
.app {{ display: flex; min-height: 100vh; }}
.nav {{
  position: sticky; top: 0; align-self: flex-start;
  width: var(--nav-w); max-height: 100vh; overflow: auto;
  padding: 16px 12px; background: #f7f8fa; border-right: 1px solid var(--line);
}}
.nav h1 {{ font-size: 15px; margin: 0 0 8px; }}
.nav .sub {{ font-size: 12px; color: var(--muted); margin-bottom: 12px; }}
.nav a {{
  display: block; padding: 6px 8px; border-radius: 6px;
  color: var(--text); text-decoration: none; font-size: 13px;
}}
.nav a:hover {{ background: #e8ebf0; }}
.nav .muted {{ color: var(--muted); font-size: 11px; margin-left: 4px; }}
main {{ flex: 1; padding: 20px 24px 48px; min-width: 0; }}
.day {{
  background: var(--card); border: 1px solid var(--line);
  border-radius: 10px; padding: 14px 16px 18px; margin-bottom: 18px;
}}
.day h2 {{ font-size: 16px; margin: 0 0 10px; font-weight: 600; }}
.src-tabs {{
  display: flex; gap: 4px; margin: 0 0 10px; flex-wrap: wrap;
  background: #e8ebf0; padding: 3px; border-radius: 8px;
}}
.src-tabs button {{
  flex: 1; min-width: 88px; border: 0; background: transparent; color: var(--muted);
  padding: 7px 8px; border-radius: 6px; font-size: 13px; cursor: pointer;
  font-family: inherit;
}}
.src-tabs button.active {{
  background: #fff; color: var(--text); font-weight: 600;
  box-shadow: 0 1px 2px rgba(0,0,0,.06);
}}
.src-pane {{ display: none; }}
.src-pane.active {{ display: block; }}
.block {{ margin-top: 0; }}
.block p.muted {{ font-size: 12px; margin: 0 0 8px; }}
.wrap {{ overflow-x: auto; }}
table {{ border-collapse: collapse; width: max-content; min-width: 100%; font-size: 13px; }}
th, td {{ padding: 5px 8px; border-bottom: 1px solid var(--line); white-space: nowrap; }}
th {{ text-align: right; color: var(--muted); font-weight: 500; position: sticky; top: 0; background: var(--card); }}
th:nth-child(1), th:nth-child(2), th:nth-child(3),
td:nth-child(1), td:nth-child(2), td:nth-child(3) {{ text-align: left; }}
td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
td.up {{ color: var(--up); }}
td.down {{ color: var(--down); }}
td.na, td.flat {{ color: var(--muted); }}
td.code {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }}
td.src {{ color: var(--muted); max-width: 120px; overflow: hidden; text-overflow: ellipsis; }}
a.xq {{ color: var(--accent); text-decoration: none; }}
a.xq:hover {{ text-decoration: underline; }}
tr.eq td {{ font-weight: 600; border-top: 2px solid var(--line); background: #fafbfc; }}
.muted {{ color: var(--muted); }}
@media (max-width: 800px) {{
  .app {{ flex-direction: column; }}
  .nav {{ position: relative; width: 100%; max-height: none; }}
}}
</style>
</head>
<body>
<div class="app">
  <nav class="nav">
    <h1>选股次日涨跌</h1>
    <div class="sub">{html.escape(start)} → {html.escape(end)}<br/>
    每日六表：短线 / 长线 / 涨停箱体回踩 / 底部启动 / 板后重启 / 热门<br/>
    数据来自信号池日期目录 · 名称点进雪球<br/>
    生成 {html.escape(generated_at)}</div>
    <div class="nav-dates">{nav}</div>
  </nav>
  <main>{body}</main>
</div>
<script>
(function () {{
  function activateSrc(src) {{
    document.querySelectorAll('.src-tabs button').forEach(b => {{
      b.classList.toggle('active', b.dataset.src === src);
    }});
    document.querySelectorAll('.src-pane').forEach(p => {{
      p.classList.toggle('active', p.dataset.src === src);
    }});
    try {{ localStorage.setItem('fwd-ret-src', src); }} catch (e) {{}}
  }}
  document.querySelectorAll('.src-tabs button').forEach(b => {{
    b.addEventListener('click', () => activateSrc(b.dataset.src));
  }});
  let src = 'short';
  try {{ src = localStorage.getItem('fwd-ret-src') || 'short'; }} catch (e) {{}}
  if (['short', 'watch', 'long', 'board', 'base', 'relaunch', 'hot'].indexOf(src) >= 0) activateSrc(src);
}})();
</script>
</body>
</html>
"""



def build_sections(
    dates: list[str], asof: str | None
) -> list[dict]:
    sections = []
    for pick_date in dates:
        short = as_picks(load_short_buys(pick_date), "短线")
        watch = load_watch_names(pick_date)
        long = as_picks(load_long_buys(pick_date), "长线")
        board = as_picks(load_board_buys(pick_date), "涨停箱体回踩")
        base = as_picks(load_base_buys(pick_date), "底部启动")
        relaunch = as_picks(load_relaunch_buys(pick_date), "板后重启")
        hot = load_hot_picks(pick_date)
        blocks = [
            make_block("short", "短线", short, pick_date, asof, f"可短打 {len(short)}"),
            make_block("watch", "待观察", watch, pick_date, asof, f"待观察 {len(watch)}"),
            make_block("long", "长线", long, pick_date, asof, f"可买入 {len(long)}"),
            make_block(
                "board",
                "涨停箱体回踩",
                board,
                pick_date,
                asof,
                f"续板候选 {len(board)}",
            ),
            make_block(
                "base",
                "底部启动",
                base,
                pick_date,
                asof,
                f"可买入 {len(base)}",
            ),
            make_block(
                "relaunch",
                "板后重启",
                relaunch,
                pick_date,
                asof,
                f"可买入 {len(relaunch)}",
            ),
            make_block(
                "hot",
                "热门",
                hot,
                pick_date,
                asof,
                f"可买/轻仓可买 {len(hot)}",
            ),
        ]
        sections.append({"pick_date": pick_date, "blocks": blocks})
    return sections


def main() -> None:
    parser = argparse.ArgumentParser(description="选股次日涨跌幅 HTML / 终端表")
    parser.add_argument(
        "date",
        nargs="?",
        default="",
        help="起始选股日（默认从最早信号池日期开始）",
    )
    parser.add_argument(
        "--only",
        action="store_true",
        help="只生成指定那一天（需给 date）",
    )
    parser.add_argument("--asof", default="", help="涨跌截止日 YYYY-MM-DD")
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="同时在终端打印表格",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=str(OUT_HTML),
        help=f"HTML 输出路径（默认 {OUT_HTML}）",
    )
    args = parser.parse_args()

    all_dates = list_pool_dates()
    if not all_dates:
        raise SystemExit(f"没有信号池目录: {POOL_ROOT}")

    if args.date:
        start = resolve_pick_date(args.date)
        if start not in all_dates:
            later = [d for d in all_dates if d >= start]
            if not later:
                raise SystemExit(f"{start} 之后没有信号池日期")
            log(f"注意: {start} 无信号池，从 {later[0]} 起")
            start = later[0]
        if args.only:
            dates = [start]
        else:
            dates = [d for d in all_dates if d >= start]
    else:
        if args.only:
            raise SystemExit("--only 需要指定 date")
        dates = all_dates

    asof = args.asof or None
    log(f"生成选股日 {dates[0]} → {dates[-1]}  共 {len(dates)} 段")
    sections = build_sections(dates, asof=asof)

    if args.stdout:
        for sec in sections:
            print_section(sec)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    html_text = render_html(
        sections,
        start=dates[0],
        end=dates[-1],
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
    )
    out.write_text(html_text, encoding="utf-8")
    log(f"HTML → {out}")


if __name__ == "__main__":
    main()
