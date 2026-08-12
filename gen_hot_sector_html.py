"""
生成热门行业角色 HTML：趋势热 + 超短反抽 → hot_sectors/

  .venv/bin/python gen_hot_sector_html.py
  .venv/bin/python gen_hot_sector_html.py --asof 2026-08-06
  .venv/bin/python run_daily_pool.py --hot-only --skip-update
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

import daily_cache

ROOT = Path(__file__).resolve().parent
HOT_ROOT = ROOT / "hot_sectors"
INDUSTRY = ROOT / "data" / "stock_industry.csv"
LOOKBACK = 20
MIN_MEMBERS = 5


def hot_day_dir(asof: str, mode: str = "final") -> Path:
    base = HOT_ROOT / asof
    if mode == "preview":
        return base / "preview"
    return base


def load_stock_frame(asof: str | None) -> tuple[pd.DataFrame, str]:
    ind = pd.read_csv(INDUSTRY, dtype=str)
    ind["股票代码"] = ind["股票代码"].str.zfill(6)
    ind = ind[~ind["股票名称"].str.contains("ST", na=False)]
    ind = ind[ind["股票代码"].str.startswith(("600", "601", "603", "605", "000", "001", "002"))]

    daily_cache.preload()
    rows: list[dict] = []
    for code in ind["股票代码"].unique():
        df = daily_cache.get(code, asof=asof)
        if df is None or len(df) < LOOKBACK + 5:
            continue
        i = len(df) - 1
        px = df["px"].to_numpy(float)
        amt = df["amt"].to_numpy(float)
        daily_rets = np.full(LOOKBACK, np.nan)
        for k in range(1, LOOKBACK):
            j = i - LOOKBACK + 1 + k
            daily_rets[k] = px[j] / px[j - 1] - 1
        start_idx = None
        for k in range(LOOKBACK):
            if daily_rets[k] >= 0.025:
                start_idx = k
                break
        if start_idx is None:
            best, bestv = LOOKBACK - 1, -9.0
            for k in range(2, LOOKBACK):
                j = i - LOOKBACK + 1 + k
                r3 = px[j] / px[j - 3] - 1
                if r3 > bestv:
                    bestv, best = r3, k
            start_idx = best
        amt5 = float(np.nanmean(amt[i - 4 : i + 1]))
        amt20 = float(np.nanmean(amt[i - 19 : i + 1]))
        ma20 = float(df["ma20"].iloc[i]) if "ma20" in df.columns else float("nan")
        above_ma20 = bool(np.isfinite(ma20) and px[i] >= ma20)

        def ret(n: int) -> float:
            return float(px[i] / px[i - n] - 1) * 100

        rows.append(
            {
                "code": code,
                "asof": str(df["日期"].iloc[i].date()),
                "mv": float(df["mv"].iloc[i]),
                "ret5": ret(5),
                "ret10": ret(10),
                "ret20": ret(20),
                "amt5亿": amt5 / 1e8,
                "amt_ratio": (amt5 / amt20) if amt20 > 0 else np.nan,
                "early_days": int(start_idx),
                "above_ma20": above_ma20,
            }
        )

    stock = pd.DataFrame(rows)
    meta = ind[["股票代码", "股票名称", "行业板块"]].drop_duplicates("股票代码")
    meta = meta.rename(columns={"股票代码": "code", "股票名称": "name", "行业板块": "industry"})
    stock = stock.merge(meta, on="code", how="left")
    day = str(stock["asof"].mode().iloc[0])
    return stock, day


def sector_stats(stock: pd.DataFrame) -> pd.DataFrame:
    sec = []
    for name, g in stock.groupby("industry"):
        if len(g) < MIN_MEMBERS:
            continue
        w = g["mv"].clip(lower=0)
        if w.sum() <= 0:
            continue
        r5 = float(g["ret5"].median())
        r10 = float(g["ret10"].median())
        r20 = float(g["ret20"].median())
        r20w = float(np.average(g["ret20"].fillna(0), weights=w))
        breadth = float((g["ret20"] > 0).mean() * 100)
        amt = float(g["amt5亿"].sum())
        heat = 0.35 * r5 + 0.25 * r10 + 0.25 * r20 + 0.15 * r20w + 0.05 * (breadth - 50) / 10
        sec.append(
            {
                "industry": name,
                "n": len(g),
                "med5": round(r5, 2),
                "med10": round(r10, 2),
                "med20": round(r20, 2),
                "w20": round(r20w, 2),
                "breadth20": round(breadth, 1),
                "amt5": round(amt, 1),
                "heat": round(heat, 2),
            }
        )
    return pd.DataFrame(sec).sort_values("heat", ascending=False)


def classify(g: pd.DataFrame) -> pd.DataFrame:
    g = g.copy().sort_values("mv", ascending=False).reset_index(drop=True)
    g["mv_rank"] = np.arange(1, len(g) + 1)
    n = len(g)
    mid_cut = max(3, int(np.ceil(n * 0.25)))
    mv_p50 = g["mv"].quantile(0.50)
    amt_p75 = g["amt5亿"].quantile(0.75)
    ret20_p75 = g["ret20"].quantile(0.75)
    ret20_p50 = g["ret20"].quantile(0.50)

    g["leader_score"] = g["ret20"] - g["early_days"] * 1.2 - np.where(g["mv_rank"] == 1, 3.0, 0.0)
    # 超短场景：更看 5 日涨幅 + 早启动
    g["burst_score"] = g["ret5"] - g["early_days"] * 0.8 - np.where(g["mv_rank"] == 1, 2.0, 0.0)
    g["army_score"] = (
        (n + 1 - g["mv_rank"]) * 2
        + g["amt5亿"] / max(amt_p75, 1e-6) * 5
        + (g["amt_ratio"].fillna(1) - 1) * 8
        + (g["ret5"] > 0).astype(float) * 2
    )

    role = pd.Series(["观察"] * n)
    reason = pd.Series([""] * n)

    army_cand = g[g["mv_rank"] <= mid_cut].nlargest(min(4, mid_cut), "army_score")
    for i in army_cand.index:
        if g.loc[i, "amt5亿"] >= amt_p75 * 0.65:
            role[i] = "中军"
            reason[i] = (
                f"市值排#{int(g.loc[i, 'mv_rank'])}/{n}、"
                f"近5日额{g.loc[i, 'amt5亿']:.1f}亿、额能比{g.loc[i, 'amt_ratio']:.2f}"
            )

    # 龙头：最大市值若同时是成交王，优先留中军，另选进攻龙
    lead_pool = g[(g["ret20"] >= ret20_p50) | (g["ret5"] >= g["ret5"].quantile(0.6))]
    lead_pool = lead_pool[lead_pool["early_days"] <= 12] if len(lead_pool) else g
    scored = lead_pool.nlargest(4, "leader_score")
    picked = 0
    for i in scored.index:
        if role[i] == "中军" and g.loc[i, "mv_rank"] == 1:
            continue
        role[i] = "龙头"
        reason[i] = (
            f"20日{g.loc[i, 'ret20']:+.1f}% / 5日{g.loc[i, 'ret5']:+.1f}%、"
            f"窗口第{int(g.loc[i, 'early_days']) + 1}日启动"
        )
        picked += 1
        if picked >= 2:
            break
    if picked == 0:
        i = g.nlargest(1, "burst_score").index[0]
        role[i] = "龙头"
        reason[i] = "板块内涨幅×启动综合最优"

    med_early = g["early_days"].median()
    for i in g.index:
        if role[i] != "观察":
            continue
        r = g.loc[i]
        small = r["mv"] < mv_p50
        late = r["early_days"] >= 12 or r["early_days"] > med_early
        follow = (r["ret5"] >= 5 and r["ret20"] > -5) or r["ret5"] >= 8
        if small and late and follow:
            role[i] = "尾部小票"
            reason[i] = (
                f"市值{r['mv']:.0f}亿偏低、启动偏晚(第{int(r['early_days']) + 1}日)、"
                f"近5日{r['ret5']:+.1f}%"
            )

    out = g.copy()
    out["role"] = role.values
    out["reason"] = reason.values
    # 人工纠偏：最大市值且成交断层 → 强制中军（避免紫金类被标龙头）
    top = out.iloc[0]
    if top["mv_rank"] == 1 and top["amt5亿"] >= amt_p75 and top["role"] == "龙头":
        out.loc[out.index[0], "role"] = "中军"
        out.loc[out.index[0], "reason"] = "市值/成交断层第一 → 中军（非龙头）"
        # 补一个龙头
        others = out[out["role"] != "中军"].nlargest(1, "leader_score")
        if len(others):
            j = others.index[0]
            out.loc[j, "role"] = "龙头"
            out.loc[j, "reason"] = (
                f"20日{out.loc[j, 'ret20']:+.1f}% / 5日{out.loc[j, 'ret5']:+.1f}%、"
                f"窗口第{int(out.loc[j, 'early_days']) + 1}日启动"
            )
    return out


def decide_buy(tier: str, row: dict, rally_stage: str, can_buy: bool) -> tuple[str, str]:
    """给出可操作结论：可买 / 轻仓可买 / 观望 / 不追 / 回避。"""
    role = str(row.get("role") or "")
    ret5 = float(row.get("ret5") or 0)
    ret20 = float(row.get("ret20") or 0)
    above = bool(row.get("above_ma20"))

    if role == "尾部小票":
        if tier == "超短反抽" or ret5 >= 8:
            return "回避", "尾部跟风票，盈亏比差，不做"
        return "回避", "尾部小票默认回避"

    if tier == "超短反抽":
        if ret5 >= 12:
            return "不追", "反抽已陡，现价不追"
        if role == "中军" and ret5 < 10:
            return "观望", "中军可等缩量回踩，今日不追涨买"
        if role == "龙头" and ret5 < 12:
            return "观望", "反抽龙头只等回踩，不建议现价首买"
        return "不追", "超短反抽段，现价追入风险高"

    # —— 趋势热 ——
    if can_buy or rally_stage == "可买入":
        if role == "中军":
            return "可买", "趋势中军 + 长线Entry触发，可分批"
        return "轻仓可买", "趋势龙头 + 长线Entry，仓位宜轻"

    if rally_stage == "观察埋伏":
        return "观望", "长线Setup中，等站上MA20再买"

    if rally_stage == "已偏强" or ret20 >= 32 or ret5 >= 15:
        if role == "中军" and ret5 < 12 and above:
            return "观望", "已偏强；中军可等回踩MA10/MA20"
        return "不追", "涨幅已大/已偏强，不适合当首买"

    if role == "中军":
        if above and ret5 <= 8 and ret20 <= 25:
            return "轻仓可买", "趋势中军未过度延伸，可轻仓跟随"
        if ret5 <= 12:
            return "观望", "中军方向对，优先等缩量回踩"
        return "不追", "中军近5日涨太急"

    # 龙头
    if above and ret5 <= 10 and ret20 <= 28:
        return "轻仓可买", "龙头仍在趋势、延伸尚可，轻仓跟"
    if ret5 <= 14:
        return "观望", "龙头已拉升，等回踩再跟"
    return "不追", "龙头短期过热，不追高"


def enrich_roles(
    roles: dict[str, list[dict]],
    tier: str,
    industry: str,
    name_map: dict,
    ind_map: dict,
    asof: str | None,
) -> dict[str, list[dict]]:
    import rally_buy_screener as rbs

    out: dict[str, list[dict]] = {}
    for role, rows in roles.items():
        enriched = []
        for r in rows:
            v = rbs.evaluate(r["code"], name_map=name_map, ind_map=ind_map, asof=asof)
            action, why = decide_buy(tier, r, v.stage, v.can_buy)
            enriched.append(
                {
                    **r,
                    "industry": industry,
                    "tier": tier,
                    "rally_stage": v.stage,
                    "rally_can_buy": bool(v.can_buy),
                    "action": action,
                    "action_why": why,
                }
            )
        out[role] = enriched
    return out


def pick_roles(tagged: pd.DataFrame) -> dict[str, list[dict]]:
    def take(role: str, k: int, col: str) -> list[dict]:
        sub = tagged[tagged["role"] == role].sort_values(col, ascending=False)
        if role == "中军":
            sub = tagged[tagged["role"] == "中军"].sort_values("mv", ascending=False)
        cols = [
            "code",
            "name",
            "role",
            "mv",
            "ret5",
            "ret10",
            "ret20",
            "amt5亿",
            "amt_ratio",
            "early_days",
            "above_ma20",
            "reason",
        ]
        return sub[cols].head(k).to_dict("records")

    return {
        "龙头": take("龙头", 3, "ret20"),
        "中军": take("中军", 4, "mv"),
        "尾部小票": take("尾部小票", 5, "ret5"),
    }


def flatten_actions(packs: list[tuple[dict, dict]], tier: str) -> list[dict]:
    rows: list[dict] = []
    for srow, roles in packs:
        for role_rows in roles.values():
            for r in role_rows:
                rows.append({**r, "industry": srow["industry"], "tier": tier})
    return rows


def esc(x: object) -> str:
    return html.escape("" if x is None else str(x))


def xueqiu_url(code: str) -> str:
    c = str(code).zfill(6)
    prefix = "SH" if c.startswith(("5", "6", "9")) else "SZ"
    return f"https://xueqiu.com/S/{prefix}{c}"


def name_link(code: str, name: str) -> str:
    return (
        f'<a class="xq" href="{esc(xueqiu_url(code))}" target="_blank" rel="noopener noreferrer" '
        f'title="雪球打开 {esc(code)} {esc(name)}">{esc(name)}</a>'
    )


def fmt_pct(v: float | None) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v:+.1f}%"


def fmt_n(v: float | None, nd: int = 0) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v:.{nd}f}"


def role_badge(role: str) -> str:
    cls = {"龙头": "b-lead", "中军": "b-army", "尾部小票": "b-tail", "观察": "b-watch"}.get(
        role, "b-watch"
    )
    return f'<span class="badge {cls}">{esc(role)}</span>'


def action_badge(action: str) -> str:
    cls = {
        "可买": "a-buy",
        "轻仓可买": "a-light",
        "观望": "a-wait",
        "不追": "a-skip",
        "回避": "a-avoid",
    }.get(action, "a-wait")
    return f'<span class="badge {cls}">{esc(action)}</span>'


def stock_rows_html(rows: list[dict]) -> str:
    if not rows:
        return '<p class="muted">本板块该角色暂无清晰候选</p>'
    trs = []
    for r in rows:
        trs.append(
            "<tr>"
            f"<td>{esc(r['code'])}</td>"
            f"<td>{name_link(r['code'], r['name'])}</td>"
            f"<td class='num'>{fmt_n(r['mv'], 0)}</td>"
            f"<td class='num'>{fmt_pct(r['ret20'])}</td>"
            f"<td class='num'>{fmt_pct(r['ret5'])}</td>"
            f"<td>{esc(r.get('rally_stage') or '—')}</td>"
            f"<td>{action_badge(str(r.get('action') or '观望'))}</td>"
            f"<td class='note'>{esc(r.get('action_why') or '')}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr>"
        "<th>代码</th><th>名称</th><th>流通市值亿</th><th>20日</th><th>5日</th>"
        "<th>长线阶段</th><th>买入结论</th><th>买入说明</th>"
        "</tr></thead><tbody>"
        + "".join(trs)
        + "</tbody></table>"
    )


def action_summary_html(rows: list[dict]) -> str:
    order = {"可买": 0, "轻仓可买": 1, "观望": 2, "不追": 3, "回避": 4}
    buyish = [r for r in rows if r.get("action") in ("可买", "轻仓可买")]
    wait = [r for r in rows if r.get("action") == "观望"]
    buyish.sort(key=lambda r: (order.get(str(r.get("action")), 9), -float(r.get("ret20") or 0)))
    if not buyish and not wait:
        return '<p class="muted">本轮没有「可买/轻仓可买」标的；多数已延伸或属反抽/尾部。</p>'

    def block(title: str, items: list[dict]) -> str:
        if not items:
            return ""
        trs = []
        for r in items:
            trs.append(
                "<tr>"
                f"<td>{esc(r['code'])}</td>"
                f"<td>{name_link(r['code'], r['name'])}</td>"
                f"<td>{esc(r.get('tier'))}</td>"
                f"<td>{esc(r.get('industry'))}</td>"
                f"<td>{role_badge(str(r.get('role')))}</td>"
                f"<td class='num'>{fmt_pct(r.get('ret20'))}</td>"
                f"<td class='num'>{fmt_pct(r.get('ret5'))}</td>"
                f"<td>{esc(r.get('rally_stage') or '—')}</td>"
                f"<td>{action_badge(str(r.get('action')))}</td>"
                f"<td class='note'>{esc(r.get('action_why') or '')}</td>"
                "</tr>"
            )
        return (
            f"<h3>{esc(title)}（{len(items)}）</h3>"
            "<table><thead><tr>"
            "<th>代码</th><th>名称</th><th>套别</th><th>板块</th><th>角色</th>"
            "<th>20日</th><th>5日</th><th>长线阶段</th><th>买入结论</th><th>买入说明</th>"
            "</tr></thead><tbody>"
            + "".join(trs)
            + "</tbody></table>"
        )

    return block("优先看：可买 / 轻仓可买", buyish) + block("可放观察池：观望", wait[:25])


def sector_block_html(tier: str, srow: dict, roles: dict[str, list[dict]]) -> str:
    name = srow["industry"]
    sid = f"{tier}-{name}"
    return f"""
<section class="sector" id="{esc(sid)}">
  <div class="sector-head">
    <h3>{esc(name)}</h3>
    <div class="meta">
      <span>成分 {srow['n']}</span>
      <span>热度 {srow['heat']}</span>
      <span>中位5日 {fmt_pct(srow['med5'])}</span>
      <span>中位20日 {fmt_pct(srow['med20'])}</span>
      <span>近5日额 {fmt_n(srow['amt5'], 0)}亿</span>
      <span>20日广度 {fmt_n(srow['breadth20'], 0)}%</span>
    </div>
  </div>
  <div class="role-block">
    <h4>{role_badge('龙头')} 候选龙头</h4>
    {stock_rows_html(roles['龙头'])}
  </div>
  <div class="role-block">
    <h4>{role_badge('中军')} 候选中军</h4>
    {stock_rows_html(roles['中军'])}
  </div>
  <div class="role-block">
    <h4>{role_badge('尾部小票')} 尾部小票</h4>
    {stock_rows_html(roles['尾部小票'])}
  </div>
</section>
"""


def render_html(
    asof: str,
    trend: list[tuple[dict, dict]],
    burst: list[tuple[dict, dict]],
    all_sectors: pd.DataFrame,
    action_rows: list[dict],
) -> str:
    trend_nav = "".join(
        f'<a href="#趋势热-{html.escape(s["industry"])}">{html.escape(s["industry"])}</a>'
        for s, _ in trend
    )
    burst_nav = "".join(
        f'<a href="#超短反抽-{html.escape(s["industry"])}">{html.escape(s["industry"])}</a>'
        for s, _ in burst
    )
    trend_toc_items = "".join(
        f'<li><a href="#趋势热-{html.escape(s["industry"])}">{html.escape(s["industry"])}</a></li>'
        for s, _ in trend
    )
    burst_toc_items = "".join(
        f'<li><a href="#超短反抽-{html.escape(s["industry"])}">{html.escape(s["industry"])}</a></li>'
        for s, _ in burst
    )
    page_toc = f"""
<nav class="page-toc" id="toc" aria-label="目录">
  <div class="toc-title">目录</div>
  <ol>
    <li><a href="#buy-rules">买入依据</a></li>
    <li><a href="#buy-summary">买入结论总表</a></li>
    <li>
      <a href="#tier-trend">趋势热</a>
      <ul>{trend_toc_items}</ul>
    </li>
    <li>
      <a href="#tier-burst">超短反抽</a>
      <ul>{burst_toc_items}</ul>
    </li>
    <li><a href="#tier-rank">全市场行业热度 Top40</a></li>
  </ol>
</nav>
"""
    trend_body = "".join(sector_block_html("趋势热", s, r) for s, r in trend)
    burst_body = "".join(sector_block_html("超短反抽", s, r) for s, r in burst)
    summary = action_summary_html(action_rows)

    counts: dict[str, int] = {}
    for r in action_rows:
        a = str(r.get("action") or "")
        counts[a] = counts.get(a, 0) + 1
    count_line = " · ".join(f"{k} {v}" for k, v in sorted(counts.items(), key=lambda x: -x[1]))

    rank_rows = []
    for _, r in all_sectors.head(40).iterrows():
        rank_rows.append(
            "<tr>"
            f"<td>{esc(r['industry'])}</td>"
            f"<td class='num'>{r['heat']}</td>"
            f"<td class='num'>{fmt_pct(r['med5'])}</td>"
            f"<td class='num'>{fmt_pct(r['med20'])}</td>"
            f"<td class='num'>{fmt_n(r['amt5'], 0)}</td>"
            f"<td class='num'>{int(r['n'])}</td>"
            "</tr>"
        )

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>热门行业角色 · {esc(asof)}</title>
<style>
:root {{
  --bg: #e9eef3;
  --ink: #12202c;
  --muted: #5a6b78;
  --line: #c9d4de;
  --panel: #f7fafc;
  --lead: #0b6e4f;
  --lead-bg: #d9f2e8;
  --army: #1a5f8a;
  --army-bg: #dceef8;
  --tail: #9a3412;
  --tail-bg: #ffedd5;
  --accent: #0e7490;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  font-family: "Avenir Next", "PingFang SC", "Hiragino Sans GB", "Noto Sans SC", sans-serif;
  color: var(--ink);
  background:
    linear-gradient(165deg, #d7e4ef 0%, transparent 42%),
    linear-gradient(345deg, #cfe0d8 0%, transparent 36%),
    var(--bg);
  line-height: 1.45;
}}
.wrap {{
  display: grid;
  grid-template-columns: 220px minmax(0, 1fr);
  gap: 0 28px;
  max-width: 1280px;
  margin: 0 auto;
  padding: 20px 20px 64px;
  align-items: start;
}}
.main {{ min-width: 0; }}
header.hero {{
  padding: 8px 0 22px;
  border-bottom: 1px solid var(--line);
  margin-bottom: 22px;
}}
.brand {{
  font-size: 0.78rem;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: var(--accent);
  font-weight: 700;
  margin-bottom: 8px;
}}
h1 {{
  font-size: clamp(1.75rem, 3vw, 2.25rem);
  font-weight: 750;
  margin: 0 0 8px;
  letter-spacing: -0.01em;
}}
.sub {{ color: var(--muted); font-size: 0.95rem; max-width: 52em; }}
.rules {{
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 12px;
  margin: 18px 0 8px;
}}
.rule {{
  background: var(--panel);
  border: 1px solid var(--line);
  padding: 12px 14px;
}}
.rule strong {{ display: block; margin-bottom: 4px; }}
.rule.lead strong {{ color: var(--lead); }}
.rule.army strong {{ color: var(--army); }}
.rule.tail strong {{ color: var(--tail); }}
.note-box {{
  margin-top: 14px;
  padding: 10px 12px;
  border-left: 3px solid var(--accent);
  background: #e4f1f5;
  color: #234;
  font-size: 0.9rem;
}}
nav.toc {{
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin: 10px 0 18px;
}}
nav.toc a {{
  text-decoration: none;
  color: var(--army);
  background: var(--army-bg);
  border: 1px solid #b7d3e6;
  padding: 4px 9px;
  font-size: 0.82rem;
}}
nav.toc a:hover {{ background: #cfe4f2; }}
nav.page-toc {{
  position: sticky;
  top: 12px;
  max-height: calc(100vh - 24px);
  overflow: auto;
  background: var(--panel);
  border: 1px solid var(--line);
  padding: 12px 12px 14px;
  font-size: 0.82rem;
}}
nav.page-toc .toc-title {{
  font-weight: 700;
  font-size: 0.88rem;
  margin-bottom: 8px;
  letter-spacing: 0.04em;
}}
nav.page-toc ol {{
  margin: 0;
  padding-left: 1.15em;
}}
nav.page-toc li {{ margin: 3px 0; }}
nav.page-toc ul {{
  margin: 3px 0 5px;
  padding-left: 1em;
  list-style: disc;
}}
nav.page-toc a {{
  color: var(--army);
  text-decoration: none;
  line-height: 1.35;
}}
nav.page-toc a:hover {{ text-decoration: underline; }}
h2 {{
  font-size: 1.35rem;
  margin: 28px 0 8px;
  letter-spacing: -0.01em;
  scroll-margin-top: 16px;
}}
h3 {{ font-size: 1.02rem; margin: 16px 0 8px; }}
.sector {{ scroll-margin-top: 16px; }}
.tier-desc {{ color: var(--muted); margin: 0 0 14px; font-size: 0.92rem; }}
.sector {{
  background: var(--panel);
  border: 1px solid var(--line);
  padding: 16px 16px 8px;
  margin: 0 0 16px;
}}
.sector-head h3 {{
  margin: 0 0 8px;
  font-size: 1.12rem;
}}
.meta {{
  display: flex;
  flex-wrap: wrap;
  gap: 8px 14px;
  color: var(--muted);
  font-size: 0.82rem;
  margin-bottom: 10px;
}}
.role-block {{ margin: 12px 0; }}
.role-block h4 {{
  margin: 0 0 8px;
  font-size: 0.95rem;
  display: flex;
  align-items: center;
  gap: 8px;
}}
.badge {{
  display: inline-block;
  font-size: 0.72rem;
  font-weight: 700;
  padding: 2px 7px;
  border: 1px solid transparent;
  white-space: nowrap;
}}
.b-lead {{ color: var(--lead); background: var(--lead-bg); border-color: #a7d9c5; }}
.b-army {{ color: var(--army); background: var(--army-bg); border-color: #a9cbe0; }}
.b-tail {{ color: var(--tail); background: var(--tail-bg); border-color: #fdba74; }}
.b-watch {{ color: #555; background: #eee; border-color: #ddd; }}
.a-buy {{ color: #fff; background: #0b6e4f; border-color: #0b6e4f; }}
.a-light {{ color: #0b6e4f; background: #d9f2e8; border-color: #7bc4a5; }}
.a-wait {{ color: #92400e; background: #fef3c7; border-color: #f6d88a; }}
.a-skip {{ color: #7f1d1d; background: #fee2e2; border-color: #f5b5b5; }}
.a-avoid {{ color: #fff; background: #7f1d1d; border-color: #7f1d1d; }}
table {{
  width: 100%;
  border-collapse: collapse;
  font-size: 0.86rem;
}}
th, td {{
  border-top: 1px solid var(--line);
  padding: 7px 8px;
  vertical-align: middle;
  text-align: center;
}}
th {{
  color: var(--muted);
  font-weight: 600;
  font-size: 0.78rem;
}}
td.num {{ font-variant-numeric: tabular-nums; white-space: nowrap; }}
td.note {{ color: #334; font-size: 0.8rem; text-align: center; }}
a.xq {{
  color: var(--army);
  text-decoration: none;
  font-weight: 650;
}}
a.xq:hover {{ text-decoration: underline; }}
.muted {{ color: var(--muted); font-size: 0.88rem; }}
.rank-wrap {{ overflow-x: auto; }}
.summary {{
  background: var(--panel);
  border: 1px solid var(--line);
  padding: 14px 16px 8px;
  margin: 0 0 8px;
}}
footer {{
  margin-top: 28px;
  padding-top: 14px;
  border-top: 1px solid var(--line);
  color: var(--muted);
  font-size: 0.82rem;
}}
@media (max-width: 960px) {{
  .wrap {{
    grid-template-columns: 1fr;
    padding-top: 12px;
  }}
  nav.page-toc {{
    position: relative;
    top: 0;
    max-height: none;
    margin-bottom: 16px;
  }}
  .rules {{ grid-template-columns: 1fr; }}
}}
@media (max-width: 800px) {{
  .rules {{ grid-template-columns: 1fr; }}
}}
</style>
</head>
<body>
<div class="wrap">
  {page_toc}
  <div class="main">
  <header class="hero">
    <div class="brand">SECTOR ROLES</div>
    <h1>热门行业角色划分</h1>
    <p class="sub">截面 {esc(asof)} · 东财行业三级 · 主板非ST · 流通市值排序。
    分两套：趋势热与超短反抽。每只票给出买入结论：可买 / 轻仓可买 / 观望 / 不追 / 回避
    （角色 × 涨幅延伸 × 是否站上MA20 × 长线Setup/Entry）。</p>
    <div class="rules">
      <div class="rule army"><strong>中军</strong>市值前排 + 近5日成交持续巨大、额能比抬升</div>
      <div class="rule lead"><strong>龙头</strong>谁先启动、20日/进攻涨幅最高，市值通常不是最大</div>
      <div class="rule tail"><strong>尾部小票</strong>市值小、启动晚、别人涨完才跟</div>
    </div>
    <div class="note-box">结论统计：{esc(count_line)}。「可买」只表示相对现价首买是否合适，不是保证赚钱。北向接口本轮不可用。</div>
  </header>

  <h2 id="buy-rules">买入依据（规则引擎）</h2>
  <div class="summary">
    <p class="tier-desc" style="margin-top:0">输入：板块套别（趋势热/超短反抽）× 角色（龙头/中军/尾部）× 5日/20日涨幅 × 是否站上MA20 × 长线筛选阶段（可买入/观察埋伏/已偏强）。</p>
    <table>
      <thead><tr><th>结论</th><th>主要触发条件</th></tr></thead>
      <tbody>
        <tr><td>{action_badge('可买')}</td><td class="note">仅趋势热：角色为中军，且长线 Entry 触发（rally 阶段=可买入）</td></tr>
        <tr><td>{action_badge('轻仓可买')}</td><td class="note">趋势热：①龙头且长线 Entry；或②中军站上MA20且5日≤8%、20日≤25%；或③龙头站上MA20且5日≤10%、20日≤28%</td></tr>
        <tr><td>{action_badge('观望')}</td><td class="note">长线仍在 Setup；或已偏强但中军未过热可等回踩；或超短反抽里龙头/中军未涨太陡，只等回踩不追</td></tr>
        <tr><td>{action_badge('不追')}</td><td class="note">20日≥32% 或 5日≥15% 或长线已偏强；中军5日&gt;12%；龙头5日&gt;14%；超短反抽5日≥12% 或默认不追涨</td></tr>
        <tr><td>{action_badge('回避')}</td><td class="note">尾部小票一律回避（跟风盈亏比差）</td></tr>
      </tbody>
    </table>
  </div>

  <h2 id="buy-summary">〇、买入结论总表</h2>
  <p class="tier-desc">先看这里。超短反抽默认偏「不追/观望」；尾部默认「回避」。</p>
  <div class="summary">
    {summary}
  </div>

  <h2 id="tier-trend">一、趋势热</h2>
  <p class="tier-desc">筛选：综合热度 ≥ 8，且中位5日 &gt; 1.5%。</p>
  <nav class="toc">{trend_nav}</nav>
  {trend_body}

  <h2 id="tier-burst">二、超短反抽</h2>
  <p class="tier-desc">筛选：中位5日 &gt; 8%，中位20日 &lt; 0，近5日板块成交合计 ≥ 80亿。现价追入多数不建议。</p>
  <nav class="toc">{burst_nav}</nav>
  {burst_body}

  <h2 id="tier-rank">三、全市场行业热度 Top40</h2>
  <div class="rank-wrap">
    <table>
      <thead><tr>
        <th>板块</th><th>热度</th><th>中位5日</th><th>中位20日</th><th>近5日额亿</th><th>成分</th>
      </tr></thead>
      <tbody>
        {''.join(rank_rows)}
      </tbody>
    </table>
  </div>

  <footer>
    由 gen_hot_sector_html.py 生成 · 输出目录 hot_sectors/ · 买入结论为规则引擎，实盘请再人工确认。
    · <a href="#toc">回目录</a>
  </footer>
  </div>
</div>
</body>
</html>
"""


def _clean_json(o):
    if isinstance(o, dict):
        return {k: _clean_json(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_clean_json(v) for v in o]
    if isinstance(o, (np.floating, float)):
        x = float(o)
        return None if np.isnan(x) else round(x, 3)
    if isinstance(o, (np.integer, int)):
        return int(o)
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    return o


def generate(
    *,
    asof: str | None = None,
    out_dir: Path | None = None,
    mode: str = "final",
    log_fn=print,
) -> dict:
    """
    生成热门板块角色 HTML/JSON/CSV。
    默认写入 hot_sectors/YYYY-MM-DD[/preview]/。
    返回 meta 摘要。
    """
    import rally_buy_screener as rbs

    log_fn("load hot-sector frame…")
    stock, day = load_stock_frame(asof)
    log_fn(f"asof={day} stocks={len(stock)}")
    sectors = sector_stats(stock)
    name_map, ind_map = rbs.load_maps()

    trend_sec = sectors[(sectors["heat"] >= 8) & (sectors["med5"] > 1.5)].copy()
    burst_sec = sectors[
        (sectors["med5"] > 8) & (sectors["med20"] < 0) & (sectors["amt5"] >= 80)
    ].sort_values("med5", ascending=False)
    trend_names = set(trend_sec["industry"])
    burst_sec = burst_sec[~burst_sec["industry"].isin(trend_names)].head(10)

    def build(pack: pd.DataFrame, tier: str) -> list[tuple[dict, dict]]:
        out = []
        for _, srow in pack.iterrows():
            tagged = classify(stock[stock["industry"] == srow["industry"]])
            roles = pick_roles(tagged)
            roles = enrich_roles(roles, tier, srow["industry"], name_map, ind_map, day)
            out.append((srow.to_dict(), roles))
            acts: dict[str, int] = {}
            for rr in roles.values():
                for r in rr:
                    acts[r["action"]] = acts.get(r["action"], 0) + 1
            log_fn(f"  {srow['industry']}: {acts}")
        return out

    log_fn(f"趋势热 {len(trend_sec)}")
    trend = build(trend_sec, "趋势热")
    log_fn(f"超短反抽 {len(burst_sec)}")
    burst = build(burst_sec, "超短反抽")

    action_rows = flatten_actions(trend, "趋势热") + flatten_actions(burst, "超短反抽")
    buyish = [r for r in action_rows if r.get("action") in ("可买", "轻仓可买")]
    counts: dict[str, int] = {}
    for r in action_rows:
        a = str(r.get("action") or "")
        counts[a] = counts.get(a, 0) + 1

    day_dir = out_dir if out_dir is not None else hot_day_dir(day, mode)
    day_dir.mkdir(parents=True, exist_ok=True)

    html_path = day_dir / "index.html"
    html_path.write_text(
        render_html(day, trend, burst, sectors, action_rows), encoding="utf-8"
    )
    sectors.to_csv(day_dir / "sectors.csv", index=False)

    payload = {
        "asof": day,
        "mode": mode,
        "actions_buyish": buyish,
        "action_counts": counts,
        "trend": [{"sector": s, "roles": r} for s, r in trend],
        "burst": [{"sector": s, "roles": r} for s, r in burst],
    }
    (day_dir / "roles.json").write_text(
        json.dumps(_clean_json(payload), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    meta = {
        "asof": day,
        "mode": mode,
        "day_dir": str(day_dir),
        "index_html": str(html_path),
        "n_trend": len(trend),
        "n_burst": len(burst),
        "n_buyish": len(buyish),
        "action_counts": counts,
    }
    (day_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log_fn(f"热门板块 HTML → {html_path}")
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description="热门行业角色 HTML → hot_sectors/")
    ap.add_argument("--asof", default="")
    ap.add_argument("--out", default="", help="输出目录（默认 hot_sectors/YYYY-MM-DD）")
    ap.add_argument("--preview", action="store_true")
    args = ap.parse_args()
    asof = args.asof.strip() or None
    mode = "preview" if args.preview else "final"
    out = Path(args.out) if args.out.strip() else None
    generate(asof=asof, out_dir=out, mode=mode)


if __name__ == "__main__":
    main()
