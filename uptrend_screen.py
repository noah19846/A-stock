"""
从 daily.db 里筛上涨趋势，写出可筛选的 HTML，并按行业大类发邮件。

只使用库里有日线的股票。去掉名称含 ST 的、创业板（300/301）、科创板（688/689）。
名称链到雪球。邮件先分成延续和回调，再按申万三级行业收成的大类，一类里按 60 日涨幅。

上涨：60 日均线高于 20 个交易日前；20 日均线仍在 60 日均线之上；
收盘在 60 日均线之上；近 60 个交易日涨幅为正。
收盘仍在 20 日线上为延续，落到 20 日线下为回调。

用法：
  .venv/bin/python uptrend_screen.py
  .venv/bin/python uptrend_screen.py --html uptrend.html
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent


def xueqiu_url(code: str) -> str:
    code = str(code).zfill(6)
    if code.startswith(("5", "6", "9")):
        prefix = "SH"
    elif code.startswith(("4", "8")) or code.startswith("92"):
        prefix = "BJ"
    else:
        prefix = "SZ"
    return f"https://xueqiu.com/S/{prefix}{code}"


# 先匹配更具体的行业名。申万三级大约三百个，邮件里按这些大类分段。
_GROUP_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("银行", ("银行", "商行")),
    ("非银金融", ("证券", "保险", "信托", "期货", "金融", "租赁", "资产")),
    ("交通运输", ("高速", "港口", "航运", "航空运输", "铁路运输", "机场", "快递", "物流", "公交", "货运", "供应链", "仓储")),
    ("汽车", ("汽车", "乘用车", "载客", "载货", "摩托车", "轮胎", "车身", "底盘")),
    ("家电", ("家电", "冰洗", "空调", "厨房电器", "厨房小家电", "清洁小家电", "照明", "彩电", "黑色家电", "卫浴电器")),
    ("建筑地产", ("地产", "住宅", "物业", "房屋建设", "园林", "基建", "国际工程", "装修装饰", "钢结构", "工程咨询", "专业工程")),
    ("建材", ("水泥", "板材", "管材", "瓷砖", "防水", "玻璃", "玻纤", "耐火", "建材", "卫浴制品")),
    ("煤炭油气", ("煤", "焦炭", "油气", "油田", "炼油", "石化")),
    ("公用事业", ("发电", "热力", "电能", "燃气", "水务", "环境治理", "固废", "大气治理", "环保")),
    ("电力设备", ("光伏", "风电", "锂电", "电池", "逆变器", "电网", "输变电", "配电", "电机", "电源", "线缆", "硅料", "火电", "电力设备")),
    ("化工", ("化工", "化学", "树脂", "塑料", "橡胶", "氟", "氯碱", "纯碱", "氨纶", "涤纶", "锦纶", "粘胶", "聚氨酯", "钛白", "炭黑", "膜材料", "无机盐", "有机硅", "胶黏", "民爆", "氮肥", "磷肥", "钾肥", "复合肥", "涂料")),
    ("有色钢铁", ("钢", "铁", "铜", "铝", "锂", "镍", "钴", "钼", "钨", "铅锌", "黄金", "白银", "稀土", "金属", "冶钢", "长材")),
    ("纺织服饰", ("家居", "家纺", "服装", "纺织", "鞋", "棉纺", "印染", "饰品", "钟表珠宝", "辅料")),
    ("食品饮料", ("食品", "酒", "乳品", "饮料", "调味", "零食", "烘焙", "预加工", "熟食", "粮", "果蔬", "肉制品")),
    ("农林牧渔", ("养殖", "饲料", "种植", "农产品", "食用菌", "林业", "种子", "宠物", "水产", "海洋", "农业", "动物保健", "农药", "农用")),
    ("医药", ("药", "医", "疫苗", "血液", "诊断", "生物制品", "保健品")),
    ("商贸服务", ("百货", "超市", "零售", "酒店", "旅游", "景区", "餐饮", "贸易", "专业连锁", "检测服务", "人力资源", "会展", "化妆品", "洗护", "娱乐用品", "体育")),
    ("传媒教育", ("出版", "影视", "游戏", "媒体", "广告", "教育", "院线", "培训", "门户", "电视广播", "文化用品", "营销")),
    ("机械军工", ("机械", "机床", "机器人", "轨交", "航天", "航空装备", "航海", "军工", "兵装", "自动化", "工控", "激光", "磨具", "仪器仪表", "重型设备", "楼宇设备", "印刷包装", "专用设备", "通用设备", "运输设备")),
    ("电子", ("芯片", "半导体", "面板", "光学", "LED", "元件", "电路板", "电子", "集成电路", "安防", "磁性", "分立器件")),
    ("通信计算机", ("通信", "软件", "IT服务", "计算机", "电商", "电信")),
    ("造纸包装", ("纸", "包装", "印刷")),
)


def industry_group(name: str) -> str:
    text = (name or "").replace("Ⅲ", "")
    for group, keys in _GROUP_RULES:
        if any(key in text for key in keys):
            return group
    return "其他"


def board_of(code: str) -> str:
    if code.startswith(("688", "689")):
        return "科创板"
    if code.startswith(("300", "301")):
        return "创业板"
    if code.startswith(("8", "4", "92")):
        return "北交所"
    if code.startswith(("600", "601", "603", "605", "000", "001", "002", "003")):
        return "主板"
    return "其他"


def load_maps(root: Path) -> tuple[dict[str, str], dict[str, str]]:
    names: dict[str, str] = {}
    industries: dict[str, str] = {}
    list_path = root / "data" / "stock_list.csv"
    ind_path = root / "data" / "stock_industry.csv"
    if list_path.exists():
        df = pd.read_csv(list_path, dtype=str)
        code_col = "股票代码" if "股票代码" in df.columns else df.columns[0]
        name_col = "股票名称" if "股票名称" in df.columns else df.columns[1]
        df[code_col] = df[code_col].astype(str).str.zfill(6)
        for code, name in zip(df[code_col], df[name_col].fillna("").astype(str)):
            name = name.strip()
            if name and name.lower() != "nan":
                names[code] = name
    if ind_path.exists():
        df = pd.read_csv(ind_path, dtype=str)
        code_col = "股票代码" if "股票代码" in df.columns else df.columns[0]
        df[code_col] = df[code_col].astype(str).str.zfill(6)
        if "行业板块" in df.columns:
            industries = dict(zip(df[code_col], df["行业板块"].fillna("").astype(str)))
        if "股票名称" in df.columns:
            for code, name in zip(df[code_col], df["股票名称"].fillna("").astype(str)):
                name = name.strip()
                if name and name.lower() != "nan" and code not in names:
                    names[code] = name
    return names, industries


def screen(root: Path) -> tuple[dict, list[dict]]:
    sys.path.insert(0, str(root))
    import daily_db

    db_path = root / "data" / "db" / "daily.db"
    cutoff = daily_db.recent_cutoff_date(limit=160, db_path=db_path)
    if not cutoff:
        raise SystemExit(f"没有读到日线：{db_path}")
    raw = daily_db.load_recent_bars_since(cutoff, db_path=db_path)
    if raw.empty:
        raise SystemExit(f"{cutoff} 之后没有日线")
    raw["code"] = raw["code"].astype(str).str.zfill(6)
    raw["trade_date"] = raw["trade_date"].astype(str).str.slice(0, 10)
    asof = str(raw["trade_date"].max())
    names, industries = load_maps(root)

    in_db = int(raw["code"].nunique())
    dropped_board = 0
    dropped_st = 0
    dropped_short = 0
    rows: list[dict] = []

    for code, g in raw.groupby("code", sort=False):
        board = board_of(code)
        if board in ("科创板", "创业板"):
            dropped_board += 1
            continue
        name = names.get(code, "")
        if "ST" in name.upper():
            dropped_st += 1
            continue
        g = g.sort_values("trade_date")
        if str(g["trade_date"].iloc[-1]) != asof or len(g) < 120:
            dropped_short += 1
            continue
        factor = pd.to_numeric(g["hfq_factor"], errors="coerce")
        last_f = float(factor.iloc[-1])
        if not (last_f > 0) or factor.isna().any():
            dropped_short += 1
            continue
        scale = factor.to_numpy() / last_f
        px = pd.to_numeric(g["close"], errors="coerce").to_numpy() * scale
        hi = pd.to_numeric(g["high"], errors="coerce").to_numpy() * scale
        amt = pd.to_numeric(g["amount"], errors="coerce").to_numpy()
        if not (pd.notna(px).all() and pd.notna(hi).all()) or min(px) <= 0:
            dropped_short += 1
            continue
        close = float(px[-1])
        ma20 = float(px[-20:].mean())
        ma60 = float(px[-60:].mean())
        ma60_prev = float(px[-80:-20].mean())
        ret60 = close / float(px[-61]) - 1.0
        ret20 = close / float(px[-21]) - 1.0
        if not (ma60 > ma60_prev and ma20 > ma60 and close > ma60 and ret60 > 0):
            continue
        amt20 = float(pd.Series(amt[-20:]).mean())
        industry = industries.get(code, "") or "未分类"
        rows.append(
            {
                "code": code,
                "name": name or code,
                "board": board,
                "group": industry_group(industry),
                "state": "回调" if close < ma20 else "延续",
                "close": round(close, 2),
                "ret60": round(ret60 * 100, 1),
                "ret20": round(ret20 * 100, 1),
                "dd60": round((close / float(hi[-60:].max()) - 1.0) * 100, 1),
                "above60": round((close / ma60 - 1.0) * 100, 1),
                "amt": round(amt20 / 1e8, 2) if pd.notna(amt20) else 0.0,
                "industry": industry,
                "xq": xueqiu_url(code),
            }
        )

    rows.sort(key=lambda r: (r["state"] != "延续", -r["ret60"]))
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["group"]] = counts.get(row["group"], 0) + 1
    groups = sorted(counts, key=lambda name: (name == "其他", -counts[name], name))
    meta = {
        "asof": asof,
        "in_db": in_db,
        "dropped_board": dropped_board,
        "dropped_st": dropped_st,
        "n": len(rows),
        "cont": sum(1 for r in rows if r["state"] == "延续"),
        "pull": sum(1 for r in rows if r["state"] == "回调"),
        "boards": sorted({r["board"] for r in rows}),
        "groups": groups,
    }
    _ = dropped_short
    return meta, rows


def _mail_state(state: str, rows: list[dict]) -> str:
    import html as html_mod

    picked = [row for row in rows if row["state"] == state]
    buckets: dict[str, list[dict]] = {}
    for row in picked:
        buckets.setdefault(row["group"], []).append(row)
    names = sorted(buckets, key=lambda name: (name == "其他", -len(buckets[name]), name))
    tone = "#1f4d3a" if state == "延续" else "#8a5a12"
    note = "收盘仍在 20 日线上" if state == "延续" else "收盘已落到 20 日线下，60 日趋势还在"
    parts = [
        f'<h2 style="margin:22px 0 2px;font-size:20px;color:{tone}">{state}'
        f'<span style="font-weight:500;font-size:14px"> · {len(picked)} 只</span></h2>',
        f'<p style="margin:0 0 10px;color:#5c6570;font-size:12px">{note}</p>',
    ]
    if not names:
        parts.append('<p style="color:#5c6570">暂无</p>')
        return "".join(parts)
    for name in names:
        items = buckets[name]
        lines: list[str] = []
        for row in items:
            industry = html_mod.escape(row["industry"].replace("Ⅲ", ""))
            lines.append(
                '<div style="padding:7px 0;border-top:1px solid #ece7df">'
                f'<a href="{html_mod.escape(row["xq"])}" style="color:#1a1d21;'
                'font-weight:600;text-decoration:none">'
                f'{html_mod.escape(row["code"])} {html_mod.escape(row["name"])}</a>'
                '<div style="color:#5c6570;font-size:12px;margin-top:2px">'
                f"{industry} · 60日 {row['ret60']:+.1f}% · 20日 {row['ret20']:+.1f}%"
                "</div></div>"
            )
        parts.append(
            '<section style="background:#fff;border-radius:12px;padding:12px 14px;margin:0 0 12px">'
            f'<h3 style="margin:0 0 2px;font-size:16px;color:{tone}">{html_mod.escape(name)}'
            f'<span style="color:#5c6570;font-weight:500;font-size:13px"> · {len(items)} 只</span></h3>'
            f'{"".join(lines)}'
            "</section>"
        )
    return "".join(parts)


def mail_html(meta: dict, rows: list[dict]) -> str:
    """单独一封上涨趋势邮件。先分延续和回调，再按行业大类分段。"""
    import html as html_mod

    asof = html_mod.escape(str(meta["asof"]))
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>上涨趋势 {asof}</title>
</head>
<body style="margin:0;padding:12px;background:#f0f2f5;color:#1a1d21;font-family:-apple-system,BlinkMacSystemFont,'PingFang SC','Hiragino Sans GB','Microsoft YaHei',sans-serif;font-size:15px;line-height:1.45">
  <h1 style="margin:0 0 6px;font-size:22px">上涨趋势 · {asof}</h1>
  <p style="margin:0 0 4px;color:#5c6570;font-size:13px">
    库内股票 · 共 {meta["n"]} 只。先分延续 {meta["cont"]}、回调 {meta["pull"]}，再按行业大类。
    已去掉 ST、创业板、科创板。名称打开雪球。一类里按 60 日涨幅从高到低。
  </p>
  {_mail_state("延续", rows)}
  {_mail_state("回调", rows)}
</body>
</html>
"""


def send_mail(root: Path | None = None) -> dict:
    """发上涨趋势专信。主题与信号池分开，方便在收件箱里直接看到。"""
    import pool_mail

    meta, rows = screen(root or ROOT)
    pool_mail.send_html_mail(
        subject=f"上涨趋势 {meta['asof']}",
        html_body=mail_html(meta, rows),
    )
    print(
        f"上涨趋势邮件 {meta['asof']}  入选 {meta['n']}"
        f"  延续 {meta['cont']}  回调 {meta['pull']}",
        flush=True,
    )
    return meta


def render_html(meta: dict, rows: list[dict]) -> str:
    data = json.dumps(rows, ensure_ascii=False, separators=(",", ":"))
    asof = meta["asof"]
    boards = "".join(
        f'<button type="button" data-board="{b}">{b}</button>' for b in meta["boards"]
    )
    groups = "".join(
        f'<button type="button" data-group="{g}">{g}</button>' for g in meta["groups"]
    )
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>{asof} 上涨趋势</title>
<style>
  :root {{
    --bg: #f6f5f2; --paper: #fff; --ink: #1c1b19; --muted: #6b6760;
    --line: #e4e0d8; --up: #1f4d3a; --pull: #8a5a12; --chip: #efece6;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; background: var(--bg); color: var(--ink);
    font: 14px/1.5 "Iowan Old Style", Palatino, "Songti SC", serif;
  }}
  main {{ max-width: 1280px; margin: 0 auto; padding: 28px 20px 64px; }}
  h1 {{ font-size: 28px; font-weight: 600; letter-spacing: -0.02em; margin: 0 0 8px; }}
  .lead {{ color: var(--muted); margin: 0 0 18px; max-width: 760px; }}
  .stats {{ display: flex; gap: 28px; margin: 0 0 18px; }}
  .stats b {{ display: block; font-size: 26px; font-variant-numeric: tabular-nums; }}
  .stats span {{ color: var(--muted); font-size: 12px; }}
  .note {{ background: var(--paper); border: 1px solid var(--line); padding: 12px 14px; margin-bottom: 18px; }}
  .note strong {{ display: block; margin-bottom: 4px; }}
  .controls {{ display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin-bottom: 12px; }}
  input[type="search"] {{
    font: inherit; padding: 6px 10px; border: 1px solid var(--line);
    background: var(--paper); min-width: 220px;
  }}
  button {{
    font: inherit; border: 1px solid var(--line); background: var(--paper);
    padding: 5px 10px; cursor: pointer; color: var(--ink);
  }}
  button.on {{ background: var(--ink); color: #fff; border-color: var(--ink); }}
  .meta {{ color: var(--muted); font-size: 12px; margin: 8px 0 14px; }}
  .wrap {{ overflow: auto; border: 1px solid var(--line); background: var(--paper); }}
  table {{ width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; }}
  th, td {{ padding: 7px 8px; border-bottom: 1px solid var(--line); text-align: left; white-space: nowrap; }}
  th {{ position: sticky; top: 0; background: #fbfaf7; font-size: 12px; cursor: pointer; }}
  td.num, th.num {{ text-align: right; }}
  a {{ color: var(--up); text-decoration: none; border-bottom: 1px solid rgba(31,77,58,.35); }}
  tr.pull a {{ color: var(--pull); border-bottom-color: rgba(138,90,18,.4); }}
  .tag {{ display: inline-block; padding: 0 6px; border-radius: 99px; background: var(--chip); font-size: 12px; }}
</style>
</head>
<body>
<main>
  <h1>{asof} 仍处上涨趋势的股票</h1>
  <p class="lead">只看数据库里的日线。已去掉 ST、创业板和科创板。名称打开雪球该股页面。分类是把申万三级行业收成的大类。</p>
  <div class="stats" id="stats"></div>
  <div class="note">
    <strong>上涨怎么定的</strong>
    四条同时成立：60 日均线高于 20 个交易日前的 60 日均线；20 日均线仍在 60 日均线之上；收盘价在 60 日均线之上；近 60 个交易日涨幅为正。收盘仍在 20 日线上记为延续，落到 20 日线下记为回调。价格用后复权因子换成前复权。
  </div>
  <div class="controls">
    <input id="q" type="search" placeholder="搜代码、名称或行业" />
    <span id="state">
      <button type="button" data-state="全部">全部</button>
      <button type="button" data-state="延续">延续</button>
      <button type="button" data-state="回调">回调</button>
    </span>
    <span id="board">
      <button type="button" data-board="全部">全部板块</button>
      {boards}
    </span>
    <span id="group">
      <button type="button" data-group="全部">全部行业</button>
      {groups}
    </span>
    <span id="liq">
      <button type="button" data-liq="全部">不看成交额</button>
      <button type="button" data-liq="1">20日均额≥1亿</button>
    </span>
  </div>
  <p class="meta" id="count"></p>
  <div class="wrap">
    <table>
      <thead>
        <tr>
          <th data-k="code">代码</th>
          <th data-k="name">名称</th>
          <th data-k="board">板块</th>
          <th data-k="group">分类</th>
          <th data-k="state">状态</th>
          <th class="num" data-k="close">收盘</th>
          <th class="num" data-k="ret60">60日涨幅</th>
          <th class="num" data-k="ret20">20日涨幅</th>
          <th class="num" data-k="dd60">距60日高点</th>
          <th class="num" data-k="above60">高于60日线</th>
          <th class="num" data-k="amt">20日均额(亿)</th>
          <th data-k="industry">行业</th>
        </tr>
      </thead>
      <tbody id="body"></tbody>
    </table>
  </div>
  <p class="meta">库内 {meta["in_db"]} 只；去掉创业板和科创板 {meta["dropped_board"]} 只、ST {meta["dropped_st"]} 只。入选 {meta["n"]} 只，延续 {meta["cont"]}，回调 {meta["pull"]}，分成 {len(meta["groups"])} 类。</p>
</main>
<script>
const ROWS = {data};
const state = {{ q: "", state: "全部", board: "全部", group: "全部", liq: "全部", sort: "ret60", dir: -1 }};

function pct(n) {{
  return (n > 0 ? "+" : "") + n.toFixed(1) + "%";
}}
function esc(s) {{
  return String(s).replace(/[&<>"']/g, (c) => ({{"&":"&amp;","<":"&lt;",">":"&gt;","\\"":"&quot;","'":"&#39;"}}[c]));
}}
function mark(sel, attr, cur) {{
  document.querySelectorAll(sel).forEach((b) => {{
    b.className = b.getAttribute(attr) === cur ? "on" : "";
  }});
}}
function filtered() {{
  const q = state.q.trim();
  const rows = ROWS.filter((r) => {{
    if (state.state !== "全部" && r.state !== state.state) return false;
    if (state.board !== "全部" && r.board !== state.board) return false;
    if (state.group !== "全部" && r.group !== state.group) return false;
    if (state.liq === "1" && r.amt < 1) return false;
    if (q && !(r.code + " " + r.name + " " + r.industry + " " + r.group).includes(q)) return false;
    return true;
  }});
  const k = state.sort, dir = state.dir;
  rows.sort((a, b) => {{
    const av = a[k], bv = b[k];
    if (typeof av === "number") return (av - bv) * dir;
    return String(av).localeCompare(String(bv), "zh") * dir;
  }});
  return rows;
}}
function render() {{
  mark("#state button", "data-state", state.state);
  mark("#board button", "data-board", state.board);
  mark("#group button", "data-group", state.group);
  mark("#liq button", "data-liq", state.liq);
  const rows = filtered();
  const cont = rows.filter((r) => r.state === "延续").length;
  document.getElementById("stats").innerHTML =
    `<div><b>${{rows.length}}</b><span>当前列表</span></div>` +
    `<div><b>${{cont}}</b><span>延续，收盘仍在 20 日线上</span></div>` +
    `<div><b>${{rows.length - cont}}</b><span>回调，收盘已在 20 日线下</span></div>`;
  document.getElementById("count").textContent =
    `显示 ${{rows.length}} / ${{ROWS.length}}。点表头排序。库内截至 {asof}。`;
  const body = document.getElementById("body");
  body.innerHTML = rows.map((r) => `<tr class="${{r.state === "回调" ? "pull" : "cont"}}">
    <td>${{esc(r.code)}}</td>
    <td><a href="${{esc(r.xq)}}" target="_blank" rel="noopener">${{esc(r.name)}}</a></td>
    <td>${{esc(r.board)}}</td>
    <td>${{esc(r.group)}}</td>
    <td><span class="tag">${{esc(r.state)}}</span></td>
    <td class="num">${{r.close.toFixed(2)}}</td>
    <td class="num">${{pct(r.ret60)}}</td>
    <td class="num">${{pct(r.ret20)}}</td>
    <td class="num">${{pct(r.dd60)}}</td>
    <td class="num">${{pct(r.above60)}}</td>
    <td class="num">${{r.amt.toFixed(2)}}</td>
    <td>${{esc(r.industry)}}</td>
  </tr>`).join("") || `<tr><td colspan="12">这个筛选下没有股票</td></tr>`;
}}
document.getElementById("q").addEventListener("input", (e) => {{ state.q = e.target.value; render(); }});
document.querySelectorAll("[data-state]").forEach((b) => b.addEventListener("click", () => {{ state.state = b.dataset.state; render(); }}));
document.querySelectorAll("[data-board]").forEach((b) => b.addEventListener("click", () => {{ state.board = b.dataset.board; render(); }}));
document.querySelectorAll("[data-group]").forEach((b) => b.addEventListener("click", () => {{ state.group = b.dataset.group; render(); }}));
document.querySelectorAll("[data-liq]").forEach((b) => b.addEventListener("click", () => {{ state.liq = b.dataset.liq; render(); }}));
document.querySelectorAll("th").forEach((th) => th.addEventListener("click", () => {{
  const k = th.dataset.k;
  if (state.sort === k) state.dir *= -1;
  else {{ state.sort = k; state.dir = typeof ROWS[0][k] === "number" ? -1 : 1; }}
  render();
}}));
render();
</script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="数据库内上涨趋势 HTML")
    parser.add_argument("--root", default=str(ROOT), help="仓库根目录，里面有 data/db/daily.db")
    parser.add_argument("--html", default="", help="输出 HTML 路径")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    meta, rows = screen(root)
    out = Path(args.html) if args.html else root / f"uptrend_{meta['asof']}.html"
    out.write_text(render_html(meta, rows), encoding="utf-8")
    print(
        f"{meta['asof']}  库内 {meta['in_db']}  入选 {meta['n']}"
        f"  延续 {meta['cont']}  回调 {meta['pull']}"
        f"  去掉板块 {meta['dropped_board']}  ST {meta['dropped_st']}"
    )
    print(out)


if __name__ == "__main__":
    main()
