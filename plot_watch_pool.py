"""
根据某日观察池，生成可交互 HTML 蜡烛图（浏览器内用 ECharts 绘制）。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
DAILY_DIR = ROOT / "data" / "daily_raw"
POOL_DIR = ROOT / "watch_pool"


def resolve_day_dir(date: str | None = None) -> tuple[Path, str]:
    """返回 (当日目录, YYYY-MM-DD)。未指定日期时取最新一天。"""
    if date:
        path = POOL_DIR / date
        if not path.is_dir():
            raise FileNotFoundError(f"找不到观察池目录：{path}")
        return path, date

    if not POOL_DIR.exists():
            raise FileNotFoundError(f"未找到观察池目录：{POOL_DIR}，请先运行 run_daily_pool.py")

    dated = sorted(
        p
        for p in POOL_DIR.iterdir()
        if p.is_dir() and len(p.name) == 10 and p.name[4] == "-" and p.name[7] == "-"
    )
    if not dated:
        raise FileNotFoundError(f"{POOL_DIR} 下没有日期目录，请先运行 run_daily_pool.py")
    path = dated[-1]
    return path, path.name


def load_pool_rows(day_dir: Path) -> pd.DataFrame:
    """读取当日汇总 pool.csv。"""
    path = day_dir / "pool.csv"
    if not path.exists():
        raise FileNotFoundError(f"找不到 {path}")
    df = pd.read_csv(path, dtype={"股票代码": str})
    if df.empty:
        raise FileNotFoundError(f"{path} 为空")
    df["股票代码"] = df["股票代码"].astype(str).str.zfill(6)
    return df


def load_qfq_bars(code: str, days: int = 365) -> tuple[list[dict], dict] | None:
    """前复权日线 + 最新一日换手/成交额。

    返回 (bars, spot)；spot 含 turnover_pct、amount_yi、volume_yi、float_yi
    （成交额按亿元，成交量/流通股本按亿股，均 4 位小数）。
    """
    code = str(code).zfill(6)
    df = None
    db = ROOT / "data" / "db" / "daily.db"
    if db.exists() and db.stat().st_size > 0:
        try:
            import daily_db

            raw = daily_db.load_bars_df(code, limit=max(days + 5, 80))
            if not raw.empty:
                df = raw
        except Exception:
            df = None
    if df is None:
        path = DAILY_DIR / f"{code}.csv"
        if not path.exists():
            return None
        df = pd.read_csv(path, dtype={"股票代码": str})

    df["日期"] = pd.to_datetime(df["日期"])
    df = df.sort_values("日期")
    latest = df.iloc[-1]
    amount = float(latest["成交额"]) if pd.notna(latest["成交额"]) else None
    turn = float(latest["换手率"]) if pd.notna(latest["换手率"]) else None
    float_shares = float(latest["流通股本"]) if pd.notna(latest["流通股本"]) else None
    spot = {
        "turnover_pct": None if turn is None else round(turn * 100.0, 2),
        "amount_yi": None if amount is None else round(amount / 1e8, 4),
        "volume_yi": (
            None
            if pd.isna(latest["成交量"])
            else round(float(latest["成交量"]) / 1e8, 4)
        ),
        "float_yi": None if float_shares is None else round(float_shares / 1e8, 4),
    }

    latest_f = float(df["后复权因子"].iloc[-1])
    for col in ["开盘", "最高", "最低", "收盘"]:
        df[col] = df[col] * df["后复权因子"] / latest_f
    end = df["日期"].iloc[-1]
    start = end - pd.Timedelta(days=days)
    df = df[df["日期"] >= start]
    if df.empty:
        return None

    bars = []
    for _, r in df.iterrows():
        amt = float(r["成交额"]) if pd.notna(r["成交额"]) else 0.0
        vol = int(r["成交量"]) if pd.notna(r["成交量"]) else 0
        bars.append(
            {
                "date": r["日期"].strftime("%Y-%m-%d"),
                "open": round(float(r["开盘"]), 4),
                "high": round(float(r["最高"]), 4),
                "low": round(float(r["最低"]), 4),
                "close": round(float(r["收盘"]), 4),
                "volume": vol,
                "amount": amt,
                "amountYi": round(amt / 1e8, 4),
            }
        )
    return bars, spot


_UI_JS_PATH = ROOT / "plot_pool_ui.js"


def _load_chart_js() -> str:
    js = _UI_JS_PATH.read_text(encoding="utf-8")
    mark = ROOT / "data" / "images" / "xueqiu_mark.png"
    if mark.exists() and "__XUEQIU_DATA_URI__" in js:
        import base64

        uri = "data:image/png;base64," + base64.b64encode(mark.read_bytes()).decode("ascii")
        js = js.replace("__XUEQIU_DATA_URI__", uri)
    elif "__XUEQIU_DATA_URI__" in js:
        # 回退：相对路径（signal_pool/日/ → ../../data/images/）
        js = js.replace(
            "__XUEQIU_DATA_URI__",
            "../../data/images/xueqiu_mark.png",
        )
    return js


_HTML_SHELL = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>__TITLE__</title>
  <script src="https://cdn.jsdelivr.net/npm/echarts@5.5.1/dist/echarts.min.js"></script>
  <style>
__CSS__
  </style>
</head>
<body>
  <div class="app">
    <aside class="sidebar">
      <div class="sidebar-title">目录</div>
      <nav id="nav"></nav>
    </aside>
    <div class="content">
      <header>
        <h1>__H1__</h1>
        __TABS__
        <p class="sub">__SUB__</p>
      </header>
      <main id="main"></main>
    </div>
  </div>
  <script>
    __DATA_JS__
    const TAB_LABEL = { long: '📈 中长线', short: '⚡ 短线', scalp: '🎯 Scalp', treasure: '💎 宝藏观察', board: '🟥 无量首板', base: '📦 底部启动', relaunch: '🔄 板后重启' };
    let currentTab = 'long';
    let charts = [];
__CHART_JS__
    if (typeof echarts === 'undefined') {
      document.getElementById('main').innerHTML =
        '<p style="padding:24px;color:#c0392b">加载 ECharts 失败，请检查网络后刷新。</p>';
    } else {
      if (typeof bindTabs === 'function') bindTabs();
      render();
      if (typeof syncHeaderOffset === 'function') syncHeaderOffset();
      if (typeof bindNavAnchors === 'function') bindNavAnchors();
      window.addEventListener('resize', () => {
        charts.forEach(c => c.resize());
        if (typeof syncHeaderOffset === 'function') syncHeaderOffset();
      });
      if (location.hash && typeof scrollToCard === 'function') {
        requestAnimationFrame(() => scrollToCard(location.hash.slice(1)));
      }
    }
  </script>
</body>
</html>
"""

_CSS = """

:root {
  --bg: #f0f2f5;
  --card: #ffffff;
  --text: #1a1d21;
  --muted: #5c6570;
  --line: #d8dee6;
  --sidebar: #f7f8fa;
  --sidebar-w: 260px;
  /* 目录锚点跳转时避开 sticky 顶栏；JS 会按实测高度覆盖 */
  --sticky-header-h: 120px;
}
* { box-sizing: border-box; }
html {
  scroll-padding-top: calc(var(--sticky-header-h) + 10px);
}
body {
  margin: 0;
  font-family: "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", sans-serif;
  background: var(--bg);
  color: var(--text);
  line-height: 1.5;
}
.app {
  display: flex;
  min-height: 100vh;
  align-items: stretch;
}
.sidebar {
  position: sticky;
  top: 0;
  align-self: flex-start;
  width: var(--sidebar-w);
  min-width: var(--sidebar-w);
  height: 100vh;
  overflow: auto;
  background: var(--sidebar);
  border-right: 1px solid var(--line);
  padding: 12px 10px 24px;
  z-index: 30;
}
.sidebar-title {
  font-size: 0.78rem;
  font-weight: 700;
  color: var(--muted);
  letter-spacing: 0.06em;
  text-transform: uppercase;
  margin: 0 6px 10px;
}
#nav {
  display: flex;
  flex-direction: column;
  gap: 2px;
  font-size: 0.84rem;
}
.nav-item {
  display: flex;
  flex-direction: column;
  align-items: stretch;
  gap: 3px;
  padding: 6px 8px;
  border-radius: 6px;
}
.nav-row {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 4px;
  min-width: 0;
}
.nav-row-main {
  width: 100%;
}
.nav-row-main .anchor {
  flex: 1 1 auto;
  min-width: 0;
}
.nav-row-tags {
  width: 100%;
  padding-left: 0;
}
.nav-item:hover { background: #e8eef8; }
.nav-item .sep { display: none; }
nav a.anchor { color: #0b57d0; text-decoration: none; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
nav a.anchor:hover { text-decoration: underline; }
nav a.anchor.buy { color: #c62828; font-weight: 700; }
.nav-industry {
  display: inline-flex;
  align-items: center;
  max-width: 7.5em;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  padding: 0 5px;
  border-radius: 4px;
  font-size: 0.68rem;
  font-weight: 650;
  line-height: 1.45;
  background: #e8f1ff;
  color: #0b57d0;
  border: 1px solid #b6d0fe;
  flex-shrink: 1;
}
.nav-industry-hot {
  background: #fff3e0;
  color: #c05600;
  border-color: #ffcc80;
}
.nav-hot {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  padding: 0;
  border: none;
  background: transparent;
  line-height: 0;
  flex-shrink: 0;
}
.nav-hot .hot-torch-icon { display: block; }
.nav-buy {
  display: inline-flex;
  align-items: center;
  padding: 0 5px;
  border-radius: 4px;
  font-size: 0.7rem;
  font-weight: 700;
  line-height: 1.45;
  background: #ffe8e6;
  color: #c62828;
  border: 1px solid #ffc1bc;
  flex-shrink: 0;
}
.nav-watch {
  display: inline-flex;
  align-items: center;
  padding: 0 5px;
  border-radius: 4px;
  font-size: 0.7rem;
  font-weight: 700;
  line-height: 1.45;
  background: #fff6e0;
  color: #b36b00;
  border: 1px solid #ffe0a3;
  flex-shrink: 0;
}
a.xq-link {
  display: inline-flex;
  align-items: center;
  line-height: 0;
  border-radius: 50%;
  opacity: 0.95;
  flex-shrink: 0;
}
a.xq-link:hover { opacity: 1; outline: 1px solid rgba(11,87,208,0.35); }
a.xq-link img,
a.xq-link svg { width: 14px; height: 14px; display: block; }
.content {
  flex: 1;
  min-width: 0;
}
header {
  position: sticky;
  top: 0;
  z-index: 20;
  background: rgba(240,242,245,0.94);
  backdrop-filter: blur(8px);
  border-bottom: 1px solid var(--line);
  padding: 14px 20px 10px;
}
h1 { margin: 0 0 4px; font-size: 1.35rem; font-weight: 650; }
.sub { margin: 0 0 4px; color: var(--muted); font-size: 0.9rem; }
main {
  max-width: 1180px;
  margin: 0 auto;
  padding: 16px 16px 48px;
  display: grid;
  gap: 14px;
}
/* Tab 模式下卡片挂在 .tab-pane 下，间距要加在这里 */
#main > .tab-pane {
  display: grid;
  gap: 14px;
}
.card {
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: 10px;
  padding: 12px 12px 4px;
  scroll-margin-top: calc(var(--sticky-header-h) + 10px);
}
.card-head {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 8px 10px;
  padding: 0 4px 8px;
}
.card-head h2 { margin: 0; font-size: 1.1rem; font-weight: 600; }
.badges {
  display: inline-flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 8px;
}
.badges-strategy {
  flex: 1 1 100%;
  width: 100%;
  margin-top: 2px;
}
.badge {
  display: inline-flex;
  align-items: center;
  padding: 3px 10px;
  border-radius: 6px;
  font-size: 0.86rem;
  font-weight: 650;
  line-height: 1.4;
  letter-spacing: 0.02em;
}
.badge-industry {
  background: #e8f1ff;
  color: #0b57d0;
  border: 1px solid #b6d0fe;
}
.badge-industry-hot {
  background: #fff3e0;
  color: #c05600;
  border-color: #ffcc80;
}
.badge-hot {
  padding: 2px 4px;
  background: transparent;
  border: none;
  line-height: 0;
}
.badge-hot .hot-torch-icon { display: block; }
.badge-advice { border: 1px solid transparent; }
.badge-advice-buy {
  background: #ffe8e6;
  color: #c62828;
  border-color: #ffc1bc;
}
.badge-advice-watch {
  background: #fff6e0;
  color: #b36b00;
  border-color: #ffe0a3;
}
.badge-advice-hot {
  background: #f0e8ff;
  color: #6a1b9a;
  border-color: #d7bff0;
}
.badge-advice-other {
  background: #eef1f4;
  color: #3c4650;
  border-color: #d5dbe3;
}
.badge-strategy {
  border: 1px solid transparent;
}
.badge-strategy-default {
  background: #eef2f7;
  color: #3c4650;
  border-color: #d5dbe3;
}
.badge-strategy-strict {
  background: #e8f1ff;
  color: #0b57d0;
  border-color: #b6d0fe;
}
.badge-strategy-r3 {
  background: #f3eef8;
  color: #6a1b9a;
  border-color: #d7bff0;
}
.badge-strategy-scalp {
  background: #e8faf4;
  color: #0d7a5f;
  border-color: #a8e6d5;
}
.badge-strategy-dry_stable,
.badge-strategy-double_trough {
  background: #fff6e0;
  color: #b36b00;
  border-color: #ffe0a3;
}
.badge-strategy-expand_after_dry,
.badge-strategy-consol_vol_up {
  background: #e8f1ff;
  color: #0b57d0;
  border-color: #b6d0fe;
}
.badge-strategy-quiet_limit_up,
.badge-strategy-low_limit_up,
.badge-strategy-quiet_first {
  background: #ffe8e6;
  color: #c62828;
  border-color: #ffc1bc;
}
.badge-strategy-bottom_base {
  background: #f3e8d8;
  color: #8d4e00;
  border-color: #e8c89a;
}
.badge-strategy-board_relaunch {
  background: #e8f5e9;
  color: #2e7d32;
  border-color: #a5d6a7;
}
.nav-strategy {
  display: inline-flex;
  align-items: center;
  padding: 0 5px;
  border-radius: 4px;
  font-size: 0.66rem;
  font-weight: 700;
  line-height: 1.45;
  flex-shrink: 0;
}
.nav-strategy-default { background: #eef2f7; color: #3c4650; border: 1px solid #d5dbe3; }
.nav-strategy-strict { background: #e8f1ff; color: #0b57d0; border: 1px solid #b6d0fe; }
.nav-strategy-r3 { background: #f3eef8; color: #6a1b9a; border: 1px solid #d7bff0; }
.nav-strategy-scalp { background: #e8faf4; color: #0d7a5f; border: 1px solid #a8e6d5; }
.nav-strategy-dry_stable, .nav-strategy-double_trough { background: #fff6e0; color: #b36b00; border: 1px solid #ffe0a3; }
.nav-strategy-expand_after_dry, .nav-strategy-consol_vol_up { background: #e8f1ff; color: #0b57d0; border: 1px solid #b6d0fe; }
.nav-strategy-quiet_limit_up, .nav-strategy-low_limit_up, .nav-strategy-quiet_first { background: #ffe8e6; color: #c62828; border: 1px solid #ffc1bc; }
.nav-strategy-bottom_base { background: #f3e8d8; color: #8d4e00; border: 1px solid #e8c89a; }
.nav-strategy-board_relaunch { background: #e8f5e9; color: #2e7d32; border: 1px solid #a5d6a7; }

.cond-tip {
  color: var(--muted);
  font-size: 0.86rem;
  font-weight: 500;
  max-width: 100%;
  line-height: 1.45;
}
.cond-tip .cond-label {
  color: #3c4650;
  font-weight: 700;
  margin-right: 2px;
}
.meta {
  display: inline-flex;
  flex-wrap: wrap;
  gap: 6px;
  align-items: center;
  width: 100%;
  margin-top: 2px;
}
.metric {
  display: inline-flex;
  align-items: baseline;
  gap: 4px;
  padding: 3px 9px;
  border-radius: 6px;
  background: #eef2f7;
  border: 1px solid #d5dbe3;
  line-height: 1.35;
}
.metric-k {
  color: #5c6570;
  font-size: 0.72rem;
  font-weight: 650;
  letter-spacing: 0.02em;
}
.metric-v {
  color: #1a1d21;
  font-size: 0.92rem;
  font-weight: 700;
  font-variant-numeric: tabular-nums;
}
.metric.metric-turn { background: #fff6e0; border-color: #ffe0a3; }
.metric.metric-turn .metric-v { color: #b36b00; }
.metric.metric-amt { background: #e8f1ff; border-color: #b6d0fe; }
.metric.metric-amt .metric-v { color: #0b57d0; }
.metric.metric-vol { background: #e8faf4; border-color: #a8e6d5; }
.metric.metric-vol .metric-v { color: #0d7a5f; }
.metric.metric-float { background: #f3eef8; border-color: #d7bff0; }
.metric.metric-float .metric-v { color: #6a1b9a; }
.metric.metric-score { background: #f0f2f5; border-color: #d8dee6; }
.metric.metric-bars { background: #f0f2f5; border-color: #d8dee6; }
.metric.metric-bars .metric-v { font-weight: 650; color: #5c6570; }
.metric.chg-up { background: #ffe8e6; border-color: #ffc1bc; }
.metric.chg-up .metric-v { color: #c62828; }
.metric.chg-down { background: #e6f7ee; border-color: #a8e0c0; }
.metric.chg-down .metric-v { color: #0f8a4b; }
.metric.chg-flat { background: #eef2f7; border-color: #d5dbe3; }
.metric.chg-flat .metric-v { color: #8a939e; }
.chg-up { color: #e74c3c; font-weight: 600; }
.chg-down { color: #14b15b; font-weight: 600; }
.chg-flat { color: #8a939e; font-weight: 600; }
.chart-toggle {
  appearance: none;
  border: none;
  background: transparent;
  color: #0b57d0;
  font-size: 0.88rem;
  font-weight: 650;
  padding: 0;
  margin: 0;
  cursor: pointer;
  text-decoration: underline;
  text-underline-offset: 2px;
  white-space: nowrap;
}
.chart-toggle:hover { color: #0842a0; }
.chart-wrap { width: 100%; }
.chart-wrap.open { margin-top: 4px; }
.chart-loading, .chart-error {
  margin: 4px 0 8px;
  padding: 10px 12px;
  color: var(--muted);
  font-size: 0.88rem;
  background: #f7f9fc;
  border-radius: 8px;
}
.chart-error { color: #c0392b; }
.chart-error code {
  font-size: 0.84rem;
  background: #ffe8e6;
  padding: 1px 6px;
  border-radius: 4px;
}
.chart { width: 100%; height: 560px; }
.tabs {
  display: flex;
  gap: 8px;
  margin: 6px 0 10px;
  flex-wrap: wrap;
}
.tab-btn {
  appearance: none;
  border: 1px solid var(--line);
  background: #fff;
  color: var(--muted);
  border-radius: 8px;
  padding: 6px 14px;
  font-size: 0.92rem;
  font-weight: 650;
  cursor: pointer;
}
.tab-btn:hover { border-color: #9db7e8; color: #0b57d0; }
.tab-btn.active {
  background: #e8f1ff;
  border-color: #0b57d0;
  color: #0b57d0;
}
.tab-pane[hidden] {
  display: none !important;
}
#main > .tab-pane.tab-enter {
  animation: tabContentIn 0.32s ease both;
}
@keyframes tabContentIn {
  from {
    opacity: 0;
    transform: translateY(10px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}
@media (prefers-reduced-motion: reduce) {
  #main > .tab-pane.tab-enter { animation: none; }
}
@media (max-width: 860px) {
  .app { flex-direction: column; }
  .sidebar {
    position: relative;
    width: 100%;
    min-width: 0;
    height: auto;
    max-height: 180px;
    border-right: none;
    border-bottom: 1px solid var(--line);
  }
  #nav { flex-direction: row; flex-wrap: wrap; }
  .chart { height: 480px; }
}

  """

def build_html(
    date: str,
    stocks: list[dict],
    out_html: Path,
    days: int,
    *,
    panels: dict[str, list[dict]] | None = None,
) -> None:
    """生成观察/信号池 HTML。panels 含 long/short 时启用 Tab，导航标注可买。"""
    tabbed = panels is not None
    payload_obj: dict | list = panels if tabbed else stocks
    payload = json.dumps(payload_obj, ensure_ascii=False, separators=(',', ':'))
    payload_js = payload.replace("<", "\u003c").replace(">", "\u003e")

    if tabbed:
        data_js = (
            f"const PANELS = {payload_js};\n"
            "    const STOCKS = [];\n"
            "    const TABBED = true;"
        )
        title = f"信号池 {date}"
        h1 = f"信号池 · {date}"
        tabs = """
    <div class="tabs" id="tabs">
      <button type="button" class="tab-btn active" data-tab="long">📈 中长线 <span id="n-long">0</span></button>
      <button type="button" class="tab-btn" data-tab="short">⚡ 短线 <span id="n-short">0</span></button>
      <button type="button" class="tab-btn" data-tab="scalp">🎯 Scalp <span id="n-scalp">0</span></button>
      <button type="button" class="tab-btn" data-tab="treasure">💎 宝藏观察 <span id="n-treasure">0</span></button>
      <button type="button" class="tab-btn" data-tab="board">🟥 无量首板 <span id="n-board">0</span></button>
      <button type="button" class="tab-btn" data-tab="base">📦 底部启动 <span id="n-base">0</span></button>
      <button type="button" class="tab-btn" data-tab="relaunch">🔄 板后重启 <span id="n-relaunch">0</span></button>
    </div>"""
        sub = (
            f'<span id="tab-label">📈 中长线</span> · 共 <span id="count">0</span> 只'
            f'（可买 <span id="buy-count">0</span>）· 近 {days} 日前复权 · 按评分排序 · 左侧目录 · 红字=买入 / 黄标=观察'
        )
    else:
        data_js = (
            f"const STOCKS = {payload_js};\n"
            "    const PANELS = {};\n"
            "    const TABBED = false;"
        )
        title = f"每日观察池 {date}"
        h1 = f"每日观察池 · {date}"
        tabs = ""
        sub = f'共 <span id="count">0</span> 只 · 近 {days} 日前复权 · 浏览器内绘制（可缩放拖拽）'

    doc = (
        _HTML_SHELL
        .replace("__TITLE__", title)
        .replace("__H1__", h1)
        .replace("__TABS__", tabs)
        .replace("__SUB__", sub)
        .replace("__DATA_JS__", data_js)
        .replace("__CHART_JS__", _load_chart_js())
        .replace("__CSS__", _CSS)
    )
    out_html.write_text(doc, encoding="utf-8")
    print(f"HTML 已生成：{out_html}")


def day_return_pct(bars: list[dict]) -> float | None:
    """最近一根相对前一日收盘的涨跌幅（%）。"""
    if len(bars) < 2:
        return None
    prev = bars[-2]["close"]
    cur = bars[-1]["close"]
    if not prev:
        return None
    return (cur / prev - 1.0) * 100.0


def run(date: str | None = None, days: int = 365) -> Path:
    day_dir, date = resolve_day_dir(date)
    pool = load_pool_rows(day_dir)

    stocks: list[dict] = []
    for _, row in pool.iterrows():
        code = row["股票代码"]
        name = str(row.get("股票名称", "") or code)
        loaded = load_qfq_bars(code, days=days)
        if not loaded:
            print(f"  跳过 {code}：无日线")
            continue
        bars, spot = loaded
        ret60 = (
            float(row["近60日涨幅"])
            if "近60日涨幅" in row and pd.notna(row["近60日涨幅"])
            else None
        )
        close = float(row["收盘"]) if "收盘" in row and pd.notna(row["收盘"]) else None
        ret1d = day_return_pct(bars)
        stocks.append(
            {
                "code": code,
                "name": name,
                "industry": str(row.get("行业板块", "") or ""),
                "close": close,
                "ret1d": None if ret1d is None else round(ret1d, 2),
                "ret60": None if ret60 is None else round(ret60, 2),
                "turnover": spot["turnover_pct"],
                "amountYi": spot["amount_yi"],
                "volumeYi": spot["volume_yi"],
                "floatYi": spot["float_yi"],
                "bars": bars,
            }
        )
        print(
            f"  {code} {name}  {len(bars)} bars  今日{ret1d:+.2f}%  换手{spot['turnover_pct']}%  "
            f"成交额{spot['amount_yi']}亿  成交量{spot['volume_yi']}亿股  流通{spot['float_yi']}亿股"
            if ret1d is not None
            else f"  {code} {name}  {len(bars)} bars"
        )

    out_html = day_dir / "index.html"
    build_html(date, stocks, out_html, days)
    return out_html


def main() -> None:
    parser = argparse.ArgumentParser(description="观察池交互式 HTML 蜡烛图")
    parser.add_argument("--date", default=None, help="观察池日期 YYYY-MM-DD，默认最新")
    parser.add_argument("--days", type=int, default=365, help="K 线回溯自然日，默认 365")
    args = parser.parse_args()
    run(date=args.date, days=args.days)


if __name__ == "__main__":
    main()
