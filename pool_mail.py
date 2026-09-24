"""
信号池邮件摘要：无 K 线、静态 HTML、适配手机。

配置见 .env.example。发送走 QQ SMTP（授权码与 POP3/IMAP 设置里同一份）。

用法：
  .venv/bin/python run_daily_pool.py --mail
  .venv/bin/python run_daily_pool.py --close --mail
  make daily ARGS='--mail'
  make close ARGS='--mail'
"""

from __future__ import annotations

import html
import os
import smtplib
import ssl
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ENV_PATH = ROOT / ".env"

# 重点：短线可买 + 观察簿全量；其它 tab 仅可买类
MAIL_SECTIONS: list[tuple[str, str, str]] = [
    ("short", "⚡ 短线可短打", "buy"),
    ("watch", "👀 观察簿", "all"),
    ("long", "📈 中长线可买入", "buy"),
    ("scalp", "🎯 Scalp 可短打", "buy"),
    ("treasure", "💎 宝藏", "buy"),
    ("board", "📦 涨停箱体回踩", "buy"),
    ("base", "📦 底部启动", "buy"),
    ("relaunch", "🔄 板后重启", "buy"),
    ("wyckoff", "🧪 威科夫 · 实验", "all"),
]


def log(msg: str) -> None:
    print(msg, flush=True)


def load_dotenv(path: Path | None = None) -> None:
    """把 .env 写入 os.environ（已存在的键不覆盖）。"""
    p = path or ENV_PATH
    if not p.exists():
        return
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = val


def mail_config() -> dict:
    load_dotenv()
    user = (os.environ.get("MAIL_USER") or "").strip()
    auth = (
        os.environ.get("MAIL_AUTH_CODE")
        or os.environ.get("MAIL_PASSWORD")
        or os.environ.get("MAIL_PASS")
        or ""
    ).strip()
    to_addr = (os.environ.get("MAIL_TO") or user).strip()
    host = (os.environ.get("MAIL_SMTP_HOST") or "smtp.qq.com").strip()
    port = int(os.environ.get("MAIL_SMTP_PORT") or "465")
    if not user or not auth:
        raise SystemExit(
            "发信缺配置：请在项目根目录 .env 填写 MAIL_USER / MAIL_AUTH_CODE"
            "（可参考 .env.example）"
        )
    if not to_addr:
        raise SystemExit("发信缺收件人：设置 MAIL_TO 或 MAIL_USER")
    return {
        "user": user,
        "auth": auth,
        "to": to_addr,
        "host": host,
        "port": port,
    }


def _is_buy(s: dict) -> bool:
    if s.get("canBuy") is True:
        return True
    a = str(s.get("advice") or "")
    return ("可买入" in a) or ("可短打" in a) or a == "买入"


def _strip_stock(s: dict) -> dict:
    """去掉 bars 等大字段，邮件只保留摘要。"""
    keep = {
        "code",
        "name",
        "industry",
        "advice",
        "canBuy",
        "when",
        "close",
        "ret1d",
        "ret60",
        "turnover",
        "amountYi",
        "score",
        "hardScore",
        "scoreEntry",
        "scoreSetup",
        "dist60LowPct",
        "ret20Pct",
        "warnFarFromLow",
        "warnImpulse",
        "hotSector",
        "hotTier",
        "strategyTags",
        "boxBottom",
        "takeProfit",
    }
    return {k: s[k] for k in keep if k in s and s[k] is not None}


def filter_panels_for_mail(
    panels: dict[str, list[dict]],
    sections: list[tuple[str, str, str]] | None = None,
) -> dict[str, list[dict]]:
    sections = sections or MAIL_SECTIONS
    out: dict[str, list[dict]] = {}
    for key, _title, mode in sections:
        rows = panels.get(key) or []
        if mode == "buy":
            rows = [s for s in rows if _is_buy(s)]
        out[key] = [_strip_stock(s) for s in rows]
    return out


def _xueqiu_url(code: str) -> str:
    c = str(code).zfill(6)
    prefix = "SH" if c.startswith(("5", "6", "9")) else "SZ"
    return f"https://xueqiu.com/S/{prefix}{c}"


def _fmt_pct(x) -> str:
    if x is None:
        return "—"
    try:
        v = float(x)
    except Exception:
        return "—"
    return f"{v:+.2f}%"


def _fmt_price(x) -> str:
    if x is None:
        return "—"
    try:
        return f"{float(x):.2f}"
    except Exception:
        return "—"


def _chg_class(x) -> str:
    try:
        v = float(x)
    except Exception:
        return ""
    if v > 0:
        return "up"
    if v < 0:
        return "down"
    return ""


def _esc(s) -> str:
    return html.escape("" if s is None else str(s))


def _name_map() -> dict[str, str]:
    try:
        from analyze_short_burst_features import load_maps

        return load_maps()[0]
    except Exception:
        return {}


def _stock_name(s: dict, name_map: dict[str, str] | None = None) -> str:
    code = str(s.get("code") or "").zfill(6)
    name = str(s.get("name") or "").strip()
    if name.lower() == "nan":
        name = ""
    if name and name != code:
        return name
    nm = name_map if name_map is not None else _name_map()
    return nm.get(code) or name or code


def _card_html(s: dict, name_map: dict[str, str] | None = None) -> str:
    code = str(s.get("code") or "").zfill(6)
    name = _stock_name(s, name_map)
    advice = str(s.get("advice") or "")
    when = str(s.get("when") or "").strip()
    ind = str(s.get("industry") or "")
    tags = s.get("strategyTags") or []
    if isinstance(tags, str):
        tags = [t for t in tags.split("|") if t]
    badge_bits = []
    if advice:
        cls = "buy" if _is_buy(s) else "watch"
        badge_bits.append(f'<span class="badge {cls}">{_esc(advice)}</span>')
    if s.get("hotSector"):
        badge_bits.append(
            f'<span class="badge hot">{_esc(s.get("hotTier") or "热门")}</span>'
        )
    if s.get("warnFarFromLow"):
        badge_bits.append('<span class="badge warn">离底远</span>')
    if s.get("warnImpulse"):
        badge_bits.append('<span class="badge warn">急涨浅回</span>')
    for t in tags[:4]:
        badge_bits.append(f'<span class="badge strat">{_esc(t)}</span>')
    if ind:
        badge_bits.append(f'<span class="badge ind">{_esc(ind)}</span>')
    metrics = [
        f'<span class="m"><b>收盘</b> {_esc(_fmt_price(s.get("close")))}</span>',
        f'<span class="m {_chg_class(s.get("ret1d"))}">'
        f'<b>今日</b> {_esc(_fmt_pct(s.get("ret1d")))}</span>',
    ]
    if s.get("turnover") is not None:
        metrics.append(
            f'<span class="m"><b>换手</b> {float(s["turnover"]):.2f}%</span>'
        )
    if s.get("amountYi") is not None:
        metrics.append(
            f'<span class="m"><b>额</b> {float(s["amountYi"]):.2f}亿</span>'
        )
    if s.get("dist60LowPct") is not None:
        metrics.append(
            f'<span class="m"><b>距60低</b> +{float(s["dist60LowPct"]):.1f}%</span>'
        )
    when_html = f'<div class="when">{_esc(when)}</div>' if when else ""
    return (
        f'<article class="card">'
        f'<a class="title" href="{_esc(_xueqiu_url(code))}">'
        f"{_esc(code)} {_esc(name)}</a>"
        f'<div class="badges">{"".join(badge_bits)}</div>'
        f'<div class="metrics">{"".join(metrics)}</div>'
        f"{when_html}</article>"
    )


def render_mail_html(
    asof: str,
    panels: dict[str, list[dict]],
    *,
    sections: list[tuple[str, str, str]] | None = None,
    show_toc: bool = True,
    page_title: str | None = None,
    subtitle: str | None = None,
) -> str:
    """静态邮件 HTML：无 JS / 无 K 线，单栏适合手机。"""
    sections = sections or MAIL_SECTIONS
    filtered = filter_panels_for_mail(panels, sections)
    name_map = _name_map()
    sections_html: list[str] = []
    toc_items: list[str] = []
    total = 0
    for key, title, _mode in sections:
        rows = filtered.get(key) or []
        n = len(rows)
        total += n
        anchor = f"sec-{key}"
        if show_toc:
            short = title.split(" ", 1)[-1] if " " in title else title
            toc_cls = "toc-item" if n else "toc-item empty"
            toc_items.append(
                f'<a class="{toc_cls}" href="#{anchor}">'
                f'<span class="toc-t">{_esc(short)}</span>'
                f'<span class="toc-n">{n}</span></a>'
            )
        if not rows:
            sections_html.append(
                f'<section class="sec" id="{anchor}">'
                f'<div class="sec-head"><h2>{_esc(title)}</h2>'
                f'<span class="n">0</span></div>'
                f'<p class="empty">暂无</p></section>'
            )
            continue
        cards = [_card_html(s, name_map) for s in rows]
        sections_html.append(
            f'<section class="sec" id="{anchor}">'
            f'<div class="sec-head"><h2>{_esc(title)}</h2>'
            f'<span class="n">{n}</span></div>'
            f'{"".join(cards)}</section>'
        )

    generated = datetime.now().strftime("%Y-%m-%d %H:%M")
    h1 = page_title or f"信号池 · {asof}"
    sub = subtitle or f"无 K 线摘要 · 共 {total} 条 · 生成于 {generated}"
    toc_html = (
        f'<nav class="toc">{"".join(toc_items)}</nav>' if show_toc else ""
    )
    sub_margin = "0 0 12px" if show_toc else "0"
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1"/>
<title>{_esc(h1)}</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    padding: 12px;
    background: #f0f2f5;
    color: #1a1d21;
    font-family: -apple-system, BlinkMacSystemFont, "PingFang SC",
      "Hiragino Sans GB", "Microsoft YaHei", sans-serif;
    font-size: 15px;
    line-height: 1.45;
    -webkit-text-size-adjust: 100%;
  }}
  .head {{
    background: #fff;
    border-radius: 12px;
    padding: 14px 16px;
    margin-bottom: 12px;
  }}
  .head h1 {{ margin: 0 0 6px; font-size: 18px; }}
  .head .sub {{ margin: {sub_margin}; color: #5c6570; font-size: 13px; }}
  .toc {{
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
  }}
  .toc-item {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 8px 10px;
    border-radius: 10px;
    background: #eef4ff;
    color: #1a56db;
    text-decoration: none;
    font-size: 13px;
    font-weight: 600;
    border: 1px solid #d6e4ff;
  }}
  .toc-item.empty {{
    background: #f3f5f8;
    color: #8a939e;
    border-color: #e5e9ef;
    font-weight: 500;
  }}
  .toc-n {{
    min-width: 18px;
    text-align: center;
    padding: 0 6px;
    border-radius: 999px;
    background: rgba(26, 86, 219, 0.12);
    font-size: 12px;
  }}
  .toc-item.empty .toc-n {{ background: rgba(0,0,0,0.06); }}
  .sec {{
    background: #fff;
    border-radius: 12px;
    padding: 0 14px 4px;
    margin-bottom: 12px;
    overflow: hidden;
  }}
  .sec-head {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 10px;
    margin: 0 -14px 10px;
    padding: 12px 14px;
    background: #1a56db;
    color: #fff;
  }}
  .sec-head h2 {{
    margin: 0;
    font-size: 17px;
    font-weight: 700;
    letter-spacing: 0.02em;
  }}
  .sec-head .n {{
    color: #fff;
    background: rgba(255,255,255,0.22);
    border-radius: 999px;
    padding: 2px 10px;
    font-weight: 700;
    font-size: 13px;
  }}
  .empty {{ color: #8a939e; font-size: 13px; margin: 0 0 10px; }}
  .card {{
    border-top: 1px solid #e8ebf0;
    padding: 12px 0;
  }}
  .card:first-of-type {{ border-top: none; }}
  .title {{
    display: block;
    font-weight: 600;
    font-size: 16px;
    color: #1a56db;
    text-decoration: none;
    margin-bottom: 6px;
  }}
  .badges {{ display: flex; flex-wrap: wrap; gap: 4px; margin-bottom: 6px; }}
  .badge {{
    display: inline-block;
    padding: 2px 7px;
    border-radius: 999px;
    font-size: 12px;
    background: #eef1f5;
    color: #3d4450;
  }}
  .badge.buy {{ background: #ffe8e6; color: #c62828; }}
  .badge.watch {{ background: #fff6e0; color: #a15c00; }}
  .badge.hot {{ background: #e8f0ff; color: #1a56db; }}
  .badge.warn {{ background: #fff0e6; color: #d35400; }}
  .badge.strat {{ background: #f0eaff; color: #5b3cc4; }}
  .badge.ind {{ background: #eef1f5; color: #5c6570; }}
  .metrics {{
    display: flex;
    flex-wrap: wrap;
    gap: 8px 12px;
    font-size: 13px;
    color: #3d4450;
  }}
  .metrics .m b {{
    font-weight: 500;
    color: #8a939e;
    margin-right: 2px;
  }}
  .up {{ color: #c62828 !important; }}
  .down {{ color: #2e7d32 !important; }}
  .when {{
    margin-top: 6px;
    font-size: 12px;
    color: #5c6570;
    word-break: break-word;
  }}
  .foot {{
    text-align: center;
    color: #8a939e;
    font-size: 12px;
    padding: 8px 0 16px;
  }}
  @media (min-width: 560px) {{
    body {{ max-width: 560px; margin: 0 auto; }}
  }}
</style>
</head>
<body>
  <header class="head">
    <h1>{_esc(h1)}</h1>
    <p class="sub">{_esc(sub)}</p>
    {toc_html}
  </header>
  {"".join(sections_html)}
  <p class="foot">完整带图版本请打开电脑上的 signal_pool/{html.escape(asof)}/index.html</p>
</body>
</html>
"""


def send_html_mail(
    *,
    subject: str,
    html_body: str,
    cfg: dict | None = None,
) -> None:
    cfg = cfg or mail_config()
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = cfg["user"]
    msg["To"] = cfg["to"]
    msg.attach(MIMEText("请使用支持 HTML 的客户端查看信号池摘要。", "plain", "utf-8"))
    msg.attach(MIMEText(html_body, "html", "utf-8"))

    context = ssl.create_default_context()
    port = int(cfg["port"])
    host = cfg["host"]
    to_disp = cfg["to"]
    if "@" in to_disp:
        local, _, domain = to_disp.partition("@")
        to_disp = f"{local[:3]}***@{domain}"
    log(f"  发信 → {to_disp}  via {host}:{port}")
    if port == 465:
        with smtplib.SMTP_SSL(host, port, context=context, timeout=60) as smtp:
            smtp.login(cfg["user"], cfg["auth"])
            smtp.sendmail(cfg["user"], [cfg["to"]], msg.as_string())
    else:
        with smtplib.SMTP(host, port, timeout=60) as smtp:
            smtp.ehlo()
            smtp.starttls(context=context)
            smtp.ehlo()
            smtp.login(cfg["user"], cfg["auth"])
            smtp.sendmail(cfg["user"], [cfg["to"]], msg.as_string())
    log("  邮件已发送")


def send_pool_mail(
    asof: str,
    panels: dict[str, list[dict]],
) -> None:
    """正式日更：多 tab 摘要（含目录）。"""
    body = render_mail_html(asof, panels)
    send_html_mail(subject=f"信号池 {asof}", html_body=body)


WATCH_MAIL_SECTIONS: list[tuple[str, str, str]] = [
    ("watch", "👀 观察簿", "all"),
]


def send_watch_mail(asof: str, watch_stocks: list[dict]) -> None:
    """收盘：仅观察簿名单，无目录。"""
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")
    body = render_mail_html(
        asof,
        {"watch": watch_stocks or []},
        sections=WATCH_MAIL_SECTIONS,
        show_toc=False,
        page_title=f"观察簿 · {asof}",
        subtitle=f"收盘观察池 · 共 {len(watch_stocks or [])} 条 · 生成于 {generated}",
    )
    send_html_mail(subject=f"观察簿 {asof}", html_body=body)
