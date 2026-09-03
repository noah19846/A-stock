"""
短线爆发信号回测

信号日 T（收盘后判定）：
  hard/soft 门槛来自 bands JSON（signal_hard / signal_soft）
  + 流动性/过热硬过滤

成交：
  默认次日开盘买入；对照：次日收盘买入

出场（相对买入价，持有最多 hold_n 日）：
  - 盘中最高触及 +target → 按目标价出
  - 盘中最低触及 -stop → 按止损价出
  - 同日既触目标又触止损 → 按「先止损」
  - 否则持有期满按收盘出

成本：双边合计 cost_bps（默认 15bp）

加速：
  - daily_cache 一次预载
  - 每股向量化预计算特征（避免 features_at 逐日 O(n²)）
  - --targets 0.15,0.08 同一次扫信号、多套出场

用法：
  .venv/bin/python short_burst_backtest.py
  .venv/bin/python short_burst_backtest.py --bands data/short_burst_feature_bands_hw.json --targets 0.15,0.08
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_short_burst_features import impulse_blocks_trade, load_maps, mv_bounds, score_row

ROOT = Path(__file__).resolve().parent
DAILY = ROOT / "data" / "daily_raw"
BANDS = ROOT / "data" / "short_burst_feature_bands.json"

LOOKBACK = 520
HOLD_N = 8
TARGET = 0.15
STOP = 0.03
COST_BPS = 15.0



def load_bands(path: Path | None = None) -> dict:
    p = path or BANDS
    if not p.exists():
        raise SystemExit(f"缺少 {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def is_signal(feat: dict, soft: float, hard: float, bands: dict) -> bool:
    signal_soft = float(bands.get("signal_soft", bands.get("min_score", 0.72)))
    signal_hard = float(bands.get("signal_hard", 0.85))
    if hard < signal_hard or soft < signal_soft:
        return False
    mv_lo, mv_hi = mv_bounds(bands)
    if feat["流通市值亿"] < mv_lo or feat["流通市值亿"] > mv_hi:
        return False
    if feat["换手率%"] > 12 or feat["换手率%"] < 0.8:
        return False
    if feat["前5日涨幅%"] > 8 or feat["距20日高点%"] > -2:
        return False
    if feat.get("前1日涨幅%", 0) > 3:
        return False
    if impulse_blocks_trade(feat, bands):
        return False
    # 离 60 日低点涨太多：底部段已走完，默认不进信号
    max_from_low = bands.get("max_dist60_low_pct")
    if max_from_low is not None:
        v = feat.get("距60日低点%")
        if v is None or not np.isfinite(float(v)) or float(v) > float(max_from_low):
            return False
    return True


def _roll_mean(a: np.ndarray, n: int) -> np.ndarray:
    return pd.Series(a).rolling(n, min_periods=n).mean().to_numpy()


def _roll_max(a: np.ndarray, n: int) -> np.ndarray:
    return pd.Series(a).rolling(n, min_periods=1).max().to_numpy()


def _roll_min(a: np.ndarray, n: int) -> np.ndarray:
    return pd.Series(a).rolling(n, min_periods=1).min().to_numpy()


def _roll_std(a: np.ndarray, n: int) -> np.ndarray:
    return pd.Series(a).rolling(n, min_periods=n).std().to_numpy()


def precompute_feature_table(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """一次性算好全序列特征，供快速逐日打分。"""
    px = df["px"].to_numpy(dtype=float)
    hi = df["hi"].to_numpy(dtype=float)
    lo = df["lo"].to_numpy(dtype=float)
    vol = df["vol"].to_numpy(dtype=float)
    amt = df["amt"].to_numpy(dtype=float)
    ret = df["ret"].to_numpy(dtype=float)
    turn = df["turn"].to_numpy(dtype=float)
    mv = df["mv"].to_numpy(dtype=float)
    ma5 = df["ma5"].to_numpy(dtype=float)
    ma10 = df["ma10"].to_numpy(dtype=float)
    ma20 = df["ma20"].to_numpy(dtype=float)
    n = len(df)

    def lag_ret(k: int) -> np.ndarray:
        out = np.full(n, np.nan)
        if n > k:
            prev = px[:-k]
            cur = px[k:]
            ok = prev > 0
            out[k:] = np.where(ok, (cur / prev - 1.0) * 100.0, np.nan)
        return out

    amt5 = _roll_mean(amt, 5)
    amt20 = _roll_mean(amt, 20)
    vol5 = _roll_mean(vol, 5)
    vol20 = _roll_mean(vol, 20)
    hi20 = _roll_max(px, 20)
    hi60 = _roll_max(px, 60)
    lo60 = _roll_min(lo, 60)

    # 近 10 日上涨占比 / 波动 / 安静日：用 rolling
    ret_pos = (ret > 0).astype(float)
    ret_abs = np.abs(ret)
    up10 = _roll_mean(ret_pos, 10) * 100.0
    vol10 = _roll_std(ret, 10) * 100.0
    quiet10 = _roll_mean((ret_abs < 0.015).astype(float), 10) * 100.0

    # 距上次大涨>5%：O(n)
    big = np.isfinite(ret) & (ret > 0.05)
    last = -1
    days_since = np.empty(n, dtype=float)
    for i in range(n):
        if big[i]:
            last = i
        days_since[i] = float(i - last) if last >= 0 else 999.0

    with np.errstate(divide="ignore", invalid="ignore"):
        table = {
            "前1日涨幅%": lag_ret(1),
            "前3日涨幅%": lag_ret(3),
            "前5日涨幅%": lag_ret(5),
            "前10日涨幅%": lag_ret(10),
            "前20日涨幅%": lag_ret(20),
            "收盘/MA5": px / ma5,
            "收盘/MA10": px / ma10,
            "收盘/MA20": px / ma20,
            "MA5/MA10": ma5 / ma10,
            "MA10/MA20": ma10 / ma20,
            "距20日高点%": (px / hi20 - 1.0) * 100.0,
            "距60日高点%": (px / hi60 - 1.0) * 100.0,
            "距60日低点%": (px / lo60 - 1.0) * 100.0,
            "当日振幅%": (hi - lo) / px * 100.0,
            "额能比1_5": amt / amt5,
            "额能比5_20": amt5 / amt20,
            "量能比5_20": vol5 / vol20,
            "换手率%": turn,
            "流通市值亿": mv,
            "前10日上涨占比%": up10,
            "前10日波动%": vol10,
            "安静日占比10%": quiet10,
            "距上次大涨>5%天数": days_since,
        }
    return table


def feat_at_table(table: dict[str, np.ndarray], i: int, ma5: float, ma10: float, ma20: float) -> dict | None:
    if not all(np.isfinite([ma5, ma10, ma20])) or min(ma5, ma10, ma20) <= 0:
        return None
    feat = {k: float(v[i]) for k, v in table.items()}
    if not np.isfinite(feat["收盘/MA5"]):
        return None
    return feat


def simulate_trade_arr(
    op: np.ndarray,
    hi: np.ndarray,
    lo: np.ndarray,
    px: np.ndarray,
    dates: pd.Series,
    signal_i: int,
    entry_mode: str,
    hold_n: int,
    target: float,
    stop: float,
    cost: float,
    *,
    protect_at: float | None = None,
    protect_floor: float = 0.0,
    trail: float | None = None,
    protect_delay: int = 0,
) -> dict | None:
    """
    protect_at: 浮盈达到该比例后启动保护（相对买入价）
    protect_floor: 启动后止损抬到 entry*(1+floor)，0=保本
    trail: 启动后按最高价回撤 trail 跟踪（价=peak_hi*(1-trail)）
    protect_delay: 0=当日高点触及即抬止损；1=收盘确认后次日开盘才生效（更稳）
    """
    buy_i = signal_i + 1
    if buy_i >= len(px):
        return None
    entry = float(op[buy_i] if entry_mode == "open" else px[buy_i])
    if not np.isfinite(entry) or entry <= 0:
        return None

    prev = float(px[signal_i])
    if entry_mode == "open" and prev > 0 and (entry / prev - 1.0) >= 0.095:
        return {
            "skipped": True,
            "reason": "次日高开近涨停，放弃",
            "signal_i": signal_i,
            "buy_i": buy_i,
            "exit_i": buy_i,
        }

    last = min(buy_i + hold_n - 1, len(px) - 1)
    exit_i = None
    exit_px = None
    reason = None
    tp = entry * (1.0 + target)
    hard_sl = entry * (1.0 - stop)
    sl = hard_sl
    peak_hi = entry
    armed = False
    pending_arm = False
    floor_px = entry * (1.0 + float(protect_floor))
    delay = int(protect_delay or 0)

    for j in range(buy_i, last + 1):
        # 次日生效：昨日收盘确认后，今日开盘抬止损 / 更新跟踪
        if delay >= 1 and pending_arm and not armed:
            armed = True
            pending_arm = False
            if floor_px > sl:
                sl = floor_px
        if delay >= 1 and armed and trail is not None:
            trail_px = peak_hi * (1.0 - float(trail))
            if trail_px > sl:
                sl = trail_px

        day_hi = float(hi[j])
        day_lo = float(lo[j])

        if delay < 1:
            # 当日抬止损：先用高点更新保护，再判断是否触及止损（偏乐观）
            if np.isfinite(day_hi) and day_hi > peak_hi:
                peak_hi = day_hi
            peak_ret = peak_hi / entry - 1.0
            if protect_at is not None and peak_ret >= float(protect_at):
                armed = True
                if floor_px > sl:
                    sl = floor_px
            if trail is not None and armed:
                trail_px = peak_hi * (1.0 - float(trail))
                if trail_px > sl:
                    sl = trail_px

        hit_tp = day_hi >= tp
        hit_sl = day_lo <= sl
        if hit_tp and hit_sl:
            exit_i, exit_px = j, sl
            reason = "同日止损优先"
            break
        if hit_sl:
            exit_i, exit_px = j, sl
            if armed and sl > hard_sl + 1e-12:
                reason = (
                    "跟踪止损"
                    if (trail is not None and sl > floor_px + 1e-12)
                    else "保本止损"
                )
            else:
                reason = "止损"
            break
        if hit_tp:
            exit_i, exit_px, reason = j, tp, "止盈"
            break

        # 收盘后更新峰值；delay>=1 时仅挂起，次日再生效
        if np.isfinite(day_hi) and day_hi > peak_hi:
            peak_hi = day_hi
        if protect_at is not None and (peak_hi / entry - 1.0) >= float(protect_at):
            if delay >= 1:
                pending_arm = True
            else:
                armed = True

    if exit_i is None:
        exit_i = last
        exit_px = float(px[exit_i])
        reason = "到期"

    raw_ret = exit_px / entry - 1.0
    net_ret = raw_ret - cost
    hold_days = int(exit_i - buy_i + 1)
    mfe = float(hi[buy_i : exit_i + 1].max() / entry - 1.0)
    mae = float(lo[buy_i : exit_i + 1].min() / entry - 1.0)

    return {
        "skipped": False,
        "signal_date": str(dates.iloc[signal_i].date()),
        "entry_date": str(dates.iloc[buy_i].date()),
        "exit_date": str(dates.iloc[exit_i].date()),
        "entry": round(entry, 4),
        "exit": round(exit_px, 4),
        "hold_days": hold_days,
        "exit_reason": reason,
        "raw_ret%": round(raw_ret * 100.0, 3),
        "net_ret%": round(net_ret * 100.0, 3),
        "mfe%": round(mfe * 100.0, 3),
        "mae%": round(mae * 100.0, 3),
        "buy_i": buy_i,
        "exit_i": exit_i,
        "target": target,
        "armed": armed,
    }


def filter_trades_daily_top_k(
    trades: list[dict],
    k: int,
    hard_w: float = 1.2,
) -> list[dict]:
    """按 signal_date 取每日 TopK（用于事后过滤）；持仓去重逻辑已在原 trades 中。"""
    if k <= 0 or not trades:
        return trades
    df = pd.DataFrame(trades)
    if "signal_date" not in df.columns:
        return trades
    df["_rank"] = df["hard"].fillna(0) * hard_w + df["soft"].fillna(0)
    keep = (
        df.sort_values(["signal_date", "_rank"], ascending=[True, False])
        .groupby("signal_date", group_keys=False)
        .head(k)
    )
    # 保持原时间顺序
    keep = keep.drop(columns=["_rank"], errors="ignore")
    return keep.to_dict("records")


def backtest_universe_topk(
    codes_dfs: dict[str, pd.DataFrame],
    bands: dict,
    entry_mode: str,
    hold_n: int,
    targets: list[float],
    stop: float,
    cost: float,
    top_k: int,
) -> dict[float, list[dict]]:
    """全市场按信号日 TopK 选股，再模拟出场（每股持仓不重叠）。"""
    hard_w = float(bands.get("rank_hard_weight", 1.2))
    # 1) 收集全部信号
    all_sigs: list[dict] = []
    arrays: dict[str, dict] = {}
    for code, df in codes_dfs.items():
        start = max(60, len(df) - LOOKBACK)
        end = len(df) - hold_n - 2
        if end <= start:
            continue
        table = precompute_feature_table(df)
        op = df["op"].to_numpy(dtype=float)
        hi = df["hi"].to_numpy(dtype=float)
        lo = df["lo"].to_numpy(dtype=float)
        px = df["px"].to_numpy(dtype=float)
        ma5 = df["ma5"].to_numpy(dtype=float)
        ma10 = df["ma10"].to_numpy(dtype=float)
        ma20 = df["ma20"].to_numpy(dtype=float)
        dates = df["日期"]
        arrays[code] = {
            "op": op,
            "hi": hi,
            "lo": lo,
            "px": px,
            "dates": dates,
        }
        for i in range(start, end + 1):
            feat = feat_at_table(table, i, float(ma5[i]), float(ma10[i]), float(ma20[i]))
            if feat is None:
                continue
            soft, hard, _ = score_row(feat, bands)
            if not is_signal(feat, soft, hard, bands):
                continue
            all_sigs.append(
                {
                    "code": code,
                    "i": i,
                    "signal_date": str(dates.iloc[i].date()),
                    "soft": soft,
                    "hard": hard,
                    "rank": hard * hard_w + soft,
                }
            )
    if not all_sigs:
        return {t: [] for t in targets}

    sig_df = pd.DataFrame(all_sigs)
    picked = (
        sig_df.sort_values(["signal_date", "rank"], ascending=[True, False])
        .groupby("signal_date", group_keys=False)
        .head(top_k)
        .sort_values(["signal_date", "rank"], ascending=[True, False])
    )

    out: dict[float, list[dict]] = {t: [] for t in targets}
    next_free = {c: 0 for c in arrays}
    primary = targets[0]
    for _, row in picked.iterrows():
        code = row["code"]
        i = int(row["i"])
        if i < next_free[code]:
            continue
        arr = arrays[code]
        primary_exit = i + 1
        taken = False
        for t in targets:
            tr = simulate_trade_arr(
                arr["op"],
                arr["hi"],
                arr["lo"],
                arr["px"],
                arr["dates"],
                i,
                entry_mode,
                hold_n,
                t,
                stop,
                cost,
            )
            if tr is None:
                continue
            if tr.get("skipped"):
                primary_exit = max(primary_exit, int(tr["exit_i"]))
                continue
            tr["code"] = code
            tr["soft"] = round(float(row["soft"]), 3)
            tr["hard"] = round(float(row["hard"]), 3)
            out[t].append(tr)
            taken = True
            if t == primary:
                primary_exit = int(tr["exit_i"])
        if taken or primary_exit > i + 1:
            next_free[code] = primary_exit + 1
    return out


def backtest_stock(
    code: str,
    df: pd.DataFrame,
    bands: dict,
    entry_mode: str,
    hold_n: int,
    targets: list[float],
    stop: float,
    cost: float,
) -> dict[float, list[dict]]:
    """返回 {target: trades}。信号间隔按第一个 target（通常 0.08）的出场推进，保证多 target 同入场集。"""
    out: dict[float, list[dict]] = {t: [] for t in targets}
    start = max(60, len(df) - LOOKBACK)
    end = len(df) - hold_n - 2
    if end <= start:
        return out

    table = precompute_feature_table(df)
    op = df["op"].to_numpy(dtype=float)
    hi = df["hi"].to_numpy(dtype=float)
    lo = df["lo"].to_numpy(dtype=float)
    px = df["px"].to_numpy(dtype=float)
    ma5 = df["ma5"].to_numpy(dtype=float)
    ma10 = df["ma10"].to_numpy(dtype=float)
    ma20 = df["ma20"].to_numpy(dtype=float)
    dates = df["日期"]
    primary = targets[0]

    i = start
    while i <= end:
        feat = feat_at_table(table, i, float(ma5[i]), float(ma10[i]), float(ma20[i]))
        if feat is None:
            i += 1
            continue
        soft, hard, _ = score_row(feat, bands)
        if not is_signal(feat, soft, hard, bands):
            i += 1
            continue

        primary_exit = i + 1
        for t in targets:
            tr = simulate_trade_arr(
                op, hi, lo, px, dates, i, entry_mode, hold_n, t, stop, cost
            )
            if tr is None:
                continue
            if tr.get("skipped"):
                primary_exit = max(primary_exit, int(tr["exit_i"]))
                continue
            tr["code"] = code
            tr["soft"] = round(soft, 3)
            tr["hard"] = round(hard, 3)
            out[t].append(tr)
            if t == primary:
                primary_exit = int(tr["exit_i"])
        i = primary_exit + 1
    return out


def summarize(trades: pd.DataFrame, label: str) -> str:
    if trades.empty:
        return f"[{label}] 无成交"
    r = trades["net_ret%"]
    win = r > 0
    reasons = trades["exit_reason"].value_counts().to_dict()
    equity = (1.0 + r / 100.0).cumprod()
    peak = equity.cummax()
    dd = (equity / peak - 1.0).min() * 100.0
    trades = trades.copy()
    trades["year"] = pd.to_datetime(trades["entry_date"]).dt.year
    by_year = trades.groupby("year").agg(
        n=("net_ret%", "count"),
        winrate=("net_ret%", lambda s: float((s > 0).mean() * 100)),
        avg=("net_ret%", "mean"),
        med=("net_ret%", "median"),
        sum=("net_ret%", "sum"),
    )

    lines = [
        f"=== {label} ===",
        f"成交笔数: {len(trades)}",
        f"覆盖股票: {trades['code'].nunique()}",
        f"胜率: {win.mean()*100:.1f}%",
        f"平均净收益: {r.mean():.3f}%",
        f"中位净收益: {r.median():.3f}%",
        f"盈亏比(均盈/|均亏|): "
        f"{(r[win].mean() / abs(r[~win].mean())) if (~win).any() and win.any() else float('nan'):.2f}",
        f"平均持有天数: {trades['hold_days'].mean():.2f}",
        f"止盈占比: {reasons.get('止盈', 0)/len(trades)*100:.1f}%",
        f"止损占比: {(reasons.get('止损', 0)+reasons.get('同日止损优先', 0))/len(trades)*100:.1f}%",
        f"到期占比: {reasons.get('到期', 0)/len(trades)*100:.1f}%",
        f"单笔接力权益总回报: {(equity.iloc[-1]-1)*100:.1f}%",
        f"单笔接力最大回撤: {dd:.1f}%",
        f"期望值(每笔): {r.mean():.3f}%",
        "",
        "出场原因:",
        str(pd.Series(reasons)),
        "",
        "分年:",
        by_year.round(2).to_string(),
    ]
    return "\n".join(lines)


def _write_result(
    trades: list[dict],
    bands_path: Path,
    target: float,
    entry: str,
    hold: int,
    stop: float,
    cost_bps: float,
    bands: dict,
    name_map: dict,
    ind_map: dict,
    top_k: int = 0,
) -> None:
    tag = bands_path.stem.replace("short_burst_feature_bands", "").strip("_") or "default"
    tp_tag = f"tp{int(round(target * 100))}"
    if top_k > 0:
        tp_tag = f"top{top_k}_{tp_tag}"
    out_trades = ROOT / "data" / f"short_burst_backtest_trades_{tag}_{tp_tag}.csv"
    out_summary = ROOT / "data" / f"short_burst_backtest_summary_{tag}_{tp_tag}.txt"

    rows = []
    for t in trades:
        t = dict(t)
        code = t["code"]
        t["name"] = name_map.get(code, "")
        t["industry"] = ind_map.get(code, "")
        rows.append(t)
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("entry_date").reset_index(drop=True)
        drop_cols = [c for c in ("buy_i", "exit_i", "skipped", "target") if c in out.columns]
        out = out.drop(columns=drop_cols, errors="ignore")

    out_trades.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_trades, index=False, encoding="utf-8-sig")

    sh = bands.get("signal_hard", 0.85)
    ss = bands.get("signal_soft", bands.get("min_score", 0.72))
    top_s = f", daily_top_k={top_k}" if top_k > 0 else ""
    label = (
        f"bands={bands_path.name}{top_s}, signal≥hard{sh}/soft{ss}, "
        f"entry={entry}, hold={hold}, "
        f"tp=+{target*100:.0f}%, sl=-{stop*100:.0f}%, cost={cost_bps:.0f}bp"
    )
    text = summarize(out, label)
    if not out.empty:
        daily_n = out.groupby("entry_date").size()
        text += (
            f"\n\n同日新开仓笔数 中位/P90: {daily_n.median():.0f} / {daily_n.quantile(0.9):.0f}"
            f"\n说明: 「单笔接力权益」是假设每次只做一笔串起来；实盘若并行多票，回撤与复合会不同。"
        )
    out_summary.write_text(text, encoding="utf-8")
    print(text, flush=True)
    print(f"\n→ {out_trades}\n→ {out_summary}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="短线爆发回测")
    parser.add_argument("--entry", choices=["open", "close"], default=None)
    parser.add_argument("--hold", type=int, default=None)
    parser.add_argument("--target", type=float, default=None, help="单个止盈；若设则覆盖 --targets")
    parser.add_argument(
        "--targets",
        type=str,
        default="",
        help="逗号分隔多个止盈，如 0.15,0.08；默认读 bands.exit 或 +15%",
    )
    parser.add_argument("--stop", type=float, default=None)
    parser.add_argument("--cost-bps", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None, help="每日信号 TopK；默认读 bands.daily_top_k")
    parser.add_argument(
        "--bands",
        type=str,
        default=str(BANDS),
        help="特征区间 JSON",
    )
    args = parser.parse_args()

    bands_path = Path(args.bands)
    bands = load_bands(bands_path)
    exit_cfg = bands.get("exit") or {}

    entry = args.entry or exit_cfg.get("entry") or "open"
    hold = int(args.hold if args.hold is not None else exit_cfg.get("hold", HOLD_N))
    stop = float(args.stop if args.stop is not None else exit_cfg.get("stop", STOP))
    cost_bps = float(
        args.cost_bps if args.cost_bps is not None else exit_cfg.get("cost_bps", COST_BPS)
    )
    top_k = args.top_k if args.top_k is not None else int(bands.get("daily_top_k") or 0)

    if args.target is not None:
        targets = [float(args.target)]
    elif args.targets.strip():
        targets = [float(x) for x in args.targets.split(",") if x.strip()]
    elif "target" in exit_cfg:
        targets = [float(exit_cfg["target"])]
    else:
        targets = [TARGET]
    if not targets:
        raise SystemExit("需要至少一个 target")

    name_map, ind_map = load_maps()
    cost = cost_bps / 10000.0

    import daily_cache

    t0 = time.time()
    daily_cache.preload(force=False, prefer_process=True)
    print(f"[cache] 预载完成 {time.time()-t0:.1f}s", flush=True)

    codes = [
        c
        for c in daily_cache.cached_codes()
        if c.startswith(("600", "601", "603", "605", "000", "001", "002"))
        and "ST" not in name_map.get(c, "").upper()
    ]
    dfs = {c: daily_cache.get(c) for c in codes}
    dfs = {c: df for c, df in dfs.items() if df is not None}

    t1 = time.time()
    if top_k > 0:
        print(f"模式: 每日 Top{top_k} + 出场 tp={targets} sl={stop} hold={hold}", flush=True)
        bucket = backtest_universe_topk(
            dfs, bands, entry, hold, targets, stop, cost, top_k
        )
    else:
        bucket = {t: [] for t in targets}
        n = 0
        for code, df in dfs.items():
            n += 1
            by_t = backtest_stock(code, df, bands, entry, hold, targets, stop, cost)
            for t, trades in by_t.items():
                bucket[t].extend(trades)
            if n % 500 == 0:
                print(
                    f"…已扫 {n}/{len(dfs)}，主target成交 {len(bucket[targets[0]])} "
                    f"({time.time()-t1:.0f}s)",
                    flush=True,
                )

    print(f"扫描完成 {len(dfs)} 只，耗时 {time.time()-t1:.1f}s", flush=True)
    for t in targets:
        _write_result(
            bucket[t],
            bands_path,
            t,
            entry,
            hold,
            stop,
            cost_bps,
            bands,
            name_map,
            ind_map,
            top_k=top_k,
        )


if __name__ == "__main__":
    main()
