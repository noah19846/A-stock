"""三连小阳策略回测。

入场：信号日尾盘（收盘价）买入
止损：-3%（持有期内最低价触及则按止损价出；若低开跌破止损则按开盘出）
止盈：无
持有：满 HOLD_DAYS 个交易日收盘卖（默认 5 ≈ 一周）
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_short_burst_features import load_maps
from three_yang_screener import (
    BASE_ABS_RET_MAX,
    BASE_DAYS,
    BASE_DIST60_LOW_MAX,
    BASE_RANGE_MAX,
    MAINBOARD_PREFIX,
    match_at,
)

ROOT = Path(__file__).resolve().parent
WARMUP = "2024-01-01"
SIGNAL_START = "2024-06-01"
HOLD_DAYS = 5
STOP = 0.03
COST_RT = 0.001  # 双边合计约 10bp（印花+佣金粗算）


def summarize(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {"n": 0}
    win = x > 0
    loss = x < 0
    return {
        "n": int(len(x)),
        "mean": round(float(np.mean(x)) * 100, 2),
        "med": round(float(np.median(x)) * 100, 2),
        "p10": round(float(np.percentile(x, 10)) * 100, 2),
        "p25": round(float(np.percentile(x, 25)) * 100, 2),
        "p75": round(float(np.percentile(x, 75)) * 100, 2),
        "p90": round(float(np.percentile(x, 90)) * 100, 2),
        "min": round(float(np.min(x)) * 100, 2),
        "max": round(float(np.max(x)) * 100, 2),
        "win": round(float(win.mean()) * 100, 1),
        "loss": round(float(loss.mean()) * 100, 1),
        "avg_win": round(float(x[win].mean()) * 100, 2) if win.any() else None,
        "avg_loss": round(float(x[loss].mean()) * 100, 2) if loss.any() else None,
        "le_stop": round(float((x <= -STOP + 1e-9).mean()) * 100, 1),
        "ge3": round(float((x >= 0.03).mean()) * 100, 1),
        "ge5": round(float((x >= 0.05).mean()) * 100, 1),
        "ge10": round(float((x >= 0.10).mean()) * 100, 1),
    }


def simulate_trade(
    entry: float,
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
) -> tuple[float, str, int]:
    """持有 opens/highs/lows/closes 为入场后连续 HOLD_DAYS 根。

    返回 (毛收益, 退出原因, 持有交易日数)。
    """
    stop_px = entry * (1.0 - STOP)
    n = len(closes)
    for j in range(n):
        op = float(opens[j])
        lo = float(lows[j])
        if not np.isfinite(op) or not np.isfinite(lo):
            continue
        # 低开直接砸破止损
        if op <= stop_px:
            return op / entry - 1.0, "stop_gap", j + 1
        if lo <= stop_px:
            return stop_px / entry - 1.0, "stop", j + 1
    last = float(closes[-1])
    return last / entry - 1.0, "time", n


def scan_code(
    code: str,
    g: pd.DataFrame,
    name: str,
    *,
    require_base: bool = True,
) -> list[dict]:
    if "ST" in name.upper() or name.startswith("*"):
        return []
    n = len(g)
    if n < 80:
        return []
    close = g["close"].to_numpy(dtype=np.float64)
    high = g["high"].to_numpy(dtype=np.float64)
    low = g["low"].to_numpy(dtype=np.float64)
    open_ = g["open"].to_numpy(dtype=np.float64)
    vol = g["volume"].to_numpy(dtype=np.float64)
    fac = g["hfq_factor"].to_numpy(dtype=np.float64)
    dates = pd.to_datetime(g["trade_date"].to_numpy())
    px = close * fac
    hi = high * fac
    lo = low * fac
    op = open_ * fac
    ret = np.full(n, np.nan)
    ret[1:] = px[1:] / px[:-1] - 1.0

    start_ts = pd.Timestamp(SIGNAL_START)
    rows: list[dict] = []
    for i in range(3, n):
        d = dates[i]
        if d < start_ts:
            continue
        hit = match_at(
            px, op, vol, ret, i, hi=hi, lo=lo, require_base=require_base
        )
        if hit is None:
            continue
        # 需要完整持有窗口
        if i + HOLD_DAYS >= n:
            continue
        entry = float(px[i])
        if entry <= 0:
            continue
        gross, reason, held = simulate_trade(
            entry,
            op[i + 1 : i + 1 + HOLD_DAYS],
            hi[i + 1 : i + 1 + HOLD_DAYS],
            lo[i + 1 : i + 1 + HOLD_DAYS],
            px[i + 1 : i + 1 + HOLD_DAYS],
        )
        net = (1.0 + gross) * (1.0 - COST_RT) - 1.0
        # 持有期 MFE/MAE（相对入场）
        path_hi = hi[i + 1 : i + 1 + HOLD_DAYS]
        path_lo = lo[i + 1 : i + 1 + HOLD_DAYS]
        mfe = float(np.nanmax(path_hi) / entry - 1.0) if len(path_hi) else np.nan
        mae = float(np.nanmin(path_lo) / entry - 1.0) if len(path_lo) else np.nan
        rows.append(
            {
                "date": str(d.date()),
                "code": code,
                "name": name,
                "cum3": round(hit.cum3 * 100, 2),
                "ret1": round(hit.ret1 * 100, 2),
                "ret2": round(hit.ret2 * 100, 2),
                "ret3": round(hit.ret3 * 100, 2),
                "vr1": round(hit.vr1, 2),
                "vr2": round(hit.vr2, 2),
                "vr3": round(hit.vr3, 2),
                "vol_growth": round(hit.vol_growth, 2),
                "base_range": None
                if hit.base_range is None
                else round(hit.base_range * 100, 2),
                "base_ret": None if hit.base_ret is None else round(hit.base_ret * 100, 2),
                "dist60_low": None
                if hit.dist60_low is None
                else round(hit.dist60_low * 100, 2),
                "ret": float(gross),
                "ret_net": float(net),
                "exit": reason,
                "held": held,
                "mfe": float(mfe),
                "mae": float(mae),
                "exit_date": str(dates[i + held].date()),
            }
        )
    return rows


def pack(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"n": 0}
    out: dict = {
        "n_trades": int(len(df)),
        "n_days": int(df["date"].nunique()),
        "n_codes": int(df["code"].nunique()),
        "ret_gross": summarize(df["ret"].to_numpy()),
        "ret_net": summarize(df["ret_net"].to_numpy()),
        "mfe": summarize(df["mfe"].to_numpy()),
        "mae": summarize(df["mae"].to_numpy()),
        "exit_mix": {
            k: int(v)
            for k, v in df["exit"].value_counts().to_dict().items()
        },
        "held_mean": round(float(df["held"].mean()), 2),
    }
    # 等权按信号日：当日多票取均值，再看日胜率 / 复利（示意，非仓位约束）
    g = df.groupby("date")["ret_net"].mean()
    out["eq_day_mean"] = round(float(g.mean()) * 100, 2)
    out["eq_day_win"] = round(float((g > 0).mean()) * 100, 1)
    out["eq_day_n"] = int(len(g))
    out["eq_compound"] = round(float(np.prod(1.0 + g.to_numpy()) - 1.0) * 100, 1)

    by_year = {}
    tmp = df.copy()
    tmp["year"] = tmp["date"].str[:4]
    for y, gy in tmp.groupby("year"):
        by_year[y] = {
            "n": int(len(gy)),
            "ret_net": summarize(gy["ret_net"].to_numpy()),
            "exit_mix": {k: int(v) for k, v in gy["exit"].value_counts().to_dict().items()},
        }
    out["by_year"] = by_year

    tmp["ym"] = tmp["date"].str[:7]
    monthly = (
        tmp.groupby("ym")
        .agg(
            n=("ret_net", "size"),
            mean=("ret_net", "mean"),
            win=("ret_net", lambda s: float((s > 0).mean())),
            stop=("exit", lambda s: float((s.isin(["stop", "stop_gap"])).mean())),
        )
        .reset_index()
    )
    out["monthly"] = [
        {
            "ym": r.ym,
            "n": int(r.n),
            "mean": round(float(r.mean) * 100, 2),
            "win": round(float(r.win) * 100, 1),
            "stop_rate": round(float(r.stop) * 100, 1),
        }
        for r in monthly.itertuples()
    ]
    # 最好 / 最差若干笔
    top = df.nlargest(8, "ret_net")[
        ["date", "code", "name", "ret_net", "exit", "held", "cum3"]
    ]
    bot = df.nsmallest(8, "ret_net")[
        ["date", "code", "name", "ret_net", "exit", "held", "cum3"]
    ]
    out["best"] = [
        {
            "date": r.date,
            "code": r.code,
            "name": r.name,
            "ret": round(float(r.ret_net) * 100, 2),
            "exit": r.exit,
            "held": int(r.held),
            "cum3": r.cum3,
        }
        for r in top.itertuples()
    ]
    out["worst"] = [
        {
            "date": r.date,
            "code": r.code,
            "name": r.name,
            "ret": round(float(r.ret_net) * 100, 2),
            "exit": r.exit,
            "held": int(r.held),
            "cum3": r.cum3,
        }
        for r in bot.itertuples()
    ]
    return out


def main() -> None:
    import argparse

    import daily_db

    ap = argparse.ArgumentParser(description="三连小阳回测")
    ap.add_argument(
        "--no-base",
        action="store_true",
        help="关闭底部横盘过滤（对照旧版）",
    )
    args = ap.parse_args()
    require_base = not args.no_base

    name_map, _ = load_maps()
    print("loading bars …", flush=True)
    raw = daily_db.load_recent_bars_since(WARMUP, prefixes=MAINBOARD_PREFIX)
    raw["code"] = raw["code"].astype(str).str.zfill(6)
    print(f"rows {len(raw)} codes {raw['code'].nunique()}", flush=True)
    print(
        f"require_base={require_base} "
        f"(days≥{BASE_DAYS} range≤{BASE_RANGE_MAX*100:.0f}% "
        f"|ret|≤{BASE_ABS_RET_MAX*100:.0f}% dist60low≤{BASE_DIST60_LOW_MAX*100:.0f}%)",
        flush=True,
    )

    trades: list[dict] = []
    for code, g in raw.groupby("code", sort=False):
        g = g.sort_values("trade_date")
        trades.extend(
            scan_code(str(code), g, name_map.get(str(code), ""), require_base=require_base)
        )

    df = pd.DataFrame(trades)
    print(f"trades {len(df)}", flush=True)
    if df.empty:
        print("{}")
        return

    payload = {
        "rules": {
            "entry": "信号日尾盘（收盘）买入",
            "stop": f"-{STOP * 100:.0f}%（盘中最低触及；低开破止损按开盘）",
            "take_profit": "无",
            "hold_days": HOLD_DAYS,
            "cost_roundtrip": COST_RT,
            "pattern": "三连小阳+每日量比1.05~2.0+整体放量",
            "base_filter": {
                "enabled": require_base,
                "days": BASE_DAYS,
                "range_max_pct": BASE_RANGE_MAX * 100,
                "abs_ret_max_pct": BASE_ABS_RET_MAX * 100,
                "dist60_low_max_pct": BASE_DIST60_LOW_MAX * 100,
            },
            "universe": "主板（含中小板）非 ST",
            "signal_start": SIGNAL_START,
        },
        "summary": pack(df),
    }
    tag = "base" if require_base else "nobase"
    out_json = ROOT / "data" / f"three_yang_backtest_{tag}.json"
    out_csv = ROOT / "data" / f"three_yang_trades_{tag}.csv"
    # 默认路径兼容：带底部过滤时也写主文件名
    if require_base:
        (ROOT / "data" / "three_yang_backtest.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        df.to_csv(ROOT / "data" / "three_yang_trades.csv", index=False, encoding="utf-8-sig")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(json.dumps(payload["summary"]["ret_net"], ensure_ascii=False, indent=2))
    print("exit_mix", payload["summary"].get("exit_mix"))
    print(f"n_trades={payload['summary']['n_trades']} eq_compound={payload['summary'].get('eq_compound')}")
    print(f"→ {out_json}", flush=True)
    print(f"→ {out_csv}", flush=True)


if __name__ == "__main__":
    main()
