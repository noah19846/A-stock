"""无量首板：近两年尾盘买 → 次日收盘卖 / 持有 5 个交易日。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_short_burst_features import load_maps
from quiet_limit_screener import (
    DIST60_BUY,
    DIST60_WATCH,
    LIMIT,
    MV_HI,
    MV_LO,
    PX_MA20_MAX,
    R20_MAX,
    TURN_QUIET,
    VOL_QUIET,
)
from volume_buy_screener import MAINBOARD_PREFIX, rolling_mean

ROOT = Path(__file__).resolve().parent
WARMUP = "2024-04-01"
SIGNAL_START = "2024-08-19"


def roll_max(x: np.ndarray, w: int) -> np.ndarray:
    return pd.Series(x).rolling(w, min_periods=w).max().to_numpy()


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
        "ge2": round(float((x >= 0.02).mean()) * 100, 1),
        "le2": round(float((x <= -0.02).mean()) * 100, 1),
        "le5": round(float((x <= -0.05).mean()) * 100, 1),
        "ge5": round(float((x >= 0.05).mean()) * 100, 1),
        "ge95": round(float((x >= 0.095).mean()) * 100, 1),
    }


def scan_code(code: str, g: pd.DataFrame, name: str) -> list[dict]:
    if "ST" in name.upper() or name.startswith("*"):
        return []
    n = len(g)
    if n < 90:
        return []
    close = g["close"].to_numpy(dtype=np.float64)
    high = g["high"].to_numpy(dtype=np.float64)
    low = g["low"].to_numpy(dtype=np.float64)
    open_ = g["open"].to_numpy(dtype=np.float64)
    vol = g["volume"].to_numpy(dtype=np.float64)
    turn = g["turnover"].to_numpy(dtype=np.float64) * 100.0
    fac = g["hfq_factor"].to_numpy(dtype=np.float64)
    fs = g["float_shares"].to_numpy(dtype=np.float64)
    dates = pd.to_datetime(g["trade_date"].to_numpy())
    px = close * fac
    hi = high * fac
    lo = low * fac
    op = open_ * fac
    ret = np.empty(n, dtype=np.float64)
    ret[0] = np.nan
    ret[1:] = px[1:] / px[:-1] - 1.0
    mv = fs * close / 1e8
    vol_ma20 = rolling_mean(vol, 20)
    turn20 = pd.Series(turn).rolling(20, min_periods=20).mean().shift(1).to_numpy()
    hi60 = roll_max(hi, 60)
    ma20 = rolling_mean(px, 20)
    is_lim = ret >= LIMIT
    n_limit10 = (
        pd.Series(is_lim.astype(np.float64)).rolling(10, min_periods=10).sum().shift(1).to_numpy()
    )
    r20 = np.full(n, np.nan)
    r20[20:] = px[20:] / px[:-20] - 1.0

    start_ts = pd.Timestamp(SIGNAL_START)
    rows = []
    for i in range(60, n):
        d = dates[i]
        if d < start_ts:
            continue
        if not np.isfinite(ret[i]) or ret[i] < LIMIT:
            continue
        if vol[i] <= 0:
            continue
        if not np.isfinite(mv[i]) or mv[i] < MV_LO or mv[i] > MV_HI:
            continue
        if i < 1 or is_lim[i - 1]:
            continue
        if not np.isfinite(n_limit10[i]) or n_limit10[i] != 0:
            continue
        v20 = vol_ma20[i]
        vol_ratio = vol[i] / v20 if np.isfinite(v20) and v20 > 0 else np.nan
        t20 = turn20[i]
        turn_ratio = (
            turn[i] / t20 if np.isfinite(turn[i]) and np.isfinite(t20) and t20 > 0 else np.nan
        )
        quiet = (np.isfinite(vol_ratio) and vol_ratio < VOL_QUIET) or (
            np.isfinite(turn_ratio) and turn_ratio < TURN_QUIET
        )
        if not quiet:
            continue
        if not np.isfinite(hi60[i]) or hi60[i] <= 0:
            continue
        dist60h = px[i] / hi60[i] - 1.0
        if dist60h > DIST60_WATCH:
            continue
        px_ma20 = px[i] / ma20[i] if np.isfinite(ma20[i]) and ma20[i] > 0 else np.nan
        low_pos = dist60h <= DIST60_BUY
        not_hot20 = (not np.isfinite(r20[i])) or r20[i] <= R20_MAX
        not_stretched = (not np.isfinite(px_ma20)) or px_ma20 <= PX_MA20_MAX
        very_quiet = np.isfinite(vol_ratio) and vol_ratio < VOL_QUIET
        if low_pos and not_hot20 and not_stretched and very_quiet:
            stage = "可买入"
        elif low_pos or dist60h <= DIST60_WATCH:
            stage = "观察"
        else:
            continue
        rec = {
            "date": str(d.date()),
            "code": code,
            "name": name,
            "stage": stage,
            "vol_ratio": None if not np.isfinite(vol_ratio) else round(float(vol_ratio), 2),
            "dist60h": round(float(dist60h) * 100, 1),
        }
        if i + 1 < n:
            rec["r1"] = float(px[i + 1] / px[i] - 1.0)
            rec["hi1"] = float(hi[i + 1] / px[i] - 1.0)
            rec["lo1"] = float(lo[i + 1] / px[i] - 1.0)
            rec["op1"] = float(op[i + 1] / px[i] - 1.0)
        if i + 5 < n:
            rec["r5"] = float(px[i + 5] / px[i] - 1.0)
            rec["mfe5"] = float(np.max(hi[i + 1 : i + 6]) / px[i] - 1.0)
            rec["mae5"] = float(np.min(lo[i + 1 : i + 6]) / px[i] - 1.0)
        rows.append(rec)
    return rows


def main() -> None:
    import daily_db

    name_map, _ = load_maps()
    print("loading bars …", flush=True)
    raw = daily_db.load_recent_bars_since(WARMUP, prefixes=MAINBOARD_PREFIX)
    raw["code"] = raw["code"].astype(str).str.zfill(6)
    print(f"rows {len(raw)} codes {raw['code'].nunique()}", flush=True)

    trades: list[dict] = []
    for code, g in raw.groupby("code", sort=False):
        g = g.sort_values("trade_date")
        trades.extend(scan_code(code, g, name_map.get(code, "")))
    df = pd.DataFrame(trades)
    print(f"signals {len(df)}", flush=True)
    if df.empty:
        print("{}")
        return

    buy = df[df["stage"] == "可买入"].copy()
    watch = df[df["stage"] == "观察"].copy()

    def pack(sub: pd.DataFrame, label: str) -> dict:
        d1 = sub[sub["r1"].notna()]
        d5 = sub[sub["r5"].notna()]
        out = {
            "label": label,
            "n_sig": int(len(sub)),
            "n_days": int(sub["date"].nunique()),
            "r1": summarize(d1["r1"].to_numpy()) if len(d1) else {"n": 0},
            "hi1": summarize(d1["hi1"].to_numpy()) if len(d1) else {"n": 0},
            "lo1": summarize(d1["lo1"].to_numpy()) if len(d1) else {"n": 0},
            "op1": summarize(d1["op1"].to_numpy()) if len(d1) else {"n": 0},
            "r5": summarize(d5["r5"].to_numpy()) if len(d5) else {"n": 0},
            "mfe5": summarize(d5["mfe5"].to_numpy()) if len(d5) else {"n": 0},
            "mae5": summarize(d5["mae5"].to_numpy()) if len(d5) else {"n": 0},
        }
        if len(d1):
            g = d1.groupby("date")["r1"].mean()
            out["r1_eq_day"] = round(float(g.mean()) * 100, 2)
            out["r1_day_win"] = round(float((g > 0).mean()) * 100, 1)
            out["r1_n_days"] = int(len(g))
            cum = float(np.prod(1.0 + g.to_numpy()) - 1.0)
            out["r1_eq_compound"] = round(cum * 100, 1)
        if len(d5):
            g5 = d5.groupby("date")["r5"].mean()
            out["r5_eq_day"] = round(float(g5.mean()) * 100, 2)
            out["r5_day_win"] = round(float((g5 > 0).mean()) * 100, 1)
            out["r5_n_days"] = int(len(g5))
        # year split
        years = {}
        for y, gy in d1.groupby(d1["date"].str[:4]):
            years[y] = {
                "r1": summarize(gy["r1"].to_numpy()),
                "r5": summarize(sub.loc[gy.index.intersection(d5.index), "r5"].to_numpy())
                if len(d5)
                else {"n": 0},
            }
        # fix year r5
        years = {}
        d1 = d1.copy()
        d1["year"] = d1["date"].str[:4]
        d5c = d5.copy()
        d5c["year"] = d5c["date"].str[:4]
        for y in sorted(set(d1["year"]) | set(d5c["year"])):
            years[y] = {
                "r1": summarize(d1.loc[d1["year"] == y, "r1"].to_numpy()),
                "r5": summarize(d5c.loc[d5c["year"] == y, "r5"].to_numpy()),
            }
        out["by_year"] = years

        # monthly EW r1
        d1["ym"] = d1["date"].str[:7]
        monthly = (
            d1.groupby("ym")
            .agg(n=("r1", "size"), r1=("r1", "mean"), win=("r1", lambda s: float((s > 0).mean())))
            .reset_index()
        )
        out["monthly"] = [
            {
                "ym": r.ym,
                "n": int(r.n),
                "r1": round(float(r.r1) * 100, 2),
                "win": round(float(r.win) * 100, 1),
            }
            for r in monthly.itertuples()
        ]
        d5c["ym"] = d5c["date"].str[:7]
        monthly5 = (
            d5c.groupby("ym")
            .agg(n=("r5", "size"), r5=("r5", "mean"), win=("r5", lambda s: float((s > 0).mean())))
            .reset_index()
        )
        out["monthly5"] = [
            {
                "ym": r.ym,
                "n": int(r.n),
                "r5": round(float(r.r5) * 100, 2),
                "win": round(float(r.win) * 100, 1),
            }
            for r in monthly5.itertuples()
        ]
        return out

    payload = {
        "range": f"{SIGNAL_START} → 有次日的最晚选股日",
        "buy": pack(buy, "续板候选"),
        "watch": pack(watch, "观察"),
    }
    out_path = ROOT / "data" / "quiet_limit_backtest.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"→ {out_path}", flush=True)


if __name__ == "__main__":
    main()
