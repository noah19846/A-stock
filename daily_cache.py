"""
主板日线共享缓存：长/短线筛选只读盘一遍。

- 若存在 data/db/daily.db：从 SQLite 批量读（推荐）
- 否则：多进程并行读 CSV
- 一次性算好后复权/均线；只保留近 MAX_BARS 根
- get(code, asof=...) 内存切片，不再读盘
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
DAILY = ROOT / "data" / "daily_raw"
DB_PATH = ROOT / "data" / "db" / "daily.db"

USECOLS = [
    "日期",
    "开盘",
    "收盘",
    "最高",
    "最低",
    "成交量",
    "成交额",
    "流通股本",
    "换手率",
    "后复权因子",
]
MAX_BARS = 320  # 交易日；覆盖 ma60 + Setup lookback
MAINBOARD_PREFIX = ("600", "601", "603", "605", "000", "001", "002")

_CACHE: dict[str, pd.DataFrame | None] = {}
_PRELOADED = False
_SOURCE: str = ""  # "sqlite" | "csv" | ""


def clear() -> None:
    global _PRELOADED, _SOURCE
    _CACHE.clear()
    _PRELOADED = False
    _SOURCE = ""


def use_sqlite() -> bool:
    return DB_PATH.exists() and DB_PATH.stat().st_size > 0


def _prepare(df: pd.DataFrame) -> pd.DataFrame | None:
    df["日期"] = pd.to_datetime(df["日期"])
    df = df.sort_values("日期", kind="mergesort")
    if len(df) > MAX_BARS:
        df = df.iloc[-MAX_BARS:]
    df = df.reset_index(drop=True)
    if len(df) < 80:
        return None
    f = df["后复权因子"].to_numpy(dtype=float, copy=False)
    close = df["收盘"].to_numpy(dtype=float, copy=False)
    df["px"] = close * f
    df["op"] = df["开盘"].to_numpy(dtype=float, copy=False) * f
    df["hi"] = df["最高"].to_numpy(dtype=float, copy=False) * f
    df["lo"] = df["最低"].to_numpy(dtype=float, copy=False) * f
    df["amt"] = df["成交额"].astype(float)
    df["vol"] = df["成交量"].astype(float)
    df["turn"] = df["换手率"].astype(float) * 100.0
    df["mv"] = df["流通股本"].astype(float) * close / 1e8
    px = df["px"]
    df["ret"] = px.pct_change()
    df["ma5"] = px.rolling(5).mean()
    df["ma10"] = px.rolling(10).mean()
    df["ma20"] = px.rolling(20).mean()
    df["ma60"] = px.rolling(60).mean()
    return df


def _load_one(path: str) -> tuple[str, pd.DataFrame | None]:
    """path 用 str，便于 ProcessPool 序列化。"""
    p = Path(path)
    code = p.stem.zfill(6)
    try:
        df = pd.read_csv(p, usecols=USECOLS)
        return code, _prepare(df)
    except Exception:
        return code, None


def list_mainboard_paths() -> list[Path]:
    out: list[Path] = []
    for p in DAILY.glob("*.csv"):
        if not p.name[0].isdigit():
            continue
        code = p.stem.zfill(6)
        if code.startswith(MAINBOARD_PREFIX):
            out.append(p)
    return sorted(out)


def list_mainboard_codes_sqlite() -> list[str]:
    import daily_db

    return [
        c
        for c in daily_db.list_codes()
        if c.startswith(MAINBOARD_PREFIX)
    ]


def _preload_sqlite() -> int:
    """单连接批量读 SQLite（比几千次 open CSV 快）。"""
    import daily_db

    codes = list_mainboard_codes_sqlite()
    conn = daily_db.connect()
    loaded = 0
    try:
        for i, code in enumerate(codes, 1):
            try:
                df = daily_db.load_bars_df(code, limit=MAX_BARS, conn=conn)
                prepared = _prepare(df) if not df.empty else None
            except Exception:
                prepared = None
            _CACHE[code] = prepared
            if prepared is not None:
                loaded += 1
            if i % 500 == 0:
                print(f"  sqlite preload {i}/{len(codes)}", flush=True)
    finally:
        conn.close()
    return loaded


def _preload_csv(workers: int | None, prefer_process: bool) -> int:
    paths = [str(p) for p in list_mainboard_paths()]
    cpu = os.cpu_count() or 4
    n_workers = workers or min(cpu, 12)
    loaded = 0

    try:
        if prefer_process:
            with ProcessPoolExecutor(max_workers=n_workers) as pool:
                for code, df in pool.map(_load_one, paths, chunksize=32):
                    _CACHE[code] = df
                    if df is not None:
                        loaded += 1
        else:
            with ThreadPoolExecutor(max_workers=min(32, max(8, cpu * 2))) as pool:
                futs = [pool.submit(_load_one, p) for p in paths]
                for fut in as_completed(futs):
                    code, df = fut.result()
                    _CACHE[code] = df
                    if df is not None:
                        loaded += 1
    except Exception:
        _CACHE.clear()
        loaded = 0
        with ThreadPoolExecutor(max_workers=min(32, max(8, cpu * 2))) as pool:
            futs = [pool.submit(_load_one, p) for p in paths]
            for fut in as_completed(futs):
                code, df = fut.result()
                _CACHE[code] = df
                if df is not None:
                    loaded += 1
    return loaded


def preload(workers: int | None = None, force: bool = False, prefer_process: bool = True) -> int:
    """并行预载主板日线。返回成功条数。优先 SQLite。"""
    global _PRELOADED, _SOURCE
    if _PRELOADED and not force and _CACHE:
        return sum(1 for v in _CACHE.values() if v is not None)

    if use_sqlite():
        loaded = _preload_sqlite()
        _SOURCE = "sqlite"
    else:
        loaded = _preload_csv(workers, prefer_process)
        _SOURCE = "csv"

    _PRELOADED = True
    return loaded


def get(code: str, asof: str | None = None) -> pd.DataFrame | None:
    """取缓存；未预载时单票懒加载。asof 时返回截断副本，不改缓存。"""
    code = str(code).zfill(6)
    if code not in _CACHE:
        if use_sqlite():
            try:
                import daily_db

                df_raw = daily_db.load_bars_df(code, limit=MAX_BARS)
                _CACHE[code] = _prepare(df_raw) if not df_raw.empty else None
            except Exception:
                _CACHE[code] = None
        else:
            path = DAILY / f"{code}.csv"
            if not path.exists():
                _CACHE[code] = None
                return None
            _, df = _load_one(str(path))
            _CACHE[code] = df
    df = _CACHE.get(code)
    if df is None:
        return None
    if asof:
        cutoff = pd.Timestamp(asof)
        out = df.loc[df["日期"] <= cutoff].reset_index(drop=True)
        if len(out) < 80:
            return None
        return out
    return df


def cached_codes() -> list[str]:
    return sorted(c for c, df in _CACHE.items() if df is not None)


def source() -> str:
    return _SOURCE or ("sqlite" if use_sqlite() else "csv")
