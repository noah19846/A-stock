"""
主板日线共享缓存：长/短线筛选只读盘一遍。

- 若存在 data/db/daily.db：从 SQLite 批量读（推荐）
- 否则：多进程并行读 CSV
- 一次性算好后复权/均线；只保留近 MAX_BARS 根
- get(code, asof=...) 内存切片，不再读盘
- 同日预载结果可落盘 data/db/preload_cache.pkl，二次启动秒级命中
"""

from __future__ import annotations

import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DAILY = ROOT / "data" / "daily_raw"
DB_PATH = ROOT / "data" / "db" / "daily.db"
PRELOAD_CACHE = ROOT / "data" / "db" / "preload_cache.pkl"

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
_CACHE_KEY: str = ""


def clear() -> None:
    global _PRELOADED, _SOURCE, _CACHE_KEY
    _CACHE.clear()
    _PRELOADED = False
    _SOURCE = ""
    _CACHE_KEY = ""


def use_sqlite() -> bool:
    return DB_PATH.exists() and DB_PATH.stat().st_size > 0


def _rolling_mean(x: np.ndarray, w: int) -> np.ndarray:
    n = len(x)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < w:
        return out
    c = np.cumsum(x, dtype=np.float64)
    out[w - 1] = c[w - 1] / w
    if n > w:
        out[w:] = (c[w:] - c[:-w]) / w
    return out


def _prepare(df: pd.DataFrame) -> pd.DataFrame | None:
    """中文列 DataFrame → 带 px/均线的缓存帧（CSV 路径）。"""
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
    px = df["px"].to_numpy(dtype=np.float64, copy=False)
    ret = np.empty(len(px), dtype=np.float64)
    ret[0] = np.nan
    ret[1:] = px[1:] / px[:-1] - 1.0
    df["ret"] = ret
    df["ma5"] = _rolling_mean(px, 5)
    df["ma10"] = _rolling_mean(px, 10)
    df["ma20"] = _rolling_mean(px, 20)
    df["ma60"] = _rolling_mean(px, 60)
    return df


def _prepare_en(g: pd.DataFrame) -> pd.DataFrame | None:
    """英文列近端片段 → 缓存帧（SQLite 批量路径，少一次 rename）。"""
    if len(g) > MAX_BARS:
        g = g.iloc[-MAX_BARS:]
    n = len(g)
    if n < 80:
        return None
    trade_date = pd.to_datetime(g["trade_date"].to_numpy())
    o = g["open"].to_numpy(dtype=np.float64, copy=False)
    h = g["high"].to_numpy(dtype=np.float64, copy=False)
    low = g["low"].to_numpy(dtype=np.float64, copy=False)
    c = g["close"].to_numpy(dtype=np.float64, copy=False)
    vol = g["volume"].to_numpy(dtype=np.float64, copy=False)
    amt = g["amount"].to_numpy(dtype=np.float64, copy=False)
    fs = g["float_shares"].to_numpy(dtype=np.float64, copy=False)
    turn = g["turnover"].to_numpy(dtype=np.float64, copy=False)
    f = g["hfq_factor"].to_numpy(dtype=np.float64, copy=False)
    px = c * f
    ret = np.empty(n, dtype=np.float64)
    ret[0] = np.nan
    ret[1:] = px[1:] / px[:-1] - 1.0
    return pd.DataFrame(
        {
            "日期": trade_date,
            "开盘": o,
            "最高": h,
            "最低": low,
            "收盘": c,
            "成交量": vol,
            "成交额": amt,
            "流通股本": fs,
            "换手率": turn,
            "后复权因子": f,
            "px": px,
            "op": o * f,
            "hi": h * f,
            "lo": low * f,
            "amt": amt,
            "vol": vol,
            "turn": turn * 100.0,
            "mv": fs * c / 1e8,
            "ret": ret,
            "ma5": _rolling_mean(px, 5),
            "ma10": _rolling_mean(px, 10),
            "ma20": _rolling_mean(px, 20),
            "ma60": _rolling_mean(px, 60),
        }
    )


def _disk_cache_key() -> str:
    import daily_db

    st = daily_db.stats()
    mx = str(st.get("max_date") or "")
    try:
        mtime = int(DB_PATH.stat().st_mtime)
    except OSError:
        mtime = 0
    # mtime：同日 force 重写 bar 后必须失效
    return f"{mx}|mtime={mtime}|bars={MAX_BARS}|v2"


def _try_load_disk_cache(key: str) -> int | None:
    if not PRELOAD_CACHE.exists():
        return None
    try:
        t0 = time.time()
        with PRELOAD_CACHE.open("rb") as f:
            payload = pickle.load(f)
        if not isinstance(payload, dict) or payload.get("key") != key:
            return None
        data = payload.get("data") or {}
        if not data:
            return None
        _CACHE.clear()
        _CACHE.update(data)
        n = sum(1 for v in _CACHE.values() if v is not None)
        print(f"  disk cache hit {n} codes  {time.time() - t0:.1f}s", flush=True)
        return n
    except Exception as e:
        print(f"  disk cache skip: {e}", flush=True)
        return None


def _save_disk_cache(key: str) -> None:
    try:
        PRELOAD_CACHE.parent.mkdir(parents=True, exist_ok=True)
        tmp = PRELOAD_CACHE.with_suffix(".pkl.tmp")
        t0 = time.time()
        with tmp.open("wb") as f:
            pickle.dump({"key": key, "data": dict(_CACHE)}, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(PRELOAD_CACHE)
        mb = PRELOAD_CACHE.stat().st_size / 1e6
        print(f"  disk cache saved {mb:.0f}MB  {time.time() - t0:.1f}s", flush=True)
    except Exception as e:
        print(f"  disk cache save fail: {e}", flush=True)


def _preload_sqlite() -> int:
    """无 ORDER BY 批量拉 + 单进程 numpy prepare；可命中磁盘缓存。"""
    global _CACHE_KEY
    import daily_db

    daily_db.init_db().close()
    key = _disk_cache_key()
    hit = _try_load_disk_cache(key)
    if hit is not None:
        _CACHE_KEY = key
        return hit

    cutoff = daily_db.recent_cutoff_date(limit=MAX_BARS)
    if not cutoff:
        return 0
    print(f"  sqlite since {cutoff} …", flush=True)
    t0 = time.time()
    bulk = daily_db.load_recent_bars_since(cutoff, prefixes=MAINBOARD_PREFIX, order=False)
    if bulk.empty:
        return 0
    bulk = bulk.sort_values(["code", "trade_date"], kind="mergesort")
    n_codes = int(bulk["code"].nunique())
    print(f"  sqlite bulk rows={len(bulk)} codes={n_codes}  {time.time() - t0:.1f}s", flush=True)

    loaded = 0
    t1 = time.time()
    for i, (code, g) in enumerate(bulk.groupby("code", sort=False), 1):
        code = str(code).zfill(6)
        try:
            prepared = _prepare_en(g)
        except Exception:
            prepared = None
        _CACHE[code] = prepared
        if prepared is not None:
            loaded += 1
        if i % 1000 == 0:
            print(f"  sqlite prepare {i}/{n_codes}", flush=True)
    del bulk
    print(f"  sqlite prepare done {loaded}  {time.time() - t1:.1f}s", flush=True)
    _CACHE_KEY = key
    _save_disk_cache(key)
    return loaded


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

    return [c for c in daily_db.list_codes() if c.startswith(MAINBOARD_PREFIX)]


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
        last = df["日期"].iloc[-1]
        # 已是最新（日常扫描）时跳过拷贝
        if last <= cutoff:
            return df
        out = df.loc[df["日期"] <= cutoff].reset_index(drop=True)
        if len(out) < 80:
            return None
        return out
    return df


def cached_codes() -> list[str]:
    return sorted(c for c, df in _CACHE.items() if df is not None)


def source() -> str:
    return _SOURCE or ("sqlite" if use_sqlite() else "csv")
