"""
拉取沪深主板日线：不复权行情 + 后复权因子（hfq_factor）。
排除 ST、创业板、科创板、北交所。
每支股票一个 CSV；workers>1 时用进程池（新浪解码器非线程安全）。
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import akshare as ak
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "data" / "daily_raw"
LIST_CSV = ROOT / "a_stock_list.csv"
FAIL_LOG_NAME = "_failed.txt"


def is_main_board(code: str) -> bool:
    code = str(code).zfill(6)
    if code.startswith(("300", "301", "688", "689")):
        return False
    if code.startswith(("8", "4", "92")):
        return False
    return code.startswith(("600", "601", "603", "605", "000", "001", "002"))


def to_sina_symbol(code: str) -> str:
    code = str(code).zfill(6)
    return f"sh{code}" if code.startswith(("5", "6", "9")) else f"sz{code}"


def load_universe() -> pd.DataFrame:
    if LIST_CSV.exists():
        df = pd.read_csv(LIST_CSV, dtype={"code": str})
        df["code"] = df["code"].str.zfill(6)
    else:
        df = ak.stock_info_a_code_name()
        df["code"] = df["code"].astype(str).str.zfill(6)
        df.to_csv(LIST_CSV, index=False, encoding="utf-8-sig")

    df = df[df["code"].map(is_main_board)].copy()
    df = df[~df["name"].astype(str).str.contains("ST", case=False, na=False)]
    df = df.reset_index(drop=True)
    return df


def merge_raw_and_factor(raw: pd.DataFrame, factor: pd.DataFrame) -> pd.DataFrame:
    raw = raw.copy()
    factor = factor.copy()
    raw["date"] = pd.to_datetime(raw["date"])
    factor["date"] = pd.to_datetime(factor["date"])
    factor = factor[factor["date"] >= "1990-01-01"]

    merged = pd.merge(raw, factor, on="date", how="outer", sort=True)
    merged["hfq_factor"] = merged["hfq_factor"].ffill().bfill()
    merged = merged.dropna(subset=["close"]).copy()
    merged["date"] = merged["date"].dt.strftime("%Y-%m-%d")
    return merged


def fetch_one(code: str) -> pd.DataFrame:
    symbol = to_sina_symbol(code)
    raw = ak.stock_zh_a_daily(symbol=symbol, adjust="")
    try:
        factor = ak.stock_zh_a_daily(symbol=symbol, adjust="hfq-factor")
    except Exception:
        factor = pd.DataFrame({"date": raw["date"], "hfq_factor": 1.0})

    out = merge_raw_and_factor(raw, factor)
    out.insert(0, "code", code)
    out = out.rename(
        columns={
            "date": "日期",
            "open": "开盘",
            "high": "最高",
            "low": "最低",
            "close": "收盘",
            "volume": "成交量",
            "amount": "成交额",
            "outstanding_share": "流通股本",
            "turnover": "换手率",
            "hfq_factor": "后复权因子",
            "code": "股票代码",
        }
    )
    return out[
        [
            "股票代码",
            "日期",
            "开盘",
            "最高",
            "最低",
            "收盘",
            "成交量",
            "成交额",
            "流通股本",
            "换手率",
            "后复权因子",
        ]
    ]


def _worker(payload: tuple[str, str, str, bool]) -> tuple[str, str, str]:
    """子进程任务：返回 (status, code, detail)。"""
    code, name, out_dir, force = payload
    path = Path(out_dir) / f"{code}.csv"
    if path.exists() and not force:
        return "skip", code, name
    try:
        df = fetch_one(code)
        df.to_csv(path, index=False, encoding="utf-8-sig")
        return "ok", code, f"{name} rows={len(df)}"
    except Exception as e:
        return "fail", code, f"{name} {e}"


def main() -> None:
    parser = argparse.ArgumentParser(description="拉取主板不复权日线+后复权因子")
    parser.add_argument("--sleep", type=float, default=0.0, help="串行模式下每支间隔秒数")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="并发进程数（>1 用进程池；新浪 JS 解码不能多线程）",
    )
    parser.add_argument("--limit", type=int, default=0, help="只拉前 N 支，0 表示全部")
    parser.add_argument("--force", action="store_true", help="已存在也重新拉取")
    parser.add_argument("--out-dir", type=str, default=str(OUT_DIR), help="输出目录")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fail_log = out_dir / FAIL_LOG_NAME

    universe = load_universe()
    if args.limit > 0:
        universe = universe.head(args.limit)

    total = len(universe)
    workers = max(1, args.workers)

    def log(msg: str) -> None:
        print(msg, flush=True)

    log(f"待拉取 {total} 支 | workers={workers} | out={out_dir}")
    t0 = time.time()
    ok = skip = fail = done = 0

    payloads = [
        (row["code"], row["name"], str(out_dir), args.force)
        for _, row in universe.iterrows()
    ]

    def handle(status: str, code: str, detail: str) -> None:
        nonlocal ok, skip, fail, done
        done += 1
        if status == "ok":
            ok += 1
        elif status == "skip":
            skip += 1
        else:
            fail += 1
            with fail_log.open("a", encoding="utf-8") as f:
                f.write(f"{code},{detail}\n")
        if done % 50 == 0 or done == total or status == "fail":
            elapsed = time.time() - t0
            speed = done / elapsed if elapsed > 0 else 0
            eta = (total - done) / speed if speed > 0 else 0
            log(
                f"[{done}/{total}] {status:4} {code} {detail} | "
                f"{elapsed:.0f}s elapsed, ~{eta:.0f}s eta, {speed:.1f}/s"
            )

    if workers == 1:
        for payload in payloads:
            status, code, detail = _worker(payload)
            handle(status, code, detail)
            if args.sleep > 0 and status != "skip":
                time.sleep(args.sleep)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_worker, p) for p in payloads]
            for fut in as_completed(futs):
                status, code, detail = fut.result()
                handle(status, code, detail)

    log(f"完成：成功 {ok}，跳过 {skip}，失败 {fail}，耗时 {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
