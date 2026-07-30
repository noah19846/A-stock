"""
日更：拉东财全市场当日截面，补进 data/daily_raw/{代码}.csv（固定文件名）。

更新后汇总每支股票最后两天日期；正常应只有一种
{last_date, last_second_date}。若出现多种模式，对少数派股票
自动全量重拉（新浪）并再补当日截面。

用法：
  .venv/bin/python update_daily.py              # 已有当日则跳过
  .venv/bin/python update_daily.py --force      # 强制覆盖当日
  .venv/bin/python update_daily.py --date 20260721
"""

from __future__ import annotations

import argparse
import time
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import requests

from fetch_daily import fetch_one, is_main_board

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "data" / "daily_raw"
COLS = [
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

CLIST_URL = "https://push2delay.eastmoney.com/api/qt/clist/get"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "Referer": "https://quote.eastmoney.com/",
}
FS = "m:0 t:6,m:0 t:80,m:1 t:2,m:1 t:23,m:0 t:81 s:2048"
FIELDS = "f12,f14,f2,f15,f16,f17,f18,f5,f6,f8,f20,f21"


def log(msg: str) -> None:
    print(msg, flush=True)


def _session() -> requests.Session:
    s = requests.Session()
    s.trust_env = False
    s.headers.update(HEADERS)
    return s


def normalize_date(s: str) -> str:
    s = s.strip().replace("/", "-")
    if len(s) == 8 and s.isdigit():
        return f"{s[:4]}-{s[4:6]}-{s[6:]}"
    return pd.Timestamp(s).strftime("%Y-%m-%d")


def latest_trade_date(asof: str | None = None) -> str:
    import akshare as ak

    td = ak.tool_trade_date_hist_sina()
    td["trade_date"] = pd.to_datetime(td["trade_date"])
    cutoff = pd.Timestamp(asof) if asof else pd.Timestamp.today().normalize()
    latest = td.loc[td["trade_date"] <= cutoff, "trade_date"].max()
    if pd.isna(latest):
        raise RuntimeError("无法确定最近交易日")
    return latest.strftime("%Y-%m-%d")


def fetch_em_spot() -> pd.DataFrame:
    session = _session()
    params = {
        "pn": "1",
        "pz": "100",
        "po": "1",
        "np": "1",
        "ut": "bd1d9ddb04089700cf9c27f6f7426281",
        "fltt": "2",
        "invt": "2",
        "fid": "f12",
        "fs": FS,
        "fields": FIELDS,
    }
    r = session.get(CLIST_URL, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()["data"]
    total = int(data["total"])
    rows = list(data["diff"])
    pages = (total + 99) // 100
    for pn in range(2, pages + 1):
        params["pn"] = str(pn)
        r = session.get(CLIST_URL, params=params, timeout=20)
        r.raise_for_status()
        rows.extend(r.json()["data"]["diff"])

    df = pd.DataFrame(rows).rename(
        columns={
            "f12": "代码",
            "f14": "名称",
            "f2": "最新价",
            "f15": "最高",
            "f16": "最低",
            "f17": "今开",
            "f18": "昨收",
            "f5": "成交量_手",
            "f6": "成交额",
            "f8": "换手率_pct",
            "f20": "总市值",
            "f21": "流通市值",
        }
    )
    df["代码"] = df["代码"].astype(str).str.zfill(6)
    return df


def to_num(v) -> float | None:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    s = str(v).strip()
    if s in ("", "-", "None", "nan"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def spot_to_bar(
    code: str,
    row: pd.Series,
    trade_date: str,
    prev_factor: float,
    prev_share: float,
) -> dict | None:
    price = to_num(row["最新价"])
    if price is None or price <= 0:
        return None

    float_mv = to_num(row["流通市值"])
    outstanding = (
        float_mv / price if float_mv is not None and float_mv > 0 else prev_share
    )
    vol_hand = to_num(row["成交量_手"])
    turn_pct = to_num(row["换手率_pct"])

    return {
        "股票代码": code,
        "日期": trade_date,
        "开盘": to_num(row["今开"]),
        "最高": to_num(row["最高"]),
        "最低": to_num(row["最低"]),
        "收盘": price,
        "成交量": vol_hand * 100 if vol_hand is not None else None,
        "成交额": to_num(row["成交额"]),
        "流通股本": outstanding,
        "换手率": turn_pct / 100.0 if turn_pct is not None else None,
        "后复权因子": prev_factor,
    }


def list_local_csvs(out_dir: Path) -> list[Path]:
    files = []
    for p in out_dir.glob("*.csv"):
        if p.name.startswith("_") or p.name.startswith("."):
            continue
        files.append(p)
    return sorted(files)


def code_from_path(path: Path) -> str:
    return path.stem.split("_")[0].zfill(6)


def migrate_to_fixed_name(path: Path, code: str, out_dir: Path) -> Path:
    """把 代码_日期.csv / 代码.csv 统一成 代码.csv，并清理同代码其它命名。"""
    target = out_dir / f"{code}.csv"
    if path.resolve() != target.resolve():
        if target.exists():
            target.unlink()
        path.rename(target)
    for old in out_dir.glob(f"{code}_*.csv"):
        old.unlink(missing_ok=True)
    return target


def peek_tail_dates(path: Path) -> tuple[str, str]:
    """只读文件尾部，返回 (last_second_date, last_date)。"""
    try:
        with path.open("rb") as f:
            f.seek(0, 2)
            size = f.tell()
            if size == 0:
                return "", ""
            f.seek(max(0, size - 8192))
            chunk = f.read().decode("utf-8-sig", errors="ignore")
        lines = [ln.strip() for ln in chunk.splitlines() if ln.strip()]
        data_lines = [ln for ln in lines if not ln.startswith("股票代码")]
        if not data_lines:
            return "", ""

        def date_of(line: str) -> str:
            parts = line.split(",")
            return parts[1] if len(parts) > 1 else ""

        d2 = date_of(data_lines[-1])
        d1 = date_of(data_lines[-2]) if len(data_lines) > 1 else ""
        return d1, d2
    except Exception:
        return "", ""


def append_spot_bar(
    path: Path,
    code: str,
    trade_date: str,
    spot_row: pd.Series | None,
    force: bool,
) -> tuple[str, str, str]:
    """
    返回 (status, last_second_date, last_date)
    status: updated / skipped / fail:...

    默认：若最后一行日期已是 trade_date，直接跳过（不读全表、不写盘）。
    --force：即使已有当日也覆盖重写。
    """
    try:
        if path.stat().st_size == 0:
            return "fail:空文件", "", ""

        d1, d2 = peek_tail_dates(path)
        if d2 == trade_date and not force:
            return "skipped", d1, d2

        df = pd.read_csv(path, dtype={"股票代码": str})
        if df.empty or "日期" not in df.columns:
            return "fail:无有效列", "", ""

        df["股票代码"] = df["股票代码"].astype(str).str.zfill(6)
        df["日期"] = df["日期"].astype(str)

        if spot_row is None:
            last2 = df["日期"].tail(2).tolist()
            d2 = last2[-1] if last2 else ""
            d1 = last2[-2] if len(last2) > 1 else ""
            return "fail:截面中无此代码", d1, d2

        already = trade_date in set(df["日期"])
        prev_factor = float(df["后复权因子"].iloc[-1])
        prev_share = float(df["流通股本"].iloc[-1])
        bar = spot_to_bar(code, spot_row, trade_date, prev_factor, prev_share)
        if bar is None:
            last2 = df["日期"].tail(2).tolist()
            d2 = last2[-1] if last2 else ""
            d1 = last2[-2] if len(last2) > 1 else ""
            return "fail:截面无有效报价", d1, d2

        if already:
            df = df[df["日期"] != trade_date]
        df = pd.concat([df, pd.DataFrame([bar])], ignore_index=True)
        df = df.sort_values("日期").reset_index(drop=True)
        df = df[COLS]
        df.to_csv(path, index=False, encoding="utf-8-sig")

        last2 = df["日期"].tail(2).tolist()
        d2 = last2[-1] if last2 else ""
        d1 = last2[-2] if len(last2) > 1 else ""
        return "updated", d1, d2
    except Exception as e:
        return f"fail:{e}", "", ""


def record_timing(out_dir: Path, info: dict) -> Path:
    """追加一次 update 耗时记录。"""
    path = out_dir / "_update_timing.csv"
    row = pd.DataFrame([info])
    if path.exists() and path.stat().st_size > 0:
        row.to_csv(path, mode="a", header=False, index=False, encoding="utf-8-sig")
    else:
        row.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def collect_tail_patterns(out_dir: Path) -> dict[tuple[str, str], list[str]]:
    """扫描所有 CSV，按 (last_second_date, last_date) 分组。"""
    groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for path in list_local_csvs(out_dir):
        code = code_from_path(path)
        try:
            df = pd.read_csv(path, dtype={"股票代码": str}, usecols=["日期"])
            ds = df["日期"].astype(str).tolist()
            d2 = ds[-1] if ds else ""
            d1 = ds[-2] if len(ds) > 1 else ""
        except Exception:
            d1, d2 = "", ""
        groups[(d1, d2)].append(code)
    return dict(groups)


def refetch_full_and_patch(
    code: str,
    out_dir: Path,
    trade_date: str,
    spot_row: pd.Series | None,
    *,
    use_sqlite: bool = False,
) -> str:
    """全量重拉 + 补当日。返回状态说明。"""
    path = out_dir / f"{code}.csv"
    try:
        df = fetch_one(code)
        df.to_csv(path, index=False, encoding="utf-8-sig")
        if use_sqlite:
            import daily_db

            conn = daily_db.init_db()
            daily_db.replace_code_df(conn, code, df)
            conn.close()
        for old in out_dir.glob(f"{code}_*.csv"):
            old.unlink(missing_ok=True)
        status, d1, d2 = append_spot_bar(path, code, trade_date, spot_row, force=True)
        if use_sqlite and status == "updated":
            # 同步当日 bar 进库（append_spot_bar 已写 CSV）
            import daily_db

            conn = daily_db.init_db()
            try:
                df2 = pd.read_csv(path, dtype={"股票代码": str})
                row = df2[df2["日期"].astype(str).str.slice(0, 10) == trade_date]
                if not row.empty:
                    daily_db.upsert_bars(conn, [daily_db.bar_dict_from_cn(row.iloc[0].to_dict())])
                    conn.commit()
            finally:
                conn.close()
        return f"{status}; tail=({d1},{d2})"
    except Exception as e:
        return f"refetch_fail:{e}"


def update_all(
    out_dir: Path,
    trade_date: str,
    spot: pd.DataFrame,
    force: bool,
) -> tuple[list[dict], list[dict]]:
    spot_map = {r["代码"]: r for _, r in spot.iterrows()}
    files = list_local_csvs(out_dir)

    tails: list[dict] = []
    fails: list[dict] = []

    for i, path in enumerate(files, 1):
        code = code_from_path(path)
        path = migrate_to_fixed_name(path, code, out_dir)
        status, d1, d2 = append_spot_bar(
            path, code, trade_date, spot_map.get(code), force=force
        )
        item = {
            "股票代码": code,
            "last_second_date": d1,
            "last_date": d2,
            "status": status,
        }
        tails.append(item)
        if status.startswith("fail"):
            fails.append({"股票代码": code, "原因": status})

        if i % 500 == 0:
            log(f"  进度 {i}/{len(files)}")

    return tails, fails


def update_all_sqlite(
    trade_date: str,
    spot: pd.DataFrame,
    force: bool,
    *,
    also_csv: bool = False,
    out_dir: Path | None = None,
) -> tuple[list[dict], list[dict]]:
    """SQLite 批量日更：单事务 upsert，避免逐文件重写 CSV。"""
    import daily_db

    out_dir = out_dir or OUT_DIR
    conn = daily_db.init_db()
    last_map = daily_db.last_two_by_code(conn)
    spot_map = {r["代码"]: r for _, r in spot.iterrows()}

    codes = sorted(last_map.keys())
    if not codes:
        codes = [code_from_path(p) for p in list_local_csvs(out_dir)]

    tails: list[dict] = []
    fails: list[dict] = []
    upsert_rows: list[tuple] = []
    n_skip = 0
    n_upd = 0

    for i, code in enumerate(codes, 1):
        info = last_map.get(code) or {
            "d1": "",
            "d2": "",
            "hfq_factor": None,
            "float_shares": None,
        }
        d1, d2 = info["d1"], info["d2"]
        if d2 == trade_date and not force:
            tails.append(
                {
                    "股票代码": code,
                    "last_second_date": d1,
                    "last_date": d2,
                    "status": "skipped",
                }
            )
            n_skip += 1
            continue

        spot_row = spot_map.get(code)
        if spot_row is None:
            tails.append(
                {
                    "股票代码": code,
                    "last_second_date": d1,
                    "last_date": d2,
                    "status": "fail:截面中无此代码",
                }
            )
            fails.append({"股票代码": code, "原因": "fail:截面中无此代码"})
            continue

        prev_factor = info["hfq_factor"]
        prev_share = info["float_shares"]
        if prev_factor is None or prev_share is None:
            path = out_dir / f"{code}.csv"
            if path.exists():
                try:
                    df = pd.read_csv(path, dtype={"股票代码": str})
                    prev_factor = float(df["后复权因子"].iloc[-1])
                    prev_share = float(df["流通股本"].iloc[-1])
                    ds = df["日期"].astype(str).tolist()
                    d2 = ds[-1] if ds else d2
                    d1 = ds[-2] if len(ds) > 1 else d1
                except Exception:
                    pass
        if prev_factor is None or prev_share is None:
            tails.append(
                {
                    "股票代码": code,
                    "last_second_date": d1,
                    "last_date": d2,
                    "status": "fail:无历史后复权/股本",
                }
            )
            fails.append({"股票代码": code, "原因": "fail:无历史后复权/股本"})
            continue

        bar = spot_to_bar(code, spot_row, trade_date, float(prev_factor), float(prev_share))
        if bar is None:
            tails.append(
                {
                    "股票代码": code,
                    "last_second_date": d1,
                    "last_date": d2,
                    "status": "fail:截面无有效报价",
                }
            )
            fails.append({"股票代码": code, "原因": "fail:截面无有效报价"})
            continue

        upsert_rows.append(daily_db.bar_dict_from_cn(bar))
        new_d1 = d2 if d2 != trade_date else d1
        tails.append(
            {
                "股票代码": code,
                "last_second_date": new_d1,
                "last_date": trade_date,
                "status": "updated",
            }
        )
        n_upd += 1

        if also_csv:
            path = out_dir / f"{code}.csv"
            if path.exists():
                append_spot_bar(path, code, trade_date, spot_row, force=force)

        if i % 500 == 0:
            log(f"  进度 {i}/{len(codes)}")

    if upsert_rows:
        daily_db.upsert_bars(conn, upsert_rows)
        conn.commit()
    conn.close()
    log(f"  SQLite upsert {n_upd}  skipped {n_skip}  fail {len(fails)}")
    return tails, fails


def collect_tail_patterns_sqlite() -> dict[tuple[str, str], list[str]]:
    import daily_db

    conn = daily_db.connect()
    try:
        last_map = daily_db.last_two_by_code(conn)
    finally:
        conn.close()
    groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for code, info in last_map.items():
        groups[(info["d1"], info["d2"])].append(code)
    return dict(groups)


def repair_minority(
    out_dir: Path,
    trade_date: str,
    spot: pd.DataFrame,
    groups: dict[tuple[str, str], list[str]],
    *,
    use_sqlite: bool = False,
) -> None:
    if len(groups) <= 1:
        log("尾部日期模式唯一，数据齐整。")
        return

    # 按数量排序：多数派保留，少数派重拉
    ranked = sorted(groups.items(), key=lambda x: len(x[1]), reverse=True)
    log("\n======== 尾部日期模式分布 ========")
    for (d1, d2), codes in ranked:
        log(f"  count={len(codes):4d}  {{last_second_date: '{d1}', last_date: '{d2}'}}")

    majority_key, majority_codes = ranked[0]
    minority = []
    for key, codes in ranked[1:]:
        minority.extend(codes)

    log(
        f"\n多数派: {{last_second_date: '{majority_key[0]}', last_date: '{majority_key[1]}'}} "
        f"共 {len(majority_codes)} 支"
    )
    log(f"少数派共 {len(minority)} 支，将全量重拉并补当日：")
    log(", ".join(minority))

    spot_map = {r["代码"]: r for _, r in spot.iterrows()}
    for i, code in enumerate(minority, 1):
        # 跳过非主板（理论上不应出现）
        if not is_main_board(code):
            log(f"[{i}/{len(minority)}] skip non-main {code}")
            continue
        result = refetch_full_and_patch(
            code, out_dir, trade_date, spot_map.get(code), use_sqlite=use_sqlite
        )
        log(f"[{i}/{len(minority)}] refetch {code} -> {result}")
        time.sleep(0.8)


def main() -> None:
    parser = argparse.ArgumentParser(description="东财截面日更 daily_raw（固定文件名）")
    parser.add_argument("--date", default="", help="交易日 YYYYMMDD / YYYY-MM-DD，默认最近交易日")
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制覆盖：即使最后一行已是当日也重写",
    )
    parser.add_argument(
        "--allow-mismatch",
        action="store_true",
        help="允许指定日期与最近交易日不一致（慎用）",
    )
    parser.add_argument(
        "--no-repair",
        action="store_true",
        help="发现少数派时只打印提醒，不自动重拉",
    )
    parser.add_argument(
        "--sqlite",
        action="store_true",
        help="写入 data/db/daily.db（批量 upsert，远快于重写 CSV）",
    )
    parser.add_argument(
        "--also-csv",
        action="store_true",
        help="与 --sqlite 联用：同时写 CSV（较慢，双写）",
    )
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    use_sqlite = bool(args.sqlite)
    if use_sqlite:
        import daily_db

        daily_db.init_db().close()
        st = daily_db.stats()
        if not st.get("codes"):
            log("提示：SQLite 库为空，请先 .venv/bin/python daily_db.py migrate")

    t0 = time.time()
    latest = latest_trade_date()
    trade_date = normalize_date(args.date) if args.date else latest
    log(
        f"目标日期={trade_date}  最近交易日={latest}  force={args.force}"
        f"  sqlite={use_sqlite} also_csv={args.also_csv}"
    )

    if trade_date != latest and not args.allow_mismatch:
        raise SystemExit(
            f"--date={trade_date} 与最近交易日 {latest} 不一致；"
            f"确认无误请加 --allow-mismatch"
        )

    t_spot = time.time()
    log("拉取东财全市场延迟行情...")
    spot = fetch_em_spot()
    spot_sec = time.time() - t_spot
    log(f"截面行数={len(spot)}  拉取耗时 {spot_sec:.1f}s")

    t_upd = time.time()
    if use_sqlite:
        log("SQLite 批量补当日...")
        tails, fails = update_all_sqlite(
            trade_date,
            spot,
            force=args.force,
            also_csv=args.also_csv,
            out_dir=out_dir,
        )
    else:
        log("遍历本地 CSV（统一为 代码.csv）并补当日...")
        tails, fails = update_all(out_dir, trade_date, spot, force=args.force)
    upd_sec = time.time() - t_upd

    n_updated = sum(1 for r in tails if r["status"] == "updated")
    n_skipped = sum(1 for r in tails if r["status"] == "skipped")
    n_failed = len(fails)

    log("\n======== 未写成功 ========")
    if not fails:
        log("(无)")
    else:
        for r in fails:
            log(f"{r['股票代码']}  {r['原因']}")

    report = pd.DataFrame(tails)
    report_path = out_dir / "_tail_dates.csv"
    report.to_csv(report_path, index=False, encoding="utf-8-sig")

    counter = Counter((r["last_second_date"], r["last_date"]) for r in tails)
    log("\n======== 尾部日期模式 ========")
    for (d1, d2), n in counter.most_common():
        log(f"  count={n:4d}  {{last_second_date: '{d1}', last_date: '{d2}'}}")

    t_repair = time.time()
    repair_sec = 0.0
    groups = defaultdict(list)
    for r in tails:
        groups[(r["last_second_date"], r["last_date"])].append(r["股票代码"])
    groups = dict(groups)

    if len(groups) > 1 and not args.no_repair:
        repair_minority(
            out_dir, trade_date, spot, groups, use_sqlite=use_sqlite
        )
        repair_sec = time.time() - t_repair
        groups2 = (
            collect_tail_patterns_sqlite() if use_sqlite else collect_tail_patterns(out_dir)
        )
        log("\n======== 修复后模式 ========")
        for (d1, d2), codes in sorted(groups2.items(), key=lambda x: -len(x[1])):
            log(f"  count={len(codes):4d}  {{last_second_date: '{d1}', last_date: '{d2}'}}")
        report2 = [
            {"股票代码": c, "last_second_date": d1, "last_date": d2}
            for (d1, d2), codes in groups2.items()
            for c in codes
        ]
        pd.DataFrame(report2).sort_values("股票代码").to_csv(
            report_path, index=False, encoding="utf-8-sig"
        )
        n_patterns = len(groups2)
    elif len(groups) > 1:
        ranked = sorted(groups.items(), key=lambda x: len(x[1]), reverse=True)
        minority = []
        for key, codes in ranked[1:]:
            minority.extend(codes)
        log(f"\n少数派 {len(minority)} 支（未自动修复）：")
        log(", ".join(minority))
        n_patterns = len(groups)
    else:
        log("尾部日期模式唯一，数据齐整。")
        n_patterns = 1

    total_sec = time.time() - t0
    timing_path = record_timing(
        out_dir,
        {
            "time": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "trade_date": trade_date,
            "force": args.force,
            "spot_sec": round(spot_sec, 2),
            "update_sec": round(upd_sec, 2),
            "repair_sec": round(repair_sec, 2),
            "total_sec": round(total_sec, 2),
            "updated": n_updated,
            "skipped": n_skipped,
            "failed": n_failed,
            "patterns": n_patterns,
            "sqlite": int(use_sqlite),
        },
    )

    log(
        f"\n完成：updated={n_updated} skipped={n_skipped} failed={n_failed} "
        f"patterns={n_patterns} 总耗时 {total_sec:.1f}s"
        + (" [sqlite]" if use_sqlite else "")
    )
    log(f"报告：{report_path}")
    log(f"耗时记录：{timing_path}")


if __name__ == "__main__":
    main()
