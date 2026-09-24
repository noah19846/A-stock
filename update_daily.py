"""
日更：拉东财全市场当日截面，补进 data/daily_raw/{代码}.csv 或 data/db/daily.db。

更新后汇总每支股票最后两天日期；正常应只有一种
{last_date, last_second_date}。若出现多种模式，写入少数派标记文件，
不自动重拉（避免卡住）；需要时再单独修复。

用法：
  .venv/bin/python update_daily.py --sqlite --force
  .venv/bin/python update_daily.py --list-minority
  .venv/bin/python update_daily.py --repair-minority --sqlite
  .venv/bin/python update_daily.py --repair-minority --sqlite --codes 002828,600439
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

from fetch_daily import fetch_one, is_main_board

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "data" / "daily_raw"
MINORITY_MARK = ROOT / "data" / "db" / "tail_minority.json"
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

CLIST_URLS = (
    "https://push2delay.eastmoney.com/api/qt/clist/get",
    "https://push2.eastmoney.com/api/qt/clist/get",
)
SINA_COUNT_URL = (
    "http://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/"
    "Market_Center.getHQNodeStockCount"
)
SINA_SPOT_URL = (
    "http://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/"
    "Market_Center.getHQNodeData"
)
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "Referer": "https://quote.eastmoney.com/",
}
FS = "m:0 t:6,m:0 t:80,m:1 t:2,m:1 t:23,m:0 t:81 s:2048"
FIELDS = "f12,f14,f2,f15,f16,f17,f18,f5,f6,f8,f20,f21"
MIN_SPOT_ROWS = 4000
REQUEST_ATTEMPTS = 3


def log(msg: str) -> None:
    print(msg, flush=True)


def _session() -> requests.Session:
    s = requests.Session()
    s.trust_env = False
    s.headers.update(HEADERS)
    return s


def _get_text_with_retry(
    urls: tuple[str, ...],
    params: dict,
    label: str,
    *,
    attempts: int = REQUEST_ATTEMPTS,
) -> str:
    errors: list[str] = []
    for attempt in range(1, attempts + 1):
        for url in urls:
            session = _session()
            try:
                response = session.get(url, params=params, timeout=20)
                response.raise_for_status()
                return response.text
            except (requests.RequestException, ValueError) as exc:
                errors.append(f"{url}: {type(exc).__name__}: {exc}")
            finally:
                session.close()
        if attempt < attempts:
            delay = 0.5 * (2 ** (attempt - 1))
            log(f"  {label} 第 {attempt} 次失败，{delay:.1f}s 后重试...")
            time.sleep(delay)
    detail = errors[-1] if errors else "unknown error"
    raise RuntimeError(f"{label} 连续失败 {attempts} 次；最后错误：{detail}")


def _validate_spot(df: pd.DataFrame, source: str) -> pd.DataFrame:
    required = {
        "代码",
        "名称",
        "最新价",
        "最高",
        "最低",
        "今开",
        "昨收",
        "成交量_手",
        "成交额",
        "换手率_pct",
        "总市值",
        "流通市值",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(f"{source} 截面缺少字段：{', '.join(missing)}")
    df = df.drop_duplicates(subset=["代码"], keep="last").reset_index(drop=True)
    if len(df) < MIN_SPOT_ROWS:
        raise RuntimeError(
            f"{source} 截面仅 {len(df)} 行，低于完整性下限 {MIN_SPOT_ROWS}"
        )
    return df


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
    text = _get_text_with_retry(CLIST_URLS, params, "东财截面第 1 页")
    data = json.loads(text).get("data")
    if not data or not data.get("diff"):
        raise RuntimeError("东财截面第 1 页返回空数据")
    total = int(data["total"])
    rows = list(data["diff"])
    pages = (total + 99) // 100
    for pn in range(2, pages + 1):
        params["pn"] = str(pn)
        text = _get_text_with_retry(CLIST_URLS, params, f"东财截面第 {pn} 页")
        page_data = json.loads(text).get("data")
        if not page_data or not page_data.get("diff"):
            raise RuntimeError(f"东财截面第 {pn} 页返回空数据")
        rows.extend(page_data["diff"])

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
    return _validate_spot(df, "东财")


def _sina_rows_to_spot(rows: list[dict]) -> pd.DataFrame:
    raw = pd.DataFrame(rows)
    return pd.DataFrame(
        {
            "代码": raw["code"].astype(str).str.zfill(6),
            "名称": raw["name"],
            "最新价": pd.to_numeric(raw["trade"], errors="coerce"),
            "最高": pd.to_numeric(raw["high"], errors="coerce"),
            "最低": pd.to_numeric(raw["low"], errors="coerce"),
            "今开": pd.to_numeric(raw["open"], errors="coerce"),
            "昨收": pd.to_numeric(raw["settlement"], errors="coerce"),
            # 新浪成交量单位为股；下游统一接收“手”。
            "成交量_手": pd.to_numeric(raw["volume"], errors="coerce") / 100.0,
            "成交额": pd.to_numeric(raw["amount"], errors="coerce"),
            # turnoverratio 已是百分数；市值字段单位为万元。
            "换手率_pct": pd.to_numeric(raw["turnoverratio"], errors="coerce"),
            "总市值": pd.to_numeric(raw["mktcap"], errors="coerce") * 10000.0,
            "流通市值": pd.to_numeric(raw["nmc"], errors="coerce") * 10000.0,
        }
    )


def fetch_sina_spot() -> pd.DataFrame:
    from akshare.utils import demjson

    count_text = _get_text_with_retry(
        (SINA_COUNT_URL,), {"node": "hs_a"}, "新浪截面页数"
    )
    match = re.search(r"\d+", count_text)
    if not match:
        raise RuntimeError(f"新浪截面页数无法解析：{count_text[:100]!r}")
    page_count = (int(match.group()) + 79) // 80
    base_params = {
        "page": "1",
        "num": "80",
        "sort": "symbol",
        "asc": "1",
        "node": "hs_a",
        "symbol": "",
        "_s_r_a": "page",
    }
    rows: list[dict] = []
    for page in range(1, page_count + 1):
        params = {**base_params, "page": str(page)}
        text = _get_text_with_retry(
            (SINA_SPOT_URL,), params, f"新浪截面第 {page} 页"
        )
        page_rows = demjson.decode(text)
        if not isinstance(page_rows, list) or not page_rows:
            raise RuntimeError(f"新浪截面第 {page} 页返回空数据")
        rows.extend(page_rows)

    df = _sina_rows_to_spot(rows)
    return _validate_spot(df, "新浪")


def fetch_market_spot() -> pd.DataFrame:
    try:
        spot = fetch_em_spot()
        log("  行情源：东财")
        return spot
    except Exception as exc:
        log(f"  东财截面失败，切换新浪备用源：{type(exc).__name__}: {exc}")
    spot = fetch_sina_spot()
    log("  行情源：新浪（备用）")
    return spot


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


def minority_mark_path(out_dir: Path | None = None) -> Path:
    MINORITY_MARK.parent.mkdir(parents=True, exist_ok=True)
    return MINORITY_MARK


def write_minority_mark(
    trade_date: str,
    groups: dict[tuple[str, str], list[str]],
    fails: list[dict],
    *,
    out_dir: Path | None = None,
) -> Path | None:
    """标记少数派 / 失败票，不自动修复。齐整时清除旧标记。"""
    path = minority_mark_path(out_dir)
    ranked = sorted(groups.items(), key=lambda x: len(x[1]), reverse=True)
    fail_codes = {
        str(r.get("股票代码", "")).zfill(6)
        for r in fails
        if r.get("股票代码")
    }

    if len(ranked) <= 1 and not fail_codes:
        if path.exists():
            path.unlink(missing_ok=True)
            log(f"尾部齐整，已清除标记：{path}")
        return None

    majority_key, majority_codes = ranked[0] if ranked else (("", ""), [])
    minority_rows: list[dict] = []
    for (d1, d2), codes in ranked[1:]:
        for c in codes:
            minority_rows.append(
                {
                    "code": str(c).zfill(6),
                    "last_second_date": d1,
                    "last_date": d2,
                    "kind": "tail_pattern",
                }
            )
    for r in fails:
        code = str(r.get("股票代码", "")).zfill(6)
        if not code or code == "000000":
            continue
        minority_rows.append(
            {
                "code": code,
                "last_second_date": "",
                "last_date": "",
                "kind": "fail",
                "reason": str(r.get("原因", "")),
            }
        )

    # 去重：同 code 合并；保留尾部日期 + fail 原因
    by_code: dict[str, dict] = {}
    for row in minority_rows:
        code = row["code"]
        cur = by_code.get(code)
        if cur is None:
            by_code[code] = dict(row)
            continue
        if row.get("last_date"):
            cur["last_second_date"] = row["last_second_date"]
            cur["last_date"] = row["last_date"]
        if row.get("kind") == "fail":
            cur["kind"] = "fail"
            if row.get("reason"):
                cur["reason"] = row["reason"]
        elif row.get("reason") and not cur.get("reason"):
            cur["reason"] = row["reason"]

    payload = {
        "trade_date": trade_date,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "majority": {
            "last_second_date": majority_key[0],
            "last_date": majority_key[1],
            "count": len(majority_codes),
        },
        "minority": sorted(by_code.values(), key=lambda x: x["code"]),
        "hint": (
            "日更不再自动重拉。修复："
            ".venv/bin/python update_daily.py --repair-minority --sqlite"
        ),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    # 同步一份到 daily_raw 便于肉眼翻
    mirror = (out_dir or OUT_DIR) / "_tail_minority.json"
    try:
        mirror.parent.mkdir(parents=True, exist_ok=True)
        mirror.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    except Exception:
        pass
    log(
        f"\n少数派/失败共 {len(by_code)} 支，已标记 → {path}\n"
        f"  修复: .venv/bin/python update_daily.py --repair-minority --sqlite"
    )
    if by_code:
        log("  " + ", ".join(sorted(by_code)))
    return path


def read_minority_mark(out_dir: Path | None = None) -> dict:
    path = minority_mark_path(out_dir)
    if not path.exists():
        alt = (out_dir or OUT_DIR) / "_tail_minority.json"
        path = alt if alt.exists() else path
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def list_minority(out_dir: Path | None = None) -> None:
    info = read_minority_mark(out_dir)
    if not info:
        log("无少数派标记（齐整或尚未日更）。")
        return
    maj = info.get("majority") or {}
    log(
        f"trade_date={info.get('trade_date')}  updated_at={info.get('updated_at')}\n"
        f"majority count={maj.get('count')} "
        f"{{last_second_date: '{maj.get('last_second_date')}', "
        f"last_date: '{maj.get('last_date')}'}}"
    )
    rows = info.get("minority") or []
    log(f"minority/fail n={len(rows)}")
    for r in rows:
        extra = f"  {r.get('reason')}" if r.get("reason") else ""
        log(
            f"  {r.get('code')}  kind={r.get('kind')}  "
            f"tail=({r.get('last_second_date')},{r.get('last_date')}){extra}"
        )


def repair_minority_codes(
    codes: list[str],
    out_dir: Path,
    trade_date: str,
    spot: pd.DataFrame | None = None,
    *,
    use_sqlite: bool = False,
) -> None:
    """对指定代码全量重拉并补当日（手动修复入口）。"""
    codes = [str(c).zfill(6) for c in codes if str(c).strip()]
    if not codes:
        log("无待修复代码。")
        return
    if spot is None:
        log("拉取全市场截面供补当日...")
        spot = fetch_market_spot()
    spot_map = {str(r["代码"]).zfill(6): r for _, r in spot.iterrows()}
    log(f"开始修复 {len(codes)} 支：{', '.join(codes)}")
    for i, code in enumerate(codes, 1):
        if not is_main_board(code):
            log(f"[{i}/{len(codes)}] skip non-main {code}")
            continue
        result = refetch_full_and_patch(
            code, out_dir, trade_date, spot_map.get(code), use_sqlite=use_sqlite
        )
        log(f"[{i}/{len(codes)}] refetch {code} -> {result}")
        time.sleep(0.5)


def repair_minority_from_mark(
    out_dir: Path,
    *,
    use_sqlite: bool = False,
    codes: list[str] | None = None,
    trade_date: str = "",
) -> None:
    info = read_minority_mark(out_dir)
    if codes:
        todo = [str(c).zfill(6) for c in codes]
    else:
        todo = [str(r["code"]).zfill(6) for r in (info.get("minority") or [])]
    if not todo:
        log("标记为空且未指定 --codes，退出。")
        return
    td = trade_date or str(info.get("trade_date") or "") or latest_trade_date()
    repair_minority_codes(todo, out_dir, td, use_sqlite=use_sqlite)
    # 修完后根据当前尾部重写/清除标记
    groups = collect_tail_patterns_sqlite() if use_sqlite else collect_tail_patterns(out_dir)
    write_minority_mark(td, groups, [], out_dir=out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="东财截面日更 daily_raw / SQLite")
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
        "--sqlite",
        action="store_true",
        help="写入 data/db/daily.db（批量 upsert，远快于重写 CSV）",
    )
    parser.add_argument(
        "--also-csv",
        action="store_true",
        help="与 --sqlite 联用：同时写 CSV（较慢，双写）",
    )
    parser.add_argument(
        "--list-minority",
        action="store_true",
        help="列出少数派/失败标记后退出",
    )
    parser.add_argument(
        "--repair-minority",
        action="store_true",
        help="按标记（或 --codes）全量重拉修复，不跑全日更",
    )
    parser.add_argument(
        "--codes",
        default="",
        help="配合 --repair-minority：逗号分隔代码，覆盖标记列表",
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

    if args.list_minority:
        list_minority(out_dir)
        return

    if args.repair_minority:
        codes = [c.strip() for c in args.codes.split(",") if c.strip()] or None
        td = normalize_date(args.date) if args.date else ""
        repair_minority_from_mark(
            out_dir, use_sqlite=use_sqlite, codes=codes, trade_date=td
        )
        return

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
    log("拉取全市场行情（东财主源 / 新浪备用）...")
    spot = fetch_market_spot()
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

    groups = defaultdict(list)
    for r in tails:
        groups[(r["last_second_date"], r["last_date"])].append(r["股票代码"])
    groups = dict(groups)
    n_patterns = len(groups)

    # 不再自动重拉；只打标记
    write_minority_mark(trade_date, groups, fails, out_dir=out_dir)
    if n_patterns <= 1 and not fails:
        log("尾部日期模式唯一，数据齐整。")

    total_sec = time.time() - t0
    timing_path = record_timing(
        out_dir,
        {
            "time": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "trade_date": trade_date,
            "force": args.force,
            "spot_sec": round(spot_sec, 2),
            "update_sec": round(upd_sec, 2),
            "repair_sec": 0.0,
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
