"""
日线 SQLite 存储（data/db/daily.db）。

用法：
  .venv/bin/python daily_db.py migrate          # 从 data/daily_raw/*.csv 导入
  .venv/bin/python daily_db.py migrate --limit 50
  .venv/bin/python daily_db.py stats
"""

from __future__ import annotations

import argparse
import sqlite3
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
DAILY_DIR = ROOT / "data" / "daily_raw"
DB_DIR = ROOT / "data" / "db"
DB_PATH = DB_DIR / "daily.db"

COLS_CN = [
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

# DB columns (English) <-> CSV
COL_MAP = {
    "股票代码": "code",
    "日期": "trade_date",
    "开盘": "open",
    "最高": "high",
    "最低": "low",
    "收盘": "close",
    "成交量": "volume",
    "成交额": "amount",
    "流通股本": "float_shares",
    "换手率": "turnover",
    "后复权因子": "hfq_factor",
}
COL_MAP_REV = {v: k for k, v in COL_MAP.items()}

SCHEMA = """
CREATE TABLE IF NOT EXISTS daily_bars (
  code         TEXT NOT NULL,
  trade_date   TEXT NOT NULL,
  open         REAL,
  high         REAL,
  low          REAL,
  close        REAL,
  volume       REAL,
  amount       REAL,
  float_shares REAL,
  turnover     REAL,
  hfq_factor   REAL,
  PRIMARY KEY (code, trade_date)
);
CREATE INDEX IF NOT EXISTS idx_daily_bars_code_date
  ON daily_bars(code, trade_date DESC);
CREATE INDEX IF NOT EXISTS idx_daily_bars_date
  ON daily_bars(trade_date);
"""


def log(msg: str) -> None:
    print(msg, flush=True)


def connect(db_path: Path | None = None, *, wal: bool = True) -> sqlite3.Connection:
    path = db_path or DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=60)
    conn.row_factory = sqlite3.Row
    if wal:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA temp_store=MEMORY")
        conn.execute("PRAGMA cache_size=-80000")  # ~80MB
    return conn


def init_db(conn: sqlite3.Connection | None = None) -> sqlite3.Connection:
    own = conn is None
    conn = conn or connect()
    conn.executescript(SCHEMA)
    conn.commit()
    if own:
        return conn
    return conn


def db_exists(db_path: Path | None = None) -> bool:
    p = db_path or DB_PATH
    return p.exists() and p.stat().st_size > 0


def bar_dict_from_cn(row: dict) -> tuple:
    code = str(row["股票代码"]).zfill(6)
    return (
        code,
        str(row["日期"])[:10],
        float(row["开盘"]),
        float(row["最高"]),
        float(row["最低"]),
        float(row["收盘"]),
        float(row["成交量"]),
        float(row["成交额"]),
        float(row["流通股本"]),
        float(row["换手率"]),
        float(row["后复权因子"]),
    )


UPSERT_SQL = """
INSERT INTO daily_bars(
  code, trade_date, open, high, low, close,
  volume, amount, float_shares, turnover, hfq_factor
) VALUES (?,?,?,?,?,?,?,?,?,?,?)
ON CONFLICT(code, trade_date) DO UPDATE SET
  open=excluded.open,
  high=excluded.high,
  low=excluded.low,
  close=excluded.close,
  volume=excluded.volume,
  amount=excluded.amount,
  float_shares=excluded.float_shares,
  turnover=excluded.turnover,
  hfq_factor=excluded.hfq_factor
"""


def upsert_bars(conn: sqlite3.Connection, rows: list[tuple]) -> None:
    if not rows:
        return
    conn.executemany(UPSERT_SQL, rows)


def replace_code_df(conn: sqlite3.Connection, code: str, df: pd.DataFrame) -> int:
    """用整表 DataFrame（中文列）替换某代码全部行情。"""
    code = str(code).zfill(6)
    conn.execute("DELETE FROM daily_bars WHERE code=?", (code,))
    if df.empty:
        conn.commit()
        return 0
    out = df.copy()
    if "股票代码" not in out.columns:
        out["股票代码"] = code
    out["股票代码"] = out["股票代码"].astype(str).str.zfill(6)
    out["日期"] = out["日期"].astype(str).str.slice(0, 10)
    rows = [bar_dict_from_cn(r) for r in out[COLS_CN].to_dict("records")]
    upsert_bars(conn, rows)
    conn.commit()
    return len(rows)


def load_bars_df(
    code: str,
    *,
    limit: int | None = 320,
    db_path: Path | None = None,
    conn: sqlite3.Connection | None = None,
) -> pd.DataFrame:
    """返回中文列名 DataFrame，按日期升序；空则 empty。"""
    code = str(code).zfill(6)
    own = conn is None
    conn = conn or connect(db_path)
    try:
        sql = """
          SELECT code, trade_date, open, high, low, close,
                 volume, amount, float_shares, turnover, hfq_factor
          FROM daily_bars
          WHERE code=?
          ORDER BY trade_date
        """
        if limit and limit > 0:
            # 取最近 limit 根：子查询倒序再正序
            sql = f"""
              SELECT * FROM (
                SELECT code, trade_date, open, high, low, close,
                       volume, amount, float_shares, turnover, hfq_factor
                FROM daily_bars
                WHERE code=?
                ORDER BY trade_date DESC
                LIMIT {int(limit)}
              ) ORDER BY trade_date
            """
        df = pd.read_sql_query(sql, conn, params=(code,))
    finally:
        if own:
            conn.close()
    if df.empty:
        return df
    df = df.rename(columns=COL_MAP_REV)
    return df[COLS_CN]


def load_recent_bars_since(
    cutoff: str,
    *,
    prefixes: tuple[str, ...] | None = None,
    db_path: Path | None = None,
) -> pd.DataFrame:
    """按 trade_date >= cutoff 批量拉（需 idx_daily_bars_date）。英文列。"""
    conn = connect(db_path)
    try:
        where = "trade_date >= ?"
        params: list = [str(cutoff)[:10]]
        if prefixes:
            ors = " OR ".join(["code LIKE ?" for _ in prefixes])
            where = f"({where}) AND ({ors})"
            params.extend(f"{p}%" for p in prefixes)
        sql = f"""
          SELECT code, trade_date, open, high, low, close,
                 volume, amount, float_shares, turnover, hfq_factor
          FROM daily_bars
          WHERE {where}
          ORDER BY code, trade_date
        """
        return pd.read_sql_query(sql, conn, params=params)
    finally:
        conn.close()


def recent_cutoff_date(
    *,
    limit: int = 320,
    sample_code: str = "600000",
    db_path: Path | None = None,
) -> str | None:
    """取样本股倒数第 limit 根的日期，作为近端批量截断。"""
    conn = connect(db_path)
    try:
        row = conn.execute(
            """
            SELECT trade_date FROM daily_bars
            WHERE code=?
            ORDER BY trade_date DESC
            LIMIT 1 OFFSET ?
            """,
            (str(sample_code).zfill(6), int(limit) - 1),
        ).fetchone()
        if row:
            return str(row[0])[:10]
        mx = conn.execute("SELECT MIN(trade_date) FROM daily_bars").fetchone()[0]
        return str(mx)[:10] if mx else None
    finally:
        conn.close()


def list_codes(db_path: Path | None = None) -> list[str]:
    conn = connect(db_path)
    try:
        rows = conn.execute("SELECT DISTINCT code FROM daily_bars ORDER BY code").fetchall()
        return [str(r[0]).zfill(6) for r in rows]
    finally:
        conn.close()


def last_two_by_code(conn: sqlite3.Connection) -> dict[str, dict]:
    """
    code -> {d1, d2, hfq_factor, float_shares}
    d2=最新日, d1=次新日；hfq/float 取自最新日。
    """
    sql = """
      SELECT code, trade_date, hfq_factor, float_shares,
             ROW_NUMBER() OVER (PARTITION BY code ORDER BY trade_date DESC) AS rn
      FROM daily_bars
    """
    # 兼容无窗口函数的旧 SQLite：退化为两轮查询
    try:
        cur = conn.execute(
            """
            WITH ranked AS (
              SELECT code, trade_date, hfq_factor, float_shares,
                     ROW_NUMBER() OVER (
                       PARTITION BY code ORDER BY trade_date DESC
                     ) AS rn
              FROM daily_bars
            )
            SELECT code, trade_date, hfq_factor, float_shares, rn
            FROM ranked WHERE rn <= 2
            """
        )
        rows = cur.fetchall()
    except sqlite3.OperationalError:
        rows = []
        for (code,) in conn.execute("SELECT DISTINCT code FROM daily_bars"):
            sub = conn.execute(
                """
                SELECT trade_date, hfq_factor, float_shares
                FROM daily_bars WHERE code=?
                ORDER BY trade_date DESC LIMIT 2
                """,
                (code,),
            ).fetchall()
            for i, r in enumerate(sub, 1):
                rows.append((code, r[0], r[1], r[2], i))

    out: dict[str, dict] = {}
    for code, trade_date, hfq, shares, rn in rows:
        code = str(code).zfill(6)
        item = out.setdefault(
            code, {"d1": "", "d2": "", "hfq_factor": None, "float_shares": None}
        )
        if int(rn) == 1:
            item["d2"] = str(trade_date)[:10]
            item["hfq_factor"] = float(hfq) if hfq is not None else None
            item["float_shares"] = float(shares) if shares is not None else None
        elif int(rn) == 2:
            item["d1"] = str(trade_date)[:10]
    return out


def stats(db_path: Path | None = None) -> dict:
    if not db_exists(db_path):
        return {"exists": False}
    conn = connect(db_path)
    try:
        n_codes = conn.execute("SELECT COUNT(DISTINCT code) FROM daily_bars").fetchone()[0]
        n_rows = conn.execute("SELECT COUNT(*) FROM daily_bars").fetchone()[0]
        mx = conn.execute("SELECT MAX(trade_date) FROM daily_bars").fetchone()[0]
        mn = conn.execute("SELECT MIN(trade_date) FROM daily_bars").fetchone()[0]
        size = (db_path or DB_PATH).stat().st_size
        return {
            "exists": True,
            "path": str(db_path or DB_PATH),
            "codes": int(n_codes),
            "rows": int(n_rows),
            "min_date": mn,
            "max_date": mx,
            "size_mb": round(size / 1e6, 1),
        }
    finally:
        conn.close()


def migrate_from_csv(
    daily_dir: Path | None = None,
    db_path: Path | None = None,
    *,
    limit: int | None = None,
    rebuild: bool = False,
) -> dict:
    """从 CSV 全量导入 SQLite。"""
    daily_dir = daily_dir or DAILY_DIR
    db_path = db_path or DB_PATH
    t0 = time.time()

    if rebuild and db_path.exists():
        for p in (db_path, Path(str(db_path) + "-wal"), Path(str(db_path) + "-shm")):
            p.unlink(missing_ok=True)

    conn = init_db(connect(db_path))
    files = sorted(
        p
        for p in daily_dir.glob("*.csv")
        if p.name[0].isdigit() and "_" not in p.stem
    )
    if limit:
        files = files[: int(limit)]

    log(f"导入 {len(files)} 个 CSV → {db_path}")
    n_rows = 0
    n_ok = 0
    n_fail = 0
    batch: list[tuple] = []

    for i, path in enumerate(files, 1):
        code = path.stem.zfill(6)
        try:
            df = pd.read_csv(path, dtype={"股票代码": str})
            if df.empty or "日期" not in df.columns:
                n_fail += 1
                continue
            df["股票代码"] = code
            df["日期"] = df["日期"].astype(str).str.slice(0, 10)
            for c in COLS_CN:
                if c not in df.columns:
                    raise ValueError(f"缺列 {c}")
            # 替换该 code
            conn.execute("DELETE FROM daily_bars WHERE code=?", (code,))
            rows = [bar_dict_from_cn(r) for r in df[COLS_CN].to_dict("records")]
            batch.extend(rows)
            n_rows += len(rows)
            n_ok += 1
            if len(batch) >= 50000:
                upsert_bars(conn, batch)
                conn.commit()
                batch.clear()
        except Exception as e:
            n_fail += 1
            if n_fail <= 5:
                log(f"  fail {code}: {e}")
        if i % 200 == 0:
            if batch:
                upsert_bars(conn, batch)
                conn.commit()
                batch.clear()
            log(f"  进度 {i}/{len(files)}  rows={n_rows}")

    if batch:
        upsert_bars(conn, batch)
        conn.commit()
    conn.execute("ANALYZE")
    conn.close()
    info = stats(db_path)
    info.update(
        {
            "imported_files": n_ok,
            "failed_files": n_fail,
            "elapsed_sec": round(time.time() - t0, 1),
        }
    )
    log(
        f"完成：codes={info.get('codes')} rows={info.get('rows')} "
        f"size={info.get('size_mb')}MB 耗时 {info['elapsed_sec']}s"
    )
    return info


def main() -> None:
    parser = argparse.ArgumentParser(description="日线 SQLite")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_m = sub.add_parser("migrate", help="从 daily_raw CSV 导入")
    p_m.add_argument("--limit", type=int, default=0)
    p_m.add_argument("--rebuild", action="store_true", help="删除旧库重建")
    p_m.add_argument("--db", default=str(DB_PATH))
    p_m.add_argument("--daily-dir", default=str(DAILY_DIR))

    p_s = sub.add_parser("stats", help="库统计")
    p_s.add_argument("--db", default=str(DB_PATH))

    args = parser.parse_args()
    if args.cmd == "migrate":
        migrate_from_csv(
            Path(args.daily_dir),
            Path(args.db),
            limit=args.limit or None,
            rebuild=args.rebuild,
        )
    elif args.cmd == "stats":
        info = stats(Path(args.db))
        for k, v in info.items():
            print(f"{k}: {v}")


if __name__ == "__main__":
    main()
