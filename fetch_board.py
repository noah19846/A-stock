"""
拉取东财行业/概念板块及成分，缓存到 data/。

输出：
  data/board_industry.csv     行业板块列表
  data/board_concept.csv      概念板块列表
  data/stock_industry.csv     股票 -> 行业板块（一对一，后者覆盖）
  data/stock_concept.csv      股票 -> 概念板块（一对多）

用法：
  .venv/bin/python fetch_board.py
  .venv/bin/python fetch_board.py --skip-concept
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

CLIST_URL = "https://push2delay.eastmoney.com/api/qt/clist/get"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "Referer": "https://quote.eastmoney.com/",
}


def _session() -> requests.Session:
    s = requests.Session()
    s.trust_env = False
    s.headers.update(HEADERS)
    return s


def fetch_clist(fs: str, fields: str = "f12,f14,f2,f3,f4,f8,f20,f104,f105") -> pd.DataFrame:
    session = _session()
    params = {
        "pn": "1",
        "pz": "100",
        "po": "1",
        "np": "1",
        "ut": "bd1d9ddb04089700cf9c27f6f7426281",
        "fltt": "2",
        "invt": "2",
        "fid": "f3",
        "fs": fs,
        "fields": fields,
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
        time.sleep(0.05)
    return pd.DataFrame(rows)


def fetch_board_list(board_type: str) -> pd.DataFrame:
    """board_type: industry | concept"""
    # t:2 行业, t:3 概念
    t = "2" if board_type == "industry" else "3"
    raw = fetch_clist(fs=f"m:90 t:{t} f:!50")
    df = pd.DataFrame(
        {
            "板块代码": raw["f12"].astype(str),
            "板块名称": raw["f14"].astype(str),
            "最新价": pd.to_numeric(raw.get("f2"), errors="coerce"),
            "涨跌幅": pd.to_numeric(raw.get("f3"), errors="coerce"),
            "涨跌额": pd.to_numeric(raw.get("f4"), errors="coerce"),
            "换手率": pd.to_numeric(raw.get("f8"), errors="coerce"),
            "总市值": pd.to_numeric(raw.get("f20"), errors="coerce"),
            "上涨家数": pd.to_numeric(raw.get("f104"), errors="coerce"),
            "下跌家数": pd.to_numeric(raw.get("f105"), errors="coerce"),
        }
    )
    return df.drop_duplicates(subset=["板块代码"]).reset_index(drop=True)


def fetch_board_cons(board_code: str) -> pd.DataFrame:
    raw = fetch_clist(
        fs=f"b:{board_code} f:!50",
        fields="f12,f14,f2,f3",
    )
    if raw.empty:
        return pd.DataFrame(columns=["股票代码", "股票名称"])
    return pd.DataFrame(
        {
            "股票代码": raw["f12"].astype(str).str.zfill(6),
            "股票名称": raw["f14"].astype(str),
        }
    )


def build_membership(boards: pd.DataFrame, kind: str) -> pd.DataFrame:
    """遍历板块拉成分。kind=industry/concept。"""
    frames = []
    total = len(boards)
    for i, row in boards.iterrows():
        code = row["板块代码"]
        name = row["板块名称"]
        try:
            cons = fetch_board_cons(code)
            if cons.empty:
                print(f"[{i+1}/{total}] empty {code} {name}", flush=True)
                continue
            cons = cons.copy()
            cons["板块代码"] = code
            cons["板块名称"] = name
            frames.append(cons)
            print(f"[{i+1}/{total}] ok {code} {name} n={len(cons)}", flush=True)
        except Exception as e:
            print(f"[{i+1}/{total}] fail {code} {name} {e}", flush=True)
        time.sleep(0.08)

    if not frames:
        return pd.DataFrame(columns=["股票代码", "股票名称", "板块代码", "板块名称"])
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="拉取东财行业/概念板块成分")
    parser.add_argument("--skip-concept", action="store_true", help="只拉行业，不拉概念")
    parser.add_argument("--only-concept", action="store_true", help="只拉概念（复用已有行业列表）")
    args = parser.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    if not args.only_concept:
        print("拉取行业板块列表...", flush=True)
        ind = fetch_board_list("industry")
        ind_path = DATA / "board_industry.csv"
        ind.to_csv(ind_path, index=False, encoding="utf-8-sig")
        print(f"行业板块 {len(ind)} -> {ind_path}", flush=True)

        print("拉取行业成分...", flush=True)
        ind_mem = build_membership(ind, "industry")
        stock_ind = (
            ind_mem.sort_values(["股票代码", "板块代码"])
            .drop_duplicates(subset=["股票代码"], keep="last")
            [["股票代码", "股票名称", "板块代码", "板块名称"]]
            .rename(columns={"板块名称": "行业板块", "板块代码": "行业代码"})
            .reset_index(drop=True)
        )
        stock_ind_path = DATA / "stock_industry.csv"
        stock_ind.to_csv(stock_ind_path, index=False, encoding="utf-8-sig")
        print(f"股票-行业 {len(stock_ind)} -> {stock_ind_path}", flush=True)

    if not args.skip_concept:
        print("拉取概念板块列表...", flush=True)
        concept = fetch_board_list("concept")
        concept_path = DATA / "board_concept.csv"
        concept.to_csv(concept_path, index=False, encoding="utf-8-sig")
        print(f"概念板块 {len(concept)} -> {concept_path}", flush=True)

        print("拉取概念成分（较慢）...", flush=True)
        con_mem = build_membership(concept, "concept")
        con_mem = con_mem.rename(
            columns={"板块名称": "概念板块", "板块代码": "概念代码"}
        )
        con_path = DATA / "stock_concept.csv"
        con_mem.to_csv(con_path, index=False, encoding="utf-8-sig")
        print(f"股票-概念 行数 {len(con_mem)} -> {con_path}", flush=True)

    print(f"完成，耗时 {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
