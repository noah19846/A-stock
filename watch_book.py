"""
短线观察簿：把「观察」单独落库，每日用**独立升级规则**复检（不是再跑一遍 default）。

急涨浅回：距20高回到 -18%～-8%、收盘贴回 MA20、止跌 → 升级可短打。
近画像观察：阴线后收盘贴 MA5、硬条件不太差 → 升级。
不要求当天再出现 default 可短打画像（消化期本来就不会再长成那个样）。

剔除：已偏强 / 条件散尽 / 满 12 个交易日。

存：data/db/daily.db → watch_book + watch_review
导出：data/watch_book/open.csv、reviews/YYYY-MM-DD.csv
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import pandas as pd

import daily_db
from analyze_short_burst_features import (
    features_at,
    impulse_watch_when,
    is_impulse_digesting,
    is_impulse_leftover,
    load_maps,
    watch_promote_status,
)
from short_burst_screener import load_bands, verdict_from_feat

ROOT = Path(__file__).resolve().parent
BANDS = ROOT / "data" / "short_burst_feature_bands.json"
OUT_DIR = ROOT / "data" / "watch_book"
KIND = "short"
CAL_CODE = "000001"


def log(msg: str) -> None:
    print(msg, flush=True)


def _conn(db_path: Path | None = None) -> sqlite3.Connection:
    return daily_db.init_db(daily_db.connect(db_path))


def trading_days_inclusive(conn: sqlite3.Connection, d0: str, d1: str) -> int:
    """用 000001 日历数交易日（含两端）。"""
    n = conn.execute(
        """
        SELECT COUNT(DISTINCT trade_date) FROM daily_bars
        WHERE code=? AND trade_date>=? AND trade_date<=?
        """,
        (CAL_CODE, d0, d1),
    ).fetchone()[0]
    if int(n) > 0:
        return int(n)
    n = conn.execute(
        """
        SELECT COUNT(DISTINCT trade_date) FROM daily_bars
        WHERE trade_date>=? AND trade_date<=?
        """,
        (d0, d1),
    ).fetchone()[0]
    return max(int(n), 1)


def open_rows(conn: sqlite3.Connection, kind: str = KIND) -> list[sqlite3.Row]:
    return list(
        conn.execute(
            """
            SELECT * FROM watch_book
            WHERE kind=? AND status='open'
            ORDER BY opened_date, code
            """,
            (kind,),
        )
    )


def _metric(feat: dict | None, key: str) -> float | None:
    if not feat:
        return None
    v = feat.get(key)
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f != f:
        return None
    return f


def opened_reason_for(feat: dict, stage: str, bands: dict) -> str:
    if is_impulse_leftover(feat, bands):
        return "impulse_leftover"
    if is_impulse_digesting(feat, bands):
        return "impulse_digesting"
    if stage == "观察":
        return "near_setup"
    return "other"


def decide_action(
    *,
    stage: str,
    can_trade: bool,
    feat: dict,
    bands: dict,
    days_open: int,
    max_days: int,
    when: str,
    in_today_watch: bool,
    opened_reason: str | None,
    hard_score: float | None,
) -> tuple[str, str]:
    """观察簿动作。升级走独立规则，不要求 default 可短打。"""
    ready, promo_note, _gaps = watch_promote_status(
        feat, bands, opened_reason, hard_score=hard_score
    )
    if ready:
        return "upgrade", promo_note
    if days_open >= max_days:
        return "drop_timeout", f"观察满{max_days}个交易日未升级，剔除"
    d20 = feat.get("距20日高点%")
    floor = float(
        ((bands.get("watch_promote") or {}).get("impulse") or {}).get(
            "dist20_high_floor", -18
        )
    )
    if (opened_reason or "").startswith("impulse") and d20 is not None:
        try:
            if float(d20) < floor:
                return "drop_gone", f"相对20日高回撤过深（{float(d20):.1f}%<{floor:.0f}%），观察结构坏了"
        except (TypeError, ValueError):
            pass
    if stage == "已偏强":
        return "drop_hot", "已偏强/贴20高，错过浅回买点，剔除"
    if (opened_reason or "").startswith("impulse"):
        return "stay", promo_note if promo_note.startswith("观察簿未升级") else impulse_watch_when(feat, bands)
    if stage == "观察" or opened_reason == "near_setup":
        if promo_note.startswith("观察簿未升级"):
            return "stay", promo_note
        return "stay", when or "仍接近短线画像，继续观察"
    if in_today_watch:
        return "stay", "当日短线池仍列观察，继续用观察簿规则等触发"
    # 已在册但结构散尽：剔除
    return "drop_gone", promo_note or "画像消失且观察簿条件未到，剔除"


def _evaluate_code(code: str, asof: str, bands: dict, name_map, ind_map):
    import daily_cache

    code = str(code).zfill(6)
    name = name_map.get(code, "")
    industry = ind_map.get(code, "")
    df = daily_cache.get(code, asof=asof)
    if df is None or df.empty:
        return None, None, None
    i = len(df) - 1
    asof_s = str(df["日期"].iloc[i].date())
    feat = features_at(df, i)
    if feat is None:
        return None, None, None
    feat = dict(feat)
    try:
        feat["今日涨跌%"] = float(df["ret"].iloc[i]) * 100.0
    except Exception:
        pass
    v = verdict_from_feat(code, name, industry, asof_s, df, i, feat, bands)
    return v, feat, name


def _watch_bands(bands: dict) -> dict:
    """观察簿不设流通市值上下限；市值限制仍由短线可交易阶段执行。"""
    out = dict(bands)
    out["mv_yi"] = [0.0, 1.0e12]
    return out


def _upsert_review(
    conn: sqlite3.Connection,
    *,
    review_date: str,
    code: str,
    kind: str,
    opened_date: str,
    action: str,
    stage: str | None,
    note: str,
    opened_reason: str | None,
    feat: dict | None,
    hard: float | None,
    soft: float | None,
) -> None:
    conn.execute(
        """
        INSERT INTO watch_review(
          review_date, code, kind, opened_date, action, stage, note, opened_reason,
          dist60_low_pct, ret20_pct, dist20_high_pct, px_ma20, hard_score, soft_score
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(review_date, code, kind, opened_date) DO UPDATE SET
          action=excluded.action,
          stage=excluded.stage,
          note=excluded.note,
          opened_reason=excluded.opened_reason,
          dist60_low_pct=excluded.dist60_low_pct,
          ret20_pct=excluded.ret20_pct,
          dist20_high_pct=excluded.dist20_high_pct,
          px_ma20=excluded.px_ma20,
          hard_score=excluded.hard_score,
          soft_score=excluded.soft_score
        """,
        (
            review_date,
            code,
            kind,
            opened_date,
            action,
            stage,
            note,
            opened_reason,
            _metric(feat, "距60日低点%"),
            _metric(feat, "前20日涨幅%"),
            _metric(feat, "距20日高点%"),
            _metric(feat, "收盘/MA20"),
            hard,
            soft,
        ),
    )


def review_day(
    asof: str,
    *,
    short_stocks: list[dict] | None = None,
    day_dir: Path | None = None,
    bands_path: Path | None = None,
    db_path: Path | None = None,
    kind: str = KIND,
) -> dict:
    """
    复检观察簿 + 纳入当日短线「观察」。
    short_stocks: run_daily_pool 的 short HTML 列表（advice=观察 的会尝试入簿）。
    """
    asof = str(asof)[:10]
    bands = load_bands(bands_path or BANDS)
    watch_bands = _watch_bands(bands)
    wb = bands.get("watch_book") or {}
    max_days = int(wb.get("max_days", 12))
    name_map, ind_map = load_maps()

    today_watch = set()
    name_hint: dict[str, str] = {}
    if short_stocks:
        for s in short_stocks:
            if str(s.get("advice") or s.get("stage") or "") != "观察":
                continue
            c = str(s.get("code") or "").zfill(6)
            today_watch.add(c)
            if s.get("name"):
                name_hint[c] = str(s["name"])

    conn = _conn(db_path)
    counts = {
        "enroll": 0,
        "stay": 0,
        "upgrade": 0,
        "drop_hot": 0,
        "drop_gone": 0,
        "drop_timeout": 0,
        "skip": 0,
    }
    review_rows: list[dict] = []
    promoted: list[dict] = []

    try:
        open_map = {
            str(r["code"]).zfill(6): r for r in open_rows(conn, kind=kind)
        }
        codes = sorted(set(open_map) | today_watch)

        for code in codes:
            v, feat, name = _evaluate_code(code, asof, watch_bands, name_map, ind_map)
            if v is None or feat is None:
                counts["skip"] += 1
                continue
            name = name or name_hint.get(code, "") or v.name

            row = open_map.get(code)
            in_today = code in today_watch
            if row is None:
                # 当日短线观察名单入簿（含 strict/r3 落选、急涨浅回）
                if not in_today:
                    counts["skip"] += 1
                    continue
                reason = opened_reason_for(feat, v.stage, bands)
                days_open = 1
                conn.execute(
                    """
                    INSERT INTO watch_book(
                      code, kind, opened_date, name, opened_reason, status,
                      last_review, last_stage, last_note, days_open
                    ) VALUES (?,?,?,?,?,'open',?,?,?,?)
                    """,
                    (
                        code,
                        kind,
                        asof,
                        name,
                        reason,
                        asof,
                        v.stage,
                        "新入观察簿",
                        days_open,
                    ),
                )
                counts["enroll"] += 1
                opened_date = asof
                opened_reason = reason
                is_new = True
            else:
                opened_date = str(row["opened_date"])[:10]
                opened_reason = row["opened_reason"] or opened_reason_for(
                    feat, v.stage, bands
                )
                days_open = trading_days_inclusive(conn, opened_date, asof)
                is_new = False

            action, note = decide_action(
                stage=v.stage,
                can_trade=bool(v.can_trade),
                feat=feat,
                bands=watch_bands,
                days_open=days_open,
                max_days=max_days,
                when=v.when,
                in_today_watch=in_today,
                opened_reason=opened_reason,
                hard_score=float(v.hard_score) if v.hard_score == v.hard_score else None,
            )
            if is_new and action == "upgrade":
                action = "stay"
                note = f"新入（{opened_reason}），当日不算升级。{note}"
            elif is_new and action.startswith("drop"):
                pass
            elif is_new and action == "stay":
                note = f"新入（{opened_reason}）。{note}"

            counts[action] = counts.get(action, 0) + 1

            if action == "upgrade":
                status, close_action = "promoted", action
                closed = asof
                promoted.append(
                    {
                        "code": code,
                        "name": name,
                        "when": note,
                        "stage": "可短打",
                        "can_trade": True,
                        "opened_reason": opened_reason,
                        "score": v.score,
                        "hard_score": v.hard_score,
                        "距60日低点%": _metric(feat, "距60日低点%"),
                        "前20日涨幅%": _metric(feat, "前20日涨幅%"),
                        "距20日高点%": _metric(feat, "距20日高点%"),
                        "收盘/MA20": _metric(feat, "收盘/MA20"),
                    }
                )
            elif action.startswith("drop"):
                status, close_action = "dropped", action
                closed = asof
            else:
                status, close_action, closed = "open", None, None

            conn.execute(
                """
                UPDATE watch_book SET
                  name=?,
                  opened_reason=COALESCE(?, opened_reason),
                  status=?,
                  closed_date=?,
                  close_action=?,
                  last_review=?,
                  last_stage=?,
                  last_note=?,
                  days_open=?
                WHERE code=? AND kind=? AND opened_date=?
                """,
                (
                    name,
                    opened_reason,
                    status,
                    closed,
                    close_action,
                    asof,
                    v.stage,
                    note[:500] if note else "",
                    days_open,
                    code,
                    kind,
                    opened_date,
                ),
            )
            _upsert_review(
                conn,
                review_date=asof,
                code=code,
                kind=kind,
                opened_date=opened_date,
                action="enroll" if is_new else action,
                stage=v.stage,
                note=note[:500] if note else "",
                opened_reason=opened_reason,
                feat=feat,
                hard=float(v.hard_score) if v.hard_score == v.hard_score else None,
                soft=float(v.score) if v.score == v.score else None,
            )
            # 新入若同日又 drop/upgrade，补一条最终动作（便于读 reviews）
            if is_new and action != "stay":
                _upsert_review(
                    conn,
                    review_date=asof,
                    code=code,
                    kind=kind,
                    opened_date=opened_date,
                    action=action,
                    stage=v.stage,
                    note=note[:500] if note else "",
                    opened_reason=opened_reason,
                    feat=feat,
                    hard=float(v.hard_score) if v.hard_score == v.hard_score else None,
                    soft=float(v.score) if v.score == v.score else None,
                )

            review_rows.append(
                {
                    "code": code,
                    "name": name,
                    "opened_date": opened_date,
                    "days_open": days_open,
                    "action": "enroll" if is_new and action == "stay" else action,
                    "stage": v.stage,
                    "opened_reason": opened_reason,
                    "note": note,
                    "距60日低点%": _metric(feat, "距60日低点%"),
                    "前20日涨幅%": _metric(feat, "前20日涨幅%"),
                    "距20日高点%": _metric(feat, "距20日高点%"),
                    "收盘/MA20": _metric(feat, "收盘/MA20"),
                    "hard": v.hard_score,
                    "soft": v.score,
                }
            )

        conn.commit()
    finally:
        conn.close()

    export_paths = export_snapshots(asof, day_dir=day_dir, db_path=db_path, kind=kind)
    n_open = int(counts["enroll"] + counts["stay"] - counts["upgrade"] - counts["drop_hot"] - counts["drop_gone"] - counts["drop_timeout"])
    # enroll+stay can double-count new stays; n_open from file is safer
    try:
        open_df = pd.read_csv(export_paths["open"], dtype={"code": str})
        n_open = 0 if open_df.empty else len(open_df)
    except Exception:
        n_open = max(n_open, 0)
    log(
        f"  观察簿: 新入 {counts['enroll']} / 留观 {counts['stay']} / "
        f"升级 {counts['upgrade']} / 剔除 "
        f"{counts['drop_hot'] + counts['drop_gone'] + counts['drop_timeout']}"
        f"（偏强{counts['drop_hot']} 消失{counts['drop_gone']} 超时{counts['drop_timeout']}）"
        f" / 在册 {n_open} → {export_paths['open']}"
    )
    return {
        "counts": counts,
        "n_open": n_open,
        "promoted": promoted,
        **{k: str(v) for k, v in export_paths.items()},
    }


def enroll_new_only(
    asof: str,
    *,
    short_stocks: list[dict] | None = None,
    bands_path: Path | None = None,
    db_path: Path | None = None,
    kind: str = KIND,
) -> dict:
    """仅把当日短线「观察」增量入簿：已在册 open 不动、不复检升降级。"""
    asof = str(asof)[:10]
    bands = load_bands(bands_path or BANDS)
    name_map, ind_map = load_maps()

    today_watch: list[tuple[str, str]] = []
    if short_stocks:
        for s in short_stocks:
            if str(s.get("advice") or s.get("stage") or "") != "观察":
                continue
            c = str(s.get("code") or "").zfill(6)
            today_watch.append((c, str(s.get("name") or "")))

    conn = _conn(db_path)
    enrolled = 0
    skipped = 0
    try:
        open_codes = {
            str(r["code"]).zfill(6) for r in open_rows(conn, kind=kind)
        }
        for code, name_hint in today_watch:
            if code in open_codes:
                skipped += 1
                continue
            v, feat, name = _evaluate_code(code, asof, bands, name_map, ind_map)
            if v is None or feat is None:
                skipped += 1
                continue
            name = name or name_hint or v.name
            reason = opened_reason_for(feat, v.stage, bands)
            conn.execute(
                """
                INSERT INTO watch_book(
                  code, kind, opened_date, name, opened_reason, status,
                  last_review, last_stage, last_note, days_open
                ) VALUES (?,?,?,?,?,'open',?,?,?,?)
                """,
                (
                    code,
                    kind,
                    asof,
                    name,
                    reason,
                    asof,
                    v.stage,
                    f"收盘增量入簿（{reason}）",
                    1,
                ),
            )
            _upsert_review(
                conn,
                review_date=asof,
                code=code,
                kind=kind,
                opened_date=asof,
                action="enroll",
                stage=v.stage,
                note=f"收盘增量入簿（{reason}）",
                opened_reason=reason,
                feat=feat,
                hard=float(v.hard_score) if v.hard_score == v.hard_score else None,
                soft=float(v.score) if v.score == v.score else None,
            )
            open_codes.add(code)
            enrolled += 1
        conn.commit()
    finally:
        conn.close()

    export_snapshots(asof, db_path=db_path, kind=kind)
    log(f"  观察簿增量入簿: 新入 {enrolled} / 已在册跳过 {skipped}")
    return {"enroll": enrolled, "skip": skipped}


def export_snapshots(
    asof: str | None = None,
    *,
    day_dir: Path | None = None,
    db_path: Path | None = None,
    kind: str = KIND,
) -> dict[str, Path]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "reviews").mkdir(parents=True, exist_ok=True)
    conn = _conn(db_path)
    try:
        opens = [dict(r) for r in open_rows(conn, kind=kind)]
        open_df = pd.DataFrame(opens)
        if open_df.empty:
            open_df = pd.DataFrame(
                columns=[
                    "code",
                    "kind",
                    "opened_date",
                    "name",
                    "opened_reason",
                    "status",
                    "closed_date",
                    "close_action",
                    "last_review",
                    "last_stage",
                    "last_note",
                    "days_open",
                ]
            )
        open_path = OUT_DIR / "open.csv"
        open_df.to_csv(open_path, index=False, encoding="utf-8-sig")

        review_path = OUT_DIR / "reviews" / f"{asof}.csv" if asof else None
        if asof:
            rows = conn.execute(
                """
                SELECT * FROM watch_review
                WHERE review_date=? AND kind=?
                ORDER BY action, code
                """,
                (asof, kind),
            ).fetchall()
            pd.DataFrame([dict(r) for r in rows]).to_csv(
                review_path, index=False, encoding="utf-8-sig"
            )
            if day_dir is not None:
                day_dir.mkdir(parents=True, exist_ok=True)
                day_csv = day_dir / "watch_book.csv"
                open_df.to_csv(day_csv, index=False, encoding="utf-8-sig")
    finally:
        conn.close()
    out = {"open": open_path}
    if review_path:
        out["review"] = review_path
    return out


def open_as_signal_df(
    *,
    db_path: Path | None = None,
    kind: str = KIND,
) -> pd.DataFrame:
    """在册观察 → 与短线 signals 同结构的 DataFrame，供 HTML tab 使用。"""
    conn = _conn(db_path)
    try:
        rows = open_rows(conn, kind=kind)
    finally:
        conn.close()
    if not rows:
        return pd.DataFrame()
    out = []
    for r in rows:
        reason = str(r["opened_reason"] or "")
        impulse = reason.startswith("impulse")
        note = str(r["last_note"] or "").strip()
        if not note:
            note = "观察簿在册，等待独立升级条件"
        tags = "急涨浅回" if impulse else "观察簿"
        out.append(
            {
                "code": str(r["code"]).zfill(6),
                "name": r["name"] or "",
                "stage": "观察",
                "can_trade": False,
                "when": note,
                "opened_date": r["opened_date"],
                "opened_reason": reason,
                "days_open": r["days_open"],
                "last_stage": r["last_stage"],
                "急涨浅回提醒": impulse,
                "strategy_tags": tags,
                "strategy_ids": "watch",
            }
        )
    return pd.DataFrame(out)


def write_watch_panel_csv(day_dir: Path, df: pd.DataFrame | None = None) -> Path:
    """写入 day_dir/watch/signals.csv，便于 --rebuild-html。"""
    out_dir = day_dir / "watch"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "signals.csv"
    if df is None:
        df = open_as_signal_df()
    if df is None or df.empty:
        pd.DataFrame(
            columns=[
                "code",
                "name",
                "stage",
                "can_trade",
                "when",
                "opened_date",
                "opened_reason",
                "days_open",
                "急涨浅回提醒",
                "strategy_tags",
                "strategy_ids",
            ]
        ).to_csv(path, index=False, encoding="utf-8-sig")
    else:
        df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def print_status(db_path: Path | None = None, kind: str = KIND) -> None:
    conn = _conn(db_path)
    try:
        rows = open_rows(conn, kind=kind)
        if not rows:
            log("观察簿空（无 open）")
            return
        log(f"在册观察 {len(rows)} 只：")
        log(
            f"{'代码':<8} {'名称':<10} {'入簿':<12} {'天数':>4} {'理由':<18} {'末次阶段':<8} 备注"
        )
        for r in rows:
            note = (r["last_note"] or "")[:60]
            log(
                f"{r['code']:<8} {(r['name'] or ''):<10} {r['opened_date']:<12} "
                f"{int(r['days_open'] or 0):>4} {(r['opened_reason'] or ''):<18} "
                f"{(r['last_stage'] or ''):<8} {note}"
            )
    finally:
        conn.close()


def print_history(code: str, db_path: Path | None = None, kind: str = KIND) -> None:
    code = str(code).zfill(6)
    conn = _conn(db_path)
    try:
        rows = conn.execute(
            """
            SELECT * FROM watch_review
            WHERE code=? AND kind=?
            ORDER BY review_date, opened_date
            """,
            (code, kind),
        ).fetchall()
        if not rows:
            log(f"{code} 无观察簿记录")
            return
        for r in rows:
            log(
                f"{r['review_date']}  {r['action']:<12}  stage={r['stage']}  "
                f"open={r['opened_date']}  {r['note']}"
            )
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="短线观察簿")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_r = sub.add_parser("review", help="按日复检（可从当日 short signals 扩入）")
    p_r.add_argument("--date", required=True, help="YYYY-MM-DD")
    p_r.add_argument(
        "--from-pool",
        action="store_true",
        help="从 signal_pool/日期/short/signals.csv 取当日观察入簿",
    )
    p_r.add_argument("--bands", default=str(BANDS))

    sub.add_parser("status", help="打印当前 open")
    p_h = sub.add_parser("history", help="某代码复检史")
    p_h.add_argument("code")

    args = parser.parse_args()
    if args.cmd == "status":
        print_status()
        return
    if args.cmd == "history":
        print_history(args.code)
        return

    # review
    short_stocks: list[dict] | None = None
    day_dir = None
    if args.from_pool:
        day_dir = ROOT / "signal_pool" / args.date
        sig = day_dir / "short" / "signals.csv"
        if sig.exists():
            df = pd.read_csv(sig, dtype={"code": str})
            short_stocks = []
            for _, row in df.iterrows():
                short_stocks.append(
                    {
                        "code": str(row["code"]).zfill(6),
                        "name": str(row.get("name") or ""),
                        "advice": str(row.get("stage") or ""),
                    }
                )
        else:
            log(f"无 {sig}，仅复检已在册")
    review_day(
        args.date,
        short_stocks=short_stocks,
        day_dir=day_dir,
        bands_path=Path(args.bands),
    )
    print_status()


if __name__ == "__main__":
    main()
