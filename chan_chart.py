"""
单只股票的日 K 缠论标注图（分型、笔、线段、中枢、买卖点）。

用法：
  pip install pandas
  python chan_chart.py --code 000001 --days 250

日线沿用 plot_watch_pool.load_qfq_bars（优先 data/db/daily.db，否则
data/daily_raw/{code}.csv）。``--days`` 与观察池图相同，是回溯的自然日，不是 K 线根数。

缠论计算不自写分型/笔/中枢/买卖点，调用 Vespa314/chan.py（MIT，固定提交
429d6ed3043e27c93a003ba2b10e70a05575e1f5）。首次运行会把该仓库克隆到
``.cache/chan.py``（已在 .gitignore 里）。本仓库没有 requirements.txt，
因此不另建依赖清单；除 pandas 外，chan.py 的形态计算本身不再需要 numpy。

采用的是 chan.py 哪一层（不要和 0.9 文档或 czsc 的信号函数混用）：

- 合并 K 线与分型：``Combiner/KLine_Combiner.py`` 的 ``update_fx`` / ``get_peak_klu``
- 笔：``Bi/BiList.py`` ``update_bi``，配置 ``bi_strict=True``、``bi_fx_check=strict``
- 线段：``Seg/SegListChan.py``（``seg_algo="chan"``，特征序列分型；虚段 ``is_sure=False``）
- 笔中枢：``ZS/ZSList.py`` ``cal_bi_zs``，``zs_algo="over_seg"``（三笔重叠即可，允许跨线段）
  ``CZS.low/high`` 即 ZD/ZG，``peak_low/peak_high`` 即 DD/GG。
  不用默认的 ``normal``：它只在已确认线段内部收反向笔，平安银行最近 250 个自然日
  线段都还是虚段，笔中枢和买卖点会是空的。
- 线段中枢：``CKLine_List.segzs_list`` 只在「线段的线段」里收反向段。样本里每条父段
  只有一段回拉，列表是空的。因此在 chan.py 已经画出的线段上，用 ``CZS`` 和
  ``ZSList.try_construct_zs`` 的同一条件（连续三段 ``min(_high) > max(_low)``，
  之后 ``CZS.try_add_to_end`` 延伸，重叠的用 ``CZS.combine`` 扩展）再扫一遍。
- 买卖点：``BuySellPoint/BSPointList.py`` 的 ``cal_seg_bs1point`` / ``cal_seg_bs2point`` /
  ``cal_seg_bs3point``。笔级别力度用 ``MACD_ALGO.FULL_AREA``，即 ``CBi.Cal_MACD_area``
  （该笔起止之间、与笔方向同色的 MACD 柱面积）。离开笔面积 <= 0.9 * 进入（或前一同向）笔
  才算背驰。0.9 来自 quick_guide 写明的常用阈值；chan.py 代码默认 ``divergence_rate=inf``
  是机器学习取特征时的“保送”，不适合直接画买卖点。
- 线段买卖点沿用库内覆盖：``CSeg.cal_macd_metric`` 只实现 slope/amp，配置里被设成 slope。

上涨 / 下跌 / 盘整中枢不是 chan.py 的字段，按缠中说禅原文套在它算出的中枢上：
同一父段里只有一个中枢为盘整；至少两个且后一个 ZD > 前一个 ZG 为上涨；
后一个 ZG < 前一个 ZD 为下跌。
延伸：中枢内笔（或段）超过 3 且没有被合并。
扩展：``CZS.sub_zs_lst`` 非空（``zs_combine`` 把重叠中枢收成更宽的 ZD/ZG）。
新生：与上一个同级中枢的 ZD/ZG 没有重叠。

也看过 waditu/czsc（Apache-2.0，星标更高）。1.0 起笔和中枢在 Rust 扩展里，
买卖点是 200 多个信号函数，不是这里要的 1/1p/2/2s/3a/3b 对象，依赖也更重，所以没有接入。
GitHub 上其他 chanlun 仓库要么星标很低，要么许可证不明，没有采用。
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "chan_charts"
CHAN_REPO = "https://github.com/Vespa314/chan.py.git"
CHAN_COMMIT = "429d6ed3043e27c93a003ba2b10e70a05575e1f5"
CHAN_CACHE = ROOT / ".cache" / "chan.py"

# quick_guide 的常用背驰阈值。代码默认 inf 会把没背驰的笔也收成一类。
DIVERGENCE_RATE = 0.9

BUY_SHORT = {
    "1": "一买",
    "1p": "盘一买",
    "2": "二买",
    "2s": "类二买",
    "3a": "三买",
    "3b": "三买b",
}
SELL_SHORT = {
    "1": "一卖",
    "1p": "盘一卖",
    "2": "二卖",
    "2s": "类二卖",
    "3a": "三卖",
    "3b": "三卖b",
}
BUY_FULL = {
    "1": "第一类买点（下跌趋势背驰）",
    "1p": "盘整背驰买点",
    "2": "第二类买点",
    "2s": "类二买",
    "3a": "第三类买点（中枢在一类之后）",
    "3b": "第三类买点（中枢在一类之前）",
}
SELL_FULL = {
    "1": "第一类卖点（上涨趋势背驰）",
    "1p": "盘整背驰卖点",
    "2": "第二类卖点",
    "2s": "类二卖",
    "3a": "第三类卖点（中枢在一类之后）",
    "3b": "第三类卖点（中枢在一类之前）",
}
TYPE_ORDER = ("1", "1p", "2", "2s", "3a", "3b")
POINT_COLOR = {
    "1": "#c0392b",
    "1p": "#e67e22",
    "2": "#d35400",
    "2s": "#f5b041",
    "3a": "#6c3483",
    "3b": "#5b2c6f",
}
SELL_COLOR = {
    "1": "#0e6655",
    "1p": "#148f77",
    "2": "#1a7f37",
    "2s": "#27ae60",
    "3a": "#1a5276",
    "3b": "#2471a3",
}


def ensure_chan_py() -> Path:
    """把 chan.py 放到 .cache，不把第三方源码写进本仓库。"""
    marker = CHAN_CACHE / "Chan.py"
    pin = CHAN_CACHE / ".commit"
    if marker.is_file() and pin.is_file() and pin.read_text(encoding="utf-8").strip() == CHAN_COMMIT:
        return CHAN_CACHE
    if CHAN_CACHE.exists():
        shutil.rmtree(CHAN_CACHE)
    CHAN_CACHE.parent.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(["git", "clone", "--depth", "1", CHAN_REPO, str(CHAN_CACHE)])
    head = subprocess.check_output(
        ["git", "-C", str(CHAN_CACHE), "rev-parse", "HEAD"], text=True
    ).strip()
    if head != CHAN_COMMIT:
        subprocess.check_call(
            ["git", "-C", str(CHAN_CACHE), "fetch", "--depth", "1", "origin", CHAN_COMMIT]
        )
        subprocess.check_call(["git", "-C", str(CHAN_CACHE), "checkout", CHAN_COMMIT])
    pin.write_text(CHAN_COMMIT + "\n", encoding="utf-8")
    return CHAN_CACHE


def _import_chan():
    cache = ensure_chan_py()
    if str(cache) not in sys.path:
        sys.path.insert(0, str(cache))
    from Chan import CChan
    from ChanConfig import CChanConfig
    from Common.CEnum import AUTYPE, BI_DIR, DATA_FIELD, FX_TYPE, KL_TYPE
    from Common.CTime import CTime

    return CChan, CChanConfig, AUTYPE, BI_DIR, DATA_FIELD, FX_TYPE, KL_TYPE, CTime


def _bind_local_source(klu_dicts: list[dict]) -> str:
    """把本地 K 线登记成 chan.py 的 custom 数据源。类必须能被 DataAPI.<模块> import。"""
    from DataAPI.CommonStockAPI import CCommonStockApi
    from KLine.KLine_Unit import CKLine_Unit

    class CLocalBars(CCommonStockApi):
        payload = klu_dicts

        def SetBasciInfo(self):
            self.name = str(self.code)
            self.is_stock = True

        @classmethod
        def do_init(cls):
            return None

        @classmethod
        def do_close(cls):
            return None

        def get_kl_data(self):
            for item in self.payload:
                yield CKLine_Unit(item, autofix=True)

    module = types.ModuleType("DataAPI.local_bars")
    module.CLocalBars = CLocalBars
    sys.modules["DataAPI.local_bars"] = module
    return "custom:local_bars.CLocalBars"


def load_bars(code: str, days: int) -> tuple[list[dict], dict]:
    from plot_watch_pool import load_qfq_bars

    loaded = load_qfq_bars(code, days=days)
    if not loaded:
        raise SystemExit(
            f"没有读到 {code} 的日线。请确认 data/db/daily.db 或 data/daily_raw/{code}.csv"
        )
    bars, spot = loaded
    if len(bars) < 30:
        raise SystemExit(f"{code} 在最近 {days} 个自然日里只有 {len(bars)} 根 K 线，不够画笔")
    return bars, spot


def _px(value: float) -> float:
    return round(float(value), 4)


def _date_of(klu) -> str:
    return klu.time.toDateStr("-")


def _dir_name(line) -> str:
    return "up" if line.is_up() else "down"


def _dump_line(line) -> dict:
    return {
        "idx": int(line.idx),
        "dir": _dir_name(line),
        "sure": bool(line.is_sure),
        "start": _date_of(line.get_begin_klu()),
        "end": _date_of(line.get_end_klu()),
        "start_price": _px(line.get_begin_val()),
        "end_price": _px(line.get_end_val()),
        "high": _px(line._high()),
        "low": _px(line._low()),
    }


def _overlap(low_a: float, high_a: float, low_b: float, high_b: float) -> bool:
    """与 Common.func_util.has_overlap 的严格条件一致。"""
    return high_b > low_a and high_a > low_b


def _trend_name(rows: list[dict]) -> None:
    """按时间比较同级相邻中枢：后 ZD>前 ZG 为上涨，后 ZG<前 ZD 为下跌，否则盘整。

    只有一个中枢时就是盘整。这是缠中说禅对走势方向的区分，chan.py 本身不写这个字段。
    """
    rows.sort(key=lambda item: (item["start"], item["end"], item["idx"]))
    for i, row in enumerate(rows):
        if i == 0:
            row["trend"] = "盘整"
            continue
        prev = rows[i - 1]
        if row["zd"] > prev["zg"]:
            row["trend"] = "上涨"
        elif row["zg"] < prev["zd"]:
            row["trend"] = "下跌"
        else:
            row["trend"] = "盘整"


def _event_tags(rows: list[dict]) -> None:
    rows.sort(key=lambda item: (item["start"], item["end"], item["idx"]))
    for i, row in enumerate(rows):
        tags: list[str] = []
        if row["sub_cnt"] > 0:
            tags.append("扩展")
        elif row["element_count"] > 3:
            tags.append("延伸")
        prev = rows[i - 1] if i else None
        if prev is None or not _overlap(prev["zd"], prev["zg"], row["zd"], row["zg"]):
            tags.append("新生")
        row["events"] = tags


def centers_from_lines(lines) -> list:
    """连续三元素重叠成中枢，规则同 ZSList.try_construct_zs / CZS.try_add_to_end。

    只用于线段中枢。笔中枢走 CChan 的 zs_list，避免和买卖点引用的中枢各算各的。
    """
    from ZS.ZS import CZS

    lines = list(lines)
    found = []
    index = 0
    while index + 2 < len(lines):
        triple = lines[index : index + 3]
        zg = min(item._high() for item in triple)
        zd = max(item._low() for item in triple)
        if not zg > zd:
            index += 1
            continue
        zs = CZS(triple, is_sure=all(bool(item.is_sure) for item in triple))
        end = index + 3
        while end < len(lines) and zs.try_add_to_end(lines[end]):
            end += 1
        covered = lines[index:end]
        zs.set_bi_lst(covered)
        found.append(zs)
        index = end if end > index else index + 1

    merged = []
    for zs in found:
        if merged and merged[-1].combine(zs, "zs"):
            host = merged[-1]
            seen = {item.idx for item in host.bi_lst}
            extra = [item for item in zs.bi_lst if item.idx not in seen]
            host.set_bi_lst(list(host.bi_lst) + extra)
            continue
        merged.append(zs)
    return merged


def _dump_zs(zs, parent_idx: int) -> dict:
    elements = [_dump_line(item) for item in zs.bi_lst]
    return {
        "idx": int(zs.begin_bi.idx),
        "parent_idx": int(parent_idx),
        "start": _date_of(zs.begin),
        "end": _date_of(zs.end),
        "zd": _px(zs.low),
        "zg": _px(zs.high),
        "dd": _px(zs.peak_low),
        "gg": _px(zs.peak_high),
        "element_count": len(elements),
        "elements": elements,
        "sub_cnt": len(zs.sub_zs_lst),
        # 延伸进去的虚段也要反映到中枢上，不能只看最初三元素
        "sure": bool(zs.is_sure) and all(item["sure"] for item in elements),
    }


def _point_basis(bsp, level: str) -> str:
    is_buy = bool(bsp.is_buy)
    full = BUY_FULL if is_buy else SELL_FULL
    codes = [item.value for item in bsp.type]
    parts = ["、".join(full[code] for code in codes if code in full)]
    if level == "seg":
        parts.append("级别=线段（力度用 CSeg.Cal_MACD_slope，库内线段不支持 MACD 面积）")
    else:
        parts.append("级别=笔")
    feats = {key: value for key, value in bsp.features.items()}
    if feats.get("divergence_rate") is not None:
        algo = "CBi.Cal_MACD_area" if level == "bi" else "CSeg.Cal_MACD_slope"
        parts.append(f"背驰力度比={float(feats['divergence_rate']):.3f}（{algo}，离开段不大于 {DIVERGENCE_RATE} 倍）")
    if feats.get("zs_cnt") is not None:
        parts.append(f"线段内中枢数={int(feats['zs_cnt'])}")
    if feats.get("bsp2_retrace_rate") is not None:
        parts.append(f"二类回撤={float(feats['bsp2_retrace_rate']):.3f}")
    if feats.get("bsp2s_retrace_rate") is not None:
        parts.append(
            f"类二回撤={float(feats['bsp2s_retrace_rate']):.3f}，层数={feats.get('bsp2s_lv')}"
        )
    if feats.get("bsp3_zs_height") is not None:
        parts.append(f"离开中枢高度比={float(feats['bsp3_zs_height']):.3f}，回抽未进入 ZD/ZG")
    line = bsp.bi
    parts.append(
        f"端点笔 { _date_of(line.get_begin_klu()) }→{ _date_of(line.get_end_klu()) }"
        f" {_px(line.get_begin_val())}→{_px(line.get_end_val())}"
    )
    parent = getattr(line, "parent_seg", None)
    if parent is not None and getattr(parent, "zs_lst", None):
        zs = parent.zs_lst[-1]
        parts.append(
            f"参考中枢 {_date_of(zs.begin)}~{_date_of(zs.end)}"
            f" ZD={_px(zs.low)} ZG={_px(zs.high)} DD={_px(zs.peak_low)} GG={_px(zs.peak_high)}"
        )
    relate = bsp.relate_bsp1
    if relate is not None:
        parts.append(
            f"对应一类 {_date_of(relate.klu)} 价格 {_px(relate.bi.get_end_val())}"
        )
    return "；".join(parts)


def _dump_bsp(bsp, level: str) -> dict:
    is_buy = bool(bsp.is_buy)
    codes = [item.value for item in bsp.type]
    codes.sort(key=lambda code: TYPE_ORDER.index(code) if code in TYPE_ORDER else 99)
    short = BUY_SHORT if is_buy else SELL_SHORT
    labels = [short[code] for code in codes if code in short]
    if level == "seg":
        labels = [f"段{label}" for label in labels]
    primary = codes[0] if codes else "2"
    color = (POINT_COLOR if is_buy else SELL_COLOR).get(primary, "#333333")
    return {
        "level": level,
        "side": "buy" if is_buy else "sell",
        "types": codes,
        "label": "/".join(labels),
        "date": _date_of(bsp.klu),
        "price": _px(bsp.bi.get_end_val()),
        "color": color,
        "basis": _point_basis(bsp, level),
    }


def _parent_map(seg_list) -> dict[int, int]:
    found: dict[int, int] = {}
    for seg in seg_list:
        for zs in seg.zs_lst:
            found[id(zs)] = int(seg.idx)
    return found


def analyze(code: str, bars: list[dict]) -> dict:
    """用 chan.py 算结构。返回可直接写入 JSON / 画图的字典。"""
    CChan, CChanConfig, AUTYPE, _BI_DIR, DATA_FIELD, FX_TYPE, KL_TYPE, CTime = _import_chan()
    klu_dicts = []
    for bar in bars:
        year, month, day = (int(part) for part in bar["date"].split("-"))
        item = {
            DATA_FIELD.FIELD_TIME: CTime(year, month, day, 0, 0),
            DATA_FIELD.FIELD_OPEN: float(bar["open"]),
            DATA_FIELD.FIELD_CLOSE: float(bar["close"]),
            DATA_FIELD.FIELD_HIGH: float(bar["high"]),
            DATA_FIELD.FIELD_LOW: float(bar["low"]),
        }
        if bar.get("volume"):
            item[DATA_FIELD.FIELD_VOLUME] = float(bar["volume"])
        if bar.get("amount"):
            item[DATA_FIELD.FIELD_TURNOVER] = float(bar["amount"])
        klu_dicts.append(item)

    data_src = _bind_local_source(klu_dicts)
    conf = CChanConfig(
        {
            "bi_strict": True,
            "bi_fx_check": "strict",
            "seg_algo": "chan",
            "zs_algo": "over_seg",
            "zs_combine": True,
            "zs_combine_mode": "zs",
            "one_bi_zs": False,
            "print_warning": False,
            "print_err_time": True,
            "divergence_rate": DIVERGENCE_RATE,
            "min_zs_cnt": 1,
            "bsp1_only_multibi_zs": True,
            "max_bs2_rate": 0.9999,
            "macd_algo": "full_area",
            "bs1_peak": True,
            "bs_type": "1,1p,2,2s,3a,3b",
            "bsp2_follow_1": True,
            "bsp3_follow_1": False,
            "bsp3_peak": False,
            "bsp2s_follow_2": False,
            "strict_bsp3": False,
        }
    )
    chan = CChan(
        code=code,
        begin_time=bars[0]["date"],
        end_time=bars[-1]["date"],
        data_src=data_src,
        lv_list=[KL_TYPE.K_DAY],
        config=conf,
        autype=AUTYPE.QFQ,
    )
    kl = chan[0]

    fractals = []
    for klc in kl.lst:
        if klc.fx == FX_TYPE.TOP:
            klu = klc.get_peak_klu(True)
            fractals.append({"type": "top", "date": _date_of(klu), "price": _px(klu.high)})
        elif klc.fx == FX_TYPE.BOTTOM:
            klu = klc.get_peak_klu(False)
            fractals.append({"type": "bottom", "date": _date_of(klu), "price": _px(klu.low)})

    bis = [_dump_line(bi) for bi in kl.bi_list]
    segs = [_dump_line(seg) for seg in kl.seg_list]

    bi_parents = _parent_map(kl.seg_list)
    seg_parents = _parent_map(kl.segseg_list)
    bi_zs = [_dump_zs(zs, bi_parents.get(id(zs), -1)) for zs in kl.zs_list]
    # segzs_list 在「父段里只有一段回拉」时是空的，用同一重叠式补线段中枢。
    seg_centers = list(kl.segzs_list) or centers_from_lines(kl.seg_list)
    seg_zs = [_dump_zs(zs, seg_parents.get(id(zs), -1)) for zs in seg_centers]
    for row in bi_zs:
        row["level"] = "bi"
    for row in seg_zs:
        row["level"] = "seg"
    _trend_name(bi_zs)
    _trend_name(seg_zs)
    _event_tags(bi_zs)
    _event_tags(seg_zs)

    points = [_dump_bsp(bsp, "bi") for bsp in kl.bs_point_lst.getSortedBspList()]
    points.extend(_dump_bsp(bsp, "seg") for bsp in kl.seg_bs_point_lst.getSortedBspList())
    points.sort(key=lambda item: (item["date"], item["label"]))

    return {
        "code": code,
        "engine": f"Vespa314/chan.py@{CHAN_COMMIT}",
        "license": "MIT",
        "config": {
            "bi_strict": True,
            "seg_algo": "chan",
            "zs_algo": "over_seg",
            "macd_algo_bi": "full_area",
            "macd_algo_seg": "slope",
            "divergence_rate": DIVERGENCE_RATE,
            "min_zs_cnt": 1,
            "bs1_peak": True,
            "bsp2_follow_1": True,
            "bsp3_follow_1": False,
        },
        "bars": bars,
        "fx": fractals,
        "bi": bis,
        "seg": segs,
        "zs": bi_zs + seg_zs,
        "bsp": points,
    }


def _triple_overlap(elements: list[dict]) -> bool:
    if len(elements) < 3:
        return False
    high = min(item["high"] for item in elements[:3])
    low = max(item["low"] for item in elements[:3])
    return high > low


def validate(result: dict) -> list[str]:
    """中枢区间合法，买卖点落在图上的 K 线日期。"""
    dates = {bar["date"] for bar in result["bars"]}
    errors: list[str] = []
    if not result["bi"]:
        errors.append("没有识别出笔")
    for zs in result["zs"]:
        tag = f"{zs['level']}中枢 {zs['start']}~{zs['end']}"
        if not (zs["zd"] < zs["zg"]):
            errors.append(f"{tag} ZD({zs['zd']}) 没有低于 ZG({zs['zg']})")
        if zs["dd"] > zs["zd"] or zs["gg"] < zs["zg"]:
            errors.append(f"{tag} DD/GG 没有包住 ZD/ZG")
        if zs["element_count"] < 3:
            errors.append(f"{tag} 元素只有 {zs['element_count']} 段，不足三段重叠")
        elif not _triple_overlap(zs["elements"]):
            errors.append(f"{tag} 前三个元素没有重叠")
        if zs["start"] not in dates or zs["end"] not in dates:
            errors.append(f"{tag} 起止日期不在 K 线上")
    for point in result["bsp"]:
        if point["date"] not in dates:
            errors.append(f"买卖点 {point['label']} 日期 {point['date']} 不在 K 线上")
        bar = next(item for item in result["bars"] if item["date"] == point["date"])
        if point["price"] > bar["high"] + 1e-4 or point["price"] < bar["low"] - 1e-4:
            errors.append(
                f"买卖点 {point['label']} {point['date']} 价格 {point['price']} 超出当日高低"
            )
    return errors


def _counts(result: dict) -> dict:
    zs_counts: dict[str, int] = {}
    for zs in result["zs"]:
        key = f"{'笔' if zs['level'] == 'bi' else '线段'}{zs['trend']}中枢"
        extra = "+".join(zs["events"])
        if extra:
            key = f"{key}({extra})"
        zs_counts[key] = zs_counts.get(key, 0) + 1
    bsp_counts: dict[str, int] = {}
    for point in result["bsp"]:
        bsp_counts[point["label"]] = bsp_counts.get(point["label"], 0) + 1
    return {"zs": zs_counts, "bsp": bsp_counts, "n_zs": len(result["zs"]), "n_bsp": len(result["bsp"])}


_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>__TITLE__</title>
  <script src="https://cdn.jsdelivr.net/npm/echarts@5.5.1/dist/echarts.min.js"></script>
  <style>
    :root { --bg:#f4f6f8; --card:#fff; --text:#1a1d21; --muted:#5c6570; --line:#d8dee6; }
    * { box-sizing: border-box; }
    body { margin:0; font-family:"PingFang SC","Hiragino Sans GB","Microsoft YaHei",sans-serif; background:var(--bg); color:var(--text); }
    header { padding:16px 20px 0; }
    h1 { margin:0 0 6px; font-size:1.25rem; }
    .sub { margin:0; color:var(--muted); font-size:0.86rem; line-height:1.55; }
    #chart { height: 760px; margin: 8px 12px; background:var(--card); border:1px solid var(--line); border-radius:8px; }
    section { margin: 0 12px 24px; background:var(--card); border:1px solid var(--line); border-radius:8px; padding:12px 14px; }
    h2 { margin:0 0 8px; font-size:1rem; }
    table { width:100%; border-collapse:collapse; font-size:0.82rem; }
    th, td { border-bottom:1px solid var(--line); padding:4px 6px; text-align:left; vertical-align:top; }
    td:nth-child(1), td:nth-child(2) { white-space: nowrap; }
    th { color:var(--muted); font-weight:600; }
    .basis { color:#333; }
  </style>
</head>
<body>
  <header>
    <h1>__TITLE__</h1>
    <p class="sub">__SUB__</p>
  </header>
  <div id="chart"></div>
  <section>
    <h2>中枢（__N_ZS__）</h2>
    <table id="zs-table"></table>
  </section>
  <section>
    <h2>买卖点（__N_BSP__）</h2>
    <table id="bsp-table"></table>
  </section>
  <script>
    const DATA = __DATA__;
    const TREND_RGB = { "上涨": "231,76,60", "下跌": "20,177,91", "盘整": "52,152,219" };
    const TREND_EDGE = { "上涨": "#c0392b", "下跌": "#0e8a43", "盘整": "#2471a3" };

    function seriesOf(name, extra) {
      return Object.assign({ name, type: "line", data: [], showSymbol: false, tooltip: { show: false } }, extra || {});
    }

    function brokenLine(rows, wantDir, sure) {
      const data = [];
      rows.forEach(row => {
        if (wantDir && row.dir !== wantDir) return;
        if (row.sure !== sure) return;
        data.push([row.start, row.start_price], [row.end, row.end_price], null);
      });
      return data;
    }

    function zoneSeries(name, rows, rgb, edge) {
      const areas = [];
      const dots = [];
      const color = "rgba(" + rgb + ",0.28)";
      const outer = "rgba(" + rgb + ",0.12)";
      rows.forEach(zs => {
        const label = zs.legend_label;
        areas.push([
          { name: label + " GG/DD", xAxis: zs.start, yAxis: zs.dd,
            itemStyle: { color: outer, borderColor: edge, borderWidth: 1, borderType: "dashed" } },
          { xAxis: zs.end, yAxis: zs.gg }
        ]);
        areas.push([
          { name: label, xAxis: zs.start, yAxis: zs.zd,
            itemStyle: { color: color, borderColor: edge, borderWidth: 1 } },
          { xAxis: zs.end, yAxis: zs.zg }
        ]);
        dots.push({
          value: [zs.mid, (zs.zd + zs.zg) / 2],
          symbolSize: 1,
          label: { show: true, formatter: label, color: edge, fontSize: 10 }
        });
      });
      return {
        name: name,
        type: "scatter",
        data: dots,
        itemStyle: { color: edge },
        markArea: { silent: true, label: { show: false }, data: areas },
        z: 1
      };
    }

    function renderTable(id, headers, rows) {
      const table = document.getElementById(id);
      const head = "<tr>" + headers.map(h => "<th>" + h + "</th>").join("") + "</tr>";
      const body = rows.map(cols => "<tr>" + cols.map(c => "<td>" + c + "</td>").join("") + "</tr>").join("");
      table.innerHTML = head + (body || "<tr><td colspan='" + headers.length + "'>无</td></tr>");
    }

    function buildOption() {
      const bars = DATA.bars;
      const dates = bars.map(b => b.date);
      const ohlc = bars.map(b => [b.open, b.close, b.low, b.high]);
      const volumes = bars.map(b => ({
        value: +(b.volume / 1e8).toFixed(4),
        itemStyle: { color: b.close >= b.open ? "#e74c3c" : "#14b15b" }
      }));
      const zsByKey = {};
      DATA.zs.forEach(zs => {
        const level = zs.level === "bi" ? "笔" : "线段";
        const key = level + zs.trend + "中枢";
        (zsByKey[key] = zsByKey[key] || []).push(zs);
      });
      const zoneOrder = ["笔上涨中枢", "笔下跌中枢", "笔盘整中枢", "线段上涨中枢", "线段下跌中枢", "线段盘整中枢"];
      const zoneSeriesList = zoneOrder.filter(name => (zsByKey[name] || []).length).map(name => {
        const rows = zsByKey[name];
        const trend = name.indexOf("上涨") >= 0 ? "上涨" : (name.indexOf("下跌") >= 0 ? "下跌" : "盘整");
        return zoneSeries(name, rows, TREND_RGB[trend], TREND_EDGE[trend]);
      });
      const buys = DATA.bsp.filter(p => p.side === "buy");
      const sells = DATA.bsp.filter(p => p.side === "sell");
      function scatter(name, rows, symbol, rotate) {
        return {
          name, type: "scatter", z: 12,
          data: rows.map(p => ({
            value: [p.date, p.price],
            symbol, symbolRotate: rotate, symbolSize: 13,
            itemStyle: { color: p.color, borderColor: "#fff", borderWidth: 0.6 },
            label: {
              show: true, formatter: p.label,
              position: p.side === "buy" ? "bottom" : "top",
              color: p.color, fontSize: 11, fontWeight: 700
            }
          }))
        };
      }
      const legend = ["K线", "成交量", "上升笔", "下降笔", "未确定笔", "上升线段", "下降线段", "未确定线段", "顶分型", "底分型"]
        .concat(zoneOrder).concat(["买点", "卖点"]);
      return {
        animation: false,
        legend: { type: "scroll", top: 8, data: legend, textStyle: { fontSize: 11 } },
        axisPointer: { link: [{ xAxisIndex: "all" }] },
        tooltip: {
          trigger: "axis",
          axisPointer: { type: "cross" },
          formatter: function (params) {
            if (!params || !params.length) return "";
            const i = params[0].dataIndex;
            const b = bars[i];
            if (!b) return "";
            let html = "<b>" + b.date + "</b><br/>开 " + b.open + " 高 " + b.high
              + "<br/>低 " + b.low + " 收 " + b.close
              + "<br/>成交量 " + (b.volume / 1e8).toFixed(4) + " 亿股";
            DATA.bsp.filter(p => p.date === b.date).forEach(p => {
              html += "<br/><b>" + p.label + "</b> " + p.price + "<br/>" + p.basis;
            });
            DATA.zs.filter(zs => zs.start <= b.date && b.date <= zs.end).forEach(zs => {
              html += "<br/>" + zs.legend_label
                + " ZD " + zs.zd + " / ZG " + zs.zg
                + " · DD " + zs.dd + " / GG " + zs.gg
                + "<br/>" + zs.start + " ~ " + zs.end;
            });
            return html;
          }
        },
        grid: [
          { left: 58, right: 18, top: 78, height: "62%" },
          { left: 58, right: 18, top: "78%", height: "12%" }
        ],
        xAxis: [
          { type: "category", data: dates, boundaryGap: true, axisLabel: { show: false }, min: "dataMin", max: "dataMax" },
          { type: "category", gridIndex: 1, data: dates, boundaryGap: true, axisLabel: { fontSize: 10, hideOverlap: true }, min: "dataMin", max: "dataMax" }
        ],
        yAxis: [
          { scale: true, splitArea: { show: true } },
          { scale: true, gridIndex: 1, splitNumber: 2, axisLabel: { show: false }, axisLine: { show: false }, axisTick: { show: false }, splitLine: { show: false } }
        ],
        dataZoom: [
          { type: "inside", xAxisIndex: [0, 1], start: 0, end: 100 },
          { type: "slider", xAxisIndex: [0, 1], bottom: 6, height: 16, start: 0, end: 100 }
        ],
        series: [
          {
            name: "K线", type: "candlestick", data: ohlc,
            itemStyle: { color: "#e74c3c", color0: "#14b15b", borderColor: "#e74c3c", borderColor0: "#14b15b" }
          },
          { name: "成交量", type: "bar", xAxisIndex: 1, yAxisIndex: 1, data: volumes, itemStyle: { color: "#7f8c8d" } },
          seriesOf("上升笔", { data: brokenLine(DATA.bi, "up", true), lineStyle: { width: 1.6, color: "#e74c3c" }, itemStyle: { color: "#e74c3c" } }),
          seriesOf("下降笔", { data: brokenLine(DATA.bi, "down", true), lineStyle: { width: 1.6, color: "#14b15b" }, itemStyle: { color: "#14b15b" } }),
          seriesOf("未确定笔", { data: brokenLine(DATA.bi, "", false), lineStyle: { width: 1.4, type: "dashed", color: "#7f8c8d" }, itemStyle: { color: "#7f8c8d" } }),
          seriesOf("上升线段", { data: brokenLine(DATA.seg, "up", true), lineStyle: { width: 2, color: "#8e44ad" }, itemStyle: { color: "#8e44ad" } }),
          seriesOf("下降线段", { data: brokenLine(DATA.seg, "down", true), lineStyle: { width: 2, color: "#6c3483" }, itemStyle: { color: "#6c3483" } }),
          seriesOf("未确定线段", { data: brokenLine(DATA.seg, "", false), lineStyle: { width: 1.6, type: "dashed", color: "#9b59b6" }, itemStyle: { color: "#9b59b6" } }),
          {
            name: "顶分型", type: "scatter", symbol: "diamond", symbolSize: 8, z: 6,
            itemStyle: { color: "#c0392b" },
            data: DATA.fx.filter(f => f.type === "top").map(f => [f.date, f.price])
          },
          {
            name: "底分型", type: "scatter", symbol: "diamond", symbolSize: 8, z: 6,
            itemStyle: { color: "#1e8449" },
            data: DATA.fx.filter(f => f.type === "bottom").map(f => [f.date, f.price])
          }
        ].concat(zoneSeriesList).concat([
          scatter("买点", buys, "triangle", 0),
          scatter("卖点", sells, "triangle", 180)
        ])
      };
    }

    const dates = DATA.bars.map(b => b.date);
    DATA.zs.forEach(zs => {
      const level = zs.level === "bi" ? "笔" : "线段";
      const events = (zs.events || []).join("·");
      zs.legend_label = level + "·" + zs.trend + (events ? "·" + events : "") + (zs.sure ? "" : "·未确定");
      const i = Math.max(0, dates.indexOf(zs.start));
      const j = Math.max(i, dates.indexOf(zs.end));
      zs.mid = dates[Math.floor((i + j) / 2)] || zs.start;
    });
    renderTable("zs-table",
      ["级别", "方向", "形态", "起", "止", "ZD", "ZG", "DD", "GG", "元素数"],
      DATA.zs.map(zs => [
        zs.level === "bi" ? "笔中枢" : "线段中枢",
        zs.trend, (zs.events || []).join("、") || "—",
        zs.start, zs.end, zs.zd, zs.zg, zs.dd, zs.gg, zs.element_count
      ]));
    renderTable("bsp-table",
      ["标签", "日期", "价格", "依据"],
      DATA.bsp.map(p => [p.label, p.date, p.price, "<span class='basis'>" + p.basis + "</span>"]));
    if (typeof echarts === "undefined") {
      document.getElementById("chart").innerHTML = "<p style='padding:16px;color:#c0392b'>ECharts 没有加载，请联网后刷新。</p>";
    } else {
      const chart = echarts.init(document.getElementById("chart"));
      chart.setOption(buildOption());
      window.addEventListener("resize", () => chart.resize());
    }
  </script>
</body>
</html>
"""


def render_html(result: dict, counts: dict) -> str:
    code = result["code"]
    bars = result["bars"]
    title = f"{code} 日K缠论  {bars[0]['date']} ~ {bars[-1]['date']}"
    sub = (
        f"引擎 {result['engine']}（MIT）。笔中枢/线段中枢的浅色虚框是 DD-GG，实心框是 ZD-ZG。"
        f" 图例可单独关闭笔、线段、分型、各类中枢、买点、卖点。"
        f" 中枢 {counts['n_zs']} 个，买卖点 {counts['n_bsp']} 个。"
        " 悬停 K 线可看买卖点依据。"
    )
    payload = json.dumps(
        {key: result[key] for key in ("bars", "fx", "bi", "seg", "zs", "bsp")},
        ensure_ascii=False,
    ).replace("<", "\\u003c")
    html = _HTML.replace("__TITLE__", title).replace("__SUB__", sub)
    html = html.replace("__N_ZS__", str(counts["n_zs"])).replace("__N_BSP__", str(counts["n_bsp"]))
    return html.replace("__DATA__", payload)


def write_outputs(result: dict, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    code = result["code"]
    end = result["bars"][-1]["date"]
    html_path = out_dir / f"{code}_{end}_chan.html"
    json_path = out_dir / f"{code}_{end}_chan.json"
    counts = _counts(result)
    public = {
        key: result[key]
        for key in ("code", "engine", "license", "config", "fx", "bi", "seg", "zs", "bsp")
    }
    public["bar_count"] = len(result["bars"])
    public["start"] = result["bars"][0]["date"]
    public["end"] = end
    public["counts"] = counts
    json_path.write_text(json.dumps(public, ensure_ascii=False, indent=2), encoding="utf-8")
    html_path.write_text(render_html(result, counts), encoding="utf-8")
    return html_path, json_path


def main() -> None:
    parser = argparse.ArgumentParser(description="单股日 K 缠论结构图（ECharts）")
    parser.add_argument("--code", required=True, help="6 位股票代码，如 000001")
    parser.add_argument("--days", type=int, default=250, help="回溯自然日，默认 250")
    parser.add_argument("--out", default=None, help="输出目录，默认 chan_charts/")
    args = parser.parse_args()
    code = str(args.code).zfill(6)
    if args.days < 30:
        raise SystemExit("--days 至少 30")
    bars, _spot = load_bars(code, args.days)
    result = analyze(code, bars)
    errors = validate(result)
    if errors:
        raise SystemExit("缠论结果未通过校验：\n- " + "\n- ".join(errors))
    out_dir = Path(args.out) if args.out else OUT_DIR
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    html_path, json_path = write_outputs(result, out_dir)
    counts = _counts(result)
    print(f"{code}  {bars[0]['date']} ~ {bars[-1]['date']}  {len(bars)} 根")
    print(f"中枢 {counts['n_zs']}  {counts['zs']}")
    print(f"买卖点 {counts['n_bsp']}  {counts['bsp']}")
    print(f"JSON {json_path}")
    print(html_path)


if __name__ == "__main__":
    main()
