"""三连小阳 + 量能抬升：形态判定（与江苏新能 9/8–9/10 同类）。

严格相似：
  · 三日均为小阳，日涨约 0.2%～3.5%
  · 每日量比 1.05～2.0；相对三日前量 ≥1.15×
  · 前 22 日横盘：振幅 ≤16%、|净涨跌| ≤10%、距 60 日低 ≤8%
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SIGNAL_RET_MAX = 0.095
LOOKBACK_60 = 60
BASE_DAYS = 22

RET_LO = 0.002
RET_HI = 0.035
VOL_RATIO_LO = 1.05
VOL_RATIO_HI = 2.0
VOL_GROWTH_MIN = 1.15
BASE_RANGE_MAX = 0.16
BASE_ABS_RET_MAX = 0.10
BASE_DIST60_LOW_MAX = 0.08

MAINBOARD_PREFIX = ("600", "601", "603", "605", "000", "001", "002")


@dataclass
class PatternHit:
    i: int
    ret1: float
    ret2: float
    ret3: float
    vr1: float
    vr2: float
    vr3: float
    cum3: float
    vol_growth: float
    base_range: float | None = None
    base_ret: float | None = None
    dist60_low: float | None = None


def base_ok(
    px: np.ndarray,
    hi: np.ndarray,
    lo: np.ndarray,
    i: int,
) -> tuple[bool, float | None, float | None, float | None]:
    """三连阳起点前一日为横盘末日（下标 i-3）。"""
    end = i - 3
    need = BASE_DAYS + LOOKBACK_60
    if end < need - 1:
        return False, None, None, None
    a = end - BASE_DAYS + 1
    hh = float(np.nanmax(hi[a : end + 1]))
    ll = float(np.nanmin(lo[a : end + 1]))
    if not np.isfinite(hh) or not np.isfinite(ll) or ll <= 0:
        return False, None, None, None
    rng = hh / ll - 1.0
    if rng > BASE_RANGE_MAX:
        return False, rng, None, None
    p0 = float(px[a])
    p1 = float(px[end])
    if p0 <= 0 or not np.isfinite(p0) or not np.isfinite(p1):
        return False, rng, None, None
    base_ret = p1 / p0 - 1.0
    if abs(base_ret) > BASE_ABS_RET_MAX:
        return False, rng, base_ret, None
    b = end - LOOKBACK_60 + 1
    low60 = float(np.nanmin(lo[b : end + 1]))
    if not np.isfinite(low60) or low60 <= 0:
        return False, rng, base_ret, None
    dist_low = p1 / low60 - 1.0
    if dist_low > BASE_DIST60_LOW_MAX:
        return False, rng, base_ret, dist_low
    return True, rng, base_ret, dist_low


def match_at(
    px: np.ndarray,
    op: np.ndarray,
    vol: np.ndarray,
    ret: np.ndarray,
    i: int,
    *,
    hi: np.ndarray | None = None,
    lo: np.ndarray | None = None,
    require_base: bool = True,
) -> PatternHit | None:
    """检查以 i 为末日的三连小阳+量抬升。"""
    if i < 3:
        return None
    if not np.isfinite(ret[i]) or ret[i] > SIGNAL_RET_MAX:
        return None
    rets: list[float] = []
    vrs: list[float] = []
    for k in (i - 2, i - 1, i):
        if not np.isfinite(ret[k]) or not np.isfinite(px[k]) or not np.isfinite(op[k]):
            return None
        if px[k] < op[k]:
            return None
        if ret[k] < RET_LO or ret[k] > RET_HI:
            return None
        prev_vol = float(vol[k - 1])
        cur_vol = float(vol[k])
        if prev_vol <= 0 or cur_vol <= 0 or not np.isfinite(prev_vol) or not np.isfinite(cur_vol):
            return None
        vr = cur_vol / prev_vol
        if vr < VOL_RATIO_LO or vr > VOL_RATIO_HI:
            return None
        rets.append(float(ret[k]))
        vrs.append(vr)
    base_vol = float(vol[i - 3])
    if base_vol <= 0 or not np.isfinite(base_vol):
        return None
    growth = float(vol[i]) / base_vol
    if growth < VOL_GROWTH_MIN:
        return None
    base_px = float(px[i - 3])
    if base_px <= 0 or not np.isfinite(base_px):
        return None

    brng = bret = dlow = None
    if require_base:
        if hi is None or lo is None:
            return None
        ok, brng, bret, dlow = base_ok(px, hi, lo, i)
        if not ok:
            return None

    return PatternHit(
        i=i,
        ret1=rets[0],
        ret2=rets[1],
        ret3=rets[2],
        vr1=vrs[0],
        vr2=vrs[1],
        vr3=vrs[2],
        cum3=float(px[i]) / base_px - 1.0,
        vol_growth=growth,
        base_range=brng,
        base_ret=bret,
        dist60_low=dlow,
    )
