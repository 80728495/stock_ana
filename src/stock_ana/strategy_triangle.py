"""
上升收敛形态策略模块 (V3)
========================

检测上升三角形 / 上升楔形。
三维度筛选:
  1. 趋势 — Minervini Stage 2
  2. 几何 — 支撑线(low 连线)显著上升 + 收敛(sup_slope > res_slope)
           阻力线角度 ≤ 15°，排除上升通道
  3. 微观 — K 线波幅递缩 + 量能枯竭 (VCP 原理)
"""

import math

import numpy as np
import pandas as pd
from loguru import logger

from stock_ana.data_fetcher import load_all_ndx100_data
from stock_ana.strategy_base import check_trend_template, find_swing_points


# ─────────── 工具 ───────────

def _fit_line(pts: list[tuple[int, float]]) -> tuple[float, float, float]:
    """OLS 线性回归。返回 (slope, intercept, r²)。"""
    if len(pts) < 2:
        return 0.0, 0.0, 0.0
    xs = np.array([p[0] for p in pts], dtype=float)
    ys = np.array([p[1] for p in pts], dtype=float)
    s, i = np.polyfit(xs, ys, 1)
    yh = s * xs + i
    ssr = float(np.sum((ys - yh) ** 2))
    sst = float(np.sum((ys - ys.mean()) ** 2))
    r2 = max(0.0, 1.0 - ssr / sst) if sst > 1e-12 else 0.0
    return float(s), float(i), r2


def _lv(s: float, i: float, x: float) -> float:
    """线上取值。"""
    return s * x + i


# ─────────── 主检测 ───────────

def screen_ascending_triangle(
    df: pd.DataFrame,
    min_period: int = 25,
    max_period: int = 120,
) -> dict | None:
    """
    检测上升收敛形态（上升三角形 / 上升楔形）。

    维度 1 — 趋势: 必须满足 Minervini Stage 2 模板。
    维度 2 — 几何:
        · OLS 回归 swing highs → 阻力线, swing lows → 支撑线
        · 支撑线上倾 (slope_pct ≥ 0.02 %/day)
        · 阻力线角度 ≤ 15°，排除上升通道
        · 收敛比 (1 − end_gap/start_gap) ≥ 15 %
    维度 3 — 微观 (VCP):
        · 右 1/3 平均波幅 < 左 1/3（至少缩 10 %）
        · 右 1/3 成交量 < 左 2/3

    Returns:
        形态信息 dict 或 None
    """
    # ── 维度 1: 趋势 ──
    if not check_trend_template(df):
        return None
    if len(df) < min_period:
        return None

    best: dict | None = None
    best_sc = -1.0

    for period in range(min_period, min(max_period + 1, len(df) + 1), 5):
        w = df.iloc[-period:]
        hi = w["high"]
        lo = w["low"]
        cl = w["close"]
        avg = float(cl.mean())

        # 自适应 swing order
        order = 3 if period < 40 else 4 if period < 80 else 5

        # ── 维度 2a: swing 极值 ──
        sw_h = [(i, v) for i, v, t in find_swing_points(hi, order=order) if t == "high"]
        sw_l = [(i, v) for i, v, t in find_swing_points(lo, order=order) if t == "low"]
        if len(sw_h) < 3 or len(sw_l) < 3:
            continue

        # ── 维度 2b: OLS 回归 ──
        rs, ri, rr2 = _fit_line(sw_h)
        ss, si, sr2 = _fit_line(sw_l)

        # 支撑线上倾
        ss_pct = ss / avg * 100
        if ss_pct < 0.02:
            continue

        # 阻力线角度 ≤ 15°，排除上升通道
        res_angle_deg = abs(math.atan(rs / avg * period)) * 180 / math.pi
        if res_angle_deg > 15:
            continue

        # 线性拟合质量: 至少一条线 R² ≥ 0.4
        if max(rr2, sr2) < 0.4:
            continue

        # ── 维度 2c: 触线测试 ──
        # swing 点距回归线 ≤ 1.5% 才算有效 touch
        tol = avg * 0.015
        res_touches = sum(
            1 for x, y in sw_h if abs(y - _lv(rs, ri, x)) <= tol
        )
        sup_touches = sum(
            1 for x, y in sw_l if abs(y - _lv(ss, si, x)) <= tol
        )
        if res_touches < 3 or sup_touches < 3:
            continue

        # 收敛
        if ss <= rs:
            continue

        g0 = _lv(rs, ri, 0) - _lv(ss, si, 0)
        ge = _lv(rs, ri, period - 1) - _lv(ss, si, period - 1)
        if g0 <= 0 or ge <= 0:
            continue
        conv = 1.0 - ge / g0
        if conv < 0.15:
            continue

        # 穿越统计
        hv = hi.values
        lv_arr = lo.values
        rv = sum(1 for x in range(period) if hv[x] > _lv(rs, ri, x) * 1.015)
        sv = sum(1 for x in range(period) if lv_arr[x] < _lv(ss, si, x) * 0.985)
        if rv > period * 0.12 or sv > period * 0.12:
            continue

        # 当前价在通道内
        cur = float(cl.iloc[-1])
        re = _lv(rs, ri, period - 1)
        se = _lv(ss, si, period - 1)
        ch = re - se
        if cur < se - ch * 0.05 or cur > re * 1.03:
            continue
        pos = (cur - se) / ch if ch > 0 else 0.5

        # ── 维度 3: VCP 微观 ──
        seg = period // 3
        if seg < 5:
            continue

        rng = hv - lv_arr
        sp1 = float(np.mean(rng[:seg]))
        sp3 = float(np.mean(rng[-seg:]))
        spr = sp3 / sp1 if sp1 > 0 else 1.0
        if spr > 0.90:
            continue

        vr = 1.0
        if "volume" in w.columns:
            vol = w["volume"]
            if not vol.isna().all() and vol.sum() > 0:
                vv = vol.values.astype(float)
                vl = float(np.mean(vv[: seg * 2]))
                vri = float(np.mean(vv[-seg:]))
                vr = vri / vl if vl > 0 else 1.0

        # ── 分类 ──
        rs_pct = rs / avg * 100
        if abs(rs_pct) < 0.05:
            pt = "ascending_triangle"
        else:
            pt = "rising_wedge"

        # ── 评分 ──
        total_touches = res_touches + sup_touches
        sc = (
            conv * 20
            + (1 - spr) * 20
            + max(0, 1 - vr) * 10
            + pos * 10
            + min(rr2, sr2) * 10
            + total_touches * 3
            - (rv + sv) * 1.0
        )

        if sc > best_sc:
            best_sc = sc
            ws = len(df) - period
            sd = rs - ss
            dtc = (si - ri) / sd - (period - 1) if abs(sd) > 1e-12 else float("inf")
            norm = avg / period if avg > 0 else 1.0
            angle = abs(math.atan(ss / norm) - math.atan(rs / norm)) * 180 / math.pi

            best = {
                "pattern": pt,
                "period": period,
                "window_start": ws,
                "convergence_ratio": round(conv, 3),
                "convergence_angle_deg": round(angle, 1),
                "days_to_convergence": round(dtc, 1),
                "convergence_status": (
                    "converged" if dtc <= 0
                    else "imminent" if dtc <= 20
                    else "future"
                ),
                "spread_contraction": round(spr, 3),
                "vol_contraction": round(vr, 3),
                "position_in_channel": round(pos, 3),
                "score": round(sc, 1),
                "resistance": {
                    "slope": rs, "intercept": ri,
                    "r_squared": round(rr2, 3),
                    "touches": res_touches, "breaches": rv,
                    "total_swings": len(sw_h),
                    "anchors": [(ws + x, y) for x, y in sw_h],
                },
                "support": {
                    "slope": ss, "intercept": si,
                    "r_squared": round(sr2, 3),
                    "touches": sup_touches, "breaches": sv,
                    "total_swings": len(sw_l),
                    "anchors": [(ws + x, y) for x, y in sw_l],
                },
                "swing_highs": [(ws + x, y) for x, y in sw_h],
                "swing_lows": [(ws + x, y) for x, y in sw_l],
            }

    return best


def scan_ndx100_ascending_triangle(
    min_period: int = 25,
    max_period: int = 120,
) -> list[dict]:
    """扫描纳指100中呈现上升收敛形态的股票。"""
    stock_data = load_all_ndx100_data()
    if not stock_data:
        logger.error("本地无数据！请先运行 update_ndx100_data() 下载数据")
        return []

    _CN = {
        "ascending_triangle": "上升三角形",
        "rising_wedge": "上升楔形",
    }

    hits: list[dict] = []
    processed = 0

    for ticker, df in stock_data.items():
        try:
            if len(df) < 260:
                continue
            processed += 1
            result = screen_ascending_triangle(df, min_period, max_period)
            if result is not None:
                ptype = _CN.get(result["pattern"], result["pattern"])
                logger.success(
                    f"✅ {ticker} {ptype} "
                    f"({result['period']}d, "
                    f"收敛{result['convergence_ratio']:.0%}, "
                    f"波幅缩{1 - result['spread_contraction']:.0%}, "
                    f"量缩{1 - result['vol_contraction']:.0%})"
                )
                hits.append({"ticker": ticker, "df": df, "pattern_info": result})
        except Exception as e:
            logger.error(f"{ticker}: {e}")

    logger.info(
        f"上升收敛扫描完成：{len(stock_data)} 只股票，"
        f"有效 {processed} 只，{len(hits)} 只呈现形态"
    )
    return hits
