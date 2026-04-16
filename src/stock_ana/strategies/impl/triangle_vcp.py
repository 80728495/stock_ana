"""
VCP 三角形形态发现模块
========================

从宏观前高（macro peak）到当前价格区间，检测 VCP 风格的三角形收敛：
  - 上边缘：水平 或 向下（排除上升楔形）
  - 下边缘：向上（低点抬升）
  - 几何上表现为"股价波动逐步收缩"

边缘点检测方法：
  - 第一遍：argrelextrema 提取局部极值候选点（宽覆盖）
  - 第二遍：ZigZag 幅度过滤掉噪声点（精筛）
  - 取两者交集作为最终 swing points

前置条件：
  - Minervini Stage 2 均线趋势过滤
  - 前高由 find_vcp.find_macro_peaks 提供
"""

import numpy as np
import pandas as pd
from loguru import logger

from stock_ana.strategies.primitives import (
    check_trend_template,
    line_value,
    merge_pivots_with_zigzag,
    ols_fit,
    zigzag_indices,
)


# ═══════════════════════════════════════════════════════════
#  VCP 三角形主检测函数
# ═══════════════════════════════════════════════════════════

def screen_triangle_vcp(
    df: pd.DataFrame,
    peak_iloc: int | None = None,
    min_period: int = 25,
    max_period: int = 250,
    require_trend: bool = True,
) -> dict | None:
    """
    在给定价格区间内检测 VCP 三角形（上平/下降 + 下升 = 收敛）。

    形态定义：
      - 上边缘斜率 ≤ 0（水平或向下），排除上升楔形
      - 下边缘斜率 > 0（低点抬升）
      - 两线收敛，波幅递缩

    Args:
        df: K 线 DataFrame（需含 open/high/low/close/volume）
        peak_iloc: 前高在 df 中的行索引位置。
                   若提供，则检测 [peak_iloc, end] 区间；
                   若为 None，则自动滑窗搜索最优解。
        min_period: 最小检测窗口（交易日）
        max_period: 最大检测窗口（交易日）
        require_trend: 是否执行 Minervini 趋势过滤（默认 True）

    Returns:
        形态信息 dict 或 None
    """
    n = len(df)
    if n < min_period:
        return None

    if require_trend and not check_trend_template(df):
        return None

    highs_arr = df["high"].values.astype(float)
    lows_arr = df["low"].values.astype(float)
    closes_arr = df["close"].values.astype(float)

    # 自适应 ZigZag 阈值：基于整段波动率
    full_range = (highs_arr.max() - lows_arr.min()) / highs_arr.mean() * 100
    zz_threshold = max(3.0, min(full_range * 0.15, 8.0))

    # ── 确定搜索区间 ──
    if peak_iloc is not None:
        # 从前高出发时，跳过初始下跌段，在回落后的盘整区搜索三角形。
        # 找前高之后的第一个显著低点（ZigZag Low），以此为三角形起点。
        post_peak_highs = highs_arr[peak_iloc:]
        post_peak_lows = lows_arr[peak_iloc:]
        _, zz_lows = zigzag_indices(post_peak_highs, post_peak_lows, zz_threshold)

        if len(zz_lows) == 0:
            # 回退：取前高后最低点
            first_low_offset = int(np.argmin(post_peak_lows[:min(len(post_peak_lows), 60)]))
        else:
            first_low_offset = int(zz_lows[0])

        # 三角形起点 = 前高 + 第一个低点偏移（至少跳过 10 根 K 线）
        tri_start = peak_iloc + max(first_low_offset, 10)
        if tri_start >= n - min_period:
            return None

        # 在 [tri_start, n] 内搜索多个窗口，同时也尝试从前高后不同位置开始
        periods = []
        for offset in range(0, min(first_low_offset, 40) + 1, 10):
            ws = peak_iloc + max(offset, 10)
            if ws < n - min_period:
                periods.append(n - ws)
        # 补充标准滑窗
        for p in range(min_period, min(max_period, n - peak_iloc) + 1, 5):
            if p not in periods:
                periods.append(p)
    else:
        # 滑窗搜索
        periods = list(range(min_period, min(max_period, n) + 1, 5))

    best: dict | None = None
    best_sc = -1.0

    for period in periods:
        w_start = n - period
        if w_start < 0:
            continue

        win_highs = highs_arr[w_start:]
        win_lows = lows_arr[w_start:]
        win_closes = closes_arr[w_start:]

        # ── 提取边缘点 ──
        merged_hi, merged_lo = merge_pivots_with_zigzag(win_highs, win_lows, zz_threshold)

        if len(merged_hi) < 3 or len(merged_lo) < 3:
            continue

        hi_vals = win_highs[merged_hi]
        lo_vals = win_lows[merged_lo]
        avg = float(win_closes.mean())

        # ── 上沿 OLS ──
        rs, ri, rr2, r_max_res = ols_fit(merged_hi.astype(float), hi_vals)
        rs_pct = rs / avg * 100 if avg > 0 else 0.0

        # 核心规则：上沿必须平或向下（排除上升楔形）
        if rs_pct > 0.03:  # 允许极小正斜率（≤0.03%/日 视为水平）
            continue

        # 残差上限：从前高出发的形态天然残差更大，按周期自适应
        max_res = 4.0 + period / 40.0  # 40天→5%, 120天→7%, 200天→9%
        if r_max_res > max_res:
            continue

        # ── 下沿 OLS ──
        ss, si, sr2, s_max_res = ols_fit(merged_lo.astype(float), lo_vals)
        ss_pct = ss / avg * 100 if avg > 0 else 0.0

        # 下沿必须上升
        if ss_pct < 0.01:
            continue
        if s_max_res > max_res:
            continue

        # ── 收敛检查 ──
        g0 = line_value(rs, ri, 0) - line_value(ss, si, 0)
        ge = line_value(rs, ri, period - 1) - line_value(ss, si, period - 1)

        if g0 <= 0 or ge <= 0:
            continue

        conv = 1.0 - ge / g0
        if conv < 0.10:
            continue

        # ── 穿越统计（从前高出发允许更多穿越） ──
        rv = sum(
            1 for x in range(period)
            if win_highs[x] > line_value(rs, ri, x) * 1.02
        )
        sv = sum(
            1 for x in range(period)
            if win_lows[x] < line_value(ss, si, x) * 0.98
        )
        if rv > period * 0.18 or sv > period * 0.18:
            continue

        # ── 当前价位置 ──
        cur = float(closes_arr[-1])
        re = line_value(rs, ri, period - 1)
        se = line_value(ss, si, period - 1)
        ch = re - se
        if ch <= 0:
            continue
        if cur < se - ch * 0.05 or cur > re * 1.03:
            continue
        pos = (cur - se) / ch

        # ── 波幅递缩 ──
        seg = period // 3
        if seg < 5:
            continue
        rng = win_highs - win_lows
        sp1 = float(np.mean(rng[:seg]))
        sp3 = float(np.mean(rng[-seg:]))
        spr = sp3 / sp1 if sp1 > 0 else 1.0
        if spr > 0.95:
            continue

        # ── 量能收缩 ──
        vr = 1.0
        if "volume" in df.columns:
            vol = df["volume"].values[w_start:].astype(float)
            if not np.isnan(vol).all() and vol.sum() > 0:
                vl = float(np.mean(vol[: seg * 2]))
                vri = float(np.mean(vol[-seg:]))
                vr = vri / vl if vl > 0 else 1.0

        # ── 触线计数 ──
        tol = avg * 0.015
        res_touches = int(np.sum(np.abs(hi_vals - line_value(rs, ri, merged_hi.astype(float))) <= tol))
        sup_touches = int(np.sum(np.abs(lo_vals - line_value(ss, si, merged_lo.astype(float))) <= tol))
        total_touches = res_touches + sup_touches

        # ── 形态子类型 ──
        if abs(rs_pct) <= 0.03:
            pattern = "ascending_triangle"
        else:
            pattern = "descending_triangle"  # 上沿向下 + 低点抬升 = 对称/下降三角

        # ── 评分 ──
        sc = (
            conv * 20
            + (1 - spr) * 25
            + max(0, 1 - vr) * 10
            + pos * 10
            + min(rr2, sr2) * 10
            + total_touches * 3
            - (rv + sv) * 1.0
        )

        if sc > best_sc:
            best_sc = sc

            # 收敛点距离
            slope_diff = rs - ss
            if abs(slope_diff) > 1e-12:
                x_cross = (si - ri) / (rs - ss)
                dtc = x_cross - (period - 1)
            else:
                dtc = float("inf")

            best = {
                "pattern": pattern,
                "period": period,
                "window_start": w_start,
                "convergence_ratio": round(conv, 3),
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
                "upper_slope_pct": round(rs_pct, 4),
                "lower_slope_pct": round(ss_pct, 4),
                "resistance": {
                    "slope": rs, "intercept": ri,
                    "r_squared": round(rr2, 3),
                    "touches": res_touches, "breaches": rv,
                    "total_swings": len(merged_hi),
                    "anchors": [(int(w_start + x), float(y))
                                for x, y in zip(merged_hi, hi_vals)],
                },
                "support": {
                    "slope": ss, "intercept": si,
                    "r_squared": round(sr2, 3),
                    "touches": sup_touches, "breaches": sv,
                    "total_swings": len(merged_lo),
                    "anchors": [(int(w_start + x), float(y))
                                for x, y in zip(merged_lo, lo_vals)],
                },
            }

    return best
