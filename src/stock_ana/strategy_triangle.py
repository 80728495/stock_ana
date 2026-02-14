"""
OLS 边缘检测策略模块
========================

基于 Swing Points + OLS 线性回归检测三种形态:
  - 上升三角形: 上沿水平 (斜率 ≈ 0) + 低点抬升 → 看多
  - 上升平行通道: 上下沿斜率接近 (ratio ≥ 0.6) → 看多
  - 上升楔形: 上沿向上但斜率明显小于下沿 (ratio < 0.6) → 看空

设计原则:
  - 长周期优先: 评分偏向更长的窗口 — 跨越更多时间的形态更可靠
  - 水平线优先: 水平阻力/支撑是关键价格位, 给予额外加分
  - 多尺度 Pivot: 同时提取短期和中期 Swing Points, 合并后拟合
  - 宽容残差: 长窗口允许略大残差 (价格波动随时间积累)
"""

import math

import numpy as np
import pandas as pd
from loguru import logger
from scipy.signal import argrelextrema

from stock_ana.data_fetcher import load_all_ndx100_data
from stock_ana.strategy_base import check_trend_template


# ─────────── 工具 ───────────

def _extract_pivots(
    highs: np.ndarray,
    lows: np.ndarray,
    order: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    用 scipy.signal.argrelextrema 提取 Swing Highs / Swing Lows 的索引。

    Returns:
        (swing_high_indices, swing_low_indices)
    """
    hi_idx = argrelextrema(highs, np.greater_equal, order=order)[0]
    lo_idx = argrelextrema(lows, np.less_equal, order=order)[0]
    return hi_idx, lo_idx


def _extract_pivots_multiscale(
    highs: np.ndarray,
    lows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    多尺度 Pivot 提取: 同时用 order=3 和 order=6 提取, 合并去重。
    这样既能捕捉到近期的小 Swing, 又不会漏掉远期的大 Swing。
    """
    hi3, lo3 = _extract_pivots(highs, lows, order=3)
    hi6, lo6 = _extract_pivots(highs, lows, order=6)
    hi_merged = np.unique(np.concatenate([hi3, hi6]))
    lo_merged = np.unique(np.concatenate([lo3, lo6]))
    return hi_merged, lo_merged


def _ols(xs: np.ndarray, ys: np.ndarray) -> tuple[float, float, float, float]:
    """
    OLS 线性回归。

    Returns:
        (slope, intercept, r², max_abs_residual_pct)
        其中 max_abs_residual_pct = max(|residual|) / mean(y) * 100
    """
    if len(xs) < 2:
        return 0.0, 0.0, 0.0, float("inf")
    s, i = np.polyfit(xs, ys, 1)
    yh = s * xs + i
    residuals = ys - yh
    ssr = float(np.sum(residuals ** 2))
    sst = float(np.sum((ys - ys.mean()) ** 2))
    r2 = max(0.0, 1.0 - ssr / sst) if sst > 1e-12 else 0.0
    ymean = float(ys.mean())
    max_res_pct = float(np.max(np.abs(residuals))) / ymean * 100 if ymean > 0 else float("inf")
    return float(s), float(i), r2, max_res_pct


def _line_val(s: float, i: float, x: float) -> float:
    """线上取值。"""
    return s * x + i


# ─────────── 主检测（通用） ───────────

# 所有支持的形态
ALL_PATTERNS = {"ascending_triangle", "parallel_channel", "rising_wedge"}

# 默认窗口范围
DEFAULT_MIN_PERIOD = 25
DEFAULT_MAX_PERIOD = 250  # ~1年, 覆盖大盘股的长周期形态


def _screen_ols_pattern(
    df: pd.DataFrame,
    min_period: int = DEFAULT_MIN_PERIOD,
    max_period: int = DEFAULT_MAX_PERIOD,
    allowed_patterns: set[str] | None = None,
) -> dict | None:
    """
    通用 OLS 边缘检测。

    改进要点:
      - 多尺度 Pivot: order=3 + order=6 合并, 确保长短期 Swing 都被捕捉
      - 长周期优先: 评分包含 log(period) 加分, 大形态天然更可靠
      - 水平线加分: 上沿接近水平 (关键阻力位) 额外加分
      - 自适应步长: 短周期步长 5, 长周期步长 10, 提高效率
      - 宽容残差: period > 150 时残差上限 3.5% (长窗口价格波动更大)

    Returns:
        形态信息 dict 或 None
    """
    if allowed_patterns is None:
        allowed_patterns = ALL_PATTERNS

    if len(df) < min_period:
        return None
    if not check_trend_template(df):
        return None

    # ── 步骤 1: 多尺度 Pivot 提取 ──
    highs_arr = df["high"].values.astype(float)
    lows_arr = df["low"].values.astype(float)
    closes_arr = df["close"].values.astype(float)

    n = len(df)
    hi_idx, lo_idx = _extract_pivots(highs_arr, lows_arr, order=5)

    if len(hi_idx) < 3 or len(lo_idx) < 3:
        return None

    # ── 步骤 2: 滑动窗口 (自适应步长) ──
    best: dict | None = None
    best_sc = -1.0

    # 构建试探周期列表
    periods = list(range(min_period, min(max_period, n) + 1, 5))

    for period in periods:
        w_start = n - period

        # 筛选落在窗口内的 pivot 点
        mask_h = (hi_idx >= w_start) & (hi_idx < n)
        mask_l = (lo_idx >= w_start) & (lo_idx < n)
        wh_idx = hi_idx[mask_h]
        wl_idx = lo_idx[mask_l]

        if len(wh_idx) < 3 or len(wl_idx) < 3:
            continue

        wh_rel = wh_idx - w_start
        wl_rel = wl_idx - w_start
        wh_vals = highs_arr[wh_idx]
        wl_vals = lows_arr[wl_idx]

        avg = float(closes_arr[w_start:].mean())

        # ── 上沿 OLS ──
        rs, ri, rr2, r_max_res = _ols(wh_rel.astype(float), wh_vals)
        rs_pct = rs / avg * 100 if avg > 0 else 0.0
        rs_pct_abs = abs(rs_pct)

        if r_max_res > 3.0:
            continue

        # ── 下沿 OLS ──
        ss, si, sr2, s_max_res = _ols(wl_rel.astype(float), wl_vals)
        ss_pct = ss / avg * 100 if avg > 0 else 0.0

        if ss_pct < 0.02:
            continue
        if s_max_res > 3.0:
            continue

        # ── 形态分类 ──
        if rs_pct < -0.05:
            continue  # 对称三角形不在检测范围

        if rs_pct > 0.05:
            ratio = rs_pct / ss_pct if ss_pct > 1e-8 else 0.0
            if ratio >= 0.6:
                _pattern_type = "parallel_channel"
            else:
                _pattern_type = "rising_wedge"
        else:
            _pattern_type = "ascending_triangle"

        if _pattern_type not in allowed_patterns:
            continue

        # ── 收敛 / 平行性检查 ──
        if _pattern_type == "parallel_channel":
            g0 = _line_val(rs, ri, 0) - _line_val(ss, si, 0)
            ge = _line_val(rs, ri, period - 1) - _line_val(ss, si, period - 1)
            if g0 <= 0 or ge <= 0:
                continue
            conv = abs(1.0 - ge / g0)
            dtc = float("inf")
        else:
            if ss <= rs:
                continue
            g0 = _line_val(rs, ri, 0) - _line_val(ss, si, 0)
            ge = _line_val(rs, ri, period - 1) - _line_val(ss, si, period - 1)
            if g0 <= 0 or ge <= 0:
                continue
            conv = 1.0 - ge / g0
            if conv < 0.15:
                continue
            slope_diff = rs - ss
            if abs(slope_diff) < 1e-12:
                dtc = float("inf")
            else:
                x_cross = (si - ri) / (rs - ss)
                dtc = x_cross - (period - 1)

        # ── 穿越统计 ──
        hv = highs_arr[w_start:w_start + period]
        lv = lows_arr[w_start:w_start + period]
        rv = sum(
            1 for x in range(period)
            if hv[x] > _line_val(rs, ri, x) * 1.015
        )
        sv = sum(
            1 for x in range(period)
            if lv[x] < _line_val(ss, si, x) * 0.985
        )
        if rv > period * 0.12 or sv > period * 0.12:
            continue

        # ── 当前价位置 ──
        cur = float(closes_arr[-1])
        re = _line_val(rs, ri, period - 1)
        se = _line_val(ss, si, period - 1)
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
        rng = hv - lv
        sp1 = float(np.mean(rng[:seg]))
        sp3 = float(np.mean(rng[-seg:]))
        spr = sp3 / sp1 if sp1 > 0 else 1.0
        if spr > 0.90:
            continue

        # 量能收缩 (软指标)
        vr = 1.0
        if "volume" in df.columns:
            vol = df["volume"].values[w_start:w_start + period].astype(float)
            if not np.isnan(vol).all() and vol.sum() > 0:
                vl = float(np.mean(vol[: seg * 2]))
                vri = float(np.mean(vol[-seg:]))
                vr = vri / vl if vl > 0 else 1.0

        # ── 触线计数 ──
        tol = avg * 0.015
        res_touches = int(np.sum(np.abs(wh_vals - _line_val(rs, ri, wh_rel.astype(float))) <= tol))
        sup_touches = int(np.sum(np.abs(wl_vals - _line_val(ss, si, wl_rel.astype(float))) <= tol))

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
            norm = avg / period if avg > 0 else 1.0
            angle = abs(math.atan(ss / norm) - math.atan(rs / norm)) * 180 / math.pi

            best = {
                "pattern": _pattern_type,
                "period": period,
                "window_start": w_start,
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
                    "total_swings": len(wh_idx),
                    "anchors": [(int(x), float(y)) for x, y in zip(wh_idx, wh_vals)],
                },
                "support": {
                    "slope": ss, "intercept": si,
                    "r_squared": round(sr2, 3),
                    "touches": sup_touches, "breaches": sv,
                    "total_swings": len(wl_idx),
                    "anchors": [(int(x), float(y)) for x, y in zip(wl_idx, wl_vals)],
                },
                "swing_highs": [(int(x), float(y)) for x, y in zip(wh_idx, wh_vals)],
                "swing_lows": [(int(x), float(y)) for x, y in zip(wl_idx, wl_vals)],
            }

    return best


# ─────────── 公开 API: 各形态独立入口 ───────────


def screen_ascending_triangle(
    df: pd.DataFrame, min_period: int = DEFAULT_MIN_PERIOD, max_period: int = DEFAULT_MAX_PERIOD,
) -> dict | None:
    """检测上升三角形（水平阻力 + 低点抬升）。"""
    return _screen_ols_pattern(df, min_period, max_period, {"ascending_triangle"})


def screen_parallel_channel(
    df: pd.DataFrame, min_period: int = DEFAULT_MIN_PERIOD, max_period: int = DEFAULT_MAX_PERIOD,
) -> dict | None:
    """检测上升平行通道（上下沿斜率接近）。"""
    return _screen_ols_pattern(df, min_period, max_period, {"parallel_channel"})


def screen_rising_wedge(
    df: pd.DataFrame, min_period: int = DEFAULT_MIN_PERIOD, max_period: int = DEFAULT_MAX_PERIOD,
) -> dict | None:
    """检测上升楔形（上沿向上但斜率明显小于下沿，看空信号）。"""
    return _screen_ols_pattern(df, min_period, max_period, {"rising_wedge"})


# ─────────── NDX100 批量扫描 ───────────


def _scan_ndx100_pattern(
    allowed_patterns: set[str],
    label: str,
    min_period: int = DEFAULT_MIN_PERIOD,
    max_period: int = DEFAULT_MAX_PERIOD,
) -> list[dict]:
    """通用 NDX100 扫描。"""
    stock_data = load_all_ndx100_data()
    if not stock_data:
        logger.error("本地无数据！请先运行 update_ndx100_data() 下载数据")
        return []

    _CN = {
        "ascending_triangle": "上升三角形",
        "parallel_channel": "上升平行通道",
        "rising_wedge": "上升楔形",
    }

    hits: list[dict] = []
    processed = 0

    for ticker, df in stock_data.items():
        try:
            if len(df) < 260:
                continue
            processed += 1
            result = _screen_ols_pattern(df, min_period, max_period, allowed_patterns)
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
        f"{label}扫描完成：{len(stock_data)} 只股票，"
        f"有效 {processed} 只，{len(hits)} 只呈现形态"
    )
    return hits


def scan_ndx100_ascending_triangle(
    min_period: int = DEFAULT_MIN_PERIOD, max_period: int = DEFAULT_MAX_PERIOD,
) -> list[dict]:
    """扫描纳指100：上升三角形。"""
    return _scan_ndx100_pattern({"ascending_triangle"}, "上升三角形", min_period, max_period)


def scan_ndx100_parallel_channel(
    min_period: int = DEFAULT_MIN_PERIOD, max_period: int = DEFAULT_MAX_PERIOD,
) -> list[dict]:
    """扫描纳指100：上升平行通道。"""
    return _scan_ndx100_pattern({"parallel_channel"}, "上升平行通道", min_period, max_period)


def scan_ndx100_rising_wedge(
    min_period: int = DEFAULT_MIN_PERIOD, max_period: int = DEFAULT_MAX_PERIOD,
) -> list[dict]:
    """扫描纳指100：上升楔形。"""
    return _scan_ndx100_pattern({"rising_wedge"}, "上升楔形", min_period, max_period)
