"""
上升三角形策略 — KDE (核密度估计) 版
=====================================

利用核密度估计识别高密度成交区 (水平阻力)，
结合低点线性回归确认上升支撑，构成上升三角形。

算法流程:
  1. 预处理: argrelextrema 提取 Swing Highs / Lows（一次性）
  2. 对窗口内 Swing Highs 的价格做 Gaussian KDE，
     寻找概率密度函数的最大峰值 → 阻力区中心价
  3. 峰值尖锐度检查: 峰值附近密度显著高于两侧 → 强阻力
  4. 阻力区位于当前价格区间顶部（上 1/3）
  5. 低点 OLS 线性回归斜率 > 0 → 低点抬升确认
  6. 收敛性: 当前价格已接近阻力区
  7. Minervini Stage 2 趋势过滤
"""

import math

import numpy as np
import pandas as pd
from loguru import logger
from scipy.signal import argrelextrema
from scipy.stats import gaussian_kde

from stock_ana.data_fetcher import load_all_ndx100_data
from stock_ana.strategy_base import check_trend_template


# ─────────── 工具函数 ───────────


def _extract_pivots(
    highs: np.ndarray,
    lows: np.ndarray,
    order: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """用 argrelextrema 提取 Swing Highs / Lows 的索引。"""
    hi_idx = argrelextrema(highs, np.greater_equal, order=order)[0]
    lo_idx = argrelextrema(lows, np.less_equal, order=order)[0]
    return hi_idx, lo_idx


def _ols(xs: np.ndarray, ys: np.ndarray) -> tuple[float, float, float, float]:
    """
    OLS 线性回归。

    Returns:
        (slope, intercept, r², max_abs_residual_pct)
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


def _find_kde_resistance(
    prices: np.ndarray,
    n_eval: int = 512,
    bw_method: str | float = "silverman",
) -> tuple[float, float, np.ndarray, np.ndarray] | None:
    """
    对价格序列做 Gaussian KDE, 找到最大密度峰值。

    Args:
        prices: 用于 KDE 的价格数组 (通常是 Swing Highs)
        n_eval: 评估点数量
        bw_method: 带宽方法 ("silverman", "scott", 或浮点数因子)

    Returns:
        (peak_price, peak_density, eval_grid, density_values) 或 None
    """
    if len(prices) < 3:
        return None

    # 归一化价格以得到更好的 KDE 效果
    p_min, p_max = float(prices.min()), float(prices.max())
    p_range = p_max - p_min
    if p_range < 1e-8:
        return None

    try:
        kde = gaussian_kde(prices, bw_method=bw_method)
    except Exception:
        return None

    # 在价格范围内均匀采样评估密度
    margin = p_range * 0.1
    grid = np.linspace(p_min - margin, p_max + margin, n_eval)
    density = kde(grid)

    # 找到全局最大峰值
    peak_idx = int(np.argmax(density))
    peak_price = float(grid[peak_idx])
    peak_density = float(density[peak_idx])

    return peak_price, peak_density, grid, density


def _peak_sharpness(
    grid: np.ndarray, density: np.ndarray,
    peak_price: float, peak_density: float,
    band_pct: float = 0.02,
) -> float:
    """
    衡量峰值的尖锐度: 峰值密度 / 峰值两侧 band_pct 范围外的平均密度。
    值越大说明阻力区越集中。

    Args:
        grid: KDE 评估网格
        density: KDE 密度值
        peak_price: 峰值价格
        peak_density: 峰值密度
        band_pct: 阻力带宽度 (占价格百分比)

    Returns:
        sharpness ratio (≥ 1.0)
    """
    band = abs(peak_price) * band_pct
    outside_mask = np.abs(grid - peak_price) > band
    if outside_mask.sum() < 2:
        return 1.0
    avg_outside = float(density[outside_mask].mean())
    if avg_outside < 1e-12:
        return float("inf")
    return peak_density / avg_outside


# ─────────── 主检测函数 ───────────


def screen_ascending_triangle_kde(
    df: pd.DataFrame,
    min_period: int = 25,
    max_period: int = 120,
) -> dict | None:
    """
    基于 KDE 阻力位确认的上升三角形检测。

    步骤:
      1. 预处理: argrelextrema 提取 Swing Highs / Lows
      2. 滑动窗口: 遍历不同周期
         a. 对窗口内 Swing Highs 做 Gaussian KDE
         b. 找到最大密度峰值 (阻力区中心)
         c. 检查峰值尖锐度 (集中度 ≥ 2.0)
         d. 检查峰值位于价格区间顶部 (上 1/3)
         e. 对窗口内 Swing Lows 做 OLS 回归, 要求正斜率
      3. 收敛性: 当前价接近阻力区
      4. 趋势过滤: Minervini Stage 2

    Returns:
        形态信息 dict 或 None
    """
    # ── 前置检查 ──
    if len(df) < min_period:
        return None
    if not check_trend_template(df):
        return None

    # ── 步骤 1: 预处理 — 提取 Pivot Points ──
    highs_arr = df["high"].values.astype(float)
    lows_arr = df["low"].values.astype(float)
    closes_arr = df["close"].values.astype(float)

    n = len(df)
    order = 3 if n < 100 else 5 if n < 300 else 7
    hi_idx, lo_idx = _extract_pivots(highs_arr, lows_arr, order=order)

    if len(hi_idx) < 3 or len(lo_idx) < 3:
        return None

    # ── 步骤 2: 滑动窗口 ──
    best: dict | None = None
    best_sc = -1.0

    for period in range(min_period, min(max_period + 1, n + 1), 5):
        w_start = n - period

        # 筛选落在窗口内的 pivot 点
        mask_h = (hi_idx >= w_start) & (hi_idx < n)
        mask_l = (lo_idx >= w_start) & (lo_idx < n)
        wh_idx = hi_idx[mask_h]
        wl_idx = lo_idx[mask_l]

        if len(wh_idx) < 3 or len(wl_idx) < 3:
            continue

        wh_vals = highs_arr[wh_idx]
        wl_vals = lows_arr[wl_idx]

        # 窗口内价格范围
        win_high = float(highs_arr[w_start:].max())
        win_low = float(lows_arr[w_start:].min())
        win_range = win_high - win_low
        if win_range < 1e-8:
            continue

        avg_price = float(closes_arr[w_start:].mean())

        # ── 步骤 2a: KDE 阻力区检测 ──
        kde_result = _find_kde_resistance(wh_vals)
        if kde_result is None:
            continue

        peak_price, peak_density, grid, density = kde_result

        # ── 步骤 2b: 峰值尖锐度检查 ──
        # 尖锐度 ≥ 1.5 → 存在集中的阻力区
        sharpness = _peak_sharpness(grid, density, peak_price, peak_density, band_pct=0.02)
        if sharpness < 1.5:
            continue

        # ── 步骤 2c: 峰值必须位于价格区间顶部 (上 1/3) ──
        # peak_price 应该在 win_low + win_range * 2/3 以上
        if peak_price < win_low + win_range * 0.60:
            continue

        # ── 步骤 2d: Swing Highs 必须聚集在峰值附近 ──
        # 至少 50% 的 Swing Highs 落在 peak_price ± 2% 的范围内
        band = abs(peak_price) * 0.025
        near_peak = np.sum(np.abs(wh_vals - peak_price) <= band)
        if near_peak < len(wh_vals) * 0.40:
            continue

        # ── 步骤 2e: 低点 OLS 回归 — 要求低点抬升 ──
        wl_rel = (wl_idx - w_start).astype(float)
        ss, si, sr2, s_max_res = _ols(wl_rel, wl_vals)

        # 支撑线斜率必须显著 > 0: 日均斜率占均价比 ≥ 0.02%
        ss_pct = ss / avg_price * 100 if avg_price > 0 else 0.0
        if ss_pct < 0.02:
            continue

        # 残差不能过大 (≤ 4%)
        if s_max_res > 4.0:
            continue

        # ── 步骤 3: 收敛性检查 ──
        # 当前支撑线终点值
        support_end = ss * (period - 1) + si
        # 阻力区 = peak_price (水平)
        gap = peak_price - support_end
        gap_initial = peak_price - (ss * 0 + si)

        if gap_initial <= 0:
            continue

        # 收敛 = 间距缩窄比例
        conv = 1.0 - gap / gap_initial if gap_initial > 0 else 0.0
        if conv < 0.15:
            continue

        # 当前价位置
        cur = float(closes_arr[-1])
        # 当前价必须在支撑与阻力之间，且偏向阻力
        if cur < support_end - win_range * 0.05:
            continue
        if cur > peak_price * 1.03:  # 允许小幅突破
            continue

        ch = peak_price - support_end
        if ch <= 0:
            continue
        pos = (cur - support_end) / ch

        # ── VCP 微观: 波幅递缩 ──
        hv = highs_arr[w_start:w_start + period]
        lv = lows_arr[w_start:w_start + period]
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

        # ── 穿越统计 ──
        rv = sum(
            1 for x in range(period)
            if hv[x] > peak_price * 1.015
        )
        sv = sum(
            1 for x in range(period)
            if lv[x] < (ss * x + si) * 0.985
        )
        if rv > period * 0.12 or sv > period * 0.12:
            continue

        # ── 评分 ──
        sc = (
            conv * 20
            + sharpness * 5        # KDE 尖锐度加分
            + near_peak * 3         # 聚集在阻力区的 swing high 数量
            + (1 - spr) * 20        # 波幅收缩
            + max(0, 1 - vr) * 10   # 量能收缩
            + pos * 10              # 价格接近阻力
            + sr2 * 10              # 支撑线拟合好坏
            - (rv + sv) * 1.0       # 穿越扣分
        )

        if sc > best_sc:
            best_sc = sc

            # 收敛点估计 (线性外推)
            if ss > 1e-12:
                x_cross = (peak_price - si) / ss
                dtc = x_cross - (period - 1)
            else:
                dtc = float("inf")

            norm = avg_price / period if avg_price > 0 else 1.0
            angle = abs(math.atan(ss / norm)) * 180 / math.pi

            # 构造与 V4 兼容的返回结构
            # 阻力线: 水平线 slope=0, intercept=peak_price
            rs, ri = 0.0, peak_price

            best = {
                "pattern": "ascending_triangle",
                "method": "kde",
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
                "kde_peak_price": round(peak_price, 2),
                "kde_sharpness": round(sharpness, 2),
                "kde_near_peak_count": int(near_peak),
                "resistance": {
                    "slope": rs,
                    "intercept": ri,
                    "r_squared": 0.0,  # KDE 不用 R²
                    "touches": int(near_peak),
                    "breaches": rv,
                    "total_swings": len(wh_idx),
                    "anchors": [(int(x), float(y)) for x, y in zip(wh_idx, wh_vals)],
                },
                "support": {
                    "slope": ss,
                    "intercept": si,
                    "r_squared": round(sr2, 3),
                    "touches": len(wl_idx),
                    "breaches": sv,
                    "total_swings": len(wl_idx),
                    "anchors": [(int(x), float(y)) for x, y in zip(wl_idx, wl_vals)],
                },
                "swing_highs": [(int(x), float(y)) for x, y in zip(wh_idx, wh_vals)],
                "swing_lows": [(int(x), float(y)) for x, y in zip(wl_idx, wl_vals)],
            }

    return best


# ─────────── NDX100 批量扫描 ───────────


def scan_ndx100_ascending_triangle_kde(
    min_period: int = 25,
    max_period: int = 120,
) -> list[dict]:
    """扫描纳指100中呈现上升三角形的股票 (KDE 版)。"""
    stock_data = load_all_ndx100_data()
    if not stock_data:
        logger.error("本地无数据！请先运行 update_ndx100_data() 下载数据")
        return []

    hits: list[dict] = []
    processed = 0

    for ticker, df in stock_data.items():
        try:
            if len(df) < 260:
                continue
            processed += 1
            result = screen_ascending_triangle_kde(df, min_period, max_period)
            if result is not None:
                logger.success(
                    f"✅ {ticker} 上升三角形(KDE) "
                    f"({result['period']}d, "
                    f"收敛{result['convergence_ratio']:.0%}, "
                    f"KDE尖锐度{result['kde_sharpness']:.1f}, "
                    f"波幅缩{1 - result['spread_contraction']:.0%})"
                )
                hits.append({"ticker": ticker, "df": df, "pattern_info": result})
        except Exception as e:
            logger.error(f"{ticker}: {e}")

    logger.info(
        f"KDE上升三角形扫描完成：{len(stock_data)} 只股票，"
        f"有效 {processed} 只，{len(hits)} 只呈现形态"
    )
    return hits
