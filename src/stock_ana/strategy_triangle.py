"""
上升三角形 / 上升楔形策略模块

基于极值边界法检测收敛三角形形态。
"""

import math

import pandas as pd
from loguru import logger

from stock_ana.data_fetcher import load_all_ndx100_data
from stock_ana.strategy_base import (
    find_swing_points,
    line_from_two_points,
    line_value,
    count_breaches_and_touches,
    find_best_boundary_line,
)


def screen_ascending_triangle(
    df: pd.DataFrame,
    min_period: int = 40,
    max_period: int = 120,
    swing_order: int = 5,
    min_touches: int = 2,
    min_convergence_angle_deg: float = 3.0,
    touch_tolerance: float = 0.015,
    max_breach_ratio: float = 0.05,
) -> dict | None:
    """
    检测上升三角形 / 上升楔形（基于极值边界法，非回归）。

    算法：
    1. 在 min_period ~ max_period 窗口中找 swing highs / lows
    2. 枚举两个极值点组合，构造上轨线和下轨线
    3. 要求 K 线不穿越边界（穿越比例 < 5%）
    4. 要求两条边界线收敛，且收敛角度 > min_convergence_angle_deg
    5. 要求边界线有多次有效触及（测试）

    上升三角形：上轨基本水平，下轨上倾
    上升楔形：  两线都上倾，下轨斜率 > 上轨斜率

    Returns:
        形态信息 dict 或 None
    """
    if len(df) < min_period:
        return None

    best_result = None
    best_score = -1

    for period in range(min_period, min(max_period + 1, len(df) + 1), 10):
        window = df.iloc[-period:]
        highs = window["high"]
        lows = window["low"]
        avg_price = window["close"].mean()

        swing_all = find_swing_points(highs, order=swing_order)
        swing_all_low = find_swing_points(lows, order=swing_order)

        sw_highs = [(i, v) for i, v, t in swing_all if t == "high"]
        sw_lows = [(i, v) for i, v, t in swing_all_low if t == "low"]

        if len(sw_highs) < 2 or len(sw_lows) < 2:
            continue

        # 找最佳上轨（连接 swing highs，K 线不穿越上方）
        upper = find_best_boundary_line(
            sw_highs, highs, period, is_upper=True,
            touch_tolerance=touch_tolerance,
            max_breach_ratio=max_breach_ratio,
            min_touches=min_touches,
        )
        if upper is None:
            continue

        res_slope_tmp = upper[0]

        # 找最佳下轨（优先选择与上轨收敛的线）
        lower = find_best_boundary_line(
            sw_lows, lows, period, is_upper=False,
            touch_tolerance=touch_tolerance,
            max_breach_ratio=max_breach_ratio,
            min_touches=min_touches,
            converge_with_slope=res_slope_tmp,
        )
        if lower is None:
            continue

        res_slope, res_intercept, res_touches, res_breaches, res_anchors = upper
        sup_slope, sup_intercept, sup_touches, sup_breaches, sup_anchors = lower

        # ---- 收敛检查 ----
        start_gap = abs(line_value(res_slope, res_intercept, 0)
                        - line_value(sup_slope, sup_intercept, 0))
        end_gap = abs(line_value(res_slope, res_intercept, period - 1)
                      - line_value(sup_slope, sup_intercept, period - 1))

        if end_gap >= start_gap:
            continue

        convergence_ratio = 1 - end_gap / start_gap if start_gap > 0 else 0
        if convergence_ratio < 0.15:
            continue

        # ---- 角度检查 ----
        norm = avg_price / period if avg_price > 0 else 1
        angle_upper = math.atan(res_slope / norm)
        angle_lower = math.atan(sup_slope / norm)
        convergence_angle_deg = abs(angle_upper - angle_lower) * 180 / math.pi

        if convergence_angle_deg < min_convergence_angle_deg:
            continue

        # ---- 下轨必须上倾 ----
        sup_slope_pct = (sup_slope / avg_price) * 100 if avg_price > 0 else 0
        res_slope_pct = (res_slope / avg_price) * 100 if avg_price > 0 else 0

        if sup_slope <= res_slope:
            continue

        # ---- 当前价在通道内 ----
        last_x = period - 1
        res_val = line_value(res_slope, res_intercept, last_x)
        sup_val = line_value(sup_slope, sup_intercept, last_x)
        curr_close = window["close"].iloc[-1]
        margin = max(end_gap * 0.1, avg_price * 0.005)
        if curr_close < sup_val - margin or curr_close > res_val + margin:
            continue

        # ---- 判断形态类型（只保留看涨形态） ----
        if abs(res_slope_pct) < 0.08 and sup_slope_pct > 0.03:
            pattern_type = "ascending_triangle"
        elif res_slope_pct > 0 and sup_slope_pct > res_slope_pct:
            pattern_type = "rising_wedge"
        else:
            continue

        # ---- 收敛时间估算 ----
        slope_diff = res_slope - sup_slope
        if abs(slope_diff) > 1e-12:
            x_convergence = (sup_intercept - res_intercept) / (res_slope - sup_slope)
            days_to_convergence = x_convergence - (period - 1)
        else:
            days_to_convergence = float("inf")

        max_future_days = 20
        if days_to_convergence <= 0:
            convergence_status = "converged"
        elif days_to_convergence <= max_future_days:
            convergence_status = "imminent"
        else:
            continue

        # ---- 评分 ----
        urgency_bonus = max(0, (max_future_days - days_to_convergence)) * 0.5
        score = (
            (res_touches + sup_touches) * 5
            - (res_breaches + sup_breaches) * 20
            + convergence_angle_deg * 2
            + convergence_ratio * 10
            + urgency_bonus
        )

        if score > best_score:
            best_score = score
            window_start = len(df) - period
            best_result = {
                "pattern": pattern_type,
                "period": period,
                "window_start": window_start,
                "convergence_angle_deg": convergence_angle_deg,
                "convergence_ratio": convergence_ratio,
                "days_to_convergence": round(days_to_convergence, 1),
                "convergence_status": convergence_status,
                "resistance": {
                    "slope": res_slope, "intercept": res_intercept,
                    "touches": res_touches, "breaches": res_breaches,
                    "anchors": [(window_start + x, y) for x, y in res_anchors],
                },
                "support": {
                    "slope": sup_slope, "intercept": sup_intercept,
                    "touches": sup_touches, "breaches": sup_breaches,
                    "anchors": [(window_start + x, y) for x, y in sup_anchors],
                },
                "swing_highs": [(window_start + x, y) for x, y in sw_highs],
                "swing_lows": [(window_start + x, y) for x, y in sw_lows],
            }

    return best_result


def scan_ndx100_ascending_triangle(
    min_period: int = 40,
    max_period: int = 120,
) -> list[dict]:
    """
    扫描纳指100中呈现上升三角形 / 上升楔形的股票。
    """
    stock_data = load_all_ndx100_data()

    if not stock_data:
        logger.error("本地无数据！请先运行 update_ndx100_data() 下载数据")
        return []

    hits: list[dict] = []
    processed = 0

    for ticker, df in stock_data.items():
        try:
            if len(df) < min_period:
                continue

            processed += 1
            result = screen_ascending_triangle(
                df, min_period=min_period, max_period=max_period,
            )
            if result is not None:
                _PATTERN_CN = {
                    "ascending_triangle": "上升三角形",
                    "rising_wedge": "上升楔形",
                    "symmetrical_triangle": "对称三角形",
                    "descending_wedge": "下降楔形",
                }
                ptype = _PATTERN_CN.get(result["pattern"], result["pattern"])
                status_cn = "已收敛" if result["convergence_status"] == "converged" else "即将收敛"
                dtc = result["days_to_convergence"]
                dtc_str = f"{dtc:.0f}日后" if dtc > 0 else "已过"
                logger.success(
                    f"✅ {ticker} 呈现{ptype}【{status_cn}】"
                    f"（周期 {result['period']} 日，"
                    f"收敛点 {dtc_str}，"
                    f"角度 {result['convergence_angle_deg']:.1f}°，"
                    f"上轨测试 {result['resistance']['touches']} 次，"
                    f"下轨测试 {result['support']['touches']} 次）"
                )
                hits.append({"ticker": ticker, "df": df, "pattern_info": result})
        except Exception as e:
            logger.error(f"{ticker}: 处理失败 - {e}")
            continue

    logger.info(
        f"上升三角形扫描完成：本地共 {len(stock_data)} 只股票，"
        f"有效处理 {processed} 只，{len(hits)} 只呈现形态"
    )
    return hits
