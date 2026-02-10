"""
策略公共基础模块 - 趋势模板、几何工具函数
"""

import math

import numpy as np
import pandas as pd
from loguru import logger


# ──────────────────────────────────────────
# Minervini 趋势模板 (Stage 2)
# ──────────────────────────────────────────

def check_trend_template(df: pd.DataFrame) -> bool:
    """
    Minervini 趋势模板 (Stage 2) 严格检查：
    1. 股价 > 150日均线 > 200日均线
    2. 200日均线呈上升趋势（至少 1 个月）
    3. 50日均线 > 150日均线
    4. 股价 > 50日均线（或非常接近，允许回踩）
    5. 股价较 52周低点至少上涨 25%
    6. 股价处于 52周高点的 25% 范围内（靠近新高）
    """
    if len(df) < 260:
        return False

    curr = df.iloc[-1]
    close = curr["close"]

    # 计算均线
    ma_50 = df["close"].rolling(50).mean().iloc[-1]
    ma_150 = df["close"].rolling(150).mean().iloc[-1]
    ma_200 = df["close"].rolling(200).mean().iloc[-1]

    # 计算 200日均线 趋势 (比较当前与 20 天前的 MA200)
    ma_200_prev = df["close"].rolling(200).mean().iloc[-20]

    # 52周高低点
    high_52w = df["high"].iloc[-260:].max()
    low_52w = df["low"].iloc[-260:].min()

    # --- 核心判定逻辑 ---

    # 1. 均线多头排列
    if not (close > ma_150 and ma_150 > ma_200):
        return False

    # 2. 200日线向上
    if ma_200 <= ma_200_prev:
        return False

    # 3. 50日线在 150/200 之上
    if ma_50 <= ma_150:
        return False

    # 4. 强力支撑：价格不应深跌破 50日线 (允许短期跌破但需在附近)
    # 稍微放宽一点：允许跌破 50日线，但不能跌破 200日线
    if close < ma_200:
        return False

    # 5. 底部脱离与头部临近
    if close < low_52w * 1.25:  # 较底部上涨 25%
        return False
    if close < high_52w * 0.75:  # 处于高点的 25% (放宽到 75% 水位以上)
        return False

    return True


# ──────────────────────────────────────────
# 几何工具函数（供三角形等策略使用）
# ──────────────────────────────────────────

def find_swing_points(series: pd.Series, order: int = 5) -> list[tuple[int, float, str]]:
    """
    寻找局部极值点（swing high / swing low）。

    Args:
        series: 价格序列
        order: 极值点左右各需要 order 根K线比它低(高)才算

    Returns:
        [(index_位置, 价格值, "high"|"low"), ...]
    """
    points = []
    values = series.values
    for i in range(order, len(values) - order):
        val = values[i]
        yield_high = True
        yield_low = True
        for j in range(1, order + 1):
            if values[i - j] >= val or values[i + j] >= val:
                yield_high = False
            if values[i - j] <= val or values[i + j] <= val:
                yield_low = False
        if yield_high or yield_low:
            points.append((i, val, "high" if yield_high else "low"))
    return points


def line_from_two_points(x1: float, y1: float, x2: float, y2: float):
    """两点确定一条线，返回 (slope, intercept)"""
    if abs(x2 - x1) < 1e-12:
        return 0.0, (y1 + y2) / 2
    slope = (y2 - y1) / (x2 - x1)
    intercept = y1 - slope * x1
    return slope, intercept


def line_value(slope: float, intercept: float, x: float) -> float:
    return slope * x + intercept


def count_breaches_and_touches(
    slope: float, intercept: float,
    candle_series: pd.Series,
    is_upper: bool,
    touch_tolerance: float,
    max_brief_days: int = 2,
) -> tuple[int, int]:
    """
    统计K线对趋势线的穿越次数和触及次数。

    突破后在 max_brief_days 天内收回的不计为有效穿越（"假突破"宽容）。

    Args:
        slope, intercept: 趋势线参数
        candle_series: high 序列(上轨) 或 low 序列(下轨)
        is_upper: True=上轨(检查high是否穿越), False=下轨(检查low是否穿越)
        touch_tolerance: 触及容差（价格的百分比）
        max_brief_days: 允许的最长"假突破"天数（<=该天数的突破被宽容）

    Returns:
        (breach_count, touch_count)
    """
    touches = 0
    vals = candle_series.values
    n = len(vals)

    # 先标记每根K线的状态
    breach_flags = [False] * n
    for i in range(n):
        lv = line_value(slope, intercept, i)
        diff = vals[i] - lv
        tol = abs(lv) * touch_tolerance

        if is_upper:
            if diff > tol:
                breach_flags[i] = True
            elif abs(diff) <= tol:
                touches += 1
        else:
            if diff < -tol:
                breach_flags[i] = True
            elif abs(diff) <= tol:
                touches += 1

    # 将连续突破分组为"事件"，短事件（<=max_brief_days天）宽容
    breaches = 0
    i = 0
    while i < n:
        if breach_flags[i]:
            event_start = i
            while i < n and breach_flags[i]:
                i += 1
            event_len = i - event_start
            if event_len > max_brief_days:
                breaches += event_len  # 持续突破，全部计入
            # else: 短时突破，宽容不计
        else:
            i += 1

    return breaches, touches


def find_best_boundary_line(
    swing_points: list[tuple[int, float]],
    candle_series: pd.Series,
    period: int,
    is_upper: bool,
    touch_tolerance: float = 0.015,
    max_breach_ratio: float = 0.05,
    min_touches: int = 3,
    converge_with_slope: float | None = None,
) -> tuple[float, float, int, int, list[tuple[int, float]]] | None:
    """
    在候选极值点中，枚举两点组合找最佳边界线。

    最佳 = 穿越最少 + 触及最多 的组合。

    Args:
        swing_points: [(x, y), ...] 极值点
        candle_series: high(上轨) 或 low(下轨) 序列
        period: 窗口长度
        is_upper: True=找上轨, False=找下轨
        touch_tolerance: 触及容差
        max_breach_ratio: 最大允许穿越比例
        min_touches: 最少触及次数
        converge_with_slope: 如果提供，优先选择与该斜率收敛的线

    Returns:
        (slope, intercept, touches, breaches, anchor_points) 或 None
    """
    if len(swing_points) < 2:
        return None

    best = None
    best_score = -1
    max_breaches = int(period * max_breach_ratio)

    for i in range(len(swing_points)):
        for j in range(i + 1, len(swing_points)):
            x1, y1 = swing_points[i]
            x2, y2 = swing_points[j]

            slope, intercept = line_from_two_points(x1, y1, x2, y2)

            # 检查其他极值点是否大致在线的正确侧
            brief_tol = touch_tolerance * 2.5
            violations = 0
            max_violations = 2
            valid = True
            for k, (px, py) in enumerate(swing_points):
                if k == i or k == j:
                    continue
                line_at_px = line_value(slope, intercept, px)
                if is_upper and py > line_at_px * (1 + touch_tolerance):
                    if py > line_at_px * (1 + brief_tol):
                        valid = False
                        break
                    violations += 1
                    if violations > max_violations:
                        valid = False
                        break
                if not is_upper and py < line_at_px * (1 - touch_tolerance):
                    if py < line_at_px * (1 - brief_tol):
                        valid = False
                        break
                    violations += 1
                    if violations > max_violations:
                        valid = False
                        break
            if not valid:
                continue

            b, t = count_breaches_and_touches(
                slope, intercept, candle_series, is_upper, touch_tolerance,
            )

            if b > max_breaches:
                continue
            if t < min_touches:
                continue

            # 评分：触及越多越好，穿越越少越好
            score = t * 10 - b * 50

            # 如果指定了收敛目标斜率，优先选择收敛方向的线
            if converge_with_slope is not None:
                if is_upper:
                    converges = slope < converge_with_slope
                else:
                    converges = slope > converge_with_slope
                if converges:
                    score += 100

            if score > best_score:
                best_score = score
                anchors = []
                for px, py in swing_points:
                    line_at_px = line_value(slope, intercept, px)
                    if abs(py - line_at_px) <= abs(line_at_px) * touch_tolerance:
                        anchors.append((px, py))
                best = (slope, intercept, t, b, anchors)

    return best
