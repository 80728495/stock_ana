"""
股票筛选模块 - 基于技术指标的筛选策略
"""

import math

import numpy as np
import pandas as pd
from loguru import logger

from stock_ana.data_fetcher import load_all_ndx100_data
from stock_ana.indicators import add_macd, add_vegas_channel


def screen_golden_cross(df: pd.DataFrame, short_ma: str = "sma_5", long_ma: str = "sma_20") -> bool:
    """
    金叉筛选：短期均线上穿长期均线

    Args:
        df: 带有均线指标的 DataFrame
        short_ma: 短期均线列名
        long_ma: 长期均线列名

    Returns:
        True 如果最近发生金叉
    """
    if len(df) < 2:
        return False

    prev = df.iloc[-2]
    curr = df.iloc[-1]

    return prev[short_ma] <= prev[long_ma] and curr[short_ma] > curr[long_ma]


def screen_rsi_oversold(df: pd.DataFrame, threshold: float = 30.0) -> bool:
    """RSI 超卖筛选：RSI 低于阈值"""
    if "rsi" not in df.columns or df["rsi"].isna().all():
        return False
    return df["rsi"].iloc[-1] < threshold


def screen_macd_bullish(df: pd.DataFrame) -> bool:
    """MACD 看涨筛选：MACD 柱状图由负转正（仅检测最后一天）"""
    if "macd_hist" not in df.columns or len(df) < 2:
        return False
    return df["macd_hist"].iloc[-2] < 0 and df["macd_hist"].iloc[-1] > 0


def screen_macd_cross_in_period(df: pd.DataFrame, lookback_days: int = 5) -> bool:
    """
    检测最近 N 个交易日内是否发生了 MACD 金叉
    （MACD 线上穿信号线，即 macd_hist 由负转正）

    过滤噪声的条件：
    1. 正面阈值：转正后 macd_hist > 收盘价 × 0.1%（避免微弱穿越）
    2. 负面深度：交叉前的负值期内，最低 hist < -收盘价 × 0.1%
       （排除 hist 长期为正、只短暂碰一下负值又弹回的假金叉）
    3. 金叉后若又发生死叉，则该金叉无效（以最后一次有意义的交叉为准）

    Args:
        df: 带有 macd_hist 和 close 列的 DataFrame
        lookback_days: 回看交易日数（1周 ≈ 5个交易日）

    Returns:
        True 如果在回看期内最后一次有意义的交叉是 MACD 金叉
    """
    if "macd_hist" not in df.columns or "close" not in df.columns or len(df) < 2:
        return False

    hist_all = df["macd_hist"]
    close_all = df["close"]

    # 取最近 lookback_days + 1 天（需要前一天做对比）
    n = lookback_days + 1
    start_idx = len(df) - n

    # 记录最后一次有意义的交叉类型：True=金叉, False=死叉, None=无
    last_cross_is_golden = None

    for i in range(max(start_idx + 1, 1), len(df)):
        prev_hist = hist_all.iloc[i - 1]
        curr_hist = hist_all.iloc[i]
        curr_close = close_all.iloc[i]
        threshold = curr_close * 0.001

        # 检查金叉：负 → 正
        if prev_hist < 0 and curr_hist > 0:
            # 条件1: 正面阈值
            if curr_hist <= threshold:
                continue

            # 条件2: 负面深度 —— 回溯交叉前的连续负值期
            min_neg_hist = prev_hist
            for j in range(i - 2, -1, -1):
                h = hist_all.iloc[j]
                if h >= 0:
                    break
                if h < min_neg_hist:
                    min_neg_hist = h

            if abs(min_neg_hist) < threshold:
                continue  # 负值期太浅，噪声波动

            last_cross_is_golden = True

        # 检查死叉：正 → 负（同样需要超过阈值才算有意义）
        elif prev_hist > 0 and curr_hist < 0:
            if abs(curr_hist) > threshold:
                last_cross_is_golden = False

    return last_cross_is_golden is True


def screen_bollinger_squeeze(df: pd.DataFrame, threshold: float = 0.05) -> bool:
    """布林带收窄筛选：带宽占比低于阈值"""
    if "bb_upper" not in df.columns:
        return False
    curr = df.iloc[-1]
    bandwidth = (curr["bb_upper"] - curr["bb_lower"]) / curr["bb_mid"]
    return bandwidth < threshold


def run_screen(df: pd.DataFrame, strategies: list[str] | None = None) -> dict[str, bool]:
    """
    运行指定的筛选策略

    Args:
        df: 带有技术指标的 DataFrame
        strategies: 要运行的策略列表，默认全部运行

    Returns:
        策略名称 -> 是否通过筛选
    """
    all_strategies = {
        "golden_cross": screen_golden_cross,
        "rsi_oversold": screen_rsi_oversold,
        "macd_bullish": screen_macd_bullish,
        "bollinger_squeeze": screen_bollinger_squeeze,
    }

    if strategies is None:
        strategies = list(all_strategies.keys())

    results = {}
    for name in strategies:
        if name in all_strategies:
            results[name] = all_strategies[name](df)

    return results


def scan_ndx100_macd_cross(lookback_days: int = 5) -> list[dict]:
    """
    基于本地已存储的数据，扫描纳指100成分股中最近 lookback_days 个交易日
    内发生 MACD 金叉的股票。

    注意：运行此函数前，需先调用 update_ndx100_data() 更新本地数据。

    Args:
        lookback_days: 回看交易日数（默认 5 天，约 1 周）

    Returns:
        包含金叉股票信息的列表，每项为 {"ticker": str, "df": DataFrame}
    """
    stock_data = load_all_ndx100_data()

    if not stock_data:
        logger.error("本地无数据！请先运行 update_ndx100_data() 下载数据")
        return []

    hits: list[dict] = []
    processed = 0

    for ticker, df in stock_data.items():
        try:
            if len(df) < 35:
                logger.debug(f"{ticker}: 数据不足（{len(df)} 行），跳过")
                continue

            processed += 1
            df = add_macd(df.copy())
            if screen_macd_cross_in_period(df, lookback_days=lookback_days):
                logger.success(f"✅ {ticker} 在最近 {lookback_days} 个交易日内发生 MACD 金叉")
                hits.append({"ticker": ticker, "df": df})
        except Exception as e:
            logger.error(f"{ticker}: 处理失败 - {e}")
            continue

    logger.info(f"扫描完成：本地共 {len(stock_data)} 只股票，"
                f"有效处理 {processed} 只，{len(hits)} 只发生 MACD 金叉")
    return hits


# ──────── Vegas 长期通道策略 ────────


def screen_vegas_channel_touch(df: pd.DataFrame, lookback_days: int = 5,
                                half_year_days: int = 120) -> bool:
    """
    Vegas 长期通道回踩策略：

    条件：
    1. 股价在最近半年（~120 个交易日）内达到最高点后开始波动下行
    2. 最近 lookback_days 个交易日内，价格触及 Vegas 通道（EMA144/EMA169 区域）
    3. 没有有效跌破 EMA169：
       - 从未跌破 EMA169，或
       - 跌破后当日或次日收盘价收回 EMA169 上方

    Args:
        df: 带有 close, low, high, ema_144, ema_169 列的 DataFrame
        lookback_days: 回看交易日数
        half_year_days: 半年的交易日数

    Returns:
        True 如果满足 Vegas 通道回踩条件
    """
    required = {"close", "low", "high", "ema_144", "ema_169"}
    if not required.issubset(df.columns) or len(df) < half_year_days:
        return False

    # ---- 条件1: 半年内到达过高点，且当前已从高点回落 ----
    half_year = df.iloc[-half_year_days:]
    peak_idx = half_year["high"].idxmax()
    peak_price = half_year.loc[peak_idx, "high"]

    # 高点不能在最近 lookback_days 内（必须已开始下行）
    recent_start = df.index[-lookback_days]
    if peak_idx >= recent_start:
        return False

    # 高点后价格确实下行：当前收盘 < 高点的 95%
    curr_close = df["close"].iloc[-1]
    if curr_close >= peak_price * 0.95:
        return False

    # ---- 条件2: 最近 lookback_days 内触及 Vegas 通道 ----
    recent = df.iloc[-lookback_days:]
    touched = False
    for _, row in recent.iterrows():
        ema_upper = max(row["ema_144"], row["ema_169"])
        ema_lower = min(row["ema_144"], row["ema_169"])
        # 触及通道：最低价 <= 通道上沿（即价格下探到通道区域）
        if row["low"] <= ema_upper:
            touched = True
            break

    if not touched:
        return False

    # ---- 条件3: 没有有效跌破 EMA169（通道下沿） ----
    # 检查最近 lookback_days：如果跌破 EMA169，当日或次日必须收回
    for i in range(len(df) - lookback_days, len(df)):
        row = df.iloc[i]
        ema_lower = min(row["ema_144"], row["ema_169"])
        close_i = row["close"]

        if close_i < ema_lower:
            # 收盘价跌破了通道下沿，检查次日是否收回
            if i + 1 < len(df):
                next_row = df.iloc[i + 1]
                next_ema_lower = min(next_row["ema_144"], next_row["ema_169"])
                if next_row["close"] < next_ema_lower:
                    return False  # 次日也没收回 → 有效跌破
            else:
                # 跌破发生在最后一天，没有次日数据 → 暂不算有效跌破
                # 但收盘在通道下沿以下，风险大，保守过滤
                return False

    return True


def scan_ndx100_vegas_touch(lookback_days: int = 5) -> list[dict]:
    """
    基于本地数据，扫描纳指100中满足 Vegas 通道回踩条件的股票。

    Args:
        lookback_days: 回看交易日数（默认 5 天）

    Returns:
        [{"ticker": str, "df": DataFrame}, ...]
    """
    stock_data = load_all_ndx100_data()

    if not stock_data:
        logger.error("本地无数据！请先运行 update_ndx100_data() 下载数据")
        return []

    hits: list[dict] = []
    processed = 0

    for ticker, df in stock_data.items():
        try:
            if len(df) < 170:
                logger.debug(f"{ticker}: 数据不足（{len(df)} 行），需至少 170 行计算 EMA169")
                continue

            processed += 1
            df = add_vegas_channel(df.copy())
            if screen_vegas_channel_touch(df, lookback_days=lookback_days):
                logger.success(f"✅ {ticker} 在最近 {lookback_days} 个交易日触及 Vegas 通道回踩")
                hits.append({"ticker": ticker, "df": df})
        except Exception as e:
            logger.error(f"{ticker}: 处理失败 - {e}")
            continue

    logger.info(f"Vegas 通道扫描完成：本地共 {len(stock_data)} 只股票，"
                f"有效处理 {processed} 只，{len(hits)} 只满足条件")
    return hits


# ──────── 上升三角形 / 上升楔形策略 ────────


def _find_swing_points(series: pd.Series, order: int = 5) -> list[tuple[int, float, str]]:
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


def _line_from_two_points(x1: float, y1: float, x2: float, y2: float):
    """两点确定一条线，返回 (slope, intercept)"""
    if abs(x2 - x1) < 1e-12:
        return 0.0, (y1 + y2) / 2
    slope = (y2 - y1) / (x2 - x1)
    intercept = y1 - slope * x1
    return slope, intercept


def _line_value(slope: float, intercept: float, x: float) -> float:
    return slope * x + intercept


def _count_breaches_and_touches(
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
        line_val = _line_value(slope, intercept, i)
        diff = vals[i] - line_val
        tol = abs(line_val) * touch_tolerance

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


def _find_best_boundary_line(
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
                             （对下轨：slope > converge_with_slope 表示收敛）
                             （对上轨：slope < converge_with_slope 表示收敛）

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

            slope, intercept = _line_from_two_points(x1, y1, x2, y2)

            # 上轨：极值点应大致在线的下方（上轨）或上方（下轨）
            # 允许少量极值点短暂超出边界（宽容度：2 倍容差，最多 2 个超出点)
            brief_tol = touch_tolerance * 2.5
            violations = 0
            max_violations = 2
            valid = True
            for k, (px, py) in enumerate(swing_points):
                if k == i or k == j:
                    continue
                line_at_px = _line_value(slope, intercept, px)
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

            breaches, touches = _count_breaches_and_touches(
                slope, intercept, candle_series, is_upper, touch_tolerance,
            )

            if breaches > max_breaches:
                continue
            if touches < min_touches:
                continue

            # 评分：触及越多越好，穿越越少越好
            score = touches * 10 - breaches * 50

            # 如果指定了收敛目标斜率，优先选择收敛方向的线
            if converge_with_slope is not None:
                if is_upper:
                    # 上轨需要 slope < converge_with_slope 才能收敛
                    converges = slope < converge_with_slope
                else:
                    # 下轨需要 slope > converge_with_slope 才能收敛
                    converges = slope > converge_with_slope
                if converges:
                    score += 100  # 大幅加分

            if score > best_score:
                best_score = score
                # 找线上所有贴近的极值点作为锚点
                anchors = []
                for px, py in swing_points:
                    line_at_px = _line_value(slope, intercept, px)
                    if abs(py - line_at_px) <= abs(line_at_px) * touch_tolerance:
                        anchors.append((px, py))
                best = (slope, intercept, touches, breaches, anchors)

    return best


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

        swing_all = _find_swing_points(highs, order=swing_order)
        swing_all_low = _find_swing_points(lows, order=swing_order)

        sw_highs = [(i, v) for i, v, t in swing_all if t == "high"]
        sw_lows = [(i, v) for i, v, t in swing_all_low if t == "low"]

        if len(sw_highs) < 2 or len(sw_lows) < 2:
            continue

        # 找最佳上轨（连接 swing highs，K 线不穿越上方）
        upper = _find_best_boundary_line(
            sw_highs, highs, period, is_upper=True,
            touch_tolerance=touch_tolerance,
            max_breach_ratio=max_breach_ratio,
            min_touches=min_touches,
        )
        if upper is None:
            continue

        res_slope_tmp = upper[0]

        # 找最佳下轨（优先选择与上轨收敛的线）
        lower = _find_best_boundary_line(
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
        start_gap = abs(_line_value(res_slope, res_intercept, 0)
                        - _line_value(sup_slope, sup_intercept, 0))
        end_gap = abs(_line_value(res_slope, res_intercept, period - 1)
                      - _line_value(sup_slope, sup_intercept, period - 1))

        if end_gap >= start_gap:
            continue  # 不收敛

        # 收敛比例
        convergence_ratio = 1 - end_gap / start_gap if start_gap > 0 else 0
        if convergence_ratio < 0.15:
            continue  # 收敛率太低，基本平行

        # ---- 角度检查：两线的夹角 ----
        # 归一化斜率：x 轴 1 单位 = 1 天, y 轴按价格百分比
        norm = avg_price / period if avg_price > 0 else 1
        angle_upper = math.atan(res_slope / norm)
        angle_lower = math.atan(sup_slope / norm)
        convergence_angle_deg = abs(angle_upper - angle_lower) * 180 / math.pi

        if convergence_angle_deg < min_convergence_angle_deg:
            continue  # 角度太小

        # ---- 下轨必须上倾（收敛的必要条件：下轨斜率 > 上轨斜率） ----
        sup_slope_pct = (sup_slope / avg_price) * 100 if avg_price > 0 else 0
        res_slope_pct = (res_slope / avg_price) * 100 if avg_price > 0 else 0

        if sup_slope <= res_slope:
            continue  # 下轨斜率不大于上轨斜率，无法收敛

        # ---- 当前价在通道内 ----
        last_x = period - 1
        res_val = _line_value(res_slope, res_intercept, last_x)
        sup_val = _line_value(sup_slope, sup_intercept, last_x)
        curr_close = window["close"].iloc[-1]
        margin = max(end_gap * 0.1, avg_price * 0.005)
        if curr_close < sup_val - margin or curr_close > res_val + margin:
            continue

        # ---- 判断形态类型 ----
        if abs(res_slope_pct) < 0.08 and sup_slope_pct > 0.03:
            pattern_type = "ascending_triangle"      # 上轨水平，下轨上倾
        elif res_slope_pct > 0 and sup_slope_pct > res_slope_pct:
            pattern_type = "rising_wedge"            # 两线都上倾，下轨更陡
        elif res_slope_pct < -0.03 and sup_slope_pct > 0.03:
            pattern_type = "symmetrical_triangle"    # 上轨下倾，下轨上倾
        elif res_slope_pct < -0.03 and abs(sup_slope_pct) < 0.08:
            pattern_type = "descending_wedge"        # 上轨下倾，下轨水平
        else:
            continue

        # ---- 收敛时间估算 ----
        # 两线在 x = period-1 处的间距为 end_gap
        # 两线的斜率差决定每天收敛速度
        slope_diff = res_slope - sup_slope  # 上轨斜率 - 下轨斜率（收敛时为负）
        if abs(slope_diff) > 1e-12:
            # 从当前位置追算交汇点
            # res_slope * x_conv + res_intercept = sup_slope * x_conv + sup_intercept
            x_convergence = (sup_intercept - res_intercept) / (res_slope - sup_slope)
            days_to_convergence = x_convergence - (period - 1)
        else:
            days_to_convergence = float("inf")

        # 分类：已收敛 / 即将收敛
        # 已收敛: 交汇点在窗口内或已过 (days <= 0)
        # 即将收敛: 交汇点在未来 20 个交易日内
        max_future_days = 20
        if days_to_convergence <= 0:
            convergence_status = "converged"       # 已收敛
        elif days_to_convergence <= max_future_days:
            convergence_status = "imminent"         # 即将收敛
        else:
            continue  # 收敛太远，跳过

        # ---- 评分 ----
        # 越快收敛分越高
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
                dtc = result['days_to_convergence']
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
