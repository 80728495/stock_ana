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
                                half_year_days: int = 120) -> dict | None:
    """
    Vegas 长期通道回踩策略（从上方测试）：

    条件：
    1. 股价在最近半年（~120 个交易日）内达到最高点后开始波动下行
    2. 高点到触及通道之间，股价必须大部分时间在 Vegas 通道上方运行
       （确认是"从上方回踩"而非"从下方突破"）
    3. 最近 lookback_days 个交易日内，价格触及 Vegas 通道（EMA144/EMA169 区域）
    4. 没有有效跌破 EMA169：
       - 从未跌破 EMA169，或
       - 跌破后当日或次日收盘价收回 EMA169 上方

    Args:
        df: 带有 close, low, high, ema_144, ema_169 列的 DataFrame
        lookback_days: 回看交易日数
        half_year_days: 半年的交易日数

    Returns:
        形态信息 dict 或 None
    """
    required = {"close", "low", "high", "ema_144", "ema_169"}
    if not required.issubset(df.columns) or len(df) < half_year_days:
        return None

    # ---- 条件1: 半年内到达过高点，且当前已从高点回落 ----
    half_year = df.iloc[-half_year_days:]
    peak_idx = half_year["high"].idxmax()
    peak_price = half_year.loc[peak_idx, "high"]

    # 高点不能在最近 lookback_days 内（必须已开始下行）
    recent_start = df.index[-lookback_days]
    if peak_idx >= recent_start:
        return None

    # 高点后价格确实下行：当前收盘 < 高点的 95%
    curr_close = df["close"].iloc[-1]
    if curr_close >= peak_price * 0.95:
        return None

    # ---- 条件2: 从高点到触及通道之间，股价必须在通道上方运行 ----
    # 确保这是"从上方回踩"而非"从下方靠近"
    peak_iloc = df.index.get_loc(peak_idx)
    touch_start = len(df) - lookback_days

    # 从高点到触及区域之间的 K 线
    between_section = df.iloc[peak_iloc:touch_start]
    if len(between_section) < 3:
        return None  # 高点和触及日太近，不算有效回踩

    # 高点到触及之间，至少 70% 的收盘价在通道上沿之上
    above_count = 0
    for _, row in between_section.iterrows():
        ema_upper = max(row["ema_144"], row["ema_169"])
        if row["close"] > ema_upper:
            above_count += 1

    above_ratio = above_count / len(between_section) if len(between_section) > 0 else 0
    if above_ratio < 0.70:
        return None  # 之前大部分时间不在通道上方 → 不是从上方回踩

    # ---- 条件3: 最近 lookback_days 内触及 Vegas 通道 ----
    recent = df.iloc[-lookback_days:]
    touched = False
    touch_date = None
    for idx_label, row in recent.iterrows():
        ema_upper = max(row["ema_144"], row["ema_169"])
        ema_lower = min(row["ema_144"], row["ema_169"])
        # 触及通道：最低价 <= 通道上沿（即价格下探到通道区域）
        if row["low"] <= ema_upper:
            touched = True
            touch_date = idx_label
            break

    if not touched:
        return None

    # ---- 条件4: 没有有效跌破 EMA169（通道下沿） ----
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
                    return None  # 次日也没收回 → 有效跌破
            else:
                return None

    return {
        "peak_price": float(peak_price),
        "peak_date": str(peak_idx),
        "current_price": float(curr_close),
        "above_ratio": round(above_ratio, 2),
        "touch_date": str(touch_date) if touch_date else None,
    }


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
            result = screen_vegas_channel_touch(df, lookback_days=lookback_days)
            if result is not None:
                logger.success(f"✅ {ticker} 在最近 {lookback_days} 个交易日触及 Vegas 通道回踩（从上方）")
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

        # ---- 判断形态类型（只保留看涨形态，排除下降/对称三角形） ----
        if abs(res_slope_pct) < 0.08 and sup_slope_pct > 0.03:
            pattern_type = "ascending_triangle"      # 上轨水平，下轨上倾
        elif res_slope_pct > 0 and sup_slope_pct > res_slope_pct:
            pattern_type = "rising_wedge"            # 两线都上倾，下轨更陡
        else:
            # 跳过 descending_wedge（下降三角形）、symmetrical_triangle（对称三角形）
            # 及其他非看涨模式
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


# ──────── VCP（波动率收缩形态）+ 杯柄形态策略 ────────
#
# Mark Minervini "Trade Like a Stock Market Wizard" 核心选股形态
#
# VCP 要素：
#   1) 前置上升趋势（Stage 2）
#   2) 2~7 次连续收缩（T1→T2→T3…），每次幅度递减
#   3) 成交量随收缩枯竭（最终收缩 < 50日均量的 50%）
#   4) 基底周期 30~180 个交易日
#
# 杯柄（Cup & Handle）= VCP 的一种特例
#   - 一次大幅 U 形底（杯身 12%~33%）
#   - 杯身后跟随一次小幅回撤（柄部 <15%）靠近前高
#   - 柄部成交量明显缩减


def _find_contractions(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    swing_order: int = 5,
) -> list[dict]:
    """
    在给定价格窗口中识别所有收缩段（从局部高点到后续低点）。

    返回值中每个 contraction 为 dict:
        {
            "high_idx": int,   # 局部高点位置
            "high_val": float, # 局部高点价格
            "low_idx": int,    # 后续局部低点位置
            "low_val": float,  # 局部低点价格
            "depth_pct": float # 收缩幅度 = (high - low) / high * 100
        }
    """
    n = len(highs)

    # 找局部极大值（基于 highs）
    local_max_idx = []
    for i in range(swing_order, n - swing_order):
        is_max = True
        for j in range(1, swing_order + 1):
            if highs[i - j] > highs[i] or highs[i + j] > highs[i]:
                is_max = False
                break
        if is_max:
            local_max_idx.append(i)

    # 找局部极小值（基于 lows）
    local_min_idx = []
    for i in range(swing_order, n - swing_order):
        is_min = True
        for j in range(1, swing_order + 1):
            if lows[i - j] < lows[i] or lows[i + j] < lows[i]:
                is_min = False
                break
        if is_min:
            local_min_idx.append(i)

    local_max_idx = np.array(local_max_idx)
    local_min_idx = np.array(local_min_idx)

    if len(local_max_idx) == 0 or len(local_min_idx) == 0:
        return []

    contractions: list[dict] = []

    for hi_idx in local_max_idx:
        # 后续的局部低点里，选最深的那个（在下一个高点之前）
        later_mins = local_min_idx[local_min_idx > hi_idx]
        later_maxs = local_max_idx[local_max_idx > hi_idx]

        if len(later_mins) == 0:
            continue

        # 截取到下一个高点之前的低点
        if len(later_maxs) > 0:
            boundary = later_maxs[0]
            candidate_mins = later_mins[later_mins < boundary]
        else:
            candidate_mins = later_mins

        if len(candidate_mins) == 0:
            continue

        # 找最深的低点
        deepest_idx = candidate_mins[np.argmin(lows[candidate_mins])]
        hi_val = highs[hi_idx]
        lo_val = lows[deepest_idx]
        depth_pct = (hi_val - lo_val) / hi_val * 100 if hi_val > 0 else 0

        if depth_pct > 1.0:  # 忽略 <1% 的噪声
            contractions.append({
                "high_idx": int(hi_idx),
                "high_val": float(hi_val),
                "low_idx": int(deepest_idx),
                "low_val": float(lo_val),
                "depth_pct": round(depth_pct, 2),
            })

    return contractions


def _is_u_shaped(closes: np.ndarray, start: int, end: int) -> bool:
    """
    判断 closes[start:end] 是否呈 U 形底部（而非 V 形）。

    规则：底部 1/3 区域内的收盘价标准差 < 顶部 1/3 的 50%
    （U 形底部平坦, V 形底部尖锐）
    """
    segment = closes[start:end + 1]
    n = len(segment)
    if n < 10:
        return False  # 太短无法判断

    bottom_val = np.min(segment)
    top_val = np.max(segment)
    rng = top_val - bottom_val
    if rng < 1e-6:
        return False

    # 底部 1/3 价格范围
    low_thresh = bottom_val + rng * 0.33
    bottom_mask = segment <= low_thresh
    bottom_count = np.sum(bottom_mask)

    # U 形底部：至少 20% 的 K 线在底部 1/3 区域（说明底部平坦停留时间长）
    if bottom_count < n * 0.15:
        return False

    # 额外检查：最低点不在收尾 15% 处（V 形的特征是尖底在中间瞬间触达后反弹）
    min_pos = np.argmin(segment)
    relative_pos = min_pos / n
    # U 形底：底部区域宽，所以最低点位置可以在 15%-85% 之间
    if relative_pos < 0.1 or relative_pos > 0.9:
        return False

    return True


def screen_vcp(
    df: pd.DataFrame,
    min_base_days: int = 30,
    max_base_days: int = 180,
    min_t1_depth: float = 8.0,
    max_t1_depth: float = 50.0,
    min_contractions: int = 2,
    max_contractions: int = 7,
    vol_dry_ratio: float = 0.85,
    near_pivot_pct: float = 15.0,
) -> dict | None:
    """
    检测 VCP（波动率收缩形态）和杯柄形态。

    改进的收缩递减检测：
    - 从全部收缩中寻找最佳递减子序列（≥2 个，每次深度 ≤ 前次 × 0.85）
    - 也检查后半段均值 < 前半段均值的整体趋势
    - 最终收缩深度 < 最大收缩的 60%

    Args:
        df: 带有 OHLCV 的 DataFrame
        min_base_days: 基底最短天数
        max_base_days: 基底最长天数
        min_t1_depth: T1 最小深度 %
        max_t1_depth: T1 最大深度 %
        min_contractions: 最少收缩次数（VCP 至少 2 次）
        max_contractions: 最多收缩次数
        vol_dry_ratio: 最终收缩区域成交量 / 50日均量的最大比率
        near_pivot_pct: 当前价距离前高的最大距离 %

    Returns:
        VCP 信息 dict 或 None
    """
    if len(df) < max(min_base_days + 50, 200):
        return None

    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values
    volumes = df["volume"].values.astype(float)

    if len(df) < 200:
        return None

    sma150 = pd.Series(closes).rolling(150).mean().values
    sma200 = pd.Series(closes).rolling(200).mean().values

    best_result = None
    best_score = -1

    for lookback in range(min_base_days, min(max_base_days + 1, len(df) - 50), 10):
        window_start = len(df) - lookback
        w_highs = highs[window_start:]
        w_lows = lows[window_start:]
        w_closes = closes[window_start:]
        w_volumes = volumes[window_start:]

        # 自适应 swing_order：窗口越大，需要更大的 order 过滤噪声
        swing_order = max(3, min(8, lookback // 20))

        contractions = _find_contractions(w_highs, w_lows, w_closes, swing_order)

        if len(contractions) < min_contractions:
            continue

        # ─── 3. 寻找最佳递减子序列 ───
        # 从所有收缩中提取最长的递减子序列
        # 递减条件：后续收缩 ≤ 前次 × 0.85（允许 15% 的宽容度）
        all_depths = [c["depth_pct"] for c in contractions]

        # 方法A：寻找以 T1 开始的最长递减子序列
        best_seq = None
        best_seq_score = -1

        for start_i in range(len(contractions)):
            t1_d = all_depths[start_i]
            if t1_d < min_t1_depth or t1_d > max_t1_depth:
                continue

            seq_indices = [start_i]
            prev_d = t1_d

            for j in range(start_i + 1, len(contractions)):
                d = all_depths[j]
                # 允许宽松递减：≤ 前次 × 0.85，或者绝对值足够小 (< 8%)
                if d <= prev_d * 0.85 or (d < 8.0 and d < prev_d):
                    seq_indices.append(j)
                    prev_d = d

            if len(seq_indices) >= min_contractions:
                # 序列评分：越长越好，递减比越大越好
                seq_depths = [all_depths[i] for i in seq_indices]
                ratio = seq_depths[-1] / seq_depths[0] if seq_depths[0] > 0 else 1
                s = len(seq_indices) * 10 + (1 - ratio) * 20
                if s > best_seq_score:
                    best_seq_score = s
                    best_seq = seq_indices

        # 方法B：整体趋势检查（后半段均值 < 前半段均值 × 0.7）
        if best_seq is None and len(contractions) >= 3:
            mid = len(all_depths) // 2
            first_half_avg = np.mean(all_depths[:mid])
            second_half_avg = np.mean(all_depths[mid:])

            if second_half_avg < first_half_avg * 0.70:
                # 整体递减趋势明显，取最大的作为 T1
                max_depth_idx = int(np.argmax(all_depths[:mid + 1]))
                t1_d = all_depths[max_depth_idx]
                if min_t1_depth <= t1_d <= max_t1_depth:
                    best_seq = list(range(max_depth_idx, len(contractions)))

        if best_seq is None or len(best_seq) < min_contractions:
            continue

        # 按子序列筛选
        contractions = [contractions[i] for i in best_seq]
        if len(contractions) > max_contractions:
            contractions = contractions[-max_contractions:]

        depths = [c["depth_pct"] for c in contractions]
        t1_depth = depths[0]
        final_depth = depths[-1]

        # 最终收缩应明显小于 T1（< 60% of T1，且 < 20%）
        if final_depth > t1_depth * 0.65 or final_depth > 20.0:
            continue

        # ─── 4. 成交量枯竭检查 ───
        pre_base_start = max(0, window_start - 50)
        vol_50ma = np.mean(volumes[pre_base_start:window_start]) if window_start > 0 else np.mean(volumes[:50])

        if vol_50ma < 1:
            continue

        # 最终收缩区域的量能检测
        # 对大盘股：取最近 15 天内滚动 5 日均量的最小值，与基底前 50 日均量比较
        # 这捕捉到了 VCP 紧缩处的短暂量能枯竭（大盘股通常不会持续缩量）
        last_c = contractions[-1]
        tail_days = min(20, len(w_volumes))
        tail_vols = w_volumes[-tail_days:]
        if len(tail_vols) >= 5:
            rolling_5d = pd.Series(tail_vols).rolling(5).mean().dropna().values
            min_5d_vol = float(np.min(rolling_5d)) if len(rolling_5d) > 0 else np.mean(tail_vols)
        else:
            min_5d_vol = np.mean(tail_vols)

        vol_ratio = min_5d_vol / vol_50ma if vol_50ma > 0 else 1.0

        if vol_ratio > vol_dry_ratio:
            continue

        # ─── 5. 前置上升趋势检查 ───
        if window_start < 200:
            continue

        curr_price = closes[-1]
        curr_sma150 = sma150[-1]
        curr_sma200 = sma200[-1]

        if np.isnan(curr_sma150) or np.isnan(curr_sma200):
            continue
        if curr_price < curr_sma200:
            continue
        if curr_sma150 < curr_sma200 * 0.97:
            continue

        # ─── 6. 当前价靠近枢轴点 ───
        pivot_price = float(np.max(w_highs))
        distance_to_pivot_pct = (pivot_price - curr_price) / pivot_price * 100

        if distance_to_pivot_pct > near_pivot_pct:
            continue

        # ─── 7. 杯柄形态特例判断 ───
        is_cup_handle = False
        cup_info = None

        if len(contractions) == 2:
            main_c = contractions[0]
            handle_c = contractions[1]
            cup_depth = main_c["depth_pct"]
            handle_depth = handle_c["depth_pct"]

            if (10.0 <= cup_depth <= 40.0
                    and handle_depth < 15.0
                    and _is_u_shaped(w_closes, main_c["high_idx"], main_c["low_idx"])
                    and handle_c["high_val"] >= main_c["high_val"] * 0.90):
                is_cup_handle = True
                cup_info = {
                    "cup_depth_pct": cup_depth,
                    "cup_start_idx": main_c["high_idx"],
                    "cup_bottom_idx": main_c["low_idx"],
                    "handle_depth_pct": handle_depth,
                    "handle_start_idx": handle_c["high_idx"],
                    "handle_bottom_idx": handle_c["low_idx"],
                }

        # ─── 8. 评分 ───
        # 收缩次数越多越好（更多收缩→更紧实的VCP）
        contraction_score = len(contractions) * 10
        # 收缩递减比率越好（每次都明显缩小 → 得分更高）
        ratio_scores = []
        for k in range(1, len(depths)):
            r = depths[k] / depths[k - 1] if depths[k - 1] > 0 else 1
            ratio_scores.append(max(0, (1 - r)) * 20)
        ratio_score = sum(ratio_scores)
        # 成交量越枯竭越好
        vol_score = max(0, (1 - vol_ratio)) * 30
        # 距离枢轴点越近越好
        pivot_score = max(0, (near_pivot_pct - distance_to_pivot_pct)) * 2
        # 杯柄加分
        cup_bonus = 15 if is_cup_handle else 0

        total_score = contraction_score + ratio_score + vol_score + pivot_score + cup_bonus

        if total_score > best_score:
            best_score = total_score
            pattern_type = "cup_and_handle" if is_cup_handle else "vcp"

            best_result = {
                "pattern": pattern_type,
                "base_days": lookback,
                "window_start": window_start,
                "num_contractions": len(contractions),
                "contractions": contractions,
                "depths": depths,
                "t1_depth_pct": t1_depth,
                "final_depth_pct": final_depth,
                "vol_ratio": round(vol_ratio, 3),
                "pivot_price": round(pivot_price, 2),
                "current_price": round(curr_price, 2),
                "distance_to_pivot_pct": round(distance_to_pivot_pct, 2),
                "score": round(total_score, 1),
                "cup_handle_info": cup_info,
            }

    return best_result


def scan_ndx100_vcp(
    min_base_days: int = 30,
    max_base_days: int = 180,
) -> list[dict]:
    """
    扫描纳指100中呈现 VCP（波动率收缩形态）或杯柄形态的股票。

    Args:
        min_base_days: 基底最短天数
        max_base_days: 基底最长天数

    Returns:
        [{"ticker": str, "df": DataFrame, "vcp_info": dict}, ...]
    """
    stock_data = load_all_ndx100_data()

    if not stock_data:
        logger.error("本地无数据！请先运行 update_ndx100_data() 下载数据")
        return []

    hits: list[dict] = []
    processed = 0

    _PATTERN_CN = {
        "vcp": "VCP（波动率收缩）",
        "cup_and_handle": "杯柄形态",
    }

    for ticker, df in stock_data.items():
        try:
            if len(df) < 200:
                logger.debug(f"{ticker}: 数据不足（{len(df)} 行），需至少 200 行检测 VCP")
                continue

            processed += 1
            result = screen_vcp(
                df, min_base_days=min_base_days, max_base_days=max_base_days,
            )
            if result is not None:
                ptype = _PATTERN_CN.get(result["pattern"], result["pattern"])
                depths_str = "→".join(f"{d:.0f}%" for d in result["depths"])
                logger.success(
                    f"✅ {ticker} 呈现{ptype}"
                    f"（基底 {result['base_days']} 日，"
                    f"{result['num_contractions']} 次收缩 [{depths_str}]，"
                    f"量缩比 {result['vol_ratio']:.0%}，"
                    f"距前高 {result['distance_to_pivot_pct']:.1f}%）"
                )
                hits.append({"ticker": ticker, "df": df, "vcp_info": result})
        except Exception as e:
            logger.error(f"{ticker}: VCP 检测失败 - {e}")
            continue

    logger.info(
        f"VCP 扫描完成：本地共 {len(stock_data)} 只股票，"
        f"有效处理 {processed} 只，{len(hits)} 只呈现形态"
    )
    return hits
