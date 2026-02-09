"""
股票筛选模块 - 基于技术指标的基础筛选策略 + Vegas 通道策略

VCP 和三角形策略已拆分至独立模块，此处保留向后兼容的 re-export。
"""

import pandas as pd
from loguru import logger

from stock_ana.data_fetcher import load_all_ndx100_data
from stock_ana.indicators import add_macd, add_vegas_channel

# ──────── 向后兼容 re-export ────────
from stock_ana.strategy_base import check_trend_template  # noqa: F401
from stock_ana.strategy_vcp import (  # noqa: F401
    screen_vcp,
    scan_ndx100_vcp,
)
from stock_ana.strategy_triangle import (  # noqa: F401
    screen_ascending_triangle,
    scan_ndx100_ascending_triangle,
)


# ──────── 基础策略 ────────


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
    3. 金叉后若又发生死叉，则该金叉无效

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

    n = lookback_days + 1
    start_idx = len(df) - n

    last_cross_is_golden = None

    for i in range(max(start_idx + 1, 1), len(df)):
        prev_hist = hist_all.iloc[i - 1]
        curr_hist = hist_all.iloc[i]
        curr_close = close_all.iloc[i]
        threshold = curr_close * 0.001

        # 金叉：负 → 正
        if prev_hist < 0 and curr_hist > 0:
            if curr_hist <= threshold:
                continue

            min_neg_hist = prev_hist
            for j in range(i - 2, -1, -1):
                h = hist_all.iloc[j]
                if h >= 0:
                    break
                if h < min_neg_hist:
                    min_neg_hist = h

            if abs(min_neg_hist) < threshold:
                continue

            last_cross_is_golden = True

        # 死叉：正 → 负
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
    扫描纳指100中最近 lookback_days 个交易日内发生 MACD 金叉的股票。

    Args:
        lookback_days: 回看交易日数（默认 5 天，约 1 周）

    Returns:
        包含金叉股票信息的列表
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
    1. 股价在最近半年内达到最高点后开始波动下行
    2. 高点到触及通道之间，股价必须大部分时间在 Vegas 通道上方运行
       （确认是"从上方回踩"而非"从下方突破"）
    3. 最近 lookback_days 个交易日内，价格触及 Vegas 通道（EMA144/EMA169 区域）
    4. 没有有效跌破 EMA169：
       - 从未跌破 EMA169，或
       - 跌破后当日或次日收盘价收回 EMA169 上方
    5. Vegas 通道必须保持上升趋势（EMA 中轨 90 天内上升）
    6. 最近 90 个交易日收盘价必须全部在通道上沿之上，不能多次横穿
       （多次穿越视为横盘无效）
    7. 90 天内最多只有 2 次有效触碰通道，第 3 次视为弱势无效

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

    # ---- 条件5: Vegas 通道保持上升趋势 ----
    # 比较 90 天前和当前的 EMA 中轨值，要求上升
    trend_window = min(90, len(df) - 1)
    ema_mid_now = (df["ema_144"].iloc[-1] + df["ema_169"].iloc[-1]) / 2
    ema_mid_ago = (df["ema_144"].iloc[-trend_window] + df["ema_169"].iloc[-trend_window]) / 2
    if ema_mid_now <= ema_mid_ago:
        return None  # 通道平或向下，不是上升趋势

    # ---- 条件6: 最近 90 天收盘价不能多次横穿通道 ----
    # 统计从通道上方跌到下方的"下穿"次数，超过 1 次视为横盘
    lookback_90 = min(90, len(df))
    section_90 = df.iloc[-lookback_90:]
    cross_below_count = 0
    was_above = True  # 假设初始在上方

    for _, row in section_90.iterrows():
        ema_upper = max(row["ema_144"], row["ema_169"])
        is_above = row["close"] > ema_upper
        if was_above and not is_above:
            cross_below_count += 1
        was_above = is_above

    if cross_below_count > 1:
        return None  # 多次下穿通道 → 横盘，无效

    # ---- 条件7: 90 天内最多 2 次触碰通道 ----
    # 扫描 90 天内的触碰事件（low <= ema_upper），合并连续触碰为一次
    touch_events = []
    in_touch = False
    for i in range(len(section_90)):
        row = section_90.iloc[i]
        ema_upper = max(row["ema_144"], row["ema_169"])
        if row["low"] <= ema_upper:
            if not in_touch:
                touch_events.append(i)
                in_touch = True
        else:
            # 需要离开通道区域至少 3 天才算结束一次触碰
            if in_touch:
                # 检查后续是否有 3 天连续在通道上方
                still_away = True
                for j in range(1, 4):
                    if i + j < len(section_90):
                        r2 = section_90.iloc[i + j]
                        eu2 = max(r2["ema_144"], r2["ema_169"])
                        if r2["low"] <= eu2:
                            still_away = False
                            break
                if still_away:
                    in_touch = False

    if len(touch_events) >= 3:
        return None  # 第 3 次触碰，弱势无效

    # ---- 条件1: 半年内到达过高点，且当前已从高点回落 ----
    half_year = df.iloc[-half_year_days:]
    peak_idx = half_year["high"].idxmax()
    peak_price = half_year.loc[peak_idx, "high"]

    recent_start = df.index[-lookback_days]
    if peak_idx >= recent_start:
        return None

    curr_close = df["close"].iloc[-1]
    if curr_close >= peak_price * 0.95:
        return None

    peak_iloc = df.index.get_loc(peak_idx)
    touch_start = len(df) - lookback_days

    between_section = df.iloc[peak_iloc:touch_start]
    if len(between_section) < 3:
        return None

    above_count = 0
    for _, row in between_section.iterrows():
        ema_upper = max(row["ema_144"], row["ema_169"])
        if row["close"] > ema_upper:
            above_count += 1

    above_ratio = above_count / len(between_section) if len(between_section) > 0 else 0
    if above_ratio < 0.70:
        return None

    recent = df.iloc[-lookback_days:]
    touched = False
    touch_date = None
    for idx_label, row in recent.iterrows():
        ema_upper = max(row["ema_144"], row["ema_169"])
        if row["low"] <= ema_upper:
            touched = True
            touch_date = idx_label
            break

    if not touched:
        return None

    for i in range(len(df) - lookback_days, len(df)):
        row = df.iloc[i]
        ema_lower = min(row["ema_144"], row["ema_169"])
        close_i = row["close"]

        if close_i < ema_lower:
            if i + 1 < len(df):
                next_row = df.iloc[i + 1]
                next_ema_lower = min(next_row["ema_144"], next_row["ema_169"])
                if next_row["close"] < next_ema_lower:
                    return None
            else:
                return None

    return {
        "peak_price": float(peak_price),
        "peak_date": str(peak_idx),
        "current_price": float(curr_close),
        "above_ratio": round(above_ratio, 2),
        "touch_date": str(touch_date) if touch_date else None,
        "channel_trend_pct": round((ema_mid_now / ema_mid_ago - 1) * 100, 2),
        "cross_below_count": cross_below_count,
        "touch_events_90d": len(touch_events),
    }


def scan_ndx100_vegas_touch(lookback_days: int = 5) -> list[dict]:
    """
    扫描纳指100中满足 Vegas 通道回踩条件的股票。

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
    min_period: int = 30,
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

        # ---- 触点间距检查 ----
        # 上下边各自至少 2 个触点，且相邻触点间距 ≥ period/4
        min_spacing = period / 4

        def _has_valid_spacing(anchors: list[tuple[int, float]], min_gap: float) -> bool:
            """检查锚点中是否存在至少 2 个间距 ≥ min_gap 的触点。"""
            if len(anchors) < 2:
                return False
            xs = sorted(a[0] for a in anchors)
            # 检查是否有任意两个相邻锚点间距达标
            for i in range(1, len(xs)):
                if xs[i] - xs[i - 1] >= min_gap:
                    return True
            return False

        if not _has_valid_spacing(res_anchors, min_spacing):
            continue
        if not _has_valid_spacing(sup_anchors, min_spacing):
            continue

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
        # 越长周期越好（同时存在长短两个趋势，选长不选短）
        period_bonus = period * 0.5  # 每多 1 天 +0.5 分
        urgency_bonus = max(0, (max_future_days - days_to_convergence)) * 0.5
        score = (
            (res_touches + sup_touches) * 5
            - (res_breaches + sup_breaches) * 20
            + convergence_angle_deg * 2
            + convergence_ratio * 10
            + urgency_bonus
            + period_bonus
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
    min_period: int = 30,
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


# ──────── VCP（波动率收缩形态）策略 ────────
#
# Mark Minervini "Trade Like a Stock Market Wizard" 严格 VCP 检测
#
# 核心逻辑：
#   1) 在 8~20 周（40~100 交易日）窗口内找到 Base 高点（Base_Start）
#   2) 当前价在 Base_Start 的 10% 以内
#   3) 在 Base 内识别 ≥2 次递减收缩（depth(T1) > depth(T2) > ...）
#   4) 最终收缩区成交量枯竭（存在单日量 < 50日均量 × 50%）
#   5) 前置上升趋势（价格 > SMA200）
#   6) 绿圈 = T1 高点（最大收缩起点），红圈 = 最终收缩高点（突破枢轴）


def _find_swing_highs_lows(
    highs: np.ndarray,
    lows: np.ndarray,
    order: int = 5,
) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    """
    寻找局部极大值和极小值。

    Args:
        highs: high 价格数组
        lows: low 价格数组
        order: 极值点左右各需要 order 根K线确认

    Returns:
        (swing_highs, swing_lows) 各为 [(index, value), ...]
    """
    n = len(highs)
    swing_highs = []
    swing_lows = []

    for i in range(order, n - order):
        # 检查高点
        is_high = True
        for j in range(1, order + 1):
            if highs[i - j] >= highs[i] or highs[i + j] >= highs[i]:
                is_high = False
                break
        if is_high:
            swing_highs.append((i, float(highs[i])))

        # 检查低点
        is_low = True
        for j in range(1, order + 1):
            if lows[i - j] <= lows[i] or lows[i + j] <= lows[i]:
                is_low = False
                break
        if is_low:
            swing_lows.append((i, float(lows[i])))

    return swing_highs, swing_lows


def _find_contractions_strict(
    swing_highs: list[tuple[int, float]],
    swing_lows: list[tuple[int, float]],
    highs: np.ndarray,
    lows: np.ndarray,
) -> list[dict]:
    """
    识别严格的收缩段：每个收缩 = 从一个 swing high 到其后续最深 swing low。

    要求 swing low 出现在当前 swing high 之后、下一个 swing high 之前。
    然后合并相邻的高低点，形成 peak→trough 配对。

    Returns:
        [{"high_idx": int, "high_val": float,
          "low_idx": int, "low_val": float,
          "depth_pct": float}, ...]
    """
    if not swing_highs or not swing_lows:
        return []

    contractions = []

    for i, (hi_idx, hi_val) in enumerate(swing_highs):
        # 找下一个 swing high 的位置作为边界
        next_hi_idx = swing_highs[i + 1][0] if i + 1 < len(swing_highs) else len(highs)

        # 在 hi_idx 和 next_hi_idx 之间找最深的 swing low
        candidate_lows = [
            (lo_idx, lo_val) for lo_idx, lo_val in swing_lows
            if hi_idx < lo_idx < next_hi_idx
        ]

        if not candidate_lows:
            # 没有 swing low，直接用区间最低点
            lo_range = lows[hi_idx + 1:next_hi_idx]
            if len(lo_range) == 0:
                continue
            rel_lo_idx = int(np.argmin(lo_range))
            lo_idx = hi_idx + 1 + rel_lo_idx
            lo_val = float(lows[lo_idx])
        else:
            # 取最深的 swing low
            lo_idx, lo_val = min(candidate_lows, key=lambda x: x[1])

        depth_pct = (hi_val - lo_val) / hi_val * 100 if hi_val > 0 else 0

        if depth_pct > 1.5:  # 忽略 <1.5% 噪声
            contractions.append({
                "high_idx": int(hi_idx),
                "high_val": float(hi_val),
                "low_idx": int(lo_idx),
                "low_val": float(lo_val),
                "depth_pct": round(depth_pct, 2),
            })

    return contractions


def _find_best_decreasing_sequence(
    contractions: list[dict],
    min_count: int = 2,
) -> list[dict] | None:
    """
    从收缩列表中找最佳的严格递减子序列。

    递减条件：depth(T_{k+1}) < depth(T_k)（严格小于）。
    优先选最长序列；长度相同时，选递减比最大的。

    Returns:
        递减子序列的收缩列表，或 None
    """
    if len(contractions) < min_count:
        return None

    best_seq = None
    best_len = 0
    best_ratio = 0.0

    for start_i in range(len(contractions)):
        seq = [contractions[start_i]]
        for j in range(start_i + 1, len(contractions)):
            if contractions[j]["depth_pct"] < seq[-1]["depth_pct"]:
                seq.append(contractions[j])

        if len(seq) >= min_count:
            ratio = 1.0 - seq[-1]["depth_pct"] / seq[0]["depth_pct"]
            if len(seq) > best_len or (len(seq) == best_len and ratio > best_ratio):
                best_len = len(seq)
                best_ratio = ratio
                best_seq = list(seq)

    return best_seq


def screen_vcp(
    df: pd.DataFrame,
    min_base_weeks: int = 8,
    max_base_weeks: int = 20,
    max_distance_to_base_pct: float = 10.0,
    min_t1_depth: float = 8.0,
    max_t1_depth: float = 50.0,
    min_contractions: int = 2,
    vol_dry_factor: float = 0.50,
    min_green_red_gap_days: int = 15,
) -> dict | None:
    """
    严格检测 Mark Minervini VCP（波动率收缩形态）。

    算法流程：
    1. 在 8~20 周窗口内确定 Base 结构（Base_Start = 窗口最高点）
    2. 当前价必须在 Base_Start 的 max_distance_to_base_pct 以内
    3. 在 Base 内找 swing highs/lows，配对为收缩段
    4. 找严格递减子序列（≥2 次, depth(T1) > depth(T2) > …）
    5. 最终收缩区成交量枯竭（存在 ≥1 天 vol < 50日MA × vol_dry_factor）
    6. 前置上升趋势（price > SMA200）
    7. 绿圈 = T1 高点，红圈 = 最终收缩高点

    Args:
        df: 带有 OHLCV 的 DataFrame
        min_base_weeks / max_base_weeks: Base 窗口范围（周）
        max_distance_to_base_pct: 当前价距离 Base 高点的最大距离 %
        min_t1_depth: T1 最小深度 %
        max_t1_depth: T1 最大深度 %
        min_contractions: 最少递减收缩次数
        vol_dry_factor: 量能枯竭阈值（相对 50日均量的倍数）
        min_green_red_gap_days: 绿圈到红圈的最小间距（交易日）

    Returns:
        VCP 信息 dict 或 None
    """
    min_base_days = min_base_weeks * 5
    max_base_days = max_base_weeks * 5

    if len(df) < max(max_base_days + 50, 200):
        return None

    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values
    volumes = df["volume"].values.astype(float)

    sma200 = pd.Series(closes).rolling(200).mean().values

    best_result = None
    best_score = -1

    for lookback in range(min_base_days, min(max_base_days + 1, len(df) - 50), 5):
        window_start = len(df) - lookback
        w_highs = highs[window_start:]
        w_lows = lows[window_start:]
        w_closes = closes[window_start:]
        w_volumes = volumes[window_start:]
        n = len(w_highs)

        # ─── 1. Base 结构：找窗口最高点 ───
        base_high_rel = int(np.argmax(w_highs))
        base_high_val = float(w_highs[base_high_rel])

        # 当前价必须在 Base 高点的 max_distance_to_base_pct 以内
        curr_price = float(w_closes[-1])
        distance_pct = (base_high_val - curr_price) / base_high_val * 100
        if distance_pct > max_distance_to_base_pct or distance_pct < -2.0:
            continue  # 太远或已大幅突破

        # ─── 2. 找 swing highs/lows ───
        # 自适应 swing_order：窗口越大用更大的 order 过滤噪声
        swing_order = max(3, min(7, lookback // 15))

        sw_highs, sw_lows = _find_swing_highs_lows(w_highs, w_lows, order=swing_order)

        if len(sw_highs) < 2 or len(sw_lows) < 1:
            continue

        # ─── 3. 配对收缩段 ───
        contractions = _find_contractions_strict(sw_highs, sw_lows, w_highs, w_lows)

        if len(contractions) < min_contractions:
            continue

        # ─── 4. 找严格递减子序列 ───
        dec_seq = _find_best_decreasing_sequence(contractions, min_count=min_contractions)

        if dec_seq is None:
            continue

        t1_depth = dec_seq[0]["depth_pct"]
        final_depth = dec_seq[-1]["depth_pct"]

        # T1 深度约束
        if t1_depth < min_t1_depth or t1_depth > max_t1_depth:
            continue

        # 最终收缩必须明显小于 T1
        if final_depth >= t1_depth * 0.80:
            continue

        # ─── 5. 绿圈到红圈间距检查 ───
        green_idx = dec_seq[0]["high_idx"]      # T1 高点 (Base_Start)
        red_idx = dec_seq[-1]["high_idx"]        # 最终收缩高点 (Pivot)
        gap_days = red_idx - green_idx
        if gap_days < min_green_red_gap_days:
            continue  # 间距太短，可能是短期 pennant

        # ─── 6. 成交量枯竭检查 ───
        # 计算 base 之前 50 天的均量作为基准
        pre_base_start = max(0, window_start - 50)
        vol_50ma = float(np.mean(volumes[pre_base_start:window_start])) if window_start > 50 else float(np.mean(volumes[:50]))

        if vol_50ma < 1:
            continue

        # 最终收缩区（从最后一个收缩的高点到窗口末尾）内
        # 必须存在至少 1 天 volume < 50日均量 × vol_dry_factor
        final_start = dec_seq[-1]["high_idx"]
        tail_vols = w_volumes[final_start:]

        if len(tail_vols) < 3:
            continue

        has_dry_up = bool(np.any(tail_vols < vol_50ma * vol_dry_factor))
        if not has_dry_up:
            continue

        # 计算最终收缩区的均量比值（用于评分）
        vol_ratio = float(np.mean(tail_vols)) / vol_50ma if vol_50ma > 0 else 1.0

        # ─── 7. 前置上升趋势检查 ───
        curr_sma200 = sma200[-1]
        if np.isnan(curr_sma200):
            continue
        if curr_price < curr_sma200:
            continue  # 价格必须在 200 日均线之上

        # SMA200 本身应该上升（最近 20 天）
        sma200_recent = sma200[-20:]
        if np.any(np.isnan(sma200_recent)):
            continue
        if sma200_recent[-1] < sma200_recent[0]:
            continue  # SMA200 下降，不是 Stage 2

        # ─── 8. Pivot 价格 ───
        # 红圈处的高点价 = 突破枢轴价
        pivot_price = float(dec_seq[-1]["high_val"])
        distance_to_pivot_pct = (pivot_price - curr_price) / pivot_price * 100

        # 枢轴价距离基底最高点不能太远（确认是同一个 base）
        if pivot_price < base_high_val * 0.85:
            continue

        # ─── 9. 评分 ───
        # 收缩次数越多越好
        contraction_score = len(dec_seq) * 10
        # 递减比越大越好（T1→Tn 收缩幅度减少越多越好）
        ratio = final_depth / t1_depth if t1_depth > 0 else 1.0
        ratio_score = max(0, (1.0 - ratio)) * 30
        # 量能越枯竭越好
        vol_score = max(0, (1.0 - vol_ratio)) * 20
        # 距离枢轴越近越好（负值表示已突破）
        proximity_score = max(0, (10.0 - abs(distance_to_pivot_pct))) * 3
        # base 时间越长越稳固
        base_score = lookback * 0.1

        total_score = contraction_score + ratio_score + vol_score + proximity_score + base_score

        if total_score > best_score:
            best_score = total_score

            # 将窗口内相对索引转为 df 绝对索引
            abs_contractions = []
            for c in dec_seq:
                abs_contractions.append({
                    "high_idx": c["high_idx"],  # 相对窗口的索引
                    "high_val": c["high_val"],
                    "low_idx": c["low_idx"],
                    "low_val": c["low_val"],
                    "depth_pct": c["depth_pct"],
                })

            best_result = {
                "pattern": "vcp",
                "base_days": lookback,
                "window_start": window_start,
                "num_contractions": len(dec_seq),
                "contractions": abs_contractions,
                "depths": [c["depth_pct"] for c in dec_seq],
                "t1_depth_pct": t1_depth,
                "final_depth_pct": final_depth,
                "vol_ratio": round(vol_ratio, 3),
                "pivot_price": round(pivot_price, 2),
                "current_price": round(curr_price, 2),
                "distance_to_pivot_pct": round(distance_to_pivot_pct, 2),
                "base_high_val": round(base_high_val, 2),
                "green_idx_rel": green_idx,    # T1 高点，窗口内相对索引
                "red_idx_rel": red_idx,        # 最终收缩高点，窗口内相对索引
                "gap_days": gap_days,
                "score": round(total_score, 1),
            }

    return best_result


def scan_ndx100_vcp(
    min_base_weeks: int = 8,
    max_base_weeks: int = 20,
) -> list[dict]:
    """
    扫描纳指100中呈现 VCP（波动率收缩形态）的股票。

    Args:
        min_base_weeks: Base 最短周数
        max_base_weeks: Base 最长周数

    Returns:
        [{"ticker": str, "df": DataFrame, "vcp_info": dict}, ...]
    """
    stock_data = load_all_ndx100_data()

    if not stock_data:
        logger.error("本地无数据！请先运行 update_ndx100_data() 下载数据")
        return []

    hits: list[dict] = []
    processed = 0

    for ticker, df in stock_data.items():
        try:
            if len(df) < 200:
                logger.debug(f"{ticker}: 数据不足（{len(df)} 行），需至少 200 行检测 VCP")
                continue

            processed += 1
            result = screen_vcp(
                df, min_base_weeks=min_base_weeks, max_base_weeks=max_base_weeks,
            )
            if result is not None:
                depths_str = "→".join(f"{d:.0f}%" for d in result["depths"])
                logger.success(
                    f"✅ {ticker} 呈现 VCP"
                    f"（基底 {result['base_days']} 日，"
                    f"{result['num_contractions']} 次收缩 [{depths_str}]，"
                    f"量缩比 {result['vol_ratio']:.0%}，"
                    f"距枢轴 {result['distance_to_pivot_pct']:.1f}%，"
                    f"绿→红间距 {result['gap_days']} 日）"
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
