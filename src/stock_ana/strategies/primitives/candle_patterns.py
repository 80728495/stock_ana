"""
蜡烛图形态识别模块（纯 pandas/numpy 实现，无 TA-Lib 依赖）

支持的形态（均返回 pandas Series，值含义：1=看涨, -1=看跌, 0=无形态）：

单根K线：
  - cdl_doji              十字星
  - cdl_hammer            锤子线 / 倒锤子线
  - cdl_spinning_top      纺锤线
  - cdl_marubozu          光头光脚线（实体极大）

双根K线：
  - cdl_engulfing         吞没形态
  - cdl_harami            孕线形态
  - cdl_piercing          穿线/乌云盖顶

三根K线：
  - cdl_morning_star      早晨之星
  - cdl_evening_star      黄昏之星
  - cdl_three_white_soldiers  三白兵
  - cdl_three_black_crows     三乌鸦

批量扫描：
  - scan_candle_patterns  扫描所有形态，返回 DataFrame
"""

from __future__ import annotations

import pandas as pd
import numpy as np


# ─────────────────────────────────────────────────────────────────
#  内部工具函数
# ─────────────────────────────────────────────────────────────────

def _body(o: pd.Series, c: pd.Series) -> pd.Series:
    """实体大小（绝对值）"""
    return (c - o).abs()


def _range(h: pd.Series, l: pd.Series) -> pd.Series:
    """K线总长（高-低），避免除零"""
    r = h - l
    return r.replace(0, np.nan)


def _upper_shadow(o: pd.Series, h: pd.Series, c: pd.Series) -> pd.Series:
    """上影线长度"""
    return h - pd.concat([o, c], axis=1).max(axis=1)


def _lower_shadow(o: pd.Series, l: pd.Series, c: pd.Series) -> pd.Series:
    """下影线长度"""
    return pd.concat([o, c], axis=1).min(axis=1) - l


def _body_ratio(o: pd.Series, h: pd.Series, l: pd.Series, c: pd.Series) -> pd.Series:
    """实体占总振幅比例"""
    return _body(o, c) / _range(h, l).fillna(1e-8)


def _is_bullish(o: pd.Series, c: pd.Series) -> pd.Series:
    return c > o


def _is_bearish(o: pd.Series, c: pd.Series) -> pd.Series:
    return c < o


# ─────────────────────────────────────────────────────────────────
#  单根K线形态
# ─────────────────────────────────────────────────────────────────

def cdl_doji(
    df: pd.DataFrame,
    body_ratio_threshold: float = 0.05,
) -> pd.Series:
    """
    十字星：实体极小（实体/总振幅 < threshold）。

    返回：1（看涨语境中出现）/ -1（看跌语境中出现）/ 0
    注意：十字星本身无方向，方向由前一根K线的涨跌决定。
    """
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    is_doji = _body_ratio(o, h, l, c) < body_ratio_threshold

    prev_bull = c.shift(1) > o.shift(1)

    result = pd.Series(0, index=df.index, name="cdl_doji")
    result[is_doji & prev_bull] = -1   # 上涨后十字星，潜在看跌
    result[is_doji & ~prev_bull] = 1   # 下跌后十字星，潜在看涨
    return result


def cdl_hammer(
    df: pd.DataFrame,
    body_ratio_max: float = 0.35,
    shadow_ratio_min: float = 2.0,
) -> pd.Series:
    """
    锤子线（下跌后出现 → 看涨）/ 倒锤子线（上涨后出现 → 看跌）。

    条件：
      - 实体占总振幅 < body_ratio_max
      - 下影线 >= body * shadow_ratio_min（锤子）
        或上影线 >= body * shadow_ratio_min（倒锤子）
    """
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    body = _body(o, c).replace(0, 1e-8)
    upper = _upper_shadow(o, h, c)
    lower = _lower_shadow(o, l, c)
    br = _body_ratio(o, h, l, c)

    # 锤子线：下影线长，上影线短
    is_hammer = (
        (br < body_ratio_max)
        & (lower >= body * shadow_ratio_min)
        & (upper < body * 0.5)
    )
    # 倒锤子线：上影线长，下影线短
    is_inv_hammer = (
        (br < body_ratio_max)
        & (upper >= body * shadow_ratio_min)
        & (lower < body * 0.5)
    )

    prev_bearish = c.shift(1) < o.shift(1)
    prev_bullish = c.shift(1) > o.shift(1)

    result = pd.Series(0, index=df.index, name="cdl_hammer")
    result[is_hammer & prev_bearish] = 1    # 下跌趋势中的锤子线 → 看涨
    result[is_inv_hammer & prev_bullish] = -1  # 上涨趋势中的倒锤子 → 看跌
    return result


def cdl_spinning_top(
    df: pd.DataFrame,
    body_ratio_min: float = 0.05,
    body_ratio_max: float = 0.35,
) -> pd.Series:
    """
    纺锤线：实体较小但有上下影线，表示多空犹豫。

    返回：1=下跌后（潜在反转看涨）/ -1=上涨后（潜在反转看跌）/ 0
    """
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    br = _body_ratio(o, h, l, c)
    is_spinning = (br >= body_ratio_min) & (br <= body_ratio_max)

    prev_bull = c.shift(1) > o.shift(1)

    result = pd.Series(0, index=df.index, name="cdl_spinning_top")
    result[is_spinning & prev_bull] = -1
    result[is_spinning & ~prev_bull] = 1
    return result


def cdl_marubozu(
    df: pd.DataFrame,
    body_ratio_min: float = 0.92,
) -> pd.Series:
    """
    光头光脚线：实体占总振幅极大，几乎无影线，代表强势单边行情。

    返回：1=阳线光头光脚（强势看涨）/ -1=阴线光头光脚（强势看跌）/ 0
    """
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    br = _body_ratio(o, h, l, c)
    bullish = _is_bullish(o, c)
    bearish = _is_bearish(o, c)

    result = pd.Series(0, index=df.index, name="cdl_marubozu")
    result[(br >= body_ratio_min) & bullish] = 1
    result[(br >= body_ratio_min) & bearish] = -1
    return result


# ─────────────────────────────────────────────────────────────────
#  双根K线形态
# ─────────────────────────────────────────────────────────────────

def cdl_engulfing(df: pd.DataFrame) -> pd.Series:
    """
    吞没形态：
      - 看涨吞没：前根阴线，当根阳线实体完全覆盖前根实体
      - 看跌吞没：前根阳线，当根阴线实体完全覆盖前根实体

    返回：1=看涨吞没 / -1=看跌吞没 / 0
    """
    o, c = df["open"], df["close"]
    o1, c1 = o.shift(1), c.shift(1)

    # 看涨吞没：前根阴 + 当根阳 + 当根 open < 前根 close + 当根 close > 前根 open
    bull_engulf = (
        (c1 < o1)           # 前根阴线
        & (c > o)           # 当根阳线
        & (o <= c1)         # 当根开盘 <= 前根收盘（覆盖下方）
        & (c >= o1)         # 当根收盘 >= 前根开盘（覆盖上方）
    )
    # 看跌吞没：前根阳 + 当根阴 + 当根 open > 前根 close + 当根 close < 前根 open
    bear_engulf = (
        (c1 > o1)           # 前根阳线
        & (c < o)           # 当根阴线
        & (o >= c1)         # 当根开盘 >= 前根收盘
        & (c <= o1)         # 当根收盘 <= 前根开盘
    )

    result = pd.Series(0, index=df.index, name="cdl_engulfing")
    result[bull_engulf] = 1
    result[bear_engulf] = -1
    return result


def cdl_harami(df: pd.DataFrame) -> pd.Series:
    """
    孕线形态（吞没的反面）：当根K线实体完全在前根K线实体内部。

      - 看涨孕线：前根阴线包住当根较小阳线
      - 看跌孕线：前根阳线包住当根较小阴线

    返回：1=看涨孕线 / -1=看跌孕线 / 0
    """
    o, c = df["open"], df["close"]
    o1, c1 = o.shift(1), c.shift(1)

    body_high = pd.concat([o, c], axis=1).max(axis=1)
    body_low = pd.concat([o, c], axis=1).min(axis=1)
    body_high1 = pd.concat([o1, c1], axis=1).max(axis=1)
    body_low1 = pd.concat([o1, c1], axis=1).min(axis=1)

    is_inside = (body_high <= body_high1) & (body_low >= body_low1)

    bull_harami = (c1 < o1) & (c > o) & is_inside   # 前阴包当阳
    bear_harami = (c1 > o1) & (c < o) & is_inside   # 前阳包当阴

    result = pd.Series(0, index=df.index, name="cdl_harami")
    result[bull_harami] = 1
    result[bear_harami] = -1
    return result


def cdl_piercing(df: pd.DataFrame) -> pd.Series:
    """
    穿线 / 乌云盖顶：

      - 穿线（看涨）：前根阴线，当根阳线开盘低于前根最低，收盘穿过前根实体中点之上
      - 乌云盖顶（看跌）：前根阳线，当根阴线开盘高于前根最高，收盘跌入前根实体中点之下

    返回：1=穿线 / -1=乌云盖顶 / 0
    """
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    o1, h1, l1, c1 = o.shift(1), h.shift(1), l.shift(1), c.shift(1)

    mid1 = (o1 + c1) / 2

    piercing = (
        (c1 < o1)            # 前根阴线
        & (c > o)            # 当根阳线
        & (o < l1)           # 开盘低跳空
        & (c > mid1)         # 收盘穿过前根实体中点
        & (c < o1)           # 但未完全吞没（否则是吞没形态）
    )
    dark_cloud = (
        (c1 > o1)            # 前根阳线
        & (c < o)            # 当根阴线
        & (o > h1)           # 开盘高跳空
        & (c < mid1)         # 收盘跌破前根实体中点
        & (c > o1)           # 但未完全吞没
    )

    result = pd.Series(0, index=df.index, name="cdl_piercing")
    result[piercing] = 1
    result[dark_cloud] = -1
    return result


# ─────────────────────────────────────────────────────────────────
#  三根K线形态
# ─────────────────────────────────────────────────────────────────

def cdl_morning_star(
    df: pd.DataFrame,
    star_body_ratio_max: float = 0.25,
    gap: bool = False,
) -> pd.Series:
    """
    早晨之星（看涨三线反转）：
      1. 第一根：较大阴线
      2. 第二根：小实体（星线），可以是阴或阳
      3. 第三根：较大阳线，收盘进入第一根实体内至少 50%

    Args:
        gap: True 时要求第二根与前后根有跳空（严格模式）
    """
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    o1, c1 = o.shift(1), c.shift(1)  # 星线
    o2, c2 = o.shift(2), c.shift(2)  # 第一根阴线

    body = _body(o, c)
    body1 = _body(o1, c1)
    body2 = _body(o2, c2)
    rng2 = _range(h.shift(2), l.shift(2)).fillna(1e-8)

    # 第一根：阴线，实体占比较大
    first_bear = (c2 < o2) & (body2 / rng2 > 0.4)
    # 第二根：小实体
    star = body1 < (body2 * star_body_ratio_max * 4)
    # 第三根：阳线，收盘进入第一根实体中点以上
    mid2 = (o2 + c2) / 2
    third_bull = (c > o) & (c > mid2)

    cond = first_bear & star & third_bull

    if gap:
        # 星线最高点 < 前根实体低点（向下跳空）
        gap1 = pd.concat([o1, c1], axis=1).max(axis=1) < pd.concat([o2, c2], axis=1).min(axis=1)
        cond = cond & gap1

    result = pd.Series(0, index=df.index, name="cdl_morning_star")
    result[cond] = 1
    return result


def cdl_evening_star(
    df: pd.DataFrame,
    star_body_ratio_max: float = 0.25,
    gap: bool = False,
) -> pd.Series:
    """
    黄昏之星（看跌三线反转）：
      1. 第一根：较大阳线
      2. 第二根：小实体（星线）
      3. 第三根：较大阴线，收盘跌入第一根实体内至少 50%
    """
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    o1, c1 = o.shift(1), c.shift(1)
    o2, c2 = o.shift(2), c.shift(2)

    body = _body(o, c)
    body1 = _body(o1, c1)
    body2 = _body(o2, c2)
    rng2 = _range(h.shift(2), l.shift(2)).fillna(1e-8)

    first_bull = (c2 > o2) & (body2 / rng2 > 0.4)
    star = body1 < (body2 * star_body_ratio_max * 4)
    mid2 = (o2 + c2) / 2
    third_bear = (c < o) & (c < mid2)

    cond = first_bull & star & third_bear

    if gap:
        gap1 = pd.concat([o1, c1], axis=1).min(axis=1) > pd.concat([o2, c2], axis=1).max(axis=1)
        cond = cond & gap1

    result = pd.Series(0, index=df.index, name="cdl_evening_star")
    result[cond] = -1
    return result


def cdl_three_white_soldiers(
    df: pd.DataFrame,
    min_body_ratio: float = 0.5,
    max_shadow_ratio: float = 0.3,
) -> pd.Series:
    """
    三白兵（强势看涨）：
      - 连续三根阳线
      - 每根阳线实体占振幅比大
      - 每根开盘在前根实体内
      - 每根收盘高于前根收盘（逐步走高）
    """
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    o1, h1, l1, c1 = o.shift(1), h.shift(1), l.shift(1), c.shift(1)
    o2, h2, l2, c2 = o.shift(2), h.shift(2), l.shift(2), c.shift(2)

    def is_soldier(op, hi, lo, cl, op_prev, cl_prev):
        br = _body(op, cl) / _range(hi, lo).fillna(1e-8)
        upper = _upper_shadow(op, hi, cl) / _range(hi, lo).fillna(1e-8)
        return (
            (cl > op)                       # 阳线
            & (br >= min_body_ratio)        # 实体够大
            & (upper <= max_shadow_ratio)   # 上影线较短
            & (op > op_prev)                # 开盘在前根实体内
            & (op < cl_prev)
            & (cl > cl_prev)                # 收盘创新高
        )

    s3 = is_soldier(o, h, l, c, o1, c1)
    s2 = is_soldier(o1, h1, l1, c1, o2, c2)
    s1 = (c2 > o2)  # 第一根只要求阳线

    result = pd.Series(0, index=df.index, name="cdl_three_white_soldiers")
    result[s1 & s2 & s3] = 1
    return result


def cdl_three_black_crows(
    df: pd.DataFrame,
    min_body_ratio: float = 0.5,
    max_shadow_ratio: float = 0.3,
) -> pd.Series:
    """
    三乌鸦（强势看跌）：
      - 连续三根阴线
      - 每根阴线实体占振幅比大
      - 每根开盘在前根实体内
      - 每根收盘低于前根收盘（逐步走低）
    """
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    o1, h1, l1, c1 = o.shift(1), h.shift(1), l.shift(1), c.shift(1)
    o2, h2, l2, c2 = o.shift(2), h.shift(2), l.shift(2), c.shift(2)

    def is_crow(op, hi, lo, cl, op_prev, cl_prev):
        br = _body(op, cl) / _range(hi, lo).fillna(1e-8)
        lower = _lower_shadow(op, lo, cl) / _range(hi, lo).fillna(1e-8)
        return (
            (cl < op)                       # 阴线
            & (br >= min_body_ratio)        # 实体够大
            & (lower <= max_shadow_ratio)   # 下影线较短
            & (op < op_prev)                # 开盘在前根实体内
            & (op > cl_prev)
            & (cl < cl_prev)                # 收盘创新低
        )

    c3 = is_crow(o, h, l, c, o1, c1)
    c2_ = is_crow(o1, h1, l1, c1, o2, c2)
    c1_ = (c2 < o2)  # 第一根只要求阴线

    result = pd.Series(0, index=df.index, name="cdl_three_black_crows")
    result[c1_ & c2_ & c3] = -1
    return result


# ─────────────────────────────────────────────────────────────────
#  批量扫描入口
# ─────────────────────────────────────────────────────────────────

#: 所有支持的形态函数映射
_PATTERN_FUNCS: dict[str, callable] = {
    "doji":                  cdl_doji,
    "hammer":                cdl_hammer,
    "spinning_top":          cdl_spinning_top,
    "marubozu":              cdl_marubozu,
    "engulfing":             cdl_engulfing,
    "harami":                cdl_harami,
    "piercing":              cdl_piercing,
    "morning_star":          cdl_morning_star,
    "evening_star":          cdl_evening_star,
    "three_white_soldiers":  cdl_three_white_soldiers,
    "three_black_crows":     cdl_three_black_crows,
}


def scan_candle_patterns(
    df: pd.DataFrame,
    patterns: list[str] | None = None,
) -> pd.DataFrame:
    """
    批量扫描蜡烛图形态，返回一个 DataFrame，每列对应一种形态。

    Args:
        df:       K 线 DataFrame，必须包含 open/high/low/close 列
        patterns: 要扫描的形态名称列表（None 表示扫描全部）

    Returns:
        DataFrame，列名为形态名称，值含义：
          1  = 看涨信号
         -1  = 看跌信号
          0  = 无信号

    Example:
        >>> result = scan_candle_patterns(df)
        >>> signals = result[result.any(axis=1)]  # 只保留有信号的行
        >>> bullish = result[(result == 1).any(axis=1)]  # 看涨信号
    """
    keys = patterns if patterns is not None else list(_PATTERN_FUNCS.keys())
    unknown = set(keys) - set(_PATTERN_FUNCS)
    if unknown:
        raise ValueError(f"未知形态：{unknown}，可用：{list(_PATTERN_FUNCS)}")

    results = {name: _PATTERN_FUNCS[name](df) for name in keys}
    return pd.DataFrame(results, index=df.index)


def get_latest_signals(
    df: pd.DataFrame,
    lookback: int = 3,
    patterns: list[str] | None = None,
) -> pd.DataFrame:
    """
    提取最近 N 根 K 线内触发的所有形态信号。

    Args:
        df:       K 线 DataFrame
        lookback: 向前查看的 K 线根数（默认 3）
        patterns: 要检测的形态列表（None=全部）

    Returns:
        DataFrame，只包含最近 lookback 根内有信号的行
    """
    result = scan_candle_patterns(df, patterns)
    recent = result.iloc[-lookback:]
    mask = (recent != 0).any(axis=1)
    return recent[mask]


# ─────────────────────────────────────────────────────────────────
#  与 ZigZag 结合：在 pivot 附近检测形态（旧接口，保留兼容）
# ─────────────────────────────────────────────────────────────────

#: 与底部（L 点）相关的看涨形态
_BULLISH_PATTERNS = [
    "engulfing",
    "hammer",
    "morning_star",
    "piercing",
    "harami",
    "doji",
    "spinning_top",
    "three_white_soldiers",
    "marubozu",
]

#: 与顶部（H 点）相关的看跌形态
_BEARISH_PATTERNS = [
    "engulfing",
    "hammer",
    "evening_star",
    "piercing",
    "harami",
    "doji",
    "spinning_top",
    "three_black_crows",
    "marubozu",
]


def near_pivot_signals(
    df: pd.DataFrame,
    pivots: list[dict],
    window: int = 3,
) -> list[dict]:
    """
    在每个 ZigZag pivot 的附近窗口内检测蜡烛图形态。

    对于 H 点（顶部）：只收集看跌信号（signal == -1）
    对于 L 点（底部）：只收集看涨信号（signal == 1）

    Args:
        df:      完整 K 线 DataFrame（open/high/low/close）
        pivots:  swing_pivots / swing_current_state 返回的 pivot 列表
        window:  在 pivot iloc 前后各扩展多少根 K 线（默认 3）
                 实际扫描范围：[iloc - window, iloc + window]

    Returns:
        list of dict，每项包含：
          - pivot_type:   "H" 或 "L"
          - pivot_date:   pivot 的日期字符串
          - pivot_value:  pivot 的价格
          - pivot_iloc:   pivot 在 df 中的位置
          - signal_iloc:  触发形态的 K 线在 df 中的位置
          - signal_date:  触发形态的日期字符串
          - signal_value: 触发形态那根 K 线的收盘价
          - pattern:      形态名称
          - direction:    "bearish"（顶部）或 "bullish"（底部）
    """
    if df is None or len(df) < 5 or not pivots:
        return []

    # 用全量 df 计算一次所有形态，避免重复计算
    all_signals = scan_candle_patterns(df)
    idx = df.index
    n = len(df)

    results: list[dict] = []

    for p in pivots:
        iloc = p["iloc"]
        ptype = p["type"]

        # 扫描范围
        lo = max(0, iloc - window)
        hi = min(n - 1, iloc + window)

        if ptype == "H":
            # 顶部：找看跌信号
            relevant_patterns = _BEARISH_PATTERNS
            target_signal = -1
            direction = "bearish"
        else:
            # 底部：找看涨信号
            relevant_patterns = _BULLISH_PATTERNS
            target_signal = 1
            direction = "bullish"

        for pattern in relevant_patterns:
            if pattern not in all_signals.columns:
                continue
            col = all_signals[pattern].iloc[lo: hi + 1]
            hits = col[col == target_signal]
            for sig_iloc_rel, _ in hits.items():
                sig_iloc = idx.get_loc(sig_iloc_rel) if not isinstance(sig_iloc_rel, int) else sig_iloc_rel
                # 用 index 查找确保正确
                sig_pos = all_signals.index.get_loc(sig_iloc_rel)
                close_val = float(df["close"].iloc[sig_pos])
                dt = idx[sig_pos]
                date_str = str(dt.date()) if hasattr(dt, "date") else str(dt)

                results.append({
                    "pivot_type":   ptype,
                    "pivot_date":   p["date"],
                    "pivot_value":  p["value"],
                    "pivot_iloc":   iloc,
                    "signal_iloc":  sig_pos,
                    "signal_date":  date_str,
                    "signal_value": close_val,
                    "pattern":      pattern,
                    "direction":    direction,
                })

    # 去重（同一根 K 线同一形态可能被多个相邻 pivot 捡到）
    seen: set[tuple] = set()
    deduped: list[dict] = []
    for r in results:
        key = (r["signal_iloc"], r["pattern"])
        if key not in seen:
            seen.add(key)
            deduped.append(r)

    return deduped


# ─────────────────────────────────────────────────────────────────
#  趋势感知形态识别（核心接口）
# ─────────────────────────────────────────────────────────────────

def trend_aware_hammer_star(
    df: pd.DataFrame,
    trend_series: "pd.Series",
    body_ratio_max: float = 0.35,
    shadow_ratio_min: float = 2.0,
) -> pd.Series:
    """
    基于 ZigZag 趋势方向的锤子线 / 墓碑线（射击之星）识别。

    与 cdl_hammer 的关键区别：
      - cdl_hammer 用「前一根K线涨跌」判断趋势（只看1根）
      - 本函数用 ZigZag 确定的趋势段（多根K线构成的方向）判断
        → 在更可靠的趋势背景下触发，减少误报

    逻辑：
      - 下跌段（trend == "down"）中出现锤子线（长下影/短上影）→ 1（看涨反转信号）
      - 上涨段（trend == "up"）中出现墓碑线/射击之星（长上影/短下影）→ -1（看跌反转信号）

    锤子线条件：实体 < 总振幅 * body_ratio_max，下影线 >= 实体 * shadow_ratio_min，上影线短
    墓碑线条件：实体 < 总振幅 * body_ratio_max，上影线 >= 实体 * shadow_ratio_min，下影线短

    Args:
        df:               K 线 DataFrame（open/high/low/close）
        trend_series:     由 trend_series_from_pivots 生成的趋势 Series，
                          值为 "up" | "down" | "unknown"
        body_ratio_max:   实体占总振幅最大比例（默认 0.35）
        shadow_ratio_min: 影线与实体的最小倍数（默认 2.0）

    Returns:
        pd.Series，值含义：
          1  = 下跌段中的锤子线（潜在底部反转）
         -1  = 上涨段中的墓碑线/射击之星（潜在顶部反转）
          0  = 无信号
    """
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    body  = _body(o, c).replace(0, 1e-8)
    upper = _upper_shadow(o, h, c)
    lower = _lower_shadow(o, l, c)
    br    = _body_ratio(o, h, l, c)

    # 锤子线：实体小，下影长，上影短
    is_hammer = (
        (br < body_ratio_max)
        & (lower >= body * shadow_ratio_min)
        & (upper < body * 0.5)
    )
    # 墓碑线 / 射击之星：实体小，上影长，下影短
    is_gravestone = (
        (br < body_ratio_max)
        & (upper >= body * shadow_ratio_min)
        & (lower < body * 0.5)
    )

    in_downtrend = trend_series == "down"
    in_uptrend   = trend_series == "up"

    result = pd.Series(0, index=df.index, name="trend_aware_hammer_star")
    result[is_hammer    & in_downtrend] = 1    # 下跌中锤子 → 看涨
    result[is_gravestone & in_uptrend]  = -1   # 上涨中墓碑 → 看跌
    return result
