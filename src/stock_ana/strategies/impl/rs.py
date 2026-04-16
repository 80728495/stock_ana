"""
RS 加速策略 (Relative Strength Acceleration)
=============================================

核心思路：
    不买当前最强的股票 (RS 90+)，而是买 **正在从普通变强** 的股票。
    捕捉 RS Line 从有意义的底部拐头加速、价格同时在蓄力的时刻。

RS Line = Stock Close / QQQ Close

两组参数：
    - rs_strict: 更严格，更高精度
    - rs_loose:  更宽松，更多信号量

条件：
    1. RS 百分位排名在中间区间（非顶部，非底部）
    2. RS 短期加速（21 日变化率 > 63 日变化率，且 >= 最低门槛）
    3. RS Line 刚突破自身 EMA21（近 N 天上穿），且之前在 EMA 下方停留足够久
    4. 价格结构健康（SMA50 > SMA200，价格接近 52 周高点）
    5. 波动率收缩（ATR10/ATR50 < 阈值，价格在蓄力）
"""

import numpy as np
import pandas as pd
from loguru import logger

from stock_ana.data.fetcher import update_qqq_data
from stock_ana.strategies.primitives.rs import (
    compute_rs_line as primitive_compute_rs_line,
    compute_rs_rank_63d as primitive_compute_rs_rank_63d,
    compute_rs_rank_at_cutoff as primitive_compute_rs_rank_at_cutoff,
)

# ═══════════════════════════════════════════════════════════
# 参数组
# ═══════════════════════════════════════════════════════════

RS_PARAMS = {
    "rs_strict": {
        "rank_lo": 40,            # RS 百分位排名下限
        "rank_hi": 80,            # RS 百分位排名上限
        "cross_window": 5,        # RS Line 上穿 EMA21 的窗口（天）
        "min_below_days": 15,     # 上穿前 RS 在 EMA 下方的最少天数
        "min_accel": 2.0,         # 21 日 RS 变化率最低 %
        "atr_ratio_max": 0.85,    # ATR(10)/ATR(50) < 此值 = 波动率收缩
        "price_from_high": 0.80,  # 价格 >= 52 周高点 × 此值
    },
    "rs_loose": {
        "rank_lo": 35,            # 更宽的排名区间
        "rank_hi": 85,
        "cross_window": 10,       # 更长的上穿窗口
        "min_below_days": 10,     # 更短的底部要求
        "min_accel": 1.0,         # 更低的加速门槛
        "atr_ratio_max": 1.0,     # 不要求波动率收缩
        "price_from_high": 0.75,  # 更宽的价格位置
    },
}


# ═══════════════════════════════════════════════════════════
# QQQ 数据加载
# ═══════════════════════════════════════════════════════════

def _load_qqq() -> pd.DataFrame | None:
    """加载本地 QQQ 数据（与个股存放在同目录）。"""
    from stock_ana.data.fetcher import load_local_data
    df = load_local_data("QQQ")
    if df is None or df.empty:
        logger.warning("QQQ 本地数据不存在，请先运行 update_qqq_data()")
    return df


# ═══════════════════════════════════════════════════════════
# RS Line 计算
# ═══════════════════════════════════════════════════════════

def compute_rs_line(
    df_stock: pd.DataFrame,
    df_market: pd.DataFrame,
) -> pd.Series | None:
    """Compatibility wrapper for the shared RS-line primitive."""
    return primitive_compute_rs_line(df_stock, df_market)


def compute_rs_rank_63d(
    stock_data: dict[str, pd.DataFrame],
    df_market: pd.DataFrame,
) -> dict[str, float]:
    """Compatibility wrapper for the shared 63-day RS rank primitive."""
    return primitive_compute_rs_rank_63d(stock_data, df_market)


# ═══════════════════════════════════════════════════════════
# 单股筛选
# ═══════════════════════════════════════════════════════════

def screen_relative_strength(
    df_stock: pd.DataFrame,
    df_market: pd.DataFrame,
    rs_rank: float,
    variant: str = "rs_strict",
) -> dict | None:
    """
    RS 加速策略筛选。

    Args:
        df_stock: 个股 DataFrame（需含 close, high, low, volume）
        df_market: 市场基准 DataFrame（QQQ，需含 close）
        rs_rank: 该股在全体中的 63 日收益率百分位排名 (0-100)
        variant: "rs_strict" 或 "rs_loose"

    Returns:
        信号详情 dict，或 None（不满足条件）
    """
    params = RS_PARAMS[variant]

    # ── 条件 0: 数据最低要求 ──
    if len(df_stock) < 252:
        return None

    closes = df_stock["close"].values.astype(float)
    highs = df_stock["high"].values.astype(float)
    lows = df_stock["low"].values.astype(float)

    # ── 条件 4a: Price > SMA(200) 且 SMA(50) > SMA(200) ──
    c_series = pd.Series(closes)
    sma200 = c_series.rolling(200).mean().iloc[-1]
    sma50 = c_series.rolling(50).mean().iloc[-1]
    if np.isnan(sma200) or np.isnan(sma50):
        return None
    if closes[-1] <= sma200:
        return None
    if sma50 <= sma200:
        return None  # 均线死叉，趋势不健康

    # ── 条件 4b: 价格接近 52 周高点 ──
    lookback_52w = min(260, len(df_stock))
    high_52w = float(np.max(highs[-lookback_52w:]))
    if closes[-1] < high_52w * params["price_from_high"]:
        return None  # 距高点太远，可能在下跌趋势中

    # ── 计算 RS Line ──
    rs_line = compute_rs_line(df_stock, df_market)
    if rs_line is None or len(rs_line) < 252:
        return None

    rs_vals = rs_line.values.astype(float)

    # ── 条件 1: RS 百分位排名在中间区间 ──
    if rs_rank < params["rank_lo"] or rs_rank > params["rank_hi"]:
        return None

    # ── 条件 2: RS 短期加速 ──
    if len(rs_vals) < 63:
        return None
    rs_chg_21d = (rs_vals[-1] / rs_vals[-21] - 1) * 100 if rs_vals[-21] > 0 else 0
    rs_chg_63d = (rs_vals[-1] / rs_vals[-63] - 1) * 100 if rs_vals[-63] > 0 else 0

    if rs_chg_21d < params["min_accel"]:
        return None  # 加速度不够
    if rs_chg_21d <= rs_chg_63d:
        return None  # 没有在加速

    # ── 条件 3: RS Line 刚上穿 EMA21，且之前在 EMA 下方待了足够久 ──
    rs_series = pd.Series(rs_vals, index=rs_line.index)
    rs_ema21 = rs_series.ewm(span=21, adjust=False).mean().values

    cross_window = params["cross_window"]
    min_below_days = params["min_below_days"]

    # 当前 RS > EMA21
    if rs_vals[-1] <= rs_ema21[-1]:
        return None

    # 检查近 cross_window 天内发生过上穿
    recently_below = False
    check_start = max(1, len(rs_vals) - cross_window - 1)
    for i in range(check_start, len(rs_vals) - 1):
        if rs_vals[i] <= rs_ema21[i]:
            recently_below = True
            break
    if not recently_below:
        return None

    # RS 在 EMA 下方的持续天数（往前找最多 60 天）
    below_count = 0
    scan_start = max(0, len(rs_vals) - 60)
    for i in range(len(rs_vals) - 2, scan_start - 1, -1):
        if rs_vals[i] <= rs_ema21[i]:
            below_count += 1
        elif below_count > 0:
            break  # 遇到上方天就停止，只计最近一段连续下方
    if below_count < min_below_days:
        return None  # 底部太短，可能是噪声交叉

    # ── 条件 5: 波动率收缩 ──
    n = len(closes)
    tr = np.empty(n)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        tr[i] = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
    atr10 = float(np.mean(tr[-10:]))
    atr50 = float(np.mean(tr[-50:])) if n >= 50 else atr10
    atr_ratio = atr10 / atr50 if atr50 > 0 else 999

    if atr_ratio > params["atr_ratio_max"]:
        return None  # 波动率没有收缩，不是蓄力状态

    # ── 通过所有条件 ──
    return {
        "rs_rank": round(rs_rank, 1),
        "rs_chg_21d": round(rs_chg_21d, 2),
        "rs_chg_63d": round(rs_chg_63d, 2),
        "acceleration": round(rs_chg_21d - rs_chg_63d, 2),
        "rs_line": round(float(rs_vals[-1]), 6),
        "rs_ema21": round(float(rs_ema21[-1]), 6),
        "rs_below_days": below_count,
        "atr_ratio": round(atr_ratio, 3),
        "price_vs_52w_high": round(closes[-1] / high_52w * 100, 1),
        "sma50": round(float(sma50), 2),
        "sma200": round(float(sma200), 2),
        "close": round(float(closes[-1]), 2),
        "variant": variant,
    }


# ═══════════════════════════════════════════════════════════
# 回测专用：在给定截止点计算 RS 排名并筛选
# ═══════════════════════════════════════════════════════════

def compute_rs_rank_at_cutoff(
    stock_data: dict[str, pd.DataFrame],
    df_market: pd.DataFrame,
    cutoff_idx: int,
) -> dict[str, float]:
    """Compatibility wrapper for the shared cutoff RS rank primitive."""
    return primitive_compute_rs_rank_at_cutoff(stock_data, df_market, cutoff_idx)


# ═══════════════════════════════════════════════════════════
# RS 陷阱预警 (RS Trap / Illusion Detector)
# ═══════════════════════════════════════════════════════════
#
# 场景：大盘下跌时某只股票反而抗跌甚至上涨，表面上相对强度很好，
#       但实际上中长期 RS Line 处于下降趋势。短期的"相对强势"
#       只是噪声反弹，容易诱人追入。
#
# 信号条件（全部满足才会预警）：
#   1. 市场近期下跌: QQQ 近 lookback 天收益 < 阈值
#   2. 个股表面抗跌: 个股近 lookback 天跑赢市场 >= gap_min
#   3. 中期 RS 趋势走弱: RS Line < EMA(RS, 50)
#   4. 长期 RS 排名低: 63 日 RS 百分位排名 < rank_ceil
#   5. RS Line 在下降: 63 日 RS 变化率 < 0
#   6. 价格结构恶化: Price < SMA(200) 或 SMA(50) < SMA(200)
#
# 两组参数对比：rs_trap_strict / rs_trap_loose

RS_TRAP_PARAMS = {
    "rs_trap_strict": {
        "lookback": 5,           # 短期观察窗口（天）
        "mkt_drop_min": -1.0,    # 市场至少下跌 1%
        "outperform_gap": 2.0,   # 个股相对市场至少跑赢 2ppt
        "rs_ema_span": 50,       # RS EMA 周期
        "rank_ceil": 40,         # RS 排名 < 40% 才算弱
        "require_price_weak": True,  # 要求价格结构恶化
    },
    "rs_trap_loose": {
        "lookback": 10,          # 更长窗口
        "mkt_drop_min": -0.5,    # 市场跌 0.5% 即可
        "outperform_gap": 1.0,   # 跑赢 1ppt 即可
        "rs_ema_span": 50,
        "rank_ceil": 50,         # 排名 < 50%
        "require_price_weak": False,  # 不强制要求价格结构恶化
    },
}


def screen_rs_trap(
    df_stock: pd.DataFrame,
    df_market: pd.DataFrame,
    rs_rank: float,
    variant: str = "rs_trap_strict",
) -> dict | None:
    """
    RS 陷阱预警：检测"看起来强但实际弱"的股票。

    Args:
        df_stock: 个股 DataFrame
        df_market: QQQ DataFrame
        rs_rank: 63 日 RS 百分位排名 (0-100)
        variant: "rs_trap_strict" 或 "rs_trap_loose"

    Returns:
        预警详情 dict，或 None（非陷阱）
    """
    params = RS_TRAP_PARAMS[variant]

    if len(df_stock) < 252:
        return None

    closes = df_stock["close"].values.astype(float)

    # ── 计算 RS Line ──
    # 去掉 freq 避免 pandas intersection 做 freq 推断（在 iloc 切片后很慢）
    if hasattr(df_market.index, 'freq') and df_market.index.freq is None:
        df_market = df_market.copy()
        df_market.index = df_market.index._with_freq(None)  # explicit no-freq
    rs_line = compute_rs_line(df_stock, df_market)
    if rs_line is None or len(rs_line) < 200:
        return None

    rs_vals = rs_line.values.astype(float)
    lookback = params["lookback"]

    # ── 条件 1: 市场近期下跌 ──
    # 复用 rs_line 的 index 作为 common_idx（已由 compute_rs_line 对齐）
    common_idx = rs_line.index
    mkt_close = df_market.loc[common_idx, "close"].values.astype(float)
    if len(mkt_close) < lookback + 1:
        return None
    mkt_ret = (mkt_close[-1] / mkt_close[-lookback - 1] - 1) * 100
    if mkt_ret > params["mkt_drop_min"]:
        return None  # 市场没怎么跌，不算陷阱场景

    # ── 条件 2: 个股表面抗跌/逆势上涨 ──
    stk_aligned = df_stock.loc[common_idx, "close"].values.astype(float)
    if len(stk_aligned) < lookback + 1:
        return None
    stk_ret = (stk_aligned[-1] / stk_aligned[-lookback - 1] - 1) * 100
    outperform = stk_ret - mkt_ret
    if outperform < params["outperform_gap"]:
        return None  # 并没有明显跑赢，不是"引人注目的抗跌"

    # ── 条件 3: 中期 RS 趋势走弱 ──
    rs_series = pd.Series(rs_vals, index=rs_line.index)
    rs_ema = rs_series.ewm(span=params["rs_ema_span"], adjust=False).mean().values
    if rs_vals[-1] >= rs_ema[-1]:
        return None  # RS 在均线上方，中期趋势不弱

    # ── 条件 4: 长期 RS 排名低 ──
    if rs_rank >= params["rank_ceil"]:
        return None  # 排名不够低，这股确实可能是强的

    # ── 条件 5: RS Line 在下降 ──
    if len(rs_vals) < 63:
        return None
    rs_chg_63d = (rs_vals[-1] / rs_vals[-63] - 1) * 100 if rs_vals[-63] > 0 else 0
    if rs_chg_63d >= 0:
        return None  # 63 日 RS 没有下降

    # ── 条件 6 (可选): 价格结构恶化 ──
    c_series = pd.Series(closes)
    sma200 = c_series.rolling(200).mean().iloc[-1]
    sma50 = c_series.rolling(50).mean().iloc[-1]
    price_below_200 = closes[-1] < sma200 if not np.isnan(sma200) else False
    death_cross = sma50 < sma200 if (not np.isnan(sma50) and not np.isnan(sma200)) else False
    price_weak = price_below_200 or death_cross

    if params["require_price_weak"] and not price_weak:
        return None

    # ── RS Line 距 EMA 的偏离程度（越负说明越弱）──
    rs_deviation = (rs_vals[-1] / rs_ema[-1] - 1) * 100

    # ── 短期 RS 变化率（正 = 近期反弹）──
    rs_chg_short = (rs_vals[-1] / rs_vals[-lookback - 1] - 1) * 100 if rs_vals[-lookback - 1] > 0 else 0

    return {
        "rs_rank": round(rs_rank, 1),
        "mkt_ret": round(mkt_ret, 2),
        "stk_ret": round(stk_ret, 2),
        "outperform": round(outperform, 2),
        "rs_chg_63d": round(rs_chg_63d, 2),
        "rs_chg_short": round(rs_chg_short, 2),
        "rs_vs_ema": round(rs_deviation, 2),
        "price_below_sma200": price_below_200,
        "death_cross": death_cross,
        "sma50": round(float(sma50), 2) if not np.isnan(sma50) else None,
        "sma200": round(float(sma200), 2) if not np.isnan(sma200) else None,
        "close": round(float(closes[-1]), 2),
        "variant": variant,
    }


