"""
Vegas 均线区带定义与 EMA 计算工具。

Vegas 体系由两组 EMA 构成：
  - Mid Vegas：EMA34 / EMA55 / EMA60（中期通道，用于升浪中的回踩触碰）
  - Long Vegas：EMA144 / EMA169 / EMA200（长期通道，用于浪结构起点与趋势判断）

本模块是这两组 EMA 的唯一计算来源，供以下模块共用：
  - strategies/primitives/wave.py        — 波段结构识别
  - strategies/impl/vegas_mid.py         — Mid Vegas 触碰检测与评分
  - strategies/impl/main_rally_pullback.py — 主升浪结构确认（mid + long 双区）

Exports
-------
MID_EMAS, LONG_EMAS                          — EMA span 常量列表
compute_vegas_emas(close_s) -> dict          — 计算全部 6 条 EMA，返回 {span: ndarray}
vegas_ema_series(close_s) -> dict            — 同上但返回 pd.Series，便于 DataFrame 操作
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ── Vegas EMA 配置 ──────────────────────────────────────────────────────────

MID_EMAS: list[int] = [34, 55, 60]
LONG_EMAS: list[int] = [144, 169, 200]
ALL_VEGAS_EMAS: list[int] = MID_EMAS + LONG_EMAS

# 结构判断窗口常量（被 vegas_mid 和 main_rally_pullback 共用）
LONG_SLOPE_WINDOW: int = 20   # Long Vegas 斜率计算窗口（交易日）
PRICE_ABOVE_LONG_WINDOW: int = 30  # 股价须持续高于 Long Vegas 的最少天数（~1.5 个月）


# ── EMA 计算 ────────────────────────────────────────────────────────────────

def compute_vegas_emas(close_s: pd.Series) -> dict[int, np.ndarray]:
    """
    计算所有 Vegas EMA，返回 {span: ndarray}。

    涵盖 Mid Vegas（EMA34/55/60）和 Long Vegas（EMA144/169/200）。
    结果为 numpy ndarray，可直接用下标索引，适合逐 bar 扫描型算法。

    Args:
        close_s: 收盘价 Series（已按时间升序排列）

    Returns:
        {34: ndarray, 55: ndarray, 60: ndarray, 144: ndarray, 169: ndarray, 200: ndarray}
    """
    return {
        s: close_s.ewm(span=s, adjust=False).mean().values
        for s in ALL_VEGAS_EMAS
    }


def vegas_ema_series(close_s: pd.Series) -> dict[int, pd.Series]:
    """
    计算所有 Vegas EMA，返回 {span: pd.Series}。

    与 compute_vegas_emas 相同，但返回 Series 而非 ndarray，
    便于挂载到 DataFrame 列或进行 .iloc 切片操作，适合批量结构判断型算法。

    Args:
        close_s: 收盘价 Series（已按时间升序排列）

    Returns:
        {34: Series, 55: Series, 60: Series, 144: Series, 169: Series, 200: Series}
    """
    return {
        s: close_s.ewm(span=s, adjust=False).mean()
        for s in ALL_VEGAS_EMAS
    }
