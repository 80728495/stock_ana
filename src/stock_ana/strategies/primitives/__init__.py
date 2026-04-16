"""策略基础积木（primitives）。

存放可被多个策略复用的低层算法组件：
  pivots    — 高低点提取（argrel + ZigZag）
  peaks     — 宏观峰值识别
  cup       — 杯底结构分析
  wave      — EMA 波段结构分析
  regression — OLS 拟合
  rs        — 相对强度计算
  momentum  — 量价异动评分
  squeeze   — 均线收敛度量
  trend     — Minervini Stage 2 趋势过滤
  vcp       — VCP 波幅收缩检测

只放策略基础能力，不放完整策略语义。
"""

from stock_ana.strategies.primitives.pivots import (
    argrel_pivots,
    merge_pivots_with_zigzag,
    multiscale_argrel_pivots,
    zigzag_indices,
    zigzag_points,
)
from stock_ana.strategies.primitives.cup import (
    analyze_cup_base_waves,
    bottom_channel_ratio,
    check_cup_ma_trend,
    find_cup_structure,
    poly_smoothness,
)
from stock_ana.strategies.primitives.peaks import find_macro_peaks
from stock_ana.strategies.primitives.regression import line_value, ols_fit
from stock_ana.strategies.primitives.momentum import (
    score_abnormal_return,
    score_accumulation,
    score_breakout,
    score_gap_up,
    score_ma_breakout,
    score_volume_surge,
)
from stock_ana.strategies.primitives.rs import compute_rs_line, compute_rs_rank_63d, compute_rs_rank_at_cutoff
from stock_ana.strategies.primitives.squeeze import (
    compute_ma_squeeze_ratio,
    compute_volume_trend_ratio,
    is_recent_crossover,
    normalized_price_range,
)
from stock_ana.strategies.primitives.trend import check_trend_template
from stock_ana.strategies.primitives.vcp import detect_vcp_micro_structure
from stock_ana.strategies.primitives.wave import analyze_wave_structure, detect_ema8_swings

__all__ = [
    "argrel_pivots",
    "merge_pivots_with_zigzag",
    "multiscale_argrel_pivots",
    "zigzag_indices",
    "zigzag_points",
    "poly_smoothness",
    "bottom_channel_ratio",
    "analyze_cup_base_waves",
    "find_cup_structure",
    "check_cup_ma_trend",
    "find_macro_peaks",
    "score_volume_surge",
    "score_abnormal_return",
    "score_breakout",
    "score_gap_up",
    "score_ma_breakout",
    "score_accumulation",
    "normalized_price_range",
    "compute_ma_squeeze_ratio",
    "compute_volume_trend_ratio",
    "is_recent_crossover",
    "ols_fit",
    "line_value",
    "compute_rs_line",
    "compute_rs_rank_63d",
    "compute_rs_rank_at_cutoff",
    "check_trend_template",
    "detect_vcp_micro_structure",
    "detect_ema8_swings",
    "analyze_wave_structure",
]
