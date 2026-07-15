"""Vegas 回踩模型的特征目录。

信号日特征（signal_features）在本包定义；基本面 / 估值 / 宏观 / 前高等跨表
特征直接复用 top_reversal 的 context 模块特征清单，避免重复造轮子。

REALTIME_FEATURE_COLS = 训练与打分共用的因果特征全集。
FEATURE_GROUPS 供重要性报告按组归类。
"""

from __future__ import annotations

from dataclasses import dataclass

from stock_ana.research.vegas_pullback.signal_features import (
    STRUCTURE_FEATURES,
    CHANNEL_FEATURES,
    MOMENTUM_FEATURES,
    VOLATILITY_FEATURES,
    MOMENTUM_CTX_FEATURES,
    EARLY_TREND_FEATURES,
    MICRO_FEATURES,
    CLUSTER_FEATURES,
    FIB_FEATURES,
)
from stock_ana.research.vegas_pullback.fund_inflection import FUND_INFLECTION_FEATURES
from stock_ana.research.vegas_pullback.rs_features import RS_FEATURES
from stock_ana.research.vegas_pullback.sqz_features import SQZ_FEATURES
from stock_ana.research.top_reversal.valuation_context import VALUATION_FEATURES as _VAL_ALL
from stock_ana.research.top_reversal.growth_context import GROWTH_FEATURES

# 估值特征只保留 PIT 绝对值 + 行业内 as-of 分位；剔除 *_pct_mkt——它把全部候选
# 跨时间池化排名（一行的分位取决于未来候选的估值分布），是已知泄漏模式
# （top_reversal 在行业分位上实测过 CN 0.83→0.71）。消融显示其 OOS 贡献≈0。
VALUATION_FEATURES: tuple[str, ...] = tuple(
    c for c in _VAL_ALL if not c.endswith("_pct_mkt")
)
from stock_ana.research.top_reversal.macro_micro_context import MACRO_MICRO_FEATURES
# 注：不复用 top_reversal 的 prior_high —— 它锚定「顶部」事件（top_pos，判断顶是否
# 靠近前高 / M 顶），对回踩（一个低点）语义不适用，实测特征全空、gain=0。
# 回踩相关的「距前高」已由 mom_dist_52w_high_pct / chan_peak_gap_pct 覆盖。


@dataclass(frozen=True)
class FeatureGroup:
    """A named group of model features."""

    name: str
    description: str
    columns: tuple[str, ...]


FEATURE_GROUPS: tuple[FeatureGroup, ...] = (
    FeatureGroup("structure", "浪结构上下文（回踩序次/连续浪/浪顶涨幅/子浪）", STRUCTURE_FEATURES),
    FeatureGroup("channel", "Vegas 通道几何（斜率/上方占比/浪顶落差/触碰深度）", CHANNEL_FEATURES),
    FeatureGroup("momentum", "动量与趋势（多窗口收益/RSI/MACD/距 52 周高低）", MOMENTUM_FEATURES),
    FeatureGroup("volatility", "波动与量能（ATR/已实现波动/触碰量比）", VOLATILITY_FEATURES),
    FeatureGroup("momentum_ctx", "动能上下文（大浪连续/浪龄/站上LV成熟度/回踩浅度/吸筹/新高近度）", MOMENTUM_CTX_FEATURES),
    FeatureGroup("early_trend", "早期趋势（长底部占比/趋势年龄/量能扩张——第1浪→大二浪识别）", EARLY_TREND_FEATURES),
    FeatureGroup("micro", "触碰日微观形态（下影线/收盘位置/弹起力度ATR/触碰量能/climax振幅）", MICRO_FEATURES),
    FeatureGroup("cluster", "踩线簇（触线密度/刺破滞留/低点趋势/振幅收敛/量能枯竭/线自身斜率）", CLUSTER_FEATURES),
    FeatureGroup("fib", "斐波那契回撤（浪锚 swing 回吐比例/最近档位/档位共振度ATR/波段幅度）", FIB_FEATURES),
    FeatureGroup("fund_inflection", "基本面拐点（增长加速度 PIT 一阶差分——困境反转 W1 识别）", FUND_INFLECTION_FEATURES),
    FeatureGroup("rs", "相对强度（RS 系统：超额收益/RS动量/市场内百分位/benchmark beta·R²）", RS_FEATURES),
    FeatureGroup("squeeze", "LazyBear挤压动量（自归一动量/加速度/squeeze态/蓄势时长/零轴年龄）", SQZ_FEATURES),
    FeatureGroup("macro_micro", "大盘/板块状态（复用 top_reversal）", MACRO_MICRO_FEATURES),
    FeatureGroup("growth", "PIT 增长（复用 top_reversal）", GROWTH_FEATURES),
    FeatureGroup("valuation", "PIT 估值分位（复用 top_reversal）", VALUATION_FEATURES),
)


# 训练/打分共用的因果特征全集（顺序稳定）
REALTIME_FEATURE_COLS: tuple[str, ...] = tuple(
    col for group in FEATURE_GROUPS for col in group.columns
)


_FEATURE_TO_GROUP: dict[str, str] = {
    col: group.name for group in FEATURE_GROUPS for col in group.columns
}


def feature_group_for(feature: str) -> str:
    """Return the group name for a feature column (or 'other')."""
    return _FEATURE_TO_GROUP.get(feature, "other")
