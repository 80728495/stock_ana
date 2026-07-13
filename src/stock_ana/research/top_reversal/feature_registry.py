"""Feature registry for top-reversal research models.

The registry is intentionally explicit: each feature belongs to a strategy
module so new ideas such as SMC or resistance tests can be added
without burying them in the training script.
"""

from __future__ import annotations

from dataclasses import dataclass

from stock_ana.research.top_reversal.macro_micro_context import MACRO_MICRO_FEATURES
from stock_ana.research.top_reversal.prior_high_context import PRIOR_HIGH_FEATURES
from stock_ana.research.top_reversal.growth_context import GROWTH_FEATURES
from stock_ana.research.top_reversal.valuation_context import VALUATION_FEATURES
from stock_ana.research.top_reversal.vegas_context import VEGAS_TREND_FEATURES


@dataclass(frozen=True)
class FeatureGroup:
    """A named group of model features."""

    name: str
    description: str
    columns: tuple[str, ...]


CANDLE_PATTERN_FEATURES = (
    "has_shadow", "has_doji", "has_gap_fail", "signal_count", "score_max", "score_sum",
    "shadow_shadow_ratio", "shadow_shadow_pct", "shadow_shadow_atr", "shadow_prior_rise_pct",
    "shadow_vol_ratio", "shadow_confirm_body_ratio", "shadow_d2_break_d1_low",
    "doji_d1_body_ratio", "doji_d1_vol_ratio", "doji_d0_vol_ratio",
    "gap_fail_gap_open_pct", "gap_fail_true_gap_pct", "gap_fail_gap_fill_ratio",
    "gap_fail_true_gap_fill_ratio", "gap_fail_effective_gap_fill_ratio",
    "gap_fail_open_to_close_drop_pct",
    "gap_fail_body_pct", "gap_fail_body_ratio", "gap_fail_body_vs_avg20",
    "gap_fail_close_position_pct", "gap_fail_vol_ratio", "gap_fail_prior_rise_pct",
    "gap_fail_top_is_20d_high", "gap_fail_close_below_prev_high",
    "gap_fail_close_below_prev_close", "gap_fail_close_below_prev_low",
)

CANDLE_INTERACTION_FEATURES = (
    "candle_top_pattern",
    "candle_top_pattern_count",
    "candle_old_top_pattern",
    "candle_old_top_pattern_count",
    "candle_strict_old_top_pattern",
    "smc_early_with_shadow",
    "smc_early_with_doji",
    "smc_early_with_gap_fail",
    "smc_early_with_any_candle",
    "smc_early_with_old_candle",
    "smc_early_with_strict_old_candle",
    "smc_early_candle_score_max",
)

CANDIDATE_RECALL_FEATURES = (
    "recalled_by_candle",
    "recalled_by_shadow",
    "recalled_by_doji",
    "recalled_by_gap_fail",
    "recalled_by_double_top",
    "recalled_by_smc_raw",
    "recalled_by_smc_appear",
    "recalled_by_smc_confirmed",
    "recalled_by_smc_early",
    "recall_source_count",
    "score_lag_bars",
    "smc_raw_recall_count",
    "smc_raw_recall_score_max",
    "smc_raw_recall_detect_lag_min",
    "smc_raw_recall_confirm_lag_min",
    "smc_appear_recall_count",
    "smc_appear_recall_confirm_lag_min",
    "smc_appear_recall_leave_pct_max",
    "smc_appear_recall_zone_width_pct",
    "smc_confirmed_recall_count",
    "smc_confirmed_recall_score_max",
    "smc_confirmed_recall_struct_score_max",
    "smc_confirmed_recall_confirm_lag_min",
    "smc_confirmed_recall_zone_width_pct",
    "smc_confirmed_recall_volume_ratio",
    "smc_early_recall_count",
    "smc_early_recall_score_max",
    "smc_early_recall_raw_score_max",
    "smc_early_recall_struct_score_max",
    "smc_early_recall_confirm_lag_min",
    "double_top_recall_count",
    "double_top_recall_confirm_lag_min",
    "double_top_recall_neckline_break_pct",
    "double_top_recall_failed_rebound_vs_neckline_pct",
)

PRICE_CONTEXT_FEATURES = (
    "confirm_drop_from_top_pct", "vol_ratio_confirm_50", "atr14_pct",
    "prior_ret_5d", "prior_ret_10d", "prior_ret_20d", "prior_ret_40d", "prior_ret_60d", "prior_ret_120d",
    "dist_ema8_pct", "dist_ema20_pct", "dist_ema34_pct", "dist_ema55_pct", "dist_ema144_pct", "dist_ema200_pct",
    "top_dist_ema144_pct", "top_dist_ema200_pct",
    "ema55_slope_20d_pct", "range20_before_pct", "top_vs_252high_pct", "top_vs_252low_pct",
)

TECHNICAL_EXHAUSTION_FEATURES = (
    "rsi14_top", "rsi14_confirm", "rsi14_overbought_70", "rsi14_overbought_80",
    "stoch14_top", "stoch14_overbought_90", "bb20_zscore_top", "bb20_above_upper_pct",
    "macd_line_top_pct", "macd_hist_top_pct", "macd_hist_confirm_pct",
    "rsi14_divergence_pts_60d", "rsi14_bear_div_60d",
    "macd_line_divergence_pct_60d", "macd_line_bear_div_60d",
    "macd_hist_divergence_pct_60d", "macd_hist_bear_div_60d",
    "overbought_score",
    "vol_ratio_top_20", "vol_ratio_top_50", "high_volume_top_50",
    "vol5_vs_vol20_top", "vol10_vs_vol50_top", "vol5_ret20_pct",
    "volume_dryup_rise20", "price_up_volume_down_20d",
    "top_close_position_pct", "top_upper_shadow_pct", "high_volume_stall_score",
)

# Realtime/as-of ZigZag features.  These columns must be generated from data
# truncated at the candidate's score_asof_pos; global/full-series ZigZag
# values belong in oracle_* diagnostic columns and must not enter models.
ZIGZAG_ANCHOR_FEATURES = (
    "rise_from_recent_zigzag_low_pct", "rise_from_anchor_low_pct", "bars_from_anchor_low",
    "middle_vs_pre_head_pct", "anchor_is_middle_reset", "anchor_is_pre_head_low",
)

INDEX_SQUEEZE_FEATURES = (
    "china_hk_focus", "max_ret_5_10_20", "short_spike_like", "weak_confirm_short_spike",
    "china_hk_short_spike", "china_hk_hstech_ret_5d", "china_hk_hstech_ret_10d",
    "china_hk_hstech_ret_20d", "china_hk_hstech_ret_40d", "china_hk_hsi_ret_10d",
    "china_hk_hsi_ret_20d", "hstech_squeeze_10d", "hstech_squeeze_20d",
    "china_hk_index_squeeze_spike", "china_hk_index_squeeze_weak_confirm",
)

WAVE_STRUCTURE_FEATURES = (
    "major_wave_rise_pct", "major_wave_number", "major_sub_wave_count", "top_cluster_high_count",
    # 浪龄/浪内节奏（2026-07-13 自 vegas_pullback momentum_ctx 语义引入；字段本就在
    # analyze_wave_structure 输出里，as-of 因果由 _causal_zigzag_context 保证）：
    # 浪走得越久/浪内 mid 回踩次数越多 → 趋势越成熟，顶部风险语义。
    "major_wave_duration_days", "major_wave_mid_pullback_count",
)

# 相对强度 + 大盘方向（2026-07-13 自 vegas_pullback 移植，实现复用其 rs_features
# ——as-of 读取 RS 系统单股逐日 parquet，零前瞻）。对顶部的语义：
#   rs_line_dd_63 / rs_rank_delta_20 : RS 自身见顶回落 = 领涨股补跌前兆（vegas 实证
#     rank>80 且 RS 仍挂峰顶时回踩胜率最差 0.400——价格弱而 RS 挂顶 = 补跌起点）；
#   bench_ret_21d/63d, bench_dist_ma50_pct : 大盘方向（此前 top_reversal 只有截面，
#     US 完全无指数方向）——大盘转弱时个股顶更可能是真顶；
#   rs_benchmark_r2 : 系统性 vs 个股性行情判别器。
# caveat：US RS 史起 2023-02、HK/CN 起 2020-01，更早样本该组 NaN（lgb 原生处理）。
RS_STRENGTH_FEATURES = (
    "rs_return_21d", "rs_return_63d", "rs_momentum_21d", "rs_momentum_63d",
    "rs_rank_63d", "rs_benchmark_beta", "rs_benchmark_r2",
    "rs_line_dd_63", "rs_rank_delta_20",
    "bench_ret_21d", "bench_ret_63d", "bench_dist_ma50_pct",
)

SMC_LIVE_FEATURES = (
    "smc_live_bull_ob_count_60d",
    "smc_live_bull_ob_score_sum_60d",
    "smc_live_last_bull_ob_age",
    "smc_live_nearest_bull_ob_dist_pct",
    "smc_live_bull_ob_mitigated_10d",
    "smc_live_bear_ob_count_20d",
    "smc_live_bear_ob_score_max_20d",
    "smc_live_bear_ob_struct_score_max_20d",
    "smc_live_ob_regime_score",
)

SMC_RAW_FEATURES = (
    "smc_raw_bear_present_3d",
    "smc_raw_bear_score_max_3d",
    "smc_raw_bear_detect_lag",
    "smc_raw_bear_zone_overlap_top",
    "smc_raw_bear_displacement_atr",
    "smc_raw_bear_has_fvg",
    "smc_raw_bear_has_sweep",
    "smc_raw_bear_zone_width_atr",
    "smc_raw_bear_volume_ratio",
)

SMC_EARLY_FEATURES = (
    "smc_early_internal_choch_down_3d",
    "smc_early_internal_bos_down_3d",
    "smc_early_micro_low_break_3d",
    "smc_early_top_low_break_3d",
    "smc_early_bear_fvg_3d",
    "smc_early_bull_ob_mitigated_3d",
    "smc_early_bull_ladder_intact",
    "smc_early_liquidity_sweep_high_3d",
    "smc_early_retest_reject_5d",
    "smc_early_score_3d",
)

SMC_DELAYED_FEATURES = (
    "smc_d5_bear_ob_confirmed",
    "smc_d10_bear_ob_confirmed",
    "smc_d10_bear_ob_score_max",
    "smc_d10_bear_ob_struct_score_max",
    "smc_d10_bull_ob_mitigated_count",
    "smc_d10_ob_regime_flip",
)

SMC_DIAGNOSTIC_FEATURES = (
    "smc_diag_bear_ob_confirmed_near_top",
    "smc_diag_top_inside_bear_ob_confirmed_zone",
    "smc_diag_bear_ob_confirm_delay",
    "smc_diag_bear_ob_confirm_has_structure",
)

SMC_CAUSAL_FEATURES = SMC_LIVE_FEATURES + SMC_RAW_FEATURES + SMC_EARLY_FEATURES
SMC_STRUCTURE_FEATURES = SMC_CAUSAL_FEATURES + SMC_DELAYED_FEATURES + SMC_DIAGNOSTIC_FEATURES

RESISTANCE_FEATURES: tuple[str, ...] = ()


FEATURE_GROUPS = (
    FeatureGroup("candidate_recall", "Which realtime recall sources discovered the candidate.", CANDIDATE_RECALL_FEATURES),
    FeatureGroup("candle_pattern", "Top-candle candidate sources and their raw scores.", CANDLE_PATTERN_FEATURES),
    FeatureGroup("candle_interaction", "Explicit SMC and candle-overlap features for scoring recalled candidates.", CANDLE_INTERACTION_FEATURES),
    FeatureGroup("mid_vegas_trend", "Strict Mid Vegas uptrend structure at the candidate origin and score date.", VEGAS_TREND_FEATURES),
    FeatureGroup("price_context", "Returns, volume, EMA distance, range, and confirmation behavior.", PRICE_CONTEXT_FEATURES),
    FeatureGroup("technical_exhaustion", "RSI/MACD divergence, overbought, volume dry-up, and high-volume stall.", TECHNICAL_EXHAUSTION_FEATURES),
    FeatureGroup("zigzag_anchor", "As-of ZigZag anchor, M-head reset, and prior low context.", ZIGZAG_ANCHOR_FEATURES),
    FeatureGroup("index_squeeze", "China/HK index squeeze and short-spike context.", INDEX_SQUEEZE_FEATURES),
    FeatureGroup("wave_structure", "As-of wave structure and high-cluster context.", WAVE_STRUCTURE_FEATURES),
    FeatureGroup(
        "smc_structure",
        "SMC live OB, raw, early, confirmed, delayed, and diagnostic structure features.",
        SMC_STRUCTURE_FEATURES,
    ),
    FeatureGroup("prior_high_structure", "Causal prior-high hold and strict M/double-top shape.", PRIOR_HIGH_FEATURES),
    FeatureGroup("macro_micro", "Sector regime, cross-sectional rank, and parabolic over-extension.", MACRO_MICRO_FEATURES),
    FeatureGroup("valuation", "Market-relative PE (US forward / HK-CN trailing); see valuation_context caveat.", VALUATION_FEATURES),
    FeatureGroup("growth", "Fundamental EPS/revenue growth (causal) + sub-sector growth heat; general replacement for industry tags.", GROWTH_FEATURES),
    FeatureGroup("rs_strength", "Relative strength vs benchmark + benchmark regime direction (as-of).",
                 RS_STRENGTH_FEATURES),
    FeatureGroup("resistance", "Future prior-high and resistance-zone features.", RESISTANCE_FEATURES),
)

FEATURE_COLS = [col for group in FEATURE_GROUPS for col in group.columns]

REALTIME_FEATURE_GROUPS = (
    FeatureGroup("candidate_recall", "Which realtime recall sources discovered the candidate.", CANDIDATE_RECALL_FEATURES),
    FeatureGroup("candle_pattern", "Top-candle candidate sources and their raw scores.", CANDLE_PATTERN_FEATURES),
    FeatureGroup("candle_interaction", "Explicit SMC and candle-overlap features visible by the score date.", CANDLE_INTERACTION_FEATURES),
    FeatureGroup("mid_vegas_trend", "Strict Mid Vegas uptrend structure visible by the candidate score date.", VEGAS_TREND_FEATURES),
    FeatureGroup("price_context", "Returns, volume, EMA distance, range, and confirmation behavior.", PRICE_CONTEXT_FEATURES),
    FeatureGroup("technical_exhaustion", "Realtime RSI/MACD divergence, overbought, and volume exhaustion features.", TECHNICAL_EXHAUSTION_FEATURES),
    FeatureGroup("zigzag_anchor", "As-of ZigZag anchor, M-head reset, and prior low context.", ZIGZAG_ANCHOR_FEATURES),
    FeatureGroup("index_squeeze", "China/HK index squeeze and short-spike context.", INDEX_SQUEEZE_FEATURES),
    FeatureGroup("wave_structure", "As-of wave structure and high-cluster context.", WAVE_STRUCTURE_FEATURES),
    FeatureGroup("smc_causal", "SMC live, raw, and early features visible by the score date.", SMC_CAUSAL_FEATURES),
    FeatureGroup("prior_high_structure", "Causal prior-high hold and strict M/double-top shape.", PRIOR_HIGH_FEATURES),
    FeatureGroup("macro_micro", "Sector regime, cross-sectional rank, and parabolic over-extension.", MACRO_MICRO_FEATURES),
    FeatureGroup("valuation", "Market-relative PE (US forward / HK-CN trailing); see valuation_context caveat.", VALUATION_FEATURES),
    FeatureGroup("growth", "Fundamental EPS/revenue growth (causal) + sub-sector growth heat; general replacement for industry tags.", GROWTH_FEATURES),
    FeatureGroup("rs_strength", "Relative strength vs benchmark + benchmark regime direction (as-of).",
                 RS_STRENGTH_FEATURES),
    FeatureGroup("resistance", "Future prior-high and resistance-zone features.", RESISTANCE_FEATURES),
)

REALTIME_FEATURE_COLS = [col for group in REALTIME_FEATURE_GROUPS for col in group.columns]

# 顶后确认型特征：必须等顶后跌出结构/确认才发育——确认 K 线读数(confirm_drop_from_top_pct、
# vol_ratio_confirm_50、rsi14_confirm、macd_hist_confirm_pct)、swing-CHoCH 确认召回
# (smc_confirmed_recall_*，当前该召回禁用→恒为空)、双顶颈线破位(double_top_recall_neckline/
# failed)、两根 K 线破位确认(shadow_d2_break_d1_low)。它们语义属于"进一步确认"，
# 对早期发现(L0，顶后 lag 1-2 天)零 OOS 增益却系统性压低新鲜点分数
# (实测剔除后 watchlist-OOS lgb AUC 变动 <0.006)。发现打分应剔除，"确认"交给段 B 结构判定
# (swing CHoCH↓，见 escape_signal_tracker)。详见 memory: top-reversal-discovery-no-confirm-features。
POST_CONFIRMATION_FEATURE_COLS = (
    "recalled_by_smc_confirmed",
    "smc_confirmed_recall_count", "smc_confirmed_recall_score_max",
    "smc_confirmed_recall_struct_score_max", "smc_confirmed_recall_confirm_lag_min",
    "smc_confirmed_recall_zone_width_pct", "smc_confirmed_recall_volume_ratio",
    "double_top_recall_neckline_break_pct", "double_top_recall_failed_rebound_vs_neckline_pct",
    "confirm_drop_from_top_pct", "vol_ratio_confirm_50",
    "rsi14_confirm", "macd_hist_confirm_pct", "shadow_d2_break_d1_low",
)

# 早期发现(L0)打分用特征 = 实时特征剔除顶后确认型。段 A 早发现模型专用。
DISCOVERY_FEATURE_COLS = [c for c in REALTIME_FEATURE_COLS if c not in set(POST_CONFIRMATION_FEATURE_COLS)]

ORACLE_ZIGZAG_CONTEXT_COLS = tuple(f"oracle_{col}" for col in ZIGZAG_ANCHOR_FEATURES + WAVE_STRUCTURE_FEATURES)

BUCKET_COLS = [
    "recall_source_count", "score_lag_bars", "smc_confirmed_recall_score_max",
    "mid_vegas_live_passed", "mid_vegas_top_days_above_long", "mid_vegas_top_days_above_mid",
    "mid_vegas_top_mid_long_gap_pct", "mid_vegas_top_close_dist_mid_pct",
    "smc_appear_recall_leave_pct_max", "smc_appear_recall_confirm_lag_min",
    "smc_early_recall_score_max", "smc_early_recall_confirm_lag_min",
    "rise_from_anchor_low_pct", "rise_from_recent_zigzag_low_pct", "prior_ret_60d", "prior_ret_120d",
    "top_dist_ema144_pct", "top_dist_ema200_pct",
    "dist_ema55_pct", "dist_ema144_pct", "confirm_drop_from_top_pct", "score_max",
    "candle_top_pattern", "candle_old_top_pattern", "smc_early_with_old_candle", "smc_early_with_gap_fail",
    "gap_fail_effective_gap_fill_ratio", "gap_fail_body_vs_avg20",
    "rsi14_top", "rsi14_divergence_pts_60d", "macd_hist_divergence_pct_60d",
    "bb20_zscore_top", "vol5_vs_vol20_top", "vol_ratio_top_50", "high_volume_stall_score",
    "middle_vs_pre_head_pct",
    "max_ret_5_10_20", "china_hk_hstech_ret_20d",
    "top_cluster_high_count", "major_sub_wave_count", "smc_live_ob_regime_score",
    "smc_live_bear_ob_score_max_20d", "smc_raw_bear_score_max_3d",
    "smc_early_score_3d", "smc_d10_bear_ob_score_max",
]


def feature_group_for(column: str) -> str:
    """Return the group name a feature column belongs to."""

    for group in FEATURE_GROUPS:
        if column in group.columns:
            return group.name
    return "unregistered"


def feature_group_summary() -> list[dict[str, object]]:
    """Small table-friendly summary of registered feature groups."""

    return [
        {
            "group": group.name,
            "n_features": len(group.columns),
            "description": group.description,
        }
        for group in FEATURE_GROUPS
    ]


LEGACY_FEATURE_ALIASES = {
    "recalled_by_smc_origin": "recalled_by_smc_confirmed",
    "covered_by_smc_origin": "covered_by_smc_confirmed",
    "smc_origin_recall_count": "smc_confirmed_recall_count",
    "smc_origin_recall_score_max": "smc_confirmed_recall_score_max",
    "smc_origin_recall_struct_score_max": "smc_confirmed_recall_struct_score_max",
    "smc_origin_recall_confirm_lag_min": "smc_confirmed_recall_confirm_lag_min",
    "smc_origin_recall_zone_width_pct": "smc_confirmed_recall_zone_width_pct",
    "smc_origin_recall_volume_ratio": "smc_confirmed_recall_volume_ratio",
    "smc_origin_recall_origin_date": "smc_confirmed_recall_formed_date",
    "smc_origin_recall_confirm_date": "smc_confirmed_recall_confirmed_date",
    "smc_confirmed_recall_origin_date": "smc_confirmed_recall_formed_date",
    "smc_confirmed_recall_confirm_date": "smc_confirmed_recall_confirmed_date",
    "smc_origin_bear_present_3d": "smc_raw_bear_present_3d",
    "smc_origin_bear_score_max_3d": "smc_raw_bear_score_max_3d",
    "smc_origin_bear_detect_lag": "smc_raw_bear_detect_lag",
    "smc_origin_bear_zone_overlap_top": "smc_raw_bear_zone_overlap_top",
    "smc_origin_bear_displacement_atr": "smc_raw_bear_displacement_atr",
    "smc_origin_bear_has_fvg": "smc_raw_bear_has_fvg",
    "smc_origin_bear_has_sweep": "smc_raw_bear_has_sweep",
    "smc_origin_bear_zone_width_atr": "smc_raw_bear_zone_width_atr",
    "smc_origin_bear_volume_ratio": "smc_raw_bear_volume_ratio",
    "smc_early_recall_origin_score_max": "smc_early_recall_raw_score_max",
    "smc_diag_bear_ob_origin_near_top": "smc_diag_bear_ob_confirmed_near_top",
    "smc_diag_top_inside_bear_ob_origin_zone": "smc_diag_top_inside_bear_ob_confirmed_zone",
}


def apply_legacy_feature_aliases(df):
    """Return a dataframe with old SMC column names mirrored to the new names."""

    if df.empty:
        return df
    out = df.copy()
    for old, new in LEGACY_FEATURE_ALIASES.items():
        if old not in out.columns:
            continue
        if new not in out.columns:
            out[new] = out[old]
    return out
