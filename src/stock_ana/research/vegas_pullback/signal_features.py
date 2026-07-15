"""信号日特征（在回踩触碰 bar t 处，仅用 ≤t 的因果数据）。

分四组，均在 build 阶段逐候选内联计算（build 已持有 df/emas/waves）：

  structure_* : 浪结构上下文（回踩序次、连续浪、是否浪终点、本浪涨幅/时长、
                子浪数、连续性）
  chan_*      : Vegas 通道几何（斜率、上方占比、浪顶落差、年内涨幅、通道宽度、
                触碰深度、收盘距通道）
  mom_*       : 动量 / 趋势（多窗口收益、EMA 斜率、RSI、MACD 柱、距 52 周高低）
  vol_*       : 波动 / 量能（ATR%、已实现波动、触碰量比、量趋势）

基本面 / 估值 / 宏观 / 前高 等跨表特征在 feature_pipeline 里复用 top_reversal
的 context 模块（按 market/sym/as-of 日查外部数据），不在此处。
"""

from __future__ import annotations

import numpy as np

from stock_ana.strategies.primitives.vegas_zones import MID_EMAS, LONG_EMAS


# ── 特征列清单（feature_registry 引用）────────────────────────────────────────

STRUCTURE_FEATURES: tuple[str, ...] = (
    "structure_pullback_seq", "structure_consec_waves", "structure_is_wave_end",
    "structure_connected_prev", "structure_wave_rise_pct", "structure_wave_duration",
    "structure_days_since_wave_start", "structure_sub_wave_count",
)

CHANNEL_FEATURES: tuple[str, ...] = (
    "chan_long_slope_pct", "chan_above_ratio", "chan_peak_gap_pct",
    "chan_rise_from_1y_low_pct", "chan_mid_long_gap_pct", "chan_touch_depth_pct",
    "chan_close_dist_pct", "chan_ema34_slope_pct", "chan_ema55_slope_pct",
    "chan_ema_order_ok", "chan_width_pct",
)

MOMENTUM_FEATURES: tuple[str, ...] = (
    "mom_ret_20", "mom_ret_60", "mom_ret_120", "mom_rsi14", "mom_macd_hist_norm",
    "mom_dist_52w_high_pct", "mom_dist_52w_low_pct", "mom_above_ema34",
)

VOLATILITY_FEATURES: tuple[str, ...] = (
    "vol_atr_pct", "vol_realized_20", "vol_realized_60",
    "vol_touch_vol_ratio", "vol_trend_20_60",
)

# 动能上下文（v2）：衡量「这只股在回踩点是不是真·强势动量」——vegas 回踩的本意
# 就是抄动量股的呼吸。用多路证据，不单靠浪结构：
#   momctx_wave_*      : 触碰所在「大浪」的上下文（连续浪链、当前浪涨幅、浪龄）
#   momctx_days_above_long / dist_above_long : 站上 Long Vegas 的成熟度与延展度
#   momctx_pct_above_ema34 / ema_fan         : 趋势持续性与加速度
#   momctx_pullback_depth                    : 本次回踩深度（浅=强势）
#   momctx_up_down_vol / new_high_recency    : 吸筹与新高近度
MOMENTUM_CTX_FEATURES: tuple[str, ...] = (
    "momctx_wave_number", "momctx_consec_waves", "momctx_cur_wave_rise_pct",
    "momctx_days_above_long", "momctx_dist_above_long_pct",
    "momctx_pct_above_ema34_60", "momctx_ema_fan_pct",
    "momctx_pullback_depth_pct", "momctx_up_down_vol_20", "momctx_new_high_recency",
)

# 早期趋势上下文（第 1 浪回踩 → 大二浪识别）：第 1 浪回踩时「连续浪链」证据
# 尚不存在，需要该阶段就已可见的证据——长期底部之后的第一浪最有力（弹簧
# 效应）、突破须放量、趋势越年轻潜在空间越大：
#   early_base_ratio_250  : 过去 250 根收盘 ≤ LV×1.03 的占比（刚脱离长底部=高）
#   early_breakout_age    : 距最近一次收盘深破 LV(×0.97) 的 bar 数（趋势年龄，
#                           capped 500；第 1 浪回踩时应当很小）
#   early_vol_expansion   : 近 40 根均量 / 之前 40~160 根均量（第 1 浪 vs 底部
#                           的量能扩张倍数）
EARLY_TREND_FEATURES: tuple[str, ...] = (
    "early_base_ratio_250", "early_breakout_age", "early_vol_expansion",
)

# 触碰日微观形态（v3，用户第一阶段）：「踩线当天怎么弹起来」的量化。
# 因果注意：锚点=确认日（93% 与触碰同日、7% 晚 1-2 天），全部特征只用
# [touch_bar, anchor] 窗口——同日确认时"反弹"即触碰日盘中收复本身。
#   micro_shadow_ratio    : 触碰日下影线 / 全振幅（长下影=承接）
#   micro_touch_close_pos : 触碰日收盘在当日振幅中的位置 0~1（收在高处=强）
#   micro_anchor_close_pos: 锚点日收盘位置（两日确认时=确认日的收盘质量）
#   micro_bounce_atr      : 触碰日最低→锚点收盘的反弹幅度 / ATR14（弹起力度，
#                           跨股票可比）
#   micro_touch_vol_ratio : 触碰日量 / 前20日均量（恐慌抛售/放量承接的量能）
#   micro_touch_range_atr : 触碰日振幅 / ATR14（是否 climax bar）
# v2 扩展（2026-07-12 锤子线深挖后泛化）：核心命题 =「止跌形态 + 有效量能」。
# 实证（2万mid触碰交叉统计）：量能是主轴——缩量触碰(vol<0.8)一律差(0.37-0.43)，
# 适度放量(1.5-2.5×)一律好(0.44-0.54)，巨量(>2.5×)是 climax 最差(0.367)；
# 形态在量能之后做二次区分：未刺破摸线弹 > 浅刺破 > 深刺破收回。
#   micro_downleg_vol_ratio : 下跌段均量/高点前均量（缩量回调=健康 vs 放量派发）
#   micro_downleg_days      : 近期高点→触碰的天数（快杀 vs 阴跌）
#   micro_last2_slope_atr   : 触碰前2日收盘斜率/ATR（跌势减速度，趋0=止跌）
#   micro_pierce_state      : 刺破序数 0=未刺破摸线弹/1=浅刺(0~-2%)/2=深刺(<-2%)
#   micro_vol_band          : 量能四档 0缩量/1常量/2放量甜区/3巨量climax（倒U显式化）
#   micro_stab_confirm      : 止跌确认位 = 量比∈[1.2,2.5] 且 未深刺破
MICRO_FEATURES: tuple[str, ...] = (
    "micro_shadow_ratio", "micro_touch_close_pos", "micro_anchor_close_pos",
    "micro_bounce_atr", "micro_touch_vol_ratio", "micro_touch_range_atr",
    "micro_downleg_vol_ratio", "micro_downleg_days", "micro_last2_slope_atr",
    "micro_pierce_state", "micro_vol_band", "micro_stab_confirm",
)

# 踩线簇特征（v3.1，2026-07-13 修正）：价格在**本次信号所踩的那条线**附近
# 反复震荡时的状态。两个关键语义修正（用户指出）：
#   ① 按信号自己的 ema_span 计算（踩 ema34 和踩到 ema55 深度不同，绝不混线）；
#   ② episode 锚定：簇从「价格上次明确离开该线（收盘 ≥ 线×1.05）之后、
#      首次(几乎)触到该线」开始，而非固定回看窗口——不会把上一段上涨或
#      别的线的触碰扫进来。episode 上限 40 bar。
#   cluster_ep_len            : episode 长度（贴线被钉了多少天）
#   cluster_touch_days        : episode 内触该线天数（low ≤ 线×1.02）
#   cluster_below_close_days  : episode 内收盘在该线下方的天数（刺破滞留度）
#   cluster_touch_low_trend_atr: episode 首末触线日低点之差/ATR（抬高=承接）
#   cluster_range_contract    : 近5日振幅/前5日振幅（<1 贴线收敛，线无关）
#   cluster_vol_fade          : episode 均量 / episode 前20日均量（枯竭 vs 放大）
#   cluster_line_slope        : 该线 episode 期间斜率%（每10bar 归一）
CLUSTER_FEATURES: tuple[str, ...] = (
    "cluster_ep_len", "cluster_touch_days", "cluster_below_close_days",
    "cluster_touch_low_trend_atr", "cluster_range_contract",
    "cluster_vol_fade", "cluster_line_slope",
)

# 斐波那契回撤特征（2026-07-13）：核心是锚定「当前区间的低点」——
# 首选浪锚（浪起点=LV 触点确认的结构低点，因果峰=起点→锚点最高价），
# 无浪上下文时兜底（近120日高点→该高点前250日最低点）。
#   fib_retrace_pct    : 触碰低点回吐主导波段涨幅的比例（0.382/0.5/0.618 连续版）
#   fib_nearest_level  : 最近的斐波档位值（0.236/0.382/0.5/0.618/0.786）
#   fib_dist_level_atr : 触碰低点距最近档位价格 / ATR（档位共振度，斐波独有主张）
#   fib_range_pct      : 主导波段幅度 %（区间上下文）
#   fib_anchor_wave    : 锚定质量 1=浪锚 0=兜底
FIB_FEATURES: tuple[str, ...] = (
    "fib_retrace_pct", "fib_nearest_level", "fib_dist_level_atr",
    "fib_range_pct", "fib_anchor_wave",
)

SIGNAL_FEATURE_COLS: tuple[str, ...] = (
    STRUCTURE_FEATURES + CHANNEL_FEATURES + MOMENTUM_FEATURES
    + VOLATILITY_FEATURES + MOMENTUM_CTX_FEATURES + EARLY_TREND_FEATURES
    + MICRO_FEATURES + CLUSTER_FEATURES + FIB_FEATURES
)

_FIB_LEVELS = (0.236, 0.382, 0.5, 0.618, 0.786)


def compute_fib_features(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    anchor: int,
    wave_ctx: dict | None,
) -> dict:
    """斐波那契回撤特征（零前瞻）。swing 锚定：浪锚优先，滚动兜底。"""
    a = anchor
    out = {c: np.nan for c in FIB_FEATURES}

    # ── 锚定 swing_low / swing_high ──
    if wave_ctx is not None:
        sp = int(wave_ctx["start_pivot"]["iloc"])
        swing_low = float(np.min(low[max(0, sp - 2) : sp + 3]))
        swing_high = float(np.max(high[sp : a + 1]))
        out["fib_anchor_wave"] = 1
    else:
        lb = max(0, a - 120)
        hi_pos = lb + int(np.argmax(high[lb : a + 1]))
        lo_lb = max(0, hi_pos - 250)
        swing_low = float(np.min(low[lo_lb : hi_pos + 1]))
        swing_high = float(np.max(high[hi_pos : a + 1]))
        out["fib_anchor_wave"] = 0

    rng = swing_high - swing_low
    if rng <= 0 or swing_low <= 0:
        return out
    out["fib_range_pct"] = round(rng / swing_low * 100, 2)

    touch_low = float(low[a])
    retrace = (swing_high - touch_low) / rng
    out["fib_retrace_pct"] = round(retrace, 3)

    atr = _atr_pct(high, low, close, a, 14) / 100 * float(close[a])
    if atr > 0:
        dists = [(lv, abs(touch_low - (swing_high - lv * rng)) / atr) for lv in _FIB_LEVELS]
        lv, dist = min(dists, key=lambda t: t[1])
        out["fib_nearest_level"] = lv
        out["fib_dist_level_atr"] = round(dist, 3)
    return out


def compute_cluster_features(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    band: np.ndarray,
    anchor: int,
    touch_margin: float = 0.02,
    depart_pct: float = 0.05,
    max_ep: int = 40,
) -> dict:
    """踩线簇特征 v3.1（零前瞻：只用 ≤ anchor 的数据）。

    Args:
        band: 本次信号所踩的那条 EMA 序列（如 emas[55]）——绝不混线。
        depart_pct: episode 起点判定：自 anchor 回溯，收盘 ≥ band×(1+5%)
                    即视为「价格明确在该线上方」，episode 从其后一根开始。
    """
    a = anchor
    line = band.astype(float)
    out: dict = {}

    # ── episode 起点：上次明确离开该线之后 ──
    ep_start = max(0, a - max_ep + 1)
    for j in range(a - 1, max(0, a - max_ep) - 1, -1):
        if float(close[j]) >= float(line[j]) * (1 + depart_pct):
            ep_start = j + 1
            break
    win = slice(ep_start, a + 1)
    ep_len = a - ep_start + 1
    out["cluster_ep_len"] = int(ep_len)

    touch_mask = low[win] <= line[win] * (1 + touch_margin)
    below_mask = close[win] < line[win]
    out["cluster_touch_days"] = int(np.sum(touch_mask))
    out["cluster_below_close_days"] = int(np.sum(below_mask))

    atr = _atr_pct(high, low, close, a, 14) / 100 * float(close[a])
    tidx = np.where(touch_mask)[0]
    if len(tidx) >= 2 and atr > 0:
        out["cluster_touch_low_trend_atr"] = round(
            (float(low[win][tidx[-1]]) - float(low[win][tidx[0]])) / atr, 3
        )
    else:
        # 单触 episode：无先例可比 → 0（中性）。若填 NaN，覆盖率仅 ~28% 会被
        # usable_features 的 50% 门槛整体剔除——真·多次踩线人群的信息反而丢失。
        out["cluster_touch_low_trend_atr"] = 0.0

    if a >= 10:
        r_recent = float(np.max(high[a - 4 : a + 1])) - float(np.min(low[a - 4 : a + 1]))
        r_prior = float(np.max(high[a - 9 : a - 4])) - float(np.min(low[a - 9 : a - 4]))
        out["cluster_range_contract"] = round(r_recent / r_prior, 3) if r_prior > 0 else np.nan
    else:
        out["cluster_range_contract"] = np.nan

    # episode 均量 vs episode 前 20 日均量
    v = np.asarray(volume, dtype=float)
    pre_lb = max(0, ep_start - 20)
    if ep_start - pre_lb >= 5 and np.mean(v[pre_lb:ep_start]) > 0:
        out["cluster_vol_fade"] = round(float(np.mean(v[win])) / float(np.mean(v[pre_lb:ep_start])), 3)
    else:
        out["cluster_vol_fade"] = np.nan

    # 该线 episode 期间斜率（每 10 bar 归一，便于跨 episode 长度比较）
    if ep_len >= 2 and float(line[ep_start]) > 0:
        out["cluster_line_slope"] = round(
            (float(line[a]) / float(line[ep_start]) - 1) * 100 / ep_len * 10, 3
        )
    else:
        out["cluster_line_slope"] = np.nan
    return out


def compute_micro_features(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    touch_bar: int,
    anchor: int,
    emas: dict[int, np.ndarray] | None = None,
    support: str = "mid",
) -> dict:
    """触碰日微观形态特征（零前瞻：只用 [touch_bar, anchor] 及之前数据）。

    v2：核心命题「止跌形态 + 有效量能」——含下跌段量能收缩、跌势减速度、
    刺破序数、量能倒U分档与止跌确认位。emas/support 用于计算触碰相对
    支撑线的刺破深度（None 时刺破类特征为 NaN）。
    """
    t = touch_bar
    a = anchor
    o, h, l, c = float(open_[t]), float(high[t]), float(low[t]), float(close[t])
    rng = h - l
    out: dict = {}
    if rng > 0:
        out["micro_shadow_ratio"] = round((min(o, c) - l) / rng, 3)
        out["micro_touch_close_pos"] = round((c - l) / rng, 3)
    else:
        out["micro_shadow_ratio"] = np.nan
        out["micro_touch_close_pos"] = np.nan
    rng_a = float(high[a]) - float(low[a])
    out["micro_anchor_close_pos"] = round((float(close[a]) - float(low[a])) / rng_a, 3) if rng_a > 0 else np.nan

    atr = _atr_pct(high, low, close, a, 14) / 100 * float(close[a])  # 绝对 ATR
    out["micro_bounce_atr"] = round((float(close[a]) - l) / atr, 3) if atr > 0 else np.nan
    out["micro_touch_range_atr"] = round(rng / atr, 3) if atr > 0 else np.nan

    v = np.asarray(volume, dtype=float)
    vol_ratio = np.nan
    if t >= 20 and np.mean(v[t - 20 : t]) > 0:
        vol_ratio = float(v[t]) / float(np.mean(v[t - 20 : t]))
    out["micro_touch_vol_ratio"] = round(vol_ratio, 2) if pd_notna(vol_ratio) else np.nan

    # ── v2: 止跌形态 + 有效量能 ──
    # 下跌段：近 20 日最高价 bar → 触碰前一日
    lb = max(0, t - 20)
    hi_pos = lb + int(np.argmax(high[lb : t + 1]))
    out["micro_downleg_days"] = int(t - hi_pos)
    if t - hi_pos >= 2 and hi_pos >= 20:
        leg_v = float(np.mean(v[hi_pos : t]))
        pre_v = float(np.mean(v[hi_pos - 20 : hi_pos]))
        out["micro_downleg_vol_ratio"] = round(leg_v / pre_v, 2) if pre_v > 0 else np.nan
    else:
        out["micro_downleg_vol_ratio"] = np.nan
    # 跌势减速度：触碰前 2 日平均收盘变化 / ATR（趋 0 = 止跌）
    if t >= 2 and atr > 0:
        out["micro_last2_slope_atr"] = round((float(close[t]) - float(close[t - 2])) / 2 / atr, 3)
    else:
        out["micro_last2_slope_atr"] = np.nan
    # 刺破序数（相对支撑通道上沿）
    pierce = np.nan
    if emas is not None:
        spans = MID_EMAS if support == "mid" else LONG_EMAS
        line = max(float(emas[s][t]) for s in spans)
        if line > 0:
            depth = (l / line - 1) * 100
            pierce = 0 if depth >= 0 else (1 if depth > -2 else 2)
    out["micro_pierce_state"] = pierce
    # 量能四档（倒U显式化）与止跌确认位
    if pd_notna(vol_ratio):
        out["micro_vol_band"] = 0 if vol_ratio < 0.8 else (1 if vol_ratio < 1.5 else (2 if vol_ratio <= 2.5 else 3))
        out["micro_stab_confirm"] = int(
            1.2 <= vol_ratio <= 2.5 and pd_notna(pierce) and pierce <= 1
        )
    else:
        out["micro_vol_band"] = np.nan
        out["micro_stab_confirm"] = np.nan
    return out


def pd_notna(x) -> bool:
    """轻量 notna（避免顶层引入 pandas 依赖习惯不一致）。"""
    return x == x and x is not None


def _slope_pct(arr: np.ndarray, bar: int, window: int) -> float:
    if bar < window:
        return 0.0
    prev = float(arr[bar - window])
    return (float(arr[bar]) / prev - 1) * 100 if prev > 0 else 0.0


def _rsi(close: np.ndarray, bar: int, period: int = 14) -> float:
    if bar < period:
        return 50.0
    diff = np.diff(close[bar - period : bar + 1].astype(float))
    up = diff[diff > 0].sum()
    down = -diff[diff < 0].sum()
    if down == 0:
        return 100.0 if up > 0 else 50.0
    rs = up / down
    return 100 - 100 / (1 + rs)


def compute_signal_features(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    volume: np.ndarray,
    emas: dict[int, np.ndarray],
    touch_bar: int,
    support: str,
    wave_pb: dict,
    wave_ctx: dict | None = None,
    long_slope_window: int = 20,
    above_window: int = 60,
) -> dict:
    """在 touch_bar 处产出全部信号日特征（因果，仅用 ≤touch_bar 的数据）。

    通道几何全部在此直接从 emas/close/low 计算，mid / long 口径一致，不依赖
    上层门控函数返回的具体 key。

    Args:
        wave_pb: locate_wave_pullback 的返回（seq / is_wave_end / wave / wave_rise_pct）。
        wave_ctx: find_wave_context 的返回（触碰所「在」的大浪；mid 浪内回踩靠它
                  才能吃到大浪动能上下文，而非要求触碰恰是浪终点）。
        support: "mid" | "long" —— 决定 touch_depth / close_dist 用哪组通道。
    """
    i = touch_bar
    c0 = float(close[i])
    spans = MID_EMAS if support == "mid" else LONG_EMAS
    lv = emas[spans[0]].astype(float).copy()
    for s in spans[1:]:
        lv = np.maximum(lv, emas[s].astype(float))
    lv_i = float(lv[i])
    long_arr = np.maximum(np.maximum(emas[144].astype(float), emas[169].astype(float)), emas[200].astype(float))
    mid_upper = max(float(emas[34][i]), float(emas[55][i]))
    long_upper = float(long_arr[i])

    out: dict = {}

    # ── structure ──
    wave = wave_pb.get("wave")
    out["structure_pullback_seq"] = int(wave_pb.get("seq", 0))
    out["structure_is_wave_end"] = int(bool(wave_pb.get("is_wave_end", False)))
    out["structure_wave_rise_pct"] = round(float(wave_pb.get("wave_rise_pct", 0.0)), 2)
    if wave is not None:
        out["structure_connected_prev"] = int(bool(wave.get("connected_prev", False)))
        out["structure_sub_wave_count"] = int(wave.get("sub_wave_count", 0))
        sp = wave["start_pivot"]["iloc"]
        pk = wave["peak_pivot"]["iloc"]
        out["structure_wave_duration"] = int(pk - sp)
        out["structure_days_since_wave_start"] = int(i - sp)
    else:
        out["structure_connected_prev"] = 0
        out["structure_sub_wave_count"] = 0
        out["structure_wave_duration"] = 0
        out["structure_days_since_wave_start"] = 0
    out["structure_consec_waves"] = int(wave_pb.get("consec_waves", 0))

    # ── channel geometry（全部直接从 emas/close/low 计算，mid/long 口径一致）──
    out["chan_long_slope_pct"] = round(_slope_pct(long_arr, i, long_slope_window), 2)
    win_start = max(0, i - above_window + 1)
    win_close = close[win_start : i + 1].astype(float)
    win_lv = long_arr[win_start : i + 1]
    out["chan_above_ratio"] = round(float(np.mean(win_close >= win_lv)), 3) if win_close.size else np.nan
    peak_close = float(np.max(win_close)) if win_close.size else 0.0
    out["chan_peak_gap_pct"] = round((peak_close / long_upper - 1) * 100, 2) if long_upper > 0 else np.nan
    lb1y = max(0, i - 252 + 1)
    low_1y = float(np.min(low[lb1y : i + 1]))
    out["chan_rise_from_1y_low_pct"] = round((c0 / low_1y - 1) * 100, 2) if low_1y > 0 else np.nan
    out["chan_mid_long_gap_pct"] = round((mid_upper / long_upper - 1) * 100, 2) if long_upper > 0 else np.nan
    touch_low = float(low[i])
    out["chan_touch_depth_pct"] = round((touch_low / lv_i - 1) * 100, 2) if lv_i > 0 else np.nan
    out["chan_close_dist_pct"] = round((c0 / lv_i - 1) * 100, 2) if lv_i > 0 else np.nan
    out["chan_ema34_slope_pct"] = round(_slope_pct(emas[34], i, 20), 2)
    out["chan_ema55_slope_pct"] = round(_slope_pct(emas[55], i, 20), 2)
    out["chan_ema_order_ok"] = int(
        emas[34][i] > emas[55][i] > emas[144][i] > emas[169][i]
    )
    out["chan_width_pct"] = round((mid_upper / long_upper - 1) * 100, 2) if long_upper > 0 else np.nan

    # ── momentum / trend ──
    for w, col in [(20, "mom_ret_20"), (60, "mom_ret_60"), (120, "mom_ret_120")]:
        out[col] = round(_slope_pct(close, i, w), 2)
    out["mom_rsi14"] = round(_rsi(close, i), 1)
    # MACD 柱（12/26/9）归一到收盘价
    ema12 = emas.get(12)
    if ema12 is None:
        s = np.asarray(close, dtype=float)
        e12 = _ewm(s, 12); e26 = _ewm(s, 26)
        macd = e12 - e26
        signal = _ewm(macd, 9)
        hist = float(macd[i] - signal[i])
    else:
        hist = 0.0
    out["mom_macd_hist_norm"] = round(hist / c0 * 100, 3) if c0 > 0 else 0.0
    lb = max(0, i - 252)
    hi_52 = float(np.max(high[lb : i + 1]))
    lo_52 = float(np.min(low[lb : i + 1]))
    out["mom_dist_52w_high_pct"] = round((c0 / hi_52 - 1) * 100, 2) if hi_52 > 0 else 0.0
    out["mom_dist_52w_low_pct"] = round((c0 / lo_52 - 1) * 100, 2) if lo_52 > 0 else 0.0
    out["mom_above_ema34"] = int(c0 >= float(emas[34][i]))

    # ── volatility / volume ──
    out["vol_atr_pct"] = round(_atr_pct(high, low, close, i, 14), 2)
    out["vol_realized_20"] = round(_realized_vol(close, i, 20), 2)
    out["vol_realized_60"] = round(_realized_vol(close, i, 60), 2)
    v = np.asarray(volume, dtype=float)
    if i >= 20 and np.mean(v[i - 20 : i]) > 0:
        out["vol_touch_vol_ratio"] = round(float(v[i]) / float(np.mean(v[i - 20 : i])), 2)
    else:
        out["vol_touch_vol_ratio"] = np.nan
    if i >= 60 and np.mean(v[i - 60 : i - 20]) > 0:
        out["vol_trend_20_60"] = round(
            float(np.mean(v[i - 20 : i])) / float(np.mean(v[i - 60 : i - 20])), 2
        )
    else:
        out["vol_trend_20_60"] = np.nan

    # ── momentum context（v2 动能证据）──
    # 大浪上下文：触碰所在的连续大浪（find_wave_context），衡量底层动能
    if wave_ctx is not None:
        out["momctx_wave_number"] = int(wave_ctx.get("wave_number", 0))
        out["momctx_consec_waves"] = int(wave_ctx.get("_consec_waves", 0))
        sp_val = float(wave_ctx["start_pivot"]["value"])
        out["momctx_cur_wave_rise_pct"] = round((c0 / sp_val - 1) * 100, 2) if sp_val > 0 else np.nan
    else:
        out["momctx_wave_number"] = 0
        out["momctx_consec_waves"] = 0
        out["momctx_cur_wave_rise_pct"] = np.nan
    # 站上 Long Vegas 的成熟度与延展度
    w120 = min(120, i)
    if w120 > 0:
        out["momctx_days_above_long"] = round(
            float(np.mean(close[i - w120 : i + 1].astype(float) >= long_arr[i - w120 : i + 1])) * 100, 1
        )
    else:
        out["momctx_days_above_long"] = np.nan
    out["momctx_dist_above_long_pct"] = round((c0 / long_upper - 1) * 100, 2) if long_upper > 0 else np.nan
    # 趋势持续性与加速度
    w60 = min(60, i)
    if w60 > 0:
        out["momctx_pct_above_ema34_60"] = round(
            float(np.mean(close[i - w60 : i + 1].astype(float) >= emas[34][i - w60 : i + 1])) * 100, 1
        )
    else:
        out["momctx_pct_above_ema34_60"] = np.nan
    e144 = float(emas[144][i])
    out["momctx_ema_fan_pct"] = round((float(emas[34][i]) / e144 - 1) * 100, 2) if e144 > 0 else np.nan
    # 本次回踩深度（相对近 30 日最高价；浅=强）
    rh = float(np.max(high[max(0, i - 30) : i + 1]))
    out["momctx_pullback_depth_pct"] = round((float(low[i]) / rh - 1) * 100, 2) if rh > 0 else np.nan
    # 上涨日/下跌日量比（近 20 日吸筹）
    if i >= 20:
        seg_c = close[i - 20 : i + 1].astype(float)
        seg_v = np.asarray(volume[i - 20 : i + 1], dtype=float)
        dchg = np.diff(seg_c)
        up_v = seg_v[1:][dchg > 0].sum()
        dn_v = seg_v[1:][dchg < 0].sum()
        out["momctx_up_down_vol_20"] = round(up_v / dn_v, 2) if dn_v > 0 else np.nan
    else:
        out["momctx_up_down_vol_20"] = np.nan
    # 距上次 52 周新高的近度（bars 越小=动量越新鲜；归一到 0~1，1=当日即新高）
    lb2 = max(0, i - 252)
    hh = high[lb2 : i + 1].astype(float)
    if hh.size:
        last_high_off = int(len(hh) - 1 - int(np.argmax(hh)))
        out["momctx_new_high_recency"] = round(1.0 / (1.0 + last_high_off), 3)
    else:
        out["momctx_new_high_recency"] = np.nan

    # ── early trend context（第 1 浪回踩 → 大二浪识别）──
    lb250 = max(0, i - 250)
    seg_c = close[lb250 : i + 1].astype(float)
    seg_lv = long_arr[lb250 : i + 1]
    out["early_base_ratio_250"] = round(float(np.mean(seg_c <= seg_lv * 1.03)), 3) if seg_c.size else np.nan
    # 趋势年龄：距最近一次收盘深破 LV(×0.97) 的 bar 数（capped 500）
    below = np.where(close[: i + 1].astype(float) < long_arr[: i + 1] * 0.97)[0]
    out["early_breakout_age"] = int(min(500, i - below[-1])) if below.size else 500
    # 量能扩张：近 40 根均量 / 之前 40~160 根均量
    if i >= 160:
        recent_v = float(np.mean(v[i - 40 : i + 1]))
        base_v = float(np.mean(v[i - 160 : i - 40]))
        out["early_vol_expansion"] = round(recent_v / base_v, 2) if base_v > 0 else np.nan
    else:
        out["early_vol_expansion"] = np.nan

    return out


def _ewm(arr: np.ndarray, span: int) -> np.ndarray:
    alpha = 2 / (span + 1)
    out = np.empty_like(arr, dtype=float)
    out[0] = arr[0]
    for k in range(1, len(arr)):
        out[k] = alpha * arr[k] + (1 - alpha) * out[k - 1]
    return out


def _atr_pct(high, low, close, bar, period=14) -> float:
    if bar < period:
        return 0.0
    h = high[bar - period + 1 : bar + 1].astype(float)
    l = low[bar - period + 1 : bar + 1].astype(float)
    pc = close[bar - period : bar].astype(float)
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    atr = float(np.mean(tr))
    c0 = float(close[bar])
    return atr / c0 * 100 if c0 > 0 else 0.0


def _realized_vol(close, bar, window) -> float:
    if bar < window:
        return 0.0
    seg = close[bar - window : bar + 1].astype(float)
    rets = np.diff(np.log(np.clip(seg, 1e-9, None)))
    return float(np.std(rets) * np.sqrt(252) * 100)
