"""
Smart Money Concepts (SMC) 指标封装

基于 smartmoneyconcepts 库（https://github.com/joshyattridge/smart-money-concepts）
对 ICT 核心概念进行统一封装，供扫描器和策略层调用。

支持的指标：
  - FVG (公允价值缺口)
  - Swing Highs & Lows (摆动高低点)
  - BOS / CHoCH (结构突破 / 角色转换)
  - Order Blocks (订单块)
  - Liquidity (流动性)
  - Previous High/Low (前期高低点，仅限日线+时间索引数据)
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
from pandas import DataFrame
from loguru import logger

os.environ.setdefault("SMC_CREDIT", "0")  # 隐藏库的版权打印

from smartmoneyconcepts import smc  # noqa: E402


# ═════════════════════════════════════════════════════════════════════════════
# OB 因果适配器 — 修正上游 swing look-ahead 问题
# ═════════════════════════════════════════════════════════════════════════════
#
# 上游 smc.ob() 的问题：
#   swing_highs_lows() 使用未来 K 线确认摆动点（前后各 swing_length 根），
#   结果是"含未来信息"的。而 ob() 在逐 bar 遍历时直接引用这些未来已知的
#   swing 点，导致尚未被确认的 swing HIGH 会抢先遮蔽合法的 OB 突破。
#
# 修复策略：
#   自行实现 OB 检测逻辑，处理 bar i 时只考虑 bar k + swing_length ≤ i
#   的 swing 点（即在 bar i 时已有足够后续 K 线确认的 swing）。
#   检测逻辑与上游完全一致，仅增加可见性过滤。
# ═════════════════════════════════════════════════════════════════════════════


def _ob_causal(
    ohlc: DataFrame,
    swing_highs_lows: DataFrame,
    swing_length: int = 5,
    close_mitigation: bool = False,
) -> DataFrame:
    """因果修正版 OB 检测。

    与上游 smc.ob() 逻辑完全一致，唯一区别：
    处理 bar i 时，只有 ``k + swing_length <= i`` 的 swing 点才可见，
    模拟实时场景下的信息可见范围。

    参数与返回值格式均与 smc.ob() 相同。
    """
    ohlc_len = len(ohlc)
    _open = ohlc["open"].values
    _high = ohlc["high"].values
    _low = ohlc["low"].values
    _close = ohlc["close"].values
    _volume = ohlc["volume"].values
    swing_hl = swing_highs_lows["HighLow"].values

    # 预分配
    crossed = np.full(ohlc_len, False, dtype=bool)
    ob = np.zeros(ohlc_len, dtype=np.int32)
    top_arr = np.zeros(ohlc_len, dtype=np.float64)
    bottom_arr = np.zeros(ohlc_len, dtype=np.float64)
    obVolume = np.zeros(ohlc_len, dtype=np.float64)
    lowVolume = np.zeros(ohlc_len, dtype=np.float64)
    highVolume = np.zeros(ohlc_len, dtype=np.float64)
    percentage = np.zeros(ohlc_len, dtype=np.float64)
    mitigated_index = np.zeros(ohlc_len, dtype=np.int32)
    breaker = np.full(ohlc_len, False, dtype=bool)

    # 全量 swing 索引（与上游一致）
    all_swing_high_indices = np.flatnonzero(swing_hl == 1)
    all_swing_low_indices = np.flatnonzero(swing_hl == -1)

    # ── 看涨 OB ─────────────────────────────────────────────────────────────
    active_bullish: list[int] = []
    for i in range(ohlc_len):
        # 维护已有的看涨 OB（消除 / breaker 清除）
        for idx in active_bullish.copy():
            if breaker[idx]:
                if _high[i] > top_arr[idx]:
                    ob[idx] = 0
                    top_arr[idx] = 0.0
                    bottom_arr[idx] = 0.0
                    obVolume[idx] = 0.0
                    lowVolume[idx] = 0.0
                    highVolume[idx] = 0.0
                    mitigated_index[idx] = 0
                    percentage[idx] = 0.0
                    active_bullish.remove(idx)
            else:
                if (
                    (not close_mitigation and _low[i] < bottom_arr[idx])
                    or (close_mitigation and min(_open[i], _close[i]) < bottom_arr[idx])
                ):
                    breaker[idx] = True
                    mitigated_index[idx] = i - 1

        # ▸ 因果过滤：只取 k + swing_length <= i 的 swing HIGH
        visible_mask = all_swing_high_indices + swing_length <= i
        visible_highs = all_swing_high_indices[visible_mask]

        # 找 bar i 之前最近的可见 swing HIGH
        pos = np.searchsorted(visible_highs, i)
        last_top_index = visible_highs[pos - 1] if pos > 0 else None

        if last_top_index is not None:
            if _close[i] > _high[last_top_index] and not crossed[last_top_index]:
                crossed[last_top_index] = True
                # 默认：前一根 K 线
                obBtm = _high[i - 1]
                obTop = _low[i - 1]
                obIndex = i - 1
                # 在 swing HIGH 与当前 bar 之间找最低 low 的 K 线
                if i - last_top_index > 1:
                    start = last_top_index + 1
                    end = i
                    if end > start:
                        segment = _low[start:end]
                        min_val = segment.min()
                        candidates = np.nonzero(segment == min_val)[0]
                        if candidates.size:
                            ci = start + candidates[-1]
                            obBtm = _low[ci]
                            obTop = _high[ci]
                            obIndex = ci
                ob[obIndex] = 1
                top_arr[obIndex] = obTop
                bottom_arr[obIndex] = obBtm
                v0 = _volume[i]
                v1 = _volume[i - 1] if i >= 1 else 0.0
                v2 = _volume[i - 2] if i >= 2 else 0.0
                obVolume[obIndex] = v0 + v1 + v2
                lowVolume[obIndex] = v2
                highVolume[obIndex] = v0 + v1
                mx = max(highVolume[obIndex], lowVolume[obIndex])
                percentage[obIndex] = (min(highVolume[obIndex], lowVolume[obIndex]) / mx * 100.0) if mx != 0 else 100.0
                active_bullish.append(obIndex)

    # ── 看跌 OB ─────────────────────────────────────────────────────────────
    active_bearish: list[int] = []
    for i in range(ohlc_len):
        for idx in active_bearish.copy():
            if breaker[idx]:
                if _low[i] < bottom_arr[idx]:
                    ob[idx] = 0
                    top_arr[idx] = 0.0
                    bottom_arr[idx] = 0.0
                    obVolume[idx] = 0.0
                    lowVolume[idx] = 0.0
                    highVolume[idx] = 0.0
                    mitigated_index[idx] = 0
                    percentage[idx] = 0.0
                    active_bearish.remove(idx)
            else:
                if (
                    (not close_mitigation and _high[i] > top_arr[idx])
                    or (close_mitigation and max(_open[i], _close[i]) > top_arr[idx])
                ):
                    breaker[idx] = True
                    mitigated_index[idx] = i

        # ▸ 因果过滤：只取 k + swing_length <= i 的 swing LOW
        visible_mask = all_swing_low_indices + swing_length <= i
        visible_lows = all_swing_low_indices[visible_mask]

        pos = np.searchsorted(visible_lows, i)
        last_btm_index = visible_lows[pos - 1] if pos > 0 else None

        if last_btm_index is not None:
            if _close[i] < _low[last_btm_index] and not crossed[last_btm_index]:
                crossed[last_btm_index] = True
                obTop = _high[i - 1]
                obBtm = _low[i - 1]
                obIndex = i - 1
                if i - last_btm_index > 1:
                    start = last_btm_index + 1
                    end = i
                    if end > start:
                        segment = _high[start:end]
                        max_val = segment.max()
                        candidates = np.nonzero(segment == max_val)[0]
                        if candidates.size:
                            ci = start + candidates[-1]
                            obTop = _high[ci]
                            obBtm = _low[ci]
                            obIndex = ci
                ob[obIndex] = -1
                top_arr[obIndex] = obTop
                bottom_arr[obIndex] = obBtm
                v0 = _volume[i]
                v1 = _volume[i - 1] if i >= 1 else 0.0
                v2 = _volume[i - 2] if i >= 2 else 0.0
                obVolume[obIndex] = v0 + v1 + v2
                lowVolume[obIndex] = v0 + v1
                highVolume[obIndex] = v2
                mx = max(highVolume[obIndex], lowVolume[obIndex])
                percentage[obIndex] = (min(highVolume[obIndex], lowVolume[obIndex]) / mx * 100.0) if mx != 0 else 100.0
                active_bearish.append(obIndex)

    # 转 NaN
    ob = np.where(ob != 0, ob, np.nan)
    top_arr = np.where(~np.isnan(ob), top_arr, np.nan)
    bottom_arr = np.where(~np.isnan(ob), bottom_arr, np.nan)
    obVolume = np.where(~np.isnan(ob), obVolume, np.nan)
    mitigated_index = np.where(~np.isnan(ob), mitigated_index, np.nan)
    percentage = np.where(~np.isnan(ob), percentage, np.nan)

    return pd.concat([
        pd.Series(ob, name="OB"),
        pd.Series(top_arr, name="Top"),
        pd.Series(bottom_arr, name="Bottom"),
        pd.Series(obVolume, name="OBVolume"),
        pd.Series(mitigated_index, name="MitigatedIndex"),
        pd.Series(percentage, name="Percentage"),
    ], axis=1)


# ═════════════════════════════════════════════════════════════════════════════
# OB 质量筛选器 — 基于回测特征分析得出的阈值
# ═════════════════════════════════════════════════════════════════════════════
#
# 回测结论（461 只美股科技股，2023-04 ~ 2026-05）：
#
# 看涨 OB 关键特征:
#   - ob_width_pct ≥ 7.5%  → 97.1% 胜率（基线 92.1%）
#   - trend_before_5d ≤ -7.8% → 96.9%（OB 前有显著下跌）
#   - 两者组合 → 99.0%，保留 15%
#
# 看跌 OB 关键特征:
#   - vol_ratio ≥ 1.15 + percentage ≤ 24.5 → 89.2%（基线 78.0%）
#   - ob_atr_ratio ≥ 1.63 + percentage ≤ 42 → 87.5%
#
# Touch 关键特征:
#   - 看涨: days_to_touch ≤ 9 + trend_before_5d ≤ -7.4 → 96.8%
#   - 看跌: percentage ≤ 25 + days_to_touch ≤ 8 → 75.9%
# ═════════════════════════════════════════════════════════════════════════════

# 默认阈值（可在调用时覆盖）
OB_QUALITY_DEFAULTS = {
    "bull": {
        "ob_width_pct_min": 4.0,      # OB 区间宽度 ≥ 4%
        "trend_before_5d_max": -3.0,   # OB 前5日跌幅 ≤ -3%（有像样的回调）
    },
    "bear": {
        "vol_ratio_min": 1.0,          # OB bar 成交量 ≥ 20日均量
        "ob_atr_ratio_min": 1.0,       # OB 宽度 ≥ 1x ATR
        "percentage_max": 45.0,        # volume split 不能太对称
    },
}

# ── OB 质量评分规则 ─────────────────────────────────────────────────────────
# 将筛选特征转化为连续打分，类似 Vegas 的强弱信号打分。
# 每条规则: (特征名, zero_val, full_val, weight)
#   zero_val → 0 分, full_val → weight 满分, 线性插值, 截断到 [0, weight]
#   权重总和 = 100, 所以评分范围 0~100
# 权重分配依据: 回测中各特征与胜率的相关性强弱
OB_SCORE_RULES: dict[str, list[tuple[str, float, float, int]]] = {
    "bull": [
        # 特征名            0分值    满分值    权重
        ("ob_width_pct",     1.0,     10.0,   30),  # 区间宽 → 需求区厚实
        ("trend_before_5d",  0.0,    -10.0,   25),  # 前5日跌得深 → 买盘介入有意义
        ("vol_ratio",        0.5,      2.5,   15),  # 放量 → 机构参与
        ("ob_atr_ratio",     0.3,      2.0,   15),  # 宽度相对ATR大 → 信号明显
        ("body_ratio",       0.2,      0.8,   10),  # 实体大 → 方向明确
        ("percentage",      50.0,     10.0,    5),  # 成交量不对称 → 单边力量
    ],
    "bear": [
        ("ob_atr_ratio",     0.3,      2.5,   25),  # 宽度相对ATR大 → 供应区显著
        ("vol_ratio",        0.5,      2.5,   20),  # 放量 → 机构参与
        ("percentage",      50.0,     15.0,   20),  # 成交量不对称 → 卖压集中
        ("ob_width_pct",     1.0,      8.0,   15),  # 区间宽 → 供应区厚实
        ("trend_before_5d",  0.0,      8.0,   10),  # 前5日涨得多 → 获利盘堆积
        ("body_ratio",       0.2,      0.8,   10),  # 实体大 → 方向明确
    ],
}


def _linear_score(value: float, zero_val: float, full_val: float) -> float:
    """线性插值: zero_val → 0, full_val → 1, 截断到 [0, 1]。"""
    if full_val == zero_val:
        return 0.5
    return max(0.0, min(1.0, (value - zero_val) / (full_val - zero_val)))


def ob_quality_rating(
    ohlcv: DataFrame,
    ob_df: DataFrame,
    bar_idx: int,
    features: dict | None = None,
) -> tuple[float, dict[str, float]]:
    """计算 OB 质量评分 (0~100) 及各分项得分。

    基于 ob_quality_score 返回的特征，按方向分别加权打分。
    类似 Vegas 的连续打分机制——分数越高代表 OB 越"强"，
    而非二元的"通过/不通过"。

    Returns:
        (total_score, breakdown)
        - total_score: 0~100 综合分
        - breakdown:   {特征名: 该特征得分} — 便于调试和展示
    """
    if features is None:
        features = ob_quality_score(ohlcv, ob_df, bar_idx)

    direction = features["direction"]
    rules = OB_SCORE_RULES.get("bull" if direction == 1 else "bear", [])

    breakdown: dict[str, float] = {}
    total = 0.0
    for feat_name, zero_val, full_val, weight in rules:
        val = features.get(feat_name, 0.0)
        s = _linear_score(val, zero_val, full_val) * weight
        breakdown[feat_name] = round(s, 1)
        total += s

    return round(total, 1), breakdown


def ob_quality_score(
    ohlcv: DataFrame,
    ob_df: DataFrame,
    bar_idx: int,
) -> dict:
    """计算单个 OB 的质量特征。

    返回字典包含:
        direction, ob_width_pct, body_ratio, vol_ratio,
        trend_before_5d, ob_atr_ratio, percentage
    """
    row = ob_df.iloc[bar_idx]
    direction = int(row["OB"])
    top = float(row["Top"])
    bottom = float(row["Bottom"])
    percentage_val = float(row["Percentage"]) if pd.notna(row.get("Percentage")) else 50.0

    mid = (top + bottom) / 2
    ob_width_pct = (top - bottom) / mid * 100 if mid > 0 else 0

    o = ohlcv["open"].iloc[bar_idx]
    h = ohlcv["high"].iloc[bar_idx]
    low = ohlcv["low"].iloc[bar_idx]
    c = ohlcv["close"].iloc[bar_idx]
    body = abs(c - o)
    rng = h - low
    body_ratio = body / rng if rng > 0 else 0

    vol = ohlcv["volume"].iloc[bar_idx]
    start = max(0, bar_idx - 20)
    avg_vol = ohlcv["volume"].iloc[start:bar_idx].mean() if bar_idx > 0 else vol
    vol_ratio = vol / avg_vol if avg_vol > 0 else 1.0

    lb = 5
    if bar_idx >= lb:
        trend_before_5d = (c - ohlcv["close"].iloc[bar_idx - lb]) / ohlcv["close"].iloc[bar_idx - lb] * 100
    else:
        trend_before_5d = 0.0

    if bar_idx >= 15:
        hi = ohlcv["high"].values[bar_idx-14:bar_idx]
        lo = ohlcv["low"].values[bar_idx-14:bar_idx]
        prev_c = ohlcv["close"].values[bar_idx-15:bar_idx-1]
        tr = np.maximum(hi - lo, np.maximum(np.abs(hi - prev_c), np.abs(lo - prev_c)))
        atr14 = float(tr.mean())
    else:
        atr14 = rng if rng > 0 else 1.0

    ob_atr_ratio = (top - bottom) / atr14 if atr14 > 0 else 0

    return {
        "direction": direction,
        "ob_width_pct": ob_width_pct,
        "body_ratio": body_ratio,
        "vol_ratio": vol_ratio,
        "trend_before_5d": trend_before_5d,
        "ob_atr_ratio": ob_atr_ratio,
        "percentage": percentage_val,
    }


def ob_passes_quality(
    ohlcv: DataFrame,
    ob_df: DataFrame,
    bar_idx: int,
    thresholds: dict | None = None,
) -> bool:
    """判断 bar_idx 处的 OB 是否通过质量筛选。

    thresholds 格式同 OB_QUALITY_DEFAULTS，为 None 时使用默认值。
    """
    row = ob_df.iloc[bar_idx]
    if pd.isna(row.get("OB")):
        return False

    feat = ob_quality_score(ohlcv, ob_df, bar_idx)
    direction = feat["direction"]
    cfg = (thresholds or OB_QUALITY_DEFAULTS).get(
        "bull" if direction == 1 else "bear", {}
    )

    # 逐条检查
    if "ob_width_pct_min" in cfg and feat["ob_width_pct"] < cfg["ob_width_pct_min"]:
        return False
    if "ob_width_pct_max" in cfg and feat["ob_width_pct"] > cfg["ob_width_pct_max"]:
        return False
    if "trend_before_5d_max" in cfg and feat["trend_before_5d"] > cfg["trend_before_5d_max"]:
        return False
    if "trend_before_5d_min" in cfg and feat["trend_before_5d"] < cfg["trend_before_5d_min"]:
        return False
    if "vol_ratio_min" in cfg and feat["vol_ratio"] < cfg["vol_ratio_min"]:
        return False
    if "ob_atr_ratio_min" in cfg and feat["ob_atr_ratio"] < cfg["ob_atr_ratio_min"]:
        return False
    if "ob_atr_ratio_max" in cfg and feat["ob_atr_ratio"] > cfg["ob_atr_ratio_max"]:
        return False
    if "percentage_max" in cfg and feat["percentage"] > cfg["percentage_max"]:
        return False
    if "percentage_min" in cfg and feat["percentage"] < cfg["percentage_min"]:
        return False
    if "body_ratio_min" in cfg and feat["body_ratio"] < cfg["body_ratio_min"]:
        return False

    return True


# ─────────────────────── 公共入口 ───────────────────────


def compute_smc_full(
    ohlcv: DataFrame,
    swing_length: int = 5,
    close_break: bool = True,
    close_mitigation: bool = False,
    join_consecutive_fvg: bool = False,
) -> dict[str, DataFrame]:
    """计算全套 SMC 指标，返回各指标 DataFrame 字典。

    parameters:
        ohlcv:               标准 OHLCV DataFrame（列名小写：open/high/low/close/volume）
        swing_length:        摆动高低点回看/向前 K 线根数（默认 5）
        close_break:         BOS/CHoCH 基于收盘价判断突破（True）还是高低点（False）
        close_mitigation:    Order Block 基于收盘价判断消除（True）还是高低点（False）
        join_consecutive_fvg:合并连续同方向 FVG

    returns: dict with keys:
        "swing_hl"     : HighLow / Level
        "fvg"          : FVG / Top / Bottom / MitigatedIndex
        "bos_choch"    : BOS / CHOCH / Level / BrokenIndex
        "ob"           : OB / Top / Bottom / OBVolume / MitigatedIndex / Percentage
        "liquidity"    : Liquidity / Level / End / Swept
    """
    # 确保列名小写
    df = ohlcv.rename(columns={c: c.lower() for c in ohlcv.columns})

    swing_hl = smc.swing_highs_lows(df, swing_length=swing_length)
    fvg = smc.fvg(df, join_consecutive=join_consecutive_fvg)
    bos_choch = smc.bos_choch(df, swing_hl, close_break=close_break)
    ob = _ob_causal(df, swing_hl, swing_length=swing_length,
                    close_mitigation=close_mitigation)
    liquidity = smc.liquidity(df, swing_hl)

    return {
        "swing_hl": swing_hl,
        "fvg": fvg,
        "bos_choch": bos_choch,
        "ob": ob,
        "liquidity": liquidity,
    }


def compute_fvg(ohlcv: DataFrame, join_consecutive: bool = False) -> DataFrame:
    """单独计算公允价值缺口（FVG）。"""
    df = ohlcv.rename(columns={c: c.lower() for c in ohlcv.columns})
    return smc.fvg(df, join_consecutive=join_consecutive)


def compute_swing_hl(ohlcv: DataFrame, swing_length: int = 5) -> DataFrame:
    """单独计算摆动高低点。"""
    df = ohlcv.rename(columns={c: c.lower() for c in ohlcv.columns})
    return smc.swing_highs_lows(df, swing_length=swing_length)


def compute_bos_choch(
    ohlcv: DataFrame,
    swing_hl: DataFrame | None = None,
    swing_length: int = 5,
    close_break: bool = True,
) -> DataFrame:
    """计算 BOS / CHoCH（需要摆动高低点）。"""
    df = ohlcv.rename(columns={c: c.lower() for c in ohlcv.columns})
    if swing_hl is None:
        swing_hl = smc.swing_highs_lows(df, swing_length=swing_length)
    return smc.bos_choch(df, swing_hl, close_break=close_break)


def compute_ob(
    ohlcv: DataFrame,
    swing_hl: DataFrame | None = None,
    swing_length: int = 5,
    close_mitigation: bool = False,
) -> DataFrame:
    """计算订单块（Order Block）— 使用因果修正版检测。"""
    df = ohlcv.rename(columns={c: c.lower() for c in ohlcv.columns})
    if swing_hl is None:
        swing_hl = smc.swing_highs_lows(df, swing_length=swing_length)
    return _ob_causal(df, swing_hl, swing_length=swing_length,
                      close_mitigation=close_mitigation)


def compute_liquidity(
    ohlcv: DataFrame,
    swing_hl: DataFrame | None = None,
    swing_length: int = 5,
    range_percent: float = 0.01,
) -> DataFrame:
    """计算流动性聚集区（Liquidity）。"""
    df = ohlcv.rename(columns={c: c.lower() for c in ohlcv.columns})
    if swing_hl is None:
        swing_hl = smc.swing_highs_lows(df, swing_length=swing_length)
    return smc.liquidity(df, swing_hl, range_percent=range_percent)


# ─────────────────────── 便捷摘要工具 ───────────────────────


def get_recent_fvg(ohlcv: DataFrame, n_recent: int = 5) -> DataFrame:
    """返回最近 n_recent 个未被消除的 FVG（Top/Bottom/方向）。"""
    fvg_df = compute_fvg(ohlcv)
    # 未被消除的 FVG：MitigatedIndex == 0
    active = fvg_df[fvg_df["FVG"].notna() & (fvg_df["MitigatedIndex"] == 0)].copy()
    active.index = ohlcv.index[active.index] if not ohlcv.index.equals(fvg_df.index) else active.index
    return active.tail(n_recent)


def get_active_ob(ohlcv: DataFrame, n_recent: int = 5) -> DataFrame:
    """返回最近 n_recent 个未被消除的订单块。"""
    ob_df = compute_ob(ohlcv)
    # MitigatedIndex 为 0 表示仍有效
    active = ob_df[ob_df["OB"].notna() & (ob_df["MitigatedIndex"] == 0)].copy()
    return active.tail(n_recent)


def get_latest_structure(ohlcv: DataFrame) -> dict:
    """返回最新的市场结构摘要（最后一个 BOS / CHoCH 信号）。

    returns dict:
        last_bos:   最近 BOS 方向（1=看涨, -1=看跌, None=无）
        last_choch: 最近 CHoCH 方向（同上）
        last_level: 对应价格水平
        last_date:  对应 K 线日期索引
    """
    bos_df = compute_bos_choch(ohlcv)
    result: dict = {"last_bos": None, "last_choch": None, "last_level": None, "last_date": None}

    bos_rows = bos_df[bos_df["BOS"].notna() & (bos_df["BOS"] != 0)]
    if not bos_rows.empty:
        last = bos_rows.iloc[-1]
        idx = bos_rows.index[-1]
        result["last_bos"] = int(last["BOS"])
        result["last_level"] = float(last["Level"])
        result["last_date"] = ohlcv.index[idx] if idx < len(ohlcv) else None

    choch_rows = bos_df[bos_df["CHOCH"].notna() & (bos_df["CHOCH"] != 0)]
    if not choch_rows.empty:
        last_c = choch_rows.iloc[-1]
        idx_c = choch_rows.index[-1]
        result["last_choch"] = int(last_c["CHOCH"])
        if result["last_bos"] is None:
            result["last_level"] = float(last_c["Level"])
            result["last_date"] = ohlcv.index[idx_c] if idx_c < len(ohlcv) else None

    return result
