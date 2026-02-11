"""
VCP V7 — ZigZag 多波收缩检测器

V7 核心改进：
  1. ZigZag 拐点检测替代简单 high/low 搜索，识别多波结构
  2. 严格左侧锚点验证：必须是显著局部高点（非中继/平台延续）
  3. 单波谷规则：右侧 ≥ 左侧（不接受"半杯"形态）
  4. 长周期深幅杯柄特例：>6 个月 + 多波浪允许右侧低 10-15%
  5. 底部通道高度 ≤ 总振幅 30%（anti-V 增强）
  6. ZigZag 多波收缩验证：波浪幅度逐级递减（VCP 本质）
  7. 聚焦中期形态（基底 ≥ 60 交易日 ≈ 3 个月）
"""

import numpy as np
import pandas as pd
from loguru import logger
from stock_ana.data_fetcher import load_all_ndx100_data


# =============================================================================
#  ZigZag 拐点检测
# =============================================================================

def _zigzag_pivots(
    highs: np.ndarray,
    lows: np.ndarray,
    threshold_pct: float = 5.0,
) -> list[dict]:
    """
    ZigZag 拐点检测：识别价格序列中交替出现的显著高低拐点。

    当价格从候选高点回落 ≥ threshold_pct% 时确认该高点；
    当价格从候选低点反弹 ≥ threshold_pct% 时确认该低点。

    Args:
        highs: 每根 K 线的最高价
        lows:  每根 K 线的最低价
        threshold_pct: 反转幅度阈值（%），用于过滤噪声

    Returns:
        交替出现的 H/L 拐点: [{"type":"H"|"L", "iloc":int, "value":float}, ...]
    """
    n = len(highs)
    if n < 5:
        return []

    pivots: list[dict] = []
    trend = 0  # 0=初始化, 1=上行寻高, -1=下行寻低

    ch_i, ch_v = 0, highs[0]    # candidate high
    cl_i, cl_v = 0, lows[0]     # candidate low

    for i in range(1, n):
        if trend == 0:
            if highs[i] > ch_v:
                ch_i, ch_v = i, highs[i]
            if lows[i] < cl_v:
                cl_i, cl_v = i, lows[i]
            if ch_v > 0 and (ch_v - lows[i]) / ch_v * 100 >= threshold_pct:
                pivots.append({"type": "H", "iloc": ch_i, "value": float(ch_v)})
                trend = -1
                cl_i, cl_v = i, lows[i]
            elif cl_v > 0 and (highs[i] - cl_v) / cl_v * 100 >= threshold_pct:
                pivots.append({"type": "L", "iloc": cl_i, "value": float(cl_v)})
                trend = 1
                ch_i, ch_v = i, highs[i]

        elif trend == 1:  # 上行寻高
            if highs[i] > ch_v:
                ch_i, ch_v = i, highs[i]
            if ch_v > 0 and (ch_v - lows[i]) / ch_v * 100 >= threshold_pct:
                pivots.append({"type": "H", "iloc": ch_i, "value": float(ch_v)})
                trend = -1
                cl_i, cl_v = i, lows[i]

        elif trend == -1:  # 下行寻低
            if lows[i] < cl_v:
                cl_i, cl_v = i, lows[i]
            if cl_v > 0 and (highs[i] - cl_v) / cl_v * 100 >= threshold_pct:
                pivots.append({"type": "L", "iloc": cl_i, "value": float(cl_v)})
                trend = 1
                ch_i, ch_v = i, highs[i]

    return pivots


# =============================================================================
#  左侧锚点验证
# =============================================================================

def _is_local_high(highs: np.ndarray, iloc: int, order: int = 15) -> bool:
    """
    验证 iloc 位置是否为显著局部高点（非中继/平台延续点）。

    条件：
      1. 左右各至少 min(order, 5) 根 K 线
      2. 该点是 [iloc-order, iloc+order] 窗口内的最高价
      3. 左侧窗口最低点到该点至少上涨 3%（排除平台中继）
      4. 右侧窗口最低点到该点至少下跌 3%（确认为"顶"）
    """
    n = len(highs)
    pivot_val = highs[iloc]
    left_start = max(0, iloc - order)
    right_end = min(n, iloc + order + 1)

    if iloc - left_start < min(order, 5):
        return False
    if right_end - iloc - 1 < min(order, 5):
        return False

    # 窗口内最高
    if pivot_val < np.max(highs[left_start:right_end]):
        return False

    # 左侧有明显上涨
    left_min = np.min(highs[left_start:iloc])
    if pivot_val > 0 and (pivot_val - left_min) / pivot_val * 100 < 3.0:
        return False

    # 右侧有明显下跌
    right_min = np.min(highs[iloc + 1:right_end])
    if pivot_val > 0 and (pivot_val - right_min) / pivot_val * 100 < 3.0:
        return False

    return True


# =============================================================================
#  辅助分析函数
# =============================================================================

def _poly_smoothness(closes: np.ndarray, degree: int = 3) -> float:
    """多项式拟合 R²：衡量基底整体形状平滑度。"""
    n = len(closes)
    if n < 10:
        return 0.0
    x = np.arange(n, dtype=float)
    coeffs = np.polyfit(x, closes, degree)
    fitted = np.polyval(coeffs, x)
    ss_res = np.sum((closes - fitted) ** 2)
    ss_tot = np.sum((closes - closes.mean()) ** 2)
    if ss_tot == 0:
        return 1.0
    return float(1.0 - ss_res / ss_tot)


def _bottom_channel_ratio(
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    low_val: float,
    amplitude: float,
) -> tuple[float, float]:
    """
    基于价格的底部通道分析（anti-V 检测）。

    1. 找出所有 close 在 [low_val, low_val + amplitude×0.40] 内的 K 线（底部区域）
    2. 计算底部区域的 high-low 极差 / 总振幅 = 通道比
    3. 同时返回底部停留比（anti-V：底部天数占比）

    Returns:
        (channel_ratio, dwell_ratio)
    """
    if amplitude <= 0 or len(closes) == 0:
        return 1.0, 0.0
    bottom_threshold = low_val + amplitude * 0.30
    mask = closes <= bottom_threshold
    dwell = float(np.sum(mask)) / len(closes)
    if not np.any(mask):
        return 0.0, 0.0
    ch_height = float(highs[mask].max() - lows[mask].min())
    return ch_height / amplitude, dwell


# =============================================================================
#  ZigZag 多波分析
# =============================================================================

def _analyze_base_waves(
    base_highs: np.ndarray,
    base_lows: np.ndarray,
    depth_pct: float,
) -> dict:
    """
    对基底区间执行 ZigZag 分析，提取多波结构信息。

    自适应阈值 = clamp(depth_pct × 0.20, 2.5, 6.0)

    Returns:
        pivots         : 完整拐点列表
        trough_count   : 波谷（ZigZag Low）数量
        peak_count     : 波峰（ZigZag High）数量
        wave_depths    : 每段 H→L 跌幅 %
        is_contracting : 波浪振幅是否逐级递减
        waves_detail   : chart.py 兼容格式（含 high_idx/low_idx/high_val/low_val）
    """
    threshold = max(3.5, min(depth_pct * 0.25, 6.0))
    pivots = _zigzag_pivots(base_highs, base_lows, threshold)

    troughs = [p for p in pivots if p["type"] == "L"]
    peaks = [p for p in pivots if p["type"] == "H"]

    waves_detail: list[dict] = []
    wave_depths: list[float] = []

    for i in range(len(pivots) - 1):
        if pivots[i]["type"] == "H" and pivots[i + 1]["type"] == "L":
            h_v, l_v = pivots[i]["value"], pivots[i + 1]["value"]
            depth = (h_v - l_v) / h_v * 100 if h_v > 0 else 0
            waves_detail.append({
                "high_idx": pivots[i]["iloc"],
                "low_idx": pivots[i + 1]["iloc"],
                "high_val": float(h_v),
                "low_val": float(l_v),
            })
            wave_depths.append(round(depth, 2))

    # ── 收缩判定 ──
    is_contracting = False
    if len(wave_depths) >= 2:
        # 严格递减
        strict = all(
            wave_depths[j] > wave_depths[j + 1]
            for j in range(len(wave_depths) - 1)
        )
        # 宽松递减：每波 < 前波 × 1.10 且末波 < 首波 × 0.80
        relaxed = (
            all(
                wave_depths[j + 1] < wave_depths[j] * 1.10
                for j in range(len(wave_depths) - 1)
            )
            and wave_depths[-1] < wave_depths[0] * 0.80
        )
        is_contracting = strict or relaxed

    return {
        "pivots": pivots,
        "trough_count": len(troughs),
        "peak_count": len(peaks),
        "wave_depths": wave_depths,
        "is_contracting": is_contracting,
        "waves_detail": waves_detail,
    }


# =============================================================================
#  V7 主函数
# =============================================================================

def screen_vcp(
    df: pd.DataFrame,
    min_base_days: int = 60,
    max_base_days: int = 300,
) -> dict | None:
    """
    VCP V7 — ZigZag 多波收缩检测。

    核心逻辑：
      1. ZigZag 拐点检测：识别基底多波结构
      2. 左侧锚点必须是显著局部高点（非中继）
      3. 单波谷：右侧 ≥ 左侧（anti-half-cup）
      4. 长周期(≥6月) + 多波浪：允许右侧低 10-15%（深幅杯柄特例）
      5. 底部通道 ≤ 总振幅 30%（anti-V）
      6. 多波收缩验证（VCP 本质：波浪振幅逐级递减）
      7. 聚焦中期形态（≥60 交易日）
    """

    # ── 0. 数据充分性 ──
    if len(df) < 300:
        return None

    # ── 1. Stage 2 趋势模板 ──
    curr_close = df["close"].iloc[-1]
    ma_50 = df["close"].rolling(50).mean().iloc[-1]
    ma_200 = df["close"].rolling(200).mean().iloc[-1]
    ma_200_prev = df["close"].rolling(200).mean().iloc[-22]

    if not (curr_close > ma_50 and ma_50 > ma_200):
        return None
    if ma_200 <= ma_200_prev:
        return None

    # ── 2. 定位左侧高点（基底起点） ──
    search_start = max(len(df) - max_base_days, 0)
    search_end = len(df) - 15
    lookback = df.iloc[search_start:search_end]
    if len(lookback) < min_base_days:
        return None

    left_idx = lookback["high"].idxmax()
    left_val = lookback.loc[left_idx, "high"]
    left_iloc = df.index.get_loc(left_idx)

    # ── 3. ★ 左侧锚点必须是显著局部高点 ★ ──
    all_highs = df["high"].values.astype(float)
    if not _is_local_high(all_highs, left_iloc, order=15):
        return None

    # ── 4. 定义基底区间 ──
    base_region = df.iloc[left_iloc:]
    base_len = len(base_region)
    if base_len < min_base_days:
        return None

    # ── 5. 定位基底最低点 ──
    low_idx = base_region["low"].idxmin()
    low_val = base_region.loc[low_idx, "low"]
    low_iloc_in_base = base_region.index.get_loc(low_idx)

    # 低点位置约束（10%-80%）
    low_pos = low_iloc_in_base / base_len if base_len > 0 else 0
    if not (0.10 <= low_pos <= 0.80):
        return None

    # ── 6. 深度范围 ──
    depth_pct = (left_val - low_val) / left_val * 100
    amplitude = left_val - low_val
    if depth_pct < 8.0 or depth_pct > 45.0:
        return None

    # ── 7. ★ ZigZag 多波分析 ★ ──
    base_highs = base_region["high"].values.astype(float)
    base_lows = base_region["low"].values.astype(float)
    wave = _analyze_base_waves(base_highs, base_lows, depth_pct)

    trough_count = wave["trough_count"]
    if trough_count == 0:
        return None  # ZigZag 未检测到任何波谷

    # ── 8. ★ 形态规则 ★ ──
    reject: list[str] = []

    # --- 8a. 单波谷：右侧 ≥ 左侧（不接受半杯） ---
    if trough_count == 1:
        if curr_close < left_val * 0.98:
            reject.append(
                f"单波谷:右侧{curr_close:.1f}<左侧98%({left_val * 0.98:.1f})"
            )
        pattern_type = "Flat Base" if depth_pct <= 15 else "Cup with Handle"

    # --- 8b. 多波谷 + 长周期 (≥120d ≈ 6 个月) ---
    #     允许右侧低于左侧 10-15%，必须波浪收缩
    elif trough_count >= 2 and base_len >= 120:
        if curr_close < left_val * 0.85:
            reject.append(
                f"深幅杯:右侧{curr_close:.1f}<左侧85%({left_val * 0.85:.1f})"
            )
        if not wave["is_contracting"]:
            reject.append("多波浪未呈收缩（VCP 要求波幅递减）")
        pattern_type = "Cup with Handle"

    # --- 8c. 多波谷 + 短周期：右侧仍需接近左侧 ---
    else:
        if curr_close < left_val * 0.95:
            reject.append(
                f"短周期多波:右侧{curr_close:.1f}<左侧95%({left_val * 0.95:.1f})"
            )
        pattern_type = "Flat Base" if depth_pct <= 15 else "Cup with Handle"

    # ── 9. ★ 底部通道 + 停留时间（anti-V）★ ──
    base_closes = base_region["close"].values.astype(float)
    ch_ratio, dwell = _bottom_channel_ratio(
        base_closes, base_highs, base_lows, low_val, amplitude,
    )
    max_ch = 0.50 if base_len >= 120 else 0.40
    if ch_ratio > max_ch:
        reject.append(f"底部通道={ch_ratio:.0%}>{max_ch:.0%}")
    if dwell < 0.08:
        reject.append(f"底部停留={dwell:.0%}<8%(V形)")

    # ── 10. R² 平滑度 ──
    r_sq = _poly_smoothness(base_closes, degree=3)
    min_r2 = 0.35 if base_len >= 150 else (0.40 if base_len >= 100 else 0.45)
    if r_sq < min_r2:
        reject.append(f"R²={r_sq:.2f}<{min_r2}")

    # ── 拒绝汇总 ──
    if reject:
        logger.debug(
            f"V7 reject: {', '.join(reject)} | "
            f"base={base_len}d depth={depth_pct:.1f}% "
            f"troughs={trough_count} channel={ch_ratio:.2f} R²={r_sq:.3f}"
        )
        return None

    # ── 11. 手柄检查 ──
    handle_days = max(5, min(30, base_len // 6))
    recent = df.iloc[-handle_days:]
    handle_high = recent["high"].max()
    handle_low = recent["low"].min()
    handle_depth = (handle_high - handle_low) / handle_high * 100

    max_handle = 8.0 if depth_pct > 25 else 6.0
    if handle_depth > max_handle:
        return None

    # 手柄位置
    min_h_low = left_val * 0.85 if (trough_count >= 2 and base_len >= 120) else left_val * 0.90
    if handle_low < min_h_low:
        return None

    # 手柄缩量
    vol_50 = df["volume"].rolling(50).mean().iloc[-1]
    vol_handle = recent["volume"].mean()
    vol_ratio = vol_handle / vol_50 if vol_50 > 0 else 1.0
    if vol_ratio > 1.20:
        return None

    # 手柄日振幅
    h_spread = ((recent["high"] - recent["low"]) / recent["open"]).mean() * 100
    if h_spread > 2.5:
        return None

    # ── 12. 构造返回结果 ──
    depths = wave["wave_depths"][:] if wave["wave_depths"] else [round(depth_pct, 2)]
    depths.append(round(handle_depth, 2))

    dist_to_pivot = (left_val - curr_close) / left_val * 100

    return {
        "pattern": pattern_type,
        "base_days": base_len,
        "base_high": float(left_val),
        "base_depth_pct": round(depth_pct, 2),
        "depths": depths,
        "wave_count": trough_count + 1,
        "num_contractions": len(wave["wave_depths"]),
        "waves": wave["waves_detail"],
        "tightness": round(handle_depth, 2),
        "vol_ratio": round(vol_ratio, 2),
        "distance_to_pivot_pct": round(dist_to_pivot, 2),
        # V7 诊断
        "handle_days": handle_days,
        "r_squared": round(r_sq, 3),
        "trough_count": trough_count,
        "channel_ratio": round(ch_ratio, 3),
        "is_contracting": wave["is_contracting"],
        "zigzag_pivots": len(wave["pivots"]),
    }


# =============================================================================
#  扫描入口
# =============================================================================

def scan_ndx100_vcp(
    min_base_days: int = 60,
    max_base_days: int = 300,
) -> list[dict]:
    """扫描纳指 100 中呈现 VCP / 杯柄 / 平底形态的股票（V7）。"""

    stock_data = load_all_ndx100_data()
    if not stock_data:
        logger.error("本地无数据！请先运行 update_ndx100_data() 下载数据")
        return []

    hits: list[dict] = []
    processed = 0

    _PATTERN_CN = {
        "Flat Base": "平底形态",
        "Cup with Handle": "杯柄形态",
    }

    for ticker, df in stock_data.items():
        try:
            if len(df) < 300:
                continue
            processed += 1
            result = screen_vcp(df, min_base_days, max_base_days)
            if result is not None:
                ptype = _PATTERN_CN.get(result["pattern"], result["pattern"])
                depths_s = "→".join(f"{d:.0f}%" for d in result["depths"])
                logger.success(
                    f"✅ {ticker} {ptype} "
                    f"| 基底:{result['base_days']}d(深{result['base_depth_pct']:.1f}%) "
                    f"| 波谷:{result['trough_count']} "
                    f"收缩:{'是' if result['is_contracting'] else '否'} "
                    f"| 收缩序列:{depths_s} "
                    f"| 底部通道:{result['channel_ratio']:.0%} "
                    f"| R²={result['r_squared']:.2f} "
                    f"| 量缩:{result['vol_ratio']:.0%}"
                )
                hits.append({"ticker": ticker, "df": df, "vcp_info": result})
        except Exception as e:
            logger.error(f"{ticker}: VCP V7 检测失败 - {e}")
            continue

    logger.info(
        f"VCP V7 扫描完成：{len(stock_data)} 只 → "
        f"有效 {processed} 只 → 命中 {len(hits)} 只"
    )
    return hits
