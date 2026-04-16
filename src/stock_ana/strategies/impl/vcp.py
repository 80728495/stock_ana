"""
VCP V8 — 显式杯身 P1/P2/P3 + ZigZag 多波收缩检测器

V8 核心改进（基于 V7 重构）：
  1. 均线趋势过滤：四条规则（绝对位置/排列/SMA200斜率/前期涨幅≥30%）
  2. 显式杯身结构识别：ZigZag H→L→H 三点匹配（P1-P2-P3）
     · 对称性 ±5%、杯深 15-33%、U型验证、7-65 周跨度
  3. 跨波搜索：枚举所有 H-L-H 组合找最优杯底，而非仅连续三元组
  4. 底部通道 + R² 平滑度质量过滤
  5. 手柄检查（深度/位置/缩量/日振幅）
"""

import pandas as pd
from loguru import logger
from stock_ana.data.market_data import load_market_data
from stock_ana.strategies.primitives import (
    bottom_channel_ratio,
    check_cup_ma_trend,
    find_cup_structure,
    poly_smoothness,
)


# =============================================================================
#  V7 主函数
# =============================================================================

def screen_vcp(
    df: pd.DataFrame,
    min_base_days: int = 60,
    max_base_days: int = 300,
    loose: bool = False,
) -> dict | None:
    """
    VCP V8 — 显式 P1/P2/P3 杯身 + ZigZag 多波收缩检测。

    核心逻辑：
      1. 均线趋势过滤（四条规则）
    2. ★ 杯身结构识别（find_cup_structure）★
           P1 左杯沿 → P2 杯底 → P3 右杯沿
           · 对称性：|P3-P1|/P1 ≤ 5%
           · 杯深：15%–33%，U 形（非 V 形）
           · 时间：7–65 周
      3. ZigZag 多波收缩验证（杯身内波幅递减）
      4. 手柄检查

    Args:
        loose: 宽松模式 — 手柄深度上限放宽至 12%/15%
    """

    # ── 0. 数据充分性（需覆盖 200MA + 至少 7 周杯身） ──
    if len(df) < 252:
        return None

    # ── 1. 预定位搜索窗口，并先做均线过滤 ──
    #   搜索范围：[now - max_base_days, now - 15]，杯身右沿至少在 15 root 前
    search_start = max(len(df) - max_base_days, 0)
    search_end   = len(df) - 15

    if search_end - search_start < min_base_days:
        return None

    # 用搜索窗口内的最高点做初步均线过滤（快速剪枝）
    lookback = df.iloc[search_start:search_end]
    left_idx  = lookback["high"].idxmax()
    left_val  = float(lookback.loc[left_idx, "high"])
    left_iloc = df.index.get_loc(left_idx)

    # ── 2. 初步均线过滤（用窗口最高价快速剪枝）──
    passed, reason = check_cup_ma_trend(df, left_val, left_iloc)
    if not passed:
        logger.debug(f"MA trend reject (prelim): {reason}")
        return None

    # ── 3. ★ 杯身结构识别（P1→P2→P3）★ ──
    cup = find_cup_structure(
        df,
        search_start_iloc=search_start,
        search_end_iloc=search_end,
        data_end_iloc=len(df) - 1,
        symmetry_tol=0.05,
        min_depth_pct=15.0,
        max_depth_pct=33.0,
        min_weeks=7,
        max_weeks=65,
    )
    if cup is None:
        logger.debug("Cup structure not found (P1/P2/P3 conditions failed)")
        return None

    p1_iloc = cup["p1_iloc"]
    p1_val  = cup["p1_val"]
    p2_iloc = cup["p2_iloc"]
    p2_val  = cup["p2_val"]
    p3_iloc = cup["p3_iloc"]
    p3_val  = cup["p3_val"]
    depth_pct  = cup["depth_pct"]
    cup_days   = cup["cup_days"]
    amplitude  = p1_val - p2_val
    wave       = cup["wave_info"]

    # ── 3b. ★ 用实际 P1 重新验证均线条件 ★ ──
    #   初步过滤用了窗口最高价（可能 ≠ P1），需用真实 P1 再验证一次
    if p1_iloc != left_iloc:
        passed2, reason2 = check_cup_ma_trend(df, p1_val, p1_iloc)
        if not passed2:
            logger.debug(f"MA trend reject (P1 recheck): {reason2}")
            return None

    # ── 4. ZigZag 多波收缩验证（在杯身区间内） ──
    trough_count = wave["trough_count"]
    if trough_count == 0:
        logger.debug("Cup: ZigZag no trough found in cup region")
        return None

    # 右沿接近程度（P3 相对 P1 的偏差）
    sym_pct = cup["symmetry_pct"]  # 负=右低，正=右高

    # ── 5. 底部通道 + 停留时间（anti-V 二次确认） ──
    cup_region   = df.iloc[p1_iloc : p3_iloc + 1]
    base_closes  = cup_region["close"].values.astype(float)
    base_highs   = cup_region["high"].values.astype(float)
    base_lows    = cup_region["low"].values.astype(float)

    ch_ratio, dwell = bottom_channel_ratio(
        base_closes, base_highs, base_lows, p2_val, amplitude,
    )
    reject: list[str] = []
    max_ch = 0.55 if cup_days >= 120 else 0.45
    if ch_ratio > max_ch:
        reject.append(f"底部通道={ch_ratio:.0%}>{max_ch:.0%}")
    if dwell < 0.08:
        reject.append(f"底部停留={dwell:.0%}<8%(V形)")

    # ── 6. R² 平滑度 ──
    r_sq = poly_smoothness(base_closes, degree=3)
    min_r2 = 0.35 if cup_days >= 150 else (0.40 if cup_days >= 100 else 0.45)
    if r_sq < min_r2:
        reject.append(f"R²={r_sq:.2f}<{min_r2}")

    if reject:
        logger.debug(
            f"Cup quality reject: {', '.join(reject)} | "
            f"days={cup_days} depth={depth_pct:.1f}% sym={sym_pct:+.1f}%"
        )
        return None

    curr_close   = df["close"].iloc[-1]
    current_iloc = len(df) - 1
    pattern_type = "Cup with Handle"

    # ── 7. 手柄检查（P3 右侧到当前） ──
    handle_days = max(5, min(30, cup_days // 6))
    recent       = df.iloc[-handle_days:]
    handle_high  = recent["high"].max()
    handle_low   = recent["low"].min()
    handle_depth = (handle_high - handle_low) / handle_high * 100

    if loose:
        max_handle = 15.0 if depth_pct > 25 else 12.0
    else:
        max_handle = 8.0 if depth_pct > 25 else 6.0
    if handle_depth > max_handle:
        return None

    # 手柄最低价须高于 P1 的 90%（宽松模式：85%）
    min_h_low = p1_val * (0.85 if loose else 0.90)
    if handle_low < min_h_low:
        return None

    # 手柄缩量
    vol_50     = df["volume"].rolling(50).mean().iloc[-1]
    vol_handle = recent["volume"].mean()
    vol_ratio  = vol_handle / vol_50 if vol_50 > 0 else 1.0
    if vol_ratio > 1.20:
        return None

    # 手柄日振幅
    h_spread = ((recent["high"] - recent["low"]) / recent["open"]).mean() * 100
    if h_spread > 2.5:
        return None

    # ── 8. 构造返回结果 ──
    depths = wave["wave_depths"][:] if wave["wave_depths"] else [round(depth_pct, 2)]
    depths.append(round(handle_depth, 2))

    dist_to_pivot      = (p1_val - curr_close) / p1_val * 100
    prior_start        = max(0, p1_iloc - 252)
    prior_52w_low      = float(df["low"].values[prior_start : p1_iloc + 1].min())
    prior_advance_pct  = ((p1_val - prior_52w_low) / prior_52w_low * 100
                          if prior_52w_low > 0 else 0.0)

    logger.debug(
        f"Cup V8 pass | P1={p1_val:.2f} P2={p2_val:.2f} P3={p3_val:.2f} "
        f"depth={depth_pct:.1f}% sym={sym_pct:+.1f}% days={cup_days} "
        f"dwell={dwell:.0%} R²={r_sq:.2f}"
    )

    return {
        "pattern":               pattern_type,
        # 杯身三点
        "p1_val":                float(p1_val),
        "p2_val":                float(p2_val),
        "p3_val":                float(p3_val),
        "symmetry_pct":          round(sym_pct, 2),
        # 基底参数（向后兼容旧字段名）
        "base_days":             cup_days,
        "base_high":             float(p1_val),
        "base_depth_pct":        round(depth_pct, 2),
        "depths":                depths,
        "wave_count":            trough_count + 1,
        "num_contractions":      len(wave["wave_depths"]),
        "waves":                 wave["waves_detail"],
        "tightness":             round(handle_depth, 2),
        "vol_ratio":             round(vol_ratio, 2),
        "distance_to_pivot_pct": round(dist_to_pivot, 2),
        # 诊断
        "handle_days":           handle_days,
        "r_squared":             round(r_sq, 3),
        "trough_count":          trough_count,
        "channel_ratio":         round(ch_ratio, 3),
        "dwell_ratio":           cup["dwell_ratio"],
        "is_contracting":        wave["is_contracting"],
        "zigzag_pivots":         len(wave["pivots"]),
        "prior_advance_pct":     round(prior_advance_pct, 1),
        # 关键位置索引（图表标注用）
        "base_start_iloc":       p1_iloc,
        "cup_bottom_iloc":       p2_iloc,
        "cup_bottom_val":        float(p2_val),
        "handle_start_iloc":     current_iloc - handle_days + 1,
    }


def scan_us_vcp(
    min_base_days: int = 60,
    max_base_days: int = 300,
    loose: bool = False,
) -> list[dict]:
    """扫描 data/cache/us/ 中全部美股，找到杯柄形态的标的（V8）。

    Args:
        loose: 宽松模式 — 手柄深度上限放宽，适合市场高波动期
    """

    stock_data = load_market_data("us")
    if not stock_data:
        logger.error("本地无数据！请先下载美股数据到 data/cache/us/")
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
            result = screen_vcp(df, min_base_days, max_base_days, loose=loose)
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
            logger.error(f"{ticker}: VCP V8 检测失败 - {e}")
            continue

    logger.info(
        f"US VCP V8 扫描完成：{len(stock_data)} 只 → "
        f"有效 {processed} 只 → 命中 {len(hits)} 只"
    )
    return hits
