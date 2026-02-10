"""
VCP的杯柄和锅底形态识别

1. 经典杯柄形态 (Cup with Handle) —— VCP 的原型
这是最标准、最容易识别的 VCP 形态。

宏观形状： 像一个咖啡杯。

左侧（杯身）： 经历一次明显的下跌（15%-35%），这里是“清洗”过程，把不坚定的筹码洗出去。杯底通常是圆弧状（U形），因为机构吸筹需要时间，不会是尖底（V形）。

右侧（手柄）： 价格回到前高附近后，不再大跌，而是横向微调。这就是 VCP 的核心——波动率收缩。

物理意义： 巨大的抛压（左侧）被消化后，只需微小的回撤（手柄）就能稳住价格。

关键特征： 手柄必须位于杯身上半部分（Upper half），且必须极度惜售（缩量 + 窄幅）。

2. 平底形态 / 碟形 (Flat Base / Saucer) —— 强势的 VCP
这种形态比杯柄更强，因为它意味着回调幅度极浅。

宏观形状： 像一个扁平的盘子或长方形盒子。

特征： 股票在经过一波上涨后，拒绝大幅回调（通常跌幅 < 15%）。价格在一个水平区间内长时间（5-7周以上）横盘震荡。

波动收缩的表现： 在这个扁平箱体内部，你会看到波浪越来越小。

刚开始可能振幅 8%。

中间变成 5%。

最后变成 2% 的一条死线。

物理意义： 机构买盘极其强劲，哪怕在大盘调整时，他们也在 50日均线附近托住了价格，不让它深跌。

同时注意几个原则：
1）杯身要平滑，不能宽幅震荡，否则吸筹不稳
2）一定是从前部高点下跌，当前位置已经回到高点，经历完整周期。

"""

import numpy as np
import pandas as pd
from loguru import logger

from stock_ana.data_fetcher import load_all_ndx100_data
from stock_ana.strategy_base import check_trend_template


# -----------------------------------------------------------------------------
#  辅助函数：平滑度与结构计算 (V5 核心组件)
# -----------------------------------------------------------------------------

def _calculate_smoothness(
    df_base: pd.DataFrame,
    high_val: float,
    low_val: float,
    low_iloc_in_base: int,
) -> dict:
    """
    计算基底的平滑度和几何结构，用于过滤"上蹿下跳"和"V型"形态。

    Args:
        df_base: 从左侧高点到当前的 DataFrame
        high_val: 左侧高点价格
        low_val: 基底最低价
        low_iloc_in_base: 最低点在 df_base 中的 iloc 位置
    """
    n = len(df_base)
    if n < 10:
        return {"is_u_shaped": False, "is_smooth": False, "avg_volatility": 100.0, "bottom_flatness": 0.0}

    # 1. 计算"混乱度" (平均日内振幅)
    # [Fix#2] 排除初始急跌段：只检查从低点前5天开始到末尾的区域
    smooth_start = max(0, low_iloc_in_base - 5)
    smooth_region = df_base.iloc[smooth_start:]
    if len(smooth_region) < 5:
        smooth_region = df_base

    daily_spreads = (smooth_region["high"] - smooth_region["low"]) / smooth_region["open"]
    avg_volatility = daily_spreads.mean() * 100
    is_smooth = avg_volatility < 3.5

    # 2. [Fix#3] U 形结构检测 (拒绝 V 形)
    # 取低点为中心的窗口，计算窗口内价格相对低点的平均偏离
    bottom_window = max(5, n // 8)
    bw_start = max(0, low_iloc_in_base - bottom_window)
    bw_end = min(n, low_iloc_in_base + bottom_window + 1)
    bottom_region = df_base.iloc[bw_start:bw_end]

    amplitude = high_val - low_val
    if amplitude > 0 and len(bottom_region) >= 3:
        deviations = (bottom_region["close"] - low_val) / amplitude
        bottom_flatness = float(deviations.mean())
    else:
        bottom_flatness = 1.0

    low_position_ratio = low_iloc_in_base / n if n > 0 else 0
    position_ok = 0.15 <= low_position_ratio <= 0.85
    is_u_shaped = bottom_flatness < 0.35 and position_ok

    return {
        "is_u_shaped": is_u_shaped,
        "is_smooth": is_smooth,
        "avg_volatility": round(avg_volatility, 2),
        "bottom_flatness": round(bottom_flatness, 2),
    }

# -----------------------------------------------------------------------------
#  核心策略：VCP/杯柄识别 (V5 - 平滑度与区域定位版)
# -----------------------------------------------------------------------------

def screen_vcp(
    df: pd.DataFrame, 
    min_base_days: int = 30,  # 新增参数支持
    max_base_days: int = 250  # 默认放宽到250天以覆盖大周期
) -> dict | None:
    """
    识别 Cup with Handle (杯柄) 和 Flat Base (平底) - V5版
    
    逻辑：
    1. Stage 2 趋势过滤。
    2. 定位左侧高点(L)与随后低点(B)。
    3. 基于深度分类 (Flat: <15%, Cup: 15-35%)。
    4. 平滑度与U形结构过滤 (拒绝宽幅震荡)。
    5. 右侧手柄(Handle)确认。
    """
    # --- 1. 基础数据检查 ---
    if len(df) < 250: return None
    
    # --- 2. 趋势模板 (Stage 2 Check) ---
    curr_close = df["close"].iloc[-1]
    ma_50 = df["close"].rolling(50).mean().iloc[-1]
    ma_200 = df["close"].rolling(200).mean().iloc[-1]
    ma_200_prev = df["close"].rolling(200).mean().iloc[-22]
    
    if not (curr_close > ma_50 and ma_50 > ma_200): return None
    if ma_200 <= ma_200_prev: return None # 200日线必须走平或向上

    # --- 3. 寻找左侧基底高点 (Left Pivot High) ---
    # 搜索范围：max_base_days 到 20天前
    search_start = max(len(df) - max_base_days, 0)
    search_end = len(df) - 15 # 给右侧手柄留出空间
    
    lookback_window = df.iloc[search_start:search_end]
    if len(lookback_window) < min_base_days: return None
    
    left_idx = lookback_window["high"].idxmax()
    left_val = lookback_window.loc[left_idx, "high"]
    left_iloc = df.index.get_loc(left_idx)
    
    # --- 4. 寻找基底低点 (Base Low) ---
    base_region = df.iloc[left_iloc:]
    if len(base_region) < min_base_days: return None 
    
    low_idx = base_region["low"].idxmin()
    low_val = base_region.loc[low_idx, "low"]
    low_iloc_in_base = base_region.index.get_loc(low_idx)
    base_len = len(base_region)

    # [Fix#1] 低点时间约束：低点必须在基底的 15%~85% 区间内
    # 如果低点在最左边 → 没有左侧下跌（不完整）
    # 如果低点在最右边 → 没有右侧恢复（还在下跌）
    low_position = low_iloc_in_base / base_len if base_len > 0 else 0
    if not (0.15 <= low_position <= 0.85):
        return None
    
    # --- 5. 计算深度与形态分类 ---
    depth_pct = (left_val - low_val) / left_val * 100
    
    # [Fix#6] 深度阈值与注释一致：Flat < 15%, Cup 15-35%
    pattern_type = "Unknown"
    if 8.0 <= depth_pct <= 15.0:
        pattern_type = "Flat Base"
    elif 15.0 < depth_pct <= 35.0:
        pattern_type = "Cup with Handle"
    else:
        return None # 太浅或太深

    # --- 6. [核心] 平滑度与结构检查 ---
    # [Fix#2] 平滑度只检查底部+恢复段（排除初始急跌）
    # [Fix#3] U 形判断：检查底部平坦度 + 低点位置
    smooth_metrics = _calculate_smoothness(base_region, left_val, low_val, low_iloc_in_base)
    
    if not smooth_metrics["is_smooth"]: return None 
    if not smooth_metrics["is_u_shaped"]: return None

    # --- 7. 右侧恢复与手柄检查 (Handle Check) ---
    # 价格必须回到左侧高点的 90% 以上
    if curr_close < left_val * 0.90: return None
    
    # [Fix#4] 手柄长度按基底比例取（10%-20% 的基底长度，最少 5 天最多 30 天）
    handle_days = max(5, min(30, base_len // 6))
    recent_df = df.iloc[-handle_days:]
    
    handle_high = recent_df["high"].max()
    handle_low = recent_df["low"].min()
    handle_depth = (handle_high - handle_low) / handle_high * 100
    
    # 手柄深度限制
    max_handle_depth = 6.0 if "Flat" in pattern_type else 10.0
    if handle_depth > max_handle_depth: return None
    
    # 手柄位置限制 (必须在上半区)
    if handle_low < left_val * 0.85: return None

    # 量能确认 (缩量)
    vol_50 = df["volume"].rolling(50).mean().iloc[-1]
    vol_5 = df["volume"].iloc[-5:].mean()
    vol_ratio = vol_5 / vol_50 if vol_50 > 0 else 1.0
    
    if vol_ratio > 1.25: return None
    
    # 距离枢轴点(前高)的距离
    dist_to_pivot = (left_val - curr_close) / left_val * 100

    # [Fix#5] 返回完整兼容格式（供 backtest_vcp.py / chart.py 消费）
    return {
        "pattern": pattern_type,
        "base_days": base_len,
        "base_high": float(left_val),
        "base_depth_pct": round(depth_pct, 2),
        "depths": [round(depth_pct, 2), round(handle_depth, 2)],
        "wave_count": 2,
        "num_contractions": 2,
        "waves": [],            # V5 不再用 ZigZag 波浪，chart.py 会安全跳过空列表
        "tightness": round(handle_depth, 2),
        "vol_ratio": round(vol_ratio, 2),
        "distance_to_pivot_pct": round(dist_to_pivot, 2),
        # 额外信息
        "handle_days": handle_days,
        "volatility_avg": smooth_metrics["avg_volatility"],
        "bottom_flatness": smooth_metrics["bottom_flatness"],
    }

# -----------------------------------------------------------------------------
#  扫描入口函数 (接口保持不变，内部逻辑适配 V5)
# -----------------------------------------------------------------------------

def scan_ndx100_vcp(
    min_base_days: int = 30,
    max_base_days: int = 180,
) -> list[dict]:
    """
    扫描纳指100中呈现 VCP（波动率收缩形态）或杯柄形态的股票。

    Args:
        min_base_days: 基底最短天数
        max_base_days: 基底最长天数

    Returns:
        [{"ticker": str, "df": DataFrame, "vcp_info": dict}, ...]
    """
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
            if len(df) < 250:
                logger.debug(f"{ticker}: 数据不足（{len(df)} 行），需至少 250 行检测 VCP")
                continue

            processed += 1
            
            # 调用 V5 核心逻辑，并传递参数
            result = screen_vcp(df, min_base_days, max_base_days)

            if result is not None:
                ptype = _PATTERN_CN.get(result["pattern"], result["pattern"])
                
                base_d = result["depths"][0]
                handle_d = result["depths"][1]
                
                logger.success(
                    f"✅ {ticker} 呈现{ptype} "
                    f"| 基底:{result['base_days']}日(深{base_d:.1f}%) "
                    f"| 手柄:{result['handle_days']}日(深{handle_d:.1f}%) "
                    f"| 量缩:{result['vol_ratio']:.0%} "
                    f"| 距前高:{result['distance_to_pivot_pct']:.1f}%"
                )
                hits.append({"ticker": ticker, "df": df, "vcp_info": result})
        except Exception as e:
            logger.error(f"{ticker}: VCP 检测失败 - {e}")
            continue

    logger.info(
        f"VCP 扫描完成：本地共 {len(stock_data)} 只股票，"
        f"有效处理 {processed} 只，{len(hits)} 只呈现形态"
    )
    return hits