#!/usr/bin/env python3
"""
NDX100 每日分析流水线

用法：
    # 完整流程（更新数据 → 技术筛选 → Gemini 分析 → 综合排序）
    python -m stock_ana.workflows.ndx100_daily_pipeline

    # 只运行某一步
    python -m stock_ana.workflows.ndx100_daily_pipeline --step 1      # 仅更新数据
    python -m stock_ana.workflows.ndx100_daily_pipeline --step 2      # 仅技术筛选
    python -m stock_ana.workflows.ndx100_daily_pipeline --step 3      # 仅 Gemini 分析 + 排序
    python -m stock_ana.workflows.ndx100_daily_pipeline --step 2,3    # 筛选 + 分析

    # 跳过数据更新（使用已有数据）
    python -m stock_ana.workflows.ndx100_daily_pipeline --skip-update
"""

import argparse
import asyncio
import shutil

from loguru import logger

from stock_ana.config import OUTPUT_DIR


def _to_plot_hits(scan_result, stock_data: dict, info_key: str | None = None, df_transform=None) -> list[dict]:
    """把标准 ScanResult 转成现有绘图函数可消费的命中结构。"""
    hits: list[dict] = []
    for hit in scan_result.hits:
        df = stock_data.get(hit.symbol)
        if df is None:
            continue
        if df_transform is not None:
            df = df_transform(df.copy())
        item = {
            "ticker": hit.symbol,
            "df": df,
        }
        if info_key is not None:
            item[info_key] = hit.decision.features
        hits.append(item)
    return hits


def _clean_output():
    """清空 output 目录（保留目录本身）"""
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ════════════════════════════════════════
# Step 1: 更新当日数据
# ════════════════════════════════════════
def step1_update_data():
    """更新纳指100全部股票的日线数据"""
    logger.info("=" * 60)
    logger.info("【Step 1】更新纳指100日线数据...")
    logger.info("=" * 60)

    from stock_ana.data.fetcher import update_ndx100_data

    data = update_ndx100_data()
    logger.success(f"✅ 数据更新完成，共 {len(data)} 只股票")
    return data


# ════════════════════════════════════════
# Step 2: 技术面筛选（Vegas + 收敛三角形）
# ════════════════════════════════════════
def step2_screen() -> list[dict]:
    """
    运行 Vegas 通道回踩 + 收敛三角形/楔形 两套策略。
    返回合并去重后的 hit 列表。
    """
    logger.info("=" * 60)
    logger.info("【Step 2】技术面筛选...")
    logger.info("=" * 60)

    from stock_ana.data.indicators import add_vegas_channel
    from stock_ana.data.market_data import load_market_data
    from stock_ana.strategies.registry import scan_strategy
    from stock_ana.backtest.chart import plot_strategy_hits

    data = load_market_data("ndx100")
    if not data:
        logger.error("本地无数据！请先运行 Step 1 更新数据")
        return []

    all_hits = []

    # 策略1：Vegas 通道回踩
    vegas_result = scan_strategy("vegas", market="ndx100", lookback_days=5)
    vegas_hits = _to_plot_hits(
        vegas_result,
        data,
        df_transform=add_vegas_channel,
    )
    logger.info(f"【策略1】Vegas 通道回踩：{len(vegas_hits)} 只")
    if vegas_hits:
        plot_strategy_hits(vegas_hits, plot_mode="single_signal_vegas")
    all_hits.extend(vegas_hits)

    # 策略2：收敛三角形/楔形
    tri_result = scan_strategy("triangle_ascending")
    tri_hits = _to_plot_hits(tri_result, data, info_key="pattern_info")
    logger.info(f"【策略2】收敛三角形/楔形：{len(tri_hits)} 只")
    if tri_hits:
        plot_strategy_hits(tri_hits, plot_mode="pattern_triangle")
    all_hits.extend(tri_hits)

    # 策略3：VCP + 杯柄形态
    vcp_result = scan_strategy(
        "vcp",
        universe="ndx100",
        min_base_days=30,
        max_base_days=180,
    )
    vcp_hits = _to_plot_hits(vcp_result, data, info_key="vcp_info")
    logger.info(f"【策略3】VCP / 杯柄形态：{len(vcp_hits)} 只")
    if vcp_hits:
        plot_strategy_hits(vcp_hits, plot_mode="pattern_vcp")
    all_hits.extend(vcp_hits)

    # 去重保序
    unique_tickers = list(dict.fromkeys(h["ticker"] for h in all_hits))
    logger.success(f"✅ 技术筛选完成，共 {len(unique_tickers)} 只不重复股票：{unique_tickers}")

    return all_hits


# ════════════════════════════════════════
# Step 3: Gemini 基本面分析 + 综合排序
# ════════════════════════════════════════
def step3_analyze_and_rank(hits: list[dict] | None = None):
    """
    对筛选结果进行 Gemini 基本面分析，然后上传报告综合排序。
    如果 hits 为 None，则扫描 output 目录中已有的技术筛选图表推断 tickers。
    """
    logger.info("=" * 60)
    logger.info("【Step 3】Gemini 基本面分析 + 综合排序...")
    logger.info("=" * 60)

    from stock_ana.utils.gemini_analyst import batch_analyze, rank_and_summarize

    # 确定待分析的 tickers
    if hits:
        tickers = list(dict.fromkeys(h["ticker"] for h in hits))
    else:
        # 从 output 目录的图表文件推断
        chart_files = (list(OUTPUT_DIR.glob("*_triangle.png"))
                      + list(OUTPUT_DIR.glob("*_vegas*.png"))
                      + list(OUTPUT_DIR.glob("*_vcp.png")))
        tickers = list(dict.fromkeys(p.stem.split("_")[0] for p in chart_files))
        if not tickers:
            logger.error(f"未找到筛选结果！请先运行 Step 2，或在 {OUTPUT_DIR} 中放入图表文件")
            return

    logger.info(f"待分析标的（{len(tickers)} 只）：{tickers}")

    # 逐只分析
    paths = asyncio.run(batch_analyze(tickers, delay=5.0))
    logger.info(f"Gemini 分析完成，共生成 {len(paths)} 份报告")

    # 综合排序
    if len(paths) >= 2:
        logger.info("开始综合排序...")
        rank_path = asyncio.run(rank_and_summarize(paths))
        logger.success(f"📊 综合排序报告：{rank_path}")
    elif len(paths) == 1:
        logger.info("仅 1 份报告，跳过排序")
    else:
        logger.warning("无成功报告，跳过排序")

    logger.success(f"✅ 全部完成！输出目录：{OUTPUT_DIR}")


# ════════════════════════════════════════
# 主入口
# ════════════════════════════════════════
def main():
    """Run the staged end-to-end pipeline for data refresh, screening, and Gemini analysis."""
    parser = argparse.ArgumentParser(
        description="NDX100 每日分析流水线",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python -m stock_ana.workflows.ndx100_daily_pipeline           # 完整流程
  python -m stock_ana.workflows.ndx100_daily_pipeline --step 2,3  # 跳过数据更新
  python -m stock_ana.workflows.ndx100_daily_pipeline --step 3    # 仅分析+排序
  python -m stock_ana.workflows.ndx100_daily_pipeline --skip-update
        """,
    )
    parser.add_argument(
        "--step",
        type=str,
        default=None,
        help="要运行的步骤，用逗号分隔。1=更新数据，2=技术筛选，3=Gemini分析+排序",
    )
    parser.add_argument(
        "--skip-update",
        action="store_true",
        help="跳过数据更新（等同于 --step 2,3）",
    )
    args = parser.parse_args()

    if args.step:
        steps = [int(s.strip()) for s in args.step.split(",")]
    elif args.skip_update:
        steps = [2, 3]
    else:
        steps = [1, 2, 3]

    logger.info(f"🚀 NDX100 流水线启动，执行步骤：{steps}")
    print()

    hits = None

    if 1 in steps:
        step1_update_data()
        print()

    if 2 in steps:
        _clean_output()
        hits = step2_screen()
        print()

    if 3 in steps:
        step3_analyze_and_rank(hits)
        print()

    logger.info("🏁 流水线结束")


if __name__ == "__main__":
    main()
