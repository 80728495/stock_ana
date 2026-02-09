#!/usr/bin/env python3
"""
stock_ana 一键流水线脚本

用法：
    # 完整流程（更新数据 → 技术筛选 → Gemini 分析 → 综合排序）
    python run_pipeline.py

    # 只运行某一步
    python run_pipeline.py --step 1      # 仅更新数据
    python run_pipeline.py --step 2      # 仅技术筛选
    python run_pipeline.py --step 3      # 仅 Gemini 分析 + 排序
    python run_pipeline.py --step 2,3    # 筛选 + 分析

    # 跳过数据更新（使用已有数据）
    python run_pipeline.py --skip-update
"""

import argparse
import asyncio
import shutil
import sys
from pathlib import Path

from loguru import logger

# ─── 项目路径 ───
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

OUTPUT_DIR = PROJECT_ROOT / "data" / "output"


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

    from stock_ana.data_fetcher import update_ndx100_data

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

    from stock_ana.data_fetcher import load_all_ndx100_data
    from stock_ana.screener import scan_ndx100_vegas_touch, scan_ndx100_ascending_triangle, scan_ndx100_vcp
    from stock_ana.chart import plot_vegas_touch_results, plot_ascending_triangle_results, plot_vcp_results

    data = load_all_ndx100_data()
    if not data:
        logger.error("本地无数据！请先运行 Step 1 更新数据")
        return []

    all_hits = []

    # 策略1：Vegas 通道回踩
    vegas_hits = scan_ndx100_vegas_touch(lookback_days=5)
    logger.info(f"【策略1】Vegas 通道回踩：{len(vegas_hits)} 只")
    if vegas_hits:
        plot_vegas_touch_results(vegas_hits)
    all_hits.extend(vegas_hits)

    # 策略2：收敛三角形/楔形
    tri_hits = scan_ndx100_ascending_triangle(min_period=40, max_period=120)
    logger.info(f"【策略2】收敛三角形/楔形：{len(tri_hits)} 只")
    if tri_hits:
        plot_ascending_triangle_results(tri_hits)
    all_hits.extend(tri_hits)

    # 策略3：VCP + 杯柄形态
    vcp_hits = scan_ndx100_vcp(min_base_days=30, max_base_days=180)
    logger.info(f"【策略3】VCP / 杯柄形态：{len(vcp_hits)} 只")
    if vcp_hits:
        plot_vcp_results(vcp_hits)
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

    from stock_ana.gemini_analyst import batch_analyze, rank_and_summarize

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

    # 第二步：逐只分析
    paths = asyncio.run(batch_analyze(tickers, delay=5.0))
    logger.info(f"Gemini 分析完成，共生成 {len(paths)} 份报告")

    # 第三步：综合排序
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
    parser = argparse.ArgumentParser(
        description="stock_ana 一键分析流水线",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python run_pipeline.py                # 完整流程
  python run_pipeline.py --step 2,3     # 跳过数据更新
  python run_pipeline.py --step 3       # 仅分析+排序（使用已有筛选结果）
  python run_pipeline.py --skip-update  # 等同于 --step 2,3
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

    # 确定要运行的步骤
    if args.step:
        steps = [int(s.strip()) for s in args.step.split(",")]
    elif args.skip_update:
        steps = [2, 3]
    else:
        steps = [1, 2, 3]

    logger.info(f"🚀 stock_ana 流水线启动，执行步骤：{steps}")
    print()

    hits = None

    # Step 1
    if 1 in steps:
        step1_update_data()
        print()

    # Step 2
    if 2 in steps:
        _clean_output()
        hits = step2_screen()
        print()

    # Step 3
    if 3 in steps:
        step3_analyze_and_rank(hits)
        print()

    logger.info("🏁 流水线结束")


if __name__ == "__main__":
    main()
