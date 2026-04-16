"""
美股杯柄形态扫描 — 全量 1400+ 美股列表
结果图片输出到 data/output/vcp/，文件名格式: {TICKER}_cup_handle.png

图例：
  ▲ 绿色上三角 = 左杯口（基底高点）
  ▼ 蓝色下三角 = 杯底（最低点）
  ▲ 橙色上三角 = 右杯口（手柄起点）
  -- 品红虚线   = 枢轴参考线（左杯口价格）

用法：
  python -m stock_ana.scan.cup_handle_scan            # 标准模式（手柄深度 ≤6/8%）
  python -m stock_ana.scan.cup_handle_scan --loose    # 宽松模式（手柄深度 ≤12/15%，适合高波动市场）
"""

import sys
from loguru import logger
from stock_ana.strategies.impl.vcp import scan_us_vcp
from stock_ana.utils.plot_renderers import plot_cup_handle_annotated


def main() -> None:
    """Scan the US universe for cup-with-handle setups and render annotated charts."""
    loose = "--loose" in sys.argv
    mode_str = "宽松模式（手柄≤12/15%）" if loose else "标准模式（手柄≤6/8%）"
    logger.info(f"=== 美股杯柄形态扫描开始 [{mode_str}] ===")

    hits = scan_us_vcp(min_base_days=60, max_base_days=300, loose=loose)

    if not hits:
        logger.warning(
            "未发现符合杯柄条件的标的。"
            + ("" if loose else " 提示：当前市场波动较大，可尝试 --loose 模式放宽手柄限制。")
        )
        return

    logger.info(f"共命中 {len(hits)} 只，开始绘制标注图...")
    plot_cup_handle_annotated(hits)
    logger.info("=== 扫描完成 ===")


if __name__ == "__main__":
    main()
