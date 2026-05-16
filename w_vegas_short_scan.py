#!/usr/bin/env python3
"""
Weekly Vegas Short 回踩周线扫描 CLI 入口

用法:
    python w_vegas_short_scan.py                  # 扫描自选列表（watchlist.md）
    python w_vegas_short_scan.py --hk             # 港股宇宙池
    python w_vegas_short_scan.py --us             # 美股科技板块
    python w_vegas_short_scan.py --us-full        # 美股全量
    python w_vegas_short_scan.py --lookback 4     # 放宽到最近 4 周
    python w_vegas_short_scan.py --min-signal BUY # 最低信号等级（默认 BUY）

输出目录：data/output/w_vegas_scan/{YYYY-MM-DD}/
  signals.json       ← 纯信号（不含 base64）
  signals_full.json  ← 含 base64 图表
  W_*.png            ← 各标的周线 K 线图
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from loguru import logger

LOG_DIR = PROJECT_ROOT / "data" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
logger.add(
    LOG_DIR / "w_vegas_scan_{time:YYYY-MM-DD}.log",
    rotation="1 day",
    retention="30 days",
    level="INFO",
    encoding="utf-8",
    enqueue=True,
)

from stock_ana.scan.w_vegas_short_scan import main

if __name__ == "__main__":
    main()
