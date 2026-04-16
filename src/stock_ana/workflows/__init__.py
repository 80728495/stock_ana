"""工作流任务层。

编排多步骤的完整业务流程，组合 data/strategies/utils 层能力产生最终输出：
  ndx100_daily_pipeline  — NDX100 每日流水线（数据更新→策略筛选→AI分析→排名）
  weekly_sector_report   — 美股板块异动周报（扫描→板块聚合→Gemini深度分析→Markdown落盘）
"""
