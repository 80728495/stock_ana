"""回测引擎与图表辅助工具包。

对每种策略提供独立的全量回测脚本（逐日滚动扫描 + 前瞻收益统计），
同时包含 diagnostics/ 单股诊断工具和 research/ 后验分析脚本。
所有回测模块通过 stock_ana.strategies.api 调用策略，不直接依赖 impl 内部实现。
"""
