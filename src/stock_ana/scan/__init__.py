"""每日扫描与即席研究脚本集。

分为两类：
  - 每日扫描主程序（有 main() 入口，直接执行即可读取最新数据并输出结果）：
      ma_squeeze_scan, vegas_mid_scan, cup_handle_scan, hk_rs_scan
  - 历史回测辅助模块（仅暴露单股扫描函数，由 backtest/ 层调用）：
      triangle_scan, triangle_vcp_scan, vcp_scan, main_rally_scan
"""
