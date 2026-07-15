"""Vegas 回踩机器学习研究包（mid / long 两套，按市场分离）。

对标 research/top_reversal，为 Mid Vegas / Long Vegas 回踩信号构建
「抄底 vs 跑路」二分类模型：

  candidates  — 用 impl.vegas_mid / vegas_long 的触碰检测器逐标的产出回踩事件
  labels      — 浪结构确认标签（止于通道并创新高=bounce / 深破且破位=breakdown）
  features    — 信号日特征（浪结构 + 通道几何 + 动量/波动/量能/RS + PIT 基本面/估值）
  build       — 组装 per-strategy 带标签数据集（含 market 列）
  modeling    — LR / LightGBM 训练（复用 top_reversal.modeling 的数值实现）
  eval_oos    — 严格 OOS：train=科技池−watchlist，test=watchlist，按策略×市场分离

两套策略绝不混训；每套内部再按 CN/HK/US 分离（共 2×3=6 模型）。
"""
