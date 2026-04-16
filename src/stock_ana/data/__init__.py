"""数据层：获取、缓存、中间计算与市场数据门面。

  fetcher / fetcher_hk       — OHLCV 行情拉取与增量更新（美股 / 港股）
  indicators / indicators_store — 技术指标计算与持久化
  market_data                — 统一市场数据读取门面（不触发网络请求）
  list_manager               — 股票列表文件（.md）的读写同步
  wave_store / peak_store    — 波段结构与宏观峰值的持久化存储
  sec_fetcher / labeler      — SEC 档案抓取与 LLM 行业标签分类
  us_universe_builder / hk_universe_builder — 股票池初始构建
"""
