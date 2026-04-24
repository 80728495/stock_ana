# Stock Analyzer (stock_ana)

多市场股票技术分析与筛选系统，覆盖 **美股（~1500 只）**、**NASDAQ-100** 和 **港股（恒指 + 恒科）**，集成图表形态识别、动量异动检测与 Gemini AI 深度分析。

## 系统架构

```
数据采集 → 技术指标 → 策略筛选 → 回测验证 → AI 分析 → 报告输出
```

依赖层次（单向，不允许反向引用）：

```
data/ → strategies/primitives/ → strategies/impl/ → strategies/api → scan/ → workflows/
                                                                    ↘ backtest/
```

### 项目结构

```
stock_ana/
├── daily_update.py                 # 每日行情 + 指标 + Wave 结构更新（cron 入口）
├── full_refresh_pipeline.py        # 全市场一键全量刷新流水线
├── cron_daily_update.sh            # cron 定时任务 shell 脚本
├── scripts/
│   ├── check_legacy_wrapper_imports.py   # 守卫：禁止引入已删除旧 wrapper 路径
│   └── check_strategy_layer_boundaries.py # 守卫：禁止策略层越界导入
├── tests/
│   ├── test_legacy_wrapper_guard.py      # 架构边界回归测试
│   └── test_screener.py                  # 端到端集成测试
└── src/stock_ana/
    ├── config.py                   # 路径与全局配置（PROJECT_ROOT / DATA_DIR / CACHE_DIR / OUTPUT_DIR）
    ├── data/                       # 数据层（价格、指标、列表、波段、峰值、AI 标注）
    │   ├── fetcher.py              # 美股 / NDX100 行情拉取 + QQQ benchmark 更新
    │   ├── fetcher_hk.py           # 港股行情拉取
    │   ├── indicators.py           # 单 DataFrame 指标计算（EMA、成交量均线、前高）
    │   ├── indicators_store.py     # 指标批量持久化（全市场 US + NDX100 + HK）
    │   ├── market_data.py          # 统一市场数据读取门面（不触发网络请求）
    │   ├── list_manager.py         # 股票列表文件（.md）的读写同步
    │   ├── wave_store.py           # Wave 波段结构批量持久化
    │   ├── peak_store.py           # 宏观峰值缓存与持久化
    │   ├── labeler.py              # LLM 行业子标签分配（Volcengine ARK）
    │   ├── sec_fetcher.py          # SEC EDGAR 公司业务描述抓取
    │   ├── us_universe_builder.py  # 通过 Finviz 构建美股投资标的池
    │   └── hk_universe_builder.py  # 构建港股主板大市值列表
    ├── strategies/
    │   ├── contracts.py            # 标准化返回契约：StrategyDecision / ScanHit / ScanResult
    │   ├── api.py                  # 策略统一门面（screen_* / scan_* / explain_*）
    │   ├── registry.py             # 策略注册与调度（scan_strategy / STRATEGY_SCAN_REGISTRY）
    │   ├── screener.py             # 基础技术指标筛选函数（金叉、MACD、布林）
    │   ├── impl/                   # 各策略算法实现（仅依赖 primitives + data）
    │   │   ├── vegas_mid.py        # Mid Vegas（EMA34/55/60）回踩检测与评分
    │   │   ├── ma_squeeze.py       # 多均线收敛两阶段检测
    │   │   ├── vcp.py              # VCP V8：显式 P1/P2/P3 + ZigZag 多波收缩
    │   │   ├── triangle.py         # OLS 三角形态（上升三角 / 平行通道 / 上升楔形）
    │   │   ├── triangle_kde.py     # KDE 核密度上升三角形检测
    │   │   ├── triangle_vcp.py     # VCP + 三角收敛组合形态
    │   │   ├── rs.py               # RS 加速策略（rs_strict / rs_loose / rs_trap）
    │   │   ├── momentum_detector.py # 6 维动量异动检测
    │   │   └── main_rally_pullback.py # 主升段 Vegas 中期回踩
    │   └── primitives/             # 可复用基础算法积木
    │       ├── pivots.py           # 高低点提取（argrel + ZigZag）
    │       ├── peaks.py            # 宏观峰值识别
    │       ├── cup.py              # 杯底结构分析（P1/P2/P3）
    │       ├── wave.py             # EMA8 波段结构分析
    │       ├── regression.py       # OLS 回归拟合
    │       ├── rs.py               # RS Line / RS 排名计算
    │       ├── momentum.py         # 量价异动评分函数
    │       ├── squeeze.py          # 均线收敛度量
    │       ├── trend.py            # Minervini Stage 2 趋势过滤
    │       └── vcp.py              # VCP 摆幅收缩检测
    ├── scan/                       # 扫描层
    │   ├── ma_squeeze_scan.py      # MA 多线压缩每日扫描（US + HK，输出 JSON + 图表）[有 main()]
    │   ├── vegas_mid_scan.py       # Vegas Mid 回踩每日扫描 [有 main()]
    │   ├── cup_handle_scan.py      # 杯柄形态全量扫描 [有 main()]
    │   ├── hk_rs_scan.py           # 港股 RS 每日计算与报告输出 [有 main()]
    │   ├── triangle_scan.py        # 三角形态历史扫描（回测辅助模块）
    │   ├── triangle_vcp_scan.py    # VCP 三角形历史扫描（回测辅助模块）
    │   ├── vcp_scan.py             # VCP 历史扫描（回测辅助模块）
    │   ├── vegas_wave_scan.py      # Vegas 升浪历史扫描（回测辅助模块）
    │   └── main_rally_scan.py      # 主升浪回调历史扫描（回测辅助模块）
    ├── backtest/                   # 回测层
    │   ├── backtest_multi_strategy.py    # NDX100 多策略综合回测
    │   ├── backtest_triangle.py          # 三角形态专项回测
    │   ├── backtest_triangle_vcp.py      # VCP 三角形专项回测（Shawn List）
    │   ├── backtest_vcp.py               # VCP 专项回测
    │   ├── backtest_vegas_mid.py           # Vegas Mid-Vegas 回踩策略回测
    │   ├── backtest_ma_squeeze.py        # 多线压缩全量回测
    │   ├── backtest_main_rally_pullback.py # 主升浪回调联合回测
    │   ├── backtest_momentum.py          # 动量异动半年期回测
    │   ├── diagnostics/            # 单股诊断工具（非生产，研究用）
    │   └── research/               # 回测后验统计分析脚本
    ├── workflows/                  # 工作流层（完整业务流程编排）
    │   ├── ndx100_daily_pipeline.py  # NDX100 每日一键流水线（更新→筛选→AI分析→排名）
    │   └── weekly_sector_report.py  # 美股板块异动周报（扫描→聚合→Gemini深度分析→Markdown落盘）
    └── utils/
        ├── plot_renderers.py       # 统一图表渲染层（K线 + 策略标注 + 趋势线）
        ├── gemini_analyst.py       # Gemini AI 个股基本面分析与批量评分
        └── ai_code_reviewer.py     # 上传回测图表 + 源码，调用 AI 提出改进建议

data/
├── lists/                          # 股票列表 Markdown 文件（手动维护 + 自动同步）
│   ├── watchlist.md               # 关注列表（手动维护＋自动同步持仓，表格格式，含中英文名）
│   ├── ndx100_list.md              # NASDAQ-100 成分股
│   ├── us_universe_list.md         # 美股全市场标的池
│   └── hk_focus_list.md / hk_full_list.md
├── taxonomy_v2.yaml                # 三级行业分类体系
├── us_sec_profiles.csv             # 美股档案（SIC 行业码 + 公司描述 + sub_label）
├── cache/                          # 本地价格 Parquet 缓存（us / ndx100 / hk / indicators / wave_structure）
└── output/                         # 扫描结果 / 回测报告 / AI 分析报告
```

## 安装

```bash
git clone https://github.com/80728495/stock_ana.git
cd stock_ana

python3 -m venv .venv
source .venv/bin/activate     # macOS / Linux

pip install -e ".[dev]"

cp .env.example .env
# 编辑 .env 填写 API 密钥
```

## 1) 每日 + 每周任务（cron 实际发生内容）

### 每日任务

脚本入口：

```bash
bash cron_daily_update.sh
bash cron_daily_scan_notify.sh
```

cron 实际执行顺序：

1. `cron_daily_update.sh`
2. `python3 daily_update.py --lists`
3. `python3 daily_update.py`
4. `cron_daily_scan_notify.sh`
5. `python3 vegas_mid_daily_scan.py`
6. `python3 notify_daily_scan_result.py --scan-exit-code <SCAN_EXIT>`

手工复现命令：

```bash
# Step A: 列表同步 + 全量日更
python3 daily_update.py --lists
python3 daily_update.py

# Step B: 每日扫描 + Gemini
python3 vegas_mid_daily_scan.py

# Step C: 推送当日结果（透传扫描退出码）
SCAN_EXIT=$?
python3 notify_daily_scan_result.py --scan-exit-code "$SCAN_EXIT"
```

### 每周任务

脚本入口：

```bash
bash cron_weekly_sector_notify.sh
```

cron 实际执行顺序：

1. `cron_weekly_sector_notify.sh`
2. `python3 -m stock_ana.workflows.weekly_sector_report`
3. `python3 notify_weekly_sector_report.py --workflow-exit-code <WORKFLOW_EXIT>`

手工复现命令：

```bash
# Step A: 周报流水线（更新价格 -> 扫描异动 -> 板块聚合 -> Gemini 生成报告）
python3 -m stock_ana.workflows.weekly_sector_report

# Step B: 推送周报结果（透传 workflow 退出码）
WORKFLOW_EXIT=$?
python3 notify_weekly_sector_report.py --workflow-exit-code "$WORKFLOW_EXIT"
```

## 2) 策略说明（不含流程）

### 可用策略一览

| 策略名 | 入口 | 实现模块 | 说明 |
|--------|------|----------|------|
| `vegas` | `api.scan_vegas_touches` | `api` 内联 | EMA144/169 通道上方回踩 |
| `vegas_mid` | `api.scan_vegas_mid_pullbacks` | `impl.vegas_mid` | EMA34/55/60 中期通道回踩，含波浪评分 |
| `ma_squeeze` | `api.scan_ma_squeeze` | `impl.ma_squeeze` | 多均线两阶段收敛检测 |
| `vcp` | `api.scan_vcp_setups` | `impl.vcp` | 显式 P1/P2/P3 杯柄 + ZigZag 多波收缩 |
| `triangle_ascending` | `api.scan_triangle_ascending` | `impl.triangle` | OLS 上升三角形 |
| `triangle_parallel_channel` | `api.scan_triangle_parallel_channel` | `impl.triangle` | OLS 平行上升通道 |
| `triangle_rising_wedge` | `api.scan_triangle_rising_wedge` | `impl.triangle` | OLS 上升楔形 |
| `triangle_kde` | `api.scan_triangle_kde_setups` | `impl.triangle_kde` | KDE 核密度阻力区上升三角 |
| `triangle_vcp` | `api.scan_triangle_vcp_setups` | `impl.triangle_vcp` | VCP + 三角收敛组合 |
| `rs_acceleration` | `api.scan_rs_acceleration` | `impl.rs` | RS Line 从低位拐头加速 |
| `rs_trap` | `api.scan_rs_trap_alert` | `impl.rs` | 虚假强势陷阱预警 |
| `momentum` | `api.scan_momentum` | `impl.momentum_detector` | 6 维量价异动评分 |
| `main_rally_pullback` | `api.scan_main_rally_pullback_setups` | `impl.main_rally_pullback` | 主升段中轨回踩 |

### 统一调用方式

```python
from stock_ana.strategies.api import scan_vcp_setups, screen_vcp_setup
from stock_ana.strategies.registry import scan_strategy

# 标准单策略调用
result = scan_vcp_setups(universe="ndx100")          # ScanResult
decision = screen_vcp_setup(df)                       # StrategyDecision

# 通过 registry 统一调度
result = scan_strategy("vcp", universe="ndx100")
```

### 6 维动量评分体系

| 维度 | 分值 | 含义 |
|------|------|------|
| 量比 | 0-2 | 近 5 日均量 vs 近 50 日均量 |
| 异常收益 | 0-2 | 收益率 Z-Score（相对历史波动） |
| 突破 | 0-2 | 创 20 日 / 60 日新高 |
| 跳空 | 0-1 | 显著向上跳空缺口（>=2%） |
| 均线突破 | 0-1 | 站上 50MA / 200MA |
| 吸筹 | 0-2 | 连续放量上涨天数 |

## 3) 其他工具和脚本（非每日 cron 主链路）

### 扫描工具（手动/研究）

```bash
# MA 多线压缩（US + HK）
python -m stock_ana.scan.ma_squeeze_scan

# Vegas Mid 回踩扫描器
python -m stock_ana.scan.vegas_mid_scan

# 港股 RS 排名与图表
python -m stock_ana.scan.hk_rs_scan

# 杯柄形态扫描（全量美股）
python -m stock_ana.scan.cup_handle_scan
```

### 工作流与报告工具

```bash
# NDX100 一键流水线
python -m stock_ana.workflows.ndx100_daily_pipeline
python -m stock_ana.workflows.ndx100_daily_pipeline --step 2,3
python -m stock_ana.workflows.ndx100_daily_pipeline --step 3
python -m stock_ana.workflows.ndx100_daily_pipeline --skip-update

# 周报预览
python -m stock_ana.workflows.weekly_sector_report --preview
```

### 数据构建与维护工具

```bash
# 首次构建美股池与档案
python -m stock_ana.data.us_universe_builder
python -m stock_ana.data.sec_fetcher
python -m stock_ana.data.labeler

# 全市场手工全刷
python full_refresh_pipeline.py
```

### 回测工具

```bash
python -m stock_ana.backtest.backtest_multi_strategy
python -m stock_ana.backtest.backtest_triangle
python -m stock_ana.backtest.backtest_vcp
python -m stock_ana.backtest.backtest_triangle_vcp
python -m stock_ana.backtest.backtest_vegas_mid
python -m stock_ana.backtest.backtest_momentum
```

回测图表输出：`data/backtest_charts/`

### AI 与开发工具

| 功能 | 模块 | 说明 |
|------|------|------|
| 个股基本面 | `utils.gemini_analyst` | 成长性评分 + 估值合理性 + 买入建议 |
| 板块深度研判 | `workflows.weekly_sector_report` | 周度板块聚合 + Gemini 分析 + Markdown 周报 |
| 策略代码审查 | `utils.ai_code_reviewer` | 上传回测图表 + 源码，AI 提改进建议 |

```bash
# 架构边界检查
python3 scripts/check_legacy_wrapper_imports.py
python3 scripts/check_strategy_layer_boundaries.py

# 语法/测试/代码风格
python3 -m py_compile src/stock_ana/**/*.py
pytest tests/ -v
ruff check src/
ruff format src/
```

## 4) 目录结构与文件说明 + 模块创建/实现原则

### 依赖与分层

系统链路：

```text
数据采集 -> 技术指标 -> 策略筛选 -> 回测验证 -> AI 分析 -> 报告输出
```

单向依赖（禁止反向引用）：

```text
data/ -> strategies/primitives/ -> strategies/impl/ -> strategies/api -> scan/ -> workflows/
                                                                    \-> backtest/
```

### 目录结构（核心）

```text
stock_ana/
├── daily_update.py
├── full_refresh_pipeline.py
├── cron_daily_update.sh
├── cron_daily_scan_notify.sh
├── cron_weekly_sector_notify.sh
├── notify_daily_scan_result.py
├── notify_weekly_sector_report.py
├── scripts/
│   ├── check_legacy_wrapper_imports.py
│   └── check_strategy_layer_boundaries.py
├── tests/
│   ├── test_legacy_wrapper_guard.py
│   └── test_screener.py
└── src/stock_ana/
    ├── config.py
    ├── data/
    ├── strategies/
    │   ├── contracts.py
    │   ├── api.py
    │   ├── registry.py
    │   ├── impl/
    │   └── primitives/
    ├── scan/
    ├── backtest/
    ├── workflows/
    └── utils/

data/
├── lists/
├── cache/
├── output/
├── taxonomy_v2.yaml
└── us_sec_profiles.csv
```

### 各模块职责

- `data/`: 行情拉取、缓存、指标持久化、列表管理、波段结构与基础元数据。
- `strategies/primitives/`: 可复用算法积木（pivots、regression、trend、vcp 等）。
- `strategies/impl/`: 具体策略实现，仅组合 primitives 与 data。
- `strategies/api.py`: 对外统一 `screen_* / scan_* / explain_*` 门面。
- `scan/`: 面向日常扫描与报告产出的 CLI 层。
- `backtest/`: 各策略历史回测与研究验证。
- `workflows/`: 多步骤编排任务（如 NDX100 流程、周报流程）。
- `utils/`: 图表渲染、AI 客户端适配、工具函数。

### 模块创建与实现原则

1. 先定层级：新能力必须落在正确层，禁止跨层倒置调用。
2. 先抽 primitives：跨策略复用的算法先下沉到 `primitives/`。
3. impl 保持纯净：`impl/` 内不互相导入，通过 `api.py` 统一编排。
4. 统一返回契约：策略输出使用 `StrategyDecision / ScanResult / ScanHit`。
5. 图表统一渲染：可视化仅在 `utils/plot_renderers.py`，策略层不直接画图。
6. 新策略接入路径：`primitives (可选) -> impl -> api -> registry (可选) -> scan/backtest/workflows`。
7. 用 guard 兜底：新增/重构后运行边界检查脚本与测试，防止架构回退。

## 数据源

| 市场 | 数据源 | 说明 |
|------|--------|------|
| 美股 (NDX100) | akshare（新浪） | Yahoo Finance 已被封禁（403） |
| 美股（全市场） | akshare | ~1500 只 Parquet 缓存 |
| 港股 | akshare（东财 / 新浪） | 恒指 + 恒科成分股 |
| 美股档案 | SEC EDGAR | SIC 行业码、10-K Item 1 业务描述 |
| 美股筛选 | Finviz | 市值 >= $2B，均量 >= 50 万，价格 >= $5 |
| 行业分类 | Volcengine ARK | LLM 分配 77 个子标签 |

## 环境变量

```ini
# Volcengine ARK API（行业标签分类）
ARK_API_KEY=your_key_here

# Gemini API（如使用直连模式）
GEMINI_API_KEY=your_key_here
```

Gemini Web 模式需在 Chrome 中登录 [gemini.google.com](https://gemini.google.com)，`gemini-webapi` 会自动读取 Cookie。
