# Stock Analyzer (stock_ana)

多市场股票技术分析与筛选系统，覆盖 **美股（~1500 只）**、**NASDAQ-100** 和 **港股（恒指 + 恒科）**，集成图表形态识别、动量异动检测与 Gemini AI 深度分析。

## 系统架构

```
数据采集 → 技术指标 → 策略筛选 → 回测验证 → AI 分析 → 报告输出
```

### 项目结构

```
stock_ana/
├── pyproject.toml                  # 依赖与项目配置
├── run_pipeline.py                 # NDX100 一键流水线
├── backtest.py / backtest_*.py     # 回测脚本
├── src/stock_ana/
│   ├── config.py                   # 跨平台路径配置
│   │
│   │  ── 数据层 ──
│   ├── data_fetcher.py             # A股 / 美股数据 (akshare + yfinance)
│   ├── data_fetcher_hk.py          # 港股数据 (东方财富 / 新浪)
│   ├── data_fetcher_us.py          # 美股宇宙构建 (Finviz 筛选)
│   ├── sec_fetcher.py              # SEC EDGAR 10-K 抓取与 SIC 分类
│   ├── labeler.py                  # LLM 三级行业标签 (Kimi K2.5)
│   │
│   │  ── 指标层 ──
│   ├── indicators.py               # MA/MACD/RSI/BB/OBV/Vegas 技术指标
│   │
│   │  ── 策略层 ──
│   ├── strategy_base.py            # Minervini Stage 2 趋势模板 + 几何工具
│   ├── strategy_vcp.py             # VCP V7 ZigZag 多波收缩
│   ├── strategy_triangle.py        # OLS 升三角 / 平行通道 / 上升楔形
│   ├── strategy_triangle_kde.py    # KDE 核密度升三角变体
│   ├── strategy_rs.py              # RS 加速 + RS 陷阱预警 (NDX100)
│   ├── strategy_rs_hk.py           # 港股 RS 日报
│   ├── momentum_detector.py        # 6维动量异动检测 (美股全市场)
│   ├── screener.py                 # 策略汇总与批量扫描
│   │
│   │  ── 回测层 ──
│   ├── backtest_momentum.py        # 6个月滚动动量回测
│   │
│   │  ── AI 分析层 ──
│   ├── gemini_analyst.py           # 个股 Gemini 基本面分析
│   ├── sector_gemini_analyst.py    # 板块级 Gemini 深度研判
│   ├── ai_code_reviewer.py         # AI 策略代码审查
│   │
│   │  ── 可视化层 ──
│   └── chart.py                    # K线图 + 技术指标叠加
│
├── data/
│   ├── taxonomy_v2.yaml            # 三级行业分类 (23 SIC 组, 77 子标签)
│   ├── us_sec_profiles.csv         # 1557只美股档案
│   ├── hk_list.txt                 # 港股股票池
│   ├── cache/                      # 价格数据缓存 (parquet, 不入库)
│   └── output/                     # 筛选结果 / 回测报告 / AI 分析
└── tests/
```

## 安装

```bash
# 克隆仓库
git clone https://github.com/80728495/stock_ana.git
cd stock_ana

# 创建虚拟环境
python3 -m venv .venv
source .venv/bin/activate     # macOS / Linux
# .venv\Scripts\activate      # Windows

# 安装项目 + 开发依赖
pip install -e ".[dev]"

# 配置环境变量
cp .env.example .env
# 编辑 .env 填写 API 密钥
```

## 功能模块

### 1. NDX100 一键流水线

```bash
python run_pipeline.py                # 完整流程：数据更新 → 策略筛选 → AI 分析
python run_pipeline.py --step 2,3     # 跳过数据更新
python run_pipeline.py --step 3       # 仅 AI 分析
```

流程覆盖：Vegas 通道回踩 → 升三角形态 → VCP 波动收缩 → Gemini 基本面评分 → 综合排名

### 2. 策略筛选

| 策略 | 模块 | 说明 |
|------|------|------|
| **VCP** | `strategy_vcp` | ZigZag 多波收缩（V7），自动识别杯柄 / 平底 / VCP 形态 |
| **升三角** | `strategy_triangle` | OLS 回归拟合阻力线 + 支撑线，检测收敛与突破 |
| **KDE 升三角** | `strategy_triangle_kde` | 核密度估计定位高密度阻力区 |
| **RS 加速** | `strategy_rs` | 相对强度从中等区加速，RS Line 穿越 EMA21 |
| **RS 陷阱** | `strategy_rs` | 识别"看似强势实则走弱"的虚假强度信号 |
| **Vegas 回踩** | `screener` | 价格从 EMA144/169 通道上方回踩 |
| **MACD 金叉** | `screener` | MACD 柱状图由负转正 |
| **布林收窄** | `screener` | Bollinger 带宽低于阈值 |

所有策略基于 **Minervini Stage 2 趋势模板** 预过滤（`strategy_base`）。

### 3. 美股板块动量系统

从 Finviz 筛选 ~1500 只美股 → SEC EDGAR 抓取 SIC 行业码 → LLM 分配三级子标签 → 6 维动量异动检测 → 板块聚合 → AI 深度研判。

```bash
# 构建股票池（仅首次）
python -m stock_ana.data_fetcher_us     # Finviz → us_universe.csv
python -m stock_ana.sec_fetcher         # SEC EDGAR → us_sec_profiles.csv
python -m stock_ana.labeler             # LLM → sub_label 列

# 更新价格 & 扫描异动
python -m stock_ana.momentum_detector --update    # 下载全部价格数据
python -m stock_ana.momentum_detector --scan      # 扫描当日异动
python -m stock_ana.momentum_detector --ticker NVDA  # 单只查看

# 6个月回测
python -m stock_ana.backtest_momentum --months 6 --min-score 3.0

# 板块 Gemini 分析
python -m stock_ana.sector_gemini_analyst --days 5 --preview   # 预览数据
python -m stock_ana.sector_gemini_analyst --days 5             # 发送 Gemini 分析
```

**6 维动量评分体系** (满分 10 分，≥3 分触发):

| 维度 | 分值 | 含义 |
|------|------|------|
| 量比 | 0–2 | 近期成交量 vs 参考期均量 |
| 异常收益 | 0–2 | 收益率 Z-Score (相对历史波动) |
| 突破 | 0–2 | 创 20 日 / 60 日新高 |
| 跳空 | 0–1 | 显著向上跳空缺口 |
| 均线突破 | 0–1 | 站上 50MA / 200MA |
| 吸筹 | 0–2 | 连续放量上涨天数 |

### 4. 港股 RS 日报

```bash
python -m stock_ana.strategy_rs_hk    # 计算恒指/恒科成分股 RS 排名
```

输出：RS 排名 (63d/21d)、RS 动量 Z-Score、强度分类、RS 序列 CSV、个股 RS 图表。

### 5. AI 分析

| 功能 | 命令 | 说明 |
|------|------|------|
| 个股基本面 | `gemini_analyst.py` | 成长性评分 + 估值合理性 + 买入建议 (docx) |
| 板块深度研判 | `sector_gemini_analyst.py` | 异动板块驱动因素 + 持续性判断 + TOP5 布局 |
| 策略代码审查 | `ai_code_reviewer.py` | 上传回测图表 + 源码，AI 提改进建议 |

AI 后端：`gemini-webapi` (Gemini Web 逆向接口，需 Chrome 登录 Gemini 获取 Cookie)。

### 6. 回测

```bash
python backtest.py            # NDX100 Vegas / 三角 / VCP 综合回测
python backtest_triangle.py   # 三角形态专项回测（含 OLS + KDE）
python backtest_vcp.py        # VCP 专项回测
```

回测输出图表保存在 `data/backtest_charts/` 下各子目录。

## 数据源

| 市场 | 数据源 | 说明 |
|------|--------|------|
| A股 | akshare | 日线行情 |
| 美股 (NDX100) | akshare (新浪) | Yahoo Finance 已被封禁 (403) |
| 美股 (全市场) | akshare | ~1500 只，parquet 缓存 |
| 港股 | akshare (东财/新浪) | 恒指 + 恒科成分股 |
| 美股档案 | SEC EDGAR | SIC 行业码、10-K Item 1 业务描述 |
| 美股筛选 | Finviz | 市值≥$2B, 均量≥50万, 价格≥$5 |
| 行业分类 | Volcengine ARK (Kimi K2.5) | 77 个子标签的 LLM 分类 |

## 关键设计决策

| 决策 | 说明 |
|------|------|
| `ta` 而非 `ta-lib` | 纯 Python，免编译，跨平台零障碍 |
| `pathlib.Path` | 统一 Windows / macOS 路径 |
| Parquet 缓存 + 增量更新 | 仅下载缺失日期，避免重复请求 |
| Minervini Stage 2 预过滤 | 过滤掉非上升趋势个股，减少假信号 |
| 三级行业分类 | Sector > SIC > Sub-label，精准定位板块轮动 |
| 6 维评分而非单指标 | 量价共振信号更可靠 |

## 环境变量

在 `.env` 文件中配置：

```ini
# Volcengine ARK API (行业标签分类)
ARK_API_KEY=your_key_here

# Gemini API (如使用直连模式)
GEMINI_API_KEY=your_key_here
```

Gemini Web 模式需在 Chrome 中登录 [gemini.google.com](https://gemini.google.com)，`gemini-webapi` 会自动读取 Cookie。

## 开发

```bash
# 运行测试
pytest tests/ -v

# 代码检查
ruff check src/

# 格式化
ruff format src/
```
