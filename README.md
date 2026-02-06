# Stock Analyzer (stock_ana)

股票技术分析与筛选工具，支持 A 股和美股数据。

## 项目结构

```
stock_ana/
├── pyproject.toml          # 依赖与项目配置
├── .editorconfig           # 跨平台编辑器配置
├── .gitignore
├── .env.example            # 环境变量模板
├── README.md
├── src/
│   └── stock_ana/
│       ├── __init__.py
│       ├── config.py       # 跨平台配置
│       ├── data_fetcher.py # 数据获取（A股 + 美股）
│       ├── indicators.py   # 技术指标计算
│       ├── screener.py     # 股票筛选策略
│       └── chart.py        # K线图与可视化
├── tests/
│   └── test_screener.py
└── data/                   # 本地数据目录（不同步）
    ├── cache/
    └── output/
```

## 跨平台设置指南（Windows + macOS）

### 0. 前置条件

两台电脑都需要安装：

- **Python 3.10+** — [python.org](https://www.python.org/downloads/)
- **Git** — [git-scm.com](https://git-scm.com/downloads)
- **VS Code** — [code.visualstudio.com](https://code.visualstudio.com/)

### 1. 首次设置（Windows 当前电脑）

```bash
# 1) 安装 Git 后，进入项目目录
cd stock_ana

# 2) 初始化 Git 仓库
git init
git add -A
git commit -m "init: 项目初始化"

# 3) 在 GitHub 创建一个新的空仓库（不要勾选 README），然后：
git remote add origin https://github.com/<你的用户名>/stock_ana.git
git branch -M main
git push -u origin main

# 4) 创建 Python 虚拟环境并安装依赖
python -m venv .venv
.venv\Scripts\activate        # Windows 激活
pip install -e ".[dev]"       # 安装项目 + 开发依赖
```

### 2. MacBook 设置

```bash
# 1) 克隆仓库
git clone https://github.com/<你的用户名>/stock_ana.git
cd stock_ana

# 2) 创建虚拟环境
python3 -m venv .venv
source .venv/bin/activate     # macOS 激活
pip install -e ".[dev]"

# 3) 复制环境变量文件
cp .env.example .env
```

### 3. 日常同步工作流

```bash
# 开始工作前：拉取最新代码
git pull

# 完成工作后：提交并推送
git add -A
git commit -m "feat: 你的改动描述"
git push
```

## 快速使用

```python
from stock_ana.data_fetcher import fetch_cn_stock, fetch_us_stock
from stock_ana.indicators import add_all_indicators
from stock_ana.screener import run_screen

# 获取贵州茅台数据
df = fetch_cn_stock("600519", "20240101", "20241231")

# 计算技术指标
df = add_all_indicators(df)

# 运行筛选策略
results = run_screen(df)
print(results)
# {'golden_cross': False, 'rsi_oversold': False, 'macd_bullish': True, 'bollinger_squeeze': False}
```

## 关键设计决策（跨平台兼容）

| 决策 | 说明 |
|------|------|
| 使用 `ta` 而非 `ta-lib` | ta-lib 需要编译 C 库，跨平台安装困难；ta 是纯 Python |
| 使用 `pathlib.Path` | 统一处理 Windows `\` 和 macOS `/` 路径差异 |
| `.editorconfig` | 统一换行符为 LF，避免 CRLF/LF 混乱 |
| `.env` + `.gitignore` | API 密钥不入库，每台电脑独立配置 |
| 虚拟环境 `.venv/` | 不同步到 Git，各平台独立安装 |
| `pyproject.toml` | 统一依赖管理，两台电脑 `pip install -e .` 即装即用 |

## VS Code 推荐设置

在两台电脑的 VS Code 中安装以下扩展：
- Python (ms-python.python)
- Pylance (ms-python.vscode-pylance)
- Ruff (charliermarsh.ruff)
- GitLens (eamodio.gitlens)
- EditorConfig (editorconfig.editorconfig)
