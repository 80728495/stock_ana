#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/Users/wl/stock_ana"
PYTHON_BIN="/Users/wl/.pyenv/shims/python3"

cd "$PROJECT_DIR"

# 0) 同步 Futu 账户持仓到 watchlist.md（建议在 6:00 运行，K 线更新之前）
"$PYTHON_BIN" sync_holdings.py || echo "⚠️ sync_holdings.py 失败（OpenD 未连接？），继续执行后续步骤"

# 1) 同步自动列表，再执行全量每日 K 线更新
"$PYTHON_BIN" daily_update.py --lists
"$PYTHON_BIN" daily_update.py
