#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/Users/wl/stock_ana"
PYTHON_BIN="$(command -v python3)"

cd "$PROJECT_DIR"

# 建议先同步自动列表，再执行全量每日更新
"$PYTHON_BIN" daily_update.py --lists
"$PYTHON_BIN" daily_update.py
