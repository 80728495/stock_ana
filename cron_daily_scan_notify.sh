#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/Users/wl/stock_ana"
PYTHON_BIN="/Users/wl/.pyenv/shims/python3"

cd "$PROJECT_DIR"

# 1) Vegas 扫描 + Gemini 汇总
SCAN_EXIT=0
"$PYTHON_BIN" vegas_mid_daily_scan.py || SCAN_EXIT=$?

# 2) 扫描完成后推送给 main agent（无论扫描成功或失败都推送）
"$PYTHON_BIN" notify_daily_scan_result.py --scan-exit-code "$SCAN_EXIT"

exit "$SCAN_EXIT"
