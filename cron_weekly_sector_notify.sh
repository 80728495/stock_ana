#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/Users/wl/stock_ana"
PYTHON_BIN="/Users/wl/.pyenv/shims/python3"

cd "$PROJECT_DIR"

WORKFLOW_EXIT=0
"$PYTHON_BIN" -m stock_ana.workflows.weekly_sector_report || WORKFLOW_EXIT=$?

"$PYTHON_BIN" notify_weekly_sector_report.py --workflow-exit-code "$WORKFLOW_EXIT"

exit "$WORKFLOW_EXIT"
