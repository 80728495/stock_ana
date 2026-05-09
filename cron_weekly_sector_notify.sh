#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$PROJECT_DIR/.venv/bin/python3" ]]; then
    PYTHON_BIN="$PROJECT_DIR/.venv/bin/python3"
else
    PYTHON_BIN="$(command -v python3)"
fi

cd "$PROJECT_DIR"

WORKFLOW_EXIT=0
"$PYTHON_BIN" -m stock_ana.workflows.weekly_sector_report --skip-update || WORKFLOW_EXIT=$?

"$PYTHON_BIN" notify_weekly_sector_report.py --workflow-exit-code "$WORKFLOW_EXIT"

exit "$WORKFLOW_EXIT"
