#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$PROJECT_DIR/.venv/bin/python3" ]]; then
    PYTHON_BIN="$PROJECT_DIR/.venv/bin/python3"
else
    PYTHON_BIN="$(command -v python3)"
fi

cd "$PROJECT_DIR"

# Full pipeline: refresh weekly indicators -> weekly short vegas scan -> Codex analysis -> PDF -> Feishu send.
"$PYTHON_BIN" weekly_vegas_short_notify.py --list combined --lookback 1
