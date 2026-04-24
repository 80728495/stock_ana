#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/Users/wl/stock_ana"
PYTHON_BIN="/Users/wl/.pyenv/shims/python3"
STAMP_FILE="$PROJECT_DIR/data/logs/.last_scan_stamp"

cd "$PROJECT_DIR"

# ── 数据新鲜度检测 ──────────────────────────────────────────────
# 取 US + HK cache 目录中是否有比上次扫描更新的 parquet 文件
# 使用 find -quit 避免 pipefail + SIGPIPE 问题
if [[ -f "$STAMP_FILE" ]]; then
    HAS_NEW=$(find \
        "$PROJECT_DIR/data/cache/us" \
        "$PROJECT_DIR/data/cache/hk" \
        -name "*.parquet" -type f \
        -newer "$STAMP_FILE" -print -quit 2>/dev/null)
    if [[ -z "$HAS_NEW" ]]; then
        echo "[daily-scan] 数据自上次扫描以来未更新，跳过本次扫描。"
        exit 0
    fi
fi

echo "[daily-scan] 检测到新数据，开始扫描..."

# 1) Vegas 扫描 + Gemini 汇总（先美股 tech，再港股）
SCAN_EXIT=0
"$PYTHON_BIN" vegas_mid_daily_scan.py --list combined || SCAN_EXIT=$?

# 2) 扫描完成后推送给 main agent（无论扫描成功或失败都推送）
"$PYTHON_BIN" notify_daily_scan_result.py --scan-exit-code "$SCAN_EXIT" --no-email

# 3) 更新时间戳（只在扫描实际执行后才更新）
touch "$STAMP_FILE"

exit "$SCAN_EXIT"
