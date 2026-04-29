#!/bin/bash
# RhinoFinance 每日任务启动脚本

# 设置 PATH 确保能找到 pyenv python / yt-dlp
export PATH="/Users/wl/.pyenv/shims:/opt/homebrew/bin:/usr/local/bin:$PATH"
export LANG=en_US.UTF-8

# 代理配置
export HTTPS_PROXY=http://127.0.0.1:5782
export HTTP_PROXY=http://127.0.0.1:5782

LOG_DIR="/tmp/openclaw"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/rhino_finance_daily.log"

echo "" >> "$LOG_FILE"
echo "════════════════════════════════════════" >> "$LOG_FILE"
/Users/wl/.pyenv/shims/python3 /Users/wl/rhino_finance_daily.py >> "$LOG_FILE" 2>&1