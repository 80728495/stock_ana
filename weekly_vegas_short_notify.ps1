# weekly_vegas_short_notify.ps1 - Windows Task Scheduler equivalent of cron_weekly_vegas_short_notify.sh
# Runs every Saturday at 09:10 via StockAna_WeeklyVegasShort scheduled task.
# Flow: refresh weekly indicators -> weekly short vegas scan -> Gemini analysis -> PDF -> Feishu send.

$ErrorActionPreference = "Stop"

$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$PythonBin  = Join-Path $ProjectDir ".venv\Scripts\python.exe"

Set-Location $ProjectDir

# Full pipeline (includes weekly indicator refresh by default)
& $PythonBin weekly_vegas_short_notify.py --list combined --lookback 1

exit $LASTEXITCODE
