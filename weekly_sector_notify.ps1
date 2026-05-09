# weekly_sector_notify.ps1 - Windows Task Scheduler equivalent of cron_weekly_sector_notify.sh
# Runs every Saturday at 09:00 via StockAna_WeeklySector scheduled task.
# US price data is already kept fresh by the daily update task, so --skip-update is passed.

$ErrorActionPreference = "Stop"

$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$PythonBin  = Join-Path $ProjectDir ".venv\Scripts\python.exe"

Set-Location $ProjectDir

$workflowExit = 0
try {
    & $PythonBin -m stock_ana.workflows.weekly_sector_report --skip-update
    $workflowExit = $LASTEXITCODE
} catch {
    $workflowExit = 1
    Write-Host "[weekly-sector] Workflow error: $_"
}

& $PythonBin notify_weekly_sector_report.py --workflow-exit-code $workflowExit --no-email

exit $workflowExit
