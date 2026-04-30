# daily_scan_notify.ps1 - Windows Task Scheduler equivalent of cron_daily_scan_notify.sh
# Usage: powershell -ExecutionPolicy Bypass -File daily_scan_notify.ps1

$ErrorActionPreference = "Stop"

$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$PythonBin  = Join-Path $ProjectDir ".venv\Scripts\python.exe"
$StampFile  = Join-Path $ProjectDir "data\logs\.last_scan_stamp"
$CacheUs    = Join-Path $ProjectDir "data\cache\us"
$CacheHk    = Join-Path $ProjectDir "data\cache\hk"

Set-Location $ProjectDir

# Check data freshness
if (Test-Path $StampFile) {
    $stampTime = (Get-Item $StampFile).LastWriteTime
    $hasNew = $false
    foreach ($dir in @($CacheUs, $CacheHk)) {
        if (Test-Path $dir) {
            $newer = Get-ChildItem -Path $dir -Filter "*.parquet" -Recurse -ErrorAction SilentlyContinue |
                     Where-Object { $_.LastWriteTime -gt $stampTime } |
                     Select-Object -First 1
            if ($newer) { $hasNew = $true; break }
        }
    }
    if (-not $hasNew) {
        Write-Host "[daily-scan] No new data since last scan, skipping."
        exit 0
    }
}

Write-Host "[daily-scan] New data detected, starting scan..."

# Check proxy/network reachability to Google before running Gemini-dependent scan
try {
    $null = Invoke-WebRequest -Uri "https://gemini.google.com" -UseDefaultCredentials -TimeoutSec 10 -ErrorAction Stop
} catch {
    Write-Host "[daily-scan] WARNING: gemini.google.com unreachable (VPN may not be active). Proceeding anyway..."
}

# Ensure log dir exists
$logDir = Join-Path $ProjectDir "data\logs"
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir | Out-Null }

# 1) Vegas scan + Gemini analysis
$scanExit = 0
try {
    & $PythonBin vegas_mid_daily_scan.py --list combined
    $scanExit = $LASTEXITCODE
} catch {
    $scanExit = 1
    Write-Host "[daily-scan] Scan error: $_"
}

# 2) Notify main agent via Feishu (always runs regardless of scan exit code)
& $PythonBin notify_daily_scan_result.py --scan-exit-code $scanExit --no-email

# 3) Watchlist Vegas touch scan (Mid + Long) → Feishu push if any signals
try {
    & $PythonBin watchlist_vegas_scan.py
} catch {
    Write-Host "[daily-scan] watchlist_vegas_scan error: $_"
}

# 4) Update timestamp
Set-Content -Path $StampFile -Value (Get-Date -Format "o") -Encoding ASCII

exit $scanExit

