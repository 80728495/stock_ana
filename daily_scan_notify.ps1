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

# Wait for Google/Gemini to be reachable (proxy may need a moment to become active)
$maxWaitSec = 180   # 最多等 3 分钟
$intervalSec = 20
$elapsed = 0
$reachable = $false
while (-not $reachable -and $elapsed -lt $maxWaitSec) {
    try {
        $null = Invoke-WebRequest -Uri "https://gemini.google.com" -UseBasicParsing -TimeoutSec 10 -ErrorAction Stop
        $reachable = $true
        Write-Host "[daily-scan] gemini.google.com reachable."
    } catch {
        Write-Host "[daily-scan] gemini.google.com unreachable, waiting ${intervalSec}s... (${elapsed}/${maxWaitSec}s)"
        Start-Sleep -Seconds $intervalSec
        $elapsed += $intervalSec
    }
}
if (-not $reachable) {
    Write-Host "[daily-scan] WARNING: gemini.google.com still unreachable after ${maxWaitSec}s, proceeding anyway..."
}

# Ensure log dir exists
$logDir = Join-Path $ProjectDir "data\logs"
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir | Out-Null }

# 1) Watchlist Vegas touch scan (Mid + Long) — 不依赖 Gemini，最先运行，完成即推送飞书
Write-Host "[daily-scan] Step 1/3: Watchlist Vegas scan..."
try {
    & $PythonBin watchlist_vegas_scan.py
} catch {
    Write-Host "[daily-scan] watchlist_vegas_scan error: $_"
}

# 2) 美股 scan + Gemini → 完成后立即发飞书（含每日数据更新状态）
Write-Host "[daily-scan] Step 2/3: US scan + Gemini..."
$usScanExit = 0
try {
    & $PythonBin vegas_mid_daily_scan.py --list tech
    $usScanExit = $LASTEXITCODE
} catch {
    $usScanExit = 1
    Write-Host "[daily-scan] US scan error: $_"
}
& $PythonBin notify_daily_scan_result.py --market us --scan-exit-code $usScanExit

# 3) 港股 scan + Gemini → 完成后立即发飞书（跳过重复的数据更新状态）
Write-Host "[daily-scan] Step 3/3: HK scan + Gemini..."
$hkScanExit = 0
try {
    & $PythonBin vegas_mid_daily_scan.py --list hk
    $hkScanExit = $LASTEXITCODE
} catch {
    $hkScanExit = 1
    Write-Host "[daily-scan] HK scan error: $_"
}
& $PythonBin notify_daily_scan_result.py --market hk --skip-update

# 4) Update timestamp
Set-Content -Path $StampFile -Value (Get-Date -Format "o") -Encoding ASCII

# 整体退出码：任一非零则返回非零
$scanExit = if ($usScanExit -ne 0) { $usScanExit } elseif ($hkScanExit -ne 0) { $hkScanExit } else { 0 }
exit $scanExit

