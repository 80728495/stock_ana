$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$venvPython = Join-Path $scriptDir "..\.venv\Scripts\python.exe"
$python = if (Test-Path $venvPython) { $venvPython } else { "python" }

Write-Host "Using Python: $python"
& $python -m pip install --upgrade pip
& $python -m pip install -r (Join-Path $scriptDir "requirements.txt")

if (-not (Get-Command yt-dlp -ErrorAction SilentlyContinue)) {
    & $python -m pip install yt-dlp
}

if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    Write-Host "ffmpeg not found in PATH. Install with: winget install Gyan.FFmpeg" -ForegroundColor Yellow
} else {
    Write-Host "ffmpeg detected"
}

if (-not (Test-Path (Join-Path $scriptDir ".env"))) {
    Copy-Item (Join-Path $scriptDir ".env.example") (Join-Path $scriptDir ".env")
    Write-Host "Created youtube_trans/.env from .env.example"
}

Write-Host "Install complete"
