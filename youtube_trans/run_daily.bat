@echo off
chcp 65001 >nul
setlocal

set "SCRIPT_DIR=%~dp0"
if "%SCRIPT_DIR:~-1%"=="\" set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

if exist "%SCRIPT_DIR%\.env" (
  for /f "usebackq tokens=1,* delims==" %%A in ("%SCRIPT_DIR%\.env") do (
    if not "%%A"=="" if /I not "%%A:~0,1"=="#" set "%%A=%%B"
  )
)

if not defined RHINO_PYTHON set "RHINO_PYTHON=%SCRIPT_DIR%\..\.venv\Scripts\python.exe"
if not exist "%RHINO_PYTHON%" set "RHINO_PYTHON=python"

set "PYTHONIOENCODING=utf-8"
set "HF_HUB_DISABLE_SYMLINKS_WARNING=1"

set "LOG_DIR=%TEMP%\openclaw"
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
set "LOG_FILE=%LOG_DIR%\rhino_finance_daily.log"

cd /d "%SCRIPT_DIR%"
echo.>> "%LOG_FILE%"
echo ========================================>> "%LOG_FILE%"
"%RHINO_PYTHON%" "%SCRIPT_DIR%\rhino_finance_daily.py" >> "%LOG_FILE%" 2>&1

endlocal
