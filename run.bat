@echo off
title LoRA Maker
echo.
echo  LoRA Maker - Starting...
echo  ================================
echo.

cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" (
    echo [ERROR] venv\Scripts\python.exe not found
    echo         Run install.py first!
    echo.
    pause
    exit /b 1
)

if not exist "app.py" (
    echo [ERROR] app.py not found
    echo         Make sure you are running from the lora-maker folder.
    echo         Current folder: %CD%
    echo.
    pause
    exit /b 1
)

echo  Open browser: http://localhost:7860
echo  Press Ctrl+C to stop
echo.

venv\Scripts\python.exe app.py
if errorlevel 1 (
    echo.
    echo [ERROR] App crashed - see error above
    pause
)
