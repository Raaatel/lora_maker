@echo off
echo.
echo  LoRA Maker (Desktop) - Starting...
echo.

if not exist "electron\node_modules" (
    echo [ERROR] Electron not installed. Run install.bat first.
    pause
    exit /b 1
)

cd electron
npx electron .
