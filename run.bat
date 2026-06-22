@echo off
title LoRA Maker
chcp 65001 >nul
echo.
echo  LoRA Maker
echo  ================================
echo.

cd /d "%~dp0"

REM Python 있는지 확인
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Install Python 3.10+ from https://python.org
    pause
    exit /b 1
)

REM 의존성 설치 및 서버 시작
python setup_and_run.py
if errorlevel 1 (
    echo.
    echo [ERROR] 오류가 발생했습니다. 위 메시지를 확인하세요.
    pause
)
