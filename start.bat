@echo off
echo.
echo  LoRA Maker - Starting...
echo  http://localhost:7860
echo.

if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
)

python app.py
