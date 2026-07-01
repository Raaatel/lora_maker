@echo off
setlocal EnableDelayedExpansion

echo.
echo  LoRA Maker - Installing...
echo  ================================
echo.

REM ── Path safety check ──────────────────────────────────────────
set "CWD=%CD%"
echo !CWD! | findstr /i "AppData Packages LocalCache" > nul
if not errorlevel 1 (
    echo [WARNING] Detected restricted path:
    echo           !CWD!
    echo.
    echo  Windows blocks venv creation inside AppData/Packages folders.
    echo  Please move the lora-maker folder to a simpler path, for example:
    echo.
    echo    C:\lora-maker
    echo    D:\lora-maker
    echo.
    echo  Then run install.bat again from there.
    echo.
    pause
    exit /b 1
)

REM ── Check Python ───────────────────────────────────────────────
python --version > nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found.
    echo         Please install Python 3.10+ from https://www.python.org
    echo         Make sure to check "Add Python to PATH" during install.
    pause
    exit /b 1
)
for /f "tokens=2" %%v in ('python --version') do set PY_VER=%%v
echo [OK] Python %PY_VER% found

REM ── Check Python version (need 3.10+) ──────────────────────────
for /f "tokens=1,2 delims=." %%a in ("%PY_VER%") do (
    set PY_MAJOR=%%a
    set PY_MINOR=%%b
)
if %PY_MAJOR% LSS 3 (
    echo [ERROR] Python 3.10+ required, found %PY_VER%
    pause
    exit /b 1
)
if %PY_MAJOR% EQU 3 if %PY_MINOR% LSS 10 (
    echo [ERROR] Python 3.10+ required, found %PY_VER%
    pause
    exit /b 1
)

REM ── Create virtualenv ──────────────────────────────────────────
if not exist "venv" (
    echo.
    echo [1/4] Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo [WARN] Standard venv failed, trying --without-pip fallback...
        python -m venv --without-pip venv
        if errorlevel 1 (
            echo [ERROR] Failed to create virtual environment.
            echo         Try running: python -m pip install --upgrade virtualenv
            pause
            exit /b 1
        )
        REM Manually install pip via get-pip.py
        echo        Downloading pip installer...
        powershell -Command "Invoke-WebRequest -Uri https://bootstrap.pypa.io/get-pip.py -OutFile get-pip.py" > nul 2>&1
        if exist "get-pip.py" (
            python get-pip.py --quiet
            del get-pip.py
        )
    )
    echo [OK] Virtual environment created
) else (
    echo [OK] Virtual environment already exists
)

REM ── Activate and upgrade pip ───────────────────────────────────
call venv\Scripts\activate.bat
echo.
echo [2/4] Upgrading pip...
python -m pip install --upgrade pip --quiet

REM ── Install PyTorch with CUDA 12.1 ────────────────────────────
echo.
echo [3/4] Installing PyTorch (CUDA 12.1)... This may take a while.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --quiet
if errorlevel 1 (
    echo [WARN] CUDA install failed, trying CPU-only PyTorch...
    pip install torch torchvision --quiet
)
echo [OK] PyTorch installed

REM ── Install remaining packages ─────────────────────────────────
echo.
echo [4/4] Installing dependencies...
pip install fastapi "uvicorn[standard]" jinja2 aiofiles aiosqlite python-multipart websockets pyyaml --quiet
if errorlevel 1 (
    echo [ERROR] Failed to install web framework packages
    pause
    exit /b 1
)
pip install diffusers transformers accelerate safetensors peft bitsandbytes --quiet
pip install opencv-python Pillow tqdm numpy huggingface-hub timm pandas psutil toml einops imagesize --quiet
pip install onnxruntime-gpu --quiet 2>nul
if errorlevel 1 (
    pip install onnxruntime --quiet
)
echo [OK] Dependencies installed

REM ── kohya-sd-scripts ──────────────────────────────────────────
if not exist "kohya-sd-scripts" (
    echo.
    echo [5/5] Cloning kohya-sd-scripts (training backend)...
    where git > nul 2>&1
    if not errorlevel 1 (
        git clone https://github.com/kohya-ss/sd-scripts kohya-sd-scripts --quiet
        if not errorlevel 1 (
            pip install -r kohya-sd-scripts\requirements.txt --quiet
            echo [OK] kohya-sd-scripts installed
        ) else (
            echo [WARN] git clone failed - check your internet connection
            echo        You can clone manually later:
            echo        git clone https://github.com/kohya-ss/sd-scripts kohya-sd-scripts
        )
    ) else (
        echo [WARN] git not found. Install Git from https://git-scm.com
        echo        Then run:  git clone https://github.com/kohya-ss/sd-scripts kohya-sd-scripts
        echo                   pip install -r kohya-sd-scripts\requirements.txt
    )
) else (
    echo [OK] kohya-sd-scripts already present
)

REM ── Electron (optional) ───────────────────────────────────────
if exist "electron\package.json" (
    where node > nul 2>&1
    if not errorlevel 1 (
        echo.
        echo [Optional] Installing Electron...
        cd electron
        call npm install --quiet 2>nul
        cd ..
        echo [OK] Electron installed
    ) else (
        echo [INFO] Node.js not found - skipping Electron (web mode works fine)
    )
)

echo.
echo  ================================
echo  Installation complete!
echo.
echo  To start (web browser):    start.bat
echo  To start (desktop app):    start_electron.bat
echo  ================================
echo.
pause
