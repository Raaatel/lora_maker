"""
LoRA Maker - Installer
Run with:  python install.py
"""
import os
import sys
import subprocess
import shutil
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent.resolve()
VENV = HERE / "venv"
PY = sys.executable

def run(cmd, check=True, **kw):
    print(f"  > {' '.join(str(c) for c in cmd)}")
    return subprocess.run(cmd, check=check, **kw)

def pip(*args):
    run([VENV / "Scripts" / "pip", *args])

def section(title):
    print(f"\n{'='*50}")
    print(f"  {title}")
    print('='*50)

# ── Path warning ────────────────────────────────────────
section("LoRA Maker - Installer")

bad_keywords = ["AppData", "Packages", "LocalCache", "Temp"]
path_str = str(HERE)
if any(k in path_str for k in bad_keywords):
    print(f"\n[WARNING] You are running from a restricted path:")
    print(f"  {HERE}")
    print()
    print("  Windows may block venv/exe creation here.")
    print("  Recommended: move lora-maker to C:\\lora-maker")
    print()
    ans = input("  Continue anyway? (y/N): ").strip().lower()
    if ans != "y":
        print("  Aborting. Please move the folder and retry.")
        input("\nPress Enter to exit...")
        sys.exit(1)

# ── Python version check ────────────────────────────────
section("1/5  Checking Python")
if sys.version_info < (3, 10):
    print(f"[ERROR] Python 3.10+ required, found {sys.version}")
    input("Press Enter to exit...")
    sys.exit(1)
print(f"[OK] Python {sys.version.split()[0]}")

# ── Virtual environment ─────────────────────────────────
section("2/5  Creating virtual environment")
if not VENV.exists():
    try:
        run([PY, "-m", "venv", str(VENV)])
        print("[OK] venv created")
    except subprocess.CalledProcessError:
        print("[WARN] Standard venv failed, trying --without-pip ...")
        try:
            run([PY, "-m", "venv", "--without-pip", str(VENV)])
            # Install pip via get-pip.py
            get_pip = HERE / "get-pip.py"
            print("  Downloading pip installer ...")
            urllib.request.urlretrieve("https://bootstrap.pypa.io/get-pip.py", get_pip)
            run([str(VENV / "Scripts" / "python"), str(get_pip), "--quiet"])
            get_pip.unlink(missing_ok=True)
            print("[OK] venv created (fallback mode)")
        except Exception as e:
            print(f"[ERROR] Could not create venv: {e}")
            print("  Try:  pip install virtualenv  then  python install.py")
            input("Press Enter to exit...")
            sys.exit(1)
else:
    print("[OK] venv already exists")

VENV_PY  = VENV / "Scripts" / "python"
VENV_PIP = VENV / "Scripts" / "pip"

# ── Upgrade pip ─────────────────────────────────────────
run([str(VENV_PY), "-m", "pip", "install", "--upgrade", "pip", "--quiet"])

# ── PyTorch ─────────────────────────────────────────────
section("3/5  Installing PyTorch (CUDA 12.1)")
print("  This can take 5-10 minutes depending on your connection...")
try:
    pip("install", "torch", "torchvision",
        "--index-url", "https://download.pytorch.org/whl/cu121", "--quiet")
    print("[OK] PyTorch (CUDA) installed")
except subprocess.CalledProcessError:
    print("[WARN] CUDA build failed, installing CPU version...")
    pip("install", "torch", "torchvision", "--quiet")
    print("[OK] PyTorch (CPU) installed")

# ── Other dependencies ──────────────────────────────────
section("4/5  Installing dependencies")
groups = [
    ["fastapi", "uvicorn[standard]", "jinja2", "aiofiles", "aiosqlite",
     "python-multipart", "websockets", "pyyaml"],
    ["diffusers", "transformers", "accelerate", "safetensors", "peft", "bitsandbytes"],
    ["opencv-python", "Pillow", "tqdm", "numpy", "huggingface-hub",
     "timm", "pandas", "psutil", "toml", "einops", "imagesize"],
]
for group in groups:
    pip("install", *group, "--quiet")

# onnxruntime: try GPU first, fall back to CPU
try:
    pip("install", "onnxruntime-gpu", "--quiet")
except subprocess.CalledProcessError:
    pip("install", "onnxruntime", "--quiet")

print("[OK] All dependencies installed")

# ── kohya-sd-scripts ────────────────────────────────────
section("5/5  kohya-sd-scripts (training backend)")
kohya = HERE / "kohya-sd-scripts"
if not kohya.exists():
    git = shutil.which("git")
    if git:
        print("  Cloning repository...")
        run(["git", "clone",
             "https://github.com/kohya-ss/sd-scripts",
             str(kohya), "--quiet"])
        req = kohya / "requirements.txt"
        if req.exists():
            pip("install", "-r", str(req), "--quiet")
        print("[OK] kohya-sd-scripts cloned and installed")
    else:
        print("[WARN] git not found.")
        print("  Install Git from https://git-scm.com then run install.py again,")
        print("  or manually clone:")
        print("    git clone https://github.com/kohya-ss/sd-scripts kohya-sd-scripts")
else:
    print("[OK] kohya-sd-scripts already present")

# ── Electron (optional) ─────────────────────────────────
electron_pkg = HERE / "electron" / "package.json"
if electron_pkg.exists():
    node = shutil.which("node")
    npm  = shutil.which("npm")
    if node and npm:
        print("\n[Optional] Installing Electron desktop wrapper...")
        run(["npm", "install", "--prefix", str(HERE / "electron"), "--quiet"],
            check=False)
        print("[OK] Electron installed")
    else:
        print("\n[INFO] Node.js not found - skipping Electron (web mode works fine)")

# ── Done ────────────────────────────────────────────────
print(f"""
{'='*50}
  Installation complete!

  Web mode:      python start.py
               (then open http://localhost:7860)

  Desktop app:   cd electron && npx electron .
{'='*50}
""")
input("Press Enter to exit...")
