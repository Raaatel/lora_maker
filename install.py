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
section("3/5  Installing PyTorch (GPU auto-detect)")
print("  Detecting GPU...")

def get_compute_capability():
    """Return (major, minor) compute capability, or None if no NVIDIA GPU."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            cap = result.stdout.strip().split("\n")[0].strip()
            if cap:
                major, minor = cap.split(".")
                return int(major), int(minor)
    except Exception:
        pass
    return None

def get_torch_index_url():
    """Pick the right wheel index for the detected GPU."""
    cap = get_compute_capability()
    if cap is None:
        print("  [INFO] No NVIDIA GPU detected — installing CPU PyTorch")
        return None, "cpu"
    major, minor = cap
    sm = major * 10 + minor
    gpu_str = f"sm_{sm}"
    print(f"  [INFO] Detected GPU compute capability: {gpu_str}")

    if sm >= 120:
        # Blackwell (RTX 50xx) — needs CUDA 12.8 + PyTorch 2.6+
        print("  [INFO] Blackwell GPU (RTX 50xx) detected → CUDA 12.8 / PyTorch 2.6+")
        return "https://download.pytorch.org/whl/cu128", "cu128"
    elif sm >= 89:
        # Ada Lovelace / Hopper (RTX 40xx, H100) — CUDA 12.1
        print("  [INFO] Ada/Hopper GPU detected → CUDA 12.1")
        return "https://download.pytorch.org/whl/cu121", "cu121"
    elif sm >= 80:
        # Ampere (RTX 30xx, A100) — CUDA 12.1
        print("  [INFO] Ampere GPU detected → CUDA 12.1")
        return "https://download.pytorch.org/whl/cu121", "cu121"
    elif sm >= 75:
        # Turing (RTX 20xx, GTX 16xx) — CUDA 12.1
        print("  [INFO] Turing GPU detected → CUDA 12.1")
        return "https://download.pytorch.org/whl/cu121", "cu121"
    else:
        print(f"  [INFO] Older GPU (sm_{sm}) → CUDA 11.8")
        return "https://download.pytorch.org/whl/cu118", "cu118"

index_url, cuda_tag = get_torch_index_url()
torch_pkg = "torch>=2.6.0" if cuda_tag == "cu128" else "torch>=2.1.0"
torchvision_pkg = "torchvision>=0.21.0" if cuda_tag == "cu128" else "torchvision>=0.16.0"

print(f"  Installing PyTorch for {cuda_tag}... (5-10 min)")
try:
    if index_url:
        pip("install", torch_pkg, torchvision_pkg,
            "--index-url", index_url, "--quiet")
    else:
        pip("install", torch_pkg, torchvision_pkg, "--quiet")
    print(f"[OK] PyTorch ({cuda_tag}) installed")
except subprocess.CalledProcessError:
    print("[WARN] GPU build failed, falling back to CPU PyTorch...")
    pip("install", "torch", "torchvision", "--quiet")
    print("[OK] PyTorch (CPU) installed")

# ── Other dependencies ──────────────────────────────────
section("4/5  Installing dependencies")
groups = [
    ["fastapi", "uvicorn[standard]", "jinja2", "aiofiles", "aiosqlite",
     "python-multipart", "websockets", "pyyaml"],
    ["diffusers>=0.28.0", "transformers>=4.40.0", "accelerate", "safetensors", "peft", "bitsandbytes"],
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
