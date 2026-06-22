"""
LoRA Maker - setup_and_run.py
모든 의존성을 자동으로 설치하고 서버를 시작합니다.
"""
import sys, os, subprocess, shutil
from pathlib import Path

HERE = Path(__file__).parent.resolve()
VENV = HERE / "venv"
VENV_PY  = VENV / "Scripts" / "python.exe"
VENV_PIP = VENV / "Scripts" / "pip.exe"


def run(cmd, **kw):
    return subprocess.run(cmd, **kw)

def pip_install(*packages, upgrade=False):
    cmd = [str(VENV_PIP), "install"] + list(packages)
    if upgrade:
        cmd.append("--upgrade")
    result = run(cmd)
    return result.returncode == 0

def is_importable(*modules):
    for m in modules:
        r = run([str(VENV_PY), "-c", f"import {m}"], capture_output=True)
        if r.returncode != 0:
            return False
    return True

def section(msg):
    print(f"\n[{msg}]")

# ── 1. venv ──────────────────────────────────────────────────────────────────
section("가상환경 확인")
if not VENV_PY.exists():
    print("  venv 생성 중...")
    run([sys.executable, "-m", "venv", str(VENV)], check=True)
    print("  venv 생성 완료")
else:
    print("  venv OK")

# pip 업그레이드
run([str(VENV_PY), "-m", "pip", "install", "--upgrade", "pip", "--quiet"])

# ── 2. 웹 서버 패키지 ────────────────────────────────────────────────────────
section("웹 서버 패키지 확인")
web_pkgs = ["fastapi", "uvicorn", "jinja2", "aiofiles", "aiosqlite",
            "multipart", "websockets", "yaml"]
if not is_importable(*web_pkgs):
    print("  설치 중...")
    pip_install("fastapi", "uvicorn[standard]", "jinja2", "aiofiles",
                "aiosqlite", "python-multipart", "websockets", "pyyaml")
else:
    print("  OK")

# ── 3. PyTorch ───────────────────────────────────────────────────────────────
section("PyTorch 확인")
if not is_importable("torch"):
    print("  GPU 감지 중...")
    sm = 0
    try:
        r = run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=10)
        if r.returncode == 0 and r.stdout.strip():
            major, minor = r.stdout.strip().split("\n")[0].strip().split(".")
            sm = int(major) * 10 + int(minor)
    except Exception:
        pass

    if sm >= 120:
        print(f"  RTX 50xx 감지 (sm_{sm}) → CUDA 12.8")
        pip_install("torch>=2.6.0", "torchvision>=0.21.0",
                    "--index-url", "https://download.pytorch.org/whl/cu128")
    elif sm >= 75:
        print(f"  GPU 감지 (sm_{sm}) → CUDA 12.1")
        pip_install("torch>=2.1.0", "torchvision>=0.16.0",
                    "--index-url", "https://download.pytorch.org/whl/cu121")
    elif sm > 0:
        print(f"  구형 GPU (sm_{sm}) → CUDA 11.8")
        pip_install("torch>=2.1.0", "torchvision>=0.16.0",
                    "--index-url", "https://download.pytorch.org/whl/cu118")
    else:
        print("  GPU 없음 → CPU 버전")
        pip_install("torch", "torchvision")
    print("  PyTorch 설치 완료")
else:
    print("  OK")

# ── 4. ML 패키지 ─────────────────────────────────────────────────────────────
section("ML 패키지 확인")
ml_pkgs_check = ["diffusers", "transformers", "safetensors", "cv2",
                 "PIL", "accelerate", "peft"]
if not is_importable(*ml_pkgs_check):
    print("  설치 중...")
    pip_install("diffusers>=0.28.0", "transformers>=4.40.0,<5.0.0", "accelerate",
                "safetensors", "peft", "bitsandbytes", "opencv-python",
                "Pillow", "tqdm", "numpy", "huggingface-hub", "timm",
                "pandas", "psutil", "toml", "einops", "imagesize")
    print("  완료")
else:
    # diffusers 버전만 확인해서 구버전이면 업그레이드
    r = run([str(VENV_PY), "-c",
             "import diffusers,transformers; dv=list(map(int,diffusers.__version__.split('.')[:2])); tv=list(map(int,transformers.__version__.split('.')[:2])); exit(0 if dv>=[0,28] and tv<[5,0] else 1)"],
            capture_output=True)
    if r.returncode != 0:
        print("  diffusers/transformers 버전 조정 중 (transformers 5.x → 4.x 포함)...")
        pip_install("diffusers>=0.28.0", "transformers>=4.40.0,<5.0.0", upgrade=True)
        print("  완료")
    else:
        print("  OK")

# ── 5. onnxruntime ───────────────────────────────────────────────────────────
if not is_importable("onnxruntime"):
    print("  onnxruntime 설치 중...")
    ok = pip_install("onnxruntime-gpu")
    if not ok:
        pip_install("onnxruntime")

# ── 6. kohya-sd-scripts ──────────────────────────────────────────────────────
section("kohya-sd-scripts 확인")
kohya = HERE / "kohya-sd-scripts"
if not (kohya / "train_network.py").exists():
    git = shutil.which("git")
    if git:
        print("  클론 중... (1-2분 소요)")
        r = run(["git", "clone", "https://github.com/kohya-ss/sd-scripts",
                 str(kohya), "--depth=1"])
        if r.returncode == 0:
            req = kohya / "requirements.txt"
            if req.exists():
                pip_install("-r", str(req))
            print("  완료")
        else:
            print("  [WARN] 클론 실패 - 학습 기능 사용 불가")
    else:
        print("  [WARN] git 없음 - https://git-scm.com 에서 설치 후 재시작")
else:
    print("  OK")

# ── 7. 서버 시작 ─────────────────────────────────────────────────────────────
print()
print("=" * 40)
print("  모든 의존성 OK")
print("  브라우저: http://localhost:7860")
print("  종료: Ctrl+C")
print("=" * 40)
print()

os.chdir(str(HERE))
os.execv(str(VENV_PY), [str(VENV_PY), "app.py"])
