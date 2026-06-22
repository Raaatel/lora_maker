import subprocess, sys

def get_sm():
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10
        )
        if r.returncode == 0 and r.stdout.strip():
            major, minor = r.stdout.strip().split("\n")[0].strip().split(".")
            return int(major) * 10 + int(minor)
    except Exception:
        pass
    return 0

sm = get_sm()
print(f"  GPU compute capability: sm_{sm}" if sm else "  No NVIDIA GPU detected")

if sm >= 120:
    url = "https://download.pytorch.org/whl/cu128"
    pkg = ["torch>=2.6.0", "torchvision>=0.21.0"]
    print("  -> CUDA 12.8 (RTX 50xx Blackwell)")
elif sm >= 75:
    url = "https://download.pytorch.org/whl/cu121"
    pkg = ["torch>=2.1.0", "torchvision>=0.16.0"]
    print("  -> CUDA 12.1")
elif sm > 0:
    url = "https://download.pytorch.org/whl/cu118"
    pkg = ["torch>=2.1.0", "torchvision>=0.16.0"]
    print("  -> CUDA 11.8")
else:
    url = None
    pkg = ["torch", "torchvision"]
    print("  -> CPU only")

cmd = [sys.executable, "-m", "pip", "install"] + pkg + ["--quiet"]
if url:
    cmd += ["--index-url", url]

print(f"  Installing: {' '.join(pkg)}")
subprocess.run(cmd, check=True)
print("  [OK] PyTorch installed")
