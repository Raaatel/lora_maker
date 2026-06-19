"""
LoRA Maker - Start (web mode)
Run with:  python start.py
"""
import sys
import subprocess
from pathlib import Path

HERE = Path(__file__).parent.resolve()
VENV_PY = HERE / "venv" / "Scripts" / "python.exe"

if not VENV_PY.exists():
    print("[ERROR] venv not found. Run install.py first.")
    input("Press Enter to exit...")
    sys.exit(1)

print("LoRA Maker starting at http://localhost:7860 ...")
print("Open your browser and go to: http://localhost:7860")
print("(Press Ctrl+C to stop)")
subprocess.run([str(VENV_PY), str(HERE / "app.py")])
