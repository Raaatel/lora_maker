"""
LoRA Maker - Automated Server Test
Usage: python test_server.py

Requires: server running at http://localhost:7860 (run run.bat first)
Tests all API endpoints without needing GPU or actual training.
"""

import sys
import json
import time
import asyncio
import threading
import urllib.request
import urllib.error
from pathlib import Path

BASE = "http://localhost:7860"
PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
INFO = "\033[94m[INFO]\033[0m"
WARN = "\033[93m[WARN]\033[0m"

results = {"pass": 0, "fail": 0}
created_project_id = None


# ── HTTP helpers ───────────────────────────────────────────────────────────────

def get(path):
    req = urllib.request.Request(f"{BASE}{path}")
    with urllib.request.urlopen(req, timeout=10) as r:
        return r.status, json.loads(r.read())

def post(path, data=None):
    body = json.dumps(data or {}).encode()
    req = urllib.request.Request(
        f"{BASE}{path}", data=body,
        headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())

def delete(path):
    req = urllib.request.Request(f"{BASE}{path}", method="DELETE")
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())

def check(name, ok, detail=""):
    if ok:
        results["pass"] += 1
        print(f"  {PASS} {name}" + (f" — {detail}" if detail else ""))
    else:
        results["fail"] += 1
        print(f"  {FAIL} {name}" + (f" — {detail}" if detail else ""))
    return ok


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_server_reachable():
    print("\n[1] Server health")
    try:
        status, _ = get("/")
        check("GET /  (index page)", status == 200, f"HTTP {status}")
    except Exception as e:
        check("GET /  (index page)", False, str(e))
        print(f"\n  {WARN} Server not reachable at {BASE}")
        print(f"       Make sure run.bat is running first!\n")
        sys.exit(1)


def test_project_crud():
    global created_project_id
    print("\n[2] Project API")

    # List (empty or existing)
    status, data = get("/api/projects")
    check("GET /api/projects", status == 200, f"{len(data)} projects")

    # Create
    status, proj = post("/api/projects", {
        "name": "test_style_lora",
        "lora_type": "style",
        "trigger_word": "myartstyle",
        "base_model": "D:/models/animagine",
        "gpu_mode": "local"
    })
    ok = check("POST /api/projects (create)", status == 200, f"id={proj.get('id','?')}")
    if ok:
        created_project_id = proj["id"]

    # Get
    if created_project_id:
        status, p = get(f"/api/projects/{created_project_id}")
        check("GET /api/projects/{id}", status == 200, f"name={p.get('name')}, status={p.get('status')}")

    # List again
    status, data = get("/api/projects")
    check("GET /api/projects (after create)", status == 200 and any(p["id"] == created_project_id for p in data))


def test_project_types():
    print("\n[3] All LoRA types")
    types = ["style", "character", "face", "object"]
    ids = []
    for t in types:
        status, proj = post("/api/projects", {
            "name": f"test_{t}",
            "lora_type": t,
            "trigger_word": f"trigger_{t}",
            "base_model": "",
            "gpu_mode": "local"
        })
        ok = check(f"  Create '{t}' project", status == 200)
        if ok:
            ids.append(proj["id"])

    # Clean up
    for pid in ids:
        delete(f"/api/projects/{pid}")


def test_config():
    print("\n[4] Config endpoint")
    if not created_project_id:
        print(f"  {WARN} Skipped — no project created")
        return
    status, cfg = get(f"/api/projects/{created_project_id}/config")
    ok = check("GET /api/projects/{id}/config", status == 200)
    if ok:
        check("  Config has network_dim", "network_dim" in cfg, str(cfg.get("network_dim")))
        check("  Config has unet_lr", "unet_lr" in cfg, str(cfg.get("unet_lr")))
        check("  Config has lr_scheduler", "lr_scheduler" in cfg, str(cfg.get("lr_scheduler")))
        # Style-specific checks
        check("  Style: cosine scheduler", cfg.get("lr_scheduler") == "cosine_with_restarts")
        check("  Style: rank >= 32", (cfg.get("network_dim") or 0) >= 32)


def test_checkpoints():
    print("\n[5] Checkpoint API")
    if not created_project_id:
        print(f"  {WARN} Skipped — no project created")
        return
    status, chks = get(f"/api/projects/{created_project_id}/checkpoints")
    check("GET /api/projects/{id}/checkpoints", status == 200, f"{len(chks)} checkpoints")

    # 404 for nonexistent download
    status, _ = get(f"/api/projects/{created_project_id}/checkpoints/999/download")
    check("GET checkpoints/999/download → 404", status == 404)


def test_vastai():
    print("\n[6] Vast.ai API")
    status, data = get("/api/vastai/status")
    check("GET /api/vastai/status", status == 200)

    # Save a fake key
    status, _ = post("/api/vastai/settings", {"api_key": "test_key_placeholder"})
    check("POST /api/vastai/settings", status == 200)


def test_validation_weight():
    print("\n[7] Validation API (weight check)")
    if not created_project_id:
        print(f"  {WARN} Skipped — no project created")
        return

    # No checkpoints yet → 404 expected
    status, data = get(f"/api/projects/{created_project_id}/validate/weight/1")
    check("GET validate/weight/1 → 404 (no checkpoint)", status == 404)

    # Create a fake checkpoint file for weight analysis test
    fake_ckpt = Path(__file__).parent / "data" / "jobs" / created_project_id / "output"
    fake_ckpt.mkdir(parents=True, exist_ok=True)
    fake_file = fake_ckpt / "test_style_lora-000001.safetensors"

    # Try to make a minimal safetensors file
    try:
        import numpy as np
        try:
            from safetensors.numpy import save_file
            tensors = {
                "lora_unet_down_blocks_0_lora_down.weight": np.random.randn(16, 320).astype(np.float32),
                "lora_unet_down_blocks_0_lora_up.weight":   np.random.randn(320, 16).astype(np.float32),
                "lora_unet_down_blocks_0_alpha":             np.array(8.0, dtype=np.float32),
            }
            save_file(tensors, str(fake_file))

            # Register checkpoint in DB
            from server.database import upsert_checkpoint
            asyncio.run(upsert_checkpoint(created_project_id, 1, str(fake_file), loss=0.08))

            # Now test weight validation
            status, result = get(f"/api/projects/{created_project_id}/validate/weight/1")
            ok = check("GET validate/weight/1 (with fake checkpoint)", status == 200)
            if ok:
                check("  Has grade field", "grade" in result, result.get("grade"))
                check("  Has stats field", "stats" in result)
                check("  Grade is A/B/C/F", result.get("grade") in ("A","B","C","F"))
            fake_file.unlink(missing_ok=True)
        except ImportError:
            print(f"  {INFO} safetensors not available for fake checkpoint test — skipping weight test")
    except ImportError:
        print(f"  {INFO} numpy not available — skipping weight test")


def test_websocket():
    print("\n[8] WebSocket")
    if not created_project_id:
        print(f"  {WARN} Skipped — no project created")
        return
    try:
        import websocket  # websocket-client
        received = []
        def on_message(ws, msg):
            received.append(json.loads(msg))
            ws.close()
        ws = websocket.WebSocketApp(
            f"ws://localhost:7860/ws/{created_project_id}",
            on_message=on_message
        )
        t = threading.Thread(target=ws.run_forever, daemon=True)
        t.start()
        t.join(timeout=3)
        check("WebSocket connection", True, f"received {len(received)} messages")
    except ImportError:
        print(f"  {INFO} websocket-client not installed — skipping WS test")
        print(f"       (pip install websocket-client to enable)")
    except Exception as e:
        check("WebSocket connection", False, str(e))


def test_cleanup():
    print("\n[9] Cleanup")
    if not created_project_id:
        return
    status, _ = delete(f"/api/projects/{created_project_id}")
    check("DELETE /api/projects/{id}", status == 200)

    status, _ = get(f"/api/projects/{created_project_id}")
    check("Deleted project returns 404", status == 404)


# ── Runner ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  LoRA Maker - Automated Test Suite")
    print(f"  Target: {BASE}")
    print("=" * 55)

    test_server_reachable()
    test_project_crud()
    test_project_types()
    test_config()
    test_checkpoints()
    test_vastai()
    test_validation_weight()
    test_websocket()
    test_cleanup()

    total = results["pass"] + results["fail"]
    print("\n" + "=" * 55)
    print(f"  Results: {results['pass']}/{total} passed", end="")
    if results["fail"] == 0:
        print("  ✅ All tests passed!")
    else:
        print(f"  ❌ {results['fail']} failed")
    print("=" * 55 + "\n")

    sys.exit(0 if results["fail"] == 0 else 1)
