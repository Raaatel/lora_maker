"""Vast.ai integration service for remote GPU training."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

VASTAI_API = "https://console.vast.ai/api/v0"


class VastAIError(Exception):
    pass


async def _request(method: str, path: str, api_key: str, data: Any = None) -> Any:
    """Make an async HTTP request to Vast.ai API."""
    import urllib.request
    import urllib.error

    url = f"{VASTAI_API}{path}?api_key={api_key}"
    headers = {"Content-Type": "application/json", "Accept": "application/json"}

    body = json.dumps(data).encode() if data else None

    def _sync_request():
        req = urllib.request.Request(url, data=body, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            raise VastAIError(f"API error {e.code}: {e.read().decode()}")

    return await asyncio.to_thread(_sync_request)


async def search_instances(
    api_key: str,
    min_vram_gb: float = 20,
    gpu_name: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Search for available GPU instances on Vast.ai."""
    query = {
        "verified": {"eq": True},
        "rentable": {"eq": True},
        "num_gpus": {"eq": 1},
        "gpu_ram": {"gte": min_vram_gb * 1024},  # MB
        "reliability2": {"gte": 0.95},
        "cuda_max_good": {"gte": 12.0},
    }
    if gpu_name:
        query["gpu_name"] = {"eq": gpu_name}

    try:
        result = await _request("GET", f"/bundles?q={json.dumps(query)}", api_key)
        offers = result.get("offers", [])
        # Sort by price
        return sorted(offers, key=lambda x: x.get("dph_total", 999))
    except VastAIError as e:
        raise VastAIError(f"Failed to search instances: {e}")


async def create_instance(
    api_key: str,
    offer_id: int,
    disk_gb: float = 80,
    docker_image: str = "pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime",
    ssh_key: str = "",
) -> dict[str, Any]:
    """Create (rent) a Vast.ai instance."""
    payload = {
        "client_id": "me",
        "image": docker_image,
        "disk": disk_gb,
        "onstart": "pip install -q kohya_ss 2>/dev/null; echo READY",
        "env": {
            "PYTHONUNBUFFERED": "1",
        },
    }
    if ssh_key:
        payload["ssh_key"] = ssh_key

    try:
        result = await _request("PUT", f"/asks/{offer_id}/", api_key, payload)
        return result
    except VastAIError as e:
        raise VastAIError(f"Failed to create instance: {e}")


async def get_instance(api_key: str, instance_id: int) -> dict[str, Any]:
    """Get instance details."""
    result = await _request("GET", f"/instances/{instance_id}/", api_key)
    instances = result.get("instances", [])
    if not instances:
        raise VastAIError(f"Instance {instance_id} not found")
    return instances[0]


async def destroy_instance(api_key: str, instance_id: int) -> None:
    """Destroy (stop & delete) a Vast.ai instance."""
    await _request("DELETE", f"/instances/{instance_id}/", api_key)


async def wait_for_instance(
    api_key: str,
    instance_id: int,
    callback: Optional[Callable[[str], None]] = None,
    timeout: int = 300,
) -> dict[str, Any]:
    """Wait for instance to reach 'running' state."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            inst = await get_instance(api_key, instance_id)
            status = inst.get("actual_status", "")
            if callback:
                callback(f"Instance status: {status}")
            if status == "running":
                return inst
            if status in ("error", "exited", "deleted"):
                raise VastAIError(f"Instance failed with status: {status}")
        except VastAIError:
            raise
        except Exception as e:
            logger.warning("Instance status check failed: %s", e)
        await asyncio.sleep(5)

    raise VastAIError(f"Instance did not start within {timeout}s")


async def run_remote_training(
    config: dict[str, Any],
    api_key: str,
    ssh_key_path: str,
    offer_id: int,
    callback: Callable[[str, dict[str, Any]], None],
    cancel_event,
) -> dict[str, Any]:
    """Full remote training flow on Vast.ai."""
    instance_id = None

    try:
        # 1) Create instance
        callback("log", {"message": "Vast.ai 인스턴스 생성 중...", "level": "info"})
        disk_gb = config.get("vastai", {}).get("disk_space_gb", 80)
        docker_image = config.get("vastai", {}).get("docker_image", "pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime")

        with open(ssh_key_path + ".pub", "r") as f:
            pub_key = f.read().strip()

        result = await create_instance(api_key, offer_id, disk_gb, docker_image, pub_key)
        instance_id = result.get("new_contract")
        if not instance_id:
            raise VastAIError("Failed to get instance ID from creation response")

        callback("log", {"message": f"인스턴스 생성됨 (ID: {instance_id}), 부팅 대기 중...", "level": "info"})

        # 2) Wait for ready
        inst = await wait_for_instance(
            api_key, instance_id,
            callback=lambda msg: callback("log", {"message": msg, "level": "info"}),
        )

        ssh_host = inst.get("ssh_host")
        ssh_port = inst.get("ssh_port", 22)

        callback("log", {"message": f"인스턴스 준비 완료: {ssh_host}:{ssh_port}", "level": "info"})

        # 3) Transfer dataset
        callback("log", {"message": "데이터셋 전송 중...", "level": "info"})
        processed_dir = config["data"]["processed_dir"]
        remote_dir = "/workspace/lora_training"

        await asyncio.to_thread(
            _rsync_to_remote,
            processed_dir, ssh_key_path, ssh_host, ssh_port,
            f"{remote_dir}/processed"
        )

        # 4) Generate remote training script
        script = _build_remote_script(config, remote_dir)
        script_path = Path(config["data"]["output_dir"]) / "remote_train.sh"
        script_path.write_text(script, encoding="utf-8")

        await asyncio.to_thread(
            _scp_file,
            str(script_path), ssh_key_path, ssh_host, ssh_port,
            f"{remote_dir}/train.sh"
        )

        # 5) Run training
        callback("log", {"message": "원격 학습 시작...", "level": "info"})
        callback("training_start", {"message": "Vast.ai에서 학습 중..."})

        await asyncio.to_thread(
            _ssh_exec,
            f"bash {remote_dir}/train.sh",
            ssh_key_path, ssh_host, ssh_port,
            lambda line: callback("log", {"message": line, "level": "info"}),
        )

        # 6) Download results
        callback("log", {"message": "결과물 다운로드 중...", "level": "info"})
        output_dir = config["data"]["output_dir"]
        lora_name = config["training"].get("lora_name", "lora_output")

        await asyncio.to_thread(
            _rsync_from_remote,
            ssh_key_path, ssh_host, ssh_port,
            f"{remote_dir}/output/*.safetensors",
            output_dir,
        )

        lora_path = Path(output_dir) / f"{lora_name}.safetensors"
        callback("completed", {"lora_path": str(lora_path), "final_loss": 0})
        return {"lora_path": str(lora_path)}

    except Exception as e:
        callback("error", {"message": f"원격 학습 실패: {e}"})
        return {"error": str(e)}
    finally:
        if instance_id:
            try:
                callback("log", {"message": "인스턴스 정리 중...", "level": "info"})
                await destroy_instance(api_key, instance_id)
            except Exception as e:
                logger.warning("Failed to destroy instance %s: %s", instance_id, e)


def _ssh_exec(command: str, key_path: str, host: str, port: int, line_callback: Callable) -> None:
    ssh_cmd = [
        "ssh", "-i", key_path,
        "-o", "StrictHostKeyChecking=no",
        "-o", "BatchMode=yes",
        "-p", str(port),
        f"root@{host}",
        command,
    ]
    process = subprocess.Popen(
        ssh_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace",
    )
    for line in process.stdout:
        line_callback(line.strip())
    process.wait()
    if process.returncode != 0:
        raise VastAIError(f"SSH command failed with exit code {process.returncode}")


def _rsync_to_remote(local_dir: str, key_path: str, host: str, port: int, remote_path: str) -> None:
    cmd = [
        "rsync", "-avz", "--progress",
        "-e", f"ssh -i {key_path} -p {port} -o StrictHostKeyChecking=no",
        local_dir + "/",
        f"root@{host}:{remote_path}/",
    ]
    subprocess.run(cmd, check=True)


def _rsync_from_remote(key_path: str, host: str, port: int, remote_path: str, local_dir: str) -> None:
    cmd = [
        "rsync", "-avz", "--progress",
        "-e", f"ssh -i {key_path} -p {port} -o StrictHostKeyChecking=no",
        f"root@{host}:{remote_path}",
        local_dir + "/",
    ]
    subprocess.run(cmd, check=True)


def _scp_file(local_path: str, key_path: str, host: str, port: int, remote_path: str) -> None:
    cmd = [
        "scp", "-i", key_path, "-P", str(port),
        "-o", "StrictHostKeyChecking=no",
        local_path, f"root@{host}:{remote_path}",
    ]
    subprocess.run(cmd, check=True)


def _build_remote_script(config: dict[str, Any], remote_dir: str) -> str:
    """Generate a shell script that runs kohya training on the remote machine."""
    train = config["training"]
    model = config["model"]
    lora_name = train.get("lora_name", "lora_output")
    num_repeats = train.get("num_repeats", 10)
    trigger = train.get("trigger_word", "sks")

    return f"""#!/bin/bash
set -e
pip install -q accelerate transformers diffusers safetensors bitsandbytes 2>/dev/null

mkdir -p {remote_dir}/output {remote_dir}/dataset/{num_repeats}_{trigger}
cp {remote_dir}/processed/*.* {remote_dir}/dataset/{num_repeats}_{trigger}/

# Download model if needed
MODEL="{model.get('base_model', '')}"
if [[ "$MODEL" != /* ]] && [[ "$MODEL" != "" ]]; then
    echo "모델 다운로드: $MODEL"
    python -c "from huggingface_hub import snapshot_download; snapshot_download('$MODEL', local_dir='/workspace/model')"
    MODEL="/workspace/model"
fi

# Clone kohya-ss if not present
if [ ! -d /workspace/kohya-sd-scripts ]; then
    git clone https://github.com/kohya-ss/sd-scripts /workspace/kohya-sd-scripts
    pip install -r /workspace/kohya-sd-scripts/requirements.txt -q
fi

cd /workspace/kohya-sd-scripts
python sdxl_train_network.py \\
    --pretrained_model_name_or_path "$MODEL" \\
    --train_data_dir {remote_dir}/dataset \\
    --output_dir {remote_dir}/output \\
    --output_name {lora_name} \\
    --save_model_as safetensors \\
    --save_precision fp16 \\
    --resolution {train.get('resolution', 1024)},{train.get('resolution', 1024)} \\
    --max_train_epochs {train.get('num_epochs', 10)} \\
    --train_batch_size {train.get('train_batch_size', 1)} \\
    --unet_lr {train.get('unet_lr', 0.0001)} \\
    --network_module networks.lora \\
    --network_dim {train.get('lora_rank', 32)} \\
    --network_alpha {train.get('lora_alpha', 16)} \\
    --lr_scheduler {train.get('lr_scheduler', 'cosine_with_restarts')} \\
    --lr_scheduler_num_cycles {train.get('lr_scheduler_num_cycles', 1)} \\
    --lr_warmup_steps {train.get('lr_warmup_steps', 100)} \\
    --min_snr_gamma {train.get('min_snr_gamma', 5)} \\
    --noise_offset {train.get('noise_offset', 0.05)} \\
    --mixed_precision fp16 \\
    --save_every_n_epochs 1 \\
    --caption_extension .txt \\
    --shuffle_caption \\
    --keep_tokens 1 \\
    --gradient_checkpointing \\
    --optimizer_type AdamW8bit \\
    --enable_bucket \\
    --cache_latents \\
    --no_half_vae \\
    --sdpa

echo "학습 완료!"
"""
