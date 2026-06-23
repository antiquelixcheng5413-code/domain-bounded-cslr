from __future__ import annotations

import shutil
import subprocess
from typing import Any

import torch


def gpu_preflight() -> dict[str, Any]:
    available = torch.cuda.is_available()
    payload: dict[str, Any] = {
        "torch_version": torch.__version__,
        "cuda_available": available,
        "device_count": torch.cuda.device_count() if available else 0,
        "nvidia_smi": None,
    }
    executable = shutil.which("nvidia-smi")
    if executable:
        result = subprocess.run(
            [
                executable,
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        payload["nvidia_smi"] = {
            "return_code": result.returncode,
            "output": result.stdout.strip(),
            "error": result.stderr.strip(),
        }
    if not available:
        payload["status"] = "unavailable"
        return payload

    device = torch.device("cuda:0")
    try:
        values = torch.ones(1024, device=device)
        payload["device_name"] = torch.cuda.get_device_name(device)
        payload["device_total_memory"] = torch.cuda.get_device_properties(device).total_memory
        payload["tensor_sum"] = float(values.sum().item())
        payload["status"] = "ok"
    except RuntimeError as exc:
        payload["status"] = "failed"
        payload["error"] = str(exc)
    return payload
