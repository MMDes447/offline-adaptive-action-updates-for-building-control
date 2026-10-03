#!/usr/bin/env python3
"""Environment check that writes only inside the AAMAS 2027 work tree."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path


AAMAS_DIR = Path(__file__).resolve().parents[1]


def command_output(command: str) -> str:
    try:
        completed = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        output = completed.stdout.strip()
        if completed.returncode:
            stderr = completed.stderr.strip()
            return f"ERR exit={completed.returncode}: {stderr or output}"
        return output
    except Exception as exc:
        return f"ERR {exc}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-conda-env", default="offrl_5zone")
    args = parser.parse_args()

    output_path = Path(args.output).resolve()
    if AAMAS_DIR != output_path.parent and AAMAS_DIR not in output_path.parents:
        parser.error(f"--output must be inside {AAMAS_DIR}")
    if output_path.exists():
        parser.error(f"refusing to overwrite existing file: {output_path}")

    result = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "cwd": os.getcwd(),
        "cpu_count": os.cpu_count(),
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
    }
    result["nvidia_smi"] = command_output(
        "nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu,driver_version --format=csv"
    )
    result["nvidia_procs"] = command_output(
        "nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv"
    )
    result["mem"] = command_output("free -g")
    result["disk"] = command_output(f"df -h {os.getcwd()}")

    failures: list[str] = []
    if result["conda_env"] != args.expected_conda_env:
        failures.append(
            f"expected conda env {args.expected_conda_env!r}, got {result['conda_env']!r}"
        )

    try:
        import torch

        result["torch"] = torch.__version__
        result["cuda_available"] = torch.cuda.is_available()
        if not result["cuda_available"]:
            failures.append("torch.cuda.is_available() is false")
        else:
            result["cuda_device"] = torch.cuda.get_device_name(0)
            result["cuda_version"] = torch.version.cuda
            x = torch.randn(4096, 4096, device="cuda")
            torch.cuda.synchronize()
            started = time.time()
            for _ in range(20):
                x @ x
            torch.cuda.synchronize()
            result["matmul_20x_4096_s"] = round(time.time() - started, 3)
    except Exception as exc:
        result["torch_error"] = repr(exc)
        failures.append(f"torch/CUDA check failed: {exc!r}")

    for module_name in ["numpy", "pandas", "frads", "pyradiance", "pywincalc"]:
        try:
            module = __import__(module_name)
            result[module_name] = getattr(module, "__version__", "ok")
        except Exception as exc:
            result[module_name] = f"ERR {exc!r}"
            failures.append(f"{module_name} import failed: {exc!r}")

    try:
        from pyenergyplus.api import EnergyPlusAPI

        functional = EnergyPlusAPI().functional
        result["energyplus"] = (
            str(functional.ep_version()) if hasattr(functional, "ep_version") else "import ok"
        )
    except Exception as exc:
        result["energyplus"] = f"ERR {exc!r}"
        failures.append(f"EnergyPlus import failed: {exc!r}")

    result["failures"] = failures
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"\nWrote {output_path}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
