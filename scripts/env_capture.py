#!/usr/bin/env python3
"""Record everything that defines the measurement environment into <run>/env/.

Runs on the VM before any measurement. A profile without its versions and clocks
is not reproducible, so this goes with every run.

Usage: python scripts/env_capture.py --run-dir runs/<id>
"""
import argparse
import importlib
import json
import platform
import subprocess
from pathlib import Path


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"ERROR: {e}"


def version(mod):
    try:
        return getattr(importlib.import_module(mod), "__version__", "unknown")
    except Exception:  # noqa: BLE001
        return None


def cuda_device_attrs(dev=0):
    """Hardware attributes from the CUDA driver API (libcuda), the inputs for the peak-bandwidth
    and peak-FLOP/s derivation in notebook 02. nvidia-smi does not report the memory bus width."""
    import ctypes
    attrs = {  # CUdevice_attribute enum values (cuda.h)
        "sm_count": 16, "sm_clock_max_khz": 13, "mem_clock_max_khz": 36, "mem_bus_width_bits": 37,
        "l2_bytes": 38, "cc_major": 75, "cc_minor": 76,
    }
    try:
        cuda = ctypes.CDLL("libcuda.so.1")
        assert cuda.cuInit(0) == 0
        handle = ctypes.c_int()
        assert cuda.cuDeviceGet(ctypes.byref(handle), dev) == 0
        out = {}
        for k, a in attrs.items():
            v = ctypes.c_int()
            assert cuda.cuDeviceGetAttribute(ctypes.byref(v), a, handle) == 0, k
            out[k] = v.value
        total = ctypes.c_size_t()
        if cuda.cuDeviceTotalMem_v2(ctypes.byref(total), handle) == 0:
            out["total_mem_bytes"] = total.value
        return out
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    a = ap.parse_args()
    run = Path(a.run_dir)
    out = run / "env"
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.load(open(run / "config.json"))
    model = cfg["vllm"]["model"]

    raw = {
        "nvidia-smi-q.txt": "nvidia-smi -q",
        "nvidia-smi-clocks.txt": "nvidia-smi -q -d CLOCK,PERFORMANCE,POWER",
        "nvidia-smi-topo.txt": "nvidia-smi topo -m",
        "pip-freeze.txt": "python -m pip freeze 2>/dev/null || uv pip freeze",
        "lscpu.txt": "lscpu",
        "nsys-version.txt": "nsys --version",
        "ncu-version.txt": "ncu --version",
        "nvcc-version.txt": "nvcc --version",
    }
    for fname, cmd in raw.items():
        (out / fname).write_text(sh(cmd) + "\n")

    import torch  # noqa: E402

    snap = sh(f"ls -1 ~/.cache/huggingface/hub/models--{model.replace('/', '--')}/snapshots 2>/dev/null | tail -1")
    env = {
        "gpu": sh("nvidia-smi --query-gpu=name,memory.total,driver_version,pstate,clocks.max.sm,clocks.max.mem "
                  "--format=csv,noheader"),
        "gpu_attrs": cuda_device_attrs(),
        "cuda_runtime_torch": torch.version.cuda,
        "torch": torch.__version__,
        "vllm": version("vllm"),
        "transformers": version("transformers"),
        "flashinfer": version("flashinfer"),
        "triton": version("triton"),
        "nsys": sh("nsys --version").splitlines()[-1] if sh("nsys --version") else None,
        "ncu": sh("ncu --version").splitlines()[-1] if sh("ncu --version") else None,
        "python": platform.python_version(),
        "kernel": platform.release(),
        "os": sh(". /etc/os-release && echo $PRETTY_NAME"),
        "cpu": sh("lscpu | grep 'Model name' | sed 's/.*: *//'"),
        "vcpus": sh("nproc"),
        "machine_type": sh("curl -s -H 'Metadata-Flavor: Google' "
                           "http://metadata.google.internal/computeMetadata/v1/instance/machine-type | sed 's#.*/##'"),
        "zone": sh("curl -s -H 'Metadata-Flavor: Google' "
                   "http://metadata.google.internal/computeMetadata/v1/instance/zone | sed 's#.*/##'"),
        "model": model,
        "model_revision": snap or None,
        "gpu_processes_at_start": sh("nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader"),
    }
    (out / "env.json").write_text(json.dumps(env, indent=2))
    print(json.dumps(env, indent=2))


if __name__ == "__main__":
    main()
