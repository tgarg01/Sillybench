#!/usr/bin/env python3
"""Profiling harness: runs vLLM in-process and opens a capture window for nsys / ncu.

Runs on the GPU VM. The engine arguments come from the run's config.json (the
"vllm" section that you edit in the setup notebook), so benchmarks and profiles always
use exactly the same engine settings.

Timeline produced (NVTX ranges):
    prefill_b{B}            -> one forward pass over the prompts (max_tokens=1)
    prefill_decode{OUT}_b{B} -> prefill + (OUT-1) decode steps

Usage:
    python scripts/prof_run.py --config runs/<id>/config.json --batch 1 --mode graphs --out 64

Environment overrides (used by ncu runs to leave GPU memory free for kernel replay):
    GMU=0.72   -> gpu_memory_utilization
    MNS=64     -> max_num_seqs
"""
import argparse
import json
import os

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")  # keep the engine in this process
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import torch  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402


def engine_kwargs(cfg, mode):
    kw = dict(cfg["vllm"])
    kw.update(cfg.get("vllm_profile_overrides", {}))
    if mode == "eager":
        kw["enforce_eager"] = True
    elif mode == "graphs":
        kw["enforce_eager"] = False
    if os.environ.get("GMU"):
        kw["gpu_memory_utilization"] = float(os.environ["GMU"])
    if os.environ.get("MNS"):
        kw["max_num_seqs"] = int(os.environ["MNS"])
    return kw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--batch", type=int, required=True)
    ap.add_argument("--mode", choices=["graphs", "eager"], required=True)
    ap.add_argument("--out", type=int, default=64, help="max_tokens in the measured decode range")
    ap.add_argument("--input-len", type=int, default=None)
    args = ap.parse_args()

    cfg = json.load(open(args.config))
    input_len = args.input_len or cfg["workload"]["input_len"]
    kw = engine_kwargs(cfg, args.mode)
    print("engine kwargs:", kw, flush=True)

    llm = LLM(**kw)
    B = args.batch
    # Distinct synthetic prompts (different token ids per sequence) so nothing can be shared.
    prompts = [{"prompt_token_ids": [1000 + i] * input_len} for i in range(B)]
    sp = lambda n: SamplingParams(max_tokens=n, ignore_eos=True, temperature=0.0)  # noqa: E731

    # Warm-up outside the capture window: JIT, autotuning, first-call costs.
    llm.generate(prompts, sp(1), use_tqdm=False)
    llm.generate(prompts, sp(args.out), use_tqdm=False)
    torch.cuda.synchronize()

    torch.cuda.profiler.start()  # nsys --capture-range=cudaProfilerApi / ncu --profile-from-start off
    with torch.cuda.nvtx.range(f"prefill_b{B}"):
        llm.generate(prompts, sp(1), use_tqdm=False)
        torch.cuda.synchronize()
    with torch.cuda.nvtx.range(f"prefill_decode{args.out}_b{B}"):
        llm.generate(prompts, sp(args.out), use_tqdm=False)
        torch.cuda.synchronize()
    torch.cuda.profiler.stop()
    print("PROF_RUN_OK", flush=True)


if __name__ == "__main__":
    main()
