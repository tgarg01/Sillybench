# Manual runbook: every step done by hand in the first study

The setup notebook automates all of this. This page keeps the manual commands, with what each one does, for learning, debugging, or running a single step yourself. **Laptop** = a terminal on your laptop; **VM** = a shell on the GPU VM (`gcloud compute ssh h100-prof --zone=us-central1-a`).

## 1. GCP project and quotas
| Where | Command | What it does |
|---|---|---|
| Laptop | `gcloud auth login` | Lets gcloud act as you (opens a browser). |
| Laptop | `gcloud projects create h100-qwen27b --name="H100Qwen27b"` | New project. The **ID** is lowercase; the name can have capitals. |
| Laptop | `gcloud config set project h100-qwen27b` | Default project for every command. |
| Laptop | `gcloud billing accounts list` then `gcloud billing projects link h100-qwen27b --billing-account=XXXX` | Attach billing; required to create VMs. |
| Laptop | `gcloud services enable compute.googleapis.com` | Turn on Compute Engine. |
| Laptop | `gcloud compute regions describe us-central1 --flatten="quotas[]" --format="table(quotas.metric,quotas.limit)" \| grep -E "CPUS\|H100"` | Regional quotas. Need `PREEMPTIBLE_NVIDIA_H100_GPUS ≥ 1`. `PREEMPTIBLE_CPUS = 0` is fine (Spot then uses `CPUS`). |
| Laptop | `gcloud compute project-info describe --flatten="quotas[]" --format="table(quotas.metric,quotas.limit)" \| grep ALL_REGIONS` | Global quotas. Need `GPUS_ALL_REGIONS ≥ 1` and `CPUS_ALL_REGIONS ≥ 26` (default was 12). |

Quota increases: <https://console.cloud.google.com/iam-admin/quotas?project=h100-qwen27b>, filter by Name (`Preemptible NVIDIA H100 GPUs`, `GPUs (all regions)`, `CPUs (all regions)`), tick the row, **Edit**, set the value, add a one-line justification (e.g. "Single a3-highgpu-1g Spot VM for LLM inference profiling, ~8 h"), submit. Small requests were approved in minutes.

## 2. Create, start, stop the VM
| Where | Command | What it does |
|---|---|---|
| Laptop | `gcloud compute accelerator-types list --filter="name=nvidia-h100-80gb AND zone~us-central1" --format="value(zone)"` | Zones with H100s. |
| Laptop | `gcloud compute images list --project=deeplearning-platform-release --no-standard-images --filter="family~'common-cu12' AND family~'ubuntu-2204'" --format="value(family)" \| sort -u \| tail -5` | Current Deep Learning VM images (driver + CUDA + nsys/ncu). |
| Laptop | `gcloud compute instances create h100-prof --zone=us-central1-a --machine-type=a3-highgpu-1g --provisioning-model=SPOT --instance-termination-action=STOP --maintenance-policy=TERMINATE --image-family=common-cu129-ubuntu-2204-nvidia-580 --image-project=deeplearning-platform-release --boot-disk-size=240GB --metadata=install-nvidia-driver=True` | 1x H100 Spot VM. `STOP` on preemption keeps the disk; GPU VMs need `TERMINATE` maintenance; the metadata installs the driver on first boot. |
| Laptop | `gcloud compute ssh h100-prof --zone=us-central1-a` | Shell on the VM (creates SSH keys the first time). |
| VM | `nvidia-smi` | Must show `NVIDIA H100 80GB HBM3`. |
| Laptop | `gcloud compute instances describe h100-prof --zone=us-central1-a --format="value(status)"` | `RUNNING` bills the GPU; `TERMINATED` bills only the disk. |
| Laptop | `gcloud compute instances start h100-prof --zone=us-central1-a && gcloud compute config-ssh` | Restart after a stop/preemption; `config-ssh` refreshes the IP for VS Code Remote-SSH. |
| Laptop | `gcloud compute instances stop h100-prof --zone=us-central1-a --discard-local-ssd=true` | Stop billing. The flag is required because `a3-highgpu-1g` has a Local SSD (unused here). |
| VM | `sudo shutdown -h +240` | Safety net: the VM stops itself after 240 min. |

## 3. VM software setup (`scripts/vm_setup.sh` does this)
| Command | What it does |
|---|---|
| `sudo apt-get install -y build-essential` | g++ for FlashInfer's JIT kernels (the image lacks it: `cannot execute 'cc1plus'`). |
| `curl -LsSf https://astral.sh/uv/install.sh \| sh` | uv, a fast Python package manager. |
| `uv venv ~/venv --python 3.12 && source ~/venv/bin/activate` | Isolated environment. |
| `uv pip install vllm "transformers>=5.8.0"` | vLLM with its matching PyTorch/CUDA wheels. |
| `hf download Qwen/Qwen3.8-27B` | ~52 GB of weights into `~/.cache/huggingface` (resumable). Run it in `tmux new -s dl`; detach with Ctrl+B then D. |
| `export PATH=/usr/local/cuda/bin:$PATH` | nsys and ncu ship with the image's CUDA toolkit. |
| `sudo -E env "PATH=$PATH" ncu --version` | ncu needs root for GPU performance counters. |

## 4. Benchmarks (`run_stage.py bench`)
```bash
vllm bench latency --model Qwen/Qwen3.8-27B --language-model-only --max-model-len 8192 --max-num-seqs 256 \
  --batch-size 1 --input-len 512 --output-len 128 --num-iters-warmup 3 --num-iters 5 --output-json lat.json
```
Loads the model, runs 3 warm-up + 5 timed passes. Decode time per token = (latency at out=128 − latency at out=1) / 127. The sweep repeats it for batch 1, 8, 32, 64, 128, 256 and out 1, 128.

## 5. Nsight Systems (`run_stage.py nsys`)
```bash
nsys profile -o b1_graphs --force-overwrite=true --trace=cuda,nvtx,osrt --cuda-graph-trace=node \
  --trace-fork-before-exec=true --capture-range=cudaProfilerApi --capture-range-end=stop \
  python scripts/prof_run.py --config runs/<id>/config.json --batch 1 --mode graphs --out 64
nsys stats --report cuda_gpu_kern_sum,nvtx_sum --format csv --output b1_graphs b1_graphs.nsys-rep
```
* `--capture-range=cudaProfilerApi`: records only between `torch.cuda.profiler.start/stop` in `prof_run.py`, so loading and warm-up are excluded.
* `--cuda-graph-trace=node`: shows every kernel inside CUDA graphs.
* `nsys stats` writes the kernel/NVTX summary CSVs and the `.sqlite` that notebook 02 reads.
* View on your laptop: open the `.nsys-rep` in the Nsight Systems app (2025.1 or newer).

## 6. Nsight Compute (`run_stage.py ncu` or the generated plan)
```bash
sudo -E env PATH=$PATH HOME=$HOME HF_HOME=$HOME/.cache/huggingface VLLM_ENABLE_V1_MULTIPROCESSING=0 GMU=0.72 MNS=64 \
  ncu --target-processes all --profile-from-start off --graph-profiling node \
      --kernel-name regex:'^nvjet_sm90_tst_384x8_64x4_4x1_v_bz_TNT$' --launch-skip 128 --launch-count 3 \
      --set full --clock-control none -o ncu/gate_up -f \
  python scripts/prof_run.py --config runs/<id>/config.json --batch 1 --mode graphs --out 8
ncu --import ncu/gate_up.ncu-rep --csv --page raw > ncu/gate_up.csv
```
* `GMU=0.72` leaves ~20 GB free for ncu's replay buffers. The first attempt at ~80 GB used ran at ~1 kernel per minute.
* `--launch-skip` (computed by notebook 02) skips the prefill instances, so ncu measures steady-state decode.
* `--clock-control none` keeps production clocks so ncu times match nsys.

## 7. Copy results to your laptop
```bash
gcloud compute scp --recurse h100-prof:~/sillybench/runs/<id> runs/ --zone=us-central1-a
```
The setup notebook instead packs a tarball with a `MANIFEST.sha256` and verifies every file after the copy.

## 8. Cost
* Credits: <https://console.cloud.google.com/billing/credits> · Spend: <https://console.cloud.google.com/billing/reports?project=h100-qwen27b> · Budget alerts: <https://console.cloud.google.com/billing/budgets>
* VM list (check nothing is RUNNING): <https://console.cloud.google.com/compute/instances?project=h100-qwen27b>
