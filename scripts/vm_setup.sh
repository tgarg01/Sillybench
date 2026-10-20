#!/bin/bash
# One-time (idempotent) setup of the GPU VM. Safe to re-run: every step checks first.
# Run on the VM from the repo root:   bash scripts/vm_setup.sh <model_id>
# The notebook runs this for you inside tmux (stage "setup").
set -euo pipefail
MODEL_ID="${1:-Qwen/Qwen3.8-27B}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$HOME/venv"

step() { echo; echo "=== $* ==="; }

step "GPU and driver"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

step "System packages (C++ compiler for FlashInfer JIT, tmux)"
NEED=""
command -v g++  >/dev/null || NEED="$NEED build-essential"
command -v tmux >/dev/null || NEED="$NEED tmux"
if [ -n "$NEED" ]; then
  sudo DEBIAN_FRONTEND=noninteractive apt-get update -qq
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq $NEED
else
  echo "g++ and tmux already present"
fi

step "CUDA tools on PATH (nsys, ncu)"
grep -q '/usr/local/cuda/bin' "$HOME/.bashrc" || echo 'export PATH=/usr/local/cuda/bin:$PATH' >> "$HOME/.bashrc"
export PATH=/usr/local/cuda/bin:$PATH
for t in nsys ncu; do
  if command -v $t >/dev/null; then echo "$t: $($t --version | tail -1)"; else echo "WARNING: $t not found on this image"; fi
done

step "uv + Python venv"
command -v uv >/dev/null || [ -x "$HOME/.local/bin/uv" ] || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
[ -x "$VENV/bin/python" ] || uv venv "$VENV" --python 3.12
# shellcheck disable=SC1091
source "$VENV/bin/activate"

step "Python packages (requirements-vm.txt, pinned)"
REQ_HASH=$(sha256sum "$REPO/requirements-vm.txt" | cut -c1-16)
if [ "$(cat "$VENV/.req_hash" 2>/dev/null)" != "$REQ_HASH" ]; then
  uv pip install -r "$REPO/requirements-vm.txt"
  echo "$REQ_HASH" > "$VENV/.req_hash"
else
  echo "requirements unchanged, skipping install"
fi
python -c "import vllm, torch; print('vllm', vllm.__version__, '| torch', torch.__version__, '|', torch.cuda.get_device_name())"

step "Model weights: $MODEL_ID"
# hf download is incremental: files already in ~/.cache/huggingface are not fetched again.
hf download "$MODEL_ID"
du -shL "$HOME/.cache/huggingface/hub/models--${MODEL_ID//\//--}/snapshots/"* | tail -1

step "Setup complete"
