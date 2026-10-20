#!/bin/bash
# End-to-end test of the setup notebook + 02 WITHOUT GCP or a GPU.
#
# A fake `gcloud` (tests/fake_gcloud.py) simulates the VM in a temp directory. The fake VM is
# pre-seeded with the real results of the initial run, so the stages find their outputs already
# present and skip them (exactly what happens after a preemption), the transfer is verified,
# the VM is "stopped", and notebook 02 analyzes the pulled data. One Spot preemption is injected.
#
#   usage: bash tests/run_offline_test.sh
set -euo pipefail
REPO_CHECK="$(cd "$(dirname "$0")/.." && pwd)"
if head -c 40 "$REPO_CHECK"/runs/*/nsys/b1_graphs.sqlite 2>/dev/null | grep -q git-lfs; then
  echo "The seed traces are Git LFS pointers. Install Git LFS, then run: git lfs install && git lfs pull"; exit 1
fi
REPO="$(cd "$(dirname "$0")/.." && pwd)"
SEED_RUN="${SEED_RUN:-2026-10-09_qwen3.8-27b_bf16_initial}"
TEST_RUN="9999-01-01_offline-test"
WORK="$(mktemp -d)"
trap 'tmux -L fakevm kill-server 2>/dev/null || true; rm -rf "$WORK"' EXIT

# --- fake VM home + stub GPU tools
export FAKE_VM_HOME="$WORK/vmhome"
mkdir -p "$FAKE_VM_HOME/sillybench/runs/$TEST_RUN" "$WORK/bin" "$WORK/vmbin"
cp -r "$REPO/runs/$SEED_RUN/bench" "$REPO/runs/$SEED_RUN/nsys" "$REPO/runs/$SEED_RUN/logs" "$FAKE_VM_HOME/sillybench/runs/$TEST_RUN/"
cat > "$WORK/vmbin/nvidia-smi" <<'EOF'
#!/bin/bash
case "$*" in *query-compute-apps*) exit 0;; *) echo "NVIDIA H100 80GB HBM3, 81559 MiB, 580.178.04";; esac
EOF
printf '#!/bin/bash\nexit 0\n' > "$WORK/vmbin/shutdown"
# the transfer step runs sha256sum on the "VM"; macOS keeps it in /sbin (or lacks it on older versions)
if command -v sha256sum >/dev/null; then ln -s "$(command -v sha256sum)" "$WORK/vmbin/sha256sum"
else printf '#!/bin/bash\nexec shasum -a 256 "$@"\n' > "$WORK/vmbin/sha256sum"; fi
cat > "$WORK/vmbin/sudo" <<'EOF'
#!/bin/bash
# drop sudo's own flags and an optional "env K=V ..." prefix, then run the command
while [[ "$1" == -* ]]; do shift; done
if [[ "$1" == "env" ]]; then shift; while [[ "$1" == *=* ]]; do export "$1"; shift; done; fi
exec "$@"
EOF
ln -s "$(command -v python3)" "$WORK/vmbin/python"
chmod +x "$WORK/vmbin/"*
# the fake VM runs jobs in a real tmux on this machine (Homebrew: /opt/homebrew/bin on Apple Silicon)
command -v tmux >/dev/null || { echo "tmux is required for the offline test (macOS: brew install tmux; Ubuntu: sudo apt install tmux)"; exit 1; }
export FAKE_VM_PATH="$WORK/vmbin:$(dirname "$(command -v tmux)"):/usr/local/bin:/usr/bin:/bin"

printf '#!/bin/bash\nexec %q %q "$@"\n' "$(command -v python3)" "$REPO/tests/fake_gcloud.py" > "$WORK/bin/gcloud"
chmod +x "$WORK/bin/gcloud"
export PATH="$WORK/bin:$PATH"
export FAKE_PREEMPT_AFTER=9          # one simulated Spot preemption during the first stage

# --- the setup notebook with test settings: fast polling, dry-run stages, no setup/env (they need a real GPU VM)
python3 - "$REPO" "$TEST_RUN" "$WORK/nb01.ipynb" <<'EOF'
import json, sys
repo, run_id, out = sys.argv[1:]
nb = json.load(open(f"{repo}/notebooks/setup.ipynb"))
for c in nb["cells"]:
    if c["cell_type"] != "code":
        continue
    src = "".join(c["source"])
    src = src.replace('POLL_SECONDS = 60', 'POLL_SECONDS = 1')
    src = src.replace('STAGES = ["setup", "env", "bench", "nsys", "ncu"]', 'STAGES = ["bench", "nsys", "ncu"]')
    src = src.replace('RUN_ID = None ', f'RUN_ID = "{run_id}" ')
    src = src.replace('DRY_RUN_STAGES = False', 'DRY_RUN_STAGES = True')
    src = src.replace('"gpu_metrics": True', '"gpu_metrics": False')
    c["source"] = src
json.dump(nb, open(out, "w"))
EOF
# gcp.wait_ssh sleeps 20 s between tries and start() 120 s on capacity errors; speed them up for the test
export PYTHONPATH="$REPO"
cd "$REPO/notebooks"
python3 - "$WORK/nb01.ipynb" <<'EOF'
import sys, nbformat
from nbclient import NotebookClient
import gpuprof.gcp as g
nb = nbformat.read(sys.argv[1], as_version=4)
# make the waits in gcp.py fast for the test run
nb.cells.insert(1, nbformat.v4.new_code_cell("import gpuprof.gcp as _g, time as _t\n_real_sleep=_t.sleep\n_g.time.sleep=lambda s: _real_sleep(min(s,0.5))"))
NotebookClient(nb, timeout=900, resources={"metadata": {"path": "."}}).execute()
for i, c in enumerate(nb.cells):
    for o in c.get("outputs", []):
        t = o.get("text") or "".join(o.get("data", {}).get("text/plain", ""))
        if t:
            print(f"--- cell {i}\n{t[-1500:]}")
nbformat.write(nb, sys.argv[1].replace(".ipynb", ".executed.ipynb"))
EOF

# --- checks (any failure exits non-zero)
R="$REPO/runs/$TEST_RUN"
check() { if eval "$2"; then echo "OK   $1"; else echo "FAIL $1"; exit 1; fi; }
check "manifest pulled"                 'test -f "$R/MANIFEST.sha256"'
check "traces pulled"                   'test -s "$R/nsys/b1_graphs.sqlite"'
check "notebook 02 ran on the pulled run" 'test -f "$R/analysis/findings.md"'
check "VM stopped at the end"           'grep -q TERMINATED "$WORK/fake_vm_state.json"'
check "survived a simulated preemption" 'grep -q "\"preempted\": true" "$WORK/fake_vm_state.json"'
cat "$R/analysis/findings.md"
rm -rf "$R"
echo "offline test passed"
