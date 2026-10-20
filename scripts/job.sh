#!/bin/bash
# Run one long job on the VM so it survives SSH drops (the notebook starts it inside tmux).
#   usage: bash scripts/job.sh <run_dir> <job_name> <command...>
# Writes the command's output to <run_dir>/logs/<job_name>.log and its exit code to
# <run_dir>/status/<job_name>.exit -- the notebook polls for that file.
RUN="$1"; NAME="$2"; shift 2
cd "$(dirname "$0")/.." || exit 1
mkdir -p "$RUN/status" "$RUN/logs"
rm -f "$RUN/status/$NAME.exit"
export PATH="$HOME/.local/bin:/usr/local/cuda/bin:$PATH"
# shellcheck disable=SC1091
[ -f "$HOME/venv/bin/activate" ] && source "$HOME/venv/bin/activate"
echo "[$(date -u +%FT%TZ)] start: $*" > "$RUN/logs/$NAME.log"
"$@" >> "$RUN/logs/$NAME.log" 2>&1
RC=$?
echo "[$(date -u +%FT%TZ)] exit code $RC" >> "$RUN/logs/$NAME.log"
echo $RC > "$RUN/status/$NAME.exit"
