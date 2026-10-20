#!/usr/bin/env python3
"""A stand-in for `gcloud` that simulates the VM on this machine, for testing the setup notebook for free.

The "VM" is a directory ($FAKE_VM_HOME) used as the VM user's home; `compute ssh --command` runs
the command there with bash, `compute scp` copies files in/out, and instance state lives in a JSON
file. FAKE_PREEMPT_AFTER=N simulates one Spot preemption after N ssh calls.
GPU tools are stubbed by tests/fake_bin (nvidia-smi, sudo, shutdown).
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HOME = Path(os.environ["FAKE_VM_HOME"])
STATE = HOME.parent / "fake_vm_state.json"


def load():
    return json.loads(STATE.read_text()) if STATE.exists() else {}


def save(s):
    STATE.write_text(json.dumps(s))


def opt(args, name):
    for a in args:
        if a.startswith(f"--{name}="):
            return a.split("=", 1)[1]
    return None


def vm_path(p):
    p = p.split(":", 1)[1]
    return Path(p) if p.startswith("/") else HOME / p


def main(argv):
    args = [a for a in argv if not a.startswith("--project=")]
    s = load()
    if args[:3] == ["config", "get-value", "account"]:
        print("tester@example.com"); return 0
    if args[:3] == ["compute", "regions", "describe"]:
        print(json.dumps({"quotas": [{"metric": "PREEMPTIBLE_NVIDIA_H100_GPUS", "limit": 1, "usage": 0},
                                     {"metric": "CPUS", "limit": 200, "usage": 0},
                                     {"metric": "PREEMPTIBLE_CPUS", "limit": 0, "usage": 0}]})); return 0
    if args[:3] == ["compute", "project-info", "describe"]:
        print(json.dumps({"quotas": [{"metric": "GPUS_ALL_REGIONS", "limit": 1, "usage": 0},
                                     {"metric": "CPUS_ALL_REGIONS", "limit": 32, "usage": 0}]})); return 0
    if args[:3] == ["compute", "instances", "list"]:
        if s.get("status"):
            print(f"{s['status']}\t{s['zone']}")
        return 0
    if args[:2] == ["compute", "instances"]:
        verb = args[2]
        if verb == "create":
            s.update(status="RUNNING", zone=opt(args, "zone"), created=True); save(s); HOME.mkdir(parents=True, exist_ok=True)
            print("Created", file=sys.stderr); return 0
        if verb == "start":
            s["status"] = "RUNNING"; save(s); return 0
        if verb == "stop":
            if "--discard-local-ssd=true" not in args:
                print("ERROR: VM has a Local SSD attached but an undefined value for `discard-local-ssd`", file=sys.stderr)
                return 1
            s["status"] = "TERMINATED"; save(s)
            subprocess.call(["tmux", "-L", "fakevm", "kill-server"], stderr=subprocess.DEVNULL)
            return 0
        if verb == "delete":
            save({}); return 0
    if args[:2] == ["compute", "ssh"]:
        if s.get("status") != "RUNNING":
            print("ssh: connect to host: Connection refused", file=sys.stderr); return 255
        s["ssh_calls"] = s.get("ssh_calls", 0) + 1
        n = int(os.environ.get("FAKE_PREEMPT_AFTER", "0"))
        if n and s["ssh_calls"] == n and not s.get("preempted"):
            s.update(status="TERMINATED", preempted=True); save(s)
            subprocess.call(["tmux", "-L", "fakevm", "kill-server"], stderr=subprocess.DEVNULL)
            print("Connection closed by remote host", file=sys.stderr); return 255
        save(s)
        cmd = opt(args, "command")
        env = dict(os.environ, HOME=str(HOME), PATH=os.environ["FAKE_VM_PATH"],
                   TMUX_TMPDIR=str(HOME.parent / "tmux"))
        (HOME.parent / "tmux").mkdir(exist_ok=True)
        # the real VM uses a login shell; here a login shell would reset PATH (root's /etc/profile)
        cmd = cmd.replace("bash -lc ", "bash -c ", 1).replace("tmux ", "tmux -L fakevm ")
        p = subprocess.run(["bash", "-c", cmd], env=env, cwd=HOME, capture_output=True, text=True)
        sys.stdout.write(p.stdout); sys.stderr.write(p.stderr)
        return p.returncode
    if args[:2] == ["compute", "scp"]:
        if s.get("status") != "RUNNING":
            return 255
        src, dst = [a for a in args[2:] if not a.startswith("--")][:2]
        src_p = vm_path(src) if ":" in src else Path(src)
        dst_p = vm_path(dst) if ":" in dst else Path(dst)
        dst_p.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src_p, dst_p); return 0
    print("fake gcloud: unhandled", args, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
