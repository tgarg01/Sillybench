"""Drive a GCP GPU VM from your laptop with the `gcloud` CLI.

Everything here runs on your laptop. Every gcloud command is printed before it runs, so the
notebook doubles as a record of exactly what was done. Nothing here measures performance:
all timing happens on the VM.

Main pieces
    GCP.status / create / start / stop / delete      VM lifecycle
    GCP.ssh(cmd)                                      run a short command on the VM
    GCP.upload_code(...)                              copy scripts + run config to the VM
    GCP.launch(job, cmd) / GCP.wait(job)              long jobs in tmux, polled every N s,
                                                      with automatic resume after Spot preemption
    GCP.pull_run(...)                                 tar + sha256-verified copy back to the laptop
"""
from __future__ import annotations

import hashlib
import io
import json
import shlex
import subprocess
import tarfile
import time
from pathlib import Path


class GCPError(RuntimeError):
    pass


class Preempted(GCPError):
    pass


def _sha256(path: Path, chunk=1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


MAX_SSH_RETRIES = 10    # consecutive ssh failures on a RUNNING VM before giving up (a real error, not a hiccup)


class GCP:
    def __init__(self, project: str, zone: str, vm: str, remote_dir: str = "sillybench",
                 verbose: bool = True):
        self.project, self.zone, self.vm = project, zone, vm
        self.remote_rel = remote_dir                 # relative to the VM user's home (used by scp)
        self.remote_repo = f"$HOME/{remote_dir}"     # expanded by the VM shell
        self.verbose = verbose
        self.running_since = None      # wall-clock start of the current RUNNING period (cost tracking)
        self.running_seconds = 0.0     # accumulated RUNNING time in this notebook session

    # ------------------------------------------------------------------ low level
    def gcloud(self, *args, check=True, timeout=None, quiet_echo=False) -> subprocess.CompletedProcess:
        cmd = ["gcloud", *args, f"--project={self.project}"]
        if self.verbose and not quiet_echo:
            print("$ " + " ".join(shlex.quote(c) for c in cmd), flush=True)
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if check and p.returncode != 0:
            raise GCPError(f"gcloud failed (rc={p.returncode}):\n{p.stderr.strip()[-3000:]}")
        return p

    def _ssh_args(self):
        return ["compute", "ssh", self.vm, f"--zone={self.zone}", "--quiet",
                "--ssh-flag=-oServerAliveInterval=30", "--ssh-flag=-oConnectTimeout=20"]

    def ssh(self, command: str, check=True, timeout=600, echo=True) -> str:
        """Run a short command on the VM in a login shell; returns stdout."""
        remote = f"bash -lc {shlex.quote(command)}"
        if self.verbose and echo:
            print(f"[vm] {command}", flush=True)
        p = self.gcloud(*self._ssh_args(), f"--command={remote}", check=False, timeout=timeout, quiet_echo=True)
        if check and p.returncode != 0:
            raise GCPError(f"ssh command failed (rc={p.returncode}): {command}\n{p.stderr.strip()[-3000:]}")
        return p.stdout

    def scp_to(self, local: str | Path, remote: str):
        self.gcloud("compute", "scp", str(local), f"{self.vm}:{remote}", f"--zone={self.zone}", "--quiet")

    def scp_from(self, remote: str, local: str | Path):
        self.gcloud("compute", "scp", f"{self.vm}:{remote}", str(local), f"--zone={self.zone}", "--quiet")

    # ------------------------------------------------------------------ preflight
    def account(self) -> str:
        return subprocess.run(["gcloud", "config", "get-value", "account"], capture_output=True, text=True).stdout.strip()

    def quotas(self, region: str) -> dict:
        """The quotas that decide whether an a3-highgpu-1g Spot VM can be created."""
        want_regional = {"PREEMPTIBLE_NVIDIA_H100_GPUS", "NVIDIA_H100_GPUS", "CPUS", "PREEMPTIBLE_CPUS"}
        want_global = {"GPUS_ALL_REGIONS", "CPUS_ALL_REGIONS"}
        out = {}
        r = json.loads(self.gcloud("compute", "regions", "describe", region, "--format=json").stdout)
        for q in r.get("quotas", []):
            if q["metric"] in want_regional:
                out[q["metric"]] = (q.get("usage", 0), q["limit"])
        g = json.loads(self.gcloud("compute", "project-info", "describe", "--format=json").stdout)
        for q in g.get("quotas", []):
            if q["metric"] in want_global:
                out[q["metric"]] = (q.get("usage", 0), q["limit"])
        return out

    # ------------------------------------------------------------------ lifecycle
    def status(self) -> str:
        p = self.gcloud("compute", "instances", "list", f"--filter=name=({self.vm})",
                        "--format=value(status,zone.basename())", check=False, quiet_echo=True)
        line = p.stdout.strip().splitlines()
        if not line:
            return "NOT_FOUND"
        st, zone = (line[0].split() + [""])[:2]
        if zone and zone != self.zone:
            print(f"note: VM {self.vm} lives in {zone}; using that zone")
            self.zone = zone
        return st

    def create(self, machine_type, image_family, image_project, disk_gb, zones=None, spot=True, extra=()):
        """Create the VM, trying each zone in turn if one has no Spot capacity."""
        for z in (zones or [self.zone]):
            self.zone = z
            args = ["compute", "instances", "create", self.vm, f"--zone={z}", f"--machine-type={machine_type}",
                    "--maintenance-policy=TERMINATE", f"--image-family={image_family}",
                    f"--image-project={image_project}", f"--boot-disk-size={disk_gb}GB",
                    "--metadata=install-nvidia-driver=True", *extra]
            if spot:
                args += ["--provisioning-model=SPOT", "--instance-termination-action=STOP"]
            p = self.gcloud(*args, check=False)
            if p.returncode == 0:
                print(f"created {self.vm} in {z}")
                self._mark_running()
                return z
            err = p.stderr
            if any(s in err for s in ("ZONE_RESOURCE_POOL_EXHAUSTED", "does not have enough resources", "stockout")):
                print(f"  no capacity in {z}, trying next zone")
                continue
            raise GCPError(err.strip()[-3000:])
        raise GCPError("No zone had capacity. Wait 10-15 minutes and try again, or add zones to ZONES.")

    def start(self, retries=6, wait_s=120):
        for i in range(retries):
            p = self.gcloud("compute", "instances", "start", self.vm, f"--zone={self.zone}", check=False)
            if p.returncode == 0:
                self._mark_running()
                return
            if "RESOURCE_POOL_EXHAUSTED" in p.stderr or "does not have enough resources" in p.stderr:
                print(f"  no Spot capacity right now (attempt {i+1}/{retries}); retrying in {wait_s}s")
                time.sleep(wait_s)
                continue
            raise GCPError(p.stderr.strip()[-3000:])
        raise GCPError("Could not start the VM: no Spot capacity. Try again later.")

    def stop(self):
        """Stop the VM: GPU/CPU billing ends, the boot disk (model, venv, results) is kept."""
        self.gcloud("compute", "instances", "stop", self.vm, f"--zone={self.zone}", "--discard-local-ssd=true")
        self._mark_stopped()

    def delete(self):
        """Delete the VM and its boot disk. Only call after results are verified on the laptop."""
        self.gcloud("compute", "instances", "delete", self.vm, f"--zone={self.zone}", "--quiet")
        self._mark_stopped()

    def ensure_running(self, create_spec: dict | None = None):
        st = self.status()
        print(f"VM {self.vm}: {st}")
        if st == "NOT_FOUND":
            if not create_spec:
                raise GCPError("VM does not exist and no create spec was given.")
            self.create(**create_spec)
        elif st in ("TERMINATED", "STOPPED", "SUSPENDED"):
            self.start()
        elif st in ("STOPPING", "SUSPENDING"):
            while self.status() in ("STOPPING", "SUSPENDING"):
                time.sleep(10)
            self.start()
        elif st == "RUNNING":
            self._mark_running()
        else:
            raise GCPError(f"Unexpected VM status {st}")
        self.wait_ssh()

    def wait_ssh(self, timeout=900):
        """Wait until SSH works and the NVIDIA driver answers (first boot installs the driver)."""
        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                out = self.ssh("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader",
                               timeout=90, echo=False)
                if "H100" in out or "NVIDIA" in out:
                    print(f"VM ready after {time.time()-t0:.0f}s: {out.strip()}")
                    return out.strip()
            except Exception:  # noqa: BLE001
                pass
            print("  waiting for SSH / NVIDIA driver ...", flush=True)
            time.sleep(20)
        raise GCPError("VM did not become ready (SSH + nvidia-smi) in time.")

    def set_shutdown_timer(self, minutes: int):
        """Safety net: the VM powers itself off after `minutes` even if the notebook dies."""
        self.ssh(f"sudo shutdown -c 2>/dev/null; sudo shutdown -h +{int(minutes)}")
        print(f"auto-shutdown armed: VM stops itself in {minutes} min")

    # ------------------------------------------------------------------ code + config upload
    def upload_code(self, repo: Path, run_dir: Path, run_id: str):
        """Send scripts/, requirements-vm.txt and this run's config (+ ncu plan) as one tarball."""
        repo, run_dir = Path(repo), Path(run_dir)
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            tar.add(repo / "scripts", arcname="scripts")
            tar.add(repo / "requirements-vm.txt", arcname="requirements-vm.txt")
            tar.add(run_dir / "config.json", arcname=f"runs/{run_id}/config.json")
            plan = run_dir / "ncu_plan"
            if plan.exists():
                tar.add(plan, arcname=f"runs/{run_id}/ncu_plan")
        tmp = repo / ".upload.tgz"
        tmp.write_bytes(buf.getvalue())
        try:
            self.ssh(f"mkdir -p {self.remote_repo}")
            self.scp_to(tmp, f"{self.remote_rel}/.upload.tgz")
            self.ssh(f"cd {self.remote_repo} && tar xzf .upload.tgz && rm .upload.tgz && chmod +x scripts/*.sh "
                     f"&& ls scripts runs/{run_id}")
        finally:
            tmp.unlink(missing_ok=True)

    # ------------------------------------------------------------------ background jobs
    def launch(self, run_id: str, job: str, command: str, force=False):
        """Start `command` on the VM inside tmux session `job` (survives SSH drops)."""
        run = f"runs/{run_id}"
        if not force:
            done = self.ssh(f"cat {self.remote_repo}/{run}/status/{job}.exit 2>/dev/null || true", echo=False).strip()
            if done == "0":
                print(f"[{job}] already completed successfully on this VM; skipping (force=True to rerun)")
                return False
        alive = self.ssh(f"tmux has-session -t {job} 2>/dev/null && echo yes || echo no", echo=False).strip()
        if alive == "yes":
            print(f"[{job}] already running in tmux; attaching to it")
            return True
        # clear any old exit code first, so the poller can't mistake a previous run for this one
        self.ssh(f"cd {self.remote_repo} && rm -f {run}/status/{job}.exit && "
                 f"tmux new-session -d -s {job} bash scripts/job.sh {run} {job} {command}")
        print(f"[{job}] launched. Watch it live from a terminal: "
              f"gcloud compute ssh {self.vm} --zone {self.zone} -- tmux attach -t {job}")
        return True

    def wait(self, run_id: str, job: str, command: str, poll_s=60, auto_resume=True, max_resumes=3,
             on_resume=None, show_lines=4):
        """Poll until the job writes its exit code. Handles Spot preemption by restarting + relaunching."""
        run = f"{self.remote_repo}/runs/{run_id}"
        resumes, t0, last, hiccups, relaunches = 0, time.time(), "", 0, 0
        while True:
            try:
                out = self.ssh(
                    f"cat {run}/status/{job}.exit 2>/dev/null; echo '---'; "
                    f"tmux has-session -t {job} 2>/dev/null && echo alive || echo gone; echo '---'; "
                    f"tail -n {show_lines} {run}/logs/{job}.log 2>/dev/null; echo '---'; "
                    f"python3 -c \"import json;d=json.load(open('{run}/status/{job}.json'));"
                    f"it=d['items'];print(sum(i['state']!='running' for i in it),'of',len(it),'items finished')\" "
                    f"2>/dev/null || true", timeout=120, echo=False)
                rc, alive, tail, prog = (out.split("---") + ["", "", "", ""])[:4]
                rc, alive = rc.strip(), alive.strip()
                hiccups = 0
            except GCPError as e:
                st = self.status()
                if st != "RUNNING":
                    self._mark_stopped()
                    if not auto_resume or resumes >= max_resumes:
                        raise Preempted(f"VM is {st} (Spot preemption?). Restart it and rerun this cell; "
                                        f"finished items are skipped.")
                    resumes += 1
                    print(f"\n!!! VM is {st} (likely Spot preemption). Auto-resume {resumes}/{max_resumes} ...")
                    self.start()
                    self.wait_ssh()
                    if on_resume:
                        on_resume()
                    self.launch(run_id, job, command, force=True)
                    continue
                hiccups += 1
                if hiccups >= MAX_SSH_RETRIES:
                    raise GCPError(f"[{job}] polling failed {hiccups} times in a row on a RUNNING VM:\n{e}")
                print(f"  (ssh hiccup {hiccups}/{MAX_SSH_RETRIES}, retrying)")
                time.sleep(15)
                continue
            mins = (time.time() - t0) / 60
            msg = f"[{job}] {mins:5.1f} min | {prog.strip()} | {tail.strip().splitlines()[-1][:150] if tail.strip() else ''}"
            if msg != last:
                print(msg, flush=True)
                last = msg
            if rc != "":
                if rc != "0":
                    log = self.ssh(f"tail -n 40 {run}/logs/{job}.log", check=False, echo=False)
                    raise GCPError(f"[{job}] failed with exit code {rc}. Last log lines:\n{log}")
                print(f"[{job}] done in {mins:.1f} min")
                return
            if alive == "gone":
                # VM rebooted or tmux died without writing an exit code -> relaunch (stages are idempotent)
                relaunches += 1
                if relaunches > 3:
                    raise GCPError(f"[{job}] tmux session keeps vanishing without an exit code. "
                                   f"Last log lines:\n{tail.strip()}")
                print(f"[{job}] tmux session vanished without an exit code; relaunching")
                self.launch(run_id, job, command, force=True)
            time.sleep(poll_s)

    def _recover(self, err, on_resume=None):
        """After an ssh failure: if the VM is no longer RUNNING (Spot preemption), restart it."""
        st = self.status()
        if st == "RUNNING":
            return False
        self._mark_stopped()
        print(f"\n!!! VM is {st} (likely Spot preemption): restarting it ...")
        self.start()
        self.wait_ssh()
        if on_resume:
            on_resume()
        return True

    def run_job(self, run_id, job, command, poll_s=60, auto_resume=True, on_resume=None, force=False,
                max_resumes=3):
        """Launch a job and wait for it, surviving Spot preemptions at any point (launch or polling)."""
        resumes, hiccups = 0, 0
        while True:
            try:
                if self.launch(run_id, job, command, force=force or resumes > 0):
                    self.wait(run_id, job, command, poll_s=poll_s, auto_resume=auto_resume,
                              max_resumes=max_resumes - resumes, on_resume=on_resume)
                return
            except Preempted:
                raise
            except GCPError as e:
                if "failed with exit code" in str(e):
                    raise                                   # the job itself failed: show its log
                if not auto_resume or resumes >= max_resumes:
                    raise Preempted(f"{e}\nVM status: {self.status()}. Restart the VM and rerun this cell; "
                                    f"finished items are skipped.") from e
                if self._recover(e, on_resume):
                    resumes += 1
                    hiccups = 0
                    continue
                hiccups += 1
                if hiccups >= MAX_SSH_RETRIES:
                    raise GCPError(f"[{job}] failed {hiccups} times in a row on a RUNNING VM "
                                   f"(not a preemption):\n{e}") from e
                print(f"  (ssh hiccup {hiccups}/{MAX_SSH_RETRIES}, retrying)")
                time.sleep(15)

    # ------------------------------------------------------------------ results transfer
    def pull_run(self, run_id: str, local_runs: Path) -> dict:
        """Copy runs/<run_id> from the VM to the laptop and verify every file with sha256.

        1. On the VM: write MANIFEST.sha256 for every file, tar+gzip the run folder.
        2. Copy the tarball, check its sha256 against the VM's.
        3. Extract into runs/<run_id> on the laptop and re-check every file against the manifest.
        """
        local_runs = Path(local_runs)
        remote_run = f"{self.remote_repo}/runs"
        tgz = f"/tmp/{run_id}.tgz"
        out = self.ssh(f"cd {remote_run} && find {run_id} -type f ! -name MANIFEST.sha256 -print0 | sort -z | "
                       f"xargs -0 sha256sum > {run_id}/MANIFEST.sha256 && tar czf {tgz} {run_id} && "
                       f"sha256sum {tgz} | cut -d' ' -f1 && du -sh {tgz} | cut -f1 && wc -l < {run_id}/MANIFEST.sha256")
        remote_sha, size, nfiles = out.strip().splitlines()[-3:]
        print(f"VM packed {nfiles} files, {size}")
        local_tgz = local_runs / f".{run_id}.tgz"
        local_runs.mkdir(parents=True, exist_ok=True)
        self.scp_from(tgz, local_tgz)
        if _sha256(local_tgz) != remote_sha:
            raise GCPError("Tarball checksum mismatch after copy. Nothing was extracted; run pull again.")
        with tarfile.open(local_tgz) as tar:
            try:
                tar.extractall(local_runs, filter="data")
            except TypeError:  # Python < 3.12
                tar.extractall(local_runs)
        bad, n = [], 0
        run_dir = local_runs / run_id
        for line in (run_dir / "MANIFEST.sha256").read_text().splitlines():
            digest, rel = line.split(maxsplit=1)
            n += 1
            p = local_runs / rel.strip()
            if not p.exists() or _sha256(p) != digest:
                bad.append(rel)
        local_tgz.unlink()
        self.ssh(f"rm -f {tgz}", echo=False)
        report = dict(files=n, mismatched=bad, verified=not bad, size=size)
        print(f"verified {n - len(bad)}/{n} files on the laptop" + ("" if not bad else f"; MISMATCH: {bad[:5]}"))
        return report

    # ------------------------------------------------------------------ cost
    def _mark_running(self):
        if self.running_since is None:
            self.running_since = time.time()

    def _mark_stopped(self):
        if self.running_since is not None:
            self.running_seconds += time.time() - self.running_since
            self.running_since = None

    def running_hours(self):
        extra = time.time() - self.running_since if self.running_since else 0
        return (self.running_seconds + extra) / 3600
