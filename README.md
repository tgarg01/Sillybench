# SillyBench

**Understand LLM inference by visualizing it on real hardware with real numbers.**

SillyBench is a teaching series on AI performance engineering: where models meet hardware. Each post takes one idea
about LLM inference, predicts the result from first principles, measures it on a real GPU, and explains every
millisecond. This repository holds everything behind the posts, so you can rerun them yourself:

* **a notebook for every post** that recomputes all of the post's numbers and charts from the raw measurements,
* **the raw measurements themselves** (benchmarks, Nsight Systems traces, Nsight Compute reports) in `runs/`,
* **the harness that produced them**, which you can point at your own GPU, model and settings.

Read the posts at **[tgarg01.github.io/sillybench-site](https://tgarg01.github.io/sillybench-site/)**.

## Posts

**Series 1: [LLM inference through the lens of a GPU](https://tgarg01.github.io/sillybench-site/series/llm-inference-gpu/)**

| # | Post | Reproduce it |
|---|---|---|
| 1 | [A loop of 902 kernels](https://tgarg01.github.io/sillybench-site/posts/01-a-loop-of-902-kernels/) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tgarg01/SillyBench/blob/post-01/notebooks/post1.ipynb) · [`notebooks/post1.ipynb`](notebooks/post1.ipynb) |

Every post's links point at a git tag (`post-01`, `post-02`, …), so the notebook you open always matches what
the post shows, even after `main` moves on.

## Reproduce a post

There are three ways, from zero setup to your own measurements.

### 1. In your browser (no install)
Click **Open in Colab** in the table above and choose **Runtime → Run all**. The first cell fetches this repo at the
post's tag; everything else runs as is. No GPU needed: the notebooks read the stored measurements.

### 2. On your laptop
Python 3.10+ and git:
```bash
git clone https://github.com/tgarg01/SillyBench.git
cd SillyBench
bash setup_laptop.sh                      # .venv + the "sillybench" Jupyter kernel (macOS or Linux)
.venv/bin/jupyter lab notebooks/post1.ipynb
```
Windows (PowerShell): `python -m venv .venv`, `.venv\Scripts\pip install -r requirements.txt`, then
`.venv\Scripts\jupyter lab notebooks\post1.ipynb`.

In VS Code, open the notebook and pick the kernel: **Select Kernel** → **Select Another Kernel…** →
**Jupyter Kernel…** → **sillybench** (not listed? run **Developer: Reload Window**).

### 3. With your own measurements
Every post notebook reads one **run folder**, `runs/<run_id>/`, and names it in a `RUN_ID` line near the top.
Make your own run on a cloud GPU with [`notebooks/setup.ipynb`](notebooks/setup.ipynb) (below), set `RUN_ID` to
your new folder, and run the post notebook again: every number and chart is recomputed from your data.

## Measure on a GPU

[`notebooks/setup.ipynb`](notebooks/setup.ipynb) runs on your laptop and drives a cloud GPU VM (Google Cloud today)
through the `gcloud` CLI, printing every command it runs. You choose the GPU machine, the model and the vLLM engine
arguments in its first section; the notebook then:

1. creates or starts the VM and installs the pinned software (`scripts/vm_setup.sh`),
2. records the environment: GPU attributes, driver, clocks, every package version, the model revision,
3. runs the benchmark sweep (`vllm bench latency`),
4. captures Nsight Systems timelines and, in a second session, Nsight Compute profiles,
5. copies the run folder back to your laptop, sha256-verifying every file,
6. stops the VM.

**Before the first run:** install the [gcloud CLI](https://cloud.google.com/sdk/docs/install), run
`gcloud auth login`, and have a project with billing and GPU quota for the machine you pick (new projects start at 0;
see [docs/manual_runbook.md](docs/manual_runbook.md#1-gcp-project-and-quotas)). The study pages under
[`docs/studies/`](docs/studies/) list the exact quotas each study needed.

**Typical sessions**
* **Session 1, measure:** edit sections 1a–1c of the setup notebook, keep `STAGES = ["setup", "env", "bench", "nsys"]`,
  run all cells. It ends by stopping the VM and running the analysis notebook
  [`02_prediction_nsys_analysis.ipynb`](notebooks/02_prediction_nsys_analysis.ipynb) on the new run, which also
  writes a plan of which kernels deserve a deeper look.
* **Session 2, Nsight Compute:** rerun the setup notebook with the same `RUN_ID` and
  `STAGES = ["setup", "env", "ncu"]`. The plan is executed on the VM and the reports come back to `runs/<id>/ncu/`.

**Changing what's measured:** every key of the `VLLM` dict (section 1b) is a vLLM engine argument; the same dict
drives `vllm bench latency` and the profiling harness `scripts/prof_run.py`. The sweep is set by the `BENCH`, `NSYS`
and `NCU` dicts (section 1c). Give each configuration its own `RUN_TAG` so runs never mix settings.

### Safe, cheap and clean by design
```
Laptop (notebook kernel)                        Cloud GPU VM (Spot)
------------------------                        ----------------------------------------
config.json  --upload (tar via gcloud scp)-->   ~/sillybench/runs/<id>/config.json
launch(stage) --gcloud ssh: tmux new -d -->     scripts/job.sh -> run_stage.py <stage>
poll every 60 s <-- exit code, log tail, --     status/<stage>.exit, status/<stage>.json
                    items finished
pull_run  <--tar + sha256 manifest, verified--  runs/<id>/
stop (--discard-local-ssd=true) or delete
```
* **No measurement crosses the network.** All timers, trace timestamps and counters are taken on the VM; your laptop
  only starts jobs and copies files after they finish.
* **Survives disconnects and Spot preemption.** Jobs run in `tmux` on the VM. If the VM is reclaimed, the notebook
  restarts it and resumes; finished items are skipped.
* **Can't run away.** The VM arms a self-shutdown timer, and a failed stage pulls the logs and stops the VM
  (`ON_FAILURE = "stop"`).
* **Stopped, not deleted, by default.** The disk keeps the model and the installed software, so the next session
  starts in about a minute. `FINAL_ACTION = "delete"` is refused unless the transfer was verified.
* **Reproducible.** `config.json` and `env/` are stored with every run, and the VM's packages are pinned.

### Test without spending money
```bash
# needs tmux: the simulated VM runs its jobs in a local tmux (macOS: brew install tmux; Ubuntu: sudo apt install tmux)
source .venv/bin/activate && bash tests/run_offline_test.sh
```
Runs the setup notebook against a fake `gcloud` that simulates the VM on your laptop (seeded with a stored run, one
injected preemption), then the analysis notebook on the pulled data. Expected last line: `offline test passed`.

## Repository layout
```
README.md                 this file
notebooks/
  postN.ipynb             one notebook per post: recomputes the post's numbers from a run folder
  setup.ipynb             measure: VM -> setup -> bench -> nsys -> ncu -> verified transfer -> stop
  02_prediction_nsys_analysis.ipynb   prediction + Nsight Systems analysis + Nsight Compute plan for a run
  README.md               how the post notebooks work
runs/<run_id>/            one folder per measurement run
  config.json             every setting used (model, vLLM args, sweep, nsys/ncu config)
  env/                    environment capture (GPU, driver, clocks, versions)
  bench/ nsys/ ncu/ logs/ raw results; Nsight reports too large for git are release assets (their CSVs are in git)
  analysis/               analysis notebook output: figures, CSV tables, findings.md
  ncu_plan/               generated Nsight Compute commands + targets
  MANIFEST.sha256         checksums of every file pulled from the VM
docs/
  studies/                one page per study: setup, results, pitfalls, versions
  manual_runbook.md       every step of the setup notebook as a manual command, with what it does
scripts/                  run ON THE VM: setup, environment capture, profiling harness, stage runner
gpuprof/                  laptop-side gcloud helpers used by the setup notebook
tests/                    offline end-to-end test with a simulated VM
requirements.txt          laptop environment for the notebooks (no vLLM)
requirements-vm.txt       GPU VM environment, pinned
setup_laptop.sh           creates .venv + the "sillybench" Jupyter kernel
```

## Studies so far

| Study | Runs | Notes |
|---|---|---|
| Qwen3.8-27B on vLLM, 1x NVIDIA H100 | [`2026-10-09_…_initial`](runs/2026-10-09_qwen3.8-27b_bf16_initial/), [`2026-10-13_…_repro`](runs/2026-10-13_qwen3.8-27b_bf16_repro/) | [results, reproduction, pitfalls](docs/studies/qwen3.8-27b_1x-h100.md) |

## Corrections

Found a number that doesn't add up? Open an issue on this repo. Corrections are credited in the post.

## License

* **Code** is under the [MIT License](LICENSE): `scripts/`, `gpuprof/`, `tests/`, `notebooks/`, `setup_laptop.sh`
  and the `requirements*.txt` files.
* **Content** is under [CC BY 4.0](LICENSE-CC-BY-4.0): the measurements in `runs/` (benchmarks, traces, reports,
  figures, tables), `docs/` and the prose of the READMEs. Reuse it freely with credit, e.g.
  "Tushar Garg, [SillyBench](https://tgarg01.github.io/sillybench-site/)".
* **Third-party files keep their own licenses:** `runs/*/analysis/model_config.json` is a copy of the model's
  `config.json` (Qwen3.8-27B, Apache-2.0), and the tool output in `runs/*/logs/` and `runs/*/env/` comes from vLLM
  and NVIDIA's tools.
