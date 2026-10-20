# Notebooks

| Notebook | What it does | Runs on |
|---|---|---|
| [`setup.ipynb`](setup.ipynb) | Creates the GCP GPU VM, runs the benchmarks and the Nsight Systems / Nsight Compute captures with the vLLM settings you choose, and copies everything back into `runs/<run_id>/` | your laptop, drives a GPU VM |
| [`02_prediction_nsys_analysis.ipynb`](02_prediction_nsys_analysis.ipynb) | The full study: first-principles prediction, Nsight Systems analysis, Nsight Compute plan | any laptop |
| [`post1.ipynb`](post1.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tgarg01/SillyBench/blob/post-01/notebooks/post1.ipynb) | Post 1, a loop of 902 kernels: model shape, the 512-token prompt, prefill and decode time, kernels per decode step and each kernel family's share of GPU time, from the Nsight Systems trace | any laptop or Colab |

## How the post notebooks work

Every post notebook reads **one run folder** made by `setup.ipynb` (`runs/<run_id>/`: `config.json`, `bench/`, `nsys/`)
and computes the post's numbers from it with the code shown in the notebook. Anything else a post needs (the
prompt text, the model's `config.json`, the tokenizer) is fetched or written inside the notebook itself.

* **Default:** the run behind the post, named by the `RUN_ID` line near the top of the notebook. Its `config.json` and
  `env/` record the model, GPU and software it was measured with.
* **Your own data:** run `setup.ipynb` on your GPU, then set `RUN_ID` in the post notebook to your new run folder.

## Open a post notebook

**In your browser:** click **Open in Colab** above and Run All. Nothing to install.

**On your machine** (Python 3.10+ and git):
```bash
git clone https://github.com/tgarg01/SillyBench.git
cd SillyBench
bash setup_laptop.sh                    # .venv + the "sillybench" Jupyter kernel (macOS or Linux)
.venv/bin/jupyter lab notebooks/post1.ipynb
```
Windows (PowerShell): `python -m venv .venv`, `.venv\Scripts\pip install -r requirements.txt`, then
`.venv\Scripts\jupyter lab notebooks\post1.ipynb`.
