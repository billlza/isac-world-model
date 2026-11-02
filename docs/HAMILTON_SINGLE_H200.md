# Hamilton: first run on one whole H200

This pilot validates CUDA execution and checkpoint continuation of the existing
ToyScene trainer. It does **not** move `active_observation_study.py` to the GPU,
validate the complete PHY-to-world-model loop, or establish scientific superiority.
The model starts from random initialization.

## Allocation and existing environment

Hamilton's official GPU examples specify `--partition=cuda` and
`--gres=gpu:h200_nvl:1` for one whole H200. `--gres=gpu:1` currently defaults to
an 18 GB slice. CPU cores, host RAM and temporary disk are allocated automatically;
do not add `-n`, `-c`, `--cpus-per-task`, `--mem` or temporary-disk requests.

Use an **existing** Linux CUDA-enabled Python interpreter with this project's
dependencies already installed. The pilot creates no environment and installs
nothing. In particular, it does not run `uv sync`: the existing Linux default
dependency selection uses CPU PyTorch. Loading a system CUDA module or changing
`--device` cannot turn a CPU-only wheel into a CUDA build.

From the project checkout root, confirm the commit you intend to test, and set the
absolute interpreter path appropriate to your already prepared environment:

```bash
export ISAC_PYTHON=/absolute/path/to/existing/cuda-python
git rev-parse HEAD
sbatch scripts/hamilton_h200_pilot.sh
```

Replace the example interpreter path before submission. This documentation does
not assert that such an environment is already installed for your account. An
optional `ISAC_RUNS_DIR` selects an existing approved output location; otherwise
the job writes `runs/hamilton-h200-<job-id>` in the checkout. A repeated output
directory is rejected. `slurm-<job-id>.out` is written in the submission directory.
The script uses `SLURM_SUBMIT_DIR`, since Slurm executes a spooled copy of the script;
set `ISAC_PROJECT_ROOT` explicitly if submitting from a different directory.

## What the job verifies

1. `scontrol` confirms a running CUDA job requesting/allocated one whole H200.
2. PyTorch was built with CUDA, CUDA is available, and exactly one device is visible.
3. The CUDA device UUID matches an H200 reported by `nvidia-smi`, with MIG disabled.
   A loose 120 GiB floor also rejects the documented 18 GB slice; advertised memory
   size is not compared for exact equality. Actual bytes and driver are recorded.
4. A small MLP completes two real CUDA forward/backward/AdamW steps with finite
   gradients and a parameter update. This smoke check is separate from project training.
5. The project's actual CLI trains `structured-attention` for one epoch, using
   eight episodes, batch size four, 16 steps and seed seven on CUDA.
6. It resumes `train/resume.pt` into a **new** `resume` output directory with
   `--epochs 2`. Epochs are a **total target**, so continuation performs one more
   epoch. All data/model/optimizer options remain unchanged.

The job preserves Slurm's `CUDA_VISIBLE_DEVICES`; do not replace it with a host GPU
index. If identity, allocation, import, kernel execution or training fails, the
process exits nonzero. There is no CPU fallback and no automatic dependency repair.

## Evidence and interpretation

- `preflight.json`: running/passed/failed status; Git commit, preflight source hash,
  Python/PyTorch/CUDA versions, Slurm metadata, GPU UUID/memory/driver, CUDA smoke result.
- `pilot.json`: overall and per-stage status, actual CLI arguments, wall time and
  peak allocated/reserved CUDA memory, final checkpoint hashes.
- `train/` and `resume/`: trainer manifests, history, best `model.pt`, and `resume.pt`.
- Slurm output: errors and CLI logs, including failures before a Python report exists.

`wall_seconds` includes that CLI invocation's data generation and validation;
`training_seconds` comes from the existing trainer and starts after dataset creation.
Memory peaks are process-local PyTorch allocator observations, not a measurement of
every byte reserved by the driver. The two-epoch run checks execution/continuation,
not convergence, exact resumed-vs-uninterrupted equality, or comparative accuracy.

A completed CPU CI or the unit tests below does not constitute H200 validation.
The latter mock hardware to check rejection and evidence handling without importing
PyTorch; actual GPU validation requires the submitted job's reports.

```bash
python3 -m unittest discover -s tests -p test_hamilton_preflight.py -v
bash -n scripts/hamilton_h200_pilot.sh
```

## Connection and official references

Hamilton's login host is `hamilton8.dur.ac.uk`. Follow the university's MFA/VPN
instructions. Its login nodes have a Tuesday 08:00–10:00 UK maintenance risk period;
the official page says batch jobs on compute nodes continue normally. Do not run
training on a login node.

Checked 2026-11-02:

- [Running jobs and allocation rules](https://dur.ac.uk/research/institutes-and-centres/advanced-research-computing/hamilton-supercomputer/usage/jobs/)
- [Official GPU scripts](https://dur.ac.uk/research/institutes-and-centres/advanced-research-computing/hamilton-supercomputer/usage/jobs/examplegpujobs/)
- [Login instructions](https://dur.ac.uk/research/institutes-and-centres/advanced-research-computing/hamilton-supercomputer/usage/login/)
