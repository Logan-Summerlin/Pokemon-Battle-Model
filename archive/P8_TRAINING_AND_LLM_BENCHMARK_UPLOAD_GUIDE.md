# P8 Model (Phase 4) Training + LLM Benchmark Upload Guide

This guide is tailored to the repository's Phase 4 experiment plan and provides:
1. a concrete P8 design rationale,
2. a reproducible command sequence to train on 1,000 replays,
3. a standard way to package and upload results so an LLM can benchmark runs,
4. updated throughput tuning guidance for the newer DataLoader/input-pipeline CLI flags.

> **Important update:** recent training-script commits added DataLoader throughput controls (`--num-workers`, `--prefetch-factor`, `--persistent-workers`, `--pin-memory`, and transfer mode toggles), plus mixed precision (`--amp`) support in the underlying trainer. Use the updated commands below when running on GPU.

---

## 1) P8 design rationale (from Phase 4 experiment ladder)

The Phase 4 experiment document defines **P8** as:
- **4 layers**
- **256 hidden size**
- **4 attention heads**
- **window length 20**
- **auxiliary head ON** (`aux_weight=0.2`)
- **value head ON**

Why this is a strong 1,000-replay candidate:
- It preserves the full history window (`max_window=20`), which protects long-horizon tactical signal.
- It is materially cheaper than 6L/384d and 8L/512d, making local iteration practical.
- It keeps both auxiliary and value supervision, which improves representation stability and calibration in this codebase.

The new script `scripts/train_p8_1k.py` automates exactly that configuration by wrapping `scripts/train_phase4.py` and then writing a consolidated benchmark JSON.

---

## 2) Prerequisites

From repository root, use the command block that matches your shell:

Mac/Linux (bash/zsh):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

Windows Command Prompt (`cmd.exe`):

```bat
python -m venv .venv
.\.venv\Scripts\activate.bat
pip install -e ".[dev]"
```

> If you see `'source' is not recognized as an internal or external command`, you are on Windows `cmd.exe`; use either the PowerShell or `cmd` activation command above instead of `source`.

Verify that processed data exists:

Mac/Linux:

```bash
test -d data/processed/battles && echo "processed data found"
```

Windows PowerShell:

```powershell
if (Test-Path "data/processed/battles") { Write-Host "processed data found" }
```

If you need to build processed tensors from raw replays, run the repo data pipeline first (download/parse/process scripts in `scripts/`).

Quick data sanity check before spending compute:

```bash
python -c "from pathlib import Path; p=Path('data/processed/battles'); files=list(p.glob('*.pt')); print('battle tensor files:', len(files)); print('sample:', files[0] if files else 'none')"
```

### Windows NVIDIA GPU setup (GTX 1650)

If training logs show `Device: cpu`, PyTorch cannot see CUDA yet. On Windows PowerShell, run:

```powershell
python -c "import torch; print('torch', torch.__version__); print('cuda available', torch.cuda.is_available()); print('torch cuda build', torch.version.cuda); print('gpu count', torch.cuda.device_count())"
```

If `cuda available` is `False`, install CUDA-enabled PyTorch wheels pinned to this repo's version range (`torch<2.4`, `torchvision<0.19`):

```powershell
pip uninstall -y torch torchvision torchaudio
pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1
```

Then run `pip install -e ".[dev]"` to ensure project dependencies are consistent, and verify again with the same `python -c ...` command. If it prints `cuda available True`, rerun the training command and it will use your NVIDIA GPU automatically.

### Super-simple version (for first-time command line users)

If you are new to Python projects, follow these exact steps slowly:

1. **Open a shell**: Terminal (Mac/Linux), **PowerShell** (Windows), or **Command Prompt** (Windows).
2. **Go into this project folder** (the folder where this repository is saved):

```bash
cd /path/to/Pokemon-Battle-Model
```

3. **Create a project-only Python environment** (this keeps packages isolated):

```bash
python -m venv .venv
```

4. **Turn the environment on**:

Mac/Linux:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Windows Command Prompt (`cmd.exe`):

```bat
.\.venv\Scripts\activate.bat
```

5. **Install this project's required Python packages**:

```bash
pip install -e ".[dev]"
```

6. **Check that prepared training data exists**:

Mac/Linux:

```bash
test -d data/processed/battles && echo "processed data found"
```

Windows PowerShell:

```powershell
if (Test-Path "data/processed/battles") { Write-Host "processed data found" }
```

If you do not see `processed data found`, pause here and prepare data first.

7. **Run a safe test command first** (shows what training would do, but does not train yet):

```bash
python scripts/train_p8_1k.py --dry-run --seeds 42
```

8. **Start real training**:

```bash
python scripts/train_p8_1k.py --num-battles 1000 --seeds 42 --batch-size 32 --epochs 30 --patience 7 --output-root checkpoints/phase4_p8_1k
```

9. **When training finishes**, your main result file is:

```text
checkpoints/phase4_p8_1k/p8_benchmark_summary.json
```

10. **Next time you come back**, open Terminal, `cd` into the repo again, and reactivate the environment before running any Python command.

---

## 3) P8 training stack summary (what the wrapper now configures)

`scripts/train_p8_1k.py` is a strict P8 wrapper around `scripts/train_phase4.py`. For each seed it pins:

- `--num-layers 4`
- `--hidden-dim 256`
- `--num-heads 4`
- `--max-window 20`
- `--aux-weight 0.2`
- value head enabled (it does **not** pass `--no-value-head`)

Recent updates also mean the wrapper now forwards input-pipeline flags to the base trainer, so you can tune host-side loading and host→device transfer behavior without editing Python.

---

## 4) DataLoader input-pipeline CLI flags (new, recommended)

These flags are accepted by `scripts/train_p8_1k.py` and forwarded to `scripts/train_phase4.py`.

- `--num-workers N`
  - Number of DataLoader worker processes.
  - If omitted, trainer auto-selects:
    - CUDA: `min(8, max(1, cpu_count // 2))`
    - CPU-only: `0`
- `--prefetch-factor K`
  - Prefetched batches per worker when `num_workers > 0`.
  - Default: `4`.
- `--persistent-workers` / `--no-persistent-workers`
  - Keep workers alive across epochs (or disable).
  - Auto default: enabled when `num_workers > 0`.
- `--pin-memory` / `--no-pin-memory`
  - Enable pinned host memory for faster H2D copies (or disable).
  - Auto default: enabled on CUDA.
- `--non-blocking-transfer` / `--blocking-transfer`
  - Control `tensor.to(device, non_blocking=...)` during train/val/test loops.
  - Auto default: non-blocking when CUDA + pinned memory are enabled.

### Practical tuning presets

**Preset A (safe GPU default):**

```bash
python scripts/train_p8_1k.py \
  --num-battles 1000 --seeds 42 \
  --batch-size 32 --epochs 30 --patience 7 \
  --num-workers 4 --prefetch-factor 4 \
  --persistent-workers --pin-memory --non-blocking-transfer \
  --output-root checkpoints/phase4_p8_1k
```

**Preset B (higher-throughput GPU host):**

```bash
python scripts/train_p8_1k.py \
  --num-battles 1000 --seeds 42 43 44 \
  --batch-size 32 --epochs 30 --patience 7 \
  --num-workers 8 --prefetch-factor 6 \
  --persistent-workers --pin-memory --non-blocking-transfer \
  --output-root checkpoints/phase4_p8_1k
```

**Preset C (CPU-only / debugging stability):**

```bash
python scripts/train_p8_1k.py \
  --num-battles 1000 --seeds 42 \
  --batch-size 16 --epochs 10 --patience 3 \
  --num-workers 0 --no-pin-memory --blocking-transfer \
  --output-root checkpoints/phase4_p8_1k_cpu_debug
```

> Tip: monitor epoch `examples_per_sec` and wall time in `training_report.json` and choose the fastest stable setting that does not trigger DataLoader worker crashes or host RAM pressure.

---

## 5) Quick dry run (sanity check)

Use dry-run to print exact training commands without starting training:

```bash
python scripts/train_p8_1k.py --dry-run --seeds 42 43 44
```

This ensures output directories and command formatting are correct before spending compute.

---

## 6) Train P8 on 1,000 replays

### Recommended single-seed run

```bash
python scripts/train_p8_1k.py \
  --num-battles 1000 \
  --seeds 42 \
  --batch-size 32 \
  --epochs 30 \
  --patience 7 \
  --num-workers 4 \
  --prefetch-factor 4 \
  --persistent-workers \
  --pin-memory \
  --non-blocking-transfer \
  --output-root checkpoints/phase4_p8_1k
```

### Recommended 3-seed stability run

```bash
python scripts/train_p8_1k.py \
  --num-battles 1000 \
  --seeds 42 43 44 \
  --batch-size 32 \
  --epochs 30 \
  --patience 7 \
  --num-workers 8 \
  --prefetch-factor 4 \
  --persistent-workers \
  --pin-memory \
  --non-blocking-transfer \
  --output-root checkpoints/phase4_p8_1k
```

What this script does:
1. launches `scripts/train_phase4.py` once per seed with strict P8 architecture,
2. stores per-seed artifacts in `checkpoints/phase4_p8_1k/seed_<seed>/`,
3. reads each `training_report.json`,
4. writes aggregate metrics to `checkpoints/phase4_p8_1k/p8_benchmark_summary.json`.

## 7) Optional advanced run: direct `train_phase4.py` with AMP

If you need explicit AMP control (not exposed by `train_p8_1k.py`), run the base trainer directly with the same P8 architecture:

```bash
python scripts/train_phase4.py \
  --mode full \
  --data-dir data/processed \
  --num-battles 1000 \
  --num-layers 4 --hidden-dim 256 --num-heads 4 \
  --max-window 20 --aux-weight 0.2 \
  --batch-size 32 --epochs 30 --patience 7 \
  --amp auto \
  --num-workers 4 --prefetch-factor 4 \
  --persistent-workers --pin-memory --non-blocking-transfer \
  --seed 42 \
  --checkpoint-dir checkpoints/phase4_p8_1k/seed_42 \
  --report-path checkpoints/phase4_p8_1k/seed_42/training_report.json
```

`--amp auto` picks bf16 on supported CUDA devices, otherwise fp16; use `--amp off` for deterministic debugging.

---

## 8) Output files you should keep

After training, preserve these files:

- `checkpoints/phase4_p8_1k/seed_42/training_report.json` (and other seeds)
- `checkpoints/phase4_p8_1k/seed_42/best_model.pt` (and other seeds)
- `checkpoints/phase4_p8_1k/p8_benchmark_summary.json`

The **benchmark summary JSON** is the preferred single file for LLM analysis because it already includes per-seed and mean/std metrics.

---

## 9) Upload workflow for LLM benchmark comparison

### A. Prepare upload bundle

```bash
tar -czf p8_1k_results.tar.gz checkpoints/phase4_p8_1k
```

### B. Upload to your LLM workspace

Upload either:
- `p8_1k_results.tar.gz`, or
- just `p8_benchmark_summary.json` + selected seed `training_report.json` files.

### C. Prompt template for LLM benchmarking

Use this template with your uploaded files:

```text
You are benchmarking Pokemon Battle Model Phase 4 runs.
Use p8_benchmark_summary.json as the primary source of truth.

Tasks:
1) Report mean/std of top-1, top-3, NLL, ECE, wall time.
2) Identify the best seed by top-1 and summarize trade-offs.
3) Compare P8 against my baseline run(s) if provided.
4) Check whether P8 satisfies the Phase 4 "no meaningful loss" thresholds relative to baseline.
5) Produce a concise recommendation: keep P8, tune P8, or move to P5/P6/P4.

Output a markdown table and a final recommendation section.
```

---

## 10) Interpreting results (what to look for)

Prioritize these from `p8_benchmark_summary.json`:
- `aggregate.test_top1_accuracy.mean`
- `aggregate.test_top3_accuracy.mean`
- `aggregate.test_nll.mean`
- `aggregate.test_ece.mean`
- `aggregate.wall_time_min.mean`
- `aggregate.*.std` across seeds (stability)

Operational guidance:
- If top-1/top-3 are close to your stronger baseline while wall time is much lower, P8 is a good deployment candidate.
- If seed std is high, rerun with 3-5 seeds before concluding.
- If ECE is weak but top-1 is good, consider calibration/post-hoc temperature scaling before ladder play.

---

## 11) Optional: compare two experiment folders locally

If you produce another run folder (e.g., `checkpoints/phase4_p6_1k`), compare both summaries in your LLM prompt. Keep both summary JSON files in the same upload so the LLM can directly contrast means/stds.

---

## 12) Reproducibility checklist

- Pin the same code commit hash.
- Keep `num-battles`, architecture, and split seed policy fixed.
- Report hardware (CPU/GPU, RAM).
- Run at least 3 seeds for final benchmark statements.
- Archive full checkpoint folder and summary JSON.
