# P8-Lean Training on RunPod + LLM Benchmark Upload Guide

Step-by-step guide to train the P8-Lean BattleTransformer on a RunPod GPU instance, package results, and upload for LLM benchmarking.

---

## Table of Contents

1. [P8-Lean Design Rationale](#1-p8-lean-design-rationale)
2. [RunPod SSH Key Setup](#2-runpod-ssh-key-setup)
3. [Create and Connect to a RunPod Instance](#3-create-and-connect-to-a-runpod-instance)
4. [Environment Setup on RunPod](#4-environment-setup-on-runpod)
5. [Data Preparation](#5-data-preparation)
6. [Dry Run (Sanity Check)](#6-dry-run-sanity-check)
7. [Training Commands](#7-training-commands)
8. [DataLoader Tuning Presets](#8-dataloader-tuning-presets)
9. [Advanced: Direct `train_phase4.py` with AMP](#9-advanced-direct-train_phase4py-with-amp)
10. [Output Files](#10-output-files)
11. [Download Results from RunPod](#11-download-results-from-runpod)
12. [LLM Benchmark Upload Workflow](#12-llm-benchmark-upload-workflow)
13. [Interpreting Results](#13-interpreting-results)
14. [Reproducibility Checklist](#14-reproducibility-checklist)

---

## 1) P8-Lean Design Rationale

P8-Lean is a compact, fast-iterating variant of the BattleTransformer optimized for Gen 3 OU:

| Parameter | Value |
|-----------|-------|
| Layers | 3 |
| Hidden dim | 224 |
| Attention heads | 4 |
| FFN multiplier | 3x |
| Parameters | ~1.95M |
| Max window | 2 (minimal history; Gen 3's smaller metagame needs less context) |
| Auxiliary head | ON (aux_weight=0.2) — predicts opponent items, speed tiers, roles, move families |
| Value head | OFF |
| Dead feature pruning | ON |
| Dropout | 0.1 |
| Embeddings | Compressed (species=48, moves=24, items=16, abilities=16, types=12) |

Why P8-Lean over P8:
- ~2x fewer parameters (1.95M vs 3.6M) — trains faster, lower GPU memory.
- Compressed embeddings reduce overfitting on the smaller Gen 3 vocabulary.
- Window of 2 is sufficient for Gen 3's less pivot-heavy metagame (no U-turn/Volt Switch).
- Ideal for rapid experimentation and hyperparameter sweeps before scaling up.

The wrapper script `scripts/train_p8_lean.py` pins this exact architecture by calling `scripts/train_phase4.py` with the correct flags.

---

## 2) RunPod SSH Key Setup

You need an SSH key to securely connect to your RunPod instance. If you already have one, skip to step 2d.

### 2a) Generate an SSH key pair (one-time setup)

On your **local machine** (Mac/Linux terminal or Windows PowerShell):

```bash
ssh-keygen -t ed25519 -C "your-email@example.com"
```

- When prompted for a file location, press **Enter** to accept the default (`~/.ssh/id_ed25519`).
- When prompted for a passphrase, either enter one (recommended) or press **Enter** twice for no passphrase.

This creates two files:
- `~/.ssh/id_ed25519` — your **private** key (never share this)
- `~/.ssh/id_ed25519.pub` — your **public** key (this goes to RunPod)

### 2b) Copy your public key

Mac/Linux:

```bash
cat ~/.ssh/id_ed25519.pub
```

Windows PowerShell:

```powershell
Get-Content $env:USERPROFILE\.ssh\id_ed25519.pub
```

Copy the entire output (starts with `ssh-ed25519 ...`).

### 2c) Add the public key to RunPod

1. Log in to [runpod.io](https://www.runpod.io/).
2. Click your profile icon (top-right) → **Settings**.
3. Scroll down to **SSH Public Keys**.
4. Click **Add SSH Key**.
5. Paste your public key into the text box.
6. Give it a name (e.g., "my-laptop") and click **Save**.

All new pods you create will automatically include this key in their `authorized_keys`.

### 2d) Verify your key is configured

After adding the key, you should see it listed under **Settings → SSH Public Keys** in the RunPod dashboard.

---

## 3) Create and Connect to a RunPod Instance

### 3a) Choose a GPU pod

1. Go to [runpod.io](https://www.runpod.io/) → **Pods** → **+ New Pod**.
2. **Recommended GPU**: Any CUDA-capable GPU with ≥8 GB VRAM. Good options:
   - **RTX 3090 / RTX 4090** — fast and cost-effective for P8-Lean's ~1.95M params.
   - **A100 (40 GB)** — overkill for P8-Lean, but useful if running multi-seed or P4 later.
   - **RTX A4000 / A5000** — budget-friendly, sufficient for P8-Lean.
3. **Template**: Select **RunPod PyTorch 2.x** (comes with CUDA, cuDNN, and Python pre-installed).
4. **Disk**: At least **20 GB** container disk (code + data + checkpoints).
5. **Volume** (optional): Attach a persistent network volume if you want data/checkpoints to survive pod restarts.
6. Click **Deploy**.

### 3b) Find your SSH connection info

1. Once the pod status shows **Running**, click on it to expand details.
2. Look for **SSH over exposed TCP** — it will show something like:

   ```
   ssh root@<pod-ip> -p <port> -i ~/.ssh/id_ed25519
   ```

3. Note the IP address and port number.

### 3c) Connect via SSH

From your **local machine** terminal:

```bash
ssh root@<pod-ip> -p <port> -i ~/.ssh/id_ed25519
```

Replace `<pod-ip>` and `<port>` with the values from the RunPod dashboard.

If prompted "Are you sure you want to continue connecting?", type `yes`.

You should now have a root shell on the RunPod instance.

### 3d) Troubleshooting SSH

| Problem | Fix |
|---------|-----|
| `Permission denied (publickey)` | Verify the key in RunPod Settings matches `~/.ssh/id_ed25519.pub` on your local machine. Recreate the pod if needed. |
| `Connection refused` | Pod may still be starting. Wait 30s and retry. Check pod status on dashboard. |
| `Connection timed out` | Check that your local network allows outbound SSH on the listed port. Some corporate firewalls block non-standard ports. |
| Key not found | Specify the full path: `ssh -i /full/path/to/id_ed25519 root@<pod-ip> -p <port>` |

---

## 4) Environment Setup on RunPod

Once connected via SSH, run these commands:

### 4a) Clone the repository

```bash
cd /workspace
git clone https://github.com/Logan-Summerlin/Pokemon-Battle-Model.git
cd Pokemon-Battle-Model
```

> If the repo is private, use HTTPS with a personal access token or set up a deploy key.

### 4b) Create a virtual environment and install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### 4c) Verify GPU access

```bash
python -c "import torch; print('torch', torch.__version__); print('cuda available', torch.cuda.is_available()); print('device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```

Expected output (example):

```
torch 2.3.x
cuda available True
device NVIDIA GeForce RTX 4090
```

If `cuda available` is `False`, the RunPod PyTorch template may need a different CUDA wheel. Run:

```bash
pip uninstall -y torch torchvision torchaudio
pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1
pip install -e ".[dev]"
```

---

## 5) Data Preparation

### 5a) Check for existing processed data

```bash
test -d data/processed/battles && echo "processed data found" || echo "no processed data"
```

### 5b) If processed data is not present

Option 1 — **Upload from your local machine** (if you already have processed data):

From your **local machine**:

```bash
scp -P <port> -i ~/.ssh/id_ed25519 -r /path/to/data/processed root@<pod-ip>:/workspace/Pokemon-Battle-Model/data/processed
```

Option 2 — **Download and process from scratch**:

```bash
# Download raw replays from Metamon dataset
python scripts/download_replays_stratified.py --generation gen3ou --output-dir data/raw

# Process into tensorized training data
python scripts/process_dataset.py --generation gen3ou --input-dir data/raw --output-dir data/processed
```

### 5c) Verify data integrity

```bash
python -c "
from pathlib import Path
battles = list(Path('data/processed/battles').glob('*.npz'))
print(f'Battle tensor files: {len(battles)}')
print(f'Sample: {battles[0].name if battles else \"none\"}')
"
```

You should see thousands of `.npz` files. If the count is 0, data processing did not complete successfully.

---

## 6) Dry Run (Sanity Check)

Before spending GPU hours, print the exact training commands without executing:

```bash
python scripts/train_p8_lean.py --dry-run --seeds 42
```

This shows the full `train_phase4.py` invocation that would run. Verify the architecture flags match P8-Lean:
- `--num-layers 3 --hidden-dim 224 --num-heads 4 --ffn-multiplier 3`
- `--no-value-head --prune-dead-features`
- Compressed embedding dims

---

## 7) Training Commands

### Single-seed run (recommended first run)

```bash
python scripts/train_p8_lean.py \
  --num-battles 10000 \
  --seeds 42 \
  --batch-size 64 \
  --epochs 30 \
  --patience 7 \
  --num-workers 4 \
  --prefetch-factor 4 \
  --persistent-workers \
  --pin-memory \
  --non-blocking-transfer \
  --output-root checkpoints/phase4_gen3_p8_lean
```

### 3-seed stability run (for benchmark-quality results)

```bash
python scripts/train_p8_lean.py \
  --num-battles 10000 \
  --seeds 42 43 44 \
  --batch-size 64 \
  --epochs 30 \
  --patience 7 \
  --num-workers 8 \
  --prefetch-factor 4 \
  --persistent-workers \
  --pin-memory \
  --non-blocking-transfer \
  --output-root checkpoints/phase4_gen3_p8_lean
```

### What the wrapper does

1. Launches `scripts/train_phase4.py` once per seed with strict P8-Lean architecture.
2. Stores per-seed artifacts in `checkpoints/phase4_gen3_p8_lean/seed_<seed>/`.
3. Reads each `training_report.json`.
4. Writes aggregate metrics to `checkpoints/phase4_gen3_p8_lean/p8_lean_benchmark_summary.json`.

### Running in the background (recommended for long runs)

Use `tmux` or `screen` so training survives SSH disconnects:

```bash
tmux new -s train

# Inside tmux, run the training command above
python scripts/train_p8_lean.py \
  --num-battles 10000 \
  --seeds 42 43 44 \
  --batch-size 64 \
  --epochs 30 \
  --patience 7 \
  --num-workers 8 \
  --prefetch-factor 4 \
  --persistent-workers \
  --pin-memory \
  --non-blocking-transfer \
  --output-root checkpoints/phase4_gen3_p8_lean

# Detach: Ctrl+B, then D
# Reattach later: tmux attach -t train
```

---

## 8) DataLoader Tuning Presets

All flags are accepted by `train_p8_lean.py` and forwarded to `train_phase4.py`.

| Flag | Description |
|------|-------------|
| `--num-workers N` | DataLoader worker processes. Auto: `min(8, cpu_count // 2)` on CUDA, `0` on CPU. |
| `--prefetch-factor K` | Batches prefetched per worker (default 4). |
| `--persistent-workers` | Keep workers alive across epochs. |
| `--pin-memory` | Pin host memory for faster GPU transfers. |
| `--non-blocking-transfer` | Async host→device copies (requires CUDA + pinned memory). |

### Preset A — Safe GPU default (start here)

```bash
--num-workers 4 --prefetch-factor 4 --persistent-workers --pin-memory --non-blocking-transfer
```

### Preset B — High-throughput GPU (multi-seed, large data)

```bash
--num-workers 8 --prefetch-factor 6 --persistent-workers --pin-memory --non-blocking-transfer
```

### Preset C — CPU-only / debugging

```bash
--num-workers 0 --no-pin-memory --blocking-transfer --batch-size 16 --epochs 5
```

Monitor `examples_per_sec` and wall time in `training_report.json` to find the fastest stable setting for your hardware.

---

## 9) Advanced: Direct `train_phase4.py` with AMP

For explicit mixed-precision control (not exposed by the wrapper), run the base trainer directly:

```bash
python scripts/train_phase4.py \
  --mode full \
  --data-dir data/processed \
  --num-battles 10000 \
  --num-layers 3 --hidden-dim 224 --num-heads 4 \
  --ffn-multiplier 3 \
  --species-embedding-dim 48 --move-embedding-dim 24 \
  --item-embedding-dim 16 --ability-embedding-dim 16 --type-embedding-dim 12 \
  --max-window 2 --aux-weight 0.2 --no-value-head --prune-dead-features \
  --dropout 0.1 \
  --batch-size 64 --epochs 30 --patience 7 \
  --lr 1e-4 --weight-decay 0.01 --warmup-steps 300 \
  --amp auto \
  --num-workers 4 --prefetch-factor 4 \
  --persistent-workers --pin-memory --non-blocking-transfer \
  --seed 42 \
  --checkpoint-dir checkpoints/phase4_gen3_p8_lean/seed_42 \
  --report-path checkpoints/phase4_gen3_p8_lean/seed_42/training_report.json
```

`--amp auto` picks bf16 on supported CUDA devices (A100, RTX 30xx/40xx), otherwise fp16. Use `--amp off` for deterministic debugging.

---

## 10) Output Files

After training completes, preserve these files:

```
checkpoints/phase4_gen3_p8_lean/
├── seed_42/
│   ├── training_report.json    ← per-epoch metrics, test results, calibration
│   └── best_model.pt           ← best checkpoint by validation loss
├── seed_43/
│   ├── training_report.json
│   └── best_model.pt
├── seed_44/
│   ├── training_report.json
│   └── best_model.pt
└── p8_lean_benchmark_summary.json  ← aggregate: per-seed + mean/std metrics
```

The **benchmark summary JSON** is the single file for LLM analysis — it already includes per-seed and aggregate mean/std metrics.

---

## 11) Download Results from RunPod

### Option A — Download the full checkpoint folder

From your **local machine**:

```bash
scp -P <port> -i ~/.ssh/id_ed25519 -r \
  root@<pod-ip>:/workspace/Pokemon-Battle-Model/checkpoints/phase4_gen3_p8_lean \
  ./p8_lean_results/
```

### Option B — Download just the benchmark summary

```bash
scp -P <port> -i ~/.ssh/id_ed25519 \
  root@<pod-ip>:/workspace/Pokemon-Battle-Model/checkpoints/phase4_gen3_p8_lean/p8_lean_benchmark_summary.json \
  ./
```

### Option C — Create a tarball on the pod, then download

On RunPod:

```bash
cd /workspace/Pokemon-Battle-Model
tar -czf p8_lean_results.tar.gz checkpoints/phase4_gen3_p8_lean
```

From your local machine:

```bash
scp -P <port> -i ~/.ssh/id_ed25519 \
  root@<pod-ip>:/workspace/Pokemon-Battle-Model/p8_lean_results.tar.gz \
  ./
```

---

## 12) LLM Benchmark Upload Workflow

### A) Upload to your LLM workspace

Upload either:
- `p8_lean_results.tar.gz` (full artifacts), or
- `p8_lean_benchmark_summary.json` + selected seed `training_report.json` files.

### B) Prompt template for LLM benchmarking

```text
You are benchmarking Pokemon Battle Model Phase 4 runs.
Use p8_lean_benchmark_summary.json as the primary source of truth.

Tasks:
1) Report mean/std of top-1, top-3, NLL, ECE, wall time.
2) Identify the best seed by top-1 and summarize trade-offs.
3) Compare P8-Lean against my baseline run(s) if provided.
4) Check whether P8-Lean satisfies the Phase 4 "no meaningful loss" thresholds relative to baseline.
5) Produce a concise recommendation: keep P8-Lean, tune P8-Lean, or scale to P8/P4.

Output a markdown table and a final recommendation section.
```

### C) Compare multiple experiments

If you have other run folders (e.g., `checkpoints/phase4_p8_1k/`), upload both summary JSONs in the same prompt so the LLM can directly contrast means/stds.

---

## 13) Interpreting Results

Prioritize these from `p8_lean_benchmark_summary.json`:

| Metric | JSON Path | What to Look For |
|--------|-----------|------------------|
| Top-1 accuracy | `aggregate.test_top1_accuracy.mean` | Higher is better; compare to baseline |
| Top-3 accuracy | `aggregate.test_top3_accuracy.mean` | Policy spread quality |
| NLL | `aggregate.test_nll.mean` | Lower is better; calibration-aware loss |
| ECE | `aggregate.test_ece.mean` | Lower is better; <0.05 is well-calibrated |
| Wall time | `aggregate.wall_time_min.mean` | Training speed; P8-Lean should be ~2x faster than P8 |
| Seed stability | `aggregate.*.std` | Low std across seeds = reliable result |

Decision guidelines:
- **Top-1/top-3 close to baseline + much lower wall time** → P8-Lean is a good iteration candidate.
- **High seed std** → rerun with 3–5 seeds before drawing conclusions.
- **Weak ECE but good top-1** → consider post-hoc temperature scaling before ladder play.
- **Significant accuracy loss vs P8** → P8-Lean may be too compressed; try P8 or increase window.

---

## 14) Reproducibility Checklist

- [ ] Pin the same git commit hash across runs.
- [ ] Keep `--num-battles`, architecture flags, and split seed fixed.
- [ ] Record hardware: GPU model, VRAM, CPU cores, RAM.
- [ ] Run at least 3 seeds for final benchmark statements.
- [ ] Archive the full checkpoint folder and summary JSON.
- [ ] Note the RunPod template/image used (for CUDA/driver reproducibility).
- [ ] Save `pip freeze` output alongside results:
  ```bash
  pip freeze > requirements_frozen.txt
  ```
