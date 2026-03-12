# Phase 4 Cloud A100 Execution Plan (25K+ Battles)

## Goal

Run the Phase 4 pipeline on an A100-class cloud GPU instance so we can train the windowed BattleTransformer on at least 25,000 processed battles in a few hours, while keeping the project’s hidden-information constraints and reproducibility standards intact.

## What already exists in this repo

The repository already has the key scripts needed for this workflow:

1. **Raw replay download**: `scripts/download_replays.py`
2. **Replay processing / tensorization**: `scripts/process_dataset.py`
3. **Phase 4 training**: `scripts/train_phase4.py`

`scripts/train_phase4.py` supports smoke/small/full modes plus overrides for model size, number of battles, batch size, max history window, optimizer settings, checkpoint location, and report output.

## Recommended cloud providers (A100)

Use the provider that best matches your team’s existing account + storage stack:

1. **AWS (recommended for production)**
   - Typical instance classes: `p4d`, `p4de`, or any A100-backed GPU VM available in your region.
   - Best when you want durable artifact storage in S3 and controlled IAM access.

2. **GCP**
   - A2 family with A100 GPUs.
   - Best when using GCS and managed networking in Google Cloud projects.

3. **Azure**
   - NC A100 v4 family.
   - Best for teams already standardized on Azure networking and RBAC.

4. **Specialized GPU platforms (fastest setup)**
   - Lambda, RunPod, CoreWeave, or Paperspace.
   - Usually easiest path for rapid experimentation and short training bursts.

## Single-node target setup (practical default)

For 25K battles, use **one A100 80GB** (or one 40GB with slightly smaller per-device batches), plus enough CPU and disk for preprocessing.

- GPU: 1× A100
- CPU: 16+ vCPU
- RAM: 64+ GB
- Storage: 500+ GB NVMe (or attach high-throughput block volume)
- OS: Ubuntu 22.04
- Python: 3.11

## End-to-end runbook

### 1) Provision and connect

```bash
# after instance creation
ssh -i <key>.pem ubuntu@<public-ip>
```

### 2) System dependencies

```bash
sudo apt-get update
sudo apt-get install -y git python3.11 python3.11-venv python3-pip build-essential
```

### 3) Clone and install project

```bash
git clone <your-repo-url> Pokemon-Battle-Model
cd Pokemon-Battle-Model
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"
```

### 4) Verify GPU + torch

```bash
nvidia-smi
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### 5) Build / refresh data (if needed)

If raw data is not already copied to the machine:

```bash
python scripts/download_replays.py --format gen9ou --output-dir data/raw
```

Then process to model-ready tensors:

```bash
python scripts/process_dataset.py --input-dir data/raw --output-dir data/processed --max-battles 25000 --max-turns 20
```

### 6) Run Phase 4 training on 25K battles

Start from the small preset but override for A100 throughput:

```bash
python scripts/train_phase4.py \
  --mode small \
  --data-dir data/processed \
  --num-battles 25000 \
  --hidden-dim 384 \
  --num-layers 6 \
  --num-heads 6 \
  --batch-size 256 \
  --epochs 20 \
  --max-window 20 \
  --lr 1e-4 \
  --warmup-steps 500 \
  --grad-accum 1 \
  --checkpoint-dir checkpoints/phase4_a100_25k \
  --report-path checkpoints/phase4_a100_25k/training_report.json
```

If you hit OOM on a 40GB card, drop to `--batch-size 128` and set `--grad-accum 2` to keep an effective 256 batch.

### 7) Persist artifacts off-node

```bash
# AWS example
aws s3 sync checkpoints/phase4_a100_25k s3://<bucket>/phase4_a100_25k/
aws s3 cp checkpoints/phase4_a100_25k/training_report.json s3://<bucket>/phase4_a100_25k/
```

## Throughput strategy to stay in “few hours” range

1. **Keep windowed per-turn dataset** (already implemented) to maximize training signal per battle.
2. **Use max window = 20** for full context unless throughput becomes the bottleneck.
3. **Prefer bf16/AMP on A100** via CUDA-enabled PyTorch runtime.
4. **Scale effective batch size to ~256** using gradient accumulation as needed.
5. **Use 6L/384d first**, then optionally re-run 8L/512d once baseline throughput is confirmed.

## Suggested experiment sequence

1. **Smoke check (10–15 min)**

```bash
python scripts/train_phase4.py --mode smoke --num-battles 500 --checkpoint-dir checkpoints/smoke_a100
```

2. **Pilot (1–2 hours)**

```bash
python scripts/train_phase4.py --mode small --num-battles 5000 --batch-size 256 --epochs 10 --checkpoint-dir checkpoints/pilot_a100
```

3. **Main 25K run (target: few hours)**

Use the full command in Step 6.

## Multi-GPU extension (optional)

For faster turnaround or larger model variants:

- Use `torchrun --nproc_per_node=<num_gpus>` and DistributedDataParallel integration in `train_phase4.py`.
- Keep global batch fixed while scaling per-GPU micro-batch.
- Store one checkpoint per epoch and keep best validation model.

(DDP wiring is not currently in `train_phase4.py`; add it only after stable single-GPU runs.)

## Operational checklist

- Pin git commit hash for each run.
- Save full training command in run notes.
- Archive `training_report.json` + best checkpoint together.
- Record `nvidia-smi` output and CUDA/PyTorch versions.
- Keep train/val/test split strategy battle-level (already enforced in the processing flow).

## Cost-control tips

- Preprocess once, train many times from `data/processed` snapshots.
- Use spot/preemptible A100 only if you checkpoint every epoch.
- Auto-stop instance at job completion.
- Sync checkpoints every epoch to object storage.

## Recommended default

If you want a single default path with minimal ops complexity:

- **Provider**: AWS (or whichever your team already uses)
- **Compute**: 1× A100 80GB, 16+ vCPU, 64+ GB RAM
- **Run**: `train_phase4.py` at 25K battles, 6L/384d, window 20, effective batch 256
- **Artifacts**: checkpoint dir + JSON report pushed to object storage at end of run
