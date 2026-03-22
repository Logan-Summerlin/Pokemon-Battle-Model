# RunPod AutoResearch Setup Guide

## 1. RunPod Instance Selection

### Recommended Configuration

| Setting | Value | Reason |
|---------|-------|--------|
| **GPU** | NVIDIA A40 (48GB VRAM) | Large VRAM for batch size scaling, good bf16 throughput, cost-effective |
| **vCPU** | 8+ cores | DataLoader workers need CPU headroom |
| **RAM** | 32+ GB | Full dataset + model + DataLoader buffers |
| **Disk** | 100 GB persistent volume | Dataset (~15GB), checkpoints, experiment logs |
| **Template** | PyTorch 2.2+ / CUDA 12.1 | Matches project requirements |

### Alternative GPUs

| GPU | VRAM | Notes |
|-----|------|-------|
| A6000 | 48GB | Same VRAM, similar performance |
| RTX 4090 | 24GB | Faster per-op but half VRAM; reduce batch size |
| A100 40GB | 40GB | Faster but more expensive |
| L40S | 48GB | Newer Ada Lovelace; good availability |

### Launching

1. Go to runpod.io → GPU Cloud → Deploy
2. Select Community Cloud (cheaper) or Secure Cloud
3. Choose A40 GPU
4. Select PyTorch template (e.g., `runpod/pytorch:2.2.0-py3.11-cuda12.1.1-devel-ubuntu22.04`)
5. Set volume size to 100 GB (persistent)
6. Deploy

---

## 2. Initial Setup

### System Dependencies

```bash
apt-get update && apt-get install -y \
    tmux htop nvtop git curl wget unzip \
    build-essential python3-pip nodejs npm
nvidia-smi  # verify GPU
```

### tmux (Essential)

```bash
tmux new-session -s autoresearch
# Panes: 0=Training, 1=Claude Code, 2=Codex, 3=Monitoring
# Ctrl-b %  (vertical split)
# Ctrl-b "  (horizontal split)
# Reconnect: tmux attach -t autoresearch
```

---

## 3. Repository Setup

```bash
cd /workspace
git clone https://github.com/Logan-Summerlin/Pokemon-Battle-Model.git
cd Pokemon-Battle-Model
pip install -e .
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"
```

### Dataset

```bash
# Download and process (if not already done):
python scripts/download_replays_stratified.py --format gen3ou --num-battles 100000
python scripts/process_dataset.py --input data/raw --output data/processed

# Verify:
python -c "import os; print(f'Battles: {len(os.listdir(\"data/processed/battles\"))}')"
```

### Smoke Test

```bash
python scripts/train_phase4.py --mode smoke
```

---

## 4. Agent Installation

### Claude Code

```bash
npm install -g @anthropic-ai/claude-code
export ANTHROPIC_API_KEY="sk-ant-..."
echo 'export ANTHROPIC_API_KEY="sk-ant-..."' >> ~/.bashrc

cd /workspace/Pokemon-Battle-Model
claude
```

### Codex

```bash
npm install -g @openai/codex
export OPENAI_API_KEY="sk-..."
echo 'export OPENAI_API_KEY="sk-..."' >> ~/.bashrc

cd /workspace/Pokemon-Battle-Model
codex "Your task here"
```

---

## 5. Running Experiments

### First Cycle

```bash
# 1. Evaluate anchor
python Autoresearch/eval_harness.py \
    --checkpoint checkpoints/phase4_p8_lean_50k/seed_42/best_model.pt \
    --data-dir data/processed \
    --output Autoresearch/results/anchor.json

# 2. Run Tier 1 experiment
python Autoresearch/run_experiment.py \
    --name "window_size_5" \
    --parent anchor \
    --tier 1 \
    --config-override max_window=5 \
    --budget-epochs 10 \
    --budget-minutes 30

# 3. Review leaderboard
python Autoresearch/leaderboard.py

# 4. Analyze with Claude Code
claude "Compare experiments/window_size_5 to the anchor. Write analysis to Autoresearch/notes/"
```

### Recommended Experiment Order

```
Day 1 (AR-0 + AR-1 + AR-2):
  1. Register anchor checkpoint
  2. Debug aux heads (speed/role at 0%)
  3. Batch size sweep: 64 → 256 → 512 → 1024
  4. torch.compile() test
  5. DataLoader worker count sweep

Day 2 (AR-3):
  6. Window size: 2 → 3 → 5 → 8
  7. Full dataset with best window size
  8. Class-weighted loss for switches
  9. Label smoothing 0.05

Day 3 (AR-4):
  10. Scale to P8 (4L/256d) with best data config
  11. Scale to P4 (6L/384d) if P8 helps
  12. Dropout sweep on best model
  13. Hierarchical action head test

Day 4 (AR-5):
  14. Combined best config, 3-seed confirmation
  15. Full test evaluation
  16. Write final report
```

---

## 6. Monitoring

```bash
# GPU (dedicated tmux pane):
nvtop
# or: watch -n 1 nvidia-smi

# Training progress:
tail -f experiments/<name>/training.log

# Quick metrics:
python -c "
import json
with open('experiments/<name>/training_report.json') as f:
    r = json.load(f)
for e in r['epoch_metrics']:
    print(f'Epoch {e[\"epoch\"]}: val_acc={e[\"val_accuracy\"]:.4f} loss={e[\"val_loss\"]:.4f}')
"
```

### Verify A40 bf16

```bash
python -c "
import torch
print(f'bf16 supported: {torch.cuda.is_bf16_supported()}')
print(f'Device: {torch.cuda.get_device_name(0)}')
print(f'Compute capability: {torch.cuda.get_device_capability()}')
"
```

---

## 7. Data Management

### Persistent Volume Layout

```
/workspace/
├── Pokemon-Battle-Model/
│   ├── data/processed/        # Dataset
│   ├── checkpoints/           # Model checkpoints
│   ├── Autoresearch/          # Harness & logs
│   └── experiments/           # Per-experiment outputs
└── backups/                   # Periodic backups
```

### Backup

```bash
# Push experiment results to git regularly
git add Autoresearch/
git commit -m "AutoResearch: results through <experiment-id>"
git push
```

---

## 8. Cost Estimate

| Phase | Hours | Cost (~$0.40/hr A40) |
|-------|-------|---------------------|
| AR-0: Setup | 2 | $0.80 |
| AR-1: Fix Issues | 4 | $1.60 |
| AR-2: Throughput | 4 | $1.60 |
| AR-3: Data | 8 | $3.20 |
| AR-4: Architecture | 8 | $3.20 |
| AR-5: Consolidation | 4 | $1.60 |
| **Total** | **~30** | **~$12.00** |

Plus API costs for Claude Code and Codex.

---

## 9. Troubleshooting

| Issue | Fix |
|-------|-----|
| CUDA OOM | Reduce batch size, use gradient accumulation |
| DataLoader hangs | Reduce num_workers, increase shm: `mount -o remount,size=16G /dev/shm` |
| Training NaN | Reduce LR, check data for NaN |
| Pod terminated | Use persistent volume, checkpoint frequently, commit often |

---

## 10. Quick-Start Checklist

```
[ ] Launch RunPod A40 pod with PyTorch template
[ ] Install tmux, htop, nvtop, nodejs
[ ] Clone repository to persistent volume
[ ] pip install -e . and verify CUDA
[ ] Verify dataset
[ ] Run smoke test
[ ] Install Claude Code and Codex
[ ] Set API keys
[ ] Create tmux session
[ ] Register anchor checkpoint
[ ] Run first experiment
[ ] Begin AutoResearch loop
```
