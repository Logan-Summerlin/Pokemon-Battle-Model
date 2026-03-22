# RunPod AutoResearch Setup Guide

## Setting Up and Running the AutoResearch Experiment on RunPod with Claude Code and Codex

This guide covers end-to-end setup of a RunPod A40 GPU instance for running the Pokemon Battle
Model AutoResearch pipeline with Claude Code and Codex agents.

---

## 1. RunPod Instance Selection

### Recommended Configuration

| Setting | Value | Reason |
|---------|-------|--------|
| **GPU** | NVIDIA A40 (48GB VRAM) | Sweet spot: large VRAM for batch size scaling, good bf16 throughput, cost-effective |
| **vCPU** | 8+ cores | DataLoader workers need CPU headroom |
| **RAM** | 32+ GB | 100K battles dataset + model + DataLoader buffers |
| **Disk** | 100 GB persistent volume | Dataset (~15GB), checkpoints, experiment logs |
| **Template** | PyTorch 2.2+ / CUDA 12.1 | Matches project requirements |

### Alternative GPUs (if A40 unavailable)

| GPU | VRAM | Notes |
|-----|------|-------|
| A6000 | 48GB | Same VRAM, similar performance, sometimes cheaper |
| RTX 4090 | 24GB | Faster per-op but half the VRAM; reduce batch size ceiling |
| A100 40GB | 40GB | Faster but more expensive; good if budget allows |
| L40S | 48GB | Newer Ada Lovelace; good availability on RunPod |

### Launching the Pod

1. Go to [runpod.io](https://runpod.io) → **GPU Cloud** → **Deploy**
2. Select **Community Cloud** or **Secure Cloud** (community is cheaper)
3. Choose the A40 GPU
4. Select a PyTorch template (e.g., `runpod/pytorch:2.2.0-py3.11-cuda12.1.1-devel-ubuntu22.04`)
5. Set volume size to 100 GB (persistent)
6. Deploy

---

## 2. Initial Pod Setup

### Connect via SSH or Web Terminal

```bash
# SSH (recommended for persistent sessions)
ssh root@<pod-ip> -p <port> -i ~/.ssh/id_rsa

# Or use RunPod's web terminal
```

### Install System Dependencies

```bash
apt-get update && apt-get install -y \
    tmux htop nvtop git curl wget unzip \
    build-essential python3-pip nodejs npm

# Verify GPU
nvidia-smi
```

### Set Up tmux (Essential for Surviving Disconnects)

```bash
# Create a persistent tmux session
tmux new-session -s autoresearch

# Pane layout (recommended):
# Pane 0: Training/experiments
# Pane 1: Claude Code
# Pane 2: Codex
# Pane 3: Monitoring (htop/nvtop)

# Split panes:
# Ctrl-b %  (vertical split)
# Ctrl-b "  (horizontal split)
# Ctrl-b o  (switch pane)
```

### Reconnecting After Disconnect

```bash
tmux attach -t autoresearch
```

---

## 3. Repository Setup

### Clone and Configure

```bash
cd /workspace

# Clone the repository
git clone https://github.com/Logan-Summerlin/Pokemon-Battle-Model.git
cd Pokemon-Battle-Model

# Install Python dependencies
pip install -e .
# Or if no setup.py/pyproject.toml install:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install numpy hydra-core wandb websockets pytest

# Verify installation
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"
```

### Prepare the Dataset

```bash
# If data is already processed and in the repo:
ls data/processed/battles/ | head -5
ls data/processed/vocabs/

# If you need to download/process:
python scripts/download_replays_stratified.py --format gen3ou --num-battles 100000
python scripts/process_dataset.py --input data/raw --output data/processed

# Verify data integrity
python -c "
import os
battles = os.listdir('data/processed/battles')
print(f'Battles: {len(battles)}')
"
```

### Verify Training Pipeline

```bash
# Quick smoke test to ensure everything works on A40
python scripts/train_phase4.py --mode smoke
```

---

## 4. Installing Claude Code

### Install via npm

```bash
# Install Claude Code globally
npm install -g @anthropic-ai/claude-code

# Set your API key
export ANTHROPIC_API_KEY="sk-ant-..."

# Add to .bashrc for persistence
echo 'export ANTHROPIC_API_KEY="sk-ant-..."' >> ~/.bashrc
```

### Launch Claude Code

```bash
# Navigate to the project
cd /workspace/Pokemon-Battle-Model

# Start Claude Code
claude

# Or start with a specific task
claude "Analyze the experiment leaderboard and propose the next 3 experiments"
```

### Claude Code Configuration for AutoResearch

Create a `.claude/settings.json` if needed:

```json
{
  "permissions": {
    "allow": [
      "Read",
      "Write",
      "Edit",
      "Bash(python *)",
      "Bash(pytest *)",
      "Bash(git *)",
      "Bash(ls *)",
      "Bash(cat *)",
      "Bash(nvidia-smi)"
    ]
  }
}
```

### Claude Code Workflow for AutoResearch

In a dedicated tmux pane:

```bash
# 1. Start Claude Code
claude

# 2. Ask it to analyze the current state
> "Read the experiment registry and leaderboard. What is the current champion?
>  What are the top 3 highest-value experiments to run next?"

# 3. Ask it to implement an experiment
> "Implement experiment AR3-02: increase window size from 2 to 10.
>  Modify the config and create the run command. Do not modify eval_harness.py."

# 4. Ask it to analyze results
> "Read the training report from experiments/AR3-02a/training_report.json.
>  Compare to the anchor. Write a note to autoresearch/notes/AR3-02a.md."
```

---

## 5. Installing and Using Codex CLI

### Install Codex

```bash
# Install Codex CLI
npm install -g @openai/codex

# Set your API key
export OPENAI_API_KEY="sk-..."
echo 'export OPENAI_API_KEY="sk-..."' >> ~/.bashrc
```

### Launch Codex

```bash
cd /workspace/Pokemon-Battle-Model

# Start Codex with a bounded task
codex "Optimize the DataLoader in scripts/train_phase4.py:
- Increase num_workers to 8
- Add persistent_workers=True
- Benchmark the change by running a 2-epoch smoke test"
```

### Codex Workflow for AutoResearch

Codex is best for bounded code-edit tasks:

```bash
# Throughput optimization
codex "Profile the training loop and identify the bottleneck.
Add timing instrumentation to the dataloader and forward pass."

# Config generation
codex "Create autoresearch/configs/ar3_02_window10.yaml with:
max_window=10, batch_size=256, all other settings from anchor.yaml"

# Script improvements
codex "Add a --max-wall-time flag to autoresearch/run_experiment.py
that kills the run after the specified minutes."
```

---

## 6. Monitoring & Profiling

### GPU Monitoring

```bash
# Real-time GPU utilization (in a dedicated tmux pane)
nvtop

# Or simpler:
watch -n 1 nvidia-smi
```

### Training Monitoring

```bash
# Watch training progress in real-time
tail -f experiments/AR3-02a/training.log

# Quick metrics check
python -c "
import json
with open('experiments/AR3-02a/training_report.json') as f:
    r = json.load(f)
for e in r['epoch_metrics']:
    print(f'Epoch {e[\"epoch\"]}: val_acc={e[\"val_accuracy\"]:.4f} loss={e[\"val_loss\"]:.4f}')
"
```

### Profiling Training Throughput

```bash
# Profile a short run to identify bottlenecks
python -c "
import torch
from torch.profiler import profile, ProfilerActivity

# Run 100 steps with profiling
# (integrate into train_phase4.py as needed)
"

# Check if CPU or GPU is the bottleneck:
# - GPU util < 80%: likely CPU/dataloader bottleneck
# - GPU util > 90%: compute-bound (good)
# - GPU memory < 50%: increase batch size
```

---

## 7. Running AutoResearch Experiments

### Step-by-Step: First Experiment Cycle

```bash
# 1. Ensure anchor is registered
python autoresearch/eval_harness.py \
    --checkpoint checkpoints/phase4_p8_lean_50k/seed_42/best_model.pt \
    --data-dir data/processed \
    --output autoresearch/results/anchor.json

# 2. Run a Tier 1 experiment (e.g., window size)
python autoresearch/run_experiment.py \
    --name "window_size_10" \
    --parent anchor \
    --tier 1 \
    --config-override max_window=10 \
    --budget-epochs 10 \
    --budget-minutes 30

# 3. Review results
python autoresearch/leaderboard.py

# 4. Use Claude Code to analyze
claude "Compare experiments/window_size_10 to the anchor.
Write a note to autoresearch/notes/AR3-02a.md with your analysis."
```

### Running Multiple Experiments in Parallel

The A40 has 48GB VRAM. For small models (1.7M params), you can potentially run
2 experiments simultaneously if each uses < 20GB.

```bash
# Terminal 1: Experiment A
CUDA_VISIBLE_DEVICES=0 python autoresearch/run_experiment.py \
    --name "window_10" --config-override max_window=10 &

# However, for clean results, prefer sequential runs.
# Parallel runs risk contention and make throughput metrics unreliable.
```

### Recommended Experiment Order

Run these in order for maximum information gain:

```
Day 1 (AR-0 + AR-1 + AR-2):
  1. Register anchor checkpoint
  2. Debug aux heads (speed/role at 0%)
  3. Batch size sweep: 64 → 256 → 512 → 1024
  4. torch.compile() test
  5. DataLoader worker count sweep

Day 2 (AR-3):
  6. Window size: 2 → 5 → 10 → 20 (most important experiment)
  7. Full 100K battles (with best window size from #6)
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

## 8. Agent Coordination Protocol

### Rule: One Agent Per File

Never let Claude Code and Codex edit the same file simultaneously.

**Ownership assignments:**

| Workstream | Owner | Files |
|-----------|-------|-------|
| Experiment planning & analysis | Claude Code | `autoresearch/notes/`, `autoresearch/experiment_registry.json` |
| Model architecture changes | Claude Code | `src/models/battle_transformer.py` |
| Config generation | Codex | `autoresearch/configs/` |
| Training script patches | Codex | `scripts/train_phase4.py` |
| Profiling & throughput | Codex | `autoresearch/profiling/` |
| Evaluation harness | Claude Code | `autoresearch/eval_harness.py` |
| Data pipeline fixes | Claude Code | `src/data/` |

### Communication Pattern

```
Human decides experiment direction
  → Claude Code designs experiment, writes hypothesis
    → Codex implements config/code changes
      → Human reviews diff
        → Training runs
          → Claude Code analyzes results
            → Human decides next direction
```

---

## 9. Data Management on RunPod

### Persistent Volume Layout

```
/workspace/
├── Pokemon-Battle-Model/      # Repository (on persistent volume)
│   ├── data/processed/        # Dataset (15GB+)
│   ├── checkpoints/           # Model checkpoints
│   ├── autoresearch/          # AutoResearch harness & logs
│   └── experiments/           # Per-experiment outputs
└── backups/                   # Periodic checkpoint backups
```

### Backup Strategy

RunPod pods can be terminated. Protect your work:

```bash
# Back up critical files to persistent volume
cp -r autoresearch/experiment_registry.json /workspace/backups/
cp -r autoresearch/notes/ /workspace/backups/

# Push experiment results to git regularly
cd /workspace/Pokemon-Battle-Model
git add autoresearch/
git commit -m "AutoResearch: experiment results through AR3-02a"
git push

# Or use runpodctl to transfer files
runpodctl send checkpoints/best_model.pt
```

### Downloading Results

```bash
# From your local machine:
scp -P <port> root@<pod-ip>:/workspace/Pokemon-Battle-Model/autoresearch/experiment_registry.json ./

# Or use git
git pull origin <branch>
```

---

## 10. Cost Management

### A40 Pricing (approximate, March 2026)

| Provider | Hourly Rate | Notes |
|----------|------------|-------|
| RunPod Community | $0.39–0.44/hr | Variable availability |
| RunPod Secure | $0.69/hr | Guaranteed uptime |

### Budget Planning

| Phase | Estimated Hours | Cost (Community) |
|-------|----------------|------------------|
| AR-0: Setup & Benchmark | 2 hr | $0.88 |
| AR-1: Fix Issues | 4 hr | $1.76 |
| AR-2: Throughput Optimization | 4 hr | $1.76 |
| AR-3: Data Experiments | 8 hr | $3.52 |
| AR-4: Architecture Tuning | 8 hr | $3.52 |
| AR-5: Consolidation | 4 hr | $1.76 |
| **Total** | **~30 hr** | **~$13.20** |

Plus API costs for Claude Code and Codex usage.

### Cost-Saving Tips

1. **Use spot/community instances** for Tier 1 experiments (interruptible is fine)
2. **Stop the pod** when not actively training (persistent volume retains data)
3. **Profile before scaling** — don't pay for A40 hours debugging DataLoader issues
4. **Tier your experiments** — kill bad ideas in 30 minutes, not 4 hours
5. **Cache preprocessed tensors** — don't re-process the dataset every run

---

## 11. Troubleshooting

### Common Issues

| Issue | Cause | Fix |
|-------|-------|-----|
| CUDA OOM on large batch | Batch too large for model + activations | Reduce batch size, use gradient accumulation |
| DataLoader hangs | Too many workers, shared memory exhaustion | Reduce num_workers, increase shm size |
| Training NaN | Learning rate too high, bad data | Reduce LR, check for NaN in data |
| Claude Code API timeout | Anthropic rate limits | Retry after 60s, use smaller context |
| Codex not finding files | Wrong working directory | Always `cd /workspace/Pokemon-Battle-Model` |
| Pod terminated mid-run | Spot instance preempted | Use persistent volume, checkpoint frequently |

### Increasing Shared Memory (if DataLoader hangs)

```bash
# RunPod default shm is sometimes small
# Check current size:
df -h /dev/shm

# If < 4GB, mount larger:
mount -o remount,size=16G /dev/shm
```

### Verifying A40 bf16 Support

```bash
python -c "
import torch
print(f'bf16 supported: {torch.cuda.is_bf16_supported()}')
print(f'Device: {torch.cuda.get_device_name(0)}')
print(f'Compute capability: {torch.cuda.get_device_capability()}')
"
# A40 is compute capability 8.6 — full bf16 support
```

---

## 12. Quick-Start Checklist

```
[ ] 1. Launch RunPod A40 pod with PyTorch template
[ ] 2. Install tmux, htop, nvtop, nodejs
[ ] 3. Clone repository to persistent volume
[ ] 4. Install Python dependencies, verify CUDA
[ ] 5. Verify dataset (100K battles in data/processed/)
[ ] 6. Run smoke test: python scripts/train_phase4.py --mode smoke
[ ] 7. Install Claude Code: npm install -g @anthropic-ai/claude-code
[ ] 8. Install Codex: npm install -g @openai/codex
[ ] 9. Set API keys (ANTHROPIC_API_KEY, OPENAI_API_KEY)
[ ] 10. Create tmux session with 4 panes
[ ] 11. Register anchor checkpoint in experiment registry
[ ] 12. Run first experiment: batch size sweep on A40
[ ] 13. Begin AutoResearch loop
```
