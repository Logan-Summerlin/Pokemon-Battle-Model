# Phase 4 Training Optimization Report

## Overview

This document records all optimizations applied during Phase 4 BattleTransformer training, along with benchmarks showing estimated speedups. These were necessary due to CPU-only training constraints. **When GPU compute becomes available, many of these constraints can be relaxed.**

## Hardware Environment

| Resource | Specification |
|----------|--------------|
| CPU | Intel Xeon @ 2.10GHz, 4 cores |
| RAM | 15.7 GB |
| GPU | None (CPU-only training) |
| Storage | 252 GB total |
| PyTorch | 2.10.0+cu128 |

## Data Pipeline Optimization: Windowed Per-Turn Dataset

### Problem
The original `TransformerTrainer` treats each battle as one training example and only computes loss at the **last step** per battle. With ~10K battles, this yields only ~10K training examples per epoch, wasting the majority of labeled turns.

### Solution
Created `WindowedTurnDataset` that unrolls each battle into **per-turn examples** with causal history windows. A battle with L valid turns produces L training examples, each using a sliding window of the previous W turns as context.

### Impact
| Metric | Original (last-step only) | Windowed (per-turn) |
|--------|--------------------------|-------------------|
| Train examples (500 battles) | ~400 | ~7,019 |
| Train examples (10K battles) | ~8,000 | ~139,942 |
| Training signal utilization | ~5% of turns | ~100% of valid turns |

### Recommendation for More Compute
Keep the windowed approach regardless of compute - it's strictly better for learning. With GPU, you can increase `max_window` from 10-20 to the full 20+ turns and use larger batch sizes.

## Model Size: Scaling Ladder

### Phase 4 Design Spec (from IMPLEMENTATION_PLAN.md)
| Configuration | Layers | Hidden Dim | Heads | Params | Target Data |
|--------------|--------|-----------|-------|--------|-------------|
| Smoke Test | 2 | 128 | 4 | 606K | 5K turns |
| Small | 4 | 256 | 4 | 3.6M | 25K turns |
| Medium | 6 | 384 | 6 | 11.5M | 100K turns |
| Full | 8 | 512 | 8 | ~25M | 500K+ turns |

### Benchmarked Training Times (CPU, 500 battles, batch=32, window=20)

| Config | Params | Time/Step | Time/Epoch | 30 Epochs |
|--------|--------|----------|-----------|-----------|
| 2L/128d/4H | 606K | 0.20s | 0.7 min | 22 min |
| 4L/256d/4H | 3.6M | 1.35s | 5.0 min | 2.5 hrs |
| 6L/384d/6H | 11.5M | 3.94s | 14.4 min | 7.2 hrs |

### Benchmarked Training Times (CPU, 10K battles, batch=64, window=10)

| Config | Params | Time/Step | Time/Epoch | 20 Epochs |
|--------|--------|----------|-----------|-----------|
| 2L/192d/4H | 1.2M | 1.14s | 20.8 min | 6.9 hrs |
| 3L/192d/4H | 1.7M | 1.47s | 26.9 min | 9.0 hrs |
| 4L/256d/4H | 3.6M | 1.29s | 47.0 min | 15.7 hrs |

### Recommendation for More Compute
- With a single GPU (e.g., A100): Use 6L/384d/6H or 8L/512d/8H with the full 10K battles
- With multiple GPUs: Scale to full 8L/512d/8H on 100K+ battles as specified in the implementation plan
- Estimated GPU speedup: 10-50x depending on GPU, enabling the full-spec model in under 1 hour per epoch on 10K battles

## Batch Size and Gradient Accumulation

### Applied Optimization
- Using batch_size=32 with window=20 for 500-battle runs
- For full 10K battles, would need batch_size=64-128 with window=10 to fit in reasonable time

### Recommendation for More Compute
- Target effective batch size of 256 (per the spec: `configs/training/bc.yaml`)
- Use gradient_accumulation_steps to achieve this: e.g., batch=64, accum=4 for effective 256
- GPU memory permits larger batch sizes without accumulation

## History Window Length

### Applied Optimization
- Using max_window=20 for 500-battle runs (full history since max seq_len is 20)
- For 10K-battle runs on CPU, would reduce to window=10 to halve transformer sequence length (140 tokens -> 70 tokens per example)

### Impact of Window Reduction
The transformer processes `window * 14` tokens (14 tokens per turn: 6 own + 6 opponent + field + context). Reducing the window from 20 to 10 halves the sequence length and approximately halves forward/backward time.

| Window | Tokens/Example | Approx Speedup |
|--------|---------------|----------------|
| 20 | 280 | 1.0x (baseline) |
| 15 | 210 | 1.3x |
| 10 | 140 | 2.0x |
| 5 | 70 | 3.5x |

### Recommendation for More Compute
Use max_window=20 (the full history). The transformer's attention over full battle history is a core architectural advantage over MLP/GRU baselines.

## Mixed Precision Training

### Current Status
- Mixed precision (AMP) is **disabled** on CPU (no benefit on CPU, only useful with GPU)
- The code supports `torch.amp.autocast("cuda")` automatically when GPU is detected

### Recommendation for More Compute
- Enable AMP with bf16 (preferred) or fp16 on GPU
- Expected speedup: 1.5-2x on modern GPUs (A100, H100)
- The training script already has the infrastructure; just run on a CUDA device

## Data Workers

### Applied Optimization
- Using `num_workers=0` (inline loading) since CPU compute is the bottleneck, not data loading
- Tested `num_workers=2`: slightly slower due to process overhead on this machine

### Recommendation for More Compute
- With GPU training, set `num_workers=4-8` to keep the GPU fed
- Data loading becomes a bottleneck when GPU forward/backward is fast

## torch.compile

### Current Status
- Not applied (requires GPU for meaningful benefit on this PyTorch version)

### Recommendation for More Compute
- Enable `torch.compile(model)` for potential 10-30% speedup on GPU
- Use `mode="reduce-overhead"` for training

## Summary: Scaling Roadmap

| Compute Level | Battles | Model | Window | Batch | Expected Time |
|--------------|---------|-------|--------|-------|--------------|
| CPU (current) | 100 | 4L/256d | 20 | 32 | ~30 min |
| CPU (current) | 500 | 4L/256d | 20 | 32 | ~2.5 hrs |
| CPU (current) | 500 | 6L/384d | 20 | 32 | ~7.2 hrs |
| Single GPU (A100) | 10K | 6L/384d | 20 | 256 | ~2-4 hrs |
| Single GPU (A100) | 10K | 8L/512d | 20 | 256 | ~4-8 hrs |
| Multi-GPU | 100K+ | 8L/512d | 20 | 256 | ~8-16 hrs |

## Key Decisions That Should NOT Be Reverted

1. **Windowed per-turn dataset**: Always better than last-step-only. Keep regardless of compute.
2. **Hidden Information Doctrine**: Opponent info uses only revealed data + "unknown" markers. Never change this.
3. **Auxiliary head training**: Even if compute-constrained, the auxiliary hidden-info head adds minimal overhead and improves representation quality.
4. **80/10/10 battle-level split**: No data leakage between splits. Do not change.
