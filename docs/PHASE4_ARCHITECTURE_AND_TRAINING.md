# Phase 4: Architecture, Training, and Experiment Reference

_Consolidated from prior Phase 4 documentation. March 2026._

---

## Current Model Variants

Training uses `scripts/train_phase4.py` as the core trainer. Wrapper scripts pin specific architectures.

| Variant | Layers | Hidden | Heads | FFN | Window | Value Head | Params | Wrapper Script |
|---------|--------|--------|-------|-----|--------|------------|--------|----------------|
| P8 | 4 | 256 | 4 | 4x | 20 | On | 3.6M | `train_p8_1k.py` |
| P8-Lean | 3 | 224 | 4 | 3x | 5 | Off | ~1.95M | `train_p8_lean.py` |
| P4 | 6 | 384 | 6 | 4x | 20 | On | 11.5M | `train_p4_25k.py` |

All variants use auxiliary head with `aux_weight=0.2` and `prune_dead_features` (P8-Lean always, others optional).

### P8-Lean Design Rationale

P8-Lean reduces P8 from ~3.6M to ~1.95M parameters via:
- Depth 4→3 layers (largest savings, ~790K params)
- Width 256→224 hidden dim
- FFN multiplier 4x→3x (reduces per-layer FFN from 525K to 301K params)
- Value head removed
- Dead features pruned (always-zero `terastallized` flag, 16 zero field binary channels)
- Simplified auxiliary head (3 sub-heads instead of 5, H/4 intermediate dim)

---

## Training Pipeline

### Data Flow
1. Tensorized `.npz` files from `data/processed/battles/`
2. Battle-level 80/10/10 train/val/test split (fixed seed, no leakage)
3. `add_auxiliary_labels()` adds item targets (speed/role/tera/move-family targets are placeholder -1)
4. `WindowedTurnDataset` expands each battle to per-turn examples with sliding window
5. `BattleTransformer` trained with masked policy loss + auxiliary loss (+ optional value loss)

### Key Training Flags
```
--num-battles N          # Subsample battles
--num-layers L           # Transformer depth
--hidden-dim D           # Hidden dimension
--num-heads H            # Attention heads
--max-window W           # History window length
--aux-weight F           # Auxiliary loss weight (default 0.2)
--no-value-head          # Disable value head
--prune-dead-features    # Remove always-zero features
--batch-size B           # Micro-batch size
--grad-accum K           # Gradient accumulation steps
--amp auto|off           # Mixed precision (auto picks bf16/fp16)
--torch-compile          # Enable torch.compile
--num-workers N          # DataLoader workers
--persistent-workers     # Keep workers alive across epochs
--pin-memory             # Pinned host memory for CUDA
```

### Running Training

```bash
# P8 on 1K battles (single seed)
python scripts/train_p8_1k.py --num-battles 1000 --seeds 42 --batch-size 32 --epochs 30 --patience 7

# P8-Lean on 10K battles
python scripts/train_p8_lean.py --num-battles 10000 --seeds 42 43 44 --batch-size 64 --epochs 30

# P4 on 25K battles
python scripts/train_p4_25k.py --num-battles 25000 --seeds 42 43 44 --batch-size 32 --epochs 30

# Direct trainer with full control
python scripts/train_phase4.py --mode full --num-battles 10000 --num-layers 4 --hidden-dim 256 --num-heads 4 --max-window 20 --aux-weight 0.2 --seed 42
```

### Output Artifacts
- Per-seed: `checkpoints/<run>/seed_<N>/training_report.json` + `best_model.pt`
- Aggregate: `checkpoints/<run>/<variant>_benchmark_summary.json`

---

## Compute-vs-Quality Experiment Framework

### Decision Rule ("No Meaningful Loss")
A lower-compute variant is acceptable if all conditions hold vs the reference:
- Test top-1 drop ≤ 1.0 absolute point
- Test top-3 drop ≤ 0.8 absolute point
- Generalization gap increase ≤ 1.0 absolute point
- Seed std-dev increase ≤ 0.5 absolute point

### Experiment Protocol
- Fixed battle cohort, battle-level splits, 3 random seeds per config
- Metrics: top-1/top-3 accuracy, NLL, ECE, wall time, generalization gap
- Construct Pareto frontier on compute vs accuracy

### 10-Point Architecture Ladder (25K battles)
| ID | Config | Relative Compute |
|----|--------|-----------------|
| P1 | 8L/512d/8H, W20, aux+value | 1.00x |
| P4 | 6L/384d/6H, W20, aux+value | 0.55x |
| P8 | 4L/256d/4H, W20, aux+value | 0.24x |
| P10 | 2L/192d/4H, W10, aux only | 0.08x |

---

## VRAM and Speed Optimization

### Optimization Priority (lowest risk first)
1. **Mixed precision (AMP/bf16)** — usually largest practical VRAM/throughput win
2. **Gradient accumulation** — reduce micro-batch, keep effective batch constant
3. **Reduce `max-window`** — major attention memory reduction (quality tradeoff)
4. **Gradient checkpointing** — trades compute for lower activation memory
5. **Disable value/auxiliary heads** — smaller incremental savings

### Speed Optimization Opportunities (validated for current pipeline)
1. Remove duplicate validation forward pass (validation calls forward twice)
2. Remove ghost auxiliary target tensors (speed/role/tera/move-family are all placeholder -1)
3. Pre-convert numpy arrays to tensors at dataset init (avoid per-sample copy churn)
4. `torch.compile` for multi-epoch runs
5. Tune DataLoader workers/prefetch empirically

---

## Feature Analysis Summary

### Currently Dead Features
- `terastallized` binary flag: 0.0% activation rate in processed data
- Field binary side conditions (16 dims): 0.0% nonzero rate (pipeline extraction issue)

### Highest-Priority Feature Gaps (Not Yet Implemented)
| Priority | Feature | Impact | Effort |
|----------|---------|--------|--------|
| P0 | Move properties (power, accuracy, priority, category, STAB) | Very High | Medium |
| P0 | Opponent base stats from Pokedex (fill existing zeros) | Very High | Low |
| P0 | Type effectiveness matchup features | Very High | Medium |
| P1 | Speed comparison signals | High | Low |
| P1 | Metagame prior integration (already computed, never tensorized) | High | Medium |
| P1 | Damage estimation features | High | Medium-High |

### Hidden Information Doctrine Compliance
All proposed features use only information available to the acting player at decision time. Opponent base stats are public Pokedex data (species-intrinsic), not hidden EVs/IVs.

---

## Cloud/GPU Training

For A100-class GPUs with 25K+ battles:
- Target: 1× A100 80GB, 16+ vCPU, 64+ GB RAM
- Use `--amp auto` for bf16, effective batch ~256, full window 20
- Preprocess once with `process_dataset.py`, train many times
- Archive `training_report.json` + `best_model.pt` per run
