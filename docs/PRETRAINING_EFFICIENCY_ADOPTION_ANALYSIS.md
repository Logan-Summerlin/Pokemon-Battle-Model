# Pre-training Efficiency Adoption Analysis for BattleTransformer

## Executive Summary

This document analyzes the techniques from the [Pre-training Efficiency Report](Pre-training%20Efficiency%20Report.md) (Chinese AI labs, 2023-2026) and identifies which are adoptable for the BattleTransformer project. The project operates at a fundamentally different scale (~2M params, ~100K battles, single GPU) than the frontier models in the report (1B-1T params, 1K+ GPUs), so most techniques require careful translation rather than direct adoption.

**Bottom line**: 6 techniques are directly adoptable with concrete expected gains. 4 more are worth investigating. The rest are inapplicable at our scale. Combined, the adoptable techniques could reduce wall-clock training time by 30-50% and/or improve model quality by 2-5% accuracy.

---

## Technique Assessment Matrix

| # | Technique | Applicable? | Expected Impact | Implementation Effort | Priority |
|---|-----------|-------------|-----------------|----------------------|----------|
| 1 | Padding-free batching / token packing | **Yes** | High (15-25% throughput) | Medium | **P0** |
| 2 | Warmup-Stable-Decay (WSD) LR schedule | **Yes** | Medium (1-3% accuracy) | Low | **P0** |
| 3 | torch.compile optimization | **Yes** | Medium (10-20% throughput) | Low | **P0** |
| 4 | Gradient checkpointing for larger windows | **Yes** | Medium (enables W10-20) | Low | **P1** |
| 5 | AdamW beta tuning for auxiliary heads | **Yes** | Low-Medium (stability) | Low | **P1** |
| 6 | Max-z logit stabilization loss | **Yes** | Low-Medium (calibration) | Low | **P1** |
| 7 | Staged context window extension | Partially | Low-Medium (1-3%) | Medium | **P2** |
| 8 | Data quality scoring / Elo weighting | **Yes** | Medium (2-4% accuracy) | Medium | **P2** |
| 9 | Mixed precision (BF16) | Already done | — | — | Done |
| 10 | Dead feature pruning | Already done | — | — | Done |
| 11 | MoE / sparse computation | No | — | — | N/A |
| 12 | ZeRO sharding / pipeline parallelism | No | — | — | N/A |
| 13 | FP8 training | No | — | — | N/A |
| 14 | Sequence parallelism | No | — | — | N/A |
| 15 | Hardware-aware NPU optimization | No | — | — | N/A |

---

## Detailed Analysis of Adoptable Techniques

### 1. Padding-Free Batching / Token Packing (P0)

**Source**: ERNIE 4.5 describes "packing data within a batch into a sequence to avoid padding" to reduce memory use and accelerate training.

**Current problem**: The `collate_windowed()` function in `train_phase4.py:209` right-pads all sequences in a batch to the maximum window length in that batch. With P8-Lean's `max_window=5`, early turns (window 1-2) are padded to match turns with full 5-step history. The transformer then processes padding tokens through all layers, wasting compute.

From the efficiency stats (`archive/artifacts/p8_efficiency_stats.json`):
- Window 5: 70 tokens, 4,900 attention elements
- Window 2: 28 tokens, 784 attention elements (6.25x cheaper)
- Window 1: 14 tokens, 196 attention elements (25x cheaper)

Early turns in each battle always have short windows (turn 1 = window 1, turn 2 = window 2, etc.), so a significant fraction of training examples are short-windowed. Padding these to W=5 wastes ~40-60% of attention compute on those examples.

**Implementation**:

```python
# Option A: Bucket batching by sequence length (simplest)
# Group examples into length buckets before batching
class BucketedSampler:
    """Groups examples by window length for minimal padding."""
    def __init__(self, dataset, batch_size, num_buckets=5):
        # Bucket examples by actual sequence length
        self.buckets = defaultdict(list)
        for idx in range(len(dataset)):
            b_idx, t_idx = dataset.examples[idx]
            actual_len = min(t_idx + 1, dataset.max_window)
            self.buckets[actual_len].append(idx)

# Option B: Nested tensor packing (PyTorch 2.x)
# Pack variable-length sequences into a single NestedTensor
# Requires enable_nested_tensor=True in TransformerEncoder
# (currently disabled at battle_transformer.py:483)

# Option C: Manual attention mask with packed sequences
# Concatenate all sequences in a batch into one long sequence
# with block-diagonal attention mask. Most efficient but
# requires custom attention mask construction.
```

**Recommendation**: Start with Option A (bucket batching). It requires only a custom sampler, no model changes. Option B is promising but requires testing compatibility with the current architecture. Option C is the approach ERNIE uses but is too complex for our scale.

**Expected gain**: 15-25% throughput improvement from reduced padding waste.

---

### 2. Warmup-Stable-Decay (WSD) LR Schedule (P0)

**Source**: MiniCPM (OpenBMB) introduces WSD scheduler: separate warmup, stable (constant high LR), and decay phases. Key finding: ~10% of total tokens in the decay phase is sufficient for best results. Enables checkpoint reuse and cheaper data-scaling experiments.

**Current state**: `WarmupCosineScheduler` in `train_phase4.py:242` does linear warmup → cosine decay to `min_lr`. The LR immediately starts decaying after warmup, meaning the model spends most of training at a lower-than-peak LR.

**Why WSD is better for this project**:
1. The cosine schedule "wastes" the middle portion of training at a medium LR where the model is neither exploring (high LR) nor fine-tuning (low LR).
2. WSD's stable phase keeps the LR high for most of training, enabling faster convergence.
3. The short decay phase at the end consolidates learning without the gradual decay.
4. Critical for staged training: if we later adopt multi-stage pretraining, WSD allows stopping the stable phase at any checkpoint and running a new decay phase — enabling "free" data-scaling experiments.
5. The multi-stage pretraining report notes that "curriculum advantages largely disappear under standard cosine LR decay" — WSD directly addresses this.

**Implementation**:

```python
class WarmupStableDecayScheduler:
    """MiniCPM-style WSD scheduler.

    Three phases:
    1. Warmup: linear ramp from 0 to peak_lr over warmup_steps
    2. Stable: constant peak_lr for stable_fraction of total steps
    3. Decay: cosine decay from peak_lr to min_lr over remaining steps
    """
    def __init__(self, optimizer, warmup_steps, total_steps,
                 stable_fraction=0.7, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.stable_fraction = stable_fraction
        self.min_lr = min_lr
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        self._step = 0

        # Calculate phase boundaries
        self.stable_end = warmup_steps + int(
            (total_steps - warmup_steps) * stable_fraction
        )

    def step(self):
        self._step += 1
        lr = self._get_lr()
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

    def _get_lr(self):
        if self._step < self.warmup_steps:
            return self.base_lrs[0] * self._step / max(self.warmup_steps, 1)
        elif self._step < self.stable_end:
            return self.base_lrs[0]  # Constant high LR
        else:
            # Cosine decay over remaining steps
            decay_steps = self.total_steps - self.stable_end
            progress = (self._step - self.stable_end) / max(decay_steps, 1)
            progress = min(progress, 1.0)
            return self.min_lr + 0.5 * (self.base_lrs[0] - self.min_lr) * (
                1 + math.cos(math.pi * progress)
            )
```

**Integration**: Add `--lr-schedule {cosine,wsd}` flag to `train_phase4.py` and `--stable-fraction` parameter.

**Expected gain**: 1-3% test accuracy improvement; cleaner loss curves; better foundation for future staged training.

---

### 3. torch.compile Optimization (P0)

**Source**: BladeDISC and similar compiler optimizations emphasize kernel fusion and operator optimization as "boring wins" that consistently improve throughput.

**Current state**: `train_phase4.py` already has `--torch-compile` flag (line 801) that calls `torch.compile(model)`. However, it's not enabled by default and the default compile mode may not be optimal.

**Improvements**:
1. Use `torch.compile(model, mode="reduce-overhead")` for small models — this mode is optimized for models where kernel launch overhead dominates (which is the case for ~2M param models).
2. Alternatively, use `mode="max-autotune"` for first run, which takes longer to compile but produces faster kernels.
3. Enable `torch.set_float32_matmul_precision('high')` which allows TF32 on Ampere+ GPUs for faster matmuls.

**Implementation**:

```python
# In train_phase4.py main(), before training:
if device == "cuda":
    torch.set_float32_matmul_precision('high')

if args.torch_compile:
    compile_mode = "reduce-overhead"  # Best for small models
    model = torch.compile(model, mode=compile_mode)
    logger.info(f"Enabled torch.compile (mode={compile_mode})")
```

**Expected gain**: 10-20% throughput improvement on CUDA with minimal code change.

---

### 4. Gradient Checkpointing for Larger Windows (P1)

**Source**: InternEvo and Baichuan 2 both emphasize activation checkpointing as a standard "boring win" for memory efficiency, enabling larger batch sizes or longer sequences.

**Current state**: No gradient checkpointing. P8-Lean uses `max_window=5` partly for memory reasons. The efficiency stats show that going from W5 → W20 increases attention cost 16x (4,900 → 78,400 elements). Gradient checkpointing could make W10 or W20 feasible without proportional memory increase.

**Implementation**:

```python
# In BattleTransformerEncoder.__init__():
if config.gradient_checkpointing:
    from torch.utils.checkpoint import checkpoint

# In BattleTransformerEncoder.forward(), wrap transformer call:
if self.config.gradient_checkpointing and self.training:
    output = torch.utils.checkpoint.checkpoint(
        self.transformer, tokens, src_key_padding_mask=attn_mask,
        use_reentrant=False
    )
else:
    output = self.transformer(tokens, src_key_padding_mask=attn_mask)
```

Add `gradient_checkpointing: bool = False` to `TransformerConfig`.

**Expected gain**: Enables W10+ on memory-constrained GPUs. With larger windows, expect 1-3% accuracy from additional context, based on the multi-stage report's analysis.

---

### 5. AdamW Beta Tuning for Auxiliary Heads (P1)

**Source**: PanGu-Σ discloses a "Hybrid Hyper-parameter ADAM Optimizer" with different epsilon values for different layer types — ε₁=1e-8 for standard layers and ε₂=1e-20 for sparse layers — because gradients in sparse layers are smaller. Also uses β₁=0.8, β₂=0.95 (vs. the standard 0.9, 0.999).

**Current state**: Single AdamW with β₁=0.9, β₂=0.999 for all parameters.

**Why this matters for BattleTransformer**: The auxiliary head (item/speed/role/move-family prediction) has different gradient characteristics than the policy head. The auxiliary head operates on per-slot opponent tokens and trains on sparse targets (many -1 / unknown values), similar to PanGu's sparse layers. The policy head is dense and well-conditioned.

**Implementation**:

```python
# In train_phase4.py, replace single optimizer with param-group optimizer:
encoder_params = list(model.encoder.parameters())
policy_params = list(model.policy_head.parameters())
aux_params = list(model.auxiliary_head.parameters())

optimizer = torch.optim.AdamW([
    {"params": encoder_params, "lr": args.lr, "betas": (0.9, 0.95)},
    {"params": policy_params, "lr": args.lr, "betas": (0.9, 0.95)},
    {"params": aux_params, "lr": args.lr * 0.5, "betas": (0.9, 0.999), "eps": 1e-12},
], weight_decay=args.weight_decay)
```

**Rationale**: Lower β₂ (0.95 vs 0.999) for encoder+policy makes the optimizer more responsive to recent gradient magnitudes, which helps when the loss landscape changes rapidly (as during curriculum or when learning from diverse Elo games). Higher β₂ and lower LR for auxiliary head stabilizes training on the sparse auxiliary targets.

**Expected gain**: More stable training, slightly better convergence. Low risk since it only changes optimizer hyperparameters.

---

### 6. Max-z Logit Stabilization Loss (P1)

**Source**: Baichuan 2 adds a max-z loss term `2e-4 * z²` (z = maximum logit) to "stabilize training and reduce sensitivity to repetition penalty at inference." Inspired by PaLM.

**Current state**: No logit regularization. The policy head outputs raw logits that are masked by legal actions then softmaxed.

**Why this matters**: In Pokemon battle prediction, certain actions (e.g., a move that OHKOs the opponent) can produce extremely high logits. This causes:
1. Very peaked softmax distributions (poor calibration)
2. Gradient saturation for non-selected actions
3. Sensitivity to numerical precision in legal masking

The max-z loss penalizes extremely confident predictions, encouraging better-calibrated probabilities.

**Implementation**:

```python
# In compute_total_loss() or as a separate function:
def compute_maxz_loss(logits: torch.Tensor, legal_mask: torch.Tensor,
                      coefficient: float = 2e-4) -> torch.Tensor:
    """Max-z logit stabilization (Baichuan 2 / PaLM style)."""
    # Only penalize logits for legal actions
    masked = logits.masked_fill(legal_mask == 0, float("-inf"))
    z = masked.max(dim=-1).values  # (batch,)
    return coefficient * (z ** 2).mean()

# Add to compute_total_loss:
if config.maxz_loss_weight > 0:
    maxz_loss = compute_maxz_loss(output.policy_logits, legal_mask)
    total = total + config.maxz_loss_weight * maxz_loss
    loss_dict["maxz"] = maxz_loss.item()
```

**Expected gain**: Better calibration (lower ECE), more robust inference behavior. The calibration data in test results already shows this could matter — the current model may be overconfident in some confidence bins.

---

### 7. Staged Context Window Extension (P2)

**Source**: Qwen2.5 uses two-phase pretraining (4K → 32K context); SkyLadder achieves up to 3.7% gains with context scheduling; Cerebras VSL shows 29% fewer FLOPs for short→long vs. constant length.

**Current state**: The multi-stage pretraining report already analyzed this extensively. P8-Lean uses fixed W5. The sinusoidal positional encoding supports arbitrary window sizes up to the 100-position buffer.

**What to adopt**: Rather than the full 3-stage pipeline from the multi-stage report, adopt a simpler within-training-run window curriculum:

```python
# In train_epoch(), dynamically set window size based on epoch progress:
def get_window_for_epoch(epoch, max_epochs, max_window):
    """Linearly ramp window size from 1 to max_window."""
    progress = epoch / max_epochs
    if progress < 0.3:
        return min(2, max_window)
    elif progress < 0.6:
        return min(3, max_window)
    else:
        return max_window
```

This captures the "short then long" benefit without requiring separate data stages or multiple training runs. The model learns local action patterns first, then incorporates history.

**Caveat**: This requires rebuilding the dataset each time the window changes, or pre-building datasets for each window size. A simpler alternative: always use max_window but apply a "window warmup mask" that zeros out early turns during the first portion of training.

**Expected gain**: 1-3% accuracy improvement; slightly faster early training (shorter attention sequences).

---

### 8. Data Quality Scoring / Elo Weighting (P2)

**Source**: Baichuan 2 uses "large-scale deduplication/clustering and scoring for sampling." ERNIE 5.0 uses data quality signals. MiniCPM uses data mixing strategies.

**Current state**: All training examples are weighted equally regardless of the Elo of the source battle.

**Why Elo weighting helps**: Higher-Elo games have less noisy action labels (better players make more consistent, strategically sound decisions). Lower-Elo games contribute noise that the model must "average out." By upweighting high-Elo examples, the model learns cleaner decision patterns.

**Implementation**:

```python
# In WindowedTurnDataset, add per-example weights based on battle Elo:
class WindowedTurnDataset(Dataset):
    def __init__(self, battles, max_window=20, elo_weighting=False):
        ...
        for b_idx, battle in enumerate(battles):
            elo = battle.get("elo", 1300)
            # Weight: 1.0 for Elo 1300, up to 2.0 for Elo 1800+
            weight = 1.0 + max(0, (elo - 1300)) / 500
            ...
            for t in range(seq_len):
                if action >= 0:
                    self.examples.append((b_idx, t))
                    self.weights.append(weight)

# Use WeightedRandomSampler in DataLoader:
sampler = WeightedRandomSampler(dataset.weights, len(dataset))
train_loader = DataLoader(dataset, batch_size=bs, sampler=sampler, ...)
```

**Expected gain**: 2-4% accuracy improvement from learning cleaner decision patterns.

---

## Already Implemented Techniques

### 9. Mixed Precision (BF16) — Already Done

The `--amp auto` flag (line 791) automatically selects BF16 on supported GPUs, with FP16+GradScaler fallback. This matches the "BF16 everywhere" recommendation from InternEvo, Baichuan 2, and other Chinese labs.

### 10. Dead Feature Pruning — Already Done

The `--prune-dead-features` flag and `FieldEmbedding.prune_dead_features` (line 322) already skip zero-valued field binary features. This is a small but "free" optimization.

---

## Inapplicable Techniques (with reasoning)

### 11. Mixture-of-Experts (MoE) — N/A
MoE is for models with billions of parameters where you want to expand capacity without proportional compute increase. At ~2M params, the model is too small for meaningful expert specialization. The routing overhead alone would likely negate any benefit.

### 12. ZeRO Sharding / Pipeline Parallelism — N/A
These are multi-GPU techniques. BattleTransformer trains on a single GPU. The full model, optimizer state, and gradients fit comfortably in memory at ~2M params.

### 13. FP8 Training — N/A
FP8 is a frontier technique for very large models where the throughput gain from reduced precision outweighs the complexity. At ~2M params, BF16 is already excellent and the model is too small for FP8 quantization noise to be manageable. PyTorch FP8 support is also immature for small models.

### 14. Sequence Parallelism / FlashAttention — N/A
With max 70 tokens (W5 × 14 tokens/step) or even 280 tokens (W20 × 14), the sequences are far too short for FlashAttention to provide meaningful benefit. Standard PyTorch `nn.TransformerEncoder` is sufficient. The overhead of FlashAttention's tiling would exceed its savings at these sequence lengths.

### 15. Hardware-Aware NPU Optimization — N/A
The project uses standard NVIDIA GPUs (or CPU), not Ascend NPUs. The Hermes and TeleChat3 optimizations are specific to Huawei's Ascend hardware stack.

---

## Implementation Roadmap

### Phase 1: Quick Wins (P0, ~1-2 days)

1. **WSD Scheduler**: Add `WarmupStableDecayScheduler` class alongside existing `WarmupCosineScheduler`. Wire up `--lr-schedule` flag.
2. **torch.compile tuning**: Set `mode="reduce-overhead"` and add `torch.set_float32_matmul_precision('high')`.
3. **Bucket batching**: Implement `BucketedBatchSampler` that groups examples by window length.

### Phase 2: Architecture Enhancements (P1, ~2-3 days)

4. **Gradient checkpointing**: Add config flag and checkpoint wrapper in encoder.
5. **Param-group optimizer**: Separate encoder/policy/auxiliary parameter groups with different betas.
6. **Max-z loss**: Add logit stabilization term to `compute_total_loss()`.

### Phase 3: Data & Curriculum (P2, ~3-5 days)

7. **Window curriculum**: Implement within-run window warmup.
8. **Elo weighting**: Add per-example weights and `WeightedRandomSampler`.

### Measurement Plan

For each technique, measure on a standardized evaluation:
- **Throughput**: examples/sec, wall-clock per epoch
- **Quality**: test top-1 accuracy, top-3 accuracy, NLL, ECE
- **Memory**: peak GPU memory, RAM usage

Use the existing P8-Lean config (3L/224d/4H, W5, 10K battles, seed 42) as the control.

---

## Summary

The Pre-training Efficiency Report describes techniques spanning the full stack from hardware to data. For a ~2M parameter model on a single GPU, the most impactful techniques are:

1. **Padding-free batching** (reduce wasted compute)
2. **WSD scheduler** (better LR utilization)
3. **torch.compile** (compiler-level kernel optimization)
4. **Gradient checkpointing** (enable longer windows)
5. **Optimizer tuning** (stabilize multi-head training)
6. **Max-z loss** (improve calibration)
7. **Data quality weighting** (learn from better demonstrations)

These are the "boring wins" that the report emphasizes Chinese production stacks consistently prioritize — and they translate directly to our much smaller scale.
