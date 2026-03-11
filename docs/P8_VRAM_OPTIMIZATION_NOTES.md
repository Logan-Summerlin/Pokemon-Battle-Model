# P8 VRAM Optimization Notes (for scaling to ~4,000 battles)

## Key takeaways

- **Yes, mixed precision should help** reduce VRAM (typically largest practical win) and can often improve throughput on modern GPUs.
- **Gradient checkpointing can also help** by trading extra compute for lower activation memory, which is useful if batch/window/model size are memory-bound.
- Moving from 1,000 to 4,000 battles does **not** inherently require 4x VRAM in this codebase, because GPU memory is primarily driven by model + current batch/window tensors (dataset size mostly impacts CPU RAM and wall time).

## Why dataset size has weak direct VRAM impact here

- The training loop sends one batch at a time to device (`batch = {k: v.to(device) for k, v in batch.items()}`), so GPU memory scales mainly with **batch shape**, not total battle count.
- The windowed dataset creates per-turn examples with configurable `max_window`, so sequence length and batch size are dominant memory levers.

## Existing memory/compute levers already in repo

1. **Gradient accumulation** (`--grad-accum`) is already supported in `train_phase4.py` and `train_p8_1k.py`, enabling smaller micro-batches for same effective batch size.
2. **Model scale knobs** (`--num-layers`, `--hidden-dim`, `--num-heads`) are exposed.
3. **Window size control** (`--max-window`) is exposed and strongly affects attention/activation memory.
4. **Optional value head off switch** (`--no-value-head`) and adjustable auxiliary loss weight (`--aux-weight`) are available.

## Recommended optimization order (lowest risk first)

1. **Mixed precision (AMP/bfloat16)**
   - Usually easiest and high-impact for VRAM.
   - Expected result: lower activation/gradient memory; often enables larger micro-batch at same VRAM.

2. **Increase `--grad-accum` and reduce micro-batch size**
   - Already implemented; no architecture change needed.
   - Maintains effective batch while lowering per-step VRAM.

3. **Reduce `--max-window` slightly (20 -> 15/10)**
   - Major activation/attention memory reduction.
   - Quality tradeoff must be re-checked (repo docs already treat this as an ablation axis).

4. **Gradient checkpointing in transformer blocks**
   - Good when still VRAM-bound after AMP + micro-batch tuning.
   - Saves activation memory at cost of additional compute (longer training).

5. **Disable value head / simplify auxiliary heads when necessary**
   - Smaller incremental VRAM and compute savings.

6. **Use memory-efficient optimizer/state settings**
   - Consider 8-bit optimizer states (if acceptable dependency-wise) to reduce optimizer-state memory.

## Practical recommendation for 4,000 battles on similar VRAM

- Keep the **P8 architecture** (4L/256d/4H, window 20) as baseline.
- Add **AMP first**.
- If still memory-limited, lower micro-batch and raise `--grad-accum`.
- If still memory-limited, add **checkpointing** and/or reduce window to 15.

This combination should let you train on 4,000 battles within similar VRAM, with longer wall-clock time as the main tradeoff.
