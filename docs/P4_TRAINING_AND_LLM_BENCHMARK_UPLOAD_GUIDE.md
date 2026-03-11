# P4 Model (Phase 4) Training + LLM Benchmark Upload Guide

This guide mirrors the P8 workflow, but for **P4** from the Phase 4 experiment ladder.

## 1) P4 design rationale (from the Phase 4 experiment plan)

From `docs/PHASE4_25K_COMPUTE_GENERALIZATION_EXPERIMENT.md`, **P4** is defined as:

- **6 transformer layers**
- **hidden dim 384**
- **6 attention heads**
- **max window 20**
- **auxiliary head ON** (`aux_weight=0.2`)
- **value head ON**
- intended for **25,000 battle** training runs, as the practical full-mode anchor.

The script `scripts/train_p4_25k.py` is a strict wrapper around `scripts/train_phase4.py` with this architecture pinned.

---

## 2) What the P4 wrapper does

For each requested seed, `scripts/train_p4_25k.py`:

1. launches `scripts/train_phase4.py` in `--mode full`,
2. enforces the P4 architecture and task-head setup,
3. writes per-seed outputs into `checkpoints/phase4_p4_25k/seed_<seed>/`,
4. reads each seed `training_report.json`,
5. writes an aggregate benchmark JSON to:

```text
checkpoints/phase4_p4_25k/p4_benchmark_summary.json
```

The aggregate file includes mean/std for key metrics like top-1, top-3, NLL, ECE, and wall time.

---

## 3) Quick sanity check (dry run)

Before starting long runs:

```bash
python scripts/train_p4_25k.py --dry-run --seeds 42
```

This prints the exact command(s) that would run without training.

---

## 4) Recommended full P4 run (3 seeds)

```bash
python scripts/train_p4_25k.py \
  --num-battles 25000 \
  --seeds 42 43 44 \
  --batch-size 32 \
  --epochs 30 \
  --patience 7 \
  --output-root checkpoints/phase4_p4_25k
```

This follows the Phase 4 experiment document defaults for dataset size and seed protocol.

---

## 5) Optional tuning flags

The wrapper forwards core trainer knobs. Typical examples:

```bash
python scripts/train_p4_25k.py \
  --num-battles 25000 \
  --seeds 42 43 44 \
  --batch-size 16 \
  --grad-accum 2 \
  --num-workers 8 \
  --pin-memory \
  --persistent-workers
```

Useful if VRAM is tight or input pipeline throughput needs improvement.

---

## 6) Artifacts to collect for LLM benchmarking

Primary file:

- `checkpoints/phase4_p4_25k/p4_benchmark_summary.json`

Per-seed details:

- `checkpoints/phase4_p4_25k/seed_42/training_report.json`
- `checkpoints/phase4_p4_25k/seed_43/training_report.json`
- `checkpoints/phase4_p4_25k/seed_44/training_report.json`

(plus each seed's `best_model.pt`).

---

## 7) Suggested prompt for external LLM analysis

Use this prompt with your uploaded P4 summary/report files:

```text
You are helping evaluate a Phase 4 behavior-cloning transformer for Pokemon battles.

Please:
1) Read p4_benchmark_summary.json and per-seed training_report.json files.
2) Report mean ± std for test top-1 accuracy, top-3 accuracy, NLL, ECE, and wall time.
3) Compare P4 against any provided baseline(s).
4) Apply the Phase 4 "no meaningful loss" criteria where applicable.
5) Provide a concise recommendation: keep P4, tune P4, or move to another permutation.
```

---

## 8) One-command recap

```bash
python scripts/train_p4_25k.py --num-battles 25000 --seeds 42 43 44 --batch-size 32 --epochs 30 --patience 7 --output-root checkpoints/phase4_p4_25k
```

When complete, share `p4_benchmark_summary.json` first; it is the fastest high-signal artifact for cross-run comparison.
