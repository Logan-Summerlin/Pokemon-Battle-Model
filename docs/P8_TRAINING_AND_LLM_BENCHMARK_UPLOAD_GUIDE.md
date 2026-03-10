# P8 Model (Phase 4) Training + LLM Benchmark Upload Guide

This guide is tailored to the repository's Phase 4 experiment plan and provides:
1. a concrete P8 design rationale,
2. a reproducible command sequence to train on 1,000 replays,
3. a standard way to package and upload results so an LLM can benchmark runs.

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

From repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Verify that processed data exists:

```bash
test -d data/processed/battles && echo "processed data found"
```

If you need to build processed tensors from raw replays, run the repo data pipeline first (download/parse/process scripts in `scripts/`).

---

## 3) Quick dry run (sanity check)

Use dry-run to print exact training commands without starting training:

```bash
python scripts/train_p8_1k.py --dry-run --seeds 42 43 44
```

This ensures output directories and command formatting are correct before spending compute.

---

## 4) Train P8 on 1,000 replays

### Recommended single-seed run

```bash
python scripts/train_p8_1k.py \
  --num-battles 1000 \
  --seeds 42 \
  --batch-size 32 \
  --epochs 30 \
  --patience 7 \
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
  --output-root checkpoints/phase4_p8_1k
```

What this script does:
1. launches `scripts/train_phase4.py` once per seed with strict P8 architecture,
2. stores per-seed artifacts in `checkpoints/phase4_p8_1k/seed_<seed>/`,
3. reads each `training_report.json`,
4. writes aggregate metrics to `checkpoints/phase4_p8_1k/p8_benchmark_summary.json`.

---

## 5) Output files you should keep

After training, preserve these files:

- `checkpoints/phase4_p8_1k/seed_42/training_report.json` (and other seeds)
- `checkpoints/phase4_p8_1k/seed_42/best_model.pt` (and other seeds)
- `checkpoints/phase4_p8_1k/p8_benchmark_summary.json`

The **benchmark summary JSON** is the preferred single file for LLM analysis because it already includes per-seed and mean/std metrics.

---

## 6) Upload workflow for LLM benchmark comparison

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

## 7) Interpreting results (what to look for)

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

## 8) Optional: compare two experiment folders locally

If you produce another run folder (e.g., `checkpoints/phase4_p6_1k`), compare both summaries in your LLM prompt. Keep both summary JSON files in the same upload so the LLM can directly contrast means/stds.

---

## 9) Reproducibility checklist

- Pin the same code commit hash.
- Keep `num-battles`, architecture, and split seed policy fixed.
- Report hardware (CPU/GPU, RAM).
- Run at least 3 seeds for final benchmark statements.
- Archive full checkpoint folder and summary JSON.
