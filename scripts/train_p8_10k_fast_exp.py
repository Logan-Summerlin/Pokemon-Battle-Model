#!/usr/bin/env python3
"""Train an experimental fast P8 variant on 10,000 battles.

Applies the first 3 recommendations from docs/P8_10K_TRAINING_TIME_REDUCTION_PLAN.md:
1) Dead-feature pruning in embeddings.
2) Value head disabled.
3) Throughput-tuned CUDA defaults (AMP + pinned memory + non-blocking + persistent workers).

Also sets max window to 5 for this experiment.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_SCRIPT = PROJECT_ROOT / "scripts" / "train_phase4.py"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train P8 fast experiment on 10,000 battles")
    p.add_argument("--data-dir", type=str, default="data/processed")
    p.add_argument("--num-battles", type=int, default=10000)
    p.add_argument("--seeds", type=int, nargs="+", default=[42])
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=7)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-steps", type=int, default=300)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--prefetch-factor", type=int, default=6)
    p.add_argument("--amp", choices=["off", "fp16", "bf16", "auto"], default="auto")
    p.add_argument("--torch-compile", action="store_true")
    p.add_argument("--output-root", type=str, default="checkpoints/phase4_p8_10k_fast_w5")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def run_command(cmd: list[str], dry_run: bool = False) -> None:
    print("\n>>> " + " ".join(cmd))
    if not dry_run:
        subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def mean_std(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    if len(values) == 1:
        return {"mean": values[0], "std": 0.0}
    return {"mean": statistics.fmean(values), "std": statistics.stdev(values)}


def summarize_seed(report: dict[str, Any], seed: int) -> dict[str, Any]:
    final = report.get("final_results", {})
    test = report.get("test_results", {})
    cfg = report.get("training_config", {})
    return {
        "seed": seed,
        "best_epoch": final.get("best_epoch"),
        "best_val_loss": final.get("best_val_loss"),
        "test_top1_accuracy": test.get("test_accuracy"),
        "test_top3_accuracy": test.get("test_top3_accuracy"),
        "test_loss": test.get("test_loss"),
        "test_nll": test.get("test_nll"),
        "test_ece": test.get("expected_calibration_error"),
        "wall_time_min": final.get("total_wall_time_min"),
        "train_time_min": final.get("total_training_time_min"),
        "parameter_count": cfg.get("parameter_count"),
    }


def aggregate(results: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    get = lambda k: [r[k] for r in results if r.get(k) is not None]
    return {
        "experiment": "phase4_p8_10k_fast_w5",
        "created_at": datetime.now(UTC).isoformat(),
        "model_config": {
            "num_layers": 4,
            "hidden_dim": 256,
            "num_heads": 4,
            "max_window": 5,
            "aux_weight": 0.2,
            "use_value_head": False,
            "prune_dead_features": True,
        },
        "throughput_config": {
            "amp": args.amp,
            "num_workers": args.num_workers,
            "prefetch_factor": args.prefetch_factor,
            "persistent_workers": args.num_workers > 0,
            "pin_memory": True,
            "non_blocking_transfer": True,
            "torch_compile": args.torch_compile,
        },
        "train_hparams": {
            "num_battles": args.num_battles,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "patience": args.patience,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "warmup_steps": args.warmup_steps,
            "grad_accum": args.grad_accum,
            "dropout": args.dropout,
            "seeds": args.seeds,
        },
        "seed_results": results,
        "aggregate": {
            "test_top1_accuracy": mean_std(get("test_top1_accuracy")),
            "test_top3_accuracy": mean_std(get("test_top3_accuracy")),
            "test_nll": mean_std(get("test_nll")),
            "test_ece": mean_std(get("test_ece")),
            "wall_time_min": mean_std(get("wall_time_min")),
        },
    }


def main() -> int:
    args = parse_args()
    if not TRAIN_SCRIPT.exists():
        raise FileNotFoundError(f"Missing training script: {TRAIN_SCRIPT}")

    output_root = PROJECT_ROOT / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, Any]] = []

    for seed in args.seeds:
        seed_dir = output_root / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        report_path = seed_dir / "training_report.json"

        cmd = [
            sys.executable,
            str(TRAIN_SCRIPT),
            "--mode", "full",
            "--data-dir", args.data_dir,
            "--num-battles", str(args.num_battles),
            "--num-layers", "4",
            "--hidden-dim", "256",
            "--num-heads", "4",
            "--max-window", "5",
            "--aux-weight", "0.2",
            "--no-value-head",
            "--prune-dead-features",
            "--dropout", str(args.dropout),
            "--batch-size", str(args.batch_size),
            "--epochs", str(args.epochs),
            "--lr", str(args.lr),
            "--weight-decay", str(args.weight_decay),
            "--warmup-steps", str(args.warmup_steps),
            "--patience", str(args.patience),
            "--grad-accum", str(args.grad_accum),
            "--seed", str(seed),
            "--num-workers", str(args.num_workers),
            "--prefetch-factor", str(args.prefetch_factor),
            "--persistent-workers",
            "--pin-memory",
            "--non-blocking-transfer",
            "--amp", args.amp,
            "--checkpoint-dir", str(seed_dir),
            "--report-path", str(report_path),
        ]
        if args.torch_compile:
            cmd.append("--torch-compile")

        run_command(cmd, dry_run=args.dry_run)

        if not args.dry_run:
            summaries.append(summarize_seed(load_json(report_path), seed))

    if args.dry_run:
        print("\nDry-run complete. No training executed.")
        return 0

    out = aggregate(summaries, args)
    out_path = output_root / "p8_fast_w5_benchmark_summary.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved aggregate benchmark summary: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
