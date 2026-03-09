# Pokemon Battle Model: Evaluation Specification

_Frozen: March 2026 — Updated only when new evaluation dimensions are added (never removed)._

---

## Overview

This document defines the success metrics and evaluation criteria for the Pokemon Battle Model. Metrics are organized into tiers: **gate metrics** (must pass to proceed), **tracking metrics** (monitored but not blocking), and **diagnostic metrics** (for debugging and analysis).

---

## 1. Battle Win Rate Metrics (Gate)

These are measured by playing N-game series (N >= 100 for statistical significance) against each opponent type.

| Metric | Floor | Target | Opponent |
|--------|-------|--------|----------|
| Win rate vs. random legal bot | 95% | 99% | Uniformly random legal action selection |
| Win rate vs. heuristic bot | 70% | 80% | Rule-based bot (see Phase 3 spec) |
| Win rate vs. simple BC baseline | >50% | >60% | MLP/GRU behavior cloning model |

**Statistical requirements:**
- All win rates reported with 95% Wilson confidence intervals.
- Minimum 200 games per matchup for gate evaluations; 500+ for final reporting.
- Use fixed random seeds for reproducibility across evaluation runs.

---

## 2. Action Prediction Accuracy (Gate)

Measured on the held-out test split of replay data (battles never seen during training).

| Metric | Floor | Target | Description |
|--------|-------|--------|-------------|
| Top-1 action accuracy | 35% | 45% | Model's highest-probability action matches the human's action |
| Top-3 action accuracy | 65% | 75% | Human's action is within the model's top 3 predictions |
| Negative log-likelihood (NLL) | — | Minimize | Average NLL of the human action under the model's distribution |

**Notes:**
- Action accuracy is measured only over turns where the player had >= 2 legal actions (exclude forced moves/switches).
- Report accuracy broken down by game phase (early: turns 1–10, mid: 11–25, late: 25+).
- Top-1 accuracy floors are deliberately modest — human play is noisy and multiple actions are often defensible.

---

## 3. Hidden-Information Auxiliary Prediction (Gate)

Evaluated on test-set turns where the ground truth was eventually revealed later in the same game.

| Metric | Floor | Target | Description |
|--------|-------|--------|-------------|
| Item prediction accuracy (top-1) | 40% | 55% | Correct item predicted for opponent Pokemon |
| Item prediction accuracy (top-3) | 70% | 80% | Correct item in top 3 predictions |
| Speed tier bucket accuracy | 50% | 65% | Correct speed bucket (5 ordinal categories) |
| Role archetype accuracy | 45% | 60% | Correct role (sweeper, wall, pivot, setter, etc.) |
| Move-family presence (avg F1) | 0.50 | 0.65 | Multi-label F1 for move categories (priority, recovery, hazards, etc.) |

**Calibration requirement (Gate):**
- Expected Calibration Error (ECE) for item predictions must be < 0.15.
- When the model assigns 70% probability to an item class, it should be correct roughly 70% of the time.
- Calibration is measured using 10 equal-width probability bins.

---

## 4. Cross-Archetype Robustness (Gate)

Win rate must be evaluated separately against teams representing each major archetype.

| Archetype | Minimum Win Rate vs. Heuristic Bot |
|-----------|--------------------------------------|
| Hyper Offense (HO) | 55% |
| Bulky Offense | 60% |
| Balance | 60% |
| Stall | 50% |
| Weather (Rain/Sun/Sand) | 55% |
| Trick Room | 50% |

**Gate criterion:** Cross-archetype win rate variance must be below 0.04 (i.e., standard deviation of per-archetype win rates < 0.20). The model must not be a specialist that collapses against unfamiliar styles.

---

## 5. Rare/Uncommon Set Robustness (Gate)

| Metric | Floor | Description |
|--------|-------|-------------|
| Win rate vs. off-meta sets | 55% vs. heuristic | Opponent uses legal but uncommon item/move/ability combinations |
| Performance drop vs. standard sets | < 15% | Win rate loss when opponent switches from standard to uncommon sets |

**Test procedure:**
- Construct 50+ uncommon-but-legal team sets using moves/items/abilities outside the top 80% usage.
- Play 20+ games per set.
- Model must not produce illegal actions or exhibit degenerate behavior (e.g., always switching, always using the same move).

---

## 6. Tracking Metrics (Non-Blocking)

These are logged and monitored but do not gate phase transitions.

| Metric | Description |
|--------|-------------|
| Value head accuracy | Binary cross-entropy of win probability predictions vs. actual game outcomes |
| Average game length (model vs. baselines) | Number of turns per game — anomalous shortening/lengthening flags issues |
| Action diversity | Entropy of action distributions — very low entropy suggests degenerate policy |
| Tera usage rate | How often the model chooses to Terastallize — compare against human rates |
| Switch rate | How often the model switches — compare against human rates |
| Illegal action attempt rate | Must be exactly 0% after Phase 1, but tracked as a safety net |

---

## 7. Diagnostic Metrics (Analysis Only)

| Metric | Description |
|--------|-------------|
| Per-turn confidence | Model's maximum action probability per turn — tracks uncertainty |
| Hidden-info prediction accuracy by turn number | Does item/speed prediction improve as game progresses? |
| Win rate by game length bucket | Does the model perform differently in short vs. long games? |
| Loss decomposition | Policy loss vs. auxiliary loss vs. value loss trends |
| Attention pattern analysis | Which tokens does the model attend to most? (Spot-check only) |

---

## 8. Evaluation Protocol

### Offline evaluation (replays)
- **Frequency:** After every training checkpoint.
- **Dataset:** Fixed held-out test split (never used for training or validation).
- **Report:** Action accuracy (top-1, top-3), NLL, hidden-info metrics, calibration.

### Online evaluation (live battles)
- **Frequency:** After each phase completion and after significant model changes.
- **Opponents:** Random bot, heuristic bot, BC baseline, and (later) previous model checkpoints.
- **Games per matchup:** 200 minimum for gate evaluation.
- **Report:** Win rate with confidence intervals, per-archetype breakdown, game length stats.

### Robustness evaluation
- **Frequency:** At phase exit gates.
- **Tests:** Uncommon-set suite, mirror-team suite, temporal holdout set.
- **Report:** Win rates, performance drop measurements, failure mode categorization.

---

## 9. Phase-Specific Exit Gate Summary

| Phase | Gate Metrics |
|-------|-------------|
| Phase 3 (Baselines) | Simple BC > 90% vs. random, > 55% vs. heuristic |
| Phase 4 (Transformer BC) | All Section 1–5 floors met; 100% legal actions; reproducible across seeds |
| Phase 5 (Synthetic) | Targeted failure rates decrease; no regression on real-replay metrics |
| Phase 7 (Offline RL) | Improvement over BC+synthetic on gate metrics without robustness regression |

---

_All metrics are computed using the evaluation harness built in Phase 6. Automated evaluation runs are tracked in Weights & Biases._
