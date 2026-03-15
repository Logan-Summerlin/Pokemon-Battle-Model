# Pokemon Battle Model: Evaluation Specification (Gen 3 OU)

_Frozen: March 2026 — Updated only when new evaluation dimensions are added (never removed)._
_Migrated to Gen 3 OU: March 2026_

---

## Overview

This document defines the success metrics and evaluation criteria for the Pokemon Battle Model targeting **Gen 3 OU (ADV)**. Metrics are organized into tiers: **gate metrics** (must pass to proceed), **tracking metrics** (monitored but not blocking), and **diagnostic metrics** (for debugging and analysis).

Gen 3-specific considerations:
- **No team preview**: The opponent's team is unknown at battle start, making hidden-info prediction harder but more valuable.
- **9-action space**: 4 moves + 5 switches (no Terastallization).
- **Type-based physical/special split**: Move category determined by type, not individual move.
- **Permanent weather**: Ability-set weather (Sand Stream) lasts indefinitely.
- **Compact metagame**: ~50-60 viable Pokemon, smaller vocabulary sizes.

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
| Top-1 action accuracy | 38% | 48% | Model's highest-probability action matches the human's action |
| Top-3 action accuracy | 68% | 78% | Human's action is within the model's top 3 predictions |
| Negative log-likelihood (NLL) | — | Minimize | Average NLL of the human action under the model's distribution |

**Notes:**
- Action accuracy is measured only over turns where the player had >= 2 legal actions (exclude forced moves/switches).
- Report accuracy broken down by game phase (early: turns 1–10, mid: 11–25, late: 25+).
- Top-1 accuracy floors are slightly higher than Gen 9 — Gen 3's smaller action space (9 vs 13) and more predictable metagame should yield better prediction accuracy.
- Gen 3 battles tend to be longer (more defensive meta), so report per-phase breakdowns carefully.
- Lead matchup accuracy (turn 1 action) should be reported separately — lead play is a distinct sub-game in Gen 3.

---

## 3. Hidden-Information Auxiliary Prediction (Gate)

Evaluated on test-set turns where the ground truth was eventually revealed later in the same game.

| Metric | Floor | Target | Description |
|--------|-------|--------|-------------|
| Item prediction accuracy (top-1) | 45% | 60% | Correct item predicted for opponent Pokemon (Gen 3: Leftovers-dominated distribution) |
| Item prediction accuracy (top-3) | 75% | 85% | Correct item in top 3 predictions |
| Speed tier bucket accuracy | 50% | 65% | Correct speed bucket (5 ordinal categories, Gen 3 thresholds: [110, 90, 65, 40]) |
| Role archetype accuracy | 45% | 60% | Correct role (sweeper, wall, trapper, setter, etc.) |
| Move-family presence (avg F1) | 0.50 | 0.65 | Multi-label F1 for move categories (priority, recovery, Spikes, status, etc.) |

**Calibration requirement (Gate):**
- Expected Calibration Error (ECE) for item predictions must be < 0.15.
- When the model assigns 70% probability to an item class, it should be correct roughly 70% of the time.
- Calibration is measured using 10 equal-width probability bins.

**Gen 3-specific hidden-info notes:**
- Item prediction floors are higher because Gen 3's item distribution is heavily concentrated (Leftovers dominates, Choice Band is the only Choice item).
- No team preview means hidden-info prediction accuracy should be measured across the full game, not just post-preview.
- Report **opponent team composition prediction**: given revealed Pokemon, can the model predict likely unrevealed teammates?
- Report **scouting efficiency**: does the model appropriately switch to reveal opponent team composition when advantageous?

---

## 4. Cross-Archetype Robustness (Gate)

Win rate must be evaluated separately against teams representing each major Gen 3 OU archetype.

| Archetype | Minimum Win Rate vs. Heuristic Bot | Gen 3 Examples |
|-----------|--------------------------------------|----------------|
| TSS (Toxic/Spikes/Sandstorm) | 55% | Tyranitar + Skarmory + Blissey core with Spikes + Toxic stalling under permanent Sand Stream |
| Bulky Offense | 60% | Salamence/Metagross/Suicune cores with Dragon Dance or Calm Mind sweepers |
| Hyper Offense (HO) | 55% | Dragon Dance Salamence, Swords Dance Heracross, mixed Tyranitar, Dugtrio trapping |
| Stall | 50% | Skarmory/Blissey/Milotic/Celebi cores with Wish + Protect + Spikes |
| Weather (Rain/Sun) | 55% | Rain Dance teams (Kingdra, Ludicolo sweepers) or Sunny Day teams (Exeggutor, Houndoom) |
| Baton Pass chains | 50% | Baton Pass teams chaining Speed/Attack boosts (Ninjask, Smeargle, Celebi) |

**Gen 3-specific archetype notes:**
- Trick Room does not exist in Gen 3 (introduced Gen 4). Replaced with Baton Pass chains, a legitimate Gen 3 strategy.
- TSS is the signature Gen 3 stall strategy and the most common archetype. It exploits permanent Sand Stream weather.
- "Trapper" is a distinct Gen 3 sub-strategy: Dugtrio (Arena Trap) and Magneton (Magnet Pull) remove specific threats.

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
| Average game length (model vs. baselines) | Number of turns per game — Gen 3 games tend to be longer (defensive meta); anomalous shortening flags issues |
| Action diversity | Entropy of action distributions — very low entropy suggests degenerate policy |
| Switch rate | How often the model switches — compare against human rates |
| Spikes usage rate | How often the model uses Spikes when available — the primary hazard in Gen 3 |
| Lead matchup win rate | Win rate broken down by whether the model won/lost the lead matchup (Gen 3-specific) |
| Weather exploitation rate | How effectively the model leverages or plays around permanent sandstorm |
| Illegal action attempt rate | Must be exactly 0% after Phase 1, but tracked as a safety net |

---

## 7. Diagnostic Metrics (Analysis Only)

| Metric | Description |
|--------|-------------|
| Per-turn confidence | Model's maximum action probability per turn — tracks uncertainty |
| Hidden-info prediction accuracy by turn number | Does item/speed prediction improve as game progresses? (Critical for no-team-preview) |
| Opponent team reveal curve | How many opponent Pokemon are revealed by turn N? Does the model scout appropriately? |
| Win rate by game length bucket | Does the model perform differently in short vs. long games? |
| Loss decomposition | Policy loss vs. auxiliary loss vs. value loss trends |
| Attention pattern analysis | Which tokens does the model attend to most? Especially: does it attend more to revealed vs. unrevealed opponent slots? |
| Physical/special damage calc accuracy | Does the model correctly account for the type-based phys/special split in damage estimation? |

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
_Gen 3 OU migration: All metrics calibrated for the ADV metagame. No Terastallization, no Stealth Rock, 9-action space._
