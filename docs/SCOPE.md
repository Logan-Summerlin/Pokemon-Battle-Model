# Pokemon Battle Model: Scope Document

_Frozen: March 2026 — Changes require explicit versioning and justification._
_Migrated from Gen 9 OU to Gen 3 OU: March 2026_

---

## Format

**Generation:** Gen 3 (Ruby/Sapphire/Emerald, also known as ADV)
**Tier:** OU (OverUsed) singles
**Simulator:** Pokemon Showdown
**Ruleset version:** Pinned to the latest stable `pokemon-showdown` server release as of March 2026. The exact commit SHA will be recorded in `configs/environment/showdown.yaml` once the server is cloned in Phase 1.

### Why Gen 3?

Gen 3 presents a more challenging hidden-information problem than Gen 9:
- **No team preview** — the opponent's entire team is unknown at battle start, making the auxiliary hidden-info head more valuable.
- **Simpler action space** — no Terastallization reduces from 13 to 9 canonical actions (4 moves + 5 switches).
- **Compact metagame** — ~50-60 viable OU Pokemon, ~100 commonly seen moves, smaller item pool (Leftovers/Choice Band dominated).
- **Research alignment** — the Metamon paper (Grigsby et al., RLC 2025) specifically chose Gens 1-4 as the most partially observed competitive settings.
- **Interesting strategic depth** — permanent weather (Sand Stream), Spikes-only hazard game, type-based physical/special split, rich lead metagame.

## Information Regime

**Team sheet visibility:** No team preview (Gen 3 does not have team preview).
- The opponent's team is entirely unknown at battle start.
- Species are revealed only when they switch in for the first time.
- Items, abilities, and moves are revealed through in-battle events.
- Information is revealed only through in-battle events (switching in, using moves, taking damage, activating abilities/items).
- The model must **never** have access to information that the acting player would not have at decision time.

## Data Window

**Replay source:** Metamon parsed replay dataset (jakegrigsby/metamon-parsed-replays on Hugging Face)
**Format:** gen3ou
**Elo threshold:** Both players must be rated 1300+ at the time of the game.
**Target volume:** 200K–500K battles (start with 10K random sample for development, scale to full corpus for training).

## Task

**In-scope:** In-battle move and switch selection, given a game state observed from the first-person perspective.

**Explicitly deferred (Version 2+):**
- Team building / team selection
- Doubles / VGC / other formats
- Other generations (Gen 1–2, Gen 4–9)
- Monte Carlo tree search or any inference-time search
- LLM-based reasoning or chain-of-thought decision making
- Opponent modeling beyond what the auxiliary head provides
- Self-play training
- Real-time ladder play on the public Showdown server

## Action Space

**9 canonical actions** (Gen 3 — no Terastallization):
- 4 move slots (move 1–4)
- 5 switch targets (switch to team slots 2–6)
- Legal action mask restricts available actions per turn

## Model Architecture

**Type:** Structured candidate-action-scoring transformer
- Not a generic language model. Purpose-built for the Pokemon battle decision space.
- Encodes battle state as 14 structured tokens per turn (6 own-team + 6 opponent-team + field + context).
- Per-pokemon features: 28 dims (9 categorical + 14 continuous + 5 binary).
- Field features: 19 dims. Context features: 7 dims.
- Policy head scores candidate actions via cross-attention to encoder output.
- Auxiliary hidden-info head predicts opponent hidden state (items, speed tiers, roles, move families).
- Optional value head predicts win probability.

## Training Strategy

1. **Behavior cloning (BC)** on human replay data — establishes the primary policy.
2. **Narrow synthetic fine-tuning** — patches specific identified BC failure modes (Stage 2, post-BC only).
3. **Self-play RL** — conditional on BC+synthetic model passing all evaluation gates (Stage 3).
4. **Edge-case synthetic repair** — fixes local minima and rare failures post-RL (Stage 4).

## Hidden Information Doctrine

1. Never train on omniscient features unavailable at decision time.
2. Represent uncertainty explicitly — use "unknown" markers, not zeros.
3. Use metagame priors as soft hints, not leaked truth.
4. Separate hidden-state inference from move selection (even if shared encoder).
5. Evaluate rare-set robustness and ambiguity handling directly.

---

_This document is the authoritative scope reference. All implementation decisions must be consistent with these constraints._
