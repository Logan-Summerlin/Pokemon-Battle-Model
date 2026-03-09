# Pokemon Battle Model: Scope Document

_Frozen: March 2026 — Changes require explicit versioning and justification._

---

## Format

**Generation:** Gen 9 (Scarlet & Violet)
**Tier:** OU (OverUsed) singles
**Simulator:** Pokemon Showdown
**Ruleset version:** Pinned to the latest stable `pokemon-showdown` server release as of March 2026. The exact commit SHA will be recorded in `configs/environment/showdown.yaml` once the server is cloned in Phase 1.

## Information Regime

**Team sheet visibility:** Closed team sheet with standard team preview.
- Both players see all 6 species at team preview, but not items, abilities, moves, EVs, IVs, natures, or Tera types.
- Information is revealed only through in-battle events (switching in, using moves, taking damage, activating abilities/items).
- The model must **never** have access to information that the acting player would not have at decision time.

## Data Window

**Replay source:** Pokemon Showdown public replay API
**Date range:** January 1, 2026 — February 28, 2026
**Elo threshold:** Both players must be rated 1500+ at the time of the game.
**Target volume:** 200K–500K battles (start with 200K for prototype, scale to 500K+ for full training).

## Task

**In-scope:** In-battle move and switch selection, given a game state observed from the first-person perspective.

**Explicitly deferred (Version 2+):**
- Team building / team selection
- Doubles / VGC / other formats
- Other generations (Gen 1–8)
- Monte Carlo tree search or any inference-time search
- LLM-based reasoning or chain-of-thought decision making
- Opponent modeling beyond what the auxiliary head provides
- Self-play training
- Real-time ladder play on the public Showdown server

## Model Architecture

**Type:** Structured candidate-action-scoring transformer
- Not a generic language model. Purpose-built for the Pokemon battle decision space.
- Encodes battle state as structured tokens (slot tokens, field token, history tokens, candidate action tokens).
- Policy head scores candidate actions via cross-attention to encoder output.
- Auxiliary hidden-info head predicts opponent hidden state (items, speed tiers, roles, move families).
- Optional value head predicts win probability.

## Training Strategy

1. **Behavior cloning (BC)** on human replay data — establishes the primary policy.
2. **Narrow synthetic fine-tuning** — patches specific identified BC failure modes (Phase 5, post-BC only).
3. **Offline RL** — conditional on BC+synthetic model passing all evaluation gates (Phase 7).

## Hidden Information Doctrine

1. Never train on omniscient features unavailable at decision time.
2. Represent uncertainty explicitly — use "unknown" markers, not zeros.
3. Use metagame priors as soft hints, not leaked truth.
4. Separate hidden-state inference from move selection (even if shared encoder).
5. Evaluate rare-set robustness and ambiguity handling directly.

---

_This document is the authoritative scope reference. All implementation decisions must be consistent with these constraints._
