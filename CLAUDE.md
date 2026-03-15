# Pokemon Battle Model

Structured transformer for Gen 3 OU (ADV) singles on Pokemon Showdown. Behavior cloning on human replay data with hidden-information-aware architecture.

## Architecture
- **BattleTransformer** (`src/models/battle_transformer.py`): 14 tokens/turn (6 own + 6 opponent + field + context), transformer encoder, candidate-action-scoring policy head
- **Auxiliary head**: predicts opponent hidden info (items, speed tier, role, move families) from encoder output
- **Action space**: 9 canonical actions (4 moves + 5 switches), legal mask applied before softmax
- **Observation**: per-pokemon 28 dims (9 categorical + 14 continuous + 5 binary), field 19 dims, context 7 dims
- **No team preview**: Gen 3 has no team preview — opponent team entirely unknown at battle start

## Hidden Information Doctrine (Non-Negotiable)
1. Never train on omniscient features unavailable at decision time
2. Represent uncertainty explicitly with "unknown" markers, not zeros
3. Metagame priors are soft hints, not leaked truth
4. Separate hidden-state inference (auxiliary head) from move selection (policy head)

## Training
- Core trainer: `scripts/train_phase4.py` with wrapper scripts for specific configs
- Current variants: P8 (4L/256d/4H, 3.6M params), P8-Lean (3L/224d/4H, ~1.95M), P4 (6L/384d/6H, 11.5M)
- Data: Metamon dataset (gen3ou), 10K battles processed, 80/10/10 battle-level splits, 1300+ Elo
- Pipeline: `.npz` tensors → `WindowedTurnDataset` → per-turn examples with sliding window
- Loss: masked cross-entropy (policy) + auxiliary loss (weighted 0.2)

## Key Files
- `src/models/battle_transformer.py` — main model
- `src/data/observation.py` — observation construction
- `src/data/tensorizer.py` — tensorization pipeline
- `src/data/dataset.py` — windowed dataset
- `scripts/train_phase4.py` — training entry point

## Project Status
- Phases 0–3 complete (env, data pipeline, baselines)
- Phase 4 in progress (transformer model, experiments, optimization)
- Phases 5–8 not started (synthetic fine-tuning, evaluation harness, offline RL, enhancements)
- Migrated from Gen 9 OU to Gen 3 OU (March 2026)

## Testing
```bash
pytest                          # all tests
pytest tests/test_transformer.py  # transformer-specific
```
