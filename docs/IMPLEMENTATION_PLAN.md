# Pokemon Battle Model: Implementation Plan

_Created: March 9, 2026_

---

## Design Philosophy

This plan synthesizes the original project plan and the red-team critique into **one executable build order**. The red-team report's central insight is correct: partial observability is not a side issue — it is the core game design constraint. Every architectural, training, and evaluation decision must be subordinated to that fact.

The plan follows one ruthless decision rule: **maximize expected progress per month under hidden information, data noise, and finite engineering bandwidth.** Breadth is deferred. Commitment is required.

---

## Frozen Scope for Version 1

| Decision | Choice | Rationale |
|---|---|---|
| Format | Gen 9 OU singles | Modern, data-rich, strategically relevant, difficult without being maximally pathological |
| Information regime | Closed team sheet with standard team preview | Preserves real competitive ambiguity while reducing roster uncertainty |
| Data window | Fixed replay date range (1-2 month snapshot) | Reduces metagame drift during training |
| Task | In-battle move/switch selection only | Team building is a separate project |
| Model | Structured candidate-action-scoring transformer | Captures sequence context without paying the generic-LLM tax |
| Training | Behavior cloning first, then narrow synthetic repair | Establishes baseline before adding complexity |
| State design | Public state + soft priors + explicit unknowns + auxiliary hidden-info head | Smallest design that respects the domain |
| Offline RL | Conditional — only after BC passes hard evaluation gates | Prevents compounding uncertainty prematurely |
| Search | Deferred until after RL proves gains | Search in hidden-info games is only as good as the belief model |

---

## Hidden Information Doctrine (Non-Negotiable)

These five rules govern every engineering decision:

1. **Never train on omniscient features** that would not be available to the player at decision time.
2. **Represent uncertainty explicitly** — use "unknown" markers, not zeros or defaults that could be confused with real values.
3. **Use metagame priors as soft hints**, not as leaked truth. Usage statistics inform probability channels, not ground-truth labels.
4. **Separate hidden-state inference from move selection**, even if they share an encoder. The auxiliary head predicts; the policy head decides.
5. **Evaluate rare-set robustness and ambiguity handling directly**, not just aggregate win rate.

---

## Phase 0: Research, Scoping, and Environment Decisions

**Duration target: ~1 week**

### Step 0.1: Write a one-page design memo
Freeze the following in a `SCOPE.md` file committed to the repo:
- Exact format: Gen 9 OU singles
- Exact ruleset version (Showdown server tag/commit)
- Information regime: closed team sheet, standard team preview
- Replay date window (e.g., January-February 2026)
- Minimum Elo threshold for replay inclusion (e.g., 1500+)
- Explicit list of what is deferred (team building, doubles, search, LLM integration)

### Step 0.2: Define success metrics
Create an `EVALUATION_SPEC.md` with:
- Win rate vs. random legal bot (floor: >95%)
- Win rate vs. heuristic bot (floor: >70%)
- Win rate vs. simple BC baseline (main model must exceed)
- Top-1 and top-3 action accuracy on held-out replays
- Hidden-info auxiliary prediction calibration scores
- Cross-archetype win rate variance (must be below threshold)
- Rare/uncommon set robustness score

### Step 0.3: Set up project infrastructure
- Git repository with branching strategy
- `pyproject.toml` or `setup.py` with pinned dependencies
- Experiment tracking (Weights & Biases or MLflow)
- Config system (Hydra or dataclass-based)
- CI with `pytest` for data/parser/legality tests
- Checkpoint naming convention and registry

**Exit gate:** Design memo, evaluation spec, and project skeleton all committed. No code yet — this is pure scoping.

---

## Phase 1: Simulator Environment and Battle Harness

**Duration target: ~2 weeks**

### Step 1.1: Stand up a local Pokemon Showdown server
- Clone and install `pokemon-showdown` at a pinned commit
- Verify it runs locally with the Gen 9 OU ruleset
- Write a smoke test: two random-move bots complete a full battle
- Document the exact server version and any patches applied

### Step 1.2: Build a Python battle interface
- Create a `BattleEnv` wrapper that:
  - Connects to the local Showdown server via websocket
  - Parses incoming messages into structured Python objects
  - Sends actions (move, switch, terastallize) as properly formatted commands
  - Tracks game state from the **first-person perspective only** (critical: never expose hidden state)
  - Exposes a `step(action) -> observation, reward, done, info` interface
- **Key design decision:** The observation must reconstruct only what the acting player knew at decision time. This is the single most important engineering constraint in the project.

### Step 1.3: Implement legality and action encoding
- Build a canonical action vocabulary:
  - 4 moves (each optionally + Tera) + up to 5 switches = variable-size legal action set per turn
  - Handle forced switches, Struggle, disabled moves, choice-locked moves, trapped states
- Implement `get_legal_actions(state) -> action_mask`
- **Unit test every edge case exhaustively:** trapping, choice lock, Encore, disabled moves, fainted forced-switch, Tera availability, etc.
- Compare legality masks against simulator acceptance on 1000+ sampled states

### Step 1.4: Build the battle harness
- Framework that can pit any `Bot` against any `Bot`
- Standard `Bot` interface: `choose_action(observation, legal_actions) -> action`
- Support for:
  - Single matches
  - Round-robin tournaments
  - N-game series with aggregated stats
- Logging: full replay export, per-turn action logs, timing info
- Deterministic seeds where possible

**Exit gate:** Two scripted bots can play 100 legal games to completion with zero crashes or legality violations. Full replay logs are generated and parseable.

---

## Phase 2: Data Pipeline

**Duration target: ~2-3 weeks**

### Step 2.1: Collect replay data
- Source: Pokemon Showdown replay API or bulk download
- Target: Gen 9 OU replays only, within the frozen date window
- Minimum: 200K battles for prototype phase; target 500K-1M for serious training
- Apply Elo filter: keep only games where both players are above the threshold
- Store raw logs immutably in a `data/raw/` directory
- Create a metadata table: `battle_id, date, player1_elo, player2_elo, winner, num_turns`

### Step 2.2: Write the replay parser
- Parse Showdown log format into an event stream:
  - `|switch|`, `|move|`, `|-damage|`, `|-status|`, `|faint|`, `|turn|`, etc.
- Extract per-turn:
  - Public battle state (both sides)
  - Actions taken by both players
  - Reveals (new moves, items, abilities shown)
  - Result
- **Critical:** Implement first-person reconstruction. For each turn and each player:
  - What did this player know at this point?
  - What was still hidden?
  - What was revealed by this turn's events?
- Flag and quarantine malformed, incomplete, or inconsistent logs

### Step 2.3: Build the observation constructor
The observation at turn `t` for player `p` contains:

**Own team (full info):**
- For each of 6 Pokemon: species, typing, current HP/max HP, status, stat boosts, known moves, item, ability, Tera type, Tera used flag

**Opponent team (partial info — first-person only):**
- Species identities from team preview (all 6 known)
- For each: HP fraction (visible), revealed moves, revealed item, revealed ability, revealed Tera
- Explicit `unknown` flags for unrevealed slots
- Soft prior features from usage statistics (optional but recommended):
  - Most common item distribution for this species
  - Most common ability distribution
  - Most common move families
  - Speed tier bucket probability

**Battlefield state:**
- Weather, terrain, trick room, gravity
- Entry hazards per side
- Screens per side
- Tailwind per side
- Other side conditions

**Turn context:**
- Turn number
- Last N actions (history window, recommend N=10-20 turns)
- Speed order observations (who moved first)
- Game phase indicator (early/mid/late based on remaining Pokemon)

**Legal action mask:**
- Binary mask over the canonical action vocabulary

### Step 2.4: Build the tensorization pipeline
- Convert observations to fixed-size tensor representations
- Use categorical embeddings for species, moves, items, abilities, types
- Use continuous features for HP fractions, stat boosts, turn number
- Use binary flags for status conditions, reveals, unknowns
- Output: `(batch, seq_len, feature_dim)` tensors for transformer input
- Cache processed tensors as Parquet or binary shards for fast loading

### Step 2.5: Create train/validation/test splits
- Split by **battle**, not by turn (prevent leakage)
- Recommended split: 80/10/10
- Additionally hold out:
  - A time-slice test set (later date range)
  - A team-archetype test set (specific archetypes reserved for test only)
- Create manifest files listing battle IDs in each split

### Step 2.6: Data quality audit
- Run the parser on the full corpus
- Compute statistics: battles per day, average turns, action distributions, Elo distribution
- Manually inspect 50+ random battles to verify parser correctness
- Check that reconstructed first-person observations match what the player actually saw
- Remove battles with parsing errors (target <1% loss rate)

**Exit gate:** Clean dataset of 200K+ battles with verified first-person observations, legal action masks, and train/val/test splits. Parser passes all unit tests. Manual inspection confirms no hidden-state leakage.

---

## Phase 3: Baselines

**Duration target: ~1-2 weeks**

### Step 3.1: Random legal bot
- Uniformly selects from legal actions
- Serves as absolute floor for evaluation
- Also stress-tests legality masks

### Step 3.2: Heuristic bot
Implement a scripted bot with these rules (priority order):
1. If a move guarantees a KO, use it (requires damage estimation)
2. If the active Pokemon is at a severe type disadvantage, switch to the best available counter
3. Use the highest-expected-damage legal move
4. Break ties randomly

This bot should be competent but clearly beatable by skilled play.

### Step 3.3: Simple BC baseline (MLP/GRU)
- Architecture: 2-layer MLP or small GRU over the flattened observation
- Training: masked cross-entropy loss over legal actions
- Dataset: full training split
- Train for convergence (early stopping on validation loss)
- Evaluate:
  - Action prediction accuracy on held-out replays
  - Win rate vs. random bot
  - Win rate vs. heuristic bot

**Exit gate:** Simple BC baseline beats random bot >90% of the time and beats heuristic bot >55% of the time. Training pipeline runs without data corruption or legality bugs.

---

## Phase 4: Main Transformer Model with Hidden-Info Auxiliary Head

**Duration target: ~3-4 weeks**

This is the core of the project. Every decision here is load-bearing.

### Step 4.1: Design the transformer architecture

**Input tokenization:**
- Each Pokemon (own and opponent) becomes a "slot token" — a learned embedding from its features
- Global state (weather, terrain, hazards, screens) becomes a "field token"
- Each turn in the history window becomes a "turn token" encoding the actions and key events
- Each legal candidate action becomes a "candidate token"

**Encoder:**
- Transformer encoder (4-8 layers, 256-512 hidden dim to start)
- Self-attention over all tokens (slot tokens, field token, history tokens)
- Output: contextual embeddings for all tokens

**Policy head (candidate-action scoring):**
- For each candidate action, compute a score by attending the candidate token to the encoder output
- Apply legal action mask
- Softmax over legal actions
- Loss: cross-entropy against the action taken in the replay

**Auxiliary hidden-info head:**
- Branches off the encoder output
- For each opponent Pokemon, predicts:
  - Item class (categorical: choice, boots, leftovers, life orb, etc.)
  - Speed bucket (ordinal: very fast, fast, medium, slow, very slow)
  - Role archetype (categorical: sweeper, wall, pivot, hazard setter, etc.)
  - Tera category (categorical: offensive, defensive, STAB, coverage)
  - Move-family presence (multi-label: has priority, has recovery, has hazards, has status, etc.)
- Loss: cross-entropy or binary cross-entropy for each sub-head
- **Label source:** Only labels derivable from the replay without leaking decision-time truth. E.g., if an item is revealed on turn 15, it can be used as a label for predictions made on turns 1-14 of that game. Labels from games where the info was never revealed should be excluded, NOT filled in from external databases.
- Loss weighting: auxiliary losses weighted at ~0.1-0.3x the policy loss (tune via ablation)

**Value head (optional but recommended):**
- Single scalar predicting win probability from current state
- Loss: binary cross-entropy against game outcome
- Useful for later search and RL stages

### Step 4.2: Implement compute-efficient training

**Scaling ladder (from the project plan's guidance):**

| Stage | Dataset | Model | Device | Purpose |
|---|---|---|---|---|
| Smoke test | 5K battles | Tiny (2L, 128d) | CPU | Verify pipeline end-to-end |
| Small | 25K battles | Small (4L, 256d) | 1 GPU | Validate learning signal |
| Medium | 100K battles | Medium (6L, 384d) | 1 GPU | First real ablations |
| Full | 500K+ battles | Target (8L, 512d) | 1 GPU | Main training run |

For each stage, record: examples/sec, peak VRAM, epoch time, validation loss, win rate vs. baselines.

**Training details:**
- Optimizer: AdamW, lr=1e-4 with cosine decay and warmup
- Batch size: as large as VRAM allows (start with 256, scale)
- Sequence length: 20-turn history window
- Mixed precision (fp16/bf16) from the start
- Gradient accumulation if batch size is VRAM-limited
- Early stopping on validation action-prediction loss
- Save checkpoints every epoch + best validation checkpoint

### Step 4.3: Run systematic ablations
After the first full training run, ablate these factors (one at a time):

| Ablation | Variants | What it tests |
|---|---|---|
| History length | 5, 10, 20, 30 turns | How much context matters |
| Elo threshold | 1200+, 1500+, 1700+ | Quality vs. quantity tradeoff |
| Auxiliary loss weight | 0, 0.1, 0.3, 0.5 | Value of hidden-info prediction |
| Model size | 4L/256d, 6L/384d, 8L/512d | Scaling behavior |
| Usage priors | with vs. without | Value of metagame soft features |
| Unknown markers | explicit vs. implicit | How to represent missing info |

### Step 4.4: Evaluate thoroughly
Run the full evaluation battery after each significant checkpoint:

**Offline metrics:**
- Top-1 action accuracy
- Top-3 action accuracy
- NLL on held-out replays
- Calibration error (are 80%-confidence moves right 80% of the time?)

**Battle metrics:**
- Win rate vs. random bot (target: >95%)
- Win rate vs. heuristic bot (target: >75%)
- Win rate vs. simple BC baseline (must exceed)
- Cross-archetype win rate breakdown (stall, HO, balance, etc.)

**Hidden-info metrics:**
- Auxiliary head accuracy on held-out reveals
- Item prediction accuracy by turn number
- Role prediction accuracy
- Calibration of hidden-info predictions

**Robustness metrics:**
- Win rate against uncommon but legal sets
- Performance when opponent uses off-meta strategies
- Decision quality in ambiguous states (human-reviewed sample)

**Exit gate:** Main transformer BC model:
- Clearly beats all baselines
- Shows reasonable hidden-info prediction calibration
- Achieves stable cross-archetype performance
- Produces legal actions 100% of the time
- Reproduces across random seeds

---

## Phase 5: Narrow Synthetic Fine-Tuning (Post-BC Repair Stage)

**Duration target: ~2 weeks**

This phase exists **only** to patch specific, identified weaknesses in the BC model. It is NOT a second pretraining regime.

### Step 5.1: Audit BC failures
- Play 500+ games against baselines and record all losses
- Categorize failure modes:
  - Mechanics errors (wrong damage calc intuition, status mishandling)
  - Tactical blunders (missing forced KOs, bad sack decisions)
  - Endgame conversion failures (winning position -> loss)
  - Ambiguity failures (overcommitting to one hidden-state hypothesis)
  - Rare-situation failures (unfamiliar sets, unusual board states)
- Rank failure categories by frequency and severity

### Step 5.2: Build a targeted scenario factory
For each identified failure cluster:
- Extract real replay states where the failure pattern occurs (anchor states)
- Apply controlled perturbations:
  - HP variation within narrow buckets
  - Status toggling
  - Hazard/weather/terrain toggling
  - Bench Pokemon swaps with usage-plausible alternatives
- **Never** create impossible or format-illegal combinations
- Tag every perturbation for later analysis

### Step 5.3: Label scenarios with three label types only
1. **Hard labels:** Only when the position is genuinely forced (one legal move, guaranteed KO with no downside, forced sack)
2. **Acceptable-action-set labels:** When 2-3 actions are all strategically reasonable
3. **Opponent-distribution labels:** When predicting opponent behavior, use probability targets, not single forced answers

**Label stability requirement:** A scenario enters the training set only if its label is stable across:
- Multiple opponent-policy assumptions
- Multiple random seeds
- Small HP/damage-roll perturbations

### Step 5.4: Fine-tune with strict constraints
- Mix: 70-80% real replay batches + 20-30% synthetic batches
- Keep the stage short: 1-3 epochs over the synthetic set
- Monitor validation performance on real replays — if it degrades, reduce synthetic weight or stop
- Balance synthetic examples by concept family (no flood of "obvious KO" puzzles)
- Use label smoothing and ranking losses for acceptable-action-set scenarios

### Step 5.5: Validate the repair
- Re-run the full evaluation battery
- Specifically re-test the identified failure clusters
- Confirm:
  - Failure rates decreased on targeted categories
  - Real-replay validation performance did NOT degrade
  - No new failure modes introduced
  - Hidden-info calibration maintained or improved

**Exit gate:** Measurable reduction in targeted failure categories without regression on general performance. If this gate fails, roll back and revisit the BC model or observation design — do not force synthetic data harder.

---

## Phase 6: Brutal Evaluation Harness

**Duration target: ongoing, but initial build ~1 week**

This is built in parallel with Phase 4 but expanded here. The evaluation harness must be harder than you think you need.

### Fixed benchmark suite components:

1. **Archetype-stratified test set:** Games involving stall, hyper offense, balance, rain, sun, sand, trick room, etc. Performance must be reasonable across ALL archetypes, not just common ones.

2. **Uncommon-set stress tests:** Inject opponents using legal but off-meta sets. The model must not collapse when expectations are violated.

3. **Mirror-team tests:** Both sides use identical teams. This isolates pure decision quality from team matchup advantage.

4. **Hidden-info calibration suite:** Curated positions where hidden information is critical. Score the model's hidden-info predictions against ground truth.

5. **Synthetic gate tests:** The targeted failure-cluster tests from Phase 5.

6. **Decision-quality review:** 100+ curated difficult states reviewed by a human expert. Score: did the model make a defensible choice?

7. **Temporal holdout:** Test on replays from a later date window to measure generalization beyond the training metagame snapshot.

---

## Phase 7: Conditional Advancement to Offline RL

**Duration target: ~3-4 weeks IF gates pass**

### Promotion gate (ALL must be met):
- BC+synthetic model clearly beats heuristic and scripted baselines
- Robust across archetypes (no archetype win rate below 40%)
- Hidden-info auxiliary heads show useful calibration
- Improvements reflect real battle play, not only synthetic scores
- Model does not collapse on uncommon but legal sets

**If any gate fails:** Do NOT add RL. Instead, fix the observation model, action model, data quality, or benchmark. Return to Phase 4 or 5.

### Step 7.1: Define reward structure
- Primary: terminal win/loss (+1/-1)
- Optional shaping (keep conservative):
  - HP differential at end of turn (small weight)
  - Faint advantage (small weight)
- **Caution:** Shaping can cause degenerate behavior. Start with terminal-only reward and add shaping only if needed.

### Step 7.2: Initialize from best BC checkpoint
- Use the BC+synthetic checkpoint as policy initialization
- This is much safer than training RL from scratch

### Step 7.3: Train with offline RL
- Compare at least two approaches:
  - Conservative Q-Learning (CQL) style
  - Decision-transformer style (return-conditioned)
- Use the same replay dataset
- Monitor for training instability and reward hacking

### Step 7.4: Evaluate RL improvement
- Full evaluation battery comparison: RL checkpoint vs. BC+synthetic checkpoint
- **Kill criteria:**
  - If RL improves in-distribution but worsens robustness → roll back
  - If RL produces overconfident policies with worse calibration → roll back
  - If RL gains are smaller than BC ablation gains → deprioritize RL and invest in better data/features

---

## Phase 8: Later-Stage Enhancements (Version 2+)

These are explicitly deferred until Phases 0-7 produce a strong, evaluated system.

### 8.1: Self-play fine-tuning
- Generate self-play games from the best checkpoint
- Fine-tune on self-play trajectories
- Monitor for meta collapse (keep population snapshots, evaluate cross-play)

### 8.2: Shallow search at inference time
- Policy proposes top-k actions
- Value head scores each after a 1-2 turn rollout
- Rerank and select
- Only viable after the value head and hidden-info model are calibrated

### 8.3: Opponent modeling module
- Track opponent tendencies within a single game
- Update beliefs about opponent sets based on revealed information
- Feed updated beliefs into the policy

### 8.4: Population training / PSRO
- Maintain a population of policy checkpoints
- Train against diverse opponents
- Reduce exploitability

### 8.5: Team building (separate project)
- Only after the battle policy is strong
- Autoregressive team generator or retrieval-based system
- Coupled with battle policy for evaluation

---

## Software Stack

| Component | Choice | Reason |
|---|---|---|
| Language | Python 3.11+ | Standard for ML |
| ML framework | PyTorch | Best ecosystem for research |
| Simulator | Pokemon Showdown (local) | Standard, well-maintained |
| Battle interface | Custom websocket client | Direct control over observation construction |
| Experiment tracking | Weights & Biases | Industry standard |
| Config | Hydra or dataclass-based | Reproducible experiments |
| Data format | Parquet + binary tensor shards | Fast I/O, schema enforcement |
| Testing | pytest | Data, parser, legality, and model tests |
| Version control | Git with clear branching | Standard |

---

## Project Directory Structure

```
pokemon-battle-model/
├── SCOPE.md                    # Frozen scope decisions
├── EVALUATION_SPEC.md          # Success metrics
├── configs/                    # Hydra configs
│   ├── model/
│   ├── training/
│   └── evaluation/
├── src/
│   ├── environment/            # Showdown interface, battle env
│   │   ├── showdown_client.py
│   │   ├── battle_env.py
│   │   ├── action_space.py
│   │   └── legality.py
│   ├── data/                   # Replay parsing, observation construction
│   │   ├── replay_parser.py
│   │   ├── observation.py
│   │   ├── tensorizer.py
│   │   ├── dataset.py
│   │   └── priors.py           # Usage statistics / metagame priors
│   ├── models/                 # Model definitions
│   │   ├── baseline_mlp.py
│   │   ├── battle_transformer.py
│   │   ├── heads.py            # Policy, value, hidden-info heads
│   │   └── embeddings.py
│   ├── training/               # Training loops
│   │   ├── bc_trainer.py
│   │   ├── synthetic_trainer.py
│   │   └── rl_trainer.py
│   ├── bots/                   # Bot implementations
│   │   ├── random_bot.py
│   │   ├── heuristic_bot.py
│   │   └── model_bot.py
│   ├── evaluation/             # Evaluation harness
│   │   ├── battle_evaluator.py
│   │   ├── offline_metrics.py
│   │   ├── benchmark_suite.py
│   │   └── calibration.py
│   └── synthetic/              # Synthetic scenario factory
│       ├── scenario_factory.py
│       ├── perturbations.py
│       └── labeling.py
├── tests/                      # Unit and integration tests
│   ├── test_legality.py
│   ├── test_parser.py
│   ├── test_observation.py
│   └── test_tensorizer.py
├── scripts/                    # One-off scripts
│   ├── download_replays.py
│   ├── process_dataset.py
│   ├── train.py
│   └── evaluate.py
├── data/
│   ├── raw/                    # Immutable raw replay logs
│   ├── processed/              # Tensorized datasets
│   └── splits/                 # Train/val/test manifests
└── checkpoints/                # Model checkpoints
```

---

## Kill Criteria (When to Stop and Reassess)

| Signal | Action |
|---|---|
| BC plateaus without cross-archetype gains | Stop scaling model size. Revisit state design and data quality. |
| Hidden-info auxiliary heads are uncalibrated | Do not add search. Fix the observation model. |
| Offline RL improves only in-distribution but worsens robustness | Roll back RL. Improve data or try different RL approach. |
| Synthetic fine-tuning degrades real-replay validation | Reduce synthetic weight or remove the stage entirely. |
| Legality violations persist after Phase 1 | Stop all model work. Fix the action space and legality system. |
| Parser reconstruction doesn't match simulator output | Stop all model work. Fix the parser. |

---

## Compute Budget Priorities (Ordered)

1. **Correct data pipeline** — no model can overcome bad data
2. **Sufficient storage and RAM** for dataset processing
3. **One empirically justified GPU** for transformer BC (determined by scaling ladder)
4. **Evaluation automation** — must be cheap to run evaluations frequently
5. **Only then** broader RL/self-play scaling

Start with CPU smoke tests. Scale empirically. Buy/allocate GPU only after the scaling ladder proves what you need.

---

## Summary: The One Path

```
Phase 0: Freeze scope, metrics, infrastructure
    ↓
Phase 1: Local Showdown + battle harness + legality
    ↓
Phase 2: Replay parser → first-person observations → tensorized dataset
    ↓
Phase 3: Random bot + heuristic bot + simple BC baseline
    ↓
Phase 4: Structured transformer BC + hidden-info auxiliary head ← CORE
    ↓  [evaluation gate]
Phase 5: Narrow synthetic fine-tuning to repair identified BC failures
    ↓  [evaluation gate]
Phase 6: Brutal evaluation harness (ongoing)
    ↓  [promotion gate: ALL criteria must pass]
Phase 7: Offline RL (conditional)
    ↓  [kill criteria active]
Phase 8+: Self-play, search, opponent modeling, team building (deferred)
```

Each phase has explicit exit gates. No phase begins until the previous gate is passed. This is how you turn a project plan into an actual build order.
