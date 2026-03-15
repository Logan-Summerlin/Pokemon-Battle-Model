# Pokemon Model Pipeline Plan

_Created: 2026-03-15_
_Migrated to Gen 3 OU: March 2026_

A high-level plan for training a competitive Gen 3 OU (ADV) Pokemon battle agent across four stages: imitation learning, targeted synthetic fine-tuning, self-play reinforcement learning, and edge-case synthetic repair. Each stage builds on the previous, with explicit gates between them.

---

## Pipeline Overview

```
Stage 1: Imitation Learning (BC)          ← Working prototype exists
    ↓  [evaluation gate]
Stage 2: Targeted Synthetic Fine-Tuning   ← Nudge toward correct decisions
    ↓  [evaluation gate]
Stage 3: Self-Play Reinforcement Learning ← Learn to win, not just imitate
    ↓  [evaluation gate]
Stage 4: Edge-Case Synthetic Repair       ← Fix local minima and rare failures
```

The intuition (from the project transcript): Stage 1 teaches the model what high-Elo players *do*. Stage 2 sharpens it on clear-cut decisions before RL. Stage 3 teaches it to *win*. Stage 4 patches the gaps that RL's sparse reward signal misses — the rare board states where two options look nearly equal but one is strategically superior (e.g., "take the free KO" vs. "use the free turn to Swords Dance and sweep").

---

## Stage 1: Imitation Learning (Behavior Cloning)

**Status: Working prototype complete.**

### What We Have

- **BattleTransformer** (`src/models/battle_transformer.py`): 14 tokens/turn structured transformer with policy head (candidate-action scoring), auxiliary hidden-info head, and optional value head.
- **Three model variants** trained and evaluated:
  - P8 (4L/256d/4H, 3.6M params)
  - P8-Lean (3L/224d/4H, ~1.95M params)
  - P4 (6L/384d/6H, 11.5M params)
- **9-action space** (Gen 3: 4 moves + 5 switches, no Terastallization).
- **~50% top-1 action accuracy** on held-out replays (vs. ~11% random baseline across 9 possible actions).
- **Training pipeline**: `.npz` tensors → `WindowedTurnDataset` → masked cross-entropy + auxiliary loss (0.2 weight).
- **Data**: 10K battles, Gen 3 OU, 1300+ Elo, 80/10/10 battle-level splits.
- **No team preview**: opponent team entirely unknown at battle start — auxiliary head predicts hidden opponent info.

### What BC Does Well

- Learns fundamental move selection patterns from human play
- Captures type matchup reasoning, switching patterns, and momentum concepts
- Auxiliary head develops rudimentary opponent modeling (item/speed/role prediction)
- Respects hidden information doctrine (first-person observations only)

### What BC Cannot Learn

- **Winning** — BC predicts what humans *did*, not what *wins*. It learns the average human policy, including their mistakes.
- **Long-horizon planning** — Cross-entropy on single-turn actions cannot capture multi-turn strategies (setting up a sweep over 3-5 turns).
- **Adaptation** — BC produces a static policy; it cannot adjust to a specific opponent's tendencies mid-game.
- **Edge-case reasoning** — Rare but important situations (sweeper setup opportunities, endgame conversions) are underrepresented in the loss landscape.

### Exit Gate for Stage 1

Before proceeding to Stage 2, the BC model must:
- Produce legal actions 100% of the time
- Beat random bot >95% win rate
- Beat heuristic bot >70% win rate
- Show reasonable cross-archetype performance (no archetype below 40% win rate)
- Auxiliary head calibration non-degenerate

---

## Stage 2: Targeted Synthetic Fine-Tuning (Pre-RL Sharpening)

**Goal:** Tighten the BC policy on clear-cut decisions before the noise of RL. This is a "nudge" — short, focused, and constrained.

### Why This Stage Exists (Before RL)

The transcript captures the key insight: going directly from BC to RL forces the RL algorithm to simultaneously learn basic tactical correctness *and* complex winning strategy. By first sharpening the policy on unambiguous scenarios, we give RL a much better starting policy, which means:
- Faster RL convergence (fewer episodes wasted on obvious blunders)
- More stable training (policy already avoids catastrophic moves)
- RL can focus its learning budget on *strategy*, not *mechanics*

### Approach Options for Stage 2

#### Option A: Supervised Fine-Tuning (SFT) on Curated Scenarios (Recommended First)

**How it works:** Build a catalog of synthetic battle scenarios where the correct action (or acceptable action set) is known. Fine-tune the BC checkpoint using standard cross-entropy loss, mixing real replay batches with synthetic batches.

**Scenario taxonomy:**
1. **Forced mechanics** — forced switches, trap/choice-lock, no-safe-switch endgames
2. **Tactical one-turn motifs** — guaranteed KO available, sack-vs-preserve, setup-vs-attack
3. **Hidden-info ambiguity** — unknown item/ability affecting damage calcs or speed
4. **Tempo and game-phase** — early momentum pivots, hazard management, late-game conversion

**Label types:**
- **Hard-optimal**: One clearly correct action (forced KO, only legal move)
- **Acceptable set**: 2-3 strategically defensible actions (label smoothing or ranking loss)
- **Distributional**: For ambiguous positions, score probabilistic alignment rather than top-1

**Training constraints:**
- Mix ratio: 70-80% real replay batches + 20-30% synthetic batches
- Duration: 1-3 epochs over the synthetic set (not a second pretraining)
- Monitor real-replay validation — halt if it degrades
- Balance scenarios by concept family (no flooding of easy KO puzzles)

**Pros:** Simple, stable, directly addresses identified weaknesses, uses existing training infrastructure.
**Cons:** Requires human effort to build scenario catalog and labels. Does not teach the model to *win*, only to make better single-turn decisions.

#### Option B: GRPO (Group Relative Policy Optimization) on Synthetic Scenarios

**How it works:** For each scenario, sample K actions from the policy, score them with a reward function (damage dealt, KO achieved, position quality heuristic), and update the policy to favor higher-reward actions relative to the group.

**Why consider GRPO here:**
- No need for a separate value network (reward is relative within each group)
- Works well for single-turn decision quality improvement
- Naturally handles scenarios with multiple acceptable actions (the group ranking preserves them)
- More sample-efficient than full PPO for this narrow setting

**Reward function for Stage 2 GRPO:**
- +1.0 for hard-optimal action
- +0.5 for acceptable-set action
- -0.5 for suboptimal action
- -2.0 for catastrophic action (e.g., switching in a Pokemon to get OHKO'd)

**Training constraints:**
- Group size K = 8-16 samples per scenario
- KL penalty to BC prior to prevent drift
- Same mix ratio with real replay batches as Option A

**Pros:** Reward-based learning without needing full RL infrastructure. Naturally handles multi-answer scenarios.
**Cons:** Requires a reward function (even if simple). More complex than SFT. Risk of reward hacking on the heuristic reward.

#### Option C: DPO (Direct Preference Optimization) on Action Pairs

**How it works:** For each scenario, construct (preferred action, dispreferred action) pairs. Train the model to increase the log-probability gap between preferred and dispreferred actions, using the implicit reward formulation from DPO.

**Pair construction:**
- Expert-labeled action vs. model's current top prediction (when different)
- Hard-optimal action vs. random legal action
- Acceptable-set action vs. non-acceptable action

**Pros:** No reward function needed — just preference pairs. Stable training dynamics. Well-studied optimization landscape.
**Cons:** Requires constructing preference pairs (labor). Binary preference loses nuance in multi-way comparisons. Less flexible than GRPO for N-way decisions.

### Recommended Stage 2 Approach

**Start with SFT (Option A)** because it uses existing infrastructure, is simplest to implement, and directly validates the scenario catalog. If SFT plateaus but the scenario catalog reveals the model struggles with *ranking* multiple acceptable actions, **upgrade to GRPO (Option B)** using the same scenarios.

### Exit Gate for Stage 2

- Targeted failure categories show measurable reduction
- Real-replay validation performance NOT degraded
- Legality rate remains 100%
- No new failure modes introduced
- Hidden-info auxiliary calibration maintained or improved

---

## Stage 3: Self-Play Reinforcement Learning

**Goal:** Teach the model to *win games*, not just imitate human moves. This is where the agent develops genuine strategy — multi-turn planning, sweeper setup, endgame conversion, risk management.

### Why RL After SFT (Not Before)

The Stage 2 SFT/GRPO sharpening means the RL starting policy already:
- Avoids catastrophic tactical blunders
- Makes correct forced decisions (KO when available, switch when trapped)
- Has reasonable hidden-info estimates

This lets RL focus its exploration budget on *strategic* improvement rather than *mechanical* correction. Without Stage 2, early RL episodes would be dominated by tactical errors, producing noisy gradients that slow convergence.

### RL Algorithm Options

#### Option A: PPO (Proximal Policy Optimization) — Recommended Starting Point

**How it works:** On-policy actor-critic. The policy (actor) plays battles. A value function (critic) estimates expected return from each state. Policy is updated using clipped surrogate objective to prevent large policy steps.

**Why PPO for Pokemon:**
- Battle episodes are short enough (20-40 turns) that on-policy rollouts are tractable
- Clipped objective prevents catastrophic policy collapse
- Well-understood hyperparameter tuning
- Natural integration with the existing value head in BattleTransformer

**Architecture for PPO:**
- **Actor**: BattleTransformer policy head (already exists)
- **Critic**: BattleTransformer value head (already exists, currently optional)
- Shared encoder, separate heads — standard actor-critic weight sharing
- Legal mask applied before policy softmax (maintained throughout RL)

**Reward structure:**
- **Primary**: Terminal win/loss (+1 / -1)
- **Optional shaping** (add conservatively, one at a time):
  - Faint advantage: +0.1 per net faint advantage gained (you KO opponent - opponent KOs you)
  - HP differential: tiny per-turn bonus proportional to HP advantage (weight ≤0.01)
  - Hazard control: small bonus for maintaining hazard advantage
- **Danger**: Shaping can cause degenerate play. A model that maximizes "faint advantage" might sack its own sweeper to trade 1-for-1 even when preserving the sweeper leads to a sweep. Start terminal-only, add shaping only if convergence is too slow.

**Training loop per iteration:**
1. Sample opponents from weighted pool
2. Roll out N battles (512-2048) with stochastic policy (temperature > 0)
3. Compute advantages using GAE (Generalized Advantage Estimation)
4. Update policy and value for K minibatch epochs (K = 3-4)
5. Run evaluation battery
6. Checkpoint if multi-metric gate passes

**Hyperparameter starting points:**
- Learning rate: 3e-4 with linear decay
- Clip ratio: 0.2
- GAE lambda: 0.95
- Discount gamma: 0.99
- Entropy bonus: 0.01 (encourage exploration early, decay over time)
- Minibatch size: 256 turns
- Rollout batch: 1024-2048 battles per iteration

**Pros:** Battle-tested (pun intended), stable with clipping, on-policy means no distributional shift issues, well-suited to episodic games.
**Cons:** Sample-inefficient (needs many battles per update), requires well-tuned value function, on-policy rollouts can be slow.

#### Option B: GRPO (Group Relative Policy Optimization) for Full Games

**How it works:** Instead of a learned value function, GRPO uses *relative* rewards within a group of rollouts from the same state. For Pokemon, this means playing K games from similar starting positions and updating the policy to favor the trajectories that won.

**Why consider GRPO for full RL:**
- **No value network needed** — avoids the instability of critic training in a stochastic game
- In Pokemon, the same position can lead to very different outcomes due to damage rolls, crits, and opponent decisions. Relative ranking within a group naturally handles this variance.
- Successfully used in LLM fine-tuning (DeepSeek-R1) and has shown promise in game-playing agents.

**GRPO design for Pokemon:**
- Group size: K = 8-16 games from each starting configuration
- Reward: terminal win/loss (+1/-1)
- Advantage: normalize rewards within each group (winners get positive advantage, losers negative)
- KL penalty to reference policy (BC checkpoint) to prevent drift

**Challenge:** Unlike PPO, GRPO doesn't have a value function for per-turn credit assignment. In a 30-turn game, the model must figure out which *specific turns* contributed to winning or losing. This is the classic credit assignment problem, and it's harder without a critic.

**Mitigation:**
- Use short-horizon games initially (opponent teams designed for 10-15 turn games)
- Apply temporal discounting to weight later turns more heavily
- Use auxiliary signals (faint events, HP changes) as group-level features, not per-turn rewards

**Pros:** Simpler architecture (no critic), robust to reward noise, handles stochastic outcomes naturally.
**Cons:** Higher variance gradients, poor credit assignment in long episodes, requires many rollouts per group.

#### Option C: Offline RL (CQL / Decision Transformer) on Replay Data

**How it works:** Train on the existing replay dataset, but condition on game outcomes. Rather than predicting "what did the human do?", predict "what did the human do *in games they won*?"

**Conservative Q-Learning (CQL):**
- Learn a Q-function from replay data
- Add a penalty that pushes down Q-values for actions not in the data (prevents overestimation)
- Extract policy by selecting highest-Q legal action

**Decision Transformer:**
- Condition the transformer on desired return (e.g., "win")
- At inference, condition on return=+1 and sample actions
- Essentially filters the BC training data by outcome

**Why consider offline RL:**
- Uses existing replay data — no simulator needed
- No sample efficiency concerns (data already exists)
- Decision Transformer is architecturally similar to what we have

**Why it's limited for Pokemon:**
- **Ceiling is human skill level** — offline RL can only be as good as the best trajectories in the data. It cannot discover strategies that humans didn't use.
- **Credit assignment in replays is ambiguous** — a player who won might have made several suboptimal moves but won anyway due to team matchup or luck.
- CQL requires careful tuning of the conservatism penalty.

**Pros:** No simulator needed, uses existing data, stable training.
**Cons:** Cannot exceed human performance, limited exploration, credit assignment issues.

### Self-Play Design

Regardless of the RL algorithm, the model needs *opponents* to play against. Self-play is the gold standard for game-playing AI, but naive self-play in Pokemon has specific failure modes.

#### Opponent Curriculum

**Stage R1 — Stability (first 1-2 weeks):**
- Random legal bot (floor test)
- Heuristic bot (basic competence)
- Frozen BC checkpoint (baseline)
- Goal: Verify the RL loop works without crashing, policy doesn't degenerate

**Stage R2 — Competence (weeks 2-4):**
- Stronger heuristic variants (damage calc + switching logic + hazard awareness)
- Scripted tactical specialists (setup sweeper bot, stall bot, pivot-heavy bot)
- Frozen snapshots from earlier RL iterations
- Goal: Win rate >70% against all R2 opponents

**Stage R3 — Self-Play (weeks 4+):**
- Current policy vs. itself (true self-play)
- Current policy vs. population of frozen checkpoints (PSRO-lite)
- Current policy vs. off-meta/unusual team bots (robustness)
- Goal: Policy improves without cycling or collapse

#### Self-Play Failure Modes in Pokemon

1. **Meta collapse / cycling**: Policy A beats Policy B, B beats C, C beats A. The model oscillates between strategies rather than improving.
   - **Mitigation**: Population-based evaluation. Keep a pool of frozen checkpoints. A new checkpoint must beat the *population*, not just the previous version.

2. **Strategy narrowing**: The model discovers one dominant strategy and over-specializes (e.g., always hyper-offense because it beats its own earlier defensive policies).
   - **Mitigation**: Diverse opponent pool including scripted specialists. Archetype-stratified win rate metrics — block promotion if any archetype drops below threshold.

3. **Reward hacking with shaping rewards**: Model learns to maximize HP differential rather than winning (e.g., using recovery moves endlessly to maintain HP advantage in a position it could sweep from).
   - **Mitigation**: Terminal-only reward for initial RL. Add shaping only after confirming terminal-only converges too slowly.

4. **Hidden-info exploitation**: In self-play, both sides use the same model. The model could learn to "signal" information through its move choices in a way that exploits shared weights — an artifact that wouldn't transfer to playing against real opponents.
   - **Mitigation**: Asymmetric information enforcement. Each side sees only its first-person observation. Evaluate periodically against non-self opponents to detect self-play overfitting.

5. **Forgetting BC calibration**: Extended RL training can erase the useful patterns learned from human replays.
   - **Mitigation**: KL regularization to BC prior. Replay buffer mixing (keep a fraction of BC replay data in each training batch).

#### Anti-Collapse Safeguards (Mandatory for All RL Approaches)

1. **BC replay anchoring**: Each RL training batch includes 10-20% real replay data with BC loss. This prevents catastrophic forgetting of human-learned patterns.
2. **KL penalty to BC prior**: Constrain policy updates so the RL policy doesn't drift too far from the BC foundation. Start with KL coefficient = 0.1, tune down as RL stabilizes.
3. **Population checkpoints**: Save a checkpoint every N iterations. Evaluate new checkpoints against the full population. Only promote if population-wide win rate improves.
4. **Calibration watchdog**: Monitor auxiliary head accuracy and policy calibration each iteration. If confidence rises while correctness drops, stop training and rollback.
5. **Archetype-level metrics**: Track win rate per team archetype (HO, balance, stall, weather, etc.). Block promotion if any archetype regresses significantly.

### Recommended Stage 3 Approach

**Start with PPO** because:
- The BattleTransformer already has a value head designed for this
- PPO is the most well-understood algorithm for episodic game-playing
- Per-turn credit assignment via the value function is important for 30-turn Pokemon battles
- Battle infrastructure (`BattleEnv`, `BattleEvaluator`) already supports the rollout loop

**If PPO's value function is unstable** (common in stochastic games), **try GRPO** as an alternative that eliminates the critic.

**Keep offline RL (Decision Transformer) as a complementary approach** — use it to mine the existing replay dataset for high-quality trajectories, and mix those into the RL training buffer.

### Exit Gate for Stage 3

- Win rate vs. heuristic bot >85%
- Win rate vs. BC checkpoint >65%
- No archetype win rate below 45%
- Hidden-info calibration not degraded vs. post-Stage-2 checkpoint
- Policy produces legal actions 100% of the time
- Robustness on uncommon sets maintained

---

## Stage 4: Edge-Case Synthetic Repair (Post-RL)

**Goal:** Fix the local minima and rare-situation failures that RL's sparse reward signal misses. This is the "intuition injection" stage.

### Why This Stage Exists (After RL)

The transcript captures this perfectly: RL optimizes for *expected* reward, which means it focuses learning on *common* situations. Rare but strategically important scenarios — the ones that occur maybe 1 in 100 games — may not have enough gradient signal to move the model's weights.

**The sweeper example from the transcript:** Your sweeper faces a sleeping opponent. Option A: hit the super-effective move and KO the sleeper. Option B: use the free turn to Swords Dance, then sweep the remaining 3 opponents. Option B has higher expected value, but the model might prefer Option A because "KO opponent → closer to winning" is a pattern reinforced across thousands of RL games. The model lacks the *intuition* that a sweeper's job is to sweep, and a free setup turn is more valuable than one KO.

RL might find this eventually given enough games, but it's a local minimum — the reward difference between Options A and B is small (both often lead to winning), and the scenario is rare. Targeted synthetic repair is the efficient solution.

### Approach Options for Stage 4

#### Option A: SFT Repair on Expert-Labeled Edge Cases (Recommended)

**How it works:** Same as Stage 2 SFT, but the scenario catalog is built *specifically* from Stage 3 RL failure analysis.

**Failure analysis process:**
1. Run the RL-trained policy through 1000+ games, logging all decisions
2. Have domain experts review games where the model won/lost narrowly, or made decisions that look correct by outcome but wrong by reasoning
3. Cluster failures into families:
   - **Setup opportunity missed** (could have boosted but attacked)
   - **Endgame conversion failure** (winning position misplayed)
   - **Risk miscalculation** (safe play when behind, risky play when ahead)
   - **Opponent modeling failure** (ignored revealed information)
      - **Momentum mismanagement** (gave free switch after KO, didn't pivot)

**Scenario construction:**
- Extract real game states from failure clusters
- Apply controlled perturbations (HP variation, status toggling, bench swaps)
- Label with expert knowledge (hard-optimal or acceptable-set)
- Verify label stability across perturbations

**Training:**
- Mix: 70% real replay + 20% RL trajectory replay + 10% synthetic edge cases
- Short duration: 1-2 epochs over the synthetic set
- Monitor all metrics from previous stages

#### Option B: GRPO on Edge-Case Scenarios

**How it works:** Use the same scenario catalog but apply GRPO rather than SFT. For each edge-case scenario, sample K actions, score them with a domain-expert reward function, and update the policy to favor expert-preferred actions.

**Advantage over SFT:** GRPO naturally handles the "multiple acceptable actions" case that's common in edge-case positions. It also doesn't require hard labels — just a reward function that scores actions.

**Domain-expert reward function for edge cases:**
- **Setup opportunity**: +1.0 for boosting when safe, +0.3 for attacking, -0.5 for switching away
- **Endgame conversion**: +1.0 for actions that maintain winning advantage, -1.0 for throwing
- **Risk calibration**: reward function depends on game-state advantage (behind → aggressive rewarded, ahead → safe rewarded)

#### Option C: Targeted RL Curriculum on Edge-Case Scenarios

**How it works:** Rather than SFT/GRPO on synthetic snapshots, run additional RL (PPO) episodes that are *enriched* with edge-case starting positions. The opponent pool includes bots that specifically create edge-case situations (e.g., a bot that always puts something to sleep, forcing the setup-vs-KO decision).

**Pros:** The model learns through actual game outcomes, not expert labels. More likely to find truly optimal play.
**Cons:** Requires specialized opponent bots. RL signal in rare scenarios is inherently noisy. Risk of disrupting non-edge-case performance.

### Recommended Stage 4 Approach

**Start with SFT repair (Option A)** because it's most directly targeted and least risky. **Use GRPO (Option B) for scenarios where multiple actions are defensible** and hard labels would be reductive. **Reserve targeted RL curriculum (Option C) for cases where neither SFT nor GRPO produces satisfactory results** — situations where the model needs to learn through *playing out* the scenario, not just seeing the right answer.

### Exit Gate for Stage 4

- Edge-case failure rates decreased on all targeted categories
- Full evaluation battery shows no regression on any metric
- Cross-archetype performance maintained or improved
- Model passes expert review on curated difficult positions
- Calibration and hidden-info quality maintained

---

## Cross-Cutting Concerns

### Evaluation Battery (Runs After Every Stage and Major Checkpoint)

| Metric | Target | Purpose |
|--------|--------|---------|
| Legal action rate | 100% | Safety floor |
| Top-1 action accuracy (held-out replays) | >48% | BC quality preservation |
| Win rate vs. random bot | >95% | Sanity check |
| Win rate vs. heuristic bot | >85% (post-RL) | Competence |
| Win rate vs. BC checkpoint | >60% (post-RL) | Improvement signal |
| Cross-archetype win rate min | >45% | No blind spots |
| Hidden-info aux accuracy | Non-regressing | Calibration health |
| ECE (Expected Calibration Error) | <0.15 | Policy confidence quality |
| Uncommon-set robustness | >40% win rate | Generalization |

### Hidden Information Doctrine (Enforced at All Stages)

1. **Never train on omniscient features** — first-person observations only, always
2. **Represent uncertainty explicitly** — "unknown" markers, not zeros
3. **Metagame priors are soft hints** — never leaked truth
4. **Separate inference from decision** — auxiliary head predicts, policy head decides
5. **In self-play, enforce information barriers** — each side sees only its own observation

### Compute Considerations

| Stage | Compute Profile | Bottleneck |
|-------|----------------|------------|
| Stage 1 (BC) | GPU training on static dataset | Dataset I/O, model size |
| Stage 2 (SFT/GRPO) | Same as Stage 1 + scenario generation | Expert labeling effort |
| Stage 3 (RL) | GPU training + simulator rollouts | Battle throughput (simulator speed) |
| Stage 4 (Repair) | Same as Stage 2 | Expert analysis of RL failures |

**Stage 3 is by far the most compute-intensive.** Each PPO iteration requires thousands of full battles. Parallelizing the simulator (multiple Showdown instances) is likely required for reasonable training times.

### Implementation Priority

The existing codebase already has most of the infrastructure needed:
- `BattleTransformer` with policy + aux + value heads → PPO actor-critic
- `BattleEnv` + `BattleEvaluator` → rollout and evaluation loop
- `ModelBot` → inference wrapper for playing games
- `WindowedTurnDataset` → BC replay mixing
- Heuristic/random bots → opponent curriculum stages R1-R2

**New infrastructure needed:**
1. **Trajectory buffer** (`src/training/trajectory_buffer.py`) — store battle rollouts with observations, actions, rewards, legal masks
2. **RL trainer** (`src/training/rl_trainer.py`) — PPO/GRPO update loop
3. **Opponent pool** (`src/bots/opponent_pool.py`) — weighted sampling with curriculum stages
4. **Scenario factory** (`src/synthetic/scenario_factory.py`) — synthetic scenario generation and labeling
5. **Synthetic trainer** (`src/training/synthetic_trainer.py`) — SFT/GRPO on mixed real+synthetic data

---

## Decision Summary

| Decision Point | Recommended | Fallback |
|---------------|-------------|----------|
| Stage 2 method | SFT (simple, stable) | GRPO (if ranking matters) |
| Stage 3 algorithm | PPO (credit assignment via critic) | GRPO (if critic is unstable) |
| Stage 3 reward | Terminal-only (+1/-1) | Add shaping if convergence is slow |
| Stage 3 opponents | Curriculum: random → heuristic → self-play | Add scripted specialists if needed |
| Stage 4 method | SFT repair (targeted) | GRPO for multi-answer, RL curriculum for play-it-out |
| Self-play format | PSRO-lite (population of checkpoints) | Simple self-play with KL anchor |

---

## Risk Register

| Risk | Impact | Mitigation |
|------|--------|------------|
| RL causes catastrophic forgetting of BC patterns | High | KL penalty + replay mixing + rollback checkpoints |
| Self-play meta-cycles (A beats B beats C beats A) | Medium | Population evaluation, diverse opponent pool |
| Reward shaping causes degenerate play | High | Start terminal-only, add shaping conservatively |
| Synthetic scenarios don't transfer to real games | Medium | Verify label stability, always mix with real data |
| Compute bottleneck at Stage 3 | High | Parallelize simulator, start with small battle budgets |
| Hidden-info exploitation in self-play | Medium | Information barrier enforcement, non-self evaluation |
| Edge-case repair (Stage 4) introduces new failures | Low | Short training, heavy monitoring, regression suite |
