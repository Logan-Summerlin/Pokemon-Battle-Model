# Project Plan: Building a Machine-Learning Pokémon Battle Model

_Last updated: March 9, 2026_

## 1. Purpose and target outcome

This project plan describes how to build a strong Pokémon battle model using machine learning, with the attached notes and recent public systems as context. It is designed to be practical rather than purely academic: the goal is to produce a reproducible training and evaluation pipeline that can start with a simple baseline and scale toward stronger battle play.

The plan assumes you want a system that can eventually do most or all of the following:

1. parse and learn from Pokémon Showdown battle data,
2. make legal and strategically reasonable in-battle decisions,
3. support multiple model families rather than locking into one early,
4. evaluate itself rigorously against bots, held-out matchups, and optionally humans,
5. remain extensible to team-building, set prediction, opponent modeling, and search.

This plan is informed especially by:

- the uploaded notes on recent Pokémon battle models,
- the uploaded summary of the Metamon offline RL paper,
- the uploaded long note on ML for team building and battle play,
- the Hugging Face `cameronangliss/vgc-bench-models` page,
- and the Hugging Face `jakegrigsby/metamon` page.

Those sources point to two useful recent directions for trained Pokémon agents:

- **Metamon** for large-scale offline RL with transformers on replay data in singles.
- **VGC-Bench** as a useful benchmark and infrastructure reference, even though this project plan is explicitly scoped to singles rather than doubles/VGC.

## 2. High-level strategic recommendation

Do **not** begin by trying to train the strongest possible end-state model immediately.

Instead, structure the project into five layers:

1. **Environment and data layer**: local simulator, replay parser, canonical schema, legality masks.
2. **Baseline policy layer**: heuristic bot, random legal bot, simple behavior cloning model.
3. **Scaled learning layer**: larger transformer policy, offline RL, self-play fine-tuning, opponent-conditioning.
4. **Inference-time improvement layer**: search, value heads, ensemble voting, set prediction, opponent belief tracking.
5. **Optional upstream meta layer**: team generation, moveset recommendation, matchup-aware team optimization.

This staged approach is the safest because recent work shows that Pokémon agents improve from a combination of:

- good state representation,
- high-quality battle trajectories,
- legality-aware action heads,
- offline learning,
- and, in some systems, search or game-theoretic population training.

## 3. What you should build first

Your initial target should be **one singles format only**.
Do not split effort across singles and doubles, and do not treat VGC as part of version 1.

Recommended default choices:

### Option A: Gen 9 OU singles
Best if you want an active modern metagame, abundant replay data, and a format that is strategically rich without the added complexity of doubles targeting.

Advantages:
- large amount of replay data,
- modern mechanics,
- large player base,
- easier benchmarking against common bots and ladder-style play.

Disadvantages:
- modern mechanics increase state complexity,
- metagame can shift over time,
- hidden information still matters heavily.

### Option B: one older-generation singles format
Best if you want a more stable ruleset, easier historical comparison, or closer alignment with recent singles-focused offline RL work.

Advantages:
- simpler mechanics in some generations,
- easier to freeze the ruleset and metagame,
- often cleaner for controlled experimentation.

Disadvantages:
- may have less directly relevant modern replay data,
- some generations have narrower communities or more specialized matchup knowledge.

### Recommendation
Make the project explicitly about **single battles only**.
Start with **one singles format first**, build the full data/training/evaluation pipeline around that format, and only consider any other format after the singles system is working well.
## 4. Project scope boundaries

To prevent scope explosion, define what is in scope for version 1.

### Version 1 should include
- one chosen format,
- local simulator integration,
- replay ingestion,
- state encoding,
- action masking,
- at least one heuristic bot,
- at least one behavior cloning model,
- at least one stronger transformer-based model,
- offline evaluation harness,
- cross-play evaluation against baselines,
- reproducible experiment tracking.

### Version 1 should not require
- perfect hidden-state inference,
- universal multi-format generalization,
- team generation,
- ladder deployment against humans,
- search-based hybridization,
- LLM integration,
- real-time production latency optimization.

Those are version 2 or version 3 goals.

## 5. Iterative compute-scaling plan for determining system requirements

Do **not** lock in final GPU requirements at the start.
Instead, treat compute planning as an empirical scaling exercise: begin with a tiny dataset on CPU, measure throughput and memory use, then iteratively scale data size and model size until you can estimate what GPU hardware is actually needed to train the full model within a reasonable amount of time.

The goal of this section is not to guess the final machine in advance.
The goal is to create a repeatable process for discovering the smallest hardware configuration that can train the target singles model fast enough to be practical.

### Core idea
For each model family you care about, measure:
- examples processed per second,
- tokens or turns processed per second if sequence-based,
- peak RAM usage on CPU runs,
- peak VRAM usage on GPU runs,
- wall-clock time per epoch,
- validation improvement per unit of compute,
- and checkpoint/evaluation turnaround time.

Then extrapolate from the measured scaling curve rather than guessing.

### Step 5.1: Start with a tiny CPU-only pipeline test
Use a very small slice of the singles dataset and a deliberately small model.

Suggested initial scale:
- 1,000 to 5,000 battles,
- short history windows,
- small batch size,
- compact MLP, GRU, or tiny transformer baseline,
- and one full train/validation/evaluation loop run entirely on CPU.

Purpose:
- verify the replay parser and tensorization,
- measure preprocessing bottlenecks,
- verify legality masking,
- estimate dataset caching needs,
- and identify whether data loading or model compute is the first bottleneck.

Deliverables:
- CPU throughput log,
- memory log,
- one sample checkpoint,
- and a timing breakdown for parsing, loading, training, and evaluation.

### Step 5.2: Create a scaling ladder of dataset sizes
Do not jump directly from tiny data to full-scale training.
Create a fixed ladder such as:
- 5k battles,
- 25k battles,
- 100k battles,
- 250k battles,
- 500k battles,
- 1M battles,
- and full target scale if larger.

At each rung, keep the evaluation slice fixed enough that runs remain comparable.

### Step 5.3: Create a scaling ladder of model sizes
For each dataset rung, test more than one model size.
For example:
- tiny baseline model,
- small production-like model,
- medium transformer,
- and target transformer.

Record for each run:
- parameter count,
- sequence length or history length,
- batch size,
- optimizer settings,
- gradient accumulation,
- epoch time,
- total train time to a chosen validation target,
- and memory usage.

### Step 5.4: Define “reasonable amount of time” before buying hardware
You need an explicit target for acceptable training speed.
For example:
- baseline BC experiments should finish within hours,
- medium transformer experiments should finish within about 1 to 2 days,
- and major ablations or offline RL runs should finish within a few days rather than multiple weeks.

Use your own research cadence, but make the target explicit.
Without that, you cannot decide whether a GPU setup is sufficient.

### Step 5.5: Run the first GPU comparison only after the CPU baseline is stable
Once the CPU pipeline works, repeat the same run on available GPU hardware.
Compare:
- CPU-only run,
- 1 consumer GPU run,
- 1 higher-memory GPU run if available,
- and multi-GPU only after single-GPU utilization is good.

What to measure:
- training speedup relative to CPU,
- VRAM ceiling,
- dataloader starvation or under-utilization,
- effect of mixed precision,
- and batch-size scaling.

### Step 5.6: Estimate full-training hardware from measured throughput
After several runs, estimate the hardware requirement using back-of-the-envelope extrapolation anchored in real measurements.

For each target model, estimate:
- total examples or turns in the full run,
- number of epochs or optimizer steps required,
- measured examples per second on candidate hardware,
- expected training time,
- expected evaluation overhead,
- and buffer for failed runs, logging, and checkpointing.

Then classify hardware candidates like this:

#### Candidate class A: CPU-only or weak GPU
Use only for:
- parser debugging,
- tiny dataset tests,
- unit tests,
- and very small baselines.

Reject this class for the main model if projected training time is too slow for iterative research.

#### Candidate class B: One strong consumer GPU
Use if:
- the main BC transformer can finish within your acceptable training window,
- ablations still fit into a manageable schedule,
- and VRAM is sufficient without extreme compromises.

This is often the first serious target to test.

#### Candidate class C: One higher-memory professional GPU or two strong GPUs
Use if:
- your target model does not fit comfortably on a single consumer card,
- sequence length or batch size is too constrained,
- or offline RL and synthetic-data runs become too slow.

#### Candidate class D: Multi-GPU workstation or cluster
Use only if the measurements show that:
- single-GPU iteration is too slow,
- the target model benefits materially from data/model parallelism,
- and you already have evidence that the pipeline scales efficiently enough to justify the complexity.

### Step 5.7: Use the same process for synthetic curriculum and offline RL
Do not assume that the compute profile of BC training will match later stages.
Repeat the scaling exercise for:
- synthetic tactical curriculum fine-tuning,
- offline RL training,
- and any later self-play or rollout generation.

Some stages may be data-loader bound; others may be simulator bound; others may be GPU bound.
Measure each stage separately.

### Step 5.8: Maintain a compute decision table
Create a table for every major training stage with columns such as:
- dataset size,
- model size,
- device,
- RAM usage,
- VRAM usage,
- examples per second,
- epoch time,
- validation score,
- projected full-run time,
- and decision.

The decision column should say things like:
- acceptable for debugging,
- acceptable for baseline experiments,
- acceptable for main training,
- or too slow / too memory-constrained.

### Step 5.9: Choose GPUs only after the scaling table is populated
After the above process, select the GPU target based on evidence.
For example:
- if one strong GPU can train the main singles BC model in an acceptable time, stay simple;
- if VRAM limits force crippling batch sizes or sequence lengths, move to a higher-memory GPU;
- if the project requires many parallel ablations or larger offline RL stages, consider 2 GPUs;
- and only move beyond that if measured speedups justify the added engineering complexity.

### Practical recommendation
The project should begin with:
- a CPU-only smoke test on tiny data,
- then progressively larger CPU and single-GPU runs,
- then a measured projection for full-scale singles training,
- and only then a final GPU procurement or hardware allocation decision.

This makes the system-requirements section evidence-based rather than speculative.
## 6. Training data requirements

The most important asset in this project is **battle trajectory data**.

### Core training data types

#### A. Replay logs
These are the foundation.

You want:
- battle replays or logs,
- turn-by-turn actions,
- observed battle state changes,
- result labels,
- format metadata,
- player ratings if available,
- timestamps if available,
- team information where available.

#### B. Parsed trajectories
Raw logs are not enough. You need a canonical dataset where each training example includes:

- observation at time `t`,
- legal action mask at time `t`,
- action taken,
- next observation,
- reward or terminal outcome,
- optional turn history prefix,
- optional per-player hidden-state estimates.

#### C. Auxiliary data
Helpful but not strictly required at the start:

- Pokédex and move databases,
- usage stats,
- moveset stats,
- teammate statistics,
- damage calculator outputs,
- team archetype labels,
- player Elo or Glicko proxies,
- matchup labels.

### Data quantity guidance

#### Prototype phase
- **50k to 200k battles** can be enough for pipeline debugging and early BC.

#### Serious baseline phase
- **200k to 1M battles** is a strong target for a credible behavior cloning system.

#### Scaled research phase
- **1M+ battles or multi-million turn trajectories** becomes very valuable for transformer scaling, offline RL, and opponent adaptation.

### Data quality guidance
Data quality matters more than raw volume at the beginning.

Prioritize:
- legal and correctly reconstructed transitions,
- one consistent simulator version,
- one consistent format,
- rating-filtered or quality-filtered games,
- removal of corrupted or incomplete logs,
- robust handling of edge cases.

### What to store for each example
A strong schema should include:

- format id,
- battle id,
- seed if reconstructable,
- turn index,
- side to act,
- encoded observation,
- public battle history summary,
- legal action ids,
- selected action id,
- outcome label,
- player rating metadata,
- team tokens if known,
- revealed-opponent-state features,
- mask for unknown information.

## 7. Data representation design

Your data schema is a make-or-break design decision.

### Recommendation: canonical structured state, not raw text first
Even though LLM-based Pokémon agents exist, the best recent fully trained systems tend to rely on **structured battle representations** rather than end-to-end chat fine-tuning.

Build a structured representation with these components:

1. **Per-Pokémon features**
   - species
   - typing
   - HP fraction
   - status
   - stat boosts
   - item known or unknown
   - ability known or unknown
   - moves revealed
   - tera state / relevant mechanic state if format supports it

2. **Per-side features**
   - remaining Pokémon count
   - hazards
   - weather / terrain / room effects
   - screens
   - side conditions
   - switch availability

3. **Global turn-context features**
   - turn number
   - last actions
   - speed order signals if known
   - game phase features
   - reveal depth

4. **Action features**
   - move id or switch target id
   - legality mask
   - optional learned action embeddings

### Sequence framing
The model should see **history**, not just the current frame.

You can store training inputs as:
- fixed-length history windows,
- whole-turn sequences,
- autoregressive token sequences,
- or transformer-friendly event sequences.

### Hidden information handling
Pokémon is partially observed.

For version 1, do not attempt perfect belief-state inference.

Instead:
- encode unknown values explicitly as unknown,
- include revealed-opponent-set features,
- include optional inferred priors as separate channels later.

## 8. Candidate model families

You should not commit to one architecture at the start. The project should compare at least four model families, all for singles play.


### Concise comparison of the most relevant model paths

| Model path | Best role in this project | Main advantages | Main drawbacks | Relative compute | Recommendation |
|---|---|---|---|---|---|
| Heuristic / scripted bot | Infrastructure baseline and regression test | Fast to build, debugs legality and simulator integration, gives a floor for evaluation | Very low ceiling, brittle, not a real competitive solution | Low | Mandatory baseline only |
| Simple behavior cloning model (MLP / GRU / small transformer) | First learned baseline | Stable training, cheap experiments, validates data pipeline, learns common play patterns | Copies average play, weak planning, limited robustness under distribution shift | Low to medium | Best first ML model |
| **Structured Pokémon battle transformer** | Mainline battle policy for singles | Strong sequence modeling, efficient legality masking, handles battle history, supports policy/value/opponent heads, cheaper than generic LLMs | Needs careful state schema and sequence design, more engineering than simple BC | Medium | **Recommended core architecture** |
| Offline RL on top of the battle transformer | Phase-2 improvement beyond imitation | Best path to exceed replay-only performance, can learn stronger long-horizon strategy from large corpora | Training instability, reward and dataset quality matter a lot, evaluation is harder | Medium to high | Recommended after BC transformer is stable |
| Limited self-play fine-tuning | Robustness and exploitability reduction | Covers rare states, can improve robustness and adaptation once core model works | Expensive, can overfit to self-generated meta loops, requires stronger evaluation discipline | High | Later-stage upgrade only |
| Search hybrid (policy + value + shallow search) | Inference-time tactical improvement | Sharpens candidate selection, adds tactical precision, can improve difficult turn choices | Adds latency and system complexity; hidden information limits exact search | Medium at inference | Good optional enhancement |
| Generic medium LLM fine-tuning on battle text | Research branch only | Flexible for explanations or mixed-text tasks | Pays an unnecessary LLM tax, slower, awkward legality handling, inefficient for core battle play | High | Not recommended as the primary path |

### Bottom-line comparison

For this project, the strongest cost-benefit sequence is:

1. heuristic baseline,
2. simple BC baseline,
3. **structured Pokémon battle transformer**,
4. offline RL on top of that transformer,
5. limited self-play,
6. optional shallow search.

That makes the **Pokémon battle transformer** the central comparison point: it preserves the sequence-model advantages you want, but is much more natural and efficient for battle states, legal actions, and offline RL than a generic LLM.


### Family 1: Heuristic / scripted baseline
Purpose:
- sanity check legality,
- provide a weak baseline,
- detect evaluation bugs.

Examples:
- random legal move bot,
- damage-maximizing move bot,
- simple type-advantage switch heuristic,
- scripted singles heuristic.

Why this matters:
If your ML model cannot beat these, the problem is likely in the pipeline rather than the model.

### Family 2: Behavior cloning classifier
Architecture options:
- MLP over flattened features,
- small transformer encoder over Pokémon slots,
- small GRU/LSTM over turn history.

Use case:
- first real learned baseline.

Strengths:
- stable,
- simple objective,
- easy debugging.

Weaknesses:
- copies average human play,
- poor out-of-distribution robustness,
- may not learn deep planning.

### Family 3: Transformer battle policy
Architecture options:
- encoder-only transformer over structured tokens,
- set transformer over team slots + battle context,
- autoregressive trajectory transformer,
- action-value transformer with policy and value heads.

Use case:
- mainline strong model.

Strengths:
- handles long context,
- good scaling behavior,
- aligns with recent successful work.

Weaknesses:
- more data hungry,
- more expensive,
- can overfit superficial replay regularities.

### Family 4: Offline RL model
Possible forms:
- decision-transformer-like policy,
- conservative offline RL over latent states,
- implicit Q-learning variant,
- transformer policy initialized by BC and improved with offline RL.

Use case:
- moving beyond imitation.

Strengths:
- can improve over demonstrators,
- leverages large replay corpora,
- aligns strongly with Metamon-style evidence.

Weaknesses:
- offline RL can be unstable,
- reward design and calibration matter,
- dataset quality becomes even more important.

### Family 5: Self-play fine-tuned policy
Possible forms:
- PPO fine-tuning from BC initialization,
- actor-critic fine-tuning,
- population training,
- PSRO / fictitious play style cycles, only after the singles pipeline is already strong.

Use case:
- robustness and reduced exploitability.

Strengths:
- can exceed demonstration quality,
- improves adaptation to strategic ecosystems,
- useful in fixed-format competitive settings.

Weaknesses:
- expensive,
- vulnerable to degenerate meta loops,
- harder to evaluate correctly.

### Family 6: Search hybrid
Possible forms:
- learned policy prior + shallow search,
- learned value head + beam search,
- minimax-style search with action proposal model,
- policy/value ensemble plus damage calculator.

Use case:
- later-stage quality improvement.

Strengths:
- often boosts tactical sharpness,
- can recover from policy uncertainty,
- more interpretable during evaluation.

Weaknesses:
- latency,
- implementation complexity,
- hidden information complicates exact search.

### Family 7: LLM-assisted hybrid
Possible forms:
- LLM for explanation or candidate proposal only,
- LLM for opponent set inference,
- LLM for action reranking,
- LLM-guided search.

Recommendation:
Do **not** make this the primary architecture at the start. Treat it as an optional research branch after the structured pipeline works.

## 9. Recommended primary architecture path

If I were choosing one default path for a practical ML battle project, I would choose this sequence:

1. **Structured state encoder**
2. **Transformer behavior cloning policy**
3. **Policy + value heads**
4. **Offline RL fine-tuning**
5. **Small-scale self-play fine-tuning**
6. **Optional shallow search at inference**

Why this path is strong:
- it starts simple enough to debug,
- it matches recent strong evidence from Pokémon battle research,
- it does not rely on expensive frontier chat models,
- it can scale upward without discarding earlier work.

## 10. Step-by-step implementation plan

## Phase 0: Research and planning

### Step 0.1: Choose the target format
Deliverable:
- one-page design memo naming the exact format, ruleset version, battle source, and initial evaluation targets.

Questions to settle:
- which singles format and ruleset are active?
- generation?
- open team sheets or hidden information?
- ladder-like environment or fixed-matchup benchmark?

### Step 0.2: Define success metrics
Deliverable:
- evaluation spec.

Minimum metrics:
- win rate versus random legal bot,
- win rate versus heuristic bot,
- cross-play matrix against ablations,
- held-out matchup generalization,
- calibration of action probabilities,
- optional ladder evaluation later.

### Step 0.3: Freeze version-1 scope
Deliverable:
- versioned scope file.

Explicitly state what is deferred.

## Phase 1: Infrastructure and environment

### Step 1.1: Stand up a local simulator environment
Deliverable:
- reproducible local battle server and Python interface.

Tasks:
- install Pokémon Showdown locally,
- verify scripted self-play matches can run,
- log request JSON and action strings,
- ensure reproducibility for offline evaluation.

### Step 1.2: Create a battle harness
Deliverable:
- framework that can pit any bot against any bot.

Requirements:
- standard bot API,
- logging hooks,
- deterministic seeds where possible,
- support for tournament or round-robin evaluation,
- replay export and metrics collection.

### Step 1.3: Implement legality and action encoding
Deliverable:
- canonical action vocabulary.

Tasks:
- enumerate all legal moves and switches,
- encode special mechanics consistently,
- create legality mask generation,
- unit test every edge case.

### Step 1.4: Build experiment tracking
Deliverable:
- run registry and metric logger.

Use:
- checkpoint naming conventions,
- exact config capture,
- train/validation/test split records,
- seed logging,
- evaluation summaries.

## Phase 2: Data pipeline

### Step 2.1: Collect replays or logs
Deliverable:
- raw replay archive.

Tasks:
- identify legal and ethical data sources,
- collect one format only,
- store raw logs immutably,
- keep metadata tables separate from parsed trajectories.

### Step 2.2: Write a replay parser
Deliverable:
- parser that converts logs into event streams.

Tasks:
- parse turn events,
- parse side conditions,
- parse revealed moves/items/abilities,
- parse win/loss labels,
- identify malformed or incomplete logs.

### Step 2.3: Build state reconstruction
Deliverable:
- canonical observation builder.

Tasks:
- reconstruct public battle state turn by turn,
- align actions with observations,
- attach legality masks,
- verify parser-state consistency against simulator outputs.

### Step 2.4: Create train/validation/test splits
Deliverable:
- reproducible dataset partitions.

Best practices:
- split by battle rather than turn,
- preserve held-out time slices if possible,
- include held-out team archetypes or matchup families,
- avoid leakage from near-duplicate battles.

### Step 2.5: Add quality filters
Deliverable:
- cleaned dataset.

Filters may include:
- minimum player rating,
- log completeness,
- length sanity checks,
- no corrupted transitions,
- consistent ruleset version.

## Phase 3: Baselines

### Step 3.1: Random legal bot
Deliverable:
- legal weak baseline.

Purpose:
- test harness and serve as floor performance.

### Step 3.2: Heuristic bot
Deliverable:
- stronger scripted bot.

Possible heuristics:
- maximize expected damage,
- avoid obvious type disadvantage,
- prefer KO lines,
- switch out of trapped losing positions only when legal.

### Step 3.3: Small MLP or GRU behavior cloning baseline
Deliverable:
- first learned model.

Tasks:
- build simple structured features,
- train with masked cross-entropy,
- evaluate action prediction accuracy,
- evaluate win rate versus scripted bots.

Exit criterion for phase 3:
- learned baseline beats random legal bot comfortably,
- system trains without data corruption or legality bugs.

## Phase 4: Main transformer BC model

### Step 4.1: Design the transformer input format
Deliverable:
- model spec.

Recommended design:
- tokens for each Pokémon slot,
- tokens for side/global state,
- optional short event-history tokens,
- learned embeddings for action candidates.

### Step 4.2: Train a medium-size transformer BC model
Deliverable:
- mainline BC checkpoint.

Training objective:
- masked next-action prediction.

Recommended outputs:
- policy head,
- optional value head,
- optional auxiliary heads such as KO prediction or opponent-item revelation probability.

### Step 4.3: Run systematic ablations
Deliverable:
- ablation table.

Ablate:
- history length,
- rating threshold,
- input features,
- model size,
- loss weighting,
- unknown-state handling.

### Step 4.4: Calibrate the policy
Deliverable:
- calibration report.

Why:
- action confidence affects search and offline RL later.


### Step 4.5: Synthetic tactical curriculum and pre-flight validation
Deliverable:
- synthetic scenario generator,
- held-out tactical benchmark suite,
- post-BC calibration checkpoint.

Purpose:
- verify that the model understands core battle mechanics before offline RL and self-play,
- reduce embarrassing tactical mistakes that replay imitation often leaves behind,
- improve coverage of rare but important states,
- and create a controlled benchmark for “basic competence” that is separate from ladder-style evaluation.

This stage should **not** assume that every battle state has exactly one objectively correct move.
Instead, it should construct a large bank of **carefully labeled tactical and mechanics-grounding scenarios** with one of three target types:

1. **Hard-label scenarios**
   - use only when a position is genuinely forced or near-forced.
   - examples: only one legal switch, guaranteed KO with no meaningful downside, forced sack or forced switch cases, and clearly dominant immediate tactical lines.

2. **Acceptable-action-set scenarios**
   - use when a small set of actions are all strategically reasonable.
   - examples: two similar attacks that both preserve a winning line, two safe switch-ins, or multiple equivalent endgame conversions.

3. **Opponent-distribution scenarios**
   - use when the goal is to predict the opponent’s likely move rather than a single hidden “correct” intention.
   - label these as probability targets or top-k targets rather than one forced answer.

Recommended scope for the synthetic curriculum:
- legality and mechanics checks,
- one-turn tactical puzzles,
- two-turn tactical setups,
- speed-order and damage-threshold scenarios,
- status, hazards, weather, and terrain interactions,
- forced-switch and sack-decision spots,
- opponent-action prediction in constrained states,
- and simple endgame conversion positions in singles.

#### Algorithmic plan for generating the synthetic data

The synthetic set should be generated by a **scenario factory** rather than by manually writing puzzle positions.
The factory should create diverse positions by sampling from real replay states, perturbing them in controlled ways, and only labeling positions that pass strict quality tests.

##### A. Build scenarios from real state anchors
Start with real battle states extracted from high-quality replay data rather than generating everything from scratch.
For each anchor state, store:
- public battle state,
- legal action mask,
- recent action history,
- inferred format metadata,
- and downstream outcome information from the replay.

Why this matters:
- real anchors reduce the risk of unrealistic board states,
- preserve natural team compositions and move combinations,
- and make synthetic examples closer to the distribution the model will actually face.

##### B. Apply controlled perturbations
For each anchor state, generate variants by applying limited edits such as:
- changing HP ranges within narrow buckets,
- changing status states where legal and mechanically plausible,
- toggling hazards, weather, terrain, or screens,
- replacing one revealed move with another common legal move from a plausible set,
- varying speed control or boost stages,
- and swapping one bench option for another usage-plausible alternative.

Perturbation rules:
- never create impossible combinations,
- never violate format legality,
- do not modify so many variables that the state becomes out-of-distribution,
- and tag every perturbation so later analysis can detect which edits produce unstable labels.

##### C. Use a labeling engine with explicit objectives
Labels should come from a transparent evaluation pipeline, not from intuition alone.
For each scenario, compute labels under a defined objective such as:
- highest estimated win probability over a short rollout horizon,
- safest move against an opponent action distribution,
- best move under a minimax or worst-case assumption,
- or best immediate tactical conversion.

Possible labeling sources:
- short simulator rollouts from the current best policy population,
- value-head estimates combined with legality-aware candidate search,
- expert-scripted tactical solvers for narrow mechanics cases,
- and consensus labels where multiple policies or evaluators agree.

Preferred rule:
- only assign a **hard single label** when the margin over the next-best action is clearly large and stable across evaluators.

##### D. Require label stability before inclusion
A scenario should enter the training set only if its label is stable.
Measure stability by checking whether the same preferred action or acceptable action set survives across:
- multiple opponent-policy assumptions,
- multiple random seeds or rollout samples,
- small perturbations of HP or damage rolls,
- and more than one evaluator or checkpoint.

If the “best” action changes easily, do **not** force a single answer.
Instead:
- convert the scenario into an acceptable-action-set label,
- convert it into a probability target,
- or discard it from the fine-tuning set and keep it only as an ambiguity test case.

##### E. Balance the dataset by concept, not only by raw count
Do not let the generator flood the dataset with repetitive “obvious KO” examples.
Set quotas by concept family, for example:
- legality and mechanics,
- forced switches,
- status interactions,
- speed control,
- endgames,
- switching-position and tempo management in singles,
- opponent prediction,
- and ambiguous-but-safe decisions.

Also balance across:
- early-, mid-, and late-game states,
- team archetypes,
- common and rare matchup structures,
- and rating tiers or source quality bands.

##### F. Deduplicate by battle semantics, not just exact state string
Prevent the model from memorizing templates.
Use semantic deduplication rules such as:
- clustering near-identical positions,
- removing repeated states that differ only in cosmetic identifiers,
- limiting per-template counts,
- and capping the contribution of any one replay, team core, or generated scenario family.

##### G. Separate training, validation, and benchmark generators
Do not generate one giant pool and split it randomly.
Instead, split at the level of:
- source replay clusters,
- matchup families,
- team archetypes,
- and scenario templates.

This helps prevent leakage where the model sees nearly identical states in both training and evaluation.

Recommended split:
- **training synthetic set** for curriculum fine-tuning,
- **validation synthetic set** for early stopping and ablations,
- **held-out benchmark suite** for final pre-flight testing only.

#### Anti-overfitting and anti-simplistic-overlearning safeguards

To avoid teaching brittle “puzzle-solving” habits, apply the following safeguards.

##### 1. Keep the synthetic stage smaller than the real replay corpus
The synthetic curriculum should sharpen fundamentals, not replace real play data.
A good default is to mix it as a minority share of post-BC training batches or use it in a short calibration phase.

##### 2. Mix synthetic and real states during fine-tuning
Do not run a long pure-synthetic training block.
Instead use interleaved batches such as:
- mostly real replay states,
- some synthetic tactical states,
- and a small amount of opponent-prediction supervision.

This preserves distributional grounding.

##### 3. Prefer soft targets when uncertainty is real
Use:
- label smoothing,
- acceptable-action-set losses,
- ranking losses over top actions,
- and KL-style losses for opponent distributions,
whenever the state is not truly forced.

##### 4. Train for invariance to harmless perturbations
For near-identical states with the same tactical meaning, encourage similar outputs.
This can be done with consistency losses across small HP, roll, or metadata perturbations.
The goal is to reduce memorization of brittle thresholds that do not generalize.

##### 5. Penalize overconfidence on ambiguous states
Track calibration on held-out ambiguous scenarios.
If the model becomes more certain while not becoming more accurate, reduce the weight of hard-label synthetic losses.

##### 6. Audit concept-level generalization
Evaluate by concept family, not just aggregate accuracy.
The model should generalize to unseen hazard states, unseen endgame motifs, unseen speed-control structures, and unseen singles switching patterns.
If gains appear only on seen templates, the generator is too narrow.

##### 7. Rotate scenario templates over time
Version the generator and periodically refresh the training pool.
Avoid letting the same handful of templates dominate every run.
This makes it harder for the model to overfit to generator quirks.

##### 8. Keep ambiguous cases in evaluation even if excluded from training
Some states are strategically debatable.
Those should still appear in the benchmark suite with annotations such as:
- acceptable action set,
- tactical risk class,
- and disagreement rate across evaluators.

That gives a more honest picture of whether the model is becoming stronger or merely more rigid.

#### Recommended training use

Use the synthetic curriculum in three ways:

1. **Post-BC calibration step**
   - run a short fine-tuning stage after the main BC model converges.
   - objective: tighten mechanics, legality, and common tactical competence.

2. **Auxiliary multitask supervision**
   - train policy, opponent-prediction, and optional value heads jointly on selected synthetic batches.
   - objective: improve internal battle understanding rather than only top-1 move prediction.

3. **Pre-flight benchmark gate**
   - require the checkpoint to pass a held-out tactical and mechanics suite before promoting it to offline RL or self-play evaluation.
   - objective: catch preventable weaknesses early.

#### Exit criteria for this stage
- no major legality failures on the held-out suite,
- strong accuracy on truly forced mechanics and tactical cases,
- acceptable calibration on ambiguous states,
- measurable reduction in simple blunders against scripted baselines,
- and no sign that ladder-like validation performance degraded because of synthetic overfitting.

## Phase 5: Offline RL

### Step 5.1: Initialize from the best BC checkpoint
Deliverable:
- BC-initialized RL run.

Rationale:
- safer than training RL from scratch.

### Step 5.2: Define reward structure carefully
Deliverable:
- reward spec.

Core recommendation:
- terminal win/loss reward should dominate.

Possible shaping terms:
- faint advantage,
- HP differential,
- board control proxies,
- strategic objective proxies.

Caution:
- shaping can cause weird incentives, so keep it conservative.

### Step 5.3: Train offline RL variants
Deliverable:
- comparison of at least two offline RL approaches.

Possible candidates:
- BC + value head only,
- BC + conservative offline RL,
- BC + sequence-model RL fine-tuning,
- BC + implicit Q-learning style variant.

### Step 5.4: Evaluate improvement over BC
Deliverable:
- offline RL vs BC report.

Measure:
- win rate versus heuristic baselines,
- held-out matchup performance,
- robustness to minor meta shifts,
- calibration degradation or improvement.

## Phase 6: Synthetic self-play and online fine-tuning

### Step 6.1: Generate self-play data from BC/RL policies
Deliverable:
- synthetic replay corpus.

Use cases:
- improve rare-state coverage,
- stress-test matchup behavior,
- create curriculum data.

### Step 6.2: Fine-tune with self-play
Deliverable:
- self-play-enhanced checkpoint.

Possible methods:
- PPO from policy initialization,
- actor-critic fine-tuning,
- offline fine-tuning on self-play trajectories.

### Step 6.3: Prevent degenerate meta collapse
Deliverable:
- diversity and exploitability diagnostics.

Methods:
- keep policy population snapshots,
- evaluate cross-play rather than only self-play reward,
- monitor collapse into narrow strategies,
- inject held-out opponents into evaluation.

## Phase 7: Population and game-theoretic extensions

### Step 7.1: Maintain a policy population
Deliverable:
- versioned set of policies.

### Step 7.2: Run cross-play evaluation matrices
Deliverable:
- policy-vs-policy matrix.

### Step 7.3: Add fictitious play / PSRO experiments
Deliverable:
- population-training experiment branch.

Only do this after single-policy training is stable.

## Phase 8: Optional inference-time upgrades

### Step 8.1: Add a value head or win-probability head
Deliverable:
- policy/value joint model.

### Step 8.2: Add shallow search
Deliverable:
- hybrid inference mode.

Good early version:
- use policy to propose top-k actions,
- use shallow rollout or heuristic/value reranking,
- do not try perfect full-depth search initially.

### Step 8.3: Add opponent set prediction
Deliverable:
- opponent belief module.

This can improve action quality without redesigning the whole policy.

## Phase 9: Optional team-building and moveset modeling

This phase is a separate subproject. Do not block battle-policy work on it.

### Step 9.1: Train set and moveset priors
Deliverable:
- conditional set recommender.

### Step 9.2: Train team generator
Deliverable:
- team prior model.

Potential model types:
- autoregressive transformer over team tokens,
- set transformer with masked completion,
- retrieval-based archetype model.

### Step 9.3: Couple team generation with battle policy evaluation
Deliverable:
- outer-loop team optimization framework.

Methods:
- evolutionary search,
- Bayesian optimization,
- quality-diversity search,
- matchup portfolio generation.

## 11. Detailed model architecture possibilities

## A. Simple baseline architecture

### Input
- current structured state,
- small history window.

### Encoder
- MLP or GRU.

### Output
- masked action logits.

### Why use it
- fast debug baseline,
- good for testing feature usefulness.

## B. Set-transformer battle encoder

### Input
- team slots for both sides,
- global state token,
- short event-history summary.

### Encoder
- permutation-aware set transformer or attention pooling over Pokémon slots.

### Output
- action logits,
- optional value estimate.

### Why use it
- natural fit for sets of Pokémon,
- interpretable attention over board pieces.

## C. Trajectory transformer

### Input
- event or state-action sequence prefix.

### Encoder/decoder
- transformer over temporal sequence.

### Output
- next-action distribution,
- optional return or win-probability head.

### Why use it
- strong long-context handling,
- closely aligned with recent sequence-model RL trends.

## D. BC-initialized offline RL transformer

### Input
- same as trajectory transformer.

### Training path
- supervised BC pretraining,
- offline RL objective,
- optional synthetic fine-tuning.

### Why use it
- strongest all-around research candidate.

## E. Policy + search hybrid

### Components
- learned policy prior,
- learned or heuristic value,
- top-k action proposal,
- shallow search or rollout reranking.

### Why use it
- can materially improve tactical precision.

## F. LLM-augmented hybrid

### Components
- structured core battle policy,
- optional text-based reasoning layer,
- optional retrieval from move/item/type knowledge,
- optional search assistant.

### Why use it
- interesting research branch,
- useful for explanations or candidate proposals.

### Why not lead with it
- more moving parts,
- harder to make reproducible,
- often slower and more expensive than structured models.

## 12. Training strategy possibilities

You should compare several training regimes rather than assume one is best.

### Regime 1: Pure imitation learning
Best for:
- first strong baseline.

Expected result:
- stable but limited ceiling.

### Regime 2: Imitation learning + quality filtering
Best for:
- improving action quality by learning from higher-rated play.

Expected result:
- often stronger than unfiltered BC.

### Regime 3: BC + offline RL
Best for:
- improving beyond average demonstrators.

Expected result:
- likely the best default research path.

### Regime 4: BC + self-play fine-tuning
Best for:
- closing the gap to stronger strategic adaptation.

Expected result:
- stronger, but more compute intensive and evaluation-sensitive.

### Regime 5: Population training
Best for:
- robustness and reduced exploitability.

Expected result:
- especially useful in stable fixed formats, but expensive.

### Regime 6: Search-only improvement on top of learned policy
Best for:
- sharpening tactics without retraining the full system.

Expected result:
- meaningful tactical gains if the search and value components are reliable.

## 13. Evaluation plan

You should treat evaluation as its own engineering subsystem.

### Core evaluation buckets

#### A. Offline imitation metrics
- top-1 action accuracy,
- top-k action accuracy,
- negative log likelihood,
- calibration error.

#### B. Bot-vs-bot performance
- win rate vs random legal bot,
- win rate vs heuristic bot,
- win rate vs previous checkpoints.

#### C. Cross-play generalization
- policy-vs-policy matrices,
- held-out matchup testing,
- held-out team family testing.

#### D. Robustness tests
- noisy hidden-state information,
- altered reveal order,
- rule variations inside the same format family,
- time-budget limits if relevant.

#### E. Human evaluation later
- anonymous ladder or controlled human scrimmages,
- careful logging,
- conservative claims.

### Minimum benchmark standard for claiming success
By the end of version 1, your main model should:

1. clearly beat random and simple heuristic bots,
2. beat the small BC baseline,
3. show stable generalization on held-out test slices,
4. remain legal and robust across edge cases,
5. produce reproducible results across seeds.

## 14. Data engineering requirements

This project needs real data engineering, not just model code.

### Essential requirements
- immutable raw logs,
- reproducible parser versions,
- versioned processed datasets,
- dataset cards with schema documentation,
- cached tensor pipeline,
- corrupted-example quarantine,
- reprocessing scripts,
- train/val/test manifest files.

### Helpful enhancements
- feature-store style cache,
- battle-level metadata DB,
- experiment registry,
- replay viewer for debugging failed cases.

## 15. Software stack recommendation

Recommended default stack:

- **Python** for orchestration and modeling,
- **PyTorch** for training,
- **Pokémon Showdown** local server for simulation,
- **a Python environment wrapper** for bot interaction,
- **Weights & Biases or equivalent** for experiment tracking,
- **Parquet / Arrow / binary tensor shards** for processed datasets,
- **Hydra or structured config system** for experiment configs,
- **pytest** for data/parser/unit tests.

## 16. Risks and failure modes

### Risk 1: replay reconstruction errors
Mitigation:
- build aggressive validation checks,
- compare reconstructed states against simulator transitions,
- inspect random battle slices manually.

### Risk 2: action-space bugs
Mitigation:
- legality-mask tests,
- unit tests for every special mechanic and switch condition,
- compare chosen action strings to simulator acceptance.

### Risk 3: data leakage
Mitigation:
- split carefully by battle and archetype,
- deduplicate near-identical logs,
- freeze test sets early.

### Risk 4: imitation ceiling
Mitigation:
- plan for offline RL and self-play from the start.

### Risk 5: self-play overfitting
Mitigation:
- use held-out external baselines,
- track exploitability and cross-play,
- preserve older population checkpoints.

### Risk 6: overly ambitious architecture branching
Mitigation:
- choose one mainline path,
- keep other branches as optional experiments.

## 17. Suggested milestone schedule

### Milestone 1: Infrastructure complete
Target outcome:
- local simulator and bot harness working,
- random and heuristic baselines running,
- replay parser producing valid trajectories.

### Milestone 2: First learned agent
Target outcome:
- small BC model trains and beats weak baselines.

### Milestone 3: Main transformer BC agent
Target outcome:
- medium transformer outperforms simpler BC and shows stable calibration.

### Milestone 4: Offline RL improvement
Target outcome:
- BC-initialized offline RL beats BC on meaningful evaluation suites.

### Milestone 5: Self-play or population extension
Target outcome:
- stronger robustness and matchup handling.

### Milestone 6: Optional hybridization
Target outcome:
- value head, shallow search, or opponent-set prediction improves tactical play.

## 18. Recommended order of experiments

To stay efficient, run experiments in this exact order:

1. random legal bot,
2. heuristic bot,
3. MLP or GRU BC,
4. medium transformer BC,
5. transformer BC with better history,
6. transformer BC with rating filtering,
7. transformer BC + value head,
8. offline RL from best BC,
9. synthetic self-play fine-tuning,
10. shallow search hybrid,
11. opponent set prediction,
12. population methods,
13. team generation and optimization.

Do not skip the early steps.

## 19. Resource budgeting guidance

### Compute budget priorities
If resources are limited, prioritize spending them in this order:

1. correct data pipeline,
2. enough storage and RAM for dataset processing,
3. one empirically justified GPU target for transformer BC after scaling tests,
4. evaluation automation,
5. only then broader RL scaling.

### Time budget priorities
If researcher time is limited, prioritize:

1. legality correctness,
2. replay reconstruction,
3. baseline evaluation harness,
4. transformer BC,
5. offline RL only after those are stable.

## 20. What not to do early

Avoid these mistakes:

- trying to support every format at once,
- starting with an LLM-only system,
- skipping weak baselines,
- using unreproducible scraped data without schema control,
- claiming human-level play before rigorous external evaluation,
- building team generation before the battle policy works,
- running expensive self-play before your offline baseline is strong.

## 21. Final recommended project blueprint

If you want one concise final blueprint for a singles-only project, it is this:

### Stage 1
Build a local Pokémon battle harness with legal action masking and replay parsing.

### Stage 2
Create a cleaned structured trajectory dataset for one chosen format.

### Stage 3
Train and evaluate weak scripted baselines plus a small BC model.

### Stage 4
Train a medium transformer behavior cloning model over structured battle sequences.

### Stage 5
Add value prediction and offline RL fine-tuning.

### Stage 6
Add synthetic self-play or light online fine-tuning for robustness.

### Stage 7
Only after the above works, explore shallow search, opponent belief modules, and optional team-building models.

## 22. Concrete deliverables checklist

By the end of the project’s first serious phase, you should have:

- a reproducible environment setup guide,
- a local simulator harness,
- a replay parser,
- a processed trajectory dataset,
- dataset documentation,
- random and heuristic bots,
- a small BC baseline,
- a transformer BC baseline,
- an offline RL experiment branch,
- evaluation scripts and cross-play tables,
- ablation reports,
- model cards and checkpoint documentation,
- a clear next-step roadmap.

## 23. Notes from the referenced recent systems

These are the main practical lessons the recent public systems suggest:

### From Metamon
- Large-scale replay-derived trajectory data matters.
- A progression from behavior cloning to offline RL to synthetic fine-tuning is promising.
- Large sequence models can be strong even without inference-time search.

### From VGC-Bench
- Standardized evaluation and cross-play matter.
- Matchup dependence is severe, especially when matchup dependence is severe.
- Population and game-theoretic training methods are worth considering, but only after simpler baselines are in place.

### From LLM-agent systems like PokéLLMon and PokéChamp
- LLMs can be useful, especially with retrieval or search.
- But the strongest reproducible trained battle systems do not necessarily depend on fine-tuning a chat LLM.
- A structured ML pipeline is still the most robust foundation for a serious battle-model project.

## 24. Recommended default decision

If you want one default decision set without revisiting every choice:

- **Format**: one singles format only for version 1
- **State representation**: structured tensors plus short history
- **Baseline model**: small BC classifier
- **Main model**: transformer BC policy with optional value head
- **Improvement path**: offline RL, then synthetic self-play
- **Evaluation**: cross-play plus held-out test slices
- **Compute planning**: start with CPU smoke tests, then scale dataset and model size empirically to determine the GPU setup needed for full training
- **Optional later branch**: shallow search and opponent-set prediction

That path gives the best balance of feasibility, technical depth, reproducibility, and alignment with the strongest recent Pokémon battle-model work.

## 25. Reference links

- Hugging Face model card: `cameronangliss/vgc-bench-models`
- Hugging Face model card: `jakegrigsby/metamon`
- Associated paper for VGC-Bench: `VGC-Bench: Towards Mastering Diverse Team Strategies in Competitive Pokémon`
- Associated paper for Metamon: `Human-Level Competitive Pokémon via Scalable Offline Reinforcement Learning with Transformers`
- Context files used: `recent_pokemon_battle_ai_models_2023_onward.md`, `pokemon_offline_rl_paper_summary.md`, `Pokemon-bot-LLM.md`
