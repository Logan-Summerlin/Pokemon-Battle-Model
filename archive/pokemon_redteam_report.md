# Pokémon Battle Model Project: Partial Observability, Red-Team Critique, and a Single Recommended Path

_Last updated: March 9, 2026_

## Executive summary

The project plan has a real strength: it knows most of the right ideas. Its central weakness is that it still behaves like a **menu of plausible research branches** instead of a **single disciplined build order**. The biggest technical reason that matters is partial observability. In competitive Pokémon, the agent never acts in a fully known world. Even when team preview exists, the policy still does not know the opponent’s exact moves, items, EVs, abilities, Tera choice, or how the opponent will trade uncertainty for tempo. That means the project cannot safely treat replay imitation, offline RL, search, or synthetic tactics as plug-and-play improvements. Each method behaves differently once hidden information is central.

The practical conclusion of this review is blunt:

**Do not start with a broad “Pokémon ML platform.” Start with one frozen environment, one observation model, one architecture, one training target, and one benchmark regime.**

My recommended single path is:

1. **Freeze the environment to Gen 9 OU singles on Pokémon Showdown, closed-team-sheet, with standard team preview, using a fixed metagame snapshot and replay date window.**
2. **Represent uncertainty explicitly**: public state + known preview species + unknown latent set fields + lightweight prior features from usage statistics. Do not attempt full belief-state search in v1.
3. **Train one main model only**: a structured candidate-action-scoring transformer, trained first with behavior cloning on high-quality human replays.
4. **Add one auxiliary prediction task only**: opponent set / hidden-info inference, because partial observability is the main strategic bottleneck.
5. **Add one narrow synthetic fine-tuning stage after BC only**: use replay-anchored synthetic states as a short repair and calibration pass for mechanics, tactics, and ambiguity handling, then delay offline RL and search until that combined system proves real gains on a brutal evaluation harness.

That path is not the most ambitious. It is the one most likely to survive contact with reality.

---

## 1. Partial observability is not a side issue; it is the core game design constraint

Competitive Pokémon is a partially observable stochastic game. Recent Pokémon AI work explicitly frames singles as a setting with imperfect information, opponent modeling, stochasticity, and long planning horizons. Metamon, for example, emphasizes reconstructing the agent’s first-person partially observed perspective from spectator logs rather than training on omniscient state traces, and studies its results in older generations that it calls the “most partially observed” settings. That is the right instinct: if the training view leaks too much hidden information, the model will learn a game that humans do not actually play. citeturn1view1turn1view4

For this project, the practical lesson is simple: **state design is not merely a feature-engineering step. It defines the task itself.** If the observation model is wrong, then evaluation, imitation targets, offline RL updates, and search enhancements all become misleading.

### What is hidden in Pokémon battles?

Even in modern singles with team preview, the agent usually does **not** know the opponent’s:

- exact moveset,
- exact item,
- exact ability when multiple are possible,
- EV and IV spread,
- speed tier relative to small benchmarks,
- chosen Tera type until revealed or inferred,
- switching thresholds and sacrifice preferences,
- long-horizon plan,
- bluffing behavior.

In older generations without team preview, the hidden-information burden is even worse because the unrevealed backline is unknown until shown in battle. Metamon’s framing of older generations as especially partially observed is therefore not just academic language; it has direct implications for project scope and difficulty. citeturn1view4

### Why this changes the ML problem

In many board-game-style projects, the observation is almost the state. In Pokémon, the observation is only a **thin slice** of the state. That creates four distinct ML problems that the project plan currently blends together too loosely:

1. **Action selection under uncertainty**: what should I do now, given hidden information?
2. **Belief update**: what do I think the opponent likely is carrying?
3. **Opponent modeling**: what kind of player / style is this opponent likely to be?
4. **Robust decision-making**: what move has the best downside protection across multiple plausible hidden states?

A plan that says “we will do behavior cloning, then offline RL, then maybe search” without sharply specifying which of those four tasks it is solving is underspecified.

---

## 2. Team preview changed the information structure of Pokémon, but it did not remove partial observability

The user’s example is correct: **team preview changed over time**.

Bulbapedia notes that **team previews were added to Link Battles in Generation V**. In other words, the standard modern expectation that both players see the opponent’s six species before turn 1 is not timeless; it is a historical rules change. citeturn1view0

Smogon’s discussions and tutorials make clear what that means competitively. Team Preview lets each player see the opponent’s whole roster before choosing a lead, and it became one of the defining strategic steps in modern play. But those same discussions also make clear that Team Preview does **not** reveal sets. It changes the game from “unknown roster plus unknown sets” to “known roster but still largely unknown configuration.” citeturn1view2turn1view3

That distinction matters enormously for project design.

### Information regimes the plan should distinguish explicitly

The current project plan discusses hidden information, but it does not separate the different information regimes cleanly enough. It should. At minimum there are four:

#### Regime A: Pre-Gen-5 style no team preview
- Opponent backline is hidden until revealed.
- Opponent sets are hidden.
- Lead choice includes much more roster uncertainty.
- Search becomes much less reliable.
- Replay imitation becomes more confounded because observed actions often respond to unseen roster possibilities.

#### Regime B: Modern closed-team-sheet singles with team preview
- Opponent six species are known at preview.
- Sets, items, abilities, EVs, Tera, and move choices remain hidden.
- Strategic planning improves, but robust play under uncertainty is still central.
- This is the most practical modern setting for a first project because the uncertainty is severe but not maximal.

#### Regime C: Open-team-sheet formats
Official VGC rules documents for the 2025 and 2026 seasons specify **open team list** formats. That greatly reduces hidden information compared with standard closed-sheet ladder play. citeturn2search1turn2search2

- Species, items, moves, abilities, and more may be disclosed.
- The game becomes much closer to a tactical optimization problem.
- Search and policy learning get easier.
- But this is not the same task as modern Smogon-style ladder singles.

#### Regime D: Spectator-log reconstruction vs true first-person observation
- Showdown replay logs may contain information in an order or form different from what the acting player knew at the time.
- Reconstruction mistakes can accidentally leak hidden state.
- Any training set built from replays needs a principled first-person reconstruction pipeline.

Metamon stresses this reconstruction issue explicitly, which is a warning the project plan should treat as first-order rather than optional engineering detail. citeturn1view1turn1view4

### Bottom line on team preview

Team Preview did **not** make Pokémon fully observable. It merely changed the partial observability profile.

That means the project should not ask, “Does the model have team preview?” It should ask:

- Which hidden variables remain after preview?
- Which of those hidden variables matter most for decisions?
- Which of those hidden variables can be inferred from replay history, metagame priors, or turn-level evidence?
- Which ones should remain explicitly unresolved and be handled by robust policies rather than guessed exactly?

That is the right framing.

---

## 3. How partial observability sharpens the harsh criticism of the current plan

The earlier harsh criticism becomes stronger, not weaker, once partial observability is taken seriously.

### Criticism 1: The plan is too broad

This fault becomes more serious under hidden information. A broad project menu might be survivable in a mostly observable domain. In Pokémon it is dangerous, because every extra branch interacts with uncertainty differently:

- behavior cloning tends to average across hidden-state contingencies,
- offline RL can mis-credit returns because counterfactual hidden states are unobserved,
- synthetic curricula often simplify away the real uncertainty,
- and search can look much stronger than it really is if the belief model is unrealistic.

So breadth is not just a scheduling problem. It is a task-definition problem.

### Criticism 2: The plan is still too weak on partial observability

This remains the single biggest technical flaw.

The plan says version 1 should not require “perfect hidden-state inference,” which is fair, but that is not an actual design choice. The real decision must be between:

- **public-state-only policy**,
- **public state plus metagame priors**,
- **public state plus explicit latent set prediction**,
- **belief-state modeling over plausible opponent sets**,
- or **search over sampled hidden-state hypotheses**.

At the moment the plan gestures toward several of these without committing. That is exactly the kind of ambiguity that later produces a huge amount of engineering churn.

### Criticism 3: Replay imitation is even noisier than it first appears

The project plan correctly treats behavior cloning as the first learned baseline. That is fine. But under partial observability, BC is noisier than the plan admits.

A human move in a replay may be “correct” under one inferred set and mediocre under another. The replay alone rarely tells you the actor’s exact belief state. So BC is not learning “the best move”; it is learning a mixture of:

- move quality,
- player priors,
- metagame habits,
- stylistic preferences,
- and hidden-state guesses.

That does not make BC useless. It means BC should be treated as **policy initialization under uncertainty**, not as ground truth supervision for optimal play.

### Criticism 4: The evaluation plan is not hard enough

This becomes even more important once the game is recognized as partially observed. Weak evaluation lets the model exploit shortcuts such as:

- overreliance on common usage priors,
- brittle assumptions about standard sets,
- strong play only against common archetypes,
- or confidence without true hidden-state calibration.

A serious benchmark needs to ask not only “did the model win?” but also:

- did it handle hidden-info ambiguity well?
- was it robust to uncommon but legal sets?
- did it preserve outs under uncertainty?
- did it overcommit to one hidden-state guess too early?

### Criticism 5: The synthetic tactical curriculum is especially risky

This section of the project plan is the clearest example of an appealing idea that may become counterproductive.

Synthetic tactical puzzles can help teach mechanics, damage thresholds, or forced tactical patterns. But Pokémon losses often come from bad uncertainty management rather than bad arithmetic. If the synthetic distribution mostly contains states with one clearly correct move, then it will systematically under-train the exact thing that makes good Pokémon play hard: acting well when multiple hidden-state hypotheses are still live.

In short, synthetic curriculum is most likely to teach the model to solve the wrong version of the game.

### Criticism 6: Offline RL is high-risk because the hidden state is not logged cleanly

Offline RL already struggles with support mismatch and biased datasets. In Pokémon it inherits an additional problem: the logged action came from a human acting on a hidden belief state that is only partly recoverable from the replay.

So offline RL can easily reward the wrong causal explanation of a good outcome. A move that worked because the opponent happened not to have a coverage move may be misread as generally strong. Unless the observation reconstruction and evaluation regime are excellent, offline RL can produce a policy that is more confident but not actually stronger.

### Criticism 7: Search is not automatically a fix

Search is tempting because Pokémon looks like a turn-based tactics game. But hidden information means that exact search over the true state is unavailable. A search layer is only as good as its candidate hidden-state model or value approximation. In a closed-sheet environment, search can still help, but it is not a clean escape hatch from uncertainty. That makes it a later-stage enhancement, not a first-resort solution.

---

## 4. The plan needs an explicit doctrine for handling hidden information

A good v1 plan should state a doctrine like this:

### Doctrine for version 1

1. **Never train on omniscient features that would not be available to the player at decision time.**
2. **Represent uncertainty explicitly instead of pretending it is solved.**
3. **Use metagame priors as soft hints, not as leaked truth.**
4. **Separate hidden-state inference from move selection, even if they share an encoder.**
5. **Evaluate rare-set robustness and ambiguity handling directly, not just aggregate win rate.**

That would convert partial observability from a vague caution into an actual engineering rulebook.

### Concretely, what should be in the observation?

For the most practical v1, the observation should include:

- public battlefield state,
- both revealed active Pokémon states,
- own full team state,
- opponent preview species identities,
- opponent revealed move, item, ability, and Tera evidence,
- turn history,
- legality mask,
- optional soft prior features from public usage statistics or replay-frequency priors,
- explicit “unknown” markers for unrevealed information.

What should **not** be in v1:

- exact hidden set labels taken from replay hindsight,
- full belief search over millions of set combinations,
- brittle handcrafted omniscient evaluators.

### One crucial addition the project plan should make

The model should have an **auxiliary hidden-info head**.

That head does not need to solve perfect belief inference. It only needs to predict a useful compressed approximation such as:

- likely item class,
- likely speed bucket,
- likely role archetype,
- likely Tera category,
- likely move-family presence.

This is much more practical than full exact set reconstruction, and far more aligned with how strong human players actually reason.

---

## 5. From a menu of options to one single optimal project path

The current project plan presents many plausible options. The problem is not that the options are bad. The problem is that they are not ranked by one ruthless decision rule.

The practical solution is to stop asking, “What are all the reasonable paths?” and instead ask, **“Which path maximizes expected progress per month under hidden information, data noise, and finite engineering bandwidth?”**

### A practical decision rule

Use a forced-choice scorecard with five criteria, each scored 1 to 5:

1. **Faithfulness to real play**: does the path match the actual information available in the target format?
2. **Implementation risk**: how likely is the path to collapse under parsing, action-space, or simulator complexity?
3. **Data efficiency**: how well can it learn from realistic replay corpora?
4. **Evaluation clarity**: can you clearly tell whether it improved?
5. **Upgrade path**: if it works, can it later support RL, search, or self-play without redesigning everything?

Then require every candidate project path to earn its place against those criteria.

### Applying that rule to the main choices

#### Choice 1: Which format?

- **Older gens without team preview**: strategically interesting, but worse for a first build because hidden-info burden is highest.
- **Modern Gen 9 OU singles with closed team sheets**: still hard, but easier than no-preview generations because roster uncertainty is reduced while preserving real competitive ambiguity.
- **Open-team-sheet formats**: easier, but they solve a different task and risk teaching the wrong assumptions if your long-term target is ladder singles.

**Winner for v1:** Gen 9 OU singles, closed-team-sheet, fixed metagame snapshot.

Reason: it is modern, data-rich, strategically relevant, and difficult without being maximally pathological.

#### Choice 2: Which state design?

- **Public-state only**: simplest, but too blind.
- **Full belief-state modeling from day 1**: too ambitious and fragile.
- **Public state + soft prior features + explicit unknowns + auxiliary hidden-info head**: best balance.

**Winner for v1:** public state plus explicit uncertainty and a lightweight hidden-info auxiliary task.

Reason: this is the smallest design that actually respects the domain.

#### Choice 3: Which model family?

- **Small MLP / GRU baseline**: necessary as a sanity check, but too weak as the mainline.
- **Structured transformer over battle tokens and candidate actions**: strong enough, flexible enough, and aligned with replay-sequence learning.
- **Generic LLM fine-tuning**: unnecessary overhead for the core task.

**Winner for v1:** structured battle transformer with candidate-action scoring.

Reason: it captures sequence context without paying the generic-LLM tax.

#### Choice 4: Which training regime?

- **Behavior cloning only**: cleanest initial signal and cheapest iteration.
- **Offline RL immediately**: too much compounding uncertainty before the baseline is understood.
- **Synthetic curriculum before replay competence**: likely misallocates effort.

**Winner for v1:** behavior cloning first, with hidden-info auxiliary prediction, no offline RL until the benchmark proves the BC model is genuinely strong.

Reason: the first question is whether the model can imitate strong human decision patterns in the real information regime.

#### Choice 5: Which enhancements belong in v1?

- **Synthetic curriculum**: no.
- **Search**: no.
- **Opponent belief module as an auxiliary head**: yes.
- **Brutal evaluation harness**: yes.

**Winner for v1:** only the minimal additions that make the baseline faithful and measurable.

### The single recommended path

Here is the one path I would actually execute.

#### Phase A: Freeze the target task
- Format: **Gen 9 OU singles**.
- Environment: **Pokémon Showdown**.
- Information regime: **closed team sheet with standard team preview**.
- Data window: fixed replay date range chosen to reduce metagame drift.
- Goal: strong in-battle move selection, not team building.

#### Phase B: Build the observation and action interface
- Reconstruct first-person observations only.
- Encode public battlefield state and turn history.
- Include opponent preview species.
- Track revealed hidden-info evidence explicitly.
- Add unknown markers for unrevealed slots.
- Define one clean legal action space with move / switch / Tera combinations handled consistently.

#### Phase C: Build two baselines only
1. simple heuristic bot,
2. small BC baseline.

These exist only to validate legality, parsing, and evaluation.

#### Phase D: Train the main model
- Structured transformer encoder.
- Candidate-action scoring head.
- Main loss: action prediction from strong human replays.
- Auxiliary losses: hidden-info inference targets such as item class, speed bucket, Tera category, or role cluster when labels are inferable from replay outcome without leaking future decision-time truth.

#### Phase E: Run a narrow synthetic fine-tuning stage
Only after the first BC checkpoint is stable:
- audit BC failures on real evaluations,
- generate replay-anchored synthetic states for those failure clusters,
- use three label types only: forced-action labels, acceptable-action-set labels, and opponent-distribution labels,
- mix synthetic data as a minority share of batches,
- and keep this stage short and explicitly repair-oriented.

This stage exists to patch sparse mechanics holes, obvious tactical blunders, endgame conversion mistakes, and poor calibration in ambiguous states. It should not become a second pretraining regime.

#### Phase F: Evaluate brutally
Use a fixed benchmark suite:
- stratified team archetypes,
- held-out replay slices,
- common vs uncommon set distributions,
- mirror-team tests,
- benchmark scripted/search baselines,
- calibration of hidden-info predictions,
- synthetic gate tests for the targeted failure clusters,
- and win-rate plus decision-quality review on curated difficult states.

#### Phase G: Only then decide whether to continue
Promote to offline RL **only if** the BC-plus-synthetic system:
- clearly beats heuristic and scripted baselines,
- is robust across archetypes,
- shows useful hidden-info calibration,
- improves real battle play rather than only synthetic scores,
- and does not collapse on uncommon but legal sets.

If those conditions are not met, do not add RL. Fix the observation model, action model, or benchmark first.

---

## 6. Why this path is better than the other tempting options

### Why not start with older generations?

Because the no-team-preview regimes are more partially observed, not less. They are excellent research environments for later work, but worse for a first build unless your primary goal is specifically to study maximal hidden-information play. Metamon’s success there is impressive, but it should be read as evidence that these formats are hard, not as evidence that they are the easiest place to start. citeturn1view4

### Why not use open-team-sheet formats first?

Because they simplify the hidden-information problem so much that your resulting system may optimize for a different game. Open sheets are a legitimate target if that is your intended deployment environment, and official VGC does use them. But if your main target is closed-sheet singles play, then open-sheet pretraining is not the cleanest first objective. citeturn2search1turn2search2

### Why not jump directly to offline RL?

Because you do not yet know whether performance gains will reflect stronger play or just better exploitation of replay biases. Under partial observability, that ambiguity is especially dangerous.

### Why not make synthetic tactics the centerpiece?

Because Pokémon is not mainly lost on arithmetic; it is often lost on uncertainty management, long-horizon preservation, and sequencing discipline. Synthetic puzzles usually oversimplify all three.

The right role is narrower: use synthetic fine-tuning as a **post-BC repair layer**, not as the main training distribution. That means:
- start from real replay anchor states,
- create only small legal perturbations,
- prefer acceptable-action-set labels when the state is genuinely ambiguous,
- keep synthetic examples a minority share of training,
- and require improvement on real battle benchmarks before keeping the stage.

That preserves the good part of the idea without letting it distort the project.

### Why not add search early?

Because search in a hidden-information game is only as sound as the belief model under it. A shaky search layer can create an illusion of rigor while quietly baking in unrealistic assumptions.

---

## 7. Concrete rewrites the original project plan should make

If the original project plan is revised, the following changes would make it much stronger.

### Rewrite 1: Replace the broad model menu with a single mainline
Instead of describing many model families in near-parallel, say:

> Version 1 will target Gen 9 OU closed-team-sheet singles with team preview. The main model will be a structured transformer trained with behavior cloning on first-person reconstructed replay data, augmented with auxiliary hidden-information prediction. All other model families are deferred until this path passes a fixed evaluation threshold.

### Rewrite 2: Add an explicit information-regime section
The plan should state exactly which hidden variables are known at preview, which become known later, and which remain latent throughout action selection.

### Rewrite 3: Add a hidden-info doctrine
The plan should explicitly forbid hindsight leakage and define how latent opponent information is represented.

### Rewrite 4: Rewrite synthetic curriculum into a narrow version-1 repair stage
Do not cut it entirely. Instead, rewrite it as a short post-BC stage with tight constraints:
- replay-anchored state generation,
- minority synthetic batch share,
- forced-label states only when the answer is highly robust,
- acceptable-action-set labels for ambiguous states,
- and a promotion gate based on real battle metrics rather than synthetic benchmark gains.

### Rewrite 5: Demote offline RL from “recommended next step” to “conditional branch”
RL should only happen if the BC baseline clears hard metrics first.

### Rewrite 6: Make the evaluation section much harsher
Require robustness to uncommon sets, archetype diversity, and uncertainty calibration.

### Rewrite 7: Add kill criteria
For example:
- if BC plateaus without robust cross-archetype gains, stop scaling model size and revisit state design;
- if hidden-info auxiliary heads are uncalibrated, do not add search;
- if offline RL improves only in-distribution but worsens robustness, roll it back.


### Rewrite 8: Add a decision rule for collapsing the project menu into one path
The original plan offers too many plausible branches. A practical way to choose one path is to score each branch against a small set of criteria and then commit to the best total.

Use a simple weighted rubric:

- **Task fidelity**: does this branch match the actual closed-sheet singles game you want to solve?
- **Partial-observability realism**: does it respect hidden information without leaking future truth?
- **Engineering tractability**: can a small team really implement and debug it?
- **Data efficiency**: can it learn from the replay data you can actually trust?
- **Evaluation clarity**: can success and failure be measured cleanly?
- **Upgrade compatibility**: does it leave a clean path to later search or RL?

Under that rubric, the best version-1 path is:
1. closed-team-sheet Gen 9 OU singles with team preview,
2. first-person reconstructed public-state representation plus latent hidden-info slots,
3. one structured candidate-action-scoring transformer,
4. BC on strong replay data,
5. one short replay-anchored synthetic fine-tuning stage,
6. then a hard promotion gate before any RL or search.

This is the optimal path not because it is theoretically maximal, but because it best balances realism, tractability, and clean evaluation.

---

## 8. Final verdict

The original plan is thoughtful but over-complete. Its real flaw is not ignorance. It is insufficient commitment.

Pokémon is not merely a large action-space game. It is a **hidden-information decision problem with long-horizon planning, heavy metagame priors, and replay labels that are noisier than they first appear**. Once that is admitted, the project should become more conservative, not more expansive.

The practical answer is therefore:

- choose one exact format,
- choose one exact observation regime,
- choose one exact model,
- choose one exact training target,
- add only one bounded post-BC repair stage,
- choose one exact benchmark,
- and refuse to branch until the baseline survives a hard test.

That is how to turn a strong-looking project memo into an actual build order.

---

## Sources consulted

- Bulbapedia, **Generation V** — for the introduction of Team Preview to Link Battles. citeturn1view0
- Smogon Forums, **Team Preview** — for contemporary discussion of what Team Preview changed and what it did not reveal. citeturn1view2
- Smogon University, **Getting Started with Competitive Battling** — for the central strategic role of Team Preview in modern play. citeturn1view3
- Grigsby et al., **Human-Level Competitive Pokémon via Scalable Offline Reinforcement Learning with Transformers** — for first-person reconstruction, partial observability, and the importance of avoiding omniscient training views. citeturn1view1turn1view4
- Play! Pokémon VGC rules materials for 2025 and 2026 — for the use of open team lists in official VGC. citeturn2search1turn2search2
