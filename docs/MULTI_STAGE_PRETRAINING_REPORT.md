# Multi-Stage Pretraining Pipeline: Research Report & Implementation Plan

## Executive Summary

This report evaluates a proposed 3-stage pretraining pipeline for the P8-Lean BattleTransformer (~1.95M params) trained on Gen 3 OU (ADV) Pokemon Showdown replays via behavior cloning. The pipeline progressively increases data quality (Elo distribution) and context window size across stages:

| Stage | Battles | Elo Distribution | Max Window | Purpose |
|-------|---------|-------------------|------------|---------|
| 1 | 60,000 | All Elos (proportional) | 2 | Broad pattern learning |
| 2 | 20,000 | Overweight >1300 Elo | 5 | Quality refinement + context extension |
| 3 | 20,000 | Overweight >1500 Elo | 5 | Expert-level specialization |

**Bottom line**: The pipeline is well-motivated by recent research and should modestly improve model quality over single-stage training. However, the gains are likely to be incremental (1-5% accuracy), not transformative, and the pipeline introduces significant complexity. The honest assessment is that it's a reasonable experiment worth running, but should not be expected to produce dramatic improvements for a ~2M parameter behavior cloning model.

---

## 1. Research Findings

### 1.1 Multi-Stage Pretraining: The Emerging Standard

Multi-stage pretraining has become the dominant paradigm for training language models. OLMo 2 (Allen AI, 2024) trains in two stages: a long initial phase on broad web data (~90% of FLOPs), followed by a shorter mid-training phase (~10% of FLOPs) on upsampled high-quality data. OLMo 3 extends this to three stages: broad pretraining → focused mid-training → long-context extension. Phi-4 (Microsoft) and DBRX (Databricks) also use staged data curricula.

**Key finding**: The "broad then curated" pattern is well-established and consistently outperforms monolithic training on a single data mix, primarily in large-scale settings (>1B parameters, >100B tokens).

**Caveat for this project**: These results come from models 100-1000x larger than P8-Lean, training on datasets 1000x+ larger. The literature on curriculum learning for small models (<10M params) on small datasets (<100K examples) is sparse. The closest relevant finding: curriculum effects are actually *more* pronounced for smaller models (up to 160M params) according to "Curriculum Learning for LLM Pretraining: An Analysis of Learning Dynamics" (Jan 2025), which showed that random ordering produces higher gradient noise in smaller models.

### 1.2 Context Window Extension

The BattleTransformer uses **sinusoidal positional encodings** with a pre-computed buffer of 100 positions (`TurnPositionEmbedding`, `battle_transformer.py:430`). This is excellent news for context extension:

- **Sinusoidal encodings generalize to unseen positions** without additional training. Unlike learned positional embeddings, they produce meaningful position representations for any length up to the buffer size.
- Going from window 2 → 5 is trivial from a positional encoding perspective. No position interpolation, NTK-aware scaling, or other tricks (commonly needed for extending decoder-only LLM context lengths from 4K to 128K) are necessary here.
- The model already has `max_seq_len=5` in its P8-Lean config, so the architecture is designed for window 5.

**Compute cost**: The transformer encoder uses full self-attention over all tokens in the window. With 14 tokens per turn step:
- Window 2: 28 tokens → attention matrix is 28×28 = 784 elements
- Window 5: 70 tokens → attention matrix is 70×70 = 4,900 elements

This is a ~6.25x increase in attention compute per example. For a 3-layer model with 224 hidden dim, this is still very cheap in absolute terms but does increase per-example training time.

**Memory**: Window 5 examples also require ~2.5x more memory per batch element (more turn history stored). The collate function (`collate_windowed`) right-pads to the max window in each batch, so batches with mixed window sizes will have some padding waste.

**Best practice for extension**: Research on context extension (primarily from the LLM literature) suggests that training on short contexts first, then fine-tuning on longer contexts, is effective and sometimes preferred over training on long contexts from scratch. The intuition: the model first learns local patterns (which move to make given the current state), then learns to leverage history (how earlier turns inform the current decision).

### 1.3 Warm-Starting Between Stages

**Learning rate**: The consensus from recent research (NVIDIA NeMo documentation; "Simple and Scalable Strategies to Continually Pre-train LLMs", ICLR) is to **re-warm the learning rate** at each stage transition. This means resetting the LR schedule with a fresh warmup phase, allowing the model to adapt to the new data distribution. The existing `WarmupCosineScheduler` in `train_phase4.py` supports this naturally — each stage gets its own warmup + cosine decay.

**Optimizer state**: "Reuse, Don't Retrain: A Recipe for Continued Pretraining of Language Models" (arXiv 2407.07263) found **negligible difference** between keeping or resetting the AdamW optimizer state during continual pretraining, even over 300B tokens. For our much smaller training runs (~100K examples per stage), this is even less likely to matter. **Recommendation**: Reset the optimizer state for simplicity — it avoids potential issues with stale momentum estimates from a different data distribution, and the cost is near zero.

**Model weights**: Always warm-start from the previous stage's best checkpoint. This is the whole point of staged training — each stage builds on the representations learned in the prior stage.

### 1.4 Catastrophic Forgetting

This is the primary risk of staged training. When the model trains on increasingly curated data, it may "forget" useful patterns from the broader dataset.

**Relevant findings**:
- "Efficient Continual Pre-training by Mitigating the Stability Gap" (arXiv 2406.14833) documents a "stability gap" where model performance first drops then gradually recovers during continual pretraining with new data.
- Experience replay (mixing in data from previous stages) is the most effective mitigation (arXiv 2402.08096).
- Mechanistic analysis shows forgetting is concentrated in query/key attention projections (67% of these weights show negative gradient alignment during domain shift).
- Larger pre-trained models are more resistant to forgetting, but P8-Lean at ~1.95M params is small.

**Risk assessment for this pipeline**: The forgetting risk is **moderate to high** because:
1. The model is small (~1.95M params), so it has limited capacity to retain broad patterns while specializing.
2. The data distribution shift between stages is significant (all Elos → >1300 → >1500).
3. The later stages use much less data (20K vs 60K battles).

**Mitigations to consider**:
- Mix 10-20% of Stage 1 data into Stages 2 and 3 (experience replay).
- Use a lower learning rate in later stages (e.g., halve the peak LR at each stage).
- Monitor Stage 1 validation loss throughout later stages to detect forgetting.

### 1.5 Elo-Based Curriculum Learning for Game AI

The research on Elo-based curriculum learning is primarily in the context of **reinforcement learning** and **self-play**, not behavior cloning. The most relevant work:

- Pommerman multi-agent training (arXiv 2407.00662) uses Elo-based matchmaking for progressive difficulty scaling during self-play.
- AlphaStar's league training creates an implicit curriculum by training against progressively stronger opponents.
- ELO-Rated Sequence Rewards (arXiv 2409.03301) uses Elo ratings as reward signals for trajectory quality.

**Critical difference**: These systems train agents via RL where the agent learns from its own experience against progressively stronger opponents. This project uses **behavior cloning** — the model learns to imitate human decisions. The curriculum is over the *quality of the demonstrations*, not the difficulty of the opponent.

**Does Elo-based data curation help for behavior cloning?** The logic is:
- Higher-Elo players make better decisions on average → the model learns better strategies.
- Lower-Elo games may have noisier/suboptimal action labels → learning from noise first might hurt.
- Counter-argument: Lower-Elo games may contain more diverse situations (unusual team compositions, different strategies) that improve generalization.

This is an open question with no definitive answer in the literature for behavior cloning specifically.

### 1.6 Does Staged Pretraining Actually Help vs. Single-Stage?

**Honest assessment**: The evidence is mixed, especially for small models.

**In favor**:
- Curriculum learning reduces gradient noise for models up to 160M params (Curriculum Learning for LLM Pretraining, 2025).
- Staged approaches achieve competitive or better results at lower total compute in LLM pretraining.
- Training on short sequences first, then extending, is a well-established practice.

**Against**:
- "How Learning Rate Decay Wastes Your Best Data in Curriculum-Based LLM Pretraining" (Nov 2025) shows that curriculum advantages **largely disappear** under standard cosine LR decay schedules. The benefit is real only with constant LR or carefully co-designed LR/data schedules.
- "Beyond Random Sampling" (June 2025) found that fine-grained difficulty groupings provide little advantage over a simpler 3-tier curriculum — suggesting that the specific Elo thresholds matter less than the broad structure.
- For small datasets (< ~10MB of text), training from scratch can actually outperform continued pretraining from a pre-trained checkpoint.
- The P8-Lean model has ~1.95M params. With 60K+20K+20K = 100K battles averaging ~40 turns each, you have ~4M training examples. This is likely more than enough data to train a 2M-param model in a single pass without curriculum.

**The uncomfortable truth**: For a ~2M parameter model on ~100K battles, the most important factors are probably:
1. Data quality (Elo filtering)
2. Model architecture and hyperparameters
3. Proper regularization (dropout, weight decay)

...rather than the curriculum ordering. Staged pretraining is most impactful when data is abundant and diverse enough that the model can't absorb it all equally well in a single pass.

---

## 2. Implementation Plan

### 2.1 Data Pipeline Changes

The current `download_replays.py` script already supports `--elo-threshold` filtering. Each stage needs its own data download and processing:

**Stage 1**: 60,000 battles, all Elos
```bash
python scripts/download_replays.py --sample-size 60000 --elo-threshold 0 \
    --output-dir data/raw_stage1
python scripts/process_dataset.py --input-dir data/raw_stage1 \
    --output-dir data/processed_stage1 --max-turns 20
```

**Stage 2**: 20,000 battles, overweight >1300 Elo

This requires modifying `download_replays.py` to support weighted sampling by Elo bracket, or downloading more battles and then subsampling with weights. The simplest approach: download battles above and below the threshold separately, then merge with desired proportions. For example, 70% of battles from >1300 Elo, 30% from all Elos.

**Stage 3**: 20,000 battles, overweight >1500 Elo

Same approach, but with higher threshold. **Data availability concern**: The current dataset shows only ~1,190 battles at 1500+ Elo out of 10K total. To get 20K battles overweighted toward >1500 Elo, you need either:
- Many more total battles from the Metamon dataset (if available)
- Heavy oversampling of the ~1,190 high-Elo battles (risking overfitting)
- A lower Elo threshold than 1500

This is a real constraint that needs investigation before committing to the pipeline.

### 2.2 Training Script Changes

Create a new `scripts/train_multistage.py` that orchestrates the three stages:

```python
# Pseudocode for the multi-stage pipeline

# Stage 1: Broad pretraining, window=2
train_phase4.py \
    --data-dir data/processed_stage1 \
    --max-window 2 \
    --epochs 20 \
    --lr 1e-4 \
    --warmup-steps 500 \
    --checkpoint-dir checkpoints/multistage/stage1

# Stage 2: Load Stage 1 best checkpoint, window=5, lower LR
train_phase4.py \
    --data-dir data/processed_stage2 \
    --max-window 5 \
    --epochs 15 \
    --lr 5e-5 \          # Reduced peak LR
    --warmup-steps 200 \
    --resume-from checkpoints/multistage/stage1/best_model.pt \
    --checkpoint-dir checkpoints/multistage/stage2

# Stage 3: Load Stage 2 best checkpoint, window=5, even lower LR
train_phase4.py \
    --data-dir data/processed_stage3 \
    --max-window 5 \
    --epochs 15 \
    --lr 2e-5 \          # Further reduced
    --warmup-steps 100 \
    --resume-from checkpoints/multistage/stage2/best_model.pt \
    --checkpoint-dir checkpoints/multistage/stage3
```

**Key modifications needed in `train_phase4.py`**:
1. Add `--resume-from` flag to load model weights from a previous checkpoint.
2. The existing `save_checkpoint` already saves `model_state_dict` — just need to add loading logic.
3. Reset optimizer and scheduler (create fresh ones for each stage).
4. Optionally add `--replay-data-dir` and `--replay-ratio` for experience replay (mixing previous stage data).

### 2.3 Hyperparameter Recommendations

| Parameter | Stage 1 | Stage 2 | Stage 3 |
|-----------|---------|---------|---------|
| Data | 60K battles, all Elo | 20K, overweight >1300 | 20K, overweight >1500 |
| Max window | 2 | 5 | 5 |
| Peak LR | 1e-4 | 5e-5 | 2e-5 |
| Warmup steps | 500 | 200 | 100 |
| Epochs | 20-25 | 15-20 | 10-15 |
| Patience | 7 | 5 | 5 |
| Batch size | 64 | 64 | 64 |
| Dropout | 0.1 | 0.1 | 0.15 |
| Weight decay | 0.01 | 0.01 | 0.02 |
| Replay ratio | — | 0.1 (10% Stage 1 data) | 0.1 (10% Stage 1+2 data) |

**Rationale**:
- Decreasing LR across stages prevents catastrophic forgetting and allows fine-grained adaptation.
- Shorter warmup in later stages since the model is already well-initialized.
- Higher dropout/weight decay in Stage 3 to prevent overfitting on the smaller, more homogeneous dataset.
- Experience replay (10% of prior stage data mixed in) is the most effective forgetting mitigation per the literature.

---

## 3. Costs and Benefits: Honest Assessment

### 3.1 Benefits

1. **Broader initial representation**: Stage 1 with 60K battles across all Elos gives the model exposure to diverse team compositions, unusual strategies, and edge cases it wouldn't see in a curated-only dataset. This is the strongest argument for the pipeline.

2. **Context window curriculum**: Training on window=2 first means the model learns to make good decisions from minimal history. Extending to window=5 then teaches it to incorporate longer-term context. This is a sound training principle — learn local patterns first, then global.

3. **Progressive quality focus**: By the final stage, the model specializes on high-Elo play where decisions are more strategic and consistent. This *should* improve the quality of the policy at test time (assuming test-time evaluation against reasonable opponents).

4. **More total data**: 100K battles vs. 10K is a 10x increase in training data. Even without curriculum effects, this alone is likely the biggest driver of improvement. **This benefit could be achieved with single-stage training on 100K battles too.**

### 3.2 Costs

1. **Pipeline complexity**: Three stages means three data processing runs, three training runs, and more hyperparameters to tune (per-stage LR, warmup, epochs, data mixing ratios). Debugging is harder. Reproducibility is harder.

2. **Total compute**: Roughly 2-3x the wall-clock time of single-stage training on the same total data, because:
   - Stage 1 uses 60K battles (largest single stage)
   - Stages 2 and 3 add 40K more
   - Each stage has warmup overhead
   - Window=5 examples are more expensive than window=2

3. **Data availability risk**: The Metamon Gen 3 OU dataset may not have enough high-Elo battles. The current 10K dataset has only ~370 battles at 1600+ Elo. To get 20K battles "overweighted" above 1500, you either need a much larger raw pool or accept heavy oversampling.

4. **Catastrophic forgetting risk**: Each stage transition risks degrading what was learned before. For a small model, this is a real concern that requires careful monitoring and mitigation (replay, lower LR).

5. **Marginal returns on curriculum**: The literature suggests curriculum learning effects are modest (1-3% accuracy improvement) and can be negated by standard LR decay schedules. For a ~2M param model on ~100K battles, the curriculum may not matter much.

6. **Opportunity cost**: The time spent implementing and debugging a 3-stage pipeline could be spent on:
   - Architectural improvements (better attention patterns, action embeddings)
   - More data (just get 100K battles at 1300+ Elo and train single-stage)
   - Synthetic data generation (Phase 5)
   - Online evaluation against bots

### 3.3 Risk Matrix

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Insufficient high-Elo data | High | Medium | Lower Elo thresholds or accept oversampling |
| Catastrophic forgetting at stage transitions | Medium | High | Experience replay, lower LR, monitoring |
| No improvement over single-stage | Medium | Low | At worst, you have a well-trained model on more data |
| LR schedule negates curriculum | Medium | Medium | Use constant LR within stages or co-design schedule |
| Overfitting on small Stage 3 dataset | Medium | Medium | Higher dropout, replay, fewer epochs |
| Pipeline bugs / reproducibility issues | Medium | Medium | Extensive logging, checkpoint saving, seed fixing |

---

## 4. Alternative Approaches to Consider

Before committing to the full 3-stage pipeline, consider these simpler alternatives that capture most of the benefit:

### 4.1 Single-Stage with More Data (Recommended Baseline)
Train on 100K battles at 1300+ Elo with window=5 in a single stage. This gets you the data volume benefit without the curriculum complexity. Compare against this baseline to isolate the curriculum effect.

### 4.2 Two-Stage Only
Skip Stage 3. Train Stage 1 (60K all Elo, window=2) then Stage 2 (40K battles >1300 Elo, window=5). This simplifies the pipeline while keeping the context extension benefit and the broad-to-curated curriculum.

### 4.3 Single-Stage with Elo Weighting
Instead of staged training, use a single stage with sample-level Elo weighting: higher-Elo battles contribute more to the loss (e.g., weight = 1.0 + 0.5 * (elo - 1300) / 500 for battles above 1300). This captures the quality curriculum without separate stages.

### 4.4 Warmup Window Curriculum (Within Single Stage)
Start training with window=2 for the first 30% of epochs, then switch to window=5 for the remaining 70%. No separate data stages needed. This captures the context extension benefit within a single run.

---

## 5. Concrete Recommendation

**If the goal is maximum model quality**: Run the full 3-stage pipeline, but **also train a single-stage baseline** on the same total data. Compare results to verify the curriculum adds value beyond just more data.

**If the goal is practical iteration speed**: Use Alternative 4.2 (two-stage) or 4.4 (within-stage window warmup). These capture the most important benefits (context extension, data volume) with much less complexity.

**If data availability is limited**: Use Alternative 4.3 (single-stage with Elo weighting). This sidesteps the data scarcity problem for high-Elo battles entirely.

**What I would do**: Start with Alternative 4.1 (single-stage baseline on 100K battles) to establish a strong baseline, then run the 3-stage pipeline and compare. If the 3-stage pipeline shows >2% accuracy improvement on the test set, it's worth the complexity. If not, stick with the simpler approach.

---

## 6. References

- "Curriculum Learning for LLM Pretraining: An Analysis of Learning Dynamics" (arXiv 2601.21698, Jan 2025)
- "How Learning Rate Decay Wastes Your Best Data in Curriculum-Based LLM Pretraining" (arXiv 2511.18903, Nov 2025)
- "Beyond Random Sampling: Efficient Language Model Pretraining via Curriculum Learning" (arXiv 2506.11300, June 2025)
- "Reuse, Don't Retrain: A Recipe for Continued Pretraining of Language Models" (arXiv 2407.07263, July 2024)
- "Simple and Scalable Strategies to Continually Pre-train LLMs" (ICLR 2024)
- "Efficient Continual Pre-training by Mitigating the Stability Gap" (arXiv 2406.14833, June 2024)
- "An Efficient Rehearsal Scheme for Catastrophic Forgetting Mitigation during Multi-stage Fine-tuning" (arXiv 2402.08096, Feb 2024)
- "STEP: Staged Parameter-Efficient Pre-training for Large Language Models" (NAACL 2025)
- "Curriculum-Guided Layer Scaling for Language Model Pretraining" (arXiv 2506.11389, 2025)
- "Multi-Agent Training for Pommerman: Curriculum Learning and Population-based Self-Play Approach" (arXiv 2407.00662, 2024)
- "ELO-Rated Sequence Rewards: Advancing Reinforcement Learning Models" (arXiv 2409.03301, 2024)
- OLMo 2 (arXiv 2501.00656, 2025)
- OLMo 3 (Allen AI blog, 2025)
- NVIDIA NeMo: Reset Learning Rate documentation
- "Data Mixing Laws: Optimizing Data Mixtures" (ICLR 2025)
