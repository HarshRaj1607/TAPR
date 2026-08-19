# TAPR — Future Work

**Status: planned, not done.** Everything in this file is proposed work. It was split out of `RESEARCH_RECORD.md` so that document contains only work actually completed.

Nothing here has been run. No result in this file exists yet.

---


# PART VI: Path to Workshop Submission

This section contains the precise next experiments needed, in order of priority, following the research reframe. The goal is a mechanistic paper about how TAPR changes training dynamics, not a benchmark paper.

**The central research question for the paper:**
> Does transition-aware process reward shaping produce qualitatively different policy updates than outcome-only reward, and does this difference manifest in generalization behavior?

## 6.1 Experiment 1: Multi-Seed Replication of Cross-Task Transfer

**This is non-negotiable. Everything else depends on this result surviving.**

### What to run:

```
GSM8K training (400 steps each), Qwen2.5-3B + LoRA:

ORM baseline: seed {1, 2, 3}
TAPR:         seed {1, 2, 3}

Evaluate each checkpoint on:
- GSM8K test (1319 problems)
- StrategyQA (200 problems, same held-out split)
```

### How to implement:

Add `--seed` argument to train_grpo.py. Seeds should control: data shuffle, model init, window sampling, dropout. Pass `seed=N` to GRPOConfig.

```bash
# Run 6 sessions (can be parallelized 2 at a time on Colab Pro+)
python train_grpo.py --task gsm8k --mode baseline --seed 1 --output_dir ./gsm_baseline_s1
python train_grpo.py --task gsm8k --mode baseline --seed 2 --output_dir ./gsm_baseline_s2
python train_grpo.py --task gsm8k --mode baseline --seed 3 --output_dir ./gsm_baseline_s3
python train_grpo.py --task gsm8k --mode tapr --seed 1 --lambda_val 0.3 --output_dir ./gsm_tapr_s1
python train_grpo.py --task gsm8k --mode tapr --seed 2 --lambda_val 0.3 --output_dir ./gsm_tapr_s2
python train_grpo.py --task gsm8k --mode tapr --seed 3 --lambda_val 0.3 --output_dir ./gsm_tapr_s3
```

Then eval each on both GSM8K and StrategyQA.

### What you're looking for:

```
                GSM8K           StrategyQA
ORM             mean ± std      mean ± std
TAPR            mean ± std      mean ± std
```

**Dream result:**
- TAPR ≈ ORM on GSM8K (small, consistent difference)
- TAPR >> ORM on StrategyQA (consistent 3-5pp gap)

**If the StrategyQA gap survives 3 seeds:** Proceed to Experiments 2-4.
**If it doesn't:** The paper needs significant reframing — the cross-task result was noise.

### Why this is first:

Every other experiment builds on the claim that TAPR produces different generalization behavior. If that claim doesn't survive seeds, the paper's central contribution collapses.

## 6.2 Experiment 2: Policy Drift Analysis

**Purpose:** Distinguish between Story A (TAPR changes model less) and Story B (TAPR changes model differently).

### What to measure throughout training (every 50 steps):

1. **KL divergence:** KL(current policy || reference policy) — already logged by TRL
2. **LoRA weight norm:** `torch.norm(param)` for each LoRA matrix
3. **Response length:** Mean completion length in tokens
4. **Reasoning step count:** Mean number of numbered steps extracted
5. **Entropy:** Already logged by TRL
6. **frac_reward_zero_std:** Already logged

### How to log:

Add a callback to GRPOTrainer that runs every `log_steps` and writes these metrics to a JSON file alongside the TRL logs.

```python
class DriftCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        # compute and save LoRA norms, response length, etc.
```

### Interpretation:

**Outcome A:** ORM shows larger KL, larger LoRA norms, larger policy drift. TAPR shows smaller everything. Then TAPR acts as an implicit regularizer.

**Outcome B:** ORM and TAPR show similar KL and LoRA norms, but different response patterns (length, step count, entropy). Then TAPR changes the model differently, not less.

**Outcome B is the stronger paper.** It says the qualitative nature of the learned behavior differs, not just the amount.

## 6.3 Experiment 3: Reward Ablation

**Purpose:** Establish that the transition-awareness (not just any dense reward) causes the cross-task preservation.

### Three conditions:

```
1. ORM:              R = R_outcome
2. Step-wise PRM:    R = R_outcome + λ * mean(individual step scores)
3. TAPR:             R = R_outcome + λ * mean(transition window scores)
```

The key difference between 2 and 3: step-wise PRM scores each step independently ("is this step good?"); TAPR scores transitions ("does this step follow from the previous?").

### How to implement step-wise PRM:

Change the judge prompt to score individual steps without context from previous steps:

```
"Evaluate this single reasoning step:
[step text]
Is this step: (a) mathematically valid, (b) clear, (c) moving toward a solution?
SCORE: [0.0 to 1.0]"
```

Score each step independently, average for R_transition.

### What you're looking for:

```
                    GSM8K   StrategyQA
ORM                  X         55.5
Step-wise PRM         X         56-57?
TAPR                  X         60.0
```

If TAPR shows significantly better cross-task preservation than step-wise PRM, you have evidence that the *transition modeling specifically* (not just dense reward) drives the effect.

**This experiment directly answers: "Is TAPR's benefit due to the transition structure or just the density of the reward signal?"**

## 6.4 Experiment 4: Transfer Dataset Characterization

**Purpose:** Map where the TAPR advantage appears and disappears. Test one dataset between GSM8K and MATH in difficulty/distribution distance.

### Recommended dataset: SVAMP

- Elementary-level math word problems (similar to GSM8K)
- Different surface form and problem structure
- Known to challenge models that memorize GSM8K patterns
- Should show transfer if TAPR learns reasoning rather than GSM8K-specific patterns

```bash
# After training, add SVAMP to eval.py
# Dataset: https://huggingface.co/datasets/ChilleD/SVAMP
```

### The transfer matrix you're building:

| Train | Eval | Expected TAPR advantage | Rationale |
|-------|------|------------------------|-----------|
| GSM8K | GSM8K | Small | In-domain, both methods work |
| GSM8K | SVAMP | Medium | Related math, different form |
| GSM8K | StrategyQA | Large | Different task type |
| GSM8K | MATH-500 | None | Too far OOD |

**If this pattern holds:** You have a story about how the TAPR advantage scales with distribution distance. That's potentially the most interesting structural finding in the paper.

## 6.5 Paper Structure (Workshop Submission)

**Target venues:**
- ICLR 2026 Workshop (reasoning/RLHF focused)
- NeurIPS 2026 Workshop
- ACL Rolling Review → EMNLP 2026 findings

**Proposed title:** "Transition-Aware Rewards Reduce Degeneracy and Preserve Generalization in Policy Gradient Fine-Tuning of Language Models"

**Abstract sketch:**
> Outcome-only GRPO training of language models suffers from reward degeneracy: when all rollouts for a problem receive identical rewards, the advantage signal is zero and the gradient contributes nothing. We show that across two benchmarks, 52-62% of GRPO training groups exhibit this degeneracy. We introduce TAPR (Transition-Aware Process Reward), which augments outcome rewards with a live judge that scores reasoning step-to-step transitions. TAPR reduces reward degeneracy to near-zero across all training problems. Training with TAPR on GSM8K produces a policy that preserves substantially more StrategyQA performance than outcome-only training (4.5pp difference), despite similar in-domain accuracy. We characterize this as a generalization-preservation effect and provide mechanistic evidence through reward degeneracy analysis, policy drift measurements, and a transition-specific ablation.

**Section structure:**

```
1. Introduction
   - Reward sparsity problem in GRPO
   - Reward degeneracy as specific failure mode
   - TAPR as mechanism + key results preview

2. Background
   - GRPO formulation + degeneracy formalization
   - Process rewards: related work + distinction from TAPR
   - Why pre-computed PRMs don't solve degeneracy

3. Method
   - TAPR reward formulation
   - Judge design (dimensions, live scoring requirement)
   - Window sampling + batching optimization

4. Experimental Setup
   - Models, benchmarks, training config
   - Judge validation procedure + results

5. Results
   5.1 Reward Degeneracy (frac_reward_zero_std)
   5.2 In-domain Accuracy (GSM8K, StrategyQA)
   5.3 Cross-Task Transfer: GSM8K→StrategyQA
   5.4 Transfer Characterization (SVAMP, MATH-500)
   5.5 Ablation: ORM vs Step-wise PRM vs TAPR

6. Analysis
   6.1 Policy Drift (Story A vs Story B)
   6.2 Reward Degeneracy Mechanics
   6.3 Boundary of Transfer Effect

7. Limitations
   - 3B student; 400 steps
   - MATH judge failure
   - Single training domain
   - Eval set size (StrategyQA)

8. Conclusion
   - Reward degeneracy is a real, measurable problem
   - Transition awareness specifically addresses it
   - Produces different generalization behavior
   - Scales to 7B+32B (projected; future work)
```
