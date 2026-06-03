# Agentic AI Usage in TAPR

**Required doc per Samsung EnnovateX AX Hackathon submission guidelines.**

This document explains how TAPR uses open weight models and agentic development tools — what we built, how it works, what worked, and what didn't.

---

## Open Weight Models Used

### Qwen2.5-3B-Instruct — the student (trained)

The model being improved. Qwen2.5-3B-Instruct is a 3B parameter instruction-tuned model selected because:
- Strong GSM8K baseline (78% zero-shot) — enough headroom to show improvement
- Efficient fine-tuning with LoRA within 24GB VRAM
- Documented sensitivity to reward design, making it a meaningful testbed for TAPR

It is fine-tuned with GRPO using the TAPR reward signal across four training stages.

### Qwen2.5-7B-Instruct — the judge agent (agentic component)

The core agentic component of TAPR. This model acts as an autonomous evaluator in the reward pipeline — it is not used for generation or reference, only for judgment.

**How it acts as an agent:**
- **Perceives:** receives a math problem + a window of W consecutive reasoning steps
- **Reasons:** evaluates the window across 5 named dimensions simultaneously
- **Acts:** produces a structured output — a score from 0 to 1 and a one-line justification
- **Tool-chained:** its outputs are stored as a JSON lookup table and consumed by the training loop as a reward signal

This is a genuine agentic workflow: the judge makes independent assessments on thousands of reasoning chains before training begins. Its outputs directly shape which reasoning behaviors get reinforced.

---

## Agentic Workflows

### Workflow 1 — Offline Judge Precomputation

```
GSM8K training split
       ↓
For each problem: generate step-by-step solution format
       ↓
Sliding window sampling (W ~ {2,3,5}, stride=1)
       ↓
Qwen2.5-7B-Instruct judge (5 dimensions per window)
       ↓
Store scores as JSON lookup table
       ↓
Training loop reads scores at training time (zero inference cost)
```

This is a **precomputed tool-chaining pattern** — the agent (judge) runs once offline, its reasoning is persisted, and the downstream system (GRPO training loop) consumes it as a tool without re-invoking the agent. This keeps training fast while benefiting from rich agentic judgment.

### Workflow 2 — Reward Signal Construction

At each GRPO training step, the reward for a generated reasoning chain is assembled from two components:

```
Generated reasoning chain
       ↓
R_outcome: exact-match answer check (live, binary)
R_transition: lookup precomputed judge score for this problem
       ↓
R_final = R_outcome + λ · R_transition
       ↓
GRPO policy update
```

The agentic judgment (R_transition) augments the simple outcome signal with process-level information — the exact transition quality the judge evaluated during precomputation.

### Workflow 3 — Easy-to-Hard Curriculum via Self-Evaluation

Before training, the base model's own zero-shot performance on GSM8K is used as a difficulty signal to order training examples. Problems the model gets wrong = hard. Problems it gets right = easy. Training proceeds easy-to-hard.

This is a form of **self-evaluation-driven planning**: the model's own behavior shapes the training schedule. No external difficulty labels needed.

---

## Reasoning & Planning Pipelines

### The 5-Dimension Judge Prompt

The judge is not asked for a generic score — it is given a structured reasoning task across five explicit dimensions. Each dimension maps directly to an empirically documented SLM failure mode identified from real Qwen2.5-3B outputs:

```
1. LOCAL VALIDITY     → does step N+1 follow from step N?
2. VALUE CONSISTENCY  → are quantities traceable? (targets Type A: wrong quantity)
3. QUESTION COHERENCE → solving the right question? (targets Type B: semantic misread)
4. CONVERGENCE        → moving toward a solution? (targets Type D: directional drift)
5. COMPLETENESS       → no bare assertions? (targets Type C: collapsed reasoning)
```

The structured prompt forces the judge to reason across multiple failure modes simultaneously rather than giving a holistic impression score. This was a deliberate design choice — early attempts at a simpler single-score prompt produced weaker separation.

### Stochastic Window Sampling as Regularization

Window size W is sampled from {2, 3, 5} with P = (0.1, 0.5, 0.4). This is an **agentic regularization mechanism** — by varying the granularity of each judgment call, the training signal prevents the model from learning to look good at one fixed evaluation window. The distribution was empirically derived: W ≥ 3 achieves r = 0.690 vs r = 0.600 for mixed W, informing the shift in probability mass toward larger windows.

---

## Development Tools

### Claude Sonnet — Research and Development Partner

Used throughout the research process as a coding assistant and thinking partner:
- Literature review (curriculum ordering, SFT effects, GRPO stability)
- Signal design iteration (progress signal → sympy → unified LLM judge)
- Validation script development (flaw generation, scoring, correlation analysis)
- Results interpretation and baseline comparisons
- Blueprint drafting

Claude was used under a strict PI/executor protocol — every design decision was made by the PI (Harsh) after discussion; Claude executed and flagged issues but did not make unilateral calls. This structure was enforced throughout and corrected when violated.

---

## What Worked

**Unified LLM judge with structured dimensions (r = 0.600, 90.3% separation)**
The key insight was replacing hand-crafted signals with a path-agnostic LLM judge that evaluates *inter-step relationships* rather than individual steps in isolation. Structuring the judgment across 5 named dimensions — each empirically motivated — was essential. A generic single-score prompt performed worse.

**Offline precomputation**
Running the judge once before training begins eliminates inference cost during RL and makes the training loop fast. The static reward has a known limitation (distribution shift as the model evolves during RL), but the live outcome signal naturally compensates for this.

**Stochastic window sampling**
The variable-W approach improves over fixed windows. Empirically: W ≥ 3 separates better than W = 2. The stochastic distribution prevents granularity gaming.

**Empirical-first design**
Every design decision was validated before committing. Failure modes were identified from real Qwen2.5-3B outputs (11 wrong answers analyzed) rather than assumed from theory. This grounded the entire signal design in observed model behavior.

---

## What Didn't Work

### Progress Signal (r = −0.690, wrong direction)

**Design:** Measure whether step N+1 reduces problem complexity — fewer variables, fewer quantities. Formula: `0.6 × %reduction_variables + 0.4 × %reduction_quantities`.

**Result:** Strong signal in the *wrong* direction. Flawed solutions scored higher on progress than ground truth.

**Why it failed:** Progress rewards simplification regardless of whether the simplification was earned. Flawed solutions — especially Type B (semantic misread) and Type C (collapsed reasoning) — solve simpler versions of the problem or skip steps. Simpler problem = fewer variables = higher progress score. The signal measured how fast the chain simplifies, not whether the simplification was legitimate.

**Key insight from failure:** A real transition signal must check whether step N+1 *follows legitimately* from step N, not just whether it reduced structural complexity. This failure directly motivated the LLM judge redesign.

### Sympy Arithmetic Checking

**Design:** Check whether arithmetic in each step is internally consistent using symbolic computation.

**Result:** Sympy scored ground truth and flawed chains nearly identically (0.799 vs 0.800). Near-zero separation.

**Why it failed:** Sympy only catches arithmetic errors, not semantic ones. Real SLM failures (Types B, C, D) are semantically wrong but arithmetically valid. The signal was checking the wrong thing.

### Atomicity Scoring

**Design:** Keyword-based classifier checking whether each step performs exactly one atomic operation.

**Result:** No meaningful separation. Qwen's clean output style inflated atomicity scores for flawed chains.

**Why it failed:** Measured formatting compliance, not reasoning quality. The model's tendency to produce clean one-operation-per-line outputs meant flawed and correct chains were indistinguishable on this signal.

---

## Honest Limitations

1. **Distribution shift:** The judge is precomputed on training examples using reference solutions. As the model's generated chains shift during RL, the static judge scores may become less calibrated. Mitigated by the live outcome signal, but not fully resolved.

2. **Type B coverage:** Semantic misread is the hardest failure type to catch (32/35 separation vs 23/24 for Type A). When the misread is subtle, the judge may score early coherent steps highly and miss the directional error.

3. **Capacity ceiling:** TAPR improves the training signal, not the model's raw capacity. If Qwen2.5-3B's reasoning failures are fundamentally capacity-limited rather than signal-limited, improvements will be bounded regardless of reward design.

4. **Judge reliability bounded by Qwen2.5-7B quality:** The transition to Qwen2.5-7B-Instruct as the open-weight judge (from GPT-4.1-mini used in Phase 1 validation) introduces a quality gap. Mitigated by testing judge quality on 20 problems before full precomputation.
