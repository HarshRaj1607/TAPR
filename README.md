# TAPR — Transition-Aware Process Reward for SLM Reasoning

* **Problem Statement Number** - 6
* **Problem Statement Title** - Improving Reasoning in Small Language Models via Reinforcement Learning
* **Team name** - Axiom
* **Team members** - Harsh Raj, Swati Jayaram Kotary
* **Institute/College Name** - Ramaiah Institute of Technology, MSR Nagar, MSRIT Post, Mathikere, Bengaluru – 560054, Karnataka
* **Final Presentation Google Drive Link** - [ADD BEFORE SUBMISSION — must be publicly accessible, no login wall]
* **Full Submission Demo Video Link** - [ADD BEFORE SUBMISSION — YouTube public/unlisted only]
* **Setup & Result Reproducibility Video Link** - [ADD BEFORE SUBMISSION — YouTube public/unlisted only]

---

## What is TAPR?

Standard outcome-based RL rewards only the final answer, leaving the entire reasoning process unsupervised. For Small Language Models (SLMs, ≤7B parameters), this is damaging — they accumulate errors across steps, reach correct answers via flawed reasoning, and are highly sensitive to poorly designed reward signals.

TAPR (Transition-Aware Process Reward) adds a process signal that rewards the *transition* from reasoning step N to step N+1 — the exact moment where reasoning either advances legitimately or breaks down.

**Reward formula:**
```
R_final = R_outcome + λ · R_transition
R_transition = mean(Qwen2.5-7B-Instruct judge scores across sliding windows)
```

**Phase 1 validation results:**
- TAPR signal: Spearman r = 0.600 (p < 0.0001) on 93 GSM8K problems
- Ground truth beats flawed reasoning: 84/93 (90.3%)
- Outcome reward alone: 0/93 separation of reasoning quality (blind by construction)
- Progress signal tested and failed (r = −0.690, wrong direction) — documented honest negative result

---

## Project Artefacts

### Technical Documentation

Full documentation in [`docs/`](docs/):

- [`docs/architecture.md`](docs/architecture.md) — System architecture, signal design, training pipeline
- [`docs/installation.md`](docs/installation.md) — Installation and environment setup
- [`docs/user_guide.md`](docs/user_guide.md) — Usage guide, training instructions, evaluation
- [`docs/ax.md`](docs/ax.md) — Agentic AI usage, open weight model integration, what worked and what didn't

### Source Code

All code in [`src/`](src/):

```
src/
├── validation/               # Phase 1 signal validation (completed)
│   ├── tapr_generate_flaws.py
│   ├── tapr_score_llm.py
│   ├── tapr_score_progress.py
│   └── tapr_signal_validation.py
├── judge_precompute.py       # Qwen2.5-7B judge precomputation (Phase 2)
├── sft_warmup.py             # Stage 1: light SFT warmup
├── train_grpo.py             # Stage 2–4: GRPO training with TAPR reward
└── eval.py                   # GSM8K / StrategyQA / MMLU evaluation
```

### Models Used

| Model | Role | Link |
|-------|------|------|
| Qwen2.5-3B-Instruct | Base SLM — fine-tuned with GRPO + TAPR | [HuggingFace](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) |
| Qwen2.5-7B-Instruct | Transition judge — reward signal | [HuggingFace](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) |

### Models Published

| Model | Description | Link |
|-------|-------------|------|
| TAPR-Qwen2.5-3B | Qwen2.5-3B fine-tuned with GRPO + TAPR reward, Apache 2.0 | [ADD AFTER TRAINING] |

### Datasets Used

| Dataset | Role | Link |
|---------|------|------|
| GSM8K | Primary benchmark — 8,500 grade school math problems | [HuggingFace](https://huggingface.co/datasets/openai/gsm8k) |
| StrategyQA | Secondary benchmark — multi-step implicit reasoning | [HuggingFace](https://huggingface.co/datasets/wics/strategy-qa) |
| MMLU | Tertiary benchmark — 57-subject reasoning | [HuggingFace](https://huggingface.co/datasets/cais/mmlu) |

### Datasets Published

| Dataset | Description | Link |
|---------|-------------|------|
| TAPR Validation Set | 100 GSM8K problems with paired flawed solutions across 3 empirically-grounded failure types (Type A: wrong quantity, Type B: semantic misread, Type C: collapsed reasoning). CC BY 4.0. | [ADD AFTER UPLOAD] |

---

## Technical Architecture

### Signal Design

The TAPR judge (Qwen2.5-7B-Instruct) evaluates windows of consecutive reasoning steps across five dimensions, each targeting an empirically documented SLM failure mode:

| Dimension | Targets |
|-----------|---------|
| Local Validity — step N+1 follows logically from N | All failure types at transition level |
| Value Consistency — quantities traceable to problem or prior steps | Type A: wrong quantity |
| Question Coherence — reasoning addresses the actual question | Type B: semantic misread |
| Convergence — steps collectively move toward a solution | Type D: directional drift |
| Completeness — no unsupported final assertions | Type C: collapsed reasoning |

Window size W sampled stochastically from {2, 3, 5} with P = (0.1, 0.5, 0.4), stride = 1. Judge precomputed **offline** before training — zero inference cost at training time.

### Training Pipeline

| Stage | Description |
|-------|-------------|
| 1 — Light SFT | 1 epoch, 200–300 examples, format compliance only |
| 2 — Curriculum | Easy-to-hard ordering via zero-shot accuracy as difficulty proxy |
| 3 — GRPO + TAPR | R_final = R_outcome + λ · R_transition |
| 4 — Stability | KL penalty, entropy monitoring, prompt augmentation |

**Base model:** Qwen2.5-3B-Instruct + LoRA (r=16, α=32, target: q_proj, v_proj)  
**RL algorithm:** GRPO (no critic model needed)

### Ablation Design

| Run | Reward | Expected result |
|-----|--------|-----------------|
| Baseline | None (zero-shot) | 78% GSM8K |
| Outcome-only | R_outcome | ~82–85% GSM8K |
| TAPR | R_outcome + λ·R_transition | Target: >85% GSM8K |

---

## Attribution

Built from scratch on open benchmarks (GSM8K, StrategyQA, MMLU) and open weight models (Qwen2.5 family, Apache 2.0). Training uses the [TRL library](https://github.com/huggingface/trl) for GRPO. No existing project forked or extended.
