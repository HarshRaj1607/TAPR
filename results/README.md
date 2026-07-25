# Results

All figures come from Qwen2.5-3B-Instruct trained with GRPO + LoRA for 400 steps on a single A100-40GB, judged by a frozen Qwen2.5-7B-Instruct.

## Provenance

Every file carries a `provenance` field. Two values appear:

- **`exact`** — transcribed directly from Colab run logs.
- **`APPROXIMATE`** — observed during a run but not captured exactly before the session ended. Currently only the MATH-500 baseline and TAPR figures. **These must be re-run before any submission.**

## Files

| File | Contents |
|---|---|
| `reward_degeneracy.json` | The core mechanistic result — `frac_reward_zero_std` for both tasks, with per-step logged values |
| `gsm8k/full_eval_summary.json` | Full 1319-problem test split, all three models, with running-accuracy checkpoints |
| `strategyqa/eval_summary.json` | Two experiments — StrategyQA-trained models, and the GSM8K→StrategyQA transfer result |
| `math500/eval_summary.json` | Null result, with the two approximate figures flagged |
| `judge_validation/summary.json` | All seven judge calibration runs, including the three MATH failures |
| `judge_validation/strategyqa_judge_validation.json` | Raw per-problem output from the production StrategyQA judge — both reasoning chains, both scores, flaw type |
| `judge_validation/math_judge_validation.json` | Raw per-problem output from the Qwen2.5-32B MATH judge, showing the 8/20 separation failure per flaw type |

Training-time artifacts — LoRA adapters and per-step TRL logs for all four runs — live in [`../runs/`](../runs/README.md), not here.

## Reading the numbers

**What is claimable:**

- Reward degeneracy elimination (0.525→0.000 GSM8K, 0.625→0.003 StrategyQA). Definitional, measured directly, replicated across two tasks.
- The 4.5pp cross-task transfer gap — *pending multi-seed replication*.
- Judge validation: an open-weight 7B judge is competitive with GPT-4.1-mini (r=0.734 vs 0.600) and removes the closed-weight API dependency. **Not** that it beats it — at n=20 the CI is [0.43, 0.89] against [0.45, 0.72] at n=93.

**What is not claimable:**

- In-domain accuracy improvements. +0.23pp on GSM8K is 3 answers out of 1319. +1.00pp on StrategyQA is inside the 3.4pp standard error at n=200.

**A cautionary data point.** The GSM8K TAPR-vs-baseline gap read as 3.0pp at n=200 and collapsed to 0.23pp at the full 1319. Early small-sample reads in RL evaluation are unreliable in both directions.
