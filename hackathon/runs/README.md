# Runs

The four 400-step GRPO runs behind every number in the top-level README. All four use Qwen2.5-3B-Instruct (4-bit) as student, LoRA r=16 / α=32 on q/k/v/o_proj, lr=1e-5, G=4, β(KL)=0.1, seed 42, single A100-40GB. The two TAPR runs additionally use a frozen Qwen2.5-7B-Instruct judge at λ=0.3.

| Run | Task | Mode | λ |
|---|---|---|---|
| `gsm8k_tapr` | GSM8K | TAPR | 0.3 |
| `gsm8k_baseline` | GSM8K | outcome-only | 0.0 |
| `sqa_tapr` | StrategyQA | TAPR | 0.3 |
| `sqa_baseline` | StrategyQA | outcome-only | 0.0 |

## What's in each folder

```
<run>/
├── adapter/              # Final LoRA adapter — load this
│   ├── adapter_model.safetensors
│   ├── adapter_config.json
│   ├── ref/              # Frozen reference adapter used for the KL term
│   ├── MODEL_CARD.md     # TRL-generated card
│   └── tokenizer files
├── run_summary.json      # Config + pre/post accuracy as written by train_grpo.py
└── training_log.json     # Per-step TRL log_history, every 10 steps (40 entries)
```

`training_log.json` was extracted from the step-400 `trainer_state.json` before intermediate checkpoints were archived, so the full training curves survive without keeping 1.4 GB of optimizer state. Each entry carries `reward`, `reward_std`, `frac_reward_zero_std`, `kl`, `entropy`, `grad_norm`, `loss`, completion-length statistics, and clip ratios.

## Loading an adapter

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B-Instruct", load_in_4bit=True)
model = PeftModel.from_pretrained(base, "runs/gsm8k_tapr/adapter")
tok = AutoTokenizer.from_pretrained("runs/gsm8k_tapr/adapter")
```

Or via `src/eval.py`:

```bash
python src/eval.py --model tapr --tapr_dir ./runs/gsm8k_tapr/adapter \
  --output_file results/eval_tapr.json
```

## Training dynamics at a glance

Read off `training_log.json`. `frac_reward_zero_std` is the mechanistic result — the fraction of groups where all G=4 rollouts got identical reward and the GRPO advantage is therefore zero for all of them.

| Run | `frac_reward_zero_std` (mean over run) | final KL | mean completion length, step 10 → 400 | entropy, step 10 → 400 |
|---|---|---|---|---|
| `gsm8k_baseline` | 0.640 | 0.0013 | 171 → 170 | 0.251 → 0.247 |
| `gsm8k_tapr` | **0.020** | 0.0089 | 172 → 143 | 0.257 → 0.215 |
| `sqa_baseline` | 0.643 | 0.0013 | 136 → 132 | 0.840 → 0.836 |
| `sqa_tapr` | **0.005** | 0.0053 | 139 → 152 | 0.852 → 0.876 |

Two things are visible here beyond the degeneracy result. TAPR runs show KL roughly 4–7× higher than their baselines — the policy moved further, though still very little in absolute terms. And the two TAPR runs move completion length in *opposite* directions (GSM8K shortens, StrategyQA lengthens), which is worth noting before anyone reads length change as a signature of the method.

## Known inconsistency — the GSM8K `run_summary.json` accuracies

`gsm8k_tapr/run_summary.json` and `gsm8k_baseline/run_summary.json` report **identical** measured values: pre 371/500 (0.742), post 402/500 (0.804), delta 0.062. Only `mode` and `lambda_val` differ.

**These two files were not copied from each other.** `mode` reads `"tapr"` / `"baseline"` and `lambda_val` reads `0.3` / `0.0` — exactly what `train_grpo.py` derives from argparse (`args.lambda_val if args.mode == "tapr" else 0.0`). A copied file would carry the source run's mode. Each file was written by its own genuine run.

**The identical *pre-training* number is expected, not a bug.** LoRA initializes with B=0, so at step 0 the adapter is an exact identity and both runs evaluate the untouched base model, greedily, on the same problems in the same order. 371/500 twice is what correct code does.

**The problem is that the eval set behind these numbers can no longer be identified.** Three schema differences show both files were written by an *older* version of `train_grpo.py` than the one in `src/`:

| | GSM8K summaries | StrategyQA summaries | current `src/train_grpo.py` |
|---|---|---|---|
| `task` field | absent | present | writes it (line 1152) |
| `unparseable` in eval dict | absent | present | writes it for StrategyQA |
| eval `total` | 500 | 200 | `N_EVAL_SAMPLES = 200` |

`N_EVAL_SAMPLES = 500` does not exist anywhere in the current code. And the obvious guess — the first 500 of the GSM8K test split — is ruled out: `eval.py` measures the untrained model at **0.7180** (359/500) on exactly that subset, against the summaries' pre-training **0.742** (371/500). Twelve problems apart, same nominal set, same greedy decoding, same prompt, same extraction function. So the old script was evaluating on some *other* 500 problems, and nothing in the repo records which.

That also explains the level shift: post 0.804 sits above both full-test figures (0.7703 / 0.7726) and above eval.py's first-500 readings (0.756 / 0.776). It is a different, apparently easier, sample.

**Given that, the identical post-training number is plausibly just a tie.** The two policies barely moved (final KL 0.0089 and 0.0013 against a base they both started from), so they agree on the overwhelming majority of problems and differ on a handful. On an easier subset where both are closer to saturation, landing on the same count is not the 1-in-50 coincidence it looks like at first glance — it is the expected outcome a meaningful fraction of the time.

**Conclusion: the GSM8K `run_summary.json` accuracies are unreproducible and should not be cited.** They are not evidence that the runs were identical — the training logs (`frac_reward_zero_std` 0.020 vs 0.640) and the adapter checksums both show the runs differed exactly as intended. Cite `results/gsm8k/full_eval_summary.json` (n=1319, provenance `exact`), which is what every document in this repo already does.

The StrategyQA summaries carry no such problem: they were written by the current script, their n=200 matches `N_EVAL_SAMPLES`, and their figures agree with `results/strategyqa/eval_summary.json`.

## Not included

Intermediate checkpoints (steps 100, 200, 300, 400) and the StrategyQA smoke-test run were moved to `_archive/` outside the repository. They contain optimizer and RNG state needed only to resume training, not to reproduce any reported result.
