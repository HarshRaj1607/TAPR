"""
TAPR GRPO Training Script
==========================
PI: Harsh | Executor: Claude
Phase 2 — Samsung EnnovateX AX Hackathon

Trains Qwen2.5-3B-Instruct with GRPO using the TAPR reward signal.

Two modes:
  --mode baseline    R_outcome only (λ=0)             → Run A: baseline
  --mode tapr        R_outcome + λ·R_transition       → Run B: TAPR

Same script, same model, same setup. One parameter differs.
The comparison between A and B is the core Phase 2 result.

Training data:  GSM8K training split, easy-to-hard curriculum
Judge:          Qwen2.5-7B-Instruct (4-bit, live scoring, tapr mode only)
Base model:     Qwen2.5-3B-Instruct + LoRA
RL algorithm:   GRPO (TRL)

Usage:
  python train_grpo.py --mode baseline --output_dir ./output_baseline
  python train_grpo.py --mode tapr --lambda_val 0.3 --output_dir ./output_tapr
"""

import argparse
import json
import os
import re
import random
import numpy as np
import torch
from dataclasses import dataclass
from typing import Optional

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import GRPOTrainer, GRPOConfig
from scipy.stats import spearmanr

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# ── models ────────────────────────────────────────────────────────────────────
STUDENT_MODEL  = "Qwen/Qwen2.5-3B-Instruct"
JUDGE_MODEL    = "Qwen/Qwen2.5-7B-Instruct"

# ── LoRA ─────────────────────────────────────────────────────────────────────
# Expanded to all 4 attention projections — literature-grounded for reasoning
LORA_R            = 16
LORA_ALPHA        = 32
LORA_TARGET_MODS  = ["q_proj", "k_proj", "v_proj", "o_proj"]
LORA_DROPOUT      = 0.05

# ── training defaults ─────────────────────────────────────────────────────────
LEARNING_RATE     = 1e-5
KL_COEF           = 0.1
NUM_GENERATIONS   = 4       # G: solutions per prompt in GRPO
BATCH_SIZE        = 2       # per device — keep small for T4
GRAD_ACCUM        = 8       # effective batch = 16
MAX_STEPS         = 1000    # covers most of GSM8K training split
MAX_COMP_LENGTH   = 512     # max tokens generated per solution
EVAL_STEPS        = 50
SAVE_STEPS        = 100
LOGGING_STEPS     = 10
TEMPERATURE       = 0.9     # sampling diversity for GRPO rollouts
N_EVAL_SAMPLES    = 500     # held-out GSM8K test problems for mid-training eval

# ── TAPR judge ────────────────────────────────────────────────────────────────
WINDOW_SIZES  = [2, 3, 5]
WINDOW_PROBS  = [0.1, 0.5, 0.4]
JUDGE_TOKENS  = 150

# ── system prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a mathematical reasoning assistant. "
    "Solve the problem step by step. Write one operation or deduction per line. "
    "End your solution with #### [final numeric answer]."
)

# ── judge prompt ──────────────────────────────────────────────────────────────
JUDGE_PROMPT = """You are evaluating the quality of reasoning transitions in a math problem solution.

Problem: {problem}

Steps to evaluate:
{window_steps}

Judge whether these steps represent legitimate reasoning across these dimensions:

1. LOCAL VALIDITY: Does each step follow logically or mathematically from the previous one?
   A step is valid even if the approach is non-obvious or creative, as long as it is a
   legitimate mathematical or logical move.

2. VALUE CONSISTENCY: Are the numbers and quantities used traceable to either the problem
   statement or prior steps? Flag any values that appear without derivation.

3. QUESTION COHERENCE: Is the reasoning moving in a direction consistent with what the
   problem is actually asking? Flag if the steps are solving a different or simpler question.

4. CONVERGENCE: Are these steps collectively moving toward a solution, even if individual
   steps temporarily increase complexity?

5. COMPLETENESS: Do the steps show their work fully? Flag if reasoning collapses into a
   bare assertion — a number stated without the computation that leads to it.

Respond with:
SCORE: [0.0 to 1.0, your honest judgment across all five dimensions]
REASONING: [one sentence explaining the score]"""


# ── argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="TAPR GRPO Training")
    parser.add_argument(
        "--mode", type=str, choices=["baseline", "tapr"], required=True,
        help="baseline: R_outcome only | tapr: R_outcome + λ·R_transition"
    )
    parser.add_argument(
        "--lambda_val", type=float, default=0.3,
        help="Weight on R_transition (ignored in baseline mode). Default: 0.3"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./tapr_output",
        help="Directory for checkpoints and final model"
    )
    parser.add_argument(
        "--max_steps", type=int, default=MAX_STEPS,
        help="Total training steps. Adjust for available compute."
    )
    parser.add_argument(
        "--num_generations", type=int, default=NUM_GENERATIONS,
        help="GRPO rollouts per prompt. Reduce to 4 if OOM on T4."
    )
    parser.add_argument(
        "--batch_size", type=int, default=BATCH_SIZE,
        help="Per-device train batch size."
    )
    parser.add_argument(
        "--lr", type=float, default=LEARNING_RATE,
        help="Learning rate. Default: 1e-5"
    )
    parser.add_argument(
        "--smoke_test", action="store_true",
        help=(
            "Run 5 steps on 20 problems to verify the full pipeline works. "
            "Always run this before committing to a full training run. "
            "Takes ~5 minutes. Catches 90%% of bugs cheaply."
        )
    )
    return parser.parse_args()


# ── model loading ─────────────────────────────────────────────────────────────

def load_student(model_name: str):
    """Load Qwen2.5-3B-Instruct in 4-bit + LoRA for GRPO training."""
    print(f"\nLoading student: {model_name}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODS,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print(f"Student loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB")
    return model, tokenizer


def load_judge(model_name: str):
    """Load Qwen2.5-7B-Instruct in 4-bit as frozen judge. No gradients."""
    print(f"\nLoading judge: {model_name}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    print(f"Judge loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB")
    return model, tokenizer


# ── dataset preparation ───────────────────────────────────────────────────────

def count_steps(answer_text: str) -> int:
    """Count solution steps as difficulty proxy for curriculum ordering."""
    lines = [l.strip() for l in answer_text.split("\n")
             if l.strip() and not l.strip().startswith("####")]
    return len(lines)


def format_prompt(question: str, tokenizer) -> str:
    """Format a GSM8K question as a chat prompt for the student model."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": question},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def load_gsm8k(tokenizer, split: str = "train"):
    """
    Load GSM8K, extract ground-truth answers, apply easy-to-hard curriculum.
    Curriculum proxy: number of GT solution steps (shorter = easier).
    """
    print(f"\nLoading GSM8K ({split} split)...")
    ds = load_dataset("openai/gsm8k", "main", split=split)

    records = []
    for item in ds:
        gt_num = extract_answer(item["answer"])
        if gt_num is None:
            continue
        records.append({
            "question":  item["question"],
            "answer":    item["answer"],
            "gt_num":    gt_num,
            "n_steps":   count_steps(item["answer"]),
        })

    # Easy-to-hard curriculum: sort by step count ascending
    if split == "train":
        records.sort(key=lambda x: x["n_steps"])
        print(f"Curriculum applied: {records[0]['n_steps']} steps (easy) → "
              f"{records[-1]['n_steps']} steps (hard)")

    # Add formatted prompt
    for r in records:
        r["prompt"] = format_prompt(r["question"], tokenizer)

    print(f"Loaded {len(records)} problems")
    return records


# ── answer extraction + R_outcome ─────────────────────────────────────────────

def extract_answer(text: str) -> Optional[str]:
    """Extract numeric answer after #### marker."""
    match = re.search(r"####\s*([+-]?\d+(?:,\d+)*(?:\.\d+)?)", text)
    if match:
        return match.group(1).replace(",", "")
    nums = re.findall(r"[+-]?\d+(?:\.\d+)?", text)
    return nums[-1] if nums else None


def r_outcome(completion: str, gt_num: str) -> float:
    """Binary correctness reward. 1.0 if answer matches GT, else 0.0."""
    pred = extract_answer(completion)
    if pred is None or gt_num is None:
        return 0.0
    try:
        return 1.0 if abs(float(pred) - float(gt_num)) < 0.01 else 0.0
    except (ValueError, TypeError):
        return 0.0


# ── judge scoring + R_transition ──────────────────────────────────────────────

def parse_steps(text: str) -> list:
    """Split solution into individual reasoning steps."""
    lines = text.strip().split("\n")
    return [l.strip() for l in lines
            if l.strip() and not l.strip().startswith("####")]


def sample_window_size() -> int:
    return random.choices(WINDOW_SIZES, weights=WINDOW_PROBS, k=1)[0]


def parse_judge_score(text: str) -> float:
    """Extract score from judge response. Falls back to 0.5 if unparseable."""
    match = re.search(r"SCORE:\s*([0-9]*\.?[0-9]+)", text)
    if match:
        return max(0.0, min(1.0, float(match.group(1))))
    nums = re.findall(r"0\.\d+|1\.0|0\.0", text)
    return float(nums[0]) if nums else 0.5


def judge_window(judge_model, judge_tok, problem: str, window: list) -> float:
    """Score one window of steps with the judge model. Returns score in [0,1]."""
    steps_text = "\n".join([f"Step {i+1}: {s}" for i, s in enumerate(window)])
    prompt = JUDGE_PROMPT.format(problem=problem, window_steps=steps_text)

    messages = [{"role": "user", "content": prompt}]
    text = judge_tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = judge_tok([text], return_tensors="pt").to(judge_model.device)

    with torch.no_grad():
        out = judge_model.generate(
            **inputs,
            max_new_tokens=JUDGE_TOKENS,
            do_sample=False,
            pad_token_id=judge_tok.eos_token_id,
        )

    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    response = judge_tok.decode(new_tokens, skip_special_tokens=True).strip()
    return parse_judge_score(response)


def r_transition(judge_model, judge_tok, completion: str, question: str) -> float:
    """
    Compute R_transition for one solution.
    Sliding window across steps, stride=1, W sampled stochastically.
    Returns mean window score. If too few steps, returns 0.5 (neutral).
    """
    steps = parse_steps(completion)
    if len(steps) < 2:
        return 0.5

    window_scores = []
    for i in range(len(steps)):
        W = sample_window_size()
        window = steps[i: i + W]
        if len(window) < 2:
            continue
        score = judge_window(judge_model, judge_tok, question, window)
        window_scores.append(score)

    return float(np.mean(window_scores)) if window_scores else 0.5


# ── reward function ───────────────────────────────────────────────────────────

class TAPRReward:
    """
    Reward function factory for GRPO.
    mode='baseline': R_outcome only
    mode='tapr':     R_outcome + lambda_val * R_transition

    Designed to be modular — judge prompt and dimensions can be
    updated per benchmark without changing the training loop.
    """

    def __init__(self, mode: str, lambda_val: float,
                 judge_model=None, judge_tok=None):
        self.mode       = mode
        self.lambda_val = lambda_val
        self.judge      = judge_model
        self.judge_tok  = judge_tok

        assert mode in ("baseline", "tapr")
        if mode == "tapr":
            assert judge_model is not None, "Judge required for tapr mode"

    def __call__(self, completions: list, **kwargs) -> list:
        """
        Called by GRPOTrainer after generating completions.
        kwargs contains dataset columns: 'gt_num', 'question'
        """
        gt_nums   = kwargs.get("gt_num",   [None]  * len(completions))
        questions = kwargs.get("question", [""]    * len(completions))

        rewards = []
        for completion, gt_num, question in zip(completions, gt_nums, questions):

            ro = r_outcome(completion, gt_num)

            if self.mode == "tapr":
                rt = r_transition(self.judge, self.judge_tok, completion, question)
                r_final = ro + self.lambda_val * rt
            else:
                r_final = ro

            rewards.append(float(r_final))

        return rewards


# ── evaluation ────────────────────────────────────────────────────────────────

def evaluate(model, tokenizer, eval_records: list, n: int = N_EVAL_SAMPLES) -> dict:
    """
    Evaluate student model on held-out GSM8K problems.
    Greedy decoding, exact-match accuracy.
    Returns dict with accuracy and sample-level results.
    """
    model.eval()
    subset = eval_records[:n]
    correct, total = 0, 0

    for item in subset:
        inputs = tokenizer(
            item["prompt"],
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(model.device)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=MAX_COMP_LENGTH,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        completion = tokenizer.decode(new_tokens, skip_special_tokens=True)
        correct += r_outcome(completion, item["gt_num"])
        total   += 1

    acc = correct / total if total > 0 else 0.0
    model.train()
    return {"accuracy": round(acc, 4), "correct": int(correct), "total": total}


# ── checkpoint resume ─────────────────────────────────────────────────────────

def find_latest_checkpoint(output_dir: str) -> Optional[str]:
    """
    Find the latest saved checkpoint in output_dir.
    Returns full path to checkpoint, or None if no checkpoints exist.
    Called automatically before training — if found, run resumes from there.
    """
    if not os.path.exists(output_dir):
        return None
    checkpoints = [
        d for d in os.listdir(output_dir)
        if d.startswith("checkpoint-") and os.path.isdir(os.path.join(output_dir, d))
    ]
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda x: int(x.split("-")[1]))
    return os.path.join(output_dir, checkpoints[-1])


# ── preflight check ───────────────────────────────────────────────────────────

def preflight_check(student, student_tok, reward_fn, sample_record: dict,
                    judge=None, judge_tok=None) -> bool:
    """
    Run before any training. Tests the full pipeline on one sample.
    Catches: output format issues, reward function column mismatches,
    judge device errors, dtype mismatches.

    Returns True if all checks pass. Raises on hard failures.
    """
    print("\n" + "="*55)
    print("PREFLIGHT CHECK")
    print("="*55)
    passed = True

    # ── 1. Student can generate ───────────────────────────────────────────────
    print("\n[1/4] Student generation...")
    try:
        inputs = student_tok(
            sample_record["prompt"],
            return_tensors="pt",
            truncation=True,
            max_length=256,
        ).to(next(student.parameters()).device)

        with torch.no_grad():
            out = student.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                pad_token_id=student_tok.eos_token_id,
            )
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        sample_completion = student_tok.decode(new_tokens, skip_special_tokens=True)
        print(f"  Generated {len(sample_completion)} chars")
        print(f"  Preview: {sample_completion[:120].strip()}")
    except Exception as e:
        print(f"  ✗ FAIL: {e}")
        raise RuntimeError("Student generation failed. Check model loading.") from e
    print("  ✓ OK")

    # ── 2. Output format contains #### ────────────────────────────────────────
    print("\n[2/4] Output format check (#### marker)...")
    if "####" in sample_completion:
        extracted = extract_answer(sample_completion)
        print(f"  ✓ OK — found ####, extracted: {extracted}")
    else:
        print("  ⚠ WARNING — no #### in output. Answer extraction will fail.")
        print("  The system prompt may need adjustment, or the model needs warmup.")
        print("  R_outcome will be 0 for all solutions without ####.")
        print("  Sample output:")
        print(f"  {sample_completion[:200]}")
        passed = False  # soft fail — training can proceed but results will be poor

    # ── 3. Reward function works ──────────────────────────────────────────────
    print("\n[3/4] Reward function...")
    try:
        dummy_rewards = reward_fn(
            completions=[sample_completion],
            gt_num=[sample_record["gt_num"]],
            question=[sample_record["question"]],
        )
        assert isinstance(dummy_rewards, list), "Reward must return list"
        assert len(dummy_rewards) == 1, f"Expected 1 reward, got {len(dummy_rewards)}"
        assert isinstance(dummy_rewards[0], float), f"Reward must be float, got {type(dummy_rewards[0])}"
        ro = r_outcome(sample_completion, sample_record["gt_num"])
        print(f"  R_final = {dummy_rewards[0]:.4f}  |  R_outcome = {ro:.1f}")
    except Exception as e:
        print(f"  ✗ FAIL: {e}")
        raise RuntimeError("Reward function failed. Check column names and return types.") from e
    print("  ✓ OK")

    # ── 4. Judge works (tapr mode only) ───────────────────────────────────────
    print("\n[4/4] Judge check...")
    if judge is not None:
        try:
            steps = parse_steps(sample_completion)
            if len(steps) >= 2:
                score = judge_window(judge, judge_tok,
                                     sample_record["question"], steps[:2])
                assert 0.0 <= score <= 1.0, f"Score out of range: {score}"
                print(f"  Judge score on first 2 steps: {score:.4f}")
            else:
                print("  Too few steps to judge — skipped (not a failure)")
        except Exception as e:
            print(f"  ✗ FAIL: {e}")
            raise RuntimeError("Judge scoring failed. Check device placement and model loading.") from e
        print("  ✓ OK")
    else:
        print("  Skipped (baseline mode — no judge)")

    # ── summary ───────────────────────────────────────────────────────────────
    print("\n" + "="*55)
    if passed:
        print("✓ PREFLIGHT PASSED — safe to start training")
    else:
        print("⚠ PREFLIGHT PASSED WITH WARNINGS — check output format above")
        print("  Training will proceed but verify #### appears in outputs early on.")
    print("="*55)
    return passed


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required. Run on Kaggle, Colab, or college GPU.")

    total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {total_vram:.1f} GB")
    print(f"\nMode:       {args.mode.upper()}")
    print(f"Lambda:     {args.lambda_val if args.mode == 'tapr' else 0.0}")
    print(f"Output:     {args.output_dir}")
    if args.smoke_test:
        print("*** SMOKE TEST MODE — 5 steps, 20 problems ***")
    else:
        print(f"Max steps:  {args.max_steps}")
    print(f"G (rollouts): {args.num_generations}")

    os.makedirs(args.output_dir, exist_ok=True)

    # ── load student ──────────────────────────────────────────────────────────
    student, student_tok = load_student(STUDENT_MODEL)

    # ── load judge (tapr mode only) ───────────────────────────────────────────
    judge, judge_tok = None, None
    if args.mode == "tapr":
        judge, judge_tok = load_judge(JUDGE_MODEL)
        print(f"\nTotal VRAM after loading both models: "
              f"{torch.cuda.memory_allocated()/1e9:.1f} GB / {total_vram:.1f} GB")

    # ── load datasets ─────────────────────────────────────────────────────────
    train_records = load_gsm8k(student_tok, split="train")
    eval_records  = load_gsm8k(student_tok, split="test")

    # Smoke test: use tiny subset
    if args.smoke_test:
        train_records = train_records[:20]
        eval_records  = eval_records[:10]
        args.max_steps = 5
        print(f"\nSmoke test: {len(train_records)} train, {len(eval_records)} eval, "
              f"{args.max_steps} steps")

    # HuggingFace dataset format for GRPOTrainer
    from datasets import Dataset
    train_dataset = Dataset.from_list([
        {
            "prompt":   r["prompt"],
            "question": r["question"],
            "gt_num":   r["gt_num"],
        }
        for r in train_records
    ])

    # ── reward function ───────────────────────────────────────────────────────
    reward_fn = TAPRReward(
        mode        = args.mode,
        lambda_val  = args.lambda_val,
        judge_model = judge,
        judge_tok   = judge_tok,
    )

    # ── preflight check ───────────────────────────────────────────────────────
    preflight_check(
        student       = student,
        student_tok   = student_tok,
        reward_fn     = reward_fn,
        sample_record = train_records[0],
        judge         = judge,
        judge_tok     = judge_tok,
    )

    # ── GRPO config ───────────────────────────────────────────────────────────
    bf16_supported = torch.cuda.is_bf16_supported()
    print(f"\nPrecision: {'bf16' if bf16_supported else 'fp16'}")

    grpo_config = GRPOConfig(
        output_dir                 = args.output_dir,
        num_generations            = args.num_generations,
        max_completion_length      = MAX_COMP_LENGTH,
        learning_rate              = args.lr,
        per_device_train_batch_size = args.batch_size,
        gradient_accumulation_steps = GRAD_ACCUM,
        max_steps                  = args.max_steps,
        eval_steps                 = EVAL_STEPS,
        save_steps                 = SAVE_STEPS,
        logging_steps              = LOGGING_STEPS,
        kl_coef                    = KL_COEF,
        temperature                = TEMPERATURE,
        bf16                       = bf16_supported,
        fp16                       = not bf16_supported,
        remove_unused_columns      = False,
        seed                       = RANDOM_SEED,
        report_to                  = "none",    # set to "wandb" if available
    )

    # ── trainer ───────────────────────────────────────────────────────────────
    trainer = GRPOTrainer(
        model         = student,
        reward_funcs  = [reward_fn],
        args          = grpo_config,
        train_dataset = train_dataset,
        tokenizer     = student_tok,
    )

    # ── baseline eval before training ────────────────────────────────────────
    print("\n" + "="*55)
    print("PRE-TRAINING EVAL (sanity check baseline)")
    print("="*55)
    pre_eval = evaluate(student, student_tok, eval_records)
    print(f"Accuracy: {pre_eval['accuracy']:.4f} ({pre_eval['correct']}/{pre_eval['total']})")
    print("Expected ~0.78 for Qwen2.5-3B-Instruct zero-shot on GSM8K")

    # ── training ──────────────────────────────────────────────────────────────
    print("\n" + "="*55)
    print(f"TRAINING — {args.mode.upper()} MODE")
    print("="*55)

    checkpoint = find_latest_checkpoint(args.output_dir)
    if checkpoint:
        print(f"Checkpoint found: {checkpoint}")
        print("Resuming from checkpoint — steps already done are skipped.")
        trainer.train(resume_from_checkpoint=checkpoint)
    else:
        print("No checkpoint found — starting fresh.")
        trainer.train()

    # ── post-training eval ────────────────────────────────────────────────────
    print("\n" + "="*55)
    print("POST-TRAINING EVAL")
    print("="*55)
    post_eval = evaluate(student, student_tok, eval_records)
    print(f"Accuracy: {post_eval['accuracy']:.4f} ({post_eval['correct']}/{post_eval['total']})")
    print(f"Delta:    {post_eval['accuracy'] - pre_eval['accuracy']:+.4f}")

    # ── save final model ──────────────────────────────────────────────────────
    final_dir = os.path.join(args.output_dir, "final")
    student.save_pretrained(final_dir)
    student_tok.save_pretrained(final_dir)
    print(f"\nModel saved to {final_dir}")

    # ── save run summary ──────────────────────────────────────────────────────
    summary = {
        "mode":          args.mode,
        "lambda_val":    args.lambda_val if args.mode == "tapr" else 0.0,
        "lora":          {"r": LORA_R, "alpha": LORA_ALPHA, "targets": LORA_TARGET_MODS},
        "training":      {"max_steps": args.max_steps, "lr": args.lr, "kl_coef": KL_COEF},
        "pre_training":  pre_eval,
        "post_training": post_eval,
        "delta":         round(post_eval["accuracy"] - pre_eval["accuracy"], 4),
    }
    summary_path = os.path.join(args.output_dir, "run_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {summary_path}")

    print("\n" + "="*55)
    print("DONE")
    print("="*55)
    print(f"Mode:  {args.mode.upper()}")
    print(f"Pre:   {pre_eval['accuracy']:.4f}")
    print(f"Post:  {post_eval['accuracy']:.4f}")
    print(f"Delta: {post_eval['accuracy'] - pre_eval['accuracy']:+.4f}")
    if args.smoke_test:
        print("\n*** SMOKE TEST COMPLETE ***")
        print("Pipeline is working. Run without --smoke_test for full training.")
    elif args.mode == "baseline":
        print("\nNext: run tapr mode and compare post-training accuracy.")
    else:
        print("\nNext: compare against baseline run_summary.json delta.")


if __name__ == "__main__":
    main()
