"""
TAPR GRPO Training Script
==========================
Harsh Raj

Trains Qwen2.5-3B-Instruct with GRPO using the TAPR reward signal.

Two modes:
  --mode baseline    R_outcome only (λ=0)             → Run A: baseline
  --mode tapr        R_outcome + λ·R_transition       → Run B: TAPR

Three tasks:
  --task gsm8k        Train on GSM8K (numeric match outcome reward)
  --task strategyqa   Train on StrategyQA (yes/no match outcome reward)
  --task math         Train on MATH (normalized LaTeX match outcome reward)

Judge dimensions per task:
  gsm8k / math:  Local Validity, Value Consistency, Question Coherence,
                 Convergence, Completeness  (same prompt — both are math)
  strategyqa:    Fact Accuracy, Chain Validity, Relevance, Convergence

Usage:
  # GSM8K baseline (Run A)
  python train_grpo.py --task gsm8k --mode baseline --output_dir ./output_gsm_baseline

  # GSM8K TAPR (Run B)
  python train_grpo.py --task gsm8k --mode tapr --lambda_val 0.3 --output_dir ./output_gsm_tapr

  # StrategyQA baseline
  python train_grpo.py --task strategyqa --mode baseline --output_dir ./output_sqa_baseline

  # StrategyQA TAPR
  python train_grpo.py --task strategyqa --mode tapr --lambda_val 0.3 --output_dir ./output_sqa_tapr

  # Smoke test (always run before full training)
  python train_grpo.py --task strategyqa --mode tapr --smoke_test --output_dir ./output_sqa_tapr
"""

import argparse
import json
import os
import re
import random
import numpy as np
import torch
import requests
from dataclasses import dataclass
from typing import Optional

from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import GRPOTrainer, GRPOConfig
from scipy.stats import spearmanr

RANDOM_SEED = 42          # default; override with --seed


def set_global_seed(seed: int) -> None:
    """
    Seed every RNG the run depends on.

    Must be called before load_task_data(): the StrategyQA loader shuffles its
    records and then takes a 90/10 split, so the seed determines which problems
    land in the held-out eval set. Two runs compared against each other must use
    the same seed or they are not evaluating on the same problems.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_global_seed(RANDOM_SEED)

# ── models ────────────────────────────────────────────────────────────────────
STUDENT_MODEL  = "Qwen/Qwen2.5-3B-Instruct"
JUDGE_MODEL    = "Qwen/Qwen2.5-7B-Instruct"

# ── LoRA ──────────────────────────────────────────────────────────────────────
LORA_R            = 16
LORA_ALPHA        = 32
LORA_TARGET_MODS  = ["q_proj", "k_proj", "v_proj", "o_proj"]
LORA_DROPOUT      = 0.05

# ── training defaults ─────────────────────────────────────────────────────────
LEARNING_RATE     = 1e-5
KL_COEF           = 0.1
NUM_GENERATIONS   = 4
BATCH_SIZE        = 4
GRAD_ACCUM        = 4
MAX_STEPS         = 400
MAX_COMP_LENGTH   = 512
EVAL_STEPS        = 50
SAVE_STEPS        = 100
LOGGING_STEPS     = 10
TEMPERATURE       = 0.9
N_EVAL_SAMPLES    = 200   # problems for mid-training eval (both tasks)

# ── TAPR judge ────────────────────────────────────────────────────────────────
WINDOW_SIZES  = [2, 3, 5]
WINDOW_PROBS  = [0.1, 0.5, 0.4]
JUDGE_TOKENS  = 80   # sufficient for both GSM8K (SCORE line) and SQA (4 score lines)


# ── system prompts ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT_GSM8K = (
    "You are a mathematical reasoning assistant. "
    "Solve the problem step by step. Write one operation or deduction per line. "
    "End your solution with #### [final numeric answer]."
)

SYSTEM_PROMPT_SQA = (
    "You are a reasoning assistant. "
    "Think through the yes/no question step by step using accurate facts. "
    "Write each reasoning step on a new line starting with a number. "
    "End your response with exactly 'Answer: yes' or 'Answer: no'."
)

SYSTEM_PROMPT_MATH = (
    "You are a mathematical reasoning assistant. "
    "Solve the problem step by step, showing all work clearly. "
    "End your solution with #### [final answer]."
)

SYSTEM_PROMPTS = {
    "gsm8k":       SYSTEM_PROMPT_GSM8K,
    "strategyqa":  SYSTEM_PROMPT_SQA,
    "math":        SYSTEM_PROMPT_MATH,
}


# ── judge prompts ──────────────────────────────────────────────────────────────

JUDGE_PROMPT_GSM8K = """You are evaluating the quality of reasoning transitions in a math problem solution.

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


JUDGE_PROMPT_SQA = """You are evaluating the quality of step-by-step reasoning for a yes/no question.

Question: {problem}

Steps to evaluate:
{window_steps}

Judge whether these steps represent legitimate reasoning across these dimensions:

1. FACT ACCURACY: Are the factual claims in these steps accurate and verifiable?
   Flag any claim that appears incorrect or unverifiable.

2. CHAIN VALIDITY: Does each step logically follow from the previous one?
   Flag any step that is a non-sequitur or introduces an unsupported jump.

3. RELEVANCE: Is the reasoning addressing what the question actually asks?
   Flag if the steps go on a tangent that does not contribute to answering the question.

4. CONVERGENCE: Are these steps collectively moving toward a yes/no conclusion,
   even if the answer is not stated yet?

Respond with:
SCORE: [0.0 to 1.0, your honest judgment across all four dimensions]
REASONING: [one sentence explaining the score]"""


JUDGE_PROMPT_MATH = """You are evaluating the quality of step-by-step reasoning in a competition math solution.

Problem: {problem}

Steps to evaluate:
{window_steps}

Judge whether these steps represent legitimate mathematical reasoning across these dimensions:

1. ALGEBRAIC VALIDITY: Are the algebraic manipulations in these steps correct?
   Flag incorrect factoring, expansion, simplification, or substitution.

2. VARIABLE CONSISTENCY: Are variables and expressions used consistently with how they were defined?
   Flag any variable reused with a different meaning, or expressions that contradict earlier definitions.

3. QUESTION COHERENCE: Is the reasoning addressing the specific problem being asked?
   Flag if the steps are solving a related but different problem.

4. CONVERGENCE: Are these steps making meaningful progress toward a solution?
   Flag circular reasoning or steps that do not reduce the problem.

5. JUSTIFICATION: Are non-trivial claims supported rather than asserted?
   Flag any step that states a result as obvious without showing the work.

Respond with:
SCORE: [0.0 to 1.0, your honest judgment across all five dimensions]
REASONING: [one sentence explaining the score]"""


JUDGE_PROMPTS = {
    "gsm8k":       JUDGE_PROMPT_GSM8K,
    "strategyqa":  JUDGE_PROMPT_SQA,
    "math":        JUDGE_PROMPT_MATH,
}


# ── argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="TAPR GRPO Training")
    parser.add_argument(
        "--task", type=str, choices=["gsm8k", "strategyqa", "math"], required=True,
        help="Training task: gsm8k or strategyqa"
    )
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
        help="Total training steps. Default: 400"
    )
    parser.add_argument(
        "--num_generations", type=int, default=NUM_GENERATIONS,
        help="GRPO rollouts per prompt. Reduce to 4 if OOM."
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
            "Run 5 steps on 20 problems to verify full pipeline. "
            "Always run before committing to a full training run."
        )
    )
    parser.add_argument(
        "--judge_model", type=str, default=JUDGE_MODEL,
        help="Judge model to use. Default: Qwen/Qwen2.5-7B-Instruct. "
             "For math task use: Qwen/Qwen2.5-Math-7B-Instruct"
    )
    parser.add_argument(
        "--seed", type=int, default=RANDOM_SEED,
        help=(
            "Random seed for data shuffling, the StrategyQA train/eval split, and "
            "GRPO rollout sampling. Default: 42 (all reported results). Conditions "
            "you intend to compare must share a seed, or they will not be evaluated "
            "on the same held-out problems."
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


# ── dataset loading ───────────────────────────────────────────────────────────

def count_steps(text: str) -> int:
    """Count steps in a solution/chain as difficulty proxy."""
    lines = [l.strip() for l in text.split("\n")
             if l.strip()
             and not l.strip().startswith("####")
             and not l.lower().strip().startswith("answer:")]
    return len(lines)


def format_prompt(question: str, tokenizer, task: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPTS[task]},
        {"role": "user",   "content": question},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def load_gsm8k(tokenizer, split: str = "train"):
    """Load GSM8K with easy-to-hard curriculum by step count."""
    print(f"\nLoading GSM8K ({split} split)...")
    ds = load_dataset("openai/gsm8k", "main", split=split)

    records = []
    for item in ds:
        gt_num = extract_answer_gsm8k(item["answer"])
        if gt_num is None:
            continue
        records.append({
            "question":  item["question"],
            "answer":    item["answer"],
            "gt_num":    gt_num,
            "n_steps":   count_steps(item["answer"]),
        })

    if split == "train":
        records.sort(key=lambda x: x["n_steps"])
        print(f"Curriculum: {records[0]['n_steps']} steps → {records[-1]['n_steps']} steps")

    for r in records:
        r["prompt"] = format_prompt(r["question"], tokenizer, "gsm8k")

    print(f"Loaded {len(records)} problems")
    return records


# StrategyQA is fetched from GitHub rather than the HF hub. Pinned to the commit
# used for every reported run so the training data cannot silently change or
# disappear from under the results. This path has exactly one commit in the
# upstream repository ("Official code release", 2021-01-16), so the pin is a
# guarantee rather than a version choice.
SQA_COMMIT    = "41190574949c1ab5e523a73b00c2f0bb9f78019f"
SQA_TRAIN_URL = (
    f"https://raw.githubusercontent.com/eladsegal/strategyqa/{SQA_COMMIT}"
    "/data/strategyqa/train.json"
)
SQA_EXPECTED_N = 2061   # records in the pinned file → 1854 train / 207 eval
                        # (matches results/strategyqa/eval_summary.json)


def load_strategyqa(tokenizer):
    """
    Load StrategyQA from the pinned upstream commit.

    Only the train split is used. The official test split ships without answer
    labels, so evaluation uses a held-out 10% of train (see load_task_data).
    Records are shuffled — unlike GSM8K there is no natural difficulty proxy to
    build a curriculum from — which means the eval split depends on the seed.
    """
    print("\nLoading StrategyQA (pinned GitHub commit "
          f"{SQA_COMMIT[:7]})...")

    r = requests.get(SQA_TRAIN_URL, timeout=30)
    r.raise_for_status()
    data = r.json()

    if len(data) != SQA_EXPECTED_N:
        print(f"  WARNING: expected {SQA_EXPECTED_N} records at this commit, "
              f"got {len(data)}. The pin may no longer resolve to the same file; "
              f"results are not comparable to the reported runs.")

    records = []
    for item in data:
        gt = "yes" if item["answer"] else "no"
        records.append({
            "question":  item["question"],
            "gt_answer": gt,
            "n_steps":   random.randint(3, 5),  # placeholder for curriculum
        })

    random.shuffle(records)

    for r in records:
        r["prompt"] = format_prompt(r["question"], tokenizer, "strategyqa")

    print(f"Loaded {len(records)} problems")
    return records


def load_task_data(tokenizer, task: str):
    """Route to correct data loader based on task."""
    if task == "gsm8k":
        train = load_gsm8k(tokenizer, split="train")
        eval_ = load_gsm8k(tokenizer, split="test")
    elif task == "strategyqa":
        all_records = load_strategyqa(tokenizer)
        # 90/10 train/eval split
        split_idx = int(len(all_records) * 0.9)
        train = all_records[:split_idx]
        eval_ = all_records[split_idx:]
        print(f"Train: {len(train)} | Eval: {len(eval_)}")
    elif task == "math":
        from datasets import load_dataset as _ld
        all_records = load_math_train(tokenizer)
        # Use MATH-500 test set for eval (held-out benchmark)
        ds_eval = _ld("HuggingFaceH4/MATH-500", split="test")
        eval_ = [
            {
                "question":  item["problem"],
                "gt_answer": item["answer"],
                "prompt":    format_prompt(item["problem"], tokenizer, "math"),
            }
            for item in ds_eval
        ]
        train = all_records
        print(f"Train: {len(train)} | Eval (MATH-500): {len(eval_)}")
    else:
        raise ValueError(f"Unknown task: {task}")
    return train, eval_


def to_hf_dataset(records: list, task: str) -> Dataset:
    """Convert records to HuggingFace Dataset for GRPOTrainer."""
    if task == "gsm8k":
        return Dataset.from_list([
            {"prompt": r["prompt"], "question": r["question"], "gt_num": r["gt_num"]}
            for r in records
        ])
    elif task in ("strategyqa", "math"):
        return Dataset.from_list([
            {"prompt": r["prompt"], "question": r["question"], "gt_answer": r["gt_answer"]}
            for r in records
        ])


# ── math-specific helpers ────────────────────────────────────────────────────

def normalize_math(text: str) -> Optional[str]:
    """Normalize a MATH answer: strip LaTeX, convert fractions, numeric compare."""
    # Extract from \boxed{...} with nested brace support
    idx = text.find(r"\boxed{")
    if idx != -1:
        start = idx + len(r"\boxed{")
        depth = 1
        for i, ch in enumerate(text[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    text = text[start:i]
                    break
    # Extract from ####
    hashed = re.search(r"####\s*(.+)", text)
    if hashed:
        text = hashed.group(1).strip()
    # Strip LaTeX formatting
    text = re.sub(r"\\text\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\left|\\right", "", text)
    text = re.sub(r"\\,|\\!|\\:|\\;", "", text)
    text = re.sub(r"\$", "", text)
    text = re.sub(r"\\%", "%", text)

    def frac_to_decimal(m):
        try:
            return str(round(float(m.group(1)) / float(m.group(2)), 6))
        except (ValueError, ZeroDivisionError):
            return f"{m.group(1)}/{m.group(2)}"

    text = re.sub(r"\\frac\{([^}]*)\}\{([^}]*)\}", frac_to_decimal, text)
    text = text.strip()
    try:
        val = float(text.replace(",", "").replace(" ", ""))
        return str(round(val, 6))
    except (ValueError, TypeError):
        return text.lower().strip() or None


def r_outcome_math(completion: str, gt_answer: str) -> float:
    """Binary correctness for MATH. Normalize both sides and compare."""
    if not gt_answer:
        return 0.0
    pred = normalize_math(completion)
    gt   = normalize_math(gt_answer)
    if not pred or not gt:
        return 0.0
    try:
        return 1.0 if abs(float(pred) - float(gt)) < 0.001 else 0.0
    except (ValueError, TypeError):
        return 1.0 if pred == gt else 0.0


def _parse_level(raw) -> int:
    """Handle level stored as int (1) or string ('Level 1')."""
    if isinstance(raw, int):
        return raw
    m = re.search(r"\d+", str(raw))
    return int(m.group()) if m else 3


MATH_SUBJECTS = [
    "algebra", "counting_and_probability", "geometry",
    "intermediate_algebra", "number_theory", "prealgebra", "precalculus",
]


def load_math_train(tokenizer):
    """
    Load MATH training set from EleutherAI/hendrycks_math.
    Loads all 7 subjects and combines them.
    Level 1-5 curriculum: easiest first.
    Answer extracted from \\boxed{} in solution field.
    """
    print("\nLoading MATH training set (EleutherAI/hendrycks_math)...")

    def extract_boxed(solution: str) -> str:
        """Extract answer from \\boxed{} with nested brace support."""
        idx = solution.find(r"\boxed{")
        if idx == -1:
            return ""
        start = idx + len(r"\boxed{")
        depth = 1
        for i, ch in enumerate(solution[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return solution[start:i].strip()
        return ""

    records = []
    for subject in MATH_SUBJECTS:
        try:
            ds = load_dataset("EleutherAI/hendrycks_math", subject, split="train")
            for item in ds:
                prob = item.get("problem", "").strip()
                ans  = extract_boxed(item.get("solution", ""))
                lev  = _parse_level(item.get("level", 3))
                if prob and ans:
                    records.append({
                        "question":  prob,
                        "gt_answer": ans,
                        "level":     lev,
                        "subject":   subject,
                    })
            print(f"  {subject}: {sum(1 for r in records if r.get('subject')==subject)} problems")
        except Exception as e:
            print(f"  {subject} failed: {e}")

    if not records:
        raise RuntimeError("Could not load any MATH training data.")

    records.sort(key=lambda x: x["level"])
    print(f"Total: {len(records)} problems | Curriculum: Level {records[0]['level']} -> {records[-1]['level']}")

    for r in records:
        r["prompt"] = format_prompt(r["question"], tokenizer, "math")
    return records


# ── answer extraction ──────────────────────────────────────────────────────────

def extract_answer_gsm8k(text: str) -> Optional[str]:
    """Extract numeric answer after #### marker."""
    match = re.search(r"####\s*([+-]?\d+(?:,\d+)*(?:\.\d+)?)", text)
    if match:
        return match.group(1).replace(",", "")
    nums = re.findall(r"[+-]?\d+(?:\.\d+)?", text)
    return nums[-1] if nums else None


def extract_answer_sqa(text: str) -> Optional[str]:
    """Extract yes/no from model output. Uses last occurrence."""
    lines = [l.strip().lower() for l in text.strip().split("\n") if l.strip()]

    # Check for "Answer: yes/no" format first
    for line in reversed(lines):
        if line.startswith("answer:"):
            rest = line.replace("answer:", "").strip()
            if "yes" in rest:
                return "yes"
            if "no" in rest:
                return "no"

    # Fallback: last standalone yes/no line
    for line in reversed(lines):
        if line in ("yes", "no"):
            return line
        if line.startswith("yes"):
            return "yes"
        if line.startswith("no"):
            return "no"

    # Last resort: find last yes/no in full text
    text_lower = text.lower()
    last_yes = text_lower.rfind("yes")
    last_no  = text_lower.rfind("no")
    if last_yes == -1 and last_no == -1:
        return None
    return "yes" if last_yes > last_no else "no"


# ── outcome rewards ───────────────────────────────────────────────────────────

def r_outcome_gsm8k(completion: str, gt_num: str) -> float:
    """Binary correctness. 1.0 if numeric answer matches GT."""
    pred = extract_answer_gsm8k(completion)
    if pred is None or gt_num is None:
        return 0.0
    try:
        return 1.0 if abs(float(pred) - float(gt_num)) < 0.01 else 0.0
    except (ValueError, TypeError):
        return 0.0


def r_outcome_sqa(completion: str, gt_answer: str) -> float:
    """Binary correctness. 1.0 if yes/no matches GT."""
    if gt_answer is None:
        return 0.0
    pred = extract_answer_sqa(completion)
    if pred is None:
        return 0.0
    return 1.0 if pred == gt_answer.lower().strip() else 0.0


def compute_r_outcome(completion: str, task: str, **kwargs) -> float:
    """Route to correct outcome reward based on task."""
    if task == "gsm8k":
        return r_outcome_gsm8k(completion, kwargs.get("gt_num"))
    elif task == "strategyqa":
        return r_outcome_sqa(completion, kwargs.get("gt_answer"))
    elif task == "math":
        return r_outcome_math(completion, kwargs.get("gt_answer"))
    raise ValueError(f"Unknown task: {task}")


# ── step parsing ───────────────────────────────────────────────────────────────

def parse_steps(text: str, task: str = "gsm8k") -> list:
    """Split solution into individual reasoning steps, excluding answer lines."""
    lines = text.strip().split("\n")
    steps = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Skip answer markers
        if line.startswith("####"):
            continue
        if task == "strategyqa" and line.lower().startswith("answer:"):
            continue
        steps.append(line)
    return steps


# ── judge scoring + R_transition ──────────────────────────────────────────────

def sample_window_size() -> int:
    return random.choices(WINDOW_SIZES, weights=WINDOW_PROBS, k=1)[0]


def parse_judge_score(text: str, task: str = "gsm8k") -> float:
    """
    Parse judge response into a [0, 1] score.
    Both gsm8k and strategyqa use SCORE: X.XX format.
    """
    match = re.search(r"SCORE:\s*([0-9]*\.?[0-9]+)", text)
    if match:
        return max(0.0, min(1.0, float(match.group(1))))
    nums = re.findall(r"0\.\d+|1\.0|0\.0", text)
    return float(nums[0]) if nums else 0.5


def build_judge_prompt(task: str, problem: str, window_steps: str) -> str:
    """Build the judge prompt for a given task and window."""
    if task in JUDGE_PROMPTS:
        return JUDGE_PROMPTS[task].format(
            problem=problem, window_steps=window_steps
        )
    raise ValueError(f"Unknown task: {task}")

def score_all_completions_batched(judge_model, judge_tok, completions: list,
                                  questions: list, task: str,
                                  gt_answers: list = None) -> list:
    """
    Score all completions in ONE batched judge call.
    Returns list of R_transition floats, one per completion.
    ~10x faster than sequential scoring on A100.
    """
    if gt_answers is None:
        gt_answers = [""] * len(completions)

    all_texts = []
    per_completion_indices = []

    for completion, question, gt_ans in zip(completions, questions, gt_answers):
        steps = parse_steps(completion, task)
        if len(steps) < 2:
            per_completion_indices.append([])
            continue

        indices = []
        for i in range(len(steps)):
            W = sample_window_size()
            window = steps[i: i + W]
            if len(window) < 2:
                continue
            steps_text = "\n".join([f"Step {j+1}: {s}" for j, s in enumerate(window)])
            prompt = build_judge_prompt(task, question, steps_text)
            messages = [{"role": "user", "content": prompt}]
            text = judge_tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            all_texts.append(text)
            indices.append(len(all_texts) - 1)

        per_completion_indices.append(indices)

    if not all_texts:
        return [0.5] * len(completions)

    judge_tok.padding_side = "left"
    inputs = judge_tok(
        all_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=768,
    ).to(next(judge_model.parameters()).device)

    with torch.no_grad():
        out_ids = judge_model.generate(
            **inputs,
            max_new_tokens=JUDGE_TOKENS,
            do_sample=False,
            pad_token_id=judge_tok.eos_token_id,
        )

    input_len = inputs["input_ids"].shape[1]
    all_scores = []
    for out in out_ids:
        new_tokens = out[input_len:]
        response = judge_tok.decode(new_tokens, skip_special_tokens=True).strip()
        all_scores.append(parse_judge_score(response, task))

    result = []
    for indices in per_completion_indices:
        if not indices:
            result.append(0.5)
        else:
            result.append(float(np.mean([all_scores[i] for i in indices])))

    return result


def r_transition(judge_model, judge_tok, completion: str,
                 question: str, task: str, gt_answer: str = "") -> float:
    """Single-completion wrapper — kept for preflight check compatibility."""
    return score_all_completions_batched(
        judge_model, judge_tok, [completion], [question], task, [gt_answer]
    )[0]


# ── reward function ───────────────────────────────────────────────────────────

class TAPRReward:
    """
    Reward function for GRPO. Task-aware.
    mode='baseline': R_outcome only
    mode='tapr':     R_outcome + lambda_val * R_transition

    Dataset columns used:
      gsm8k:       'question', 'gt_num'
      strategyqa:  'question', 'gt_answer'
    """

    def __init__(self, mode: str, lambda_val: float, task: str,
                 judge_model=None, judge_tok=None):
        self.mode       = mode
        self.lambda_val = lambda_val
        self.task       = task
        self.judge      = judge_model
        self.judge_tok  = judge_tok
        self.__name__   = f"tapr_reward_{mode}_{task}"

        assert mode in ("baseline", "tapr")
        if mode == "tapr":
            assert judge_model is not None, "Judge required for tapr mode"

    def __call__(self, completions: list, **kwargs) -> list:
        questions = kwargs.get("question", [""] * len(completions))

        # Task-specific outcome reward
        if self.task == "gsm8k":
            gt_vals = kwargs.get("gt_num", [None] * len(completions))
            r_outcomes = [r_outcome_gsm8k(c, g) for c, g in zip(completions, gt_vals)]
            gt_answers = [""] * len(completions)
        elif self.task == "strategyqa":
            gt_vals = kwargs.get("gt_answer", [None] * len(completions))
            r_outcomes = [r_outcome_sqa(c, g) for c, g in zip(completions, gt_vals)]
            gt_answers = [g or "" for g in gt_vals]
        elif self.task == "math":
            gt_vals = kwargs.get("gt_answer", [None] * len(completions))
            r_outcomes = [r_outcome_math(c, g) for c, g in zip(completions, gt_vals)]
            gt_answers = [""] * len(completions)
        else:
            raise ValueError(f"Unknown task: {self.task}")

        if self.mode == "tapr":
            r_transitions = score_all_completions_batched(
                self.judge, self.judge_tok,
                completions, questions, self.task, gt_answers
            )
            rewards = [float(ro + self.lambda_val * rt)
                       for ro, rt in zip(r_outcomes, r_transitions)]
        else:
            rewards = [float(ro) for ro in r_outcomes]

        return rewards


# ── evaluation ────────────────────────────────────────────────────────────────

def evaluate(model, tokenizer, eval_records: list, task: str,
             n: int = N_EVAL_SAMPLES) -> dict:
    """
    Evaluate student model on held-out problems.
    Greedy decoding, exact-match accuracy.
    Task-aware: numeric match for GSM8K, yes/no match for StrategyQA.
    """
    model.eval()
    subset = eval_records[:n]
    correct, total, unparseable = 0, 0, 0

    for item in subset:
        inputs = tokenizer(
            item["prompt"],
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(next(model.parameters()).device)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=MAX_COMP_LENGTH,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        completion = tokenizer.decode(new_tokens, skip_special_tokens=True)

        if task == "gsm8k":
            correct += r_outcome_gsm8k(completion, item["gt_num"])
        elif task == "strategyqa":
            pred = extract_answer_sqa(completion)
            if pred is None:
                unparseable += 1
            correct += r_outcome_sqa(completion, item["gt_answer"])
        elif task == "math":
            correct += r_outcome_math(completion, item["gt_answer"])
        total += 1

    acc = correct / total if total > 0 else 0.0
    result = {"accuracy": round(acc, 4), "correct": int(correct), "total": total}
    if task == "strategyqa":
        result["unparseable"] = unparseable
    model.train()
    return result


# ── checkpoint resume ─────────────────────────────────────────────────────────

def find_latest_checkpoint(output_dir: str) -> Optional[str]:
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
                    task: str, judge=None, judge_tok=None) -> bool:
    """
    Run before any training. Tests full pipeline on one sample.
    Catches: output format issues, reward function mismatches, judge errors.
    """
    print("\n" + "="*55)
    print(f"PREFLIGHT CHECK  [{task.upper()}]")
    print("="*55)
    passed = True

    # ── 1. Student generation ─────────────────────────────────────────────────
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
                **inputs, max_new_tokens=128, do_sample=False,
                pad_token_id=student_tok.eos_token_id,
            )
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        sample_completion = student_tok.decode(new_tokens, skip_special_tokens=True)
        print(f"  Generated {len(sample_completion)} chars")
        print(f"  Preview: {sample_completion[:120].strip()}")
    except Exception as e:
        raise RuntimeError(f"Student generation failed: {e}") from e
    print("  ✓ OK")

    # ── 2. Output format check ────────────────────────────────────────────────
    print(f"\n[2/4] Output format check...")
    if task in ("gsm8k", "math"):
        if "####" in sample_completion:
            print(f"  ✓ OK — #### found, extracted: {extract_answer_gsm8k(sample_completion)}")
        else:
            print("  ⚠ WARNING — no #### in output. R_outcome will be 0.")
            passed = False
    elif task == "strategyqa":
        pred = extract_answer_sqa(sample_completion)
        if pred:
            print(f"  ✓ OK — yes/no extracted: '{pred}'")
        else:
            print("  ⚠ WARNING — could not extract yes/no. R_outcome will be 0.")
            passed = False

    # ── 3. Reward function ────────────────────────────────────────────────────
    print("\n[3/4] Reward function...")
    try:
        kwargs = {"question": [sample_record["question"]]}
        if task == "gsm8k":
            kwargs["gt_num"] = [sample_record["gt_num"]]
        elif task == "strategyqa":
            kwargs["gt_answer"] = [sample_record["gt_answer"]]

        dummy_rewards = reward_fn(completions=[sample_completion], **kwargs)
        assert isinstance(dummy_rewards, list)
        assert len(dummy_rewards) == 1
        assert isinstance(dummy_rewards[0], float)
        print(f"  R_final = {dummy_rewards[0]:.4f}")
    except Exception as e:
        raise RuntimeError(f"Reward function failed: {e}") from e
    print("  ✓ OK")

    # ── 4. Judge check (tapr mode only) ───────────────────────────────────────
    print("\n[4/4] Judge check...")
    if judge is not None:
        try:
            steps = parse_steps(sample_completion, task)
            if len(steps) >= 2:
                score = r_transition(judge, judge_tok,
                                     sample_completion,
                                     sample_record["question"],
                                     task)
                assert 0.0 <= score <= 1.0, f"Score out of range: {score}"
                print(f"  Judge score: {score:.4f}")
            else:
                print("  Too few steps — skipped (not a failure)")
        except Exception as e:
            raise RuntimeError(f"Judge scoring failed: {e}") from e
        print("  ✓ OK")
    else:
        print("  Skipped (baseline mode)")

    print("\n" + "="*55)
    print("✓ PREFLIGHT PASSED" if passed else "⚠ PREFLIGHT PASSED WITH WARNINGS")
    print("="*55)
    return passed


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Re-seed from the CLI value. Module-level seeding already ran at import
    # with the default; this makes --seed effective and must happen before any
    # data loading, since the StrategyQA split depends on it.
    set_global_seed(args.seed)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required.")

    total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"\nGPU:    {torch.cuda.get_device_name(0)}")
    print(f"VRAM:   {total_vram:.1f} GB")
    print(f"Task:   {args.task.upper()}")
    print(f"Mode:   {args.mode.upper()}")
    print(f"Seed:   {args.seed}")
    print(f"Lambda: {args.lambda_val if args.mode == 'tapr' else 0.0}")
    if args.mode == "tapr":
        print(f"Judge:  {args.judge_model}")
    print(f"Output: {args.output_dir}")
    if args.smoke_test:
        print("*** SMOKE TEST MODE — 5 steps, 20 problems ***")
    else:
        print(f"Steps:  {args.max_steps}")

    os.makedirs(args.output_dir, exist_ok=True)

    # ── load student ──────────────────────────────────────────────────────────
    student, student_tok = load_student(STUDENT_MODEL)

    # ── load judge (tapr mode only) ───────────────────────────────────────────
    judge, judge_tok = None, None
    if args.mode == "tapr":
        judge, judge_tok = load_judge(args.judge_model)
        print(f"\nTotal VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB / {total_vram:.1f} GB")

    # ── load datasets ─────────────────────────────────────────────────────────
    train_records, eval_records = load_task_data(student_tok, args.task)

    if args.smoke_test:
        train_records = train_records[:20]
        eval_records  = eval_records[:10]
        args.max_steps = 5
        print(f"\nSmoke test: {len(train_records)} train, {len(eval_records)} eval")

    train_dataset = to_hf_dataset(train_records, args.task)

    # ── reward function ───────────────────────────────────────────────────────
    reward_fn = TAPRReward(
        mode        = args.mode,
        lambda_val  = args.lambda_val,
        task        = args.task,
        judge_model = judge,
        judge_tok   = judge_tok,
    )

    # ── preflight ─────────────────────────────────────────────────────────────
    preflight_check(
        student       = student,
        student_tok   = student_tok,
        reward_fn     = reward_fn,
        sample_record = train_records[0],
        task          = args.task,
        judge         = judge,
        judge_tok     = judge_tok,
    )

    # ── GRPO config ───────────────────────────────────────────────────────────
    bf16_supported = torch.cuda.is_bf16_supported()
    grpo_config = GRPOConfig(
        output_dir                  = args.output_dir,
        num_generations             = args.num_generations,
        max_completion_length       = MAX_COMP_LENGTH,
        learning_rate               = args.lr,
        per_device_train_batch_size = args.batch_size,
        gradient_accumulation_steps = GRAD_ACCUM,
        max_steps                   = args.max_steps,
        eval_steps                  = EVAL_STEPS,
        save_steps                  = SAVE_STEPS,
        logging_steps               = LOGGING_STEPS,
        beta                        = KL_COEF,
        temperature                 = TEMPERATURE,
        bf16                        = bf16_supported,
        fp16                        = not bf16_supported,
        remove_unused_columns       = False,
        seed                        = args.seed,
        report_to                   = "none",
    )

    # ── trainer ───────────────────────────────────────────────────────────────
    trainer = GRPOTrainer(
        model            = student,
        reward_funcs     = [reward_fn],
        args             = grpo_config,
        train_dataset    = train_dataset,
        processing_class = student_tok,
    )

    # ── pre-training eval ─────────────────────────────────────────────────────
    print("\n" + "="*55)
    print("PRE-TRAINING EVAL")
    print("="*55)
    pre_eval = evaluate(student, student_tok, eval_records, args.task)
    print(f"Accuracy: {pre_eval['accuracy']:.4f} ({pre_eval['correct']}/{pre_eval['total']})")

    # ── training ──────────────────────────────────────────────────────────────
    print("\n" + "="*55)
    print(f"TRAINING — {args.task.upper()} / {args.mode.upper()}")
    print("="*55)

    checkpoint = find_latest_checkpoint(args.output_dir)
    if checkpoint:
        print(f"Resuming from: {checkpoint}")
        trainer.train(resume_from_checkpoint=checkpoint)
    else:
        print("Starting fresh.")
        trainer.train()

    # ── post-training eval ────────────────────────────────────────────────────
    print("\n" + "="*55)
    print("POST-TRAINING EVAL")
    print("="*55)
    post_eval = evaluate(student, student_tok, eval_records, args.task)
    print(f"Accuracy: {post_eval['accuracy']:.4f} ({post_eval['correct']}/{post_eval['total']})")
    print(f"Delta:    {post_eval['accuracy'] - pre_eval['accuracy']:+.4f}")

    # ── save ──────────────────────────────────────────────────────────────────
    final_dir = os.path.join(args.output_dir, "final")
    student.save_pretrained(final_dir)
    student_tok.save_pretrained(final_dir)
    print(f"\nModel saved to: {final_dir}")

    summary = {
        "task":          args.task,
        "mode":          args.mode,
        "seed":          args.seed,
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
    print(f"Summary saved to: {summary_path}")

    print("\n" + "="*55)
    print("DONE")
    print(f"Task:  {args.task.upper()}")
    print(f"Mode:  {args.mode.upper()}")
    print(f"Pre:   {pre_eval['accuracy']:.4f}")
    print(f"Post:  {post_eval['accuracy']:.4f}")
    print(f"Delta: {post_eval['accuracy'] - pre_eval['accuracy']:+.4f}")
    print("="*55)


if __name__ == "__main__":
    main()
