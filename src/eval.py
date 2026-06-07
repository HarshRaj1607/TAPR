"""
TAPR Evaluation Script
======================
PI: Harsh | Executor: Claude
Phase 2 — Samsung EnnovateX AX Hackathon

Evaluates three models across three benchmarks:
  Models:     Zero-shot | Baseline trained | TAPR trained
  Benchmarks: GSM8K (1319) | MATH-500 (500) | StrategyQA (~490)

MATH-500 answer matching: string normalization + LLM judge fallback
StrategyQA answer matching: yes/no extraction

Output: eval_results.json + printed comparison table

Usage:
  # All three models together
  python eval.py \
    --baseline_dir /content/drive/MyDrive/tapr_baseline/final \
    --tapr_dir     /content/drive/MyDrive/tapr_run/final \
    --output_file  /content/drive/MyDrive/eval_results.json

  # Single model (parallel sessions)
  python eval.py --model zeroshot  --output_file /content/drive/MyDrive/eval_zeroshot.json
  python eval.py --model baseline  --output_file /content/drive/MyDrive/eval_baseline.json
  python eval.py --model tapr      --output_file /content/drive/MyDrive/eval_tapr.json

  # Limit problems per benchmark (for quick smoke check)
  python eval.py --model tapr --limit 100 --output_file /content/drive/MyDrive/eval_tapr.json

  # Skip LLM judge fallback for MATH-500 (faster)
  python eval.py --model tapr --skip_judge --output_file /content/drive/MyDrive/eval_tapr.json
"""

import argparse
import json
import os
import re
import random
import numpy as np
import torch
from typing import Optional

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

BASE_MODEL   = "Qwen/Qwen2.5-3B-Instruct"
JUDGE_MODEL  = "Qwen/Qwen2.5-7B-Instruct"
MAX_NEW_TOKENS = 512

# ── system prompts ─────────────────────────────────────────────────────────────

SYSTEM_PROMPTS = {
    "gsm8k": (
        "You are a mathematical reasoning assistant. "
        "Solve the problem step by step. Write one operation or deduction per line. "
        "End your solution with #### [final numeric answer]."
    ),
    "math500": (
        "You are a mathematical reasoning assistant. "
        "Solve the problem step by step. Show all work clearly. "
        "End your solution with #### [final answer]."
    ),
    "strategyqa": (
        "You are a reasoning assistant. "
        "Think through the question step by step using facts you know. "
        "End your response with exactly 'yes' or 'no' on the final line."
    ),
}

# ── argument parsing ───────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="TAPR Evaluation")
    parser.add_argument("--baseline_dir", type=str,
                        default="/content/drive/MyDrive/tapr_baseline/final",
                        help="Path to baseline trained model adapter")
    parser.add_argument("--tapr_dir", type=str,
                        default="/content/drive/MyDrive/tapr_run/final",
                        help="Path to TAPR trained model adapter")
    parser.add_argument("--output_file", type=str,
                        default="/content/drive/MyDrive/eval_results.json",
                        help="Path to save evaluation results")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit problems per benchmark (default: no limit, use full split)")
    parser.add_argument("--model", type=str,
                        choices=["all", "zeroshot", "baseline", "tapr"], default="all",
                        help="Which model to evaluate (default: all). Use for parallel sessions.")
    parser.add_argument("--skip_judge", action="store_true",
                        help="Skip LLM judge fallback for MATH-500 (faster, less accurate)")
    return parser.parse_args()


# ── model loading ──────────────────────────────────────────────────────────────

def get_bnb_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )


def load_model(adapter_path: Optional[str] = None, label: str = ""):
    """
    Load Qwen2.5-3B-Instruct in 4-bit.
    If adapter_path is given, load LoRA adapter on top (trained model).
    If None, return zero-shot base model.
    """
    print(f"\nLoading: {label}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=get_bnb_config(),
        device_map="auto",
        trust_remote_code=True,
    )

    if adapter_path:
        print(f"  Loading adapter from: {adapter_path}")
        model = PeftModel.from_pretrained(base, adapter_path)
    else:
        model = base

    model.eval()
    vram = torch.cuda.memory_allocated() / 1e9
    print(f"  VRAM used: {vram:.1f} GB")
    return model, tokenizer


def load_judge():
    """Load Qwen2.5-7B-Instruct as frozen math equivalence judge."""
    print(f"\nLoading judge: {JUDGE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(JUDGE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        JUDGE_MODEL,
        quantization_config=get_bnb_config(),
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    print(f"  Judge loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB")
    return model, tokenizer


def unload_model(model):
    """Free GPU memory after evaluating a model."""
    del model
    torch.cuda.empty_cache()


# ── dataset loading ────────────────────────────────────────────────────────────

def load_gsm8k_test():
    print("\nLoading GSM8K test split...")
    ds = load_dataset("openai/gsm8k", "main", split="test")
    records = [{"question": item["question"], "answer": item["answer"]} for item in ds]
    print(f"  {len(records)} problems")
    return records


def load_math500():
    print("\nLoading MATH-500...")
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    records = [
        {
            "question": item["problem"],
            "answer":   item["answer"],
            "subject":  item.get("subject", ""),
            "level":    item.get("level", ""),
        }
        for item in ds
    ]
    print(f"  {len(records)} problems")
    return records


def load_strategyqa():
    """
    Load StrategyQA via direct GitHub download (wics/strategy-qa HF loader is broken).
    Falls back from test split to last 490 of train if test unavailable.
    """
    import requests
    print("\nLoading StrategyQA (direct GitHub download)...")

    test_url  = "https://raw.githubusercontent.com/eladsegal/strategyqa/main/data/strategyqa/test.json"
    train_url = "https://raw.githubusercontent.com/eladsegal/strategyqa/main/data/strategyqa/train.json"

    try:
        response = requests.get(test_url, timeout=30)
        response.raise_for_status()
        data = response.json()
        print(f"  Loaded test split: {len(data)} problems")
    except Exception as e:
        print(f"  Test split failed ({e}), falling back to train...")
        response = requests.get(train_url, timeout=30)
        response.raise_for_status()
        data = response.json()[-490:]
        print(f"  Using last 490 of train split")

    records = []
    for item in data:
        gt = "yes" if item["answer"] else "no"
        records.append({"question": item["question"], "answer": gt})

    print(f"  {len(records)} problems loaded")
    return records


# ── generation ─────────────────────────────────────────────────────────────────

def generate(model, tokenizer, question: str, benchmark: str) -> str:
    """Generate a solution for one problem using greedy decoding."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPTS[benchmark]},
        {"role": "user",   "content": question},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(next(model.parameters()).device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ── answer extraction ──────────────────────────────────────────────────────────

def extract_numeric(text: str) -> Optional[str]:
    """Extract number after #### marker."""
    match = re.search(r"####\s*([+-]?\d+(?:,\d+)*(?:\.\d+)?)", text)
    if match:
        return match.group(1).replace(",", "")
    nums = re.findall(r"[+-]?\d+(?:\.\d+)?", text)
    return nums[-1] if nums else None


def r_outcome_gsm8k(completion: str, gt_answer: str) -> bool:
    """GSM8K: extract number after ####, numeric comparison."""
    gt_num = extract_numeric(gt_answer)
    pred   = extract_numeric(completion)
    if pred is None or gt_num is None:
        return False
    try:
        return abs(float(pred) - float(gt_num)) < 0.01
    except (ValueError, TypeError):
        return False


def normalize_math(text: str) -> str:
    """
    Normalize a MATH answer for comparison.
    Strips LaTeX, converts fractions to decimals where possible.
    """
    # Extract from \boxed{...} if present
    boxed = re.search(r"\\boxed\{([^}]*)\}", text)
    if boxed:
        text = boxed.group(1)

    # Extract from #### if present
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

    # Convert \frac{a}{b} → decimal where both are numeric
    def frac_to_decimal(m):
        try:
            return str(round(float(m.group(1)) / float(m.group(2)), 6))
        except (ValueError, ZeroDivisionError):
            return f"{m.group(1)}/{m.group(2)}"

    text = re.sub(r"\\frac\{([^}]*)\}\{([^}]*)\}", frac_to_decimal, text)
    text = text.strip()

    # Try numeric
    try:
        val = float(text.replace(",", "").replace(" ", ""))
        return str(round(val, 6))
    except (ValueError, TypeError):
        return text.lower().strip()


def r_outcome_math500_normalized(completion: str, gt_answer: str) -> Optional[bool]:
    """
    MATH-500: normalize both answers and compare.
    Returns True/False if confident, None if uncertain (→ use LLM judge).
    """
    pred_norm = normalize_math(completion)
    gt_norm   = normalize_math(gt_answer)

    if not pred_norm or not gt_norm:
        return None

    # Numeric comparison
    try:
        pred_val = float(pred_norm.replace(",", ""))
        gt_val   = float(gt_norm.replace(",", ""))
        return abs(pred_val - gt_val) < 0.001
    except (ValueError, TypeError):
        pass

    # Exact string match after normalization
    if pred_norm == gt_norm:
        return True

    # Close but not exact — uncertain → LLM judge
    return None


def judge_math_equiv(judge_model, judge_tok,
                     problem: str, model_ans: str, gt_ans: str) -> bool:
    """
    LLM judge fallback: ask Qwen2.5-7B if model answer is equivalent to GT.
    Used when string normalization is inconclusive.
    """
    prompt = (
        f"Are these two answers to the following math problem equivalent?\n\n"
        f"Problem: {problem[:300]}\n"
        f"Ground truth answer: {gt_ans}\n"
        f"Model answer: {model_ans}\n\n"
        f"Respond with only 'yes' or 'no'."
    )
    messages = [{"role": "user", "content": prompt}]
    text = judge_tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = judge_tok([text], return_tensors="pt").to(
        next(judge_model.parameters()).device
    )
    with torch.no_grad():
        out = judge_model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
            pad_token_id=judge_tok.eos_token_id,
        )
    response = judge_tok.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip().lower()
    return "yes" in response


def extract_yes_no(text: str) -> Optional[str]:
    """Extract yes/no from model output. Uses last occurrence."""
    lines = [l.strip().lower() for l in text.strip().split("\n") if l.strip()]

    # Check last non-empty line first
    for line in reversed(lines):
        if line in ("yes", "no"):
            return line
        if line.startswith("yes"):
            return "yes"
        if line.startswith("no"):
            return "no"

    # Scan full text for last occurrence
    text_lower = text.lower()
    last_yes = text_lower.rfind("yes")
    last_no  = text_lower.rfind("no")

    if last_yes == -1 and last_no == -1:
        return None
    if last_yes > last_no:
        return "yes"
    return "no"


# ── benchmark evaluation ───────────────────────────────────────────────────────

def evaluate_gsm8k(model, tokenizer, records: list) -> dict:
    """Evaluate on GSM8K test split. Exact numeric match."""
    correct, total = 0, 0
    results = []

    for i, item in enumerate(records):
        completion = generate(model, tokenizer, item["question"], "gsm8k")
        is_correct = r_outcome_gsm8k(completion, item["answer"])
        correct += int(is_correct)
        total   += 1
        results.append({
            "id": i, "correct": is_correct,
            "pred": extract_numeric(completion),
            "gt":   extract_numeric(item["answer"]),
        })

        if (i + 1) % 100 == 0:
            print(f"  GSM8K [{i+1}/{len(records)}] running acc: {correct/(i+1):.4f}")

    acc = correct / total if total > 0 else 0.0
    return {"accuracy": round(acc, 4), "correct": correct, "total": total, "results": results}


def evaluate_math500(model, tokenizer, records: list,
                     judge_model=None, judge_tok=None) -> dict:
    """
    Evaluate on MATH-500. Normalization + LLM judge fallback.
    Tracks: normalized match, judge calls, judge correct.
    """
    correct, total = 0, 0
    judge_calls, judge_correct = 0, 0
    results = []
    use_judge = judge_model is not None

    for i, item in enumerate(records):
        completion = generate(model, tokenizer, item["question"], "math500")

        norm_result = r_outcome_math500_normalized(completion, item["answer"])

        if norm_result is not None:
            is_correct = norm_result
        elif use_judge:
            # Normalization inconclusive — call LLM judge
            is_correct = judge_math_equiv(
                judge_model, judge_tok,
                item["question"], completion, item["answer"]
            )
            judge_calls += 1
            judge_correct += int(is_correct)
        else:
            # No judge available — treat as incorrect
            is_correct = False

        correct += int(is_correct)
        total   += 1
        results.append({
            "id": i, "subject": item.get("subject", ""),
            "level": item.get("level", ""),
            "correct": is_correct,
            "norm_result": norm_result,
            "used_judge": norm_result is None,
        })

        if (i + 1) % 50 == 0:
            print(f"  MATH-500 [{i+1}/{len(records)}] running acc: {correct/(i+1):.4f} "
                  f"| judge calls: {judge_calls}")

    acc = correct / total if total > 0 else 0.0
    return {
        "accuracy": round(acc, 4), "correct": correct, "total": total,
        "judge_calls": judge_calls, "judge_correct": judge_correct,
        "results": results,
    }


def evaluate_strategyqa(model, tokenizer, records: list) -> dict:
    """Evaluate on StrategyQA. Yes/no extraction, exact match."""
    correct, total, unparseable = 0, 0, 0
    results = []

    for i, item in enumerate(records):
        completion = generate(model, tokenizer, item["question"], "strategyqa")
        pred = extract_yes_no(completion)

        if pred is None:
            is_correct = False
            unparseable += 1
        else:
            is_correct = (pred == item["answer"].lower().strip())

        correct += int(is_correct)
        total   += 1
        results.append({
            "id": i, "correct": is_correct,
            "pred": pred, "gt": item["answer"],
        })

        if (i + 1) % 100 == 0:
            print(f"  StrategyQA [{i+1}/{len(records)}] running acc: {correct/(i+1):.4f} "
                  f"| unparseable: {unparseable}")

    acc = correct / total if total > 0 else 0.0
    return {
        "accuracy": round(acc, 4), "correct": correct, "total": total,
        "unparseable": unparseable, "results": results,
    }


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required.")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    if args.limit:
        print(f"Limit: {args.limit} problems per benchmark")
    else:
        print("Limit: none (full splits)")

    # ── load datasets once (reused across all models) ─────────────────────────
    gsm8k_records   = load_gsm8k_test()
    math500_records = load_math500()
    sqa_records     = load_strategyqa()

    # Apply limit if set
    if args.limit:
        gsm8k_records   = gsm8k_records[:args.limit]
        math500_records = math500_records[:args.limit]
        sqa_records     = sqa_records[:args.limit]

    # ── load judge once (reused across all models for MATH fallback) ──────────
    judge_model, judge_tok = None, None
    if not args.skip_judge:
        judge_model, judge_tok = load_judge()

    # ── model configurations ──────────────────────────────────────────────────
    model_configs = [
        ("Zero-shot", None),
        ("Baseline",  args.baseline_dir),
        ("TAPR",      args.tapr_dir),
    ]

    # Filter to single model if --model flag is set
    if args.model != "all":
        key_map = {"zeroshot": "Zero-shot", "baseline": "Baseline", "tapr": "TAPR"}
        target  = key_map[args.model]
        model_configs = [(n, p) for n, p in model_configs if n == target]
        print(f"\nSingle-model mode: {target}")

    all_results = {}

    for model_name, adapter_path in model_configs:
        print(f"\n{'='*60}")
        print(f"EVALUATING: {model_name}")
        print(f"{'='*60}")

        model, tokenizer = load_model(adapter_path, label=model_name)

        # GSM8K
        print(f"\n[GSM8K] {model_name}...")
        gsm8k_result = evaluate_gsm8k(model, tokenizer, gsm8k_records)
        print(f"  GSM8K accuracy: {gsm8k_result['accuracy']:.4f} "
              f"({gsm8k_result['correct']}/{gsm8k_result['total']})")

        # MATH-500
        print(f"\n[MATH-500] {model_name}...")
        math_result = evaluate_math500(
            model, tokenizer, math500_records, judge_model, judge_tok
        )
        print(f"  MATH-500 accuracy: {math_result['accuracy']:.4f} "
              f"({math_result['correct']}/{math_result['total']}) "
              f"| judge calls: {math_result['judge_calls']}")

        # StrategyQA
        print(f"\n[StrategyQA] {model_name}...")
        sqa_result = evaluate_strategyqa(model, tokenizer, sqa_records)
        print(f"  StrategyQA accuracy: {sqa_result['accuracy']:.4f} "
              f"({sqa_result['correct']}/{sqa_result['total']}) "
              f"| unparseable: {sqa_result['unparseable']}")

        all_results[model_name] = {
            "gsm8k":      gsm8k_result,
            "math500":    math_result,
            "strategyqa": sqa_result,
        }

        # Unload model to free VRAM before loading next
        unload_model(model)
        print(f"\n  VRAM after unload: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    # ── results table ─────────────────────────────────────────────────────────
    evaluated = list(all_results.keys())
    print(f"\n{'='*65}")
    print("RESULTS")
    print(f"{'='*65}")
    print(f"{'Model':<15} {'GSM8K':>10} {'MATH-500':>10} {'StrategyQA':>12}")
    print("-" * 50)

    for model_name in evaluated:
        r = all_results[model_name]
        gsm  = r["gsm8k"]["accuracy"]
        math = r["math500"]["accuracy"]
        sqa  = r["strategyqa"]["accuracy"]
        print(f"{model_name:<15} {gsm:>10.4f} {math:>10.4f} {sqa:>12.4f}")

    # Deltas vs zero-shot (only if zero-shot was evaluated in this run)
    if "Zero-shot" in all_results:
        print("\nDeltas vs Zero-shot:")
        zs = all_results["Zero-shot"]
        for model_name in ["Baseline", "TAPR"]:
            if model_name not in all_results:
                continue
            r  = all_results[model_name]
            dg = r["gsm8k"]["accuracy"]      - zs["gsm8k"]["accuracy"]
            dm = r["math500"]["accuracy"]    - zs["math500"]["accuracy"]
            ds = r["strategyqa"]["accuracy"] - zs["strategyqa"]["accuracy"]
            print(f"  {model_name:<12} GSM8K {dg:+.4f}  MATH-500 {dm:+.4f}  StrategyQA {ds:+.4f}")

    print()

    # ── save ──────────────────────────────────────────────────────────────────
    output = {
        "models": {k: {
            "gsm8k":      {kk: vv for kk, vv in v["gsm8k"].items()      if kk != "results"},
            "math500":    {kk: vv for kk, vv in v["math500"].items()    if kk != "results"},
            "strategyqa": {kk: vv for kk, vv in v["strategyqa"].items() if kk != "results"},
        } for k, v in all_results.items()},
        "config": {
            "base_model":       BASE_MODEL,
            "judge_model":      JUDGE_MODEL if not args.skip_judge else None,
            "baseline_adapter": args.baseline_dir,
            "tapr_adapter":     args.tapr_dir,
            "limit":            args.limit,
            "model_filter":     args.model,
        }
    }

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to: {args.output_file}")


if __name__ == "__main__":
    main()
