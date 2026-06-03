"""
TAPR Judge Quality Test — Qwen2.5-7B-Instruct
===============================================
PI: Harsh | Executor: Claude
Phase 2 — Risk 1 mitigation

Validates that Qwen2.5-7B-Instruct (open-weight judge) achieves
acceptable separation between ground truth and flawed reasoning chains,
before we build the full training loop around it.

Scores 20 problems from tapr_flawed_solutions.json — stratified across
flaw types (5A / 8B / 7C) — using the 5-dimension TAPR judge prompt.

Pass criteria:
  Spearman r >= 0.40
  GT beats Flawed >= 70%

If either criterion fails: STOP. Do not proceed to training. Discuss.

Phase 1 reference (GPT-4.1-mini): r=0.600, 90.3% separation.
If Qwen2.5-7B is significantly below this, flag before continuing.

Output: judge_test_results.json
Next:   If PASS → proceed to train_grpo.py
        If FAIL → diagnose: simplify prompt or fall back to Qwen2.5-14B
"""

import json
import os
import re
import random
import numpy as np
from scipy.stats import spearmanr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ── config ────────────────────────────────────────────────────────────────────
INPUT_FILE      = "tapr_flawed_solutions.json"
OUTPUT_FILE     = "judge_test_results.json"
MODEL_NAME      = "Qwen/Qwen2.5-7B-Instruct"
N_SAMPLES       = 20
RANDOM_SEED     = 42
WINDOW_SIZES    = [2, 3, 5]
WINDOW_PROBS    = [0.1, 0.5, 0.4]     # updated from Phase 1 (was 0.2, 0.6, 0.2)
MAX_NEW_TOKENS  = 150

# Pass thresholds — lower than Phase 1 since Qwen2.5-7B < GPT-4.1-mini
PASS_THRESHOLD_R   = 0.40
PASS_THRESHOLD_SEP = 0.70

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

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


# ── model loading ─────────────────────────────────────────────────────────────

def load_judge(model_name: str):
    """Load Qwen2.5-7B-Instruct in 4-bit. Fits on T4 (16GB) and A100."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "No GPU detected. This script requires CUDA. "
            "Run on Kaggle, Colab, or college GPU."
        )

    total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {total_vram:.1f} GB available")
    print(f"Loading judge: {model_name} (4-bit quantized)...")

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

    used_vram = torch.cuda.memory_allocated() / 1e9
    print(f"Judge loaded. VRAM used: {used_vram:.1f} GB\n")
    return model, tokenizer


# ── inference ─────────────────────────────────────────────────────────────────

def judge_window(model, tokenizer, problem: str, window_steps: list) -> tuple:
    """
    Run Qwen2.5-7B judge on a window of steps.
    Returns (score: float, reasoning: str).
    """
    steps_text = "\n".join([f"Step {i+1}: {s}" for i, s in enumerate(window_steps)])
    prompt = JUDGE_PROMPT.format(problem=problem, window_steps=steps_text)

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    score = parse_score(response)
    reasoning_match = re.search(r"REASONING:\s*(.+)", response, re.DOTALL)
    reasoning = reasoning_match.group(1).strip()[:120] if reasoning_match else response[:120]

    return score, reasoning


def parse_score(text: str) -> float:
    """Extract numeric score from judge response. Fallback to 0.5 if unparseable."""
    match = re.search(r"SCORE:\s*([0-9]*\.?[0-9]+)", text)
    if match:
        return max(0.0, min(1.0, float(match.group(1))))
    nums = re.findall(r"0\.\d+|1\.0|0\.0", text)
    if nums:
        return float(nums[0])
    print(f"  [WARN] Could not parse score from: {text[:80]}")
    return 0.5


# ── chain scoring ─────────────────────────────────────────────────────────────

def parse_steps(text: str) -> list:
    lines = text.strip().split("\n")
    return [l.strip() for l in lines if l.strip() and not l.strip().startswith("####")]


def sample_window_size() -> int:
    return random.choices(WINDOW_SIZES, weights=WINDOW_PROBS, k=1)[0]


def score_chain(model, tokenizer, solution_text: str, problem: str) -> dict:
    """Score a full solution chain using sliding window judge, stride=1."""
    steps = parse_steps(solution_text)

    if len(steps) < 2:
        return {"n_steps": len(steps), "window_scores": [], "mean_score": 0.5}

    window_scores = []
    for i in range(len(steps)):
        W = sample_window_size()
        window = steps[i: i + W]
        if len(window) < 2:
            continue
        score, reasoning = judge_window(model, tokenizer, problem, window)
        window_scores.append({
            "start_step": i,
            "window_size": len(window),
            "score": score,
            "reasoning": reasoning,
        })

    if not window_scores:
        return {"n_steps": len(steps), "window_scores": [], "mean_score": 0.5}

    mean_score = round(float(np.mean([w["score"] for w in window_scores])), 4)
    return {"n_steps": len(steps), "window_scores": window_scores, "mean_score": mean_score}


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*60)
    print("TAPR Judge Quality Test — Qwen2.5-7B-Instruct")
    print("Phase 2  |  Risk 1 mitigation")
    print("="*60 + "\n")

    # Load validation data
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"{INPUT_FILE} not found. "
            "Copy tapr_flawed_solutions.json to this directory."
        )

    with open(INPUT_FILE) as f:
        data = json.load(f)

    # Filter out samples where flawed answer == GT answer
    filtered = {}
    for pid, item in data.items():
        try:
            gt  = float(item["gt_num"])
            fl  = float(item["flawed_num"])
            if abs(gt - fl) > 0.01:
                filtered[pid] = item
        except (TypeError, ValueError):
            pass

    print(f"Loaded {len(data)} problems, {len(filtered)} after filtering\n")

    # Stratified sample: 5A / 8B / 7C (proportional to original 25/40/35)
    pools = {"A": [], "B": [], "C": []}
    for pid, item in filtered.items():
        pools[item["flaw_type"]].append((pid, item))

    sample = (
        random.sample(pools["A"], min(5, len(pools["A"]))) +
        random.sample(pools["B"], min(8, len(pools["B"]))) +
        random.sample(pools["C"], min(7, len(pools["C"])))
    )
    random.shuffle(sample)

    counts = {t: sum(1 for _, i in sample if i["flaw_type"] == t) for t in "ABC"}
    print(f"Sampled {len(sample)} problems  ({counts['A']}A / {counts['B']}B / {counts['C']}C)\n")

    # Load judge
    model, tokenizer = load_judge(MODEL_NAME)

    # Score all problems
    results = []
    gt_scores, fl_scores = [], []
    type_scores = {t: {"gt": [], "fl": []} for t in "ABC"}

    for rank, (pid, item) in enumerate(sample):
        flaw_type = item["flaw_type"]
        question  = item["question"]

        print(f"[{rank+1}/{len(sample)}] Problem {pid} | Type {flaw_type}")

        print("  Scoring GT...")
        gt_result = score_chain(model, tokenizer, item["ground_truth_raw"], question)
        print(f"  GT score: {gt_result['mean_score']}")

        print("  Scoring flawed...")
        fl_result = score_chain(model, tokenizer, item["flawed_solution"], question)
        print(f"  FL score: {fl_result['mean_score']}")

        gt_scores.append(gt_result["mean_score"])
        fl_scores.append(fl_result["mean_score"])
        type_scores[flaw_type]["gt"].append(gt_result["mean_score"])
        type_scores[flaw_type]["fl"].append(fl_result["mean_score"])

        results.append({
            "id": pid,
            "flaw_type": flaw_type,
            "question": question[:80],
            "gt_score": gt_result["mean_score"],
            "fl_score": fl_result["mean_score"],
            "gt_beats_flawed": gt_result["mean_score"] > fl_result["mean_score"],
            "gt_details": gt_result,
            "fl_details": fl_result,
        })

    # ── analysis ──────────────────────────────────────────────────────────────
    gt_arr = np.array(gt_scores)
    fl_arr = np.array(fl_scores)
    all_scores  = np.concatenate([gt_arr, fl_arr])
    all_correct = np.array([1] * len(gt_arr) + [0] * len(fl_arr))

    r, p    = spearmanr(all_scores, all_correct)
    wins    = sum(1 for g, f in zip(gt_scores, fl_scores) if g > f)
    ties    = sum(1 for g, f in zip(gt_scores, fl_scores) if g == f)
    losses  = sum(1 for g, f in zip(gt_scores, fl_scores) if g < f)
    sep_rate = wins / len(results)

    print("\n" + "="*60)
    print("RESULTS — Qwen2.5-7B Judge Quality")
    print("="*60)
    print(f"Spearman r    = {r:.3f}  (p={p:.4f})")
    print(f"GT mean score = {gt_arr.mean():.4f}")
    print(f"FL mean score = {fl_arr.mean():.4f}")
    print(f"Gap           = {gt_arr.mean() - fl_arr.mean():.4f}")
    print(f"GT beats FL   = {wins}/{len(results)} ({sep_rate:.1%})")
    print(f"Ties={ties}  |  GT loses={losses}")

    print("\nBreakdown by flaw type:")
    print(f"{'Type':<6} {'GT mean':>10} {'FL mean':>10} {'Gap':>8} {'GT>FL':>8}")
    print("-" * 46)
    for t in "ABC":
        gt_t = np.array(type_scores[t]["gt"])
        fl_t = np.array(type_scores[t]["fl"])
        if len(gt_t) == 0:
            continue
        w = sum(1 for g, f in zip(gt_t, fl_t) if g > f)
        print(f"Type {t:<2} {gt_t.mean():>10.4f} {fl_t.mean():>10.4f} "
              f"{gt_t.mean()-fl_t.mean():>8.4f} {w}/{len(gt_t):>5}")

    print(f"\nPhase 1 reference (GPT-4.1-mini, n=93): r=0.600, 90.3% separation")

    # ── verdict ───────────────────────────────────────────────────────────────
    passed = (r >= PASS_THRESHOLD_R) and (sep_rate >= PASS_THRESHOLD_SEP)

    print("\n" + "="*60)
    print("VERDICT")
    print("="*60)

    if passed:
        print(f"✓ PASS — Qwen2.5-7B judge meets quality threshold.")
        print(f"  r={r:.3f} >= {PASS_THRESHOLD_R} | sep={sep_rate:.1%} >= {PASS_THRESHOLD_SEP:.0%}")
        print(f"  Proceed to train_grpo.py with inline judge.")
    else:
        print(f"✗ FAIL — Judge quality below threshold. Do NOT proceed to training.")
        if r < PASS_THRESHOLD_R:
            print(f"  r={r:.3f} < {PASS_THRESHOLD_R} — weak separation. Check prompt format.")
        if sep_rate < PASS_THRESHOLD_SEP:
            print(f"  sep={sep_rate:.1%} < {PASS_THRESHOLD_SEP:.0%} — too many GT losses.")
        print(f"  → Stop. Discuss with PI. Options: simplify prompt to 3 dimensions,")
        print(f"    check response parsing, or try Qwen2.5-14B if VRAM allows.")

    # ── save ──────────────────────────────────────────────────────────────────
    output = {
        "config": {
            "judge_model": MODEL_NAME,
            "n_samples": len(results),
            "sample_breakdown": counts,
            "window_sizes": WINDOW_SIZES,
            "window_probs": WINDOW_PROBS,
            "pass_threshold_r": PASS_THRESHOLD_R,
            "pass_threshold_sep": PASS_THRESHOLD_SEP,
        },
        "results": {
            "spearman_r": round(float(r), 4),
            "p_value": round(float(p), 4),
            "gt_mean": round(float(gt_arr.mean()), 4),
            "fl_mean": round(float(fl_arr.mean()), 4),
            "gap": round(float(gt_arr.mean() - fl_arr.mean()), 4),
            "gt_beats_fl": wins,
            "ties": ties,
            "gt_loses": losses,
            "total": len(results),
            "sep_rate": round(sep_rate, 4),
            "passed": passed,
        },
        "phase1_reference": {
            "judge_model": "gpt-4.1-mini",
            "spearman_r": 0.600,
            "sep_rate": 0.903,
            "n_problems": 93,
        },
        "raw_results": results,
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved → {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
