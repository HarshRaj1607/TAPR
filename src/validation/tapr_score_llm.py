"""
TAPR LLM Judge Scoring
========================
PI: Harsh | Executor: Claude
Research Note v5

Loads tapr_flawed_solutions.json.
Scores ground truth vs flawed solutions using GPT-4.1-mini as transition judge.

Signal design:
- Sliding window across chain, stride=1
- W sampled stochastically from {2,3,5} with P=(0.2, 0.6, 0.2)
- Window clipped at chain end (use whatever steps are available)
- Unified prompt judges 5 dimensions: local validity, value consistency,
  question coherence, convergence, completeness
- Chain score = mean of all window scores

Output: correlation table + tapr_llm_results.json
"""

import json
import os
import re
import random
import time
import numpy as np
from openai import OpenAI
from scipy.stats import spearmanr

# ── config ────────────────────────────────────────────────────────────────────
INPUT_FILE     = "tapr_flawed_solutions.json"
OUTPUT_FILE    = "tapr_llm_results.json"
MODEL          = "gpt-4.1-mini"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
WINDOW_SIZES   = [2, 3, 5]
WINDOW_PROBS   = [0.2, 0.6, 0.2]
RANDOM_SEED    = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

client = OpenAI(api_key=OPENAI_API_KEY)

# ── prompt ────────────────────────────────────────────────────────────────────
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

# ── helpers ───────────────────────────────────────────────────────────────────

def parse_steps(text: str) -> list:
    lines = text.strip().split("\n")
    steps = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("####"):
            continue
        steps.append(line)
    return steps


def sample_window_size() -> int:
    return random.choices(WINDOW_SIZES, weights=WINDOW_PROBS, k=1)[0]


def extract_score(response_text: str) -> float:
    """Extract numeric score from LLM response."""
    match = re.search(r"SCORE:\s*([0-9]*\.?[0-9]+)", response_text)
    if match:
        score = float(match.group(1))
        return max(0.0, min(1.0, score))
    # Fallback: find any float in range 0-1
    nums = re.findall(r"0\.\d+|1\.0|0\.0", response_text)
    if nums:
        return float(nums[0])
    return 0.5  # neutral if unparseable


def judge_window(problem: str, window_steps: list) -> tuple:
    """Call GPT-4.1-mini to judge a window of steps. Returns (score, reasoning)."""
    steps_text = "\n".join([f"Step {i+1}: {s}" for i, s in enumerate(window_steps)])
    prompt = JUDGE_PROMPT.format(problem=problem, window_steps=steps_text)

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=150,
        )
        text = response.choices[0].message.content.strip()
        score = extract_score(text)
        reasoning_match = re.search(r"REASONING:\s*(.+)", text, re.DOTALL)
        reasoning = reasoning_match.group(1).strip() if reasoning_match else text
        return score, reasoning
    except Exception as e:
        print(f"    API error: {e}")
        time.sleep(2)
        return 0.5, "API error"


def score_chain_llm(solution_text: str, problem: str) -> dict:
    """
    Score a full solution chain using sliding window LLM judge.
    Stride = 1, W sampled stochastically, clipped at chain end.
    """
    steps = parse_steps(solution_text)

    if len(steps) < 2:
        return {"n_steps": len(steps), "window_scores": [], "mean_score": 0.5}

    window_scores = []

    for i in range(len(steps)):
        W = sample_window_size()
        window = steps[i: i + W]  # clip naturally at end
        if len(window) < 2:
            continue  # need at least 2 steps to judge a transition

        score, reasoning = judge_window(problem, window)
        window_scores.append({
            "start_step": i,
            "window_size": len(window),
            "score": score,
            "reasoning": reasoning[:120]
        })

    if not window_scores:
        return {"n_steps": len(steps), "window_scores": [], "mean_score": 0.5}

    mean_score = round(np.mean([w["score"] for w in window_scores]), 4)

    return {
        "n_steps": len(steps),
        "window_scores": window_scores,
        "mean_score": mean_score
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*60)
    print("TAPR LLM Judge Scoring — Research Note v5")
    print("="*60 + "\n")

    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not set. Run: $env:OPENAI_API_KEY='your_key'")

    with open(INPUT_FILE) as f:
        data = json.load(f)

    print(f"Loaded {len(data)} problems from {INPUT_FILE}")

    # Filter samples where flawed answer == GT answer
    filtered_out = 0
    filtered_data = {}
    for pid, item in data.items():
        try:
            gt_num = float(item["gt_num"]) if item["gt_num"] is not None else None
            fl_num = float(item["flawed_num"]) if item["flawed_num"] is not None else None
            if gt_num is not None and fl_num is not None and abs(gt_num - fl_num) < 0.01:
                filtered_out += 1
                continue
        except (ValueError, TypeError):
            pass
        filtered_data[pid] = item

    print(f"Filtered out {filtered_out} samples where flawed answer == GT answer.")
    print(f"Proceeding with {len(filtered_data)} samples.\n")
    data = filtered_data

    results = []
    gt_scores = []
    fl_scores = []
    type_scores = {"A": {"gt": [], "fl": []},
                   "B": {"gt": [], "fl": []},
                   "C": {"gt": [], "fl": []}}

    total = len(data)
    for rank, (problem_id, item) in enumerate(data.items()):
        flaw_type = item["flaw_type"]
        question = item["question"]

        print(f"[{rank+1}/{total}] Problem {problem_id} | Type {flaw_type}")

        # Score ground truth
        print(f"  Scoring GT...")
        gt_result = score_chain_llm(item["ground_truth_raw"], question)
        print(f"  GT score: {gt_result['mean_score']}")

        # Score flawed
        print(f"  Scoring flawed...")
        fl_result = score_chain_llm(item["flawed_solution"], question)
        print(f"  FL score: {fl_result['mean_score']}")

        gt_scores.append(gt_result["mean_score"])
        fl_scores.append(fl_result["mean_score"])
        type_scores[flaw_type]["gt"].append(gt_result["mean_score"])
        type_scores[flaw_type]["fl"].append(fl_result["mean_score"])

        results.append({
            "id": problem_id,
            "question": question[:80],
            "flaw_type": flaw_type,
            "gt_num": item["gt_num"],
            "flawed_num": item["flawed_num"],
            "ground_truth": gt_result,
            "flawed": fl_result,
            "gt_beats_flawed": bool(gt_result["mean_score"] > fl_result["mean_score"])
        })

        # Save incrementally every 10 problems
        if (rank + 1) % 10 == 0:
            _save_results(results, gt_scores, fl_scores, type_scores, partial=True)
            print(f"  → Checkpoint saved ({rank+1}/{total})\n")

    # ── Final analysis ────────────────────────────────────────────────────────
    gt_scores = np.array(gt_scores)
    fl_scores = np.array(fl_scores)
    all_scores = np.concatenate([gt_scores, fl_scores])
    all_correct = np.array([1]*len(gt_scores) + [0]*len(fl_scores))

    r, p = spearmanr(all_scores, all_correct)

    print("\n" + "="*55)
    print("CORRELATION: LLM Judge Score vs Correctness")
    print("="*55)
    print(f"Spearman r = {r:.3f}  |  p-value = {p:.4f}")
    strength = "strong" if abs(r) > 0.4 else ("moderate" if abs(r) > 0.2 else "weak")
    print(f"Strength: {strength}\n")

    print("="*55)
    print("MEAN SCORES")
    print("="*55)
    print(f"{'Chain type':<20} {'Mean score':>12} {'N':>5}")
    print("-"*40)
    print(f"{'Ground truth':<20} {gt_scores.mean():>12.4f} {len(gt_scores):>5}")
    print(f"{'Flawed':<20} {fl_scores.mean():>12.4f} {len(fl_scores):>5}")
    print(f"{'Gap (GT - Fl)':<20} {gt_scores.mean()-fl_scores.mean():>12.4f}")

    print("\n" + "="*55)
    print("BREAKDOWN BY FLAW TYPE")
    print("="*55)
    print(f"{'Type':<8} {'GT mean':>10} {'Fl mean':>10} {'Gap':>10} {'GT>Fl':>8}")
    print("-"*50)
    for t in ["A", "B", "C"]:
        gt_t = np.array(type_scores[t]["gt"])
        fl_t = np.array(type_scores[t]["fl"])
        wins = sum(1 for g, f in zip(gt_t, fl_t) if g > f)
        print(f"Type {t:<4} {gt_t.mean():>10.4f} {fl_t.mean():>10.4f} "
              f"{gt_t.mean()-fl_t.mean():>10.4f} {wins}/{len(gt_t):>5}")

    wins = sum(1 for g, f in zip(gt_scores, fl_scores) if g > f)
    ties = sum(1 for g, f in zip(gt_scores, fl_scores) if g == f)
    losses = sum(1 for g, f in zip(gt_scores, fl_scores) if g < f)

    print(f"\n{'='*55}")
    print("PROBLEM-LEVEL SEPARATION (GT > Flawed)")
    print(f"{'='*55}")
    print(f"GT beats Flawed: {wins}/{len(results)} ({wins/len(results):.1%})")
    print(f"Ties:            {ties}/{len(results)}")
    print(f"GT loses:        {losses}/{len(results)}")

    print(f"\n{'='*55}")
    print("VERDICT")
    print(f"{'='*55}")
    if r > 0.4 and p < 0.05:
        print("✓ STRONG: LLM judge reliably separates good from bad reasoning.")
        print("  Signal is validated — proceed to blueprint.")
    elif r > 0.2 and p < 0.05:
        print("~ MODERATE: LLM judge shows real but weak separation.")
        print("  Worth proceeding but note limitation in blueprint.")
    else:
        print("✗ WEAK: LLM judge does not reliably separate chains.")
        print("  Revisit signal design before proceeding.")

    _save_results(results, gt_scores, fl_scores, type_scores, partial=False)
    print(f"\nFull results saved to {OUTPUT_FILE}")


def _save_results(results, gt_scores, fl_scores, type_scores, partial=False):
    gt_arr = np.array(gt_scores)
    fl_arr = np.array(fl_scores)
    all_scores = np.concatenate([gt_arr, fl_arr])
    all_correct = np.array([1]*len(gt_arr) + [0]*len(fl_arr))

    try:
        r, p = spearmanr(all_scores, all_correct)
    except Exception:
        r, p = 0.0, 1.0

    output = {
        "config": {
            "model": MODEL,
            "window_sizes": WINDOW_SIZES,
            "window_probs": WINDOW_PROBS,
            "stride": 1,
            "partial": partial
        },
        "correlations": {
            "spearman_r": round(float(r), 3),
            "p_value": round(float(p), 4),
        },
        "mean_scores": {
            "ground_truth": round(float(gt_arr.mean()), 4) if len(gt_arr) > 0 else None,
            "flawed": round(float(fl_arr.mean()), 4) if len(fl_arr) > 0 else None,
        },
        "by_flaw_type": {
            t: {
                "gt_mean": round(float(np.mean(type_scores[t]["gt"])), 4) if type_scores[t]["gt"] else None,
                "fl_mean": round(float(np.mean(type_scores[t]["fl"])), 4) if type_scores[t]["fl"] else None,
            } for t in ["A", "B", "C"]
        },
        "raw_results": results
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)


if __name__ == "__main__":
    main()
