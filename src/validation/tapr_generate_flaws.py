"""
TAPR Flaw Generation Script
============================
PI: Harsh | Executor: Claude
Research Note v5

Generates 100 fresh GSM8K problems with realistic flawed solutions
using GPT-4.1-mini. Three flaw types grounded in real SLM failure modes:

  Type A (25 samples) — Wrong quantity: correct transitions, wrong number plugged in
  Type B (40 samples) — Semantic misread: locally coherent, directionally wrong from start
  Type C (35 samples) — Collapsed reasoning: correct start, assertion at the end

Output: tapr_flawed_solutions.json
Next:   Spot-check 5A / 10B / 10C, then run tapr_score_progress.py
"""

import json
import os
import re
import random
from openai import OpenAI
from datasets import load_dataset

# ── config ────────────────────────────────────────────────────────────────────
SAMPLE_SIZE  = 100
RANDOM_SEED  = 99          # different from original seed=42 — fresh 100 problems
FLAW_DIST    = {"A": 25, "B": 40, "C": 35}
MODEL        = "gpt-4.1-mini"
OUTPUT_FILE  = "tapr_flawed_solutions.json"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

random.seed(RANDOM_SEED)
client = OpenAI(api_key=OPENAI_API_KEY)

# ── prompts ───────────────────────────────────────────────────────────────────
PROMPTS = {
    "A": """You are generating a flawed math reasoning example for research purposes.

Given this problem: {problem}

Generate a step-by-step solution where:
- Each transition between steps is mathematically valid and locally correct
- The arithmetic operations chosen are appropriate
- BUT the model confuses quantities — it uses a number that appears in the problem but is the wrong one for that step
- The error should feel natural, not like a deliberate trick
- The rest of the chain should follow consistently from the wrong number (propagate the error forward)
- End with #### [wrong answer]

Do not flag or acknowledge the error. Write as if this is a genuine solution attempt.""",

    "B_step1": """Given this math problem: {problem}

Identify the single most natural misreading a student might make — for example:
- Answering a sub-question instead of the final question (e.g. computing total cost instead of change)
- Ignoring one condition in the problem (e.g. forgetting a starting amount or a discount)
- Swapping what is being asked (e.g. question asks for remaining, student computes spent)

Return only the misread in one sentence. Be specific to this problem.""",

    "B_step2": """You are generating a flawed math reasoning example for research purposes.

Given this problem: {problem}

A student misread the problem as follows: {misread}

Generate a step-by-step solution where:
- Every individual transition is locally coherent and mathematically valid
- The arithmetic is correct at each step
- BUT the student commits fully to the misread above from the very first step
- The reasoning is internally consistent but answers the wrong question throughout
- End with #### [wrong answer — must be numerically different from the correct answer]

Do not flag or acknowledge the misread. Write as if this is a genuine solution attempt.""",

    "C": """You are generating a flawed math reasoning example for research purposes.

Given this problem: {problem}

Generate a step-by-step solution where:
- The first several steps are correct, well-reasoned, and show clear computation
- BUT the reasoning collapses near the end — the final step asserts a number without showing the computation that leads to it
- It should feel like the model ran out of reasoning capacity right before the finish
- End with #### [wrong answer that doesn't follow from the shown work]
- The final answer after #### must be numerically different from the correct answer.

Do not flag or acknowledge the error. Write as if this is a genuine solution attempt."""
}

# ── helpers ───────────────────────────────────────────────────────────────────

def extract_final_answer(text: str):
    match = re.search(r"####\s*([+-]?\d+(?:,\d+)*(?:\.\d+)?)", text)
    if match:
        return match.group(1).replace(",", "")
    nums = re.findall(r"[+-]?\d+(?:\.\d+)?", text)
    return nums[-1] if nums else None


def assign_flaw_types(n: int, dist: dict) -> list:
    """Assign flaw types to n samples according to dist, randomly shuffled."""
    types = []
    for flaw_type, count in dist.items():
        types.extend([flaw_type] * count)
    assert len(types) == n, f"Flaw distribution must sum to {n}"
    random.shuffle(types)
    return types


def generate_flawed_solution(problem: str, flaw_type: str) -> tuple:
    """
    Returns (flawed_solution, misread_or_empty).
    For Type B: two API calls — first identify misread, then generate solution.
    For Type A/C: one API call.
    """
    if flaw_type == "B":
        # Step 1 — identify the misread
        step1_prompt = PROMPTS["B_step1"].format(problem=problem)
        step1_response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": step1_prompt}],
            temperature=0.7,
            max_tokens=100,
        )
        misread = step1_response.choices[0].message.content.strip()

        # Step 2 — generate solution committing to the misread
        step2_prompt = PROMPTS["B_step2"].format(problem=problem, misread=misread)
        step2_response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": step2_prompt}],
            temperature=0.7,
            max_tokens=600,
        )
        return step2_response.choices[0].message.content.strip(), misread

    else:
        prompt = PROMPTS[flaw_type].format(problem=problem)
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=600,
        )
        return response.choices[0].message.content.strip(), ""


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*60)
    print("TAPR Flaw Generation — Research Note v5")
    print("="*60 + "\n")

    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not set. Run: export OPENAI_API_KEY=your_key")

    # Load dataset
    print("Loading GSM8K...")
    ds = load_dataset("gsm8k", "main", split="train")

    # Sample fresh 100 problems (seed=99, no overlap with original seed=42 run)
    all_indices = list(range(len(ds)))
    indices = random.sample(all_indices, SAMPLE_SIZE)
    print(f"Sampled {SAMPLE_SIZE} problems (seed={RANDOM_SEED})\n")

    # Assign flaw types
    flaw_types = assign_flaw_types(SAMPLE_SIZE, FLAW_DIST)
    type_counts = {t: flaw_types.count(t) for t in ["A", "B", "C"]}
    print(f"Flaw distribution: {type_counts}\n")

    results = {}
    failed = []

    for rank, (idx, flaw_type) in enumerate(zip(indices, flaw_types)):
        item = ds[idx]
        question = item["question"]
        gt_answer_raw = item["answer"]
        gt_num = extract_final_answer(gt_answer_raw)

        print(f"[{rank+1}/{SAMPLE_SIZE}] Problem {idx} | Type {flaw_type} | {question[:60]}...")

        try:
            flawed_solution, misread = generate_flawed_solution(question, flaw_type)
            flawed_num = extract_final_answer(flawed_solution)

            results[str(idx)] = {
                "question": question,
                "ground_truth_raw": gt_answer_raw,
                "gt_num": gt_num,
                "flaw_type": flaw_type,
                "misread": misread,
                "flawed_solution": flawed_solution,
                "flawed_num": flawed_num,
            }
            if flaw_type == "B" and misread:
                print(f"  ✓ GT={gt_num} | Flawed pred={flawed_num} | Misread: {misread[:60]}...")
            else:
                print(f"  ✓ GT={gt_num} | Flawed pred={flawed_num}")

        except Exception as e:
            print(f"  ✗ Failed: {e}")
            failed.append(idx)
            continue

        # Save incrementally
        if (rank + 1) % 10 == 0 or (rank + 1) == SAMPLE_SIZE:
            with open(OUTPUT_FILE, "w") as f:
                json.dump(results, f, indent=2)
            print(f"  → Checkpoint saved ({rank+1}/{SAMPLE_SIZE})")

    print(f"\nDone. {len(results)} problems saved to {OUTPUT_FILE}")
    if failed:
        print(f"Failed problems: {failed}")

    # Print spot-check summary
    print("\n" + "="*60)
    print("SPOT-CHECK SAMPLES")
    print("="*60)
    print("Before running tapr_score_progress.py, verify these:\n")

    for flaw_type, n_check in [("A", 5), ("B", 10), ("C", 10)]:
        type_samples = [(k, v) for k, v in results.items() if v["flaw_type"] == flaw_type]
        check_samples = random.sample(type_samples, min(n_check, len(type_samples)))
        print(f"\n--- TYPE {flaw_type} — Check {n_check} samples ---")
        for k, v in check_samples:
            print(f"\nProblem {k}:")
            print(f"Q: {v['question'][:100]}")
            print(f"GT answer: {v['gt_num']}")
            print(f"Flawed solution:\n{v['flawed_solution'][:400]}")
            print(f"Flawed pred: {v['flawed_num']}")
            print("-" * 40)

    print(f"\nIf quality looks good → run: python tapr_score_progress.py")


if __name__ == "__main__":
    main()
