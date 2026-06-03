"""
TAPR Signal Validation Experiment
===================================
PI: Harsh | Executor: Claude
Research Note v4

Goal: Validate that the four-component TAPR reward signal correctly
ranks good vs bad reasoning chains on GSM8K samples — before full RL training.

Three solution types per problem:
  - ground_truth : clean step-by-step solution from GSM8K
  - flawed       : artificially corrupted (bad step injected)
  - slm_generated: Phi-3-Mini zero-shot output (loaded from JSON if available)

Four signals computed per step:
  1. sympy_valid   : was this step mathematically valid?
  2. atomicity     : is this step atomic (not a jump, not redundant)?
  3. teacher_legit : windowed GPT-4o-mini legitimacy check (precomputed or live)
  4. progress      : did this step reduce problem complexity?

Output: correlation table + per-signal strength vs final correctness.
"""

import json
import os
import re
import random
import numpy as np
from pathlib import Path
from typing import Optional

# ── third-party ──────────────────────────────────────────────────────────────
from datasets import load_dataset
from sympy import sympify, simplify, symbols, Eq, solve, SympifyError
from sympy.core.expr import Expr
import sympy as sp
from openai import OpenAI
from scipy.stats import pointbiserialr, spearmanr

# ── config ────────────────────────────────────────────────────────────────────
SAMPLE_SIZE      = 50        # start small — expand to 200 when ready
RANDOM_SEED      = 42
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY", "")
TEACHER_MODEL    = "gpt-4.1-mini"
WINDOW_PROBS     = {2: 0.2, 3: 0.6, 5: 0.2}   # stochastic W distribution
EQUAL_WEIGHT     = 0.25                         # α=β=γ=δ=0.25 (equal init)
SLM_OUTPUT_FILE  = "phi3_outputs.json"          # from Colab inference script
OUTPUT_FILE      = "tapr_validation_results.json"

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


# ══════════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_gsm8k_samples(n: int) -> list[dict]:
    """Load n problems from GSM8K train split."""
    print(f"Loading {n} GSM8K samples...")
    ds = load_dataset("gsm8k", "main", split="train")
    indices = random.sample(range(len(ds)), n)
    samples = []
    for i in indices:
        item = ds[i]
        samples.append({
            "id": i,
            "question": item["question"],
            "answer_raw": item["answer"],          # contains #### final answer
            "answer_num": extract_final_answer(item["answer"]),
        })
    print(f"  Loaded {len(samples)} samples.")
    return samples


def extract_final_answer(answer_raw: str) -> Optional[float]:
    """Extract numeric final answer from GSM8K #### format."""
    match = re.search(r"####\s*([+-]?\d+(?:,\d+)*(?:\.\d+)?)", answer_raw)
    if match:
        return float(match.group(1).replace(",", ""))
    return None


def parse_steps(solution_text: str) -> list[str]:
    """
    Split a solution into steps.
    In practice: one step per line, stripped.
    Filters out empty lines and the #### answer line.
    """
    lines = solution_text.strip().split("\n")
    steps = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("####"):
            continue
        steps.append(line)
    return steps


# ══════════════════════════════════════════════════════════════════════════════
# 2. SOLUTION CONSTRUCTION
# ══════════════════════════════════════════════════════════════════════════════

def make_flawed_solution(steps: list[str]) -> list[str]:
    """
    Inject a flaw into a ground-truth step sequence.
    Three flaw types, randomly chosen:
      - arithmetic_error : corrupt a number in a random step
      - jump             : collapse two consecutive steps into one
      - redundant        : duplicate a step
    """
    if not steps:
        return steps
    flawed = steps.copy()
    flaw_type = random.choice(["arithmetic_error", "jump", "redundant"])

    if flaw_type == "arithmetic_error" and len(flawed) >= 2:
        idx = random.randint(0, len(flawed) - 1)
        # Replace a number in the step with a wrong one
        step = flawed[idx]
        nums = re.findall(r"\d+\.?\d*", step)
        if nums:
            target = random.choice(nums)
            wrong = str(int(float(target)) + random.choice([-7, -3, 3, 7, 11]))
            flawed[idx] = step.replace(target, wrong, 1)

    elif flaw_type == "jump" and len(flawed) >= 2:
        idx = random.randint(0, len(flawed) - 2)
        # Collapse step[idx] and step[idx+1] into one (a logical jump)
        combined = flawed[idx] + " so " + flawed[idx + 1]
        flawed = flawed[:idx] + [combined] + flawed[idx + 2:]

    elif flaw_type == "redundant" and len(flawed) >= 1:
        idx = random.randint(0, len(flawed) - 1)
        # Duplicate a step
        flawed = flawed[:idx + 1] + [flawed[idx]] + flawed[idx + 1:]

    return flawed


# ══════════════════════════════════════════════════════════════════════════════
# 3. SIGNAL 1 — SYMPY VALIDITY
# ══════════════════════════════════════════════════════════════════════════════

def extract_numbers(text: str) -> list[float]:
    """Extract all numbers from a text string."""
    return [float(x.replace(",", "")) for x in re.findall(r"[+-]?\d+(?:,\d+)*(?:\.\d+)?", text)]


def sympy_valid(step: str, prev_step: Optional[str] = None) -> float:
    """
    Check if the arithmetic/algebraic operation in this step is valid.
    Strategy:
      - Extract numbers from the step
      - Try to verify the operation is internally consistent
      - If an equation is found, check if it's satisfiable
    Returns: 1.0 (valid) | 0.5 (uncertain) | 0.0 (invalid)
    """
    nums = extract_numbers(step)

    # Check for explicit arithmetic: look for patterns like "a + b = c" or "a * b = c"
    # Pattern: number op number = result
    arith_pattern = re.search(
        r"([+-]?\d+(?:\.\d+)?)\s*([+\-*/×÷])\s*([+-]?\d+(?:\.\d+)?)\s*=\s*([+-]?\d+(?:\.\d+)?)",
        step
    )
    if arith_pattern:
        a, op, b, result = arith_pattern.groups()
        a, b, result = float(a), float(b), float(result)
        ops = {"+": a + b, "-": a - b, "*": a * b, "×": a * b}
        if op in ["/", "÷"] and b != 0:
            ops[op] = a / b
        if op in ops:
            computed = ops[op]
            if abs(computed - result) < 0.01:
                return 1.0
            else:
                return 0.0  # arithmetic error detected

    # Try sympy expression parsing for equation-like steps
    eq_match = re.search(r"([a-zA-Z_]\w*)\s*=\s*(.+)", step)
    if eq_match:
        try:
            rhs = eq_match.group(2).strip().split()[0]  # take first token of rhs
            val = sympify(rhs)
            if val.is_number:
                return 1.0  # valid assignment
        except (SympifyError, Exception):
            pass

    # If no arithmetic found — can't verify, treat as uncertain
    if len(nums) == 0:
        return 0.5  # no numbers, step is text-only (might be setup)

    return 0.5  # uncertain — not invalid, not verified


# ══════════════════════════════════════════════════════════════════════════════
# 4. SIGNAL 2 — ATOMICITY
# ══════════════════════════════════════════════════════════════════════════════

# Operation type keywords (lightweight classifier substitute)
OP_KEYWORDS = {
    "addition":       ["add", "added", "plus", "sum", "total", "+"],
    "subtraction":    ["subtract", "minus", "difference", "less", "-"],
    "multiplication": ["multiply", "times", "product", "×", "*"],
    "division":       ["divide", "divided", "per", "each", "÷", "/"],
    "substitution":   ["let", "define", "set", "suppose", "denote"],
    "simplification": ["simplify", "reduce", "cancel", "combine"],
    "equation_setup": ["equation", "equals", "therefore", "so we get"],
    "interpretation": ["means", "represents", "so", "thus", "hence", "therefore"],
}


def detect_operations(step: str) -> list[str]:
    """Return list of operation types found in a step."""
    step_lower = step.lower()
    found = []
    for op_type, keywords in OP_KEYWORDS.items():
        if any(kw in step_lower for kw in keywords):
            found.append(op_type)
    return found


def atomicity_score(step: str, prev_step: Optional[str] = None) -> float:
    """
    Score atomicity of a step.
    Penalize:
      - Jumps: step is very long OR contains multiple operations OR 'so' connecting two conclusions
      - Redundancy: step is very similar to prev_step
    Returns: 0.0 to 1.0
    """
    score = 1.0

    # Length heuristic: very long steps are likely jumps
    words = step.split()
    if len(words) > 30:
        score -= 0.3
    elif len(words) > 20:
        score -= 0.1

    # Multiple operations detected = likely a jump
    ops = detect_operations(step)
    if len(ops) >= 3:
        score -= 0.4
    elif len(ops) >= 2:
        score -= 0.15

    # Jump signal: "so X, therefore Y" pattern — two conclusions in one step
    jump_patterns = [
        r"\bso\b.{5,}\btherefore\b",
        r"\bso\b.{5,}\bthus\b",
        r"\bso\b.{5,}\bhence\b",
        r"=.{3,}=.{3,}=",  # chain of equalities in one step
    ]
    for pattern in jump_patterns:
        if re.search(pattern, step, re.IGNORECASE):
            score -= 0.3
            break

    # Redundancy: high word overlap with previous step
    if prev_step:
        prev_words = set(prev_step.lower().split())
        curr_words = set(step.lower().split())
        if len(prev_words) > 0:
            overlap = len(prev_words & curr_words) / len(prev_words)
            if overlap > 0.85:
                score -= 0.5  # near-duplicate step
            elif overlap > 0.65:
                score -= 0.2

    return max(0.0, min(1.0, score))


# ══════════════════════════════════════════════════════════════════════════════
# 5. SIGNAL 3 — TEACHER WINDOW LEGITIMACY
# ══════════════════════════════════════════════════════════════════════════════

def sample_window_size() -> int:
    """Sample W from stochastic distribution P(W=2)=0.2, P(W=3)=0.6, P(W=5)=0.2."""
    return random.choices(
        population=list(WINDOW_PROBS.keys()),
        weights=list(WINDOW_PROBS.values()),
        k=1
    )[0]


def teacher_window_legitimacy(
    steps: list[str],
    step_idx: int,
    question: str,
    cache: dict
) -> float:
    """
    Query GPT-4o-mini to judge legitimacy of a window of steps.
    W is sampled stochastically. Result cached to avoid duplicate calls.
    Returns: 1.0 (legitimate) | 0.5 (uncertain) | 0.0 (not legitimate)
    """
    if not client:
        print("  [teacher] No API key — skipping teacher signal.")
        return 0.5  # neutral if no API key

    W = sample_window_size()
    start = max(0, step_idx - W + 1)
    window = steps[start: step_idx + 1]
    cache_key = f"{hash(question)}_{start}_{step_idx}"

    if cache_key in cache:
        return cache[cache_key]

    window_text = "\n".join(f"Step {i+1}: {s}" for i, s in enumerate(window))
    prompt = (
        f"Problem: {question}\n\n"
        f"Reasoning steps:\n{window_text}\n\n"
        f"Are these reasoning steps collectively a legitimate progression toward "
        f"solving the problem? Answer YES or NO, then briefly explain why in one sentence."
    )

    try:
        response = client.chat.completions.create(
            model=TEACHER_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=80,
            temperature=0.0,
        )
        reply = response.choices[0].message.content.strip().upper()
        if reply.startswith("YES"):
            result = 1.0
        elif reply.startswith("NO"):
            result = 0.0
        else:
            result = 0.5
    except Exception as e:
        print(f"  [teacher] API error: {e}")
        result = 0.5

    cache[cache_key] = result
    return result


# ══════════════════════════════════════════════════════════════════════════════
# 6. SIGNAL 4 — PROGRESS
# ══════════════════════════════════════════════════════════════════════════════

def estimate_complexity(step: str) -> float:
    """
    Estimate problem complexity at this step.
    Proxy: number of distinct unknown variables + number of remaining operations.
    Lower = simpler = more progress has been made.
    """
    # Count unknown variables (single letters not part of words)
    unknowns = set(re.findall(r"\b([a-zA-Z])\b", step))
    # Remove common English words that happen to be single letters
    unknowns -= {"a", "A", "I", "s"}
    n_unknowns = len(unknowns)

    # Count operators (more operators = more complex expression)
    n_operators = len(re.findall(r"[+\-*/=<>]", step))

    # Count distinct numbers (more numbers in play = more complex)
    n_numbers = len(extract_numbers(step))

    return float(n_unknowns * 2 + n_operators + n_numbers * 0.5)


def progress_score(step: str, prev_step: Optional[str]) -> float:
    """
    Compute directional progress: did this step reduce complexity?
    progress = complexity(prev) - complexity(curr)
    Normalized to [-1, 1] range.
    Returns: positive = moving closer to solution, negative = moving away.
    """
    if prev_step is None:
        return 0.0  # no prev step to compare

    prev_complexity = estimate_complexity(prev_step)
    curr_complexity = estimate_complexity(step)
    delta = prev_complexity - curr_complexity

    # Normalize: cap at ±5 complexity units, map to [-1, 1]
    return max(-1.0, min(1.0, delta / 5.0))


# ══════════════════════════════════════════════════════════════════════════════
# 7. COMBINED REWARD
# ══════════════════════════════════════════════════════════════════════════════

def compute_r_transition(
    step: str,
    prev_step: Optional[str],
    steps: list[str],
    step_idx: int,
    question: str,
    teacher_cache: dict,
    alpha: float = EQUAL_WEIGHT,
    beta: float = EQUAL_WEIGHT,
    gamma: float = EQUAL_WEIGHT,
    delta: float = EQUAL_WEIGHT,
) -> dict:
    """Compute all four signals and combined R_transition for one step."""
    s1 = sympy_valid(step, prev_step)
    s2 = atomicity_score(step, prev_step)
    s3 = teacher_window_legitimacy(steps, step_idx, question, teacher_cache)
    # progress_score in [-1,1], normalize to [0,1] for combination
    raw_progress = progress_score(step, prev_step)
    s4 = (raw_progress + 1.0) / 2.0

    r_transition = alpha * s1 + beta * s2 + gamma * s3 + delta * s4

    return {
        "sympy":        round(s1, 3),
        "atomicity":    round(s2, 3),
        "teacher":      round(s3, 3),
        "progress_raw": round(raw_progress, 3),
        "progress":     round(s4, 3),
        "r_transition": round(r_transition, 3),
    }


def score_solution(
    steps: list[str],
    question: str,
    teacher_cache: dict,
) -> dict:
    """Score an entire solution — average R_transition across all steps."""
    if not steps:
        return {"step_scores": [], "mean_r_transition": 0.0,
                "mean_sympy": 0.0, "mean_atomicity": 0.0,
                "mean_teacher": 0.0, "mean_progress": 0.0}

    step_scores = []
    for i, step in enumerate(steps):
        prev = steps[i - 1] if i > 0 else None
        scores = compute_r_transition(step, prev, steps, i, question, teacher_cache)
        step_scores.append(scores)

    return {
        "step_scores":      step_scores,
        "mean_r_transition": round(np.mean([s["r_transition"] for s in step_scores]), 3),
        "mean_sympy":        round(np.mean([s["sympy"] for s in step_scores]), 3),
        "mean_atomicity":    round(np.mean([s["atomicity"] for s in step_scores]), 3),
        "mean_teacher":      round(np.mean([s["teacher"] for s in step_scores]), 3),
        "mean_progress":     round(np.mean([s["progress"] for s in step_scores]), 3),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 8. MAIN EXPERIMENT
# ══════════════════════════════════════════════════════════════════════════════

def run_experiment():
    print("\n" + "="*60)
    print("TAPR Signal Validation Experiment — Research Note v4")
    print("="*60 + "\n")

    # Load data
    samples = load_gsm8k_samples(SAMPLE_SIZE)

    # Load SLM outputs if available
    slm_outputs = {}
    if Path(SLM_OUTPUT_FILE).exists():
        with open(SLM_OUTPUT_FILE) as f:
            slm_outputs = json.load(f)
        print(f"Loaded SLM outputs for {len(slm_outputs)} problems.\n")
    else:
        print(f"No SLM output file found ({SLM_OUTPUT_FILE}). Skipping SLM solutions.\n")

    results = []
    teacher_cache = {}

    for idx, sample in enumerate(samples):
        print(f"[{idx+1}/{len(samples)}] Problem ID {sample['id']}")

        gt_steps = parse_steps(sample["answer_raw"])
        fl_steps = make_flawed_solution(gt_steps)
        slm_steps = parse_steps(slm_outputs.get(str(sample["id"]), {}).get("solution", "")) \
                    if slm_outputs else []

        q = sample["question"]

        gt_scores  = score_solution(gt_steps, q, teacher_cache)
        fl_scores  = score_solution(fl_steps, q, teacher_cache)
        slm_scores = score_solution(slm_steps, q, teacher_cache) if slm_steps else None

        result = {
            "id":       sample["id"],
            "question": q[:80] + "...",
            "answer":   sample["answer_num"],
            "ground_truth": {
                "steps": gt_steps,
                "correct": True,
                **gt_scores,
            },
            "flawed": {
                "steps": fl_steps,
                "correct": False,
                **fl_scores,
            },
        }
        if slm_scores:
            # SLM correctness: check if final answer matches
            slm_answer_raw = slm_outputs.get(str(sample["id"]), {}).get("answer", "")
            slm_num = extract_final_answer(slm_answer_raw)
            slm_correct = (slm_num is not None and
                          sample["answer_num"] is not None and
                          abs(slm_num - sample["answer_num"]) < 0.01)
            result["slm"] = {
                "steps": slm_steps,
                "correct": slm_correct,
                **slm_scores,
            }

        results.append(result)

    # ── Analysis ──────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)

    # Build arrays for correlation analysis
    # Ground truth = correct (1), flawed = incorrect (0)
    all_correct = []
    all_r_trans = []
    all_sympy   = []
    all_atom    = []
    all_teacher = []
    all_progress= []

    for r in results:
        for sol_type in ["ground_truth", "flawed"]:
            sol = r[sol_type]
            all_correct.append(1 if sol["correct"] else 0)
            all_r_trans.append(sol["mean_r_transition"])
            all_sympy.append(sol["mean_sympy"])
            all_atom.append(sol["mean_atomicity"])
            all_teacher.append(sol["mean_teacher"])
            all_progress.append(sol["mean_progress"])

        if "slm" in r:
            sol = r["slm"]
            all_correct.append(1 if sol["correct"] else 0)
            all_r_trans.append(sol["mean_r_transition"])
            all_sympy.append(sol["mean_sympy"])
            all_atom.append(sol["mean_atomicity"])
            all_teacher.append(sol["mean_teacher"])
            all_progress.append(sol["mean_progress"])

    all_correct  = np.array(all_correct)
    all_r_trans  = np.array(all_r_trans)
    all_sympy    = np.array(all_sympy)
    all_atom     = np.array(all_atom)
    all_teacher  = np.array(all_teacher)
    all_progress = np.array(all_progress)

    def safe_corr(signal, name):
        try:
            r, p = spearmanr(signal, all_correct)
            return r, p
        except Exception:
            return 0.0, 1.0

    signals = {
        "R_transition (combined)": all_r_trans,
        "Sympy validity":          all_sympy,
        "Atomicity":               all_atom,
        "Teacher legitimacy":      all_teacher,
        "Progress":                all_progress,
    }

    print(f"\n{'Signal':<30} {'Spearman r':>12} {'p-value':>12} {'Strength':>12}")
    print("-" * 68)

    correlation_results = {}
    for name, signal in signals.items():
        r, p = safe_corr(signal, name)
        strength = "strong" if abs(r) > 0.4 else ("moderate" if abs(r) > 0.2 else "weak")
        print(f"{name:<30} {r:>12.3f} {p:>12.4f} {strength:>12}")
        correlation_results[name] = {"spearman_r": round(r, 3), "p_value": round(p, 4), "strength": strength}

    # Mean scores by solution type
    print(f"\n{'Solution type':<20} {'R_trans':>8} {'Sympy':>8} {'Atom':>8} {'Teacher':>8} {'Progress':>8}")
    print("-" * 64)
    for sol_type_label, correct_val in [("Ground truth", True), ("Flawed", False)]:
        mask = all_correct == (1 if correct_val else 0)
        if mask.sum() == 0:
            continue
        print(
            f"{sol_type_label:<20}"
            f"{all_r_trans[mask].mean():>8.3f}"
            f"{all_sympy[mask].mean():>8.3f}"
            f"{all_atom[mask].mean():>8.3f}"
            f"{all_teacher[mask].mean():>8.3f}"
            f"{all_progress[mask].mean():>8.3f}"
        )

    if "slm" in results[0]:
        slm_correct_mask = np.array([1 if r.get("slm", {}).get("correct", False) else 0
                                     for r in results if "slm" in r])
        if len(slm_correct_mask) > 0:
            slm_r   = np.array([r["slm"]["mean_r_transition"] for r in results if "slm" in r])
            slm_acc = slm_correct_mask.mean()
            print(f"{'SLM (correct)':<20}{slm_r[slm_correct_mask==1].mean() if slm_correct_mask.sum()>0 else 0:>8.3f}")
            print(f"{'SLM (incorrect)':<20}{slm_r[slm_correct_mask==0].mean() if (slm_correct_mask==0).sum()>0 else 0:>8.3f}")
            print(f"\nSLM accuracy on this sample: {slm_acc:.1%}")

    # Save full results
    output = {
        "config": {
            "sample_size": SAMPLE_SIZE,
            "teacher_model": TEACHER_MODEL,
            "window_distribution": WINDOW_PROBS,
            "equal_weights": EQUAL_WEIGHT,
        },
        "correlations": correlation_results,
        "raw_results": results,
    }
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nFull results saved to {OUTPUT_FILE}")
    print("\nKey question: Does progress score dominate teacher legitimacy?")
    pr = correlation_results.get("Progress", {}).get("spearman_r", 0)
    tr = correlation_results.get("Teacher legitimacy", {}).get("spearman_r", 0)
    if abs(pr) > abs(tr):
        print(f"  → YES: progress r={pr:.3f} > teacher r={tr:.3f}. δ should be upweighted.")
    else:
        print(f"  → NO: teacher r={tr:.3f} >= progress r={pr:.3f}. Equal weights hold.")


if __name__ == "__main__":
    if not OPENAI_API_KEY:
        print("WARNING: OPENAI_API_KEY not set. Teacher signal will return 0.5 (neutral).")
        print("Set it via: export OPENAI_API_KEY=your_key\n")
    run_experiment()
