"""
MATH Judge Validation
======================
Harsh Raj

Validates MATH-specific judge dimensions before training.
Uses competition math problems from HuggingFaceH4/MATH-500.

Five dimensions (MATH-specific):
  1. ALGEBRAIC VALIDITY    — correct algebraic manipulations
  2. VARIABLE CONSISTENCY  — variables used consistently with definitions
  3. QUESTION COHERENCE    — reasoning addresses the actual problem
  4. CONVERGENCE           — steps make progress toward solution
  5. JUSTIFICATION         — non-trivial claims supported, not asserted

Flaw types:
  algebraic  — wrong algebraic step (bad factoring, expansion, simplification)
  chain      — unjustified leap (step doesn't follow from previous)
  relevance  — correct math but addresses wrong question

Target: r >= 0.55, separation >= 80%

Usage:
  !python math_judge_validation.py \\
    --output_file /content/drive/MyDrive/TAPR/results/math/math_judge_validation.json
"""

import argparse
import json
import os
import re
import random
import numpy as np
import torch
from scipy.stats import spearmanr
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

JUDGE_MODEL = "Qwen/Qwen2.5-Math-7B-Instruct"

# Set True to print raw judge responses. Essential for diagnosing format
# collapse — this is how the Qwen2.5-Math-7B failure was identified.
DEBUG_JUDGE_OUTPUT = False
N_PROBLEMS  = 20

FLAW_TYPES = [
    "algebraic",   # wrong algebraic manipulation
    "chain",       # unjustified leap / step doesn't follow
    "relevance",   # correct math but off-track from actual question
]

# ── prompts ───────────────────────────────────────────────────────────────────

CORRECT_SYSTEM = (
    "You are a careful mathematical reasoning assistant. "
    "Solve the competition math problem step by step, showing all algebraic work. "
    "Write 3-5 numbered steps, then end with #### [final answer]."
)

FLAW_SYSTEM = {
    "algebraic": (
        "You are generating a flawed math solution for research purposes. "
        "Solve the problem step by step but INTRODUCE ONE ALGEBRAIC ERROR in one step "
        "such as wrong factoring, wrong expansion, wrong simplification, or wrong substitution. "
        "Everything else should look reasonable. "
        "Write 3-5 numbered steps, then end with #### [answer]."
    ),
    "chain": (
        "You are generating a flawed math solution for research purposes. "
        "Write 3-5 numbered steps but make ONE STEP that does not logically or algebraically "
        "follow from the previous step — an unjustified leap or non-sequitur. "
        "Use correct algebra otherwise. End with #### [answer]."
    ),
    "relevance": (
        "You are generating a flawed math solution for research purposes. "
        "Write 3-5 numbered steps of correct mathematics, but solve a RELATED BUT DIFFERENT "
        "problem rather than the one actually asked. The math should look plausible. "
        "End with #### [answer]."
    ),
}

JUDGE_PROMPT_TEMPLATE = (
    "You are evaluating the quality of step-by-step reasoning "
    "in a competition math solution.\n\n"
    "Problem: {question}\n\n"
    "Steps to evaluate:\n"
    "{steps}\n\n"
    "Judge whether these steps represent legitimate mathematical reasoning "
    "across these dimensions:\n\n"
    "1. ALGEBRAIC VALIDITY: Are the algebraic manipulations in these steps correct?\n"
    "   Flag incorrect factoring, expansion, simplification, or substitution.\n\n"
    "2. VARIABLE CONSISTENCY: Are variables and expressions used consistently "
    "with how they were defined?\n"
    "   Flag any variable reused with a different meaning, or contradicting earlier definitions.\n\n"
    "3. QUESTION COHERENCE: Is the reasoning addressing the specific problem being asked?\n"
    "   Flag if the steps are solving a related but different problem.\n\n"
    "4. CONVERGENCE: Are these steps making meaningful progress toward a solution?\n"
    "   Flag circular reasoning or steps that do not reduce the problem.\n\n"
    "5. JUSTIFICATION: Are non-trivial claims supported rather than asserted?\n"
    "   Flag any step that states a result as obvious without showing the work.\n\n"
    "Respond with:\n"
    "SCORE: [0.0 to 1.0, your honest judgment across all five dimensions]\n"
    "REASONING: [one sentence explaining the score]"
)


JUDGE_SYSTEM = (
    "You are a math solution grader. "
    "Your response must be exactly two lines: "
    "SCORE: [a decimal between 0.0 and 1.0] and "
    "REASONING: [one short sentence]. "
    "Do not solve the problem. Do not write anything else."
)


# ── model loading ──────────────────────────────────────────────────────────────

def get_bnb():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )


def load_model(name=JUDGE_MODEL):
    print(f"Loading {name}...")
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        name, quantization_config=get_bnb(), device_map="auto", trust_remote_code=True
    )
    model.eval()
    print(f"  VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB")
    return model, tok


# ── data loading ───────────────────────────────────────────────────────────────

def parse_level(raw) -> int:
    """Handle level stored as int (1) or string ('Level 1')."""
    if isinstance(raw, int):
        return raw
    m = re.search(r"\d+", str(raw))
    return int(m.group()) if m else 3


def load_math_problems(n=N_PROBLEMS):
    """Load from HuggingFaceH4/MATH-500 — known to exist, no trust_remote_code needed."""
    from datasets import load_dataset
    print(f"\nLoading MATH problems ({n} problems)...")
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    records = [
        {
            "question": item["problem"],
            "level":    parse_level(item.get("level", 3)),
        }
        for item in ds
        if item.get("problem", "").strip()
    ]
    # Prefer easier problems (level <= 3) for cleaner chain generation
    easy   = [r for r in records if r["level"] <= 3]
    harder = [r for r in records if r["level"] > 3]
    pool   = easy + harder  # fallback to harder if not enough easy
    random.shuffle(pool)
    selected = pool[:n]
    print(f"  Selected {len(selected)} problems "
          f"(easy={sum(1 for r in selected if r['level']<=3)}, "
          f"hard={sum(1 for r in selected if r['level']>3)})")
    return selected


# ── generation ────────────────────────────────────────────────────────────────

def generate_chain(model, tok, question, system_prompt, max_new=400):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": f"Problem: {question}"},
    ]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=512).to(
        next(model.parameters()).device
    )
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new, do_sample=True,
            temperature=0.7, pad_token_id=tok.eos_token_id,
        )
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    return tok.decode(new_tokens, skip_special_tokens=True).strip()


# ── scoring ────────────────────────────────────────────────────────────────────

def extract_steps(chain_text):
    lines = chain_text.strip().split("\n")
    steps = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("####"):
            continue
        if re.match(r"^(step\s*)?\d+[\.\):]", line, re.IGNORECASE):
            clean = re.sub(r"^(step\s*)?\d+[\.\):]\s*", "", line, flags=re.IGNORECASE)
            if clean:
                steps.append(clean)
    if not steps:
        steps = [l.strip() for l in lines
                 if l.strip() and not l.strip().startswith("####")]
    return steps


def parse_score(text):
    match = re.search(r"SCORE:\s*([0-9]*\.?[0-9]+)", text)
    if match:
        return max(0.0, min(1.0, float(match.group(1))))
    nums = re.findall(r"0\.\d+|1\.0|0\.0", text)
    return float(nums[0]) if nums else 0.5


def score_window(model, tok, question, window_steps):
    steps_text = "\n".join(f"Step {i+1}: {s}" for i, s in enumerate(window_steps))
    prompt = JUDGE_PROMPT_TEMPLATE.format(question=question, steps=steps_text)
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user",   "content": prompt},
    ]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok([text], return_tensors="pt", truncation=True, max_length=1024).to(
        next(model.parameters()).device
    )
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=80, do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    response = tok.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()
    if DEBUG_JUDGE_OUTPUT:
        print(f"    [JUDGE] {response[:200]}")
    return parse_score(response)


def score_chain(model, tok, question, chain_text, window_size=3):
    steps = extract_steps(chain_text)
    if not steps:
        return None

    windows = []
    for i in range(max(1, len(steps) - window_size + 1)):
        window = steps[i:i + window_size]
        if len(window) >= 2:
            windows.append(window)
    if not windows:
        windows = [steps]

    scores = [score_window(model, tok, question, w) for w in windows]
    return round(sum(scores) / len(scores), 3)


# ── main ───────────────────────────────────────────────────────────────────────

def run_validation(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required.")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    problems = load_math_problems(N_PROBLEMS)
    model, tok = load_model()

    results = []
    correct_scores, flawed_scores = [], []
    separation_wins = 0

    for i, prob in enumerate(problems):
        flaw_type = FLAW_TYPES[i % len(FLAW_TYPES)]
        q = prob["question"]

        print(f"\n[{i+1}/{N_PROBLEMS}] {q[:60]}...")
        print(f"  Flaw type: {flaw_type}")

        correct_chain = generate_chain(model, tok, q, CORRECT_SYSTEM)
        flawed_chain  = generate_chain(model, tok, q, FLAW_SYSTEM[flaw_type])

        c = score_chain(model, tok, q, correct_chain)
        f = score_chain(model, tok, q, flawed_chain)

        print(f"  Correct: {c} | Flawed: {f}")

        if c is not None and f is not None:
            correct_scores.append(c)
            flawed_scores.append(f)
            if c > f:
                separation_wins += 1
                print("  Separation: CORRECT > FLAWED ✓")
            else:
                print("  Separation: FAILED ✗")

        results.append({
            "id": i, "question": q[:80], "flaw_type": flaw_type,
            "correct_score": c, "flawed_score": f,
            "separation": c > f if (c is not None and f is not None) else None,
        })

    n_valid  = len(correct_scores)
    sep_rate = separation_wins / n_valid if n_valid else 0
    all_scores = correct_scores + flawed_scores
    all_labels = [1] * n_valid + [0] * n_valid
    rho, pval  = spearmanr(all_scores, all_labels) if n_valid >= 3 else (None, None)

    print(f"\n{'='*55}")
    print("MATH JUDGE VALIDATION RESULTS")
    print(f"{'='*55}")
    print(f"Problems evaluated : {n_valid}/{N_PROBLEMS}")
    if rho:
        print(f"Spearman r         : {rho:.3f} (p={pval:.4f})")
    print(f"Separation         : {separation_wins}/{n_valid} ({sep_rate:.1%})")
    if correct_scores:
        print(f"Mean correct       : {sum(correct_scores)/n_valid:.3f}")
        print(f"Mean flawed        : {sum(flawed_scores)/n_valid:.3f}")

    print("\nBreakdown by flaw type:")
    for ft in FLAW_TYPES:
        subset = [r for r in results if r["flaw_type"] == ft and r["separation"] is not None]
        wins = sum(1 for r in subset if r["separation"])
        print(f"  {ft:<12}: {wins}/{len(subset)} separated")

    if rho and rho >= 0.55 and sep_rate >= 0.80:
        print("\nPASS — proceed to MATH training.")
    elif rho and rho >= 0.45:
        print("\nMARGINAL — signal present. Review failures before training.")
    else:
        print("\nFAIL — review dimensions and prompt before training.")

    if args.output_file:
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        output = {
            "spearman_r": float(rho) if rho else None,
            "p_value":    float(pval) if pval else None,
            "separation": f"{separation_wins}/{n_valid}",
            "separation_rate": sep_rate,
            "mean_correct": sum(correct_scores)/n_valid if correct_scores else None,
            "mean_flawed":  sum(flawed_scores)/n_valid if flawed_scores else None,
            "problems": results,
        }
        with open(args.output_file, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Saved to: {args.output_file}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--output_file", type=str,
        default="/content/drive/MyDrive/TAPR/results/math/math_judge_validation.json"
    )
    return p.parse_args()


if __name__ == "__main__":
    run_validation(parse_args())
