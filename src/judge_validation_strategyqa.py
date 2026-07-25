"""
StrategyQA Judge Validation
============================
Harsh Raj

Validates the StrategyQA judge dimensions before training.
Mirrors the GSM8K validation approach.

Process:
  1. Load 20 StrategyQA problems
  2. Generate correct reasoning chains (Qwen2.5-7B)
  3. Generate flawed reasoning chains (Qwen2.5-7B with explicit flaw injection)
  4. Score all 40 chains using the 4-dimension judge
  5. Compute Spearman r and separation

Target: r >= 0.65, separation >= 17/20 (85%)

Dimensions:
  - Fact Accuracy: are factual claims correct?
  - Chain Validity: does each step follow from the previous?
  - Conclusion Consistency: do steps build toward the final yes/no?
  - Relevance: do steps address what the question actually asks?

Usage (Colab):
  !python strategyqa_judge_validation.py

  # Optional: save results to Drive
  !python strategyqa_judge_validation.py --output_file /content/drive/MyDrive/sqa_judge_validation.json
"""

import argparse
import json
import re
import random
import numpy as np
import torch
import requests
from scipy.stats import spearmanr
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

JUDGE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
N_PROBLEMS  = 20

# Flaw types rotate across problems for diversity
FLAW_TYPES = [
    "factual",       # wrong factual claim in one step
    "chain",         # one step doesn't follow from the previous
    "relevance",     # steps are correct but don't address the question
    # "conclusion" removed — wrong yes/no is caught by R_outcome, not the judge
]

# ── prompts ───────────────────────────────────────────────────────────────────

CORRECT_SYSTEM = (
    "You are a careful reasoning assistant. "
    "Think through the yes/no question step by step using accurate facts. "
    "Write 3-5 clear numbered steps, then state the final answer as 'Answer: yes' or 'Answer: no'."
)

FLAW_SYSTEM = {
    "factual": (
        "You are generating a flawed reasoning chain for research purposes. "
        "Think through the yes/no question step by step, but INTRODUCE ONE FACTUAL ERROR "
        "in one of the steps — a wrong date, wrong definition, wrong property, or wrong historical fact. "
        "Everything else should look reasonable. "
        "Write 3-5 numbered steps, then state 'Answer: yes' or 'Answer: no'."
    ),
    "chain": (
        "You are generating a flawed reasoning chain for research purposes. "
        "Think through the yes/no question step by step, but make ONE STEP that does NOT "
        "logically follow from the previous step — it should seem like a non-sequitur or jump. "
        "Use accurate facts otherwise. "
        "Write 3-5 numbered steps, then state 'Answer: yes' or 'Answer: no'."
    ),
    "conclusion": (
        "You are generating a flawed reasoning chain for research purposes. "
        "Think through the yes/no question step by step using accurate facts and logical steps, "
        "but state the WRONG final answer — if the correct answer is 'yes', say 'no', and vice versa. "
        "Write 3-5 numbered steps, then state 'Answer: yes' or 'Answer: no'."
    ),
    "relevance": (
        "You are generating a flawed reasoning chain for research purposes. "
        "Write 3-5 numbered steps that are factually true and follow from each other, "
        "but DO NOT actually address the specific question being asked — go on a tangent "
        "about related facts that don't lead to the answer. "
        "Then state 'Answer: yes' or 'Answer: no'."
    ),
}

JUDGE_PROMPT_TEMPLATE = """\
You are evaluating the quality of step-by-step reasoning for a yes/no question.

Question: {question}

Steps to evaluate:
{steps}

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

def load_strategyqa(n=N_PROBLEMS):
    print(f"\nLoading StrategyQA ({n} problems)...")
    test_url  = "https://raw.githubusercontent.com/eladsegal/strategyqa/main/data/strategyqa/test.json"
    train_url = "https://raw.githubusercontent.com/eladsegal/strategyqa/main/data/strategyqa/train.json"

    try:
        r = requests.get(test_url, timeout=30)
        r.raise_for_status()
        data = r.json()
        print(f"  Loaded test split ({len(data)} total)")
    except Exception as e:
        print(f"  Test failed ({e}), using train...")
        r = requests.get(train_url, timeout=30)
        r.raise_for_status()
        data = r.json()

    random.shuffle(data)
    selected = data[:n]
    problems = [
        {"question": item["question"], "answer": "yes" if item["answer"] else "no"}
        for item in selected
    ]
    print(f"  Selected {len(problems)} problems")
    return problems


# ── generation ────────────────────────────────────────────────────────────────

def generate_chain(model, tok, question, answer, system_prompt, max_new=300):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": f"Question: {question}\nCorrect answer: {answer}"},
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


# ── window scoring ─────────────────────────────────────────────────────────────

def extract_steps(chain_text):
    """Extract numbered steps from chain text."""
    lines = chain_text.strip().split("\n")
    steps = []
    for line in lines:
        line = line.strip()
        if re.match(r"^(step\s*)?\d+[\.\):]", line, re.IGNORECASE):
            # Remove numbering prefix
            clean = re.sub(r"^(step\s*)?\d+[\.\):]\s*", "", line, flags=re.IGNORECASE)
            if clean:
                steps.append(clean)
    # Fallback: non-empty lines that aren't the answer line
    if not steps:
        steps = [
            l.strip() for l in lines
            if l.strip() and not l.lower().startswith("answer:")
        ]
    return steps


def score_window(model, tok, question, window_steps):
    """Score one window of steps using the judge."""
    steps_text = "\n".join(f"Step {i+1}: {s}" for i, s in enumerate(window_steps))
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        question=question, steps=steps_text
    )
    messages = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok([text], return_tensors="pt", truncation=True, max_length=800).to(
        next(model.parameters()).device
    )
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=60, do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    response = tok.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()
    score = parse_score(response)
    return {"mean": score, "raw_response": response}


def parse_score(text: str) -> float:
    """Parse single SCORE: X.XX from judge response."""
    match = re.search(r"SCORE:\s*([0-9]*\.?[0-9]+)", text)
    if match:
        return max(0.0, min(1.0, float(match.group(1))))
    nums = re.findall(r"0\.\d+|1\.0|0\.0", text)
    return float(nums[0]) if nums else 0.5


def parse_scores(text):
    """Parse 4-dimension scores from judge output. Returns mean if all found."""
    dims = {
        "fact_accuracy":        r"Fact Accuracy:\s*([1-5])",
        "chain_validity":       r"Chain Validity:\s*([1-5])",
        "conclusion_consistency": r"Conclusion Consistency:\s*([1-5])",
        "relevance":            r"Relevance:\s*([1-5])",
    }
    scores = {}
    for key, pattern in dims.items():
        m = re.search(pattern, text, re.IGNORECASE)
        scores[key] = int(m.group(1)) if m else None

    valid = [v for v in scores.values() if v is not None]
    scores["mean"] = round(sum(valid) / len(valid), 3) if valid else None
    scores["raw_response"] = text
    return scores


def score_chain(model, tok, question, answer, chain_text, window_size=3):
    """Score all windows of a chain and return mean across windows."""
    steps = extract_steps(chain_text)
    if not steps:
        return {"mean": None, "windows": []}

    # Sliding windows
    windows = []
    stride = 1
    for i in range(0, max(1, len(steps) - window_size + 1), stride):
        window = steps[i:i + window_size]
        if len(window) >= 2:
            windows.append(window)

    if not windows:
        windows = [steps]  # fallback: score all steps as one window

    window_scores = []
    for w in windows:
        s = score_window(model, tok, question, w)
        if s["mean"] is not None:
            window_scores.append(s)

    if not window_scores:
        return {"mean": None, "windows": window_scores}

    chain_mean = round(sum(s["mean"] for s in window_scores) / len(window_scores), 3)
    return {"mean": chain_mean, "windows": window_scores}


# ── validation ────────────────────────────────────────────────────────────────

def run_validation(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required.")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    problems = load_strategyqa(N_PROBLEMS)
    model, tok = load_model()

    results = []
    correct_scores, flawed_scores = [], []
    separation_wins = 0

    for i, prob in enumerate(problems):
        flaw_type = FLAW_TYPES[i % len(FLAW_TYPES)]
        q, a = prob["question"], prob["answer"]

        print(f"\n[{i+1}/{N_PROBLEMS}] {q[:60]}...")
        print(f"  Answer: {a} | Flaw type: {flaw_type}")

        # Generate chains
        correct_chain = generate_chain(model, tok, q, a, CORRECT_SYSTEM)
        flawed_chain  = generate_chain(model, tok, q, a, FLAW_SYSTEM[flaw_type])

        # Score chains
        correct_scored = score_chain(model, tok, q, a, correct_chain)
        flawed_scored  = score_chain(model, tok, q, a, flawed_chain)

        c_score = correct_scored["mean"]
        f_score = flawed_scored["mean"]

        print(f"  Correct score: {c_score} | Flawed score: {f_score}")

        if c_score is not None and f_score is not None:
            correct_scores.append(c_score)
            flawed_scores.append(f_score)
            if c_score > f_score:
                separation_wins += 1
                print(f"  Separation: CORRECT > FLAWED ✓")
            else:
                print(f"  Separation: FAILED ✗")

        results.append({
            "id": i,
            "question": q,
            "answer": a,
            "flaw_type": flaw_type,
            "correct_chain": correct_chain,
            "flawed_chain":  flawed_chain,
            "correct_score": c_score,
            "flawed_score":  f_score,
            "separation": c_score > f_score if (c_score and f_score) else None,
        })

    # ── metrics ────────────────────────────────────────────────────────────────
    all_scores  = correct_scores + flawed_scores
    all_labels  = [1] * len(correct_scores) + [0] * len(flawed_scores)
    n_valid     = len(correct_scores)
    separation  = separation_wins / n_valid if n_valid > 0 else 0

    rho, pval = spearmanr(all_scores, all_labels) if n_valid >= 3 else (None, None)

    print(f"\n{'='*55}")
    print("STRATEGYQA JUDGE VALIDATION RESULTS")
    print(f"{'='*55}")
    print(f"Problems evaluated : {n_valid}/{N_PROBLEMS}")
    print(f"Spearman r         : {rho:.3f} (p={pval:.4f})" if rho else "Spearman r: N/A")
    print(f"Separation         : {separation_wins}/{n_valid} ({separation:.1%})")
    print(f"Mean correct score : {sum(correct_scores)/len(correct_scores):.3f}" if correct_scores else "")
    print(f"Mean flawed score  : {sum(flawed_scores)/len(flawed_scores):.3f}" if flawed_scores else "")
    print()

    if rho and rho >= 0.65 and separation >= 0.85:
        print("PASS — judge is sufficiently calibrated. Proceed to training.")
    elif rho and rho >= 0.55:
        print("MARGINAL — judge shows signal but below target. Review failed cases before training.")
    else:
        print("FAIL — judge not calibrated. Review dimensions and prompt before training.")

    # Per-flaw-type breakdown
    print("\nBreakdown by flaw type:")
    for ft in FLAW_TYPES:
        subset = [r for r in results if r["flaw_type"] == ft and r["separation"] is not None]
        if subset:
            wins = sum(1 for r in subset if r["separation"])
            print(f"  {ft:<12}: {wins}/{len(subset)} separated")

    # Save
    if args.output_file:
        import os
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        output = {
            "spearman_r": rho, "p_value": pval,
            "separation": f"{separation_wins}/{n_valid}",
            "separation_rate": separation,
            "mean_correct": sum(correct_scores)/len(correct_scores) if correct_scores else None,
            "mean_flawed":  sum(flawed_scores)/len(flawed_scores) if flawed_scores else None,
            "problems": results,
        }
        with open(args.output_file, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nSaved to: {args.output_file}")

    return rho, separation


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output_file", type=str,
                   default="/content/drive/MyDrive/sqa_judge_validation.json")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_validation(args)
