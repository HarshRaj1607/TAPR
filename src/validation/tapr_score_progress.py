"""
TAPR Progress Signal Scoring
==============================
PI: Harsh | Executor: Claude
Research Note v5

Loads tapr_flawed_solutions.json (output of tapr_generate_flaws.py).
Scores ground truth vs flawed solutions using the progress signal only.

Progress signal (per transition N → N+1):
  progress = 0.6 × %reduction_variables + 0.4 × %reduction_quantities
  Clipped at 0 — no penalties, only rewards forward progress.

Chain-level score = mean progress across all transitions in the chain.

Output: correlation table + tapr_progress_results.json
"""

import json
import re
import numpy as np
from scipy.stats import spearmanr

INPUT_FILE  = "tapr_flawed_solutions.json"
OUTPUT_FILE = "tapr_progress_results.json"

# ── text parsing ──────────────────────────────────────────────────────────────

STOPWORDS = {
    'the','a','an','is','are','was','were','of','to','in','for','and','or',
    'so','step','total','each','per','if','then','ounces','dollars','hours',
    'minutes','days','weeks','months','years','number','amount','times','how',
    'many','much','what','with','calculate','find','he','she','they','it',
    'his','her','their','this','that','these','those','since','as','we','get',
    'can','will','has','have','been','would','could','first','next','now',
    'therefore','thus','hence','let','use','using','given','remaining'
}

def extract_variables(text: str) -> set:
    tokens = re.findall(r'\b[a-zA-Z_][a-zA-Z_0-9]*\b', text.lower())
    return set(t for t in tokens if t not in STOPWORDS)

def extract_quantities(text: str) -> set:
    return set(re.findall(r'\d+\.?\d*', text))

def parse_steps(text: str) -> list[str]:
    lines = text.strip().split("\n")
    steps = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("####"):
            continue
        steps.append(line)
    return steps

# ── progress signal ───────────────────────────────────────────────────────────

def pct_reduction(before: set, after: set) -> float:
    """Percentage reduction in set size. Clipped at 0."""
    if len(before) == 0:
        return 0.0
    reduction = (len(before) - len(after)) / len(before)
    return max(0.0, reduction)  # no penalties

def transition_progress(step_n: str, step_n1: str) -> float:
    """
    Compute progress from step N to step N+1.
    progress = 0.6 × %reduction_variables + 0.4 × %reduction_quantities
    """
    vars_n  = extract_variables(step_n)
    vars_n1 = extract_variables(step_n1)
    qty_n   = extract_quantities(step_n)
    qty_n1  = extract_quantities(step_n1)

    var_progress = pct_reduction(vars_n, vars_n1)
    qty_progress = pct_reduction(qty_n, qty_n1)

    return round(0.6 * var_progress + 0.4 * qty_progress, 4)

def score_chain(text: str) -> dict:
    """Score a full solution chain. Returns mean progress + per-transition scores."""
    steps = parse_steps(text)

    if len(steps) < 2:
        return {
            "n_steps": len(steps),
            "transitions": [],
            "mean_progress": 0.0
        }

    transitions = []
    for i in range(1, len(steps)):
        p = transition_progress(steps[i-1], steps[i])
        transitions.append({
            "step_n": steps[i-1][:80],
            "step_n1": steps[i][:80],
            "progress": p
        })

    mean_progress = round(np.mean([t["progress"] for t in transitions]), 4)

    return {
        "n_steps": len(steps),
        "transitions": transitions,
        "mean_progress": mean_progress
    }

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*60)
    print("TAPR Progress Signal Scoring — Research Note v5")
    print("="*60 + "\n")

    with open(INPUT_FILE) as f:
        data = json.load(f)

    print(f"Loaded {len(data)} problems from {INPUT_FILE}\n")

    results = []
    gt_scores  = []
    fl_scores  = []
    type_scores = {"A": {"gt": [], "fl": []},
                   "B": {"gt": [], "fl": []},
                   "C": {"gt": [], "fl": []}}

    for problem_id, item in data.items():
        flaw_type = item["flaw_type"]

        gt_result = score_chain(item["ground_truth_raw"])
        fl_result = score_chain(item["flawed_solution"])

        gt_scores.append(gt_result["mean_progress"])
        fl_scores.append(fl_result["mean_progress"])
        type_scores[flaw_type]["gt"].append(gt_result["mean_progress"])
        type_scores[flaw_type]["fl"].append(fl_result["mean_progress"])

        results.append({
            "id": problem_id,
            "question": item["question"][:80],
            "flaw_type": flaw_type,
            "gt_num": item["gt_num"],
            "flawed_num": item["flawed_num"],
            "ground_truth": gt_result,
            "flawed": fl_result,
            "gt_beats_flawed": gt_result["mean_progress"] > fl_result["mean_progress"]
        })

    gt_scores = np.array(gt_scores)
    fl_scores = np.array(fl_scores)

    # ── correlation analysis ──────────────────────────────────────────────────
    all_scores  = np.concatenate([gt_scores, fl_scores])
    all_correct = np.array([1]*len(gt_scores) + [0]*len(fl_scores))

    r, p = spearmanr(all_scores, all_correct)

    print("=" * 55)
    print("CORRELATION: Progress Score vs Correctness")
    print("=" * 55)
    print(f"Spearman r = {r:.3f}  |  p-value = {p:.4f}")
    strength = "strong" if abs(r) > 0.4 else ("moderate" if abs(r) > 0.2 else "weak")
    print(f"Strength: {strength}\n")

    # ── mean scores ───────────────────────────────────────────────────────────
    print("=" * 55)
    print("MEAN PROGRESS SCORES")
    print("=" * 55)
    print(f"{'Chain type':<20} {'Mean progress':>15} {'N':>5}")
    print("-" * 42)
    print(f"{'Ground truth':<20} {gt_scores.mean():>15.4f} {len(gt_scores):>5}")
    print(f"{'Flawed':<20} {fl_scores.mean():>15.4f} {len(fl_scores):>5}")
    print(f"{'Gap (GT - Fl)':<20} {gt_scores.mean()-fl_scores.mean():>15.4f}")

    # ── per flaw type ─────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("BREAKDOWN BY FLAW TYPE")
    print("=" * 55)
    print(f"{'Type':<8} {'GT mean':>10} {'Fl mean':>10} {'Gap':>10} {'GT>Fl':>8}")
    print("-" * 50)
    for t in ["A", "B", "C"]:
        gt_t = np.array(type_scores[t]["gt"])
        fl_t = np.array(type_scores[t]["fl"])
        wins = sum(1 for g, f in zip(gt_t, fl_t) if g > f)
        print(f"Type {t:<4} {gt_t.mean():>10.4f} {fl_t.mean():>10.4f} "
              f"{gt_t.mean()-fl_t.mean():>10.4f} {wins}/{len(gt_t):>5}")

    # ── problem-level separation ──────────────────────────────────────────────
    wins  = sum(1 for g, f in zip(gt_scores, fl_scores) if g > f)
    ties  = sum(1 for g, f in zip(gt_scores, fl_scores) if g == f)
    losses = sum(1 for g, f in zip(gt_scores, fl_scores) if g < f)
    print(f"\n{'='*55}")
    print(f"PROBLEM-LEVEL SEPARATION (GT > Flawed)")
    print(f"{'='*55}")
    print(f"GT beats Flawed: {wins}/{len(results)} ({wins/len(results):.1%})")
    print(f"Ties:            {ties}/{len(results)}")
    print(f"GT loses:        {losses}/{len(results)}")

    # ── verdict ───────────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print("VERDICT")
    print(f"{'='*55}")
    if abs(r) > 0.4 and p < 0.05:
        print("✓ STRONG: Progress signal reliably separates good from bad reasoning.")
        print("  Signal is validated — proceed to blueprint.")
    elif abs(r) > 0.2 and p < 0.05:
        print("~ MODERATE: Progress signal shows real but weak separation.")
        print("  Worth proceeding but note the limitation in the blueprint.")
    else:
        print("✗ WEAK: Progress signal does not reliably separate chains.")
        print("  Revisit signal design before proceeding.")

    # ── save ──────────────────────────────────────────────────────────────────
    output = {
        "config": {
            "n_problems": len(results),
            "flaw_distribution": {"A": 25, "B": 40, "C": 35},
            "signal": "progress only",
            "formula": "0.6 * pct_reduction_variables + 0.4 * pct_reduction_quantities",
            "clipped_at_zero": True
        },
        "correlations": {
            "spearman_r": round(r, 3),
            "p_value": round(p, 4),
            "strength": strength
        },
        "mean_scores": {
            "ground_truth": round(float(gt_scores.mean()), 4),
            "flawed": round(float(fl_scores.mean()), 4),
            "gap": round(float(gt_scores.mean() - fl_scores.mean()), 4)
        },
        "separation": {
            "gt_beats_flawed": wins,
            "ties": ties,
            "gt_loses": losses,
            "total": len(results)
        },
        "raw_results": results
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nFull results saved to {OUTPUT_FILE}")
    print("Next step: interpret results, then draft Phase 1 blueprint.")


if __name__ == "__main__":
    main()
