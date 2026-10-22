"""
Judge accuracy against hand labels.

Ground truth: data/handlabels.jsonl, 40 traces labeled at step level. Each row gives
the FIRST step that breaks (1-based), or null if the reasoning has no break.
Rules are in notes/HANDLABEL_RULES.md.

For each scorer we ask two questions:
  1. Step level. Steps before the first break (and every step in no-break traces) are
     "ok". The first-break step is "bad". AUC = P(ok step scores above bad step).
     Steps AFTER the first break are excluded: they may be wrong or merely inherit the error.
  2. Trace level. Does the minimum step score separate no-break traces from break traces?

Confidence intervals resample PROBLEMS, not traces, because the same problem appears
several times (40 traces cover 20 problems).

Run:  python scripts/judge_vs_handlabels.py data/scores14b.jsonl
"""

import json
import random
import sys
from collections import defaultdict


def load_labels(path):
    lab = {}
    with open(path) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                lab[(r["problem_id"], r["sample_id"])] = r
    return lab


def load_scores(path):
    d = defaultdict(lambda: defaultdict(dict))  # scorer -> trace -> step_idx -> score
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("parse_failed") or not isinstance(r.get("score"), (int, float)):
                continue
            d[r["scorer"]][(r["problem_id"], r["sample_id"])][r["step_idx"]] = float(r["score"])
    return d


def auc(pos, neg):
    if not pos or not neg:
        return None
    return sum((a > b) + 0.5 * (a == b) for a in pos for b in neg) / (len(pos) * len(neg))


def step_sets(scores, lab, keys):
    ok, bad = [], []
    for k in keys:
        brk = lab[k]["first_break_step"]
        for i, s in scores.get(k, {}).items():
            j = i + 1  # step_idx is 0-based, labels are 1-based
            if brk is None or j < brk:
                ok.append(s)
            elif j == brk:
                bad.append(s)
    return ok, bad


def cluster_boot(scores, lab, keys, B=2000, seed=1):
    by_problem = defaultdict(list)
    for k in keys:
        by_problem[k[0]].append(k)
    problems = sorted(by_problem)
    rng = random.Random(seed)
    vals = []
    for _ in range(B):
        ks = [k for p in (rng.choice(problems) for _ in problems) for k in by_problem[p]]
        a = auc(*step_sets(scores, lab, ks))
        if a is not None:
            vals.append(a)
    vals.sort()
    return vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals))]


def main(scores_path, labels_path="data/handlabels.jsonl"):
    lab = load_labels(labels_path)
    d = load_scores(scores_path)
    keys = list(lab)
    n_break = sum(1 for k in keys if lab[k]["first_break_step"])
    print(f"{len(keys)} labeled traces, {n_break} with a break, "
          f"{len({k[0] for k in keys})} distinct problems\n")
    print(f"{'scorer':<12} {'step AUC':>9} {'95% CI (by problem)':>21} {'trace-min sep':>14} {'lowest=break':>13}")
    for sc in sorted(d):
        scores = d[sc]
        ok, bad = step_sets(scores, lab, keys)
        a = auc(ok, bad)
        if a is None:
            continue
        lo, hi = cluster_boot(scores, lab, keys)
        nb = [min(scores[k].values()) for k in keys if scores.get(k) and not lab[k]["first_break_step"]]
        br = [min(scores[k].values()) for k in keys if scores.get(k) and lab[k]["first_break_step"]]
        hit = tot = 0
        for k in keys:
            brk = lab[k]["first_break_step"]
            if brk and scores.get(k):
                m = min(scores[k].values())
                tot += 1
                hit += brk in [i + 1 for i, s in scores[k].items() if s == m]
        print(f"{sc:<12} {a:>9.2f} {f'[{lo:.2f}, {hi:.2f}]':>21} {auc(nb, br):>14.2f} {f'{hit}/{tot}':>13}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/scores14b.jsonl")
