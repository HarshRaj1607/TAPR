"""
Read scores.jsonl / cal.jsonl and report what we need to set the bars.

Run:  python scripts/analyze_scores.py data/cal.jsonl
"""

import json
import sys
from collections import defaultdict
from statistics import mean, stdev

SCORERS = ["isolated", "transition", "bare"]


def load(path):
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if r.get("parse_failed"):
                    continue
                s = r.get("score")
                if s is None or s == "NA":     # NA = judge says not a transition
                    continue
                rows.append(r)
    return rows


def hist(vals, width=40):
    """Crude histogram of 0-10 scores."""
    counts = defaultdict(int)
    for v in vals:
        counts[int(round(v))] += 1
    top = max(counts.values()) if counts else 1
    lines = []
    for s in range(11):
        n = counts.get(s, 0)
        bar = "#" * int(width * n / top) if top else ""
        lines.append(f"   {s:2}  {n:4}  {bar}")
    return "\n".join(lines)


def main(path):
    rows = load(path)
    print(f"{len(rows)} scored items\n")

    # ── 1. what does each scorer's distribution look like? ───────────────────
    print("=" * 62)
    print("1. SCORE DISTRIBUTION PER SCORER")
    print("   (if everything piles up at 10, the scorer isn't discriminating)")
    print("=" * 62)
    for sc in SCORERS:
        vals = [r["score"] for r in rows if r["scorer"] == sc]
        if not vals:
            continue
        sd = stdev(vals) if len(vals) > 1 else 0.0
        pct10 = sum(1 for v in vals if v >= 10) / len(vals)
        print(f"\n{sc}:  n={len(vals)}  mean={mean(vals):.2f}  sd={sd:.2f}  "
              f"at-ceiling(10)={pct10:.0%}")
        print(hist(vals))

    # ── 2. per-trace aggregation, split by answer correctness ────────────────
    # group scores by (problem, sample, scorer)
    by_trace = defaultdict(lambda: defaultdict(list))
    correct_of = {}
    for r in rows:
        key = (r["problem_id"], r["sample_id"])
        by_trace[key][r["scorer"]].append(r["score"])
        correct_of[key] = r["answer_correct"]

    print("\n" + "=" * 62)
    print("2. PER-TRACE SCORES, SPLIT BY WHETHER THE ANSWER WAS RIGHT")
    print("   primary aggregation = MIN (one broken link breaks the chain)")
    print("=" * 62)

    for agg_name, agg in [("min", min), ("mean", mean)]:
        print(f"\n--- aggregation: {agg_name} ---")
        print(f"{'scorer':<12} {'right-ans':>12} {'wrong-ans':>12} {'gap':>8}")
        for sc in SCORERS:
            right = [agg(v[sc]) for k, v in by_trace.items() if correct_of[k] and v.get(sc)]
            wrong = [agg(v[sc]) for k, v in by_trace.items() if not correct_of[k] and v.get(sc)]
            if not right or not wrong:
                continue
            gap = mean(right) - mean(wrong)
            print(f"{sc:<12} {mean(right):>12.2f} {mean(wrong):>12.2f} {gap:>8.2f}")
        # separation: pick one right + one wrong at random, how often right scores higher?
        print(f"\n{'scorer':<12} {'separation':>12}   (right > wrong, over all pairs)")
        for sc in SCORERS:
            right = [agg(v[sc]) for k, v in by_trace.items() if correct_of[k] and v.get(sc)]
            wrong = [agg(v[sc]) for k, v in by_trace.items() if not correct_of[k] and v.get(sc)]
            if not right or not wrong:
                continue
            wins = sum(1 for a in right for b in wrong if a > b)
            ties = sum(1 for a in right for b in wrong if a == b)
            total = len(right) * len(wrong)
            print(f"{sc:<12} {(wins + 0.5*ties)/total:>12.1%}")

    # ── 3. within-outcome spread ─────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("3. WITHIN-OUTCOME SPREAD")
    print("   traces of the SAME problem that got the SAME answer.")
    print("   sd must be clearly > 0 or the signal adds nothing over outcome reward.")
    print("=" * 62)

    groups = defaultdict(list)
    for key, scs in by_trace.items():
        groups[(key[0], correct_of[key])].append((key, scs))

    for sc in SCORERS:
        sds = []
        for gk, members in groups.items():
            vals = [min(m[1][sc]) for m in members if m[1].get(sc)]
            if len(vals) >= 2:
                sds.append(stdev(vals))
        if sds:
            n_zero = sum(1 for s in sds if s == 0)
            print(f"{sc:<12} groups={len(sds):3}  mean sd={mean(sds):.2f}  "
                  f"zero-variance groups={n_zero}/{len(sds)}")

    # ── 4. do the scorers agree with each other? ─────────────────────────────
    print("\n" + "=" * 62)
    print("4. SCORER AGREEMENT (per step, same step scored by two scorers)")
    print("=" * 62)
    per_step = defaultdict(dict)
    for r in rows:
        per_step[(r["problem_id"], r["sample_id"], r["step_idx"])][r["scorer"]] = r["score"]

    for a, b in [("isolated", "transition"), ("isolated", "bare"), ("transition", "bare")]:
        pairs = [(v[a], v[b]) for v in per_step.values() if a in v and b in v]
        if not pairs:
            continue
        diffs = [x - y for x, y in pairs]
        same = sum(1 for d in diffs if d == 0) / len(diffs)
        big = sum(1 for d in diffs if abs(d) >= 3) / len(diffs)
        print(f"{a} vs {b}:  n={len(pairs)}  mean diff={mean(diffs):+.2f}  "
              f"identical={same:.0%}  differ>=3={big:.0%}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/cal.jsonl")
