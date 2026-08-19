"""
Did removing the bands open up the scale?

Banded transition parked on 9 for 86% of its scores and never used 10.
Hypothesis: the discrete bands created attractors and collapsed the scale.

Run:
    python scripts/check_spread.py data/cal_free.jsonl                 # the new 0-1 scorers
    python scripts/check_spread.py data/scores14b.jsonl data/cal_free.jsonl # compare old vs new
"""

import json
import sys
from collections import Counter
from statistics import mean, stdev


def load(paths):
    per = {}
    for p in paths:
        with open(p) as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get("parse_failed"):
                    continue
                s = r.get("score")
                if s is None or s == "NA":
                    continue
                per.setdefault(r["scorer"], []).append(float(s))
    return per


def show(name, vals):
    sd = stdev(vals) if len(vals) > 1 else 0.0
    rounded = [round(v, 1) for v in vals]
    c = Counter(rounded)
    modal, modal_n = c.most_common(1)[0]
    print(f"\n{name}")
    print(f"  n={len(vals)}  mean={mean(vals):.2f}  sd={sd:.2f}")
    print(f"  modal value {modal} used by {modal_n/len(vals):.0%} of scores")
    print(f"  distinct values used: {len(c)}")
    top = max(c.values())
    for v, n in sorted(c.items()):
        bar = "#" * int(40 * n / top)
        print(f"    {v:>5}  {n:>4}  {bar}")


def main(paths):
    per = load(paths)
    if not per:
        print("no usable scores found")
        return
    print("=" * 58)
    print("SCORE SPREAD PER SCORER   (_free scorers rescaled 0-1 -> 0-10)")
    print("=" * 58)
    for sc in sorted(per):
        show(sc, per[sc])

    print("\n" + "=" * 58)
    print("VERDICT")
    print("=" * 58)
    for sc in sorted(per):
        vals = per[sc]
        c = Counter(round(v, 1) for v in vals)
        frac = c.most_common(1)[0][1] / len(vals)
        sd = stdev(vals) if len(vals) > 1 else 0.0
        flag = "COLLAPSED" if frac > 0.6 else ("spread" if len(c) >= 5 else "narrow")
        print(f"  {sc:<18} modal={frac:>4.0%}  distinct={len(c):>3}  sd={sd:.2f}  -> {flag}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["data/cal_free.jsonl"])
