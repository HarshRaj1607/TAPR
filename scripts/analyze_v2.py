"""
Analysis v2 — the question: do context-aware scorers (transition, bare) tell
broken reasoning from sound reasoning better than the context-free one (isolated)?

Two fixes over v1:
  1. EXCLUDES format failures. A trace that reasoned correctly to the gold value
     and then mistyped the marker (computed 70,000, wrote "#### 7.0") is NOT
     broken reasoning. Leaving it in the "wrong" group is contamination.
  2. Bootstrap confidence intervals on separation, so we can tell a real gap
     from noise at this sample size.

Run:  python scripts/analyze_v2.py data/scores14b.jsonl data/traces.jsonl
"""

import json
import random
import sys
from collections import defaultdict
from statistics import mean, stdev

import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1] / "src"))

from tapr.parsing import parse_trace

SCORERS = ["isolated", "transition", "bare"]
random.seed(0)


def load(scores_path, traces_path):
    # which traces are format failures rather than reasoning failures?
    fmt_fail, parsed = set(), {}
    with open(traces_path) as f:
        for line in f:
            if line.strip():
                p = parse_trace(json.loads(line))
                key = (p["problem_id"], p["sample_id"])
                parsed[key] = p
                if p.get("likely_format_failure"):
                    fmt_fail.add(key)

    rows = []
    with open(scores_path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("parse_failed"):
                continue
            s = r.get("score")
            if s is None or s == "NA":
                continue
            rows.append(r)
    return rows, fmt_fail, parsed


def separation(right, wrong):
    """P(a random sound trace scores above a random broken one). 50% = chance."""
    if not right or not wrong:
        return None
    wins = sum(1 for a in right for b in wrong if a > b)
    ties = sum(1 for a in right for b in wrong if a == b)
    return (wins + 0.5 * ties) / (len(right) * len(wrong))


def boot_ci(right, wrong, n=2000):
    """Bootstrap 95% CI: resample traces, recompute separation."""
    vals = []
    for _ in range(n):
        r = [random.choice(right) for _ in right]
        w = [random.choice(wrong) for _ in wrong]
        s = separation(r, w)
        if s is not None:
            vals.append(s)
    vals.sort()
    return vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals))]


def main(scores_path, traces_path):
    rows, fmt_fail, parsed = load(scores_path, traces_path)

    by_trace = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_trace[(r["problem_id"], r["sample_id"])][r["scorer"]].append(r["score"])

    n_wrong_all = sum(1 for k in by_trace if parsed[k]["answer_correct"] is False)
    n_fmt = len(fmt_fail & set(by_trace))
    print(f"{len(by_trace)} scored traces")
    print(f"wrong-answer traces: {n_wrong_all}")
    print(f"  of which FORMAT failures (reasoning reached gold): {n_fmt}")
    print(f"  genuinely broken reasoning: {n_wrong_all - n_fmt}\n")

    for agg_name, agg in [("min", min), ("mean", mean)]:
        print("=" * 62)
        print(f"SEPARATION — aggregation: {agg_name}")
        print("  sound = correct answer | broken = wrong answer, format failures REMOVED")
        print("=" * 62)
        print(f"{'scorer':<12} {'sound':>7} {'broken':>7} {'gap':>7} {'separation':>12} {'95% CI':>18}")
        for sc in SCORERS:
            sound, broken = [], []
            for k, v in by_trace.items():
                if not v.get(sc):
                    continue
                if parsed[k]["answer_correct"]:
                    sound.append(agg(v[sc]))
                elif k not in fmt_fail:
                    broken.append(agg(v[sc]))
            if not sound or not broken:
                continue
            sep = separation(sound, broken)
            lo, hi = boot_ci(sound, broken)
            print(f"{sc:<12} {mean(sound):>7.2f} {mean(broken):>7.2f} "
                  f"{mean(sound)-mean(broken):>7.2f} {sep:>11.1%} "
                  f"{f'[{lo:.1%}, {hi:.1%}]':>18}")
        print(f"   n sound={len(sound)}  n broken={len(broken)}\n")

    # ── context-aware vs context-free ────────────────────────────────────────
    print("=" * 62)
    print("CONTEXT-AWARE (transition, bare) vs CONTEXT-FREE (isolated)")
    print("  does seeing the prior chain help at all?")
    print("=" * 62)
    for agg_name, agg in [("min", min), ("mean", mean)]:
        ctx, free = [], []
        for k, v in by_trace.items():
            label = ("sound" if parsed[k]["answer_correct"]
                     else ("fmt" if k in fmt_fail else "broken"))
            if label == "fmt":
                continue
            for sc in ("transition", "bare"):
                if v.get(sc):
                    ctx.append((label, agg(v[sc])))
            if v.get("isolated"):
                free.append((label, agg(v["isolated"])))
        for name, data in [("context-aware", ctx), ("context-free", free)]:
            s = [x for l, x in data if l == "sound"]
            b = [x for l, x in data if l == "broken"]
            if s and b:
                print(f"  {agg_name:<5} {name:<15} separation={separation(s,b):.1%}")
        print()

    # ── within-outcome spread ────────────────────────────────────────────────
    print("=" * 62)
    print("WITHIN-OUTCOME SPREAD (the GRPO engine)")
    print("  same problem, same final answer -> do scores still differ?")
    print("  sd = 0 means no gradient: the signal adds nothing over outcome reward")
    print("=" * 62)
    groups = defaultdict(list)
    for k, v in by_trace.items():
        groups[(k[0], parsed[k]["answer_correct"])].append(v)

    print(f"{'scorer':<12} {'groups':>7} {'mean sd':>9} {'zero-var':>10} {'usable':>8}")
    for sc in SCORERS:
        sds = []
        for gk, members in groups.items():
            vals = [min(m[sc]) for m in members if m.get(sc)]
            if len(vals) >= 2:
                sds.append(stdev(vals))
        if sds:
            z = sum(1 for s in sds if s == 0)
            print(f"{sc:<12} {len(sds):>7} {mean(sds):>9.2f} "
                  f"{f'{z}/{len(sds)}':>10} {(len(sds)-z)/len(sds):>7.0%}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/scores14b.jsonl",
         sys.argv[2] if len(sys.argv) > 2 else "data/traces.jsonl")
