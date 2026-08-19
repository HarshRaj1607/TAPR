"""
Step 2 — look at what the pilot actually produced.

CPU only. No judge, no GPU. Answers:
  - how many traces are usable at all?
  - what fraction got the wrong answer?
  - how many steps does a typical trace have?  (this sets judge-call cost)
  - do we have same-problem groups for the within-outcome variance check?

Run:  python scripts/inspect_traces.py data/traces.jsonl
"""

import json
import sys
from collections import Counter, defaultdict

import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1] / "src"))

from tapr.parsing import parse_trace


def main(path):
    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    parsed = [parse_trace(r) for r in records]

    print(f"total traces: {len(parsed)}\n")

    # ── usability ────────────────────────────────────────────────────────────
    status = Counter(p["status"] for p in parsed)
    print("STATUS")
    for k, v in status.most_common():
        print(f"  {k:18} {v:4}  ({v/len(parsed):.1%})")

    ok = [p for p in parsed if p["status"] == "ok"]
    print(f"\nusable: {len(ok)}\n")

    if ok:
        methods = Counter(p["answer_method"] for p in ok)
        print("ANSWER FOUND VIA")
        for k, v in methods.most_common():
            print(f"  {str(k):18} {v:4}")
        print()
    if not ok:
        print("nothing usable — stop and check the generation prompt.")
        return

    # ── the number that sets sample size ─────────────────────────────────────
    wrong = [p for p in ok if not p["answer_correct"]]
    right = [p for p in ok if p["answer_correct"]]
    print("ANSWER CORRECTNESS  (usable traces only)")
    print(f"  correct : {len(right):4}  ({len(right)/len(ok):.1%})")
    print(f"  wrong   : {len(wrong):4}  ({len(wrong)/len(ok):.1%})   <-- drives sample size")

    # ── judge cost ───────────────────────────────────────────────────────────
    steps = [p["n_steps"] for p in ok]
    steps_sorted = sorted(steps)
    median = steps_sorted[len(steps_sorted) // 2]
    total_steps = sum(steps)
    print(f"\nSTEPS PER TRACE")
    print(f"  min {min(steps)}  median {median}  max {max(steps)}  mean {total_steps/len(ok):.1f}")
    # 3 scorers: isolated, transition, bare
    print(f"\nJUDGE CALLS if we score all usable traces with 3 scorers:")
    print(f"  {total_steps} steps x 3 = {total_steps * 3} calls")

    # ── groups for within-outcome variance ───────────────────────────────────
    by_problem = defaultdict(list)
    for p in ok:
        by_problem[p["problem_id"]].append(p)

    groups_same_outcome = 0
    for pid, group in by_problem.items():
        n_right = sum(1 for g in group if g["answer_correct"])
        n_wrong = len(group) - n_right
        if n_right >= 2 or n_wrong >= 2:
            groups_same_outcome += 1

    print(f"\nWITHIN-OUTCOME GROUPS")
    print(f"  problems with >=2 traces sharing an outcome: "
          f"{groups_same_outcome}/{len(by_problem)}")
    print("  (these are the groups where within-outcome score spread is testable)")

    # ── a peek at wrong-answer traces ────────────────────────────────────────
    print(f"\nEXAMPLES OF WRONG-ANSWER TRACES (first 5)")
    for p in wrong[:5]:
        print(f"  p{p['problem_id']}/s{p['sample_id']}: "
              f"model={p['model_answer']} gold={p['gold_answer']} steps={p['n_steps']}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/traces.jsonl")
