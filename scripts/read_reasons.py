"""
Read what the judges actually SAID on specific traces.

The numbers told us the transition scorer gives 10 to almost everything.
This tells us WHY — did it notice the problem and score 10 anyway, or did
it not notice at all? Different problems, different fixes.

Run:
    python scripts/read_reasons.py data/cal.jsonl data/traces.jsonl              # worst-scoring traces
    python scripts/read_reasons.py data/cal.jsonl data/traces.jsonl 3 0          # a specific problem/sample
"""

import json
import sys

from collections import defaultdict

import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1] / "src"))


def load_scores(path):
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_traces(path):
    d = {}
    with open(path) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                d[(r["problem_id"], r["sample_id"])] = r
    return d


def show(pid, sid, scores, traces):
    rec = traces.get((pid, sid))
    print("=" * 72)
    print(f"PROBLEM {pid} / SAMPLE {sid}")
    print("=" * 72)
    if rec:
        print("QUESTION:")
        print("  " + rec["question"][:400])
        gold = rec["gold_answer"].split("####")[-1].strip()
        print(f"\nGOLD ANSWER: {gold}")

    # gather this trace's scores, keyed by step
    by_step = defaultdict(dict)
    for r in scores:
        if r["problem_id"] == pid and r["sample_id"] == sid:
            by_step[r["step_idx"]][r["scorer"]] = r

    if rec:
        # re-segment so we can print the step text alongside
        try:
            from tapr.parsing import parse_trace
            steps = parse_trace(rec)["steps"]
        except Exception:
            steps = []
    else:
        steps = []

    for i in sorted(by_step):
        print(f"\n--- STEP {i} " + "-" * 58)
        if i < len(steps):
            txt = steps[i].replace("\n", " ")
            print(f"TEXT: {txt[:300]}")
        for sc in ("isolated", "transition", "bare"):
            r = by_step[i].get(sc)
            if not r:
                continue
            print(f"  {sc:<11} score={r['score']}")
            if r.get("reason"):
                print(f"              reason: {r['reason'][:220]}")
    print()


def main():
    scores_path = sys.argv[1] if len(sys.argv) > 1 else "data/cal.jsonl"
    traces_path = sys.argv[2] if len(sys.argv) > 2 else "data/traces.jsonl"

    scores = load_scores(scores_path)
    traces = load_traces(traces_path)

    if len(sys.argv) > 4:
        show(int(sys.argv[3]), int(sys.argv[4]), scores, traces)
        return

    # default: show wrong-answer traces, lowest transition score first
    trace_min = {}
    for r in scores:
        if r["scorer"] != "transition" or r.get("score") is None:
            continue
        k = (r["problem_id"], r["sample_id"])
        if r["answer_correct"]:
            continue
        trace_min[k] = min(trace_min.get(k, 99), r["score"])

    print(f"{len(trace_min)} wrong-answer traces in this file")
    print("showing the 3 with the LOWEST transition score "
          "(if even these scored 10, the scorer is stuck)\n")

    for k, _ in sorted(trace_min.items(), key=lambda kv: kv[1])[:3]:
        show(k[0], k[1], scores, traces)


if __name__ == "__main__":
    main()
