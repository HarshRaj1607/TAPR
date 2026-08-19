"""
Diagnose the 24% loss before regenerating.

Two different failures, two different fixes:
  no_final_marker -> probably truncation      -> raise max_new_tokens
  no_steps        -> segmentation missed      -> fix prompt or regex

Run:  python scripts/diagnose_losses.py data/traces.jsonl
"""

import json
import sys

import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1] / "src"))

from tapr.parsing import parse_trace


def main(path):
    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    parsed = [(r, parse_trace(r)) for r in records]

    no_steps = [(r, p) for r, p in parsed if p["status"] == "no_steps"]
    no_marker = [(r, p) for r, p in parsed if p["status"] == "no_final_marker"]

    # ── no_steps: what format did the model use instead? ─────────────────────
    print("=" * 70)
    print(f"NO_STEPS  ({len(no_steps)} traces) — show first 3 in full")
    print("=" * 70)
    for r, p in no_steps[:3]:
        t = r["model_trace"]
        print(f"\n--- p{r['problem_id']}/s{r['sample_id']}  ({len(t)} chars) ---")
        print(t[:900])
        if len(t) > 900:
            print(f"... [{len(t)-900} more chars]")
        print()

    # ── no_final_marker: truncated, or just missing ####? ────────────────────
    print("=" * 70)
    print(f"NO_FINAL_MARKER  ({len(no_marker)} traces) — last 200 chars of first 5")
    print("=" * 70)
    for r, p in no_marker[:5]:
        t = r["model_trace"]
        print(f"\n--- p{r['problem_id']}/s{r['sample_id']}  ({len(t)} chars, {p['n_steps']} steps) ---")
        print("..." + t[-200:])

    # ── is truncation the cause? length tells us ─────────────────────────────
    print("\n" + "=" * 70)
    print("LENGTH CHECK — traces near the token cap are truncated")
    print("=" * 70)
    ok_lens = [len(r["model_trace"]) for r, p in parsed if p["status"] == "ok"]
    nm_lens = [len(r["model_trace"]) for r, p in no_marker]
    ns_lens = [len(r["model_trace"]) for r, p in no_steps]

    def summarize(name, lens):
        if not lens:
            print(f"  {name:18} (none)")
            return
        print(f"  {name:18} min {min(lens):5}  mean {sum(lens)//len(lens):5}  max {max(lens):5}")

    summarize("ok", ok_lens)
    summarize("no_final_marker", nm_lens)
    summarize("no_steps", ns_lens)
    print("\n  If no_final_marker lengths cluster near the max, it's truncation.")
    print("  If no_steps lengths look normal, it's a FORMAT problem, not length.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/traces.jsonl")
