"""
Checkpoint 1 — parsing layer (v2).

Turns a raw trace into structured data. No LLM here, no judging.
Everything that can fail returns a flag instead of crashing, so we can
COUNT failures rather than lose traces silently.

v2 changes, driven by diagnosing a 24% loss on the pilot:
  - steps: also accept "Step 1:" / "### Step 1:" / "**Step 1:**", not just "1."
    (19/150 traces used these and segmented to zero steps — their lengths were
     normal, so it was a FORMAT problem, not truncation)
  - final answer: fall back to \boxed{...} when #### is absent
    (several traces were COMPLETE but used \boxed instead of ####)
  - guard against splitting on decimals like "3.5" inside LaTeX
"""

import re


# ── steps ─────────────────────────────────────────────────────────────────────
# Matches a line starting a numbered step in any observed format:
#   "1. "     "Step 1: "     "### Step 1: "     "**Step 1:**"
# The (?!\d) after \d+\. stops "3.5" (LaTeX) being read as a step marker.
_STEP_SPLIT_RE = re.compile(
    r"(?m)^\s*(?:#{1,6}\s*)?(?:\*\*)?\s*(?:Step\s+\d+\s*[:.]|\d+\.(?!\d))\s*",
    re.IGNORECASE,
)


def segment_steps(reasoning_text):
    """Split numbered reasoning into step strings. Text before step 1 is intro; dropped."""
    parts = _STEP_SPLIT_RE.split(reasoning_text)
    return [p.strip() for p in parts[1:] if p.strip()]


# ── numbers ───────────────────────────────────────────────────────────────────

_FINAL_RE = re.compile(r"####\s*\$?\s*(-?[\d,]*\.?\d+)")
_BOXED_RE = re.compile(r"\\boxed\{\s*\$?\s*(-?[\d,]*\.?\d+)\s*\}")


def to_number(s):
    """'70,000' -> 70000.0 ; '7.0' -> 7.0 ; junk -> None"""
    if s is None:
        return None
    try:
        return float(s.replace(",", "").replace("$", "").strip())
    except (ValueError, AttributeError):
        return None


def extract_final_answer(text):
    """
    Return (value, method).
      method: "hash" if found via ####, "boxed" if via \boxed{}, None if neither.
    Takes the LAST match — models sometimes echo the marker before the real answer.
    """
    m = _FINAL_RE.findall(text)
    if m:
        return to_number(m[-1]), "hash"
    b = _BOXED_RE.findall(text)
    if b:
        return to_number(b[-1]), "boxed"
    return None, None


# ── one trace ─────────────────────────────────────────────────────────────────

def parse_trace(record):
    """
    record: one line of traces.jsonl. Never raises.

    status:
      "ok"              usable
      "no_final_marker" no #### and no \boxed  (usually genuine truncation)
      "no_steps"        segmentation found nothing
      "no_gold_answer"  gold field didn't parse (should never happen)
    """
    trace = record["model_trace"]

    model_answer, method = extract_final_answer(trace)
    gold_answer, _ = extract_final_answer(record["gold_answer"])

    # reasoning body = everything before the final-answer marker
    reasoning = trace.split("####")[0] if "####" in trace else trace
    steps = segment_steps(reasoning)

    if gold_answer is None:
        status = "no_gold_answer"
    elif model_answer is None:
        status = "no_final_marker"
    elif not steps:
        status = "no_steps"
    else:
        status = "ok"

    answer_correct = (
        None if (model_answer is None or gold_answer is None)
        else abs(model_answer - gold_answer) < 1e-6
    )

    # CONTAMINATION CHECK.
    # Several traces reason correctly to the gold value and then print a
    # mangled marker (e.g. computes 70,000 then writes "#### 7.0").
    # Those are OUTPUT-FORMAT failures, not reasoning failures, and must not
    # be counted as broken reasoning. Flag them: the marker says wrong, but
    # the gold value literally appears in the reasoning body.
    likely_format_failure = False
    if answer_correct is False and gold_answer is not None:
        nums_in_reasoning = set()
        for tok in re.findall(r"-?[\d,]*\.?\d+", reasoning):
            v = to_number(tok)
            if v is not None:
                nums_in_reasoning.add(round(v, 6))
        if round(gold_answer, 6) in nums_in_reasoning:
            likely_format_failure = True

    return {
        "problem_id": record["problem_id"],
        "sample_id": record["sample_id"],
        "question": record["question"],
        "steps": steps,
        "n_steps": len(steps),
        "model_answer": model_answer,
        "answer_method": method,      # "hash" | "boxed" | None — worth counting
        "gold_answer": gold_answer,
        "answer_correct": answer_correct,
        "likely_format_failure": likely_format_failure,
        "status": status,
    }


# ── self-test against the real failure cases from the pilot ───────────────────

if __name__ == "__main__":
    gold = "Janet sells <<16-3-4=9>>9 eggs.\nShe makes <<9*2=18>>18.\n#### 18"

    samples = [
        ("plain 1. + ####",   "1. a\n2. b\n\n#### 18"),
        ("comma",             "1. a\n2. b\n\n#### 70,000"),
        ("decimal marker",    "1. a\n2. b\n\n#### 7.0"),
        ("Step N: format",    "Step 1: Determine feed.\n\\[20 x 3 = 60\\]\n\nStep 2: Total.\n\n#### 18"),
        ("### Step N:",       "Let's solve.\n\n### Step 1: Price\n- $5\n\n### Step 2: Cost\n\n#### 18"),
        ("boxed, no ####",    "1. a\n2. b\n\nJosh made \\(\\boxed{18}\\)"),
        ("latex 3.5 guard",   "1. Speed\n   80 \\times 3.5 = 280\n2. Next\n\n#### 18"),
        ("truncated",         "1. a\n2. b - Feed = \\(60 - 40 \\text"),
    ]

    print(f"gold -> {extract_final_answer(gold)}\n")
    for name, tr in samples:
        o = parse_trace({"problem_id": 0, "sample_id": 0, "question": "q",
                         "model_trace": tr, "gold_answer": gold})
        print(f"{name:18} status={o['status']:16} ans={str(o['model_answer']):8} "
              f"via={str(o['answer_method']):6} steps={o['n_steps']} correct={o['answer_correct']}")
