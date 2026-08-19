"""
Checkpoint 1 — scoring.

Scores every step of every usable trace with THREE scorers:

  isolated    sees: question + ONE step.            "is this statement valid?"
  transition  sees: question + ALL prior steps + this step.  "does this move follow?"
  bare        sees: same as transition, but NO criteria given.

The bare scorer is the control: if it discriminates as well as `transition`,
our three explicit conditions aren't adding anything over the model's own
notion of entailment. That turns a philosophical question into a measurement.

Prefix-only is guaranteed by SLICING (steps[:i] never includes later steps),
not by asking the model politely.

Output: scores.jsonl — one row per (trace, step, scorer). Raw response kept.

Usage:
    pip install bitsandbytes accelerate
    python scripts/score_traces.py --traces data/traces.jsonl --out data/scores.jsonl
    python scripts/score_traces.py --traces data/traces.jsonl --out data/cal.jsonl --limit 30   # calibration
"""

import argparse
import json
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1] / "src"))

from tapr.parsing import parse_trace


JUDGE_MODEL = "Qwen/Qwen2.5-7B-Instruct"   # override with --judge_model


# ── prompts ───────────────────────────────────────────────────────────────────

ISOLATED_PROMPT = """You are checking one statement from a reasoning trace.

Problem: {question}

Statement:
{step}

Judge ONLY this statement, on its own. You have no access to any other step,
so do not guess what came before.

First do the check, then score.

CHECK: State the arithmetic or mathematical claim this statement makes, and
verify it. Is the arithmetic correct? Is the claim well-defined?

Then score 0-10:
  0-3   the arithmetic is wrong
  4-6   no clear mathematical claim, or too vague to verify
  7-8   correct but imprecisely stated
  9-10  arithmetic correct and the claim is well-defined

Respond in exactly this format:
CHECK: <your check>
SCORE: <integer 0-10>"""


TRANSITION_PROMPT = """You are checking whether one reasoning move is valid.

Problem: {question}

Reasoning so far:
{prior}

Next move:
{step}

FIRST decide whether this is a transition at all.
A transition performs an inference: it concludes, substitutes, simplifies, or
derives something from what came before. A move that merely restates a fact
from the problem, or introduces a piece of common knowledge without inferring
anything, is NOT a transition. If it is not a transition, write SCORE: NA.

If it IS a transition, work through all three checks BEFORE scoring.
Be skeptical. Do not assume a number is legitimate because it sounds plausible.

CHECK 1 - PREMISES. List every number the move uses. Put each into one bucket:
  GIVEN        - stated in the problem, or derived in an earlier step
  WORLD        - a genuine fact not in the problem (e.g. a dozen is 12),
                 and nothing in the problem says otherwise
  CONTRADICTS  - the problem already specified this quantity, and the move
                 uses a different value for it
  FABRICATED   - appears from nowhere and is not a genuine world fact

CHECK 2 - OPERATION. State the operation performed. Is it legitimate here,
and is the arithmetic correct?

CHECK 3 - DIRECTION. Does this move get closer to what the problem asks, or
does it answer a different question or return to an earlier state?

Then score 0-10:
  0-3   any premise is CONTRADICTS or FABRICATED, or the arithmetic is wrong
  4-6   premises and arithmetic fine, but the move does not advance the problem
  7-8   valid and advancing, with minor issues
  9-10  all three checks pass cleanly

A correct operation on a CONTRADICTS or FABRICATED premise still scores 0-3.
WORLD premises are acceptable and cost nothing.

If you are unsure whether a premise contradicts the problem, or unsure whether
the move is valid, score it LOW. Do not resolve doubt in favour of the move.

Judge only on the reasoning shown above. Do not assume later steps.

Respond in exactly this format:
CHECK 1: <each number and its bucket>
CHECK 2: <operation>
CHECK 3: <direction>
SCORE: <integer 0-10, or NA if not a transition>"""


BARE_PROMPT = """You are checking whether one reasoning move is valid.

Problem: {question}

Reasoning so far:
{prior}

Next move:
{step}

Does this move follow from what came before? Think it through before scoring.

Respond in exactly this format:
REASONING: <your reasoning>
SCORE: <integer 0-10>"""




# ── UNBANDED VARIANTS (0-1 continuous, no bands, no penalty hierarchy) ────────
# Hypothesis: the discrete bands in TRANSITION_PROMPT collapsed the scale
# (448/519 scores landed on exactly 9, none on 10). These state the metrics
# and nothing about how to map them onto numbers.

TRANSITION_FREE_PROMPT = """You are judging one reasoning move.

Problem: {question}

Reasoning so far:
{prior}

Next move:
{step}

Judge the quality of this move on these metrics:
- the values it uses are stated in the problem or derived in the reasoning so far,
  and none contradict what the problem says
- the operation is legitimate and the arithmetic is correct
- the move advances toward what the problem asks

Score the quality of this move between 0 and 1.

Respond in exactly this format:
REASONING: <your reasoning>
SCORE: <number between 0 and 1>"""


BARE_FREE_PROMPT = """You are judging one reasoning move.

Problem: {question}

Reasoning so far:
{prior}

Next move:
{step}

Does this move follow from what came before?

Score the quality of this move between 0 and 1.

Respond in exactly this format:
REASONING: <your reasoning>
SCORE: <number between 0 and 1>"""


NO_PRIOR = "(none - this is the first step)"


# ── score parsing ─────────────────────────────────────────────────────────────

import re
_SCORE_RE = re.compile(r"SCORE\s*:\s*(NA|-?\d+(?:\.\d+)?)", re.IGNORECASE)
_REASON_RE = re.compile(r"REASON\s*:\s*(.+)", re.IGNORECASE | re.DOTALL)


def parse_response(text, scale_01=False):
    """Return (score, reason, parse_failed). Never guesses a default value."""
    m = _SCORE_RE.findall(text)
    if not m:
        return None, None, True
    tok = m[-1]                       # SCORE line comes LAST now
    if tok.upper() == "NA":           # judge says this step is not a transition
        reason = text.split("SCORE:")[0].strip()[:600]
        return "NA", reason, False
    try:
        score = float(tok)
    except ValueError:
        return None, None, True
    if not (0 <= score <= 10):
        return None, None, True          # out of range = malformed, not a real score
    if scale_01:
        if score > 1.0:                   # judge ignored the 0-1 instruction
            return None, None, True
        score = score * 10.0              # rescale to the common 0-10 axis
    reason = text.split("SCORE:")[0].strip()[:600]   # the checks, for inspection
    return score, reason, False


# ── job construction ──────────────────────────────────────────────────────────

def build_jobs(traces_path, limit=None):
    """
    One job per (trace, step, scorer). Prefix-only enforced here: for step i the
    transition/bare prompts only ever see steps[:i].
    """
    jobs = []
    n_traces = 0
    with open(traces_path) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            p = parse_trace(rec)
            if p["status"] != "ok":
                continue
            if limit is not None and n_traces >= limit:
                break
            n_traces += 1

            steps = p["steps"]
            for i, step in enumerate(steps):
                prior = "\n\n".join(steps[:i]) if i > 0 else NO_PRIOR
                base = {
                    "problem_id": p["problem_id"],
                    "sample_id": p["sample_id"],
                    "step_idx": i,
                    "n_steps": len(steps),
                    "answer_correct": p["answer_correct"],
                }
                jobs.append({**base, "scorer": "isolated",
                             "prompt": ISOLATED_PROMPT.format(question=p["question"], step=step)})
                jobs.append({**base, "scorer": "transition",
                             "prompt": TRANSITION_PROMPT.format(question=p["question"], prior=prior, step=step)})
                jobs.append({**base, "scorer": "bare",
                             "prompt": BARE_PROMPT.format(question=p["question"], prior=prior, step=step)})
                jobs.append({**base, "scorer": "transition_free",
                             "prompt": TRANSITION_FREE_PROMPT.format(question=p["question"], prior=prior, step=step)})
                jobs.append({**base, "scorer": "bare_free",
                             "prompt": BARE_FREE_PROMPT.format(question=p["question"], prior=prior, step=step)})
    return jobs, n_traces


def job_key(j):
    return (j["problem_id"], j["sample_id"], j["step_idx"], j["scorer"])


def load_done(path):
    done = set()
    if not os.path.exists(path):
        return done
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
                done.add((r["problem_id"], r["sample_id"], r["step_idx"], r["scorer"]))
            except (json.JSONDecodeError, KeyError):
                continue
    return done


# ── model ─────────────────────────────────────────────────────────────────────

def load_judge(model_name=JUDGE_MODEL):
    if not torch.cuda.is_available():
        raise RuntimeError("No GPU. Switch Colab runtime to GPU.")
    print(f"loading {model_name} (4-bit) ...")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.padding_side = "left"                 # required for batched generation
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb, device_map="auto",
    )
    model.eval()
    print(f"loaded. VRAM {torch.cuda.memory_allocated()/1e9:.1f} GB\n")
    return model, tok


def run_batch(model, tok, prompts, max_new):
    texts = [
        tok.apply_chat_template([{"role": "user", "content": p}],
                                tokenize=False, add_generation_prompt=True)
        for p in prompts
    ]
    inputs = tok(texts, return_tensors="pt", padding=True).to(model.device)
    plen = inputs["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new,
            do_sample=False,                  # deterministic judging
            pad_token_id=tok.pad_token_id,
        )
    return [tok.decode(row[plen:], skip_special_tokens=True).strip() for row in out]



def make_batches(jobs, tok, max_tokens, max_batch):
    """
    Group jobs into batches by TOKEN BUDGET, not fixed count.

    Every sequence in a batch is padded to the longest one, so cost is
    batch_size x longest_prompt. Sorting by length makes batches homogeneous:
    short prompts get large batches, long prompts get small ones. Less padding
    waste, and far less OOM risk on the long-prompt batches.
    """
    lens = [len(tok(j["prompt"])["input_ids"]) for j in jobs]
    order = sorted(range(len(jobs)), key=lambda i: lens[i])

    batches, cur, cur_max = [], [], 0
    for i in order:
        new_max = max(cur_max, lens[i])
        if cur and (new_max * (len(cur) + 1) > max_tokens or len(cur) >= max_batch):
            batches.append(cur)
            cur, cur_max = [], 0
            new_max = lens[i]
        cur.append(jobs[i])
        cur_max = new_max
    if cur:
        batches.append(cur)
    return batches


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default="data/traces.jsonl")
    ap.add_argument("--out", default="data/scores.jsonl")
    ap.add_argument("--batch_size", type=int, default=16, help="max items per batch")
    ap.add_argument("--max_tokens", type=int, default=9000,
                    help="padded token budget per batch; lower this if you OOM")
    ap.add_argument("--max_new", type=int, default=320)
    ap.add_argument("--limit", type=int, default=None,
                    help="only score the first N usable traces (use 30 for calibration)")
    ap.add_argument("--judge_model", type=str, default=JUDGE_MODEL,
                    help="e.g. Qwen/Qwen2.5-14B-Instruct or Qwen/Qwen2.5-32B-Instruct")
    ap.add_argument("--only", type=str, default=None,
                    help="score only these traces, e.g. 3/0,2/1,5/0")
    ap.add_argument("--scorers", type=str, default=None,
                    help="comma list, e.g. transition_free,bare_free")
    args = ap.parse_args()

    jobs, n_traces = build_jobs(args.traces, None if args.only else args.limit)

    if args.only:
        wanted = set()
        for pair in args.only.split(","):
            p, s = pair.strip().split("/")
            wanted.add((int(p), int(s)))
        jobs = [j for j in jobs if (j["problem_id"], j["sample_id"]) in wanted]
        n_traces = len(wanted)

    if args.scorers:
        keep = {s.strip() for s in args.scorers.split(",")}
        jobs = [j for j in jobs if j["scorer"] in keep]

    print(f"{n_traces} traces -> {len(jobs)} judge calls")
    print(f"judge: {args.judge_model}")

    done = load_done(args.out)
    if done:
        jobs = [j for j in jobs if job_key(j) not in done]
        print(f"resuming — {len(done)} already scored, {len(jobs)} remaining")
    if not jobs:
        print("nothing to do.")
        return

    model, tok = load_judge(args.judge_model)

    batches = make_batches(jobs, tok, args.max_tokens, args.batch_size)
    print(f"{len(batches)} batches (token budget {args.max_tokens}, max {args.batch_size}/batch)")

    t0 = time.time()
    n_fail = 0
    n_seen = 0
    with open(args.out, "a") as fout:
        for batch in batches:
            responses = run_batch(model, tok, [j["prompt"] for j in batch], args.max_new)

            for j, resp in zip(batch, responses):
                score, reason, failed = parse_response(resp, scale_01=j["scorer"].endswith("_free"))
                n_fail += int(failed)
                row = {k: j[k] for k in
                       ("problem_id", "sample_id", "step_idx", "n_steps",
                        "answer_correct", "scorer")}
                row.update({"score": score, "reason": reason,
                            "parse_failed": failed, "raw": resp[:900]})
                fout.write(json.dumps(row) + "\n")
            fout.flush()
            os.fsync(fout.fileno())

            n_seen += len(batch)
            n = n_seen
            el = time.time() - t0
            rate = n / el
            print(f"[{n}/{len(jobs)}]  {rate:.1f} calls/s  "
                  f"eta {(len(jobs)-n)/rate/60:.1f} min  parse_fail {n_fail}")

    print(f"\ndone in {(time.time()-t0)/60:.1f} min")
    print(f"parse failures: {n_fail}/{len(jobs)} ({n_fail/len(jobs):.1%})")
    if n_fail / len(jobs) > 0.05:
        print("WARNING: >5% parse failures — check the prompt format before trusting these.")


if __name__ == "__main__":
    main()
