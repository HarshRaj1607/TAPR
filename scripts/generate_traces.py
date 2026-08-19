"""
Checkpoint 1 — Trace generation.

Generates model traces on GSM8K for later bucketing and scoring.
Inference only. No training.

Output: traces.jsonl  (one JSON object per line)

Usage:
    python scripts/generate_traces.py --n_problems 30 --n_samples 5        # pilot
    python scripts/generate_traces.py --n_problems 150 --n_samples 5       # full run

Resumes automatically if interrupted — already-generated (problem_id, sample_id)
pairs are skipped.
"""

import argparse
import json
import os
import time

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

PROMPT = """Solve this problem step by step.
Number each step (1., 2., 3., ...).
End your response with the final answer on its own line in the form: #### <number>

{question}"""


# ── io ────────────────────────────────────────────────────────────────────────

def load_done(path):
    """
    Read existing output and return the set of (problem_id, sample_id) already
    generated, so an interrupted run can resume instead of starting over.
    """
    done = set()
    if not os.path.exists(path):
        return done
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                done.add((r["problem_id"], r["sample_id"]))
            except (json.JSONDecodeError, KeyError):
                # a partially-written final line from a hard crash — ignore it
                continue
    return done


def append_records(path, records):
    """Append records and flush to disk immediately, so a crash loses at most one batch."""
    with open(path, "a") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
        f.flush()
        os.fsync(f.fileno())


# ── generation ────────────────────────────────────────────────────────────────

def build_batch(tok, questions):
    """Turn a list of questions into a padded, tokenized batch."""
    texts = []
    for q in questions:
        messages = [{"role": "user", "content": PROMPT.format(question=q)}]
        texts.append(
            tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        )
    # padding=True pads every sequence to the longest in the batch.
    # With padding_side="left" (set at load time) the real tokens all end at the
    # same position, so generation continues from the correct place for each row.
    return tok(texts, return_tensors="pt", padding=True)


def generate_batch(model, tok, questions, max_new, temperature, top_p):
    inputs = build_batch(tok, questions).to(model.device)
    prompt_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tok.pad_token_id,
        )

    # Every row shares the same prompt_len because of left padding, so the same
    # slice index is correct for all of them.
    traces = []
    for row in out:
        new_tokens = row[prompt_len:]
        traces.append(tok.decode(new_tokens, skip_special_tokens=True).strip())
    return traces


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_problems", type=int, default=30)
    ap.add_argument("--n_samples", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=10)
    ap.add_argument("--max_new", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--out", type=str, default="data/traces.jsonl")
    ap.add_argument("--split", type=str, default="test")
    args = ap.parse_args()

    print(f"loading gsm8k ({args.split}) ...")
    ds = load_dataset("openai/gsm8k", "main", split=args.split)

    print(f"loading {MODEL_NAME} ...")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    # Left padding is REQUIRED for batched generation on a decoder-only model.
    # With right padding, pad tokens sit between the prompt and the first
    # generated token, and the model continues from padding — producing garbage
    # with no error raised.
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    print("loaded.\n")

    done = load_done(args.out)
    if done:
        print(f"resuming — {len(done)} traces already present in {args.out}\n")

    # Flat list of every (problem, sample) job still to do.
    jobs = []
    for pid in range(args.n_problems):
        for sid in range(args.n_samples):
            if (pid, sid) not in done:
                jobs.append((pid, sid))

    if not jobs:
        print("nothing to do — all traces already generated.")
        return

    print(f"{len(jobs)} traces to generate, batch size {args.batch_size}\n")

    t0 = time.time()
    n_done = 0

    for start in range(0, len(jobs), args.batch_size):
        batch = jobs[start: start + args.batch_size]
        questions = [ds[pid]["question"] for pid, _ in batch]

        traces = generate_batch(
            model, tok, questions,
            max_new=args.max_new,
            temperature=args.temperature,
            top_p=args.top_p,
        )

        records = []
        for (pid, sid), trace in zip(batch, traces):
            records.append({
                "problem_id": pid,
                "sample_id": sid,
                "question": ds[pid]["question"],
                "gold_answer": ds[pid]["answer"],   # raw field, <<>> and #### intact
                "model_trace": trace,
                "gen_config": {
                    "model": MODEL_NAME,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "max_new_tokens": args.max_new,
                },
            })

        append_records(args.out, records)

        n_done += len(batch)
        elapsed = time.time() - t0
        rate = n_done / elapsed
        remaining = (len(jobs) - n_done) / rate if rate > 0 else 0
        print(f"[{n_done}/{len(jobs)}]  {rate:.2f} traces/s  eta {remaining/60:.1f} min")

    print(f"\ndone — {n_done} traces written to {args.out} in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
