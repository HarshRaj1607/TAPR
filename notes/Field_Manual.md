# Field Manual — What We Did, and What It Generalises To

*Companion to `Intern_Machinery_Minimum.md`. Everything here is grounded in
errors and decisions from Checkpoint 1, not generic advice.*

---

# PART 1 — DEBUGGING

## The one habit that matters: read the traceback bottom-up

A traceback is a **call stack**, printed outermost-first. The last line is the
actual error. Everything above it is the path that got there — mostly library
internals you didn't write.

Most people panic and read from the top, land in `transformers/generation/utils.py`,
and conclude the library is broken. It isn't. Read the bottom line first, and only
walk upward if it isn't enough.

## The errors we actually hit, and what each one teaches

### 1. `Could not find a version that satisfies the requirement accelarate (from versions: none)`
**Cause:** typo — `accelarate` for `accelerate`.
**The tell:** `from versions: none`. That means PyPI has **no package by that name at all**.
A real package with a version conflict lists the versions it *does* have.
**Generalises to:** `none` = wrong name. A list = version problem.

### 2. `HfUriError: Repository id must be 'namespace/name', got 'gsm8k'`
**Cause:** `datasets` v4 requires `openai/gsm8k`; bare `gsm8k` was the old shorthand.
**Generalises to:** **version drift.** Code that worked six months ago breaks
because a library changed its contract. Fix: `pip freeze > requirements.txt` and
pin versions. This is the #1 cause of "it worked yesterday."

### 3. `SyntaxError: invalid syntax` on `python inspect_traces.py traces.jsonl`
**Cause:** ran a shell command in a Python cell. Needs `!` in Colab.
**Generalises to:** a SyntaxError on something that looks like a valid terminal
command is almost always a wrong-interpreter problem, not a code problem.

### 4. `ValueError: invalid literal for int() with base 10: 'Janet sells 16 - 3 - 4 = ...'`
**Cause:** `extract_final_answer` was written to receive the text *after* `####`,
but was handed the whole gold field.
**Caught by:** testing the function on CPU with a real example **before** spending GPU time.
**Generalises to:** **test parsing logic on real data before the expensive step.**
This one would have crashed all 150 traces.

### 5. `int("7.0")` → ValueError
**Cause:** assumed integer answers; the model emitted `#### 7.0`.
**Generalises to:** your parser encodes assumptions. The data will violate them.

### 6. `ValueError: not enough values to unpack (expected 2, got 1)`
**Cause:** `trace.split("####", 1)` on a truncated trace with no `####`.
**Generalises to:** any `split` that you unpack into fixed variables will explode
when the delimiter is missing. Check first, or handle the short case.

### 7. `argument --out: expected one argument`
**Cause:** the value never reached argparse (line break / stray space).
**The tell:** "expected one argument" always means the flag's value was **absent**,
never that it was *wrong*.

### 8. `FileNotFoundError: 'cal.jsonl'`
**Cause:** wrote output to Drive, read it from the local directory.
**Generalises to:** when a file "vanishes," check the path before the logic.

### 9. `ImportError: Using bitsandbytes 4-bit quantization requires bitsandbytes`
**Cause:** Colab runtime reset. `pip install`s don't survive; the base image does.
**Generalises to:** know what your environment persists and what it doesn't.

### 10. `torch.OutOfMemoryError: Tried to allocate 286.00 MiB. GPU has 41.81 MiB free`
**Cause:** batch of 8 long prompts on a 14B model on a 16 GB T4.
**Why it hit at call 112, not call 1:** prompt length varies. Memory ≈
`batch_size × longest_prompt_in_batch`. A batch of long prompts blows up where
earlier batches didn't.
**Knobs, in order:** batch size → sequence length → precision/quantization → model size.

---

## Diagnosing, not just fixing

### Case study: the 24% loss

150 traces, 36 unusable. Obvious fix: raise `max_new_tokens` and regenerate.
That would have been **wrong** — it addresses only half the problem.

Diagnosis showed **two different causes** under one symptom:
- 19 traces: `no_steps`. Model wrote `### Step 1:`, our regex only matched `1.`.
  **Their lengths were normal (mean 871 vs 873 for healthy ones)** — nothing was
  truncated. A *format* problem.
- 17 traces: `no_final_marker`. Some genuinely truncated mid-equation; others
  **complete** but ending in `\boxed{70000}` instead of `####`.

Fixing the **parser** (broader step regex + `\boxed` fallback) recovered 24 traces
for free. Usability went 76% → 92%, no regeneration.

**The lesson:** one symptom, two causes. Diagnose before fixing, or you fix half
the problem and conclude the other half is unfixable.

**The tool:** the length comparison. If failing cases have the same length
distribution as healthy ones, it isn't truncation.

---

# PART 2 — MAKING IT FASTER

## Batching: why it works, and why it broke

A GPU is massively parallel. Running one prompt uses a fraction of it. Running 16
at once costs barely more wall-clock time than 1.

**The catch — padding.** Sequences in a batch must be the same length, so shorter
ones are padded. Cost is `batch_size × longest_in_batch`. A batch of
`[100, 100, 100, 5000]` tokens costs `4 × 5000 = 20,000` — 95% wasted.

### `padding_side="left"` — the silent killer

Decoder-only models continue from the **last token**. With right padding, pad
tokens sit between your prompt and where generation starts, so the model continues
from padding.

**It does not raise an error. It returns garbage.** This is the worst class of bug:
silent corruption. Always `tok.padding_side = "left"` for batched generation.

A bonus: with left padding, every row's real tokens end at the same index, so one
slice index (`output[:, prompt_len:]`) correctly strips the prompt from all rows.

### Token-budget batching (what we built)

Instead of a fixed count, sort jobs by length and fill each batch to a **padded
token budget**:

```
sort by length
greedily add to batch while (longest_so_far × count) <= budget
```

Result: short prompts get large batches, long prompts get small ones.
**Measured: 122 batches instead of ~586.** It also removes the OOM, because the
budget caps exactly the quantity that causes it.

### Quantization

4-bit (`BitsAndBytesConfig`) stores weights in 4 bits instead of 16. A 14B model
drops from ~28 GB to ~10 GB — the difference between fitting a T4 and not.
Costs a little accuracy and some speed. It is what made a 14B judge possible at all.

### Other levers we used
- `do_sample=False` for judging — deterministic, so score differences come from
  the data, not sampling noise.
- `torch.no_grad()` — no gradient graph during inference. Large memory saving.
- Caching model weights (e.g. `HF_HOME` on Drive) — a 33-minute download becomes zero.

---

# PART 3 — EXPERIMENT INVARIANTS

*These transfer to every project. They are the actual output of this month.*

**1. Cheapest decisive test first.**
Diffed two adapters on CPU before touching a GPU. Ran 3 traces before 30 before
138. Tested one prompt change on 36 calls, not 1,758. Order your tests by
`information gained / cost`.

**2. Never default on failure — count it.**
A parser returning `0.5` on failure puts fabricated data in your results,
indistinguishable from real scores. Return `None`, set a flag, report the rate.
*We caught this in the old judge code before it corrupted anything.*

**3. Save raw, aggregate later.**
Per-item scores, not summaries. Aggregation is free and reversible; regenerating
costs GPU. *This project already paid for this lesson once — per-problem outputs
weren't saved, so a test couldn't be run without retraining.*

**4. Pre-register before looking.**
Decide the primary metric and the failure condition before you see numbers.
Otherwise you compute five aggregations and pick the flattering one. That's
p-hacking, even when unintentional.

**5. Calibrate small before scaling.**
30 traces revealed the judge was giving 10/10 to 99% of steps. The full run would
have cost 50 minutes to produce 1,758 identical numbers.

**6. Read examples, not just aggregates.**
The distributions said "transition doesn't discriminate." Reading the REASON lines
said *why*: the judge emitted a score first and rationalised afterward. That
diagnosis was invisible in the numbers.

**7. Incremental writes + resume.**
Sessions die. Append per batch, `fsync`, skip completed work on restart. Our runs
died three times and lost nothing.

**8. Provenance on every artifact.**
Which model, which config, which commit. *Untrusted, scattered results are why
this project restarted from zero.*

**9. One symptom can have two causes.**
See the 24% loss above.

**10. Distrust plausible-sounding output.**
The 7B judge wrote confident, well-formatted checks about values that weren't in
the text (`150,000 - FABRICATED` for a step containing no 150,000). Fluent ≠ correct.

---

# PART 4 — CORE KNOWLEDGE (the ML/math side)

*My judgment of what's expected, tiered. Not authoritative.*

## Tier A — assumed, will not be explained to you

**Probability & statistics**
- Random variables, expectation, variance, standard deviation
- Sampling; why sample size drives uncertainty
- **Standard error** = sd/√n — why a 200-item eval has ~3.4pp noise
- Confidence intervals; what "the CI includes zero" means
- Correlation vs causation; Spearman vs Pearson
- Multiple comparisons — test enough things and something looks significant

**Linear algebra**
- Matrix multiplication, shapes, transpose
- **Rank** — and why a low-rank product `B @ A` can express a full-size matrix
  (this is LoRA)
- Norms (L1, L2); cosine similarity
- Eigenvalues/SVD conceptually — what "low-rank approximation" means

**Calculus**
- Gradients, partial derivatives, chain rule
- Why backprop is the chain rule applied over a computation graph

## Tier B — core ML/DL

- Train/val/test; overfitting; regularization
- Bias–variance
- Cross-entropy loss; why it's used for next-token prediction
- SGD, Adam; learning rate schedules; warmup
- **Transformer**: attention (Q/K/V), multi-head, residuals, layernorm,
  positional encoding. You should be able to draw the block.
- Causal masking — why a decoder can only attend leftward
- Tokenization; why token counts ≠ word counts

## Tier C — LLM-specific (your actual area)

- Autoregressive generation; the KV cache and why it makes generation incremental
- Sampling: greedy, temperature, top-p, top-k — and what each does to the distribution
- Context windows; why long prompts cost quadratically in attention
- Fine-tuning: full vs **LoRA/PEFT** vs prompting. What each costs.
- Quantization: 4-bit/8-bit, what precision buys
- Instruct models and chat templates

## Tier D — RL for LLMs (essential for TAPR)

- MDP framing: state, action, policy, reward, trajectory
- Policy gradient / REINFORCE — the core identity
- Why variance reduction matters; baselines
- **Advantage** = reward − baseline. Why we subtract.
- PPO: clipping, trust regions
- **GRPO**: normalises advantage *within a group* of samples per prompt.
  You must know why `advantage = 0` when all samples in a group share a reward —
  that's your degeneracy argument.
- KL penalty against a reference model; what it constrains
- RLHF vs RLVR: learned reward model vs verifiable checker
- ORM vs PRM: outcome vs process supervision

## Tier E — evaluation & rigor

- Seeds and variance across runs; why single-run results are weak
- Statistical tests for paired comparisons (e.g. McNemar for paired binary outcomes)
- Effect size vs significance
- Ablations: change one thing at a time
- Pre-registration

## What you can skip

Distributed training internals, CUDA kernels, most classical ML (SVMs, trees),
optimization theory, information theory beyond entropy/KL basics.

---

## Honest gap assessment

**Strong:** Tier C (used directly), Tier E (designed this experiment around it),
Part 3 invariants (built them into working code).

**Adequate:** Tier D conceptually — you can argue about GRPO degeneracy and
within-group variance, which is the part that matters for TAPR.

**Thinner:** Tier A statistics (you've been reasoning about separation and
variance without formal grounding), Tier B transformer internals (used, not
derived), and PyTorch training loops (inference only so far).

**Highest-leverage fix:** Tier A statistics. Everything you're claiming —
separation, variance, whether a gap is real — rests on it, and it's the smallest
body of material on this list.
