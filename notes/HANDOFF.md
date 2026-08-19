# TAPR — HANDOFF (read this first in a new chat)

*Everything needed to continue. Written Aug 2026, after Checkpoint 1 full run.*

---

## 0. Read these files, in this order

1. **`WORKING_AGREEMENT.md`** — the operating contract. Present-first always, with
   strictness graded (STRICT / MEDIUM / LENIENT). Read before anything else.
2. **`TAPR_Argument.md`** — the one-page claim.
3. **This file** — current state, results, open questions.
4. `TAPR_v2_State.md` — full design record (long).
5. `Field_Manual.md`, `Intern_Machinery_Minimum.md` — skills reference.

---

## 1. The claim, in one paragraph

Outcome-only rewards say nothing about reasoning. Isolated-step process rewards
score each step alone, so a chain where every step is individually fine can score
high while going nowhere — documented independently by Reward Granularity
(process-only got best accuracy, worst trace validity), Samineni, and PROGRS.
**TAPR scores transitions instead**: given everything seen so far, does this move
follow? The differentiator, confirmed against 5 papers: every neighbour's
"coherence" needs a per-step answer key or policy document. **Ours needs neither.**

Separately: when a process reward underperforms, nobody has separated "the signal
is weak" from "mixing it with outcome reward conflicts." Everyone varies the
mixing weight and blames mixing. **Checkpoint 1 validates the signal alone, first.**

---

## 2. What Checkpoint 1 actually is

Not training. Pure inference. Score the same traces with different scorers and
compare — no GRPO, no policy, so no attribution problem.

**Scorers** (same judge model, only the prompt differs):
- `isolated` — sees question + ONE step. Context-free.
- `transition` — sees question + all prior steps + this step. Banded rubric, 0–10.
- `bare` — same context, no criteria, just "does this follow?" Banded, 0–10.
- `transition_free` — metrics stated, **no bands**, 0–1. *(new, being tested)*
- `bare_free` — no metrics, no bands, 0–1. *(new, being tested)*

**Prefix-only** is enforced by slicing (`steps[:i]`), not by instructing the model.

**Ground truth** = GSM8K's final answer. Mechanical, independent of any scorer.
Raw disagreement between scorers is uninterpretable — you need an external arbiter.

---

## 3. Results so far

### Data
- 150 traces generated (Qwen2.5-3B, 30 problems × 5 samples, temp 0.9)
- **138 usable (92%)** after parser fixes — was 76% before
- 42 wrong-answer, **but 15 of those (36%) are FORMAT failures** — reasoning
  reached the gold value, the `####` marker was mistyped (computed 70,000,
  wrote `#### 7.0`). **27 genuinely broken traces.**

### Judge scale matters — 7B failed, 14B worked
- **Qwen2.5-7B:** scored 10 on 99% of steps. On a trace where the model invented
  "7 days in a week" (problem says *3 times a week*), it wrote
  *"unsourced, but commonly known"* → 7/10. It hallucinated a `FABRICATED` value
  that wasn't in the step. **Unusable.**
- **Qwen2.5-14B:** same trace → **0/10**, correctly citing the contradiction.
- This is a reportable finding: LLM-as-judge for transition validity is
  unreliable at 7B.

### Full run (138 traces, 14B, format failures excluded, n=96 sound / 27 broken)

| scorer | separation (min) | 95% CI |
|---|---|---|
| isolated | 71.8% | [58.6, 84.2] |
| transition | 76.1% | [64.2, 86.9] |
| bare | 75.3% | [65.3, 85.0] |

**Honest reading: the CIs overlap almost completely. At n=27 broken, you cannot
rank the scorers.** All three are clearly above chance (50%), so the judges *do*
detect broken reasoning. You just can't say one is better.

### Within-outcome variance (the GRPO engine) — the weak result
Same problem, same final answer → do scores still differ? Zero variance ⇒ zero
advantage ⇒ that problem teaches nothing.
- isolated: 5/34 zero-variance groups (best)
- transition: 9/33
- bare: 15/34 (worst)

**The signal does not yet provide the within-group gradient TAPR requires.**

### Distribution problem → current open test
Banded `transition` put **448 of 519 scores on exactly 9, and zero on 10.**
That's not a scale in use; it's a default. Pattern across prompts: *the more
rubric, the less scale usage* (bare, with no rubric, spread widest).

**Harsh's hypothesis:** the discrete bands created attractors. His hackathon
prompt gave metrics only, asked for 0–1, and spread fine.
**Test in flight:** `transition_free` and `bare_free` — metrics but no bands, 0–1.
Calibration on 30 traces done (`cal_free.jsonl`, 238 calls, 0 parse failures).
**Next command:**
```
python check_spread.py scores14b.jsonl cal_free.jsonl
```
If modal fraction drops well below 86% and distinct values rise → hypothesis holds,
run the full 138. Watch whether `bare_free` also changes — bare never had bands,
so if it shifts too, the cause is the 0–1 scale, not the bands.

---

## 4. A reframe Harsh made that changes the interpretation

Originally: transition (with explicit criteria) should beat bare.
Results: bare ties transition.

Harsh's position, after reading the interwhen paper: **if judgment is baked into
criteria I wrote, I've relocated human judgment rather than removed it — the same
critique I made of interwhen's policy documents.** So bare tying transition is
*the result*, not a failure.

**Current claim: context-aware scoring beats context-free. Explicit criteria add
nothing over asking "does this follow?"** Simpler and more defensible.

---

## 5. Files

**Scripts** (Colab; `tapr_parsing.py` must sit alongside all of them)
- `generate_traces.py` — batched generation, resume, provenance
- `tapr_parsing.py` — segmentation + answer extraction + format-failure flag
- `inspect_traces.py` — usability stats
- `diagnose_losses.py` — why traces fail to parse
- `score_traces_v2.py` — the judge harness (5 scorers, token-budget batching, resume)
- `read_reasons.py` — read what the judge actually said on a given trace
- `analyze_scores.py` — v1 analysis
- `analyze_v2.py` — excludes format failures, bootstrap CIs. **Bug: the
  distribution section was appended outside `main()` and never runs. Also assumes
  all 5 scorers are present.** Use `check_spread.py` for distributions meanwhile.
- `check_spread.py` — distribution / collapse check

**Data (Drive `/content/drive/MyDrive/tapr/`)**
- `traces.jsonl` — the 150 traces (do not regenerate; temp 0.9 gives different ones)
- `cal.jsonl` (7B calibration), `cal14b.jsonl` (14B calibration)
- `scores14b.jsonl` — full run, 3 banded scorers
- `cal_free.jsonl` — 30-trace calibration of the 2 unbanded scorers

---

## 6. Environment gotchas

- **Mount Drive and verify as the first cell, every session.** Anything written
  locally dies with the runtime.
- `pip install bitsandbytes accelerate` after every reset.
- 14B re-downloads (~30 GB, 15–35 min) unless `HF_HOME` points at Drive. Set
  `os.environ["HF_HOME"] = "/content/drive/MyDrive/hf_cache"` before loading.
- **CUDA OOM on T4 with 14B:** lower `--max_tokens` (5000 works; 9000 fails on the
  long-prompt batches at the end, since sorting puts them last).
- `datasets` v4 needs `openai/gsm8k`, not `gsm8k`.
- Everything resumes — rerunning the same command skips completed work.

---

## 7. Open questions

1. **Did removing bands fix the collapse?** ← the live one
2. If yes: does within-outcome variance improve? That's the engine.
3. If variance stays near zero, TAPR has no gradient signal and the method needs
   rethinking regardless of separation.
4. n=27 broken traces is too few to rank scorers. Generate more? The wrong-answer
   rate is ~30%, so ~450 traces would give ~85 genuinely broken.
5. The 36% format-failure rate is a finding in its own right — any paper labelling
   by final-answer correctness on this setup mislabels a third of its failures.

---

## 8. Standing guardrails

1. **Predicted ≠ shown.** Arguments are not results.
2. **Pre-register before looking.** Metric and kill condition decided in advance.
3. **Cheapest decisive test first.** 3 traces → 30 → 138.
4. **Never default on failure — count it.** A parser returning 0.5 on error puts
   fabricated data in results.
5. **Save raw, aggregate later.** Aggregation is free; regenerating costs GPU.
6. **Read examples, not just aggregates.** The distributions said "no
   discrimination"; reading the judge's own text said *why* — it scored first and
   rationalised after.
7. **Distrust plausible-sounding output** — from the judge, from Claude, from
   abstracts. Fluent ≠ correct.
