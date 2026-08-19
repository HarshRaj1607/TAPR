# TAPR — Transition-Aware Process Reward

Scoring reasoning **transitions** instead of isolated steps, and separating a
confound in how process rewards are evaluated.

**Write-up:** https://HarshRaj1607.github.io/TAPR/
**Technical appendix:** https://HarshRaj1607.github.io/TAPR/appendix.html

---

## What this is

Outcome-only rewards say nothing about reasoning. Isolated-step process rewards
score each step alone, so a chain where every step is individually fine can score
high while going nowhere. TAPR scores the transition instead: *given everything
seen so far, does this move follow?*

Separately: when a process reward underperforms, the literature varies how it is
mixed with the outcome reward and blames the mixing, without ever validating the
signal on its own. Checkpoint 1 tries to separate those two causes.

## Status — read this before the results

**Checkpoint 1 ran. The hypothesis failed, and the test had a design flaw I found
afterwards.**

The claim is that a transition judge catches reasoning that is *locally valid but
globally broken* — the kind an isolated judge rates as fine. The broken set I
actually tested on was "wrong final answer, minus format failures." Those are not
the same population. Most wrong answers break at a single visible step, which the
isolated judge catches too.

So the null result does not say the signal is weak. It says the test was run on
the wrong traces. Fixing this needs step-level hand labelling, which is the next
piece of work.

Everything below is reported as-is, including what went wrong.

## Results at a glance

| | isolated | transition | bare |
|---|---|---|---|
| Separation | 71.8% | 76.1% | 75.3% |
| 95% CI | [58.6, 84.2] | [64.2, 86.9] | [65.3, 85.0] |
| Zero-variance groups | 5/34 | 9/33 | 15/34 |
| Modal score fraction | 34.3% | 86.3% | 79.3% |

All three beat chance. **None can be ranked** — n = 27 broken traces. Within-outcome
variance, the metric that matters for GRPO, went the *wrong* way: transition
produced less spread than isolated, not more.

Other findings that stand on their own:

- **36% of wrong-answer traces are format failures**, not reasoning failures.
  Labelling reasoning quality by final-answer correctness mislabels a third of
  failures on this setup.
- **LLM-as-judge is unreliable at 7B** for transition validity. Qwen2.5-7B scored
  10 on 99% of steps and fabricated its own justifications. 14B works.
- **`bare` ties `transition`.** Explicit criteria added nothing over asking "does
  this follow?"

## Repository layout

```
docs/          the write-up (GitHub Pages)
notes/         full record of Checkpoint 1, start to finish
scripts/       generation, scoring, analysis
src/tapr/      shared parsing library
data/          raw traces and every judge call, with raw responses kept
hackathon/     earlier version — trained the reward instead of validating it
```

## Reproducing

```bash
pip install torch transformers datasets bitsandbytes accelerate

python scripts/generate_traces.py --n_problems 30 --n_samples 5
python scripts/inspect_traces.py    data/traces.jsonl
python scripts/score_traces_v2.py   --traces data/traces.jsonl \
                                    --out data/scores14b.jsonl \
                                    --judge_model Qwen/Qwen2.5-14B-Instruct
python scripts/analyze_v2.py        data/scores14b.jsonl
python scripts/check_spread.py      data/scores14b.jsonl data/cal_free.jsonl
```

Every script resumes — rerunning skips completed work.

Do not regenerate `data/traces.jsonl`. Temperature is 0.9, so a rerun produces
different traces, and every number here is tied to this specific set.

## Data

| File | Judge | Traces | Scorers | Calls |
|---|---|---|---|---|
| `traces.jsonl` | — | 150 generated | — | — |
| `cal.jsonl` | 7B | 30 | isolated, transition, bare | 357 |
| `cal14b.jsonl` | 14B | 30 | isolated, transition, bare | 357 |
| `scores14b.jsonl` | 14B | 138 | isolated, transition, bare | 1,758 |
| `cal_free.jsonl` | 14B | 30 | transition_free, bare_free | 238 |

Every row keeps the judge's raw response, so any number can be traced back to the
text that produced it. Nothing defaults on a parse failure — failures are recorded
and counted, never silently filled with a midpoint.

## Known issues

- `analyze_v2.py`: the distribution section sits outside `main()` and never runs.
  It also assumes all five scorers are present. Use `check_spread.py` meanwhile.
- The banded-vs-unbanded comparison changes five things at once, not one — see
  the appendix. The bands hypothesis is untested, not confirmed.
- Banded scorers ran on 138 traces, unbanded on 30. Not the same problems.

## Next

1. Hand-label the broken traces at step level, blind, to split *locally-valid-
   globally-broken* from *plainly broken*. Run the comparison on the first group.
2. Sample and label sound traces to measure contamination in the control group.
3. Measure judge accuracy against those labels instead of assuming it.
4. Deconfound the bands test — change only the bands.

## Author

Harsh Raj — B.E. Computer Science, Ramaiah Institute of Technology, Bangalore.
Independent work, no lab affiliation.

The main write-up is written entirely by me. The technical appendix is
LLM-assisted and labelled as such.
