# Data

| File | Judge | Traces | Scorers | Calls | Status |
|---|---|---|---|---|---|
| `traces.jsonl` | — | 150 generated | — | — | committed |
| `cal.jsonl` | Qwen2.5-7B-Instruct | 30 | isolated, transition, bare | 357 | **missing** |
| `cal14b.jsonl` | Qwen2.5-14B-Instruct | 30 | isolated, transition, bare | 357 | **missing** |
| `scores14b.jsonl` | Qwen2.5-14B-Instruct | 138 | isolated, transition, bare | 1,758 | committed |
| `cal_free.jsonl` | Qwen2.5-14B-Instruct | 30 | transition_free, bare_free | 238 | committed |

**`cal.jsonl` and `cal14b.jsonl` are not in this folder.** They're referenced
in the appendix write-up (the 7B-vs-14B judge comparison) but weren't found on
disk during this pass. If they exist elsewhere — another machine, a Colab
runtime, a different local folder — add them here before relying on the
appendix's judge-comparison claims being independently re-checkable. Everything
below describes all five files as originally recorded; only three are
currently present.

## Row schema — traces.jsonl

`problem_id`, `sample_id`, `question`, `gold_answer`, `model_trace`, `gen_config`

## Row schema — score files

`problem_id`, `sample_id`, `step_idx`, `n_steps`, `answer_correct`, `scorer`,
`score`, `reason`, `parse_failed`, `raw`

`score` is a float, the string `"NA"` (transition only — judge decided the step is
not a transition), or `null` when parsing failed. **Never a default value.**
`raw` keeps the judge's response so any number can be traced to its source.

`_free` scorers emit 0–1 and are rescaled ×10 on write for a common axis. They
used one decimal place only, so effective resolution matches the 0–10 scorers.

## Do not regenerate

`traces.jsonl` was generated at temperature 0.9. A rerun produces different
traces and every published number is tied to this specific set.
