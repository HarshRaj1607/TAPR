# Hand-labeling rules (step level)

Labels live in `data/handlabels.jsonl`. 40 traces: all 27 genuinely broken traces
(wrong answer, format failures removed) plus 13 randomly sampled sound traces
(seed 7). Order was shuffled and the labeler saw neither judge scores nor the gold
answer.

## How the labels were made

- L01 to L10: labeled by Harsh without seeing Claude's labels, compared against
  Claude's independent labels, and disagreements resolved into the rules below.
  Claude labeled L04 to L06 after seeing Harsh's, so those do not count towards
  the pre-rule agreement figure.
- L11 to L40: drafted by Claude under the rules, then reviewed by Harsh. Report as
  **LLM-drafted, human-reviewed**.
- Agreement before the rules existed: on first-break step, 3 of 7 independently
  labeled traces matched clearly (L01, L03, L07). Disagreements were about
  convention, not about the problems.

## Rules

1. **First break** = earliest step containing a false claim, an invalid inference, or a
   contradiction of the problem, even if a later step is worse.
2. **Ambiguous or convention-dependent gold is not a break.** Flag `ambiguity`.
   (The lemon-tree problem: gold is 13, reasoning to 12 is defensible.)
3. **Plan-only or truncated traces with no computed numbers are not a break at the
   step level.** Flag `no_computation` or `truncated_no_computation`.
4. **Goal versus delivery.** If one step states a goal and a later step delivers
   something different, the break is at the delivering step.

## Break types

- `F` fabricated value, or contradicts the problem
- `O` invalid operation or wrong arithmetic
- `D` valid but pointless, looping, or wrong direction
- `E` reasoning fine, final interpretation off

## Known limitations

- 40 traces over 20 distinct problems, so labels are clustered. Use problem-level
  resampling for any interval.
- 19 break steps in total: 14 `O`, 4 `F`, 1 `E`, 0 `D`.
- Labels are LLM-drafted for 30 of 40 traces.
