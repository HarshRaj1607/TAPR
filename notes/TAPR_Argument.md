# TAPR — The Argument (one page)

## Finding 1 — Both existing reward types are broken, in different ways

**Outcome-only:** rewards the final answer. A completely wrong chain that lands on the right number gets full reward. No information about the reasoning at all.

**Isolated-step:** scores each step on its own. This is what almost everyone does. A step is checked against a gold reference, or by a trained PRM, or by asking "is this step correct." Either way, **the check never looks at whether the step connects to the one before it.**

## Finding 2 — Isolated-step scoring has a specific, documented failure

Because each step is judged alone, a chain where every step is individually fine but the chain as a whole goes nowhere scores high. Three papers documented this independently:

- **Reward Granularity's own table:** process-only training got the best accuracy and the worst trace validity — worse than the untrained model.
- **Samineni:** local coherence improves without correctness following.
- **PROGRS:** PRMs "assign high scores to locally coherent reasoning that ends in an incorrect answer."

Everyone sees the problem. Nobody fixes the signal.

## Finding 3 — The fix: score the transition, not the step

Instead of "is this step correct," ask **"does this move follow from everything before it."**

Three conditions: values traceable, operation legitimate, direction advancing the problem.

## Finding 4 — The differentiator, confirmed against five papers

Every neighbour's version of "coherence" turned out to be something else — matching a gold answer key, counting errors, measuring variance in a scorer's confidence, or checking the answer against the whole chain. **None asks whether one step follows from the previous one, judged from the reasoning's own content.**

And every one of them needs a per-step answer key. Ours doesn't. That's the methodological edge — it works where no gold steps exist.

## Finding 5 — The confound (a separate argument)

When a process reward fails, it's either because **the signal is bad** or because **mixing it with outcome reward conflicts**. Every paper varies the mixing weight while never independently validating the signal, then blames the mixing. Nobody separated the two causes.

Our response: validate the signal first, alone, before any mixing question. That's what Checkpoint 1 is.

## What we're actually trying to show

**That a transition signal detects broken reasoning that isolated-step scoring rates as fine.**

Not accuracy. Not transfer. Just: two scorers, the same broken traces, does ours catch what theirs misses.

## The problem just found

To show that, we need traces that are **locally valid but globally broken**, and we need to know which ones those are — independently of either scorer. Number-matching against gold is a bad proxy for "locally valid." That gap is what the redesign has to close.
