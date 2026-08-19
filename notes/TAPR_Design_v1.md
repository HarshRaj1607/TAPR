# TAPR — Research Design v1

*Transition-Aware Process Reward. Consolidated design record.*
*Status: Samineni + bundle-sweep complete. PROGRS read closely. §9 and §11 updated; core narrowed. To be consolidated into v2.*

---

## 0. How to read this doc

This is a **living design record**, not a paper draft. Its job is to state, honestly, what we are claiming, what we are only *predicting*, what each experiment must prove, and what would count as failure. Everything provenance-tainted from the hackathon is treated as untrusted and is being rebuilt clean.

A recurring convention below: **CLAIM** = something we assert and will defend; **THESIS-TO-EARN** = something we believe but have *not yet demonstrated* and must not state as fact until an experiment earns it; **GUARDRAIL** = a standing honesty check to stop us fooling ourselves.

---

## 1. The central claim

Adding a live judge that scores reasoning **transitions** — whether each step *follows from* the previous one — on top of outcome-only RL does not merely fix a training inefficiency. It changes the **direction** the policy learns in, producing reasoning that survives a **change of task format** better than either outcome-only rewards *or* step-level process rewards.

Three moving parts, each of which needs its own experiment:

1. **Mechanism** — outcome-only RL wastes most of its problems (zero gradient whenever all rollouts share the same outcome). The transition signal restores gradient on those wasted problems by breaking ties on *reasoning quality*.
2. **Nature of the change** — the resulting policy is not simply "moved less" (regularization); it is moved in a *different direction* (qualitatively different learning).
3. **Payoff** — that different direction is what transfers across formats, because "does this follow from that" is a **domain-general** operation, whereas step-correctness is a **domain-specific** surface pattern.

---

## 2. The intellectual core — stated at its true epistemic status

**THESIS-TO-EARN (the engine of the paper):**
Step-level rewards teach a model to pattern-match individual steps against what "correct steps" look like *in the training domain*. Those patterns are domain-specific (GSM8K arithmetic steps look nothing like commonsense-inference steps), so the skill does not carry to a new format. Transition-scoring instead rewards a domain-general operation ("does this follow"), so what it learns generalizes.

This is the most important idea in the project **and the one most likely to seduce us into asserting it before we've shown it.** It is a claim about *what was learned*, which requires its own evidence (Experiments 3 and 4). Until then it is a **prediction we are willing to test**, never a fact we build on.

**GUARDRAIL:** We reached "step-rewards won't transfer" by reasoning, not measurement — and partly because it is the conclusion the paper *needs*. That is exactly when to be most careful. We do not *state* that step-rewards fail at transfer; we *predict* it and then *show* it by running a step-level reward through our own transfer eval (Exp 4).

---

## 3. What is (and is not) novel

**Not novel — do not claim these:**
- "Process reward helps small-model reasoning." Settled since Uesato et al. (2022); re-confirmed by Reward Granularity (2607.02869).
- "Credit assignment matters in GRPO." Known (Thinker, 2505.21097: the whole sequence receives one scalar advantage).
- "Dense reward beats sparse reward in-domain." Reward Granularity already shows this (~10pt gap, process-only vs outcome-only).

**Novel — the defensible bundle:**
- We reward **transitions** (relational, between adjacent steps) — not **steps** (a property of each step, checked in isolation).
- We measure **cross-format transfer** — train on one format, test on another — which no paper in our reading list does.
- We explain the transfer **mechanistically** (degeneracy → direction → domain-general operation), not just report it.
- We do it with **multi-seed rigor**, which the nearest neighbor explicitly lacks.

**The one-line boundary:** *They reward steps and measure in-domain; we reward transitions and measure transfer.* Two independent axes of difference. Either alone is arguable; both together is clean space.

---

## 4. Abandoned from the hackathon version

- **The accuracy story is dropped.** At this model scale the in-domain accuracy gap is noise-level and someone else already owns the accuracy hill. Accuracy is now *measured to establish in-domain parity* (half of the Story-B claim), **not claimed** as a contribution.
- **The λ=0.3 hybrid is under review.** Our hackathon reward was outcome + 0.3·transition — a minority process signal on a dominant outcome signal. Reward Granularity's λ=0.1 anomaly (small process weight on dominant outcome *hurt*) is structurally the same shape. This motivates testing **transition-only**, not just the hybrid.
- **All old code and data are untrusted.** Scattered saves, hackathon rush, unknown final code version. We rebuild clean (see §8).

---

## 5. Experiments — hypotheses, success criteria (pre-registered), reviewer question killed

Ordered by dependency. Multi-seed is **not** its own experiment — it is *how* Exp 2–4 are run.

### Exp 0 — Judge validity (prerequisite)
- **Hypothesis:** the judge scores good transitions above bad ones, *and* its score carries information the outcome reward does not.
- **Success:** clear discrimination on a **100+ problem** validation set (not the old 20); **and** meaningful score variance *within* the correct-answer set *and* within the wrong-answer set (within-outcome variance is the entire engine of TAPR).
- **Kills:** "Your process reward just re-encodes final-answer correctness."

### Exp 1 — Degeneracy mechanism
- **Hypothesis:** outcome-only produces zero-reward-variance (zero-gradient) groups on a large fraction of problems; TAPR collapses that fraction.
- **Success:** large, stable gap in `frac_reward_zero_std` across training and seeds.
- **Kills:** "Why should this help at all?"
- **HONESTY:** nearly guaranteed by construction. This is a **demonstration, not a risky test**, and will be framed that way in the paper — not dressed up as a discovery.

### Exp 2 — THE GATE: cross-format transfer
- **Hypothesis:** trained on math (GSM8K), TAPR preserves performance on a *different-format* reasoning task better than outcome-only, while staying statistically tied in-domain.
- **Success (pre-registered, decided before running):** across seeds, TAPR transfer accuracy consistently above ORM; the gap exceeds the seed-to-seed spread; direction holds in **≥2 of 3 seeds**; **and** in-domain (GSM8K) accuracy is statistically indistinguishable (proving we are not trading in-domain for transfer).
- **Kills:** "Does any of this change something that matters?"
- **Note:** everything downstream is conditional on this. If it fails, there is no paper — stop and rethink.

### Exp 3 — Direction, not magnitude (Story A vs B) — *now load-bearing*
- **Hypothesis:** TAPR and ORM move the policy by *similar magnitude* but in *different directions*.
- **Success:** magnitude metrics (KL, weight-norm, response length / step count) comparable between conditions; directional metrics (per-layer weight-change cosine — the analysis previewed on the two adapters — plus behavioral divergence) show them diverging.
- **Two-sided by construction:** it *can* return "TAPR just moved less" (Story A). If it does, **we report Story A.**
- **Kills:** "Isn't this just implicit regularization — TAPR changes the model *less*, so of course it transfers?" This objection now has a *named theoretical backer* (Binary Rewards, 2605.02375: KL-control selects the valid distribution closest to base). Beating it is mandatory, not optional.
- **Stretch ambition:** show the transition-model's weight changes are more *similar across problem types* while the step-model's are domain-specific — the direct fingerprint of the domain-general thesis (§2).

### Exp 4 — Transitions vs *any* dense reward (the sharpened ablation)
- **Hypothesis:** the benefit comes specifically from scoring **transitions**, not from having a dense reward. A step-level PRM (dense, but scores steps in isolation — the Reward Granularity construction) will improve *in-domain* like TAPR but **fail to transfer**.
- **Success:** TAPR clearly beats step-level PRM on transfer. If step-PRM matches TAPR, the honest claim shrinks to "dense process reward helps" and the transition claim is not supported.
- **Kills:** "Is transition-awareness doing the work, or would any dense reward do this?"
- **Sharpened role:** this is also the experiment that *earns* the §2 thesis — running a step-level reward through our transfer eval directly tests domain-general vs domain-specific. Most expensive (a third training condition); **stage it after the gate (Exp 2) confirms there is an effect worth ablating.**

### Exp 5 — Shape of transfer (optional polish)
- **Hypothesis:** the advantage grows with distribution distance, then vanishes when the target is too far (both models fail).
- **Success:** a graded curve — small in-domain, larger cross-format, null far-OOD.
- **Kills:** "Is this effect suspiciously universal?" (Bounded effects read as real mechanisms, not artifacts.)
- **Status:** nice-to-have, **not** load-bearing for a first workshop submission.

---

## 6. Open design decisions — Harsh's to own

Not yet decided. Each is a real choice, not hackathon inheritance.

- **Student model.** 3B is defensible on merits (its ceiling is what makes in-domain similar while transfer diverges). Working lean: *iterate* on 1.5B (faster, cheaper, starker degeneracy), *report final numbers* on 3B for credibility. **Not committed.**
- **Judge model.** 7B same-family, already validated, standard. Be ready for the "same-family judge shares the student's blind spots" question; Exp 0's orthogonality check partly answers it.
- **Training domain.** GSM8K — clean step structure (transitions well-defined), validated judge. Keep unless a reason emerges.
- **Transfer target — THE open question.** StrategyQA is the striking headline (entirely different format) but has genuinely weak eval (≈200 problems, no real test labels). Options: (a) accept and caveat it, or (b) find a cleaner-eval cross-format task. This is the decision most worth solving before Exp 2.
- **Baseline / condition set.** Reward Granularity raises transition-*only* vs hybrid. A small grid (outcome-only, transition-only, outcome+transition) is more informative but more runs. Undecided — weigh against whether it distracts from the transfer center.
- **Exp 4 scope.** Assessed as **core, not stretch** (without it "transitions specifically" is unproven) — but staged after the gate.
- **SVAMP / Exp 5.** Deferred to optional polish for v1.

---

## 7. Standing honesty guardrails

1. Story B is the hypothesis we *chase*, but Exp 3 and Exp 4 must be able to return Story A / null / "just dense reward," and we report whatever they say.
2. The §2 domain-general thesis is **prediction-to-be-demonstrated**, earned only by Exp 4 — never stated as fact.
3. The adapter weight-diff (which leaned Story B: similar magnitude, ~143% directional difference) is **n=1 with unknown provenance** — a hint, not a finding.
4. Do not underestimate the nearest neighbor. Reward Granularity is not "weak"; it is scoped differently and honest about its limits. We beat it on *rigor + transfer + transitions*, not by dismissing it.
5. Small evals have misled this project before (200-problem eval showed a gap that collapsed at full scale). Do not shrink eval sets to go faster.

---

## 8. Rebuild plan (provenance reset)

- **Phase 0 — trusted foundation (mostly non-GPU):** one version-pinned codebase, single train + single eval script. Fix the seed trap (global RNG reseed must happen *after* arg parsing, or window sampling stays frozen). Add drift logging. Re-validate the judge (Exp 0). One smoke run to confirm the judge produces *varied* scores (not a wall of 0.5 parse-failure defaults) and that `frac_reward_zero_std` actually diverges between modes.
- **Phase 1 — gate + mechanism, instrumented once:** ORM + TAPR across seeds, full steps, drift-logged, checkpoints saved. Harvest Exp 1 (degeneracy), Exp 2 (transfer gate), Exp 3 (drift). *Instrument once, harvest three times.* Decision point: does the gap survive, and does drift say A or B?
- **Phase 2 — specificity (only if Phase 1 holds):** add the step-level PRM condition → Exp 4.
- **Phase 3 — optional:** SVAMP for the transfer-shape curve (Exp 5), MATH-500 as the boundary null.

Efficiency note: Exp 1, 2, 3 all harvest from the *same* training runs (instrument once). Exp 4 needs a new training condition, which is why it is staged after the gate.

---

## 9. Literature standing (pre-Samineni)

- **Uesato et al. (2022)** — foundational outcome-vs-process comparison. The great-grandparent; "process helps" lives here, so our novelty cannot.
- **Reward Granularity in RLVR (2607.02869)** — closest neighbor. Step-level reward (Eq. 1 = correct steps / total steps, permutation-invariant, checked against answer key in isolation). In-domain only. Single-seed, no CIs, "indicative not confirmed" (their words). Their Table 3 is our motivation *in their data*: process-only gets the highest accuracy (0.64) but the **worst** trace validity (0.22, below base's 0.56) — step-scoring trades global coherence for local correctness. Role for us: motivation + in-domain floor + boundary.
- **Binary Rewards and RL (2605.02375)** — degeneracy theory + KL-control. Gift (rigorous vocabulary for Exp 1) and threat (its KL-control story *is* the Story-A objection Exp 3 must beat).
- **Thinker (2505.21097)** — GRPO gives the whole sequence one scalar advantage. Cite as Exp 1 motivation.
- **Samineni et al. (2025)** — "local coherence vs globally correct." Closest phrase to "transitions." **NOT YET READ.** The last paper that could genuinely crowd the center. Reading next → drives v2.

---

## 10. What v2 must resolve

- Where exactly Samineni stopped, and whether "local coherence" overlaps "transitions" as reward or only as measurement.
- The transfer-target decision (§6).
- Whether the condition set is ORM vs TAPR, or the fuller grid.
- First concrete build target for Phase 0.

---

## 11. Cracks to measure against (the PROGRS critique)

**Status of this whole section: PREDICTED, not demonstrated.** These are claims about PROGRS's method (2604.02341) derived from reading its equations, plus one conceded overlap. The job is to convert each from "we argued this" to "we showed this." Nothing here is a finding yet.

### 11.0 What we must concede to PROGRS (honest overlap)
- **Within-outcome-group preference learning is theirs, not ours.** Their Eq. 3: "when all K samples share the same outcome, A_outcome = 0 and learning is driven by within-group preferences." That *is* our Exp-1 mechanism, published. We cannot claim the "outcome-only wastes agreeing-rollout problems; process signal rescues them" mechanism as novel.
- **Multi-scale window aggregation is theirs too.** Their §III-B.2 partitions trajectories into windows and offers a multi-scale form over window sizes W. Our stochastic window sampling over {2,3,5} is the same shape.
- **Consequence:** TAPR is *not* differentiated by mechanism or by windowing. Differentiation must come from the *signal content* and the *transfer + mechanistic* angles below.

### 11.1 CRACK 1 — Their "coherence" is score-stability, not logical entailment
- **The claim:** PROGRS's windowed coherence (their Eq. 5, `rcoh = µ·exp(−λvar·σ/µ)`) penalizes *variance in the frozen PRM's step scores*. It measures whether the *scorer's confidence numbers* are steady across neighboring steps — NOT whether step *i* logically follows from step *i−1*. Our transition signal measures entailment directly.
- **Why we believe it:** their score `rfine(st)` is a per-prefix confidence verdict (their Eq. 4); the window term operates only on those numbers' mean and std. Nothing compares the *content* of adjacent steps.
- **Measurement that would confirm/kill it:** construct or find traces where score-stability and entailment DISAGREE, and check which predicts actual correctness. Two diagnostic cases:
  - *Valid-but-jumpy:* a sound chain with a legitimately hard step (PRM confidence dips then recovers) → high σ → PROGRS penalizes a correct chain.
  - *Fluent-but-broken:* a smoothly-wrong chain (each step locally plausible, scores steady ~0.7) → low σ → PROGRS rewards a wrong chain. (This is the exact "locally fluent, globally wrong" failure PROGRS's own intro claims to fix.)
- **If confirmed:** this becomes a sharpened Exp 4 — *transition-entailment vs published score-variance coherence*, head-to-head on the conflict cases. If killed (their signal tracks entailment in practice): our signal distinction collapses and we lean entirely on transfer + mechanism.

### 11.2 CRACK 2 — The unstated bet: "stable scores = good reasoning"
- **The claim:** PROGRS's coherence rests on an unjustified assumption that smoothness of scores tracks soundness of reasoning. They never defend it; it is baked into Eq. 5.
- **Measurement:** correlation between per-window score-variance and human/verifier-judged step validity. If low or non-monotonic, the assumption is unsupported — and rewarding smoothness is optimizing the wrong quantity.
- **Note:** CRACK 1 and 2 are the same fault seen from two sides (1 = the two failure directions; 2 = the assumption underneath). Measure together.

### 11.3 What is cleanly OURS (not in PROGRS)
- **Cross-format transfer.** PROGRS evaluates only in-domain math (MATH-500, AMC, AIME, MinervaMath). No math→different-format (e.g. commonsense) transfer. Our transfer axis (Exp 2) is untouched.
- **Mechanistic explanation.** PROGRS reports Pass@1 gains; no Story-A-vs-B, no policy-direction geometry. Our Exp 3 (direction not magnitude) is untouched.

### 11.4 The strategic read
- Two things narrowed from open-space to *actively-defended-against-a-published-equation*: the signal distinction (CRACK 1) now must be argued against PROGRS Eq. 5, not asserted into a vacuum.
- Two things remain cleanly ours: transfer-across-format, mechanistic explanation.
- **The bottom-up idea gains weight:** "what kinds of problems does a transition-*entailment* signal repair, and does that principle transfer across formats" is a question PROGRS's score-variance method *cannot* answer and does not ask. This is now a candidate spine, not a backup.

---

## 12. Literature standing — bundle-sweep update (supersedes parts of §9)

The bundle-shaped sweep (searching combinations, not pieces) revealed an active 2025–2026 cluster doing richer-reward + GRPO + (often) transfer. "Richer reward transfers better than outcome-only" is now **table stakes, not a contribution.** Much of this postdates the hackathon and the training cutoff.

- **PROGRS (2604.02341)** — READ CLOSELY. Frozen PRM + within-outcome-group centering + windowed coherence + GRPO, in-domain math. Occupies our mechanism and windowing (§11.0); spared on signal-content, transfer, mechanism (§11.1, §11.3). The paper to beat.
- **Beyond Outcome Verification (2601.17223)** — NOT YET READ. Coherence = consistency between intermediate steps and final conclusion; GRPO; outcome vs process. Closest on *reward definition*. Must read: does it reward step→step entailment or step→answer consistency?
- **GRPO-CARE (2506.16141)** — NOT YET READ. Consistency-aware reward, GRPO, transfers to (video) benchmarks. Closest on *transfer*. Same "outcome-only degrades step-answer coherence" motivation.
- **Rubric-Grounded RL (2605.08061)** — NOT YET READ. GRPO + judge/rubric reward → improves on 4 external reasoning benchmarks. Occupies generic transfer.
- **Posterior-GRPO (2508.05170)** — NOT YET READ. Process reward conditioned on task success; generalizes code→math. Note: "reward process only on correct outcomes" is nearly OPPOSITE to our within-outcome-variance intent — worth contrasting.
- **FOVER (2505.15960)** — verifier-transfer (PRM generalizes cross-task via formal-logic training data), Best-of-K not RL-policy transfer. Different object; note but lower priority.

**Must-read-before-Phase-0 (novelty-critical):** Beyond Outcome Verification, then GRPO-CARE. These decide whether the narrowed core (§11) survives.

---

## 13. What v2 must resolve (updated)
- Read Beyond Outcome Verification + GRPO-CARE; confirm or kill the signal distinction (CRACK 1) against their reward definitions.
- Decide whether the paper's spine is (a) the narrowed top-down core (transition-entailment + transfer + mechanism) or (b) the bottom-up "what does a transition signal repair" characterization — or a fusion.
- Resolve §6 open decisions (transfer target first).
- First concrete Phase-0 build target.

---

## 14. Beyond Outcome Verification (2601.17223) — READ. Verdict: not a threat to the signal; new threat to the mechanism story.

- **Their reward (their §3.2):** each step scored on identifier-match (`sⁿ`) and label-match (`sˡ`) against a GOLD step identifier/label from rule-based task logic, summed + terminal outcome reward. This is **answer-key matching per step**, same family as Reward Granularity Eq. 1. Permutation-invariant. NOT step→step entailment, NOT step→answer consistency.
- **CRACK 1 survives cleanly.** The "coherence" that made the abstract look scary is not in the method. Our entailment signal (judged from the reasoning itself, NO gold step labels) is untouched.
- **Pattern now confirmed 3×** (Reward Granularity, Samineni, this): in this literature "coherence/consistency" ≈ "agreement with an external per-step label," not "internal logical flow." **Our sharpest, most repeatable distinction: transition-entailment needs no per-step answer key; every neighbor does.** Promote this to the lead differentiator.
- **NEW THREAT — to Exp 3, not the signal.** Their §3.3 Theorem 1 proves GRPO/DAPO advantage is positive-in-expectation for correct chains, negative for incorrect (given µ_correct > µ_incorrect). Combined with Binary Rewards (2605.02375) KL-control, the *theoretical* account of "why process reward reshapes the policy" is now a **two-paper front**. Exp 3 must distinguish: their theorem is about reward-MAGNITUDE (correct > incorrect in expectation); it says nothing about DIRECTION of policy change across problem types, or transfer. Our weight-geometry/transfer angle is what they don't touch — but Exp 3 now defends against two theoretical results, not zero.
- **Still unread, novelty-critical:** GRPO-CARE (2506.16141) — closest on transfer.

---

## 15. GRPO-CARE (2506.16141) — READ. Verdict: differentiator confirmed 4×; but §11.3 transfer claim must be softened.

- **Their consistency = reasoning→answer likelihood.** "Compare the model's reasoning-to-answer likelihood (via slowly-evolving reference) against group peers." Axis = does the ANSWER follow from the whole reasoning chain. NOT step→step entailment.
- **Distinction (finest yet):** a chain can make its answer likely (their consistency high) while having a broken internal step→step link (our signal catches it), and vice versa. Different geometry: reasoning→answer vs step→step.
- **"Consistency/coherence" now resolves to a DIFFERENT quantity in all 4 neighbors:** Reward Granularity (step vs answer-key), Samineni (error-presence), Beyond Outcome Verification (step vs answer-key), GRPO-CARE (reasoning→answer likelihood). None measure **step→step entailment from the reasoning's own content, with no gold labels.** Differentiator holds; each paper sharpens the empty spot.
- **CORRECTION to §11.3 (honest):** GRPO-CARE DOES test real transfer (in-dist / cross-environment / cross-env-task + other video benchmarks). "No one in this cluster tests transfer" was TOO STRONG. Revised claim: our transfer is differentiated by *signal* (step→step entailment) and *by testing cross-FORMAT (math→commonsense)* rather than cross-environment within multimodal — not by being the only ones who test transfer at all.
- **Small note for Exp 3:** they *replace KL penalty* with the consistency bonus, arguing strict KL over-constrains exploration. Another group treating KL-proximity as something to escape — a data point for the Story-A-vs-B framing (is our transfer "different direction" or "less KL movement"?).
