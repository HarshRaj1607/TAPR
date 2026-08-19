#!/usr/bin/env python3
"""
Build the standalone project page.

Reads the SVG figures from docs/figures/ and inlines them as base64 so the
output is a single portable file — publishable via GitHub Pages, or emailable
as-is with no external assets.

    python src/build_page.py            # -> index.html

Regenerate after changing figures (src/plot_training_curves.py).
"""

import argparse
import base64
import os

REPO_URL = "https://github.com/HarshRaj1607/TAPR"   # update if the repo moves


def embed(fig_dir, name):
    """Inline an SVG figure as a base64 data URI."""
    path = os.path.join(fig_dir, f"{name}.svg")
    if not os.path.exists(path):
        return f'<p class="missing">[figure {name} not found — run src/plot_training_curves.py]</p>'
    b64 = base64.b64encode(open(path, "rb").read()).decode()
    return f'<img src="data:image/svg+xml;base64,{b64}" alt="{name}">'


CSS = """
*,*::before,*::after{box-sizing:border-box}
:root{
  --ink:#1a1a1a; --muted:#5f6368; --faint:#8b8f94;
  --rule:#e3e5e8; --bg:#fdfdfc; --panel:#f6f7f8;
  --base:#B04A3F; --tapr:#2E6F9E; --warn:#8a6d1f;
}
html{-webkit-text-size-adjust:100%}
body{
  margin:0; background:var(--bg); color:var(--ink);
  font:16.5px/1.68 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;
}
.wrap{max-width:780px;margin:0 auto;padding:0 24px}
h1,h2,h3{line-height:1.25;font-weight:650;letter-spacing:-0.01em}
h1{font-size:2.35rem;margin:0 0 .35rem}
h2{font-size:1.5rem;margin:3.4rem 0 .2rem;padding-top:1.6rem;border-top:1px solid var(--rule)}
h3{font-size:1.09rem;margin:2.1rem 0 .5rem}
p{margin:.85rem 0}
a{color:var(--tapr);text-decoration:none;border-bottom:1px solid rgba(46,111,158,.3)}
a:hover{border-bottom-color:var(--tapr)}
code{font:13.5px/1.5 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  background:var(--panel);padding:.13em .38em;border-radius:3px}
pre{background:var(--panel);border:1px solid var(--rule);border-radius:6px;
  padding:14px 16px;overflow-x:auto;margin:1.1rem 0}
pre code{background:none;padding:0;font-size:13px}

header{padding:5rem 0 0}
.sub{font-size:1.16rem;color:var(--muted);margin:.5rem 0 1.4rem;line-height:1.5}
.byline{font-size:.93rem;color:var(--muted);margin:0 0 .3rem}
.byline strong{color:var(--ink);font-weight:600}
.origin{font-size:.86rem;color:var(--faint);margin:0 0 1.8rem}
.links{display:flex;flex-wrap:wrap;gap:9px;margin:0 0 1rem}
.links a{display:inline-block;padding:.42rem .85rem;border:1px solid var(--rule);
  border-radius:5px;font-size:.88rem;color:var(--ink);background:#fff;border-bottom:1px solid var(--rule)}
.links a:hover{border-color:var(--muted)}

.tldr{background:var(--panel);border:1px solid var(--rule);border-radius:8px;
  padding:1.3rem 1.5rem;margin:2.4rem 0 0}
.tldr h4{margin:0 0 .7rem;font-size:.78rem;text-transform:uppercase;
  letter-spacing:.09em;color:var(--faint);font-weight:650}
.tldr ul{margin:0;padding-left:1.1rem}
.tldr li{margin:.5rem 0}
.tag{display:inline-block;font-size:.7rem;font-weight:700;letter-spacing:.05em;
  padding:.1rem .42rem;border-radius:3px;vertical-align:.09em;margin-right:.4rem}
.t-solid{background:#e3f0e6;color:#2c6b3a}
.t-weak{background:#fbf0d8;color:var(--warn)}
.t-null{background:#eceef0;color:var(--muted)}

figure{margin:2rem 0}
figure img{width:100%;height:auto;display:block}
figcaption{font-size:.85rem;color:var(--muted);margin-top:.7rem;line-height:1.55}
.missing{color:var(--warn);font-size:.9rem}

table{border-collapse:collapse;width:100%;margin:1.3rem 0;font-size:.9rem}
th,td{text-align:left;padding:.52rem .7rem;border-bottom:1px solid var(--rule)}
th{font-weight:650;font-size:.79rem;text-transform:uppercase;
  letter-spacing:.05em;color:var(--faint);border-bottom:1.5px solid var(--rule)}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
tr.hi td{background:rgba(46,111,158,.045)}
.ci{color:var(--faint);font-size:.86em;white-space:nowrap}

.eq{background:var(--panel);border-left:3px solid var(--rule);padding:.9rem 1.2rem;
  margin:1.3rem 0;font:15px/1.7 ui-monospace,SFMono-Regular,Menlo,monospace;overflow-x:auto}
.note{border-left:3px solid var(--warn);background:#fdfaf2;padding:.85rem 1.2rem;
  margin:1.4rem 0;font-size:.93rem}
.note strong{color:var(--warn)}
.kicker{font-size:.78rem;text-transform:uppercase;letter-spacing:.09em;
  color:var(--faint);font-weight:650;margin:0 0 .3rem}

footer{margin:4.5rem 0 3rem;padding-top:1.6rem;border-top:1px solid var(--rule);
  font-size:.87rem;color:var(--muted)}
footer p{margin:.4rem 0}
@media (max-width:640px){
  header{padding-top:2.8rem} h1{font-size:1.85rem} .wrap{padding:0 18px}
  table{font-size:.83rem} th,td{padding:.45rem .4rem}
}
"""


def build(fig_dir):
    f1 = embed(fig_dir, "fig1_reward_degeneracy")
    f4 = embed(fig_dir, "fig4_policy_drift")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>TAPR — Transition-Aware Process Reward</title>
<meta name="description" content="A dense reward signal for GRPO fine-tuning of small language models, built by scoring step-to-step reasoning transitions with a live judge. Eliminates reward degeneracy on 52-63% of training groups.">
<style>{CSS}</style>
</head>
<body>
<div class="wrap">

<header>
  <p class="kicker">Reinforcement learning · reward design</p>
  <h1>Transition-Aware Process Reward</h1>
  <p class="sub">Outcome-only GRPO produces no gradient on the majority of its training data.
  TAPR restores it by scoring the <em>quality of reasoning transitions</em> with a live judge —
  a signal that varies even when every rollout reaches the same answer.</p>
  <p class="byline"><strong>Harsh Raj</strong> · Ramaiah Institute of Technology, Bangalore</p>
  <div class="links">
    <a href="{REPO_URL}">Code &amp; results on GitHub →</a>
    <a href="{REPO_URL}/blob/main/docs/RESEARCH_RECORD.md">Full research record</a>
  </div>

  <div class="tldr">
    <h4>What this work established</h4>
    <ul>
      <li><span class="tag t-solid">SOLID</span><strong>Reward degeneracy is eliminated.</strong>
      Groups contributing zero gradient fall from 52.5% → 0.0% (GSM8K) and 62.5% → 0.3%
      (StrategyQA). Measured directly by the trainer, and definitional rather than
      statistical — zero within-group variance <em>implies</em> zero advantage.</li>
      <li><span class="tag t-weak">UNDERPOWERED</span><strong>Cross-task transfer looks
      promising and is not yet significant.</strong> A 4.5pp gap on GSM8K→StrategyQA,
      95% CI [−5.2, +14.2]pp at n=200, single seed. Reported because the direction is
      consistent with the mechanism, not because the statistics support it.</li>
      <li><span class="tag t-null">NULL</span><strong>In-domain accuracy is unchanged.</strong>
      +0.23pp on GSM8K at n=1319 is three answers. A 3B student moved 400 steps is enough
      to demonstrate a mechanism, not an accuracy gain.</li>
    </ul>
  </div>
</header>

<h2>The problem</h2>

<p>GRPO computes each rollout's advantage by comparing it against the others in its group:</p>

<div class="eq">A<sub>i</sub> = (R<sub>i</sub> − mean(R<sub>1</sub>…R<sub>G</sub>)) / std(R<sub>1</sub>…R<sub>G</sub>)</div>

<p>When every rollout in a group earns the same reward — all correct, or all wrong — the
standard deviation is zero, so <code>A<sub>i</sub> = 0</code> for every rollout and the group
contributes <strong>no gradient at all</strong>.</p>

<p>This is not an edge case. Measured on Qwen2.5-3B:</p>

<table>
  <tr><th>Task</th><th class="num">Groups with zero reward variance</th></tr>
  <tr><td>GSM8K</td><td class="num">52.5%</td></tr>
  <tr><td>StrategyQA</td><td class="num">62.5%</td></tr>
</table>

<p>Outcome-only GRPO is blind on most of what it is shown. Easy problems (all rollouts
succeed) and hard problems (all rollouts fail) both collapse to zero signal. Only the narrow
band where some rollouts succeed and some fail teaches the model anything.</p>

<h2>The idea</h2>

<p>Four rollouts can reach the same answer by four different routes. A chain that guessed
correctly and a chain that reasoned correctly both receive <code>R<sub>outcome</sub> = 1</code>,
but they are not the same object. TAPR adds a second term that scores the reasoning itself:</p>

<div class="eq">R<sub>total</sub> = R<sub>outcome</sub> + λ · R<sub>transition</sub></div>

<p><code>R<sub>transition</sub></code> comes from a frozen judge model scoring sliding windows
of consecutive reasoning steps — asking not "is the answer right?" but "does each step follow
from the one before it?" The two signals are close to orthogonal, and that orthogonality is
what restores within-group variance.</p>

<p><strong>The judge must run live, during training.</strong> Pre-computing a transition score
per problem gives every rollout in a group the same value, leaving within-group variance at
zero and the degeneracy unsolved. Live scoring is structurally required, not merely convenient
— which is also what makes the method expensive.</p>

<h2>Result 1 — degeneracy is eliminated</h2>

<figure>
  {f1}
  <figcaption><strong>Fraction of groups contributing zero gradient, over 400 training
  steps.</strong> Outcome-only GRPO sits around 0.64 throughout. TAPR collapses to near zero
  within a few steps and stays there, on two structurally different tasks. Logged every 10
  steps by the trainer; single seed.</figcaption>
</figure>

<table>
  <tr><th>Task</th><th class="num">Outcome-only GRPO</th><th class="num">TAPR</th></tr>
  <tr><td>GSM8K</td><td class="num">0.525</td><td class="num"><strong>0.000</strong></td></tr>
  <tr><td>StrategyQA</td><td class="num">0.625</td><td class="num"><strong>0.003</strong></td></tr>
</table>

<p>Effective training data goes from roughly 47% of problems to roughly 100%. This is the one
result here that needs no significance test: zero within-group reward variance mathematically
implies zero advantage, so the metric is a direct measurement of the mechanism rather than an
estimate of an effect.</p>

<h2>Result 2 — cross-task transfer</h2>

<p>GSM8K-trained models evaluated on StrategyQA. Neither model saw a single StrategyQA example
during training.</p>

<table>
  <tr><th>Model</th><th class="num">StrategyQA</th><th class="num">vs. untrained</th><th class="num">95% CI</th></tr>
  <tr><td>Zero-shot (untrained)</td><td class="num">61.0%</td><td class="num">—</td><td class="num">—</td></tr>
  <tr><td>Outcome-only GRPO</td><td class="num">55.5%</td><td class="num">−5.5pp</td><td class="num ci">[−15.2, +4.2]</td></tr>
  <tr class="hi"><td>TAPR</td><td class="num">60.0%</td><td class="num">−1.0pp</td><td class="num ci">[−10.6, +8.6]</td></tr>
</table>

<p>Outcome-only GRPO trained on GSM8K <em>damaged</em> the model's general reasoning, landing
5.5pp below the untrained baseline. TAPR lost 1.0pp. The proposed mechanism: outcome-only
reward optimises for format compliance — numbered steps terminating in <code>####</code> —
driving over-specialisation, while TAPR optimises for step-to-step validity, a more
transferable property.</p>

<div class="note">
<p><strong>This result is underpowered and is presented as a direction, not a finding.</strong>
The 4.5pp TAPR-vs-baseline gap carries a 95% CI of [−5.2, +14.2]pp at n=200 — it includes
zero. Both models are evaluated on the identical held-out set, so the correct test is paired
(McNemar), which would be tighter; whether it clears zero depends entirely on how many
problems the two models disagree about. Per-problem outputs were not saved during the original
evaluation, so that test cannot currently be run. Re-running evaluation on the stored adapters
would settle it without any retraining.</p>
</div>

<h2>Result 3 — in-domain accuracy</h2>

<table>
  <tr><th>Benchmark</th><th class="num">n</th><th class="num">Zero-shot</th><th class="num">Outcome-only</th><th class="num">TAPR</th><th class="num">TAPR − ORM</th></tr>
  <tr><td>GSM8K</td><td class="num">1319</td><td class="num">72.86%</td><td class="num">77.03%</td><td class="num">77.26%</td><td class="num">+0.23pp <span class="ci">[−3.0, +3.4]</span></td></tr>
  <tr><td>StrategyQA</td><td class="num">200</td><td class="num">61.00%</td><td class="num">62.50%</td><td class="num">63.50%</td><td class="num">+1.00pp <span class="ci">[−8.5, +10.5]</span></td></tr>
  <tr><td>MATH-500</td><td class="num">200</td><td class="num">47.00%</td><td class="num">~47%</td><td class="num">~47%</td><td class="num">~0pp</td></tr>
</table>

<p><strong>None of these gaps is claimable.</strong> +0.23pp on GSM8K is three additional
correct answers out of 1319. MATH-500 is a clean null: GSM8K training does not transfer to
competition mathematics, which usefully bounds how far the transfer effect reaches.</p>

<p>What <em>is</em> supported: both trained models beat the untrained model on GSM8K by about
4pp, with intervals excluding zero (TAPR vs. zero-shot: +4.40pp, CI [+1.10, +7.70]). GRPO
works. The question this project asks is whether the reward <em>shape</em> changes what gets
learned, and 400 steps on a 3B student is not enough to answer it through accuracy.</p>

<div class="note">
<p><strong>A cautionary data point worth recording.</strong> At n=200 the GSM8K TAPR-vs-ORM gap
read as 3.0pp. At the full 1319 problems it collapsed to 0.23pp. Small-sample evaluation in RL
is unstable in both directions, and an early read is not a preview of the final number.</p>
</div>

<h2>Result 4 — judge validation</h2>

<p>Before training against a judge, the judge was calibrated: for each problem, generate one
correct reasoning chain and one deliberately flawed chain, score both, and measure whether the
judge separates them.</p>

<table>
  <tr><th>Task</th><th>Judge</th><th class="num">n</th><th class="num">Spearman r</th><th class="num">95% CI</th><th class="num">Separation</th></tr>
  <tr><td>GSM8K</td><td>GPT-4.1-mini</td><td class="num">93</td><td class="num">0.600</td><td class="num ci">[0.45, 0.72]</td><td class="num">90.3%</td></tr>
  <tr class="hi"><td>GSM8K</td><td>Qwen2.5-7B</td><td class="num">20</td><td class="num">0.734</td><td class="num ci">[0.43, 0.89]</td><td class="num">95%</td></tr>
  <tr><td>StrategyQA</td><td>Qwen2.5-7B</td><td class="num">20</td><td class="num">0.531</td><td class="num ci">[0.12, 0.79]</td><td class="num">85%</td></tr>
  <tr><td>StrategyQA</td><td>Qwen2.5-32B</td><td class="num">20</td><td class="num">0.388</td><td class="num ci">[−0.07, 0.71]</td><td class="num">70%</td></tr>
  <tr><td>MATH</td><td>Qwen2.5-7B</td><td class="num">20</td><td class="num">0.081</td><td class="num ci">—</td><td class="num">50%</td></tr>
  <tr><td>MATH</td><td>Qwen2.5-Math-7B</td><td class="num">20</td><td class="num">—</td><td class="num ci">—</td><td class="num">0%</td></tr>
  <tr><td>MATH</td><td>Qwen2.5-32B</td><td class="num">20</td><td class="num">0.223</td><td class="num ci">—</td><td class="num">40%</td></tr>
</table>

<p>An open-weight 7B judge is <strong>competitive with GPT-4.1-mini</strong>, which removes the
closed-weight API dependency from the method entirely. It is worth being precise about what
that does and does not say: 0.734 versus 0.600 looks decisive, but at n=20 the intervals
overlap almost completely, and the same is true of the separation rates. Establishing that the
open-weight judge is genuinely <em>better</em> would need roughly n=80 — which is inference
only, requiring no training, and is the cheapest open question in the project.</p>

<h2>What didn't work</h2>

<p><strong>MATH training was abandoned.</strong> All three candidate judges failed, each in a
different way:</p>

<ul>
  <li><strong>Qwen2.5-7B</strong> — r=0.081. No statistical signal at all.</li>
  <li><strong>Qwen2.5-Math-7B</strong> — total format collapse. The model ignores the SCORE
  instruction and emits its own solution instead. A constraining system prompt had no effect;
  its math-solving instruction tuning overrides any grader role. Every window fell back to the
  neutral 0.5 default, so correct and flawed chains scored identically — 0/20 separation.</li>
  <li><strong>Qwen2.5-32B</strong> — perfect format compliance, but the algebraic dimension
  separated 0/7.</li>
</ul>

<p>The diagnostic pattern is consistent across all three: the <em>relevance</em> dimension
separates well (4–5 of 6), while <em>algebraic validity</em> does not (0–2 of 7). Relevance is
checkable by reading. Algebraic validity requires independently solving the problem. No
open-weight model at 7B–32B scale does the latter reliably for competition mathematics. This
is a capability ceiling, not a prompt-engineering problem — which is why the task was dropped
rather than iterated on.</p>

<p><strong>MMLU was never evaluated.</strong> Multiple-choice knowledge retrieval is a poor fit
for a method that scores reasoning transitions: short answers yield too few windows for the
signal to exist.</p>

<h2>An open observation</h2>

<figure>
  {f4}
  <figcaption><strong>Completion length and entropy over training.</strong> TAPR settles near
  159 tokens on both tasks, approaching from opposite directions — down from 171 on GSM8K, up
  from 139 on StrategyQA — while the outcome-only baselines diverge to 190 and 136.</figcaption>
</figure>

<p>The two TAPR runs converge on nearly the same completion length despite training on
structurally different tasks, and they get there from opposite starting points. One
explanation: both use the same judge with the same window configuration, so the transition
reward carries a preferred chain geometry that is indifferent to the task, while the outcome
reward is entirely task-shaped.</p>

<p>If that is right, it raises a question this work cannot yet answer — whether part of the
transfer effect is a length-regularisation effect rather than a reasoning-quality effect. The
observation rests on two tasks and one seed, and two points always lie on a line. It is
recorded here because it is testable: changing the window configuration should move the
attractor if the judge geometry is causing it.</p>

<h2>Honest limitations</h2>

<ul>
  <li><strong>3B student, 400 steps.</strong> KL stayed at 0.001–0.009 throughout — the policy
  barely moved. Enough to demonstrate a mechanism; not enough to demonstrate accuracy gains.</li>
  <li><strong>Single seed everywhere.</strong> In RL, seed-to-seed variance is routinely larger
  than the effects being measured.</li>
  <li><strong>n=200 for StrategyQA and MATH-500.</strong> Standard error ≈ 3.4pp on a single
  proportion, and roughly 4.9pp on a difference between two.</li>
  <li><strong>StrategyQA has no labelled test set.</strong> Evaluation uses a held-out 10% of
  the train split. Both runs share seed 42, so the eval sets are identical across conditions.</li>
  <li><strong>λ=0.3 was tuned qualitatively</strong> in an earlier phase and never formally
  ablated at full training scale.</li>
  <li><strong>The transfer mechanism is a hypothesis.</strong> Two explanations remain live:
  TAPR as implicit regulariser (moves the policy less) versus TAPR as different learning (moves
  it comparably, in a different direction).</li>
</ul>

<h2>Reproducing</h2>

<p>Single A100-40GB. 4-bit quantised: 3B student ≈ 2.7GB, 7B judge ≈ 5.6GB. Roughly 7 hours for
400 steps.</p>

<pre><code>pip install -r requirements.txt

# validate the judge before training anything
python src/judge_validation_strategyqa.py --output_file results/sqa_judge.json

# smoke test — 5 steps, 20 problems, ~5 min
python src/train_grpo.py --task gsm8k --mode tapr --smoke_test --output_dir /tmp/smoke

# train
python src/train_grpo.py --task gsm8k --mode tapr --lambda_val 0.3 \\
  --output_dir ./runs/gsm8k_tapr</code></pre>

<p>Batching every window judge call for a training step into a single <code>model.generate()</code>
gave an <strong>11.6× speedup</strong> (772s → 67s per step), which is what made 400-step
training feasible at all. This requires <code>padding_side="left"</code> so sequences align at
the generation start; without it, attention patterns break and judge outputs become garbage.</p>

<p>Trained LoRA adapters and per-step training logs for all four runs are in
<a href="{REPO_URL}/tree/main/runs"><code>runs/</code></a>. Every results file carries a
<code>provenance</code> field marking whether figures are exact transcriptions from run logs or
approximate.</p>

<footer>
  <p><strong>Harsh Raj</strong> — B.E. Computer Science, Ramaiah Institute of Technology, Bangalore</p>
  <p style="margin-top:1rem">Built on TRL, PEFT and Qwen2.5. GRPO is from DeepSeekMath
  (Shao et al., 2024). Process-reward framing draws on Lightman et al., 2023. MIT licensed.</p>
  <p><a href="{REPO_URL}">{REPO_URL.replace('https://', '')}</a></p>
</footer>

</div>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fig_dir", default="./docs/figures")
    ap.add_argument("--out", default="./index.html")
    a = ap.parse_args()

    html = build(a.fig_dir)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"wrote {a.out}  ({len(html)/1024:.0f} KB, self-contained)")


if __name__ == "__main__":
    main()
