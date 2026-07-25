# Figures

Generated from `runs/*/training_log.json` — data already in the repository. No GPU, no model loading, no network.

```bash
python src/plot_training_curves.py
```

Each figure is written as both PNG (200 dpi, for GitHub and slides) and SVG (for the web page).

| Figure | Shows |
|---|---|
| `fig1_reward_degeneracy` | `frac_reward_zero_std` over 400 steps, both tasks. **The headline result.** Outcome-only GRPO sits around 0.64; TAPR collapses to near zero within a few steps and stays there. |
| `fig2_kl_divergence` | KL from the frozen reference policy. Both conditions stay in the 1e-3 to 1e-2 range — context for why in-domain accuracy differences are small. |
| `fig3_reward` | Mean reward and within-group reward std. The lower row is the mechanism: reward std is the denominator of the GRPO advantage. Reward *levels* are not comparable across conditions, since TAPR's reward includes the λ·transition term. |
| `fig4_policy_drift` | Mean completion length and entropy. TAPR settles near 159 tokens on both tasks while the baselines diverge to 190 (GSM8K) and 136 (StrategyQA). |

## Caveats

All four figures are **single-seed** (42), 400 steps, Qwen2.5-3B. Curves are logged every 10 steps, so each line is 40 points and the step-to-step jitter is real sampling noise, not smoothed away.

`fig1` is the only figure carrying a result that is definitional rather than statistical — zero within-group reward variance mathematically implies zero advantage, so no significance testing applies. The other three are descriptive: they show what these particular runs did, and none of them is replicated.

The length observation in `fig4` rests on two tasks. It is worth noticing and not worth claiming.
