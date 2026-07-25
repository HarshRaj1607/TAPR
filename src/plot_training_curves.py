#!/usr/bin/env python3
"""
Plot training curves for all four TAPR runs from their saved training logs.

Reads runs/<run>/training_log.json (the TRL log_history captured at step 400)
and writes figures to docs/figures/. No GPU, no model loading, no network —
this runs off data already in the repository.

Usage:
    python src/plot_training_curves.py
    python src/plot_training_curves.py --runs_dir ./runs --out_dir ./docs/figures
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── style ─────────────────────────────────────────────────────────────────────

C_BASE = "#B04A3F"   # outcome-only GRPO
C_TAPR = "#2E6F9E"   # TAPR
GRID   = {"color": "#000000", "alpha": 0.07, "linewidth": 0.8}

PAIRS = [
    ("GSM8K",      "gsm8k_baseline", "gsm8k_tapr"),
    ("StrategyQA", "sqa_baseline",   "sqa_tapr"),
]


def _style(ax, xlabel="Training step", ylabel=None, title=None):
    ax.grid(True, **GRID)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#999999")
    ax.tick_params(colors="#444444", labelsize=9)
    if xlabel: ax.set_xlabel(xlabel, fontsize=10, color="#222222")
    if ylabel: ax.set_ylabel(ylabel, fontsize=10, color="#222222")
    if title:  ax.set_title(title, fontsize=11, color="#111111", pad=10)


def load_runs(runs_dir):
    runs = {}
    for name in os.listdir(runs_dir):
        p = os.path.join(runs_dir, name, "training_log.json")
        if os.path.isfile(p):
            runs[name] = json.load(open(p))["log_history"]
    if not runs:
        raise SystemExit(f"No training_log.json found under {runs_dir}")
    return runs


def series(hist, key):
    """Extract (steps, values) for a metric, skipping entries that lack it."""
    xs, ys = [], []
    for e in hist:
        if key in e and "step" in e:
            xs.append(e["step"])
            ys.append(e[key])
    return xs, ys


def save(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    for ext, kw in (("png", {"dpi": 200}), ("svg", {})):
        path = os.path.join(out_dir, f"{name}.{ext}")
        fig.savefig(path, bbox_inches="tight", facecolor="white", **kw)
    plt.close(fig)
    print(f"  wrote {name}.png / {name}.svg")


# ── figures ───────────────────────────────────────────────────────────────────

def fig_degeneracy(runs, out_dir):
    """The headline result: fraction of groups with zero reward variance."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, (task, base, tapr) in zip(axes, PAIRS):
        for run, color, label in ((base, C_BASE, "Outcome-only GRPO"),
                                  (tapr, C_TAPR, "TAPR")):
            if run not in runs:
                continue
            xs, ys = series(runs[run], "frac_reward_zero_std")
            ax.plot(xs, ys, color=color, linewidth=2, label=label,
                    marker="o", markersize=3, markevery=4)
        ax.set_ylim(-0.04, 1.0)
        _style(ax, ylabel="frac_reward_zero_std" if task == "GSM8K" else None,
               title=task)
    axes[0].legend(frameon=False, fontsize=9, loc="center",
                   bbox_to_anchor=(0.5, 0.30))
    fig.suptitle("Groups contributing zero gradient, over training",
                 fontsize=13, y=1.02)
    fig.text(0.5, -0.06,
             "When all G=4 rollouts in a group receive the same reward, the GRPO advantage is zero "
             "for every rollout\nand the group teaches the model nothing. Lower is better.",
             ha="center", fontsize=8.5, color="#555555")
    save(fig, out_dir, "fig1_reward_degeneracy")


def fig_kl(runs, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, (task, base, tapr) in zip(axes, PAIRS):
        for run, color, label in ((base, C_BASE, "Outcome-only GRPO"),
                                  (tapr, C_TAPR, "TAPR")):
            if run not in runs:
                continue
            xs, ys = series(runs[run], "kl")
            ax.plot(xs, ys, color=color, linewidth=2, label=label)
        _style(ax, ylabel="KL(policy ‖ reference)" if task == "GSM8K" else None,
               title=task)
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle("Policy divergence from the frozen reference", fontsize=13, y=1.0)
    fig.text(0.5, -0.06,
             "Both runs stay in the 1e-3 to 1e-2 range — the policy barely moved in 400 steps. "
             "This is why\nthe in-domain accuracy differences are small and should not be over-read.",
             ha="center", fontsize=8.5, color="#555555")
    save(fig, out_dir, "fig2_kl_divergence")


def fig_reward(runs, out_dir):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for col, (task, base, tapr) in enumerate(PAIRS):
        for row, key, ylab in ((0, "reward", "Mean reward"),
                               (1, "reward_std", "Within-group reward std")):
            ax = axes[row][col]
            for run, color, label in ((base, C_BASE, "Outcome-only GRPO"),
                                      (tapr, C_TAPR, "TAPR")):
                if run not in runs:
                    continue
                xs, ys = series(runs[run], key)
                ax.plot(xs, ys, color=color, linewidth=1.8, label=label)
            _style(ax,
                   xlabel="Training step" if row == 1 else None,
                   ylabel=ylab if col == 0 else None,
                   title=task if row == 0 else None)
    axes[0][0].legend(frameon=False, fontsize=9)
    fig.suptitle("Reward level and within-group spread", fontsize=13, y=0.98)
    fig.text(0.5, 0.02,
             "TAPR's reward is on a different scale (outcome + λ·transition), so the levels are not "
             "comparable across conditions.\nThe lower row is the one that matters: reward std is the "
             "denominator of the GRPO advantage.",
             ha="center", fontsize=8.5, color="#555555")
    plt.subplots_adjust(bottom=0.13)
    save(fig, out_dir, "fig3_reward")


def fig_drift(runs, out_dir):
    """Behavioural drift: response length and entropy."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for col, (task, base, tapr) in enumerate(PAIRS):
        for row, key, ylab in ((0, "completions/mean_length", "Mean completion length (tokens)"),
                               (1, "entropy", "Entropy")):
            ax = axes[row][col]
            for run, color, label in ((base, C_BASE, "Outcome-only GRPO"),
                                      (tapr, C_TAPR, "TAPR")):
                if run not in runs:
                    continue
                xs, ys = series(runs[run], key)
                ax.plot(xs, ys, color=color, linewidth=1.8, label=label)
            _style(ax,
                   xlabel="Training step" if row == 1 else None,
                   ylabel=ylab if col == 0 else None,
                   title=task if row == 0 else None)
    axes[0][0].legend(frameon=False, fontsize=9)
    fig.suptitle("Behavioural drift during training", fontsize=13, y=0.98)
    fig.text(0.5, 0.005,
             "TAPR sits below its baseline on GSM8K (mean 159 vs 190 tokens) and above it on StrategyQA "
             "(159 vs 136) — settling at\nnearly the same length on both tasks while the outcome-only "
             "baselines diverge. Two tasks is an observation, not a finding.",
             ha="center", fontsize=8.5, color="#555555")
    plt.subplots_adjust(bottom=0.13)
    save(fig, out_dir, "fig4_policy_drift")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", default="./runs")
    ap.add_argument("--out_dir",  default="./docs/figures")
    args = ap.parse_args()

    runs = load_runs(args.runs_dir)
    print(f"Loaded {len(runs)} runs: {', '.join(sorted(runs))}")

    fig_degeneracy(runs, args.out_dir)
    fig_kl(runs, args.out_dir)
    fig_reward(runs, args.out_dir)
    fig_drift(runs, args.out_dir)
    print(f"\nFigures written to {args.out_dir}")


if __name__ == "__main__":
    main()
