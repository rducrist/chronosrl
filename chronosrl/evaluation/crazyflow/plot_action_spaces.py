"""Plot the eval learning curves of the crazyflow_position action spaces over seeds.

Each seed's curve is smoothed with a centered moving average over `--window` evals, to remove
the eval noise of 100 episodes per eval, and drawn thin; the thick line is the mean over seeds.

The eval metrics are only logged to wandb, so the runs are pulled from there. Each action space
is one `runner.tag` (see VARIANTS) whose runs differ only in `environment.seed`.

pixi run -e crazyflow python -m chronosrl.evaluation.crazyflow.plot_action_spaces
"""

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import wandb

VARIANTS = {
    "eval_normal": "SRL, absolute",
    "eval_normal_chronosrl": "ChronoSRL, absolute",
    "eval_delta_0.2_chronosrl": "ChronoSRL, delta 0.2",
}
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]


def fetch(project: str, group: str, metric: str) -> dict[str, dict[int, dict[int, float]]]:
    """Return {tag: {seed: {global_step: value}}} of ``metric`` for the VARIANTS runs."""
    curves = defaultdict(dict)
    # Newest first, so a rerun of a tag and seed replaces the older run
    for run in wandb.Api().runs(project, filters={"group": group}, order="-created_at"):
        # Older runs predate runner.tag
        tag = run.config.get("runner", {}).get("tag")
        if tag not in VARIANTS:
            continue
        seed = run.config["environment"]["seed"]
        if seed in curves[tag]:
            print(f"skip {run.name}: a newer run of {tag} seed {seed} is used", flush=True)
            continue
        # Only the rows that log the metric, i.e. the evals
        rows = run.scan_history(keys=["global_step", metric])
        curves[tag][seed] = {row["global_step"]: row[metric] for row in rows}
        print(f"use  {run.name} ({run.state}): {len(curves[tag][seed])} evals", flush=True)
    return curves


def smooth(curve: dict[int, float], window: int) -> dict[int, float]:
    """Centered moving average over ``window`` evals; at the ends only the existing evals count."""
    steps = sorted(curve)
    values = np.array([curve[step] for step in steps])
    kernel = np.ones(window)
    # The full convolution, cut so that output i is centered on eval i
    center = slice(window // 2, window // 2 + len(values))
    sums = np.convolve(values, kernel)[center]
    counts = np.convolve(np.ones(len(values)), kernel)[center]
    return dict(zip(steps, sums / counts))


def mean_over_seeds(seeds: dict[int, dict[int, float]]) -> tuple[np.ndarray, np.ndarray]:
    """Return the steps and the mean over the seeds that reached each step."""
    steps = np.array(sorted(set().union(*seeds.values())))
    values = [[curve[step] for curve in seeds.values() if step in curve] for step in steps]
    return steps, np.array([np.mean(value) for value in values])


def main():
    """Fetch the runs, print the final evals and save the plot."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metric", default="eval/goal_reached", help="e.g. eval/time_at_goal")
    parser.add_argument("--group", default="crazyflow_position", help="wandb group (exp_name)")
    parser.add_argument("--project", default="rducristian-technical-university-of-munich/chronosrl")
    parser.add_argument(
        "--window", type=int, default=5, help="Evals per moving average, odd (1: no smoothing)"
    )
    parser.add_argument("--out", type=Path, help="Default: action_spaces_<metric>.pdf")
    args = parser.parse_args()
    out = args.out or Path(f"action_spaces_{args.metric.split('/')[-1]}.pdf")

    curves = fetch(args.project, args.group, args.metric)
    # Compare only the seeds that every variant with runs has
    common = set.intersection(*(set(seeds) for seeds in curves.values()))
    print(f"common seeds: {sorted(common)}", flush=True)
    curves = {tag: {seed: seeds[seed] for seed in common} for tag, seeds in curves.items()}
    fig, ax = plt.subplots(figsize=(5, 3.5))
    for (tag, label), color in zip(VARIANTS.items(), COLORS):
        if not curves.get(tag):
            print(f"no runs with tag {tag}", flush=True)
            continue
        seeds = {seed: smooth(curve, args.window) for seed, curve in curves[tag].items()}
        for curve in seeds.values():
            ax.plot(list(curve), list(curve.values()), color=color, linewidth=1, alpha=0.3)
        steps, mean = mean_over_seeds(seeds)
        print(f"{tag}: final smoothed mean {args.metric} {mean[-1]:.3f}", flush=True)
        ax.plot(steps, mean, color=color, linewidth=2, label=f"{label} (n={len(seeds)})")

    ax.set_xlabel("Environment steps")
    ax.set_ylabel(args.metric.split("/")[-1].replace("_", " ").capitalize())
    ax.grid(color="0.9")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out)
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
