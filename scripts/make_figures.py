#!/usr/bin/env python3
"""Generate Fig-3-style plots from results CSVs.

    python scripts/make_figures.py --results results/*.csv --out figures/

Fig. 3a: empirical generalization gap |train_err - test_err| vs training-set size N, one
         curve per experiment (mean over seeds, +/- std shading), with the 0.75 maximal-gap
         dashed line.
Fig. 3b: test error vs corruption ratio r (partial_corruption), mean +/- std over seeds.
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

MAX_GAP = 0.75  # 4-class maximal generalization gap (random test accuracy = 0.25)
_LABELS = {"real_labels": "true labels", "random_labels": "random labels",
           "random_states": "random states", "partial_corruption": "partial corruption"}


def load_results(patterns: list[str]) -> pd.DataFrame:
    files = [f for p in patterns for f in glob.glob(p)]
    if not files:
        raise SystemExit(f"no result CSVs matched {patterns}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    return df


def _agg(df: pd.DataFrame, x: str, y: str):
    g = df.groupby(x)[y]
    return g.mean().index.values, g.mean().values, g.std(ddof=0).fillna(0).values


def plot_gap_vs_N(df: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for exp in [e for e in ["real_labels", "random_labels", "random_states"] if e in set(df.experiment)]:
        sub = df[df.experiment == exp]
        xs, mean, std = _agg(sub, "N", "gen_gap")
        ax.plot(xs, mean, "o-", label=_LABELS.get(exp, exp))
        ax.fill_between(xs, mean - std, mean + std, alpha=0.2)
    ax.axhline(MAX_GAP, ls="--", color="gray", lw=1, label="maximal gap (0.75)")
    ax.set_xlabel("training-set size $N$")
    ax.set_ylabel(r"generalization gap  $|\,\mathrm{err}_{train}-\mathrm{err}_{test}\,|$")
    ax.set_ylim(-0.02, 1.0)
    ax.set_title("Fig. 3a — generalization gap vs N (n=8)")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"[fig] {out}")


def plot_test_error_vs_N(df: pd.DataFrame, out: Path):
    """Robust discriminator (well-defined even when a job didn't fully memorize):
    random labels -> ~0.75 (chance); true labels -> below 0.5 and decreasing."""
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for exp in [e for e in ["real_labels", "random_labels", "random_states"] if e in set(df.experiment)]:
        sub = df[df.experiment == exp]
        xs, mean, std = _agg(sub, "N", "test_error")
        ax.plot(xs, mean, "o-", label=_LABELS.get(exp, exp))
        ax.fill_between(xs, mean - std, mean + std, alpha=0.2)
    ax.axhline(MAX_GAP, ls="--", color="gray", lw=1, label="chance (0.75)")
    ax.set_xlabel("training-set size $N$")
    ax.set_ylabel("test error")
    ax.set_ylim(0, 1.0)
    ax.set_title("Test error vs N (n=8)")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"[fig] {out}")


def plot_error_vs_ratio(df: pd.DataFrame, out: Path):
    sub = df[df.experiment == "partial_corruption"]
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for N in sorted(sub.N.unique()):
        s = sub[sub.N == N]
        xs, mean, std = _agg(s, "ratio", "test_error")
        ax.plot(xs, mean, "o-", label=f"N={N}")
        ax.fill_between(xs, mean - std, mean + std, alpha=0.2)
    ax.set_xlabel("corruption ratio $r$")
    ax.set_ylabel("test error")
    ax.set_title("Fig. 3b — test error vs corruption ratio (n=8)")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"[fig] {out}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+", default=["results/*.csv"])
    ap.add_argument("--out", default="figures")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = load_results(args.results)
    df = df[~df["name"].astype(str).str.startswith("_")]  # drop probe/scratch runs
    print(f"[fig] loaded {len(df)} rows; experiments: {sorted(df.experiment.unique())}")
    if df.gen_gap.notna().any():
        plot_gap_vs_N(df, out / "fig3a_gap_vs_N.png")
        plot_test_error_vs_N(df, out / "fig3a_test_error_vs_N.png")
    plot_error_vs_ratio(df, out / "fig3b_error_vs_ratio.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
