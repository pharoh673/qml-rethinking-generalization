#!/usr/bin/env python3
"""Benchmark per-(n,N) job runtime and estimate full-sweep wall-clock.

    python scripts/benchmark.py --experiment random_labels --n 8 --N 5 8 --seconds-cap 120

Runs a couple of real jobs with a modest budget, then extrapolates the Fig-3 sweep time
given a worker count. Run this BEFORE launching the full sweep (README compute plan).
"""
from __future__ import annotations

import argparse
import os
import time

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from src.config import ExperimentConfig  # noqa: E402
from src.runner import run_job  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="random_labels")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--N", type=int, nargs="+", default=[5, 8])
    ap.add_argument("--max-generations", type=int, default=200)
    ap.add_argument("--max-restarts", type=int, default=5)
    ap.add_argument("--workers", type=int, default=10, help="for the sweep extrapolation")
    args = ap.parse_args()

    times = {}
    for N in args.N:
        cfg = ExperimentConfig(
            args.experiment, args.n, N_values=[N], seeds=[0], test_size=200,
            max_generations=args.max_generations, max_restarts=args.max_restarts,
            name="benchmark", results_dir="results", checkpoint_dir="results/checkpoints",
        )
        t0 = time.time()
        row = run_job(cfg, cfg.jobs()[0], write_csv=False, checkpoint=False)
        dt = time.time() - t0
        times[N] = dt
        print(f"  n={args.n} N={N}: {dt:.1f}s  "
              f"(train_err={row['train_error']:.2f} test_err={row['test_error']:.2f} "
              f"restarts={row['restarts']} memorized={row['memorized']})")

    avg = sum(times.values()) / len(times)
    # full Fig-3a slice for this n: 5 N-values x 5 seeds = 25 jobs
    jobs = 25
    serial = jobs * avg
    parallel = serial / args.workers
    print(f"\nEstimated Fig-3a (n={args.n}) slice: {jobs} jobs x ~{avg:.0f}s "
          f"= {serial/60:.0f} min serial, ~{parallel/60:.0f} min on {args.workers} workers.")
    print("NOTE: random labels are HARDER to memorize than the modest budget here -> real "
          "runs use max_generations=500, max_restarts=30 and will take longer.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
