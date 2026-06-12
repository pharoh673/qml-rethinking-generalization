#!/usr/bin/env python3
"""Single entry point for config-driven experiments.

    python run_experiment.py --config configs/random_labels_n8.yaml
    python run_experiment.py --config configs/random_labels_n8.yaml --workers 10
    docker compose run --rm experiment --config configs/random_labels_n8.yaml

Pins BLAS to one thread per worker BEFORE importing numpy so the process pool doesn't
oversubscribe cores. Resumes automatically (skips jobs already in the results CSV).
"""
from __future__ import annotations

import os

# Must happen before numpy is imported anywhere (no-op inside the Docker image, which already
# sets these, but required for native/venv runs).
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402

from src.config import ExperimentConfig  # noqa: E402
from src.job_runner import run_sweep  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Run a QCNN generalization experiment.")
    ap.add_argument("--config", required=True, help="path to a YAML experiment config")
    ap.add_argument("--workers", type=int, default=None,
                    help="parallel workers (default: min(10, #jobs))")
    ap.add_argument("--serial", action="store_true", help="run jobs serially (workers=1)")
    ap.add_argument("--no-resume", action="store_true",
                    help="re-run all jobs even if present in the results CSV")
    args = ap.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    rows = run_sweep(cfg, workers=1 if args.serial else args.workers, resume=not args.no_resume)
    print(f"[done] {len(rows)} job(s) -> {cfg.results_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
