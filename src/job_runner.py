"""Parallel dispatch of (experiment, n, N, seed[, ratio]) jobs across CPU cores.

Designed for the i5-14400 (16 threads): a process pool runs many jobs at once, with each
worker pinned to a SINGLE BLAS thread (OMP/MKL/OPENBLAS/NUMEXPR = 1) so W workers don't
oversubscribe the cores. Each job writes its own log file; results are appended to the
shared CSV. Already-completed jobs are skipped, so a sweep resumes after interruption /
across Kaggle's 12 h sessions (together with per-job CMA-ES checkpoints).
"""
from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

_THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def pin_blas_threads() -> None:
    """Force single-threaded BLAS so process-level parallelism doesn't oversubscribe."""
    for v in _THREAD_VARS:
        os.environ.setdefault(v, "1")


def _job_id(job: dict) -> tuple:
    return (job["N"], job["seed"], job.get("ratio"))


def pending_jobs(cfg, jobs: list[dict]) -> list[dict]:
    """Drop jobs already present in the results CSV (resume support)."""
    csv_path = cfg.results_csv
    if not Path(csv_path).exists():
        return list(jobs)
    import csv as _csv

    done = set()
    with open(csv_path) as f:
        for r in _csv.DictReader(f):
            ratio = r.get("ratio")
            ratio = None if ratio in (None, "", "None") else float(ratio)
            done.add((int(r["N"]), int(r["seed"]), ratio))
    return [j for j in jobs if _job_id(j) not in done]


def _run_one(cfg_dict: dict, job: dict) -> dict:
    """Worker entry point: pin threads, run one job, log to its own file."""
    pin_blas_threads()
    from .config import ExperimentConfig
    from .runner import run_job

    cfg = ExperimentConfig(**cfg_dict)
    rtag = "" if job.get("ratio") is None else f"_r{job['ratio']:.3f}"
    log_path = Path(cfg.log_dir) / f"{cfg.name}_n{cfg.n}_N{job['N']}_s{job['seed']}{rtag}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(log_path, "w") as log:
        log.write(f"[start] {time.ctime()} job={job}\n")
        log.flush()
        try:
            row = run_job(cfg, job)
            log.write(f"[done ] {time.time()-t0:.1f}s row={row}\n")
            return row
        except Exception as exc:  # noqa: BLE001 - record and re-raise for the dispatcher
            log.write(f"[error] {time.time()-t0:.1f}s {type(exc).__name__}: {exc}\n")
            raise


def run_sweep(cfg, workers: int | None = None, resume: bool = True) -> list[dict]:
    """Run all jobs in `cfg` across a process pool; return the collected result rows."""
    pin_blas_threads()
    jobs = pending_jobs(cfg, cfg.jobs()) if resume else cfg.jobs()
    if not jobs:
        print(f"[sweep] {cfg.name}: nothing to do (all jobs complete).")
        return []
    workers = workers or min(10, len(jobs))
    print(f"[sweep] {cfg.name}: {len(jobs)} job(s) on {workers} worker(s) "
          f"(experiment={cfg.experiment}, n={cfg.n}).")

    cfg_dict = asdict(cfg)
    rows: list[dict] = []
    if workers == 1:                                    # serial path (debug / single core)
        for job in jobs:
            rows.append(_run_one(cfg_dict, job))
            _print_row(rows[-1])
        return rows

    with ProcessPoolExecutor(max_workers=workers, initializer=pin_blas_threads) as ex:
        futures = {ex.submit(_run_one, cfg_dict, job): job for job in jobs}
        for fut in as_completed(futures):
            job = futures[fut]
            try:
                row = fut.result()
                rows.append(row)
                _print_row(row)
            except Exception as exc:  # noqa: BLE001
                print(f"[sweep] FAILED job={job}: {type(exc).__name__}: {exc}")
    return rows


def _print_row(row: dict) -> None:
    ratio = row["ratio"]
    rtag = "" if ratio is None else f" r{ratio}"
    print(f"  [{row['experiment']} n{row['n']} N{row['N']} s{row['seed']}{rtag}] "
          f"train_err={row['train_error']:.3f} test_err={row['test_error']:.3f} "
          f"gap={row['gen_gap']:.3f} ({row['seconds']:.0f}s, {row['restarts']} restarts)")
