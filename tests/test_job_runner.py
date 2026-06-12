"""Tests for src/job_runner.py — thread pinning, resume/skip logic, serial sweep."""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from src.config import ExperimentConfig
from src import job_runner

AUTHORS_ROOT = Path(__file__).resolve().parent.parent / "data" / "authors"
_HAVE = (AUTHORS_ROOT / "random_labels" / "8_qubits").is_dir()


def test_pin_blas_threads_sets_env():
    job_runner.pin_blas_threads()
    import os
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        assert os.environ[v] == "1"


def test_pending_jobs_skips_completed(tmp_path: Path):
    cfg = ExperimentConfig("random_labels", 8, N_values=[5, 8], seeds=[0, 1],
                           name="r", results_dir=str(tmp_path))
    jobs = cfg.jobs()                                   # 4 jobs
    # mark (N=5, seed=0) and (N=8, seed=1) as done
    with open(cfg.results_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["N", "seed", "ratio"])
        w.writeheader()
        w.writerow({"N": 5, "seed": 0, "ratio": ""})
        w.writerow({"N": 8, "seed": 1, "ratio": ""})
    pending = job_runner.pending_jobs(cfg, jobs)
    assert len(pending) == 2
    assert {(j["N"], j["seed"]) for j in pending} == {(5, 1), (8, 0)}


def test_pending_jobs_handles_ratio(tmp_path: Path):
    cfg = ExperimentConfig("partial_corruption", 8, N_values=[6], seeds=[0],
                           corruption_ratios=[0.0, 0.5], name="pc", results_dir=str(tmp_path))
    with open(cfg.results_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["N", "seed", "ratio"])
        w.writeheader()
        w.writerow({"N": 6, "seed": 0, "ratio": "0.0"})
    pending = job_runner.pending_jobs(cfg, cfg.jobs())
    assert len(pending) == 1 and pending[0]["ratio"] == 0.5


@pytest.mark.skipif(not _HAVE, reason="run scripts/fetch_authors_data.py first")
def test_serial_sweep_and_resume(tmp_path: Path):
    cfg = ExperimentConfig(
        "random_labels", 8, N_values=[5], seeds=[0], test_size=30,
        max_generations=30, max_restarts=1, name="smoke",
        results_dir=str(tmp_path), checkpoint_dir=str(tmp_path / "ck"), log_dir=str(tmp_path / "logs"),
    )
    rows = job_runner.run_sweep(cfg, workers=1)
    assert len(rows) == 1
    assert cfg.results_csv.exists()
    # per-job log written
    assert list(Path(cfg.log_dir).glob("*.log"))
    # second call resumes -> nothing to do
    rows2 = job_runner.run_sweep(cfg, workers=1)
    assert rows2 == []
