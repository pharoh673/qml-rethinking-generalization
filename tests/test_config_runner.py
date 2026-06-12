"""Tests for src/config.py and src/runner.py — config schema, job expansion, single job."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.config import ExperimentConfig
from src import runner

AUTHORS_ROOT = Path(__file__).resolve().parent.parent / "data" / "authors"
_HAVE = (AUTHORS_ROOT / "random_labels" / "8_qubits").is_dir()
REPO_ROOT = Path(__file__).resolve().parent.parent


# --- config ----------------------------------------------------------------------------
def test_job_expansion_cartesian_product():
    cfg = ExperimentConfig("random_labels", 8, N_values=[5, 8], seeds=[0, 1, 2])
    jobs = cfg.jobs()
    assert len(jobs) == 2 * 3
    assert all(set(j) == {"experiment", "n", "N", "seed"} for j in jobs)


def test_partial_corruption_job_includes_ratio():
    cfg = ExperimentConfig("partial_corruption", 8, N_values=[6], seeds=[0],
                           corruption_ratios=[0.0, 0.5, 1.0])
    jobs = cfg.jobs()
    assert len(jobs) == 3
    assert {j["ratio"] for j in jobs} == {0.0, 0.5, 1.0}


def test_config_validation():
    with pytest.raises(ValueError):
        ExperimentConfig("bogus", 8, N_values=[5])
    with pytest.raises(ValueError):
        ExperimentConfig("partial_corruption", 8, N_values=[6])  # missing ratios


def test_yaml_roundtrip(tmp_path: Path):
    cfg = ExperimentConfig("random_labels", 8, N_values=[5, 8], seeds=[0, 1], name="x")
    p = tmp_path / "c.yaml"
    cfg.to_yaml(p)
    back = ExperimentConfig.from_yaml(p)
    assert back.experiment == "random_labels" and back.N_values == [5, 8] and back.seeds == [0, 1]


def test_shipped_configs_parse():
    for name in ["random_labels_n8", "partial_corruption_n8", "random_states_n8", "random_labels_n16"]:
        cfg = ExperimentConfig.from_yaml(REPO_ROOT / "configs" / f"{name}.yaml")
        assert cfg.jobs()  # non-empty


# --- runner: dataset preparation -------------------------------------------------------
@pytest.mark.skipif(not _HAVE, reason="run scripts/fetch_authors_data.py first")
def test_prepare_random_labels():
    trS, trL, teS, teL = runner.prepare_datasets("random_labels", 8, N=5, seed=0, ratio=None,
                                                 periodic=False, test_size=30)
    assert trS.shape == (5, 256) and trL.shape == (5,)
    assert teS.shape == (30, 256) and teL.shape == (30,)
    assert set(np.unique(trL)) <= {0, 1, 2, 3}


@pytest.mark.skipif(not _HAVE, reason="authors data not fetched")
def test_prepare_random_states_resamples_and_keeps_labels():
    trS, trL, teS, teL = runner.prepare_datasets("random_states", 8, N=5, seed=0, ratio=None,
                                                 periodic=False, test_size=20)
    # resampled states are real & unit norm
    assert np.max(np.abs(trS.imag)) == 0.0
    np.testing.assert_allclose(np.linalg.norm(trS, axis=1), 1.0, atol=1e-9)


@pytest.mark.skipif(not _HAVE, reason="authors data not fetched")
def test_prepare_partial_corruption_count():
    trS, trL, teS, teL = runner.prepare_datasets("partial_corruption", 8, N=6, seed=0, ratio=1 / 3,
                                                 periodic=False, test_size=10)
    # 1/3 of 6 = 2 training labels corrupted vs the clean truth (LABELS_6_0)
    _, _, truth, _ = runner._load_partial_corruption(8, 6, 10, False)
    assert (trL != truth).sum() == 2


# --- runner: full single job (small budget) --------------------------------------------
@pytest.mark.skipif(not _HAVE, reason="authors data not fetched")
def test_run_job_end_to_end(tmp_path: Path):
    cfg = ExperimentConfig(
        "random_labels", 8, N_values=[5], seeds=[0], test_size=30,
        max_generations=40, max_restarts=1, name="smoke",
        results_dir=str(tmp_path), checkpoint_dir=str(tmp_path / "ck"),
    )
    row = runner.run_job(cfg, cfg.jobs()[0])
    for key in ["train_error", "test_error", "gen_gap", "train_accuracy", "memorized"]:
        assert key in row
    assert 0.0 <= row["train_error"] <= 1.0
    assert 0.0 <= row["test_error"] <= 1.0
    assert row["gen_gap"] == pytest.approx(abs(row["train_error"] - row["test_error"]), abs=1e-6)
    assert cfg.results_csv.exists()                      # summary row appended
