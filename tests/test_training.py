"""Tests for src/training.py — CMA-ES loop, restart logic, checkpoint/resume, CSV logging."""
from __future__ import annotations

from pathlib import Path

import numpy as np

import pytest

from src.training import TrainConfig, train, train_qcnn, _load_checkpoint

AUTHORS_ROOT = Path(__file__).resolve().parent.parent / "data" / "authors"
_HAVE_AUTHORS = (AUTHORS_ROOT / "real_labels" / "8_qubits").is_dir()


def _sphere(target=0.3):
    return lambda p: float(np.sum((np.asarray(p) - target) ** 2))


def test_cma_minimizes_objective():
    obj = _sphere()
    acc = lambda p: 1.0 if obj(p) < 1e-2 else 0.0
    cfg = TrainConfig(dim=4, sigma0=0.4, tolfun=1e-10, max_generations=300,
                      max_restarts=3, seed=1)
    res = train(obj, acc, cfg)
    assert res.success
    assert res.best_loss < 1e-2
    np.testing.assert_allclose(res.best_params, 0.3, atol=0.2)


def test_fails_after_max_restarts_when_unreachable():
    obj = _sphere()
    acc = lambda p: 0.0                       # never satisfied
    cfg = TrainConfig(dim=3, sigma0=0.3, tolfun=1e-3, max_generations=5,
                      max_restarts=2, seed=2)
    res = train(obj, acc, cfg)
    assert not res.success
    assert res.restarts == 3                  # attempts 0,1,2 then exits (max_restarts+1)


def test_csv_results_are_appended(tmp_path: Path):
    obj = _sphere()
    acc = lambda p: 0.0
    csv_path = tmp_path / "results.csv"
    cfg = TrainConfig(dim=2, sigma0=0.3, tolfun=1e-2, max_generations=4,
                      max_restarts=1, seed=3)
    train(obj, acc, cfg, result_path=csv_path, tag="unit")
    text = csv_path.read_text()
    assert "train_accuracy" in text                     # header
    assert text.count("unit") == 2                      # one row per attempt (restarts 0,1)


def test_checkpoint_is_written_and_resumed(tmp_path: Path):
    obj = _sphere()
    ckpt = tmp_path / "ckpt.pkl"

    # first run: cannot succeed (acc always 0), 1 attempt of few generations -> checkpoint saved
    cfg_fail = TrainConfig(dim=3, sigma0=0.3, tolfun=1e-3, max_generations=6,
                           max_restarts=0, seed=4, checkpoint_every=2)
    r1 = train(obj, lambda p: 0.0, cfg_fail, checkpoint_path=ckpt)
    assert ckpt.exists()
    saved = _load_checkpoint(ckpt)
    assert saved["total_gen"] >= 1
    gen_after_first = r1.generations

    # second run with SAME checkpoint: resumes (loads es/total_gen) instead of restarting at 0
    cfg_ok = TrainConfig(dim=3, sigma0=0.3, tolfun=1e-3, max_generations=6,
                         max_restarts=3, seed=4, checkpoint_every=2)
    r2 = train(obj, lambda p: 1.0, cfg_ok, checkpoint_path=ckpt)  # acc now satisfied
    assert r2.success
    assert r2.generations >= gen_after_first        # continued, did not reset to 0


@pytest.mark.skipif(not _HAVE_AUTHORS, reason="run scripts/fetch_authors_data.py first")
def test_qcnn_memorizes_small_trainingset():
    """Definitive self-consistent check: QCNN + CMA-ES reaches 100% train accuracy on a
    small set (memorization — the mechanism behind the generalization-gap result). ~20s."""
    from src import data_generation as dg, qcnn

    ds = dg.load_authors_dataset("real_labels", 8, N=5, periodic=False)
    states, labels = ds.states[:3], ds.labels[:3]
    cfg = TrainConfig(dim=qcnn.num_params(8), sigma0=0.7, tolfun=5e-4,
                      max_generations=200, max_restarts=4, seed=0, checkpoint_every=10_000)
    res = train_qcnn(states, labels, 8, cfg)
    assert res.success and res.train_accuracy == 1.0
