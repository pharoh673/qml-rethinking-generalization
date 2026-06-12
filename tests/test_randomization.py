"""Tests for src/randomization.py — the three Fig-3 protocols, incl. authors'-data checks."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src import randomization as rnd

AUTHORS_ROOT = Path(__file__).resolve().parent.parent / "data" / "authors"
_HAVE = (AUTHORS_ROOT / "random_states" / "8_qubits").is_dir()


# --- (a) random labels -----------------------------------------------------------------
def test_random_labels_range_and_uniformity():
    rng = np.random.default_rng(0)
    labs = rnd.random_labels(8000, rng)
    assert labs.shape == (8000,)
    assert set(np.unique(labs)) <= {0, 1, 2, 3}
    counts = np.bincount(labs, minlength=4) / labs.size
    np.testing.assert_allclose(counts, 0.25, atol=0.03)   # ~uniform


# --- (b) partial corruption ------------------------------------------------------------
def test_corrupt_labels_exact_count_and_forced_different():
    rng = np.random.default_rng(1)
    truth = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    for c in range(len(truth) + 1):
        out = rnd.corrupt_labels(truth, c, rng)
        diff = out != truth
        assert diff.sum() == c                            # exactly c changed
        assert np.all(out[diff] != truth[diff])           # each corrupted -> different class
        assert set(np.unique(out)) <= {0, 1, 2, 3}


def test_corrupt_by_ratio_endpoints():
    rng = np.random.default_rng(2)
    truth = np.array([0, 1, 2, 3, 0, 1])
    np.testing.assert_array_equal(rnd.corrupt_labels_by_ratio(truth, 0.0, rng), truth)
    full = rnd.corrupt_labels_by_ratio(truth, 1.0, rng)
    assert np.all(full != truth)
    # 1/3 of 6 = 2 corrupted
    assert (rnd.corrupt_labels_by_ratio(truth, 1 / 3, rng) != truth).sum() == 2


def test_corrupt_labels_rejects_bad_args():
    rng = np.random.default_rng(3)
    with pytest.raises(ValueError):
        rnd.corrupt_labels(np.array([0, 1]), 3, rng)
    with pytest.raises(ValueError):
        rnd.corrupt_labels_by_ratio(np.array([0, 1]), 1.5, rng)


# --- (c) random states -----------------------------------------------------------------
def test_gaussian_resample_is_normalized_real_and_uncorrelated():
    rng = np.random.default_rng(4)
    # a structured real state
    psi = rng.normal(size=256)
    psi[0] += 5  # give it a peak / nonzero structure
    psi = psi / np.linalg.norm(psi)

    res = rnd.gaussian_resample_state(psi, rng)
    assert res.shape == psi.shape
    assert np.linalg.norm(res) == pytest.approx(1.0, abs=1e-12)
    assert np.max(np.abs(res.imag)) == 0.0                # real in -> real out
    # fresh i.i.d. draw -> essentially uncorrelated with the original
    corr = abs(np.corrcoef(psi.real, res.real)[0, 1])
    assert corr < 0.2


def test_gaussian_resample_is_reproducible():
    psi = np.random.default_rng(5).normal(size=64)
    psi /= np.linalg.norm(psi)
    a = rnd.gaussian_resample_state(psi, np.random.default_rng(7))
    b = rnd.gaussian_resample_state(psi, np.random.default_rng(7))
    np.testing.assert_array_equal(a, b)


# --- canonical validation against the authors' random-states inputs --------------------
@pytest.mark.skipif(not _HAVE, reason="run scripts/fetch_authors_data.py first")
def test_authors_random_states_match_protocol():
    """The authors' random-states inputs are real, unit-norm, and uncorrelated with the
    true ED ground state — exactly the signature of our gaussian_resample protocol."""
    from src import data_generation as dg, hamiltonian as ham

    d = dg.authors_dir("random_states", 8) / "5_training_data"
    stored = np.load(d / "train_groundstates.npy", allow_pickle=True)
    S = np.asarray(stored).reshape(len(stored), -1)
    j1 = np.atleast_1d(np.loadtxt(d / "J1coef_j1j2_5")).ravel()
    j2 = np.atleast_1d(np.loadtxt(d / "J2coef_j1j2_5")).ravel()

    assert np.max(np.abs(S.imag)) == 0.0
    np.testing.assert_allclose(np.linalg.norm(S, axis=1), 1.0, atol=1e-6)
    for k in range(len(S)):
        true = ham.ground_state(float(j1[k]), float(j2[k]), 8, periodic=False).state.real
        corr = abs(np.corrcoef(true, S[k].real)[0, 1])
        assert corr < 0.4, f"state {k} corr {corr:.3f} (expected ~0 for a fresh resample)"
