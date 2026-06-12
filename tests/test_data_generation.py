"""Tests for src/data_generation.py — generation, caching, and authors'-file indexing."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src import data_generation as dg

REPO_ROOT = Path(__file__).resolve().parent.parent
AUTHORS_ROOT = REPO_ROOT / "data" / "authors"
_HAVE_AUTHORS = (AUTHORS_ROOT / "real_labels" / "8_qubits").is_dir()


def test_generate_shape_and_normalization():
    j1 = np.array([0.0, 1.5, -2.0])
    j2 = np.array([0.0, -1.0, 3.0])
    states = dg.generate_ground_states(j1, j2, n=4, periodic=False)
    assert states.shape == (3, 2**4)
    norms = np.linalg.norm(states, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-10)


def test_generate_rejects_mismatched_couplings():
    with pytest.raises(ValueError):
        dg.generate_ground_states(np.zeros(3), np.zeros(2), n=4)


def test_n_above_ed_max_raises():
    with pytest.raises(NotImplementedError):
        dg.generate_ground_states(np.array([1.0]), np.array([1.0]), n=18)


def test_cache_roundtrip_and_reuse(tmp_path: Path):
    j1, j2 = np.array([1.0, -1.5]), np.array([2.0, 0.5])
    s1 = dg.ground_states_cached(j1, j2, n=4, cache_dir=tmp_path)
    files = list(tmp_path.glob("gs_n4_obc_*.npz"))
    assert len(files) == 1                                  # cache file written
    s2 = dg.ground_states_cached(j1, j2, n=4, cache_dir=tmp_path)
    np.testing.assert_array_equal(s1, s2)                   # second call loads identical
    assert len(list(tmp_path.glob("*.npz"))) == 1           # no duplicate generated
    # different couplings -> different cache entry
    dg.ground_states_cached(j1 + 0.1, j2, n=4, cache_dir=tmp_path)
    assert len(list(tmp_path.glob("*.npz"))) == 2


def test_obc_pbc_cache_keys_differ(tmp_path: Path):
    j1, j2 = np.array([1.0]), np.array([1.0])
    dg.ground_states_cached(j1, j2, n=4, periodic=False, cache_dir=tmp_path)
    dg.ground_states_cached(j1, j2, n=4, periodic=True, cache_dir=tmp_path)
    assert len(list(tmp_path.glob("*.npz"))) == 2


# --- authors'-file integration (skipped if data not fetched) ---------------------------
@pytest.mark.skipif(not _HAVE_AUTHORS, reason="run scripts/fetch_authors_data.py first")
def test_authors_paths_resolve():
    j1, j2, lab = dg.authors_paths("real_labels", 8, N=5)
    assert j1.exists() and j2.exists() and lab.exists()
    tj1, tj2, tlab = dg.authors_paths("real_labels", 8, N=None)
    assert tj1.name.endswith("1000_Test_Set") and tlab.exists()


@pytest.mark.skipif(not _HAVE_AUTHORS, reason="authors data not fetched")
def test_load_authors_dataset_real_labels_n8(tmp_path: Path):
    # use a fresh cache dir so the test is hermetic
    dg.CACHE_ROOT = tmp_path
    ds = dg.load_authors_dataset("real_labels", 8, N=5, periodic=False)
    assert ds.n == 8 and len(ds) == 5
    assert ds.states.shape == (5, 2**8)
    phases_ok = set(np.unique(ds.labels)) <= {0, 1, 2, 3}
    assert phases_ok
    np.testing.assert_allclose(np.linalg.norm(ds.states, axis=1), 1.0, atol=1e-9)
    # every regenerated state is (near-)ground of H(j1,j2) under OBC
    for k in range(len(ds)):
        rel = dg.near_ground_relative_excess(ds.j1[k], ds.j2[k], 8, ds.states[k], periodic=False)
        assert rel < 0.06, f"state {k} not near-ground (rel excess {rel:.3f})"
