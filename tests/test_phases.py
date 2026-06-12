"""Tests for src/phases.py — phase-label constants, loaders, and authors'-data validation."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src import phases

REPO_ROOT = Path(__file__).resolve().parent.parent
AUTHORS_ROOT = REPO_ROOT / "data" / "authors"


# --- constants -------------------------------------------------------------------------
def test_phase_encoding_is_complete_and_consistent():
    assert phases.N_CLASSES == 4
    assert set(phases.PHASE_NAMES) == {0, 1, 2, 3}
    assert phases.PHASE_NAMES == {0: "SPT", 1: "FM", 2: "AFM", 3: "trivial"}
    # label -> output bitstring must match the QCNN readout convention (Eq. 4)
    assert phases.LABEL_TO_BITSTRING == {0: "00", 1: "01", 2: "10", 3: "11"}


# --- validation ------------------------------------------------------------------------
def test_validate_labels_accepts_valid_and_rejects_invalid():
    phases.validate_labels(np.array([0, 1, 2, 3, 0]))  # no raise
    with pytest.raises(ValueError):
        phases.validate_labels(np.array([0, 1, 4]))
    with pytest.raises(ValueError):
        phases.validate_labels(np.array([-1, 0]))


# --- loaders (synthetic round-trip) ----------------------------------------------------
def test_load_dataset_roundtrip(tmp_path: Path):
    j1 = np.array([1.5, -2.0, 0.25])
    j2 = np.array([-1.0, 3.0, 0.5])
    labels = np.array([0, 2, 3])
    # authors store each vector on a single whitespace-separated line
    (tmp_path / "J1").write_text(" ".join(f"{x:.12g}" for x in j1.tolist()))
    (tmp_path / "J2").write_text(" ".join(f"{x:.12g}" for x in j2.tolist()))
    (tmp_path / "LB").write_text(" ".join(str(x) for x in labels.tolist()))

    ds = phases.load_dataset(tmp_path / "J1", tmp_path / "J2", tmp_path / "LB")
    assert len(ds) == 3
    np.testing.assert_allclose(ds.j1, j1)
    np.testing.assert_allclose(ds.j2, j2)
    np.testing.assert_array_equal(ds.labels, labels)
    assert ds.summary()["counts"] == {"SPT": 1, "FM": 0, "AFM": 1, "trivial": 1}


def test_phasedataset_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        phases.PhaseDataset(j1=np.zeros(3), j2=np.zeros(3), labels=np.zeros(2, int))


# --- approximate map: shape/range sanity (NOT a correctness claim) ----------------------
def test_approximate_phase_label_returns_valid_classes():
    for j1, j2 in [(0.0, 0.0), (3.5, -3.0), (-3.0, 2.0), (0.1, 3.9), (-0.1, -3.9)]:
        assert phases.approximate_phase_label(j1, j2) in range(4)


# --- canonical authors' data (skipped if not yet fetched) ------------------------------
@pytest.mark.skipif(
    not AUTHORS_ROOT.exists() or phases.find_test_set(AUTHORS_ROOT) is None,
    reason="authors' data not fetched (run scripts/fetch_authors_data.py)",
)
def test_authors_test_set_is_well_formed():
    ds = phases.find_test_set(AUTHORS_ROOT)
    assert ds is not None
    s = ds.summary()
    assert s["n_points"] == 1000
    # couplings are sampled Uniform[-4,4]^2
    assert -4.0 <= s["j1_range"][0] and s["j1_range"][1] <= 4.0
    assert -4.0 <= s["j2_range"][0] and s["j2_range"][1] <= 4.0
    # all four phases present; SPT dominates this domain (we observed ~451/242/244/63)
    assert all(v > 0 for v in s["counts"].values())
    assert s["counts"]["SPT"] == max(s["counts"].values())


@pytest.mark.skipif(
    not AUTHORS_ROOT.exists() or phases.find_test_set(AUTHORS_ROOT) is None,
    reason="authors' data not fetched",
)
def test_approximate_map_is_known_to_be_inexact():
    """Regression guard: documents that the analytic map is ~62% (NOT canonical)."""
    ds = phases.find_test_set(AUTHORS_ROOT)
    pred = np.array([phases.approximate_phase_label(a, b) for a, b in zip(ds.j1, ds.j2)])
    acc = float((pred == ds.labels).mean())
    # if this ever jumps to ~1.0 someone reconstructed the exact map -> revisit phases.py
    assert 0.55 <= acc <= 0.75, f"approximate-map accuracy drifted to {acc:.3f}"
