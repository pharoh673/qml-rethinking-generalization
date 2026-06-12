"""End-to-end integration: authors' trained params + our regenerated states + our QCNN.

This validates the full forward pipeline (ED ground states -> StatePrep -> ported QCNN gates
-> readout) against the authors at once. We assert on the LOSS (mean p_true), which is robust
to ground-state degeneracy, plus a lenient accuracy floor. We do NOT require 100% train
accuracy: at degenerate FM/AFM points our ED ground-state representative differs from the
authors' DMRG one, so their params don't perfectly classify our states. The definitive,
self-consistent memorization check is in the training phase.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src import data_generation as dg, loss, qcnn

AUTHORS_ROOT = Path(__file__).resolve().parent.parent / "data" / "authors"
_HAVE = (AUTHORS_ROOT / "real_labels" / "8_qubits").is_dir()


@pytest.mark.skipif(not _HAVE, reason="run scripts/fetch_authors_data.py first")
@pytest.mark.parametrize("N", [5, 8])
def test_authors_bestparams_transfer_to_our_pipeline(N):
    fn = qcnn.build_qcnn(8)
    bp = np.loadtxt(dg.authors_dir("real_labels", 8) / f"{N}_training_data" / f"BEST_PARAMS_j1j2_{N}")
    assert bp.shape == (qcnn.num_params(8),)
    ds = dg.load_authors_dataset("real_labels", 8, N=N, periodic=False)

    mean_p_true = loss.dataset_loss(fn, bp, ds.states, ds.labels)
    acc = loss.accuracy(fn, bp, ds.states, ds.labels)
    # trained QCNN suppresses the true label far below uniform (0.25) -> gates+readout correct
    assert mean_p_true < 0.20, f"mean p_true {mean_p_true:.3f} (expected << 0.25)"
    # clearly above 4-class random (0.25); not 100% due to degenerate-state representatives
    assert acc > 0.6, f"train accuracy {acc:.3f}"
