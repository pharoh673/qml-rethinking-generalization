"""Tests for src/loss.py — readout formulas, argmin prediction, and p_true loss."""
from __future__ import annotations

import numpy as np
import pytest

from src import loss


def test_readout_probs_is_a_valid_distribution():
    rng = np.random.default_rng(0)
    for _ in range(50):
        # physical-ish expvals; exact validity for true states is checked in test_qcnn
        z_a, z_b = rng.uniform(-1, 1, size=2)
        zz = rng.uniform(-1, 1)
        p = loss.readout_probs([z_a, z_b, zz])
        assert p.sum() == pytest.approx(1.0, abs=1e-12)


def test_readout_probs_hand_cases():
    # all-zero expvals -> uniform
    np.testing.assert_allclose(loss.readout_probs([0, 0, 0]), [0.25] * 4)
    # z_a=z_b=zz=1 -> all weight on label 0 (p00)
    np.testing.assert_allclose(loss.readout_probs([1, 1, 1]), [1, 0, 0, 0], atol=1e-12)
    # exact formulas for a distinct point
    z_a, z_b, zz = 0.2, -0.4, 0.5
    expected = [(1 + zz + z_a + z_b) / 4, (1 - zz - z_a + z_b) / 4,
                (1 - zz + z_a - z_b) / 4, (1 + zz - z_a - z_b) / 4]
    np.testing.assert_allclose(loss.readout_probs([z_a, z_b, zz]), expected)


def test_prediction_is_argmin():
    assert loss.predict_from_probs(np.array([0.4, 0.1, 0.3, 0.2])) == 1
    assert loss.predict_from_probs(np.array([0.05, 0.4, 0.3, 0.25])) == 0


def test_sample_loss_is_prob_of_true_label():
    expvals = [0.2, -0.4, 0.5]
    probs = loss.readout_probs(expvals)
    for label in range(4):
        assert loss.sample_loss(expvals, label) == pytest.approx(probs[label])


def test_equalize_term_penalizes_spread_among_incorrect_classes():
    # expvals giving p = [0.325, 0.025, 0.475, 0.175]; incorrect classes (for label 0) differ
    expvals = [0.2, -0.4, 0.5]
    base = loss.sample_loss(expvals, 0, equalize=False)
    eq = loss.sample_loss(expvals, 0, equalize=True)
    assert eq > base                                   # spread among incorrect -> extra > 0
    # when the three incorrect classes are equal, the extra term vanishes
    # uniform expvals -> p=[.25,.25,.25,.25]; incorrect all equal
    assert loss.sample_loss([0, 0, 0], 0, equalize=True) == pytest.approx(0.25)
    # extra term is always non-negative
    for label in range(4):
        assert loss.sample_loss(expvals, label, equalize=True) >= loss.sample_loss(expvals, label) - 1e-12


def test_dataset_loss_and_accuracy_with_fake_qcnn():
    # fake qcnn: returns fixed expvals making label 0 the MINIMUM prob (so argmin predicts 0)
    # choose z_a=z_b=zz=1 -> p=[1,0,0,0]; argmin -> label 1 (first zero). Use a clearer case:
    # p=[0.1,0.4,0.3,0.2] via solving: pick expvals so p0 is smallest.
    def fake(params, state):
        return np.array([0.2, -0.4, 0.5])  # p = [0.325,0.025,0.475,0.175] -> argmin=1
    states = np.zeros((3, 4))
    labels = np.array([1, 1, 1])
    p = loss.readout_probs([0.2, -0.4, 0.5])
    assert loss.predict_from_probs(p) == 1
    assert loss.accuracy(fake, None, states, labels) == pytest.approx(1.0)
    assert loss.error_probability(fake, None, states, labels) == pytest.approx(0.0)
    assert loss.dataset_loss(fake, None, states, labels) == pytest.approx(p[1])
