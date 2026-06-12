"""Tests for src/qcnn.py — param counts, exact gate unitaries, output shape & validity."""
from __future__ import annotations

import numpy as np
import pytest

from src import loss, qcnn

# exact tc.gates.r(0.7,1.3,0.4) matrix, extracted from TensorCircuit (scripts/extract_tc_gates.py)
_R_PROBED = np.array(
    [[0.764842 - 0.172328j, -0.241728 - 0.571741j],
     [0.241728 - 0.571741j, 0.764842 + 0.172328j]]
)


@pytest.mark.parametrize("n,expected", [(8, 45), (16, 61), (32, 77)])
def test_num_params(n, expected):
    assert qcnn.num_params(n) == expected


def test_num_params_rejects_non_power_of_two():
    with pytest.raises(NotImplementedError):
        qcnn.num_params(12)


@pytest.mark.parametrize("n,wires", [(8, (3, 7)), (16, (7, 15)), (32, (15, 31))])
def test_output_wires(n, wires):
    assert qcnn.output_wires(n) == wires


def test_r_matrix_matches_tensorcircuit():
    np.testing.assert_allclose(qcnn.r_matrix(0.7, 1.3, 0.4), _R_PROBED, atol=1e-6)


def test_r_matrix_is_unitary():
    rng = np.random.default_rng(0)
    for _ in range(10):
        U = qcnn.r_matrix(*rng.uniform(0, 2 * np.pi, 3))
        np.testing.assert_allclose(U @ U.conj().T, np.eye(2), atol=1e-12)


def test_cr_matrix_is_controlled_r():
    U = qcnn._cr_matrix(0.7, 1.3, 0.4)
    np.testing.assert_allclose(U[:2, :2], np.eye(2), atol=1e-12)         # control |0> -> identity
    np.testing.assert_allclose(U[2:, 2:], _R_PROBED, atol=1e-6)          # control |1> -> r
    np.testing.assert_allclose(U[:2, 2:], 0, atol=1e-12)


def test_isingzz_equals_exp_minus_i_theta_zz():
    import pennylane as qml

    th = 0.7
    ZZ = np.diag([1, -1, -1, 1]).astype(complex)
    got = qml.matrix(qml.IsingZZ(2 * th, wires=[0, 1]))
    expected = np.diag(np.exp(-1j * th * np.diag(ZZ)))
    np.testing.assert_allclose(got, expected, atol=1e-12)


@pytest.mark.parametrize("n", [8, 16])
def test_qcnn_forward_shape_and_valid_distribution(n):
    fn = qcnn.build_qcnn(n)
    rng = np.random.default_rng(1)
    params = rng.uniform(0, 2 * np.pi, qcnn.num_params(n))
    psi = rng.normal(size=2**n) + 1j * rng.normal(size=2**n)
    psi /= np.linalg.norm(psi)

    out = fn(params, psi)
    assert out.shape == (3,)
    assert np.all(np.abs(out) <= 1 + 1e-9)              # valid Z expectations
    probs = loss.readout_probs(out)
    assert probs.sum() == pytest.approx(1.0, abs=1e-9)
    assert np.all(probs > -1e-9) and np.all(probs < 1 + 1e-9)  # genuine probabilities


def test_qcnn_is_deterministic():
    fn = qcnn.build_qcnn(8)
    rng = np.random.default_rng(2)
    params = rng.uniform(0, 2 * np.pi, 45)
    psi = rng.normal(size=256) + 1j * rng.normal(size=256)
    psi /= np.linalg.norm(psi)
    np.testing.assert_allclose(fn(params, psi), fn(params, psi), atol=1e-12)
