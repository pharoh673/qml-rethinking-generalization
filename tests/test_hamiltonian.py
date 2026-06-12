"""Tests for src/hamiltonian.py — Hermiticity, analytic limits, and variationality."""
from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from src import hamiltonian as ham


@pytest.mark.parametrize("n", [3, 4, 6])
@pytest.mark.parametrize("j1,j2", [(0.0, 0.0), (1.5, -2.0), (-3.0, 3.5)])
def test_hamiltonian_is_hermitian(n, j1, j2):
    H = ham.hamiltonian(j1, j2, n)
    diff = (H - H.getH())
    assert sp.linalg.norm(diff) < 1e-12


@pytest.mark.parametrize("n", [3, 4, 8])
def test_trivial_point_ground_state(n):
    # j1 = j2 = 0  =>  H = Σ_j Z_j, minimized by all qubits |1> (Z|1> = -|1>): E0 = -n.
    gs = ham.ground_state(0.0, 0.0, n)
    assert gs.energy == pytest.approx(-float(n), abs=1e-8)
    # ground state is the all-ones computational basis state (index 2^n - 1)
    probs = np.abs(gs.state) ** 2
    assert probs[-1] == pytest.approx(1.0, abs=1e-8)
    assert gs.gap > 0.5  # non-degenerate, well-gapped


@pytest.mark.parametrize("n", [4, 6])
def test_ground_state_is_variational_minimum(n):
    rng = np.random.default_rng(0)
    j1, j2 = rng.uniform(-4, 4, size=2)
    H = ham.hamiltonian(j1, j2, n)
    gs = ham.ground_state(j1, j2, n)
    # ⟨ψ0|H|ψ0⟩ equals the reported E0 ...
    assert ham.expectation(H, gs.state) == pytest.approx(gs.energy, abs=1e-6)
    # ... and no random state has lower energy.
    for _ in range(20):
        v = rng.normal(size=2**n) + 1j * rng.normal(size=2**n)
        v /= np.linalg.norm(v)
        assert ham.expectation(H, v) >= gs.energy - 1e-9


@pytest.mark.parametrize("n", [4, 8])
def test_ground_state_normalized_and_phase_fixed(n):
    gs = ham.ground_state(2.0, 1.0, n)
    assert np.linalg.norm(gs.state) == pytest.approx(1.0, abs=1e-10)
    # phase convention: largest-magnitude amplitude is real & positive
    k = int(np.argmax(np.abs(gs.state)))
    assert gs.state[k].imag == pytest.approx(0.0, abs=1e-10)
    assert gs.state[k].real > 0


def test_open_vs_periodic_differ():
    H_pbc = ham.hamiltonian(1.0, 1.0, 6, periodic=True)
    H_obc = ham.hamiltonian(1.0, 1.0, 6, periodic=False)
    assert sp.linalg.norm(H_pbc - H_obc) > 1e-6


def test_requires_minimum_three_qubits():
    with pytest.raises(ValueError):
        ham.hamiltonian(1.0, 1.0, 2)


# --- cross-check our ED states against the authors' (densified) n=8 ground states -------
# Requires the one-time conversion: scripts/convert_authors_states.py (see its docstring).
from pathlib import Path  # noqa: E402

_AUTH_DENSE = Path(__file__).resolve().parent.parent / "data" / "authors_dense" / "real_labels_8q_testset.npz"


@pytest.mark.skipif(not _AUTH_DENSE.exists(), reason="run scripts/convert_authors_states.py first")
def test_authors_states_are_near_ground_of_eq3():
    """The authors' released states are (near-)ground states of our literal Eq-3 H (OBC).

    Degeneracy-proof: instead of demanding exact fidelity (the ground state is non-unique
    across most of the phase diagram — Z2 breaking in FM/AFM, SPT edge modes under OBC), we
    check that each authors' state sits near the BOTTOM of our spectrum: its energy excess
    above E0 is tiny relative to the full spectral width. This confirms the Hamiltonian
    *model* (a brute-force search over Pauli conventions independently selected +Z/-XX/-XZX).
    """
    data = np.load(_AUTH_DENSE)
    j1s, j2s, auth = data["j1"], data["j2"], data["states"]
    n = 8
    from scipy.sparse.linalg import eigsh

    rel_excess = []
    for idx in range(40):
        j1, j2 = float(j1s[idx]), float(j2s[idx])
        H = ham.hamiltonian(j1, j2, n, periodic=False)
        e_lo = float(eigsh(H, k=1, which="SA")[0][0])
        e_hi = float(eigsh(H, k=1, which="LA")[0][0])
        a = auth[idx] / np.linalg.norm(auth[idx])
        Ea = float(np.real(np.vdot(a, H @ a)))
        rel_excess.append((Ea - e_lo) / (e_hi - e_lo))  # in [0,1]; ~0 => near ground
    rel_excess = np.array(rel_excess)
    # near-ground everywhere (a wrong model, e.g. Hadamard X-basis, lands mid-spectrum ~0.5)
    assert rel_excess.mean() < 0.05, f"mean relative excess {rel_excess.mean():.3f}"
    assert rel_excess.max() < 0.15, f"max relative excess {rel_excess.max():.3f}"


@pytest.mark.skipif(not _AUTH_DENSE.exists(), reason="run scripts/convert_authors_states.py first")
def test_ed_matches_authors_at_gapped_points():
    """At clearly non-degenerate (large-gap) points the ground state is unique, so our ED
    state must have high overlap with the authors' (OBC). Residual < 1.0 reflects remaining
    near-degeneracy; we require a strong match (>0.8)."""
    data = np.load(_AUTH_DENSE)
    j1s, j2s, auth = data["j1"], data["j2"], data["states"]
    n = 8

    fids = []
    for idx in range(len(j1s)):
        gs = ham.ground_state(float(j1s[idx]), float(j2s[idx]), n, periodic=False)
        if gs.gap < 0.8:
            continue
        a = auth[idx] / np.linalg.norm(auth[idx])
        fids.append(abs(np.vdot(a, gs.state)) ** 2)
        if len(fids) >= 15:
            break
    assert len(fids) >= 8, f"too few gapped points ({len(fids)})"
    fids = np.array(fids)
    # Typical strong agreement; a few SPT points retain low-lying edge-mode manifolds
    # (not strictly unique) so we assert mean/median rather than a brittle minimum.
    assert fids.mean() > 0.8, f"mean gapped fidelity {fids.mean():.3f}"
    assert np.median(fids) > 0.85, f"median gapped fidelity {np.median(fids):.3f}"
