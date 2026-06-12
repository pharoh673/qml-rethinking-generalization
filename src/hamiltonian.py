"""Generalized cluster Hamiltonian and exact-diagonalization ground states.

    H = Σ_j ( Z_j − j1·X_j X_{j+1} − j2·X_{j−1} Z_j X_{j+1} )      (Eq. 3, periodic BC)

Used for the ED ground states at n ∈ {8, 10, 12}. (n = 16, 32 use DMRG/MPS — see
src/data_generation.py.) Couplings j1, j2 are sampled Uniform[-4, 4]^2; phase labels
come from the authors' canonical files (see src/phases.py), NOT from this module.

Qubit/index convention (kept consistent across the whole pipeline):
    site i  ==  PennyLane wire i  ==  the i-th (leftmost) Kronecker factor,
so the statevector index is the bitstring b_0 b_1 ... b_{n-1} with site 0 the MOST
significant bit. This matches `qml.StatePrep`, so ED states feed straight into the QCNN.

Boundary conditions: default is OPEN (periodic=False). We determined empirically that the
authors' released ground states are reproduced by the literal Eq-3 Pauli structure with
OPEN boundaries (DMRG convention): a brute-force search over Pauli conventions selects
+Z/-XX/-XZX, and under OBC every non-degenerate (large-gap) point matches their state at
fidelity >= 0.85, whereas PBC mismatches badly in the SPT region. The residual below 1.0 is
SPT edge-mode near-degeneracy (the ground state there is not unique). The paper does not
state the boundary condition -> see README "Open items / ZENODO-CHECK".
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

# single-qubit Pauli operators (sparse, complex)
_I = sp.identity(2, format="csr", dtype=complex)
_X = sp.csr_matrix(np.array([[0, 1], [1, 0]], dtype=complex))
_Z = sp.csr_matrix(np.array([[1, 0], [0, -1]], dtype=complex))


def _pauli_op(factors: dict[int, sp.spmatrix], n: int) -> sp.csr_matrix:
    """Kronecker product over sites 0..n-1, with `factors[i]` on listed sites else I."""
    out = factors.get(0, _I)
    for i in range(1, n):
        out = sp.kron(out, factors.get(i, _I), format="csr")
    return out


def hamiltonian(j1: float, j2: float, n: int, periodic: bool = False) -> sp.csr_matrix:
    """Build H(j1, j2) on n qubits as a sparse (2^n x 2^n) matrix.

    periodic=False (default, OPEN BC) reproduces the authors' DMRG-generated states; set
    periodic=True for the translationally-invariant ring.
    """
    if n < 3:
        raise ValueError("n >= 3 required for the 3-site cluster term")
    H = sp.csr_matrix((2**n, 2**n), dtype=complex)
    # bond ranges: under PBC every site hosts a term (indices wrap); under open BC,
    # XX needs j+1<n and the 3-site cluster needs 0<=j-1 and j+1<n (interior sites only).
    xx_sites = range(n) if periodic else range(n - 1)
    cluster_sites = range(n) if periodic else range(1, n - 1)

    for j in range(n):                                  # Σ_j Z_j
        H = H + _pauli_op({j: _Z}, n)
    for j in xx_sites:                                  # − j1 Σ_j X_j X_{j+1}
        H = H - j1 * _pauli_op({j: _X, (j + 1) % n: _X}, n)
    for j in cluster_sites:                             # − j2 Σ_j X_{j-1} Z_j X_{j+1}
        H = H - j2 * _pauli_op({(j - 1) % n: _X, j: _Z, (j + 1) % n: _X}, n)
    return H.tocsr()


@dataclass(frozen=True)
class GroundState:
    j1: float
    j2: float
    n: int
    energy: float           # lowest eigenvalue E0
    gap: float              # E1 - E0 (degeneracy indicator)
    state: np.ndarray       # shape (2^n,), complex, normalized, global phase fixed


def _fix_global_phase(psi: np.ndarray) -> np.ndarray:
    """Rotate so the largest-magnitude amplitude is real & positive (cache determinism)."""
    k = int(np.argmax(np.abs(psi)))
    phase = psi[k] / abs(psi[k]) if abs(psi[k]) > 0 else 1.0
    return psi / phase


def ground_state(j1: float, j2: float, n: int, periodic: bool = False) -> GroundState:
    """Exact-diagonalize H(j1, j2) and return its (phase-fixed) ground state.

    Computes the two lowest eigenpairs so the spectral `gap` flags (near-)degeneracy.
    """
    H = hamiltonian(j1, j2, n, periodic=periodic)
    vals, vecs = eigsh(H, k=2, which="SA")
    order = np.argsort(vals)
    vals, vecs = vals[order], vecs[:, order]
    psi = vecs[:, 0]
    psi = _fix_global_phase(psi / np.linalg.norm(psi))
    return GroundState(
        j1=float(j1), j2=float(j2), n=int(n),
        energy=float(vals[0]), gap=float(vals[1] - vals[0]), state=psi,
    )


def expectation(H: sp.spmatrix, psi: np.ndarray) -> float:
    """⟨psi|H|psi⟩ (real part); used to verify variationality in tests."""
    return float(np.real(np.vdot(psi, H @ psi)))
