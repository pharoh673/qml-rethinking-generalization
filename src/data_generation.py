"""Ground-state generation + disk caching, keyed to the authors' canonical coupling files.

Pipeline role
-------------
The authors release `(j1, j2)` coupling points and 4-class phase LABELS (adopted verbatim;
see src/phases.py), but their ground-state files are pickled TensorCircuit/Qibo objects
unusable by PennyLane. This module REGENERATES the ground states from those exact couplings
in native dense form (statevector, site i = wire i), via exact diagonalization with OPEN
boundary conditions — the convention shown (in tests/test_hamiltonian.py) to reproduce the
authors' states. Results are cached to data/cache/ keyed by a content hash so a sweep never
recomputes the same states.

n coverage: sparse ED (`eigsh`) is used for n <= 16 (2^16 = 65536-dim, tractable on CPU).
n = 32 requires DMRG/MPS (quimb) and is an optional extension — see README Phase 8.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import phases
from .hamiltonian import expectation, ground_state, hamiltonian

REPO_ROOT = Path(__file__).resolve().parent.parent
AUTHORS_ROOT = REPO_ROOT / "data" / "authors"
CACHE_ROOT = REPO_ROOT / "data" / "cache"

ED_MAX_N = 16          # sparse ED is practical up to here
TEST_SET_SIZE = 1000   # the shared i.i.d. test set the paper draws


@dataclass(frozen=True)
class Dataset:
    """Couplings + canonical labels + regenerated ground states for one experiment slice."""

    experiment: str
    n: int
    j1: np.ndarray          # (m,)
    j2: np.ndarray          # (m,)
    labels: np.ndarray      # (m,) int in {0,1,2,3}
    states: np.ndarray      # (m, 2^n) complex, unit-norm
    periodic: bool

    def __len__(self) -> int:
        return int(self.labels.shape[0])


# --------------------------------------------------------------------------------------
# Ground-state generation + caching
# --------------------------------------------------------------------------------------
def generate_ground_states(
    j1: np.ndarray, j2: np.ndarray, n: int, periodic: bool = False
) -> np.ndarray:
    """Exact-diagonalize H(j1_k, j2_k) for each k; return states as (m, 2^n) complex array."""
    if n > ED_MAX_N:
        raise NotImplementedError(
            f"n={n} exceeds ED_MAX_N={ED_MAX_N}; use the DMRG/MPS path (quimb) for n=32 "
            "(optional, README Phase 8)."
        )
    j1 = np.atleast_1d(np.asarray(j1, float))
    j2 = np.atleast_1d(np.asarray(j2, float))
    if j1.shape != j2.shape:
        raise ValueError(f"j1/j2 shape mismatch: {j1.shape} vs {j2.shape}")
    states = np.empty((j1.size, 2**n), dtype=complex)
    for k in range(j1.size):
        states[k] = ground_state(float(j1[k]), float(j2[k]), n, periodic=periodic).state
    return states


def _content_key(j1: np.ndarray, j2: np.ndarray, n: int, periodic: bool) -> str:
    h = hashlib.sha1()
    h.update(np.ascontiguousarray(j1, dtype=np.float64).tobytes())
    h.update(np.ascontiguousarray(j2, dtype=np.float64).tobytes())
    h.update(f"n={n};pbc={int(periodic)};v1".encode())
    return h.hexdigest()[:16]


def ground_states_cached(
    j1: np.ndarray, j2: np.ndarray, n: int, periodic: bool = False, cache_dir: Path = CACHE_ROOT
) -> np.ndarray:
    """Load ground states from cache if present (keyed by coupling content), else generate."""
    j1 = np.atleast_1d(np.asarray(j1, float))
    j2 = np.atleast_1d(np.asarray(j2, float))
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _content_key(j1, j2, n, periodic)
    path = cache_dir / f"gs_n{n}_{'pbc' if periodic else 'obc'}_{key}.npz"
    if path.exists():
        with np.load(path) as data:
            return data["states"]
    states = generate_ground_states(j1, j2, n, periodic=periodic)
    np.savez_compressed(path, states=states, j1=j1, j2=j2, n=n, periodic=periodic)
    return states


# --------------------------------------------------------------------------------------
# Indexing the authors' canonical files
# --------------------------------------------------------------------------------------
def authors_dir(experiment: str, n: int, root: Path = AUTHORS_ROOT) -> Path:
    d = root / experiment / f"{n}_qubits"
    if not d.is_dir():
        raise FileNotFoundError(f"missing authors dir {d} (run scripts/fetch_authors_data.py)")
    return d


def authors_paths(experiment: str, n: int, N: int | None = None, root: Path = AUTHORS_ROOT):
    """Return (j1_path, j2_path, labels_path). N=None -> the 1000-point test set."""
    d = authors_dir(experiment, n, root)
    if N is None:
        return (
            d / "J1coef_j1j2_1000_Test_Set",
            d / "J2coef_j1j2_1000_Test_Set",
            d / "LABELS_1000_Test_Set",
        )
    sub = d / f"{N}_training_data"
    return (sub / f"J1coef_j1j2_{N}", sub / f"J2coef_j1j2_{N}", sub / f"LABELS_{N}")


def load_authors_dataset(
    experiment: str, n: int, N: int | None = None, periodic: bool = False, root: Path = AUTHORS_ROOT
) -> Dataset:
    """Load canonical couplings+labels for (experiment, n, N) and attach regenerated states."""
    j1_p, j2_p, lab_p = authors_paths(experiment, n, N, root)
    pd = phases.load_dataset(j1_p, j2_p, lab_p)
    states = ground_states_cached(pd.j1, pd.j2, n, periodic=periodic)
    return Dataset(
        experiment=experiment, n=n, j1=pd.j1, j2=pd.j2, labels=pd.labels,
        states=states, periodic=periodic,
    )


def near_ground_relative_excess(j1: float, j2: float, n: int, state: np.ndarray, periodic: bool = False) -> float:
    """(⟨ψ|H|ψ⟩ − E0) / (Emax − E0): ~0 means `state` is a ground state of H(j1,j2)."""
    from scipy.sparse.linalg import eigsh

    H = hamiltonian(j1, j2, n, periodic=periodic)
    e_lo = float(eigsh(H, k=1, which="SA")[0][0])
    e_hi = float(eigsh(H, k=1, which="LA")[0][0])
    return (expectation(H, state) - e_lo) / (e_hi - e_lo)
