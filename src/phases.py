"""Phase labels for the generalized cluster Hamiltonian (4-class SPT classification).

H = Σ_j ( Z_j − j1·X_j X_{j+1} − j2·X_{j−1} Z_j X_{j+1} ),   j1, j2 ~ Uniform[-4, 4]^2

Four symmetry-protected phases (Verresen–Moessner–Pollmann, PRB 96, 165124):
    0 = SPT (symmetry-protected topological)
    1 = FM  (ferromagnetic)
    2 = AFM (antiferromagnetic)
    3 = trivial

CANONICAL LABELS
----------------
The authors' labeling *function* is NOT in their released code, and the paper defers
the boundaries to Verresen et al. We verified that the free-fermion winding / gap-closing
diagram derivable from the paper reproduces only ~62% of the authors' published labels
(their boundaries are curved). We therefore adopt the authors' released `(j1, j2) -> label`
files VERBATIM as the canonical dataset (see scripts/fetch_authors_data.py), loaded here.

`approximate_phase_label` is provided ONLY as a clearly-flagged placeholder for generating
*new* coupling points; it must NOT be used to reproduce the paper's figures.
See README -> "Open items / ZENODO-CHECK".
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# --- canonical class encoding (matches the authors' LABELS_* files & loss bitstrings) ---
SPT, FM, AFM, TRIVIAL = 0, 1, 2, 3
N_CLASSES = 4
PHASE_NAMES: dict[int, str] = {SPT: "SPT", FM: "FM", AFM: "AFM", TRIVIAL: "trivial"}

# label index -> 2-qubit output bitstring used by the QCNN readout (Eq. 4):
#   0->00, 1->01, 2->10, 3->11   (verified against the authors' main_code.py)
LABEL_TO_BITSTRING: dict[int, str] = {0: "00", 1: "01", 2: "10", 3: "11"}


@dataclass(frozen=True)
class PhaseDataset:
    """A set of coupling points with their canonical phase labels."""

    j1: np.ndarray  # shape (m,)
    j2: np.ndarray  # shape (m,)
    labels: np.ndarray  # shape (m,), int in {0,1,2,3}

    def __post_init__(self) -> None:
        if not (self.j1.shape == self.j2.shape == self.labels.shape):
            raise ValueError(
                f"shape mismatch: j1{self.j1.shape} j2{self.j2.shape} labels{self.labels.shape}"
            )
        validate_labels(self.labels)

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def summary(self) -> dict:
        return summarize(self.j1, self.j2, self.labels)


# --------------------------------------------------------------------------------------
# Loading the authors' canonical files
# --------------------------------------------------------------------------------------
def _load_row(path: str | Path) -> np.ndarray:
    """Load a whitespace-separated file (the authors store each vector on one line)."""
    arr = np.loadtxt(path)
    return np.atleast_1d(arr).ravel()


def load_couplings(j1_path: str | Path, j2_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    return _load_row(j1_path), _load_row(j2_path)


def load_labels(labels_path: str | Path) -> np.ndarray:
    return _load_row(labels_path).astype(int)


def load_dataset(
    j1_path: str | Path, j2_path: str | Path, labels_path: str | Path
) -> PhaseDataset:
    """Load an authors' (J1coef, J2coef, LABELS) file triple into a PhaseDataset."""
    j1, j2 = load_couplings(j1_path, j2_path)
    labels = load_labels(labels_path)
    return PhaseDataset(j1=j1, j2=j2, labels=labels)


def find_test_set(authors_root: str | Path) -> PhaseDataset | None:
    """Locate the shared 1000-point test set in a fetched authors' tree, if present.

    File triple: partially_corrupted_labels/{J1coef,J2coef,LABELS}_*1000_Test_Set
    """
    root = Path(authors_root)
    j1 = next(root.rglob("J1coef_*1000_Test_Set"), None)
    j2 = next(root.rglob("J2coef_*1000_Test_Set"), None)
    lb = next(root.rglob("LABELS_*1000_Test_Set"), None)
    if not (j1 and j2 and lb):
        return None
    return load_dataset(j1, j2, lb)


# --------------------------------------------------------------------------------------
# Validation / summary helpers
# --------------------------------------------------------------------------------------
def validate_labels(labels: np.ndarray) -> None:
    """Raise ValueError if any label is outside {0,1,2,3}."""
    bad = np.setdiff1d(np.unique(labels), np.arange(N_CLASSES))
    if bad.size:
        raise ValueError(f"labels outside 0..{N_CLASSES - 1}: {bad.tolist()}")


def summarize(j1: np.ndarray, j2: np.ndarray, labels: np.ndarray) -> dict:
    counts = {PHASE_NAMES[c]: int((labels == c).sum()) for c in range(N_CLASSES)}
    return {
        "n_points": int(labels.shape[0]),
        "counts": counts,
        "j1_range": (float(j1.min()), float(j1.max())),
        "j2_range": (float(j2.min()), float(j2.max())),
    }


# --------------------------------------------------------------------------------------
# APPROXIMATE analytic map — NOT canonical (do not use for paper reproduction)
# --------------------------------------------------------------------------------------
def approximate_phase_label(j1: float, j2: float) -> int:
    """Free-fermion winding + symmetry-sign heuristic.

    WARNING: reproduces only ~62% of the authors' published labels (their boundaries are
    curved and not specified in the paper). Provided solely as a placeholder for generating
    new coupling points if/when an exact map is reconstructed. Never use this to reproduce
    the paper's figures — use the authors' canonical LABELS_* files via load_dataset().

    Heuristic: winding number of the BDI symbol f(z) = 1 - j1 z - j2 z^2 (#roots inside the
    unit circle) gives 0/1/2; the w=1 (ordered) sector is split into FM (j1>0) / AFM (j1<=0).
    """
    coeffs = [j2, j1, -1.0] if abs(j2) > 1e-12 else [j1, -1.0]
    roots = np.roots(coeffs)
    w = int(np.sum(np.abs(roots) < 1.0))
    if w == 0:
        return TRIVIAL
    if w >= 2:
        return SPT
    return FM if j1 > 0 else AFM
