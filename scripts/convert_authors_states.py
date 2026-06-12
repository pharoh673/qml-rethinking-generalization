#!/usr/bin/env python3
"""One-time: densify the authors' TensorCircuit ground states for amplitude cross-checks.

The authors store their n=8 ground states as PICKLED TensorCircuit MPS objects, which can
only be read with `tensorcircuit` installed and are not directly usable by PennyLane. This
script converts the n=8 real-labels TEST set (1000 ground states) into a plain complex
ndarray of shape (1000, 256) and saves it to data/authors_dense/, so that
tests/test_hamiltonian.py can cross-check our ED states WITHOUT depending on tensorcircuit.

`tensorcircuit` is intentionally NOT in requirements.txt (it conflicts with our PennyLane
0.45 pin). Run this once in a throwaway environment, e.g.:

    docker run --rm -v "<repo>:/app" -w /app python:3.11-slim \
        sh -c "pip install -q tensorcircuit numpy && python scripts/convert_authors_states.py"

Output (data/authors_dense/) is gitignored and only needed to ENABLE the cross-check test.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
AUTHORS = REPO_ROOT / "data" / "authors"
OUT = REPO_ROOT / "data" / "authors_dense"

# (label-experiment dir, n, packed-states file, j1 file, j2 file, labels file, out name)
TARGETS = [
    (
        "real_labels/8_qubits",
        8,
        "test_set_1000examples.npy",
        "J1coef_j1j2_1000_Test_Set",
        "J2coef_j1j2_1000_Test_Set",
        "LABELS_1000_Test_Set",
        "real_labels_8q_testset.npz",
    ),
]


def _densify(packed, n: int) -> np.ndarray:
    import tensorcircuit as tc

    tc.set_backend("numpy")
    states = []
    for elem in packed:
        c = tc.Circuit(n, mps_inputs=elem)
        states.append(np.asarray(c.state()).ravel().astype(np.complex128))
    return np.stack(states, axis=0)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for subdir, n, states_f, j1_f, j2_f, lab_f, out_name in TARGETS:
        d = AUTHORS / subdir
        packed = np.load(d / states_f, allow_pickle=True)
        dense = _densify(packed, n)
        norms = np.linalg.norm(dense, axis=1)
        j1 = np.atleast_1d(np.loadtxt(d / j1_f)).ravel()
        j2 = np.atleast_1d(np.loadtxt(d / j2_f)).ravel()
        labels = np.atleast_1d(np.loadtxt(d / lab_f)).ravel().astype(int)
        np.savez_compressed(OUT / out_name, states=dense, j1=j1, j2=j2, labels=labels)
        print(
            f"[convert] {subdir}: {dense.shape[0]} states (dim {dense.shape[1]}), "
            f"norm in [{norms.min():.4f}, {norms.max():.4f}] -> {OUT / out_name}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
