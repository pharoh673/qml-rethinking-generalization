"""Readout, prediction rule, and loss — verbatim from the authors' main_code.py.

From the two output-wire expectations (z_a=⟨Z_a⟩, z_b=⟨Z_b⟩, zz=⟨Z_aZ_b⟩) the four
class probabilities are reconstructed exactly as in the authors' code:

    p00 = (1 + zz + z_a + z_b) / 4        (label 0, SPT)
    p01 = (1 - zz - z_a + z_b) / 4        (label 1, FM)
    p10 = (1 - zz + z_a - z_b) / 4        (label 2, AFM)
    p11 = (1 + zz - z_a - z_b) / 4        (label 3, trivial)

PREDICTION (Eq. 4): ŷ = argmin_b p_b   — the LOWEST-probability bitstring.
LOSS       (Eq. 5): ℓ = p_{true label}, averaged over the training set, MINIMIZED.
(These are the paper's counter-intuitive rules; do NOT substitute cross-entropy / argmax.)
"""
from __future__ import annotations

import numpy as np

N_CLASSES = 4


def readout_probs(expvals: np.ndarray) -> np.ndarray:
    """Map [z_a, z_b, zz] -> [p00, p01, p10, p11] (the authors' projector formulas)."""
    z_a, z_b, zz = expvals
    return np.array(
        [
            (1 + zz + z_a + z_b) / 4,
            (1 - zz - z_a + z_b) / 4,
            (1 - zz + z_a - z_b) / 4,
            (1 + zz - z_a - z_b) / 4,
        ],
        dtype=float,
    )


def predict_from_probs(probs: np.ndarray) -> int:
    """ŷ = argmin_b p_b (lowest-probability bitstring)."""
    return int(np.argmin(probs))


def sample_loss(expvals: np.ndarray, label: int, equalize: bool = False) -> float:
    """Per-sample loss = probability of the TRUE label (minimized).

    `equalize=True` adds the authors' optional term (commented in their main_code.py): the
    mean squared pairwise difference among the THREE incorrect-class probabilities. This
    pushes the incorrect classes together so the true class is reliably the argmin — required
    in practice to memorize the harder configs (the plain p_true objective plateaus, leaving
    an incorrect class below the true one). See README "Loss: equalize term".
    """
    p = readout_probs(expvals)
    base = float(p[label])
    if not equalize:
        return base
    others = [p[c] for c in range(N_CLASSES) if c != label]
    a, b, c = others
    return base + ((a - b) ** 2 + (a - c) ** 2 + (b - c) ** 2) / 3.0


def dataset_loss(qcnn, params: np.ndarray, states: np.ndarray, labels: np.ndarray,
                 equalize: bool = False) -> float:
    """Mean per-sample loss over the dataset — the objective handed to CMA-ES."""
    total = 0.0
    for state, label in zip(states, labels):
        total += sample_loss(qcnn(params, state), int(label), equalize=equalize)
    return total / len(labels)


def predictions(qcnn, params: np.ndarray, states: np.ndarray) -> np.ndarray:
    """argmin predictions for each state."""
    return np.array(
        [predict_from_probs(readout_probs(qcnn(params, state))) for state in states], dtype=int
    )


def accuracy(qcnn, params: np.ndarray, states: np.ndarray, labels: np.ndarray) -> float:
    """Fraction correctly classified under the argmin rule (in [0,1])."""
    preds = predictions(qcnn, params, states)
    return float(np.mean(preds == np.asarray(labels, int)))


def error_probability(qcnn, params: np.ndarray, states: np.ndarray, labels: np.ndarray) -> float:
    """Probability of error = 1 - accuracy (the paper's reported metric)."""
    return 1.0 - accuracy(qcnn, params, states, labels)
