"""The three randomization protocols of Fig. 3 (verified against the authors' data).

(a) Random labels      — labels replaced by i.i.d. Uniform{0,1,2,3} (Zhang-et-al. style).
(b) Partial corruption — exactly `c` of N labels reassigned to a DIFFERENT class
                         (verified: the authors' LABELS_{N}_c differ from the truth in
                         exactly c positions, never colliding with the true label).
(c) Random states      — each state's amplitudes resampled i.i.d. from N(μ_ψ, σ_ψ) fitted to
                         THAT state's own (real) amplitudes, then renormalized. NOT Haar.
                         (verified: the authors' random-states inputs are real, unit-norm,
                         std ≈ the true state's, and uncorrelated with the true state.)

The cluster Hamiltonian is real, so its ground states are real-valued -> the Gaussian fit is
on real amplitudes. A complex fallback (fit real & imag separately) is included for safety.

All functions take an explicit numpy Generator for reproducibility. For exact paper
reproduction we use the authors' shipped LABELS files directly; these generators are for
fresh/extended experiments and are validated to match the authors' protocols.
"""
from __future__ import annotations

import numpy as np

N_CLASSES = 4


def random_labels(m: int, rng: np.random.Generator, n_classes: int = N_CLASSES) -> np.ndarray:
    """m i.i.d. labels Uniform{0,...,n_classes-1} (full randomization)."""
    return rng.integers(0, n_classes, size=m)


def corrupt_labels(
    labels: np.ndarray, n_corrupt: int, rng: np.random.Generator, n_classes: int = N_CLASSES
) -> np.ndarray:
    """Reassign exactly `n_corrupt` randomly-chosen labels, each to a DIFFERENT class."""
    labels = np.asarray(labels).astype(int).copy()
    m = labels.size
    if not 0 <= n_corrupt <= m:
        raise ValueError(f"n_corrupt={n_corrupt} out of range [0, {m}]")
    idx = rng.choice(m, size=n_corrupt, replace=False)
    for i in idx:
        others = [c for c in range(n_classes) if c != labels[i]]
        labels[i] = rng.choice(others)
    return labels


def corrupt_labels_by_ratio(
    labels: np.ndarray, ratio: float, rng: np.random.Generator, n_classes: int = N_CLASSES
) -> np.ndarray:
    """Corrupt a fraction `ratio` (rounded to nearest count) of labels. ratio in [0,1]."""
    if not 0.0 <= ratio <= 1.0:
        raise ValueError(f"ratio={ratio} must be in [0,1]")
    n_corrupt = int(round(ratio * np.asarray(labels).size))
    return corrupt_labels(labels, n_corrupt, rng, n_classes)


def gaussian_resample_state(psi: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Resample one state's amplitudes i.i.d. from N(μ_ψ, σ_ψ), then renormalize.

    Real ground states -> fit on real amplitudes. Complex states (fallback) -> fit real and
    imaginary parts independently.
    """
    a = np.asarray(psi)
    if np.max(np.abs(a.imag)) > 1e-9:
        re = rng.normal(a.real.mean(), a.real.std(), size=a.shape)
        im = rng.normal(a.imag.mean(), a.imag.std(), size=a.shape)
        v = re + 1j * im
    else:
        r = a.real
        v = rng.normal(r.mean(), r.std(), size=r.shape).astype(complex)
    return v / np.linalg.norm(v)


def gaussian_resample_states(states: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply gaussian_resample_state independently to each row of `states`."""
    return np.stack([gaussian_resample_state(s, rng) for s in states])
