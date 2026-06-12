"""Translation-invariant QCNN (Cong–Choi–Lukin / Caro et al.), ported to PennyLane.

Ported VERBATIM from the authors' TensorCircuit `main_code.py`, with the exact gate
unitaries extracted from TensorCircuit (see scripts/extract_tc_gates.py / README):

    conv(q1,q2)  [13 params]:  r(q1)·r(q2)·exp(-iθ ZZ)(q1,q2)·r(q1)·r(q2)
    pool(q1,q2)  [ 3 params]:  controlled-r  (control q1, target q2; coherent, no measurement)

    r(θ,α,φ)   = exp(-i θ (sinα cosφ·X + sinα sinφ·Y + cosα·Z))      [tc.gates.r]
    exp(-iθ ZZ) = qml.IsingZZ(2θ)                                     [tc exp1 + _zz_matrix]
    cr(θ,α,φ)  = block_diag(I2, r(θ,α,φ))                             [tc.gates.cr]

Architecture: start with all n wires active; each layer applies a translation-invariant
conv (two rounds of nearest-neighbour pairs on the active ring, parameters shared) then a
pool that keeps every other wire — halving the active set until 2 wires remain, where a
final conv is applied. Depth = log2(n). Parameter counts: n=8->45, n=16->61, n=32->77.

Currently supports n a power of two (covers Fig-3a n in {8,16,32} and random-states n=8).
The non-power-of-two random-states sizes (n=10,12) use a different reduction and are added
later (their layout is fetched from the authors' n=10/12 main_code.py).
"""
from __future__ import annotations

import numpy as np
import pennylane as qml

# Pauli matrices for building the r gate
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)
_I2 = np.eye(2, dtype=complex)

CONV_PARAMS = 13
POOL_PARAMS = 3


def _is_power_of_two(n: int) -> bool:
    return n >= 2 and (n & (n - 1)) == 0


def num_params(n: int) -> int:
    """Total trainable parameters for the n-qubit QCNN (13 per conv, 3 per pool)."""
    if not _is_power_of_two(n):
        raise NotImplementedError(f"n={n} must be a power of two (got non-power-of-two)")
    layers = int(np.log2(n))                       # number of conv layers
    return CONV_PARAMS * layers + POOL_PARAMS * (layers - 1)


def r_matrix(theta: float, alpha: float, phi: float) -> np.ndarray:
    """tc.gates.r unitary: exp(-i θ n·σ), n = (sinα cosφ, sinα sinφ, cosα)."""
    nx = np.sin(alpha) * np.cos(phi)
    ny = np.sin(alpha) * np.sin(phi)
    nz = np.cos(alpha)
    return np.cos(theta) * _I2 - 1j * np.sin(theta) * (nx * _X + ny * _Y + nz * _Z)


def _cr_matrix(theta: float, alpha: float, phi: float) -> np.ndarray:
    """tc.gates.cr: block_diag(I2, r) — control on q1's |1>, target q2 (q1 = MSB)."""
    U = np.eye(4, dtype=complex)
    U[2:, 2:] = r_matrix(theta, alpha, phi)
    return U


def _apply_conv(p: np.ndarray, q1: int, q2: int) -> None:
    qml.QubitUnitary(r_matrix(p[0], p[1], p[2]), wires=q1)
    qml.QubitUnitary(r_matrix(p[3], p[4], p[5]), wires=q2)
    qml.IsingZZ(2.0 * p[6], wires=[q1, q2])
    qml.QubitUnitary(r_matrix(p[7], p[8], p[9]), wires=q1)
    qml.QubitUnitary(r_matrix(p[10], p[11], p[12]), wires=q2)


def _apply_pool(p: np.ndarray, q1: int, q2: int) -> None:
    qml.QubitUnitary(_cr_matrix(p[0], p[1], p[2]), wires=[q1, q2])


def output_wires(n: int) -> tuple[int, int]:
    """The two wires measured at the end (e.g. (3,7) for n=8, (15,31) for n=32)."""
    active = list(range(n))
    while len(active) > 2:
        active = active[1::2]
    return active[0], active[1]


def _build_circuit(params: np.ndarray, n: int) -> None:
    """Apply the QCNN gates (no state prep / measurement) on wires 0..n-1."""
    active = list(range(n))
    off = 0
    while len(active) > 2:
        conv = params[off : off + CONV_PARAMS]
        m = len(active)
        for i in range(0, m, 2):                              # round A: (a0,a1),(a2,a3),...
            _apply_conv(conv, active[i], active[i + 1])
        for i in range(1, m, 2):                              # round B: (a1,a2),...,(a_{m-1},a0)
            _apply_conv(conv, active[i], active[(i + 1) % m])
        pool = params[off + CONV_PARAMS : off + CONV_PARAMS + POOL_PARAMS]
        for i in range(0, m, 2):                              # pool: control a_even, keep a_odd
            _apply_pool(pool, active[i], active[i + 1])
        active = active[1::2]
        off += CONV_PARAMS + POOL_PARAMS
    _apply_conv(params[off : off + CONV_PARAMS], active[0], active[1])  # final conv


def build_qcnn(n: int, device: str = "lightning.qubit"):
    """Return `qcnn(params, state) -> [⟨Z_a⟩, ⟨Z_b⟩, ⟨Z_aZ_b⟩]` on the 2 output wires.

    `state` is a length-2^n complex statevector (site i = wire i). Gradient-free: evaluated
    forward only (CMA-ES), so the default numpy interface is used.
    """
    if not _is_power_of_two(n):
        raise NotImplementedError(f"n={n} must be a power of two")
    a, b = output_wires(n)
    dev = qml.device(device, wires=n)

    @qml.qnode(dev, interface=None)
    def circuit(params, state):
        qml.StatePrep(state, wires=range(n))
        _build_circuit(params, n)
        return [
            qml.expval(qml.PauliZ(a)),
            qml.expval(qml.PauliZ(b)),
            qml.expval(qml.PauliZ(a) @ qml.PauliZ(b)),
        ]

    def qcnn(params: np.ndarray, state: np.ndarray) -> np.ndarray:
        return np.asarray(circuit(np.asarray(params, float), np.asarray(state, complex)), float)

    qcnn.n = n
    qcnn.output_wires = (a, b)
    qcnn.num_params = num_params(n)
    return qcnn
