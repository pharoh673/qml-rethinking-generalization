import numpy as np
import tensorcircuit as tc
tc.set_backend("numpy")
np.set_printoptions(precision=6, suppress=True, linewidth=140)

th, al, ph = 0.7, 1.3, 0.4  # arbitrary distinct test angles

def unitary(nq, apply):
    c = tc.Circuit(nq); apply(c)
    try:
        return np.asarray(c.matrix())
    except Exception:
        U = np.zeros((2**nq, 2**nq), complex)
        for b in range(2**nq):
            c2 = tc.Circuit(nq);
            for q in range(nq):
                if (b >> (nq-1-q)) & 1: c2.x(q)
            apply(c2); U[:, b] = np.asarray(c2.state()).ravel()
        return U

print("=== _zz_matrix ===")
print(np.asarray(tc.gates._zz_matrix))

print("\n=== r(theta,alpha,phi) single-qubit ===")
Ur = unitary(1, lambda c: c.r(0, theta=th, alpha=al, phi=ph))
print(Ur)

print("\n=== exp1(0,1, theta, _zz_matrix) ===")
Uzz = unitary(2, lambda c: c.exp1(0, 1, theta=th, unitary=tc.gates._zz_matrix))
print(Uzz)
# compare to candidate PennyLane forms exp(-i t/2 ZZ) and exp(i t ZZ)
ZZ = np.kron([[1,0],[0,-1]], [[1,0],[0,-1]])
from scipy.linalg import expm
print("exp(i*th*ZZ) close?", np.allclose(Uzz, expm(1j*th*ZZ)))
print("exp(-i*th*ZZ) close?", np.allclose(Uzz, expm(-1j*th*ZZ)))
print("exp(-i*th/2*ZZ)(IsingZZ) close?", np.allclose(Uzz, expm(-1j*th/2*ZZ)))

print("\n=== cr(0,1, theta,alpha,phi) controlled-r ===")
Ucr = unitary(2, lambda c: c.cr(0, 1, theta=th, alpha=al, phi=ph))
print(Ucr)
# is it block-diag(I2, r) with control on qubit 0 (|1> controls)?
print("top-left 2x2 == I?", np.allclose(Ucr[:2,:2], np.eye(2)))
print("bottom-right 2x2 ==\n", Ucr[2:,2:])
print("matches r(theta,alpha,phi)?", np.allclose(Ucr[2:,2:], Ur))
