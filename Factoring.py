import numpy as np
import multiprocessing as mp
from fractions import Fraction
from math import gcd
from random import randint
from qiskit import QuantumCircuit,  transpile
from qiskit.circuit import Gate
from qiskit_aer import AerSimulator
from qiskit.circuit.library import QFT, DraperQFTAdder
from qiskit.visualization import plot_histogram

def is_trivial_factor(N):
    if N % 2 == 0:
        return 2, N // 2
    p = 2
    while p * p <= N:
        k = 0
        x = N
        while x % p == 0:
            x //= p
            k += 1
        if x == 1 and k > 1:
            return p, N // p
        p += 1
    return None

def choose_a(N):
    while True:
        a = randint(2, N - 1)
        g = gcd(a, N)
        if g > 1:
            return a, g
        if g == 1:
            return a, None

def continued_fraction_order(measured_value, t, N):
    phase = measured_value / (2 ** t)
    frac = Fraction(phase).limit_denominator(N)
    return frac.denominator, frac

def shor_postprocess(a, N, r):
    if r % 2 != 0:
        return None
    x = pow(a, r // 2, N)
    if x == N - 1:
        return None
    p1 = gcd(x - 1, N)
    p2 = gcd(x + 1, N)
    if p1 in (1, N) or p2 in (1, N):
        return None
    return p1, p2

def build_modexp_unitary(a, N):
    dim = 1
    while dim < N:
        dim *= 2
    n_qubits = int(np.log2(dim))

    U = np.zeros((dim, dim), dtype=complex)

    for x in range(dim):
        if x < N:
            y = pow(a, x, N)
        else:
            # map unused basis states to themselves
            y = x
        U[y, x] = 1.0

    gate = Gate(name=f"U_a={a}_N={N}", num_qubits=n_qubits, params=[])
    gate._define([(QuantumCircuit(n_qubits).unitary(U, range(n_qubits)), range(n_qubits))])
    return gate, n_qubits

def order_finding_qpe(a, N, t=8, shots=2048):
    U_gate, n_work = safe_build_modexp(a, N)

    qc = QuantumCircuit(t + n_work, t)

    for q in range(t):
        qc.h(q)

    qc.x(t)

    for k in range(t):
        power = 2 ** k
        U_power = QuantumCircuit(n_work)
        for _ in range(power):
            U_power.append(U_gate, range(n_work))
        U_power_gate = U_power.to_gate().control(1)
        qc.append(U_power_gate, [k] + list(range(t, t + n_work)))

    qc.append(QFT(t, inverse=True, do_swaps=True), range(t))

    qc.measure(range(t), range(t))

    backend = AerSimulator()
    compiled = transpile(qc, backend)
    result = backend.run(compiled, shots=shots).result()
    counts = result.get_counts()

    bitstring = max(counts, key=counts.get)
    measured_value = int(bitstring, 2)

    r, frac = continued_fraction_order(measured_value, t, N)
    return r, frac, counts

def shor_factor(N, t=8, shots=2048):
    trivial = is_trivial_factor(N)
    if trivial is not None:
        return trivial

    a, lucky_gcd = choose_a(N)
    if lucky_gcd is not None:
        return lucky_gcd, N // lucky_gcd

    r, frac, counts = order_finding_qpe(a, N, t=t, shots=shots)
    print(f"Chosen a = {a}")
    print(f"Measured phase ≈ {frac} → candidate order r = {r}")

    factors = shor_postprocess(a, N, r)
    if factors is None:
        print("Post-processing failed, try again with a different a.")
        return None
    return factors

def modexp_worker(a, N, q):
    try:
        gate, n = build_modexp_unitary(a, N)
        q.put((gate, n, None))
    except Exception as e:
        q.put((None, None, e))


def safe_build_modexp(a, N):
    q = mp.Queue()
    p = mp.Process(target=modexp_worker, args=(a, N, q))
    p.start()
    p.join()

    gate, n, err = q.get()
    if err is not None:
        raise err
    return gate, n

if __name__ == "__main__":
    N = 15  # change this to 21, 33, etc. (keep it small)
    print(f"Factoring N = {N} with Shor's algorithm (small-scale)...")
    factors = shor_factor(N, t=8, shots=4096)
    print("Result:", factors)