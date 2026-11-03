#!/usr/bin/env python
"""Time-evolution demo: <Z_k(t)> on the long-range Ising chain.

Starts every qubit in |0>, evolves under lr_ising, and prints each qubit's
<Z_k(t)> three ways: exact, Trotter circuit (no noise), and Trotter circuit
under gate noise. Sites evolve differently because the end qubits have fewer
and weaker couplings than the middle ones -- that site dependence is the
signal QH-FEX uses to recover the coupling law.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qiskit.quantum_info import Statevector  # noqa: E402

from src.evolution import evolved_circuit, exact_evolved_state, noisy_state  # noqa: E402
from src.hamiltonian import build_hamiltonian, build_site_observables  # noqa: E402
from src.noise import NoiseConfig  # noqa: E402
from src.solvers import expectation  # noqa: E402

N = 4
PARAMS = {"J0": 1.0, "alpha": 1.3, "h": 1.0}
STATE = "0" * N
TIMES = np.linspace(0.0, 1.0, 6)
# Every step costs gates, and gates cost noise: 4 order-2 steps on 4 all-to-all
# qubits is ~96 CX. At p2=0.01 that wipes out most of the signal, so the demo
# uses p2=0.001 (roughly current trapped-ion two-qubit error). The fixed step
# count also runs at t=0 (as zero-angle gates), which is why noisy z < 1 there.
STEPS, ORDER = 4, 2
NOISE = NoiseConfig(p1=0.0001, p2=0.001)


def main() -> None:
    H = build_hamiltonian("lr_ising", N, **PARAMS)
    z_ops = [build_site_observables(N)[f"z_{k}"] for k in range(N)]

    print(f"lr_ising  n={N}  {PARAMS}  start=|{STATE}>")
    print(f"Trotter: {STEPS} steps, order {ORDER}   noise: p1={NOISE.p1}, p2={NOISE.p2}\n")
    header = "".join(f"{f'z_{k}':>8s}" for k in range(N))
    print(f"{'t':>5s}  {'method':8s}{header}")

    for t in TIMES:
        qc = evolved_circuit(STATE, H, t, STEPS, ORDER)
        states = {
            "exact": exact_evolved_state(STATE, H, t),
            "trotter": Statevector(qc),
            "noisy": noisy_state(qc, NOISE),
        }
        for i, (name, state) in enumerate(states.items()):
            values = "".join(f"{expectation(state, op):8.3f}" for op in z_ops)
            print(f"{t:5.2f}  {name:8s}{values}" if i == 0 else f"{'':5s}  {name:8s}{values}")
        print()


if __name__ == "__main__":
    main()
