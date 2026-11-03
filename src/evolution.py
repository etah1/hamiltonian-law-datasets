"""Time evolution: product state -> exp(-iHt) -> state, ideal or noisy.

Three ways to get the evolved state, from most to least idealised::

    exact_evolved_state   exact exp(-iHt) on the statevector (reference answer)
    Statevector(evolved_circuit(...))   Trotterized circuit, no gate noise
    noisy_state(evolved_circuit(...), noise)   same circuit through the Aer noise model

The Trotter circuit is what would run on hardware. Comparing it with the exact
state measures Trotter error; ``noisy_state`` adds gate noise on top.
"""
from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import DensityMatrix, SparsePauliOp, Statevector
from qiskit.synthesis import LieTrotter, SuzukiTrotter
from qiskit_aer import AerSimulator
from scipy.sparse.linalg import expm_multiply

from src.noise import BASIS_GATES, NoiseConfig, build_noise_model
from src.states import product_state_circuit


def trotter_circuit(H: SparsePauliOp, t: float, steps: int = 10, order: int = 1) -> QuantumCircuit:
    """Circuit approximating exp(-iHt) with ``steps`` Trotter steps.

    ``order=1`` is plain Lie-Trotter (error ~ t^2/steps); ``order=2`` is the
    symmetric Suzuki formula (error ~ t^3/steps^2), at roughly twice the gates.
    """
    if steps < 1:
        raise ValueError("steps must be >= 1")
    if order == 1:
        synthesis = LieTrotter(reps=steps)
    elif order == 2:
        synthesis = SuzukiTrotter(order=2, reps=steps)
    else:
        raise ValueError("order must be 1 or 2")
    # Synthesize into explicit gates. Appending the PauliEvolutionGate itself is
    # a trap: Statevector simulates it with the exact exponential, so the "Trotter"
    # circuit would silently have no Trotter error.
    qc = synthesis.synthesize(PauliEvolutionGate(H, time=t))
    qc.name = f"trotter-t{t:g}"
    return qc


def evolved_circuit(
    code: str, H: SparsePauliOp, t: float, steps: int = 10, order: int = 1
) -> QuantumCircuit:
    """Prepare product state ``code`` (see ``src.states``), then Trotter-evolve to time t."""
    if len(code) != H.num_qubits:
        raise ValueError(f"State code has {len(code)} qubits, Hamiltonian has {H.num_qubits}.")
    qc = product_state_circuit(code)
    qc.compose(trotter_circuit(H, t, steps, order), inplace=True)
    return qc


def exact_evolved_state(code: str, H: SparsePauliOp, t: float) -> Statevector:
    """Exact exp(-iHt)|code>, no Trotter error. Dense in 2^n, so small n only."""
    psi0 = Statevector(product_state_circuit(code)).data
    psi_t = expm_multiply(-1j * t * H.to_matrix(sparse=True), psi0)
    return Statevector(psi_t)


def noisy_state(circuit: QuantumCircuit, noise: NoiseConfig | None) -> DensityMatrix:
    """Run ``circuit`` through the gate-noise model (same as the VQE pipeline).

    Returns the exact noisy density matrix (no shot noise). With no noise this
    is just the ideal state as a density matrix.
    """
    model = build_noise_model(noise)
    if model is None:
        return DensityMatrix(Statevector(circuit))
    # Decompose the evolution gate into the basis the noise model acts on.
    qc = transpile(circuit, basis_gates=BASIS_GATES, optimization_level=0)
    qc.save_density_matrix()
    result = AerSimulator(method="density_matrix", noise_model=model).run(qc).result()
    return DensityMatrix(np.asarray(result.data(0)["density_matrix"]))
