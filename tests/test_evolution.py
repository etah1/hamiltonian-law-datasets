import numpy as np
import pytest
from qiskit.quantum_info import Statevector, state_fidelity

from src.evolution import evolved_circuit, exact_evolved_state, noisy_state, trotter_circuit
from src.hamiltonian import build_hamiltonian, build_site_observables
from src.noise import NoiseConfig
from src.solvers import expectation


def _trotter_error(H, code, t, steps, order):
    """1 - fidelity between the Trotter state and the exact state."""
    approx = Statevector(evolved_circuit(code, H, t, steps, order))
    return 1 - state_fidelity(approx, exact_evolved_state(code, H, t))


def test_single_spin_precession_matches_analytic():
    # J0 = 0 leaves H = h (X_0 + X_1): each spin precesses on its own, and
    # starting from |0>, <Z(t)> = cos(2 h t). Checks the time convention exp(-iHt).
    h, t = 0.7, 0.9
    H = build_hamiltonian("lr_ising", 2, J0=0.0, h=h)
    state = Statevector(evolved_circuit("00", H, t, steps=1))
    z0 = expectation(state, build_site_observables(2)["z_0"])
    assert z0 == pytest.approx(np.cos(2 * h * t))


def test_commuting_hamiltonian_needs_one_step():
    # With h = 0 every term is ZZ, they all commute, so one step is exact.
    H = build_hamiltonian("lr_ising", 3, h=0.0)
    assert _trotter_error(H, "+r-", t=1.5, steps=1, order=1) == pytest.approx(0, abs=1e-10)


def test_trotter_converges_to_exact():
    H = build_hamiltonian("lr_ising", 4)
    assert _trotter_error(H, "0+r1", t=1.0, steps=50, order=2) < 1e-6


def test_more_steps_and_higher_order_reduce_error():
    # Also guards against the circuit secretly being exact (coarse would be 0).
    H = build_hamiltonian("lr_ising", 3)
    coarse = _trotter_error(H, "0+0", t=1.0, steps=4, order=1)
    assert coarse > 1e-4
    fine = _trotter_error(H, "0+0", t=1.0, steps=16, order=1)
    second = _trotter_error(H, "0+0", t=1.0, steps=4, order=2)
    assert fine < coarse
    assert second < coarse


def test_time_zero_is_the_initial_state():
    H = build_hamiltonian("lr_ising", 3)
    obs = build_site_observables(3)
    state = Statevector(evolved_circuit("1+0", H, t=0.0))
    assert expectation(state, obs["z_0"]) == pytest.approx(-1)
    assert expectation(state, obs["x_1"]) == pytest.approx(1)


def test_noiseless_config_matches_ideal_circuit():
    H = build_hamiltonian("lr_ising", 3)
    qc = evolved_circuit("0+r", H, t=0.8, steps=5)
    assert state_fidelity(noisy_state(qc, NoiseConfig()), Statevector(qc)) == pytest.approx(1)


def test_gate_noise_shrinks_signal():
    # Depolarizing noise pushes expectation values toward 0 but keeps a valid state.
    H = build_hamiltonian("lr_ising", 3)
    qc = evolved_circuit("000", H, t=0.3, steps=3)
    z = build_site_observables(3)["z_1"]
    ideal = expectation(Statevector(qc), z)
    rho = noisy_state(qc, NoiseConfig(p1=0.01, p2=0.05))
    assert rho.trace() == pytest.approx(1)
    assert abs(expectation(rho, z)) < abs(ideal)


def test_bad_arguments():
    H = build_hamiltonian("lr_ising", 3)
    with pytest.raises(ValueError):
        trotter_circuit(H, 1.0, steps=0)
    with pytest.raises(ValueError):
        trotter_circuit(H, 1.0, order=3)
    with pytest.raises(ValueError):
        evolved_circuit("00", H, 1.0)  # 2-qubit state, 3-qubit Hamiltonian
