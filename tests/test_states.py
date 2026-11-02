import pytest
from qiskit.quantum_info import Statevector

from src.hamiltonian import build_site_observables
from src.solvers import expectation
from src.states import STATE_LETTERS, product_state_circuit

# Expected (<X>, <Y>, <Z>) for each single-qubit state.
BLOCH = {
    "0": (0, 0, 1),
    "1": (0, 0, -1),
    "+": (1, 0, 0),
    "-": (-1, 0, 0),
    "r": (0, 1, 0),
    "l": (0, -1, 0),
}


@pytest.mark.parametrize("letter", STATE_LETTERS)
def test_single_qubit_bloch_vector(letter):
    # Each letter must land on the right Pauli eigenstate, including its sign.
    state = Statevector(product_state_circuit(letter + "0"))
    obs = build_site_observables(2)
    got = tuple(expectation(state, obs[f"{p}_0"]) for p in "xyz")
    assert got == pytest.approx(BLOCH[letter])


def test_matches_qiskit_from_label():
    # Our codes read left to right as qubit 0, 1, ...; Qiskit labels are reversed.
    code = "0+r1-l"
    ours = Statevector(product_state_circuit(code))
    assert ours.equiv(Statevector.from_label(code[::-1]))


def test_qubit_order():
    # Only qubit 2 is flipped, so only z_2 is -1.
    state = Statevector(product_state_circuit("0010"))
    obs = build_site_observables(4)
    assert [expectation(state, obs[f"z_{k}"]) for k in range(4)] == pytest.approx([1, 1, -1, 1])


def test_bad_codes():
    with pytest.raises(ValueError):
        product_state_circuit("")
    with pytest.raises(ValueError):
        product_state_circuit("0x1")
