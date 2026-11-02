"""Product-state preparation circuits for time-evolution experiments.

Each qubit starts in one of six single-qubit Pauli eigenstates, named with
Qiskit's ``Statevector.from_label`` letters::

    0  |0>     Z = +1          +  |+>     X = +1          r  |+i>    Y = +1
    1  |1>     Z = -1          -  |->     X = -1          l  |-i>    Y = -1

A state code is a string with one letter per qubit, read left to right as
qubit 0, 1, 2, ... So ``"0+r"`` puts qubit 0 in |0>, qubit 1 in |+>, and
qubit 2 in |+i>. Note this is the reverse of Qiskit's own label order,
where the rightmost character is qubit 0.
"""
from __future__ import annotations

from qiskit import QuantumCircuit

# Gates that take |0> to each state, applied in order.
_PREP_GATES: dict[str, tuple[str, ...]] = {
    "0": (),
    "1": ("x",),
    "+": ("h",),
    "-": ("x", "h"),
    "r": ("h", "s"),
    "l": ("h", "sdg"),
}

STATE_LETTERS = tuple(_PREP_GATES)


def product_state_circuit(code: str) -> QuantumCircuit:
    """Circuit preparing the product state ``code`` from |0...0>.

    Returns a circuit (not a statevector) so time evolution and measurement
    can be appended to it and the whole thing run through the noise model.
    """
    if not code:
        raise ValueError("State code must have at least one qubit.")
    bad = sorted(set(code) - set(STATE_LETTERS))
    if bad:
        raise ValueError(f"Unknown state letter(s) {bad}. Use {''.join(STATE_LETTERS)}.")
    qc = QuantumCircuit(len(code), name=f"prep-{code}")
    for q, letter in enumerate(code):
        for gate in _PREP_GATES[letter]:
            getattr(qc, gate)(q)
    return qc
