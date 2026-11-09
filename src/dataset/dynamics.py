"""Time-evolution datasets for AutoSR: neutral CSV + hidden ground-truth JSON.

One CSV row per measurement::

    n, setting_id, state, observable, t, value, stderr

``state`` is a product-state code (see ``src.states``), ``observable`` a site
label like ``z_2`` or ``xx_0_1`` (see ``build_site_observables``). Nothing in
the CSV or its filename names the model or its parameters; those go in
``truth.json``, which AutoSR must never be given.

Pipeline per (n, setting, state, t): prepare the state, Trotter-evolve,
optionally run through the gate-noise model, then sample every observable
with finite-shot Binomial noise.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from qiskit.quantum_info import Statevector

from src.evolution import evolved_circuit, noisy_state
from src.hamiltonian import build_hamiltonian, build_site_observables, model_defaults
from src.noise import NoiseConfig
from src.solvers import expectation
from src.states import STATE_LETTERS, product_state_circuit

CSV_COLUMNS = ["n", "setting_id", "state", "observable", "t", "value", "stderr"]


# --------------------------------------------------------------------------- #
# Shot noise
# --------------------------------------------------------------------------- #
def shot_noise(m: float, shots: int, rng: np.random.Generator) -> tuple[float, float]:
    """Sample a Pauli expectation ``m`` with ``shots`` measurements.

    Each shot gives +1 with probability (1+m)/2, so k ~ Binomial(shots, (1+m)/2)
    and the estimate is m_hat = 2k/shots - 1. The standard error is
    sqrt((1 - m_hat^2)/shots), floored at 1/shots: when every shot agrees
    (m_hat = +-1) the formula gives 0, which would make that point infinitely
    trusted in a chi-squared fit.
    """
    if shots < 1:
        raise ValueError("shots must be >= 1")
    p = min(max((1.0 + m) / 2.0, 0.0), 1.0)  # clip float round-off
    k = rng.binomial(shots, p)
    m_hat = 2.0 * k / shots - 1.0
    stderr = max(math.sqrt((1.0 - m_hat**2) / shots), 1.0 / shots)
    return m_hat, stderr


# --------------------------------------------------------------------------- #
# Experiment design
# --------------------------------------------------------------------------- #
def random_states(n: int, count: int, rng: np.random.Generator) -> list[str]:
    """``count`` distinct random product-state codes on ``n`` qubits.

    Random letters mix Z, X and Y eigenstates, so ZZ couplings show up at
    first order (they are invisible in <Z> from |0...0>).
    """
    count = min(count, len(STATE_LETTERS) ** n)
    out: list[str] = []
    while len(out) < count:
        code = "".join(rng.choice(STATE_LETTERS, size=n))
        if code not in out:
            out.append(code)
    return out


def select_observables(n: int, max_pair_distance: int = 1) -> dict:
    """Single-site X, Y, Z on every qubit, plus XX, YY, ZZ on pairs with
    |i-j| <= ``max_pair_distance`` (1 = nearest neighbours, as in the plan)."""
    keep = {}
    for name, op in build_site_observables(n).items():
        sites = [int(s) for s in name.split("_")[1:]]
        if len(sites) == 1 or sites[1] - sites[0] <= max_pair_distance:
            keep[name] = op
    return keep


# --------------------------------------------------------------------------- #
# Dataset generation
# --------------------------------------------------------------------------- #
def generate_dynamics_dataset(
    model: str,
    sizes: Sequence[int],
    times: Sequence[float],
    *,
    settings: Sequence[Mapping[str, float]] = ({},),
    states_per_size: int = 10,
    shots: int | None = 1000,
    noise: NoiseConfig | None = None,
    dt: float = 0.1,
    order: int = 2,
    max_pair_distance: int = 1,
    seed: int = 0,
) -> tuple[pd.DataFrame, dict]:
    """Simulate the experiments and return (data, truth).

    ``settings`` is a list of parameter overrides (one ``setting_id`` each),
    e.g. ``[{"alpha": 1.0}, {"alpha": 1.3}]``. Each time t uses
    ceil(t/dt) Trotter steps, so the step size stays at most ``dt``; t = 0
    is just the prepared state with no evolution gates. ``shots=None``
    records exact expectations with stderr 0 (for testing only).
    """
    rng = np.random.default_rng(seed)
    noise = noise or NoiseConfig()
    times = [round(float(t), 10) for t in times]  # 0.30000000000000004 -> 0.3
    defaults = model_defaults(model)

    rows = []
    for n in sizes:
        observables = select_observables(n, max_pair_distance)
        states = random_states(n, states_per_size, rng)
        for setting_id, overrides in enumerate(settings):
            H = build_hamiltonian(model, n, **{**defaults, **overrides})
            for code in states:
                for t in times:
                    steps = math.ceil(t / dt - 1e-9)
                    qc = (
                        evolved_circuit(code, H, t, steps, order)
                        if steps > 0
                        else product_state_circuit(code)
                    )
                    state = Statevector(qc) if noise.is_noiseless else noisy_state(qc, noise)
                    for obs_name, op in observables.items():
                        m = expectation(state, op)
                        value, err = (m, 0.0) if shots is None else shot_noise(m, shots, rng)
                        rows.append((n, setting_id, code, obs_name, t, value, err))

    data = pd.DataFrame(rows, columns=CSV_COLUMNS)
    truth = {
        "model": model,
        "settings": [{**defaults, **o} for o in settings],
        "sizes": list(sizes),
        "times": times,
        "states_per_size": states_per_size,
        "shots": shots,
        "noise": {k: getattr(noise, k) for k in ("p1", "p2", "t1", "t2")},
        "trotter": {"dt": dt, "order": order},
        "max_pair_distance": max_pair_distance,
        "seed": seed,
    }
    return data, truth


def save_dynamics_dataset(data: pd.DataFrame, truth: dict, out_dir: str | Path) -> Path:
    """Write ``data.csv`` (for AutoSR) and ``truth.json`` (never give to AutoSR)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    data.to_csv(out / "data.csv", index=False)
    (out / "truth.json").write_text(json.dumps(truth, indent=2))
    return out
