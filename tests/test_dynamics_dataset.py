import json

import numpy as np
import pytest
from qiskit.quantum_info import Statevector

from src.dataset import generate_dynamics_dataset, save_dynamics_dataset, shot_noise
from src.dataset.dynamics import CSV_COLUMNS, random_states, select_observables
from src.evolution import exact_evolved_state
from src.hamiltonian import build_hamiltonian
from src.noise import NoiseConfig
from src.solvers import expectation


# --------------------------------------------------------------------------- #
# Shot noise
# --------------------------------------------------------------------------- #
def test_shot_noise_mean_and_spread_match_binomial():
    rng = np.random.default_rng(0)
    m, shots = 0.4, 500
    draws = np.array([shot_noise(m, shots, rng)[0] for _ in range(4000)])
    assert draws.mean() == pytest.approx(m, abs=0.005)
    assert draws.std() == pytest.approx(np.sqrt((1 - m**2) / shots), rel=0.05)
    assert np.all(np.abs(draws) <= 1)


def test_shot_noise_stderr_formula_and_floor():
    rng = np.random.default_rng(1)
    value, err = shot_noise(0.2, 1000, rng)
    assert err == pytest.approx(np.sqrt((1 - value**2) / 1000))
    # m = 1: every shot is +1, so the formula would give 0; the floor keeps it finite.
    assert shot_noise(1.0, 1000, rng) == (1.0, pytest.approx(1 / 1000))


def test_shot_noise_is_reproducible():
    a = [shot_noise(0.3, 100, np.random.default_rng(7)) for _ in range(2)]
    assert a[0] == a[1]


# --------------------------------------------------------------------------- #
# Experiment design
# --------------------------------------------------------------------------- #
def test_random_states_are_distinct_and_valid():
    states = random_states(3, 10, np.random.default_rng(0))
    assert len(states) == len(set(states)) == 10
    assert all(len(s) == 3 and set(s) <= set("01+-rl") for s in states)
    # Asking for more than exist (6^1 = 6) returns all of them.
    assert len(random_states(1, 50, np.random.default_rng(0))) == 6


def test_select_observables_nearest_neighbour():
    obs = select_observables(4, max_pair_distance=1)
    assert len(obs) == 3 * 4 + 3 * 3  # x,y,z per site + xx,yy,zz per bond
    assert "zz_0_1" in obs and "zz_0_2" not in obs
    assert len(select_observables(4, max_pair_distance=3)) == 3 * 4 + 3 * 6


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
SMALL = dict(sizes=[2, 3], times=[0.0, 0.1, 0.2], states_per_size=2)


def test_dataset_shape_and_neutral_columns():
    data, _ = generate_dynamics_dataset("lr_ising", **SMALL, shots=100, seed=0)
    assert list(data.columns) == CSV_COLUMNS
    # per n: states * times * observables (3n singles + 3(n-1) bonds)
    expected = sum(2 * 3 * (3 * n + 3 * (n - 1)) for n in (2, 3))
    assert len(data) == expected
    assert (data["stderr"] > 0).all()


def test_csv_does_not_leak_the_answer(tmp_path):
    data, truth = generate_dynamics_dataset("lr_ising", **SMALL, seed=0)
    out = save_dynamics_dataset(data, truth, tmp_path / "run")
    text = (out / "data.csv").read_text().lower()
    for leak in ("lr_ising", "alpha", "j0", "ising"):
        assert leak not in text
    saved = json.loads((out / "truth.json").read_text())
    assert saved["model"] == "lr_ising"
    assert saved["settings"][0]["alpha"] == 1.3


def test_same_seed_same_data():
    a, _ = generate_dynamics_dataset("lr_ising", **SMALL, seed=3)
    b, _ = generate_dynamics_dataset("lr_ising", **SMALL, seed=3)
    c, _ = generate_dynamics_dataset("lr_ising", **SMALL, seed=4)
    assert a.equals(b)
    assert not a.equals(c)


def test_noiseless_values_match_exact_evolution():
    # No shots, no gate noise, small dt: values should be the exact dynamics.
    data, _ = generate_dynamics_dataset(
        "lr_ising", sizes=[3], times=[0.0, 0.3], states_per_size=2, shots=None, dt=0.01, seed=0
    )
    H = build_hamiltonian("lr_ising", 3)
    obs = select_observables(3)
    for row in data.itertuples():
        exact = expectation(exact_evolved_state(row.state, H, row.t), obs[row.observable])
        assert row.value == pytest.approx(exact, abs=1e-4)


def test_time_zero_is_the_prepared_state():
    data, _ = generate_dynamics_dataset(
        "lr_ising", sizes=[2], times=[0.0], states_per_size=3, shots=None, seed=0,
        noise=NoiseConfig(p1=0.01, p2=0.05),
    )
    # No evolution gates at t=0. Prep gates are 1-qubit only, so p2 does nothing
    # and single-site values shrink by at most a few percent of p1.
    for row in data[data["observable"].str.count("_") == 1].itertuples():
        ideal = expectation(Statevector.from_label(row.state[::-1]), select_observables(2)[row.observable])
        assert row.value == pytest.approx(ideal, abs=0.03)


def test_settings_sweep_gets_separate_ids():
    data, truth = generate_dynamics_dataset(
        "lr_ising", sizes=[2], times=[0.1], states_per_size=1,
        settings=[{"alpha": 1.0}, {"alpha": 2.0}], seed=0,
    )
    assert sorted(data["setting_id"].unique()) == [0, 1]
    assert [s["alpha"] for s in truth["settings"]] == [1.0, 2.0]
