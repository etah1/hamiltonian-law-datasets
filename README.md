# hamiltonian-law-datasets

Simulated measurement datasets for **QH-FEX**, a project that learns the *formula*
behind a Hamiltonian's coefficients from measurement data, rather than estimating
each coefficient on its own.

A chain of `n` qubits with all-to-all couplings has about `n²/2` coupling constants,
but they usually come from one physical law, e.g. `J_ij = J0 |i - j|^(-α)`. QH-FEX
uses symbolic regression (AutoSR) to recover that law directly. This repository
generates the data AutoSR is tested on.

This is a fork of Aarav Nirmal's
[Hardware-Variational-Ansatz](https://github.com/AaravNirmal/Hardware-Variational-Ansatz),
a Qiskit pipeline for noisy VQE ground-state data on spin chains. Changes are being made
here with Aarav's agreement, and results that use this pipeline should credit him.

## Status

The time-evolution dataset pipeline works end to end on simulated data. Built in Qiskit
so it can reuse the upstream gate-noise model.

| Aspect      | Upstream                                  | This fork                                        |
|-------------|-------------------------------------------|--------------------------------------------------|
| Data type   | Ground-state energies and observables     | Time-evolved expectation values `⟨O(t)⟩`         |
| Couplings   | One uniform `J` per chain                 | `lr_ising`: `J0 \|i-j\|^(-α)` on every pair       |
| Observables | Averaged over sites or nearest neighbours | Per-site and per-pair values                     |
| Noise       | Gate noise, T1/T2, and shots              | Same gate noise, plus Binomial shot noise        |
| CSV labels  | Model and parameter names in columns      | Neutral columns; ground truth in a separate file |

Still to do: more law-based models (Gaussian beam envelope, site-dependent fields, XXZ
with a law), and longer held-out times for validation.

## Making a dataset for AutoSR

```bash
python scripts/make_dynamics_dataset.py --sizes 4 5 6 --times 0:0.5:6 --states 10     --shots 1000 --p1 0.0001 --p2 0.001 --seed 0 --out data/run1
```

This writes two files:

```
data/run1/data.csv     n, setting_id, state, observable, t, value, stderr   <- give to AutoSR
data/run1/truth.json   model, parameters, noise, Trotter settings, seed     <- never give to AutoSR
```

- **state**: one letter per qubit, qubit 0 first: `0 1` (Z), `+ -` (X), `r l` (Y, i.e. `|±i⟩`).
  States are drawn at random so X and Y eigenstates appear; from `|0…0⟩` alone the ZZ
  couplings are invisible at first order.
- **observable**: `x_k, y_k, z_k` on every qubit and `xx_i_j, yy_i_j, zz_i_j` on pairs with
  `|i-j| <= --pair-distance` (default 1, nearest neighbours).
- **value / stderr**: `k ~ Binomial(M, (1+m)/2)`, `m̂ = 2k/M − 1`, standard error
  `sqrt((1 − m̂²)/M)` floored at `1/M` so points where every shot agrees aren't given
  infinite weight.
- **Trotter**: each time uses `ceil(t/dt)` steps (`--dt`, `--order`); `t = 0` is the
  prepared state with no evolution gates.

Keep the `--out` folder name neutral (no model or parameter names), since a filename hint
broke the Ion T4 benchmark. `python scripts/run_dynamics_demo.py` prints exact vs Trotter
vs noisy `⟨Z_k(t)⟩` side by side as a sanity check.

### How the existing ground-state data is still used

The upstream VQE output stays useful as a second, static data type. A first test is
whether AutoSR can recover how the XXZ ground-state energies depend on `J`, `Δ`, and `h`.
Its gate noise model also gives a harder robustness test later on. VQE results include
optimizer error on top of noise, so filter on `vqe_error` and compare against the
`*_exact` columns before fitting.

---

## Ground-state pipeline (from upstream)

Aarav's original VQE pipeline, unchanged apart from accepting `lr_ising`.

### Models

| name       | Hamiltonian                                                   | parameters        |
|------------|---------------------------------------------------------------|-------------------|
| `tfim`     | `-J ΣZZ - h ΣX`                                               | `J`, `h`          |
| `xxz`      | `J Σ(XX + YY + Δ ZZ) + h ΣZ`                                  | `J`, `delta`, `h` |
| `xy_chain` | `-Jx ΣXX - Jy ΣYY - h ΣZ` (spin form of the Kitaev chain)    | `Jx`, `Jy`, `h`   |
| `lr_ising` | `Σ_{i<j} J0 \|i-j\|^(-α) ZZ + h ΣX` (all pairs; use `--ansatz hea`) | `J0`, `alpha`, `h` |

Add a model by writing a builder in `src/hamiltonian.py` and registering it in `MODELS`
(plus an HVA in `src/ansatz/hva.py`, or just use `--ansatz hea`).

### Usage

```bash
pip install -r requirements.txt
python -m pytest                      # run tests
python scripts/run_vqe_demo.py        # single-point exact vs VQE vs noisy VQE

# parameter sweep -> CSV
python scripts/generate_sr_data.py --model tfim --n-qubits 4 \
    --grid J=1.0 h=0.2:2.0:20 --p1 0.001 --p2 0.01 --shots 4000 \
    --out data/tfim_n4.csv
```

Grid axes: `name=start:stop:num` (linspace), `name=value`, or `name=v1,v2,v3`.
Axes are combined as a Cartesian product.

### Noise model

`--p1 / --p2` depolarizing error per 1-/2-qubit gate; `--t1 / --t2` thermal relaxation
(same time units as the gate times in `NoiseConfig`); `--shots` adds finite-sampling noise
to the `*_noisy` columns. By default the VQE is trained on the ideal statevector and the
optimised circuit is then run through the noise model (density-matrix simulation). Use
`--noisy-training` to optimise the noisy energy directly (slow).

### CSV columns

| group       | columns |
|-------------|---------|
| metadata    | `model, n_qubits, periodic, ansatz, depth, p1, p2, t1, t2, shots` |
| parameters  | one per Hamiltonian parameter (`J`, `h`, `delta`, ...) |
| energies    | `E_*` total and `e_*` per site, for `* ∈ {exact, vqe, noisy}` |
| observables | `{mx,my,mz,xx,yy,zz}_{exact,vqe,noisy}` (site / nearest-neighbour averages) |
| diagnostics | `gap, vqe_error, converged, n_evals` |

`*_exact` = exact diagonalisation, `*_vqe` = optimised ansatz (noiseless),
`*_noisy` = same circuit under noise. Filter on `vqe_error` before regression.

### Layout

```
src/hamiltonian.py      models, observables (averaged and per-site)
src/states.py           product-state preparation
src/evolution.py        Trotter / exact / noisy time evolution
src/noise.py            NoiseConfig -> Aer NoiseModel
src/ansatz/             hea.py, hva.py
src/solvers/            exact_solver.py, vqe_engine.py
src/dataset/exporter.py ground-state grid sweeps -> CSV
src/dataset/dynamics.py time-evolution datasets -> data.csv + truth.json
scripts/                CLI entry points
tests/                  pytest suite
```
