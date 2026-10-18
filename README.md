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

The code currently does what the upstream repository does: VQE ground states of
uniform-coupling spin chains (TFIM, XXZ, XY chain), with gate and shot noise, exported
to CSV. None of the QH-FEX changes below are built yet.

## What this fork will add

The upstream pipeline was built for a different target, so it needs these changes
before it can feed QH-FEX:

| Aspect      | Upstream now                              | What QH-FEX needs                          |
|-------------|-------------------------------------------|--------------------------------------------|
| Data type   | Ground-state energies and observables     | Time-evolved expectation values `⟨O(t)⟩`   |
| Couplings   | One uniform `J` per chain                 | Position-dependent laws such as `J0 \|i-j\|^(-α)` |
| Observables | Averaged over sites or nearest neighbours | Per-site and per-pair values               |
| Noise       | Gate noise, T1/T2, and shots              | Shot noise first, gate noise later         |
| CSV labels  | Model and parameter names in columns      | Neutral column names                       |

Planned work, roughly in order:

1. **Distance-dependent couplings.** Add models to `src/hamiltonian.py` whose couplings
   follow a law (power law, power law with a Gaussian beam envelope, site-dependent
   fields), so there is a formula to recover instead of a single constant.
2. **Per-site and per-pair observables.** Export `⟨Z_k⟩`, `⟨X_k X_{k+1}⟩`, etc. for each
   site and pair instead of chain averages. Averaging throws away the spatial information
   AutoSR needs to find something like `J(r)`.
3. **Time-evolution mode.** Prepare product states (each qubit in one of
   `|0⟩, |1⟩, |+⟩, |−⟩, |+i⟩, |−i⟩`), evolve them with Trotterized circuits, and record
   `⟨O_k(t)⟩` at several times. Short times are used for first-order fitting; longer
   times are held back for validation.
4. **Neutral data.** The CSV AutoSR sees has no model or parameter names in its columns
   or filename. The true Hamiltonian is written to a separate JSON file that AutoSR
   never reads. Each dataset records its random seed so it can be regenerated exactly.
5. **One framework.** Either port the noise model to PennyLane or move the QH-FEX
   generator to Qiskit, so both produce the same CSV format.

### Target dataset format

```
<dataset>.csv     n, setting_id, state, observable, t, value, stderr
<dataset>.json    true Hamiltonian (operator families + coefficient laws), seed  [hidden from AutoSR]
```

Shot noise for a Pauli observable with true expectation `m` and `M` shots:
`k ~ Binomial(M, (1+m)/2)`, recorded as `m̂ = 2k/M − 1` with standard error
`sqrt((1 − m̂²)/M)`.

### How the existing ground-state data is still used

The upstream VQE output stays useful as a second, static data type. A first test is
whether AutoSR can recover how the XXZ ground-state energies depend on `J`, `Δ`, and `h`.
Its gate noise model also gives a harder robustness test later on. VQE results include
optimizer error on top of noise, so filter on `vqe_error` and compare against the
`*_exact` columns before fitting.

---

## Current pipeline (from upstream)

Everything below describes the code as it exists today.

### Models

| name       | Hamiltonian                                                   | parameters        |
|------------|---------------------------------------------------------------|-------------------|
| `tfim`     | `-J ΣZZ - h ΣX`                                               | `J`, `h`          |
| `xxz`      | `J Σ(XX + YY + Δ ZZ) + h ΣZ`                                  | `J`, `delta`, `h` |
| `xy_chain` | `-Jx ΣXX - Jy ΣYY - h ΣZ` (spin form of the Kitaev chain)    | `Jx`, `Jy`, `h`   |

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
src/hamiltonian.py      models, observables
src/noise.py            NoiseConfig -> Aer NoiseModel
src/ansatz/             hea.py, hva.py
src/solvers/            exact_solver.py, vqe_engine.py
src/dataset/exporter.py grid sweeps -> DataFrame -> CSV
scripts/                CLI entry points
tests/                  pytest suite
```
