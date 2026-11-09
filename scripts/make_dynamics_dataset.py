#!/usr/bin/env python
"""Generate a time-evolution dataset for AutoSR.

Writes two files into --out:
  data.csv     n, setting_id, state, observable, t, value, stderr   (give to AutoSR)
  truth.json   model, parameters, noise, seed                      (never give to AutoSR)

Examples
--------
  python scripts/make_dynamics_dataset.py --out data/run1

  python scripts/make_dynamics_dataset.py --model lr_ising --params alpha=1.3 \
      --sizes 4 5 6 --times 0:0.5:6 --states 10 --shots 1000 \
      --p1 0.0001 --p2 0.001 --seed 1 --out data/run2

Keep --out neutral (no model or parameter names) so the folder name
doesn't give the answer away.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset import generate_dynamics_dataset, parse_axis, save_dynamics_dataset  # noqa: E402
from src.hamiltonian import MODELS, model_defaults  # noqa: E402
from src.noise import NoiseConfig  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--model", choices=sorted(MODELS), default="lr_ising")
    p.add_argument("--params", nargs="*", default=[], help="overrides, e.g. alpha=1.3 h=0.5")
    p.add_argument("--sizes", type=int, nargs="+", default=[4, 5, 6])
    p.add_argument("--times", default="0:0.5:6", help="start:stop:num or t1,t2,...")
    p.add_argument("--states", type=int, default=10, help="random product states per size")
    p.add_argument("--shots", type=int, default=1000)
    p.add_argument("--p1", type=float, default=0.0, help="1-qubit depolarizing")
    p.add_argument("--p2", type=float, default=0.0, help="2-qubit depolarizing")
    p.add_argument("--t1", type=float, default=None)
    p.add_argument("--t2", type=float, default=None)
    p.add_argument("--dt", type=float, default=0.1, help="max Trotter step size")
    p.add_argument("--order", type=int, choices=[1, 2], default=2)
    p.add_argument("--pair-distance", type=int, default=1, help="max |i-j| for pair observables")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="data/dataset")
    args = p.parse_args()

    params = {}
    for spec in args.params:
        name, values = parse_axis(spec)
        params[name] = values[0]
    unknown = set(params) - set(model_defaults(args.model))
    if unknown:
        p.error(f"{sorted(unknown)} are not parameters of '{args.model}': "
                f"{sorted(model_defaults(args.model))}")
    _, times = parse_axis(f"t={args.times}")

    data, truth = generate_dynamics_dataset(
        args.model,
        args.sizes,
        times,
        settings=[params],
        states_per_size=args.states,
        shots=args.shots,
        noise=NoiseConfig(p1=args.p1, p2=args.p2, t1=args.t1, t2=args.t2),
        dt=args.dt,
        order=args.order,
        max_pair_distance=args.pair_distance,
        seed=args.seed,
    )
    out = save_dynamics_dataset(data, truth, args.out)
    print(f"Wrote {len(data)} rows -> {out / 'data.csv'}")
    print(f"Ground truth       -> {out / 'truth.json'}  (do not give to AutoSR)")


if __name__ == "__main__":
    main()
