"""Sample a dev-branch emulator and write a chain the main-branch scorer can read.

The runs live on dev and the scoring machinery lives on main, and the two CANNOT share a
Python process: both ship a package called `likelihood`, so importing main's wrapper into
a process that has already loaded dev's resolves `from .base import` against the wrong
abstract base. HANDOVER4 section 5 records this; the bridge it describes was lost with the
scratchpad and this is its replacement.

Main's own benchmarking/make_surrogate_chain.py cannot be pointed at a dev run either. It
expects the older layout --- `trained_model_it_N.keras` beside pickled
`scalers/x_scaler_it_N.pkl` --- whereas dev bakes the input Normalization and the output
TargetDenormalization into the model itself. So the sampling has to happen HERE, on dev,
and only the resulting array crosses the branch boundary.

Two choices are not free:

  * T = 1, not the acquisition temperature. The credible metric is a statement about the
    POSTERIOR, not about the tempered distribution the acquisition explores. Sampling at
    T=7 and scoring it would compare the emulator against a target it was never meant to
    reproduce.
  * a PINNED step count. HANDOVER4 section 5 measured the adaptive stop giving different
    arms different chain lengths, worth ~0.01 in dCM --- enough to manufacture a
    difference between two arms that are actually identical. Always pass the same -n to
    every arm of a comparison.

Output is emcee's HDFBackend layout (group `mcmc`, datasets `chain` and `log_prob`, and
the `iteration` attribute), written into a main-style run directory so that
credible_metric.py and plot_corner_reference.py read it unchanged.

Usage:
    python benchmarking/dev_surrogate_chain.py results/<dev_run> \
        -o ../client_public/results/score_B1a -n 100000 --config-for-scoring <main.yaml>
"""

import argparse
import sys
import time
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def latest_iteration(run_dir):
    models = list((Path(run_dir) / "trained_models").glob("model_it_*.keras"))
    if not models:
        raise FileNotFoundError(f"no trained_models/model_it_*.keras in {run_dir}")
    return max(int(p.stem.rsplit("_", 1)[1]) for p in models)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run_dir", help="dev run directory")
    ap.add_argument("-o", "--output", required=True,
                    help="main-style run directory to write benchmark_chains/ into")
    ap.add_argument("-it", "--iteration", type=int, default=None,
                    help="iteration to sample (default: the latest trained)")
    ap.add_argument("-n", "--n-steps", type=int, default=100000,
                    help="PIN this across every arm of a comparison (default 100000)")
    ap.add_argument("--config-for-scoring", default=None,
                    help="main-schema yaml to copy into the output, naming the target")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    out_dir = Path(args.output).resolve()
    it = args.iteration if args.iteration is not None else latest_iteration(run_dir)

    chain_dir = out_dir / "benchmark_chains"
    chain_dir.mkdir(parents=True, exist_ok=True)
    chain_path = chain_dir / f"benchmark_chain_it_{it}.h5"
    if chain_path.exists() and not args.overwrite:
        print(f"chain already exists: {chain_path}\npass --overwrite to re-sample")
        return

    import yaml as _yaml

    from config.config import Config
    from likelihood.base import build_likelihood
    from likelihood.surrogate import SurrogateLikelihood, SurrogateMetadata
    from model.network import load_model
    from sampling.base import build_sampler

    cfg_path = next(iter(run_dir.glob("*.yaml")), None)
    if cfg_path is None:
        raise FileNotFoundError(f"no config yaml in {run_dir}")
    cfg = Config.from_yaml(str(cfg_path))

    # The likelihood is loaded only for its effective prior bounds --- the surrogate is
    # what gets sampled, and no true evaluation happens here.
    likelihood = build_likelihood(wrapper=cfg.likelihood.wrapper,
                                  input_path=str(run_dir / "likelihood_input"
                                                 / Path(cfg.likelihood.input).name))
    if cfg.prior.n_sigma is not None:
        likelihood.restrict_prior_bounds(cfg.prior.n_sigma)
    bounds = np.array([[lo, hi] for lo, hi in likelihood.prior_bounds.values()], float)
    ndim = likelihood.ndim

    surrogate = SurrogateLikelihood(
        model=load_model(run_dir / "trained_models" / f"model_it_{it}.keras"),
        metadata=SurrogateMetadata.load(run_dir / "metadata.json"))

    print(f"run        : {run_dir.name}")
    print(f"iteration  : {it}")
    print(f"sampling   : {cfg.sampling.n_walkers} walkers, {args.n_steps} steps, "
          f"burn-in {cfg.sampling.burn_in}, T=1 (the posterior, not the tempered target)")

    sampler = build_sampler(name=cfg.sampling.sampler,
                            n_walkers=cfg.sampling.n_walkers, ndim=ndim,
                            log_prob_fn=surrogate.logpost)     # T = 1: no tempering
    rng = np.random.default_rng(args.seed)
    t0 = time.monotonic()
    sampler.run(n_steps=args.n_steps,
                initial_positions=rng.uniform(bounds[:, 0], bounds[:, 1],
                                              size=(cfg.sampling.n_walkers, ndim)),
                burn_in=cfg.sampling.burn_in, adaptive=None,   # pinned, never adaptive
                chunk_size=cfg.sampling.chunk_size)
    elapsed = time.monotonic() - t0

    chain = np.asarray(sampler.chain())
    log_prob = np.asarray(sampler.log_prob())
    acceptance = float(np.mean(sampler.acceptance_fraction().numpy()))
    max_tau = float(sampler.max_tau)

    with h5py.File(chain_path, "w") as f:
        g = f.create_group("mcmc")
        g.attrs["iteration"] = chain.shape[0]
        g.attrs["has_blobs"] = False
        g.create_dataset("chain", data=chain, compression=None)
        g.create_dataset("log_prob", data=log_prob, compression=None)
        g.create_dataset("accepted", data=np.zeros(chain.shape[1], dtype=np.int64))
    sampler.reset()

    if args.config_for_scoring:
        target = out_dir / Path(args.config_for_scoring).name
        target.write_text(Path(args.config_for_scoring).read_text())
        print(f"copied scoring config -> {target.name}")

    print(f"finished   : {elapsed:.0f}s, acceptance {acceptance:.3g}, max tau {max_tau:.0f}")
    print(f"             chain holds {chain.shape[0] / max(max_tau, 1):.0f} tau")
    print(f"wrote      : {chain_path}  ({chain.nbytes / 2**30:.2f} GiB)")


if __name__ == "__main__":
    main()
