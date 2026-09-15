import argparse
import time
from pathlib import Path

import numpy as np

from likelihood.base import build_likelihood
from utils.mpi_utils import (
    bcast,
    broadcast_and_evaluate,
    get_size,
    is_master,
    is_mpi_available,
    print_master,
)


def draw_peak_samples(cfg, likelihood, inputs, targets, master,
                      n_draw, seed, context=""):
    """Points drawn from the Gaussian the quadratic response surface implies.

    Returns the NEW (inputs, targets) only; the caller decides where they go. The
    surface's vertex is a good estimate of where the maximum is, but a poor training
    point on its own: its log-likelihood sits hundreds of units above the rest of the
    design, so the msre weighting collapses onto it and the network fits one point. A
    *sample* from the same fit lands 0.5*T*ndim +/- 0.5*T*sqrt(2*ndim) below the vertex
    instead -- a few loss margins down, spread over a comparable width -- which is the
    regime the weighting expects.

    Drawn at the acquisition temperature, so the injected points span the log-likelihood
    range the sampler will actually visit. Costs n_draw true evaluations per call.

    Note the interaction with training.msre_ess_floor: the injected set is bimodal, and
    the floor targets an effective sample size over the *whole* training set, so it is
    forced to a very large c unless n_draw > alpha * n_train / (1 - alpha). See the
    README.
    """
    if master:
        from dataset.peak_samples import gaussian_peak_samples

        bounds = np.array(
            [[lo, hi] for lo, hi in likelihood.prior_bounds.values()], dtype=float
        )
        injected, info = gaussian_peak_samples(
            inputs,
            np.asarray(targets).reshape(-1),
            n_draw=n_draw,
            temperature=cfg.acquisition.target_temperature,
            bounds=bounds,
            seed=seed,
        )
        print_master(
            f"{context}Injecting {len(injected)} points from the quadfit Gaussian at "
            f"T={cfg.acquisition.target_temperature:g}: fitted on "
            f"{len(targets)} points, vertex value "
            f"{info['model_value']:.1f}, predicted deficit {info['deficit'].mean():.1f} "
            f"+/- {info['deficit'].std():.1f} (law {info['deficit_mean_expected']:.1f} "
            f"+/- {info['deficit_sd_expected']:.1f}), draw acceptance "
            f"{info['acceptance']:.2f}, fit rms {info['resid']:.1f}"
        )
    else:
        injected = None

    new_inputs, new_targets = broadcast_and_evaluate(
        samples=injected, evaluator=likelihood.loglkl
    )
    if not master:
        return None, None

    ok = np.isfinite(new_targets)
    print_master(
        f"   injected log-likelihood range "
        f"[{new_targets[ok].min():.1f}, {new_targets[ok].max():.1f}]"
    )
    return new_inputs[ok], new_targets[ok]


def build_initial_design(cfg, likelihood, master):
    """The initial design: a Latin hypercube (or whatever prior.sampling_strategy asks).

    Returns (samples, info) with samples on master only, matching sample_prior's contract
    so the caller's broadcast_and_evaluate is unchanged.
    """
    if not master:
        return None, None
    from sampling.prior_sampler import sample_prior

    print_master(
        f"Generating {cfg.prior.n_samples} {cfg.prior.sampling_strategy} samples")
    return sample_prior(likelihood=likelihood, n_samples=cfg.prior.n_samples,
                        strategy=cfg.prior.sampling_strategy), None


def inject_peak_samples(cfg, likelihood, inputs, targets, master):
    """One-shot injection after the initial design (`prior.n_inject`)."""
    new_inputs, new_targets = draw_peak_samples(
        cfg, likelihood, inputs, targets, master,
        n_draw=cfg.prior.n_inject, seed=cfg.seed,
    )
    if not master:
        return inputs, targets

    inputs = np.vstack([inputs, new_inputs])
    targets = np.concatenate([targets, new_targets])
    print_master(f"   training set now {len(targets)} points")
    return inputs, targets


def main():
    using_mpi = is_mpi_available()
    master = is_master()
    mpi_status = f"Enabled ({get_size()} processes)" if using_mpi else "Disabled"
    print_master(f"\nMPI status: {mpi_status}\n")

    # ---- Arguments ----
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_or_dir", help="Input YAML file (new run) or run directory (continue)"
    )
    parser.add_argument(
        "-n", "--name", help="Run name/tag for organization (new runs only)"
    )
    parser.add_argument(
        "-o",
        "--output",
        default="results",
        help="Base output directory (new runs only)",
    )
    parser.add_argument(
        "-r",
        "--retrain",
        action="store_true",
        help="Force retrain even if a saved model exists",
    )
    parser.add_argument(
        "-s",
        "--start",
        type=int,
        help="Starting iteration (continue only, auto-detected if omitted)",
    )
    parser.add_argument(
        "-i",
        "--iterations",
        type=int,
        help="Number of (additional) iterations to run (overrides convergence criterion)",
    )
    args = parser.parse_args()

    # ---- Configuration (master loads the run and broadcasts it to workers) ----
    if master:
        from run.metrics import MetricsTracker
        from run.run import Run

        run = Run.from_args(args)
        if run.is_new:
            run.create_directories(args.input_or_dir)

        metrics_tracker = MetricsTracker(
            results_dir=run.run_dir,
            start_iteration=run.start_iteration,
            preserve_start_metrics=False if run.is_new else True,
        )

        print_master(f"Run ID: {run.run_id}")
        print_master(f"Run mode: {run.mode}")
        print_master(f"Run directory: {run.run_dir}\n")
    else:
        run = None
        metrics_tracker = None

    run = bcast(run)
    cfg = run.config

    # Seed before anything draws.
    if cfg.seed is not None:
        import tensorflow as tf

        tf.keras.utils.set_random_seed(cfg.seed)
        print_master(f"Seed: {cfg.seed} (AIES proposals excepted -- XLA discards them)")

    # ---- Likelihood ----
    print_master("initializing likelihood...")

    likelihood = build_likelihood(
        wrapper=cfg.likelihood.wrapper,
        input_path=cfg.likelihood.input,
    )
    if cfg.prior.n_sigma is not None:
        likelihood.restrict_prior_bounds(cfg.prior.n_sigma)

    # ---- Surrogate metadata ----
    if master:
        from dataset.dataset import TrainingDataset
        from likelihood.surrogate import SurrogateLikelihood, SurrogateMetadata

        surrogate_metadata = SurrogateMetadata.from_likelihood(likelihood)
        if run.is_new:
            surrogate_metadata.save(run.run_dir / "metadata.json")
    else:
        surrogate_metadata = None


    # ---- Training data (new run) ----
    if run.is_new:
        prior_samples, design_info = build_initial_design(cfg, likelihood, master)

        inputs, targets = broadcast_and_evaluate(
            samples=prior_samples, evaluator=likelihood.loglkl
        )

        if master:
            valid = np.isfinite(targets)
            if not valid.all():
                inputs, targets = inputs[valid], targets[valid]
                print_master(
                    f"Warning: filtered {valid.size - np.count_nonzero(valid)} targets with non-finite loglkl values"
                )


        if cfg.prior.n_inject > 0:
            inputs, targets = inject_peak_samples(cfg, likelihood, inputs, targets, master)

        if master:
            dataset = TrainingDataset(
                inputs=inputs,
                targets=targets,
                likelihood=likelihood,
                n_neighbors=cfg.acquisition.n_neighbors,
                target_temperature=cfg.acquisition.target_temperature,
            )
            dataset.save(run.training_data_dir / "data_it_0.csv")
        else:
            dataset = None

    # --- Training data (continue run) ----
    else:
        if master:
            dataset = TrainingDataset.load(
                training_data_dir=run.training_data_dir,
                likelihood=likelihood,
                n_neighbors=cfg.acquisition.n_neighbors,
                target_temperature=cfg.acquisition.target_temperature,
                iteration=run.start_iteration,
            )
            print_master(
                f"Loaded {len(dataset.inputs)} training samples from data_it_{run.start_iteration}.csv"
            )
        else:
            dataset = None

    # ---- Convergence metric ----
    previous_chain_summary = None
    if master:
        from convergence.convergence import build_convergence_metric

        metric = build_convergence_metric(cfg.convergence.metric)
        if not run.is_new and run.start_iteration > 0:
            previous_chain_summary = metric.load_chain_summary(
                run.convergence_stats_dir, run.start_iteration - 1
            )
            print_master(
                f"Loaded previous chain summary from chain_summary_it_{run.start_iteration - 1}.npz"
            )
    else:
        metric = None

    final_iteration = run.final_iteration
    use_convergence = run.use_convergence

    # Hoist loop-invariant values and master-only imports out of the iteration loop.
    if master:
        ndim = likelihood.ndim
        prior_bounds = np.asarray(list(likelihood.prior_bounds.values()), dtype=float)
        prior_lower = prior_bounds[:, 0]
        prior_upper = prior_bounds[:, 1]
        inverse_sampling_temperature = 1.0 / cfg.sampling.temperature

        import tensorflow as tf

        from dataset.acquisition import select_points
        from model.network import build_model, load_model
        from sampling.base import build_sampler
        from training.losses import build_loss
        from training.training import save_history, train_model

    # ---- Main loop ----
    for iteration in range(run.start_iteration, final_iteration + 1):
        print_master(f"\n--- Iteration {iteration}/{final_iteration} ---")
        if master:
            iteration_start_time = time.monotonic()
            model_path = run.trained_models_dir / f"model_it_{iteration}.keras"
            # Train for new/retrain runs, after acquisition updates, or if a continued run has no saved model.
            should_train = (
                run.is_new
                or run.retrain
                or iteration != run.start_iteration
                or not model_path.exists()
            )
            if should_train:
                if not model_path.exists() and not run.is_new:
                    print_master(
                        f"No saved model found for iteration {iteration}; retraining..."
                    )
                print_master(f"Training model on {len(dataset.inputs)} samples...")
            else:
                print_master(f"Loading existing model: {model_path}")

        # ---- Training ----
        if master and should_train:
            tf.keras.backend.clear_session()

            # Shuffle the training data to avoid biasing the validation set
            shuffle_indices = np.random.permutation(len(dataset.inputs))
            inputs, targets = (
                dataset.inputs[shuffle_indices],
                dataset.targets[shuffle_indices],
            )

            model = build_model(
                inputs=inputs,
                targets=targets,
                n_layers=cfg.model.n_layers,
                n_neurons=cfg.model.n_neurons,
                activation=cfg.model.activation,
            )
            loss = build_loss(
                name=cfg.training.loss,
                sigma_level=cfg.training.sigma_level,
                chi2_dof=ndim,
                max_loglkl=float(targets.max()),
                targets=targets,
                ess_floor=cfg.training.msre_ess_floor,
            )
            training_start_time = time.monotonic()
            history, training_metrics = train_model(
                model=model,
                inputs=inputs,
                targets=targets,
                loss=loss,
                learning_rate=cfg.training.learning_rate,
                n_epochs=cfg.training.n_epochs,
                batch_size=cfg.training.batch_size,
                validation_split=cfg.training.validation_split,
                patience=cfg.training.patience,
            )
            training_time = time.monotonic() - training_start_time
            print_master(f"Finished training in {training_time:.2f}s")
            model.save(model_path)
            save_history(
                history.history,
                run.training_history_dir / f"history_it_{iteration}.csv",
            )
            metrics_tracker.add_training_metrics(
                iteration=iteration,
                epoch=training_metrics["epoch"],
                loss=training_metrics["loss"],
                val_loss=training_metrics["val_loss"],
                training_time=training_time,
            )
            metrics_tracker.save_all_metrics()

        # ---- Surrogate ----
        if master:
            # Reuse the freshly trained model in memory; only load from disk for already-trained iterations.
            if not should_train:
                tf.keras.backend.clear_session()
                model = load_model(model_path)
            surrogate = SurrogateLikelihood(
                model=model,
                metadata=surrogate_metadata,
            )

            # ---- Sampling ----
            def tempered_logpost_fn(positions):
                return surrogate.logpost(positions) * inverse_sampling_temperature

            sampler = build_sampler(
                name=cfg.sampling.sampler,
                n_walkers=cfg.sampling.n_walkers,
                ndim=ndim,
                log_prob_fn=tempered_logpost_fn,
            )
            initial_positions = np.random.uniform(
                low=prior_lower,
                high=prior_upper,
                size=(cfg.sampling.n_walkers, ndim),
            )
            sampling_start_time = time.monotonic()
            sampler.run(
                n_steps=cfg.sampling.n_steps,
                initial_positions=initial_positions,
                burn_in=cfg.sampling.burn_in,
                adaptive=cfg.sampling.adaptive_options,
                chunk_size=cfg.sampling.chunk_size,
                thin=cfg.sampling.thin,
            )
            sampling_elapsed_time = time.monotonic() - sampling_start_time

            # chain() and log_prob() are numpy views of the sampler's own buffer, so
            # they cost nothing here; the sampler must therefore not be reset until
            # acquisition has finished with them.
            chain = sampler.chain()
            logposts = sampler.log_prob()

            acceptance = sampler.acceptance_fraction().numpy()
            steps_run = sampler.n_steps_run
            max_tau = sampler.max_tau
            sampler_converged = sampler.converged

            mean_acceptance = float(np.mean(acceptance))
            tau_note = f", max(tau): {max_tau:.1f}"
            if cfg.sampling.adaptive:
                tau_note += f", converged: {sampler_converged}"
            print_master(
                f"Finished sampling in {sampling_elapsed_time:.2f}s "
                f"({steps_run} steps, acceptance rate: {mean_acceptance:.2g}{tau_note})"
            )

            metrics_tracker.add_sampling_metrics(
                iteration=iteration,
                steps_per_walker=steps_run,
                acceptance_rate=mean_acceptance,
                sampling_time=sampling_elapsed_time,
            )
            metrics_tracker.save_all_metrics()

        # ---- Convergence check ----
        converged = False
        if master:
            chain_summary = metric.summarize(chain)
            metric.save_chain_summary(
                convergence_stats_dir=run.convergence_stats_dir,
                iteration=iteration,
                chain_summary=chain_summary,
            )
            if previous_chain_summary is None:
                print_master(
                    "No previous chain summary available, skipping convergence check"
                )
            else:
                metric_value = metric.compute_from_summaries(
                    current_chain_summary=chain_summary,
                    previous_chain_summary=previous_chain_summary,
                )
                print_master(
                    f"{metric.name}: {metric_value:.3g} (threshold: {cfg.convergence.threshold:.3g})"
                )
                converged = metric_value < cfg.convergence.threshold
                metrics_tracker.add_convergence_metrics(
                    iteration=iteration,
                    metric_value=metric_value,
                    converged=converged,
                    metric_name=metric.name,
                )
                metrics_tracker.save_all_metrics()
                if use_convergence and converged:
                    print_master("Convergence criterion met, stopping...")
            previous_chain_summary = chain_summary

        # Broadcast the stopping decision so all ranks leave the loop together.
        if use_convergence:
            converged = bcast(converged)
            if converged:
                if master:
                    iteration_elapsed_time = time.monotonic() - iteration_start_time
                    metrics_tracker.add_iteration_metrics(
                        iteration=iteration,
                        iteration_time=iteration_elapsed_time,
                    )
                    metrics_tracker.save_all_metrics()
                break

        # ---- Acquisition ----
        if iteration < final_iteration:
            print_master(
                f"Selecting {cfg.acquisition.n_append} new samples from surrogate chain"
            )

            if master:
                acquisition_start_time = time.monotonic()
                n_samples = logposts.size
                new_samples, acq_metrics = select_points(
                    dataset=dataset,
                    chain=chain,
                    logposts=logposts,
                    n_append=cfg.acquisition.n_append,
                    mcmc_temperature=cfg.sampling.temperature,
                    pool_factor=cfg.acquisition.pool_factor,
                    batch_size=cfg.acquisition.batch_size,
                )

                # The sampler output is no longer needed after acquisition; release it
                # before likelihood evaluations. chain/logposts are views into the
                # sampler's buffer, so the sampler has to go too or nothing is freed.
                chain = None
                logposts = None
                sampler.reset()
                sampler = None

                n_unique = acq_metrics["n_unique"]
                n_duplicates = n_samples - n_unique
                unique_fraction = n_unique / n_samples
                print_master(
                    f"Identified {n_unique} unique samples from {n_samples} total samples "
                    f"({n_duplicates} duplicates, unique fraction: {unique_fraction:.3g}, "
                    f"max multiplicity: {acq_metrics['max_multiplicity']})"
                )
                print_master(
                    f"Estimated the target density from a "
                    f"{acq_metrics['n_reference']}-point reference sample "
                    f"({len(dataset.inputs)} training points)"
                )
                if len(new_samples) < cfg.acquisition.n_append:
                    print_master(
                        f"Warning: selected only {len(new_samples)}/"
                        f"{cfg.acquisition.n_append} requested acquisition samples"
                    )
            else:
                new_samples = None

            new_inputs, new_targets = broadcast_and_evaluate(
                samples=new_samples, evaluator=likelihood.loglkl
            )

            if master:
                valid = np.isfinite(new_targets)
                if not valid.all():
                    new_inputs, new_targets = new_inputs[valid], new_targets[valid]
                    print_master(
                        f"Warning: filtered {valid.size - np.count_nonzero(valid)} targets with non-finite loglkl values"
                    )

                n_current_inputs = len(dataset.inputs)
                dataset.add_data(inputs=new_inputs, targets=new_targets)
                n_new_inputs = len(dataset.inputs) - n_current_inputs
                acquisition_elapsed_time = time.monotonic() - acquisition_start_time
                print_master(
                    f"Added {n_new_inputs} new training samples in {acquisition_elapsed_time:.2f}s for a total of {len(dataset.inputs)} samples"
                )
                metrics_tracker.add_acquisition_metrics(
                    iteration=iteration,
                    n_evaluated=len(new_targets),
                    n_added=n_new_inputs,
                    acquisition_time=acquisition_elapsed_time,
                    dataset_size=len(dataset.inputs),
                )
                metrics_tracker.save_all_metrics()

            # Re-fit the quadratic on EVERYTHING gathered so far and draw a fresh sample
            # from it. prior.n_inject fits the initial design once; this tracks the fit as
            # the training set improves, at n_inject evaluations per iteration.
            if cfg.acquisition.n_inject > 0:
                inj_seed = None if cfg.seed is None else cfg.seed + 1000 + iteration
                inj_inputs, inj_targets = draw_peak_samples(
                    cfg, likelihood,
                    dataset.inputs if master else None,
                    dataset.targets if master else None,
                    master,
                    n_draw=cfg.acquisition.n_inject,
                    seed=inj_seed,
                    context=f"it {iteration + 1}: ",
                )
                if master:
                    dataset.add_data(inputs=inj_inputs, targets=inj_targets)
                    print_master(
                        f"   training set now {len(dataset.inputs)} points"
                    )

            if master:
                dataset.save(run.training_data_dir / f"data_it_{iteration + 1}.csv")

        if master:
            iteration_elapsed_time = time.monotonic() - iteration_start_time
            metrics_tracker.add_iteration_metrics(
                iteration=iteration,
                iteration_time=iteration_elapsed_time,
            )
            metrics_tracker.save_all_metrics()

            # Release remaining per-iteration objects.
            chain = None
            logposts = None
            surrogate = None
            model = None

    # ---- Finalization ----
    if master:
        metrics_tracker.save_all_metrics()
        print_master(f"\nRun completed: {run.run_id}")
        print_master(f"Results saved in: {run.run_dir}")


if __name__ == "__main__":
    main()
