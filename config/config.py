from dataclasses import dataclass

import yaml


@dataclass(frozen=True)
class LikelihoodConfig:
    wrapper: str
    input: str

    @classmethod
    def from_dict(cls, d):
        return cls(
            wrapper=str(d["wrapper"]),
            input=str(d["input"]),
        )


@dataclass(frozen=True)
class PriorConfig:
    n_samples: int
    sampling_strategy: str
    n_sigma: float | None
    # Extra initial points drawn from the Gaussian implied by the quadratic response
    # surface, at the acquisition temperature; each costs one true likelihood call.
    # Default off: it measurably speeds convergence but did not improve the credible
    # metric, so it is not yet earned. See the README before switching it on.
    n_inject: int = 0

    @classmethod
    def from_dict(cls, d):
        n_sigma = d.get("n_sigma")
        return cls(
            n_samples=int(d["n_samples"]),
            sampling_strategy=str(d["sampling_strategy"]),
            n_sigma=None if n_sigma is None else float(n_sigma),
            n_inject=int(d.get("n_inject", 0)),
        )


@dataclass(frozen=True)
class AcquisitionConfig:
    n_append: int
    n_neighbors: int
    target_temperature: float
    pool_factor: int

    # Points re-drawn from the quadratic fit EVERY iteration, using the whole
    # accumulated training set rather than the initial design alone. 0 (the default)
    # leaves the loop exactly as it was. Distinct from prior.n_inject, which injects once
    # after the initial design: this one tracks the fit as it improves, and costs
    # n_inject evaluations per iteration rather than once.
    n_inject: int = 0

    # How many candidates the density-deficit selector commits per pass. 1 is the exact
    # sequential algorithm; >1 draws that many jointly, an approximation bounded by its
    # size that turns n_append passes over the candidate pool into n_append/batch_size of
    # them. Measured on the 31D ridge at n_append=32100: 3483 s at 1, 277 s at 10, 198 s
    # at 50, with per-parameter KS against the exact selection below 0.01 and no
    # measurable intra-set clustering. Set 1 to reproduce a run made before this option
    # existed, or when an arm must match one that used the exact path.
    batch_size: int = 10

    @classmethod
    def from_dict(cls, d):
        return cls(
            n_append=int(d["n_append"]),
            n_neighbors=int(d["n_neighbors"]),
            target_temperature=float(d["target_temperature"]),
            n_inject=int(d.get("n_inject", 0)),
            pool_factor=int(d["pool_factor"]),
            batch_size=int(d.get("batch_size", 10)),
        )


@dataclass(frozen=True)
class ModelConfig:
    n_layers: int
    n_neurons: int
    activation: str

    @classmethod
    def from_dict(cls, d):
        return cls(
            n_layers=int(d["n_layers"]),
            n_neurons=int(d["n_neurons"]),
            activation=str(d["activation"]),
        )


@dataclass(frozen=True)
class TrainingConfig:
    learning_rate: float
    loss: str
    sigma_level: float
    n_epochs: int
    batch_size: int
    validation_split: float
    patience: int
    # Floor on the msre effective sample size, as a fraction of the training set. The
    # default is ON: without it the loss collapses onto one or two points whenever the
    # initial design misses the peak by much, which on a 29D target scored 260x the
    # credible metric's noise floor against 8x with the floor active. It anneals itself
    # away once the training set fills in, so it costs nothing where it is not needed.
    # Set 0.0 to recover the untouched loss exactly.
    msre_ess_floor: float = 0.1

    # Learning-rate annealing. 'plateau' (the default) reduces the rate by lr_factor
    # whenever val_loss stalls for lr_patience epochs, down to lr_min, and then stops
    # lr_grace epochs later. 'none' restores the previous fixed-rate behaviour with the
    # plain `patience` EarlyStopping.
    #
    # It is the default because it is the only change tested in this project where a
    # training-side gain carried through to the credible metric. At a matched epoch
    # budget on the 31D ridge, against a fixed 1e-4:
    #
    #     validation loss   2.1x lower      credible metric   2.4x better
    #     median dCM        0.0135 vs 0.0322 (both seeds beat both fixed seeds)
    #     better on 27 of 31 parameters, 6 resolvable wins, 0 resolvable losses
    #
    # The mechanism is in the loss curve. At a fixed rate the coefficient of variation
    # over the 250 epochs before the best epoch is 124% and only 2 of those epochs come
    # within 10% of the best -- the optimiser is sampling a wide noise ball and
    # restore_best_weights catches a lucky dip. Annealed: CV 24%, and 97 epochs within
    # 10%. The weights are a settled minimum rather than a fortunate draw, which also
    # removes an uncontrolled source of run-to-run variance.
    #
    # `patience` is UNUSED when lr_schedule is 'plateau' -- see
    # training/training.py::StopAfterScheduleExhausted for why a plain patience cannot
    # terminate an annealed run at all.
    #
    # ON THE VALUE OF lr_patience. A 2x2 factorial over lr_patience 25/50 and lr_min
    # 1e-6/1e-7 (seed 42; the second seed was not run) found:
    #
    #     lr_patience  lr_min   epochs  min_lr at   val_loss
    #              50    1e-6      472        372   7.47e-05
    #              25    1e-6      305        205   1.95e-03   <- 26x WORSE
    #              50    1e-7      573        473   7.31e-05      (+2% for +101 epochs)
    #              25    1e-7      356        256   1.99e-03   <- 26x WORSE
    #
    # lr_patience 25 is catastrophic: it anneals so fast that the rate is frozen at the
    # floor by epoch 205, before the optimiser has descended, and the 305 epochs are cheap
    # only because the run is dead. The failure mode is therefore annealing TOO EARLY, and
    # 75 is set here to move further from it. lr_min 1e-7 was rejected: 2% of loss for 21%
    # more epochs, and a 2x loss spread in this family moved the credible metric by 9%,
    # far below its 0.025 floor.
    #
    # 75 IS AN EXTRAPOLATION, NOT A MEASUREMENT. The factorial establishes 25 << 50; it
    # does not establish 75 > 50. It is the cautious side of a known failure direction.
    # If it is ever measured, record the result here.
    # DEFAULT REVERTED TO 'none' ON 16 SEPTEMBER 2026 -- constant learning rate.
    #
    # Annealing remains available and is still 2.1x better on validation loss at a
    # matched budget on a DENSE training set (the section 6.2 result, measured on
    # data_it_15.csv with 32,100 points). But it is unsafe on a SPARSE one, and the
    # initial design of any high-dimensional run is sparse by construction.
    #
    # Measured on 58D iteration 0 (2,100 points, c inflated 1008x), three seeds each:
    #
    #     fixed 1e-4   1.4977e-02   1.8308e-02   1.4500e-02     spread 1.26x
    #     annealed     3.2887e-02   1.3641e-01   3.8296e-02     spread 4.15x
    #
    # The two sets do not overlap: the worst fixed run beats the best annealed run by
    # 1.80x, complete separation at one-sided p = 0.05. Annealing is also four times
    # less predictable, and its worst seed landed 9.1x above the fixed-rate result with
    # its first reduction at epoch 86.
    #
    # WHY. On sparse data val_loss is a wide noise ball -- at 58D iteration 0 it swings
    # a factor of two between neighbouring epochs (3.29e-02 at 520, 6.28e-02 at 580).
    # The plateau detector reads that as convergence and reduces the rate, after which
    # improvement is slower still, so the next plateau fires sooner and the rate
    # collapses. The best epoch is then a lucky dip caught by restore_best_weights
    # rather than a settled minimum.
    #
    # THE CONSEQUENCE IS NOT LOCAL. A 2.2x worse iteration-0 emulator produced an
    # acquisition that DISPERSED instead of contracting (selected-point spread 1.32
    # against the baseline's 0.18), which produced a reproducible iteration-1 training
    # failure, which contaminated every later training set. By iteration 10 the 58D
    # posterior sat entirely outside the truth's 95% contour in every cosmological
    # parameter, with 24 of 58 parameters above dCM 3.5 against the fixed-rate arm's 7.
    # An offline replay cleared the walker count, thinning, chain length and the batched
    # selector; swapping only the iteration-0 emulator moved the spread 1.203 -> 0.108.
    #
    # Set 'plateau' deliberately, on a target whose initial design is known to be dense.
    lr_schedule: str = "none"
    lr_factor: float = 0.3
    lr_patience: int = 75
    lr_min: float = 1e-6
    lr_grace: int = 100

    # Warm-starting: initialise iteration N's network from iteration N-1's trained
    # weights instead of at random. OFF by default -- it changes results, and the
    # evidence below is from one target.
    #
    # The scalers are NOT inherited. build_model adapts an input Normalization to the
    # current inputs and builds a TargetDenormalization from the current targets; both
    # are data statistics that move as the training set grows. Only the 22 trainable
    # weight tensors are transferred, so the learned function is inherited while
    # everything data-dependent is rebuilt for the current iteration. Carrying the
    # scalers over instead ("warm-continue") was measured and is NOT better: it starts
    # far closer (initial val_loss 4.97e-04 against cold's converged 3.96e-04 at 31D
    # iteration 10) but finishes no better, and it accumulates stale statistics.
    #
    # MEASURED on the 31D ridge, cold against warm-transfer, same data and split:
    #
    #     iteration   epochs to reach cold's best val_loss      epochs run
    #            1    164 vs 1022  (6.2x fewer)                 538 vs 1027
    #            5    251 vs 1038  (4.1x fewer)                 425 vs 1044
    #           10      2 vs  722  (warm beat cold's FINAL      402 vs  722
    #                               result in two epochs)
    #
    # Credible metric at iteration 10: median 0.0187 against cold's 0.0198 (a null --
    # both sit at the ~0.018 measurement floor), max excluding x_30 0.0431 against
    # 0.1066. So: equal-or-better accuracy for roughly half the epochs. At iteration 1
    # the dCM comparison was mixed and every arm was far from converged, so the safe
    # claim is "as accurate, much cheaper", not "more accurate".
    #
    # It also reduces exposure to a schedule that anneals too early, because a warm
    # network does not need the long from-scratch descent in the first place.
    #
    # DEFAULT TURNED ON 17 SEPTEMBER 2026, on two full runs each matched to its own
    # baseline on every setting except this one:
    #
    #     target            epochs run           it15 median dCM      it15 max
    #     58D CLOE +/-10s   7,852 vs 23,187      0.2318 vs 0.2551     0.6523 vs 1.0598
    #     31D ridge         8,539 vs 21,724      0.0081 vs 0.0209     0.0565 vs 0.0754
    #
    # i.e. 2.95x and 2.54x fewer epochs at equal or better accuracy. On 58D iteration 10
    # the worst parameter came in at 1.1704 against the baseline's 9.4522, with zero of
    # 58 parameters above dCM 3.5 where the baseline had seven -- roughly the baseline's
    # ITERATION-15 quality reached at iteration 10.
    #
    # Iteration 0 cannot be warm started, which makes it a free control: with nothing to
    # inherit it must reproduce the baseline exactly, and it did on both runs to four
    # digits (58D 2071 epochs at 1.4977e-02; 31D 1697 at 6.8004e-04). That is what
    # confirms warm_start was the only active variable in each comparison.
    #
    # THE OBJECTION DID NOT MATERIALISE. c re-solves every iteration, so inherited
    # weights are optimal for a different loss; the feared signature was val_loss
    # degrading as the weights age. It improved instead. TargetDenormalization is
    # rebuilt from the current target distribution each iteration, so the output is
    # rescaled before a single gradient step; what the weights carry is the SHAPE of the
    # function, which changes slowly.
    #
    # KNOWN BLEMISH: the 31D run is not monotone -- its iteration-10 median (0.0565) is
    # worse than both its own iteration 5 and the baseline's iteration 10, though both
    # sit near the ~0.018-0.026 measurement floor and its maxima are better at both
    # iterations. One seed per run. Set warm_start: false to restore cold starts.
    warm_start: bool = True

    @classmethod
    def from_dict(cls, d):
        return cls(
            learning_rate=float(d["learning_rate"]),
            loss=str(d["loss"]),
            sigma_level=float(d["sigma_level"]),
            n_epochs=int(d["n_epochs"]),
            batch_size=int(d["batch_size"]),
            validation_split=float(d["validation_split"]),
            patience=int(d["patience"]),
            msre_ess_floor=float(d.get("msre_ess_floor", 0.1)),
            warm_start=bool(d.get("warm_start", True)),
            lr_schedule=str(d.get("lr_schedule", "none")),
            lr_factor=float(d.get("lr_factor", 0.3)),
            lr_patience=int(d.get("lr_patience", 75)),
            lr_min=float(d.get("lr_min", 1e-6)),
            lr_grace=int(d.get("lr_grace", 100)),
        )


@dataclass(frozen=True)
class SamplingConfig:
    sampler: str
    temperature: float
    n_walkers: int
    burn_in: int
    n_steps: int
    # Optional adaptive stopping. adaptive: false (the default) keeps n_steps a fixed
    # budget, exactly as before. adaptive: true makes n_steps a maximum and stops once
    # the chain holds ess_target autocorrelation times and tau has settled.
    adaptive: bool = False
    ess_target: int = 50
    delta_tau_tol: float = 0.05
    ac_thin: int = 10
    chunk_size: int = 5000

    # Keep every thin-th step, strided AT WRITE TIME, so a thinned chain never allocates
    # the unthinned one. sampling/aies.py has supported this since the preallocated-buffer
    # change but nothing reached it: client.py did not pass it and this field did not
    # exist. 1 (the default) is the previous behaviour exactly.
    #
    # It is what makes a larger n_walkers affordable. The chain buffer is
    # n_steps/thin x n_walkers x ndim x 4 bytes, so at 100k steps and 31D:
    #
    #     216 walkers, thin 1  ->  2.49 GB      432 walkers, thin 1  ->  4.99 GB
    #     216 walkers, thin 5  ->  0.50 GB      432 walkers, thin 5  ->  1.00 GB
    #
    # Thinning does NOT reduce how many autocorrelation times a chain holds: rows and tau
    # shrink by the same factor, so rows/(tau/thin) = n_steps/tau is invariant. On the 31D
    # ridge tau ~ 5000, so thin 5 leaves 20000 rows at tau/thin = 1000 -- the same 20 tau
    # the unthinned 100k chain carried, at a fifth of the memory.
    #
    # The constraint is instead thin << tau, so the thinned series still resolves the
    # correlation: as tau/thin approaches 1 the series looks uncorrelated and tau becomes
    # unmeasurable. On these targets tau is 800-5000, so thin 5 has two to three orders of
    # magnitude of headroom. max(tau) is reported every run, so it is checkable.
    thin: int = 1

    @classmethod
    def from_dict(cls, d):
        return cls(
            sampler=str(d["sampler"]),
            temperature=float(d["temperature"]),
            n_walkers=int(d["n_walkers"]),
            burn_in=int(d["burn_in"]),
            n_steps=int(d["n_steps"]),
            adaptive=bool(d.get("adaptive", False)),
            ess_target=int(d.get("ess_target", 50)),
            delta_tau_tol=float(d.get("delta_tau_tol", 0.05)),
            ac_thin=int(d.get("ac_thin", 10)),
            chunk_size=int(d.get("chunk_size", 5000)),
            thin=int(d.get("thin", 1)),
        )

    @property
    def adaptive_options(self):
        if not self.adaptive:
            return None
        return {
            "ess_target": self.ess_target,
            "delta_tau_tol": self.delta_tau_tol,
            "ac_thin": self.ac_thin,
        }


@dataclass(frozen=True)
class ConvergenceConfig:
    threshold: float
    metric: str
    max_iterations: int

    @classmethod
    def from_dict(cls, d):
        return cls(
            threshold=float(d["threshold"]),
            metric=str(d["metric"]),
            max_iterations=int(d["max_iterations"]),
        )


@dataclass(frozen=True)
class Config:
    likelihood: LikelihoodConfig
    prior: PriorConfig
    acquisition: AcquisitionConfig
    model: ModelConfig
    training: TrainingConfig
    sampling: SamplingConfig
    convergence: ConvergenceConfig
    # Top-level `seed:`. Absent (None) leaves every RNG as it was. Set, it fixes the
    # prior design, the network initialisation, the walker start and the acquisition
    # draws -- but NOT the AIES proposals, which XLA reseeds regardless.
    seed: int = None

    @classmethod
    def from_dict(cls, d):
        return cls(
            likelihood=LikelihoodConfig.from_dict(d["likelihood"]),
            prior=PriorConfig.from_dict(d["prior"]),
            acquisition=AcquisitionConfig.from_dict(d["acquisition"]),
            model=ModelConfig.from_dict(d["model"]),
            training=TrainingConfig.from_dict(d["training"]),
            sampling=SamplingConfig.from_dict(d["sampling"]),
            convergence=ConvergenceConfig.from_dict(d["convergence"]),
            seed=(None if d.get("seed") is None else int(d["seed"])),
        )

    @classmethod
    def from_yaml(cls, path):
        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f))
