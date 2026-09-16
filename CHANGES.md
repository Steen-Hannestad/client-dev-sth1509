# Changes to CLiENT

Everything below is additive and **off by default except where stated**. A configuration
written for the previous version parses unchanged and, with the single exception in §1,
produces identical behaviour.

Contents: §1 the loss, §2 Gaussian peak injection, §3 the sampler, §4 reproducibility,
§5 new targets, §6 configuration reference, §7 what is not established.

---

## 1. The msre loss: a floor on the effective sample size

**`training/losses.py`**, config key `training.msre_ess_floor`.
**This is the one changed default: it is now `0.1`, previously the behaviour was `0.0`.**

### The problem

msre weights each training point by

```
w_i  ∝  1 / (y_max − y_i + c)² ,     c = ½ Δχ²(sigma_level, ndim)
```

`c` is a distance below the maximum inside which the loss still cares — 19.2 log-units at
ndim=16, 28.7 at ndim=29. The specification measures that distance from the *true*
maximum; the code has only `max(y_train)`, the best point of the current training set.

On a wide prior box those are not the same thing. Measured on a 2000-point Latin
hypercube over the 29D banana at ±30σ, the best point sits **4041 log-units** below the
true maximum — an anchor error of 141 `c`. The weights collapse:

| training set | effective sample size | weight on the single best point |
|---|---:|---:|
| 16D wide, 2000 points | 1.5 of 2000 | 80.7 % |
| 29D wide, 2000 points | **1.0 of 2000** | **98.8 %** |

The network is fitting one point. A visible symptom is early stopping firing on noise:
in one run it terminated after 227 epochs with a validation loss *worse* than iteration 0,
and the resulting surrogate produced a chain with 2.3 % acceptance.

Note this is **not** a scaling artefact. The loss is exactly invariant to the target
scaler — σ cancels between numerator and denominator — so the collapse must be, and is,
explained in physical units alone.

### The fix

Raise `c` until the effective sample size reaches `alpha * n`:

```
c = max(c₀, c_ESS),    c_ESS solves  ESS(c) = alpha·n,    ESS(c) = 1 / Σ wᵢ²
```

`ESS` is monotone increasing in `c` — raising it moves every pairwise weight ratio toward
one — so the root is unique and bisecting upward from `c₀` is safe. `c` can only increase,
so weight can only spread.

**It anneals itself away.** As acquisition fills the peak region the empirical maximum
rises, `ESS(c₀)` climbs past the floor on its own, and `c` returns to `c₀` exactly. There
is no schedule. Measured on the 29D wide target:

| iteration | c / c₀ | ESS at c₀ |
|---:|---:|---:|
| 0 | ×75.2 | 1.0 |
| 1 | ×9.20 | 5.8 |
| 2 | ×6.60 | 12.6 |
| 3 | ×1.16 | 673.6 |
| 4 | ×1.00 | 1850.8 |
| 5 | ×1.00 | 3066.5 |

`msre_ess_floor: 0.0` reproduces the untouched loss to 13 significant figures.

### Why the default changed

On the 29D wide target, scored against 1.6×10⁷ samples of the true posterior:

| | credible metric (68 %) | × noise floor | posterior drift |
|---|---:|---:|---:|
| `msre_ess_floor: 0.0` | 10.80 | 260× | 19.4 |
| `msre_ess_floor: 0.1` | **0.334** | **8×** | 0.551 |

Every ΛCDM parameter goes from 2–11 times its own credible width in error to within about
twice the metric's noise floor. Drift and the credible metric — independent measurements —
agree the effect is 35× and 32× respectively.

On easier targets the floor binds for one or two iterations and then switches off, so it
costs nothing there. **The harder the target, the longer it works and the more it buys.**

---

## 2. Gaussian peak injection

**`dataset/peak_samples.py`**, config key `prior.n_inject`. **Default 0 (off).**

### Idea

The quadratic response surface fitted to the initial design gives not only the location of
the maximum but the full curvature. In whitened coordinates,

```
logL(z) ≈ logL(z*) − ½ (z − z*)ᵀ P (z − z*),     P = −H  (positive definite)
```

so the tempered posterior `L^(1/T)` is Gaussian with mean `z*` and covariance `T·P⁻¹`, and

```
z = z* + √T · V Λ^(−1/2) u ,     u ~ N(0, I) ,     P = V Λ Vᵀ
```

draws points distributed as a chain at temperature `T` would be.

### Why a sample and not the vertex

Injecting the vertex alone as a single training point makes the collapse *worse*: it
raises `y_max` by the anchor error `A`, so its own weight share is about
`[1 + n c² / A²]⁻¹` — 91 % predicted, 99.7 % measured. A sample instead lands

```
deficit below the vertex  =  ½ T ‖u‖² ,   ‖u‖² ~ χ²_d
                          =  ½ T d  ±  ½ T √(2d)
```

which is **56 ± 20** log-units at d=16 and **102 ± 27** at d=29, against loss margins `c`
of 19.2 and 28.7. A few margins down, spread over a comparable width. Measured effect
before the loss does anything:

| training set | ESS (29D wide) | top point |
|---|---:|---:|
| LHC only | 1.0 | 98.8 % |
| LHC + vertex | 1.0 | 99.7 % |
| **LHC + 100 draws** | **61.6** | **4.4 %** |

`stratify_radius` (default on) draws `‖u‖` from stratified χ quantiles: in high dimension a
Gaussian sample concentrates in a thin shell, and with ~100 draws in 29D the radial
coverage of a plain sample is poor. Points outside the prior box are rejected and redrawn,
with the batch size adapted to the observed acceptance; if the box is too tight the
shortfall is clipped and **reported** in `n_clipped` rather than silently under-delivering.

### Interaction with §1 — read before enabling

Injection makes the training set bimodal: `n_inject` points near the maximum,
`n_samples` points at deficit ≈ `A`. `ESS(c)` then plateaus near `n_inject` until `c`
approaches `A`, so a floor targeting `alpha·n` over the whole set is forced to `c ≈ A`
whenever `alpha·n > n_inject`, i.e. whenever

```
n_inject  ≤  alpha · n_samples / (1 − alpha)
```

— 223 at `alpha=0.1`, `n_samples=2000`. Measured, the transition is sharp:

| n_inject | alpha·n | c chosen | c / A |
|---:|---:|---:|---:|
| 100 | 210 | 4611.3 | 1.14 |
| 200 | 220 | 1744.3 | 0.43 |
| **250** | **225** | **98.7** | **0.02** |
| 500 | 250 | 28.7 (= c₀) | — |

**Choose `n_inject` above that threshold, or expect the floor to over-correct.**

### Re-injection: `acquisition.n_inject`

`prior.n_inject` fits the quadratic to the initial design **once** and never revisits it.
`acquisition.n_inject` (NEW, **default 0 = off**) re-fits it on the **whole accumulated
training set** after every acquisition and draws afresh, so the injected points track the
fit as the data improves. `client.py::draw_peak_samples` is the shared implementation;
`inject_peak_samples` is the one-shot wrapper, unchanged in behaviour.

Cost is `n_inject` true evaluations **per iteration** rather than once: 100 per iteration
over 16 iterations is 1600 calls, ~5% on top of a 32000-point budget. Comparisons against a
one-shot arm are therefore not evaluation-matched, and should say so.

**Why it matters.** On the 58D Euclid target (§5) with a ±10σ initial design, whose closest
design point sits 242σ from the peak in the whitened metric, the base arm never improves:
after 16 iterations its *closest* training point is still the one it started with, its best
log-likelihood is −29 033 against a true maximum of +343.9, and it scores 844× the noise
floor. The same pipeline with `acquisition.n_inject: 100` reaches a minimum radius of 6.0 —
inside the posterior's own typical radius of 7.6 — and scores 91×.

**The ESS floor alone cannot cross that gap.** It reweights the points you have; it does not
produce points nearer the peak. The injected samples do, because they come from the fitted
quadratic rather than from a surrogate that has never seen the maximum.

### The feedback loop, and that it self-corrects

Re-fitting on data that includes one's own previous injections is a feedback loop, and it is
visible. On the ±10σ target the vertex starts 1.9 log-units below the true maximum (fitted on
the initial design), falls to **536 below** by iteration 2 as early acquisition points
dominate the fit, then climbs monotonically back to 12.7 below by iteration 14. Throughout,
the fit residual is 0.0 and the draw acceptance 1.00 — the quadratic describes the data
exactly; it is the *data* that moves.

It self-corrected on both widths tested (±4σ dipped 36 log-units, ±10σ dipped 536), more
slowly the further out the design starts. **It is not guaranteed to self-correct on a target
where the quadratic is not an exact description**, and the banana and ridge targets are
exactly such targets. Watch the reported vertex value if you enable this.

---

## 3. The sampler

**`sampling/aies.py`**, **`sampling/autocorr.py`** (new).

### Memory: preallocated chain buffer

The chain was accumulated as a list of per-chunk tensors, concatenated with `tf.concat`,
then copied again by `.numpy()`. All three coexisted, so the high-water mark was **2.93×
one chain**. Each chunk is now written straight into a preallocated array and `chain()`
returns a view of the filled part.

| 216 walkers × 30 000 × 16D (0.386 GiB per copy) | peak RSS |
|---|---:|
| before | 1.589 GiB |
| after | **0.983 GiB** |

At 29D with 232 walkers × 100 000 steps a chain copy is 2.5 GiB, so this is the difference
between ~3.2 GiB and ~6.8 GiB resident — measured peak across the 29D runs was 3.8–4.4 GiB.

`np.empty` is lazily faulted, so preallocating at the maximum costs nothing in resident
memory until written: a 1.29 GiB buffer with 30 000 of 100 000 rows written is 0.407 GiB.

`chain()` and `log_prob()` now return **numpy views**, not tf tensors — callers must not
call `.numpy()` on them, and must not `reset()` the sampler while still using them.

### `thin` at write time

`run(..., thin=k)` keeps every k-th step as it goes, so a thinned chain never allocates the
unthinned one. `benchmarking/benchmark.py` uses this instead of slicing an
already-materialised chain.

### Optional adaptive stopping

`run(..., adaptive={...})`, config `sampling.adaptive`. **Default off** — `n_steps` remains
a fixed budget. Enabled, `n_steps` becomes a *maximum* and sampling stops once the chain
holds `ess_target` autocorrelation times and τ has settled to within `delta_tau_tol`.

**`max(tau)` is now reported after every run regardless**, costing ~0.7 s (0.4 % of a
sampling step). This matters more than the stopping rule: it is the only way to see whether
a fixed-length chain is long enough. The shipped 16D default of 30 000 steps turned out to
deliver only **36 τ** against the 50 the rule targets — under-sampling that was previously
invisible. On the 29D wide target the true likelihood has τ ≈ 1373, so 100 000 steps buys
73 τ.

---

## 4. Reproducibility

**`utils/rng.py`** (new), **`sampling/prior_sampler.py`**, top-level config key `seed`.

`scipy.stats.qmc` samplers and `np.random.default_rng()` read OS entropy and therefore
ignore `np.random.seed`. The initial design was not reproducible even with a seed set —
two runs of one configuration drew different Latin hypercubes, which makes any paired
comparison unfalsifiable. Generators are now seeded from the legacy global via
`seeded_entropy()`: successive calls stay independent, but the whole sequence becomes a
deterministic function of `seed`. With no seed set the global is itself entropy-seeded, so
unseeded behaviour is statistically unchanged.

With `seed: N` set, two runs agree byte-for-byte on the initial design, the network
initialisation, the walker start and the acquisition draws.

> **The AIES chain itself cannot be seeded.** Its proposals come from `tf.random.uniform`
> inside a `@tf.function(jit_compile=True)`, and XLA discards the seed — TensorFlow warns
> about this at runtime. Fixing it needs `tf.random.stateless_uniform` with an explicit
> counter. Until then, chain-level differences between two otherwise identical runs are
> not attributable, and there is no noise estimate on any chain-derived quantity.

---

## 5. New targets

- `input/cobaya/banana_lcdm16.yaml` — 16D reduction of the analytic banana (6 ΛCDM + 8
  Planck nuisance + the two banana directions), with `banana_loglkl_lcdm16` added to
  `input/cobaya/banana_function.py`.
- `input/cobaya/banana_lcdm16_wide.yaml` — the same with its 8 hard nuisance priors
  widened ×5.
- `input/cobaya/banana_planck_wide.yaml` — the 29D banana with **18 of its 21** nuisance
  priors widened ×5. `calib_100T`, `calib_217T` and `A_planck` are deliberately left alone:
  the ±30σ box never reaches their hard bounds, so widening them is inert.
- `input/cobaya/banana_planck31.yaml`, `banana_planck31_wide.yaml` — the 29D target plus a
  **curved ridge** in two further directions, `banana_loglkl_planck31`:

  ```
  penalty = −½ [ (x_31 − a·x_30²) / s ]² ,     a = 1, s = 1
  x_30 ∈ [−4, 4],  x_31 ∈ [0, 16]
  ```

  A second *kind* of non-Gaussianity. The existing `−½(x_28 x_29)²` is hyperbolic, with its
  maximum set on the two prior edges; this one's maximum set is the curve `x_31 = x_30²` —
  degenerate as before, but **curved**. It stands in for the `m_eff^sterile`–`N_eff`
  degeneracy of ΛCDM plus a sterile neutrino, which is a curved band rather than an
  ellipse.

  It exists to stress the assumption both new methods share: that a *quadratic* surface
  describes the region that matters. Measured on a 2000-point design at ±30σ:

  | | 29D wide | 31D wide |
  |---|---:|---:|
  | quadratic fit residual | 8.2 | **22.4** |
  | vertex error below the true maximum | 16.5 | **39.7** |
  | Hessian eigenvalues needing a sign flip | 1 | 2 |

  The LHC shortfall is comparable (4012 against 3383), so the target isolates *fit
  quality* rather than simply moving the anchor further away. The penalty again acts on a
  disjoint variable set, so the density factorises, the 27D Gaussian marginals are
  unchanged, and the maximum is still attained (anywhere on the ridge) — verified flat
  along its whole length and unbeaten by 20 000 perturbations around it.

Widening is a difficulty knob. The best point of a 2000-point design falls this far below
the true maximum:

| target | shortfall (log-units) |
|---|---:|
| 16D, ±10σ | 81 |
| 16D wide, ±30σ | 802 |
| 29D, ±30σ | 3201 |
| **29D wide, ±30σ** | **4041** |

---

- `input/cobaya/gaussian_cloe58.yaml` — a **58D Euclid/CLOE covariance**, in the form
  `gaussian_planck.yaml` already uses (Cobaya's built-in `gaussian`, mean and covariance
  inline). Generated from a `.covmat` by
  `client_public/benchmarking/make_gaussian_config.py`, so the 3364 covariance entries are
  machine-written and the file is reproducible from its source.

  No non-Gaussian structure at all: it isolates **dimension and correlation** from shape.
  Better conditioned than the 27D Planck matrix (2.9e9 against 7.4e10) but with **18
  parameter pairs above |r| = 0.9** against one. Twelve parameters carry the Euclid IST:F
  fiducial; the other 46 are centred at zero, which is a convention — a covmat has no
  centre, and the mean is pure location.

  **The initial-design width is the whole difficulty, and it is not what it looks like.**
  CLiENT draws over `mean ± n_sigma·σ_i` with σ_i the *marginal* deviations. On a strongly
  correlated covariance that box bears little relation to the posterior: the correlations
  contribute a constant **factor 47 in χ²** (equivalently √(det C_diag/det C) = 10^17.6 in
  volume), so the best of 2000 design points sits **97σ** from the peak at n_sigma = 4 and
  **242σ** at n_sigma = 10, against 7.6σ for a posterior draw.

  | n_sigma | 2 | 3 | **4** | 5 | **10** | 30 |
  |---|---:|---:|---:|---:|---:|---:|
  | best-of-2000 shortfall | 1024 | 2304 | **4096** | 6401 | **25603** | 185856 |

  **Use n_sigma = 4 here, not the 30 the banana targets use** — that puts the anchor error
  at 4096, matching the 29D wide banana's 4012, at twice the dimension.

  Run configs: `input/gaussian_cloe58_{base,essfloor,inject,inject250,reinject}.yaml` at
  ±4σ and `_{base_n10,reinject_n10}.yaml` at ±10σ.

  **This target needs about twice the iterations the banana targets did.** At iteration 5
  five different arms all sat between 326× and 352× the noise floor, which looked like a
  ceiling; continuing to iteration 10–15 reached 59× (±4σ) and 91× (±10σ). See
  `client_public/docs/client_58d_euclid.pdf`.

## 6. Configuration reference

Every key, with its default. `(required)` means the loader raises if it is absent.

```yaml
likelihood:
  wrapper: cobaya          # (required) cobaya | montepython
  input: input/cobaya/banana_planck31_wide.yaml   # (required)

prior:
  n_samples: 2000          # (required)
  sampling_strategy: lhs   # (required)
  n_sigma: 30.0            # (required; null for the likelihood's own bounds)
  n_inject: 100            # default 0. Points from the quadfit Gaussian, ONCE after the
                           # initial design. If used, keep n_inject > alpha*n_samples/(1-alpha).

acquisition:
  n_append: 2000           # (required)
  n_neighbors: 20          # (required) k for the density estimator
  target_temperature: 7.0  # (required)
  pool_factor: 20          # (required) candidate pool = pool_factor * n_append
  n_inject: 100            # default 0. Re-fit the quadfit on the WHOLE accumulated
                           # training set after each acquisition and draw this many.
  batch_size: 10           # default 10. Candidates committed per selector pass; 1 is the
                           # exact sequential algorithm. See PERFORMANCE.md section 1.

model:
  n_layers: 5              # (required)
  n_neurons: 512           # (required)
  activation: alsing       # (required)

training:
  learning_rate: 0.0001    # (required)
  loss: msre               # (required)
  sigma_level: 3           # (required)
  n_epochs: 5000           # (required) maximum; EarlyStopping decides
  batch_size: 128          # (required)
  validation_split: 0.1    # (required)
  patience: 250            # (required) UNUSED while lr_schedule is 'plateau'
  msre_ess_floor: 0.1      # default 0.1, i.e. ON. 0.0 restores the original loss.
  lr_schedule: plateau     # default 'plateau'. 'none' restores a fixed rate with the
                           # plain `patience` EarlyStopping.
  lr_factor: 0.3           # default 0.3
  lr_patience: 75          # default 75, epochs of stalled val_loss before a reduction
  lr_min: 1.0e-6           # default 1e-6, the floor
  lr_grace: 100            # default 100, epochs to run after the floor is reached

sampling:
  sampler: aies            # (required)
  temperature: 7.0         # (required)
  n_walkers: 432           # (required)
  n_steps: 100000          # (required) a fixed budget unless adaptive
  burn_in: 5000            # (required)
  thin: 5                  # default 1. Keep every thin-th step, strided at write time.
  chunk_size: 5000         # default 5000. Steps per graph call.
  adaptive: false          # default false. Stop on the autocorrelation criterion.
  ess_target: 50           # default 50, used only when adaptive
  delta_tau_tol: 0.05      # default 0.05, used only when adaptive
  ac_thin: 10              # default 10, thinning for the tau estimate

convergence:
  metric: gaussian_posterior_drift   # (required)
  threshold: 0.01          # (required) IGNORED unless the run is launched without -i
  max_iterations: 16       # (required) iterations 0 .. max_iterations-1

seed: 42                   # default null (unseeded)
```

---

## 7. What is not established

Stated explicitly so nobody has to rediscover it:

- **The injection's value depends on the target, and can be decisive.** On the 29D wide
  target it was 2.6× better in drift and marginally *worse* in the credible metric (0.348
  against 0.334 at 68 %) — convergence, not accuracy. On the 31D ridge it was the only arm
  to recover the curved degeneracy (4.2× the floor against 9.5×). On the 58D Euclid target
  at ±10σ, with per-iteration re-injection, it is the difference between 91× and 844× — the
  base arm never converges at all. **It is still off by default**, but "improves convergence
  but not accuracy" is only true where the design already starts near the peak.
- **Untested: `n_inject` above the §2 threshold.** Every scored injection run used 100,
  which is *below* `alpha·n/(1−alpha) = 222` at n = 2000, so the floor was forced to
  over-correct every time (up to ×160). A 250-point arm exists as
  `input/banana31_wide30_inject250.yaml` and `gaussian_cloe58_inject250.yaml`, both unrun.
- **The corrections' benefit scales with how far the initial design starts from the peak.**
  On 16D targets, where the floor binds for one or two iterations, it buys 1–2 iterations
  and the arms then become statistically indistinguishable (0.70σ over six). On 58D at ±4σ
  (design 97σ out) it saves about five iterations — the uncorrected arm reaches 326× the
  noise floor on its own, five iterations behind the corrected arm's 59×. On 58D at ±10σ
  (design 242σ out) the uncorrected arm does not converge at all. Any claim that these are
  "a cold-start fix, not a route to a better posterior" is specific to close-in designs.
- **Both corrections stall at ~8× the credible metric's noise floor** on the 29D banana,
  limited by the two banana directions `x_28`, `x_29`. Both rest on a *quadratic* surface,
  so those are exactly the directions where the approximation has least purchase. The best
  58D result sits at 59× its (much tighter) floor; the residual there is unexplained.
- **Six iterations is not a universal run length.** Every experiment before the 58D work
  used six. On 58D that is roughly half what is needed, and five arms agreeing at
  326–352× at iteration 5 looked like a structural ceiling but was five arms stopped too
  early. **Agreement between under-converged runs is not evidence of a limit.**
- **Nothing chain-derived has an error bar** until the sampler seeding of §4 is fixed —
  with one exception now measured. Continuing a run re-samples its last iteration from the
  saved model without retraining, which is the freeze-one-surrogate test for the drift
  metric's own resolution. Three such pairs:

  | max τ | drift (1) | drift (2) | difference |
  |---:|---:|---:|---:|
  | ~7900 | 55.3 | 83.5 | **51 %** |
  | ~3600 | 23.1 | 22.3 | 3.5 % |
  | ~4400 | 4.72 | 4.71 | **0.2 %** |

  **Drift's reproducibility depends entirely on how well the surrogate samples.** On a
  badly-mixed surrogate two chains from the same model differ by half; on a well-mixed one
  they agree to a fraction of a percent. Treat drift differences under ~2× as uninformative
  unless τ is small relative to the chain length.
