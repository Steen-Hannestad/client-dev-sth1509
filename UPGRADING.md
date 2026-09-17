# Upgrading to client-dev-sth1509

Every change in this version relative to the upstream CLiENT it was forked from, what it
does to your results, and what you have to do about it.

**Read §1 first.** Four changes are on by default and three of them change results.
Annealing (§1.2) was turned OFF on 17 September after it failed badly on a sparse
58D design; warm-starting (§1.3) replaced it as the training-side default.

Depth lives elsewhere and is cross-referenced throughout: `CHANGES.md` for the iterative
algorithm with the experiments behind it, `PERFORMANCE.md` for the speed and memory work
with its measurements, `PACKAGE.md` for what was deliberately left out.

---

## 1. Changes that are ON by default

### 1.1 `training.msre_ess_floor: 0.1` — dynamic `c`

**Changes results.** The `msre` loss weights point *i* by `1/(max_loglkl − y_i + c)²`. With
the original fixed `c` the weights can collapse onto one or two points whenever the initial
design misses the peak by a lot — on a 29D target that scored **260× the noise floor**. The
floor raises `c` until the weights carry an effective sample size of at least
`msre_ess_floor × n`, then leaves it alone.

It anneals itself away. On the 31D ridge `c/c₀` ran ×145 → ×22.5 → ×4.43 → ×1.00 over three
iterations and the floor never engaged again; on a 58D Gaussian it took eleven. Once the
training set covers the peak the floor costs nothing.

*To restore the old behaviour exactly:* `msre_ess_floor: 0.0`.

Every run prints what it did, so you can always see whether it was active:

```
[msre] ESS floor 0.10: c 30.0 -> 4357.6 log-units (x145.04); ESS 53.3 -> 210.0 of 2100
```

### 1.2 `training.lr_schedule: none` — annealing, now OFF by default

> **DEFAULT REVERTED TO `none` ON 17 SEPTEMBER 2026.** Everything below was measured on a
> *dense* training set and still holds there. On a **sparse** one it fails, and the initial
> design of any high-dimensional run is sparse by construction. Measured on 58D iteration 0
> (2,100 points), three seeds each:
>
> | | seed 42 | seed 43 | seed 44 | spread |
> |---|---:|---:|---:|---:|
> | fixed 1e-4 | 1.4977e-02 | 1.8308e-02 | 1.4500e-02 | 1.26× |
> | annealed | 3.2887e-02 | 1.3641e-01 | 3.8296e-02 | 4.15× |
>
> The sets do not overlap — the worst fixed run beats the best annealed run by 1.80×,
> complete separation at one-sided p = 0.05 — and annealing is four times less predictable.
>
> **The consequence is not local.** On sparse data `val_loss` is a wide noise ball (a factor
> of two between neighbouring epochs), the plateau detector reads that as convergence, and
> the rate collapses. A 2.2× worse iteration-0 emulator then produced an acquisition that
> *dispersed* instead of contracting (selected-point spread 1.32 against 0.18), a
> reproducible iteration-1 training failure, and by iteration 10 a posterior sitting
> entirely outside the truth's 95% contour with 24 of 58 parameters above ΔCM 3.5 against
> the fixed-rate arm's 7. An offline replay cleared the walker count, thinning, chain length
> and the batched selector; swapping only the iteration-0 emulator moved the spread
> 1.203 → 0.108.
>
> Set `lr_schedule: plateau` deliberately, on a target whose initial design is known to be
> dense.

**Changes results, for the better — and it is the only change tested in this project where
a training-side gain carried through to the credible metric.** The rate is reduced by
`lr_factor` whenever `val_loss` stalls for `lr_patience` epochs, down to `lr_min`, and
training then stops `lr_grace` epochs later.

At a matched epoch budget on the 31D ridge, against the previous fixed 1e-4:

| | fixed | annealed |
|---|---:|---:|
| validation loss | 1.048e-04 | **4.901e-05** (2.1× lower) |
| median ΔCM | 0.0322 | **0.0135** (2.4× better) |
| max ΔCM excl. `x_30` | 0.2345 | **0.0696** |
| better on | — | **27 of 31 parameters** |

Six parameter differences clear twice the metric's noise floor in annealing's favour and
**none** clears it the other way; both annealed seeds beat both fixed seeds on every
aggregate. For contrast, input whitening bought 18–45× on validation loss and made ΔCM
*worse* — which is why the metric, not the loss, is what qualifies a change here.

**The mechanism is visible in the loss curve.** Over the 250 epochs before the best epoch,
the coefficient of variation falls from **124% to 24%** and the number of those epochs
within 10% of the best rises from **2 to 97**. At a fixed rate the optimiser was sampling a
wide noise ball and `restore_best_weights` was catching a lucky dip; annealed, it settles.
That also removes an uncontrolled source of run-to-run variance, since the shipped weights
are no longer a fortunate draw.

**`patience` is unused while this is on.** A plain `EarlyStopping` cannot terminate an
annealed run: the smoothed curve creeps downward monotonically, so with `min_delta=0` every
epoch resets the counter and it never fires — an annealed arm ran past 1500 epochs while
its fixed twin stopped at 1184. A relative-improvement test does not fire either ("no 1%
gain over 50 epochs" never triggered on either recorded curve). Hence the schedule-gated
stopper, `training/training.py::StopAfterScheduleExhausted`.

`lr_grace: 100` is a deliberate trade: simulated on the recorded curves it saved ~99 epochs
for a 1.11× val_loss cost, and a 2.01× val_loss spread within the annealed family moved
ΔCM by 9% — far below its 0.025 floor. Grinding out the tail buys a number that does not
predict the metric.

*To restore the old behaviour exactly:* `lr_schedule: none`.

**On the hyper-parameters.** A 2×2 factorial over `lr_patience` 25/50 and `lr_min`
1e-6/1e-7 (one seed) found `lr_patience: 25` **catastrophic — 26× worse loss**, because it
reaches the floor by epoch 205 and freezes the rate before the optimiser has descended; its
305 epochs are cheap only because the run is dead. `lr_min: 1e-7` was rejected at 2% of
loss for 21% more epochs. The failure mode is annealing *too early*, so the shipped
`lr_patience: 75` sits on the cautious side of it.

**`lr_patience: 75` is an extrapolation, not a measured optimum.** The factorial shows
25 ≪ 50; it does not show 75 > 50. Worth measuring.

#### Across a whole iterative loop, 16 September 2026

Everything above was measured on a *single fixed training set*. This is the first
measurement across a full iterative loop, where `c` re-solves every iteration and the loss
target therefore moves. Same 31D ridge, iterations 0–15, against the pre-annealing arm.
The two runs share a **byte-identical iteration-0 training set**, so this is a controlled
comparison rather than a matched one.

| | pre-optimisation | optimised | |
|---|---:|---:|---|
| epochs run | 21,724 | **15,319** | 1.42× fewer |
| wall clock | 9.52 h | **6.88 h** | 1.38× (1.46× less scoring contention) |
| peak RSS | — | **3.55 GiB** | measured under `/usr/bin/time -l` |
| median ΔCM, it15 | 0.0209 | **0.0084** | 2.49× |
| **max ΔCM excl. `x_30`, it15** | **0.0754** | **0.0182** | **4.14×** |
| max ΔCM at 95%, it15 | 0.0399 | **0.0128** | 3.12× |

The quality result holds up: annealing's gain survives a changing training set, and
`max excl. x_30` is the figure to trust — the median is floor-limited on both arms, and
this run's chains carry ~2× the effective sample size, which lowers its own floor by ~√2.

**Three findings that cut against the change, recorded because they are load-bearing:**

1. **The speed prediction missed by roughly a factor of two.** 2.7× fewer epochs was
   predicted; 1.42× was delivered. The cause is visible in the learning-rate ladder:
   **51–68% of every iteration is spent at the initial rate**, before the first reduction
   ever fires, descending `val_loss` from ~1.2 to ~1e-3. That is genuine descent, not a
   stalled counter, and **no stopping rule can shorten it**. The 2.7× came from
   generalising one measurement taken on the iteration-15 training set — where the descent
   phase is short because the network is fitting 32,100 points — to all sixteen iterations.

2. **The epoch saving inverts at late iterations.** The annealed schedule has a floor of
   four reductions at `lr_patience: 75` plus `lr_grace: 100`. Once the emulator converges
   enough for fixed-rate `EarlyStopping` to fire early, the baseline is *cheaper*:
   0.93×, 0.83× and 0.68× at iterations 11, 12 and 15. It is paid for — `val_loss` is
   2.3–3.2× better at exactly those iterations — but expect no epoch saving late in a run,
   and **treat this as a live reason to measure `lr_patience: 50` against the shipped 75.**

3. **The absolute `min_delta` makes behaviour depend on loss magnitude.** `ReduceLROnPlateau`
   defaults to `min_delta=1e-4` *absolute*, so as `val_loss` falls the same improvement
   stops counting. The initial-rate phase shrank from 695 to 347 epochs across the run for
   this reason alone. A *relative* `min_delta` is worth considering.

**Where the speed actually came from:** the sampling-side changes, which beat their
prediction. 1.35× faster per pass at double the walkers, ~2× the effective sample size, and
2.5× less chain memory. The training-side contribution is real but smaller than advertised.

### 1.3 `training.warm_start: true` — inherit the previous iteration's weights

**Changes results, for the better, and is the first training-side change tested here that
helps on BOTH targets.** Iteration *N* starts from iteration *N−1*'s trained weights instead
of a random initialisation.

Two full runs, each matched to its own baseline on every setting except this one:

| target | epochs run | it15 median ΔCM | it15 max |
|---|---|---|---|
| 58D CLOE ±10σ | **7,852 vs 23,187** (2.95×) | **0.2318** vs 0.2551 | **0.6523** vs 1.0598 |
| 31D ridge | **8,539 vs 21,724** (2.54×) | **0.0081** vs 0.0209 | **0.0565** vs 0.0754 |

On 58D iteration 10 the worst parameter came in at **1.1704 against the baseline's 9.4522**,
with **zero** of 58 parameters above ΔCM 3.5 where the baseline had seven — roughly the
baseline's *iteration-15* quality reached at *iteration 10*.

**Only the trainable weights are inherited.** `build_model` re-adapts the input
`Normalization` and rebuilds the output `TargetDenormalization` from the *current*
iteration's data; both are statistics that move as the training set grows. Carrying them
over as well ("warm-continue") was measured and is not better — it starts far closer
(initial `val_loss` 4.97e-04, already beating a cold network's *converged* 3.96e-04) and
finishes no better, while accumulating stale statistics.

**Iteration 0 cannot be warm started**, which makes it a free control: with nothing to
inherit it must reproduce the baseline exactly. It did on both runs to four digits (58D
2071 epochs at 1.4977e-02; 31D 1697 at 6.8004e-04).

**The obvious objection did not materialise.** `c` re-solves every iteration, so inherited
weights are optimal for a different loss; the feared signature was `val_loss` degrading as
they age. It improved instead — `TargetDenormalization` is rebuilt from the current target
distribution each iteration, so the output is rescaled before a single gradient step, and
what the weights carry is the *shape* of the function, which changes slowly.

*Known blemish:* the 31D run is not monotone — its iteration-10 median (0.0565) is worse
than both its own iteration 5 and the baseline's iteration 10, though both sit near the
~0.018–0.026 measurement floor and its maxima are better at both iterations. One seed per
run.

*To restore cold starts:* `warm_start: false`.

### 1.4 `acquisition.batch_size: 10` — batched candidate selection

**Changes which points are selected, but not their distribution.** The density-deficit
selector used to commit one candidate per pass over the candidate pool. It now commits ten,
drawn jointly, with their pool distances from one matrix product instead of ten.

On a 32 100-point request the selection went from **3 483 s to 277 s (12.6×)**. The cost is
that batch members do not see the density each other contributes, so roughly 90% of the
selected points differ from the exact algorithm's — but per-parameter KS against the exact
selection is below 0.01 on all 31 parameters, and intra-set nearest-neighbour spacing
matches to 0.4%, so the two are draws from the same density.

*If you need the exact sequential algorithm* — reproducing a run made before this option
existed, or matching an arm that used it — set `batch_size: 1`. That is still **3.97×**
faster than upstream, because the per-pass optimisations below are exact.

**Caveat worth knowing:** this was validated on one target at one batch size. The
approximation's error grows with `batch_size` and with how concentrated the deficit
landscape is. Beyond about 50 it also gets *slower*, because the pool × batch distance
block leaves cache. `PERFORMANCE.md` §1 has the check to repeat on a new target.

### 1.5 Sampler memory: the preallocated chain buffer

**Does not change results.** The chain used to be accumulated as per-chunk tensors,
concatenated, then copied again, with all three alive at once — a high-water mark of
**2.93× one chain**. Chunks now go straight into a preallocated array.

| 216 walkers × 30 000 × 16D | peak RSS |
|---|---:|
| before | 1.589 GiB |
| after | **0.983 GiB** |

**This one has an API consequence.** `sampler.chain()` and `sampler.log_prob()` now return
**numpy views into the sampler's buffer**, not TensorFlow tensors. If you have code that
touches the sampler directly:

* do **not** call `.numpy()` on them — they are already numpy;
* do **not** `sampler.reset()` while still holding them, or you are reading freed memory.

`client.py` releases the chain, the log-probabilities *and* the sampler together for this
reason, and says so at the call site.

---

## 2. Changes that are OFF by default

Each is inert unless you set it.

### 2.1 `prior.n_inject` — peak injection at iteration 0

Draws this many extra initial points from the Gaussian implied by a quadratic fit to the
initial design, at the acquisition temperature. Each costs one true likelihood call.

It reliably speeds convergence and does **not** reliably improve accuracy — see §7 of
`CHANGES.md`, which records a case where it was 2.6× better in drift and marginally *worse*
on the credible metric. It is off by default for that reason.

**Read this before enabling:** the injected points are heavily weighted by the `msre` loss,
so injection interacts with §1.1. Keep `n_inject > alpha·n_samples/(1−alpha)` where `alpha`
is `msre_ess_floor`, or the floor will fight the injection.

### 2.2 `acquisition.n_inject` — continuous re-injection

The same idea every iteration, re-fitting the quadratic on the whole accumulated training
set rather than the initial design alone. Costs `n_inject` true evaluations per iteration.

**It is not uniformly beneficial, and the reason is geometric.** Measured at iteration 10 on
the 31D ridge against an otherwise identical arm:

| | `x_28` | `x_29` | `x_30` | `x_31` |
|---|---:|---:|---:|---:|
| W₁ to truth, ratio re-inject / inject-only | 0.73× | **0.25×** | 1.62× | **5.29×** |

The banana block improves up to fourfold; the ridge block degrades up to fivefold. Injection
draws from a *quadratic*, and on a parabolic ridge a quadratic fit puts its vertex **off the
ridge** — at (0.11, 5.17), where the true log-likelihood is −13.3 rather than 0, with a 24
log-unit residual. So every iteration deposits points where the degeneracy is not.

**Rule of thumb:** useful where a quadratic can describe the peak region; harmful on curved
degeneracies. If your target has one, measure before trusting it.

### 2.3 `sampling.thin` — thinning at write time

Keeps every *k*-th step as sampling proceeds, so a thinned chain never allocates the
unthinned one. The sampler has supported this since the buffer change, but **no config field
existed until this version**, so it was unreachable.

It is what makes a larger `n_walkers` affordable:

| 100 000 steps, 31D | thin 1 | thin 5 |
|---|---:|---:|
| 216 walkers | 2.49 GB | 0.50 GB |
| 432 walkers | 4.99 GB | **1.00 GB** |

**Thinning does not reduce how many autocorrelation times a chain holds.** Rows and τ shrink
by the same factor, so `rows/(τ/thin) = n_steps/τ` is invariant. The constraint is instead
`thin ≪ τ`, so the thinned series still resolves the correlation. On these targets τ is
800–5000, so `thin: 5` has two to three orders of magnitude of headroom. `max(tau)` is
printed every run — check it rather than assuming.

### 2.4 `sampling.adaptive` — stop on the autocorrelation criterion

`n_steps` becomes a maximum; sampling stops once the chain holds `ess_target`
autocorrelation times and τ has settled to within `delta_tau_tol`. Off by default because a
fixed length is predictable.

**`max(tau)` is now reported after every run whether or not this is enabled**, at about 0.7 s
— 0.4% of a sampling step. This matters more than the stopping rule: it is the only way to
see whether a fixed-length chain was long enough. It immediately showed that the shipped 16D
default of 30 000 steps delivered only **36 τ**, which had been invisible.

### 2.5 `seed` — top-level reproducibility

Fixes the prior design, the network initialisation, the walker starting positions and the
acquisition draws. **It does not fix the AIES proposals**: those run under XLA, which
discards seeds. Two seeded runs are statistically identical, not bit-identical. Every run
prints this so it cannot be mistaken:

```
Seed: 42 (AIES proposals excepted -- XLA discards them)
```

---

## 3. Shipped config defaults that changed

The research configs in `input/` now ship with `n_walkers: 432` and `thin: 5` in place of
216 and unthinned. Measured on the 31D ridge at matched `n_steps`:

| | 216 walkers | 432 walkers |
|---|---:|---:|
| max τ | 3678 | **3610** (0.98×) |
| effective sample size | 2 936 | **5 984** (2.04×) |
| chain buffer, thin 5 | 0.25 GB | 0.50 GB |

**τ does not move**, so doubling the ensemble is free: the network is nowhere near
saturating at a batch of `n_walkers/2`, and 400 steps at 432 walkers took no longer than at
216. Twice the effective sample size tightens the measurement floor by √2, and combined with
`thin: 5` the buffer is 40% of what the old default used.

**If you are comparing against results produced at 216 walkers**, either keep 216 or
recompute your noise floor from the new chain — twice the ESS means a different floor, so
the same ΔCM value is not measuring the same thing.

---

## 4. Behavioural details that surprise people

None of these are new to this version, but all of them cost somebody time.

**`-i` disables the convergence criterion.** Passing `-i` sets `use_convergence` false, so
`convergence.threshold` cannot stop the run early. Without `-i` the run may stop as soon as
drift dips below the threshold. On the 31D ridge drift finished at 0.101 against a threshold
of 0.01, so that run would never have terminated on its own — but on an easier target it
might stop at iteration 3.

**`-i` counts differently on a continued run.** For a new run `-i N` gives iterations
`0 … N−1`. On a continue, `n_iterations = N + 1`, so `-i 5` from iteration 15 reaches 20 and
`-i 0` re-runs the start iteration alone.

**A continued run restarts at the last iteration, not the next one.** Acquisition is gated on
`iteration < final_iteration`, so the final iteration of any run never acquires and
`data_it_{N+1}.csv` is never written. `_start_iteration` takes `max(latest model, latest
data)`, and re-running iteration N regenerates the missing acquisition. The network is **not**
retrained there — the checkpoint is loaded — so the cost is one sampling pass plus
acquisition.

**Drift is not an accuracy metric.** `gaussian_posterior_drift` measures how much the
surrogate's sampled posterior moves between iterations. It carries a **factor-of-two
scatter** on a plateau: over one arm's iterations 8–14 it ran 0.128, 0.129, 0.135, 0.172,
0.202, 0.148, 0.100 while the emulator improved by every independent measure. Use it to see
that a run is progressing, never to rank two runs.

---

## 5. New likelihood targets

`input/cobaya/` carries every target used in the work behind these changes. All are analytic
or inline-Gaussian: **no Boltzmann solver, no clik, no MontePython** is needed for any of
them, despite what `README.md`'s prerequisites imply.

| target | dim | what it is for |
|---|---:|---|
| `banana_planck.yaml`, `_wide` | 29 | Planck-shaped Gaussian + hyperbolic banana |
| `banana_planck31.yaml`, `_wide` | 31 | the above **plus a parabolic ridge** — the hard case |
| `banana_lcdm16.yaml`, `_wide` | 16 | reduction; the banana block factorises exactly |
| `gaussian_planck.yaml` | 27 | the shipped Planck covariance |
| `gaussian_planck25*.yaml` | 25 | conditioned sub-models of the 27D |
| `gaussian_cloe56/58.yaml` | 56/58 | Euclid-shaped, the largest tested |

The 31D ridge deserves a note because it is the target most of §2.2's warning comes from.
Its penalty is `−½[(x₃₁ − x₃₀²)]²`, so its maximum set is a *curve* rather than a point or a
straight degeneracy, and it was built to defeat quadratic approximations — which is exactly
what peak injection is. `docs/` in the source project carries the full write-up.

---

## 6. Verifying an install

```bash
python client.py input/_smoke.yaml -n smoke -i 2
```

About 20 s on the 16D analytic banana. It exercises the LHC design, iteration-0 injection,
dynamic `c`, training, thinned sampling, the drift metric, batched selection and continuous
re-injection. You should see `[msre] ESS floor` engage, an `it 1: Injecting` line, and a
final `gaussian_posterior_drift`.

---

## 7. What is deliberately not here

Input whitening, model transfer, and the Muon / K-FAC-preconditioned optimisers were all
investigated and are all excluded — each was either measurably harmful or measurably slower.
`PACKAGE.md` gives the numbers. If you find references to them in a docstring or a comment,
they are explanatory, not a promise that the feature exists.
