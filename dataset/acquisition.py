import numpy as np
from scipy.special import gammaln, logsumexp

from dataset.dataset import _WhitenedKNN


def _log_unit_ball_volume(ndim):
    return (ndim / 2.0) * np.log(np.pi) - gammaln(ndim / 2.0 + 1.0)


def _deduplicate(chain, logposts):
    chain = np.asarray(chain)
    logposts = np.asarray(logposts)
    points = []
    point_logposts = []
    counts = []
    for walker in range(chain.shape[1]):
        x = chain[:, walker, :]
        y = logposts[:, walker]
        starts = np.r_[True, np.any(x[1:] != x[:-1], axis=1)]
        start_indices = np.flatnonzero(starts)
        points.append(x[start_indices])
        point_logposts.append(y[start_indices])
        counts.append(np.diff(np.r_[start_indices, len(x)]))

    return (
        np.concatenate(points, axis=0),
        np.concatenate(point_logposts, axis=0),
        np.concatenate(counts, axis=0),
    )


def select_points(dataset, chain, logposts, n_append, mcmc_temperature,
                  pool_factor, blas_distances=True, batch_size=10):
    knn_index = dataset.knn_index
    ndim = knn_index.ndim
    n_neighbors = dataset.n_neighbors
    target_temperature = dataset.target_temperature
    n_current = len(dataset.inputs)

    # log w_i ∝ (1/T - 1/T_MC) * loglkls_i
    # Since logposts = 1/T_MC * loglkls, we have
    # log w_i ∝ (T_MC/T - 1) * logposts_i
    log_weight_coeff = mcmc_temperature / target_temperature - 1.0

    # Collapse exact duplicate MCMC states, but preserve their total probability mass
    chain_unique, logposts_unique, counts = _deduplicate(chain, logposts)
    log_weights_unique = log_weight_coeff * logposts_unique + np.log(counts)

    # Normalize the unique-point weights
    log_weights_unique = log_weights_unique - logsumexp(log_weights_unique)
    weights = np.exp(log_weights_unique)

    # Draw the candidate pool and a disjoint reference sample in a single weighted
    # draw without replacement. Both are then samples from the target density
    # q ∝ L^(1/T); drawing them disjointly keeps a pool point from turning up as its
    # own nearest neighbor in the reference index below.
    n_available = int(np.count_nonzero(weights))
    pool_size = min(pool_factor * n_append, n_available)
    n_reference = min(n_current, n_available - pool_size)
    if n_reference <= n_neighbors:
        raise ValueError(
            f"Only {n_available} chain points carry nonzero weight, leaving "
            f"{n_reference} for the reference sample after a pool of {pool_size}; "
            f"at least {n_neighbors + 1} are needed. Lengthen the chain "
            f"(sampling.n_steps) or lower acquisition.pool_factor."
        )
    draw = np.random.choice(
        a=len(chain_unique), size=pool_size + n_reference, replace=False, p=weights
    )
    pool_indices, reference_indices = draw[:pool_size], draw[pool_size:]
    pool = chain_unique[pool_indices]
    reference = chain_unique[reference_indices]

    # Calculate the whitened coordinates for later distance computations
    pool_whitened = knn_index.whiten(pool)
    pool_rows_init = np.arange(pool_size)

    # The k-NN estimator does not report the density *at* a point: it reports the
    # density averaged over a ball enclosing a fixed mass fraction k/N, and in high
    # dimensions that ball spans much of the distribution (r_k ≈ 0.9 √n at n = 30).
    # The estimate is therefore flattened, log ρ̂ ≈ s log ρ + c with s < 1 (s ≈ 0.64 at
    # n = 30, k = 20). Comparing an *analytic* target density against an *estimated*
    # current density would let that exponent through and drive the training set to
    # L^(1/(sT)) rather than L^(1/T) -- systematically colder than requested.
    # Estimating both sides with the same estimator, the same k and the same whitening
    # makes it cancel: log D̂ = s [log ρ_target - log ρ_merged] vanishes exactly where
    # the true ratio is one, whatever s happens to be. A wrong s then only compresses
    # the size of the deficit signal, i.e. the rate of convergence, not its target.
    log_ball_vol = _log_unit_ball_volume(ndim)
    reference_index = _WhitenedKNN(
        reference, n_neighbors, transform=knn_index.transform
    )
    reference_distances, _ = reference_index.query(pool)
    r_k_reference = np.max(reference_distances, axis=1)

    # ρ_ref = k / (V_n * r_{k,ref}^n), the number density of the n_reference-point
    # reference cloud. The selection loop rescales it to the point count of the merged
    # training set, which grows as points are appended.
    log_rho_reference = (
        np.log(n_neighbors) - log_ball_vol - ndim * np.log(r_k_reference)
    )

    # Compute the k-th nearest neighbor distances for the pool points.
    #
    # Stored SQUARED from here on. Nothing downstream needs the root: the estimator wants
    # ndim * log r_k, which is (ndim/2) * log r_k^2, and the eviction test is a comparison
    # that sqrt (monotonic) does not affect. max_i sqrt(d2_i) == sqrt(max_i d2_i) bit-for-bit
    # because sqrt is monotonic and correctly rounded, so r_k is unchanged.
    pool_distances, _ = knn_index.query(pool)
    current_distances = (pool_distances ** 2)

    # ||x_i||^2 for the BLAS distance identity below: loop-invariant, computed once.
    pool_sq_norms = np.einsum('ij,ij->i', pool_whitened, pool_whitened)

    # Cache the row-wise argmax and the k-th distance it points at. The baseline recomputed
    # both with a full pool_size x k max AND a full argmax every pass -- two passes over the
    # same array computing the same quantity. Here the pair is maintained incrementally and
    # only rows whose k-th neighbour actually changed are re-maximised.
    worst_indices = np.argmax(current_distances, axis=1)
    r_k_sq_merged = current_distances[pool_rows_init, worst_indices]
    current_flat = current_distances.reshape(-1)

    selected_indices = []
    selected_mask = np.ones(pool_size, dtype=bool)

    n_to_select = min(n_append, pool_size)
    n_selected = 0
    while n_selected < n_to_select:
        # r_k is carried in the cache; ndim*log(r_k) == (ndim/2)*log(r_k^2).
        log_rho_merged = (
            np.log(n_neighbors) - log_ball_vol - (0.5 * ndim) * np.log(r_k_sq_merged)
        )

        # Compute the next-point target number density by rescaling the reference
        # cloud's density from n_reference points to n_current + 1.
        #
        # This rescale is deliberately the naive one, and it is not what an unbiased
        # estimator would need: because the estimate is flattened, the estimator's
        # response to scaling a density by λ is s log λ, not log λ, so a full log factor
        # overshoots by (1 - s) log[(n_current + 1) / n_reference]. That residual is
        # independent of θ, so it only shifts how eagerly points are added, never where
        # -- log D keeps the form (θ-independent constant) + s [log q - log ρ], whose
        # θ-dependence still vanishes exactly at ρ ∝ q for any s. It is zero at the
        # start of the batch (n_reference == n_current) and grows to (1 - s) log(1 +
        # n_append / n_current) by the end: 0.01 to 0.07 nats for batch fractions
        # between 4% and 20%.
        log_rho_target = np.log(n_current + 1) - np.log(n_reference) + log_rho_reference

        # Compute the number density ratio D = ρ_target / ρ_merged
        log_D = log_rho_target - log_rho_merged

        # Compute the positive number deficit retention factor
        # a = [1 - 1/D]_+ = [1 - e^{-logD}]_+
        a = np.zeros_like(log_D)
        positive = log_D > 0.0
        a[positive] = -np.expm1(-log_D[positive])
        a[~selected_mask] = 0.0

        # Normalize the retention factors and sample without replacement
        a_sum = np.sum(a)
        if not np.isfinite(a_sum) or a_sum <= 0.0:
            break
        a = a / a_sum

        # Draw `n_draw` candidates at once. batch_size=1 is the exact sequential
        # algorithm. batch_size>1 is an APPROXIMATION: every member of a batch is drawn
        # from the same deficit distribution, so none of them sees the density the others
        # contribute, and two can land in the same deficit region. It trades that for
        # turning `n_append` passes over the pool into `n_append/batch_size` of them, with
        # the per-point distances coming from one BLAS gemm instead of `batch_size` gemvs.
        n_draw = min(batch_size, n_to_select - n_selected, int(np.count_nonzero(a)))
        if n_draw <= 0:
            break
        if n_draw == 1:
            drawn = np.array([np.random.choice(a=pool_size, p=a)])
        else:
            drawn = np.random.choice(a=pool_size, size=n_draw, replace=False, p=a)
        selected_indices.extend(int(i) for i in drawn)
        selected_mask[drawn] = False
        n_current += n_draw
        n_selected += n_draw

        # SQUARED distance from every pool point to the newly selected point.
        #
        # ||x - v||^2 = ||x||^2 - 2 x.v + ||v||^2 turns a pool_size x ndim broadcast (which
        # materialises a 160 MB temporary and is purely memory-bound) into one BLAS gemv.
        # Exact in exact arithmetic; in float64 it agrees with the broadcast form to ~1e-15
        # relative at the r_k scale, and the cancellation that can reach 1e-2 relative is
        # confined to ||x - v||^2 -> 0, i.e. a point against itself or a near-duplicate,
        # orders of magnitude below any r_k. Clamped at zero because cancellation can
        # produce a small negative. sklearn's euclidean_distances uses the same identity.
        #
        # blas_distances=False restores the broadcast form, which makes the whole selection
        # bit-identical to the pre-optimisation implementation. See
        # tests/test_acquisition_equivalence.py.
        if blas_distances:
            # ||x - v||^2 = ||x||^2 - 2 x.v + ||v||^2. One gemv for a single draw, one
            # gemm for a batch. Replaces a pool_size x ndim broadcast that materialises a
            # 160 MB temporary and is purely memory-bound. Exact in exact arithmetic; in
            # float64 it agrees with the broadcast form to ~1e-15 relative at the r_k
            # scale, and the cancellation that can reach 1e-2 relative is confined to
            # ||x - v||^2 -> 0 -- a point against itself or a near-duplicate, orders of
            # magnitude below any r_k. Clamped at zero because cancellation can go
            # slightly negative. sklearn's euclidean_distances uses the same identity.
            #
            # blas_distances=False restores the broadcast form, which with batch_size=1
            # makes the selection bit-identical to the pre-optimisation implementation.
            if n_draw == 1:
                new_distance_sq = pool_sq_norms - 2.0 * (
                    pool_whitened @ pool_whitened[drawn[0]]
                ) + pool_sq_norms[drawn[0]]
            else:
                new_distance_sq = (
                    pool_sq_norms[:, None]
                    - 2.0 * (pool_whitened @ pool_whitened[drawn].T)
                    + pool_sq_norms[drawn][None, :]
                ).min(axis=1)
            np.maximum(new_distance_sq, 0.0, out=new_distance_sq)
        else:
            if n_draw == 1:
                new_distance_sq = (
                    (pool_whitened - pool_whitened[drawn[0]]) ** 2
                ).sum(axis=1)
            else:
                new_distance_sq = np.min([
                    ((pool_whitened - pool_whitened[i]) ** 2).sum(axis=1) for i in drawn
                ], axis=0)

        # Keep the k smallest distances from each pool point to the current training data
        # plus all selected points. Only rows that actually improve are touched, and only
        # those rows are re-maximised -- the baseline re-scanned all pool_size x k.
        #
        # For a batch only the NEAREST of the batch is inserted per row, so a row whose k
        # smallest should have absorbed two batch members keeps only one. That is the same
        # approximation as drawing the batch jointly, and it is bounded by batch_size.
        improved = np.flatnonzero(new_distance_sq < r_k_sq_merged)
        if improved.size:
            current_flat[improved * n_neighbors + worst_indices[improved]] = (
                new_distance_sq[improved]
            )
            improved_rows = current_distances[improved]
            w = np.argmax(improved_rows, axis=1)
            worst_indices[improved] = w
            r_k_sq_merged[improved] = improved_rows[np.arange(improved.size), w]

        # Selected points are not future query candidates.
        for i in drawn:
            current_flat[i * n_neighbors:(i + 1) * n_neighbors] = np.inf
        r_k_sq_merged[drawn] = np.inf
    metrics = {
        "n_unique": len(chain_unique),
        "max_multiplicity": int(np.max(counts)),
        "n_reference": n_reference,
    }

    return pool[selected_indices], metrics
