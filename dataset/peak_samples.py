import numpy as np

def quadratic_peak_estimate(x, y, ridge_eig=1e-3):
    """Estimate the maximum of y(x) from a quadratic response surface.

    Returns (x_star, model_value, rms_residual, n_projected). A log-likelihood is
    quadratic near its peak, so fitting

        y ~ c + b.z + 0.5 z^T H z,     z = (x - mean)/std

    over an existing sample and jumping to z* = -H^-1 b locates the peak without a
    dedicated optimisation. On the 29D banana this reaches within 2.9 log-units of the
    true maximum from the 2000 LHC points CLiENT evaluates anyway, where Nelder-Mead
    needs ~64,000 calls.

    model_value is the surface's own prediction at its vertex, not a fresh evaluation:
    the caller (a Keras model builder) has no likelihood to call. It overestimates by
    ~2.8 log-units here, which is the safe direction for a ceiling.

    Three details are load-bearing, each having cost a wrong answer:
      * features are z_i z_j for i <= j, so H_ii = 2 c_ii but H_ij = c_ij;
      * solve by SVD, never the normal equations -- A^T A squares a condition number
        already at ~2e4 for the Planck covariance;
      * flip non-negative eigenvalues preserving MAGNITUDE. Projecting them to -1e-6
        puts 1e6 into H^-1 along the flattest direction and the step explodes.
    """
    fit = _fit_quadratic(x, y, ridge_eig)
    z_star, coef, d = fit['z_star'], fit['coef'], fit['d']

    feat = [np.ones(1), z_star]
    for i in range(d):
        feat.append(z_star[i] * z_star[i:])
    model_value = float(np.concatenate(feat) @ coef)

    return (fit['mean'] + fit['std'] * z_star, model_value, fit['resid'],
            int((fit['w'] > 0).sum()))


def _fit_quadratic(x, y, ridge_eig=1e-3):
    """Least-squares quadratic surface in whitened coordinates, and its vertex.

    Factored out of quadratic_peak_estimate so that the vertex estimate and the Gaussian
    sampler below share one fit and cannot drift apart. Returns the whitening, the
    coefficients, the eigen-decomposition of the flipped Hessian, and the (radius-clipped)
    vertex. Behaviour is unchanged: quadratic_peak_estimate returns exactly what it did.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    n, d = x.shape

    mean, std = x.mean(0), x.std(0)
    std[std == 0] = 1.0
    z = (x - mean) / std

    cols = [np.ones((n, 1)), z]
    for i in range(d):
        cols.append(z[:, i:i + 1] * z[:, i:])
    A = np.hstack(cols)

    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = float(np.sqrt(np.mean((A @ coef - y) ** 2)))

    b = coef[1:1 + d]
    H = np.zeros((d, d))
    k = 1 + d
    for i in range(d):
        for j in range(i, d):
            if i == j:
                H[i, i] = 2.0 * coef[k]
            else:
                H[i, j] = H[j, i] = coef[k]
            k += 1

    w, V = np.linalg.eigh(H)
    w_neg = -np.maximum(np.abs(w), ridge_eig * np.abs(w).max())
    z_star = -np.linalg.solve(V @ np.diag(w_neg) @ V.T, b)

    radius = float(np.max(np.linalg.norm(z, axis=1)))
    nrm = float(np.linalg.norm(z_star))
    if nrm > radius:
        z_star *= radius / nrm

    return dict(mean=mean, std=std, z=z, coef=coef, b=b, H=H, w=w, w_neg=w_neg, V=V,
                z_star=z_star, resid=resid, radius=radius, d=d, n=n)


def gaussian_peak_samples(x, y, n_draw, temperature=1.0, bounds=None, seed=None,
                          ridge_eig=1e-3, stratify_radius=True, max_tries=50):
    """Draw training points from the Gaussian implied by the quadratic response surface.

    The vertex estimate x_star is an excellent guess at where the maximum is -- within
    2.6-3.5 log-units on the 16D banana, from data already in hand -- but injecting it as
    a single training point does not help and at 500 points actively hurts. The reason is
    a distance problem, not an accuracy one: x_star's log-likelihood sits hundreds of
    log-units above everything else in the initial LHC, so under the msre weighting it
    takes 96% of the loss and the network fits one point.

    The fix is to inject a *sample* rather than a point. If the likelihood is
    approximately Gaussian near its peak -- the same assumption that justifies the vertex
    estimate -- then the quadratic fit gives not just the location of the maximum but the
    full covariance, and one can draw points distributed exactly as a chain would be.

    Writing the fitted surface in whitened coordinates as

        logL(z) ~ logL(z*) - 0.5 (z - z*)^T P (z - z*),     P = -H,

    P is the precision (positive definite after the eigenvalue flip). The tempered
    posterior L^(1/T) is then Gaussian with the same mean and covariance T P^-1, so

        z = z* + sqrt(T) V diag(lambda^-1/2) u,     u ~ N(0, I),

    with (lambda, V) the eigen-decomposition of P. Drawing this way reproduces the density
    of a chain run at temperature T under the Gaussian approximation, which is the point:
    the injected points then span the same range of log-likelihood the sampler will visit,
    instead of piling up at the peak.

    That range is known in advance. The quadratic form is T |u|^2 with |u|^2 ~ chi^2_d, so
    the deficit below the vertex is 0.5 T chi^2_d -- mean 0.5 T d, standard deviation
    0.5 T sqrt(2d). At T=7 that is 56 +/- 20 log-units at d=16 and 102 +/- 27 at d=29,
    against an msre margin c of 19.2 and 28.7 respectively. So the sample lands a few c
    below the estimated peak and spread over a comparable width -- exactly the regime the
    weighting was designed for, and nothing like the several-hundred-log-unit gap a lone
    x_star opens up.

    stratify_radius draws |u| from stratified chi quantiles rather than at random. In high
    dimension a Gaussian sample concentrates in a thin shell at |u| ~ sqrt(d), and with
    n_draw of order 100 in 29 dimensions the radial coverage from a plain draw is poor.
    Stratifying keeps the exact radial law while removing the clumping; directions stay
    uniform on the sphere.

    Points outside `bounds` are rejected and redrawn, since clipping would pile mass onto
    the faces of the box. After max_tries rounds any shortfall is clipped instead, and the
    count is reported.

    Returns (x_new, info). info carries the predicted log-likelihood of each draw, the
    deficit statistics to compare against the law above, and the rejection bookkeeping ---
    none of which costs a likelihood evaluation.
    """
    fit = _fit_quadratic(x, y, ridge_eig)
    d, V, w_neg, z_star = fit['d'], fit['V'], fit['w_neg'], fit['z_star']
    rng = np.random.default_rng(seed)

    # P = -H has eigenvalues -w_neg > 0; the tempered draw scales by sqrt(T / lambda).
    scale = np.sqrt(temperature / (-w_neg))
    L = V * scale[None, :]                     # z - z* = L @ u

    feat = [np.ones(1), z_star]
    for i in range(d):
        feat.append(z_star[i] * z_star[i:])
    model_value = float(np.concatenate(feat) @ fit['coef'])

    lo = hi = None
    if bounds is not None:
        b = np.asarray(bounds, dtype=float)
        lo, hi = b[:, 0], b[:, 1]

    def draw(m):
        u = rng.normal(size=(m, d))
        if stratify_radius:
            from scipy.stats import chi2
            q = (rng.permutation(m) + 0.5) / m
            u *= (np.sqrt(chi2.ppf(q, d)) / np.linalg.norm(u, axis=1))[:, None]
        return u

    # Size each round from the acceptance seen so far. A wide fitted covariance against a
    # tight box can accept only a few percent, and drawing exactly n_draw a time then
    # exhausts max_tries and silently falls back to clipping.
    kept_u, tries = [], 0
    n_rejected = n_kept = 0
    batch = n_draw
    while n_kept < n_draw and tries < max_tries:
        u = draw(batch)
        xs = fit['mean'] + fit['std'] * (z_star + u @ L.T)
        if lo is None:
            ok = np.ones(len(xs), dtype=bool)
        else:
            ok = np.all((xs >= lo) & (xs <= hi), axis=1)
        n_rejected += int((~ok).sum())
        kept_u.append(u[ok])
        n_kept += int(ok.sum())
        tries += 1
        rate = max(n_kept / max(1, n_kept + n_rejected), 1e-3)
        batch = int(min(20 * n_draw, max(n_draw, np.ceil((n_draw - n_kept) / rate))))

    u = np.vstack(kept_u)[:n_draw] if kept_u else np.empty((0, d))
    n_clipped = 0
    if len(u) < n_draw:                        # box too tight for the fitted covariance
        pad = draw(n_draw - len(u))
        u = np.vstack([u, pad]) if len(u) else pad
        n_clipped = len(pad)

    x_new = fit['mean'] + fit['std'] * (z_star + u @ L.T)
    if lo is not None:
        x_new = np.clip(x_new, lo, hi)

    chi2_draw = np.sum(u * u, axis=1)
    deficit = 0.5 * temperature * chi2_draw
    info = dict(
        x_star=fit['mean'] + fit['std'] * z_star,
        model_value=model_value,
        predicted_loglkl=model_value - deficit,
        deficit=deficit,
        deficit_mean_expected=0.5 * temperature * d,
        deficit_sd_expected=0.5 * temperature * np.sqrt(2.0 * d),
        resid=fit['resid'],
        n_projected=int((fit['w'] > 0).sum()),
        n_rejected=n_rejected,
        n_clipped=n_clipped,
        acceptance=len(u) / max(1, n_rejected + len(u)),
    )
    return x_new, info
