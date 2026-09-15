import numpy as np
import tensorflow as tf


def _chi2_quantile_wilson_hilferty(sigma_level, chi2_dof):
    # Approximate a chi-squared quantile using the Wilson-Hilferty transform.
    # sigma_level is the equivalent standard-normal sigma level.
    center = 1.0 - 2.0 / (9.0 * chi2_dof)
    width = tf.sqrt(2.0 / (9.0 * chi2_dof))
    return chi2_dof * tf.pow(center + sigma_level * width, 3.0)


def _msre_ess(targets, max_loglkl, c):
    """Effective sample size of the msre weights w_i = 1/(max_loglkl - y_i + c)^2."""
    w = 1.0 / (targets - max_loglkl - c) ** 2
    w = w / w.sum()
    return 1.0 / np.sum(w**2)


def solve_ess_floor(targets, max_loglkl, c0, alpha, max_grow=1e8):
    """Smallest c >= c0 whose msre weights have effective sample size >= alpha * n.

    msre weights each training point by 1/(max_loglkl - y_i + c)^2, so c sets how far
    below the best point the loss still cares. c0 = 0.5*dchi2(sigma_level, ndim) is a
    fixed 19.2 log-units at ndim=16, sigma_level=3, chosen as a distance below the TRUE
    maximum -- but the code has only the best point of the current training set, and at
    iteration 0 on a wide prior box that is hundreds of log-units short of the truth.
    Every deficit then dwarfs c and the weights collapse onto one point. Measured on a
    2000-point LHC over the wide +/-30 sigma box: the best point sits 802 log-units below
    the true maximum, ESS is 1.5 of 2000, and the single best point takes 80.7% of the
    loss while the other 1990 share 3.9%.

    Raising c until ESS reaches alpha*n spreads the weight back. It binds only while the
    data is sparse: as acquisition fills the peak region the empirical maximum rises,
    ESS(c0) climbs past the floor on its own and c returns to c0 exactly. There is no
    schedule, and alpha=0 reproduces the untouched loss bit for bit.

    ESS is monotone increasing in c -- raising it moves every pairwise weight ratio
    toward one -- so bisecting upward from c0 has a unique root, and c can only ever
    increase, so weight can only spread.
    """
    targets = np.asarray(targets, dtype=float).ravel()
    target_ess = alpha * len(targets)
    if alpha <= 0.0 or _msre_ess(targets, max_loglkl, c0) >= target_ess:
        return c0

    hi = max(c0 * 2.0, 1e-6)
    while _msre_ess(targets, max_loglkl, hi) < target_ess:
        hi *= 2.0
        if hi > max_grow * max(c0, 1e-12):
            return hi                     # degenerate data; give up rather than hang
    lo = c0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if _msre_ess(targets, max_loglkl, mid) < target_ess:
            lo = mid
        else:
            hi = mid
    return hi


def _build_msre(sigma_level, chi2_dof, max_loglkl, targets=None, ess_floor=0.0):
    half_chi2_quantile = 0.5 * float(_chi2_quantile_wilson_hilferty(sigma_level, chi2_dof))

    if ess_floor and ess_floor > 0.0 and targets is not None:
        y = np.asarray(targets, dtype=float).ravel()
        c = solve_ess_floor(y, float(max_loglkl), half_chi2_quantile, float(ess_floor))
        w = 1.0 / (y - float(max_loglkl) - c) ** 2
        print(
            f"[msre] ESS floor {ess_floor:.2f}: c {half_chi2_quantile:.1f} -> {c:.1f} "
            f"log-units (x{c / half_chi2_quantile:.2f}); ESS "
            f"{_msre_ess(y, float(max_loglkl), half_chi2_quantile):.1f} -> "
            f"{_msre_ess(y, float(max_loglkl), c):.1f} of {len(y)}, "
            f"top point {100 * w.max() / w.sum():.2f}%",
            flush=True,
        )
        half_chi2_quantile = c

    def msre(true_loglkl, pred_loglkl):
        loglkl_scale = true_loglkl - max_loglkl - half_chi2_quantile
        relative_loglkl_error = (pred_loglkl - true_loglkl) / loglkl_scale
        return tf.reduce_mean(tf.square(relative_loglkl_error))

    return msre


def build_loss(name, sigma_level=None, chi2_dof=None, max_loglkl=None, targets=None,
               ess_floor=0.0):
    if name == "msre":
        return _build_msre(sigma_level, chi2_dof, max_loglkl, targets, ess_floor)
    return name
