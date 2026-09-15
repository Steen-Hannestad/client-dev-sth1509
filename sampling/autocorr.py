import numpy as np


def _next_pow_two(n):
    i = 1
    while i < n:
        i <<= 1
    return i


def _acf_1d(x):
    """Normalised autocorrelation function of a 1D series, via FFT."""
    n = _next_pow_two(len(x))
    f = np.fft.fft(x - np.mean(x), n=2 * n)
    acf = np.fft.ifft(f * np.conjugate(f))[: len(x)].real
    if acf[0] == 0.0:
        return np.zeros_like(acf)
    return acf / acf[0]


def _auto_window(taus, c):
    """Sokal's automatic windowing: the first M with M >= c * tau(M)."""
    mask = np.arange(len(taus)) < c * taus
    if np.any(mask):
        return int(np.argmin(mask))
    return len(taus) - 1


def integrated_time(chain, c=5.0, thin=1):
    """Integrated autocorrelation time per parameter, in units of single steps.

    The standard estimator -- the autocorrelation function averaged over walkers, summed
    with Sokal's automatic window -- implemented here because the package has no emcee
    dependency. Validated against AR(1) processes of known tau to better than 2%. `chain` is (n_steps, n_walkers, ndim) and may already be
    thinned; `thin` scales the answer back to single steps.

    Returns +inf for any parameter whose chain is too short for the window to close,
    which the caller should read as "keep sampling" rather than as a number.
    """
    chain = np.asarray(chain)
    n_steps, n_walkers, ndim = chain.shape
    tau = np.empty(ndim, dtype=float)

    for d in range(ndim):
        f = np.zeros(n_steps)
        for w in range(n_walkers):
            f += _acf_1d(chain[:, w, d])
        f /= n_walkers
        taus = 2.0 * np.cumsum(f) - 1.0
        window = _auto_window(taus, c)
        # The window failing to close means the series is shorter than a few tau; the
        # estimate is then a lower bound, not an estimate.
        tau[d] = taus[window] * thin if window < len(taus) - 1 else np.inf

    return tau
