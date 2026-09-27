"""Stage 5 - STATISTICAL SIGNIFICANCE (novel part #4: empirical false-alarm
probability per star, not a fixed SNR threshold).

On TOI-907 the search returned a second "candidate" with SDE 13 that is pure
noise. A global threshold (SDE > 7, SNR > 7.1) cannot tell that apart, because
the noise differs from star to star. So for every candidate we build this
star's own null distribution:

  1. remove the candidate's transits
  2. make N null light curves that keep the real red noise and systematics
     but cannot contain a periodic transit:
        - the INVERTED light curve (dips become bumps, bumps become dips)
        - BLOCK-SCRAMBLED copies (1-day blocks shuffled within each orbit)
  3. run the identical BLS search on each -> null distribution of max SDE
  4. fit a Gumbel (extreme-value) law -> FAP(SDE_observed) -> Gaussian sigma

Reported per candidate: SNR, SDE, empirical FAP, sigma, and the calibrated
classifier probability. A detection is claimed only if FAP < 1e-3 (~3.1 sigma)
AND the planet probability passes its threshold.
"""
import numpy as np
from scipy import stats

from .detrend import segments
from .search import bin_series, bls_search, transit_mask


def null_curves(t, y, rng, n_scramble=9, block=1.0):
    yield "inverted", t, 2 * np.median(y) - y
    for _ in range(n_scramble):
        ys = y.copy()
        for s in segments(t):
            tt = t[s]
            k = np.floor((tt - tt[0]) / block).astype(int)
            blocks = [np.flatnonzero(k == i) for i in np.unique(k)]
            order = rng.permutation(len(blocks))
            ys[s] = np.concatenate([y[s][blocks[i]] for i in order])[: tt.size]
        yield "scrambled", t, ys


def empirical_fap(time, flux, cand, rng=None, n_scramble=9, pmin=0.5, pmax=None):
    rng = rng or np.random.default_rng(0)
    t, y = bin_series(time, flux)
    m = transit_mask(t, cand["period"], cand["t0"], cand["duration"], pad=2.0)
    t, y = t[~m], y[~m] - np.median(y[~m])
    null = []
    for _, tn, yn in null_curves(t, y, rng, n_scramble):
        r = bls_search(tn, yn, pmin, pmax)
        if r is not None:
            null.append(r["sde"])
    null = np.array(null)
    if null.size < 3:
        return dict(fap=np.nan, sigma=np.nan, null_sde_mean=np.nan, null_sde_max=np.nan)
    loc, scale = stats.gumbel_r.fit(null)
    fap = float(stats.gumbel_r.sf(cand["sde"], loc, scale))
    fap = max(fap, 1e-15)
    return dict(fap=fap, sigma=float(max(stats.norm.isf(fap), 0.0)),
                null_sde_mean=float(null.mean()), null_sde_max=float(null.max()))


class TemperatureScaler:
    """Post-hoc probability calibration (Guo et al. 2017): one scalar T fitted
    on the validation set so predicted probabilities match observed rates."""

    def __init__(self, T=1.0):
        self.T = T

    def fit(self, logits, labels, mask_partial=None):
        from scipy.optimize import minimize_scalar

        def nll(T):
            z = logits / T
            z = z - z.max(1, keepdims=True)
            lp = z - np.log(np.exp(z).sum(1, keepdims=True))
            return -lp[np.arange(len(labels)), labels].mean()
        self.T = float(minimize_scalar(nll, bounds=(0.3, 10), method="bounded").x)
        return self

    def __call__(self, logits):
        z = logits / self.T
        z = z - z.max(1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(1, keepdims=True)
