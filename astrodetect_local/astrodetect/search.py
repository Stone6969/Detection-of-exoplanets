"""Stage 2 - IDENTIFICATION: iterative Box Least Squares on the detrended flux.

For each star:
  bin to 10 min (5x faster, no loss for >= 1 h transits)
  BLS over a duration grid 0.75-13 h and a frequency grid fine enough that the
  phase drift across the full baseline stays < 1/3 of the shortest duration
  take the highest-SDE peak -> candidate
  mask its transits and search again (multi-planet systems, and it stops one
  deep EB from hiding a shallow planet), until SDE < sde_min or max_cands.
"""
import numpy as np
from astropy.timeseries import BoxLeastSquares
from scipy.ndimage import median_filter

from .detrend import mad_std

DURATIONS_H = np.array([1, 1.5, 2, 3, 4, 5, 7, 10, 13])


def bin_series(t, y, minutes=10.0):
    ok = np.isfinite(y)
    t, y = t[ok], y[ok]
    w = minutes / 1440
    k = np.floor((t - t[0]) / w).astype(int)
    _, idx, cnt = np.unique(k, return_index=True, return_counts=True)
    tb = np.add.reduceat(t, idx) / cnt
    yb = np.add.reduceat(y, idx) / cnt
    good = cnt >= max(1, 0.5 * np.median(cnt))
    return tb[good], yb[good]


def transit_mask(t, period, t0, duration, pad=1.5):
    ph = ((t - t0 + 0.5 * period) % period) - 0.5 * period
    return np.abs(ph) < pad * duration / 2


def bls_search(t, y, pmin=0.5, pmax=None, max_periods=100_000, min_transits=2):
    """One BLS pass. Returns dict for the best peak + the periodogram."""
    base = t.max() - t.min()
    pmax = min(pmax or np.inf, base / min_transits)
    if pmax <= pmin * 1.05:
        return None
    durs = DURATIONS_H / 24
    durs = durs[durs < 0.9 * pmin]                  # BLS: every duration < shortest period
    qmin = durs.min()
    # frequency step so the transit drifts < qmin/2 over the whole baseline
    df = qmin / (2 * base ** 2)
    nf = int((1 / pmin - 1 / pmax) / df)
    if nf > max_periods:
        df *= nf / max_periods
    freqs = np.arange(1 / pmax, 1 / pmin, df)
    periods = np.sort(1 / freqs)
    dy = np.full(y.size, mad_std(y))
    bls = BoxLeastSquares(t, y, dy)
    res = bls.power(periods, durs, objective="likelihood")
    p = res.power
    # BLS power rises with period; flatten with a running median before SDE
    flat = p - median_filter(p, size=min(1001, (p.size // 10) | 1), mode="nearest")
    i = int(np.argmax(flat))
    sde = (flat[i] - np.mean(flat)) / np.std(flat)
    return dict(period=float(res.period[i]), t0=float(res.transit_time[i]),
                duration=float(res.duration[i]), depth=float(res.depth[i]),
                depth_snr=float(res.depth_snr[i]), sde=float(sde),
                periods=res.period, power=flat)


def iterative_search(time, flux, max_cands=3, sde_min=7.0, pmin=0.5, pmax=None,
                     keep_periodogram=False, max_periods=100_000):
    """Find up to `max_cands` candidates, masking each before the next pass."""
    t, y = bin_series(time, flux)
    y = y - np.median(y)
    cands = []
    for _ in range(max_cands):
        r = bls_search(t, y, pmin, pmax, max_periods=max_periods)
        if r is None or r["sde"] < sde_min or r["depth"] <= 0:
            break
        if not keep_periodogram:
            r.pop("periods"), r.pop("power")
        # a "transit" lasting > 20 % of the orbit is stellar variability, not
        # an eclipse; skip it (but still mask it so the next pass moves on)
        if r["duration"] <= 0.2 * r["period"]:
            cands.append(r)
        m = transit_mask(t, r["period"], r["t0"], r["duration"], pad=2.0)
        t, y = t[~m], y[~m]
        if t.size < 200:
            break
    return cands


def matches(p_found, p_true, tol=0.01):
    """True if a found period equals the true one or a simple harmonic."""
    for h in (1, 0.5, 2, 1 / 3, 3):
        if abs(p_found / (p_true * h) - 1) < tol:
            return h
    return None
