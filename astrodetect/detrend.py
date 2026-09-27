"""Stage 1 - DETRENDING (novel part #1: telemetry-aware, PDC-independent).

raw SAP flux
  -> (io) quality mask + crowding correction
  -> upward-outlier clip (cosmic rays / flares; low points kept - they may be transits)
  -> iterate:
        slow trend  = robust Gaussian low-pass with *dip protection*
        residuals   = flux / trend - 1
        systematics = ridge regression of residuals on the star's own spacecraft
                      telemetry (background, centroid x/y, pointing x/y, + squares),
                      fitted per orbit, OUTSIDE detected dips only; inside dips the
                      telemetry is interpolated before the model is applied, so a
                      real transit (or a blend's centroid motion) cannot be
                      "explained away" by the regression
  -> detrended flux = flux / (trend * systematics)

Why: a plain filter on SAP leaves fast pointing / scattered-light systematics
(60 % more noise than NASA PDCSAP on TOI-907). Telemetry regression recovered
~64 % of that gap using only the star's own file; no NASA cotrending needed.
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter

FWHM = 2 * np.sqrt(2 * np.log(2))


# ------------------------------------------------------------------ utilities
def mad_std(x):
    x = np.asarray(x)[np.isfinite(x)]
    return 1.4826 * np.median(np.abs(x - np.median(x))) if x.size else np.nan


def segments(time, gap=0.5):
    """Contiguous chunks (TESS orbits / sectors), split where dt > gap days."""
    br = np.where(np.diff(time) > gap)[0] + 1
    e = np.r_[0, br, time.size]
    return [slice(a, b) for a, b in zip(e[:-1], e[1:]) if b - a > 10]


def to_grid(t, y):
    """Regular cadence grid for one segment; missing cadences interpolated."""
    cad = np.median(np.diff(t))
    idx = np.round((t - t[0]) / cad).astype(int)
    g = np.full(idx[-1] + 1, np.nan)
    g[idx] = y
    m = ~np.isfinite(g)
    if m.any():
        k = np.arange(g.size)
        g[m] = np.interp(k[m], k[~m], g[~m])
    return g, idx, cad


def fill_gaps(y, mask, order=2):
    """Replace masked runs with a local polynomial bridged from both sides.
    (A straight line across a 6-h transit gap biased depths on curved stellar
    variability - quadratic bridge fixes that.)"""
    out = y.copy()
    if not mask.any() or (~mask).sum() < 5:
        return out
    k = np.arange(y.size)
    edges = np.flatnonzero(np.diff(np.r_[0, mask.astype(int), 0]))
    for a, b in zip(edges[::2], edges[1::2]):
        L = max(10, b - a)
        left = np.arange(max(0, a - L), a)
        right = np.arange(b, min(y.size, b + L))
        left, right = left[~mask[left]], right[~mask[right]]
        if left.size < 3 or right.size < 3:
            # run touches a segment edge: never extrapolate a polynomial,
            # hold the nearest clean level instead
            side = left if left.size >= right.size else right
            out[a:b] = np.median(y[side]) if side.size else np.median(y[~mask])
            continue
        ctx = np.r_[left, right]
        o = order if min(left.size, right.size) >= 3 * (order + 1) else 1
        out[a:b] = np.polyval(np.polyfit(ctx - a, y[ctx], o), k[a:b] - a)
    return out


def lowpass(y, window_pts):
    return gaussian_filter1d(y, sigma=window_pts / FWHM, mode="nearest")


def clip_upper_outliers(time, flux, window=0.5, nsig=5.0):
    keep = np.ones(flux.size, bool)
    for s in segments(time):
        g, idx, cad = to_grid(time[s], flux[s])
        r = g - median_filter(g, size=max(3, int(window / cad) | 1), mode="nearest")
        keep[s] = r[idx] < nsig * mad_std(r)
    return keep


# ------------------------------------------------------------------ robust trend
def robust_trend(time, flux, window=0.75, nsig=3.0, n_iter=8, known_mask=None,
                 dip_scale=1 / 24):
    """Iterative dip-protected Gaussian trend.
    Masks (a) per-point >nsig outliers and (b) dips whose residual smoothed over
    `dip_scale` is < -nsig sigma  -> shallow transits are protected even when
    each point alone is inside the noise.
    Returns trend (same length as flux) and the final dip mask."""
    trend = np.full(flux.size, np.nan)
    dips = np.zeros(flux.size, bool)
    for s in segments(time):
        g, idx, cad = to_grid(time[s], flux[s])
        known = np.zeros(g.size, bool)
        if known_mask is not None:
            known[idx] = known_mask[s]
        mask = known.copy()
        tr = lowpass(g, window / cad)
        for _ in range(n_iter):
            tr = lowpass(fill_gaps(g, mask), window / cad)
            r = g / tr - 1
            sig = mad_std(r[~mask])
            rs = lowpass(r, dip_scale / cad)
            new = known | (np.abs(r) > nsig * sig) | (rs < -nsig * mad_std(rs[~mask]))
            new = np.convolve(new, np.ones(max(5, int(0.5 * dip_scale / cad))), "same") > 0
            if np.array_equal(new, mask):
                break
            mask = new
        trend[s], dips[s] = tr[idx], mask[idx]
    return trend, dips


# ------------------------------------------------------------------ telemetry regression
def _seg_design(ts, aux_s, interp_mask=None):
    """[1, z, z^2] for one orbit from standardised telemetry. Inside
    interp_mask the telemetry is bridged by interpolation first."""
    cols = [np.ones(ts.size)]
    for v in aux_s.values():
        v = v.copy()
        if interp_mask is not None and interp_mask.any() and (~interp_mask).sum() > 5:
            v[interp_mask] = np.interp(ts[interp_mask], ts[~interp_mask], v[~interp_mask])
        sd = np.std(v)
        if not sd > 0:
            continue
        z = (v - v.mean()) / sd
        cols += [z, z ** 2]
    return np.array(cols).T


def telemetry_systematics(time, resid, aux, fit_mask, protect_mask, ridge=1e-3,
                          n_irls=5, huber_k=2.0):
    """Per orbit: robust (Huber-IRLS) ridge regression of residuals on
    telemetry, fitted on cadences with fit_mask=True.
    Generic detected dips are NOT excluded: fast systematics look like dips,
    and excluding them removes exactly what we want to model (TOI-907: no gain
    at all when they were excluded). Robust weights stop genuine transits from
    steering the fit. Once a candidate is known, its transits go in
    `protect_mask`: excluded from the fit and telemetry bridged across them at
    prediction time, so a transit - or a blend's centroid motion - cannot be
    subtracted by the model."""
    out = np.zeros_like(resid)
    for s in segments(time):
        fm = fit_mask[s]
        if fm.sum() < 50:
            continue
        aux_s = {k: v[s] for k, v in aux.items()}
        A = _seg_design(time[s], aux_s)[fm]
        b = resid[s][fm]
        w = np.ones(b.size)
        for _ in range(n_irls):
            Aw = A * w[:, None]
            coef = np.linalg.solve(A.T @ Aw + ridge * len(A) * np.eye(A.shape[1]), Aw.T @ b)
            r = b - A @ coef
            c = huber_k * mad_std(r)
            w = np.where(np.abs(r) <= c, 1.0, c / np.maximum(np.abs(r), 1e-12))
        out[s] = _seg_design(time[s], aux_s, interp_mask=protect_mask[s]) @ coef
    return out


def strong_dips(time, resid, nsig=5.0, scale=1 / 24):
    """Sustained, significant dips only: residual smoothed over `scale` below
    -nsig robust sigma, widened by 1.5x scale. Unlike the per-point dip mask
    this covers ~3 % of cadences (not ~20 %), so fast systematics stay in the
    regression while real eclipses/transits are protected."""
    m = np.zeros(time.size, bool)
    for s in segments(time):
        g, idx, cad = to_grid(time[s], resid[s])
        rs = lowpass(g, scale / cad)
        mm = rs < -nsig * mad_std(rs)
        mm = np.convolve(mm, np.ones(max(5, int(1.5 * scale / cad))), "same") > 0
        m[s] = mm[idx]
    return m


def detrend_raw(lc, window=0.75, n_outer=3, known_mask=None, use_telemetry=True):
    """Full novel detrending on raw SAP. Returns dict with detrended flux,
    trend, systematics, dip mask, and detrended centroids (for blend tests)."""
    t, f = lc.time, lc.sap
    keep = clip_upper_outliers(t, f)
    sysm = np.ones_like(f)
    usable = {k: v for k, v in lc.aux.items() if np.isfinite(v).all()}
    for _ in range(n_outer if use_telemetry and usable else 1):
        trend, dips = robust_trend(t, f / sysm, window, known_mask=known_mask)
        if not (use_telemetry and usable):
            break
        resid = f / trend - 1
        # Protect only STRONG, sustained dips (plus a known candidate). Why:
        #  - protecting every flagged dip (~20 % of cadences) hides the fast
        #    systematics themselves -> no gain (CDPP 409 on TOI-907)
        #  - protecting nothing lets the centroid regressors "explain away" a
        #    blend, whose centroid moves in step with its eclipse -> deep blends
        #    vanished (0/4 recovered in the injection test)
        #  - strong-dip protection: CDPP 316, SNR 33, 5/6 blends recovered
        prot = strong_dips(t, resid)
        if known_mask is not None:
            prot |= known_mask
        sysm = 1 + telemetry_systematics(t, resid, usable, keep & ~prot, prot)
    trend, dips = robust_trend(t, f / sysm, window, known_mask=known_mask)
    flat = f / (trend * sysm)
    flat[~keep] = np.nan
    return dict(flux=flat, trend=trend, systematics=sysm, dips=dips, keep=keep,
                cx=_detrend_centroid(t, lc.aux["cx"], dips, window),
                cy=_detrend_centroid(t, lc.aux["cy"], dips, window))


def detrend_pdc(lc, window=0.75, known_mask=None):
    """Baseline for the ablation: same robust trend but on NASA PDCSAP."""
    t, f = lc.time, lc.pdc
    keep = clip_upper_outliers(t, f) & np.isfinite(f)
    trend, dips = robust_trend(t[keep], f[keep], window,
                               known_mask=None if known_mask is None else known_mask[keep])
    flat = np.full(f.size, np.nan)
    flat[keep] = f[keep] / trend
    d = np.zeros(f.size, bool)
    d[keep] = dips
    return dict(flux=flat, dips=d, keep=keep,
                cx=_detrend_centroid(t, lc.aux["cx"], d, window),
                cy=_detrend_centroid(t, lc.aux["cy"], d, window))


def _detrend_centroid(t, c, dips, window):
    """Centroid minus its own slow trend (dips bridged), so a shift during
    transit survives -> blend diagnostic."""
    if not np.isfinite(c).all():
        return np.full(t.size, np.nan)
    out = np.full(t.size, np.nan)
    for s in segments(t):
        g, idx, cad = to_grid(t[s], c[s])
        m = np.zeros(g.size, bool)
        m[idx] = dips[s]
        out[s] = (g - lowpass(fill_gaps(g, m), window / cad))[idx]
    return out
