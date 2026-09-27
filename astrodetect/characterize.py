"""Stage 3 - CHARACTERIZATION: physical transit-model fit (batman) with errors.

Fits t0, P, Rp/R*, a/R*, impact parameter b to the candidate-pass detrended
flux (data within +-2.5 durations of each transit), quadratic limb darkening
fixed from Teff. Least squares -> covariance from the Jacobian, inflated by the
reduced chi2 and by a red-noise factor (beta) so errors are not optimistic.

Derived: depth, T14, Rp [R_earth], and the transit-derived stellar density.
Comparing that density with the catalogue density (from logg + radius) is a
strong physics check: an eclipsing binary or blend fitted as a planet
usually implies an impossible star.
"""
import warnings

import batman
import numpy as np
from scipy.optimize import least_squares

from .detrend import mad_std
from .search import transit_mask

G = 6.674e-8          # cgs
R_SUN_EARTH = 109.1
RHO_SUN = 1.408       # g/cm3


def limb_darkening(teff):
    """Rough quadratic LD for the TESS band vs Teff (Claret 2017 trend)."""
    teff = 5800 if teff is None or not np.isfinite(teff) else float(np.clip(teff, 3000, 10000))
    u1 = np.interp(teff, [3000, 4000, 5000, 5800, 7000, 10000], [0.20, 0.35, 0.42, 0.38, 0.30, 0.20])
    u2 = np.interp(teff, [3000, 4000, 5000, 5800, 7000, 10000], [0.40, 0.30, 0.22, 0.22, 0.25, 0.28])
    return [float(u1), float(u2)]


def catalog_density(logg, radius):
    """rho* [g/cc] from logg and R*/Rsun (None if unavailable)."""
    try:
        g = 10 ** float(logg)
        r = float(radius) * 6.957e10
        return 3 * g / (4 * np.pi * G * r) if np.isfinite(g * r) and r > 0 else None
    except (TypeError, ValueError):
        return None


def _model(theta, t, u, exp_time):
    t0, P, rp, a, b = theta
    p = batman.TransitParams()
    p.t0, p.per, p.rp, p.a = t0, P, rp, max(a, 1.01)
    p.inc = np.degrees(np.arccos(np.clip(b / p.a, 0, 1)))
    p.ecc, p.w, p.limb_dark, p.u = 0.0, 90.0, "quadratic", u
    ss = 7 if exp_time > 0.007 else 1                 # supersample 10/30-min data
    return batman.TransitModel(p, t, supersample_factor=ss, exp_time=exp_time).light_curve(p)


def t14(P, rp, a, b):
    arg = np.sqrt(max((1 + rp) ** 2 - b ** 2, 0)) / (a * np.sin(np.arccos(min(b / a, 1))))
    return P / np.pi * np.arcsin(min(arg, 1.0))


def fit_transit(time, flux, cand, teff=None, rstar=None, logg=None, exp_time=2 / 1440):
    """cand: dict with period, t0, duration, depth (from BLS). Returns dict."""
    P0, T00, D0 = cand["period"], cand["t0"], cand["duration"]
    depth0 = max(cand.get("depth", 1e-3), 5e-5)
    sel = transit_mask(time, P0, T00, D0, pad=5.0) & np.isfinite(flux)
    t, y = time[sel], flux[sel]
    if t.size < 30:
        return None
    u = limb_darkening(teff)
    rho = catalog_density(logg, rstar) or RHO_SUN
    a0 = (G * rho * (P0 * 86400) ** 2 / (3 * np.pi)) ** (1 / 3)
    a0 = float(np.clip(a0, 1.5, 300))
    x0 = [T00, P0, np.sqrt(depth0) * 1.05, a0, 0.3]
    lo = [T00 - 0.5 * D0, P0 * 0.995, 0.003, 1.2, 0.0]
    hi = [T00 + 0.5 * D0, P0 * 1.005, 1.2, 500, 1.5]
    x0 = np.clip(x0, np.array(lo) + 1e-9, np.array(hi) - 1e-9)
    oot = ~transit_mask(t, P0, T00, D0, 1.5)
    sig = mad_std(y[oot]) if oot.sum() > 20 else mad_std(y)
    if not (np.isfinite(sig) and sig > 0):
        return None
    res_fn = lambda th: np.nan_to_num((y - _model(th, t, u, exp_time)) / sig, nan=1e3)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = least_squares(res_fn, x0, bounds=(lo, hi), x_scale="jac", max_nfev=400)
    except ValueError:
        return None
    th = r.x
    resid = r.fun * sig
    chi2r = np.sum(r.fun ** 2) / max(t.size - 5, 1)
    beta = _red_noise_beta(t, resid, D0)
    try:
        cov = np.linalg.inv(r.jac.T @ r.jac) * max(chi2r, 1) * beta ** 2
        err = np.sqrt(np.clip(np.diag(cov), 0, None))
    except np.linalg.LinAlgError:
        err = np.full(5, np.nan)
    t0, P, rp, a, b = th
    model_min = 1 - _model(th, np.array([t0]), u, 0.0)[0]
    dur = t14(P, rp, a, b)
    rho_tr = 3 * np.pi * a ** 3 / (G * (P * 86400) ** 2)
    rho_cat = catalog_density(logg, rstar)
    out = dict(t0=t0, period=P, rp_rs=rp, a_rs=a, b=b,
               t0_err=err[0], period_err=err[1], rp_rs_err=err[2], a_rs_err=err[3], b_err=err[4],
               depth=float(model_min), duration=float(dur if dur > 0 else D0),
               rp_earth=float(rp * rstar * R_SUN_EARTH) if rstar and np.isfinite(rstar) else np.nan,
               rho_transit=float(rho_tr),
               rho_ratio=float(rho_tr / rho_cat) if rho_cat else np.nan,
               chi2r=float(chi2r), beta=float(beta), fit_ok=bool(r.success))
    return out


def _red_noise_beta(t, resid, duration):
    """Pont+2006 time-averaging beta: how much worse than white the residual
    scatter is on the transit timescale (1 = white)."""
    n = max(2, int(duration / np.median(np.diff(t)) / 2))
    if resid.size < 4 * n:
        return 1.0
    s1 = np.std(resid)
    m = resid[: resid.size // n * n].reshape(-1, n).mean(1)
    expected = s1 / np.sqrt(n) * np.sqrt(len(m) / max(len(m) - 1, 1))
    return float(max(1.0, np.std(m) / expected)) if expected > 0 else 1.0
