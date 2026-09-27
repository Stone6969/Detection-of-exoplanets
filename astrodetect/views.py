"""Stage 4b - model inputs: phase-folded "views" + scalar features.

global view : 201 bins over the full orbit (flux)              -> 1 x 201
local views : 61 bins over +-2.5 T14 around the transit        -> 7 x 61
              channels = flux, odd transits, even transits,
                         secondary (phase 0.5), centroid x, centroid y,
                         background
Flux channels are scaled so the transit is -1 (depth goes in the scalars);
centroid/background channels are in units of their own out-of-transit
scatter, so a blend shows up as a clear bump regardless of star brightness.
The telemetry channels are the part standard AstroNet/ExoMiner-style inputs
do not have.
"""
import numpy as np

from .vetting import phase

N_GLOBAL, N_LOCAL = 201, 61
LOCAL_CHANNELS = ["flux", "odd", "even", "secondary", "cx", "cy", "bkg"]

SCALARS = ["log_depth", "log_period", "dur_over_period", "snr", "log_sde",
           "odd_even_sigma", "secondary_sigma", "log_secondary_depth", "v_shape",
           "rp_rs", "b", "log_rp_earth", "log_rho_ratio", "centroid_sigma",
           "log_source_offset", "bkg_shift_sigma", "max_event_fraction",
           "log_sector_chi2", "n_transits", "teff_k", "rstar", "tmag", "beta"]


def _bin(x, y, lo, hi, n):
    e = np.linspace(lo, hi, n + 1)
    i = np.digitize(x, e) - 1
    ok = (i >= 0) & (i < n) & np.isfinite(y)
    s = np.bincount(i[ok], weights=y[ok], minlength=n)
    c = np.bincount(i[ok], minlength=n)
    v = np.where(c > 0, s / np.maximum(c, 1), np.nan)
    if np.isnan(v).all():
        return np.zeros(n)
    k = np.arange(n)
    v[np.isnan(v)] = np.interp(k[np.isnan(v)], k[~np.isnan(v)], v[~np.isnan(v)])
    return v


def make_views(t, flux, cx, cy, bkg, P, t0, dur, depth):
    ok = np.isfinite(flux)
    t, flux, cx, cy, bkg = (a[ok] for a in (t, flux, cx, cy, bkg))
    ph = phase(t, P, t0)
    epoch = np.round((t - t0) / P).astype(int)
    w = 2.5 * dur
    depth = depth if depth > 0 else max(np.std(flux), 1e-5)
    oot = np.abs(ph) > 0.75 * dur

    def norm_flux(v):
        return np.clip((v - np.median(flux[oot])) / depth, -5, 5)

    g = _bin(ph / P, flux, -0.5, 0.5, N_GLOBAL)
    glob = np.clip((g - np.median(g)) / depth, -5, 5)[None, :]

    loc = []
    for name in LOCAL_CHANNELS:
        if name in ("flux", "odd", "even"):
            m = np.ones(t.size, bool) if name == "flux" else (epoch % 2 == (1 if name == "odd" else 0))
            v = norm_flux(_bin(ph[m], flux[m], -w, w, N_LOCAL)) if m.sum() > 10 else np.zeros(N_LOCAL)
        elif name == "secondary":
            ph2 = ((t - t0) % P) - 0.5 * P
            v = norm_flux(_bin(ph2, flux, -w, w, N_LOCAL))
        else:
            c = {"cx": cx, "cy": cy, "bkg": bkg}[name]
            if not np.isfinite(c).all():
                v = np.zeros(N_LOCAL)
            else:
                b = _bin(ph, c, -w, w, N_LOCAL)
                edge = np.r_[b[:15], b[-15:]]
                s = np.std(edge) or 1.0
                v = np.clip((b - np.median(edge)) / s, -10, 10)
        loc.append(v)
    return glob.astype(np.float32), np.array(loc, np.float32)


def scalar_vector(diag, fit, sde, meta):
    L = lambda x: float(np.log10(max(abs(x), 1e-9)))
    v = dict(
        log_depth=L(diag.get("depth_ppm", 0)), log_period=L(fit["period"]),
        dur_over_period=fit["duration"] / fit["period"], snr=min(diag.get("snr", 0), 500),
        log_sde=L(sde), odd_even_sigma=min(diag["odd_even_sigma"], 50),
        secondary_sigma=float(np.clip(diag["secondary_sigma"], -20, 50)),
        log_secondary_depth=L(max(diag["secondary_depth_ppm"], 1)), v_shape=float(np.clip(diag["v_shape"], -2, 3)),
        rp_rs=fit["rp_rs"], b=fit["b"],
        log_rp_earth=L(fit["rp_earth"]) if np.isfinite(fit["rp_earth"]) else 0.0,
        log_rho_ratio=L(fit["rho_ratio"]) if np.isfinite(fit["rho_ratio"]) and fit["rho_ratio"] > 0 else 0.0,
        centroid_sigma=min(diag["centroid_sigma"], 50),
        log_source_offset=L(max(diag["source_offset_px"], 1e-3)),
        bkg_shift_sigma=float(np.clip(diag["bkg_shift_sigma"], -50, 50)),
        max_event_fraction=diag["max_event_fraction"],
        log_sector_chi2=L(max(diag["sector_depth_chi2"], 1e-3)),
        n_transits=min(diag["n_transits"], 100),
        teff_k=float(meta.get("teff") or 5800) / 1000, rstar=float(meta.get("radius") or 1.0),
        tmag=float(meta.get("tmag") or 11.0), beta=min(fit.get("beta", 1.0), 10))
    return np.array([v[k] if np.isfinite(v[k]) else 0.0 for k in SCALARS], np.float32)
