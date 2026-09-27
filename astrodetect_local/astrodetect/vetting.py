"""Stage 4a - physics diagnostics (novel part #2: centroid + telemetry vetting
from the light-curve file alone).

Each diagnostic targets one false-positive family in the problem statement:
  eclipsing binary  -> odd/even depth mismatch, secondary eclipse, V shape,
                       large radius, impossible stellar density
  blend (background EB) -> CENTROID SHIFT during transit: if the dimming star
                       is a neighbour, the flux-weighted centroid moves. The
                       implied source offset = shift / depth (pixels) - near
                       0 means on-target, >~0.3 px means somewhere else
  instrumental / other -> background change in transit, one event dominating
                       the SNR, depth inconsistent between sectors
They become scalar features for the classifier AND a human-readable report.
"""
import numpy as np

from .detrend import mad_std, segments
from .search import transit_mask


def phase(t, P, t0):
    return ((t - t0 + 0.5 * P) % P) - 0.5 * P


def _depth(y, inn, out):
    """Depth and its error using out-of-transit scatter."""
    if inn.sum() < 3 or out.sum() < 10:
        return np.nan, np.nan
    s = mad_std(y[out])
    return np.median(y[out]) - np.mean(y[inn]), s / np.sqrt(inn.sum())


def red_noise_sigma(t, y, mask_out, duration):
    """Scatter of running means over one transit duration (out of transit)."""
    vals = []
    for s in segments(t):
        yy = y[s][mask_out[s]]
        yy = yy[np.isfinite(yy)]
        n = max(1, int(round(duration / np.median(np.diff(t[s])))))
        if yy.size > 3 * n:
            c = np.cumsum(np.r_[0, yy])
            vals.append((c[n:] - c[:-n]) / n)
    return np.std(np.concatenate(vals)) if vals else np.nan


def beta_factor(t, y, mask_out, duration):
    """Red-noise inflation: real scatter of 1-duration means / white-noise
    expectation. Without it, odd/even and centroid 'sigmas' come out 3-5x too
    significant on real TESS data (seen on TOI-907)."""
    y = np.asarray(y)
    if not np.isfinite(y).any():
        return 1.0
    n = max(1, int(round(duration / np.median(np.diff(t)))))
    white = mad_std(y[mask_out]) / np.sqrt(n)
    red = red_noise_sigma(t, y, mask_out, duration)
    return float(max(1.0, red / white)) if white > 0 and np.isfinite(red) else 1.0


def diagnostics(t, flux, cx, cy, bkg, sector, P, t0, dur, depth_fit=None):
    ok = np.isfinite(flux)
    t, flux, cx, cy, bkg, sector = (a[ok] for a in (t, flux, cx, cy, bkg, sector))
    ph = phase(t, P, t0)
    inn = np.abs(ph) < 0.35 * dur            # core of transit
    near = np.abs(ph) < 3 * dur               # local window
    out_loc = near & (np.abs(ph) > 0.75 * dur)
    out_all = np.abs(ph) > 0.75 * dur
    if out_loc.sum() < 20:                   # very short period: use all out-of-transit
        out_loc = out_all
    epoch = np.round((t - t0) / P).astype(int)
    n_tr = len(np.unique(epoch[inn]))
    f = {}

    bf = beta_factor(t, flux, out_all, dur)
    depth, derr = _depth(flux, inn, out_loc)
    f["depth_ppm"] = depth * 1e6
    sig_dur = red_noise_sigma(t, flux, out_all, dur)
    f["snr"] = depth / sig_dur * np.sqrt(max(n_tr, 1)) if sig_dur > 0 else np.nan
    f["n_transits"] = n_tr
    f["beta_flux"] = bf

    # --- EB family --------------------------------------------------------
    odd, even = inn & (epoch % 2 == 1), inn & (epoch % 2 == 0)
    d_o, e_o = _depth(flux, odd, out_loc)
    d_e, e_e = _depth(flux, even, out_loc)
    f["odd_even_sigma"] = abs(d_o - d_e) / (bf * np.hypot(e_o, e_e)) if np.isfinite(d_o * d_e) else 0.0
    ph2 = ((t - t0) % P) / P - 0.5                           # 0 at phase 0.5
    sec = np.abs(ph2 * P) < 0.35 * dur
    sec_out = (np.abs(ph2 * P) < 3 * dur) & (np.abs(ph2 * P) > 0.75 * dur)
    d_s, e_s = _depth(flux, sec, sec_out)
    f["secondary_depth_ppm"] = d_s * 1e6 if np.isfinite(d_s) else 0.0
    f["secondary_sigma"] = d_s / (bf * e_s) if np.isfinite(d_s) and e_s > 0 else 0.0
    edge = (np.abs(ph) > 0.2 * dur) & (np.abs(ph) < 0.45 * dur)
    core = np.abs(ph) < 0.15 * dur
    d_edge, _ = _depth(flux, edge, out_loc)
    d_core, _ = _depth(flux, core, out_loc)
    f["v_shape"] = d_edge / d_core if np.isfinite(d_edge) and d_core > 0 else 1.0

    # --- blend family: centroid motion ------------------------------------
    for name, c in (("cx", cx), ("cy", cy)):
        if np.isfinite(c).all() and c.size:
            dc, ec = _depth(-c, inn, out_loc)       # mean in - mean out
            ec *= beta_factor(t, c, out_all, dur)
            f[f"{name}_shift_sigma"] = dc / ec if ec > 0 else 0.0
            f[f"{name}_shift_px"] = dc
        else:
            f[f"{name}_shift_sigma"], f[f"{name}_shift_px"] = 0.0, 0.0
    f["centroid_sigma"] = float(np.hypot(f["cx_shift_sigma"], f["cy_shift_sigma"]))
    shift = np.hypot(f["cx_shift_px"], f["cy_shift_px"])
    f["source_offset_px"] = float(shift / depth) if depth > 0 else 0.0

    # --- instrumental / other ---------------------------------------------
    if np.isfinite(bkg).all() and bkg.size:
        db, eb = _depth(-bkg, inn, out_loc)
        eb *= beta_factor(t, bkg, out_all, dur)
        f["bkg_shift_sigma"] = db / eb if eb > 0 else 0.0
    else:
        f["bkg_shift_sigma"] = 0.0
    # how much of the signal comes from the single strongest event
    ev = []
    for e in np.unique(epoch[inn]):
        m = inn & (epoch == e)
        ev.append(max(np.median(flux[out_loc]) - np.mean(flux[m]), 0) * np.sqrt(m.sum()))
    ev = np.array(ev) if ev else np.zeros(0)
    f["max_event_fraction"] = float(ev.max() / ev.sum()) if ev.size and ev.sum() > 0 else 1.0
    # depth consistency between sectors
    ds = []
    for s in np.unique(sector):
        m = sector == s
        d, e = _depth(flux[m], inn[m], out_loc[m])
        if np.isfinite(d) and e > 0:
            ds.append((d, e))
    if len(ds) > 1:
        d, e = np.array(ds).T
        w = 1 / e ** 2
        mu = np.sum(w * d) / w.sum()
        f["sector_depth_chi2"] = float(np.sum(((d - mu) / e) ** 2) / (len(d) - 1))
    else:
        f["sector_depth_chi2"] = 1.0
    return {k: float(v) if np.isfinite(v) else 0.0 for k, v in f.items()}
