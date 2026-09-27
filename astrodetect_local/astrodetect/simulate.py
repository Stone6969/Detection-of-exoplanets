"""Novel part #3: physics-based simulations injected into REAL TESS noise.

The dataset labels are planet / candidate / false positive only - it never says
*which* kind of false positive, but the problem statement asks to classify
transits vs eclipses vs blends vs other. So we create those classes ourselves,
with correct physics, inside real light curves (real noise, real systematics,
real telemetry), and train on them together with the real labels.

  0 planet  : batman transit, a/R* from the star's density, small centroid
              motion consistent with on-target dimming
  1 EB      : primary + secondary eclipse (brightness ratio), often grazing /
              V-shaped, optional ellipsoidal variation, sometimes twin
              eclipses (which a search then finds at half the period)
  2 blend   : an EB on a neighbour star, diluted to planet-like depth, and the
              centroid moves toward/away from the neighbour in proportion to
              the dimming (shift = depth x offset)
  3 other   : no injection (whatever the search finds in real noise), or a
              scattered-light-like dip that also appears in the background
All injections are multiplicative on SAP and PDCSAP and additive on the
telemetry, then go through exactly the same detrend -> search -> fit ->
vetting chain as real data.
"""
import batman
import numpy as np

from .characterize import G, catalog_density, limb_darkening

CLASSES = ["planet", "eclipsing_binary", "blend", "other"]


def _transit(t, P, t0, rp, a, b, u):
    p = batman.TransitParams()
    p.t0, p.per, p.rp, p.a = t0, P, rp, a
    p.inc = np.degrees(np.arccos(np.clip(b / a, 0, 1)))
    p.ecc, p.w, p.limb_dark, p.u = 0.0, 90.0, "quadratic", u
    return batman.TransitModel(p, t).light_curve(p)


def _a_rs(P, rho):
    return float(np.clip((G * rho * (P * 86400) ** 2 / (3 * np.pi)) ** (1 / 3), 1.5, 400))


def _eb_flux(t, P, t0, rng, u, rho):
    """Normalised EB light curve (primary + secondary + ellipsoidal)."""
    k = rng.uniform(0.15, 0.9)                     # radius ratio of the companion
    a = _a_rs(P, rho * rng.uniform(0.5, 2.0))
    b = rng.uniform(0.0, 1.0 + k * 0.9)            # grazing allowed
    prim = _transit(t, P, t0, k, a, b, u)
    sb = rng.uniform(0.0, 1.0) ** 1.5              # surface-brightness ratio
    if rng.random() < 0.2:
        sb = rng.uniform(0.85, 1.0)                # near-twin eclipses
    ecc_shift = rng.normal(0, 0.01) * P
    sec = _transit(t, P, t0 + P / 2 + ecc_shift, k, a, b, u)
    f = 1 - (1 - prim) - sb * (1 - sec)
    if rng.random() < 0.5:                          # ellipsoidal variation
        amp = rng.uniform(0, 0.2) * (1 - prim.min())
        f *= 1 - amp * np.cos(4 * np.pi * (t - t0) / P)
    return f


def sample_params(cls, t, meta, rng):
    base = t.max() - t.min()
    pmax = max(0.6, min(base / 2.2, 30.0))
    P = float(np.exp(rng.uniform(np.log(0.5), np.log(pmax))))
    t0 = float(t.min() + rng.uniform(0, P))
    return dict(cls=cls, period=P, t0=t0,
                rho=catalog_density(meta.get("logg"), meta.get("radius")) or 1.41,
                u=limb_darkening(meta.get("teff")))


def inject(lc, cls, rng, params=None):
    """Return (new LightCurve, truth dict). lc is not modified."""
    t = lc.time
    prm = params or sample_params(cls, t, lc.meta, rng)
    P, t0, rho, u = prm["period"], prm["t0"], prm["rho"], prm["u"]
    model = np.ones(t.size)
    dcx = np.zeros(t.size)
    dcy = np.zeros(t.size)
    dbkg = np.zeros(t.size)
    crowd = lc.meta.get("crowdsap", 1.0) or 1.0
    ang = rng.uniform(0, 2 * np.pi)

    if cls == "planet":
        depth = 10 ** rng.uniform(np.log10(150e-6), np.log10(3e-2))
        rp = np.sqrt(depth) * 1.08
        a = _a_rs(P, rho)
        b = rng.uniform(0, 0.9)
        model = _transit(t, P, t0, rp, a, b, u)
        off = rng.uniform(0, 0.3) * (1 - crowd + 0.02)      # on target: tiny offset
    elif cls == "eclipsing_binary":
        model = _eb_flux(t, P, t0, rng, u, rho)
        off = rng.uniform(0, 0.3) * (1 - crowd + 0.02)
    elif cls == "blend":
        eb = _eb_flux(t, P, t0, rng, u, rho)
        dil = 10 ** rng.uniform(np.log10(5), np.log10(300))   # target/neighbour flux
        model = 1 - (1 - eb) / (1 + dil)
        off = rng.uniform(0.4, 3.0)                            # neighbour offset, px
    else:  # other
        if rng.random() < 0.5:
            off = 0.0                                          # pure noise / variability
        else:                                                  # scattered-light-like dips
            w = rng.uniform(0.1, 0.6)
            ph = ((t - t0 + 0.5 * P) % P) - 0.5 * P
            prof = np.exp(-0.5 * (ph / (w / 2.355)) ** 2) * (rng.random() < 0.7)
            amp = 10 ** rng.uniform(-4, -2.3)
            model = 1 - amp * prof
            dbkg = amp * prof * rng.uniform(5, 50)            # also in background
            off = 0.0
    dim = 1 - model
    dcx, dcy = dim * off * np.cos(ang), dim * off * np.sin(ang)

    new = lc.select(np.ones(t.size, bool))
    new.sap = lc.sap * model
    new.pdc = lc.pdc * model
    new.aux = dict(lc.aux)
    new.aux["cx"] = lc.aux["cx"] + dcx
    new.aux["cy"] = lc.aux["cy"] + dcy
    if np.isfinite(lc.aux["bkg"]).all():
        new.aux["bkg"] = lc.aux["bkg"] * (1 + dbkg)
    truth = dict(label=CLASSES.index(cls), cls=cls, period=P, t0=t0,
                 depth=float(dim.max()), offset_px=float(off))
    return new, truth
