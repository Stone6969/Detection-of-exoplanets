"""End-to-end glue: one star in -> training samples out (build mode) or
candidates with classes + significance out (detect mode)."""
import numpy as np

from .characterize import fit_transit
from .detrend import _detrend_centroid, detrend_pdc, detrend_raw
from .io import bin_lc, read_lc, stitch
from .search import iterative_search, matches, transit_mask
from .significance import empirical_fap
from .simulate import CLASSES, inject
from .vetting import diagnostics
from .views import make_views, scalar_vector

BTJD = 2457000.0
BUILD_PERIODS = 30_000     # coarser BLS grid for training-set building (3x faster)


def load_star(paths, max_sectors=None, rng=None):
    lcs = []
    for p in paths:
        try:
            lcs.append(bin_lc(read_lc(p)))
        except Exception as e:  # corrupt / unexpected file: skip, keep going
            print(f"  skip {p}: {e}")
    if max_sectors and len(lcs) > max_sectors:
        # take CONSECUTIVE sectors: a short baseline keeps the BLS grid small and
        # fine (two sectors years apart made the search ~10x slower and coarser)
        rng = rng or np.random.default_rng(0)
        lcs = sorted(lcs, key=lambda l: int(l.sector[0]))
        i0 = int(rng.integers(0, len(lcs) - max_sectors + 1))
        lcs = lcs[i0:i0 + max_sectors]
    return stitch(lcs)


def blind_detrend(lc, variant="raw"):
    return detrend_raw(lc) if variant == "raw" else detrend_pdc(lc)


def analyse_candidate(lc, cand, variant="raw"):
    """Candidate pass: re-detrend with this candidate's transits protected,
    fit the transit model, run the diagnostics, build the model inputs."""
    km = transit_mask(lc.time, cand["period"], cand["t0"], cand["duration"], pad=1.5)
    d = detrend_raw(lc, known_mask=km) if variant == "raw" else detrend_pdc(lc, known_mask=km)
    m = lc.meta
    fit = fit_transit(lc.time, d["flux"], cand, m.get("teff"), m.get("radius"), m.get("logg"),
                      exp_time=m.get("cadence_days", 2 / 1440))
    if fit is None or not np.isfinite(fit["period"]):
        fit = dict(period=cand["period"], t0=cand["t0"], duration=cand["duration"],
                   depth=cand.get("depth", 0), rp_rs=np.sqrt(max(cand.get("depth", 0), 0)),
                   b=0.5, rp_earth=np.nan, rho_ratio=np.nan, beta=1.0, fit_ok=False)
    # guard against a fit that ran away from the searched signal
    if abs(fit["period"] / cand["period"] - 1) > 0.01 or not (0 < fit["duration"] < 0.5 * fit["period"]):
        fit.update(period=cand["period"], t0=cand["t0"], duration=cand["duration"])
    bkg = lc.aux["bkg"]
    bkg_d = (_detrend_centroid(lc.time, bkg / np.nanmedian(bkg), km, 0.75)
             if np.isfinite(bkg).all() else np.full(lc.time.size, np.nan))
    diag = diagnostics(lc.time, d["flux"], d["cx"], d["cy"], bkg_d, lc.sector,
                       fit["period"], fit["t0"], fit["duration"])
    g, l = make_views(lc.time, d["flux"], d["cx"], d["cy"], bkg_d,
                      fit["period"], fit["t0"], fit["duration"], diag["depth_ppm"] / 1e6)
    s = scalar_vector(diag, fit, cand.get("sde", 0.0), m)
    return dict(g=g, l=l, s=s, fit=fit, diag=diag, flux=d["flux"])


def _sde_at(periodogram, period):
    if periodogram is None:
        return 0.0
    P, pw = periodogram
    i = np.argmin(np.abs(P - period))
    return float((pw[i] - pw.mean()) / pw.std())


# ============================================================ BUILD (training)
def build_star_samples(paths, signals, rng, n_sims=2, variants=("raw", "pdc"),
                       max_sectors=2, max_cands=2):
    """signals: list of dicts with period, t0 (BTJD), duration (days), depth,
    label (0 planet / -2 real FP / -1 unlabeled candidate), uid.
    Returns list of sample dicts (views + scalars + label + provenance)."""
    lc = load_star(paths, max_sectors, rng)
    if lc is None or len(lc) < 1000:
        return [], []
    out, recovery = [], []

    # ---- real catalogue signals: found by OUR blind search? -----------------
    det = blind_detrend(lc, "raw")
    cands = iterative_search(lc.time, det["flux"], max_cands=min(3, max_cands + len(signals)),
                             sde_min=6.0, pmax=30.0, keep_periodogram=True, max_periods=BUILD_PERIODS)
    pg = (cands[0]["periods"], cands[0]["power"]) if cands else None
    for sig in signals:
        if not np.isfinite(sig["period"]) or sig["period"] <= 0 or sig["period"] > 60:
            continue
        found = next((c for c in cands if matches(c["period"], sig["period"]) == 1
                      and abs(((c["t0"] - sig["t0"]) / c["period"] + 0.5) % 1 - 0.5) < 0.05), None)
        cand = (dict(found) if found else
                dict(period=sig["period"], t0=sig["t0"], duration=sig["duration"],
                     depth=sig["depth"], sde=_sde_at(pg, sig["period"])))
        recovery.append(dict(uid=sig["uid"], kind="real", label=sig["label"],
                             recovered=found is not None, sde=cand["sde"]))
        for v in variants:
            a = analyse_candidate(lc, cand, v)
            out.append(_pack(a, sig["label"], v, sig["uid"], lc.tic, "real", found is not None))

    # ---- simulations inside this star's real noise --------------------------
    bg = lc
    for sig in signals:                      # cut out every known signal
        if np.isfinite(sig["period"]) and sig["period"] > 0:
            bg = bg.select(~transit_mask(bg.time, sig["period"], sig["t0"],
                                         max(sig["duration"], 0.04), pad=2.5))
    if len(bg) < 1000:
        return out, recovery
    for k in range(n_sims):
        cls = CLASSES[rng.integers(0, 4)]
        sim, truth = inject(bg, cls, rng)
        sdet = blind_detrend(sim, "raw")
        sc = iterative_search(sim.time, sdet["flux"], max_cands=max_cands, sde_min=6.0, pmax=30.0,
                              max_periods=BUILD_PERIODS)
        if cls == "other":
            hit = sc[0] if sc else None
        else:
            hit = next((c for c in sc if matches(c["period"], truth["period"])), None)
        recovery.append(dict(uid=f"sim:{lc.tic}:{k}", kind="sim", label=truth["label"],
                             recovered=hit is not None, depth=truth["depth"],
                             period=truth["period"], sde=hit["sde"] if hit else 0.0))
        if hit is None:
            continue
        for v in variants:
            a = analyse_candidate(sim, hit, v)
            out.append(_pack(a, truth["label"], v, f"sim:{lc.tic}:{k}", lc.tic, "sim", True,
                             extra=dict(true_period=truth["period"], true_depth=truth["depth"],
                                        offset_px=truth["offset_px"])))
    return out, recovery


def _pack(a, label, variant, uid, tic, kind, recovered, extra=None):
    d = dict(g=a["g"], l=a["l"], s=a["s"], label=int(label), variant=variant, uid=uid,
             tic=int(tic), kind=kind, recovered=bool(recovered),
             period=a["fit"]["period"], depth_ppm=a["diag"]["depth_ppm"], snr=a["diag"]["snr"])
    d.update(extra or {})
    return d


# ============================================================ DETECT (unseen)
def detect_star(paths, predict_fn, max_cands=3, sde_min=7.0, n_null=9, fap_max=1e-3,
                p_planet_min=0.5, rng=None):
    """Full pipeline on an unseen star. predict_fn(g, l, s) -> class probs."""
    rng = rng or np.random.default_rng(0)
    lc = load_star(paths)
    if lc is None:
        return []
    det = blind_detrend(lc, "raw")
    cands = iterative_search(lc.time, det["flux"], max_cands=max_cands, sde_min=sde_min)
    rows = []
    for i, c in enumerate(cands):
        a = analyse_candidate(lc, c, "raw")
        probs = predict_fn(a["g"], a["l"], a["s"])
        sig = empirical_fap(lc.time, det["flux"], c, rng, n_scramble=n_null)
        f, dg = a["fit"], a["diag"]
        cls = CLASSES[int(np.argmax(probs))]
        rows.append(dict(
            tic=lc.tic, cand=i + 1, sectors=",".join(map(str, lc.meta.get("sectors", []))),
            period_d=f["period"], period_err=f.get("period_err", np.nan),
            t0_btjd=f["t0"], t0_err=f.get("t0_err", np.nan),
            duration_h=f["duration"] * 24, depth_ppm=dg["depth_ppm"],
            rp_rs=f["rp_rs"], rp_rs_err=f.get("rp_rs_err", np.nan), b=f["b"],
            rp_earth=f["rp_earth"], rho_ratio=f["rho_ratio"],
            snr=dg["snr"], sde=c["sde"], fap=sig["fap"], sigma=sig["sigma"],
            odd_even_sigma=dg["odd_even_sigma"], secondary_sigma=dg["secondary_sigma"],
            centroid_sigma=dg["centroid_sigma"], source_offset_px=dg["source_offset_px"],
            **{f"p_{k}": float(p) for k, p in zip(CLASSES, probs)},
            predicted_class=cls,
            is_detection=bool(sig["fap"] < fap_max and probs[0] >= p_planet_min),
        ))
    return rows
