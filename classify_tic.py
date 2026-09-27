"""
Classify any TESS star by its TIC id.

    python classify_tic.py 305424003
    python classify_tic.py            (asks for the TIC id)

1. Finds that star's lc.fits files in your data folder. If they aren't there,
   downloads the SPOC 2-min light curves from NASA MAST (needs internet).
2. Runs the full AstroDetect pipeline: detrend -> search -> fit -> vetting -> CNN.
3. For every signal found prints planet / eclipsing binary / blend / other
   probabilities, the fitted planet parameters, and (optionally) the
   false-alarm probability, and saves a diagnostic plot per signal.
4. If the star is in the catalogue, shows NASA's answer next to ours, and
   warns if the star was used in training (then it isn't a fair test).
"""
import glob
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request

# ============================== PATHS / SETTINGS (edit if needed) ==================
CODE_DIR    = r"C:\Users\kinim\Desktop\Exoplanets\astrodetect_local"
DATA_DIR    = r"C:\Users\kinim\Desktop\Exoplanets\results"               # your lc.fits (+ jobs.json)
RESULTS_DIR = r"C:\Users\kinim\Desktop\Exoplanets\astrodetect_results"   # contains models\model_raw.pt
OUT_DIR     = r"C:\Users\kinim\Desktop\Exoplanets\classified"            # plots + csv per star
DOWNLOAD_DIR = os.path.join(OUT_DIR, "_mast_downloads")
VARIANT     = "raw"      # "raw" = our model, "pdc" = NASA-baseline model
MAX_SECTORS = 4          # sectors used per star (more = slower, more transits)
FAP         = True       # empirical false-alarm probability (~30-60 s extra)
# ===================================================================================

sys.path.insert(0, CODE_DIR)
import numpy as np  # noqa: E402

CLASSES = ["planet", "eclipsing_binary", "blend", "other"]
PRETTY = {"planet": "PLANET", "eclipsing_binary": "ECLIPSING BINARY", "blend": "BLEND (neighbour star)",
          "other": "OTHER (noise / instrumental / variability)"}


# ------------------------------------------------------------------ find / download files
def local_files(tic):
    hits = []
    for f in glob.glob(os.path.join(DATA_DIR, "**", "*.fits"), recursive=True) + \
             glob.glob(os.path.join(DOWNLOAD_DIR, "**", "*.fits"), recursive=True):
        parent = os.path.basename(os.path.dirname(f))
        m = re.search(r"-(\d{16})-", os.path.basename(f))
        if parent == f"TIC_{tic}" or (m and int(m.group(1)) == tic):
            hits.append(f)
    uniq = {}                                   # same file may exist in several folders
    for f in sorted(hits):
        uniq.setdefault(os.path.basename(f), f)
    return sorted(uniq.values())


def _mast(request):
    data = urllib.parse.urlencode({"request": json.dumps(request)}).encode()
    with urllib.request.urlopen("https://mast.stsci.edu/api/v0/invoke", data, timeout=120) as r:
        return json.loads(r.read())


def download_from_mast(tic, max_files):
    """SPOC 2-min lc.fits for this TIC from MAST (public, no login)."""
    print(f"not in your data folder -> downloading TIC {tic} from NASA MAST ...")
    obs = _mast({"service": "Mast.Caom.Filtered", "format": "json",
                 "params": {"columns": "obsid,obs_id,t_exptime",
                            "filters": [{"paramName": "obs_collection", "values": ["TESS"]},
                                        {"paramName": "dataproduct_type", "values": ["timeseries"]},
                                        {"paramName": "target_name", "values": [str(tic)]}]}})["data"]
    obs = [o for o in obs if o.get("t_exptime") == 120 and o["obs_id"].endswith("-s")]
    if not obs:
        sys.exit(f"MAST has no 2-min SPOC light curves for TIC {tic}")
    out = os.path.join(DOWNLOAD_DIR, f"TIC_{tic}")
    os.makedirs(out, exist_ok=True)
    got = []
    for o in sorted(obs, key=lambda o: o["obs_id"])[:max_files]:
        prods = _mast({"service": "Mast.Caom.Products", "format": "json",
                       "params": {"obsid": str(o["obsid"])}})["data"]
        for p in prods:
            if p["productFilename"].endswith("_lc.fits"):
                dst = os.path.join(out, p["productFilename"])
                if not os.path.exists(dst):
                    url = "https://mast.stsci.edu/api/v0.1/Download/file?uri=" + urllib.parse.quote(p["dataURI"])
                    urllib.request.urlretrieve(url, dst)
                    print("   downloaded", p["productFilename"])
                got.append(dst)
    return sorted(got)


# ------------------------------------------------------------------ catalogue answer
def catalogue_info(tic):
    jj = glob.glob(os.path.join(DATA_DIR, "**", "jobs.json"), recursive=True)
    if jj:
        for j in json.load(open(jj[0])):
            if j["tic"] == tic:
                return j["signals"], j["split"]
    return [], None


# ------------------------------------------------------------------ main
def classify(tic):
    import warnings
    warnings.filterwarnings("ignore")
    from astrodetect.detect import plot_candidate
    from astrodetect.pipeline import analyse_candidate, blind_detrend, load_star
    from astrodetect.search import iterative_search, matches
    from astrodetect.train import load_model

    model_hits = glob.glob(os.path.join(RESULTS_DIR, "**", f"model_{VARIANT}.pt"), recursive=True)
    if not model_hits:
        sys.exit(f"model_{VARIANT}.pt not found under {RESULTS_DIR}")
    predict = load_model(os.path.dirname(model_hits[0]), VARIANT)

    paths = local_files(tic) or download_from_mast(tic, MAX_SECTORS)
    signals, split = catalogue_info(tic)
    print(f"\nTIC {tic}: {len(paths)} light-curve file(s)")
    if split == "train" or split == "val":
        print(f"  NOTE: this star was in the '{split}' set - the model has seen it, so this is not a fair test.")
    elif split == "test":
        print("  this star is in the held-out TEST set - the model never saw it (fair test).")

    t0 = time.time()
    lc = load_star(paths, max_sectors=MAX_SECTORS)
    if lc is None:
        sys.exit("could not read any light curve")
    print(f"  sectors used: {lc.meta.get('sectors')}  |  {len(lc)} measurements  |  "
          f"Teff {lc.meta.get('teff')} K, R* {lc.meta.get('radius')} Rsun, Tmag {lc.meta.get('tmag')}")
    det = blind_detrend(lc, "raw")
    cands = iterative_search(lc.time, det["flux"], max_cands=3, sde_min=7.0)
    if not cands:
        print("\n  RESULT: no periodic transit-like signal found (nothing to classify).")
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []
    for i, c in enumerate(cands, 1):
        a = analyse_candidate(lc, c, "raw")
        p = predict(a["g"], a["l"], a["s"])
        f, d = a["fit"], a["diag"]
        row = dict(tic=tic, cand=i, period_d=f["period"], t0_btjd=f["t0"], duration_h=f["duration"] * 24,
                   depth_ppm=d["depth_ppm"], rp_earth=f["rp_earth"], rp_rs=f["rp_rs"], b=f["b"],
                   snr=d["snr"], sde=c["sde"], odd_even_sigma=d["odd_even_sigma"],
                   secondary_sigma=d["secondary_sigma"], centroid_sigma=d["centroid_sigma"],
                   source_offset_px=d["source_offset_px"],
                   **{f"p_{k}": float(v) for k, v in zip(CLASSES, p)},
                   predicted_class=CLASSES[int(np.argmax(p))], fap=np.nan, sigma=np.nan)
        if FAP:
            from astrodetect.significance import empirical_fap
            s = empirical_fap(lc.time, det["flux"], c, np.random.default_rng(0))
            row.update(fap=s["fap"], sigma=s["sigma"])
        rows.append(row)

        # ---------- print
        print("\n" + "-" * 66)
        print(f"  SIGNAL {i}:  period {row['period_d']:.5f} d | depth {row['depth_ppm']:.0f} ppm | "
              f"duration {row['duration_h']:.2f} h")
        print(f"  size: Rp/R* = {row['rp_rs']:.3f}  ->  Rp = {row['rp_earth']:.1f} Earth radii | impact b = {row['b']:.2f}")
        print(f"  strength: SNR {row['snr']:.1f}, SDE {row['sde']:.1f}"
              + (f", false-alarm probability {row['fap']:.1e} ({row['sigma']:.1f} sigma)" if FAP else ""))
        print(f"  checks: odd/even {row['odd_even_sigma']:.1f} sigma, secondary eclipse {row['secondary_sigma']:.1f} sigma, "
              f"centroid shift {row['centroid_sigma']:.1f} sigma (offset {row['source_offset_px']:.2f} px)")
        print("  class probabilities:")
        for k in CLASSES:
            bar = "#" * int(round(row[f"p_{k}"] * 30))
            print(f"     {k:17s} {100 * row[f'p_{k}']:5.1f} %  {bar}")
        verdict = PRETTY[row["predicted_class"]]
        if FAP and row["fap"] > 1e-3 and row["predicted_class"] != "other":
            verdict += "   (but FAP > 0.001: could be noise)"
        print(f"  ==> AstroDetect says: {verdict}")
        cat = next((s for s in signals if matches(row["period_d"], s["period"])), None)
        if cat:
            nasa = {0: "confirmed PLANET", -2: "FALSE POSITIVE", -1: "candidate (not yet confirmed)"}[cat["label"]]
            print(f"  ==> NASA catalogue ({cat['uid']}, P = {cat['period']:.4f} d): {nasa}")
        png = os.path.join(OUT_DIR, f"TIC{tic}_signal{i}.png")
        try:
            plot_candidate(paths, row, png)
            print(f"  plot: {png}")
        except Exception as e:
            print("  (plot failed:", e, ")")

    missed = [s for s in signals if not any(matches(r["period_d"], s["period"]) for r in rows)]
    for s in missed:
        print(f"\n  catalogue signal {s['uid']} (P = {s['period']:.4f} d) was NOT found in the sectors used.")
    import pandas as pd
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, f"TIC{tic}_result.csv"), index=False)
    print(f"\n  done in {time.time() - t0:.0f} s  |  table: {os.path.join(OUT_DIR, f'TIC{tic}_result.csv')}")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else input("TIC id: ")
    classify(int(re.sub(r"\D", "", arg)))
