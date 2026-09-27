"""Run the full pipeline on UNSEEN light curves.

    python -m astrodetect.detect --input /path/to/lc_folder --model models/ --out results/

Files are grouped by TIC (from the FITS header), every sector of a star is
stitched, and each star goes through detrend -> search -> fit -> vetting ->
classifier -> empirical FAP. Output: candidates.csv + one diagnostic PNG per
candidate.
"""
import argparse
import glob
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits

from .pipeline import analyse_candidate, detect_star, load_star
from .vetting import phase


def group_by_tic(folder):
    groups = defaultdict(list)
    for p in sorted(glob.glob(os.path.join(folder, "**", "*.fits"), recursive=True)):
        try:
            groups[int(fits.getheader(p, 0).get("TICID"))].append(p)
        except Exception:
            print("skip (no TICID):", p)
    return groups


def plot_candidate(paths, row, out_png):
    lc = load_star(paths)
    c = dict(period=row["period_d"], t0=row["t0_btjd"], duration=row["duration_h"] / 24,
             depth=row["depth_ppm"] / 1e6, sde=row["sde"])
    a = analyse_candidate(lc, c, "raw")
    t, f = lc.time, a["flux"]
    P, t0, D = c["period"], c["t0"], c["duration"]
    ph = phase(t, P, t0) * 24
    fig, ax = plt.subplots(2, 3, figsize=(15, 7))
    ax[0, 0].plot(t, lc.sap, ",", color="0.6")
    ax[0, 0].set_title("raw SAP (crowding-corrected)")
    ax[0, 1].plot(t, (f - 1) * 1e6, ",", color="C0")
    ax[0, 1].set_title("detrended (ours, telemetry-aware)")
    near = np.abs(ph) < 3 * D * 24
    ax[0, 2].plot(ph[near], (f[near] - 1) * 1e6, ".", ms=1, alpha=0.4)
    ax[0, 2].set_title(f"fold P={P:.5f} d  depth={row['depth_ppm']:.0f} ppm")
    ep = np.round((t - t0) / P).astype(int)
    for k, col in ((0, "C0"), (1, "C3")):
        m = near & (ep % 2 == k)
        ax[1, 0].plot(ph[m], (f[m] - 1) * 1e6, ".", ms=1, alpha=0.4, color=col,
                      label="even" if k == 0 else "odd")
    ax[1, 0].legend(markerscale=8)
    ax[1, 0].set_title(f"odd/even  ({row['odd_even_sigma']:.1f} sigma)")
    ph2 = (((t - t0) % P) - 0.5 * P) * 24
    m2 = np.abs(ph2) < 3 * D * 24
    ax[1, 1].plot(ph2[m2], (f[m2] - 1) * 1e6, ".", ms=1, alpha=0.4, color="C2")
    ax[1, 1].set_title(f"phase 0.5 (secondary {row['secondary_sigma']:.1f} sigma)")
    for key, col in (("cx", "C4"), ("cy", "C5")):
        ax[1, 2].plot(ph[near], lc.aux[key][near] - np.nanmedian(lc.aux[key][near]), ".", ms=1,
                      alpha=0.3, color=col, label=key)
    ax[1, 2].set_title(f"centroid ({row['centroid_sigma']:.1f} sigma, "
                       f"offset {row['source_offset_px']:.2f} px)")
    ax[1, 2].legend(markerscale=8)
    for x in ax[1]:
        x.set_xlabel("hours from mid-transit")
    fig.suptitle(f"TIC {row['tic']} cand {row['cand']}: {row['predicted_class']} "
                 f"(p_planet={row['p_planet']:.2f})  SNR={row['snr']:.1f}  "
                 f"FAP={row['fap']:.1e} ({row['sigma']:.1f} sigma)  Rp={row['rp_earth']:.1f} Re")
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def run_folder(folder, model_dir, out_dir, variant="raw", plots=True, **kw):
    from .train import load_model
    os.makedirs(out_dir, exist_ok=True)
    predict = load_model(model_dir, variant)
    rows = []
    groups = group_by_tic(folder)
    for i, (tic, paths) in enumerate(groups.items()):
        print(f"[{i + 1}/{len(groups)}] TIC {tic}: {len(paths)} file(s)", flush=True)
        try:
            r = detect_star(paths, predict, **kw)
        except Exception as e:
            print("   failed:", e)
            continue
        rows += r
        if plots:
            for row in r:
                plot_candidate(paths, row, os.path.join(out_dir, f"TIC{tic}_c{row['cand']}.png"))
        pd.DataFrame(rows).to_csv(os.path.join(out_dir, "candidates.csv"), index=False)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="detections")
    ap.add_argument("--no-plots", action="store_true")
    a = ap.parse_args()
    df = run_folder(a.input, a.model, a.out, plots=not a.no_plots)
    print(df.to_string())
