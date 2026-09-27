"""
Test a trained AstroDetect model on your own computer.

What it does, for every star in a folder of TESS lc.fits files:
  detrend -> BLS search -> transit fit -> vetting -> CNN classification
  (-> optional empirical false-alarm probability)
then compares the results with the catalogue labels and prints how accurate
the model is.

Usage (from the folder that contains this file and the `astrodetect/` package):

  python evaluate_local.py --data "D:/exoplanet/exoplanetdatahuggingface" --model "D:/results/models"

  --data   folder with lc.fits files (searched recursively). The folder of the
           Kaggle dataset works directly: it also contains jobs.json/signals.csv.
  --model  folder with model_raw.pt + model_raw.json (from astrodetect_results.zip)

Useful options:
  --split test      only stars the model never saw in training (default). 'all' = every star
  --max-stars 30    stop after N stars (quick check)
  --workers 4       CPU processes in parallel
  --fap             also compute the false-alarm probability (slower, ~30-60 s/star)
  --variant pdc     evaluate the NASA-baseline model instead of ours (raw)

Output (in --out, default ./eval_results):
  candidates.csv   one row per detected candidate with class probabilities
  per_signal.csv   one row per catalogue signal: found? predicted class? p_planet
  metrics.json     all numbers printed at the end
  roc.png          ROC curve for real planet vs false positive
"""
import argparse
import glob
import json
import os
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

CLASSES = ["planet", "eclipsing_binary", "blend", "other"]
LABEL_NAME = {0: "planet", -2: "false_positive", -1: "unlabelled"}


# ------------------------------------------------------------------ inputs
def find_fits(data_dir):
    """TIC id -> list of lc.fits paths (from folder name TIC_<id> or file name)."""
    groups = defaultdict(list)
    for f in glob.glob(os.path.join(data_dir, "**", "*.fits"), recursive=True):
        name = os.path.basename(f)
        parent = os.path.basename(os.path.dirname(f))
        if parent.startswith("TIC_"):
            tic = int(parent[4:])
        else:
            m = re.search(r"-(\d{16})-", name)
            if not m:
                continue
            tic = int(m.group(1))
        groups[tic].append(f)
    return groups


def load_labels(data_dir, signals_csv=None, jobs_json=None):
    """TIC -> list of catalogue signals {uid, period, t0, duration, label} and split."""
    from astrodetect.build import star_signals, split_of
    jobs_json = jobs_json or next(iter(glob.glob(os.path.join(data_dir, "**", "jobs.json"), recursive=True)), None)
    if jobs_json:
        jobs = json.load(open(jobs_json))
        return {j["tic"]: j["signals"] for j in jobs}, {j["tic"]: j["split"] for j in jobs}
    signals_csv = signals_csv or next(iter(glob.glob(os.path.join(data_dir, "**", "signals.csv"), recursive=True)), None)
    if signals_csv:
        import pandas as pd
        sig = pd.read_csv(signals_csv)
        labels = {int(t): star_signals(g) for t, g in sig.groupby("tic_id")}
        return labels, {t: split_of(t) for t in labels}
    return {}, {}


# ------------------------------------------------------------------ one star (runs in a worker)
_PREDICT = None


def _init(model_dir, variant):
    global _PREDICT
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"
    import torch
    torch.set_num_threads(1)
    from astrodetect.train import load_model
    _PREDICT = load_model(model_dir, variant)


def process_star(tic, paths, signals, use_fap, max_sectors):
    import warnings
    warnings.filterwarnings("ignore")
    from astrodetect.pipeline import analyse_candidate, blind_detrend, load_star
    from astrodetect.search import iterative_search, matches
    t0 = time.time()
    try:
        lc = load_star(sorted(paths), max_sectors=max_sectors, rng=np.random.default_rng(tic))
        if lc is None or len(lc) < 1000:
            return tic, [], [], "too little data", time.time() - t0
        det = blind_detrend(lc, "raw")
        cands = iterative_search(lc.time, det["flux"], max_cands=3, sde_min=7.0)
        rows = []
        for i, c in enumerate(cands):
            a = analyse_candidate(lc, c, "raw")
            p = _PREDICT(a["g"], a["l"], a["s"])
            row = dict(tic=tic, cand=i + 1, period_d=a["fit"]["period"], t0_btjd=a["fit"]["t0"],
                       duration_h=a["fit"]["duration"] * 24, depth_ppm=a["diag"]["depth_ppm"],
                       rp_earth=a["fit"]["rp_earth"], snr=a["diag"]["snr"], sde=c["sde"],
                       centroid_sigma=a["diag"]["centroid_sigma"],
                       odd_even_sigma=a["diag"]["odd_even_sigma"],
                       **{f"p_{k}": float(v) for k, v in zip(CLASSES, p)},
                       predicted_class=CLASSES[int(np.argmax(p))])
            if use_fap:
                from astrodetect.significance import empirical_fap
                s = empirical_fap(lc.time, det["flux"], c, np.random.default_rng(0))
                row.update(fap=s["fap"], sigma=s["sigma"])
            rows.append(row)
        per_sig = []
        for s in signals:
            hit = next((r for r in rows if matches(r["period_d"], s["period"]) == 1), None)
            per_sig.append(dict(tic=tic, uid=s["uid"], label=LABEL_NAME.get(s["label"], "?"),
                                catalogue_period=s["period"], found=hit is not None,
                                found_period=hit["period_d"] if hit else np.nan,
                                p_planet=hit["p_planet"] if hit else np.nan,
                                predicted_class=hit["predicted_class"] if hit else "not found"))
        return tic, rows, per_sig, None, time.time() - t0
    except Exception as e:
        return tic, [], [], repr(e), time.time() - t0


# ------------------------------------------------------------------ report
def report(per_sig, cands, out, variant):
    import pandas as pd
    from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
    P = pd.DataFrame(per_sig)
    C = pd.DataFrame(cands)
    C.to_csv(os.path.join(out, "candidates.csv"), index=False)
    P.to_csv(os.path.join(out, "per_signal.csv"), index=False)
    m = dict(model=variant, stars=int(C.tic.nunique()) if len(C) else 0, candidates=len(C))
    print("\n" + "=" * 64)
    print(f"  AstroDetect local evaluation  (model: {variant})")
    print("=" * 64)
    lab = P[P.label.isin(["planet", "false_positive"])] if len(P) else P
    if len(lab):
        # stage 2: did our search find the catalogue signal?
        for name in ("planet", "false_positive"):
            sub = lab[lab.label == name]
            if len(sub):
                m[f"found_{name}"] = float(sub.found.mean())
                print(f"  search found {sub.found.sum():4d} / {len(sub):4d} catalogue {name}s "
                      f"({100 * sub.found.mean():.0f} %)")
        # stage 4: classification of the ones that were found
        f = lab[lab.found]
        y = (f.label == "planet").astype(int).values
        p = f.p_planet.values
        if len(f):
            pred = p >= 0.5
            acc = float((pred == y).mean())
            m.update(n_classified=len(f), accuracy=acc,
                     planet_recall=float(pred[y == 1].mean()) if (y == 1).any() else None,
                     fp_rejected=float((~pred[y == 0]).mean()) if (y == 0).any() else None)
            print(f"\n  classification of the {len(f)} found signals (threshold p_planet >= 0.5):")
            print(f"    accuracy          {100 * acc:5.1f} %")
            if m["planet_recall"] is not None:
                print(f"    planets kept      {100 * m['planet_recall']:5.1f} %   ({pred[y == 1].sum()} / {(y == 1).sum()})")
            if m["fp_rejected"] is not None:
                print(f"    FPs rejected      {100 * m['fp_rejected']:5.1f} %   ({(~pred[y == 0]).sum()} / {(y == 0).sum()})")
            cm = confusion_matrix(y, pred.astype(int), labels=[1, 0])
            print("                      predicted planet  predicted not-planet")
            print(f"    true planet        {cm[0, 0]:10d}  {cm[0, 1]:18d}")
            print(f"    true FP            {cm[1, 0]:10d}  {cm[1, 1]:18d}")
            if len(set(y)) == 2:
                auc = float(roc_auc_score(y, p))
                m["roc_auc"] = auc
                print(f"    ROC-AUC           {auc:.3f}   (1.0 perfect, 0.5 random)")
                try:
                    import matplotlib
                    matplotlib.use("Agg")
                    import matplotlib.pyplot as plt
                    fpr, tpr, _ = roc_curve(y, p)
                    plt.figure(figsize=(4.5, 4.5))
                    plt.plot(fpr, tpr, lw=2, label=f"AUC {auc:.3f}")
                    plt.plot([0, 1], [0, 1], "k--", lw=.8)
                    plt.xlabel("false-positive rate"); plt.ylabel("planet recall")
                    plt.title(f"real planet vs false positive ({variant})"); plt.legend()
                    plt.tight_layout(); plt.savefig(os.path.join(out, "roc.png"), dpi=130)
                except Exception:
                    pass
            print("\n  predicted class of the found signals:")
            print(pd.crosstab(f.label, f.predicted_class).to_string())
    else:
        print("  no labelled catalogue signals for these stars - only candidates are listed")
    if len(C):
        print(f"\n  all candidates by predicted class: {C.predicted_class.value_counts().to_dict()}")
        if "fap" in C:
            print(f"  candidates with FAP < 1e-3: {(C.fap < 1e-3).sum()} / {len(C)}")
    json.dump(m, open(os.path.join(out, "metrics.json"), "w"), indent=2, default=float)
    print(f"\n  files written to {os.path.abspath(out)}")
    return m


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--variant", default="raw", choices=["raw", "pdc"])
    ap.add_argument("--split", default="test", choices=["test", "val", "train", "all"])
    ap.add_argument("--max-stars", type=int, default=None)
    ap.add_argument("--max-sectors", type=int, default=4)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--fap", action="store_true")
    ap.add_argument("--signals", default=None, help="signals.csv if there is no jobs.json")
    ap.add_argument("--jobs", default=None, help="jobs.json (star list + train/val/test split)")
    ap.add_argument("--out", default="eval_results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    for f in (f"model_{a.variant}.pt", f"model_{a.variant}.json"):
        if not os.path.exists(os.path.join(a.model, f)):
            sys.exit(f"missing {f} in {a.model}")
    groups = find_fits(a.data)
    labels, splits = load_labels(a.data, a.signals, a.jobs)
    tics = sorted(groups)
    if a.split != "all" and splits:
        tics = [t for t in tics if splits.get(t) == a.split]
        print(f"using the '{a.split}' split: {len(tics)} stars the model did not train on"
              if a.split == "test" else f"using the '{a.split}' split: {len(tics)} stars")
    elif a.split != "all":
        print("no jobs.json/signals.csv found: evaluating every star (no split info)")
    if a.max_stars:
        tics = tics[: a.max_stars]
    if not tics:
        sys.exit(f"no lc.fits files found under {a.data} for split '{a.split}'")
    print(f"{len(tics)} stars, {sum(len(groups[t]) for t in tics)} files, {a.workers} worker(s)"
          f"{', with FAP' if a.fap else ''}\n")

    cands, per_sig, t_start = [], [], time.time()
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(a.model, a.variant)) as ex:
        futs = [ex.submit(process_star, t, groups[t], labels.get(t, []), a.fap, a.max_sectors) for t in tics]
        for k, fu in enumerate(as_completed(futs), 1):
            tic, rows, ps, err, dt = fu.result()
            cands += rows
            per_sig += ps
            el = time.time() - t_start
            status = f"ERROR {err}" if err else (
                ", ".join(f"{r['predicted_class']} P={r['period_d']:.3f}d p_planet={r['p_planet']:.2f}" for r in rows)
                or "no candidate")
            print(f"[{k}/{len(tics)}] TIC {tic} ({dt:.0f}s): {status}   | ~{el / k * (len(tics) - k) / 60:.0f} min left",
                  flush=True)
    report(per_sig, cands, a.out, a.variant)


if __name__ == "__main__":
    main()
