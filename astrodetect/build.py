"""Build the training set from the Hugging Face dataset
saadtaleb/tess-lightcurves-planets (82 GB) without ever storing it:
each star's lc.fits files are downloaded, processed, and deleted.

Output: shards/shard_XXXXX.npz (+ recovery_XXXXX.csv), resumable.

Labels (from signals.csv, de-duplicated per star):
   0  planet          signal_label == confirmed  or TFOPWG in {CP, KP}
  -2  real FP         signal_label == false_positive or TFOPWG in {FP, FA}
  -1  unlabeled       everything else (PC / APC / candidate / unknown)
Split: by TIC hash -> train 70 / val 15 / test 15 %. Test stars are also the
"unseen" stars for the end-to-end detection benchmark.
"""
import hashlib
import json
import os
import shutil
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

REPO = "saadtaleb/tess-lightcurves-planets"
BTJD = 2457000.0


def split_of(tic):
    h = int(hashlib.md5(str(tic).encode()).hexdigest(), 16) % 100
    return "train" if h < 70 else ("val" if h < 85 else "test")


def label_of(row):
    lab = str(row.get("signal_label", "")).lower()
    tf = str(row.get("tfopwg_disposition", "")).upper()
    if lab == "confirmed" or tf in ("CP", "KP"):
        return 0
    if lab == "false_positive" or tf in ("FP", "FA"):
        return -2
    return -1


def load_metadata(cache_dir):
    from huggingface_hub import hf_hub_download
    get = lambda f: pd.read_csv(hf_hub_download(REPO, f, repo_type="dataset", cache_dir=cache_dir))
    return get("signals.csv"), get("lightcurves.csv"), get("targets.csv")


def star_signals(sig_rows):
    """De-duplicate TOI/CTOI/TCE entries of the same planet (period within 1 %)."""
    pri = {"TOI": 0, "TCE": 1, "CTOI": 2}
    rows = sorted(sig_rows.to_dict("records"), key=lambda r: pri.get(r["signal_source"], 3))
    out = []
    for r in rows:
        P = float(r.get("period_days") or np.nan)
        ep = float(r.get("epoch_bjd") or np.nan)
        dur = float(r.get("duration_hours") or np.nan) / 24
        dep = float(r.get("depth_ppm") or np.nan) / 1e6
        if not (np.isfinite(P) and P > 0 and np.isfinite(ep)):
            continue
        if ep > 2_000_000:
            ep -= BTJD
        lab = label_of(r)
        dup = next((o for o in out if abs(o["period"] / P - 1) < 0.01), None)
        if dup:
            if dup["label"] == -1 and lab != -1:     # keep the most informative label
                dup["label"] = lab
            continue
        out.append(dict(uid=r["signal_uid"], period=P, t0=ep,
                        duration=dur if np.isfinite(dur) and dur > 0 else 0.1,
                        depth=dep if np.isfinite(dep) and dep > 0 else 1e-3, label=lab))
    return out


def plan(signals, lcs, max_unlabeled_stars=1500, seed=0, file_types=("lc",), fraction=1.0):
    """Which stars to process: every star with a labelled signal + a sample of
    candidate-only stars (they supply real noise for the simulations)."""
    lcs = lcs[lcs["file_type"].isin(file_types)]
    by_tic = {tic: g for tic, g in signals.groupby("tic_id")}
    files = lcs.groupby("tic_id")["hf_path"].apply(list).to_dict()
    jobs = []
    for tic, g in by_tic.items():
        if tic not in files:
            continue
        sigs = star_signals(g)
        if sigs:
            jobs.append(dict(tic=int(tic), files=files[tic], signals=sigs, split=split_of(tic),
                             labelled=any(s["label"] != -1 for s in sigs)))
    lab = [j for j in jobs if j["labelled"]]
    unl = [j for j in jobs if not j["labelled"]]
    rng = np.random.default_rng(seed)
    unl = [unl[i] for i in rng.permutation(len(unl))[:max_unlabeled_stars]]
    if fraction < 1.0:
        # test run: random subset of stars, same fraction of labelled and
        # unlabelled stars; splits are TIC-hash based so they stay consistent
        lab = [lab[i] for i in sorted(rng.permutation(len(lab))[:int(round(fraction * len(lab)))])]
        unl = [unl[i] for i in sorted(rng.permutation(len(unl))[:int(round(fraction * len(unl)))])]
    return lab + unl


def _process(job, cfg):
    """Runs in a worker process: download -> build samples -> delete."""
    from huggingface_hub import hf_hub_download
    from .pipeline import build_star_samples
    tmp = os.path.join(cfg["tmp_dir"], f"TIC_{job['tic']}")
    local = cfg.get("local_root")
    try:
        paths = []
        files = job["files"][: cfg["max_files_per_star"]]
        if local:                    # files already pulled by notebook 00: read, never delete
            paths = [os.path.join(local, f) for f in files if os.path.exists(os.path.join(local, f))]
            files = []
        for f in files:
            for attempt in range(4):
                try:
                    paths.append(hf_hub_download(REPO, f, repo_type="dataset", local_dir=tmp))
                    break
                except Exception:
                    time.sleep(5 * (attempt + 1))
        rng = np.random.default_rng(job["tic"])
        n_sims = cfg["n_sims"] if job["split"] != "test" else cfg["n_sims_test"]
        samples, rec = build_star_samples(paths, job["signals"], rng, n_sims=n_sims,
                                          max_sectors=cfg["max_sectors"])
        for s in samples:
            s["split"] = job["split"]
        for r in rec:
            r.update(tic=job["tic"], split=job["split"])
        return job["tic"], samples, rec, None
    except Exception:
        return job["tic"], [], [], traceback.format_exc()
    finally:
        if not local:
            shutil.rmtree(tmp, ignore_errors=True)


def save_shard(samples, recs, path):
    if not samples:
        return
    arr = lambda k, dt=None: np.array([s.get(k, np.nan) for s in samples], dtype=dt)
    np.savez_compressed(
        path, G=np.stack([s["g"] for s in samples]), L=np.stack([s["l"] for s in samples]),
        S=np.stack([s["s"] for s in samples]), y=arr("label", np.int64),
        variant=arr("variant", str), kind=arr("kind", str), split=arr("split", str),
        uid=arr("uid", str), tic=arr("tic", np.int64), recovered=arr("recovered", bool),
        period=arr("period", float), depth_ppm=arr("depth_ppm", float), snr=arr("snr", float))
    pd.DataFrame(recs).to_csv(path.replace(".npz", "_recovery.csv"), index=False)


def run(out_dir, cache_dir="/tmp/hf_meta", tmp_dir="/tmp/lc", workers=4, shard_size=100,
        shard_id=0, n_shards=1, time_budget_h=11.0, jobs_file=None, local_root=None, **cfg_over):
    """jobs_file + local_root: use the star list and FITS files pulled by
    download_subset() (notebook 00) instead of downloading from Hugging Face."""
    cfg = dict(n_sims=2, n_sims_test=2, max_sectors=2, max_files_per_star=4,
               max_unlabeled_stars=1500, fraction=1.0, tmp_dir=tmp_dir, local_root=local_root)
    cfg.update(cfg_over)
    os.makedirs(out_dir, exist_ok=True)
    if jobs_file:
        jobs = json.load(open(jobs_file))
    else:
        signals, lcs, targets = load_metadata(cache_dir)
        jobs = plan(signals, lcs, cfg["max_unlabeled_stars"], fraction=cfg["fraction"])
    jobs = [j for i, j in enumerate(jobs) if i % n_shards == shard_id]     # split across sessions
    done_file = os.path.join(out_dir, f"done_{shard_id}.json")
    done = set(json.load(open(done_file))) if os.path.exists(done_file) else set()
    jobs = [j for j in jobs if j["tic"] not in done]
    print(f"{len(jobs)} stars to process (shard {shard_id}/{n_shards}), {len(done)} already done")
    t_start = time.time()
    buf_s, buf_r, k = [], [], len([f for f in os.listdir(out_dir) if f.endswith(".npz")])
    # one numeric thread per worker (4 workers x 4 BLAS threads each thrashed the
    # 4 Kaggle cores) and 'spawn' workers (forking after torch/OpenMP is loaded
    # can hang the children)
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[v] = "1"
    import multiprocessing as mp
    n_total, t_last = len(jobs), time.time()
    ctx = mp.get_context("spawn" if os.environ.get("ASTRODETECT_FORK") != "1" else "fork")
    with ProcessPoolExecutor(workers, mp_context=ctx) as ex:
        futs = {}
        it = iter(jobs)
        for j in [next(it, None) for _ in range(workers * 2)]:
            if j:
                futs[ex.submit(_process, j, cfg)] = j
        while futs:
            for fu in as_completed(list(futs)):
                futs.pop(fu)
                tic, s, r, err = fu.result()
                if err:
                    print(f"TIC {tic} failed:\n{err[-500:]}")
                buf_s += s
                buf_r += r
                done.add(tic)
                n_new = len(done) - (len(set(json.load(open(done_file)))) if os.path.exists(done_file) else 0)
                el = time.time() - t_start
                print(f"  [{len(done)}] TIC {tic}: {len(s)} samples | {el/60:.1f} min elapsed, "
                      f"~{el / max(1, n_new) * (n_total - n_new) / 60:.0f} min left", flush=True)
                if len(done) % shard_size == 0 and buf_s:
                    save_shard(buf_s, buf_r, os.path.join(out_dir, f"shard_{shard_id}_{k:04d}.npz"))
                    json.dump(sorted(done), open(done_file, "w"))
                    buf_s, buf_r, k = [], [], k + 1
                    el = (time.time() - t_start) / 3600
                    print(f"{len(done)} stars, {el:.2f} h", flush=True)
                if (time.time() - t_start) / 3600 < time_budget_h:
                    j = next(it, None)
                    if j:
                        futs[ex.submit(_process, j, cfg)] = j
                break
    save_shard(buf_s, buf_r, os.path.join(out_dir, f"shard_{shard_id}_{k:04d}.npz"))
    json.dump(sorted(done), open(done_file, "w"))
    print("finished", len(done))


# ====================================================================== pull only
def download_subset(out_dir, fraction=0.2, max_files_per_star=4, max_unlabeled_stars=1500,
                    stars_per_tar=100, threads=8, cache_dir="/tmp/hf_meta", tmp_dir="/tmp/pull",
                    time_budget_h=11.0, seed=0):
    """Notebook 00: download the raw lc.fits of a random `fraction` of the stars
    and pack them into raw_XXXX.tar files (100 stars each) + jobs.json.
    Tars keep Kaggle's output to a few dozen files; re-running skips finished tars."""
    import tarfile
    from concurrent.futures import ThreadPoolExecutor
    from huggingface_hub import hf_hub_download
    os.makedirs(out_dir, exist_ok=True)
    signals, lcs, targets = load_metadata(cache_dir)
    for name, df in (("signals.csv", signals), ("lightcurves.csv", lcs), ("targets.csv", targets)):
        df.to_csv(os.path.join(out_dir, name), index=False)
    jobs = plan(signals, lcs, max_unlabeled_stars, seed=seed, fraction=fraction)
    for j in jobs:
        j["files"] = j["files"][:max_files_per_star]
    json.dump(jobs, open(os.path.join(out_dir, "jobs.json"), "w"))
    n_files = sum(len(j["files"]) for j in jobs)
    print(f"{len(jobs)} stars, {n_files} files (~{n_files * 2 / 1000:.1f} GB)")

    def get(f):
        for attempt in range(5):
            try:
                return hf_hub_download(REPO, f, repo_type="dataset", local_dir=tmp_dir)
            except Exception as e:
                err = e
                time.sleep(5 * (attempt + 1))
        print("  failed:", f, err)
        return None

    t0 = time.time()
    chunks = [jobs[i:i + stars_per_tar] for i in range(0, len(jobs), stars_per_tar)]
    for k, chunk in enumerate(chunks):
        tar_path = os.path.join(out_dir, f"raw_{k:04d}.tar")
        if os.path.exists(tar_path):
            continue
        if (time.time() - t0) / 3600 > time_budget_h:
            print("time budget reached - run again (with this output attached) to continue")
            break
        files = [f for j in chunk for f in j["files"]]
        with ThreadPoolExecutor(threads) as ex:
            got = [p for p in ex.map(get, files) if p]
        with tarfile.open(tar_path + ".part", "w") as tar:
            for p in got:
                tar.add(p, arcname=os.path.relpath(p, tmp_dir))
        os.replace(tar_path + ".part", tar_path)
        shutil.rmtree(tmp_dir, ignore_errors=True)
        print(f"raw_{k:04d}.tar: {len(got)}/{len(files)} files, "
              f"{(time.time() - t0) / 60:.1f} min elapsed", flush=True)
    print("done")


def extract_subset(input_glob="/kaggle/input/**/raw_*.tar", dest="/tmp/raw"):
    """Notebook 01: unpack the tars from notebook 00; returns (jobs_file, dest)."""
    import glob
    import tarfile
    os.makedirs(dest, exist_ok=True)
    tars = sorted(glob.glob(input_glob, recursive=True))
    for t in tars:
        with tarfile.open(t) as tar:
            try:
                tar.extractall(dest, filter="data")
            except TypeError:
                tar.extractall(dest)
    jobs = glob.glob(os.path.join(os.path.dirname(tars[0]), "jobs.json")) if tars else []
    print(f"extracted {len(tars)} tars to {dest}")
    return (jobs[0] if jobs else None), dest
