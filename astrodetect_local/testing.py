"""
One-click evaluation for your folders. Put this file in astrodetect_local and run:

    python run_eval.py

Edit the PATHS / SETTINGS block below if anything moves.
"""
import glob
import os
import sys
import tarfile
import zipfile

# ============================== PATHS (edit if needed) ==============================
CODE_DIR    = r"C:\Users\kinim\Desktop\Exoplanets\astrodetect_local"      # evaluate_local.py + astrodetect\
DATA_DIR    = r"C:\Users\kinim\Desktop\Exoplanets\results"                # jobs.json + zipped raw TIC files
RESULTS_DIR = r"C:\Users\kinim\Desktop\Exoplanets\astrodetect_results"    # contains models\model_raw.pt
OUT_DIR     = r"C:\Users\kinim\Desktop\Exoplanets\eval_results"           # where the report goes

# ============================== SETTINGS ============================================
SPLIT     = "test"   # "test" = only stars the model never trained on; "all" = every star
MAX_STARS = 2   # e.g. 10 for a quick check, None = all
WORKERS   = max(1, (os.cpu_count() or 2) - 1)
VARIANT   = "raw"    # "raw" = our model, "pdc" = NASA-baseline model
FAP       = False    # True = also compute false-alarm probability (much slower)
# ====================================================================================

EXTRACT_DIR = os.path.join(DATA_DIR, "_extracted")


def extract_archives():
    """Unpack every .zip / .tar / .tar.gz under DATA_DIR once (nested archives too)."""
    os.makedirs(EXTRACT_DIR, exist_ok=True)
    done_flag = os.path.join(EXTRACT_DIR, ".done")
    if os.path.exists(done_flag):
        print("archives already extracted ->", EXTRACT_DIR)
        return
    seen = set()
    while True:
        archives = [a for a in glob.glob(os.path.join(DATA_DIR, "**", "*"), recursive=True)
                    if a.lower().endswith((".zip", ".tar", ".tar.gz", ".tgz")) and a not in seen]
        if not archives:
            break
        for a in archives:
            seen.add(a)
            name = os.path.splitext(os.path.basename(a))[0].replace(".tar", "")
            dest = os.path.join(EXTRACT_DIR, name)
            print("extracting", os.path.relpath(a, DATA_DIR), "->", os.path.relpath(dest, DATA_DIR))
            try:
                if a.lower().endswith(".zip"):
                    with zipfile.ZipFile(a) as z:
                        z.extractall(dest)
                else:
                    with tarfile.open(a) as t:
                        t.extractall(dest)
            except Exception as e:
                print("   could not extract:", e)
    open(done_flag, "w").close()


def find_model_dir():
    hits = glob.glob(os.path.join(RESULTS_DIR, "**", f"model_{VARIANT}.pt"), recursive=True)
    if not hits:
        sys.exit(f"model_{VARIANT}.pt not found under {RESULTS_DIR} (unzip astrodetect_results.zip there)")
    return os.path.dirname(hits[0])


def main():
    for p in (CODE_DIR, DATA_DIR, RESULTS_DIR):
        if not os.path.isdir(p):
            sys.exit(f"folder not found: {p}")
    sys.path.insert(0, CODE_DIR)
    extract_archives()
    model_dir = find_model_dir()
    n_fits = len(glob.glob(os.path.join(DATA_DIR, "**", "*.fits"), recursive=True))
    has_jobs = bool(glob.glob(os.path.join(DATA_DIR, "**", "jobs.json"), recursive=True))
    print(f"\nmodel : {model_dir}\ndata  : {DATA_DIR}  ({n_fits} .fits files, jobs.json {'found' if has_jobs else 'NOT found'})\n")
    if n_fits == 0:
        sys.exit("no .fits files found after extraction - check DATA_DIR")

    args = ["evaluate_local.py", "--data", DATA_DIR, "--model", model_dir, "--out", OUT_DIR,
            "--split", SPLIT, "--workers", str(WORKERS), "--variant", VARIANT]
    if MAX_STARS:
        args += ["--max-stars", str(MAX_STARS)]
    if FAP:
        args.append("--fap")
    sys.argv = args
    import evaluate_local
    evaluate_local.main()


if __name__ == "__main__":      # required on Windows (worker processes)
    main()