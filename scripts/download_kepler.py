"""Download Kepler long-cadence light curves for every star in kepler_selected.csv.

For each unique star (kepid) this script:
  1. searches the archive for its long-cadence Kepler light curves (one per quarter),
  2. downloads them all and stitches them into one light curve
     (each quarter is divided by its own median, so they line up at ~1.0),
  3. saves time, flux and flux_err to data/kepler_raw/<kepid>.csv.

Stars that already have a file are skipped, so you can stop the script and run it
again and it carries on where it left off. Stars that fail are written to
data/failed.csv and tried again on the next run.

Usage (run from the project root):
    python scripts/download_kepler.py            # all stars
    python scripts/download_kepler.py --limit 3  # only the first 3 stars (for testing)
    python scripts/download_kepler.py --kepid 6849046 3762468   # only these stars
"""

import argparse
import shutil
import tempfile
import time
import warnings
from datetime import datetime
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")  # lightkurve/astropy print a lot of harmless warnings
import lightkurve as lk  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
INPUT_CSV = ROOT / "data" / "kepler_selected.csv"
OUT_DIR = ROOT / "data" / "kepler_raw"
FAILED_CSV = ROOT / "data" / "failed.csv"


def download_star(kepid: int) -> pd.DataFrame:
    """Download and stitch all long-cadence Kepler quarters for one star."""
    search = lk.search_lightcurve(
        f"KIC {kepid}", mission="Kepler", author="Kepler", cadence="long"
    )
    if len(search) == 0:
        raise RuntimeError("no long-cadence Kepler light curves found")

    # Download into a throwaway folder. Otherwise lightkurve keeps every FITS file
    # in its cache (~/.lightkurve), which would grow to several GB.
    tmp_dir = tempfile.mkdtemp(prefix="kepler_")
    try:
        collection = search.download_all(download_dir=tmp_dir)
        if collection is None or len(collection) == 0:
            raise RuntimeError("download returned no data")
        lc = collection.stitch().remove_nans()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return pd.DataFrame({
        "time": lc.time.value,            # days, Kepler clock (BKJD), same as koi_time0bk
        "flux": lc.flux.value,            # brightness, normalised to ~1.0
        "flux_err": lc.flux_err.value,    # uncertainty on the brightness
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=None, help="only process the first N stars")
    parser.add_argument("--kepid", type=int, nargs="+", default=None, help="only process these kepids")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # One star can have several KOIs, so keep each kepid only once (in file order).
    kepids = pd.read_csv(INPUT_CSV)["kepid"].drop_duplicates().tolist()
    if args.kepid:
        kepids = [k for k in kepids if k in set(args.kepid)]
    if args.limit:
        kepids = kepids[: args.limit]

    # Failures from earlier runs, so we keep a record of the ones that still fail.
    failures = {}
    if FAILED_CSV.exists():
        old = pd.read_csv(FAILED_CSV)
        failures = {int(r.kepid): r._asdict() for r in old.itertuples(index=False)}

    total = len(kepids)
    n_done = n_skipped = n_failed = 0
    start = time.time()

    for i, kepid in enumerate(kepids, start=1):
        out_path = OUT_DIR / f"{kepid}.csv"
        prefix = f"[{i}/{total}] KIC {kepid}"

        if out_path.exists():
            n_skipped += 1
            failures.pop(kepid, None)
            print(f"{prefix}: already downloaded, skipping")
            continue

        try:
            df = download_star(kepid)
            # Write to a temporary name first and rename at the end, so a crash
            # half-way through never leaves a broken file that looks "done".
            tmp_path = out_path.with_suffix(".csv.part")
            df.to_csv(tmp_path, index=False)
            tmp_path.replace(out_path)
            failures.pop(kepid, None)
            n_done += 1
            print(f"{prefix}: saved {len(df):,} points")
        except KeyboardInterrupt:
            print("\nStopped by user. Run the script again to continue.")
            break
        except Exception as e:  # network errors, missing data, etc.
            n_failed += 1
            failures[kepid] = {
                "kepid": kepid,
                "error": f"{type(e).__name__}: {e}".replace("\n", " ")[:300],
                "time": datetime.now().isoformat(timespec="seconds"),
            }
            print(f"{prefix}: FAILED ({failures[kepid]['error']})")

        # Save the failure list after every star, so it's up to date if the script stops.
        pd.DataFrame(list(failures.values()), columns=["kepid", "error", "time"]).to_csv(FAILED_CSV, index=False)

    minutes = (time.time() - start) / 60
    print(
        f"\nDone in {minutes:.1f} min: {n_done} downloaded, {n_skipped} skipped "
        f"(already there), {n_failed} failed. Failures are listed in {FAILED_CSV.relative_to(ROOT)}"
    )


if __name__ == "__main__":
    main()
