"""
step07_combine_monte_carlo_parents.py
Concatenate the per-chunk summary CSVs written by 045b for one parent (truth)
into a single all-grid table, and print a console digest (global medians of
bias/SD/RMSE per fitted model, and |bias|/SD ratio -- the same diagnostic as
054). The digest is printed TWICE: for all cells, and for the FLOOD-RELEVANT
subset only ((q90>=50 OR max>=100) AND uparea>=50, from 091, matching 098/103).
Arid garbage (|bias%| or |sd%| > CAP_PCT) is dropped from the medians.

Input:  <dat_dir>/045b/<truth>/summary/summary_*.csv
        <dat_dir>/091/change_type_cells.csv   (optional; enables flood digest)
Output: <dat_dir>/045b/summary_allgrid_<truth>.csv

Usage:
  python3 step07_combine_monte_carlo_parents.py <truth{st,ln,qd}> [dat_dir]
"""
import os
import sys
import glob
import csv
import numpy as np
import importlib
excess_lib = importlib.import_module("common_target_cells")

MODELS = ["st", "ln", "qd", "ad"]
NAME = {"st": "stationary", "ln": "linear", "qd": "quadratic", "ad": "AIC-adaptive"}
STATS = ["bias", "sd", "iqr", "tail"]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Drop arid garbage: |bias%| or |sd%| above this is a truth_q100 ~ 0 cell whose
# relative error blew up to ~1e137 (also avoids the b**2 overflow warning).
CAP_PCT = 300.0


def load_flood_ids(dat_dir, x1=excess_lib.FLOOD_Q90, x2=excess_lib.FLOOD_MAX,
                   a_up=excess_lib.FLOOD_UPAREA):
    """Set of flood-relevant cell_ids from 091 (matches 098/103):
    keep = (q90 >= x1 OR max_amax >= x2) AND uparea_km2 >= a_up.
    Returns None if the classification file is absent."""
    cells_csv = os.path.join(dat_dir, "091", "change_type_cells.csv")
    if not os.path.exists(cells_csv):
        return None
    ids = set()
    with open(cells_csv, newline="") as f:
        for row in csv.DictReader(f):
            def num(k):
                v = row.get(k, "")
                return float(v) if v not in ("", "NA", "nan") else float("nan")
            if ((num("q90") >= x1) or (num("max_amax") >= x2)) and (num("uparea_km2") >= a_up):
                ids.add(int(row["cell_id"]))
    return ids


def print_digest(rows, idx, truth, label):
    """Median bias/sd/RMSE and |bias|/sd per fitted model over `rows` (in %).
    Per-cell pairing of bias & sd; arid garbage (|.| > CAP_PCT) is dropped."""
    print(f"\nparent={NAME[truth]}  [{label}: {len(rows)} cells]")
    print(f"{'fit':12s} {'median bias%':>13s} {'median sd%':>11s} "
          f"{'median RMSE%':>13s} {'median |bias|/sd':>17s}")
    for m in MODELS:
        if f"{m}_bias" not in idx:           # ad columns absent in pre-adaptive runs
            continue
        bi, si = idx[f"{m}_bias"], idx[f"{m}_sd"]
        bs, ss = [], []
        for r in rows:
            try:
                b, s = float(r[bi]) * 100.0, float(r[si]) * 100.0
            except (ValueError, IndexError):
                continue
            if np.isfinite(b) and np.isfinite(s) and abs(b) <= CAP_PCT and abs(s) <= CAP_PCT:
                bs.append(b); ss.append(s)
        if not bs:
            continue
        b, s = np.array(bs), np.array(ss)
        rmse = np.sqrt(b ** 2 + s ** 2)
        ratio = np.abs(b) / np.where(s > 0, s, np.nan)
        print(f"{NAME[m]:12s} {np.median(b):13.2f} {np.median(s):11.2f} "
              f"{np.median(rmse):13.2f} {np.nanmedian(ratio):17.2f}")


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in MODELS:
        sys.exit("Usage: python3 step07_combine_monte_carlo_parents.py <truth{st,ln,qd}> [dat_dir]")
    truth = sys.argv[1]
    dat_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(SCRIPT_DIR, "..", "data")
    in_dir = os.path.join(dat_dir, "045b", truth, "summary")
    out_csv = os.path.join(dat_dir, "045b", f"summary_allgrid_{truth}.csv")

    files = sorted(glob.glob(os.path.join(in_dir, "summary_*.csv")))
    if not files:
        sys.exit(f"No summary chunks in {in_dir} (run 045b_pbs.sh for TRUTH={truth} first)")

    header = None
    rows = []
    n_degenerate = 0
    for fp in files:
        with open(fp, newline="") as f:
            r = csv.reader(f)
            h = next(r, None)
            # All-fail chunks: 045b writes an empty data.frame -> a degenerate
            # file whose "header" is '' (no cell_id). Skipping these is essential:
            # otherwise the first such chunk becomes `header` and wipes out the
            # real column names (st_bias, ...), breaking both the output CSV and
            # the digest (KeyError: 'st_bias').
            if h is None or "cell_id" not in h:
                n_degenerate += 1
                continue
            if header is None:
                header = h
            elif h != header:
                continue   # unexpected schema; ignore to keep columns aligned
            for row in r:
                if row and len(row) == len(header):
                    rows.append(row)

    if header is None or not rows:
        sys.exit("No data rows found.")
    if n_degenerate:
        print(f"(skipped {n_degenerate} empty/all-fail chunks)")

    bad = excess_lib.load_bad_cells(dat_dir)
    if bad and "cell_id" in header:
        ci = header.index("cell_id")
        before = len(rows)
        rows = [r for r in rows if not (r[ci] not in ("", "NA") and int(r[ci]) in bad)]
        print(f"Excluding {before - len(rows)} negative-AMAX cells (031).")

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"Parent={truth}: combined {len(files)} chunks, {len(rows)} cells -> {out_csv}")

    # console digest: all cells, then the flood-relevant subset only
    idx = {c: i for i, c in enumerate(header)}
    print_digest(rows, idx, truth, "all cells")

    flood = load_flood_ids(dat_dir)
    if flood is None:
        print("\nNOTE: 091/change_type_cells.csv not found -> no flood-relevant "
              "digest (run 094_classify_pbs.sh + 095 first).")
    elif "cell_id" in idx:
        ci = idx["cell_id"]
        frows = [r for r in rows if r[ci] not in ("", "NA") and int(r[ci]) in flood]
        print_digest(frows, idx, truth, "flood-relevant")


if __name__ == "__main__":
    main()
