"""
step20_combine_aic_adaptive.py
Concatenate the per-chunk AIC-adaptive summaries written by 108 for one parent
into a single all-grid table, and print a median digest (bias/SD/RMSE of the
adaptive estimator). The output is what 107 reads to add the 'ad' fitted model.

Input:  <dat_dir>/108/<truth>/summary/ad_*.csv
Output: <dat_dir>/108/summary_allgrid_ad_<truth>.csv
          cell_id, iy, ix, truth_q100, ad_bias, ad_sd, ad_iqr, ad_tail

Usage:
  python3 step20_combine_aic_adaptive.py <truth{st,ln,qd}> [dat_dir]
"""
import os
import sys
import glob
import csv
import numpy as np
import importlib
excess_lib = importlib.import_module("common_target_cells")

CAP_PCT = 300.0
NAME = {"st": "stationary", "ln": "linear", "qd": "quadratic"}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in NAME:
        sys.exit("Usage: python3 step20_combine_aic_adaptive.py <truth{st,ln,qd}> [dat_dir]")
    truth = sys.argv[1]
    dat_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "data")
    in_dir = os.path.join(dat_dir, "108", truth, "summary")
    out_csv = os.path.join(dat_dir, "108", f"summary_allgrid_ad_{truth}.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    files = sorted(glob.glob(os.path.join(in_dir, "ad_*.csv")))
    if not files:
        sys.exit(f"No ad chunks in {in_dir} (run 108_pbs.sh for TRUTH={truth} first)")

    header, rows, n_degenerate = None, [], 0
    for fp in files:
        with open(fp, newline="") as f:
            r = csv.reader(f)
            h = next(r, None)
            if h is None or "cell_id" not in h:   # all-fail chunk -> empty header
                n_degenerate += 1
                continue
            if header is None:
                header = h
            elif h != header:
                continue
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

    idx = {c: i for i, c in enumerate(header)}

    def col_pct(name):
        """Finite, garbage-trimmed values of a column, in %."""
        if name not in idx:
            return None
        j = idx[name]
        out = []
        for r in rows:
            try:
                v = float(r[j]) * 100.0
            except (ValueError, IndexError):
                continue
            if np.isfinite(v) and abs(v) <= CAP_PCT:
                out.append(v)
        return np.array(out)

    print(f"\nparent={NAME[truth]}  (median bias/sd/RMSE %, same realizations)")
    print(f"{'fit':12s} {'bias%':>8s} {'sd%':>8s} {'rmse%':>8s}")
    for fit, lab in (("ad", "adaptive"), ("ln", "linear"), ("qd", "quadratic")):
        b, s = col_pct(f"{fit}_bias"), col_pct(f"{fit}_sd")
        if b is None or s is None or b.size == 0:
            continue
        n = min(len(b), len(s))
        rmse = np.sqrt(b[:n] ** 2 + s[:n] ** 2)
        print(f"{lab:12s} {np.median(b):8.2f} {np.median(s):8.2f} {np.median(rmse):8.2f}")

    # QD-selection frequency across cells (how often curvature is detected)
    for fcol, rule in (("ad_qd_frac", "ΔAIC=2"), ("ad_qd_frac_d0", "ΔAIC=0")):
        qf = col_pct(fcol)
        if qf is not None and qf.size:
            print(f"  QD-selection frequency ({rule}): median = {np.median(qf):.1f}%, "
                  f"mean = {np.mean(qf):.1f}%, QD>50%-of-runs in "
                  f"{100*np.mean(qf > 50):.1f}% of cells")


if __name__ == "__main__":
    main()
