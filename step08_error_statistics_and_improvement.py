"""
step08_error_statistics_and_improvement.py
Combine 040/mc_summary/*.csv (relative error statistics) into a single CSV.
Also computes improvement rates [%] of nonstationary over stationary; cells whose
|improvement| > CAP_PCT (arid blow-up) are DROPPED (-> nan), not clamped (as 052).

Relative error definition (computed in 040):
  rel_error = (estimated_Q100 - truth_Q100) / truth_Q100
  Scale-independent, perfect prediction = 0.

Input:
  ../data/040/mc_summary/summary_*.csv

Output: ../data/050/
  - summary_allgrid.csv
"""
import os
import glob
import csv
import math
import importlib
excess_lib = importlib.import_module("common_target_cells")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DAT_ROOT = os.path.join(SCRIPT_DIR, "..", "data")
SUMMARY_DIR = os.path.join(DAT_ROOT, "040", "mc_summary")
OUTPUT_DIR = os.path.join(DAT_ROOT, "050")

# Cells whose |improvement| exceeds this (%) are non-physical arid artifacts
# (stationary SD ~ 0, truth_q100 ~ 0) and are DROPPED (-> nan), not clamped to a
# boundary -- matching 052's CAP_PCT so all downstream views stay consistent.
CAP_PCT = 300.0


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    bad = excess_lib.load_bad_cells(os.path.dirname(OUTPUT_DIR))

    summary_files = sorted(glob.glob(os.path.join(SUMMARY_DIR, "summary_*.csv")))
    print(f"Summary chunks: {len(summary_files)}")

    output_csv = os.path.join(OUTPUT_DIR, "summary_allgrid.csv")
    total_cells = 0

    with open(output_csv, "w", newline="") as fout:
        writer = csv.writer(fout)
        writer.writerow([
            "cell_id", "iy", "ix",
            "truth_q100",
            "stat_err_sd", "stat_err_iqr", "stat_err_tail",
            "lin_err_sd", "lin_err_iqr", "lin_err_tail",
            "quad_err_sd", "quad_err_iqr", "quad_err_tail",
            "improve_lin_sd", "improve_lin_iqr", "improve_lin_tail",
            "improve_quad_sd", "improve_quad_iqr", "improve_quad_tail",
        ])

        for sf in summary_files:
            with open(sf, "r", newline="") as fin:
                reader = csv.DictReader(fin)
                for row in reader:
                    tq = row.get("truth_q100", "")
                    if tq == "" or tq == "NA" or tq == "nan":
                        continue
                    try:
                        if int(row["cell_id"]) in bad:
                            continue
                    except (KeyError, ValueError):
                        pass

                    try:
                        st_sd   = float(row["st_err_sd"])
                        st_iqr  = float(row["st_err_iqr"])
                        st_tail = float(row["st_err_tail"])
                        ln_sd   = float(row["ln_err_sd"])
                        ln_iqr  = float(row["ln_err_iqr"])
                        ln_tail = float(row["ln_err_tail"])
                        qd_sd   = float(row["qd_err_sd"])
                        qd_iqr  = float(row["qd_err_iqr"])
                        qd_tail = float(row["qd_err_tail"])
                    except (ValueError, KeyError):
                        continue

                    # Improvement [%]: (stat - nonstat) / stat * 100.
                    # A near-zero stationary SD (arid cells) blows the ratio up;
                    # rather than clamp such values to a boundary (which piles a
                    # spurious mass at +/-CAP), DROP them (-> nan) when non-finite
                    # or |.| > CAP_PCT, exactly as 052 does.
                    def improve(stat_val, ns_val):
                        if abs(stat_val) < 1e-9:
                            return "nan"
                        val = (stat_val - ns_val) / stat_val * 100.0
                        if not math.isfinite(val) or abs(val) > CAP_PCT:
                            return "nan"
                        return f"{val:.4f}"

                    writer.writerow([
                        row["cell_id"], row["iy"], row["ix"],
                        tq,
                        f"{st_sd:.6f}", f"{st_iqr:.6f}", f"{st_tail:.6f}",
                        f"{ln_sd:.6f}", f"{ln_iqr:.6f}", f"{ln_tail:.6f}",
                        f"{qd_sd:.6f}", f"{qd_iqr:.6f}", f"{qd_tail:.6f}",
                        improve(st_sd, ln_sd), improve(st_iqr, ln_iqr), improve(st_tail, ln_tail),
                        improve(st_sd, qd_sd), improve(st_iqr, qd_iqr), improve(st_tail, qd_tail),
                    ])
                    total_cells += 1

            print(f"Processed {os.path.basename(sf)} -> {total_cells} cells so far")

    print(f"Summary saved: {output_csv} ({total_cells} cells)")


if __name__ == "__main__":
    main()
