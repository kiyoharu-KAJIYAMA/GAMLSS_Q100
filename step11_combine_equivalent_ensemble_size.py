"""
step11_combine_equivalent_ensemble_size.py
Combine all 060/<parent>/<ns_model>/eq_ens/*.csv chunks into a single CSV.

The data-generating parent and the nonstationary estimator are both selectable
(matches 060/062):
  parent   in {st, ln, qd}   (default ln)
  ns_model in {lin, qd, ad}  (default lin)
  reads  060/<parent>/<ns_model>/eq_ens
  writes 063/eq_ens_allgrid_<parent>_<ns_model>.csv

Usage:
  python3 step11_combine_equivalent_ensemble_size.py [dat_dir] [parent: st|ln|qd] [ns_model: lin|qd|ad]
"""
import os
import sys
import glob
import csv


def main():
    dat_dir = sys.argv[1] if len(sys.argv) > 1 else \
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
    parent = sys.argv[2] if len(sys.argv) > 2 else "ln"
    ns_model = sys.argv[3] if len(sys.argv) > 3 else "lin"
    if parent not in ("st", "ln", "qd"):
        sys.exit("parent must be one of st, ln, qd")
    if ns_model not in ("lin", "qd", "ad"):
        sys.exit("ns_model must be one of lin, qd, ad")

    eq_dir = os.path.join(dat_dir, "060", parent, ns_model, "eq_ens")
    out_dir = os.path.join(dat_dir, "063")
    out_csv = os.path.join(out_dir, f"eq_ens_allgrid_{parent}_{ns_model}.csv")

    os.makedirs(out_dir, exist_ok=True)

    chunk_files = sorted(glob.glob(os.path.join(eq_dir, "eq_ens_*.csv")))
    print(f"Chunks found: {len(chunk_files)}")

    total_cells = 0

    with open(out_csv, "w", newline="") as fout:
        writer = None

        for cf in chunk_files:
            with open(cf, "r", newline="") as fin:
                reader = csv.DictReader(fin)

                if writer is None:
                    fieldnames = reader.fieldnames
                    writer = csv.DictWriter(fout, fieldnames=fieldnames)
                    writer.writeheader()

                for row in reader:
                    if row.get("status", "") == "SKIP":
                        continue
                    writer.writerow(row)
                    total_cells += 1

            print(f"Processed {os.path.basename(cf)} -> {total_cells} cells so far")

    print(f"Saved: {out_csv} ({total_cells} cells)")


if __name__ == "__main__":
    main()
