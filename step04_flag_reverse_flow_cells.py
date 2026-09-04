"""
step04_flag_reverse_flow_cells.py
Flag grid cells that are unusable for flood-frequency analysis because their
annual-maximum series contains negative discharge. These are CaMa-Flood
reverse / backwater-flow cells (river mouths, estuaries, very flat floodplains):
the negative values are genuine model output (outflw is bidirectional), but a
negative annual maximum means the cell had no positive flow that year, so it
carries no meaningful flood peak. About 0.16% of cells, scattered globally.

The combine/analysis scripts (079, 105, 054, 107, ...) drop these cells via
common_target_cells.load_bad_cells so every figure/table uses the same clean cell set,
without re-running the (expensive) Monte-Carlo stages.

Input:  <dat_dir>/030/amax_all.bin  (float32, n_cells x 120)
        <dat_dir>/010/land_cells.csv
Output: <dat_dir>/010/bad_cells.csv  (cell_id, iy, ix, n_neg_years, min_amax)

Usage: python step04_flag_reverse_flow_cells.py [dat_dir]
"""
import os
import sys
import csv
import numpy as np

N_YEARS = 120


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    dat_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "..", "data")
    amax = np.fromfile(os.path.join(dat_dir, "030", "amax_all.bin"),
                       dtype="float32").reshape(-1, N_YEARS)
    cells = np.loadtxt(os.path.join(dat_dir, "010", "land_cells.csv"),
                       delimiter=",", skiprows=1, dtype=int)
    neg = amax < 0
    bad = np.where(neg.any(axis=1))[0]
    out = os.path.join(dat_dir, "010", "bad_cells.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cell_id", "iy", "ix", "n_neg_years", "min_amax"])
        for cid in bad:
            w.writerow([int(cid), int(cells[cid, 1]), int(cells[cid, 2]),
                        int(neg[cid].sum()), f"{amax[cid].min():.4f}"])
    print(f"Flagged {len(bad)} / {amax.shape[0]} cells "
          f"({100.0 * len(bad) / amax.shape[0]:.3f}%) with negative AMAX -> {out}")


if __name__ == "__main__":
    main()
