"""
step01_extract_land_cells.py
Extract all land grid cell coordinates (uparea > 0) from the global grid.

Output: ../data/010/
  - land_cells.csv  (cell_id, iy, ix)
"""
import os
import csv
import numpy as np

# === Path settings (modify for your environment) ===
DATADIR = "/home/kk/jp_claude/gamlss/data"
UPAREA_PATH = f"{DATADIR}/uparea.bin"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DAT_DIR = os.path.join(SCRIPT_DIR, "..", "data", "010")

NY, NX = 1800, 3600


def main():
    os.makedirs(DAT_DIR, exist_ok=True)

    uparea = np.fromfile(UPAREA_PATH, "float32").reshape(NY, NX)

    # Extract all cells with positive upstream area
    iy_arr, ix_arr = np.where(uparea > 0)
    n_cells = len(iy_arr)
    print(f"Land cells found: {n_cells}")

    savefile = os.path.join(DAT_DIR, "land_cells.csv")
    with open(savefile, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["cell_id", "iy", "ix"])
        for cell_id, (iy, ix) in enumerate(zip(iy_arr, ix_arr)):
            writer.writerow([cell_id, int(iy), int(ix)])

    print(f"Saved: {savefile}")


if __name__ == "__main__":
    main()
