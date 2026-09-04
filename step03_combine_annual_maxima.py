"""
step03_combine_annual_maxima.py
Combine yearly AMAX grids (1800x3600) into a single binary file,
extracting only land cells defined in land_cells.csv.

Input:
  ../data/010/land_cells.csv
  ../data/020/amax_grid_{year}.bin

Output: ../data/030/
  - amax_all.bin  : (n_cells x 120) float32, ~960 MB for 2M cells
  - years.bin     : (120,) int32
"""
import os
import csv
import numpy as np

NY, NX = 1800, 3600
YEAR_START = 1981
YEAR_END = 2100
N_YEARS = YEAR_END - YEAR_START + 1

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DAT_ROOT = os.path.join(SCRIPT_DIR, "..", "data")
CELL_CSV = os.path.join(DAT_ROOT, "010", "land_cells.csv")
YEARLY_DIR = os.path.join(DAT_ROOT, "020")
OUTPUT_DIR = os.path.join(DAT_ROOT, "030")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load land cell coordinates
    iy_list, ix_list = [], []
    with open(CELL_CSV, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            iy_list.append(int(row["iy"]))
            ix_list.append(int(row["ix"]))

    iy_arr = np.array(iy_list, dtype=np.int32)
    ix_arr = np.array(ix_list, dtype=np.int32)
    n_cells = len(iy_arr)
    print(f"Land cells: {n_cells}")

    # Pre-allocate result array (n_cells x 120)
    amax_all = np.zeros((n_cells, N_YEARS), dtype="float32")

    # Load each yearly grid and extract land cell values via vector indexing
    for yi, year in enumerate(range(YEAR_START, YEAR_END + 1)):
        fpath = os.path.join(YEARLY_DIR, f"amax_grid_{year}.bin")
        grid = np.fromfile(fpath, "float32").reshape(NY, NX)
        amax_all[:, yi] = grid[iy_arr, ix_arr]
        if (yi + 1) % 10 == 0:
            print(f"  Loaded {yi + 1}/{N_YEARS} years")

    # Save combined binary
    output_bin = os.path.join(OUTPUT_DIR, "amax_all.bin")
    amax_all.tofile(output_bin)
    size_mb = os.path.getsize(output_bin) / 1e6
    print(f"Saved: {output_bin}  ({n_cells} cells x {N_YEARS} years = {size_mb:.0f} MB)")

    # Save year index for reference
    years = np.arange(YEAR_START, YEAR_END + 1, dtype="int32")
    years.tofile(os.path.join(OUTPUT_DIR, "years.bin"))


if __name__ == "__main__":
    main()
