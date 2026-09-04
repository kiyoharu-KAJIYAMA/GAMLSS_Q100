"""
step02_annual_maxima_per_grid.py
Extract annual maximum (AMAX) outflow for all grid cells using vectorized
numpy operations (np.max over time axis).

Usage: python step02_annual_maxima_per_grid.py <year>

Output: ../data/020/
  - amax_grid_{year}.bin  (1800 x 3600, float32)

Memory: ~9.5 GB per year (365 days x 1800 x 3600 x 4 bytes)
"""
import os
import sys
import numpy as np

NY, NX = 1800, 3600

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DAT_DIR = os.path.join(SCRIPT_DIR, "..", "data", "020")


def get_outflow_path(year):
    if year < 2015:
        return f"/work/a07/ykimura/2022_LaRC/out/C6-g06M_ACC_hist/outflw{year}.bin"
    elif year < 2021:
        return f"/work/a07/ykimura/2022_LaRC/out/C6-g06M_ACC_ssp245_y2015-2022/outflw{year}.bin"
    else:
        return f"/work/a07/ykimura/2022_LaRC/out/C6-g06M_ACC_ssp245/outflw{year}.bin"


def main(year):
    os.makedirs(DAT_DIR, exist_ok=True)

    path = get_outflow_path(year)
    print(f"Loading {path} ...")

    # Read full year and compute max along time axis (vectorized)
    data = np.fromfile(path, "float32").reshape(-1, NY, NX)  # (ndays, 1800, 3600)
    amax_grid = np.max(data, axis=0)  # (1800, 3600)
    del data  # Free memory

    outpath = os.path.join(DAT_DIR, f"amax_grid_{year}.bin")
    amax_grid.astype("float32").tofile(outpath)
    print(f"Year {year}: saved {outpath}")


if __name__ == "__main__":
    year = int(sys.argv[1])
    main(year)
