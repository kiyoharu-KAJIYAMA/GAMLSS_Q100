"""
step23_extract_climate_covariates.py   (run in the `geo` conda env: rasterio + numpy)
Extract per-cell TRUE-CLIMATE covariates onto the study's 0.1 deg grid (NY=1800,
NX=3600, row 0 = +90 N, col 0 = -180 W), from the datasets downloaded to
data/climate/:

  koppen_geiger_0p1.tif   Beck et al. 2023 Koppen-Geiger, 1991-2020, 0.1 deg
                          (1800x3600 uint8, EXACT grid match; 0=ocean, 1..30=class)
  ai_v31_yr.tif           CGIAR-CSI Global Aridity Index v3.1 (Zomer 2022),
                          30 arc-sec uint16, value = (P/PET) x 1e4.  Aggregated to
                          0.1 deg by block-mean over valid pixels (12x12 blocks;
                          its grid spans lat +90..-60, so rows for lat<-60 are NaN).

Koppen 30-class -> 5 main groups: A(1-3) B(4-7) C(8-16) D(17-28) E(29-30).

Output: data/517/cell_climate.csv   columns:
  iy, ix, koppen_code(1..30), koppen_group(A/B/C/D/E), aridity_index(P/PET)
only for land cells (koppen_code>0). Keyed by (iy,ix) so the analysis merges on the
iy/ix already present in 070/079 (no cell_id assumption).

Usage:  conda activate geo && python3 step23_extract_climate_covariates.py [dat_dir]
"""
import os
import sys
import csv
import numpy as np
import rasterio
from rasterio.windows import Window

NY, NX, RES = 1800, 3600, 0.1

# Beck 30-class code -> main group letter
GROUP = {}
for c in range(1, 31):
    GROUP[c] = ("A" if c <= 3 else "B" if c <= 7 else "C" if c <= 16
                else "D" if c <= 28 else "E")


def aggregate_ai(ai_path):
    """CGIAR AI 30arcsec -> 0.1deg block-mean of valid (P/PET) values.
    Returns float array (NY, NX) with NaN where no valid source pixels."""
    out = np.full((NY, NX), np.nan, dtype="float64")
    with rasterio.open(ai_path) as ds:
        H, W = ds.height, ds.width           # 18000 x 43200
        fy = H // (NY)                        # source rows per 0.1 row candidate
        # source spans lat +90..-60 => 15000? actually H/120=150deg -> covers 1500 rows
        rows_cov = H // 12                    # number of full 0.1-deg rows covered
        assert W == NX * 12, f"unexpected AI width {W}"
        scale = 1.0e4
        for r in range(rows_cov):
            win = Window(0, r * 12, W, 12)
            blk = ds.read(1, window=win).astype("float64")   # (12, 43200)
            valid = (blk > 0) & (blk < 65535)
            blk[~valid] = np.nan
            # reshape to (12, NX, 12) and nanmean over the two 12-axes
            blk = blk.reshape(12, NX, 12)
            with np.errstate(invalid="ignore"):
                m = np.nanmean(blk, axis=(0, 2))              # (NX,)
            out[r, :] = m / scale
    return out


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    dat_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "..", "data")
    cdir = os.path.join(dat_dir, "climate")
    kp_path = os.path.join(cdir, "koppen_geiger_0p1.tif")
    ai_path = os.path.join(cdir, "ai_v31_yr.tif")
    out_dir = os.path.join(dat_dir, "517"); os.makedirs(out_dir, exist_ok=True)
    for p in (kp_path, ai_path):
        if not os.path.exists(p):
            sys.exit(f"Not found: {p}")

    with rasterio.open(kp_path) as ds:
        assert (ds.height, ds.width) == (NY, NX), f"koppen shape {ds.height}x{ds.width}"
        kop = ds.read(1)                     # uint8 (1800,3600), 0=ocean
    print(f"Koppen loaded: {kop.shape}; land cells = {int((kop>0).sum()):,}")

    ai = aggregate_ai(ai_path)
    print(f"AI aggregated to 0.1deg; valid cells = {int(np.isfinite(ai).sum()):,}")

    out_csv = os.path.join(out_dir, "cell_climate.csv")
    n = 0
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["iy", "ix", "koppen_code", "koppen_group", "aridity_index"])
        ys, xs = np.nonzero(kop > 0)
        for iy, ix in zip(ys.tolist(), xs.tolist()):
            code = int(kop[iy, ix])
            a = ai[iy, ix]
            w.writerow([iy, ix, code, GROUP[code],
                        "" if not np.isfinite(a) else f"{a:.4f}"])
            n += 1
    print(f"Saved: {out_csv}  ({n:,} land cells)")

    # quick group tally
    codes = kop[kop > 0]
    grp = np.array([GROUP[int(c)] for c in codes])
    print("Koppen main-group land-cell counts:")
    for g in "ABCDE":
        print(f"  {g}: {int((grp==g).sum()):,}")


if __name__ == "__main__":
    main()
