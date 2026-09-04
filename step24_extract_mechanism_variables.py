"""
step24_extract_mechanism_variables.py   (run in `gam` env: numpy)
Extract the two MECHANISM variables that decompose mean_amax, so the paper can name
the cause of "small river -> large slope-of-scale" instead of the symptom.

  Mechanism 1  SPATIAL DILUTION      -> upstream cell count / Shreve source count,
               (aggregation)            from data/nextxy.bin flow network. Pure count
                                        of independent AMAX series summed at a cell;
                                        decoupled from aridity/flow magnitude.

  Mechanism 2  TEMPORAL EFFECTIVE     -> N_eff,var = (sum d^2)^2 / sum d^4, d = x-mean,
               SAMPLE SIZE              the effective number of years the scale (sigma)
                                        estimate rests on. Low when a few giant floods
                                        dominate (arid, stochastic) -> slope-of-scale
                                        emerges easily. From data/030/amax_all.bin.

Also: N_eff,part = (sum x)^2 / sum x^2 (participation ratio for the mean), n_pos
(number of non-zero flood years), n_valid.

amax_all.bin row order == land_cells.csv row order (cell_id = row index).
nextxy.bin = (2, NY, NX) int32: nextx=b[0] (1-based, -9 mouth, -10 inland, -9999 sea).

Output: data/519/mechanism_vars.csv
  cell_id, iy, ix, neff_var, neff_part, n_pos, n_valid, n_up, n_src
Usage: python3 step24_extract_mechanism_variables.py [dat_dir]
"""
import os
import sys
import csv
import numpy as np
from collections import deque

NY, NX = 1800, 3600
N_YEARS = 120


def load_land_cells(path):
    cid, iy, ix = [], [], []
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            cid.append(int(row["cell_id"])); iy.append(int(row["iy"])); ix.append(int(row["ix"]))
    return np.array(cid), np.array(iy), np.array(ix)


def neff_from_amax(amax):
    """amax: (Ncell, N_YEARS) float. Returns dict of per-cell effective-sample metrics."""
    x = amax.astype("float64")
    finite = np.isfinite(x)
    n_valid = finite.sum(axis=1)
    xz = np.where(finite, x, 0.0)
    cnt = np.where(n_valid > 0, n_valid, 1)
    mean = xz.sum(axis=1) / cnt
    d = np.where(finite, x - mean[:, None], 0.0)
    s2 = (d ** 2).sum(axis=1)
    s4 = (d ** 4).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        neff_var = np.where(s4 > 0, s2 ** 2 / s4, np.nan)
        sx = xz.sum(axis=1); sx2 = (xz ** 2).sum(axis=1)
        neff_part = np.where(sx2 > 0, sx ** 2 / sx2, np.nan)
    n_pos = (xz > 0).sum(axis=1)
    return neff_var, neff_part, n_pos, n_valid


def flow_accumulation(dat_dir, iy, ix):
    """upstream cell count (incl self) and Shreve source count for each land cell,
    via Kahn topological accumulation over data/nextxy.bin."""
    b = np.fromfile(os.path.join(dat_dir, "nextxy.bin"), dtype="<i4").reshape(2, NY, NX)
    nx_g, ny_g = b[0], b[1]
    ncell = iy.size
    lin = iy * NX + ix                                   # grid linear index of each land cell
    lin2k = np.full(NY * NX, -1, dtype="int64")
    lin2k[lin] = np.arange(ncell)
    # downstream grid coords for each land cell
    nxv = nx_g[iy, ix]; nyv = ny_g[iy, ix]
    has_ds = nxv > 0
    ds_lin = np.where(has_ds, (nyv - 1) * NX + (nxv - 1), -1)
    ds_k = np.where(ds_lin >= 0, lin2k[np.clip(ds_lin, 0, NY * NX - 1)], -1)
    ds_k = np.where(has_ds, ds_k, -1).astype("int64")    # -1 = terminal / off-land

    indeg = np.zeros(ncell, dtype="int64")
    valid_ds = ds_k[ds_k >= 0]
    np.add.at(indeg, valid_ds, 1)
    src = (indeg == 0)                                   # headwater sources (Shreve seed)
    acc = np.ones(ncell, dtype="float64")                # upstream cells incl self
    accs = src.astype("float64")                         # Shreve = upstream source count
    q = deque(np.nonzero(indeg == 0)[0].tolist())
    indeg_work = indeg.copy()
    processed = 0
    while q:
        k = q.popleft(); processed += 1
        d = ds_k[k]
        if d >= 0:
            acc[d] += acc[k]; accs[d] += accs[k]
            indeg_work[d] -= 1
            if indeg_work[d] == 0:
                q.append(d)
    print(f"  flow accumulation: processed {processed:,}/{ncell:,} cells "
          f"({'acyclic OK' if processed == ncell else 'WARNING cycles remain'})")
    return acc, accs


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    dat_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "..", "data")
    lc_path = os.path.join(dat_dir, "010", "land_cells.csv")
    amax_path = os.path.join(dat_dir, "030", "amax_all.bin")
    out_dir = os.path.join(dat_dir, "519"); os.makedirs(out_dir, exist_ok=True)
    for p in (lc_path, amax_path, os.path.join(dat_dir, "nextxy.bin")):
        if not os.path.exists(p):
            sys.exit(f"Not found: {p}")

    cid, iy, ix = load_land_cells(lc_path)
    ncell = cid.size
    print(f"land cells: {ncell:,}")

    amax = np.fromfile(amax_path, dtype="<f4")
    assert amax.size == ncell * N_YEARS, f"amax size {amax.size} != {ncell}*{N_YEARS}"
    amax = amax.reshape(ncell, N_YEARS)
    # AMAX uses 0/negative for dry-or-missing in some builds; treat <=0 as no-flood but
    # keep for moments only if finite. Nonfinite -> missing.
    neff_var, neff_part, n_pos, n_valid = neff_from_amax(amax)
    print(f"N_eff,var  : median={np.nanmedian(neff_var):.1f}  "
          f"(effective #floods for the scale; low=arid/stochastic)")
    print(f"N_eff,part : median={np.nanmedian(neff_part):.1f}")

    n_up, n_src = flow_accumulation(dat_dir, iy, ix)
    print(f"upstream cells: median={np.median(n_up):.0f}  max={n_up.max():.0f}")
    print(f"Shreve sources: median={np.median(n_src):.0f}  max={n_src.max():.0f}")

    out_csv = os.path.join(out_dir, "mechanism_vars.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cell_id", "iy", "ix", "neff_var", "neff_part", "n_pos",
                    "n_valid", "n_up", "n_src"])
        for k in range(ncell):
            nv = neff_var[k]; npt = neff_part[k]
            w.writerow([cid[k], iy[k], ix[k],
                        "" if not np.isfinite(nv) else f"{nv:.3f}",
                        "" if not np.isfinite(npt) else f"{npt:.3f}",
                        int(n_pos[k]), int(n_valid[k]),
                        int(n_up[k]), int(n_src[k])])
    print(f"Saved: {out_csv}  ({ncell:,} rows)")


if __name__ == "__main__":
    main()
