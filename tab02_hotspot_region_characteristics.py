"""
tab02_hotspot_region_characteristics.py   (run in `gam` env)
Do the five tier-2 improvement HOTSPOTS (506) fit the low-N_eff -> steep scale-slope
-> improvement theory, or do some run on a DIFFERENT pathway?

The theory predicts high improvement where N_eff is SMALL (arid/tropical, class
var/both, steep sig_slope).  But the five outlined regions span very different
climates (two African = arid/tropical; Quebec + West Siberia = cold/boreal, the
HIGH-N_eff, low-slope end).  This script tabulates, per region box, the quantities
that distinguish the two pathways:
  * median improvement
  * AIC-class composition  (mean=LOCATION trend vs var/both=SCALE trend)
  * median sig_slope, N_eff,var, aridity, dominant Koppen
So we can see which regions are SCALE-slope / low-N_eff driven (theory) and which are
LOCATION-trend driven (a second pathway the improvement inherits).

Inputs: 079/aic_allgrid.csv, 050/summary_allgrid.csv, 519/mechanism_vars.csv,
        517/cell_climate.csv
Output: data/524/region_diagnosis.csv  (+ console table)
Usage: python3 tab02_hotspot_region_characteristics.py [dat_dir] [--scope ensured|all]
"""
import os
import sys
import csv
import importlib
import numpy as np

excess_lib = importlib.import_module("common_target_cells")

RES = 0.1
# (name, lon0, lon1, lat0, lat1)
REGIONS = [
    ("Quebec-Labrador",           -78.0, -60.0,  47.0,  54.0),
    ("West Siberia / N Russia",    40.0,  95.0,  55.0,  66.0),
    ("North China Plain",         110.0, 120.0,  32.0,  41.0),
    ("West+Central Africa",       -12.0,  30.0,  -6.0,  14.0),
    ("Southern Africa (Kalahari)", 19.0,  29.0, -27.0, -18.0),
]
CLASSES = ["stationary", "mean", "var", "both"]
GROUPS = ["A", "B", "C", "D", "E"]


def sfloat(v):
    try:
        x = float(v)
        return x if np.isfinite(x) else np.nan
    except (TypeError, ValueError):
        return np.nan


def main():
    argv = sys.argv[1:]
    scope = "ensured"
    if "--scope" in argv:
        i = argv.index("--scope"); scope = argv[i + 1]; del argv[i:i + 2]
    dat_dir = argv[0] if argv and not argv[0].startswith("--") else \
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
    out_dir = os.path.join(dat_dir, "524"); os.makedirs(out_dir, exist_ok=True)

    keep = excess_lib.load_flood_ensured_ids(dat_dir) if scope == "ensured" else None
    keep = keep or None

    # 079: class, sig_slope, iy, ix
    rec = {}
    with open(os.path.join(dat_dir, "079", "aic_allgrid.csv"), newline="") as f:
        for row in csv.DictReader(f):
            try:
                cid = int(row["cell_id"]); iy = int(row["iy"]); ix = int(row["ix"])
            except (ValueError, KeyError):
                continue
            rec[cid] = [iy, ix, row.get("class", ""), sfloat(row.get("sig_slope")),
                        np.nan, np.nan, np.nan, ""]     # +imp, neff, aridity, koppen

    imp_map, cap = {}, excess_lib.CAP_PCT
    with open(os.path.join(dat_dir, "050", "summary_allgrid.csv"), newline="") as f:
        for row in csv.DictReader(f):
            try:
                cid = int(row["cell_id"])
            except (ValueError, KeyError):
                continue
            if cid in rec:
                v = sfloat(row.get("improve_lin_sd"))
                rec[cid][4] = v if (np.isfinite(v) and abs(v) <= cap) else np.nan

    with open(os.path.join(dat_dir, "519", "mechanism_vars.csv"), newline="") as f:
        for row in csv.DictReader(f):
            try:
                cid = int(row["cell_id"])
            except (ValueError, KeyError):
                continue
            if cid in rec:
                rec[cid][5] = sfloat(row.get("neff_var"))

    clim = {}
    with open(os.path.join(dat_dir, "517", "cell_climate.csv"), newline="") as f:
        for row in csv.DictReader(f):
            try:
                clim[(int(row["iy"]), int(row["ix"]))] = (row.get("koppen_group", ""),
                                                          sfloat(row.get("aridity_index")))
            except (ValueError, KeyError):
                pass
    for cid, r in rec.items():
        kg, ar = clim.get((r[0], r[1]), ("", np.nan))
        r[6] = ar; r[7] = kg

    # assemble arrays with lon/lat
    ids = [cid for cid in rec if (keep is None or cid in keep)]
    iy = np.array([rec[c][0] for c in ids]); ix = np.array([rec[c][1] for c in ids])
    lon = -180.0 + (ix + 0.5) * RES; lat = 90.0 - (iy + 0.5) * RES
    klass = np.array([rec[c][2] for c in ids])
    sig = np.array([rec[c][3] for c in ids]); imp = np.array([rec[c][4] for c in ids])
    neff = np.array([rec[c][5] for c in ids]); arid = np.array([rec[c][6] for c in ids])
    kg = np.array([rec[c][7] for c in ids])

    gmed_imp = np.nanmedian(imp)
    thr = float(np.nanpercentile(imp, 80.0))   # tier-2 top-quintile threshold (p80)
    top = np.isfinite(imp) & (imp >= thr)
    print(f"scope={scope}  cells={len(ids):,}  GLOBAL median improvement={gmed_imp:+.2f}%")
    print(f"tier-2 threshold: improvement p80 = {thr:.2f}%  "
          f"(top-20% cells = {int(top.sum()):,})")
    print(f"Restricting each region to its top-20% cells (improvement >= {thr:.2f}%), "
          f"so the MINIMUM is ~{thr:.1f}% by construction.\n")

    hdr = (f"{'region':28s} {'nTop':>6s} {'impMin':>7s} {'impMed':>7s} {'sigSlp':>8s} "
           f"{'Neff':>6s} {'arid':>5s} {'Kopp':>5s} | {'stat%':>6s} {'loc%':>6s} "
           f"{'scl%':>6s} {'both%':>6s}")
    print(hdr)
    rows = []
    for name, lo, hi, la0, la1 in REGIONS:
        box = (lon >= lo) & (lon < hi) & (lat >= la0) & (lat < la1)
        m = box & top                                  # top-20% cells only
        n = int(m.sum())
        if n == 0:
            print(f"{name:28s}  (no top-20% cells)"); continue
        km = klass[m]
        fr = {c: 100.0 * np.mean(km == c) for c in CLASSES}
        kk = kg[m]; kk = kk[kk != ""]
        dom = max(GROUPS, key=lambda g: np.sum(kk == g)) if kk.size else "?"
        imin = np.nanmin(imp[m]); im = np.nanmedian(imp[m]); ss = np.nanmedian(sig[m])
        nf = np.nanmedian(neff[m]); ar = np.nanmedian(arid[m])
        rows.append((name, n, imin, im, ss, nf, ar, dom, fr))
        print(f"{name:28s} {n:>6d} {imin:>+7.2f} {im:>+7.2f} {ss:>+8.4f} {nf:>6.1f} "
              f"{ar:>5.2f} {dom:>5s} | {fr['stationary']:>6.1f} {fr['mean']:>6.1f} "
              f"{fr['var']:>6.1f} {fr['both']:>6.1f}")

    with open(os.path.join(out_dir, "region_diagnosis.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["region", "n_top20", "min_improvement", "median_improvement",
                    "median_sig_slope", "median_neff_var", "median_aridity",
                    "dominant_koppen", "pct_stationary", "pct_mean_location",
                    "pct_var_scale", "pct_both"])
        for name, n, imin, im, ss, nf, ar, dom, fr in rows:
            w.writerow([name, n, f"{imin:.3f}", f"{im:.3f}", f"{ss:.5f}", f"{nf:.2f}",
                        f"{ar:.3f}", dom, f"{fr['stationary']:.2f}", f"{fr['mean']:.2f}",
                        f"{fr['var']:.2f}", f"{fr['both']:.2f}"])
    print(f"\nSaved: {os.path.join(out_dir, 'region_diagnosis.csv')}")
    print("\nPathway key: LOW Neff + high scl%/both%  => scale-slope/flashiness (theory)."
          "\n             HIGH Neff + high loc%(mean)  => LOCATION-trend pathway (different).")


if __name__ == "__main__":
    main()
