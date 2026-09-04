"""
figS3_aic_preferred_model_map.py
Combine the per-grid AIC of the three GAMLSS models (from 100) and visualise
which model the data prefer at each land cell.

Selection rule (forward, parsimony-aware): start from the stationary model and
adopt a more complex model only if it lowers AIC by MORE than `delta_aic`
(default 0 = plain lowest-AIC). With delta_aic = 2 (the conventional
"substantial support" threshold) the extra parameters of the quadratic model
must clearly pay for themselves, which is exactly the parsimony argument for
recommending the LINEAR model when the true parent is unknown: the quadratic
model has low bias but overfits, so AIC rarely prefers it.

Two views are produced: ALL grid cells, and the FLOOD-RELEVANT subset only
(suffix _flood). Flood relevance matches 098:
  keep = (q90 >= X1 OR max_amax >= X2) AND uparea_km2 >= A   (defaults 50/100/50)

Outputs (to <dat_dir>/103/, each also as *_flood for the flood-relevant subset):
  aic_best_model_map[_flood].png            Robinson map, cells coloured by the
                                    AIC-preferred model (Antarctica cropped;
                                    dashed river-basin outlines)
  aic_best_model_pie[_flood].png            pie chart of the grid-cell fraction
  aic_best_model_by_changetype[_flood].png  stacked bars of the per-model
                                    fraction by change type x climate (needs 091;
                                    localises where the quadratic model is
                                    actually preferred -> discussion support)
  aic_model_fractions.csv           machine-readable st/ln/qd fractions for the
                                    all-grid and flood-ensured subsets at
                                    delta_aic = 0 and 2 (subset,delta_aic,model,
                                    n,n_subset,frac)
  console: counts/percent per model for delta_aic = 0 and = 2

Input:
  <dat_dir>/100/summary/aic_*.csv     (cell_id, iy, ix, aic_st, aic_ln, aic_qd)
  <dat_dir>/091/change_type_cells.csv (type, climate, q90, max_amax, uparea_km2;
                                       optional -> enables bars + _flood view)

Usage:
  python3 figS3_aic_preferred_model_map.py [dat_dir] [delta_aic] [X1] [X2] [A]
"""
import os
import sys
import csv
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import importlib
excess_lib = importlib.import_module("common_target_cells")

RES = 0.1
MODELS = ["st", "ln", "qd"]
NAME = {"st": "Stationary", "ln": "Linear", "qd": "Quadratic"}
# Linear is the recommended model -> green; quadratic (overfits) -> orange.
# colours matched to the manuscript Figs. 2-4 palette:
# stationary = pink, linear = green, quadratic = purple
COLOR = {"st": "#e8638c", "ln": "#31a354", "qd": "#756bb1"}
BASIN_MIN_AREA = 50000.0
# Change-type x climate strata (from 091/change_type_cells.csv, added by 092/095)
TYPES = ["stationary", "trend", "step", "variance"]
CLIMATES = ["low", "high"]
# The stratified bar chart always uses the parsimony threshold (substantial
# support, dAIC > 2), independent of the map/pie delta_aic, because the
# change-type x climate breakdown is the discussion centrepiece and dAIC=2 is
# the defensible criterion (raw AIC over-selects the quadratic model).
BAR_DELTA_AIC = 2.0


def load_basin_feature(basin_dir, min_area_km2=BASIN_MIN_AREA):
    """Major river-basin outlines (HydroBASINS lev03) as a dashed feature."""
    shps = sorted(glob.glob(os.path.join(basin_dir, "hybas_*lev*.shp")))
    if not shps:
        print(f"NOTE: no basin shapefiles in {basin_dir} -> coastlines only")
        return None
    import cartopy.io.shapereader as shpreader
    from cartopy.feature import ShapelyFeature
    geoms = []
    for s in shps:
        for rec in shpreader.Reader(s).records():
            area = rec.attributes.get("SUB_AREA")
            if area is not None and area < min_area_km2:
                continue
            geoms.append(rec.geometry)
    print(f"Basin outlines: {len(geoms)} polygons")
    return ShapelyFeature(geoms, ccrs.PlateCarree(), edgecolor="#404040",
                          facecolor="none", linewidth=0.45, linestyle="--")


def load_aic(summary_dir, bad=frozenset()):
    cid, iy, ix, a_st, a_ln, a_qd = [], [], [], [], [], []
    files = sorted(glob.glob(os.path.join(summary_dir, "aic_*.csv")))
    if not files:
        sys.exit(f"No aic_*.csv in {summary_dir} (run 100/102 first)")
    print(f"Chunks: {len(files)}")

    def g(row, k):
        v = row.get(k, "")
        return float(v) if v not in ("", "NA", "nan") else np.nan

    for fp in files:
        with open(fp, "r", newline="") as f:
            for row in csv.DictReader(f):
                c = int(row["cell_id"])
                if c in bad:                  # 031 negative-AMAX reverse-flow cells
                    continue
                cid.append(c)
                iy.append(int(row["iy"]))
                ix.append(int(row["ix"]))
                a_st.append(g(row, "aic_st"))
                a_ln.append(g(row, "aic_ln"))
                a_qd.append(g(row, "aic_qd"))
    cid = np.array(cid); iy = np.array(iy); ix = np.array(ix)
    lat = 90.0 - (iy + 0.5) * RES
    lon = -180.0 + (ix + 0.5) * RES
    aic = np.vstack([a_st, a_ln, a_qd]).T   # (n, 3): st, ln, qd
    return cid, lon, lat, aic


def load_class_map(cells_csv, key):
    """cell_id -> classification string (e.g. type or climate), as in 098."""
    out = {}
    with open(cells_csv, "r", newline="") as f:
        for row in csv.DictReader(f):
            out[int(row["cell_id"])] = row.get(key, "")
    return out


def load_num_map(cells_csv, key):
    """cell_id -> float field (q90 / max_amax / uparea_km2). Missing -> NaN."""
    out = {}
    with open(cells_csv, "r", newline="") as f:
        for row in csv.DictReader(f):
            v = row.get(key, "")
            out[int(row["cell_id"])] = float(v) if v not in ("", "NA", "nan") else np.nan
    return out


def select_model(aic, delta_aic):
    """Forward parsimony selection -> integer code 0/1/2 (st/ln/qd) per cell.
    Adopt a more complex model only if it lowers AIC by more than delta_aic.
    NaN AICs are treated as +inf (a failed fit can never win)."""
    a = np.where(np.isfinite(aic), aic, np.inf)
    a_st, a_ln, a_qd = a[:, 0], a[:, 1], a[:, 2]
    code = np.zeros(len(a), dtype=int)         # 0 = stationary baseline
    best = a_st.copy()
    up = a_ln < best - delta_aic
    code[up] = 1; best[up] = a_ln[up]
    up = a_qd < best - delta_aic
    code[up] = 2; best[up] = a_qd[up]
    return code


def counts_table(code):
    n = len(code)
    return {m: int(np.sum(code == i)) for i, m in enumerate(MODELS)}, n


def write_fractions_csv(aic, flood_mask, filename):
    """Machine-readable st/ln/qd diagnosis fractions for both the all-grid and the
    flood-ensured subset, at the two parsimony thresholds (dAIC = 0 and 2). The
    flood_mask is already flood-ensured because load_aic dropped reverse-flow /
    no-flood-regime cells via load_excluded_cells. One row per
    subset x delta_aic x model, with count and fraction within that subset."""
    subsets = [("all", np.ones(len(aic), dtype=bool))]
    if flood_mask is not None:
        subsets.append(("flood_ensured", flood_mask))
    with open(filename, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subset", "delta_aic", "model", "n", "n_subset", "frac"])
        for sub_name, mask in subsets:
            n_sub = int(np.sum(mask))
            for daic in (0.0, 2.0):
                code = select_model(aic, daic)
                csub = code[mask]
                for i, m in enumerate(MODELS):
                    cnt = int(np.sum(csub == i))
                    frac = cnt / n_sub if n_sub else 0.0
                    w.writerow([sub_name, f"{daic:g}", m, cnt, n_sub, f"{frac:.4f}"])
    print(f"Saved: {filename}")


def plot_map(lon, lat, code, basins, filename):
    cmap = ListedColormap([COLOR[m] for m in MODELS])
    norm = BoundaryNorm(np.arange(-0.5, len(MODELS) + 0.5), cmap.N)
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
    # keep Antarctica cropped in the south but include the full Arctic to 90N.
    # set_extent() mis-computes the northern bound under Robinson, so set the
    # limits directly in PROJECTED coordinates: x = full width, y = [-60N..pole].
    y_bot = ax.projection.transform_point(0.0, -60.0, ccrs.PlateCarree())[1]
    ax.set_xlim(*ax.projection.x_limits)
    ax.set_ylim(y_bot, ax.projection.y_limits[1])
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    if basins is not None:
        ax.add_feature(basins)
    ax.scatter(lon, lat, c=code, s=0.3, cmap=cmap, norm=norm,
               transform=ccrs.PlateCarree(), rasterized=True, zorder=2)
    ax.set_title("Best GAMLSS model by AIC (per grid cell)", fontsize=26, pad=15)
    plt.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {filename}")

    # legend written to its OWN png (kept off the map)
    cnt, n = counts_table(code)
    handles = [plt.Line2D([0], [0], marker="s", linestyle="", markersize=16,
                          markerfacecolor=COLOR[m], markeredgecolor="none",
                          label=f"{NAME[m]}  ({100*cnt[m]/n:.1f}%)") for m in MODELS]
    figl, axl = plt.subplots(figsize=(4.6, 1.9))
    axl.axis("off")
    axl.legend(handles=handles, loc="center", fontsize=18, frameon=True,
               title="AIC-preferred model", title_fontsize=19,
               facecolor="white", edgecolor="#888888")
    leg_file = filename.replace(".png", "_legend.png")
    figl.savefig(leg_file, dpi=200, bbox_inches="tight")
    plt.close(figl)
    print(f"Saved: {leg_file}")


def plot_pie(code, filename, delta_aic):
    cnt, n = counts_table(code)
    sizes = [cnt[m] for m in MODELS]
    colors = [COLOR[m] for m in MODELS]
    labels = [f"{NAME[m]}\n{cnt[m]:,} ({100*cnt[m]/n:.1f}%)" for m in MODELS]
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(sizes, labels=labels, colors=colors, startangle=90,
           counterclock=False, textprops={"fontsize": 16},
           wedgeprops={"edgecolor": "white", "linewidth": 1.5})
    ax.set_title(f"AIC-preferred model\n(ΔAIC threshold = {delta_aic:g}; n = {n:,} cells)",
                 fontsize=18)
    ax.axis("equal")
    plt.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {filename}")


def stratified_bar(cid, code, ctype, climate, filename, delta_aic):
    """Stacked bar chart of the AIC-preferred-model fractions per change-type x
    climate stratum (plus an 'All' reference). Shows WHERE the data support the
    more complex models -- e.g. if quadratic wins mostly in 'trend/high' cells,
    that localises the regions where curvature is real, strengthening the
    discussion that linear is the safe default elsewhere."""
    strata, frac, counts = [], {m: [] for m in MODELS}, []

    def add(label, mask):
        n = int(np.sum(mask))
        strata.append(f"{label}\n(n={n:,})")
        counts.append(n)
        for i, m in enumerate(MODELS):
            frac[m].append(np.sum(code[mask] == i) / n if n else 0.0)

    add("All", np.ones(len(code), dtype=bool))
    for t in TYPES:
        for c in CLIMATES:
            add(f"{t}/{c}", (ctype == t) & (climate == c))

    x = np.arange(len(strata))
    fig, ax = plt.subplots(figsize=(15, 7))
    bottom = np.zeros(len(strata))
    for m in MODELS:
        vals = np.array(frac[m]) * 100.0
        ax.bar(x, vals, bottom=bottom, color=COLOR[m], edgecolor="white",
               linewidth=0.8, label=NAME[m])
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(strata, fontsize=12)
    ax.set_ylabel("Grid cells [%]", fontsize=16)
    ax.set_ylim(0, 100)
    ax.tick_params(axis="y", labelsize=14)
    ax.set_title(f"AIC-preferred model by change type x climate "
                 f"(ΔAIC threshold = {delta_aic:g})", fontsize=18)
    ax.legend(title="Model", fontsize=14, title_fontsize=15,
              loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.axvline(0.5, color="grey", linewidth=1.0, linestyle=":")   # split All vs strata
    fig.tight_layout()
    fig.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {filename}")

    # console table
    print(f"\nstratum            " + "  ".join(f"{NAME[m]:>10s}" for m in MODELS))
    for j, lab in enumerate([s.split("\n")[0] for s in strata]):
        print(f"{lab:18s} " + "  ".join(f"{frac[m][j]*100:9.1f}%" for m in MODELS))


def make_outputs(suffix, mask, cid, lon, lat, code, code_bar, basins, out_dir,
                 delta_aic, ctype, climate):
    """Map + pie (at delta_aic) + stratified bar (at BAR_DELTA_AIC) for a subset.
    The bar uses code_bar so the change-type breakdown is always the parsimony
    (dAIC>2) view, regardless of the map/pie threshold."""
    if int(np.sum(mask)) == 0:
        print(f"NOTE: no cells for view '{suffix or 'all'}' -> skipped.")
        return
    plot_map(lon[mask], lat[mask], code[mask], basins,
             os.path.join(out_dir, f"aic_best_model_map{suffix}.png"))
    plot_pie(code[mask], os.path.join(out_dir, f"aic_best_model_pie{suffix}.png"),
             delta_aic)
    if ctype is not None:
        keep = mask & (ctype != "?") & (climate != "?")
        if int(np.sum(keep)):
            stratified_bar(cid[keep], code_bar[keep], ctype[keep], climate[keep],
                           os.path.join(out_dir, f"aic_best_model_by_changetype{suffix}.png"),
                           BAR_DELTA_AIC)


def main():
    dat_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "data")
    delta_aic = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
    # flood-relevance thresholds, identical to 098: keep if
    #   (q90 >= X1 OR max_amax >= X2) AND uparea_km2 >= A
    x1 = float(sys.argv[3]) if len(sys.argv) > 3 else excess_lib.FLOOD_Q90
    x2 = float(sys.argv[4]) if len(sys.argv) > 4 else excess_lib.FLOOD_MAX
    a_up = float(sys.argv[5]) if len(sys.argv) > 5 else excess_lib.FLOOD_UPAREA

    summary_dir = os.path.join(dat_dir, "100", "summary")
    out_dir = os.path.join(dat_dir, "103")
    os.makedirs(out_dir, exist_ok=True)

    bad = excess_lib.load_excluded_cells(dat_dir)
    cid, lon, lat, aic = load_aic(summary_dir, bad)
    print(f"Cells: {len(lon)}" + (f"  (excluded {len(bad)} reverse-flow + "
          f"no-flood-regime cells)" if bad else ""))

    # console: effect of the parsimony threshold (all grid)
    print(f"\n{'model':11s} {'dAIC=0':>16s} {'dAIC=2':>16s}")
    c0 = select_model(aic, 0.0)
    c2 = select_model(aic, 2.0)
    n = len(aic)
    for i, m in enumerate(MODELS):
        n0, n2 = int(np.sum(c0 == i)), int(np.sum(c2 == i))
        print(f"{NAME[m]:11s} {n0:8d} ({100*n0/n:4.1f}%) {n2:8d} ({100*n2/n:4.1f}%)")

    basins = load_basin_feature(os.path.join(dat_dir, "shp", "basins"))
    code = select_model(aic, delta_aic)            # map / pie
    code_bar = select_model(aic, BAR_DELTA_AIC)    # stratified bar (parsimony)

    # classification + flood-relevance fields from 091 (optional)
    cells_csv = os.path.join(dat_dir, "091", "change_type_cells.csv")
    ctype = climate = None
    flood_mask = None
    if os.path.exists(cells_csv):
        tmap = load_class_map(cells_csv, "type")
        cmap = load_class_map(cells_csv, "climate")
        ctype = np.array([tmap.get(c, "?") for c in cid])
        climate = np.array([cmap.get(c, "?") for c in cid])
        q90 = load_num_map(cells_csv, "q90")
        maxm = load_num_map(cells_csv, "max_amax")
        upm = load_num_map(cells_csv, "uparea_km2")
        if q90 and upm:
            flood_mask = np.array([
                ((q90.get(c, np.nan) >= x1) or (maxm.get(c, np.nan) >= x2))
                and (upm.get(c, np.nan) >= a_up) for c in cid])
            nf = int(np.sum(flood_mask))
            print(f"\nFlood-relevant cells: {nf} / {n} ({100*nf/n:.1f}%)  "
                  f"[(q90>={x1:g} OR max>={x2:g}) AND uparea>={a_up:g}]")
        else:
            print("WARNING: q90/max_amax/uparea_km2 absent -> no flood-relevant view.")
    else:
        print(f"NOTE: {cells_csv} not found -> no stratified bars / flood view "
              f"(run 094_classify_pbs.sh + 095 first).")

    # machine-readable fractions (all + flood-ensured, dAIC 0 & 2)
    write_fractions_csv(aic, flood_mask,
                        os.path.join(out_dir, "aic_model_fractions.csv"))

    # View 1: all grid cells
    make_outputs("", np.ones(len(cid), dtype=bool), cid, lon, lat, code, code_bar,
                 basins, out_dir, delta_aic, ctype, climate)
    # View 2: flood-relevant cells only
    if flood_mask is not None:
        make_outputs("_flood", flood_mask, cid, lon, lat, code, code_bar,
                     basins, out_dir, delta_aic, ctype, climate)


if __name__ == "__main__":
    main()
