"""
figS2_target_cell_selection.py
Diagnostic to choose the flood-potential cutoff that excludes grids which never
flood, while KEEPING flash-flood grids (intermittent but with occasional large
peaks). Judgement quantity = the 90th percentile of the 120-yr AMAX (q90, i.e.
the ~once-per-decade peak), combined with upstream area (uparea).

For every analysis cell (max AMAX >= 1 m3/s, the same set the pipeline uses) it
computes:
  q90_amax  : 90th percentile of the 120 annual maxima  [m3/s]
  max_amax  : maximum of the 120 annual maxima          [m3/s]
  uparea    : upstream drainage area                     [km2]

Outputs (../data/099/):
  hist_q90.png, hist_max.png, hist_uparea.png   distributions (log axes)
  joint_q90_uparea.png                          2D density (log-log)
  retain_table.csv                              retained-cell fraction/count for
                                                a grid of (q90>=X) AND (uparea>=A)
  printed percentile summary + retain matrix

Retain rule examined: keep cell if  q90_amax >= X  AND  uparea >= A.

Inputs:
  ../data/030/amax_all.bin     (float32, n_landcells x 120)
  ../data/010/land_cells.csv
  uparea.bin                           (float32, 1800 x 3600, km2)

It also writes Figure S1 justifying the OR retain criterion
  keep = (q90 >= X_sel  OR  max >= X2_sel)  AND  uparea >= A_sel :
  si_threshold_panels.png
    (A) q90 and max AMAX distributions + the two cutoff lines
    (B) retained cells vs retained flood magnitude as the q90 cut varies (insensitivity)
    (C) q90 vs max density + OR keep region (red box = excluded) + flash-flood sites
    (D) global map of kept vs excluded cells

Usage (cluster):
  python3 figS2_target_cell_selection.py [dat_dir] [uparea_path] [X_sel] [A_sel] [X2_sel]
    X_sel  : q90 (recurrent) cutoff  [m3/s]  (default 10)
    A_sel  : uparea cutoff           [km2]   (default 1000)
    X2_sel : max (rare-flood) cutoff [m3/s]  (default 100)
"""
import os
import sys
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter, LogLocator
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import importlib
excess_lib = importlib.import_module("common_target_cells")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
N_YEARS = 120
NY, NX = 1800, 3600
MIN_FLOW = 1.0
CHUNK = 100000

# Candidate thresholds for the retain tables
X_Q90 = [1, 2, 5, 10, 20, 50, 100, 200]     # m3/s   (q90 of AMAX; recurrent floods)
A_UPAREA = [0, 50, 100, 500, 1000, 5000, 10000, 50000]  # km2
X2_MAX = [20, 50, 100, 200, 500]            # m3/s   (max of AMAX; rare large floods)
ZERO_COL = "#6a3d9a"                        # panel-D special colour: "retained = 0"


def main():
    dat_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(SCRIPT_DIR, "..", "data")
    uparea_path = sys.argv[2] if len(sys.argv) > 2 else \
        "/home/kk/jp_claude/gamlss/data/uparea.bin"
    X_SEL = float(sys.argv[3]) if len(sys.argv) > 3 else 10.0      # q90 (recurrent) cutoff X1
    A_SEL = float(sys.argv[4]) if len(sys.argv) > 4 else 1000.0    # uparea cutoff A
    X2_SEL = float(sys.argv[5]) if len(sys.argv) > 5 else 100.0    # max (rare-flood) cutoff X2

    amax_bin = os.path.join(dat_dir, "030", "amax_all.bin")
    cell_csv = os.path.join(dat_dir, "010", "land_cells.csv")
    out_dir = os.path.join(dat_dir, "099")
    os.makedirs(out_dir, exist_ok=True)
    for p in (amax_bin, cell_csv, uparea_path):
        if not os.path.exists(p):
            sys.exit(f"Not found: {p}")

    iy_list, ix_list = [], []
    with open(cell_csv, "r", newline="") as f:
        for row in csv.DictReader(f):
            iy_list.append(int(row["iy"])); ix_list.append(int(row["ix"]))
    iy = np.array(iy_list, np.int32); ix = np.array(ix_list, np.int32)
    n_cells = len(iy)
    print(f"Land cells: {n_cells}")

    # NOTE: glb_06min/uparea.bin is stored in m^2 (median ~2e8), so convert to km^2.
    uparea_grid = np.fromfile(uparea_path, dtype="<f4").reshape(NY, NX)
    uparea = uparea_grid[iy, ix].astype(np.float64) / 1.0e6   # m^2 -> km^2

    arr = np.memmap(amax_bin, dtype="<f4", mode="r", shape=(n_cells, N_YEARS))
    q90 = np.empty(n_cells); maxv = np.empty(n_cells); neff = np.empty(n_cells)
    for s in range(0, n_cells, CHUNK):
        e = min(s + CHUNK, n_cells)
        block = np.asarray(arr[s:e], dtype=np.float64)
        q90[s:e] = np.percentile(block, 90, axis=1)
        maxv[s:e] = block.max(axis=1)
        # effective flood count N_eff,var = (sum d^2)^2 / sum d^4  (= n / raw-kurtosis):
        # a heavy-tailed intermittent series (few floods among many dry years) has
        # huge kurtosis, so N_eff -> ~1; a regular series keeps N_eff ~ n.
        d = block - block.mean(axis=1, keepdims=True)
        s2 = (d ** 2).sum(axis=1); s4 = (d ** 4).sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            neff[s:e] = np.where(s4 > 0, s2 ** 2 / s4, np.nan)
        print(f"  computed {e}/{n_cells}", end="\r")
    print()

    valid = maxv >= MIN_FLOW
    bad = excess_lib.load_bad_cells(dat_dir)   # 031 negative-AMAX reverse-flow cells
    if bad:                                     # array position == cell_id (land order)
        ib = [c for c in bad if 0 <= c < n_cells]
        valid[ib] = False
        print(f"Excluded {len(ib)} negative-AMAX cells (031).")

    # dry-year fraction (070 zero_flow_rate) and Q100 estimation SD (050 stat_err_sd),
    # joined by cell_id into land-cell order, for the flood-ensured regime panel (E).
    dry = np.full(n_cells, np.nan); sdq = np.full(n_cells, np.nan)
    p070 = os.path.join(dat_dir, "070", "timeseries_tests.csv")
    p050 = os.path.join(dat_dir, "050", "summary_allgrid.csv")
    if os.path.exists(p070):
        with open(p070, newline="") as f:
            for row in csv.DictReader(f):
                try:
                    c = int(row["cell_id"])
                    if 0 <= c < n_cells:
                        dry[c] = float(row["zero_flow_rate"])
                except (ValueError, KeyError):
                    pass
    if os.path.exists(p050):
        with open(p050, newline="") as f:
            for row in csv.DictReader(f):
                try:
                    c = int(row["cell_id"])
                    if 0 <= c < n_cells:
                        sdq[c] = float(row["stat_err_sd"])
                except (ValueError, KeyError):
                    pass

    q90v = q90[valid]; maxvv = maxv[valid]; upv = uparea[valid]
    neffv = neff[valid]; dryv = dry[valid]; sdqv = sdq[valid]
    n_valid = int(valid.sum())
    tot_mag = q90v.sum()                      # total flood magnitude proxy (sum of q90)
    print(f"Analysis cells (max>=1): {n_valid}")

    # --- percentile summary ---
    def pct(a, ps):
        return {p: float(np.percentile(a, p)) for p in ps}
    ps = [5, 10, 25, 50, 75, 90, 95, 99]
    print("\nq90_amax [m3/s] percentiles:", {p: round(v, 2) for p, v in pct(q90v, ps).items()})
    print("uparea   [km2]   percentiles:", {p: round(v, 1) for p, v in pct(upv, ps).items()})

    # --- histograms ---
    def hist_log(data, title, xlabel, fname):
        d = data[(data > 0) & np.isfinite(data)]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(np.log10(d), bins=60, color="#4575b4", alpha=0.85)
        ax.set_xlabel(f"log10({xlabel})"); ax.set_ylabel("cells")
        ax.set_title(f"{title}  (n={len(d)}, zeros={int(np.sum(data <= 0))})")
        fig.tight_layout(); fig.savefig(os.path.join(out_dir, fname), dpi=160); plt.close(fig)
        print(f"Saved: {fname}")

    hist_log(q90v, "90th-percentile AMAX", "q90_amax [m3/s]", "hist_q90.png")
    hist_log(maxvv, "Maximum AMAX (120 yr)", "max_amax [m3/s]", "hist_max.png")
    hist_log(upv, "Upstream area", "uparea [km2]", "hist_uparea.png")

    # --- joint q90 vs uparea ---
    m = (q90v > 0) & (upv > 0)
    fig, ax = plt.subplots(figsize=(7, 6))
    hb = ax.hexbin(np.log10(upv[m]), np.log10(q90v[m]), gridsize=60, mincnt=1,
                   cmap="viridis", bins="log")
    ax.set_xlabel("log10 uparea [km2]"); ax.set_ylabel("log10 q90_amax [m3/s]")
    ax.set_title("Joint distribution (cell density)")
    fig.colorbar(hb, ax=ax, label="log10 cells")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "joint_q90_uparea.png"), dpi=160)
    plt.close(fig)
    print("Saved: joint_q90_uparea.png")

    # --- retain table: keep if q90>=X AND uparea>=A ---
    print("\nRetained fraction [%] of analysis cells  (rows q90>=X, cols uparea>=A):")
    header = "q90\\uparea " + "".join(f"{a:>9d}" for a in A_UPAREA)
    print(header)
    rows_out = [["q90_X\\uparea_A"] + [str(a) for a in A_UPAREA]]
    for X in X_Q90:
        cells_x = q90v >= X
        line = f"{X:>9.0f} "
        row = [str(X)]
        for A in A_UPAREA:
            keep = int(np.sum(cells_x & (upv >= A)))
            frac = 100.0 * keep / n_valid
            line += f"{frac:8.1f} "
            row.append(f"{keep}|{frac:.1f}")
        print(line)
        rows_out.append(row)

    # --- flood-magnitude retained [%] (sum of q90 of kept cells / total) ---
    print("\nFlood-magnitude retained [%]  (rows q90>=X, cols uparea>=A):")
    print(header)
    mag_rows = [["MAGNITUDE% q90_X\\uparea_A"] + [str(a) for a in A_UPAREA]]
    for X in X_Q90:
        cx = q90v >= X
        line = f"{X:>9.0f} "
        row = [str(X)]
        for A in A_UPAREA:
            mag = float(q90v[cx & (upv >= A)].sum() / tot_mag * 100)
            line += f"{mag:8.1f} "
            row.append(f"{mag:.1f}")
        print(line)
        mag_rows.append(row)

    # --- OR-criterion retained cells [%]: keep if (q90>=X1 OR max>=X2) AND uparea>=A_SEL ---
    print(f"\nOR-criterion retained cells [%] at uparea>={A_SEL:g}  "
          "(keep if q90>=X1 OR max>=X2):")
    print("  X1\\X2  " + "".join(f"{x2:>8.0f}" for x2 in X2_MAX))
    or_rows = [[f"OR cells% (uparea>={A_SEL:g}) X1\\X2"] + [str(x2) for x2 in X2_MAX]]
    upok = upv >= A_SEL
    for X1 in X_Q90:
        cq = q90v >= X1
        line = f"  {X1:>5.0f} "
        row = [str(X1)]
        for X2 in X2_MAX:
            keep = (cq | (maxvv >= X2)) & upok
            frac = 100.0 * keep.mean()
            line += f"{frac:8.1f} "
            row.append(f"{frac:.1f}")
        print(line)
        or_rows.append(row)

    # cells rescued ONLY by the rare-flood clause (max>=X2_SEL, but q90<X_SEL)
    rescued = ((q90v < X_SEL) & (maxvv >= X2_SEL) & upok)
    print(f"\nAt X1={X_SEL:g}, X2={X2_SEL:g}, A={A_SEL:g}: "
          f"{int(rescued.sum())} cells ({100*rescued.mean():.2f}%) kept ONLY by the "
          f"rare-flood clause (max>={X2_SEL:g} but q90<{X_SEL:g})")
    keep_or = ((q90v >= X_SEL) | (maxvv >= X2_SEL)) & upok
    print(f"  OR-criterion total kept: {int(keep_or.sum())} ({100*keep_or.mean():.1f}%)")

    with open(os.path.join(out_dir, "retain_table.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["# n_valid", n_valid])
        w.writerow(["# cells: retained_count|retained_percent ; keep if q90>=X AND uparea>=A"])
        w.writerows(rows_out)
        w.writerow([])
        w.writerow(["# flood-magnitude retained [%] (sum q90 kept / total)"])
        w.writerows(mag_rows)
        w.writerow([])
        w.writerow([f"# OR-criterion cells% (q90>=X1 OR max>=X2) AND uparea>={A_SEL:g}"])
        w.writerows(or_rows)
    print(f"\nSaved: retain_table.csv  (n_valid={n_valid})")

    # ================================================================
    # Figure S1: 4-panel justification of the OR criterion
    #   keep = (q90 >= X_SEL  OR  max >= X2_SEL)  AND  uparea >= A_SEL
    # ================================================================
    lonv = -180.0 + (ix[valid] + 0.5) * 0.1
    latv = 90.0 - (iy[valid] + 0.5) * 0.1

    upok = upv >= A_SEL
    maxok = maxvv >= X2_SEL
    keepv = ((q90v >= X_SEL) | maxok) & upok

    # log-scaled axes but labelled in REAL discharge (1, 10, 100, ...) instead of
    # log10 values, which are hard to read. (Discharge spans ~12 orders of
    # magnitude, so a truly linear axis is not usable.)
    def _mfmt(x, _pos):
        if x <= 0:
            return ""
        if x < 1:
            return f"{x:g}"
        return f"{int(round(x)):,}"
    dfmt = FuncFormatter(_mfmt)

    # complementary-rescue counts (used by panel C and the printed summary)
    n_maxonly = int(((q90v <  X_SEL) & (maxvv >= X2_SEL) & upok).sum())
    n_q90only = int(((q90v >= X_SEL) & (maxvv <  X2_SEL) & upok).sum())
    n_both    = int(((q90v >= X_SEL) & (maxvv >= X2_SEL) & upok).sum())
    n_excl    = int(((q90v <  X_SEL) & (maxvv <  X2_SEL) & upok).sum())

    # ---- panel drawers: each renders ONE panel into a given axes, so the same
    #      code produces the combined 4-panel figure AND standalone per-panel files.
    def panel_A(ax):
        """q90 and max AMAX distributions + the two cutoff lines."""
        pos_q = q90v[q90v > 0]; pos_m = maxvv[maxvv > 0]
        lo = max(1e-3, float(min(pos_q.min(), pos_m.min())))
        hi = float(max(pos_q.max(), pos_m.max()))
        bins = np.logspace(np.log10(lo), np.log10(hi), 60)
        ax.hist(pos_q, bins=bins, color="#4575b4", alpha=0.55, label="q90")
        ax.hist(pos_m, bins=bins, color="#f46d43", alpha=0.45, label="max")
        ax.axvline(X_SEL, color="#4575b4", lw=2, ls="--", label=f"q90 cut = {X_SEL:g}")
        ax.axvline(X2_SEL, color="#d73027", lw=2, ls="--", label=f"max cut = {X2_SEL:g}")
        ax.set_xscale("log"); ax.set_xlim(lo, hi)
        ax.xaxis.set_major_locator(LogLocator(numticks=12))
        ax.xaxis.set_major_formatter(dfmt)
        ax.set_xlabel("discharge [m3/s]  (log scale)"); ax.set_ylabel("cells")
        ax.set_title("(A) q90 and max AMAX distributions"); ax.legend(fontsize=8)

    def panel_B(ax):
        """Retained cells vs retained flood magnitude as the q90 cut varies."""
        Xs = np.logspace(0, 2.5, 50)
        fc = [float(np.mean(((q90v >= xi) | maxok) & upok) * 100) for xi in Xs]
        fm = [float(q90v[((q90v >= xi) | maxok) & upok].sum() / tot_mag * 100) for xi in Xs]
        ax.plot(Xs, fc, color="#4575b4", label="cells retained [%]")
        ax.plot(Xs, fm, color="#1a9850", label="flood magnitude retained [%]")
        ax.axvline(X_SEL, color="#d73027", lw=2, ls="--")
        ax.set_xscale("log"); ax.xaxis.set_major_formatter(dfmt)
        ax.set_xlabel(f"q90 threshold [m3/s]  (OR max>={X2_SEL:g}, uparea>={A_SEL:g}; log scale)")
        ax.set_ylabel("% of analysis total"); ax.set_ylim(0, 101)
        ax.set_title("(B) Retained cells vs flood magnitude"); ax.legend()

    def panel_C(ax):
        """q90 vs max SCATTER, coloured by which magnitude criterion keeps each cell.
        Core "why two criteria" justification: each criterion RESCUES a distinct
        population the other misses, so neither alone is sufficient.
          both     (q90>=X, max>=X2) : ordinary flood rivers          (grey)
          max only (q90< X, max>=X2) : flashy low-flow / flash-flood   (orange)
          q90 only (q90>=X, max< X2) : steady rivers, no extreme       (blue)
          excluded (q90< X, max< X2) : never reach flood scale         (red)
        Axes remain LOG (discharge spans ~12 orders of magnitude) but the view is
        floored at 1 m3/s (no confusing 1e-10 tick) and labelled in real m3/s. Each
        quadrant is capped at N_PER points so the small q90-only group is visible."""
        base = upok & (q90v > 0) & (maxvv > 0)
        q = q90v[base]; mx = maxvv[base]
        g_both = (q >= X_SEL) & (mx >= X2_SEL)
        g_excl = (q <  X_SEL) & (mx <  X2_SEL)
        g_maxo = (q <  X_SEL) & (mx >= X2_SEL)
        g_q90o = (q >= X_SEL) & (mx <  X2_SEL)
        rng = np.random.default_rng(0)
        N_PER = 6000
        # draw bulk first, the two rescue groups (the story) last / on top
        for gmask, col in [(g_both, "#7f7f7f"), (g_excl, "#d62728"),
                           (g_maxo, "#ff7f0e"), (g_q90o, "#1f77b4")]:
            gi = np.nonzero(gmask)[0]
            if gi.size > N_PER:
                gi = rng.choice(gi, N_PER, replace=False)
            ax.scatter(mx[gi], q[gi], s=4, c=col, alpha=0.35, edgecolors="none", zorder=2)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.axhline(X_SEL, color="#08519c", lw=1.8, ls="--", zorder=3)
        ax.axvline(X2_SEL, color="#a50f15", lw=1.8, ls="--", zorder=3)
        ax.set_xlim(1.0, float(mx.max()) * 1.4); ax.set_ylim(1.0, float(q.max()) * 1.4)
        ax.xaxis.set_major_formatter(dfmt); ax.yaxis.set_major_formatter(dfmt)
        ax.tick_params(axis="both", labelsize=14)
        bb = dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.88, edgecolor="#cccccc")
        tk = dict(transform=ax.transAxes, fontsize=14, fontweight="bold", bbox=bb, zorder=5)
        ax.text(0.03, 0.97, f"q90 only\n{n_q90only:,}", color="#08519c", ha="left",  va="top",    **tk)
        ax.text(0.97, 0.97, f"both\n{n_both:,}",        color="#4d4d4d", ha="right", va="top",    **tk)
        ax.text(0.97, 0.03, f"max only\n{n_maxonly:,}", color="#b35900", ha="right", va="bottom", **tk)
        ax.text(0.03, 0.03, f"excluded\n{n_excl:,}",    color="#a50f15", ha="left",  va="bottom", **tk)
        ax.set_xlabel("max_amax [m3/s]  (log scale)", fontsize=13)
        ax.set_ylabel("q90_amax [m3/s]  (log scale)", fontsize=13)
        ax.set_title("(C) complementary criteria: cells kept ONLY by q90 vs ONLY by max")

    def panel_D(ax):
        """Retained-fraction map aggregated to 1-deg bins (avoids point overplotting,
        which would otherwise hide excluded cells under the kept ones). Bins where
        NO cell is retained (fraction exactly 0) get a distinct colour (ZERO_COL),
        separate from the low-but-nonzero red end. The legend for that colour is
        written to a SEPARATE png (si_panelD_legend.png), not drawn on the map.
        Antarctica is cropped out."""
        ax.set_extent([-179.9, 179.9, -58.0, 84.0], crs=ccrs.PlateCarree())  # no Antarctica
        ax.add_feature(cfeature.COASTLINE, linewidth=0.4)
        binsize = 1.0
        nlat = int(180 / binsize); nlon = int(360 / binsize)
        bi = ((90.0 - latv) / binsize).astype(int).clip(0, nlat - 1)
        bj = ((lonv + 180.0) / binsize).astype(int).clip(0, nlon - 1)
        tot = np.zeros((nlat, nlon)); kep = np.zeros((nlat, nlon))
        np.add.at(tot, (bi, bj), 1.0)
        np.add.at(kep, (bi, bj), keepv.astype(float))
        frac = np.divide(kep, tot, out=np.full_like(kep, np.nan), where=tot > 0)
        lon_edges = -180.0 + np.arange(nlon + 1) * binsize
        lat_edges = 90.0 - np.arange(nlat + 1) * binsize
        LON, LAT = np.meshgrid(lon_edges, lat_edges)
        cmap = plt.get_cmap("RdYlGn").copy(); cmap.set_under(ZERO_COL)
        # vmin slightly >0 so fraction==0 falls into the "under" (special) colour,
        # while empty (no-cell) bins stay NaN and are not drawn.
        pcm = ax.pcolormesh(LON, LAT, frac, cmap=cmap, vmin=1e-6, vmax=1,
                            transform=ccrs.PlateCarree())
        ax.get_figure().colorbar(pcm, ax=ax, orientation="horizontal", pad=0.04,
                                 shrink=0.7, extend="min",
                                 label="fraction of analysis cells retained")
        ax.set_title(f"(D) Retained fraction (1deg bins)\n"
                     f"(q90>={X_SEL:g} OR max>={X2_SEL:g}) & uparea>={A_SEL:g}")

    def panel_E(ax):
        """WHY the flood-ENSURED regime cut (dry-year fraction <= 0.5) is needed.
        Among flood-relevant (magnitude-passing) cells, bin by dry-year fraction and
        show that as it rises past 0.5 the effective flood count
        N_eff = (sum d^2)^2 / sum d^4 collapses from ~14 to ~2, i.e. the 100-year
        level is extrapolated from only one or two effective events. (The Monte-Carlo
        Q100 SD itself stays bounded here -- the delta-correction keeps the Gumbel
        fit numerically stable -- so the breakdown is in effective SAMPLE SIZE, not
        fit variance; that is exactly what N_eff measures.) Median + IQR band."""
        sel = (((q90v >= X_SEL) | (maxvv >= X2_SEL)) & upok
               & np.isfinite(dryv) & np.isfinite(neffv))
        dx = dryv[sel]; ne = neffv[sel]
        edges = np.linspace(0.0, 1.0, 21)
        idx = np.clip(np.digitize(dx, edges) - 1, 0, 19)
        ctr, m25, m50, m75 = [], [], [], []
        for b in range(20):
            mb = idx == b
            if mb.sum() < 30:
                continue
            ctr.append(0.5 * (edges[b] + edges[b + 1]))
            q1, q2, q3 = np.percentile(ne[mb], [25, 50, 75])
            m25.append(float(q1)); m50.append(float(q2)); m75.append(float(q3))
        ax.axvspan(0.5, 1.0, color="#d7301f", alpha=0.08, zorder=0)
        ax.axvline(0.5, color="#d73027", lw=2, ls="--")
        ax.axhline(3.0, color="#888888", lw=1.0, ls=":")
        ax.fill_between(ctr, m25, m75, color="#1a9850", alpha=0.18, label="IQR (25-75%)")
        ax.plot(ctr, m50, "-s", color="#1a9850", ms=4, label="$N_{eff}$ median")
        ax.set_xlim(0, 1); ax.set_ylim(0, None)
        ax.set_xlabel("dry-year fraction  (070 zero_flow_rate)")
        ax.set_ylabel("effective flood count  $N_{eff}=(\\Sigma d^2)^2/\\Sigma d^4$")
        m_wet = np.median(neffv[sel & (dryv <= 0.5)])
        m_dry = np.median(neffv[sel & (dryv > 0.5)])
        ax.text(0.72, 0.93, f"removed by flood-ensured\n(dry-year > 0.5)\n"
                f"median $N_{{eff}}$: {m_wet:.1f} $\\to$ {m_dry:.1f}",
                transform=ax.transAxes, color="#7f1d0f", ha="center", va="top",
                fontsize=8, fontweight="bold")
        ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
        ax.set_title("(E) flood-ensured regime cut: dry cells lose effective floods "
                     "($N_{eff}$ collapses)")

    # ---- combined figure (A-D justify the magnitude thresholds; E justifies the
    #      flood-ensured regime cut). 3x2 grid, last slot blank. ----
    fig = plt.figure(figsize=(15, 16))
    panel_A(fig.add_subplot(3, 2, 1))
    panel_B(fig.add_subplot(3, 2, 2))
    panel_C(fig.add_subplot(3, 2, 3))
    panel_D(fig.add_subplot(3, 2, 4, projection=ccrs.Robinson()))
    panel_E(fig.add_subplot(3, 2, 5))
    fig.add_subplot(3, 2, 6).axis("off")
    fig.tight_layout()
    out = os.path.join(out_dir, "si_threshold_panels.png")
    fig.savefig(out, dpi=180, bbox_inches="tight"); plt.close(fig)
    print(f"Saved: {out}")

    # ---- standalone per-panel figures (A, C, D) for independent placement ----
    figA, axA = plt.subplots(figsize=(8, 5.2)); panel_A(axA)
    figA.tight_layout(); pA = os.path.join(out_dir, "si_panelA_distributions.png")
    figA.savefig(pA, dpi=180, bbox_inches="tight"); plt.close(figA); print(f"Saved: {pA}")

    figC, axCs = plt.subplots(figsize=(7.6, 6.6)); panel_C(axCs)
    figC.tight_layout(); pC = os.path.join(out_dir, "si_panelC_complementary.png")
    figC.savefig(pC, dpi=180, bbox_inches="tight"); plt.close(figC); print(f"Saved: {pC}")

    figD = plt.figure(figsize=(11, 6))
    panel_D(figD.add_subplot(1, 1, 1, projection=ccrs.Robinson()))
    pD = os.path.join(out_dir, "si_panelD_retained_map.png")
    figD.savefig(pD, dpi=180, bbox_inches="tight"); plt.close(figD); print(f"Saved: {pD}")

    # standalone legend for panel D's special "retained = 0" colour (kept off-map)
    from matplotlib.patches import Patch
    figL, axL = plt.subplots(figsize=(3.4, 0.8)); axL.axis("off")
    axL.legend(handles=[Patch(facecolor=ZERO_COL, edgecolor="none",
                              label="retained = 0 (never)")],
               loc="center", fontsize=12, framealpha=1.0, edgecolor="#888888")
    pL = os.path.join(out_dir, "si_panelD_legend.png")
    figL.savefig(pL, dpi=180, bbox_inches="tight"); plt.close(figL); print(f"Saved: {pL}")

    figE, axE = plt.subplots(figsize=(8, 5.4)); panel_E(axE)
    figE.tight_layout(); pE = os.path.join(out_dir, "si_panelE_regime.png")
    figE.savefig(pE, dpi=180, bbox_inches="tight"); plt.close(figE); print(f"Saved: {pE}")

    # console: contrast the two sides of the regime cut (among flood-relevant cells)
    selE = (((q90v >= X_SEL) | (maxvv >= X2_SEL)) & upok
            & np.isfinite(dryv) & np.isfinite(sdqv) & np.isfinite(neffv))
    wet = selE & (dryv <= 0.5); drym = selE & (dryv > 0.5)
    if wet.any() and drym.any():
        print(f"  regime cut (flood-relevant cells): dry<=0.5  n={int(wet.sum()):,}  "
              f"median N_eff={np.median(neffv[wet]):.1f}  "
              f"Q100 SD median={100*np.median(sdqv[wet]):.1f}% p95={100*np.percentile(sdqv[wet],95):.1f}%")
        print(f"                                     dry> 0.5  n={int(drym.sum()):,}  "
              f"median N_eff={np.median(neffv[drym]):.1f}  "
              f"Q100 SD median={100*np.median(sdqv[drym]):.1f}% p95={100*np.percentile(sdqv[drym],95):.1f}%")

    n_keep = int(keepv.sum())
    print(f"OR criterion (q90>={X_SEL:g} OR max>={X2_SEL:g}) & uparea>={A_SEL:g}: "
          f"kept {n_keep} ({100*n_keep/n_valid:.1f}%), "
          f"magnitude {q90v[keepv].sum()/tot_mag*100:.1f}%")
    print(f"  complementary rescue (why BOTH criteria): "
          f"ONLY-max (q90<{X_SEL:g}, max>={X2_SEL:g}) = {n_maxonly:,}; "
          f"ONLY-q90 (q90>={X_SEL:g}, max<{X2_SEL:g}) = {n_q90only:,}; "
          f"both = {n_both:,}; excluded = {n_excl:,}")


if __name__ == "__main__":
    main()
