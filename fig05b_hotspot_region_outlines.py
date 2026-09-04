"""
fig05b_hotspot_region_outlines.py  (EXPERIMENT, local)
Draw the SD-improvement tiers as region OUTLINES on the LINEAR-FIT SD maps:
  tier 1 EXTREME : 1-deg block-mean improvement >= 50%            -> BLACK
  tier 2 HIGH    : 1-deg block-mean improvement >= cell-p80 (~20%) -> WHITE
                   (inclusive of tier 1, so the outlines nest)
The tier-2 threshold is the TOP QUINTILE (p80): in the 5-hue x 4-shade
equal-count map the purple hue family is exactly the last 4 of 20 bins =
the top 20% of cells, so the WHITE outline traces precisely the visually
purple regions ("purple = tier 2" holds by construction, and the threshold
is easy to state in a paper: "the top quintile, ~20% improvement").

Nearby regions are MERGED by a morphological closing of the block masks
(radius CLOSE_R blocks; bridges gaps up to ~2*CLOSE_R deg), so the outlines
read as a few coherent regions instead of many small specks.

Overlays are drawn on:
  (a) 051's linear-fit SD map, linear YlOrRd scale     -> tiers_lin_sd.png
  (b) same field, equal-count 5x4 colormap             -> tiers_lin_sd_5x4.png
  (c) the SD-improvement 5x4 map itself (as 052)       -> tiers_sd_improve_lin.png

It ALSO draws two INDEPENDENT context maps carrying the SAME tier outlines, so
the reader can see the improvement hotspots sit in the ARID belts:
  (d) global mean discharge (mean AMAX)                -> tiers_discharge.png
  (e) specific discharge q_spec = mean AMAX / uparea   -> tiers_qspec.png
      (the aridity proxy: low q_spec = dry -> BROWN, wet -> teal)
Both use a log scale (fields span orders of magnitude) and carry the tier1/tier2
outlines unchanged. Discharge/q_spec are joined by cell_id: iy/ix from the 063
all-grid file, mean AMAX from 070/timeseries_tests.csv, uparea from uparea.bin
(same join as 500); scope is ALL valid cells (no flood-ensured restriction) so
the background aridity pattern is shown as fully as possible.

Reuses 052's loaders/renderers and 051's loader for the SD columns, and 069's
cell_id loaders for the discharge fields.

Outputs: ../data/506/
Usage:   python3 fig05b_hotspot_region_outlines.py [dat_dir] [parent] [ns_model]
                 [--uparea-path PATH]
    parent/ns_model (default ln/lin) only select which 063 file supplies iy/ix
    for the discharge join; they do NOT change the tier outlines.
"""
import os
import sys
import csv
import importlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import ListedColormap, BoundaryNorm, LogNorm
import cartopy.crs as ccrs

m052 = importlib.import_module("fig05a_improvement_map")
m051 = importlib.import_module("lib_global_map_renderer")
m069 = importlib.import_module("lib_equivalent_ensemble_by_flow")
m107 = importlib.import_module("fig04_parent_by_fit_boxplots")
excess_lib = importlib.import_module("common_target_cells")

RES = m052.RES
BLOCK = m052.HOT_BLOCK          # 10 cells = 1 deg
NX, NY = 3600, 1800
UPAREA_DEFAULT = "/home/kk/jp_claude/gamlss/data/uparea.bin"
Q1 = 99.0                       # tier-1 = top percentile (p99)
Q2 = 80.0                       # tier-2 = top QUINTILE (p80, ~20%): identical by
                                # construction to the PURPLE hue family of the
                                # 5x4 equal-count maps (last 4 of 20 bins)
CLOSE_R = 1                     # closing radius [blocks], CROSS element (narrow)
MIN_T1_BLOCKS = 5               # tier-1 minimum region size [1-deg blocks]
MIN_T2_BLOCKS = 60              # tier-2 minimum: ~the Australia cluster (60 blocks);
                                # smaller clusters are dropped as sub-continental noise

# --- MANUAL region edits (documented in the manuscript text) -----------------
# The Central-Asian tier-2 blocks fuse into ONE meshed cluster stretching from
# Kazakhstan down to Afghanistan. To display Afghanistan as its OWN region we
# artificially SEVER that neck: each cut is a lon/lat box of 1-deg blocks forced
# OUT of tier2, breaking the 4-connectivity bridge between the two clusters.
# (These are deliberate manual operations, justified in the paper.)
MANUAL_CUTS = [
    # sever the RUSSIA/S-Siberia lobe (north) from the Kazakhstan/Central-Asia
    # mass (south) along ~lat 50.5 (approx the Russia-Kazakhstan border). Two
    # boxes leave lon 74-79 UNCUT so the eastern lobe near lon 74-80 (lat 41-50)
    # stays attached to Russia (D) instead of being orphaned into a fragment.
    (55.0, 74.0, 50.05, 50.95),
    (79.0, 108.0, 50.05, 50.95),
]
# Seeds (lon,lat) whose containing cluster is KEPT regardless of size, so a
# small but deliberately-severed cluster still displays after a cut. (The
# severed Russia lobe is large, so none are needed here.)
MANUAL_KEEP = [
]
# Local boundary-SMOOTHING boxes (lon0,lon1,lat0,lat1,radius): inside each box
# the mask is morphologically CLOSED with the given radius, filling small
# notches/concavities so the outline reads as a simpler shape (only ADDS blocks,
# never removes). Used to tidy the jagged Kazakhstan (F) boundary.
MANUAL_SMOOTH = [
    (55.0, 73.0, 42.0, 50.0, 2),   # Kazakhstan / Central-Asia cluster
]
T1_COL, T2_COL = "#000000", "#ffffff"   # tier1 black fill, tier2 white line
T2_LW = 1.3                             # tier2 outline width [pt] (white, dark halo)
T2_HALO = 1.6                           # extra width of the dark halo under the white
N_FINE = 100                            # fine histogram bins across the colormap range
                                        # (merged from the retired 509 script)
NAME_TAG = ""                           # appended to every output filename; set to
                                        # "_nobasin" by --no-basins so the river-basin
                                        # river-basin outlines are omitted in a parallel set


def _dilate(m, r):
    """CROSS (4-neighbour) dilation -- the narrowest useful buffer: one
    orthogonal block per iteration, no diagonal spread."""
    out = m.copy()
    for _ in range(r):
        prev = out
        out = prev.copy()
        out |= np.roll(prev, 1, 0) | np.roll(prev, -1, 0)
        out |= np.roll(prev, 1, 1) | np.roll(prev, -1, 1)
    return out


def _close(m, r):
    """Morphological closing (cross element): bridges only ~r-block orthogonal
    gaps between regions -- deliberately narrow so distinct belts (e.g. Russia
    vs Central/South Asia) are not welded together."""
    if r <= 0:
        return m
    return ~_dilate(~_dilate(m, r), r)


def _bridge_diagonals(mask):
    """Gap filling that keeps diagonal chains CONTIGUOUS under 4-connectivity:
    for every pair of flagged blocks touching only at a corner, fill one shared
    orthogonal neighbour (deterministically the northern one). Repeated to
    convergence. This is the explicit rule that keeps e.g. the Australia
    cluster in one piece instead of splitting at diagonal steps."""
    out = mask.copy()
    while True:
        a = out
        d1 = a[:-1, :-1] & a[1:, 1:] & ~a[:-1, 1:] & ~a[1:, :-1]   # NW-SE corner pairs
        d2 = a[:-1, 1:] & a[1:, :-1] & ~a[:-1, :-1] & ~a[1:, 1:]   # NE-SW corner pairs
        new = a.copy()
        new[:-1, 1:] |= d1     # fill the northern shared neighbour
        new[:-1, :-1] |= d2
        if not (new & ~a).any():
            return new
        out = new


def _components4(mask):
    """4-neighbour (orthogonal ONLY) connected components -- urban-mask style:
    blocks touching only diagonally are SEPARATE clusters, so diagonal-linked
    exclaves no longer ride along with a big cluster."""
    hbk, wbk = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    comps = []
    for i0 in range(hbk):
        for j0 in range(wbk):
            if not mask[i0, j0] or seen[i0, j0]:
                continue
            stack = [(i0, j0)]; seen[i0, j0] = True; cells = []
            while stack:
                i, j = stack.pop(); cells.append((i, j))
                for ii, jj in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)):
                    if 0 <= ii < hbk and 0 <= jj < wbk and mask[ii, jj] and not seen[ii, jj]:
                        seen[ii, jj] = True; stack.append((ii, jj))
            comps.append(cells)
    return comps


def _drop_small(mask, min_blocks, keep_seeds=None):
    """Remove 4-connected clusters smaller than min_blocks (urban-mask rule:
    a cluster must contain at least min_blocks orthogonally contiguous blocks).
    A cluster CONTAINING any (i,j) in keep_seeds is kept regardless of size, so
    a deliberately-severed small region (e.g. Afghanistan) still displays."""
    keep_seeds = set(keep_seeds or ())
    out = np.zeros_like(mask)
    for cells in _components4(mask):
        if len(cells) >= min_blocks or (keep_seeds and keep_seeds.intersection(cells)):
            for c in cells:
                out[c] = True
    return out


def _block_ij(lon, lat):
    """(block_row i, block_col j) of the 1-deg block containing lon/lat."""
    return (int((90.0 - lat) / (BLOCK * RES)),
            int((lon + 180.0) / (BLOCK * RES)))


def _apply_manual_cuts(mask, cuts=MANUAL_CUTS):
    """Force every 1-deg block inside a MANUAL_CUTS lon/lat box OUT of the mask,
    severing the 4-connectivity neck between two otherwise-fused clusters."""
    out = mask.copy()
    hbk, wbk = out.shape
    for (lon0, lon1, lat0, lat1) in cuts:
        ia, ja = _block_ij(lon0, lat1)
        ib, jb = _block_ij(lon1, lat0)
        i_lo, i_hi = sorted((ia, ib)); j_lo, j_hi = sorted((ja, jb))
        i_lo = max(0, i_lo); i_hi = min(hbk - 1, i_hi)
        j_lo = max(0, j_lo); j_hi = min(wbk - 1, j_hi)
        out[i_lo:i_hi + 1, j_lo:j_hi + 1] = False
    return out


def _keep_seed_ij(seeds=MANUAL_KEEP):
    """(i,j) block indices of the MANUAL_KEEP force-keep seeds."""
    return {_block_ij(lon, lat) for (lon, lat) in seeds}


def _apply_manual_smooth(mask, boxes=MANUAL_SMOOTH):
    """Within each MANUAL_SMOOTH box, replace the mask by its morphological
    closing (radius r), filling small notches so the outline is a simpler shape.
    Closing only adds blocks, so the box's mask grows to fill concavities; the
    rest of the map is untouched."""
    out = mask.copy()
    hbk, wbk = out.shape
    for (lon0, lon1, lat0, lat1, r) in boxes:
        closed = _close(mask, r)
        ia, ja = _block_ij(lon0, lat1)
        ib, jb = _block_ij(lon1, lat0)
        i_lo, i_hi = sorted((ia, ib)); j_lo, j_hi = sorted((ja, jb))
        i_lo = max(0, i_lo); i_hi = min(hbk - 1, i_hi)
        j_lo = max(0, j_lo); j_hi = min(wbk - 1, j_hi)
        out[i_lo:i_hi + 1, j_lo:j_hi + 1] |= closed[i_lo:i_hi + 1, j_lo:j_hi + 1]
    return out


def _fill_holes(mask):
    """Fill interior holes so contouring draws only OUTER region boundaries
    (holes otherwise appear as spurious small rings inside big clusters).
    The largest background component is the outside; all others are holes."""
    comps = sorted(m052._components(~mask), key=len, reverse=True)
    out = mask.copy()
    for cells in comps[1:]:
        for c in cells:
            out[c] = True
    return out


def tier_block_masks(lon, lat, vals, q1=Q1, q2=Q2, close_r=CLOSE_R):
    """Closed (merged, de-speckled) 1-deg block masks of the two improvement
    tiers. Both thresholds are cell percentiles (paper-friendly):
      tier1 = p99 (top 1%), tier2 = p80 (top quintile = the purple family).
    tier2 is INCLUSIVE of tier1, so the outlines nest cleanly."""
    v = vals[np.isfinite(vals)]
    thr1 = float(np.percentile(v, q1))
    thr2 = float(np.percentile(v, q2))
    bm = m052._block_means(lon, lat, vals, BLOCK)
    fin = np.isfinite(bm)
    # tier1 is FILLED black on the maps, so no morphology at all: exactly the
    # raw 1-deg blocks whose mean improvement reaches p99 (simplest definition)
    t1 = fin & (bm >= thr1)
    t2raw = _bridge_diagonals(_close(fin & (bm >= thr2), close_r))
    # MANUAL boundary smoothing (fill notches) BEFORE the cut, so the closing
    # does not re-weld a severed neck
    t2raw = _apply_manual_smooth(t2raw)
    # MANUAL severance of fused clusters (e.g. Russia from Kazakhstan), applied
    # AFTER smoothing/bridging so it is not immediately re-welded
    t2raw = _apply_manual_cuts(t2raw)
    keep_seeds = _keep_seed_ij()
    # diagnostics: tier-2 cluster sizes + centroids (justifies MIN_T2_BLOCKS)
    comps = sorted(_components4(t2raw), key=len, reverse=True)
    print("tier2 clusters (blocks, centroid lon/lat):")
    for c in comps:
        kept = len(c) >= MIN_T2_BLOCKS or bool(keep_seeds.intersection(c))
        if not kept:
            continue
        iis = [p[0] for p in c]; jjs = [p[1] for p in c]
        clat = 90.0 - (np.mean(iis) + 0.5) * BLOCK * RES
        clon = -180.0 + (np.mean(jjs) + 0.5) * BLOCK * RES
        tag = "KEPT" if len(c) >= MIN_T2_BLOCKS else "KEPT(manual seed)"
        print(f"    {len(c):4d}  ({clon:+7.1f},{clat:+6.1f})  {tag}")
    n_drop = sum(1 for c in comps if len(c) < MIN_T2_BLOCKS
                 and not keep_seeds.intersection(c))
    print(f"    ... dropping {n_drop} clusters < {MIN_T2_BLOCKS} blocks")
    t2 = _fill_holes(_drop_small(t2raw, MIN_T2_BLOCKS, keep_seeds=keep_seeds))
    return t1, t2, thr1, thr2


def _mask_edge_segments(mask):
    """RECTILINEAR outline of a block mask: for every flagged block, emit its
    N/S/E/W edge wherever the neighbour across that edge is unflagged. The
    result follows the 1-deg block edges exactly -- horizontal and vertical
    line segments only, no diagonals (urban-mask style)."""
    hbk, wbk = mask.shape
    segs = []
    for i in range(hbk):
        lat_t = 90.0 - i * BLOCK * RES
        lat_b = 90.0 - (i + 1) * BLOCK * RES
        for j in range(wbk):
            if not mask[i, j]:
                continue
            lon_l = -180.0 + j * BLOCK * RES
            lon_r = -180.0 + (j + 1) * BLOCK * RES
            if i == 0 or not mask[i - 1, j]:
                segs.append([(lon_l, lat_t), (lon_r, lat_t)])
            if i == hbk - 1 or not mask[i + 1, j]:
                segs.append([(lon_l, lat_b), (lon_r, lat_b)])
            if not mask[i, (j - 1) % wbk]:
                segs.append([(lon_l, lat_t), (lon_l, lat_b)])
            if not mask[i, (j + 1) % wbk]:
                segs.append([(lon_r, lat_t), (lon_r, lat_b)])
    return segs


def _draw_tiers(ax, masks):
    """tier2 = BLACK rectilinear block-edge outline over a WHITE halo (inner
    black line, outer white rim -- readable on both dark and light fields).
    tier1 is NOT drawn at all (no fill, no black colormap bin). Each merged
    tier-2 region gets a letter A, B, C, ... assigned west-to-east by centroid.
    NO in-map legend (saved separately)."""
    from matplotlib.collections import LineCollection
    _t1, t2, _thr1, _thr2 = masks
    # tier2 outline (two passes: wide WHITE underlay, then BLACK line on top --
    # a per-segment path effect would stamp end-caps across neighbouring
    # segments and make the outline look like a chain of bricks)
    segs = _mask_edge_segments(t2)
    if segs:
        ax.add_collection(LineCollection(
            segs, colors="#ffffff", linewidths=T2_LW + T2_HALO,
            transform=ccrs.PlateCarree(), zorder=6, capstyle="projecting"))
        ax.add_collection(LineCollection(
            segs, colors="#000000", linewidths=T2_LW,
            transform=ccrs.PlateCarree(), zorder=6.1, capstyle="projecting"))
    # region letters: FIXED paper order, matched to clusters by nearest centroid
    #   A=North China Plain, B=Sahelian belt, C=Southern Africa,
    #   D=West Siberia, E=Quebec  (extra clusters, if any, get F, G, ...)
    ANCHORS = [("A", 114.0, 36.0), ("B", 8.0, 9.0), ("C", 24.0, -22.0),
               ("D", 65.0, 59.0), ("E", -69.0, 50.0)]
    comps = _components4(t2)
    cents = []
    for cells in comps:
        ci = np.mean([i for i, _j in cells]); cj = np.mean([j for _i, j in cells])
        cents.append((-180.0 + (cj + 0.5) * BLOCK * RES,
                      90.0 - (ci + 0.5) * BLOCK * RES))
    used, lab = set(), {}
    for letter, alon, alat in ANCHORS:
        best, bestd = None, np.inf
        for k, (clon, clat) in enumerate(cents):
            if k in used:
                continue
            d = (clon - alon) ** 2 + (clat - alat) ** 2
            if d < bestd:
                bestd, best = d, k
        if best is not None:
            used.add(best); lab[best] = letter
    extra = iter("FGHIJK")
    for k in range(len(comps)):
        lab.setdefault(k, next(extra))
    for k, (clon, clat) in enumerate(cents):
        ax.text(clon, clat, lab[k], transform=ccrs.PlateCarree(),
                fontsize=20, fontweight="bold", color="#000000",
                ha="center", va="center", zorder=7,
                path_effects=[pe.withStroke(linewidth=4.0, foreground="white")])


def save_legend(out_png, thr1, thr2):
    """Standalone legend file for the tier outlines (tier2 only; tier1 removed)."""
    handles = [plt.Line2D([0], [0], color="#000000", lw=2.0,
                          path_effects=[pe.withStroke(linewidth=3.8, foreground="white")])]
    labels = [f"hotspot regions: block mean $\\geq$ p80 ({thr2:.0f}%, top quintile)"]
    fig, ax = plt.subplots(figsize=(6.4, 1.1))
    ax.axis("off")
    leg = ax.legend(handles, labels, loc="center", fontsize=15, framealpha=1.0,
                    facecolor="white", edgecolor="#555555")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_png}")


def plot_tiers(lon, lat, vals, masks, title, cbar_label, out_png,
               mode="q5x4", cmap_name="YlOrRd", black_tier1_thr=None):
    """Raster map of `vals` (equal-count 5x4 or robust-linear scale) + tier
    outlines from the SD-improvement masks.

    black_tier1_thr : if given (the tier-1 threshold thr1, in the units of
        `vals`), the equal-count colormap gets an extra BLACK top bin covering
        `vals >= thr1`, so tier-1 cells render black FROM their own grid value
        (no opaque overlay). Only meaningful when `vals` IS the improvement."""
    v = vals[np.isfinite(vals)]
    fig = plt.figure(figsize=(21, 12.6))
    ax = m052._base_ax(fig)
    if mode == "q5x4":
        if black_tier1_thr is not None:
            thr = float(black_tier1_thr)
            base = m052._q5x4_edges(v)
            lower = base[base < thr]
            hi = base[-1] if base[-1] > thr else thr + max(1e-9, abs(thr) * 1e-3)
            edges = np.concatenate([lower, [thr, hi]])
            nb = edges.size - 1
            cmap = ListedColormap(list(m052.semantic_5x4_colors(nb - 1))
                                  + [(0.0, 0.0, 0.0)])
        else:
            edges = m052._q5x4_edges(v)
            nb = edges.size - 1
            cmap = ListedColormap(m052.semantic_5x4_colors(nb))
        norm = BoundaryNorm(edges, nb)
        sc = m052._imshow(ax, lon, lat, np.clip(vals, edges[0], edges[-1]),
                          cmap, norm=norm)
        ticks = list(edges[::4] if nb >= 8 else edges)
        if black_tier1_thr is not None and thr not in ticks:
            ticks = sorted(set(list(ticks) + [thr]))   # show the tier1/black edge
        # improvement map: FLAT left end (extend only the top) so the colorbar
        # reads "minimum = the bottom edge (~15%)"; the top keeps a triangle for
        # the >=p99.5 values clipped into the black tier1 bin
        extend = "max" if black_tier1_thr is not None else "both"
        cbar = plt.colorbar(sc, ax=ax, orientation="horizontal", pad=0.05,
                            shrink=0.7, aspect=40, ticks=ticks, extend=extend)
        cbar.ax.set_xticklabels([f"{e:.2g}" for e in ticks])
        lbl = f"{cbar_label} (equal-count bins" + \
              (f"; black = tier1, $\\geq${thr:.0f}%)" if black_tier1_thr is not None else ")")
        cbar.set_label(lbl, fontsize=24)
    else:
        vmin, vmax = np.percentile(v, [2, 98])
        sc = m052._imshow(ax, lon, lat, np.clip(vals, vmin, vmax), cmap_name,
                          vmin=vmin, vmax=vmax)
        cbar = plt.colorbar(sc, ax=ax, orientation="horizontal", pad=0.05,
                            shrink=0.7, aspect=40, extend="both")
        cbar.set_label(cbar_label, fontsize=24)
    cbar.ax.tick_params(labelsize=18)
    _draw_tiers(ax, masks)
    ax.set_title(title, fontsize=26, pad=15)
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_png}")


def load_ensured_yx(dat_dir):
    """(iy,ix) set of FLOOD-ENSURED cells: cell_id->iy/ix from 050 intersected
    with the flood-ensured cell_ids (excess_lib). None if the flood-ensured list
    is unavailable. cell_id is a sequential land index (NOT iy*NX+ix), so the
    (iy,ix) mapping must be read from the 050 table."""
    ids = excess_lib.load_flood_ensured_ids(dat_dir)
    if not ids:
        return None
    ids = set(ids)
    yx = set()
    with open(os.path.join(dat_dir, "050", "summary_allgrid.csv"), newline="") as f:
        for row in csv.DictReader(f):
            if int(row["cell_id"]) in ids:
                yx.add((int(row["iy"]), int(row["ix"])))
    return yx


def _yx_mask(lon, lat, yxset):
    """Boolean mask of cells whose (iy,ix) is in yxset (fast via linear index)."""
    ix = np.round((np.asarray(lon) + 180.0) / RES - 0.5).astype(np.int64)
    iy = np.round((90.0 - np.asarray(lat)) / RES - 0.5).astype(np.int64)
    keep_lin = np.fromiter((a * NX + b for (a, b) in yxset),
                           dtype=np.int64, count=len(yxset))
    return np.isin(iy * NX + ix, keep_lin)


def load_discharge_fields(dat_dir, parent, ns_model, uparea_path, ensured_yx=None):
    """Global mean discharge (mean AMAX) and specific discharge (mean AMAX /
    uparea) per cell, joined by cell_id as in 500: iy/ix from the 063 all-grid
    file, mean AMAX from 070/timeseries_tests.csv, uparea from uparea.bin.
    Scope = ALL valid land cells (finite, uparea>0, not reverse-flow/Antarctica),
    OR restricted to `ensured_yx` (a flood-ensured (iy,ix) set) when given.
    Returns lon, lat (cell centres) and the two value arrays."""
    eq_path = os.path.join(dat_dir, "063", f"eq_ens_allgrid_{parent}_{ns_model}.csv")
    mo_path = os.path.join(dat_dir, "070", "timeseries_tests.csv")
    eqmap = m069.load_eq_ens(eq_path)
    momap = m069.load_mean_outflow(mo_path)
    bad_yx = excess_lib.load_bad_yx(dat_dir)
    up_grid = np.fromfile(uparea_path, dtype="<f4").reshape(NY, NX) / 1.0e6  # km^2
    lon_l, lat_l, mo_l, qs_l = [], [], [], []
    for cid, (iy, ix, _eqv) in eqmap.items():
        if (iy, ix) in bad_yx:
            continue
        if ensured_yx is not None and (iy, ix) not in ensured_yx:
            continue
        lat = 90.0 - (iy + 0.5) * RES
        if lat < m052.LAT_CUT:
            continue
        mo = momap.get(cid, np.nan)
        upv = up_grid[iy, ix]
        if not (np.isfinite(mo) and mo > 0 and np.isfinite(upv) and upv > 0):
            continue
        lon_l.append(-180.0 + (ix + 0.5) * RES); lat_l.append(lat)
        mo_l.append(mo); qs_l.append(mo / upv)
    lon = np.array(lon_l); lat = np.array(lat_l)
    mo = np.array(mo_l); qs = np.array(qs_l)
    print(f"discharge fields: {mo.size} cells "
          f"(discharge median={np.median(mo):.3g} m3/s, "
          f"q_spec median={np.median(qs):.3g} m3/s/km2)")
    return lon, lat, mo, qs


def plot_field_with_tiers(lon, lat, vals, masks, title, cbar_label, out_png,
                          cmap="Blues", pctl=(2, 98)):
    """Log-scaled raster of a positive field (discharge / q_spec) + the tier1
    fill and tier2 outline. vmin/vmax are robust percentiles so a few extreme
    cells do not wash out the colour range."""
    v = vals[np.isfinite(vals) & (vals > 0)]
    vmin, vmax = np.percentile(v, list(pctl))
    if not (vmin > 0):
        vmin = float(v.min())
    norm = LogNorm(vmin=vmin, vmax=vmax)
    fig = plt.figure(figsize=(21, 12.6))
    ax = m052._base_ax(fig)
    sc = m052._imshow(ax, lon, lat, np.clip(vals, vmin, vmax), cmap, norm=norm)
    cbar = plt.colorbar(sc, ax=ax, orientation="horizontal", pad=0.05,
                        shrink=0.7, aspect=40, extend="both")
    cbar.set_label(cbar_label, fontsize=24)
    cbar.ax.tick_params(labelsize=18)
    _draw_tiers(ax, masks)
    ax.set_title(title, fontsize=26, pad=15)
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_png}")


def run_scope(dat_dir, out_dir, parent, ns_model, uparea_path, ensured_yx, suffix):
    """Produce the full 506 figure set for one SCOPE. suffix="" is the all-land
    version (unchanged filenames); suffix="_ensured" restricts every field to
    the flood-ensured (iy,ix) set so tiers, SD and discharge maps share the
    eq_ens (510) domain."""
    scope_lbl = "flood-ensured" if ensured_yx is not None else "all-land"
    print(f"\n===== scope: {scope_lbl} (suffix='{suffix}') =====")

    # --- improvement (as 052) -> tier masks (closed/merged) ------------------
    lons, lats, st_pct, improve_lin, _iq, _al, _aq = \
        m052.load_data(os.path.join(dat_dir, "050", "summary_allgrid.csv"))
    # drop non-physical (|imp|>CAP) AND the rare NEGATIVE-improvement cells
    # (~0.006%: linear worse than stationary), so the minimum improvement is
    # non-negative and the colormap starts at a clean positive value (~15%)
    keep = (np.isfinite(improve_lin) & (np.abs(improve_lin) <= m052.CAP_PCT)
            & (improve_lin >= 0.0))
    if ensured_yx is not None:
        keep &= _yx_mask(lons, lats, ensured_yx)
    lo, la, vv = lons[keep], lats[keep], improve_lin[keep]
    print(f"improvement cells: {vv.size}")
    masks = tier_block_masks(lo, la, vv)
    t1, t2, thr1, thr2 = masks
    print(f"tier1 blocks={int(t1.sum())}  tier2 blocks={int(t2.sum())} "
          f"(thr1=p{Q1:g}={thr1:.1f}%, thr2=p{Q2:g}={thr2:.1f}%, "
          f"closing r={CLOSE_R}, min region t1={MIN_T1_BLOCKS} t2={MIN_T2_BLOCKS} blocks)")
    save_legend(os.path.join(out_dir, f"tiers_legend{suffix}{NAME_TAG}.png"), thr1, thr2)

    # --- linear-fit SD field (051 columns) ------------------------------------
    lons51, lats51, values = m051.load_data()
    sd = values["lin_err_sd"]
    k = np.isfinite(sd) & (sd > 0) & (sd <= m052.CAP_PCT)
    if ensured_yx is not None:
        k &= _yx_mask(lons51, lats51, ensured_yx)
    title_sd = "Linear nonstationary — SD of relative error\n" \
               "(outlines: SD-improvement tiers, 1deg blocks, merged)"
    # (a) linear YlOrRd (as 051 lin_sd.png)
    plot_tiers(lons51[k], lats51[k], sd[k], masks, title_sd, "SD [%]",
               os.path.join(out_dir, f"tiers_lin_sd{suffix}{NAME_TAG}.png"), mode="linear")
    # (b) equal-count 5x4 (as 051 lin_sd_5x4.png)
    plot_tiers(lons51[k], lats51[k], sd[k], masks, title_sd, "SD [%]",
               os.path.join(out_dir, f"tiers_lin_sd_5x4{suffix}{NAME_TAG}.png"), mode="q5x4")
    # (c) the SD-improvement 5x4 map itself (purple family == tier 2 by construction)
    plot_tiers(lo, la, vv, masks,
               "SD improvement: Linear over Stationary  (hotspot outlines, merged)",
               "Improvement [%] (positive = linear better)",
               os.path.join(out_dir, f"tiers_sd_improve_lin{suffix}{NAME_TAG}.png"), mode="q5x4")

    # --- INDEPENDENT context maps: discharge + specific discharge (aridity) ----
    dlon, dlat, mo, qs = load_discharge_fields(dat_dir, parent, ns_model,
                                               uparea_path, ensured_yx)
    # (d) mean discharge: dark blue = big rivers; log scale
    plot_field_with_tiers(
        dlon, dlat, mo, masks,
        "Mean annual-maximum discharge  (tier outlines: SD-improvement hotspots)",
        "mean AMAX discharge [m$^3$ s$^{-1}$] (log)",
        os.path.join(out_dir, f"tiers_discharge{suffix}{NAME_TAG}.png"), cmap="Blues")
    # (e) specific discharge = aridity proxy: BROWN = dry (low q_spec), teal = wet
    plot_field_with_tiers(
        dlon, dlat, qs, masks,
        "Specific discharge q$_{spec}$ = mean AMAX / uparea   "
        "(brown = arid; tier outlines = SD-improvement hotspots)",
        "q$_{spec}$ [m$^3$ s$^{-1}$ km$^{-2}$] (log; brown=dry, teal=wet)",
        os.path.join(out_dir, f"tiers_qspec{suffix}{NAME_TAG}.png"), cmap="BrBG")

    # --- improvement HISTOGRAM (merged from 509), one per scope so it pairs with
    #     tiers_sd_improve_lin{suffix}.png: suffix="" -> all-land, "_ensured" ->
    #     flood-ensured. Uses this scope's improvement values `vv`. ------------
    improvement_histogram(vv, os.path.join(out_dir, f"improvement_hist_tier1{suffix}{NAME_TAG}.png"))


def improvement_histogram(v, out_png, q1=Q1):
    """Bare histogram of the SD improvement `v`, coloured by the SAME equal-count
    5-hue x 4-shade semantic palette as the paired tiers_sd_improve_lin map. Each
    bar takes the colour of the map bin its value falls in, so the histogram and
    the map read in identical colours. NO black tier1 bin (removed). Deliberately
    bare (big ticks, no labels/title/legend), log-y so the rare high tail stays
    visible.

    Merged verbatim from the retired 509 script; call once per scope so the
    histogram matches its scope's map (all-land vs flood-ensured)."""
    v = v[np.isfinite(v)]
    edges = m052._q5x4_edges(v)                       # 20 equal-count bins, no tier1 bin
    nb = edges.size - 1
    colors = list(m052.semantic_5x4_colors(nb))

    lo, hival = float(edges[0]), float(edges[-1])
    fine = np.linspace(lo, hival, N_FINE + 1)
    counts, be = np.histogram(v, bins=fine)              # values outside dropped
    centers = 0.5 * (be[:-1] + be[1:])
    idx = np.clip(np.searchsorted(edges, centers, side="right") - 1, 0, nb - 1)

    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.bar(centers, counts, width=np.diff(be),
           color=[colors[i] for i in idx], edgecolor="none")
    ax.set_yscale("log")
    ax.tick_params(axis="both", which="major", labelsize=40, length=12, width=2.2)
    ax.tick_params(axis="both", which="minor", length=6, width=1.4)
    ax.set_xlabel(""); ax.set_ylabel(""); ax.set_title("")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  improvement hist: n={v.size} min={v.min():.1f}% median={np.median(v):.1f}% "
          f"range=[{lo:.1f},{hival:.1f}]%")
    print(f"Saved: {out_png}")


def improvement_boxplot_by_parent(data, ns_fits, flood_ids, out_png):
    """Independent figure, built exactly like 514's parent boxplot but for 506's
    OWN quantity: the SD improvement % = (sd_stat - sd_ns)/sd_stat*100 of each
    nonstationary fit OVER THE STATIONARY fit, per PARENT (truth) distribution
    {st,ln,qd} x nonstationary fit {linear, quadratic, AIC-adaptive}, restricted
    to the FLOOD-ENSURED cells. NEGATIVE improvement (fit worse than stationary)
    is DROPPED so the domain matches 506's maps/histogram (improve >= 0). Reuses
    107's loaders/colours.

    The plot is deliberately bare: big tick labels, NO axis names / title, and NO
    in-figure legend -- the legend is written to a SEPARATE file (out_png stem +
    '_legend.png')."""
    TICK_FS = 22
    nf = len(ns_fits); bw = 0.8 / nf
    offs = (np.arange(nf) - (nf - 1) / 2.0) * bw
    fig, ax = plt.subplots(figsize=(9.5, 6.8))
    use_flood = bool(flood_ids)
    if not use_flood:
        print("  WARNING: flood-ensured id set empty -> boxplot falls back to ALL cells")
    allv = []
    for gi, (pk, pn) in enumerate(m107.PARENTS):
        d = data[pk]
        in_view = (np.array([c in flood_ids for c in d["cid"]]) if use_flood
                   else np.ones(len(d["cid"]), bool))
        for fi, f in enumerate(ns_fits):
            m = (in_view & m107.valid_mask(d, "st") & m107.valid_mask(d, f)
                 & (d["st"]["sd"] > 0))
            imp = (d["st"]["sd"][m] - d[f]["sd"][m]) / d["st"]["sd"][m] * 100.0
            imp = imp[imp >= 0.0]                    # drop negative improvement (as 506's maps)
            if imp.size == 0:
                continue
            allv.append(imp)
            bp = ax.boxplot(imp, positions=[gi + 1 + offs[fi]], widths=bw * 0.9,
                            whis=(5, 95), showfliers=False, patch_artist=True,
                            medianprops=dict(color="black", linewidth=1.4))
            for b in bp["boxes"]:
                b.set(facecolor=m107.FIT_COL[f], alpha=0.85, edgecolor="#555555")
            for el in bp["whiskers"] + bp["caps"]:
                el.set(color="#555555")
            print(f"  improvement {pn:16s} x {m107.FIT_NAME[f]:12s}: "
                  f"median={np.median(imp):5.1f}% (n={imp.size})")
    lo = min(np.percentile(v, 5) for v in allv); hi = max(np.percentile(v, 95) for v in allv)
    pad = 0.06 * (hi - lo); ax.set_ylim(max(0.0, lo - pad), hi + pad)
    ax.set_xticks([1, 2, 3]); ax.set_xticklabels([pn for _, pn in m107.PARENTS])
    ax.set_xlim(0.5, 3.5)
    ax.set_xlabel(""); ax.set_ylabel(""); ax.set_title("")   # no axis names / title
    ax.tick_params(axis="both", labelsize=TICK_FS); ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(out_png, dpi=200, bbox_inches="tight"); plt.close()
    print(f"Saved: {out_png}")

    # ---- standalone legend (separate file) ----
    leg_png = out_png.replace(".png", "_legend.png")
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=m107.FIT_COL[f], alpha=0.85,
                             edgecolor="#555555") for f in ns_fits]
    labels = [m107.FIT_NAME[f] for f in ns_fits]
    figl, axl = plt.subplots(figsize=(4.2, 1.7))
    axl.axis("off")
    leg = axl.legend(handles, labels, title="Nonstationary fit", loc="center",
                     fontsize=15, framealpha=1.0, facecolor="white", edgecolor="#555555")
    leg.get_title().set_fontsize(16)
    figl.savefig(leg_png, dpi=200, bbox_inches="tight"); plt.close(figl)
    print(f"Saved: {leg_png}")


def run_improvement_boxplot(dat_dir, out_dir):
    """Driver for the independent parent-distribution improvement boxplot (as 514):
    load the 107 parent files (st=045b, ln=054, qd=045b; ad from 108), attach the
    AIC-adaptive fit if present, and draw the flood-ensured boxplot."""
    files = m107.parent_files(dat_dir)
    missing = [p for p, fp in files.items() if not os.path.exists(fp)]
    if missing:
        print(f"NOTE: parent files missing {missing} -> skipping improvement boxplot "
              f"(need 045b/054 outputs).")
        return
    fits = m107.detect_fits(dat_dir)
    pdata = {pk: m107.load_parent(files[pk], m107.FITS_BASE) for pk, _ in m107.PARENTS}
    if "ad" in fits:
        m107.attach_ad(pdata, dat_dir)
    ns_fits = [f for f in fits if f != "st"]
    flood_ids = excess_lib.load_flood_ensured_ids(dat_dir)
    print(f"\n===== independent parent boxplot (flood-ensured: "
          f"{len(flood_ids) if flood_ids else 0} cells) =====")
    improvement_boxplot_by_parent(pdata, ns_fits, flood_ids,
                                  os.path.join(out_dir, "improvement_boxplot_by_parent.png"))


def main():
    global NAME_TAG
    argv = sys.argv[1:]
    uparea_path = UPAREA_DEFAULT
    scope = "both"
    no_basins = False
    if "--uparea-path" in argv:
        i = argv.index("--uparea-path"); uparea_path = argv[i + 1]; del argv[i:i + 2]
    if "--scope" in argv:
        i = argv.index("--scope"); scope = argv[i + 1]; del argv[i:i + 2]
    if "--no-basins" in argv:
        argv.remove("--no-basins"); no_basins = True; NAME_TAG = "_nobasin"
    parent = argv[1] if len(argv) > 1 else "ln"
    ns_model = argv[2] if len(argv) > 2 else "lin"
    dat_dir = argv[0] if len(argv) > 0 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "data")
    out_dir = os.path.join(dat_dir, "506")
    os.makedirs(out_dir, exist_ok=True)
    if no_basins:
        m052.BASINS = None          # omit river-basin outlines: coastlines only
        print("--no-basins: river-basin outlines OFF (outputs tagged '_nobasin')")
    else:
        m052.BASINS = m052.load_basin_feature(os.path.join(dat_dir, "shp", "basins"))

    # scope: "all" (all-land, unsuffixed), "ensured" (flood-ensured, _ensured),
    # or "both" (default: emit both, so the ensured version pairs with 510)
    jobs = []
    if scope in ("all", "both"):
        jobs.append((None, ""))
    if scope in ("ensured", "both"):
        ensured_yx = load_ensured_yx(dat_dir)
        if ensured_yx is None:
            print("WARNING: flood-ensured list unavailable -> skipping ensured version")
        else:
            print(f"flood-ensured cells: {len(ensured_yx)}")
            jobs.append((ensured_yx, "_ensured"))
    for ensured_yx, suffix in jobs:
        run_scope(dat_dir, out_dir, parent, ns_model, uparea_path, ensured_yx, suffix)

    # independent parent-distribution improvement boxplot (as 514's boxplot), one
    # flood-ensured figure regardless of the map scope above. It has no map, so the
    # basin outlines are irrelevant -> skip it on the --no-basins pass (no overwrite).
    if not no_basins:
        run_improvement_boxplot(dat_dir, out_dir)


if __name__ == "__main__":
    main()
