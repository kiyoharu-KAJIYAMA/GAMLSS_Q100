"""
fig05a_improvement_map.py
Plot SD improvement rate (%) of nonstationary models over stationary.

  improvement = (SD_stat - SD_nonstat) / SD_stat * 100 [%]
  Positive = nonstationary model reduces uncertainty.

Maps:
  - sd_improve_lin.png   : Linear vs Stationary
  - sd_improve_quad.png  : Quadratic vs Stationary

Input:
  ../data/050/summary_allgrid.csv

Antarctica (lat < -60) is cropped; non-physical arid cells (|improvement| >
CAP_PCT, from truth_q100 ~ 0) are dropped; river-basin outlines (dashed) replace
country borders. Four variants are written per panel for comparison:
  - sd_improve_{lin,quad}.png         relative %, linear robust symmetric +/- p98
  - sd_improve_{lin,quad}_symlog.png  relative %, symlog: small + large visible
  - sd_improve_{lin,quad}_5x4.png     relative %, equal-count 5-hue x 4-shade bins
  - sd_improve_{lin,quad}_abs.png     ABSOLUTE SD reduction (SD_stat - SD_nonstat)
                                      in PERCENTAGE POINTS (err SD is a relative-
                                      error fraction, x100); same % units as the
                                      SD itself; equal-count 5-hue x 4-shade bins;
                                      paper-primary, no small-denominator blow-up

The relative % maps express reduction as a fraction OF the stationary SD and
explode where SD_stat ~ 0 (arid cells), which is why CAP_PCT exists. The absolute
map instead reports the reduction in percentage points (same units the SD is read
in), so "stationary -> nonstationary shrinks the SD by X points" is read directly;
it has no small-denominator pathology and is the honest headline metric. Console
summaries are area-weighted (cos(lat)) and report the median.

Output: ../data/052/

Usage:
  python3 fig05a_improvement_map.py [dat_dir]
"""
import os
import sys
import csv
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm, SymLogNorm
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature

RES = 0.1
NLON, NLAT = 3600, 1800     # 0.1deg global grid (lon x lat)
LAT_CUT = -60.0             # omit Antarctica band
# Cells whose |improvement| exceeds this (%) are non-physical (arid cells with
# truth_q100 ~ 0 blow the relative-error SD up to ~1e137); dropped before plot.
CAP_PCT = 300.0

# River-basin outlines (drawn instead of country borders, as in 049/097/098)
BASIN_MIN_AREA = 50000.0   # km2; smaller HydroBASINS polygons are not drawn
BASINS = None              # ShapelyFeature, set once in main()


def load_basin_feature(basin_dir, min_area_km2=BASIN_MIN_AREA):
    """Major river-basin outlines (HydroBASINS lev03) as a reusable DASHED map
    feature, matching 049/097/098. Returns None if no shapefiles are found."""
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

# --- Custom 5-hue x 4-shade colormap (a semantic tab20b replacement) ---------
# The 5 hue families are ORDERED so small values read as "low" and large values
# as "high" (red -> orange -> green -> blue -> purple); within each family the 4
# shades go LIGHT (small) -> DARK (large), i.e. reversed relative to tab20b.
# Yellow is dropped and the hues are spread around the wheel so adjacent families
# stay distinguishable; light shades keep enough saturation to keep their hue.
SEMANTIC_HUES = ["#b2182b", "#e08214", "#4daf4a", "#2166ac", "#762a83"]


def _family_shades(base, n=4, toward_white=0.40, toward_black=0.38):
    """n shades of `base`, lightest first (small value) to darkest last.
    toward_white < 0.5 keeps the lightest shade saturated enough that its hue
    family is still recognisable (avoids fading to near-grey)."""
    base = np.array(mcolors.to_rgb(base))
    lightest = base + (1.0 - base) * toward_white
    darkest = base * (1.0 - toward_black)
    return [tuple(lightest + (darkest - lightest) * (k / (n - 1))) for k in range(n)]


def semantic_5x4_colors(nb, hues=SEMANTIC_HUES, shades=4):
    """Ordered list of nb colors built from the 5x4 (hue x shade) design.
    For the usual nb == len(hues)*shades the full grid is returned; otherwise
    nb colors are sampled evenly along it, preserving the low->high ordering."""
    full = []
    for h in hues:
        full.extend(_family_shades(h, shades))
    if nb == len(full):
        return full
    idx = np.linspace(0, len(full) - 1, nb).round().astype(int)
    return [full[i] for i in idx]


def _area_weighted_summary(vals, lats, valid):
    """Median and cos(lat) area-weighted mean of `vals` over `valid` cells.
    On a regular lat/lon grid a 0.1deg cell near the poles covers far less
    ground than one at the equator, so an unweighted mean over-counts high
    latitudes; cos(lat) weighting corrects that for any paper-quoted average."""
    v = vals[valid]
    if v.size == 0:
        return float("nan"), float("nan")
    w = np.cos(np.radians(lats[valid]))
    return np.median(v), float(np.sum(w * v) / np.sum(w))


def load_data(csv_path):
    """Load CSV, compute relative (%) and absolute SD improvements."""
    lons, lats = [], []
    st_sd, ln_sd, qd_sd = [], [], []

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            iy = int(row["iy"])
            ix = int(row["ix"])
            lat = 90.0 - (iy + 0.5) * RES
            lon = -180.0 + (ix + 0.5) * RES

            if lat < -60:
                continue

            try:
                s = float(row["stat_err_sd"])
                l = float(row["lin_err_sd"])
                q = float(row["quad_err_sd"])
            except (ValueError, KeyError):
                continue

            # Drop arid garbage: a non-finite SD (truth_q100 ~ 0 -> ~1e137/inf)
            # or stat SD <= 0 would make the improvement ratio inf/overflow.
            if not (np.isfinite(s) and np.isfinite(l) and np.isfinite(q)) or s <= 0:
                continue

            lons.append(lon)
            lats.append(lat)
            st_sd.append(s)
            ln_sd.append(l)
            qd_sd.append(q)

    lons = np.array(lons)
    lats = np.array(lats)
    st_sd = np.array(st_sd)
    ln_sd = np.array(ln_sd)
    qd_sd = np.array(qd_sd)

    # Relative improvement [%]: positive = nonstationary is better. Pathological
    # where st_sd ~ 0 (arid), hence the CAP_PCT filtering downstream.
    improve_lin  = (st_sd - ln_sd) / st_sd * 100
    improve_quad = (st_sd - qd_sd) / st_sd * 100

    # Absolute SD reduction in PERCENTAGE POINTS. The err SDs are relative-error
    # fractions (e.g. 0.10 = 10%), so *100 puts the reduction in the SAME % units
    # the SD is read in -- i.e. "how many points of relative-error SD the
    # nonstationary model removes" (st_sd - ns_sd, e.g. 0.018 -> 1.8 pt). This
    # keeps units aligned with the % maps; no small-denominator blow-up.
    absd_lin  = (st_sd - ln_sd) * 100
    absd_quad = (st_sd - qd_sd) * 100

    print(f"Loaded {len(lons)} grid cells")
    for nm, rel, ab in (("Linear", improve_lin, absd_lin),
                        ("Quadratic", improve_quad, absd_quad)):
        # Same physically-valid mask drives both summaries AND the maps, so the
        # relative and absolute views always describe the identical cell set.
        valid = np.isfinite(rel) & (np.abs(rel) <= CAP_PCT)
        n_bad = int(np.sum(~valid))
        r_md, r_wm = _area_weighted_summary(rel, lats, valid)
        a_md, a_wm = _area_weighted_summary(ab, lats, valid)
        print(f"{nm:9s} improvement: median={r_md:.1f}%, area-wtd mean={r_wm:.1f}%  "
              f"| abs SD reduction: median={a_md:.2f} pt, area-wtd mean={a_wm:.2f} pt  "
              f"(excluded {n_bad} non-physical/arid cells)")

    # baseline stationary SD in % (for the DeltaSD-vs-level dependence check)
    st_pct = st_sd * 100.0
    return lons, lats, st_pct, improve_lin, improve_quad, absd_lin, absd_quad


# --- raster (imshow) rendering: gap-free 0.1deg cells, as 066/120 -------------
def _raster_grid(lon, lat, vals):
    """Place per-cell values onto the regular 0.1deg global grid (NaN = no data)
    and crop rows below LAT_CUT. lon/lat are cell centres built from ix/iy, so
    the inverse index mapping is exact."""
    ix = np.round((np.asarray(lon) + 180.0) / RES - 0.5).astype(int)
    iy = np.round((90.0 - np.asarray(lat)) / RES - 0.5).astype(int)
    arr = np.full((NLAT, NLON), np.nan)
    arr[iy, ix] = vals
    n_keep = int(round((90.0 - LAT_CUT) / RES))
    return np.ma.masked_invalid(arr[:n_keep, :]), [-180.0, 180.0, LAT_CUT, 90.0]


def _imshow(ax, lon, lat, vals, cmap, norm=None, vmin=None, vmax=None):
    """imshow(nearest) replacement for the old per-cell scatter: no sub-pixel
    marker aliasing, thin land is kept, ocean/no-data stays transparent."""
    arr, extent = _raster_grid(lon, lat, vals)
    cmap = (plt.get_cmap(cmap) if isinstance(cmap, str) else cmap).copy()
    cmap.set_bad(alpha=0.0)
    kw = dict(norm=norm) if norm is not None else dict(vmin=vmin, vmax=vmax)
    return ax.imshow(arr, origin="upper", extent=extent, transform=ccrs.PlateCarree(),
                     cmap=cmap, interpolation="nearest", regrid_shape=NLON,
                     zorder=0, **kw)


# --- DeltaSD vs baseline-SD dependence (is the absolute map just the SD map?) --
def _avg_ranks(a):
    order = np.argsort(a, kind="mergesort")
    r = np.empty(a.size, float)
    r[order] = np.arange(1, a.size + 1)
    u, inv = np.unique(a, return_inverse=True)
    sums = np.bincount(inv, weights=r); cnts = np.bincount(inv)
    return (sums / cnts)[inv]


def _spearman(x, y):
    rx = _avg_ranks(np.asarray(x)); ry = _avg_ranks(np.asarray(y))
    rx = rx - rx.mean(); ry = ry - ry.mean()
    return float((rx * ry).sum() / np.sqrt((rx ** 2).sum() * (ry ** 2).sum()))


def _block_pair(lon, lat, a, b, block=10):
    """1-deg block means of two co-located cell fields; paired finite blocks."""
    aa, _ = _raster_grid(lon, lat, a)
    bb, _ = _raster_grid(lon, lat, b)
    out = []
    for arr in (aa.filled(np.nan), bb.filled(np.nan)):
        hbk, wbk = arr.shape[0] // block, arr.shape[1] // block
        ab = arr[:hbk * block, :wbk * block].reshape(hbk, block, wbk, block)
        fin = np.isfinite(ab)
        cnt = fin.sum(axis=(1, 3))
        s = np.where(fin, ab, 0.0).sum(axis=(1, 3))
        out.append(np.where(cnt >= 0.2 * block * block, s / np.maximum(cnt, 1), np.nan))
    m = np.isfinite(out[0]) & np.isfinite(out[1])
    return out[0][m], out[1][m]


def _q5x4_edges(v, n_bins=20):
    """Equal-count bin edges over the inner 0.5-99.5 pct range -- the SAME edges
    the _5x4 quantile maps use, so figures sharing them are colour-consistent."""
    lo, hi = np.percentile(v, [0.5, 99.5])
    core = v[(v >= lo) & (v <= hi)]
    if core.size < 2:
        core = v
    return np.unique(np.quantile(core, np.linspace(0.0, 1.0, n_bins + 1)))


def make_improvement_hist(vals, out_png, title):
    """STANDALONE histogram of the relative SD improvement, with each bar
    coloured by the SAME equal-count 5-hue x 4-shade bins as the _5x4 map --
    the reader can match map colours to values and their frequencies directly."""
    v = np.asarray(vals); v = v[np.isfinite(v)]
    med = float(np.median(v))
    edges = _q5x4_edges(v)
    nb = len(edges) - 1
    cols = semantic_5x4_colors(nb)
    lo, hi = np.percentile(v, [0.5, 99.5])
    counts, bedges = np.histogram(v, bins=np.linspace(lo, hi, 80))
    centers = 0.5 * (bedges[:-1] + bedges[1:])
    idx = np.clip(np.searchsorted(edges, centers, side="right") - 1, 0, nb - 1)
    fig, ax = plt.subplots(figsize=(8.5, 6))
    ax.bar(centers, counts, width=np.diff(bedges), color=[cols[i] for i in idx],
           edgecolor="none")
    ax.set_yscale("log")
    ax.axvline(med, color="crimson", linewidth=1.6, linestyle="--",
               label=f"median = {med:.1f}%")
    ax.axvline(0.0, color="#555555", linewidth=1.2, linestyle=(0, (5, 3)))
    ax.set_xlabel("SD improvement [%]", fontsize=14)
    ax.set_ylabel("number of cells (log)", fontsize=14)
    ax.tick_params(axis="both", labelsize=12)
    ax.set_title(title, fontsize=15)
    ax.legend(fontsize=12)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_png}")


def make_improvement_dist(vals, out_png, title):
    """STANDALONE distribution figure of the relative SD improvement:
    left  = histogram (log count axis: bulk AND heavy tail both visible),
    right = inverse CDF (percentile -> value) with labelled deciles, so the
    'mostly ~uniform bulk + long tail' shape is shown honestly in one figure."""
    v = np.asarray(vals); v = v[np.isfinite(v)]
    med = float(np.median(v))
    fig, (axh, axp) = plt.subplots(1, 2, figsize=(14, 5.6))
    lo, hi = np.percentile(v, [0.5, 99.5])
    axh.hist(v, bins=np.linspace(lo, hi, 80), color="#9ecae1", edgecolor="#4477aa",
             linewidth=0.3)
    axh.set_yscale("log")
    axh.axvline(med, color="crimson", linewidth=1.6, linestyle="--",
                label=f"median = {med:.1f}%")
    axh.axvline(0.0, color="#555555", linewidth=1.2, linestyle=(0, (5, 3)))
    axh.set_xlabel("SD improvement [%]"); axh.set_ylabel("number of cells (log)")
    axh.legend(fontsize=11); axh.grid(alpha=0.25)
    qs = np.linspace(0, 100, 501)
    axp.plot(qs, np.percentile(v, qs), color="#333333")
    dec = np.arange(0, 101, 10); dv = np.percentile(v, dec)
    axp.plot(dec, dv, "o", color="#0072B2")
    for d, val in zip(dec, dv):
        axp.annotate(f"{val:.1f}", (d, val), textcoords="offset points",
                     xytext=(4, -2), fontsize=8, color="#0072B2")
    axp.axhline(0.0, color="#d62728", linewidth=1.4, linestyle=(0, (7, 3)))
    axp.set_xlabel("percentile"); axp.set_ylabel("SD improvement [%]")
    axp.grid(alpha=0.25)
    fig.suptitle(title, fontsize=15)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_png}")


# --- hotspot detection: few, strictly-thresholded contiguous regions ----------
HOT_BLOCK = 10        # aggregation block in cells (10 x 0.1deg = 1deg blocks)
HOT_Q = 99.5          # STRICT percentile of block-mean improvement
HOT_MIN_BLOCKS = 3    # min contiguous 1-deg blocks for a region (drops specks)
HOT_MAX = 8           # hard cap -> a finite, labelled set of regions


def _block_means(lon, lat, vals, block=HOT_BLOCK):
    """1-deg block means of a cell field (NaN where <20% of the block has data)."""
    arr, _ = _raster_grid(lon, lat, vals)
    a = arr.filled(np.nan)
    hbk, wbk = a.shape[0] // block, a.shape[1] // block
    ab = a[:hbk * block, :wbk * block].reshape(hbk, block, wbk, block)
    fin = np.isfinite(ab)
    cnt = fin.sum(axis=(1, 3))
    s = np.where(fin, ab, 0.0).sum(axis=(1, 3))
    return np.where(cnt >= 0.2 * block * block, s / np.maximum(cnt, 1), np.nan)


def _components(mask):
    """8-neighbour connected components of a boolean grid -> list of cell lists."""
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
                for di in (-1, 0, 1):
                    for dj in (-1, 0, 1):
                        ii, jj = i + di, j + dj
                        if 0 <= ii < hbk and 0 <= jj < wbk and mask[ii, jj] and not seen[ii, jj]:
                            seen[ii, jj] = True; stack.append((ii, jj))
            comps.append(cells)
    return comps


def find_hotspots(lon, lat, vals, block=HOT_BLOCK, q=HOT_Q,
                  min_blocks=HOT_MIN_BLOCKS, max_n=HOT_MAX, abs_thr=None):
    """Contiguous regions of extremely high values. The field is block-averaged
    (1deg; suppresses cell-scale MC noise), blocks above the threshold (absolute
    abs_thr if given, else the q-th percentile of block means) are connected
    (8-neighbour); components smaller than min_blocks are dropped and the top
    max_n by mean value are returned as lon/lat bounding boxes."""
    bm = _block_means(lon, lat, vals, block)
    thr = float(abs_thr) if abs_thr is not None else float(np.nanpercentile(bm, q))
    hot = np.isfinite(bm) & (bm >= thr)
    comps = []
    for cells in _components(hot):
        if len(cells) < min_blocks:
            continue
        iis = [c[0] for c in cells]; jjs = [c[1] for c in cells]
        comps.append(dict(
            n=len(cells), mean=float(np.mean([bm[c] for c in cells])),
            lon0=-180.0 + min(jjs) * block * RES,
            lon1=-180.0 + (max(jjs) + 1) * block * RES,
            lat0=90.0 - (max(iis) + 1) * block * RES,
            lat1=90.0 - min(iis) * block * RES))
    comps.sort(key=lambda c: c["mean"], reverse=True)
    comps = comps[:max_n]
    tdesc = f"abs>={thr:g}" if abs_thr is not None else f"q{q:g}={thr:.1f}"
    print(f"hotspots: thr[{tdesc}] on {block * RES:g}deg block means "
          f"-> {len(comps)} regions (>= {min_blocks} blocks, capped at {max_n})")
    for k, c in enumerate(comps, 1):
        print(f"  H{k}: lon[{c['lon0']:.1f},{c['lon1']:.1f}] lat[{c['lat0']:.1f},{c['lat1']:.1f}] "
              f"mean={c['mean']:.1f} blocks={c['n']}")
    return comps


# latitude bands used to show the GEOGRAPHIC SPREAD of each improvement tier
LAT_BANDS = [("tropical(|lat|<23.5)", 0.0, 23.5),
             ("subtropical(23.5-35)", 23.5, 35.0),
             ("temperate(35-55)", 35.0, 55.0),
             ("boreal(>=55)", 55.0, 90.0)]


def tier_table(lon, lat, vals, out_csv, thr1=50.0, q2=95.0,
               block=HOT_BLOCK, min_blocks=HOT_MIN_BLOCKS):
    """Two-tier gradation of high improvement, written as ONE table:
      tier1 EXTREME    : improvement >= thr1 (absolute; expected arid-confined)
      tier2 HIGH(top5%): >= the q2-th percentile but < thr1 (expected widespread)
    For each tier: cell counts, 1-deg block counts, contiguous-region count and
    a latitude-band breakdown of the blocks (shows the spread across climates)."""
    v = np.asarray(vals); v = v[np.isfinite(v)]
    thr2 = float(np.percentile(v, q2))
    bm = _block_means(lon, lat, vals, block)
    hbk = bm.shape[0]
    blat = np.abs(90.0 - (np.arange(hbk) + 0.5) * block * RES)   # |lat| per block row
    tiers = [
        ("tier1_extreme", f">= {thr1:g}%", thr1, np.inf),
        ("tier2_high_top{:g}pct".format(100 - q2), f">= p{q2:g} ({thr2:.1f}%) & < {thr1:g}%",
         thr2, thr1),
    ]
    rows = []
    for name, desc, lo_t, hi_t in tiers:
        cells = (v >= lo_t) & (v < hi_t)
        bmask = np.isfinite(bm) & (bm >= lo_t) & (bm < hi_t)
        n_reg = sum(1 for c in _components(bmask) if len(c) >= min_blocks)
        band_n = []
        for _, b0, b1 in LAT_BANDS:
            rowsel = (blat >= b0) & (blat < b1)
            band_n.append(int(bmask[rowsel, :].sum()))
        rows.append([name, desc, int(cells.sum()),
                     f"{100.0 * cells.sum() / v.size:.2f}",
                     int(bmask.sum()), n_reg] + band_n)
        print(f"  {name}: {desc}  cells={cells.sum()} "
              f"({100.0 * cells.sum() / v.size:.2f}%)  blocks={bmask.sum()} "
              f"regions(>= {min_blocks}blk)={n_reg}  lat-bands={band_n}")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tier", "definition", "n_cells", "pct_cells",
                    "n_blocks_1deg", "n_contig_regions"]
                   + [b[0] for b in LAT_BANDS])
        w.writerows(rows)
    print(f"Saved: {out_csv}")


def draw_hotspots(ax, hotspots, margin=1.0):
    """Outline each hotspot bounding box (black rectangle + white-haloed Hk
    label) on top of the raster."""
    if not hotspots:
        return
    import matplotlib.patches as mpatches
    import matplotlib.patheffects as pe
    t = ccrs.PlateCarree()._as_mpl_transform(ax)
    for k, c in enumerate(hotspots, 1):
        lon0 = c["lon0"] - margin; lon1 = c["lon1"] + margin
        lat0 = c["lat0"] - margin; lat1 = c["lat1"] + margin
        ax.add_patch(mpatches.Rectangle(
            (lon0, lat0), lon1 - lon0, lat1 - lat0, transform=ccrs.PlateCarree(),
            fill=False, edgecolor="black", linewidth=2.2, zorder=6))
        ax.annotate(f"H{k}", xy=(lon0, lat1), xycoords=t, xytext=(2, 4),
                    textcoords="offset points", fontsize=20, fontweight="bold",
                    color="black", zorder=7,
                    path_effects=[pe.withStroke(linewidth=3.5, foreground="white")])


def _base_ax(fig):
    """Robinson axes; Antarctica (lat < LAT_CUT) cropped from the frame by
    trimming ONLY the southern edge in projected coords -- set_extent(...90)
    silently truncated the Arctic, so the full north (to the pole) is kept."""
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
    ax.set_global()
    y_cut = ccrs.Robinson().transform_point(0.0, LAT_CUT, ccrs.PlateCarree())[1]
    ax.set_ylim(bottom=y_cut)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    if BASINS is not None:
        ax.add_feature(BASINS)
    return ax


def plot_map(lon, lat, vals, title, cbar_label, filename, vmin, vmax,
             cmap="RdBu_r", hotspots=None):
    """Linear, robust-symmetric scale (vmin/vmax = +/- robust percentile)."""
    fig = plt.figure(figsize=(21, 12.6))   # 4200px @ dpi=200: >=1px per 0.1deg cell
    ax = _base_ax(fig)
    sc = _imshow(ax, lon, lat, np.clip(vals, vmin, vmax), cmap,
                 vmin=vmin, vmax=vmax)
    draw_hotspots(ax, hotspots)
    cbar = plt.colorbar(sc, ax=ax, orientation="horizontal",
                        pad=0.05, shrink=0.7, aspect=40, extend="both")
    cbar.set_label(cbar_label, fontsize=24)
    cbar.ax.tick_params(labelsize=20)
    ax.set_title(title, fontsize=26, pad=15)
    plt.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {filename}")


def symlog_map(lon, lat, vals, title, cbar_label, filename, cmap="RdBu_r",
               hotspots=None):
    """Symmetric-log scale: linear within +/-linthresh (so small improvements
    near 0 stay visible) and logarithmic in the tails (so the large arid-zone
    values are also shown without saturating). Handles the sign, unlike a plain
    log. A common, paper-friendly way to show both small and large values."""
    vals = np.asarray(vals)
    m = np.isfinite(vals)
    v = vals[m]
    vmax = np.percentile(np.abs(v), 99)
    nz = np.abs(v[v != 0])
    linthresh = max(np.percentile(nz, 50) if nz.size else vmax * 1e-2,
                    vmax * 1e-3)
    norm = SymLogNorm(linthresh=linthresh, vmin=-vmax, vmax=vmax)
    fig = plt.figure(figsize=(21, 12.6))   # 4200px @ dpi=200: >=1px per 0.1deg cell
    ax = _base_ax(fig)
    sc = _imshow(ax, lon[m], lat[m], np.clip(v, -vmax, vmax), cmap, norm=norm)
    draw_hotspots(ax, hotspots)
    cbar = plt.colorbar(sc, ax=ax, orientation="horizontal", pad=0.05,
                        shrink=0.7, aspect=40, extend="both")
    cbar.set_label(f"{cbar_label} (symlog)", fontsize=24)
    cbar.ax.tick_params(labelsize=18)
    ax.set_title(title, fontsize=26, pad=15)
    plt.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {filename}")


def quantile_map(lon, lat, vals, title, cbar_label, filename, n_bins=20,
                 hotspots=None):
    """Equal-count (quantile) binned version of plot_map, coloured by the custom
    5-hue x 4-shade map (semantic tab20b replacement). Bin edges sit at the
    quantiles of the improvement values, so every color covers the same number
    of cells (histogram equalisation). Hue advances red->purple as the value
    rises and shades go light->dark within each hue (see SEMANTIC_HUES)."""
    vals = np.asarray(vals)
    m = np.isfinite(vals)
    v = vals[m]
    edges = _q5x4_edges(v, n_bins)
    nb = len(edges) - 1
    if nb < 2:
        print(f"quantile_map: too few distinct values, skipped: {filename}")
        return
    cmap = ListedColormap(semantic_5x4_colors(nb))
    norm = BoundaryNorm(edges, nb)
    vc = np.clip(v, edges[0], edges[-1])

    fig = plt.figure(figsize=(21, 12.6))   # 4200px @ dpi=200: >=1px per 0.1deg cell
    ax = _base_ax(fig)
    sc = _imshow(ax, lon[m], lat[m], vc, cmap, norm=norm)
    draw_hotspots(ax, hotspots)
    ticks = edges[::4] if nb >= 8 else edges
    cbar = plt.colorbar(sc, ax=ax, orientation="horizontal", pad=0.05,
                        shrink=0.7, aspect=40, ticks=ticks, extend="both")
    cbar.ax.set_xticklabels([f"{e:.2g}" for e in ticks])
    cbar.set_label(f"{cbar_label} (equal-count bins)", fontsize=24)
    cbar.ax.tick_params(labelsize=18)
    ax.set_title(title, fontsize=26, pad=15)
    plt.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {filename}")


def main():
    if len(sys.argv) > 1:
        dat_dir = sys.argv[1]
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        dat_dir = os.path.join(script_dir, "..", "data")

    csv_path = os.path.join(dat_dir, "050", "summary_allgrid.csv")
    out_dir = os.path.join(dat_dir, "052")
    os.makedirs(out_dir, exist_ok=True)

    global BASINS
    BASINS = load_basin_feature(os.path.join(dat_dir, "shp", "basins"))

    lons, lats, st_pct, improve_lin, improve_quad, absd_lin, absd_quad = load_data(csv_path)
    corr_rows = []

    # The arid filter is defined on the RELATIVE values (|improvement| > CAP_PCT);
    # the same mask is reused for the absolute maps so both views show the same
    # cells. Robust +/- p98 color ranges are computed over the kept cells.
    rel_all  = np.concatenate([improve_lin, improve_quad])
    keep_all = np.isfinite(rel_all) & (np.abs(rel_all) <= CAP_PCT)
    vlim     = np.percentile(np.abs(rel_all[keep_all]), 98)

    cbl = "Improvement [%] (positive = {} better)"
    panels = [
        ("lin",  improve_lin,  absd_lin,  "Linear",    "SD improvement: Linear over Stationary"),
        ("quad", improve_quad, absd_quad, "Quadratic", "SD improvement: Quadratic over Stationary"),
    ]
    for key, vals, absvals, who, title in panels:
        keep = np.isfinite(vals) & (np.abs(vals) <= CAP_PCT)   # drop arid garbage
        lo, la, vv = lons[keep], lats[keep], vals[keep]
        av = absvals[keep]
        label = cbl.format(who.lower())
        # (0a) does the ABSOLUTE reduction map carry information beyond the
        # baseline-SD map? DeltaSD = improvement x SD_st, so if the relative
        # improvement is ~spatially uniform, corr(DeltaSD, SD_st) ~ 1 and the
        # absolute map is just the SD-level map re-coloured.
        st_k = st_pct[keep]
        pear = float(np.corrcoef(av, st_k)[0, 1])
        spear = _spearman(av, st_k)
        ba, bs = _block_pair(lo, la, av, st_k)
        spear_b = _spearman(ba, bs)
        print(f"[{who}] corr(DeltaSD, SD_st): Pearson={pear:.3f}  Spearman={spear:.3f}"
              f"  Spearman(1deg blocks)={spear_b:.3f}  (n={av.size}, blocks={ba.size})")
        corr_rows.append([key, av.size, f"{pear:.4f}", f"{spear:.4f}",
                          ba.size, f"{spear_b:.4f}"])
        # (0b) STANDALONE distribution figure (histogram + inverse CDF)
        make_improvement_dist(vv,
                              os.path.join(out_dir, f"improvement_dist_{key}.png"),
                              title)
        # (0c) STANDALONE histogram, bar colours = the _5x4 map's bins/colours
        make_improvement_hist(vv,
                              os.path.join(out_dir, f"improvement_hist_{key}.png"),
                              title)
        # two-tier gradation table: EXTREME (>=50%, arid-confined) vs HIGH
        # (top-5%, expected to spread across climates) -- the map boxes below
        # outline ONLY tier 1, the table quantifies both.
        print(f"[{who}] improvement tiers:")
        tier_table(lo, la, vv,
                   os.path.join(out_dir, f"improvement_tiers_{key}.csv"))
        # hotspots: few, strictly-thresholded contiguous regions of EXTREME
        # (tier-1, >=50%) improvement, outlined (H1..Hn) on every variant + CSV.
        print(f"[{who}] ", end="")
        hs = find_hotspots(lo, la, vv, abs_thr=50.0)
        with open(os.path.join(out_dir, f"hotspots_{key}.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "lon_min", "lon_max", "lat_min", "lat_max",
                        "mean_improvement_pct", "n_blocks_1deg"])
            for k, c in enumerate(hs, 1):
                w.writerow([f"H{k}", c["lon0"], c["lon1"], c["lat0"], c["lat1"],
                            f"{c['mean']:.2f}", c["n"]])
        # (1) linear, robust symmetric scale  -- the conventional choice
        plot_map(lo, la, vv, title, label,
                 os.path.join(out_dir, f"sd_improve_{key}.png"),
                 vmin=-vlim, vmax=vlim, cmap="RdBu_r", hotspots=hs)
        # (2) symlog -- small AND large improvements both visible
        symlog_map(lo, la, vv, title, label,
                   os.path.join(out_dir, f"sd_improve_{key}_symlog.png"), hotspots=hs)
        # (3) equal-count 5x4 -- histogram-equalised (unusual in papers)
        quantile_map(lo, la, vv, title, label,
                     os.path.join(out_dir, f"sd_improve_{key}_5x4.png"), hotspots=hs)
        # (4) ABSOLUTE SD reduction -- physical magnitude, no small-denominator
        #     blow-up; the honest paper-primary view (see module docstring).
        #     Drawn with the same equal-count 5-hue x 4-shade colormap as (3).
        abs_label = f"SD reduction [percentage points] (positive = {who.lower()} better)"
        quantile_map(lo, la, av, f"{title} (absolute)", abs_label,
                     os.path.join(out_dir, f"sd_improve_{key}_abs.png"), hotspots=hs)

    with open(os.path.join(out_dir, "spatial_corr_absSD_vs_stSD.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fit", "n_cells", "pearson_cell", "spearman_cell",
                    "n_blocks_1deg", "spearman_block_1deg"])
        w.writerows(corr_rows)
    print(f"Saved: {os.path.join(out_dir, 'spatial_corr_absSD_vs_stSD.csv')}")


if __name__ == "__main__":
    main()
