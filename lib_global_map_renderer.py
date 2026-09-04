"""
lib_global_map_renderer.py
Plot global maps on Robinson projection. Antarctica (lat < -60) is cropped from
the frame; river-basin outlines (dashed, HydroBASINS lev03) replace country
borders; all statistics are shown in PERCENT (x100 of the relative-error stat).
Non-physical arid cells (stat or |diff| > CAP_PCT, from truth_q100 ~ 0 blowing
the relative-error up to ~1e137) are dropped before plotting.

Maps:
  1) 9 individual maps: 3 models x 3 statistics (SD, IQR, Tail), positive
  2) 6 difference maps: Stationary - {Linear, Quadratic} for SD/IQR/Tail
     (positive = nonstationary is better), signed

Each map is written in THREE scale variants for comparison:
  - <name>.png         linear, robust percentile range (the conventional choice)
  - <name>_log.png     LogNorm   (individual maps; positive, spans magnitudes)
  - <name>_symlog.png  SymLogNorm (difference maps; signed, small+large visible)
  - <name>_5x4.png     equal-count bins, custom 5-hue x 4-shade colormap

Input:
  ../data/050/summary_allgrid.csv

Output: ../data/051/   (e.g. stat_sd.png / stat_sd_log.png / stat_sd_5x4.png,
  diff_stat_lin_sd.png / diff_stat_lin_sd_symlog.png / diff_stat_lin_sd_5x4.png)

Usage:
  python3 lib_global_map_renderer.py
"""
import os
import glob
import numpy as np
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm, SymLogNorm, LogNorm
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Cells whose statistic exceeds this (%) are non-physical (arid cells with
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
# Unlike tab20b, the 5 hue families are ORDERED so that small values read as
# "low" and large values as "high" (red -> orange -> green -> blue -> purple),
# and within each family the 4 shades go LIGHT (small) -> DARK (large), i.e.
# reversed relative to tab20b (which is dark -> light within a family). Yellow is
# dropped and the hues are spread around the wheel so adjacent families stay
# distinguishable; light shades keep enough saturation to keep their hue.
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
DAT_ROOT = os.path.join(SCRIPT_DIR, "..", "data")
INPUT_CSV = os.path.join(DAT_ROOT, "050", "summary_allgrid.csv")
OUTPUT_DIR = os.path.join(DAT_ROOT, "051")

# Grid resolution (0.1 degree)
RES = 0.1
NLON, NLAT = 3600, 1800     # 0.1deg global grid (lon x lat)
LAT_CUT = -60.0             # omit Antarctica band


def load_data():
    """Load CSV and convert iy/ix to lon/lat. Exclude Antarctica."""
    lons, lats = [], []
    values = {k: [] for k in [
        "stat_err_sd", "stat_err_iqr", "stat_err_tail",
        "lin_err_sd", "lin_err_iqr", "lin_err_tail",
        "quad_err_sd", "quad_err_iqr", "quad_err_tail",
    ]}

    with open(INPUT_CSV, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            iy = int(row["iy"])
            ix = int(row["ix"])
            lat = 90.0 - (iy + 0.5) * RES
            lon = -180.0 + (ix + 0.5) * RES

            # Exclude Antarctica
            if lat < -60:
                continue

            # Skip rows with missing values
            skip = False
            for k in values:
                v = row[k]
                if v == "" or v == "NA" or v == "nan":
                    skip = True
                    break
            if skip:
                continue

            lons.append(lon)
            lats.append(lat)
            for k in values:
                values[k].append(float(row[k]))

    lons = np.array(lons)
    lats = np.array(lats)
    # Stats are SDs/IQRs/tails of the *relative* Q100 error (dimensionless);
    # x100 -> percent, which reads more naturally on the colorbar.
    for k in values:
        values[k] = np.array(values[k]) * 100.0

    print(f"Loaded {len(lons)} grid cells (excluding Antarctica)")
    return lons, lats, values


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


def plot_map(lons, lats, vals, title, cbar_label, filename, vmin=None, vmax=None, cmap="viridis"):
    """Linear scale (the conventional choice). Values are clipped to vmin/vmax."""
    fig = plt.figure(figsize=(21, 12.6))   # 4200px @ dpi=200: >=1px per 0.1deg cell
    ax = _base_ax(fig)
    sc = _imshow(ax, lons, lats, np.clip(vals, vmin, vmax), cmap,
                 vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(sc, ax=ax, orientation="horizontal", pad=0.05,
                        shrink=0.7, aspect=40, extend="both")
    cbar.set_label(cbar_label, fontsize=24)
    cbar.ax.tick_params(labelsize=20)
    ax.set_title(title, fontsize=28, pad=15)
    plt.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {filename}")


def log_map(lons, lats, vals, title, cbar_label, filename, cmap="YlOrRd"):
    """Logarithmic scale for a POSITIVE statistic (SD/IQR/tail) that spans
    orders of magnitude, so both small and large values are resolved."""
    vals = np.asarray(vals)
    m = np.isfinite(vals) & (vals > 0)
    v = vals[m]
    vmin, vmax = np.percentile(v, [2, 98])
    vmin = max(vmin, vmax * 1e-3)
    norm = LogNorm(vmin=vmin, vmax=vmax)
    fig = plt.figure(figsize=(21, 12.6))   # 4200px @ dpi=200: >=1px per 0.1deg cell
    ax = _base_ax(fig)
    sc = _imshow(ax, lons[m], lats[m], np.clip(v, vmin, vmax), cmap, norm=norm)
    cbar = plt.colorbar(sc, ax=ax, orientation="horizontal", pad=0.05,
                        shrink=0.7, aspect=40, extend="both")
    cbar.set_label(f"{cbar_label} (log)", fontsize=24)
    cbar.ax.tick_params(labelsize=20)
    ax.set_title(title, fontsize=28, pad=15)
    plt.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {filename}")


def symlog_map(lons, lats, vals, title, cbar_label, filename, cmap="RdBu_r"):
    """Symmetric-log scale for a SIGNED quantity (difference maps): linear within
    +/-linthresh so small values near 0 stay visible, logarithmic in the tails so
    large arid-zone values also show without saturating the scale."""
    vals = np.asarray(vals)
    m = np.isfinite(vals)
    v = vals[m]
    vmax = np.percentile(np.abs(v), 99)
    nz = np.abs(v[v != 0])
    linthresh = max(np.percentile(nz, 50) if nz.size else vmax * 1e-2, vmax * 1e-3)
    norm = SymLogNorm(linthresh=linthresh, vmin=-vmax, vmax=vmax)
    fig = plt.figure(figsize=(21, 12.6))   # 4200px @ dpi=200: >=1px per 0.1deg cell
    ax = _base_ax(fig)
    sc = _imshow(ax, lons[m], lats[m], np.clip(v, -vmax, vmax), cmap, norm=norm)
    cbar = plt.colorbar(sc, ax=ax, orientation="horizontal", pad=0.05,
                        shrink=0.7, aspect=40, extend="both")
    cbar.set_label(f"{cbar_label} (symlog)", fontsize=24)
    cbar.ax.tick_params(labelsize=20)
    ax.set_title(title, fontsize=28, pad=15)
    plt.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {filename}")


def quantile_map(lons, lats, vals, title, cbar_label, filename, n_bins=20):
    """Equal-count (quantile) binned version of plot_map, coloured by the custom
    5-hue x 4-shade map (semantic tab20b replacement). Bin edges sit at the
    quantiles of the plotted values, so every color covers the same number of
    cells (histogram equalisation), which exposes spatial structure the linear
    YlOrRd/RdBu_r scales compress. Hue advances red->purple as the value rises
    and shades go light->dark within each hue (see SEMANTIC_HUES)."""
    vals = np.asarray(vals)
    m = np.isfinite(vals)
    v = vals[m]
    # Robust edges: arid cells (truth_q100 ~ 0) can blow the relative-error SD up
    # to ~1e137, which would dominate the min/max bins and wreck the colorbar.
    # Build the equal-count edges over the inner 0.5-99.5 percentile range and
    # clamp out-of-range cells into the end bins (colorbar shows extend caps).
    lo, hi = np.percentile(v, [0.5, 99.5])
    core = v[(v >= lo) & (v <= hi)]
    if core.size < 2:
        core = v
    edges = np.unique(np.quantile(core, np.linspace(0.0, 1.0, n_bins + 1)))
    nb = len(edges) - 1
    if nb < 2:
        print(f"quantile_map: too few distinct values, skipped: {filename}")
        return
    cmap = ListedColormap(semantic_5x4_colors(nb))
    norm = BoundaryNorm(edges, nb)
    vc = np.clip(v, edges[0], edges[-1])

    fig = plt.figure(figsize=(21, 12.6))   # 4200px @ dpi=200: >=1px per 0.1deg cell
    ax = _base_ax(fig)
    sc = _imshow(ax, lons[m], lats[m], vc, cmap, norm=norm)
    ticks = edges[::4] if nb >= 8 else edges
    cbar = plt.colorbar(sc, ax=ax, orientation="horizontal", pad=0.05,
                        shrink=0.7, aspect=40, ticks=ticks, extend="both")
    cbar.ax.set_xticklabels([f"{e:.2g}" for e in ticks])
    cbar.set_label(f"{cbar_label} (equal-count bins)", fontsize=24)
    cbar.ax.tick_params(labelsize=18)
    ax.set_title(title, fontsize=28, pad=15)
    plt.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {filename}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    global BASINS
    BASINS = load_basin_feature(os.path.join(DAT_ROOT, "shp", "basins"))
    lons, lats, values = load_data()

    # Define plot configurations
    models = [
        ("stat", "Stationary"),
        ("lin",  "Linear nonstationary"),
        ("quad", "Quadratic nonstationary"),
    ]
    stats = [
        ("sd",   "SD",          "Standard deviation of relative error"),
        ("iqr",  "IQR",         "Interquartile range of relative error"),
        ("tail", "Tail spread", "Tail spread (Q95-Q05) of relative error"),
    ]
    cmaps = {
        "sd":   "YlOrRd",
        "iqr":  "YlOrRd",
        "tail": "YlOrRd",
    }

    # Common vmin/vmax per statistic (across 3 models), robust + garbage-free
    vranges = {}
    for stat_key, _, _ in stats:
        all_vals = np.concatenate([values[f"{m}_err_{stat_key}"] for m, _ in models])
        valid = all_vals[np.isfinite(all_vals) & (all_vals > 0) & (all_vals <= CAP_PCT)]
        vranges[stat_key] = (np.percentile(valid, 2), np.percentile(valid, 98))

    # 9 individual maps (absolute SD/IQR/tail). Three scale variants each:
    #   <name>.png (linear), _log.png (log), _5x4.png (equal-count)
    for model_key, model_name in models:
        for stat_key, stat_short, stat_desc in stats:
            v = values[f"{model_key}_err_{stat_key}"]
            keep = np.isfinite(v) & (v > 0) & (v <= CAP_PCT)   # drop arid garbage
            lo, la, vv = lons[keep], lats[keep], v[keep]
            vmin, vmax = vranges[stat_key]
            title = f"{model_name} — {stat_desc}"
            label = f"{stat_short} [%]"
            base = os.path.join(OUTPUT_DIR, f"{model_key}_{stat_key}")
            plot_map(lo, la, vv, title, label, f"{base}.png",
                     vmin=vmin, vmax=vmax, cmap=cmaps[stat_key])
            log_map(lo, la, vv, title, label, f"{base}_log.png", cmap=cmaps[stat_key])
            quantile_map(lo, la, vv, title, label, f"{base}_5x4.png")

    # 6 difference maps (Stationary - Nonstationary; positive = nonstationary
    # better). Three scale variants each: linear / symlog / 5x4.
    diff_pairs = [("lin", "Linear"), ("quad", "Quadratic")]
    for ns_key, ns_name in diff_pairs:
        for stat_key, stat_short, stat_desc in stats:
            diff = values[f"stat_err_{stat_key}"] - values[f"{ns_key}_err_{stat_key}"]
            keep = np.isfinite(diff) & (np.abs(diff) <= CAP_PCT)   # drop arid garbage
            lo, la, dd = lons[keep], lats[keep], diff[keep]
            vlim = np.percentile(np.abs(dd), 98)
            title = f"Stationary - {ns_name} ({stat_short})\n(positive = nonstationary better)"
            # difference of two %-quantities -> PERCENTAGE POINTS, not %
            label = f"\u0394{stat_short} [% points]"
            base = os.path.join(OUTPUT_DIR, f"diff_stat_{ns_key}_{stat_key}")
            plot_map(lo, la, dd, title, label, f"{base}.png",
                     vmin=-vlim, vmax=vlim, cmap="RdBu_r")
            # (symlog variant dropped: log scaling not needed for the diff maps)
            quantile_map(lo, la, dd, title, label, f"{base}_5x4.png")


if __name__ == "__main__":
    main()
