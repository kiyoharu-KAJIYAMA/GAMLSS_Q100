"""
lib_equivalent_ensemble_loader.py
Plot global maps of equivalent ensemble size results from 063.

Parent (truth) and nonstationary estimator are both selectable (matches 063):
  reads  063/eq_ens_allgrid_<parent>_<ns_model>.csv
  writes 066/<parent>/<ns_model>/

Two equivalent-ensemble DEFINITIONS (--def):
  old : SD point crossing (column eq_ens; as before; kept).
  sig : SD 95%-CI significance = largest e at which the stationary SD is NOT YET
        significantly below the ns SD (first significant size n has st_hi < ns_lo).
        Computed here ANALYTICALLY from the existing 063 columns WITHOUT changing
        060, using the pipeline's own SD ~ 1/sqrt(e) scaling and ~constant relative
        bootstrap-CI widths:
          st_sd(e) = ns_sd*sqrt(eq_ens/e)              (st_sd = ns_sd at e = eq_ens)
          k_st = (eq_sd_hi - eq_sd)/eq_sd  (stat upper);  k_ns = (ns_sd - ns_sd_lo)/ns_sd
          n = eq_ens * ((1+k_st)/(1-k_ns))**2 ;  eq_sig = n - step
        (approximation; the exact grid n needs the full stationary SD sweep, only
         available per-cell in 067.)

Colormap for the eq_ens map (--norm):
  quantile (default) : histogram-equalised (equal cell count per colour bin), so
                       the 0.1 fractional differences are visible even where many
                       cells pile up near one integer (e.g. 4). Continuous turbo,
                       NO rounding.
  linear             : plain linear turbo between 2/98 percentiles.
  clip               : percentile-clip. The colour range is capped at the
                       --clip-hi percentile (default 95) so the full colour
                       budget is spent on the bulk (~3-5) instead of the heavy
                       tail; every cell above the cap is drawn in ONE distinct
                       over-colour and the colourbar grows a ">=" arrow
                       (extend='max'). Likewise --clip-lo (default 2) on the low
                       end. This makes the 0.1 structure near the peak visible
                       while still flagging that extreme (e.g. ~10) cells exist.

Maps: eq_ens[/eq_ens_sig], ns_sd, eq_sd, q100_true. Robinson; Antarctica excluded.

Coastlines/borders are drawn OFFLINE from local Natural Earth shapefiles (as 120;
the compute server has no internet). Defaults to <script_dir>/ne/ne_110m_coastline.shp
and .../ne_110m_admin_0_boundary_lines_land.shp if present, else falls back to
cfeature (needs a download). Override with --coastline/--borders or --no-coastlines.

Usage:
  python3 lib_equivalent_ensemble_loader.py [dat_dir] [parent st|ln|qd] [ns_model lin|qd|ad]
                             [--def old|sig] [--norm quantile|linear|clip|boundary|q5x4]
                             [--clip-lo P] [--clip-hi P] [--cmap NAME]
                             [--nbins N] [--breaks percentile|linear|manual]
                             [--bin-edges e0,e1,...] [--width-px N] [--agg K]
                             [--coastline SHP] [--borders SHP] [--no-coastlines]

The eq_ens field is dominated by CELL-SCALE Monte-Carlo noise, so a per-cell map
reads as uniform speckle. --agg K block-averages K x K cells (noise ~1/sqrt(K^2))
to expose the underlying spatial signal; combine with an equal-count norm so the
now-smooth, narrow range still spans the full colour scale:
  --agg 5 --norm quantile                                # smooth + rank colour
  --agg 5 --norm boundary --breaks percentile --nbins 10 # smooth + discrete bins
  add --cmap Spectral_r / viridis for a non-rainbow ramp; --width-px 5200 for size

An ADDITIONAL eq_ens{_sig}_5x4.png is always written using the 052-style
5-hue x 4-shade equal-count (20-bin) colormap (norm q5x4), alongside whatever
--norm produced the primary eq_ens map.
"""
import os
import sys
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, Normalize
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io.shapereader import Reader
import importlib
excess_lib = importlib.import_module("common_target_cells")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RES = 0.1
NLON, NLAT = 3600, 1800     # 0.1deg global grid (lon x lat)
LAT_CUT = -60.0             # omit Antarctica band
DPI = 200                   # width_px = fig_width_in * DPI (as 120)
WIDTH_PX = 4200             # output figure width in px (set in main via --width-px)
AGG = 1                     # spatial block-mean factor (set in main via --agg)

# offline coastline/border shapefiles (server has no internet -> cartopy cannot
# download Natural Earth); set in main(). None -> fall back to cfeature.
COASTLINE_PATH = None
BORDERS_PATH = None
NO_COASTLINES = False


def draw_boundaries(ax):
    """Draw coastlines/borders offline-robustly. Prefers local Natural Earth
    shapefiles (as 120 does); falls back to cfeature only if no shapefile is set."""
    if NO_COASTLINES:
        return
    if COASTLINE_PATH and os.path.exists(COASTLINE_PATH):
        ax.add_geometries(Reader(COASTLINE_PATH).geometries(), ccrs.PlateCarree(),
                          facecolor="none", edgecolor="black", linewidth=0.5)
    else:
        try:
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        except Exception as e:
            print(f"NOTE: coastlines unavailable ({e}); pass --coastline ne_110m_coastline.shp")
    if BORDERS_PATH and os.path.exists(BORDERS_PATH):
        ax.add_geometries(Reader(BORDERS_PATH).geometries(), ccrs.PlateCarree(),
                          facecolor="none", edgecolor="grey", linewidth=0.3, linestyle=":")
    elif BORDERS_PATH is None:
        try:
            ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle=":")
        except Exception as e:
            print(f"NOTE: borders unavailable ({e}); pass --borders <shp> or --no-coastlines")
MODEL_LABEL = {"lin": "linear", "qd": "quadratic", "ad": "adaptive"}
PARENT_LABEL = {"st": "stationary", "ln": "linear", "qd": "quadratic"}
DEF_LABEL = {"old": "SD point crossing", "sig": "SD 95%-CI significance (n-step)"}


def safe_float(v):
    if v is None or v == "" or v == "NA" or v == "nan":
        return np.nan
    try:
        return float(v)
    except ValueError:
        return np.nan


COLS = ["eq_ens", "ns_sd", "ns_sd_lo", "eq_sd", "eq_sd_hi", "q100_true"]


def load_data(csv_path, bad_yx=frozenset()):
    """Load grid indices iy/ix + the columns needed for both definitions; exclude
    Antarctica and reverse-flow cells. iy/ix let us rasterise onto the 0.1deg grid."""
    iys, ixs = [], []
    cols = {c: [] for c in COLS}
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            iy = int(row["iy"]); ix = int(row["ix"])
            if (iy, ix) in bad_yx:
                continue
            lat = 90.0 - (iy + 0.5) * RES
            if lat < LAT_CUT:
                continue
            iys.append(iy); ixs.append(ix)
            for c in COLS:
                cols[c].append(safe_float(row.get(c)))
    print(f"Loaded {len(iys)} grid cells")
    if len(iys) == 0:
        sys.exit(f"No cells in {csv_path} -- run 064/060/063 first.")
    return (np.array(iys, dtype=int), np.array(ixs, dtype=int),
            {c: np.array(cols[c]) for c in COLS})


def block_nanmean(a, k):
    """Aggregate a (H,W) grid into (H//k, W//k) by averaging each k x k block,
    ignoring NaN (so coastal blocks still get their land-cell mean; all-NaN
    blocks -> NaN). Suppresses SD of per-cell Monte-Carlo noise by ~1/sqrt(count)."""
    if k <= 1:
        return a
    h, w = a.shape
    h2, w2 = h // k, w // k
    a = a[:h2 * k, :w2 * k].reshape(h2, k, w2, k)
    mask = np.isfinite(a)
    cnt = mask.sum(axis=(1, 3))
    s = np.where(mask, a, 0.0).sum(axis=(1, 3))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cnt > 0, s / cnt, np.nan)


def rasterize(iy, ix, vals, agg=1):
    """Place per-cell values on the regular 0.1deg grid (NaN = no data), optionally
    block-average into agg x agg blocks (spatial smoothing to reveal structure
    under the cell-scale MC noise), crop rows down to LAT_CUT, and return a masked
    array + PlateCarree extent for imshow."""
    arr = np.full((NLAT, NLON), np.nan, dtype=float)
    arr[iy, ix] = vals
    arr = block_nanmean(arr, agg)                     # (NLAT/agg, NLON/agg) if agg>1
    res = RES * max(agg, 1)
    n_keep = int(round((90.0 - LAT_CUT) / res))       # rows down to lat -60
    arr = arr[:n_keep, :]
    return np.ma.masked_invalid(arr), [-180.0, 180.0, LAT_CUT, 90.0]


def compute_eq_sig(d, step=0.1, e_max=10.0):
    """Significance-based equivalent ensemble from existing 063 columns (no 060
    change). n = eq_ens*((1+k_st)/(1-k_ns))^2 ; eq_sig = n - step.
      k_st = (eq_sd_hi - eq_sd)/eq_sd ;  k_ns = (ns_sd - ns_sd_lo)/ns_sd
    Assumes SD ~ 1/sqrt(e) and ~constant relative bootstrap-CI widths."""
    eq_ens, ns_sd, ns_sd_lo = d["eq_ens"], d["ns_sd"], d["ns_sd_lo"]
    eq_sd, eq_sd_hi = d["eq_sd"], d["eq_sd_hi"]
    with np.errstate(divide="ignore", invalid="ignore"):
        k_st = (eq_sd_hi - eq_sd) / eq_sd
        k_ns = (ns_sd - ns_sd_lo) / ns_sd
        n = eq_ens * ((1.0 + k_st) / (1.0 - k_ns)) ** 2
        eq_sig = n - step
    bad = (~np.isfinite(eq_sig)) | (~np.isfinite(eq_ens)) | (eq_ens > e_max) | (k_ns >= 1.0)
    return np.where(bad, np.nan, eq_sig)


# --- 5-hue x 4-shade semantic colormap (ported from fig05a_improvement_map) ---
SEMANTIC_HUES = ["#b2182b", "#e08214", "#4daf4a", "#2166ac", "#762a83"]


def _family_shades(base, n=4, toward_white=0.40, toward_black=0.38):
    """n shades of `base`, lightest first (small value) to darkest last (as 052)."""
    base = np.array(matplotlib.colors.to_rgb(base))
    lightest = base + (1.0 - base) * toward_white
    darkest = base * (1.0 - toward_black)
    return [tuple(lightest + (darkest - lightest) * (k / (n - 1))) for k in range(n)]


def semantic_5x4_colors(nb, hues=SEMANTIC_HUES, shades=4):
    """Ordered list of nb colors from the 5x4 (hue x shade) design (as 052):
    hue advances red->purple as the value rises, shades light->dark within hue."""
    full = []
    for h in hues:
        full.extend(_family_shades(h, shades))
    if nb == len(full):
        return full
    idx = np.linspace(0, len(full) - 1, nb).round().astype(int)
    return [full[i] for i in idx]


def _boundary_edges(vfin, breaks, nbins, clip_lo, clip_hi, bin_edges):
    """Bin edges for norm_mode='boundary'. 'manual' uses bin_edges verbatim;
    'percentile' gives equal-count (non-uniform: dense where the data is dense);
    'linear' gives uniform edges between the clip percentiles."""
    if breaks == "manual":
        if not bin_edges:
            sys.exit("--breaks manual requires --bin-edges e0,e1,...")
        edges = np.array(sorted(set(float(x) for x in bin_edges)), dtype=float)
    elif breaks == "linear":
        lo = np.percentile(vfin, clip_lo); hi = np.percentile(vfin, clip_hi)
        edges = np.linspace(lo, hi, nbins + 1)
    else:  # percentile (equal cell count per bin -> non-uniform, dense at the peak)
        edges = np.percentile(vfin, np.linspace(clip_lo, clip_hi, nbins + 1))
    return np.unique(np.round(edges, 3))


def plot_eq_ens_map(iy, ix, vals, title, cbar_label, filename,
                    norm_mode="quantile", cmap_name="turbo",
                    clip_lo=2.0, clip_hi=95.0,
                    nbins=8, breaks="percentile", bin_edges=None):
    """eq_ens map as a gap-free RASTER (imshow, nearest) on the 0.1deg grid --
    NOT a scatter -- so thin land is not dropped. Robinson, no Antarctica.
    norm_mode:
      quantile : histogram-equalised continuous colour (256 rank bins).
      clip     : continuous, colour range capped at clip_lo/clip_hi pct; tail ->
                 over-colour. (Turbo over a narrow clipped range looks mosaic-y
                 because cell-scale MC noise jumps hue -- prefer 'boundary'.)
      linear   : plain linear between 2/98 pct.
      boundary : DISCRETE non-uniform bins (method B). Collapses the rainbow
                 salt-and-pepper into a few flat colours; --breaks sets the edges
                 (percentile=equal count, linear=uniform, manual=--bin-edges) and
                 out-of-range cells get the over/under arrow colour."""
    # rasterise (+ optional spatial block-mean) FIRST, so the colour norm is
    # derived from what is actually displayed (aggregation narrows the range).
    arr, extent = rasterize(iy, ix, vals, agg=AGG)
    vfin = np.asarray(arr.compressed()) if np.ma.isMaskedArray(arr) else arr[np.isfinite(arr)]
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad(alpha=0.0)                       # no-data cells transparent
    extend = "neither"
    ticks = None                                  # boundary mode ticks = bin edges

    if norm_mode == "quantile":
        # equal-count colour bins: bin edges at data percentiles (histogram-equalised)
        edges = np.unique(np.percentile(vfin, np.linspace(0, 100, 257)))
        if edges.size < 3:                       # degenerate (few distinct values)
            norm = Normalize(vmin=float(vfin.min()), vmax=float(vfin.max()))
        else:
            norm = BoundaryNorm(edges, ncolors=cmap.N, clip=True)
        vmin, vmax = float(edges[0]), float(edges[-1])
    elif norm_mode == "boundary":
        # method B: discrete non-uniform bins -> few flat colours (no hue noise)
        edges = _boundary_edges(vfin, breaks, nbins, clip_lo, clip_hi, bin_edges)
        if edges.size < 2:
            sys.exit("boundary: need >=2 distinct edges (data too degenerate)")
        # sample nbins WELL-SEPARATED colours so adjacent bins are distinguishable
        n_int = edges.size - 1
        disc = matplotlib.colors.ListedColormap(cmap(np.linspace(0, 1, n_int)))
        disc.set_bad(alpha=0.0)
        norm = BoundaryNorm(edges, ncolors=disc.N, clip=False)
        vmin, vmax = float(edges[0]), float(edges[-1])
        dmin, dmax = float(vfin.min()), float(vfin.max())
        hi = vmax < dmax; lo = vmin > dmin
        extend = "both" if (hi and lo) else "max" if hi else "min" if lo else "neither"
        if hi: disc.set_over("#7a0177")          # magenta: eq_ens above top bin (heavy tail)
        if lo: disc.set_under("#08306b")         # dark blue: eq_ens below bottom bin
        cmap = disc
        ticks = list(edges)
    elif norm_mode == "q5x4":
        # 052-style: equal-count 20 bins coloured by the 5-hue x 4-shade design
        # (hue advances red->purple with the value, light->dark within each hue).
        lo, hi = np.percentile(vfin, [0.5, 99.5])
        core = vfin[(vfin >= lo) & (vfin <= hi)]
        if core.size < 2:
            core = vfin
        edges = np.unique(np.quantile(core, np.linspace(0.0, 1.0, 21)))
        n_int = edges.size - 1
        if n_int < 2:
            sys.exit("q5x4: too few distinct values")
        disc = matplotlib.colors.ListedColormap(semantic_5x4_colors(n_int))
        disc.set_bad(alpha=0.0)
        norm = BoundaryNorm(edges, ncolors=disc.N, clip=True)
        cmap = disc
        vmin, vmax = float(edges[0]), float(edges[-1])
        extend = "both"
        ticks = list(edges[::4]) if n_int >= 8 else list(edges)
    elif norm_mode == "clip":
        # percentile-clip: full colour budget on the bulk, heavy tail -> over-colour
        vmin = float(np.percentile(vfin, clip_lo)); vmax = float(np.percentile(vfin, clip_hi))
        norm = Normalize(vmin=vmin, vmax=vmax, clip=False)
        dmin, dmax = float(vfin.min()), float(vfin.max())
        hi = vmax < dmax; lo = vmin > dmin
        extend = "both" if (hi and lo) else "max" if hi else "min" if lo else "neither"
        # distinct out-of-range colours so extreme cells are identifiable, not just clamped
        if hi:
            cmap.set_over("#7a0177")             # magenta: eq_ens >= clip-hi (heavy tail)
        if lo:
            cmap.set_under("#08306b")            # dark blue: eq_ens <= clip-lo
    else:
        vmin = float(np.percentile(vfin, 2)); vmax = float(np.percentile(vfin, 98))
        norm = Normalize(vmin=vmin, vmax=vmax)

    w_in = WIDTH_PX / DPI                          # high-res raster (as 115/120)
    fig = plt.figure(figsize=(w_in, w_in * 0.60))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
    ax.set_global()
    ax.set_extent([-179.9, 179.9, LAT_CUT + 2.0, 84.0], crs=ccrs.PlateCarree())
    draw_boundaries(ax)
    im = ax.imshow(arr, origin="upper", extent=extent, transform=ccrs.PlateCarree(),
                   cmap=cmap, norm=norm, interpolation="nearest", regrid_shape=NLON,
                   zorder=0)

    # colourbar ticks: bin edges (boundary/q5x4) or nice integers spanning the range
    if ticks is None:
        t0 = int(np.floor(vmin)); t1 = int(np.ceil(vmax))
        ticks = [t for t in range(t0, t1 + 1)]
    # q5x4 shows each equal-count bin at equal width on the bar (as 052);
    # the other modes keep value-proportional spacing.
    spacing = "uniform" if norm_mode == "q5x4" else "proportional"
    cbar = plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.05,
                        shrink=0.7, aspect=40, ticks=ticks, spacing=spacing,
                        extend=extend)
    if norm_mode == "boundary":
        cbar.ax.set_xticklabels([f"{t:g}" for t in ticks])
    elif norm_mode == "q5x4":
        cbar.ax.set_xticklabels([f"{t:.2g}" for t in ticks])
    cbar.set_label(cbar_label, fontsize=24)
    cbar.ax.tick_params(labelsize=20)
    if norm_mode == "quantile":
        cbar.ax.set_xlabel("(colour spaced by rank / equal cell count per bin)",
                           fontsize=13)
    elif norm_mode == "q5x4":
        cbar.ax.set_xlabel("(equal-count bins; 5 hues x 4 shades, low->high = red->purple)",
                           fontsize=13)
    elif norm_mode == "boundary":
        cbar.ax.set_xlabel(f"(discrete bins, breaks={breaks}, {clip_lo:g}-{clip_hi:g} pct;"
                           f" out-of-range cells in the arrow colour)", fontsize=13)
    elif norm_mode == "clip":
        cbar.ax.set_xlabel(f"(clipped at {clip_lo:g}-{clip_hi:g} pct;"
                           f" out-of-range cells in the arrow colour)", fontsize=13)
    ax.set_title(title, fontsize=26, pad=15)
    plt.savefig(filename, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved: {filename}  ({WIDTH_PX}px wide)")


def plot_map(iy, ix, vals, title, cbar_label, filename, cmap="viridis", log_scale=False):
    plot_vals = vals.copy()
    if log_scale:
        plot_vals = np.where(plot_vals > 0, np.log10(plot_vals), np.nan)
        cbar_label = f"log10({cbar_label})"
    arr, extent = rasterize(iy, ix, plot_vals, agg=AGG)
    valid = arr.compressed() if np.ma.isMaskedArray(arr) else arr[np.isfinite(arr)]
    vmin = float(np.percentile(valid, 2)); vmax = float(np.percentile(valid, 98))
    cmap = plt.get_cmap(cmap).copy()
    cmap.set_bad(alpha=0.0)                       # no-data cells transparent
    norm = Normalize(vmin=vmin, vmax=vmax)
    w_in = WIDTH_PX / DPI                          # high-res raster (as 115/120)
    fig = plt.figure(figsize=(w_in, w_in * 0.60))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
    ax.set_global()
    ax.set_extent([-179.9, 179.9, LAT_CUT + 2.0, 84.0], crs=ccrs.PlateCarree())
    draw_boundaries(ax)
    im = ax.imshow(arr, origin="upper", extent=extent, transform=ccrs.PlateCarree(),
                   cmap=cmap, norm=norm, interpolation="nearest", regrid_shape=NLON,
                   zorder=0)
    cbar = plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.05, shrink=0.7, aspect=40)
    cbar.set_label(cbar_label, fontsize=24)
    cbar.ax.tick_params(labelsize=20)
    ax.set_title(title, fontsize=26, pad=15)
    plt.savefig(filename, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved: {filename}  ({WIDTH_PX}px wide)")


def main():
    global COASTLINE_PATH, BORDERS_PATH, NO_COASTLINES, WIDTH_PX, AGG
    argv = sys.argv[1:]
    eq_def = "old"; norm_mode = "quantile"; clip_lo = 2.0; clip_hi = 95.0
    coastline = None; borders = None
    nbins = 8; breaks = "percentile"; bin_edges = None; cmap_name = "turbo"
    if "--def" in argv:
        i = argv.index("--def"); eq_def = argv[i + 1]; del argv[i:i + 2]
    if "--norm" in argv:
        i = argv.index("--norm"); norm_mode = argv[i + 1]; del argv[i:i + 2]
    if "--clip-lo" in argv:
        i = argv.index("--clip-lo"); clip_lo = float(argv[i + 1]); del argv[i:i + 2]
    if "--clip-hi" in argv:
        i = argv.index("--clip-hi"); clip_hi = float(argv[i + 1]); del argv[i:i + 2]
    if "--nbins" in argv:
        i = argv.index("--nbins"); nbins = int(argv[i + 1]); del argv[i:i + 2]
    if "--breaks" in argv:
        i = argv.index("--breaks"); breaks = argv[i + 1]; del argv[i:i + 2]
    if "--bin-edges" in argv:
        i = argv.index("--bin-edges")
        bin_edges = [float(x) for x in argv[i + 1].split(",")]; del argv[i:i + 2]
    if "--cmap" in argv:
        i = argv.index("--cmap"); cmap_name = argv[i + 1]; del argv[i:i + 2]
    if "--width-px" in argv:
        i = argv.index("--width-px"); WIDTH_PX = int(argv[i + 1]); del argv[i:i + 2]
    if "--agg" in argv:
        i = argv.index("--agg"); AGG = int(argv[i + 1]); del argv[i:i + 2]
    if "--coastline" in argv:
        i = argv.index("--coastline"); coastline = argv[i + 1]; del argv[i:i + 2]
    if "--borders" in argv:
        i = argv.index("--borders"); borders = argv[i + 1]; del argv[i:i + 2]
    if "--no-coastlines" in argv:
        NO_COASTLINES = True; argv.remove("--no-coastlines")
    if eq_def not in DEF_LABEL:
        sys.exit("--def must be old or sig")
    if norm_mode not in ("quantile", "linear", "clip", "boundary", "q5x4"):
        sys.exit("--norm must be quantile, linear, clip, boundary or q5x4")
    if breaks not in ("percentile", "linear", "manual"):
        sys.exit("--breaks must be percentile, linear or manual")

    # offline coastline/border shapefiles: default to <script_dir>/ne/... if present
    if coastline is None:
        cand = os.path.join(SCRIPT_DIR, "ne", "ne_110m_coastline.shp")
        if os.path.exists(cand): coastline = cand; print(f"using --coastline: {cand}")
    if borders is None:
        cand = os.path.join(SCRIPT_DIR, "ne", "ne_110m_admin_0_boundary_lines_land.shp")
        if os.path.exists(cand): borders = cand; print(f"using --borders: {cand}")
    COASTLINE_PATH = coastline; BORDERS_PATH = borders

    dat_dir = argv[0] if len(argv) > 0 else \
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
    parent = argv[1] if len(argv) > 1 else "ln"
    ns_model = argv[2] if len(argv) > 2 else "lin"
    if parent not in PARENT_LABEL:
        sys.exit("parent must be one of st, ln, qd")
    if ns_model not in MODEL_LABEL:
        sys.exit("ns_model must be one of lin, qd, ad")
    label = MODEL_LABEL[ns_model]; plabel = PARENT_LABEL[parent]

    csv_path = os.path.join(dat_dir, "063", f"eq_ens_allgrid_{parent}_{ns_model}.csv")
    out_dir = os.path.join(dat_dir, "066", parent, ns_model)
    os.makedirs(out_dir, exist_ok=True)

    iys, ixs, d = load_data(csv_path, excess_lib.load_bad_yx(dat_dir))
    q100_true = d["q100_true"]; ns_sd = d["ns_sd"]; eq_sd = d["eq_sd"]

    # eq values per the chosen definition
    if eq_def == "old":
        eq_vals = d["eq_ens"]
    else:
        eq_vals = compute_eq_sig(d)
        n_ok = int(np.isfinite(eq_vals).sum())
        print(f"def=sig: computed eq_ens_sig for {n_ok}/{len(eq_vals)} cells "
              f"(median={np.nanmedian(eq_vals):.2f})")

    # 1. Equivalent ensemble size (continuous, histogram-equalised colour)
    suffix = "" if eq_def == "old" else "_sig"
    plot_eq_ens_map(
        iys, ixs, eq_vals,
        title=f"Equivalent ensemble size [{DEF_LABEL[eq_def]}]\n({label} over stationary, {plabel} parent)",
        cbar_label="equivalent ensemble size",
        filename=os.path.join(out_dir, f"eq_ens{suffix}.png"),
        norm_mode=norm_mode, cmap_name=cmap_name, clip_lo=clip_lo, clip_hi=clip_hi,
        nbins=nbins, breaks=breaks, bin_edges=bin_edges,
    )
    # 1b. ADDITIONAL figure: 052-style 5-hue x 4-shade equal-count map (always)
    if norm_mode != "q5x4":
        plot_eq_ens_map(
            iys, ixs, eq_vals,
            title=f"Equivalent ensemble size [{DEF_LABEL[eq_def]}]\n({label} over stationary, {plabel} parent)",
            cbar_label="equivalent ensemble size",
            filename=os.path.join(out_dir, f"eq_ens{suffix}_5x4.png"),
            norm_mode="q5x4",
        )

    # 2. Nonstationary SD (e=3 fixed)
    plot_map(iys, ixs, ns_sd,
             title=f"Nonstationary SD ({label}, e=3; {plabel} parent)",
             cbar_label="SD of relative error",
             filename=os.path.join(out_dir, "ns_sd.png"), cmap="YlOrRd")
    # 3. Stationary SD at equivalent ensemble size
    plot_map(iys, ixs, eq_sd,
             title="Stationary SD at equivalent ensemble size",
             cbar_label="SD of relative error",
             filename=os.path.join(out_dir, "eq_sd.png"), cmap="YlOrRd")
    # 4. True Q100 (log scale)
    plot_map(iys, ixs, q100_true,
             title="True Q100 (1981-2100 mean)",
             cbar_label="Q100 [m³/s]",
             filename=os.path.join(out_dir, "q100_true.png"), cmap="YlGnBu", log_scale=True)


if __name__ == "__main__":
    main()
