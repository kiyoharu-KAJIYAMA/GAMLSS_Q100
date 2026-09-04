"""
fig02a_change_type_global_map.py  (LOCAL renderer -- run where the internet /
matplotlib+cartopy are available; the compute server is offline and has no terra)

Draw the whole-globe GU 4-class map for Figure 2 in Robinson projection as a
gap-free RASTER (imshow, nearest-neighbour, high dpi) so thin land (e.g. the Noto
peninsula, Ishikawa) is NOT dropped by sub-pixel marker aliasing. Reads the
whole-globe classification produced on the server by step16_combine_aic_change_type.sh
(077b/aic_allgrid.csv: cell_id, iy, ix, ..., class) -- copy just that CSV locally.

The 0.1deg grid is 3600 x 1800 (lon x lat); the figure is rendered wide enough
that each cell spans >= 1 output pixel, and interpolation='nearest' avoids
mixing class colours with the transparent ocean.

Usage:
  python3 fig02a_change_type_global_map.py AIC_ALLGRID_CSV [-o OUT.png]
                                       [--bad bad_cells.csv] [--width-px 4200]
Requires: numpy, pandas, matplotlib, cartopy
  (local install, one-off:  pip install numpy pandas matplotlib cartopy)
"""
import argparse, sys, os
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# renamed GU classes + Okabe-Ito colours (identical to fig02b_example_cell_panels.R)
CLASS_ORDER = ["stationary", "location", "scale", "location and scale"]
RENAME = {"stationary": "stationary", "mean": "location", "var": "scale", "both": "location and scale"}
CLASS_COL = {"stationary": "#BBBBBB", "location": "#0072B2",
             "scale": "#D55E00", "location and scale": "#009E73"}

# the eight example cells b-i (keep in sync with 115 EX): label, lon, lat  [cell_id]
EX = [("b", -97.50, 32.40),   # 901988  stationary/low  Texas
      ("c", 27.35, -3.05),    # 1283512 location/low    DR Congo
      ("d", -57.55, -36.05),  # 1538746 scale/low       Buenos Aires, AR (Samborombon)
      ("e", 176.75, -39.45),  # 1547062 loc+scale/low   Hawke's Bay, NZ (Tutaekuri)
      ("f", 44.15, -18.05),   # 1406770 stationary/high Madagascar
      ("g", 79.40, 39.80),    # 788582  location/high   Xinjiang (Yarkand)
      ("h", 115.10, 33.20),   # 891235  scale/high      Henan (Ying)
      ("i", -66.80, 52.10)]   # 549028  loc+scale/high  Quebec (Mouchalagane)

NLON, NLAT, RES = 3600, 1800, 0.1
LAT_CUT = -60.0  # omit Antarctica band


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?", default=None,
                    help="077b/aic_allgrid.csv (default: <script_dir>/../data/077b/aic_allgrid.csv)")
    ap.add_argument("-o", "--out", default=None,
                    help="output PNG (default: <dat_dir>/120/global_class_map.png, "
                         "where dat_dir is inferred from the csv path)")
    ap.add_argument("--bad", default=None,
                    help="reverse-flow cells to drop (default: <dat_dir>/010/bad_cells.csv if present)")
    ap.add_argument("--probe", action="append", default=None, metavar="lon0,lon1,lat0,lat1",
                    help="report how many classified cells fall in this bbox and their class "
                         "breakdown (repeatable). Use to check data presence, e.g. Ishikawa: "
                         "--probe 136.6,137.5,36.7,37.6")
    ap.add_argument("--width-px", type=int, default=4200, help="figure width in pixels (>=3600 keeps thin land)")
    ap.add_argument("--coastline", default=None,
                    help="Natural Earth coastline .shp for offline drawing (default: "
                         "<script_dir>/ne/ne_110m_coastline.shp if present, else cartopy)")
    ap.add_argument("--no-coastlines", action="store_true", help="skip coastlines entirely")
    ap.add_argument("--flood-ensured", action="store_true",
                    help="restrict to flood-ensured cells (common_target_cells.load_flood_ensured_ids); "
                         "outputs get a _ensured suffix")
    args = ap.parse_args()

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patheffects as pe
        from matplotlib.colors import ListedColormap, BoundaryNorm
        import cartopy.crs as ccrs
    except ImportError as e:
        sys.exit(f"Missing plotting package ({e}).\n"
                 f"Install locally:  pip install numpy pandas matplotlib cartopy")

    # default csv: <script_dir>/../data/077b/aic_allgrid.csv
    if args.csv is None:
        args.csv = os.path.join(SCRIPT_DIR, "..", "data", "077b", "aic_allgrid.csv")
    if not os.path.exists(args.csv):
        sys.exit(f"csv not found: {args.csv}\n  (run step16_combine_aic_change_type.sh first, or pass the path explicitly)")

    # dat_dir inferred from the csv path (.../<dat_dir>/077b/aic_allgrid.csv)
    dat_dir = os.path.dirname(os.path.dirname(os.path.abspath(args.csv)))
    suffix = "_ensured" if args.flood_ensured else ""

    # default output: <dat_dir>/120/global_class_map[_ensured].png
    if args.out is None:
        out_dir = os.path.join(dat_dir, "120"); os.makedirs(out_dir, exist_ok=True)
        args.out = os.path.join(out_dir, f"global_class_map{suffix}.png")
    else:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)

    # default --bad: <dat_dir>/010/bad_cells.csv (if present)
    if args.bad is None:
        cand = os.path.join(dat_dir, "010", "bad_cells.csv")
        if os.path.exists(cand): args.bad = cand; print(f"using --bad: {cand}")
    # default --coastline: <script_dir>/ne/ne_110m_coastline.shp (if present)
    if args.coastline is None:
        cand = os.path.join(SCRIPT_DIR, "ne", "ne_110m_coastline.shp")
        if os.path.exists(cand): args.coastline = cand; print(f"using --coastline: {cand}")

    df = pd.read_csv(args.csv, usecols=["cell_id", "iy", "ix", "class"])
    df = df[df["class"].isin(RENAME)]
    if args.bad:
        bad = set(pd.read_csv(args.bad)["cell_id"].tolist())
        df = df[~df["cell_id"].isin(bad)]
    if args.flood_ensured:
        sys.path.insert(0, SCRIPT_DIR)
        import importlib
        ex = importlib.import_module("common_target_cells")
        keep = ex.load_flood_ensured_ids(dat_dir)
        if not keep:
            sys.exit("flood-ensured id set is empty (common_target_cells.load_flood_ensured_ids)")
        n_before = len(df)
        df = df[df["cell_id"].isin(keep)]
        print(f"flood-ensured filter: {len(df):,} / {n_before:,} classified cells kept")
    df = df.assign(lon=-180.0 + (df["ix"] + 0.5) * RES, lat=90.0 - (df["iy"] + 0.5) * RES)

    # --probe: report data presence in a bbox (e.g. Ishikawa / Noto peninsula)
    if args.probe:
        for spec in args.probe:
            lon0, lon1, lat0, lat1 = (float(x) for x in spec.split(","))
            sub = df[(df["lon"] >= lon0) & (df["lon"] <= lon1) &
                     (df["lat"] >= lat0) & (df["lat"] <= lat1)]
            print(f"\n[probe] bbox lon[{lon0},{lon1}] lat[{lat0},{lat1}]: {len(sub)} classified cells")
            if len(sub):
                vc = sub["class"].map(RENAME).value_counts()
                for k in CLASS_ORDER:
                    if k in vc: print(f"    {k:>20}: {int(vc[k])}")
                print(sub[["cell_id", "lon", "lat", "class"]].head(12).to_string(index=False))
            else:
                print("    -> NO cell here in the classification (likely max(AMAX)<=1 m3/s "
                      "so 077b skipped it, or not a land_cell). This is a DATA gap, not a "
                      "rendering gap.")
        print()

    code = df["class"].map(RENAME).map({c: i for i, c in enumerate(CLASS_ORDER)})

    # rasterise onto the regular 0.1deg grid; NaN = no data (transparent)
    arr = np.full((NLAT, NLON), np.nan, dtype=float)
    arr[df["iy"].to_numpy(), df["ix"].to_numpy()] = code.to_numpy()

    # crop rows to lat in [LAT_CUT, 90]  (row lat = 90 - (iy+0.5)*RES)
    n_keep = int(round((90.0 - LAT_CUT) / RES))       # 1500 rows -> down to lat -60
    arr = arr[:n_keep, :]
    extent = [-180.0, 180.0, LAT_CUT, 90.0]

    cmap = ListedColormap([CLASS_COL[c] for c in CLASS_ORDER])
    cmap.set_bad(alpha=0.0)
    norm = BoundaryNorm(np.arange(-0.5, len(CLASS_ORDER) + 0.5), cmap.N)

    n = df.shape[0]
    frac = [100.0 * (code == i).sum() / n for i in range(len(CLASS_ORDER))]
    print("class fractions: " + ", ".join(f"{c}={f:.1f}%" for c, f in zip(CLASS_ORDER, frac)))

    w_in = args.width_px / 200.0
    fig = plt.figure(figsize=(w_in, w_in * 0.52), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1], projection=ccrs.Robinson())
    ax.set_global()
    ax.set_extent([-179.9, 179.9, -58.0, 84.0], crs=ccrs.PlateCarree())
    ax.imshow(np.ma.masked_invalid(arr), origin="upper", extent=extent,
              transform=ccrs.PlateCarree(), cmap=cmap, norm=norm,
              interpolation="nearest", regrid_shape=NLON)

    # coastlines: offline-robust (server has no internet -> cartopy cannot download)
    if not args.no_coastlines:
        if args.coastline:
            from cartopy.io.shapereader import Reader
            ax.add_geometries(Reader(args.coastline).geometries(), ccrs.PlateCarree(),
                              facecolor="none", edgecolor="grey", linewidth=0.4)
        else:
            try:
                ax.coastlines(resolution="110m", color="grey", linewidth=0.4)
            except Exception as e:
                print(f"NOTE: coastlines unavailable ({e}); drawing map without them.\n"
                      f"      For an offline server, pass --coastline ne_110m_coastline.shp "
                      f"or seed cartopy's data_dir.")
    # no ticks/labels, but DRAW the Robinson map outline (outer boundary)
    ax.set_xticks([]); ax.set_yticks([])
    try:
        ax.spines["geo"].set_visible(True)
        ax.spines["geo"].set_edgecolor("black")
        ax.spines["geo"].set_linewidth(0.9)
    except Exception:
        pass  # older cartopy: outline shown by default

    # example markers b-i: large bold letter lifted clear ABOVE the circle
    t = ccrs.PlateCarree()._as_mpl_transform(ax)
    for lbl, lon, lat in EX:
        ax.plot(lon, lat, marker="o", markersize=12, markerfacecolor="white",
                markeredgecolor="black", markeredgewidth=2.2,
                transform=ccrs.PlateCarree(), zorder=5)
        ax.annotate(lbl, xy=(lon, lat), xycoords=t, xytext=(0, 22),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=36, fontweight="bold", zorder=6,
                    path_effects=[pe.withStroke(linewidth=4.0, foreground="white")])

    fig.savefig(args.out, dpi=200, bbox_inches="tight", pad_inches=0.05,
                transparent=False, facecolor="white")
    print(f"Saved: {args.out}  ({df.shape[0]} cells, {args.width_px}px wide)")
    plt.close(fig)

    # ---- companion pie of the 4-class fractions (matches 115 class_pie) -------
    # %-label colour: black on the light-grey wedge, white on the saturated ones
    CLASS_TXT = {"stationary": "black", "location": "white",
                 "scale": "white", "location and scale": "white"}
    counts = [int((code == i).sum()) for i in range(len(CLASS_ORDER))]
    pie_out = os.path.join(os.path.dirname(os.path.abspath(args.out)),
                           f"class_pie{suffix}.png")
    figp, axp = plt.subplots(figsize=(8, 8))
    wedges, _txt, autotxt = axp.pie(
        counts, colors=[CLASS_COL[c] for c in CLASS_ORDER],
        startangle=90, counterclock=False,
        autopct=lambda p: f"{p:.2g}%", pctdistance=0.62,
        wedgeprops={"edgecolor": "white", "linewidth": 1.6},
        textprops={"fontsize": 20, "fontweight": "bold"})
    for a, c in zip(autotxt, CLASS_ORDER):
        a.set_color(CLASS_TXT[c])
    scope = "flood-ensured" if args.flood_ensured else "all land"
    axp.set_title(f"GU change-type classes ({scope}; n = {n:,})", fontsize=17, pad=14)
    axp.legend(wedges, [f"{c}  ({f:.1f}%)" for c, f in zip(CLASS_ORDER, frac)],
               loc="lower center", bbox_to_anchor=(0.5, -0.14), ncol=2,
               fontsize=13, frameon=False)
    axp.axis("equal")
    figp.savefig(pie_out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(figp)
    print(f"Saved: {pie_out}")


if __name__ == "__main__":
    main()
