"""
fig01b_example_cell_map.py
Locator map for the Figure-1 concept panels: the flood-relevant change-type
map (same data/colors as 097's change_type_map_flood) with the 8 cells
selected by 048 (all8 mode) marked by LARGE letters (b)-(i).

Style (per manuscript figure requirements):
  - Robinson projection, Antarctica omitted (lat < -60 cut from data AND frame)
  - NO legend, NO axis labels, NO title (all explained in the caption)
  - major river-basin outlines (HydroBASINS lev03) if available, as in 097

Inputs:
  <dat_dir>/091/change_type_cells.csv          (type + flood fields, from 095)
  <dat_dir>/048/fig1_all8_cells.csv            (8 selected cells, from 048 all8)
Output:
  <dat_dir>/049/fig1_cells_map.png

Usage:
  python3 fig01b_example_cell_map.py [dat_dir] [X1] [X2] [A] [basin_dir]
"""
import os
import sys
import csv
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import cartopy.crs as ccrs
import cartopy.feature as cfeature

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TYPES = ["stationary", "trend", "step", "variance"]
COLORS = {"stationary": "#9e9e9e", "trend": "#d73027",
          "step": "#4575b4", "variance": "#f0a30a"}
FLOOD_Q90 = 50.0
FLOOD_MAX = 100.0
FLOOD_UPAREA = 50.0
BASIN_MIN_AREA = 50000.0


def fnum(s):
    try:
        return float(s)
    except (ValueError, TypeError):
        return np.nan


def load_basin_feature(basin_dir, min_area_km2=BASIN_MIN_AREA):
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
                          facecolor="none", linewidth=0.45)


def main():
    dat_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(SCRIPT_DIR, "..", "data")
    x1 = float(sys.argv[2]) if len(sys.argv) > 2 else FLOOD_Q90
    x2 = float(sys.argv[3]) if len(sys.argv) > 3 else FLOOD_MAX
    a = float(sys.argv[4]) if len(sys.argv) > 4 else FLOOD_UPAREA
    basin_dir = sys.argv[5] if len(sys.argv) > 5 else os.path.join(dat_dir, "shp", "basins")

    cells_csv = os.path.join(dat_dir, "091", "change_type_cells.csv")
    sel_csv = os.path.join(dat_dir, "048", "fig1_all8_cells.csv")
    out_dir = os.path.join(dat_dir, "049")
    os.makedirs(out_dir, exist_ok=True)
    if not os.path.exists(cells_csv):
        sys.exit(f"Not found: {cells_csv} (run 094/095 first)")
    if not os.path.exists(sel_csv):
        sys.exit(f"Not found: {sel_csv} (run 'Rscript fig01a_methodology_flowchart.R <dat_dir> all8' first)")

    # flood-relevant change-type cells (same filter as 097)
    lon, lat, tcode = [], [], []
    tindex = {t: i for i, t in enumerate(TYPES)}
    with open(cells_csv, "r", newline="") as f:
        for row in csv.DictReader(f):
            la = float(row["lat"])
            if la < -60:
                continue
            t = row["type"]
            if t not in tindex:
                continue
            q90 = fnum(row.get("q90", "")); mx = fnum(row.get("max_amax", ""))
            up = fnum(row.get("uparea_km2", ""))
            if not (((q90 >= x1) or (mx >= x2)) and (up >= a)):
                continue
            lon.append(float(row["lon"])); lat.append(la); tcode.append(tindex[t])
    lon = np.array(lon); lat = np.array(lat); tcode = np.array(tcode)
    print(f"Flood-relevant cells: {len(lon)}")

    # selected cells from 048
    sel = []
    with open(sel_csv, "r", newline="") as f:
        for row in csv.DictReader(f):
            sel.append((row["label"], float(row["lon"]), float(row["lat"])))
    print(f"Marked cells: {len(sel)}: " + ", ".join(s[0] for s in sel))

    basins = load_basin_feature(basin_dir)

    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
    ax.set_extent([-180, 180, -60, 90], crs=ccrs.PlateCarree())  # no Antarctica
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    if basins is not None:
        ax.add_feature(basins)
    from matplotlib.colors import ListedColormap, BoundaryNorm
    cmap = ListedColormap([COLORS[t] for t in TYPES])
    norm = BoundaryNorm(np.arange(-0.5, len(TYPES) + 0.5), cmap.N)
    ax.scatter(lon, lat, c=tcode, s=0.3, cmap=cmap, norm=norm,
               transform=ccrs.PlateCarree(), rasterized=True, zorder=2)

    # big letters (b)-(i) at the 8 selected cells
    for lab, lo, la in sel:
        ax.scatter([lo], [la], s=130, facecolors="white", edgecolors="black",
                   linewidths=1.8, transform=ccrs.PlateCarree(), zorder=5)
        ax.text(lo + 3.0, la + 3.0, lab, transform=ccrs.PlateCarree(),
                fontsize=30, fontweight="bold", zorder=6,
                path_effects=[pe.withStroke(linewidth=4, foreground="white")])

    # no title, no legend, no axis labels (caption carries the explanation)
    out_png = os.path.join(out_dir, "fig1_cells_map.png")
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
