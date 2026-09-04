"""
lib_equivalent_ensemble_by_flow.py
Is the high equivalent-ensemble signal (the red arid/cold-belt areas in 066)
ONLY a low-flow artefact, or do HIGH-flow cells also show large eq_ens?

Low flow -> large excess -> large eq_ens is expected/natural. What matters for
flood forecasting is reducing HIGH-flow predictive uncertainty, so cells that are
BOTH high-flow AND high-eq_ens are the interesting, paper-worthy targets. This
script stratifies the flood-ensured cells into flow TERTILES (low/mid/high) by
mean AMAX flow and shows WHERE the high-flow high-eq_ens hotspots cluster.

Inputs (join by cell_id):
  063/eq_ens_allgrid_<parent>_<ns_model>.csv : eq_ens (+ iy, ix)
  070/timeseries_tests.csv                   : mean_outflow (mean AMAX)
  scope = flood-ensured cells (common_target_cells.load_flood_ensured_ids), minus
          reverse-flow / no-regime (load_excluded_cells) and Antarctica.

Outputs (in 069/<parent>/<ns_model>/):
  hotspot_density.png : per agg x agg block, FRACTION of high-flow cells that are
                        hotspots (high-flow AND eq_ens >= threshold). Reveals the
                        regional clustering of high-flow high-eq_ens cells.
  hotspot_cat.png     : per-cell categorical map -- low/mid flow (grey),
                        high-flow normal (blue), high-flow HOTSPOT (crimson).
  top_pct_by_flow.png : ONLY the top --top-pct% eq_ens cells (99th-pct tail),
                        coloured by mean AMAX flow with a DISCRETE high-contrast
                        colourmap; cells with mean AMAX >= --flow-hl (default 5000)
                        are HIGHLIGHTED (large, black-edged). Prints per-flow-bin
                        counts and the highest-flow tail cells (lon/lat).
  top_flow_hist.png   : bar chart of the tail's mean-AMAX bin counts, with the
                        EXPECTED count under flow-independence overlaid (black
                        diamonds) and O/E printed per bin, so ">=5000 vs <500"
                        is judged against the domain base rate, not raw counts.
  tail_enrichment.csv : per-bin n_domain, observed, expected, O/E, binomial p.
                        Plus a stdout binomial test for the >= --flow-hl group:
                        O/E << 1 & p<0.05 -> genuine depletion (attenuation
                        argument defensible); O/E ~ 1 -> tail is flow-independent.
  eq_ens_distribution.png : boxplots of eq_ens by flow tertile (whiskers = min/max,
                        no outlier points) + the percentile->value curve that shows
                        exactly where the 066 quantile colourmap changes colour.
  eq_ens_by_flow_stats.csv : n, min, Q1, median, Q3, P90, P95, max per group.

Hotspot threshold (within the HIGH-flow group): eq_ens >= Q(--hot-q, default 75)
of the high-flow subset, or an absolute --hot-abs value.

Usage:
  python3 lib_equivalent_ensemble_by_flow.py [dat_dir] [parent st|ln|qd] [ns_model lin|qd|ad]
      [--hot-q P | --hot-abs V] [--top-pct P] [--flow-hl V] [--agg K] [--width-px N]
      [--cmap NAME] [--scope ensured|all] [--coastline SHP] [--borders SHP] [--no-coastlines]
"""
import os
import sys
import csv
import importlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm, Normalize
import cartopy.crs as ccrs

m066 = importlib.import_module("lib_equivalent_ensemble_loader")
excess_lib = importlib.import_module("common_target_cells")

RES = m066.RES
PARENT_LABEL = m066.PARENT_LABEL
MODEL_LABEL = m066.MODEL_LABEL

# Discrete mean-AMAX flow bins (m3/s) + HIGH-CONTRAST distinct colours (Spectral-
# like, clearly different per bin -- not a smooth gradient). Last bin is open-ended
# (>=10000, values above the last edge are clipped into it).
FLOW_EDGES = np.array([0, 50, 100, 500, 1000, 5000, 10000, 50000], float)
FLOW_COLORS = ["#bdbdbd", "#3288bd", "#66c2a5", "#abdda4",
               "#fdae61", "#f46d43", "#d53e4f"]   # 7 bins, low->high


def safe_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def load_eq_ens(csv_path):
    """cell_id -> (iy, ix, eq_ens) from the 063 all-grid file."""
    out = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                cid = int(row["cell_id"]); iy = int(row["iy"]); ix = int(row["ix"])
            except (KeyError, ValueError):
                continue
            out[cid] = (iy, ix, safe_float(row.get("eq_ens")))
    return out


def load_mean_outflow(csv_path):
    """cell_id -> mean_outflow (mean AMAX) from 070/timeseries_tests.csv."""
    out = {}
    if not os.path.exists(csv_path):
        sys.exit(f"Not found: {csv_path} (070 timeseries_tests). Needed for the flow split.")
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                out[int(row["cell_id"])] = safe_float(row.get("mean_outflow"))
            except (KeyError, ValueError):
                continue
    return out


def describe(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if x.size == 0:
        return dict(n=0, min=np.nan, q1=np.nan, median=np.nan, q3=np.nan,
                    p90=np.nan, p95=np.nan, max=np.nan)
    return dict(n=int(x.size), min=float(x.min()),
                q1=float(np.percentile(x, 25)), median=float(np.median(x)),
                q3=float(np.percentile(x, 75)), p90=float(np.percentile(x, 90)),
                p95=float(np.percentile(x, 95)), max=float(x.max()))


def base_ax(fig_w_in):
    fig = plt.figure(figsize=(fig_w_in, fig_w_in * 0.60))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
    ax.set_global()
    ax.set_extent([-179.9, 179.9, m066.LAT_CUT + 2.0, 84.0], crs=ccrs.PlateCarree())
    m066.draw_boundaries(ax)
    return fig, ax


def main():
    argv = sys.argv[1:]
    hot_q = 75.0; hot_abs = None; agg = 5; width_px = 4200
    top_pct = 1.0                       # top-N% eq_ens tail shown in the flow map
    flow_hl = 5000.0                    # highlight cells with mean AMAX >= this
    cmap_name = "YlOrRd"; scope = "ensured"
    coastline = None; borders = None; no_coast = False

    def take(flag, cast=str):
        nonlocal argv
        if flag in argv:
            i = argv.index(flag); val = argv[i + 1]; del argv[i:i + 2]
            return cast(val)
        return None

    v = take("--hot-q", float);   hot_q = v if v is not None else hot_q
    v = take("--hot-abs", float);  hot_abs = v if v is not None else hot_abs
    v = take("--top-pct", float);  top_pct = v if v is not None else top_pct
    v = take("--flow-hl", float);  flow_hl = v if v is not None else flow_hl
    v = take("--agg", int);        agg = v if v is not None else agg
    v = take("--width-px", int);   width_px = v if v is not None else width_px
    v = take("--cmap");            cmap_name = v if v is not None else cmap_name
    v = take("--scope");           scope = v if v is not None else scope
    v = take("--coastline");       coastline = v
    v = take("--borders");         borders = v
    if "--no-coastlines" in argv:
        no_coast = True; argv.remove("--no-coastlines")
    if scope not in ("ensured", "all"):
        sys.exit("--scope must be ensured or all")

    dat_dir = argv[0] if len(argv) > 0 else \
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
    parent = argv[1] if len(argv) > 1 else "ln"
    ns_model = argv[2] if len(argv) > 2 else "lin"
    if parent not in PARENT_LABEL or ns_model not in MODEL_LABEL:
        sys.exit("parent in {st,ln,qd}; ns_model in {lin,qd,ad}")

    # wire 066's raster/coastline globals
    m066.WIDTH_PX = width_px; m066.AGG = agg
    m066.NO_COASTLINES = no_coast
    if coastline is None:
        cand = os.path.join(m066.SCRIPT_DIR, "ne", "ne_110m_coastline.shp")
        if os.path.exists(cand): coastline = cand
    if borders is None:
        cand = os.path.join(m066.SCRIPT_DIR, "ne", "ne_110m_admin_0_boundary_lines_land.shp")
        if os.path.exists(cand): borders = cand
    m066.COASTLINE_PATH = coastline; m066.BORDERS_PATH = borders

    eq_path = os.path.join(dat_dir, "063", f"eq_ens_allgrid_{parent}_{ns_model}.csv")
    mo_path = os.path.join(dat_dir, "070", "timeseries_tests.csv")
    out_dir = os.path.join(dat_dir, "069", parent, ns_model)
    os.makedirs(out_dir, exist_ok=True)

    eqmap = load_eq_ens(eq_path)
    momap = load_mean_outflow(mo_path)
    bad_yx = excess_lib.load_bad_yx(dat_dir)
    keep_ids = None
    if scope == "ensured":
        keep_ids = excess_lib.load_flood_ensured_ids(dat_dir)
        if not keep_ids:
            print("WARNING: flood-ensured set empty (091 missing?); falling back to all cells.")
            keep_ids = None

    # join eq_ens + mean_outflow over the scope, drop Antarctica / bad / non-finite
    cid_l, iy_l, ix_l, eq_l, mo_l = [], [], [], [], []
    for cid, (iy, ix, eq) in eqmap.items():
        if keep_ids is not None and cid not in keep_ids:
            continue
        if (iy, ix) in bad_yx:
            continue
        if 90.0 - (iy + 0.5) * RES < m066.LAT_CUT:
            continue
        mo = momap.get(cid, np.nan)
        if not (np.isfinite(eq) and np.isfinite(mo)):
            continue
        cid_l.append(cid); iy_l.append(iy); ix_l.append(ix); eq_l.append(eq); mo_l.append(mo)
    iy = np.array(iy_l); ix = np.array(ix_l)
    eq = np.array(eq_l); mo = np.array(mo_l)
    n = eq.size
    if n == 0:
        sys.exit("No cells after join/scope -- check 063, 070 and scope.")
    print(f"scope={scope}: {n} cells with eq_ens + mean_outflow")

    # flow TERTILES: low < 33.3 pct <= mid < 66.7 pct <= high
    t1, t2 = np.percentile(mo, [100.0 / 3, 200.0 / 3])
    grp = np.where(mo < t1, 0, np.where(mo < t2, 1, 2))     # 0 low, 1 mid, 2 high
    is_high = grp == 2
    print(f"flow tertiles (mean_outflow m3/s): low<{t1:.2f}  mid<{t2:.2f}  high>=")
    print(f"  n: low={int((grp==0).sum())} mid={int((grp==1).sum())} high={int(is_high.sum())}")

    # hotspot threshold within the HIGH-flow group
    hf_eq = eq[is_high]
    if hot_abs is not None:
        thr = float(hot_abs); thr_desc = f"eq_ens >= {thr:g} (absolute)"
    else:
        thr = float(np.percentile(hf_eq, hot_q)); thr_desc = f"eq_ens >= {thr:.2f} (Q{hot_q:g} of high-flow)"
    is_hot = is_high & (eq >= thr)
    print(f"hotspot: {thr_desc} -> {int(is_hot.sum())} cells "
          f"({100.0*is_hot.sum()/max(is_high.sum(),1):.1f}% of high-flow)")

    plabel = PARENT_LABEL[parent]; mlabel = MODEL_LABEL[ns_model]
    w_in = width_px / m066.DPI

    # ---- 1. hotspot DENSITY map: fraction of high-flow cells that are hotspots ----
    # value 1.0 for hotspot, 0.0 for other high-flow cells, NaN elsewhere;
    # block_nanmean over agg x agg then = hotspot fraction among high-flow cells.
    hf_val = np.where(is_hot[is_high], 1.0, 0.0)
    arr, extent = m066.rasterize(iy[is_high], ix[is_high], hf_val, agg=agg)
    cmap = plt.get_cmap(cmap_name).copy(); cmap.set_bad(alpha=0.0)
    fig, ax = base_ax(w_in)
    im = ax.imshow(arr, origin="upper", extent=extent, transform=ccrs.PlateCarree(),
                   cmap=cmap, norm=Normalize(0.0, 1.0), interpolation="nearest",
                   regrid_shape=m066.NLON, zorder=0)
    cbar = plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.05, shrink=0.7, aspect=40)
    cbar.set_label("hotspot fraction among high-flow cells", fontsize=22)
    cbar.ax.tick_params(labelsize=18)
    ax.set_title(f"High-flow high-eq_ens hotspot density  ({mlabel}/{plabel})\n"
                 f"{thr_desc}; {agg*RES:.1f}deg blocks", fontsize=22, pad=12)
    p = os.path.join(out_dir, "hotspot_density.png")
    plt.savefig(p, dpi=m066.DPI, bbox_inches="tight"); plt.close(); print(f"Saved: {p}")

    # ---- 2. per-cell CATEGORICAL map: low/mid grey, high-flow blue, hotspot crimson ----
    cat = np.where(is_hot, 2.0, np.where(is_high, 1.0, 0.0))
    arr_c, extent = m066.rasterize(iy, ix, cat, agg=1)         # per-cell (no averaging)
    # nearest bin: round the (unaveraged) codes back to ints for the discrete cmap
    cat_cmap = ListedColormap(["#d9d9d9", "#4575b4", "#d73027"])   # low/mid, high, hotspot
    cat_cmap.set_bad(alpha=0.0)
    cnorm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cat_cmap.N)
    fig, ax = base_ax(w_in)
    ax.imshow(arr_c, origin="upper", extent=extent, transform=ccrs.PlateCarree(),
              cmap=cat_cmap, norm=cnorm, interpolation="nearest",
              regrid_shape=m066.NLON, zorder=0)
    handles = [plt.Line2D([0], [0], marker="s", linestyle="", markersize=14, markerfacecolor=c,
                          markeredgecolor="none",
                          label=lbl) for c, lbl in
               [("#d9d9d9", "low/mid flow"), ("#4575b4", "high flow"),
                ("#d73027", "high flow AND hotspot")]]
    ax.legend(handles=handles, loc="lower left", fontsize=16, framealpha=0.9)
    ax.set_title(f"High-flow equivalent-ensemble hotspots  ({mlabel}/{plabel})\n"
                 f"{thr_desc}", fontsize=22, pad=12)
    p = os.path.join(out_dir, "hotspot_cat.png")
    plt.savefig(p, dpi=m066.DPI, bbox_inches="tight"); plt.close(); print(f"Saved: {p}")

    # ---- 2b. TOP eq_ens cells ONLY, coloured by mean AMAX flow (discrete cmap) ----
    # The eq_ens>~6 tail is ~the top 1% (99th pct of all cells). Plot ONLY those
    # tail cells and colour by mean AMAX with a DISCRETE colourmap, so we can see
    # whether any big-river / high-flow cells reach the tail (Nile, Mississippi,
    # Indus, Ganges, La Plata, ...) or whether it is essentially all low-flow.
    eq_thr_top = float(np.percentile(eq, 100.0 - top_pct))
    top = eq >= eq_thr_top
    n_top = int(top.sum())
    print(f"\ntop {top_pct:g}% eq_ens: eq_ens >= {eq_thr_top:.2f} -> {n_top} cells")
    lat_top = 90.0 - (iy[top] + 0.5) * RES
    lon_top = -180.0 + (ix[top] + 0.5) * RES
    mo_top = mo[top]
    eq_top = eq[top]

    flow_cmap = ListedColormap(FLOW_COLORS)
    flow_norm = BoundaryNorm(FLOW_EDGES, flow_cmap.N, clip=True)
    hi = mo_top >= flow_hl                       # big-river cells to HIGHLIGHT
    n_hi = int(hi.sum())
    print(f"highlight: mean_amax >= {flow_hl:g} -> {n_hi} of {n_top} tail cells")

    fig, ax = base_ax(w_in)
    # de-emphasised small dots for the low-flow majority (< flow_hl)
    if (~hi).any():
        ax.scatter(lon_top[~hi], lat_top[~hi], c=mo_top[~hi], s=4, cmap=flow_cmap,
                   norm=flow_norm, transform=ccrs.PlateCarree(), alpha=0.45,
                   zorder=4, edgecolors="none")
    # HIGHLIGHT: large, black-edged markers for >= flow_hl (the big rivers)
    sc = ax.scatter(lon_top[hi], lat_top[hi], c=mo_top[hi], s=60, cmap=flow_cmap,
                    norm=flow_norm, transform=ccrs.PlateCarree(), zorder=6,
                    edgecolors="black", linewidths=0.8)
    cbar = plt.colorbar(sc, ax=ax, orientation="horizontal",
                        pad=0.05, shrink=0.7, aspect=40, ticks=FLOW_EDGES,
                        spacing="uniform")
    cbar.set_label("mean AMAX flow [m3/s]  (top bin >=10000, clipped)", fontsize=20)
    cbar.ax.tick_params(labelsize=16)
    ax.set_title(f"Top {top_pct:g}% eq_ens cells; highlighting mean AMAX >= {flow_hl:g} m3/s "
                 f"({mlabel}/{plabel})\neq_ens >= {eq_thr_top:.2f}  (tail n={n_top}, highlighted={n_hi})",
                 fontsize=20, pad=12)
    p = os.path.join(out_dir, "top_pct_by_flow.png")
    plt.savefig(p, dpi=m066.DPI, bbox_inches="tight"); plt.close(); print(f"Saved: {p}")

    # ---- 2c. HISTOGRAM + ENRICHMENT: tail composition vs the DOMAIN base rate ----
    # "72 tail cells >= 5000 m3/s" is only interpretable against how many domain
    # cells are >= 5000 in the first place. Under flow-independence (H0), each bin
    # is expected to put p_tail = n_top/n of its cells into the tail:
    #   expected_b = n_domain_b * p_tail ;  O/E = observed_b / expected_b
    # O/E << 1 -> high-flow cells are genuinely DEPLETED from the tail (supports
    # attenuation/aggregation damping); O/E ~ 1 -> tail is flow-independent.
    p_tail = n_top / float(n)
    cbins = list(FLOW_EDGES); cbins[-1] = np.inf     # last bin open-ended
    counts, n_dom, labels = [], [], []
    for lo, hi_e in zip(cbins[:-1], cbins[1:]):
        counts.append(int(((mo_top >= lo) & (mo_top < hi_e)).sum()))
        n_dom.append(int(((mo >= lo) & (mo < hi_e)).sum()))
        labels.append(f"{lo:.0f}-{'inf' if not np.isfinite(hi_e) else f'{hi_e:.0f}'}")
    expected = [nd * p_tail for nd in n_dom]

    figh, axh = plt.subplots(figsize=(11, 6))
    bars = axh.bar(range(len(counts)), counts, color=FLOW_COLORS, edgecolor="black",
                   label="observed in tail")
    axh.plot(range(len(expected)), expected, "D-", color="black", markersize=7,
             linewidth=1.2, label=f"expected if flow-independent (n_bin x {p_tail:.4f})")
    for k_b, (b, c, e_v) in enumerate(zip(bars, counts, expected)):
        oe = c / e_v if e_v > 0 else np.nan
        axh.text(b.get_x() + b.get_width() / 2, max(c, e_v),
                 f"{c}\nO/E={oe:.2f}" if np.isfinite(oe) else f"{c}\nO/E=--",
                 ha="center", va="bottom", fontsize=9)
    axh.set_xticks(range(len(counts)))
    axh.set_xticklabels(labels, rotation=30, ha="right")
    axh.set_xlabel("mean AMAX flow bin [m3/s]")
    axh.set_ylabel(f"top {top_pct:g}% eq_ens cells (count)")
    axh.set_ylim(0, max(max(counts), max(expected)) * 1.25 if counts else 1)
    axh.legend(fontsize=10)
    axh.set_title(f"mean-AMAX distribution of the top {top_pct:g}% eq_ens tail vs base rate  "
                  f"({mlabel}/{plabel}); tail n={n_top} of {n}\n"
                  f">= {flow_hl:g} m3/s: {100.0*n_hi/max(n_top,1):.2f}% of the tail",
                  fontsize=14)
    figh.tight_layout()
    p = os.path.join(out_dir, "top_flow_hist.png")
    figh.savefig(p, dpi=140, bbox_inches="tight"); plt.close(); print(f"Saved: {p}")

    # ---- 2d. enrichment table + binomial test for the >= flow_hl group ----
    # Per-bin O/E with a 95% CI on the observed count (Poisson approx), then an
    # exact binomial test for the highlighted high-flow group:
    #   H0: a >=flow_hl cell enters the tail with the same p_tail as any cell.
    def binom_p_two_sided(k, n_trials, p0):
        """Two-sided binomial p-value; scipy if available, else normal approx."""
        try:
            from scipy.stats import binomtest
            return float(binomtest(k, n_trials, p0).pvalue)
        except Exception:
            mu = n_trials * p0
            sd = np.sqrt(n_trials * p0 * (1.0 - p0))
            if sd == 0:
                return 1.0
            z = (k - mu) / sd
            from math import erfc
            return float(erfc(abs(z) / np.sqrt(2.0)))     # 2*(1-Phi(|z|))

    enr_path = os.path.join(out_dir, "tail_enrichment.csv")
    print(f"\nenrichment vs base rate (p_tail={p_tail:.5f}):")
    print(f"{'bin':>14} {'n_domain':>9} {'observed':>9} {'expected':>9} {'O/E':>7} {'p_binom':>10}")
    with open(enr_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bin_m3s", "n_domain", "observed_tail", "expected_tail",
                    "obs_over_exp", "p_binomial"])
        for lab, nd, c, e_v in zip(labels, n_dom, counts, expected):
            oe = c / e_v if e_v > 0 else np.nan
            pv = binom_p_two_sided(c, nd, p_tail) if nd > 0 else np.nan
            print(f"{lab:>14} {nd:>9d} {c:>9d} {e_v:>9.1f} "
                  f"{oe:>7.2f} {pv:>10.3g}")
            w.writerow([lab, nd, c, f"{e_v:.2f}", f"{oe:.3f}", f"{pv:.4g}"])
    print(f"Saved: {enr_path}")

    n_hf_dom = int((mo >= flow_hl).sum())
    e_hf = n_hf_dom * p_tail
    oe_hf = n_hi / e_hf if e_hf > 0 else np.nan
    pv_hf = binom_p_two_sided(n_hi, n_hf_dom, p_tail) if n_hf_dom > 0 else np.nan
    print(f"\n>= {flow_hl:g} m3/s group: n_domain={n_hf_dom}, observed_tail={n_hi}, "
          f"expected={e_hf:.1f}, O/E={oe_hf:.2f}, p={pv_hf:.3g}")
    if np.isfinite(oe_hf):
        if oe_hf < 0.8 and pv_hf < 0.05:
            print("  -> high-flow cells are significantly DEPLETED from the tail "
                  "(supports attenuation/aggregation damping).")
        elif pv_hf >= 0.05:
            print("  -> no significant depletion/enrichment: the tail is "
                  "flow-independent at this threshold.")
        else:
            print("  -> high-flow cells are significantly ENRICHED in the tail.")

    # per-bin counts + highest-flow tail cells (candidate big rivers)
    print("\ntop-cell counts by mean-AMAX bin (m3/s):")
    for lab, c in zip(labels, counts):
        print(f"  {lab:>14}: {c:6d}  ({100.0*c/max(n_top,1):5.2f}%)")
    if n_top > 0:
        order_hi = np.argsort(mo_top)[::-1][:20]
        print("highest-flow tail cells (lon, lat, mean_amax, eq_ens):")
        for j in order_hi:
            print(f"  ({lon_top[j]:8.2f}, {lat_top[j]:7.2f})  "
                  f"mean_amax={mo_top[j]:10.1f}  eq_ens={eq_top[j]:.2f}")

    # ---- 3. distribution: boxplots by tertile (whiskers=min/max, no fliers) + pct curve ----
    groups = {"all": eq, "low": eq[grp == 0], "mid": eq[grp == 1], "high": eq[is_high]}
    fig, (axb, axp) = plt.subplots(1, 2, figsize=(15, 6))
    order = ["low", "mid", "high", "all"]
    axb.boxplot([groups[k] for k in order], tick_labels=order, whis=(0, 100), showfliers=False,
                medianprops=dict(color="crimson", linewidth=2))
    axb.axhline(thr, color="#d73027", linestyle="--", linewidth=1.2,
                label=f"hotspot thr = {thr:.2f}")
    axb.set_ylabel("equivalent ensemble size"); axb.set_xlabel("flow tertile")
    axb.set_title("eq_ens by flow tertile\n(box=Q1..Q3, whiskers=min..max, no outliers)")
    axb.legend(fontsize=9)
    # percentile -> value curve (inverse CDF) = where the 066 quantile colours change
    pr = np.linspace(0, 100, 501); yv = np.percentile(groups["all"], pr)
    axp.plot(pr, yv, color="#333333")
    dec = np.arange(0, 101, 10); dv = np.percentile(groups["all"], dec)
    axp.plot(dec, dv, "o", color="#0072B2")
    for d, val in zip(dec, dv):
        axp.annotate(f"{val:.2f}", (d, val), textcoords="offset points", xytext=(4, -2),
                     fontsize=8, color="#0072B2")
    axp.set_xlabel("percentile (= colour position in quantile map)")
    axp.set_ylabel("equivalent ensemble size")
    axp.set_title("Where the quantile colourmap changes\n(inverse-CDF of all-cell eq_ens)")
    axp.grid(alpha=0.3)
    fig.tight_layout()
    p = os.path.join(out_dir, "eq_ens_distribution.png")
    fig.savefig(p, dpi=140, bbox_inches="tight"); plt.close(); print(f"Saved: {p}")

    # ---- 4. numeric summary (stdout + CSV) ----
    stats_path = os.path.join(out_dir, "eq_ens_by_flow_stats.csv")
    cols = ["n", "min", "q1", "median", "q3", "p90", "p95", "max"]
    print("\neq_ens distribution by flow group:")
    hdr = "group     " + "".join(f"{c:>9}" for c in cols); print(hdr)
    with open(stats_path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["group"] + cols)
        for k in order:
            s = describe(groups[k])
            print(f"{k:<9} " + "".join(
                (f"{s[c]:>9d}" if c == "n" else f"{s[c]:>9.2f}") for c in cols))
            w.writerow([k] + [s[c] for c in cols])
    print(f"\nSaved: {stats_path}")
    print(f"flow tertile thresholds (mean_outflow m3/s): t1={t1:.2f}  t2={t2:.2f}")


if __name__ == "__main__":
    main()
