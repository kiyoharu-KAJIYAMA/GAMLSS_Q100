"""
fig07_improvement_vs_parameter_slope.py  (local)
Does the SD improvement track the SCALE (sigma) year-slope or the LOCATION (mu)
year-slope? Uses the AIC change-type class (079/aic_allgrid.csv 'class':
stationary / mean=location-only / var=scale-only / both) to put each cell in the
RIGHT panel, and the linear-GU truth slopes (064/ln) as the predictors:

  (1) improvement_vs_sigslope.png : SCALE-ONLY cells (class 'var').
        x = sigma_1 (log link, per-year relative scale change).
  (2) improvement_vs_muslope.png  : LOCATION-ONLY cells (class 'mean').
        x = mu_1/mu_0 (per-year relative change of the mean; mu on actual flow).
  (3) improvement_2d_both.png     : BOTH-change cells (class 'both'), a 2-D
        heatmap of mean improvement over (|sigma_1| x |mu_1/mu_0|) with the 052
        5x4 improvement colormap + marginal/partial Spearman, to read WHICH axis
        drives the improvement.

y = improvement [%] = (SD_stat - SD_lin)/SD_stat*100 from 050 (no baseline
subtraction). Scope = flood-ensured. All inputs local.

Outputs (<dat_dir>/511/): the three PNGs + slope_stats.csv
Usage: python3 fig07_improvement_vs_parameter_slope.py [dat_dir] [--scope ensured|all]
"""
import os
import sys
import glob
import csv
import importlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (enables projection='3d')

excess_lib = importlib.import_module("common_target_cells")
m052 = importlib.import_module("fig05a_improvement_map")

CAP_PCT = 300.0
LBL_FS, TICK_FS, TITLE_FS = 20, 17, 14     # enlarged axis fonts
NOISE_PCT = 90.0                            # noise floor = this pct of |slope| in stationary cells

# AIC change-type classes coloured as in Figure 2 (120 Okabe-Ito palette):
CLASS_ORDER = ["stationary", "mean", "var", "both"]
CLASS_COL = {"stationary": "#BBBBBB", "mean": "#0072B2",
             "var": "#D55E00", "both": "#009E73"}
CLASS_NAME = {"stationary": "stationary", "mean": "location",
              "var": "scale", "both": "location and scale"}


def _ranks(a):
    return np.argsort(np.argsort(a, kind="mergesort")).astype(float)


def spearman(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size < 10:
        return np.nan
    rx = _ranks(x) - (x.size - 1) / 2.0
    ry = _ranks(y) - (y.size - 1) / 2.0
    d = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    return float((rx * ry).sum() / d) if d > 0 else np.nan


def partial_spearman(imp, a, b):
    """Spearman(imp, a) averaged within tertiles of b (b controlled)."""
    t1, t2 = np.nanpercentile(b, [100 / 3, 200 / 3])
    grp = np.where(b < t1, 0, np.where(b < t2, 1, 2))
    vals = [spearman(a[grp == k], imp[grp == k]) for k in range(3)]
    return float(np.nanmean(vals))


def load_ln_coeffs(dat_dir):
    chunks = [c for c in sorted(glob.glob(os.path.join(dat_dir, "064", "ln", "mc_truth", "*.csv")))
              if not c.endswith("Zone.Identifier")]
    cols = ["cell_id", "truth_mu_0", "truth_mu_1", "truth_sigma_1"]
    frames = []
    for c in chunks:
        try:
            frames.append(pd.read_csv(c, usecols=cols))
        except (ValueError, pd.errors.EmptyDataError):
            continue
    df = pd.concat(frames, ignore_index=True).dropna(
        subset=["truth_mu_0", "truth_mu_1", "truth_sigma_1"])
    return df.set_index("cell_id")


def load_improvement(dat_dir):
    df = pd.read_csv(os.path.join(dat_dir, "050", "summary_allgrid.csv"),
                     usecols=["cell_id", "stat_err_sd", "lin_err_sd"])
    df = df[(df["stat_err_sd"] > 0) & np.isfinite(df["stat_err_sd"])
            & np.isfinite(df["lin_err_sd"])]
    imp = (df["stat_err_sd"] - df["lin_err_sd"]) / df["stat_err_sd"] * 100.0
    return pd.Series(imp.values, index=df["cell_id"].values)


def load_class(dat_dir):
    """cell_id -> AIC change-type class (stationary/mean/var/both) from 079."""
    df = pd.read_csv(os.path.join(dat_dir, "079", "aic_allgrid.csv"),
                     usecols=["cell_id", "class"])
    return pd.Series(df["class"].values, index=df["cell_id"].values)


def hexfig(x, y, xlabel, title, out_png):
    m = np.isfinite(x) & np.isfinite(y) & (np.abs(y) <= CAP_PCT)
    x, y = x[m], y[m]
    lo, hi = np.percentile(x, [1, 99]); mm = (x >= lo) & (x <= hi)
    med_y = float(np.median(y)); rho = spearman(x, y); rho_a = spearman(np.abs(x), y)
    fig, ax = plt.subplots(figsize=(8.4, 6.0))
    ax.hexbin(x[mm], y[mm], gridsize=50, bins="log", cmap="Greys", mincnt=1)
    edges = np.unique(np.quantile(x[mm], np.linspace(0, 1, 15)))
    cx, cy = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        s = (x >= a) & (x <= b)
        if s.sum() >= 30:
            cx.append(0.5 * (a + b)); cy.append(np.median(y[s]))
    ax.plot(cx, cy, "-o", color="crimson", lw=2.4, ms=5, label="binned median")
    ax.axhline(med_y, color="#555555", lw=1.2, ls=(0, (2, 2)),
               label=f"overall median = {med_y:.1f}%")
    ax.axvline(0, color="grey", lw=0.9, ls=":")
    ax.set_xlabel(xlabel, fontsize=LBL_FS)
    ax.set_ylabel("improvement [%]", fontsize=LBL_FS)
    ax.tick_params(axis="both", labelsize=TICK_FS)
    ax.set_title(f"{title}\nSpearman rho={rho:+.3f}  (|slope|: {rho_a:+.3f}),  n={x.size:,}",
                 fontsize=TITLE_FS)
    ax.legend(fontsize=12, loc="upper center")
    fig.tight_layout()
    fig.savefig(out_png, dpi=160, bbox_inches="tight"); plt.close(fig)
    print(f"Saved: {out_png}  (n={x.size}, rho={rho:+.3f}, rho_|.|={rho_a:+.3f})")
    return rho, rho_a


def hexfig_allclass(x, y, klass, xlabel, title, out_png, tau=None):
    """improvement vs a signed slope over ALL flood-ensured cells (NOT one class).
    Grey scatter cloud of all cells, with a PER-CLASS binned-median curve (improvement
    vs the slope, binned within each class) drawn in the Figure-2 palette. Each class
    traces how improvement responds to the slope over its own slope range.
    If `tau` is given (the class threshold on THIS axis), draw +/-tau as vertical lines:
    for the slope-threshold classification the classes are defined by |slope| vs tau on
    this very axis, so the inner classes' curves live inside [-tau,tau] and the outer
    classes' curves live outside -- the curves visibly CUT at +/-tau."""
    BIG_LBL, BIG_TICK = 27, 23                   # enlarged axis fonts (per request)
    m = np.isfinite(x) & np.isfinite(y) & (np.abs(y) <= CAP_PCT)
    x, yy, kk = x[m], y[m], klass[m]
    lo, hi = np.percentile(x, [0.05, 99.95]); mm = (x >= lo) & (x <= hi)  # wide x-window
    ylo, yhi = np.percentile(yy, [0.1, 99.9])                            # wide y-window
    rho = spearman(x, yy); rho_a = spearman(np.abs(x), yy)
    fig, ax = plt.subplots(figsize=(9.6, 7.0))
    ax.scatter(x[mm], yy[mm], s=2, alpha=0.04, linewidths=0, color="#444444",
               rasterized=True)                  # all cells as a scatter cloud
    ax.axvline(0, color="grey", lw=0.9, ls=":")
    handles = []
    if tau is not None and tau > 0:               # class threshold on THIS axis
        for xt in (-tau, tau):
            ax.axvline(xt, color="#d62728", lw=1.8, ls="--", alpha=0.85, zorder=6)
        handles.append(plt.Line2D([0], [0], color="#d62728", lw=1.8, ls="--",
                                  label=fr"class threshold  $\pm${tau:.4f}"))
    for ck in CLASS_ORDER:                       # per-class binned-median curve
        s = kk == ck
        if s.sum() < 100:
            continue
        xs, ys = x[s], yy[s]
        keep = (xs >= lo) & (xs <= hi)           # common x-window for comparability
        xs, ys = xs[keep], ys[keep]
        line = None
        for side in (xs < 0, xs > 0):            # split branches so no line crosses
            xb, yb = xs[side], ys[side]          # the +/-tau gap (outer classes) or x=0
            if xb.size < 60:
                continue
            edges = np.unique(np.quantile(xb, np.linspace(0, 1, 13)))
            cx, cy = [], []
            for a, b in zip(edges[:-1], edges[1:]):
                q = (xb >= a) & (xb <= b)
                if q.sum() >= 30:
                    # plot at the bin's MEDIAN x (not the edge-midpoint): with the
                    # heavily peaked, long-tailed slope distribution the wide outer
                    # quantile bins would otherwise place the point far from where the
                    # cells actually sit, offsetting the curve from the scatter.
                    cx.append(float(np.median(xb[q]))); cy.append(float(np.median(yb[q])))
            if len(cx) >= 2:
                line, = ax.plot(cx, cy, "-", color=CLASS_COL[ck], lw=3.0,
                                alpha=0.95, zorder=5)
        if line is not None:
            line.set_label(CLASS_NAME[ck])
            handles.append(line)
    ax.set_xlim(lo, hi); ax.set_ylim(float(ylo), float(yhi))
    ax.set_xlabel(xlabel, fontsize=BIG_LBL)
    ax.set_ylabel("improvement [%]", fontsize=BIG_LBL)
    ax.tick_params(axis="both", labelsize=BIG_TICK)
    ax.locator_params(axis="x", nbins=6)         # fewer x ticks so big labels don't overlap
    fig.tight_layout()                           # NO legend in the main figure
    fig.savefig(out_png, dpi=170, bbox_inches="tight"); plt.close(fig)
    # standalone legend figure
    legfig = plt.figure(figsize=(6.6, 2.2))
    legfig.legend(handles, [h.get_label() for h in handles], loc="center", ncol=1,
                  fontsize=17, title="class binned median", title_fontsize=17,
                  frameon=False)
    leg_png = out_png.replace(".png", "_legend.png")
    legfig.savefig(leg_png, dpi=170, bbox_inches="tight"); plt.close(legfig)
    print(f"Saved: {out_png}  (n={x.size}, rho={rho:+.3f}, rho_|.|={rho_a:+.3f})")
    print(f"Saved: {leg_png}")
    return rho, rho_a


def class_scatter_sigslope(sig1, y, klass, out_png):
    """079's excess_vs_sigslope idea, reworked to AVOID over-plotting: improvement
    vs the scale year-slope sigma_1 over ALL flood-ensured cells, but as 2x2 SMALL
    MULTIPLES -- one panel per AIC change-type class (Figure-2 palette), each drawn
    over a light-grey context of ALL cells. Because every class gets its own panel,
    NO class can bury another (the single-panel overlay is order-dependent; this is
    not). NO median-bin overlay. Shared axes so the panels are directly comparable.
    Shows that scale-change classes (scale, location&scale) fan out along sigma_1
    into the improvement 'U', while stationary/location sit near sigma_1=0."""
    m = np.isfinite(sig1) & np.isfinite(y) & (np.abs(y) <= CAP_PCT)
    x, yy, kk = sig1[m], y[m], klass[m]
    lo, hi = np.percentile(x, [1, 99]); mm = (x >= lo) & (x <= hi)
    x, yy, kk = x[mm], yy[mm], kk[mm]
    # axes hug the ACTUAL plotted data range (small margin only; no reach to y=0)
    xpad = 0.03 * (x.max() - x.min()); ypad = 0.03 * (yy.max() - yy.min())
    xlim = (float(x.min() - xpad), float(x.max() + xpad))
    ylim = (float(yy.min() - ypad), float(yy.max() + ypad))
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.2), sharex=True, sharey=True)
    for ax, ck in zip(axes.ravel(), CLASS_ORDER):
        ax.scatter(x, yy, s=2, alpha=0.05, linewidths=0, color="#d9d9d9",
                   rasterized=True)                         # all cells = grey context
        s = kk == ck
        ax.scatter(x[s], yy[s], s=2.6, alpha=0.28, linewidths=0,
                   color=CLASS_COL[ck], rasterized=True)    # this class, coloured
        ax.axvline(0, color="grey", lw=0.8, ls=":")
        tcol = "#555555" if ck == "stationary" else CLASS_COL[ck]
        ax.set_title(f"{CLASS_NAME[ck]}  (n={int(s.sum()):,})", fontsize=16, color=tcol)
        ax.tick_params(labelsize=13)
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.grid(alpha=0.2)
    for ax in axes[-1, :]:
        ax.set_xlabel(r"scale year-slope $\sigma_1$   ($<0$ decr. $\mid$ $>0$ incr.)",
                      fontsize=16)
    for ax in axes[:, 0]:
        ax.set_ylabel("improvement [%]", fontsize=16)
    fig.suptitle("improvement vs scale year-slope by change-type class "
                 "(flood-ensured; grey = all cells)", fontsize=16, y=0.995)
    fig.tight_layout()
    fig.savefig(out_png, dpi=170, bbox_inches="tight"); plt.close(fig)
    print(f"Saved: {out_png}  (n={x.size})")


def slope_reclassify(sig1, mu_rel, aic_klass, ok, noise_pct=NOISE_PCT):
    """INDEPENDENT 4-way class from the 064/ln BOTH-fit slopes alone (no per-cell
    AIC): a slope counts as REAL only if its magnitude clears a NOISE FLOOR, else
    negligible. The floor is the noise_pct-th percentile of |slope| among the
    AIC-stationary cells -- cells whose trend is not AIC-justified, so their fitted
    both-slope is pure noise. Then
        scale real?  |sigma_1|      > tau_s
        loc   real?  |mu_1/mubar|   > tau_m
    -> stationary / mean(location only) / var(scale only) / both.
    Returns (new_klass, tau_s, tau_m)."""
    a_s, a_m = np.abs(sig1), np.abs(mu_rel)
    st = ok & (aic_klass == "stationary") & np.isfinite(a_s) & np.isfinite(a_m)
    tau_s = float(np.nanpercentile(a_s[st], noise_pct))
    tau_m = float(np.nanpercentile(a_m[st], noise_pct))
    scale_on = np.isfinite(a_s) & (a_s > tau_s)
    loc_on = np.isfinite(a_m) & (a_m > tau_m)
    new = np.select(
        [~scale_on & ~loc_on, ~scale_on & loc_on, scale_on & ~loc_on, scale_on & loc_on],
        ["stationary", "mean", "var", "both"], default="stationary")
    return new.astype(str), tau_s, tau_m


def draw_partial_panelA(ax, a_s, a_m, y, ok, legend_loc="upper left"):
    """Draw the PARTIAL-dependence curves (scale vs location) on a given axes.
    Effect of each slope with the OTHER held to its middle tertile, x = percentile
    rank of the slope within that held-fixed subset, y = median improvement per
    decile. Returns (partial_scale, partial_location)."""
    def rankpct(a, mask):
        r = np.full(a.shape, np.nan)
        v = a[mask]
        r[mask] = (_ranks(v) + 0.5) / v.size * 100.0
        return r
    edges = np.linspace(0, 100, 11)
    both_ok = ok & np.isfinite(a_s) & np.isfinite(a_m)
    ps_s = partial_spearman(y[both_ok], a_s[both_ok], a_m[both_ok])
    ps_m = partial_spearman(y[both_ok], a_m[both_ok], a_s[both_ok])
    for a_slope, a_other, col, name, ps in [
            (a_s, a_m, CLASS_COL["var"], r"scale $|\sigma_1|$", ps_s),
            (a_m, a_s, CLASS_COL["mean"], r"location $|\mu_1/\bar{\mu}|$", ps_m)]:
        t1, t2 = np.nanpercentile(a_other[both_ok], [100 / 3, 200 / 3])
        mk = both_ok & (a_other >= t1) & (a_other <= t2)   # hold OTHER slope ~fixed
        rp = rankpct(a_slope, mk)
        cx, cy = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            s = mk & (rp >= lo) & ((rp < hi) if hi < 100 else (rp <= hi))
            if s.sum() >= 30:
                cx.append(0.5 * (lo + hi)); cy.append(float(np.median(y[s])))
        ax.plot(cx, cy, "-o", color=col, lw=2.8, ms=6,
                label=f"{name}   (partial Spearman {ps:+.2f})")
    ax.set_xlabel("percentile of |slope|  (other slope held fixed)", fontsize=LBL_FS)
    ax.set_ylabel("median improvement [%]", fontsize=LBL_FS)
    ax.tick_params(axis="both", labelsize=TICK_FS)
    ax.set_xlim(0, 100)
    ax.legend(fontsize=13, loc=legend_loc)
    ax.grid(alpha=0.25)
    return ps_s, ps_m


def panelA_standalone(sig1, mu_rel, y, klass, out_png):
    """Panel (a) on its own: partial-dependence curves, scale vs location."""
    ok = np.isfinite(y) & (np.abs(y) <= CAP_PCT)
    a_s, a_m = np.abs(sig1), np.abs(mu_rel)
    fig, ax = plt.subplots(figsize=(8.6, 6.4))
    ps_s, ps_m = draw_partial_panelA(ax, a_s, a_m, y, ok)
    ax.set_title("with the other slope held fixed, scale lifts improvement more\n"
                 f"partial Spearman: scale {ps_s:+.2f}  vs  location {ps_m:+.2f}",
                 fontsize=TITLE_FS)
    fig.tight_layout()
    fig.savefig(out_png, dpi=170, bbox_inches="tight"); plt.close(fig)
    print(f"Saved: {out_png}")


def improvement_surface3d(sig1, mu_rel, y, out_png, elev=28, azim=-60):
    """Scale slope, location slope and improvement on THREE EQUIVALENT axes (no
    colormap needed for improvement -- it is the Z HEIGHT). Bin the flood-ensured
    cells on a quantile grid over (|sigma_1|, |mu_1/mubar|); Z = median improvement
    per bin. The surface climbs steeply along the scale-slope (x) axis and only
    gently along the location-slope (y) axis, so improvement's dependence on each
    slope is read directly as the slope of the surface. Colour merely echoes Z."""
    m = np.isfinite(sig1) & np.isfinite(mu_rel) & np.isfinite(y) & (np.abs(y) <= CAP_PCT)
    x, yv, z = np.abs(sig1)[m], np.abs(mu_rel)[m], y[m]
    xlo, xhi = np.percentile(x, [1, 97]); ylo, yhi = np.percentile(yv, [1, 97])
    NB = 14
    xe = np.unique(np.quantile(x[(x >= xlo) & (x <= xhi)], np.linspace(0, 1, NB + 1)))
    ye = np.unique(np.quantile(yv[(yv >= ylo) & (yv <= yhi)], np.linspace(0, 1, NB + 1)))
    xi = np.clip(np.digitize(x, xe) - 1, 0, xe.size - 2)
    yi = np.clip(np.digitize(yv, ye) - 1, 0, ye.size - 2)
    nx, ny = xe.size - 1, ye.size - 1
    grp = pd.DataFrame({"b": yi * nx + xi, "z": z}).groupby("b")["z"].agg(["median", "size"])
    med = np.full(nx * ny, np.nan)
    good = grp["size"] >= 30
    med[grp.index[good].to_numpy()] = grp["median"][good].to_numpy()
    med = med.reshape(ny, nx)
    xc = 0.5 * (xe[:-1] + xe[1:]); yc = 0.5 * (ye[:-1] + ye[1:])
    X, Y = np.meshgrid(xc, yc)
    Z = np.ma.masked_invalid(med)

    fig = plt.figure(figsize=(11.0, 8.4))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(X, Y, Z, cmap="viridis", rstride=1, cstride=1,
                           linewidth=0.2, edgecolor="#33333322", antialiased=True,
                           vmin=np.nanpercentile(med, 2), vmax=np.nanpercentile(med, 98))
    ax.set_xlabel(r"scale slope  $|\sigma_1|$", fontsize=16, labelpad=12)
    ax.set_ylabel(r"location slope  $|\mu_1/\bar{\mu}|$", fontsize=16, labelpad=12)
    ax.set_zlabel("improvement [%]", fontsize=16, labelpad=10)
    ax.tick_params(labelsize=11)
    ax.view_init(elev=elev, azim=azim)
    ax.set_title("median improvement surface over the (scale, location) slope plane",
                 fontsize=TITLE_FS)
    cb = fig.colorbar(surf, ax=ax, shrink=0.55, pad=0.10)
    cb.set_label("improvement [%]", fontsize=13); cb.ax.tick_params(labelsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=170, bbox_inches="tight"); plt.close(fig)
    print(f"Saved: {out_png}")


def improvement_slope_plane_byclass(sig1, mu_rel, y, klass, out_png):
    """ONE figure carrying all FOUR quantities the user asked for:
      (1) scale slope    -> x = |sigma_1|
      (2) location slope -> y = |mu_1/mubar|
      (3) improvement    -> point COLOUR (052 semantic improvement colormap)
      (4) change class    -> 2x2 SMALL MULTIPLES (Figure-2 palette in titles)
    Absolute-valued slopes fold the signed U into one quadrant, so improvement
    simply deepens toward the upper-right (both slopes large). Each class occupies
    its own region: stationary near the origin, location up the y-axis, scale along
    the x-axis, location&scale filling the plane. Shared axes + one colorbar."""
    ok = np.isfinite(sig1) & np.isfinite(mu_rel) & np.isfinite(y) & (np.abs(y) <= CAP_PCT)
    xs, ys, yy, kk = np.abs(sig1)[ok], np.abs(mu_rel)[ok], y[ok], klass[ok]
    xhi = float(np.percentile(xs, 99)); yhi = float(np.percentile(ys, 99))
    edges = m052._q5x4_edges(yy); nb = max(edges.size - 1, 1)
    cmap = ListedColormap(m052.semantic_5x4_colors(nb)); norm = BoundaryNorm(edges, nb)
    fig, axes = plt.subplots(2, 2, figsize=(13.8, 10.4), sharex=True, sharey=True,
                             constrained_layout=True)
    sc = None
    for ax, ck in zip(axes.ravel(), CLASS_ORDER):
        s = (kk == ck) & (xs <= xhi) & (ys <= yhi)
        sc = ax.scatter(xs[s], ys[s], c=np.clip(yy[s], edges[0], edges[-1]),
                        s=5, alpha=0.55, linewidths=0, cmap=cmap, norm=norm,
                        rasterized=True)
        ax.set_xlim(0, xhi); ax.set_ylim(0, yhi)
        ax.set_title(f"{CLASS_NAME[ck]}   (n={int(s.sum()):,})", fontsize=15,
                     color="#555555" if ck == "stationary" else CLASS_COL[ck])
        ax.tick_params(labelsize=12); ax.grid(alpha=0.2)
    for ax in axes[-1, :]:
        ax.set_xlabel(r"scale slope  $|\sigma_1|$", fontsize=LBL_FS)
    for ax in axes[:, 0]:
        ax.set_ylabel(r"location slope  $|\mu_1/\bar{\mu}|$", fontsize=LBL_FS)
    cb = fig.colorbar(sc, ax=axes, ticks=edges[::4] if nb >= 8 else edges,
                      fraction=0.046, pad=0.02)
    cb.set_label("improvement [%]", fontsize=16); cb.ax.tick_params(labelsize=12)
    fig.savefig(out_png, dpi=170, bbox_inches="tight"); plt.close(fig)
    print(f"Saved: {out_png}")


def improvement_slope_and_class(sig1, mu_rel, y, klass, out_png):
    """Two clean panels carrying ONE message each (replaces the hard-to-read
    combined-M figure that mixed both messages onto a single sqrt(sig^2+mu^2) axis):

    (a) improvement vs the PERCENTILE RANK of |slope| magnitude, one curve for the
        SCALE slope |sigma_1| (red) and one for the LOCATION slope |mu_1/mubar|
        (blue). Ranking both slopes to a common 0-100 axis makes them directly
        comparable: the scale curve climbs steeply with its rank while the location
        curve stays flat / declines, so SCALE change binds improvement far more
        strongly. Marginal Spearman in the legend; partials (each slope controlling
        the other) in the legend title.

    (b) improvement by AIC change-type class (Figure-2 palette, notched boxes,
        median annotated). SCALE and LOCATION&SCALE sit clearly above STATIONARY and
        LOCATION in MEDIAN improvement -- the 508 ordering, in improvement units."""
    ok = np.isfinite(y) & (np.abs(y) <= CAP_PCT)
    a_s, a_m = np.abs(sig1), np.abs(mu_rel)
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(15.2, 6.4),
                                   gridspec_kw={"width_ratios": [1.18, 1.0]})

    # ---- (a) PARTIAL-dependence curves: effect of each slope with the OTHER held
    # to its middle tertile (so the curves are the partial effect, not the marginal
    # one -- marginally BOTH slopes rise because they co-vary; the honest 'scale >
    # location' claim lives in the partials). x = percentile rank of the slope
    # within the held-fixed subset; y = median improvement per decile. ----
    def rankpct(a, mask):
        r = np.full(a.shape, np.nan)
        v = a[mask]
        r[mask] = (_ranks(v) + 0.5) / v.size * 100.0
        return r
    edges = np.linspace(0, 100, 11)
    both_ok = ok & np.isfinite(a_s) & np.isfinite(a_m)
    ps_s = partial_spearman(y[both_ok], a_s[both_ok], a_m[both_ok])
    ps_m = partial_spearman(y[both_ok], a_m[both_ok], a_s[both_ok])
    for a_slope, a_other, col, name, ps in [
            (a_s, a_m, CLASS_COL["var"], r"scale $|\sigma_1|$", ps_s),
            (a_m, a_s, CLASS_COL["mean"], r"location $|\mu_1/\bar{\mu}|$", ps_m)]:
        t1, t2 = np.nanpercentile(a_other[both_ok], [100 / 3, 200 / 3])
        mk = both_ok & (a_other >= t1) & (a_other <= t2)   # hold OTHER slope ~fixed
        rp = rankpct(a_slope, mk)
        cx, cy = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            s = mk & (rp >= lo) & ((rp < hi) if hi < 100 else (rp <= hi))
            if s.sum() >= 30:
                cx.append(0.5 * (lo + hi)); cy.append(float(np.median(y[s])))
        axA.plot(cx, cy, "-o", color=col, lw=2.8, ms=6,
                 label=f"{name}   (partial Spearman {ps:+.2f})")
    axA.set_xlabel("percentile of |slope|  (other slope held fixed)", fontsize=LBL_FS)
    axA.set_ylabel("median improvement [%]", fontsize=LBL_FS)
    axA.tick_params(axis="both", labelsize=TICK_FS)
    axA.set_xlim(0, 100)
    leg = axA.legend(fontsize=13, loc="upper left")
    axA.grid(alpha=0.25)
    axA.set_title("(a) with the other slope held fixed, scale lifts improvement more",
                  fontsize=TITLE_FS)

    # ---- (b) improvement by change-type class (508 ordering, in improvement units) ----
    data = [y[ok & (klass == ck)] for ck in CLASS_ORDER]
    bp = axB.boxplot(data, notch=True, showfliers=False, widths=0.6,
                     medianprops=dict(color="black", lw=2.2), patch_artist=True,
                     whiskerprops=dict(color="#555555"),
                     capprops=dict(color="#555555"))
    for patch, ck in zip(bp["boxes"], CLASS_ORDER):
        patch.set_facecolor(CLASS_COL[ck]); patch.set_alpha(0.85)
        patch.set_edgecolor("#333333")
    meds = [float(np.median(d)) for d in data]
    for i, md in enumerate(meds, start=1):
        axB.text(i, md, f" {md:.1f}", ha="left", va="center",
                 fontsize=15, fontweight="bold", color="#111111")
    axB.set_xticks(range(1, len(CLASS_ORDER) + 1))
    axB.set_xticklabels([CLASS_NAME[ck] for ck in CLASS_ORDER],
                        fontsize=13, rotation=12)
    axB.set_ylabel("improvement [%]", fontsize=LBL_FS)
    axB.tick_params(axis="y", labelsize=TICK_FS)
    allv = np.concatenate(data)
    axB.set_ylim(float(np.percentile(allv, 3)), float(np.percentile(allv, 97)))
    axB.grid(axis="y", alpha=0.25)
    axB.set_title("(b) median improvement by class:  scale, both > stationary, location",
                  fontsize=TITLE_FS)

    fig.tight_layout()
    fig.savefig(out_png, dpi=170, bbox_inches="tight"); plt.close(fig)
    print(f"Saved: {out_png}")
    print("  class medians: " + "  ".join(
        f"{CLASS_NAME[ck]}={m:.2f}" for ck, m in zip(CLASS_ORDER, meds)))


def heat2d_both(sig_abs, mu_abs, imp, out_png):
    m = np.isfinite(sig_abs) & np.isfinite(mu_abs) & np.isfinite(imp) & (np.abs(imp) <= CAP_PCT)
    x, yv, z = sig_abs[m], mu_abs[m], imp[m]
    # clip extreme axes for stable bins
    xlo, xhi = np.percentile(x, [1, 99]); ylo, yhi = np.percentile(yv, [1, 99])
    NB = 12
    xe = np.unique(np.quantile(x[(x >= xlo) & (x <= xhi)], np.linspace(0, 1, NB + 1)))
    ye = np.unique(np.quantile(yv[(yv >= ylo) & (yv <= yhi)], np.linspace(0, 1, NB + 1)))
    xi = np.clip(np.digitize(x, xe) - 1, 0, xe.size - 2)
    yi = np.clip(np.digitize(yv, ye) - 1, 0, ye.size - 2)
    nx, ny = xe.size - 1, ye.size - 1
    ssum = np.zeros((ny, nx)); cnt = np.zeros((ny, nx))
    np.add.at(ssum, (yi, xi), z); np.add.at(cnt, (yi, xi), 1.0)
    mean2d = np.where(cnt >= 30, ssum / cnt, np.nan)
    edges = m052._q5x4_edges(mean2d[np.isfinite(mean2d)]); nb = max(edges.size - 1, 1)
    cmap = ListedColormap(m052.semantic_5x4_colors(nb)).copy(); cmap.set_bad("#ffffff")
    norm = BoundaryNorm(edges, nb)

    rs = spearman(x, z); rm = spearman(yv, z)                 # marginal
    rs_p = partial_spearman(z, x, yv); rm_p = partial_spearman(z, yv, x)  # partial

    fig, ax = plt.subplots(figsize=(9.2, 7.4))
    im = ax.imshow(np.ma.masked_invalid(np.clip(mean2d, edges[0], edges[-1])),
                   origin="lower", extent=[0, nx, 0, ny], aspect="auto",
                   cmap=cmap, norm=norm, interpolation="nearest")
    xt = np.arange(0, nx + 1, 2); yt = np.arange(0, ny + 1, 2)
    ax.set_xticks(xt); ax.set_xticklabels(["%.2g" % xe[i] for i in xt], fontsize=TICK_FS)
    ax.set_yticks(yt); ax.set_yticklabels(["%.3g" % ye[i] for i in yt], fontsize=TICK_FS)
    ax.set_xlabel(r"$|\sigma_1|$  (scale year-slope magnitude, log link)", fontsize=LBL_FS)
    ax.set_ylabel(r"$|\mu_1/\bar{\mu}|$  (location year-slope magnitude, relative)", fontsize=LBL_FS)
    cb = fig.colorbar(im, ax=ax, ticks=edges[::4] if nb >= 8 else edges)
    cb.set_label("mean improvement [%]", fontsize=16); cb.ax.tick_params(labelsize=14)
    ax.set_title("'location and scale' cells: which drives improvement?\n"
                 f"Spearman(imp,|$\\sigma_1$| scale)={rs:+.3f} (partial {rs_p:+.3f})   "
                 f"Spearman(imp,|$\\mu_1/\\bar{{\\mu}}$| location)={rm:+.3f} (partial {rm_p:+.3f})",
                 fontsize=TITLE_FS)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160, bbox_inches="tight"); plt.close(fig)
    print(f"Saved: {out_png}  (n={x.size})")
    print(f"  |sig|: marginal rho={rs:+.3f} partial={rs_p:+.3f} ; "
          f"|mu|: marginal rho={rm:+.3f} partial={rm_p:+.3f}")
    return rs, rs_p, rm, rm_p


def improvement_boxplot(y, klass, out_png, ok):
    """Standalone boxplot of improvement by change-type class (Figure-2 palette,
    black median, whiskers, no fliers), with a dashed horizontal line at the OVERALL
    median improvement as reference. No title. Used for both the AIC and the
    slope-threshold classification (distinguished by the file name)."""
    data = [y[ok & (klass == ck)] for ck in CLASS_ORDER]
    ref = float(np.median(y[ok]))
    fig, ax = plt.subplots(figsize=(9.2, 6.6))
    bp = ax.boxplot(data, widths=0.62, showfliers=False, patch_artist=True,
                    medianprops=dict(color="black", lw=2.6),
                    whiskerprops=dict(color="#555555", lw=1.4),
                    capprops=dict(color="#555555", lw=1.4))
    for patch, ck in zip(bp["boxes"], CLASS_ORDER):
        patch.set_facecolor(CLASS_COL[ck]); patch.set_alpha(0.9)
        patch.set_edgecolor("#333333")
    ax.axhline(ref, color="black", ls="--", lw=1.7, zorder=1)
    ax.set_xticks(range(1, len(CLASS_ORDER) + 1))
    ax.set_xticklabels(["stationary", "location", "scale", "location\n& scale"],
                       fontsize=19)
    ax.set_ylabel("relative SD improvement [%]", fontsize=21)
    ax.tick_params(axis="y", labelsize=18)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_png, dpi=170, bbox_inches="tight"); plt.close(fig)
    print(f"Saved: {out_png}  (ref median={ref:.2f}; "
          + "  ".join(f"{CLASS_NAME[ck]}={np.median(d):.2f}" for ck, d in zip(CLASS_ORDER, data)) + ")")


def main():
    argv = sys.argv[1:]
    scope = "ensured"
    if "--scope" in argv:
        i = argv.index("--scope"); scope = argv[i + 1]; del argv[i:i + 2]
    dat_dir = argv[0] if argv and not argv[0].startswith("--") else \
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
    out_dir = os.path.join(dat_dir, "511"); os.makedirs(out_dir, exist_ok=True)

    co = load_ln_coeffs(dat_dir)
    imp = load_improvement(dat_dir)
    cls = load_class(dat_dir)
    cids = co.index.intersection(imp.index).intersection(cls.index)
    if scope == "ensured":
        keep = excess_lib.load_flood_ensured_ids(dat_dir) - excess_lib.load_excluded_cells(dat_dir)
        cids = cids.intersection(pd.Index(list(keep)))
    co = co.loc[cids]
    y = imp.loc[cids].to_numpy()
    klass = cls.loc[cids].to_numpy()
    mu0 = co["truth_mu_0"].to_numpy(); mu1 = co["truth_mu_1"].to_numpy()
    sig1 = co["truth_sigma_1"].to_numpy()
    # location year-slope as a per-year RELATIVE rate (analog of the log-link
    # sigma_1): mu_rel = mu1 / mu_bar, where mu_bar = mu0 + mu1*YBAR is the location
    # MEAN LEVEL over the record (NOT the year-0 intercept mu0, which is a huge
    # extrapolation strongly anti-correlated with mu1 -- that broke the old mu1/mu0,
    # dropping/flipping the increasing-flow cells). Fit is on negated flow, so the
    # negation cancels in the ratio: mu_rel<0 = decreasing, >0 = increasing location.
    YBAR = 2040.5                                   # mean year over 1981..2100
    mu_bar = mu0 + mu1 * YBAR
    mu_rel = np.where(np.abs(mu_bar) > 1e-9, mu1 / mu_bar, np.nan)
    NAME = {"stationary": "stationary", "mean": "location", "var": "scale",
            "both": "location and scale"}
    for k in ("stationary", "mean", "var", "both"):
        print(f"  {NAME[k]:20s}: {int((klass == k).sum())}")

    is_var, is_mean, is_both = klass == "var", klass == "mean", klass == "both"
    rows = []
    # INDEPENDENT slope-threshold classification (both-fit slopes vs a noise floor),
    # computed up front so the sigslope/muslope curves can be drawn BOTH ways.
    ok_all = np.isfinite(y) & (np.abs(y) <= CAP_PCT)
    klass_sl, tau_s, tau_m = slope_reclassify(sig1, mu_rel, klass, ok_all)

    SIG_LBL = r"scale year-slope  $\sigma_1$"
    MU_LBL = r"location year-slope  $\mu_1/\bar{\mu}$"
    # --- AIC classification (077b): classes overlap in x, so NO threshold line ---
    r1 = hexfig_allclass(sig1, y, klass, SIG_LBL,
                "improvement vs scale year-slope (AIC classes)",
                os.path.join(out_dir, "improvement_vs_sigslope_aic.png"))
    rows.append(["aic_sigslope", *[f"{v:.4f}" for v in r1], int(y.size)])
    r2 = hexfig_allclass(mu_rel, y, klass, MU_LBL,
                "improvement vs location year-slope (AIC classes)",
                os.path.join(out_dir, "improvement_vs_muslope_aic.png"))
    rows.append(["aic_muslope", *[f"{v:.4f}" for v in r2], int(y.size)])
    # --- slope-threshold classification: curves CUT at +/-tau on this very axis ---
    hexfig_allclass(sig1, y, klass_sl, SIG_LBL,
                "improvement vs scale year-slope (slope-threshold classes)",
                os.path.join(out_dir, "improvement_vs_sigslope_slopecls.png"), tau=tau_s)
    hexfig_allclass(mu_rel, y, klass_sl, MU_LBL,
                "improvement vs location year-slope (slope-threshold classes)",
                os.path.join(out_dir, "improvement_vs_muslope_slopecls.png"), tau=tau_m)
    r3 = heat2d_both(np.abs(sig1[is_both]), np.abs(mu_rel[is_both]), y[is_both],
                     os.path.join(out_dir, "improvement_2d_both.png"))

    # (4) ALL flood-ensured cells on one scatter, coloured by change-type class
    # (Figure-2 palette), no median bin -- the reworked 079/excess_vs_sigslope idea
    class_scatter_sigslope(sig1, y, klass,
                           os.path.join(out_dir, "improvement_vs_sigslope_byclass.png"))

    # (5b) all four quantities in ONE figure: x=|sig1|, y=|mu_rel|, colour=improvement,
    # class = 2x2 small multiples
    improvement_slope_plane_byclass(sig1, mu_rel, y, klass,
                                    os.path.join(out_dir, "improvement_slopeplane_byclass.png"))
    # (5d) standalone class boxplots -- AIC and slope-threshold versions
    improvement_boxplot(y, klass, os.path.join(out_dir, "improvement_boxplot_aic.png"), ok_all)
    improvement_boxplot(y, klass_sl,
                        os.path.join(out_dir, "improvement_boxplot_slopecls.png"), ok_all)
    # (5c) scale slope, location slope, improvement on 3 EQUIVALENT axes (Z=improvement)
    improvement_surface3d(sig1, mu_rel, y,
                          os.path.join(out_dir, "improvement_surface3d.png"))

    # (6) redraw the other class-based figures with the slope-threshold classes
    # (klass_sl/tau computed up front). The partial-curve panel (a) is class-
    # INDEPENDENT, so it is unchanged.
    print(f"Slope-threshold reclassification (noise floor = stationary p{NOISE_PCT:.0f}): "
          f"tau_sigma={tau_s:.5f}  tau_mu={tau_m:.5f}")
    NAME2 = {"stationary": "stationary", "mean": "location", "var": "scale",
             "both": "location and scale"}
    for k in ("stationary", "mean", "var", "both"):
        s = ok_all & (klass_sl == k)
        med = float(np.median(y[s])) if s.any() else float("nan")
        print(f"  {NAME2[k]:20s} n={int(s.sum()):8d}  median improvement={med:.2f}")
    # cross-tab AIC class (rows) vs slope class (cols) on the plotted cells
    ct = pd.crosstab(pd.Series(klass[ok_all], name="AIC"),
                     pd.Series(klass_sl[ok_all], name="slope"))
    print("AIC-class (rows) x slope-class (cols) cross-tab:")
    print(ct.reindex(index=["stationary", "mean", "var", "both"],
                     columns=["stationary", "mean", "var", "both"], fill_value=0))
    improvement_slope_plane_byclass(sig1, mu_rel, y, klass_sl,
                                    os.path.join(out_dir, "improvement_slopeplane_slopecls.png"))
    old = os.path.join(out_dir, "improvement_vs_totalchange_byclass.png")
    if os.path.exists(old):
        os.remove(old)

    with open(os.path.join(out_dir, "slope_stats.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["figure", "spearman_signed", "spearman_abs", "n"])
        w.writerows(rows)
        w.writerow([])
        w.writerow(["both_cells: sig marginal/partial", f"{r3[0]:.4f}", f"{r3[1]:.4f}"])
        w.writerow(["both_cells: mu  marginal/partial", f"{r3[2]:.4f}", f"{r3[3]:.4f}"])
    print(f"Saved: {os.path.join(out_dir, 'slope_stats.csv')}")


if __name__ == "__main__":
    main()
