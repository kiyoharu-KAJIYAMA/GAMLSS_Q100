"""
fig04_parent_by_fit_boxplots.py
All-grid summary of the 3x3 robustness experiment: for each PARENT (truth)
distribution {st120, LIN, QD} and each FITTED model {st, ln, qd}, report the
per-cell SD / bias / RMSE of the relative Q100 error as a TABLE and a BOXPLOT,
for ALL grid cells and for the FLOOD-RELEVANT subset.

This is the all-grid counterpart of 047 (which uses the lightweight 3x3 cells).

Parents (rows) and their source files:
  st120 = stationary parent (truth fit on 120 yr)  -> 045b/summary_allgrid_st.csv
  LIN   = linear parent                            -> 054/bias_allgrid.csv
  QD    = quadratic parent                         -> 045b/summary_allgrid_qd.csv
Each file has per-cell {fit}_bias and {fit}_sd (fractions of true Q100); RMSE is
computed per cell as sqrt(bias^2 + sd^2). If an {fit}=ad (AIC-adaptive) column is
present in ALL parent files (i.e. 045b was re-run with the adaptive estimator),
it is included automatically as a fourth fitted model.

Flood relevance (matches 098/103): keep if
  (q90 >= X1 OR max_amax >= X2) AND uparea_km2 >= A   (defaults 50/100/50),
from 091/change_type_cells.csv. Arid garbage (|bias%| or |sd%| > CAP_PCT) is
dropped per cell.

Outputs (to <dat_dir>/107/):
  summary_3x3_all.csv  / summary_3x3_flood.csv     median [q25,q75] per parent x fit
  boxplot_3x3_all.png  / boxplot_3x3_flood.png     SD / Bias / RMSE boxplots
  console: the same medians

Usage:
  python3 fig04_parent_by_fit_boxplots.py [dat_dir] [X1] [X2] [A]
"""
import os
import sys
import csv
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import importlib
excess_lib = importlib.import_module("common_target_cells")

CAP_PCT = 300.0
PARENTS = [("st", "Stationary (120)"), ("ln", "Linear"), ("qd", "Quadratic")]
FITS_BASE = ["st", "ln", "qd"]
FIT_NAME = {"st": "stationary", "ln": "linear", "qd": "quadratic", "ad": "AIC-adaptive"}
FIT_COL = {"st": "#ffc2c7", "ln": "#a8d7bb", "qd": "#a4a1ff", "ad": "#9e9e9e"}
METRICS = [("sd", "SD"), ("bias", "Bias"), ("rmse", "RMSE")]


def parent_files(dat_dir):
    return {
        "st": os.path.join(dat_dir, "045b", "summary_allgrid_st.csv"),
        "ln": os.path.join(dat_dir, "054", "bias_allgrid.csv"),
        "qd": os.path.join(dat_dir, "045b", "summary_allgrid_qd.csv"),
    }


def load_flood_ids(dat_dir, x1, x2, a_up):
    cells_csv = os.path.join(dat_dir, "091", "change_type_cells.csv")
    if not os.path.exists(cells_csv):
        return None
    ids = set()
    with open(cells_csv, newline="") as f:
        for row in csv.DictReader(f):
            def num(k):
                v = row.get(k, "")
                return float(v) if v not in ("", "NA", "nan") else float("nan")
            if ((num("q90") >= x1) or (num("max_amax") >= x2)) and (num("uparea_km2") >= a_up):
                ids.add(int(row["cell_id"]))
    return ids


def load_parent(path, fits):
    """Return dict: cid array + per-fit {'bias','sd','rmse'} arrays (in %)."""
    cid = []
    raw = {fit: {"bias": [], "sd": []} for fit in fits}
    with open(path, newline="") as f:
        rd = csv.DictReader(f)
        for row in rd:
            try:
                c = int(row["cell_id"])
            except (KeyError, ValueError):
                continue
            cid.append(c)
            for fit in fits:
                def g(k):
                    v = row.get(k, "")
                    return float(v) if v not in ("", "NA", "nan") else np.nan
                raw[fit]["bias"].append(g(f"{fit}_bias"))
                raw[fit]["sd"].append(g(f"{fit}_sd"))
    out = {"cid": np.array(cid)}
    for fit in fits:
        b = np.array(raw[fit]["bias"]) * 100.0
        s = np.array(raw[fit]["sd"]) * 100.0
        out[fit] = {"bias": b, "sd": s, "rmse": np.sqrt(b ** 2 + s ** 2)}
    return out


def ad_file(dat_dir, parent):
    return os.path.join(dat_dir, "108", f"summary_allgrid_ad_{parent}.csv")


def detect_fits(dat_dir):
    """st/ln/qd always; add 'ad' (AIC-adaptive) only if 108 produced
    summary_allgrid_ad_<p>.csv for EVERY parent (run 108_pbs.sh + step20_combine_aic_adaptive
    for st, ln and qd)."""
    fits = list(FITS_BASE)
    if all(os.path.exists(ad_file(dat_dir, pk)) for pk, _ in PARENTS):
        fits.append("ad")
    return fits


def attach_ad(data, dat_dir):
    """Join 108's adaptive bias/sd onto each parent's cells by cell_id."""
    for pk, _ in PARENTS:
        m = {}
        with open(ad_file(dat_dir, pk), newline="") as f:
            for row in csv.DictReader(f):
                try:
                    cid = int(row["cell_id"])
                except (KeyError, ValueError):
                    continue
                def g(k):
                    v = row.get(k, "")
                    return float(v) if v not in ("", "NA", "nan") else np.nan
                m[cid] = (g("ad_bias"), g("ad_sd"))
        cids = data[pk]["cid"]
        b = np.array([m.get(c, (np.nan, np.nan))[0] for c in cids]) * 100.0
        s = np.array([m.get(c, (np.nan, np.nan))[1] for c in cids]) * 100.0
        data[pk]["ad"] = {"bias": b, "sd": s, "rmse": np.sqrt(b ** 2 + s ** 2)}


def valid_mask(d, fit):
    """Finite and within CAP for both bias and sd of this fit."""
    b, s = d[fit]["bias"], d[fit]["sd"]
    return np.isfinite(b) & np.isfinite(s) & (np.abs(b) <= CAP_PCT) & (np.abs(s) <= CAP_PCT)


def collect(data, fits, cell_subset, bad=frozenset()):
    """For each parent, fit, metric -> 1D array over kept cells.
    cell_subset: None (all) or a set of flood cell_ids. bad: cell_ids to drop
    (031 negative-AMAX reverse-flow cells)."""
    box = {}
    for pk, _ in PARENTS:
        d = data[pk]
        in_view = np.ones(len(d["cid"]), bool) if cell_subset is None \
            else np.array([c in cell_subset for c in d["cid"]])
        if bad:
            in_view &= np.array([c not in bad for c in d["cid"]])
        for fit in fits:
            keep = in_view & valid_mask(d, fit)
            for mk, _ in METRICS:
                box[(pk, fit, mk)] = d[fit][mk][keep]
    return box


def _avg_ranks(a):
    """Average (midrank) ranks of a 1D array + tie-group sizes (for the tie
    correction). Pure numpy so scipy is not required."""
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(a.size, float)
    ranks[order] = np.arange(1, a.size + 1)
    uniq, inv = np.unique(a, return_inverse=True)
    sums = np.bincount(inv, weights=ranks)
    cnts = np.bincount(inv)
    return (sums / cnts)[inv], cnts


def wilcoxon_p(x, y):
    """Two-sided paired Wilcoxon signed-rank p-value (normal approximation with
    tie correction; fine for the large n here). x, y: per-cell paired metrics."""
    d = x - y
    d = d[np.isfinite(d)]
    d = d[d != 0.0]
    n = d.size
    if n < 10:
        return np.nan
    ranks, cnts = _avg_ranks(np.abs(d))
    w = ranks[d > 0].sum()
    mu = n * (n + 1) / 4.0
    var = n * (n + 1) * (2 * n + 1) / 24.0 - (cnts ** 3 - cnts).sum() / 48.0
    if var <= 0:
        return np.nan
    z = (w - mu) / math.sqrt(var)
    return math.erfc(abs(z) / math.sqrt(2.0))


def stars(p):
    if not np.isfinite(p):
        return ""
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "ns"


def compute_pvals(data, fits, cell_subset, bad=frozenset()):
    """Paired (same cells; joint validity mask) two-sided Wilcoxon of each
    non-stationary fit vs the stationary fit, per parent x metric."""
    pv = {}
    for pk, _ in PARENTS:
        d = data[pk]
        in_view = np.ones(len(d["cid"]), bool) if cell_subset is None \
            else np.array([c in cell_subset for c in d["cid"]])
        if bad:
            in_view &= np.array([c not in bad for c in d["cid"]])
        for fit in fits:
            if fit == "st":
                continue
            keep = in_view & valid_mask(d, "st") & valid_mask(d, fit)
            for mk, _ in METRICS:
                pv[(pk, fit, mk)] = wilcoxon_p(d[fit][mk][keep], d["st"][mk][keep])
    return pv


def collect_paired_drmse(data, fits, cell_subset, bad=frozenset()):
    """Per-cell PAIRED RMSE difference (stationary - fit; positive = the
    nonstationary fit improves on stationary), joint validity mask, per parent."""
    out = {}
    for pk, _ in PARENTS:
        d = data[pk]
        in_view = np.ones(len(d["cid"]), bool) if cell_subset is None \
            else np.array([c in cell_subset for c in d["cid"]])
        if bad:
            in_view &= np.array([c not in bad for c in d["cid"]])
        for fit in fits:
            if fit == "st":
                continue
            keep = in_view & valid_mask(d, "st") & valid_mask(d, fit)
            out[(pk, fit)] = d["st"]["rmse"][keep] - d[fit]["rmse"][keep]
    return out


def write_pvals(pv, fits, out_csv):
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["parent", "fit"] + [f"{mk}_p" for mk, _ in METRICS]
                   + [f"{mk}_sig" for mk, _ in METRICS])
        for pk, _ in PARENTS:
            for fit in fits:
                if fit == "st":
                    continue
                ps = [pv.get((pk, fit, mk), np.nan) for mk, _ in METRICS]
                w.writerow([pk, fit] + [f"{p:.3e}" if np.isfinite(p) else "NA" for p in ps]
                           + [stars(p) for p in ps])
    print(f"Saved: {out_csv}")


def write_table(box, fits, out_csv):
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["parent", "fit"]
                   + [f"{mk}_{q}" for mk, _ in METRICS for q in ("median", "q25", "q75")])
        for pk, _ in PARENTS:
            for fit in fits:
                rowv = [pk, fit]
                for mk, _ in METRICS:
                    v = box[(pk, fit, mk)]
                    if v.size:
                        rowv += [f"{np.median(v):.3f}", f"{np.percentile(v,25):.3f}",
                                 f"{np.percentile(v,75):.3f}"]
                    else:
                        rowv += ["NA", "NA", "NA"]
                w.writerow(rowv)
    print(f"Saved: {out_csv}")


def print_table(box, fits, label):
    print(f"\n=== 3x3 medians [{label}] (relative Q100 error, %) ===")
    print(f"{'parent':12s} {'fit':12s} {'SD':>8s} {'Bias':>8s} {'RMSE':>8s}")
    for pk, pn in PARENTS:
        for fit in fits:
            vals = {mk: box[(pk, fit, mk)] for mk, _ in METRICS}
            med = {mk: (np.median(v) if v.size else float('nan')) for mk, v in vals.items()}
            print(f"{pn:12s} {FIT_NAME[fit]:12s} "
                  f"{med['sd']:8.2f} {med['bias']:8.2f} {med['rmse']:8.2f}")


def make_boxplot(box, fits, out_png, label, pvals=None):
    TITLE_FS, LABEL_FS, TICK_FS, LEG_FS = 20, 16, 14, 13
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.2))
    nf = len(fits)
    box_w = 0.8 / nf
    offs = (np.arange(nf) - (nf - 1) / 2.0) * box_w
    for ax, (mk, mlab) in zip(axes, METRICS):
        # PER-PANEL robust limits (5-95 whisker span of this metric only), so the
        # huge Linear bias/RMSE no longer compresses the other panels' scales.
        vals_mk = [box[(pk, fit, mk)] for pk, _ in PARENTS for fit in fits
                   if box[(pk, fit, mk)].size]
        lo = min(np.percentile(v, 5) for v in vals_mk)
        hi = max(np.percentile(v, 95) for v in vals_mk)
        # symlog when one group dominates the range (e.g. Linear-parent bias/RMSE):
        # linear near 0 keeps small differences readable, log tail keeps the big
        # boxes on-plot. linthresh = typical (median) magnitude of the box edges.
        edges75 = [np.percentile(np.abs(v), 75) for v in vals_mk]
        linthresh = max(1.0, float(np.median(edges75)))
        use_symlog = hi > 6.0 * linthresh
        whisk_top = {}
        for gi, (pk, _) in enumerate(PARENTS):
            for fi, fit in enumerate(fits):
                v = box[(pk, fit, mk)]
                if v.size == 0:
                    continue
                xpos = gi + 1 + offs[fi]
                bp = ax.boxplot(v, positions=[xpos], widths=box_w * 0.9,
                                patch_artist=True, showfliers=False, whis=(5, 95),
                                medianprops=dict(color="black", linewidth=1.3))
                for b in bp["boxes"]:
                    b.set(facecolor=FIT_COL[fit], alpha=0.85, edgecolor="#555555")
                for el in bp["whiskers"] + bp["caps"]:
                    el.set(color="#555555")
                whisk_top[(pk, fit)] = (xpos, np.percentile(v, 95))
        # zero = ideal reference on EVERY panel: long-dash red, unlike any other line
        ax.axhline(0.0, color="#d62728", linewidth=1.6, linestyle=(0, (7, 3)), zorder=1)
        if use_symlog:
            ax.set_yscale("symlog", linthresh=linthresh)
        pad = 0.06 * (hi - lo)
        ax.set_ylim(min(lo - pad, -0.02 * hi), hi + (2.2 * pad if pvals else pad))
        # significance stars vs the stationary fit, above each non-st whisker
        if pvals:
            for (pk, fit), (xpos, ytop) in whisk_top.items():
                if fit == "st":
                    continue
                s = stars(pvals.get((pk, fit, mk), np.nan))
                if s:
                    ax.annotate(s, xy=(xpos, ytop), xytext=(0, 3),
                                textcoords="offset points", ha="center", va="bottom",
                                fontsize=11 if s == "ns" else 14, color="#333333")
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels([pn for _, pn in PARENTS])
        ax.set_xlim(0.5, 3.5)
        ax.set_xlabel("Parent (truth) distribution", fontsize=LABEL_FS)
        ax.set_ylabel(f"{mlab} of relative Q100 error [%]", fontsize=LABEL_FS)
        ax.set_title(mlab, fontsize=TITLE_FS)
        ax.tick_params(axis="both", labelsize=TICK_FS)
        ax.grid(axis="y", alpha=0.25)
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=FIT_COL[f], alpha=0.85,
                             edgecolor="#555555") for f in fits]
    leg = axes[0].legend(handles, [FIT_NAME[f] for f in fits], title="Fitted model",
                         loc="lower left", fontsize=LEG_FS)
    leg.get_title().set_fontsize(LEG_FS + 1)
    fig.suptitle(f"3x3 robustness ({label})", fontsize=TITLE_FS + 2, y=1.02)
    if pvals:
        fig.text(0.5, -0.02,
                 "* p<0.05  ** p<0.01  *** p<0.001  (two-sided paired Wilcoxon "
                 "signed-rank vs the stationary fit, same cells)",
                 ha="center", fontsize=12, color="#333333")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_png}")


def make_paired_box(paired, fits, out_png, label):
    """STANDALONE paired-difference boxplot: per-cell dRMSE (stationary - fit;
    positive = the nonstationary fit improves on stationary), per parent."""
    TITLE_FS, LABEL_FS, TICK_FS, LEG_FS = 20, 16, 14, 13
    fits_ns = [f for f in fits if f != "st"]
    nfd = len(fits_ns)
    bw = 0.8 / nfd
    offd = (np.arange(nfd) - (nfd - 1) / 2.0) * bw
    vals_d = [paired[(pk, f)] for pk, _ in PARENTS for f in fits_ns
              if paired[(pk, f)].size]
    lo = min(np.percentile(v, 5) for v in vals_d)
    hi = max(np.percentile(v, 95) for v in vals_d)
    pad = 0.06 * (hi - lo)
    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    for gi, (pk, _) in enumerate(PARENTS):
        for fi, fit in enumerate(fits_ns):
            v = paired[(pk, fit)]
            if v.size == 0:
                continue
            bp = ax.boxplot(v, positions=[gi + 1 + offd[fi]], widths=bw * 0.9,
                            patch_artist=True, showfliers=False, whis=(5, 95),
                            medianprops=dict(color="black", linewidth=1.3))
            for b in bp["boxes"]:
                b.set(facecolor=FIT_COL[fit], alpha=0.85, edgecolor="#555555")
            for el in bp["whiskers"] + bp["caps"]:
                el.set(color="#555555")
    ax.axhline(0.0, color="#d62728", linewidth=1.6, linestyle=(0, (7, 3)), zorder=1)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels([pn for _, pn in PARENTS])
    ax.set_xlim(0.5, 3.5)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel("Parent (truth) distribution", fontsize=LABEL_FS)
    ax.set_ylabel("Paired RMSE reduction vs stationary [% points]", fontsize=LABEL_FS)
    ax.set_title(f"$\\Delta$RMSE (paired, {label})", fontsize=TITLE_FS)
    ax.tick_params(axis="both", labelsize=TICK_FS)
    ax.grid(axis="y", alpha=0.25)
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=FIT_COL[f], alpha=0.85,
                             edgecolor="#555555") for f in fits_ns]
    leg = ax.legend(handles, [FIT_NAME[f] for f in fits_ns], title="Fitted model",
                    loc="lower left", fontsize=LEG_FS)
    leg.get_title().set_fontsize(LEG_FS + 1)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_png}")


def make_paired_cdf(paired, fits, out_png, label):
    """STANDALONE empirical CDF of the paired dRMSE. Colour = fitted model,
    linestyle = parent. The share of cells left of 0 is the fraction NOT
    improved by that fit; curves fully right of 0 = systematic improvement."""
    TITLE_FS, LABEL_FS, TICK_FS, LEG_FS = 20, 16, 14, 13
    PSTYLE = {"st": "-", "ln": "--", "qd": ":"}
    fits_ns = [f for f in fits if f != "st"]
    qs = np.linspace(0.0, 100.0, 801)
    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    xs_all = []
    for pk, _ in PARENTS:
        for fit in fits_ns:
            v = paired[(pk, fit)]
            if v.size == 0:
                continue
            x = np.percentile(v, qs)
            ax.plot(x, qs / 100.0, color=FIT_COL[fit], linestyle=PSTYLE[pk],
                    linewidth=1.9)
            xs_all.append((np.percentile(v, 5), np.percentile(v, 95)))
    ax.axvline(0.0, color="#d62728", linewidth=1.6, linestyle=(0, (7, 3)), zorder=1)
    # x-range: 5-95 pct span, with the left end FLOORED so one heavy negative
    # tail (quadratic-parent linear fit) cannot crush the readable region near 0
    hi = max(x1 for _, x1 in xs_all)
    lo = max(min(x0 for x0, _ in xs_all), -1.6 * hi)
    pad = 0.05 * (hi - lo)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Paired RMSE reduction vs stationary [% points]", fontsize=LABEL_FS)
    ax.set_ylabel("CDF", fontsize=LABEL_FS)
    ax.set_title(f"$\\Delta$RMSE CDF (paired, {label})", fontsize=TITLE_FS)
    ax.tick_params(axis="both", labelsize=TICK_FS)
    ax.grid(alpha=0.25)
    fit_h = [plt.Line2D([0], [0], color=FIT_COL[f], linewidth=2.5) for f in fits_ns]
    par_h = [plt.Line2D([0], [0], color="#444444", linestyle=PSTYLE[pk], linewidth=2.0)
             for pk, _ in PARENTS]
    leg1 = ax.legend(fit_h, [FIT_NAME[f] for f in fits_ns], title="Fitted model",
                     loc="upper left", fontsize=LEG_FS)
    leg1.get_title().set_fontsize(LEG_FS + 1)
    ax.add_artist(leg1)
    leg2 = ax.legend(par_h, [pn for _, pn in PARENTS], title="Parent (truth)",
                     loc="lower right", fontsize=LEG_FS)
    leg2.get_title().set_fontsize(LEG_FS + 1)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_png}")


def main():
    dat_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "data")
    x1 = float(sys.argv[2]) if len(sys.argv) > 2 else excess_lib.FLOOD_Q90
    x2 = float(sys.argv[3]) if len(sys.argv) > 3 else excess_lib.FLOOD_MAX
    a_up = float(sys.argv[4]) if len(sys.argv) > 4 else excess_lib.FLOOD_UPAREA

    files = parent_files(dat_dir)
    for pk, path in files.items():
        if not os.path.exists(path):
            sys.exit(f"Missing parent file for {pk}: {path}\n"
                     f"  st/qd <- run 045b + 045c; ln <- run 054.")
    out_dir = os.path.join(dat_dir, "107")
    os.makedirs(out_dir, exist_ok=True)

    fits = detect_fits(dat_dir)
    print(f"Fitted models: {fits}")
    data = {pk: load_parent(files[pk], FITS_BASE) for pk, _ in PARENTS}
    if "ad" in fits:
        attach_ad(data, dat_dir)
    bad = excess_lib.load_excluded_cells(dat_dir)
    if bad:
        print(f"Excluding {len(bad)} cells (031 reverse-flow + no-flood-regime "
              f"dry_year>{excess_lib.DRY_MAX}) from all views.")
    for pk, _ in PARENTS:
        print(f"  parent {pk}: {len(data[pk]['cid'])} cells")

    flood = load_flood_ids(dat_dir, x1, x2, a_up)
    views = [("all", None)]
    if flood is not None:
        print(f"Flood-relevant cell ids: {len(flood)}")
        views.append(("flood", flood))
    else:
        print("NOTE: 091/change_type_cells.csv not found -> all-grid view only.")

    for label, subset in views:
        box = collect(data, fits, subset, bad)
        print_table(box, fits, label)
        write_table(box, fits, os.path.join(out_dir, f"summary_3x3_{label}.csv"))
        pvals = compute_pvals(data, fits, subset, bad)
        write_pvals(pvals, fits, os.path.join(out_dir, f"signif_vs_stationary_{label}.csv"))
        # stars are NOT drawn on the figure (pvals stay in the CSV above)
        paired = collect_paired_drmse(data, fits, subset, bad)
        for (pk, fit), v in sorted(paired.items()):
            if v.size:
                print(f"  paired dRMSE [{label}] {pk}/{fit}: median={np.median(v):+.2f}pt "
                      f"improved={100.0 * (v > 0).mean():.1f}%")
        make_boxplot(box, fits, os.path.join(out_dir, f"boxplot_3x3_{label}.png"), label)
        make_paired_box(paired, fits,
                        os.path.join(out_dir, f"boxplot_paired_drmse_{label}.png"), label)
        make_paired_cdf(paired, fits,
                        os.path.join(out_dir, f"cdf_paired_drmse_{label}.png"), label)


if __name__ == "__main__":
    main()
