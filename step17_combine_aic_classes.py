"""
step17_combine_aic_classes.py
Combine the 077 AIC-decomposition chunks and LINK the nonstationarity type to
BOTH the excess SD improvement AND the hydroclimate -- i.e. test the full chain

    hydroclimate (070)  ->  AIC nonstationarity type (mean / variance / both)  ->  excess improvement (050)

077 classified each flood-relevant cell by which GU nonstationarity AIC justifies
(stationary / mean-only / var-only / both) and recorded daic_mean, daic_var
(= aic_st - aic_*, >0 means that nonstationarity beats stationary). Here we:

  1. concatenate 077/aic/chunk_*.csv  -> 077/aic_allgrid.csv
  2. EXCESS vs AIC class : is the SD reduction larger where AIC justifies a
     VARIANCE trend than where it justifies only a MEAN trend?  (the decisive
     test of 076c's finding that variance non-stationarity is the real driver)
  3. excess vs daic_var and excess vs daic_mean (Spearman, continuous version)
  4. HYDROCLIMATE per AIC class : are the var-class cells the arid / flashy /
     small-basin ones?  (the left half of the chain)

excess = improve_lin_sd - baseline, baseline = median improvement over Slater
"stationary"-type cells (excess_lib), so it matches 076b/076c.

Inputs:
  <dat_dir>/077/aic/chunk_*.csv
  <dat_dir>/050/summary_allgrid.csv
  <dat_dir>/091/change_type_cells.csv     (Slater type for baseline, climate)
  <dat_dir>/070/timeseries_tests.csv       (hydroclimate predictors)
Outputs (<dat_dir>/079/):
  aic_allgrid.csv               combined 077 table
  excess_by_aicclass.png        boxplot of excess by AIC class x climate
  excess_vs_daic.png            excess vs daic_var and daic_mean (binned median)
  class_summary.csv             per AIC class: n, median excess, median hydroclimate
Usage:
  python3 step17_combine_aic_classes.py [dat_dir]
"""
import os
import sys
import csv
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

import importlib
excess_lib = importlib.import_module("common_target_cells")

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAVE_CARTOPY = True
except ImportError:
    HAVE_CARTOPY = False

RES = 0.1
AIC_CLASSES = ["stationary", "mean", "var", "both"]
# semantic colours: variance (the key driver) red, mean blue, both purple, stat grey
CLASS_COLOR = {"stationary": "#bdbdbd", "mean": "#2166ac",
               "var": "#d6604d", "both": "#762a83"}
HYDRO = ["var_ratio", "cv", "zero_flow_rate", "mean_outflow", "uparea_km2"]
# display names (CSV keys stay as-is for 070 compatibility). zero_flow_rate is the
# fraction of the 120 ANNUAL MAXIMA below 1 m3/s -> "dry years with no flood", not
# the usual daily intermittency, so it is shown as dry_year_fraction.
HYDRO_LABEL = {"var_ratio": "var_ratio", "cv": "cv",
               "zero_flow_rate": "dry_year_fraction",
               "mean_outflow": "mean_outflow", "uparea_km2": "uparea_km2"}


def sfloat(v):
    try:
        x = float(v)
        return x if np.isfinite(x) else np.nan
    except (TypeError, ValueError):
        return np.nan


def combine_aic(aic_dir, out_csv):
    """Concatenate 077 chunks -> dict cid->row, and write the combined CSV."""
    files = sorted(glob.glob(os.path.join(aic_dir, "chunk_*.csv")))
    if not files:
        sys.exit(f"No 077 chunks in {aic_dir} (run 078_pbs.sh first)")
    d = {}
    header = None
    with open(out_csv, "w", newline="") as fout:
        w = csv.writer(fout)
        for fp in files:
            with open(fp, newline="") as f:
                r = csv.DictReader(f)
                if header is None:
                    header = r.fieldnames
                    w.writerow(header)
                for row in r:
                    try:
                        cid = int(row["cell_id"])
                    except (ValueError, KeyError, TypeError):
                        continue
                    d[cid] = row
                    w.writerow([row.get(h, "") for h in header])
    print(f"Combined {len(files)} chunks -> {out_csv} ({len(d)} cells)")
    return d


def keyed(path):
    d = {}
    if not os.path.exists(path):
        return d
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                d[int(row["cell_id"])] = row
            except (ValueError, KeyError):
                pass
    return d


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    dat_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "..", "data")
    aic_dir = os.path.join(dat_dir, "077", "aic")
    summ = os.path.join(dat_dir, "050", "summary_allgrid.csv")
    cells = os.path.join(dat_dir, "091", "change_type_cells.csv")
    ts = os.path.join(dat_dir, "070", "timeseries_tests.csv")
    out_dir = os.path.join(dat_dir, "079")
    os.makedirs(out_dir, exist_ok=True)

    aic = combine_aic(aic_dir, os.path.join(out_dir, "aic_allgrid.csv"))
    c91 = keyed(cells)
    t70 = keyed(ts)
    stype_map = {cid: r.get("type", "") for cid, r in c91.items()}
    imp_map = {}
    with open(summ, newline="") as f:
        for row in csv.DictReader(f):
            try:
                imp_map[int(row["cell_id"])] = float(row["improve_lin_sd"])
            except (ValueError, KeyError):
                pass

    # merge on the 077 (flood-relevant) cells
    cid_l, imp_l, klass_l, bm_l, clim_l = [], [], [], [], []
    iy_l, ix_l, dvar_l, dmean_l, slope_l = [], [], [], [], []
    hyd = {k: [] for k in HYDRO}
    for cid, row in aic.items():
        imp = imp_map.get(cid, np.nan)
        if not np.isfinite(imp):
            continue
        cid_l.append(cid); imp_l.append(imp)
        klass_l.append(row.get("class", ""))
        bm_l.append(row.get("best_model", ""))
        try:
            iy_l.append(int(row["iy"])); ix_l.append(int(row["ix"]))
        except (ValueError, KeyError, TypeError):
            iy_l.append(-1); ix_l.append(-1)
        dvar_l.append(sfloat(row.get("daic_var")))
        dmean_l.append(sfloat(row.get("daic_mean")))
        slope_l.append(sfloat(row.get("sig_slope")))
        clim_l.append(c91.get(cid, {}).get("climate", "?"))
        r70 = t70.get(cid, {})
        for k in HYDRO:
            hyd[k].append(sfloat(r70.get(k)))

    cid = np.array(cid_l); imp = np.array(imp_l, float)
    klass = np.array(klass_l); bm = np.array(bm_l); clim = np.array(clim_l)
    iy = np.array(iy_l); ix = np.array(ix_l)
    dvar = np.array(dvar_l, float); dmean = np.array(dmean_l, float)
    slope = np.array(slope_l, float)
    for k in HYDRO:
        hyd[k] = np.array(hyd[k], float)

    bad_rev = excess_lib.load_bad_cells(dat_dir)        # reverse-flow (always invalid)
    deg = excess_lib.load_degenerate_cells(dat_dir)     # no-flood-regime (dry_year>DRY_MAX)
    cap_ok = np.abs(imp) <= excess_lib.CAP_PCT
    keep_pre = cap_ok & ~np.isin(cid, list(bad_rev))    # CAP + reverse only (KEEPS dry cells)
    keep = keep_pre & ~np.isin(cid, list(deg))          # + drop no-flood-regime = flood-ensured
    print(f"Excluding {int((cap_ok & ~keep).sum())} cells "
          f"(031 reverse-flow + no-flood-regime dry_year>{excess_lib.DRY_MAX}).")

    # snapshot of the PRE-dry set (CAP + reverse only) for the dry-threshold sweep (3g),
    # so it can span the FULL dry range (incl. dry>0.5) that justifies the flood-ensured cut.
    cid_pd, klass_pd, slope_pd = cid[keep_pre], klass[keep_pre], slope[keep_pre]
    dvar_pd, zf_pd = dvar[keep_pre], hyd["zero_flow_rate"][keep_pre]
    _, excess_pd = excess_lib.baseline_excess(cid_pd, imp[keep_pre], stype_map)

    cid, imp, klass, bm, clim = cid[keep], imp[keep], klass[keep], bm[keep], clim[keep]
    iy, ix = iy[keep], ix[keep]
    dvar, dmean, slope = dvar[keep], dmean[keep], slope[keep]
    for k in HYDRO:
        hyd[k] = hyd[k][keep]
    baseline, excess = excess_lib.baseline_excess(cid, imp, stype_map)
    print(f"Cells: {len(excess)}  baseline={baseline:.2f}%")

    # ---------- class composition ----------
    print("AIC class composition:")
    for c in AIC_CLASSES:
        n = int(np.sum(klass == c))
        print(f"  {c:11s} {n:8d}  ({100*n/len(klass):5.1f}%)")

    # ---------- (1) excess by AIC class x climate ----------
    fig, ax = plt.subplots(figsize=(10, 6))
    data, labels, pos, cols = [], [], [], []
    palette = {"low": "#e08214", "high": "#2166ac"}
    p = 1
    for c in AIC_CLASSES:
        for cl in ("low", "high"):
            d = excess[(klass == c) & (clim == cl)]
            data.append(d if len(d) else [np.nan]); labels.append(f"{c}\n{cl}")
            pos.append(p); cols.append(palette[cl]); p += 1
        p += 0.6
    bp = ax.boxplot(data, positions=pos, widths=0.7, showfliers=False, patch_artist=True)
    for patch, c in zip(bp["boxes"], cols):
        patch.set_facecolor(c); patch.set_alpha(0.6)
    ax.set_xticks(pos); ax.set_xticklabels(labels, fontsize=8)
    ax.axhline(0, color="k", ls="--", lw=1, label="baseline (excess=0)")
    ax.set_ylabel("Excess SD improvement [pt]")
    ax.set_title(f"Excess by AIC nonstationarity class (baseline={baseline:.1f}%)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "excess_by_aicclass.png"), dpi=180)
    plt.close(fig)
    print("Saved: excess_by_aicclass.png")
    print("Median EXCESS [pt] by AIC class:")
    for c in AIC_CLASSES:
        d = excess[klass == c]
        print(f"  {c:11s} {np.median(d):+6.2f}" if len(d) else f"  {c:11s}   --")

    # ---------- (2) excess vs daic_var / daic_mean ----------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, (xv, lab) in zip(axes, [(dvar, "daic_var (variance evidence)"),
                                    (dmean, "daic_mean (mean evidence)")]):
        m = np.isfinite(xv) & np.isfinite(excess)
        x, y = xv[m], excess[m]
        lo, hi = np.percentile(x, [1, 99])
        mm = (x >= lo) & (x <= hi)
        ax.hexbin(x[mm], y[mm], gridsize=50, bins="log", cmap="Greys", mincnt=1)
        edges = np.unique(np.quantile(x[mm], np.linspace(0, 1, 13)))
        cx, cy = [], []
        for a, b in zip(edges[:-1], edges[1:]):
            s = (x >= a) & (x <= b)
            if s.sum() >= 30:
                cx.append(0.5 * (a + b)); cy.append(np.median(y[s]))
        ax.plot(cx, cy, "-o", color="crimson", lw=2, ms=4)
        ax.axhline(0, color="grey", lw=0.8, ls=":")
        ax.axvline(0, color="grey", lw=0.8, ls=":")
        rho, _ = stats.spearmanr(x, y)
        ax.set_title(f"excess vs {lab}\nSpearman rho={rho:+.3f}", fontsize=11)
        ax.set_xlabel(lab); ax.set_ylabel("excess [pt]")
    fig.suptitle("Does the SD reduction track VARIANCE or MEAN nonstationarity evidence?",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(out_dir, "excess_vs_daic.png"), dpi=160)
    plt.close(fig)
    print("Saved: excess_vs_daic.png")
    rv, _ = stats.spearmanr(dvar[np.isfinite(dvar)], excess[np.isfinite(dvar)])
    rm, _ = stats.spearmanr(dmean[np.isfinite(dmean)], excess[np.isfinite(dmean)])
    print(f"Spearman(excess, daic_var) = {rv:+.3f}")
    print(f"Spearman(excess, daic_mean) = {rm:+.3f}")

    # ---------- (2b) INCREASING vs DECREASING variance ----------
    # sig_slope = sigma year-slope of the 'both' model (>0 increasing variance).
    if np.any(np.isfinite(slope)):
        inc = np.isfinite(slope) & (slope > 0)
        dec = np.isfinite(slope) & (slope < 0)
        nonst = np.isin(klass, ["var", "both"])
        print("Variance-trend DIRECTION (sigma year-slope sign):")
        for name, mk in [("increasing (all)", inc), ("decreasing (all)", dec),
                         ("increasing & var/both", inc & nonst),
                         ("decreasing & var/both", dec & nonst)]:
            n = int(np.sum(mk))
            med = np.median(excess[mk]) if n else np.nan
            print(f"  {name:24s} n={n:8d}  median excess={med:+.2f}")
        m = np.isfinite(slope)
        rs, _ = stats.spearmanr(slope[m], excess[m])
        ra, _ = stats.spearmanr(np.abs(slope[m]), excess[m])
        print(f"Spearman(excess, sig_slope)   = {rs:+.3f}  (>0 => increasing helps more)")
        print(f"Spearman(excess, |sig_slope|) = {ra:+.3f}  (>0 => any change helps)")

        fig, ax = plt.subplots(figsize=(7.5, 5.5))
        x, y = slope[m], excess[m]
        lo, hi = np.percentile(x, [1, 99])
        mm = (x >= lo) & (x <= hi)
        ax.hexbin(x[mm], y[mm], gridsize=50, bins="log", cmap="Greys", mincnt=1)
        edges = np.unique(np.quantile(x[mm], np.linspace(0, 1, 15)))
        cx, cy = [], []
        for a, b in zip(edges[:-1], edges[1:]):
            s = (x >= a) & (x <= b)
            if s.sum() >= 30:
                cx.append(0.5 * (a + b)); cy.append(np.median(y[s]))
        ax.plot(cx, cy, "-o", color="crimson", lw=2, ms=4)
        ax.axhline(0, color="grey", lw=0.8, ls=":"); ax.axvline(0, color="grey", lw=0.8, ls=":")
        ax.set_xlabel("sigma year-slope  (<0 decreasing | >0 increasing variance)")
        ax.set_ylabel("excess [pt]")
        ax.set_title(f"Excess vs variance-trend slope (rho_signed={rs:+.3f}, rho_|.|={ra:+.3f})")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "excess_vs_sigslope.png"), dpi=160)
        plt.close(fig)
        print("Saved: excess_vs_sigslope.png")
    else:
        print("NOTE: sig_slope absent in 077 output -> skipped direction analysis (re-run 077).")

    # ---------- (3) hydroclimate per AIC class ----------
    # var_ratio = Var(early)/Var(late): <1 increasing, >1 decreasing. A signed
    # median CANCELS increasing vs decreasing cells, so we ALSO report a
    # direction-free magnitude (fold = exp(median|ln var_ratio|), "variance moved
    # by x-fold either way") and the increasing fraction separately.
    vr = hyd["var_ratio"]

    def vr_stats(sel):
        v = vr[sel]
        v = v[np.isfinite(v) & (v > 0)]
        if v.size == 0:
            return np.nan, np.nan
        fold = float(np.exp(np.median(np.abs(np.log(v)))))   # direction-free size
        frac_inc = float(np.mean(v < 1.0))                   # var_ratio<1 => increasing
        return fold, frac_inc

    with open(os.path.join(out_dir, "class_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["aic_class", "n", "median_excess",
                    "fold_var_change", "frac_var_increasing"]
                   + [f"median_{HYDRO_LABEL[k]}" for k in HYDRO])
        for c in AIC_CLASSES:
            sel = klass == c
            n = int(np.sum(sel))
            if n == 0:
                w.writerow([c, 0, "", "", ""] + [""] * len(HYDRO)); continue
            fold, frac_inc = vr_stats(sel)
            row = [c, n, f"{np.median(excess[sel]):.3f}",
                   f"{fold:.3f}", f"{frac_inc:.3f}"]
            row += [f"{np.nanmedian(hyd[k][sel]):.4g}" for k in HYDRO]
            w.writerow(row)
    print("Saved: class_summary.csv")
    print("Variance change by AIC class (fold = exp median|ln var_ratio|, "
          "direction-free):")
    print(f"  {'class':11s} {'fold':>7s} {'%increasing':>12s}")
    for c in AIC_CLASSES:
        sel = klass == c
        if not np.any(sel):
            continue
        fold, frac_inc = vr_stats(sel)
        print(f"  {c:11s} {fold:7.3f} {100*frac_inc:11.1f}%")
    print("Hydroclimate (median) by AIC class:")
    print("  class       " + "  ".join(f"{HYDRO_LABEL[k]:>17s}" for k in HYDRO))
    for c in AIC_CLASSES:
        sel = klass == c
        if not np.any(sel):
            continue
        vals = "  ".join(f"{np.nanmedian(hyd[k][sel]):17.4g}" for k in HYDRO)
        print(f"  {c:11s} {vals}")

    # the median hides the right tail: are there near-permanently-dry cells (e.g.
    # 90% of annual maxima below 1 m3/s) inside a class? Report the tail explicitly.
    zf_all = hyd["zero_flow_rate"]
    print("dry_year_fraction TAIL by AIC class (median hides extremes):")
    print(f"  {'class':11s} {'median':>8s} {'P95':>8s} {'P99':>8s} {'max':>8s}"
          f" {'%cells>50%':>11s} {'%cells>90%':>11s}")
    for c in AIC_CLASSES:
        z = zf_all[klass == c]
        z = z[np.isfinite(z)]
        if z.size == 0:
            continue
        print(f"  {c:11s} {np.median(z):8.3f} {np.percentile(z,95):8.3f} "
              f"{np.percentile(z,99):8.3f} {np.max(z):8.3f} "
              f"{100*np.mean(z>0.5):11.2f} {100*np.mean(z>0.9):11.2f}")

    # ---------- (3b) BOXPLOT: variance-change distribution per class ----------
    # The median hides that each class MIXES increasing and decreasing cells; the
    # boxplot shows the full spread. Displayed as Var(late)/Var(early)=1/var_ratio
    # on a log axis (>1 = INCREASING) with a "no change" line at 1.
    fig, ax = plt.subplots(figsize=(8, 6))
    data, labels, cols = [], [], []
    for c in AIC_CLASSES:
        v = vr[klass == c]
        v = v[np.isfinite(v) & (v > 0)]
        data.append((1.0 / v) if v.size else [np.nan])       # late/early, >1 increasing
        labels.append(f"{c}\n(n={v.size})"); cols.append(CLASS_COLOR[c])
    bp = ax.boxplot(data, widths=0.6, showfliers=False, patch_artist=True)
    for patch, cc in zip(bp["boxes"], cols):
        patch.set_facecolor(cc); patch.set_alpha(0.6)
    ax.set_yscale("log")
    ax.axhline(1.0, color="k", ls="--", lw=1, label="no change (Var early = Var late)")
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Var(late) / Var(early)   ( >1 = variance INCREASING )")
    ax.set_title("Variance-change distribution by AIC nonstationarity class")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "var_ratio_by_class.png"), dpi=180)
    plt.close(fig)
    print("Saved: var_ratio_by_class.png")

    # ---------- (3b-2) BOXPLOT: GU scale-trend (sig_slope) per class ----------
    # var_ratio above is the EMPIRICAL early/late variance change; sig_slope is the
    # MODEL-BASED sigma year-slope of the 'both' GU fit (log link, >0 = variance
    # INCREASING). Showing it per class makes the variance non-stationarity that
    # drives the excess visible in the SAME estimator the paper uses, not just as a
    # two-window ratio. var/both should sit clearly off zero; stationary/mean ~ 0.
    if np.any(np.isfinite(slope)):
        fig, ax = plt.subplots(figsize=(8, 6))
        data, labels, cols = [], [], []
        for c in AIC_CLASSES:
            v = slope[klass == c]
            v = v[np.isfinite(v)]
            data.append(v if v.size else [np.nan])
            labels.append(f"{c}\n(n={v.size})"); cols.append(CLASS_COLOR[c])
        bp = ax.boxplot(data, widths=0.6, showfliers=False, patch_artist=True, notch=True)
        for patch, cc in zip(bp["boxes"], cols):
            patch.set_facecolor(cc); patch.set_alpha(0.6)
        ax.axhline(0.0, color="k", ls="--", lw=1, label="no scale trend (sigma flat)")
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel(r"$\sigma$ year-slope  ( <0 decreasing | >0 increasing variance )")
        ax.set_title("GU scale-trend (sig_slope) by AIC nonstationarity class")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "sig_slope_by_class.png"), dpi=180)
        plt.close(fig)
        print("Saved: sig_slope_by_class.png")
    else:
        print("NOTE: sig_slope absent -> sig_slope_by_class.png skipped (re-run 077).")

    # ---------- (3c) BOXPLOT: hydroclimate predictor per class ----------
    # var_ratio is shown INVERTED as Var(late)/Var(early) (>1 = increasing) on a log
    # axis, like cv. dry_year_fraction is dropped (kept only as the tail table above).
    # Notches give the 95% CI of each median (tiny at N~1e5), so even where the boxes
    # overlap heavily (cv, mean_outflow) the class-median TREND stays readable; a red
    # line connects the class medians to make that trend explicit.
    HYDRO_PLOT = [k for k in HYDRO if k != "zero_flow_rate"]
    LOG_PANELS = {"var_ratio", "cv", "mean_outflow", "uparea_km2"}
    fig, axes = plt.subplots(1, len(HYDRO_PLOT), figsize=(3.6 * len(HYDRO_PLOT), 5))
    for ax, k in zip(np.atleast_1d(axes), HYDRO_PLOT):
        invert = (k == "var_ratio")
        uselog = k in LOG_PANELS
        title = "Var(late)/Var(early)" if invert else HYDRO_LABEL[k]
        data, cols, meds = [], [], []
        for c in AIC_CLASSES:
            v = hyd[k][klass == c]
            v = v[np.isfinite(v)]
            if uselog:
                v = v[v > 0]                 # log axis: drop non-positive fill values
            if invert:
                v = 1.0 / v
            data.append(v if v.size else [np.nan])
            meds.append(np.median(v) if v.size else np.nan)
            cols.append(CLASS_COLOR[c])
        bp = ax.boxplot(data, widths=0.6, showfliers=False, patch_artist=True, notch=True)
        for patch, cc in zip(bp["boxes"], cols):
            patch.set_facecolor(cc); patch.set_alpha(0.6)
        ax.plot(np.arange(1, len(AIC_CLASSES) + 1), meds, "-o", color="crimson",
                lw=1.6, ms=4, zorder=5)                       # class-median trend
        if uselog:
            ax.set_yscale("log")
        if invert:
            ax.axhline(1.0, color="k", ls="--", lw=0.8)      # no change reference
        ax.set_xticklabels(AIC_CLASSES, rotation=45, ha="right", fontsize=8)
        ax.set_title(title, fontsize=11)
    fig.suptitle("Hydroclimate by AIC nonstationarity class", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(out_dir, "hydro_by_class.png"), dpi=160)
    plt.close(fig)
    print("Saved: hydro_by_class.png")

    # ---------- (3d) inter-index Spearman matrix ----------
    # How do the indices relate to EACH OTHER (not just to excess)? ATTENUATION
    # predicts large rivers (high uparea / mean_outflow) have LOWER variability (cv)
    # and smaller variance change (|ln var_ratio|): negative rho in those cells.
    absln = np.abs(np.log(np.where((vr > 0) & np.isfinite(vr), vr, np.nan)))
    mat_vars = [
        ("excess", excess),
        ("|ln var_ratio|", absln),          # variance-change magnitude (direction-free)
        ("cv", hyd["cv"]),
        ("dry_year_fraction", hyd["zero_flow_rate"]),
        ("mean_outflow", hyd["mean_outflow"]),
        ("uparea_km2", hyd["uparea_km2"]),
    ]
    names = [n for n, _ in mat_vars]
    arrs = [np.asarray(a, float) for _, a in mat_vars]
    M = len(arrs)
    rho = np.full((M, M), np.nan)
    for i in range(M):
        for j in range(M):
            m = np.isfinite(arrs[i]) & np.isfinite(arrs[j])
            if m.sum() >= 30:
                rho[i, j] = stats.spearmanr(arrs[i][m], arrs[j][m]).correlation
    with open(os.path.join(out_dir, "index_correlation.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([""] + names)
        for i, n in enumerate(names):
            w.writerow([n] + [f"{rho[i, j]:.3f}" for j in range(M)])
    print("Saved: index_correlation.csv")
    fig, ax = plt.subplots(figsize=(7.8, 6.6))
    im = ax.imshow(rho, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(M)); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(M)); ax.set_yticklabels(names, fontsize=9)
    for i in range(M):
        for j in range(M):
            if np.isfinite(rho[i, j]):
                ax.text(j, i, f"{rho[i, j]:.2f}", ha="center", va="center",
                        fontsize=8, color="k")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Spearman rho")
    ax.set_title("Inter-index Spearman correlations (flood-relevant cells)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "index_correlation.png"), dpi=160)
    plt.close(fig)
    print("Saved: index_correlation.png")

    # ---------- (3e) ATTENUATION: river size vs variability / variance change ----
    def size_panel(ax, x, y, xlab, ylab):
        m = np.isfinite(x) & np.isfinite(y) & (x > 0)
        x, y = x[m], y[m]
        if x.size < 50:
            ax.set_title(f"{ylab} vs {xlab} (n<50)"); return
        ax.scatter(x, y, s=2, alpha=0.05, color="0.5", rasterized=True)
        ax.set_xscale("log")
        lx = np.log10(x)
        edges = np.unique(np.quantile(lx, np.linspace(0, 1, 13)))
        cx, cy = [], []
        for a, b in zip(edges[:-1], edges[1:]):
            s = (lx >= a) & (lx <= b)
            if s.sum() >= 30:
                cx.append(10 ** (0.5 * (a + b))); cy.append(np.median(y[s]))
        ax.plot(cx, cy, "-o", color="crimson", lw=2, ms=4)
        ax.set_ylim(0, np.percentile(y, 99))
        rho_ = stats.spearmanr(x, y).correlation
        ax.set_xlabel(xlab); ax.set_ylabel(ylab)
        ax.set_title(f"{ylab} vs {xlab}  (rho={rho_:+.3f})", fontsize=10)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    size_panel(axes[0, 0], hyd["uparea_km2"], hyd["cv"], "uparea_km2", "cv")
    size_panel(axes[0, 1], hyd["mean_outflow"], hyd["cv"], "mean_outflow", "cv")
    size_panel(axes[1, 0], hyd["uparea_km2"], absln, "uparea_km2", "|ln var_ratio|")
    size_panel(axes[1, 1], hyd["mean_outflow"], absln, "mean_outflow", "|ln var_ratio|")
    fig.suptitle("ATTENUATION test: larger rivers -> less variability / variance "
                 "change?  (rho<0 supports attenuation)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(out_dir, "attenuation.png"), dpi=150)
    plt.close(fig)
    print("Saved: attenuation.png")

    # ---------- (3f) ROBUSTNESS: drop near-permanently-dry (degenerate) cells ----
    # The var class holds some ~90%-dry cells (degenerate AMAX -> unstable Q100).
    # Confirm the excess <- variance link is NOT an artifact of those cells by
    # re-running the key numbers on cells with dry_year_fraction <= MAX_DRY.
    MAX_DRY = 0.5
    zf_all = hyd["zero_flow_rate"]
    keep_dry = ~(np.isfinite(zf_all) & (zf_all > MAX_DRY))

    def summarize(mask, label):
        kk, ex = klass[mask], excess[mask]
        dv, sl = dvar[mask], slope[mask]
        md = np.isfinite(dv)
        rv = stats.spearmanr(dv[md], ex[md]).correlation if md.sum() > 30 else np.nan
        ms = np.isfinite(sl)
        ra = stats.spearmanr(np.abs(sl[ms]), ex[ms]).correlation if ms.sum() > 30 else np.nan
        print(f"  [{label:12s}] n={int(mask.sum()):8d}  "
              f"Spearman(excess,daic_var)={rv:+.3f}  "
              f"Spearman(excess,|sig_slope|)={ra:+.3f}")
        for c in AIC_CLASSES:
            s = kk == c
            if s.any():
                print(f"       {c:11s} n={int(s.sum()):8d} ({100*s.mean():5.1f}%)  "
                      f"median excess={np.median(ex[s]):+.2f}")

    print(f"ROBUSTNESS to arid/degenerate cells (drop dry_year_fraction > {MAX_DRY}):")
    summarize(np.ones(len(excess), dtype=bool), "ALL")
    summarize(keep_dry, f"dry<={MAX_DRY}")

    # ---------- (3g) dry-year threshold SWEEP (justifies the flood-ensured cut) ------
    # Computed on the PRE-dry set (CAP + reverse-flow only, dry cells KEPT) so it spans
    # the full range: keep cells with dry_year_fraction <= thr and recompute the var/
    # both median excess and the excess<->variance correlations. The var curve falls as
    # more intermittent (ill-conditioned, tiny-Q100) cells are admitted and stabilises
    # below ~0.5 -> that is why the flood-ensured definition cuts at DRY_MAX=0.5.
    THRS = [0.3, 0.5, 0.7, 0.9, 1.0]
    sw = {k: [] for k in ("thr", "n", "var_med", "both_med", "rho_sig", "rho_dvar")}
    for thr in THRS:
        km = ~(np.isfinite(zf_pd) & (zf_pd > thr))           # PRE-dry set: full range
        ex, kk, sl, dv = excess_pd[km], klass_pd[km], slope_pd[km], dvar_pd[km]
        ms, md = np.isfinite(sl), np.isfinite(dv)
        sw["thr"].append(thr); sw["n"].append(int(km.sum()))
        sw["var_med"].append(np.median(ex[kk == "var"]) if np.any(kk == "var") else np.nan)
        sw["both_med"].append(np.median(ex[kk == "both"]) if np.any(kk == "both") else np.nan)
        sw["rho_sig"].append(stats.spearmanr(np.abs(sl[ms]), ex[ms]).correlation
                             if ms.sum() > 30 else np.nan)
        sw["rho_dvar"].append(stats.spearmanr(dv[md], ex[md]).correlation
                              if md.sum() > 30 else np.nan)
    print("Dry-year threshold sweep (keep dry_year <= thr):")
    print(f"  {'thr':>5s} {'n':>9s} {'var_med':>9s} {'both_med':>9s} "
          f"{'rho|sig|':>9s} {'rho_dvar':>9s}")
    for i in range(len(THRS)):
        print(f"  {sw['thr'][i]:5.1f} {sw['n'][i]:9d} {sw['var_med'][i]:9.2f} "
              f"{sw['both_med'][i]:9.2f} {sw['rho_sig'][i]:9.3f} {sw['rho_dvar'][i]:9.3f}")
    with open(os.path.join(out_dir, "dry_threshold_sweep.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dry_thr", "n_cells", "var_median_excess", "both_median_excess",
                    "spearman_abs_sigslope", "spearman_daic_var"])
        for i in range(len(THRS)):
            w.writerow([sw["thr"][i], sw["n"][i], f"{sw['var_med'][i]:.3f}",
                        f"{sw['both_med'][i]:.3f}", f"{sw['rho_sig'][i]:.3f}",
                        f"{sw['rho_dvar'][i]:.3f}"])
    print("Saved: dry_threshold_sweep.csv")
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12, 5))
    axA.plot(sw["thr"], sw["var_med"], "-o", color=CLASS_COLOR["var"], label="var")
    axA.plot(sw["thr"], sw["both_med"], "-s", color=CLASS_COLOR["both"], label="both")
    axA.axhline(0, color="grey", ls=":", lw=0.8)
    axA.set_xlabel("dry_year_fraction cut (keep <= thr)")
    axA.set_ylabel("median excess [pt]")
    axA.set_title("Median excess by class")
    axA.legend()
    axB.plot(sw["thr"], sw["rho_sig"], "-o", color="#d6604d",
             label="rho(excess, |sig_slope|)")
    axB.plot(sw["thr"], sw["rho_dvar"], "-s", color="#2166ac",
             label="rho(excess, daic_var)")
    axB.set_ylim(0, 1)
    axB.set_xlabel("dry_year_fraction cut (keep <= thr)")
    axB.set_ylabel("Spearman rho")
    axB.set_title("Excess<->variance correlation")
    axB.legend()
    fig.suptitle("Robustness of the excess<-variance result to the intermittency cut",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(out_dir, "dry_threshold_sweep.png"), dpi=160)
    plt.close(fig)
    print("Saved: dry_threshold_sweep.png")

    # ---------- (3h) flood-MAGNITUDE threshold sensitivity ----------
    # Does the excess<-variance result depend on the flood-relevance magnitude cuts
    # (q90>=50 OR max_amax>=100) AND uparea>=50? Tighten each in turn (the base set is
    # already flood-relevant, so we can only restrict further) and recompute var/both
    # median excess and rho. Stable rows => the conclusion is not sensitive to the cut.
    def c91col(key):
        out = np.full(len(cid), np.nan)
        for i, c in enumerate(cid):
            v = c91.get(c, {}).get(key, "")
            try:
                out[i] = float(v)
            except (ValueError, TypeError):
                pass
        return out
    q90a, maxa, upa = c91col("q90"), c91col("max_amax"), c91col("uparea_km2")
    MAG = [("uparea_km2", upa, [50, 100, 500, 1000]),
           ("q90", q90a, [50, 100, 200]),
           ("max_amax", maxa, [100, 200, 500])]
    print("Flood-MAGNITUDE sensitivity (tighten one cut; var/both median excess, rho|sig|):")
    print(f"  {'cut':18s} {'n':>9s} {'var':>7s} {'both':>7s} {'rho|sig|':>9s}")
    rows_mag = []
    for name, arr, vals in MAG:
        for thr in vals:
            m = np.isfinite(arr) & (arr >= thr)
            ex, kk, sl = excess[m], klass[m], slope[m]
            ms = np.isfinite(sl)
            rho = (stats.spearmanr(np.abs(sl[ms]), ex[ms]).correlation
                   if ms.sum() > 30 else np.nan)
            vmed = np.median(ex[kk == "var"]) if np.any(kk == "var") else np.nan
            bmed = np.median(ex[kk == "both"]) if np.any(kk == "both") else np.nan
            print(f"  {name + '>=' + str(thr):18s} {int(m.sum()):9d} "
                  f"{vmed:7.2f} {bmed:7.2f} {rho:9.3f}")
            rows_mag.append([name, thr, int(m.sum()), f"{vmed:.3f}", f"{bmed:.3f}",
                             f"{rho:.3f}"])
    with open(os.path.join(out_dir, "flood_magnitude_sweep.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["criterion", "threshold", "n_cells", "var_median_excess",
                    "both_median_excess", "spearman_abs_sigslope"])
        w.writerows(rows_mag)
    print("Saved: flood_magnitude_sweep.csv")

    # ---------- (4) world map of the best AIC model per cell ----------
    if not HAVE_CARTOPY:
        print("NOTE: cartopy not available -> skipped best_model_map.png")
        return
    valid = (iy >= 0) & (ix >= 0)
    lon = -180.0 + (ix[valid] + 0.5) * RES
    lat = 90.0 - (iy[valid] + 0.5) * RES
    bmv = bm[valid]
    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
    ax.set_global()
    ax.add_feature(cfeature.COASTLINE, linewidth=0.4)
    # draw stationary first (bottom) so the nonstationary classes stay visible
    for name in AIC_CLASSES:
        mm = bmv == name
        if np.any(mm):
            ax.scatter(lon[mm], lat[mm], s=0.3, color=CLASS_COLOR[name],
                       transform=ccrs.PlateCarree(), rasterized=True)
    handles = [mpatches.Patch(color=CLASS_COLOR[n],
                              label=f"{n} (n={int(np.sum(bmv == n))})")
               for n in AIC_CLASSES]
    ax.legend(handles=handles, loc="lower left", fontsize=12, title="best AIC model")
    ax.set_title("Best AIC nonstationarity model per cell (flood-relevant)", fontsize=18)
    fig.savefig(os.path.join(out_dir, "best_model_map.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Saved: best_model_map.png")

    # ---------- (4b) spatial co-location: mean discharge vs excess hotspots ----------
    # Make the attenuation link spatial: colour = mean discharge (log), black contour =
    # excess hotspots (1.5-deg mean excess >= 2 pt). The hotspots fall on LOW-discharge
    # (small, flashy) rivers, tying the excess geography (Fig R1) to river physics.
    from matplotlib.colors import LogNorm
    mo = hyd["mean_outflow"][valid]; ex = excess[valid]
    good = np.isfinite(mo) & (mo > 0)
    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
    ax.set_extent([-180, 180, -60, 90], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, linewidth=0.4)
    sc = ax.scatter(lon[good], lat[good], c=mo[good], s=0.3, cmap="viridis",
                    norm=LogNorm(vmin=max(np.percentile(mo[good], 2), 1e-3),
                                 vmax=np.percentile(mo[good], 98)),
                    transform=ccrs.PlateCarree(), rasterized=True)
    res = 1.5
    le = np.arange(-180, 180 + res, res); la = np.arange(-60, 90 + res, res)
    s_, _, _ = np.histogram2d(lon, lat, bins=[le, la], weights=ex)
    n_, _, _ = np.histogram2d(lon, lat, bins=[le, la])
    with np.errstate(invalid="ignore"):
        me = np.where(n_ >= 3, s_ / np.maximum(n_, 1), 0.0)
    try:
        from scipy.ndimage import gaussian_filter
        me = gaussian_filter(me, 1.0)
    except ImportError:
        pass
    xc = 0.5 * (le[:-1] + le[1:]); yc = 0.5 * (la[:-1] + la[1:])
    ax.contour(xc, yc, me.T, levels=[2.0], colors="k", linewidths=1.0,
               transform=ccrs.PlateCarree())
    cb = plt.colorbar(sc, ax=ax, orientation="horizontal", pad=0.05, shrink=0.7,
                      aspect=40, extend="both")
    cb.set_label("mean discharge [m3 s-1]", fontsize=16); cb.ax.tick_params(labelsize=13)
    ax.set_title("Mean discharge with excess hotspots (black: 1.5deg mean excess >= 2 pt)",
                 fontsize=16)
    fig.savefig(os.path.join(out_dir, "mean_outflow_vs_excess_map.png"), dpi=200,
                bbox_inches="tight")
    plt.close(fig)
    print("Saved: mean_outflow_vs_excess_map.png")


if __name__ == "__main__":
    main()
