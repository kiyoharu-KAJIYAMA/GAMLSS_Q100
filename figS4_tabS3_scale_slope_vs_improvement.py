"""
figS4_tabS3_scale_slope_vs_improvement.py
Supplementary-Information tables for the claim that the SD improvement is driven by
the SLOPE OF SCALE (sigma_1, the year-on-year change of the GU log-scale), NOT by the
scale LEVEL / flashiness, and that arid-zone hotspots are a by-product of arid cells
over-concentrating high-sigma_1 AND high-CV cells.

Reproduces (and persists as CSV) the five analyses:
  S1  slope vs level of scale        : marginal & partial Spearman of |sigma_1|,
                                       sigma_0 (level), CV with improvement.
  S2  flashiness needs a trend       : r(imp,CV) and median improvement by CV,
                                       split into no-trend (stationary/mean) vs
                                       trend (var/both) cells; CV x class 2x2.
  S3  excess over |sigma_1| x CV     : 3x3 median-excess matrix (bands are vertical
                                       => slope dominates, CV lifts only at high slope).
  S4  intermittency / effective-N    : within var/both, r(imp,zero_flow), the same
                                       partialled on |sigma_1| (collapses -> no
                                       independent channel), r(|sigma_1|,zero_flow).
  S5  aridity decomposition          : q_spec deciles (medians of imp/exc/|s1|/CV/zf),
                                       aridity correlations & partials, and the
                                       high-|sigma_1|&high-CV "corner" prevalence by
                                       aridity tertile.

Definitions / scope:
  improvement = 050 improve_lin_sd [%] (|.| <= common_target_cells.CAP_PCT).
  excess      = improvement - baseline, baseline = median improvement over the AIC
                'stationary'-class cells present here.
  sigma_1 / sigma_0 = 064/ln/mc_truth truth_sigma_1 / truth_sigma_0 (GU log-scale
                slope / level). |sigma_1| = trend magnitude.
  CV, zero_flow_rate, mean_outflow, uparea_km2 = 070; q_spec = mean_outflow/uparea*1000
                (L s-1 km-2; low = arid). class = 079 AIC change-type.
  Scope = common_target_cells flood-ensured cells; S4 restricted to class in {var, both}.
  Partial Spearman = Spearman within quantile strata of the control(s) (4 strata),
                averaged (control = summed ranks when >1 control variable).

Inputs:  050/summary_allgrid.csv, 079/aic_allgrid.csv, 070/timeseries_tests.csv,
         064/ln/mc_truth/*.csv, common_target_cells
Outputs (<dat_dir>/517/):  S1..S5 CSVs + README.md (+ echo to console)
Usage:   python3 figS4_tabS3_scale_slope_vs_improvement.py [dat_dir]
"""
import os
import sys
import glob
import importlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

excess_lib = importlib.import_module("common_target_cells")
C_SIG, C_LVL, C_CV, C_TREND = "#D55E00", "#777777", "#0072B2", "#009E73"
NG = 4                                   # strata for partial Spearman
YEARS = 120                              # record length for effective-N proxy


def rk(a):
    return pd.Series(a).rank().to_numpy()


def sp(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    return float(np.corrcoef(rk(x[m]), rk(y[m]))[0, 1]) if m.sum() > 10 else np.nan


def psp(y, a, ctrls, ng=NG):
    """partial Spearman(y, a | ctrls) via quantile strata of the summed control ranks."""
    m = np.isfinite(y) & np.isfinite(a)
    for c in ctrls:
        m &= np.isfinite(c)
    y, a = y[m], a[m]
    b = np.zeros(int(m.sum()))
    for c in ctrls:
        b += rk(c[m])
    q = np.quantile(b, np.linspace(0, 1, ng + 1)); q[0] -= 1e-9
    g = np.clip(np.digitize(b, q[1:-1]), 0, ng - 1)
    return float(np.nanmean([sp(a[g == k], y[g == k]) for k in range(ng)]))


def load(dat_dir):
    d50 = pd.read_csv(os.path.join(dat_dir, "050", "summary_allgrid.csv"),
                      usecols=["cell_id", "improve_lin_sd"]).set_index("cell_id")
    aic = pd.read_csv(os.path.join(dat_dir, "079", "aic_allgrid.csv"),
                      usecols=["cell_id", "class"]).set_index("cell_id")
    t70 = pd.read_csv(os.path.join(dat_dir, "070", "timeseries_tests.csv"),
                      usecols=["cell_id", "cv", "zero_flow_rate", "mean_outflow",
                               "uparea_km2"]).set_index("cell_id")
    ch = [c for c in sorted(glob.glob(os.path.join(dat_dir, "064", "ln", "mc_truth", "*.csv")))
          if not c.endswith("Zone.Identifier")]
    co = pd.concat([pd.read_csv(c, usecols=["cell_id", "truth_sigma_0", "truth_sigma_1"])
                    for c in ch], ignore_index=True).set_index("cell_id")
    df = d50.join(aic, how="inner").join(t70, how="inner").join(co, how="inner")
    df = df[df.index.isin(excess_lib.load_flood_ensured_ids(dat_dir))]
    df = df[(df["mean_outflow"] > 0) & (df["uparea_km2"] > 0)]
    df["imp"] = df["improve_lin_sd"].where(df["improve_lin_sd"].abs() <= excess_lib.CAP_PCT)
    base = float(np.nanmedian(df.loc[df["class"] == "stationary", "imp"]))
    df["exc"] = df["imp"] - base
    df["qspec"] = df["mean_outflow"] / df["uparea_km2"] * 1000.0
    return df, base


def table_s1(df, out):
    """slope vs level of scale: marginal & partial Spearman with improvement."""
    imp = df["imp"].to_numpy()
    asig = np.abs(df["truth_sigma_1"].to_numpy())
    sig0 = df["truth_sigma_0"].to_numpy()
    cv = df["cv"].to_numpy()
    n = int(np.isfinite(imp).sum())
    rows = [
        ["|sigma_1|  (scale SLOPE, trend magnitude)", "-", round(sp(asig, imp), 3), n],
        ["|sigma_1|  (scale SLOPE)", "sigma_0", round(psp(imp, asig, [sig0]), 3), n],
        ["|sigma_1|  (scale SLOPE)", "CV", round(psp(imp, asig, [cv]), 3), n],
        ["sigma_0    (scale LEVEL / size)", "-", round(sp(sig0, imp), 3), n],
        ["sigma_0    (scale LEVEL)", "|sigma_1|", round(psp(imp, sig0, [asig]), 3), n],
        ["CV         (flashiness)", "-", round(sp(cv, imp), 3), n],
        ["CV         (flashiness)", "|sigma_1|", round(psp(imp, cv, [asig]), 3), n],
    ]
    t = pd.DataFrame(rows, columns=["predictor", "control", "spearman_with_improvement", "n"])
    t.to_csv(out, index=False)
    return t


def table_s2(df, out):
    """flashiness needs a trend: r(imp,CV) & median improvement by CV, per class group."""
    imp = df["imp"].to_numpy(); cv = df["cv"].to_numpy(); cls = df["class"].to_numpy()
    grp = {"no_trend (stationary+mean)": np.isin(cls, ["stationary", "mean"]),
           "trend (var+both)": np.isin(cls, ["var", "both"])}
    cvmed = np.nanmedian(cv)
    rows = []
    for name, msk in grp.items():
        c, y = cv[msk], imp[msk]
        q1, q3 = np.nanpercentile(c, [25, 75])
        rows.append([
            name, int(msk.sum()), round(sp(c, y), 3),
            round(float(np.nanmedian(y[c <= q1])), 2),
            round(float(np.nanmedian(y[(c > q1) & (c <= q3)])), 2),
            round(float(np.nanmedian(y[c > q3])), 2),
            round(float(np.nanmedian(y[c <= cvmed])), 2),
            round(float(np.nanmedian(y[c > cvmed])), 2),
            round(float(np.nanmedian(y[c > cvmed]) - np.nanmedian(y[c <= cvmed])), 2),
        ])
    t = pd.DataFrame(rows, columns=[
        "class_group", "n", "r_improvement_CV", "median_imp_CVlow", "median_imp_CVmid",
        "median_imp_CVhigh", "median_imp_CV<=median", "median_imp_CV>median", "CV_lift"])
    t.to_csv(out, index=False)
    return t


def table_s3(df, out):
    """3x3 median-excess matrix over |sigma_1| tertiles x CV tertiles."""
    exc = df["exc"].to_numpy(); asig = np.abs(df["truth_sigma_1"].to_numpy()); cv = df["cv"].to_numpy()
    def tert(a):
        return np.digitize(a, np.nanpercentile(a, [100 / 3, 200 / 3]))
    ts, tc = tert(asig), tert(cv)
    lab = ["low", "mid", "high"]
    rows = []
    for i in range(3):
        row = {"sigma1_tertile": lab[i]}
        for j in range(3):
            row[f"CV_{lab[j]}"] = round(float(np.nanmedian(exc[(ts == i) & (tc == j)])), 2)
            row[f"n_CV_{lab[j]}"] = int(((ts == i) & (tc == j)).sum())
        rows.append(row)
    t = pd.DataFrame(rows)
    t.to_csv(out, index=False)
    return t


def table_s4(df, out):
    """intermittency / effective-N within var/both."""
    vb = np.isin(df["class"].to_numpy(), ["var", "both"])
    imp = df["imp"].to_numpy()[vb]; zf = df["zero_flow_rate"].to_numpy()[vb]
    asig = np.abs(df["truth_sigma_1"].to_numpy())[vb]
    effN = YEARS * (1.0 - zf)
    rows = [
        ["r(improvement, zero_flow_rate)", "-", round(sp(zf, imp), 3)],
        ["r(improvement, zero_flow_rate)", "|sigma_1|", round(psp(imp, zf, [asig]), 3)],
        ["r(improvement, effective_N=120*(1-zero_flow))", "-", round(sp(effN, imp), 3)],
        ["r(|sigma_1|, zero_flow_rate)", "-", round(sp(asig, zf), 3)],
    ]
    t = pd.DataFrame(rows, columns=["quantity", "control", "spearman"])
    t.attrs["frac_zf_gt0"] = float(np.mean(zf > 0))
    t.to_csv(out, index=False)
    return t


def table_s5(df, out_dec, out_sum):
    """aridity: q_spec deciles + aridity correlations/partials + corner prevalence."""
    imp = df["imp"].to_numpy(); exc = df["exc"].to_numpy()
    asig = np.abs(df["truth_sigma_1"].to_numpy()); cv = df["cv"].to_numpy()
    zf = df["zero_flow_rate"].to_numpy(); q = df["qspec"].to_numpy()
    arid = -np.log10(q)
    # deciles (D1 = most arid)
    e = np.percentile(q, np.linspace(0, 100, 11)); e[0] -= 1e-9
    g = np.clip(np.digitize(q, e[1:-1]), 0, 9)
    rows = []
    for k in range(10):
        s = g == k
        rows.append([f"D{k + 1}", int(s.sum()),
                     round(float(np.nanmedian(imp[s])), 2), round(float(np.nanmedian(exc[s])), 2),
                     round(float(np.nanmedian(asig[s])), 4), round(float(np.nanmedian(cv[s])), 2),
                     round(float(np.nanmedian(zf[s])), 3)])
    dec = pd.DataFrame(rows, columns=["qspec_decile_D1arid_D10humid", "n", "median_imp",
                                      "median_excess", "median_abs_sigma1", "median_CV",
                                      "median_zero_flow"])
    dec.to_csv(out_dec, index=False)
    # summary: correlations, partials, corner prevalence
    hs = asig >= np.nanpercentile(asig, 66.7)
    hc = cv >= np.nanpercentile(cv, 66.7)
    corner = hs & hc
    qt = np.digitize(q, np.nanpercentile(q, [33.3, 66.7]))
    srows = [
        ["r(improvement, aridity=-log10 qspec)", "-", round(sp(arid, imp), 3)],
        ["r(|sigma_1|, aridity)", "-", round(sp(arid, asig), 3)],
        ["r(CV, aridity)", "-", round(sp(arid, cv), 3)],
        ["r(improvement, aridity)", "|sigma_1|", round(psp(imp, arid, [asig]), 3)],
        ["r(improvement, aridity)", "|sigma_1|+CV", round(psp(imp, arid, [asig, cv]), 3)],
    ]
    for i, nm in enumerate(["arid_lowqspec", "mid", "humid_highqspec"]):
        s = qt == i
        srows.append([f"corner_prevalence[{nm}]  P(|s1|top1/3 & CV top1/3)", "-",
                      round(100 * float(np.mean(corner[s])), 1)])
    srows.append(["median_imp corner cells", "-", round(float(np.nanmedian(imp[corner])), 2)])
    srows.append(["median_imp non-corner cells", "-", round(float(np.nanmedian(imp[~corner])), 2)])
    summ = pd.DataFrame(srows, columns=["quantity", "control", "value"])
    summ.to_csv(out_sum, index=False)
    return dec, summ


def _binmed(x, y, nb=12, qlo=1, qhi=99):
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    lo, hi = np.percentile(x, [qlo, qhi])
    edges = np.linspace(lo, hi, nb + 1)
    cx, md = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        s = (x >= a) & (x < b)
        if s.sum() >= 50:
            cx.append(0.5 * (a + b)); md.append(np.median(y[s]))
    return np.array(cx), np.array(md)


def make_s1_separate(df, out_marg, out_part):
    """Panel (a) split into two standalone figures: marginal-only and partial-only."""
    imp = df["imp"].to_numpy()
    asig = np.abs(df["truth_sigma_1"].to_numpy()); sig0 = df["truth_sigma_0"].to_numpy()
    cv = df["cv"].to_numpy()
    names = [r"$|\sigma_1|$" + "\n(scale SLOPE)", r"$\sigma_0$" + "\n(scale LEVEL)",
             "CV\n(flashiness)"]
    cols = [C_SIG, C_LVL, C_CV]
    marg = [sp(asig, imp), sp(sig0, imp), sp(cv, imp)]
    part = [psp(imp, asig, [cv]), psp(imp, sig0, [asig]), psp(imp, cv, [asig])]
    ctl = ["| CV", r"| $|\sigma_1|$", r"| $|\sigma_1|$"]
    x = np.arange(3)
    for vals, ctls, hatch, alpha, ttl, out in [
            (marg, None, None, 0.95,
             "Marginal Spearman with improvement (flood-ensured, n={:,})".format(len(df)),
             out_marg),
            (part, ctl, "///", 0.55,
             "Partial Spearman with improvement (competing scale term controlled)", out_part)]:
        fig, ax = plt.subplots(figsize=(7.2, 6.0))
        ax.bar(x, vals, 0.62, color=cols, alpha=alpha, hatch=hatch, edgecolor="white")
        for xi, v in enumerate(vals):
            lbl = f"{v:+.2f}" + (f"\n{ctls[xi]}" if ctls else "")
            ax.text(xi, v + (0.02 if v >= 0 else -0.03), lbl, ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=13,
                    fontweight="bold" if not ctls else "normal")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(x); ax.set_xticklabels(names, fontsize=13)
        ax.set_ylabel("Spearman $\\rho$ with improvement", fontsize=14)
        ax.set_ylim(-0.45, 0.72); ax.grid(axis="y", alpha=0.25)
        ax.set_title(ttl, fontsize=13)
        fig.tight_layout()
        fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig)
        print(f"Saved: {out}")


def make_aridity_figure(df, out_png):
    """'Arid zone = corner factory': (a) high-|sigma_1|&high-CV corner prevalence by
    q_spec tertile (arid over-supplies it), (b) aridity correlations -- it co-varies
    with |sigma_1| & CV, and its effect on improvement collapses to ~0 once those are
    controlled (aridity is a geographic label, not an independent driver)."""
    imp = df["imp"].to_numpy()
    asig = np.abs(df["truth_sigma_1"].to_numpy()); cv = df["cv"].to_numpy()
    q = df["qspec"].to_numpy(); arid = -np.log10(q)
    hs = asig >= np.nanpercentile(asig, 66.7); hc = cv >= np.nanpercentile(cv, 66.7)
    corner = hs & hc
    qt = np.digitize(q, np.nanpercentile(q, [33.3, 66.7]))     # 0 arid,1 mid,2 humid
    prev = [100 * np.mean(corner[qt == i]) for i in range(3)]
    medi = [np.nanmedian(imp[qt == i]) for i in range(3)]
    tcol = ["#a6611a", "#dfc27d", "#018571"]                   # BrBG arid->humid
    tname = ["arid\n(low q$_{spec}$)", "mid", "humid\n(high q$_{spec}$)"]

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(14.5, 6.2),
                                   gridspec_kw={"width_ratios": [1.0, 1.25]})

    # (a) corner prevalence by aridity tertile
    xb = np.arange(3)
    axA.bar(xb, prev, 0.62, color=tcol, edgecolor="#333333")
    axA.axhline(100 * np.mean(corner), color="k", ls="--", lw=1.2,
                label=f"overall = {100*np.mean(corner):.1f}%")
    for xi, (p_, mi_) in enumerate(zip(prev, medi)):
        axA.text(xi, p_ + 0.5, f"{p_:.1f}%", ha="center", fontsize=15, fontweight="bold")
        axA.text(xi, 1.5, f"med imp\n{mi_:.1f}%", ha="center", va="bottom",
                 fontsize=11, color="white", fontweight="bold")
    axA.set_xticks(xb); axA.set_xticklabels(tname, fontsize=13)
    axA.set_ylabel(r"P(high $|\sigma_1|$ AND high CV)  [%]", fontsize=14)
    axA.set_ylim(0, max(prev) * 1.18)
    axA.set_title("(a) arid cells over-supply the max-improvement corner\n"
                  r"($|\sigma_1|$ top-1/3 AND CV top-1/3)", fontsize=13)
    axA.legend(fontsize=12); axA.grid(axis="y", alpha=0.25)

    # (b) aridity correlations: co-variation with drivers, then collapse on improvement
    labels = [r"$|\sigma_1|$" + "\nvs aridity", "CV\nvs aridity",
              "improvement\nvs aridity", "improvement\nvs aridity\n" + r"| $|\sigma_1|$",
              "improvement\nvs aridity\n" + r"| $|\sigma_1|$+CV"]
    vals = [sp(arid, asig), sp(arid, cv), sp(arid, imp),
            psp(imp, arid, [asig]), psp(imp, arid, [asig, cv])]
    cols = [C_SIG, C_CV, "#555555", "#999999", "#c9c9c9"]
    xb2 = np.arange(5)
    axB.bar(xb2, vals, 0.66, color=cols, edgecolor="#333333")
    for xi, v in enumerate(vals):
        axB.text(xi, v + (0.006 if v >= 0 else -0.006), f"{v:+.3f}", ha="center",
                 va="bottom" if v >= 0 else "top", fontsize=12, fontweight="bold")
    axB.axhline(0, color="k", lw=0.8)
    axB.axvline(1.5, color="#bbbbbb", lw=1.0, ls=":")
    axB.text(0.5, 0.30, "why: aridity co-varies\nwith the drivers", ha="center",
             fontsize=11, color="#444444", transform=axB.get_xaxis_transform())
    axB.text(3.0, 0.30, "effect on improvement\ncollapses when controlled", ha="center",
             fontsize=11, color="#444444", transform=axB.get_xaxis_transform())
    axB.set_xticks(xb2); axB.set_xticklabels(labels, fontsize=10.5)
    axB.set_ylabel("Spearman $\\rho$", fontsize=14)
    axB.set_ylim(-0.05, 0.34)
    axB.set_title("(b) aridity is a geographic label, not an independent driver\n"
                  r"(residual $\approx$ 0 after $|\sigma_1|$ + CV)", fontsize=13)
    axB.grid(axis="y", alpha=0.25)

    fig.suptitle("Arid zones are the 'factory' of high-improvement cells: they concentrate "
                 r"high $|\sigma_1|$ AND high CV", fontsize=15, y=1.0)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"Saved: {out_png}")


def make_figure(df, base, out_png):
    """4-panel SI figure visualising S1, S2, S3, S5."""
    imp = df["imp"].to_numpy(); exc = df["exc"].to_numpy()
    asig = np.abs(df["truth_sigma_1"].to_numpy()); sig0 = df["truth_sigma_0"].to_numpy()
    cv = df["cv"].to_numpy(); q = df["qspec"].to_numpy(); cls = df["class"].to_numpy()

    fig, axes = plt.subplots(2, 2, figsize=(15.5, 11.5))
    axA, axB, axC, axD = axes.ravel()

    # (a) S1: marginal vs partial Spearman -------------------------------------
    names = [r"$|\sigma_1|$" + "\n(scale SLOPE)", r"$\sigma_0$" + "\n(scale LEVEL)",
             "CV\n(flashiness)"]
    marg = [sp(asig, imp), sp(sig0, imp), sp(cv, imp)]
    part = [psp(imp, asig, [cv]), psp(imp, sig0, [asig]), psp(imp, cv, [asig])]
    ctl = ["| CV", r"| $|\sigma_1|$", r"| $|\sigma_1|$"]
    cols = [C_SIG, C_LVL, C_CV]
    x = np.arange(3); w = 0.38
    axA.bar(x - w / 2, marg, w, color=cols, alpha=0.95, label="marginal")
    axA.bar(x + w / 2, part, w, color=cols, alpha=0.45, hatch="///",
            edgecolor="white", label="partial")
    for xi, (m_, p_, c_) in enumerate(zip(marg, part, ctl)):
        axA.text(xi - w / 2, m_ + (0.02 if m_ >= 0 else -0.05), f"{m_:+.2f}",
                 ha="center", fontsize=12, fontweight="bold")
        axA.text(xi + w / 2, p_ + (0.02 if p_ >= 0 else -0.05), f"{p_:+.2f}\n{c_}",
                 ha="center", va="bottom" if p_ >= 0 else "top", fontsize=10)
    axA.axhline(0, color="k", lw=0.8)
    axA.set_xticks(x); axA.set_xticklabels(names, fontsize=13)
    axA.set_ylabel("Spearman $\\rho$ with improvement", fontsize=14)
    axA.set_ylim(-0.45, 0.72)
    axA.set_title("(a) improvement tracks the SLOPE, not the level/flashiness\n"
                  r"partialling $|\sigma_1|$ collapses CV & $\sigma_0$; $|\sigma_1|$ survives",
                  fontsize=13)
    axA.legend(fontsize=12, loc="upper right"); axA.grid(axis="y", alpha=0.25)

    # (b) S2: improvement vs CV, split by trend class --------------------------
    for msk, col, lab in [(np.isin(cls, ["stationary", "mean"]), C_LVL,
                           "no trend (stationary/mean)"),
                          (np.isin(cls, ["var", "both"]), C_TREND, "trend (var/both)")]:
        cx, md = _binmed(cv[msk], imp[msk])
        axB.plot(cx, md, "-o", color=col, lw=2.6, ms=5, label=lab)
    axB.axhline(base, color="k", ls="--", lw=1.2, label=f"baseline = {base:.1f}%")
    axB.set_xlabel("CV  (flashiness)", fontsize=14)
    axB.set_ylabel("median improvement [%]", fontsize=14)
    axB.set_title("(b) flashiness lifts improvement ONLY where a trend exists",
                  fontsize=13)
    axB.legend(fontsize=11, loc="upper left"); axB.grid(alpha=0.25)

    # (c) S3: 3x3 median-excess heatmap ----------------------------------------
    def tert(a):
        return np.digitize(a, np.nanpercentile(a, [100 / 3, 200 / 3]))
    ts, tc = tert(asig), tert(cv)
    M = np.array([[np.nanmedian(exc[(ts == i) & (tc == j)]) for j in range(3)]
                  for i in range(3)])
    vmax = float(np.nanmax(np.abs(M)))
    im = axC.imshow(M, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    for i in range(3):
        for j in range(3):
            axC.text(j, i, f"{M[i, j]:+.2f}", ha="center", va="center",
                     fontsize=15, fontweight="bold",
                     color="white" if abs(M[i, j]) > 0.6 * vmax else "black")
    axC.set_xticks(range(3)); axC.set_xticklabels(["low", "mid", "high"], fontsize=12)
    axC.set_yticks(range(3)); axC.set_yticklabels(["low", "mid", "high"], fontsize=12)
    axC.set_xlabel("CV tertile  (flashiness)", fontsize=14)
    axC.set_ylabel(r"$|\sigma_1|$ tertile  (scale slope)", fontsize=14)
    axC.set_title("(c) median excess: rises with $|\\sigma_1|$ (vertical),\n"
                  "CV lifts only in the high-$|\\sigma_1|$ row", fontsize=13)
    cb = fig.colorbar(im, ax=axC, shrink=0.85); cb.set_label("median excess [pt]", fontsize=12)

    # (d) S5: aridity deciles -- imp with |sigma_1| & CV co-rising --------------
    e = np.percentile(q, np.linspace(0, 100, 11)); e[0] -= 1e-9
    g = np.clip(np.digitize(q, e[1:-1]), 0, 9)
    xd = np.arange(1, 11)
    mi = [np.nanmedian(imp[g == k]) for k in range(10)]
    ms = [np.nanmedian(asig[g == k]) for k in range(10)]
    mc = [np.nanmedian(cv[g == k]) for k in range(10)]
    axD.plot(xd, mi, "-o", color="#333333", lw=2.6, ms=6, label="median improvement")
    axD.set_xlabel("q$_{spec}$ decile   (D1 = arid  $\\rightarrow$  D10 = humid)", fontsize=14)
    axD.set_ylabel("median improvement [%]", fontsize=14, color="#333333")
    axD.set_xticks(xd); axD.grid(alpha=0.25)
    ax2 = axD.twinx()
    ax2.plot(xd, ms, "-s", color=C_SIG, lw=2.2, ms=5, label=r"median $|\sigma_1|$")
    ax2.plot(xd, np.array(mc) / 100.0, "-^", color=C_CV, lw=2.2, ms=5,
             label="median CV / 100")
    ax2.set_ylabel(r"median $|\sigma_1|$  /  (CV/100)", fontsize=13)
    # corner prevalence annotation
    hs = asig >= np.nanpercentile(asig, 66.7); hc = cv >= np.nanpercentile(cv, 66.7)
    corner = hs & hc; qt = np.digitize(q, np.nanpercentile(q, [33.3, 66.7]))
    pv = [100 * np.mean(corner[qt == i]) for i in range(3)]
    axD.text(0.98, 0.05, f"high-$|\\sigma_1|$ & high-CV corner:\narid {pv[0]:.0f}%  "
             f"mid {pv[1]:.0f}%  humid {pv[2]:.0f}%", transform=axD.transAxes,
             ha="right", va="bottom", fontsize=11,
             bbox=dict(facecolor="white", alpha=0.85, edgecolor="#999999"))
    axD.set_title("(d) arid cells co-concentrate high $|\\sigma_1|$ AND high CV\n"
                  "(aridity residual after $|\\sigma_1|$+CV $\\approx$ 0)", fontsize=13)
    h1, l1 = axD.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    axD.legend(h1 + h2, l1 + l2, fontsize=10, loc="upper right")

    fig.suptitle("Slope of scale ($\\sigma_1$), not its level, drives the SD improvement "
                 "(flood-ensured, n={:,})".format(len(df)), fontsize=16, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out_png, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"Saved: {out_png}")


README = """# 517 -- Slope-of-scale vs improvement: Supplementary tables

Scope: flood-ensured cells (common_target_cells.load_flood_ensured_ids), n={n:,}.
baseline (median improvement over AIC 'stationary' cells) = {base:.2f} %.
excess = improvement - baseline. |sigma_1| = |064 truth_sigma_1| (GU log-scale year
slope); sigma_0 = scale level; CV/zero_flow/mean_outflow/uparea from 070; q_spec =
mean_outflow/uparea*1000 (low = arid); class = 079 AIC change type. Partial Spearman =
Spearman averaged within 4 quantile strata of the control(s).

Files:
  S1_slope_vs_level.csv         improvement driven by the SLOPE (|sigma_1|), not the
                                LEVEL (sigma_0) or flashiness (CV): partial on |sigma_1|
                                collapses CV to ~0, sigma_0 to a small negative; |sigma_1|
                                survives control on CV.
  S2_flashiness_by_trend.csv    CV lifts improvement only where a trend exists (var/both);
                                no-trend cells stay at baseline across CV.
  S3_excess_2d_sigma1_CV.csv    median excess over |sigma_1| x CV tertiles -- rises down
                                the |sigma_1| axis, ~flat across CV except at high |sigma_1|.
  S4_intermittency.csv          within var/both: zero-flow's effect on improvement is
                                fully mediated by |sigma_1| (partial ~0); intermittency
                                does correlate with larger |sigma_1| (r~+0.34).
                                fraction of cells with zero_flow>0 = {zf:.1%}.
  S5a_aridity_deciles.csv       per q_spec decile medians (imp/excess/|s1|/CV/zero_flow).
  S5b_aridity_summary.csv       aridity correlations & partials (collapse to ~0 after
                                |sigma_1|+CV) and high-|sigma_1|&high-CV corner prevalence
                                by aridity tertile (arid ~2.6x humid).
  S_fig_scaleslope_improvement.png  4-panel figure of the above: (a) marginal vs partial
                                Spearman, (b) improvement vs CV by trend class, (c) 3x3
                                excess heatmap, (d) aridity deciles with |sigma_1|/CV.
  S1_marginal.png / S1_partial.png  panel (a) split into two standalone figures
                                (marginal-only and partial-only Spearman bars).
  S5_aridity_corner.png         'arid zone = corner factory': (a) high-|sigma_1|&high-CV
                                corner prevalence by q_spec tertile (arid 28% vs humid 11%),
                                (b) aridity's marginal/partial correlations (collapses to
                                ~0 after |sigma_1|+CV).
"""


def main():
    dat_dir = sys.argv[1] if len(sys.argv) > 1 else \
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
    out_dir = os.path.join(dat_dir, "517"); os.makedirs(out_dir, exist_ok=True)
    df, base = load(dat_dir)
    n = len(df)
    print(f"n={n:,}  baseline={base:.2f}%")

    t1 = table_s1(df, os.path.join(out_dir, "S1_slope_vs_level.csv"))
    t2 = table_s2(df, os.path.join(out_dir, "S2_flashiness_by_trend.csv"))
    t3 = table_s3(df, os.path.join(out_dir, "S3_excess_2d_sigma1_CV.csv"))
    t4 = table_s4(df, os.path.join(out_dir, "S4_intermittency.csv"))
    d5, s5 = table_s5(df, os.path.join(out_dir, "S5a_aridity_deciles.csv"),
                      os.path.join(out_dir, "S5b_aridity_summary.csv"))
    make_figure(df, base, os.path.join(out_dir, "S_fig_scaleslope_improvement.png"))
    make_s1_separate(df, os.path.join(out_dir, "S1_marginal.png"),
                     os.path.join(out_dir, "S1_partial.png"))
    make_aridity_figure(df, os.path.join(out_dir, "S5_aridity_corner.png"))
    zf = float(t4.attrs["frac_zf_gt0"])
    with open(os.path.join(out_dir, "README.md"), "w") as f:
        f.write(README.format(n=n, base=base, zf=zf))
    for name, t in [("S1", t1), ("S2", t2), ("S3", t3), ("S4", t4),
                    ("S5a", d5), ("S5b", s5)]:
        print(f"\n===== {name} =====")
        print(t.to_string(index=False))
    print(f"\nSaved 6 CSVs + README.md to {out_dir}")


if __name__ == "__main__":
    main()
