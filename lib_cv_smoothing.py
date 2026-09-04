"""
lib_cv_smoothing.py
Direct measurement of the SMOOTHING mechanism behind the secondary uparea effect
found by 500 (at fixed aridity, larger catchments are 2-4x less likely to reach
the eq_ens top-1% tail). No change-type classification is used anywhere.

Hypothesis chain, each link measured directly from the AMAX series:
  (i)  SMOOTHNESS vs AGGREGATION: within each specific-discharge (aridity)
       quartile, the median CV (= sd/mean of the 120-yr AMAX) declines as
       upstream area grows -> "bigger basins have smoother series".
         -> cv_2d_qspec_uparea.png / .csv   (median CV heatmap)
  (ii) TAIL vs SMOOTHNESS: O/E of the eq_ens tail across CV quantile bins is a
       steep monotone increase -> "rougher series enter the tail".
         -> oe_by_cv.png / tail_enrichment_cv.csv
  (iii) MEDIATION: in a 2-D CV x uparea O/E table, the gradient runs along CV
       and the rows are ~flat -> once smoothness is fixed, basin size adds
       nothing; CV mediates the uparea effect.
         -> oe_2d_cv_uparea.png / .csv   (expected < 5 masked)

If (i)+(ii)+(iii) hold, the 500 result is explained mechanistically: aridity ->
high relative variability; aggregation -> smoothing; smoothness -> tail entry.

Scope mirrors 069/500 (flood-ensured, minus reverse-flow/Antarctica).

Outputs: <dat_dir>/501/<parent>/<ns_model>/

Usage:
  python3 lib_cv_smoothing.py [dat_dir] [parent st|ln|qd] [ns_model lin|qd|ad]
      [--top-pct P] [--nq N] [--uparea-path PATH] [--scope ensured|all]
"""
import os
import sys
import csv
import importlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

m066 = importlib.import_module("lib_equivalent_ensemble_loader")
m069 = importlib.import_module("lib_equivalent_ensemble_by_flow")
m069c = importlib.import_module("lib_attenuation_tests")
excess_lib = importlib.import_module("common_target_cells")

RES = m066.RES
NX, NY = 3600, 1800
N_YEARS = 120
UPAREA_DEFAULT = "/home/kk/jp_claude/gamlss/data/uparea.bin"


def fmt(v):
    return f"{v:.2g}"


def cell_cv(amax_path, cell_ids, chunk=100000):
    """CV = sd/mean of the 120-yr AMAX for each cell_id (record = cell_id row).
    Memory-mapped, chunked; mean <= 0 -> NaN."""
    n_total = os.path.getsize(amax_path) // (N_YEARS * 4)
    mm = np.memmap(amax_path, dtype="<f4", mode="r", shape=(n_total, N_YEARS))
    cv = np.full(cell_ids.size, np.nan)
    for s in range(0, cell_ids.size, chunk):
        ids = cell_ids[s:s + chunk]
        a = np.asarray(mm[ids], dtype=np.float64)
        mu = a.mean(axis=1)
        sd = a.std(axis=1, ddof=1)
        ok = mu > 0
        cv[s:s + chunk][ok] = sd[ok] / mu[ok]
    return cv


def main():
    argv = sys.argv[1:]
    top_pct = 1.0; nq = 6
    uparea_path = UPAREA_DEFAULT; scope = "ensured"

    def take(flag, cast=str):
        nonlocal argv
        if flag in argv:
            i = argv.index(flag); val = argv[i + 1]; del argv[i:i + 2]
            return cast(val)
        return None

    v = take("--top-pct", float); top_pct = v if v is not None else top_pct
    v = take("--nq", int);        nq = v if v is not None else nq
    v = take("--uparea-path");    uparea_path = v if v is not None else uparea_path
    v = take("--scope");          scope = v if v is not None else scope

    dat_dir = argv[0] if len(argv) > 0 else \
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
    parent = argv[1] if len(argv) > 1 else "ln"
    ns_model = argv[2] if len(argv) > 2 else "lin"

    out_dir = os.path.join(dat_dir, "501", parent, ns_model)
    os.makedirs(out_dir, exist_ok=True)

    # ---- join eq_ens + mean_outflow + uparea + CV over the 069 scope ----
    eq_path = os.path.join(dat_dir, "063", f"eq_ens_allgrid_{parent}_{ns_model}.csv")
    mo_path = os.path.join(dat_dir, "070", "timeseries_tests.csv")
    amax_path = os.path.join(dat_dir, "030", "amax_all.bin")
    eqmap = m069.load_eq_ens(eq_path)
    momap = m069.load_mean_outflow(mo_path)
    bad_yx = excess_lib.load_bad_yx(dat_dir)
    keep_ids = None
    if scope == "ensured":
        keep_ids = excess_lib.load_flood_ensured_ids(dat_dir)
        if not keep_ids:
            keep_ids = None
    if not os.path.exists(uparea_path):
        sys.exit(f"uparea not found: {uparea_path}")
    if not os.path.exists(amax_path):
        sys.exit(f"amax_all.bin not found: {amax_path}")
    up_grid = np.fromfile(uparea_path, dtype="<f4").reshape(NY, NX) / 1.0e6

    cid_l, eq_l, mo_l, up_l = [], [], [], []
    for cid, (iy, ix, eqv) in eqmap.items():
        if keep_ids is not None and cid not in keep_ids:
            continue
        if (iy, ix) in bad_yx:
            continue
        if 90.0 - (iy + 0.5) * RES < m066.LAT_CUT:
            continue
        mo = momap.get(cid, np.nan)
        upv = up_grid[iy, ix]
        if not (np.isfinite(eqv) and np.isfinite(mo) and np.isfinite(upv) and upv > 0):
            continue
        cid_l.append(cid); eq_l.append(eqv); mo_l.append(mo); up_l.append(upv)
    cid = np.array(cid_l, dtype=np.int64)
    eq = np.array(eq_l); mo = np.array(mo_l); up = np.array(up_l)
    qs = mo / up
    print(f"scope cells: {eq.size}; computing CV from {amax_path} ...")
    cv = cell_cv(amax_path, cid)
    ok = np.isfinite(cv)
    cid, eq, mo, up, qs, cv = cid[ok], eq[ok], mo[ok], up[ok], qs[ok], cv[ok]
    n = eq.size
    eq_thr = float(np.percentile(eq, 100.0 - top_pct))
    tail = eq >= eq_thr
    p_tail = tail.sum() / float(n)
    print(f"usable cells: {n}; tail: {int(tail.sum())} (p_tail={p_tail:.5f}); "
          f"CV median={np.median(cv):.3f}")

    def qbins(vals, k):
        e = np.quantile(vals, np.linspace(0, 1, k + 1))
        e[0] -= 1e-12; e[-1] += 1e-12
        idx = np.clip(np.searchsorted(e, vals, side="right") - 1, 0, k - 1)
        labs = [f"{fmt(e[b])}–{fmt(e[b+1])}" for b in range(k)]
        return idx, labs

    # ================= (i) median CV in q_spec x uparea cells =================
    nqq = nuu = 4
    qi, q_lab = qbins(qs, nqq)
    ui, u_lab = qbins(up, nuu)
    cv2 = np.full((nuu, nqq), np.nan)
    n2 = np.zeros((nuu, nqq), int)
    for bu in range(nuu):
        for bq in range(nqq):
            inb = (ui == bu) & (qi == bq)
            n2[bu, bq] = int(inb.sum())
            if n2[bu, bq] >= 100:
                cv2[bu, bq] = float(np.median(cv[inb]))
    csv1 = os.path.join(out_dir, "cv_2d_qspec_uparea.csv")
    with open(csv1, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["uparea_bin_km2", "qspec_bin", "n", "median_cv"])
        for bu in range(nuu):
            for bq in range(nqq):
                w.writerow([u_lab[bu], q_lab[bq], n2[bu, bq],
                            "" if np.isnan(cv2[bu, bq]) else f"{cv2[bu, bq]:.4f}"])
    print(f"Saved: {csv1}")

    fig1, ax1 = plt.subplots(figsize=(8.4, 6.0))
    cmap1 = plt.get_cmap("YlOrBr").copy(); cmap1.set_bad("#e0e0e0")
    im1 = ax1.imshow(cv2, origin="lower", cmap=cmap1, aspect="auto")
    for bu in range(nuu):
        for bq in range(nqq):
            txt = ("--" if np.isnan(cv2[bu, bq])
                   else f"{cv2[bu, bq]:.2f}\nn={n2[bu, bq]:,}")
            ax1.text(bq, bu, txt, ha="center", va="center", fontsize=8.5)
    ax1.set_xticks(range(nqq)); ax1.set_xticklabels(q_lab, rotation=20, ha="right")
    ax1.set_yticks(range(nuu)); ax1.set_yticklabels(u_lab)
    ax1.set_xlabel("specific discharge quartile [m$^3$ s$^{-1}$ km$^{-2}$]  (dry → wet)")
    ax1.set_ylabel("upstream area quartile [km$^2$]")
    cb1 = plt.colorbar(im1, ax=ax1); cb1.set_label("median CV of 120-yr AMAX")
    ax1.set_title("(i) Smoothness vs aggregation: median CV in q_spec x uparea cells\n"
                  "smoothing prediction: CV declines upward within each column",
                  fontsize=11)
    fig1.tight_layout()
    p = os.path.join(out_dir, "cv_2d_qspec_uparea.png")
    fig1.savefig(p, dpi=150, bbox_inches="tight"); plt.close(); print(f"Saved: {p}")

    # ================= (ii) O/E by CV quantile bins =================
    ci_idx, c_lab = qbins(cv, nq)
    rows = []
    enr_csv = os.path.join(out_dir, "tail_enrichment_cv.csv")
    with open(enr_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bin_cv", "n_domain", "observed_tail", "expected_tail",
                    "obs_over_exp", "p_binomial"])
        print(f"{'CV bin':>14} {'n_dom':>8} {'obs':>6} {'exp':>8} {'O/E':>6} {'p':>10}")
        for b in range(nq):
            inb = ci_idx == b
            nd = int(inb.sum()); k = int((inb & tail).sum()); e = nd * p_tail
            oe = k / e if e > 0 else np.nan
            pv = m069c.binom_p(k, nd, p_tail) if nd > 0 else np.nan
            lo, hi = m069c.poisson_ci(k)
            rows.append((c_lab[b], nd, k, e, oe, pv,
                         lo / e if e > 0 else np.nan, hi / e if e > 0 else np.nan))
            print(f"{c_lab[b]:>14} {nd:>8d} {k:>6d} {e:>8.1f} {oe:>6.2f} {pv:>10.3g}")
            w.writerow([c_lab[b], nd, k, f"{e:.2f}", f"{oe:.3f}", f"{pv:.4g}"])
    print(f"Saved: {enr_csv}")

    fig2, ax2 = plt.subplots(figsize=(8.2, 5.2))
    xs = np.arange(nq)
    ax2.axhline(1.0, color="#666666", lw=0.8, ls="--")
    ax2.plot(xs, [r[4] for r in rows], "-", color="#555555", lw=1.0)
    for i, r in enumerate(rows):
        col = "#9e9e9e" if r[5] >= 0.05 else ("#2166ac" if r[4] < 1 else "#b2182b")
        ax2.plot([i, i], [r[6], r[7]], color=col, lw=1.5)
        ax2.plot(i, r[4], "o", color=col, markersize=7)
        ax2.text(i, r[7] * 1.15, f"{r[4]:.2f}", ha="center", fontsize=9)
    ax2.set_yscale("log")
    ax2.set_xticks(xs); ax2.set_xticklabels(c_lab, rotation=30, ha="right")
    ax2.set_xlabel("CV of 120-yr AMAX  (smooth → rough)")
    ax2.set_ylabel("observed / expected (O/E)")
    ax2.grid(axis="y", alpha=0.3)
    ax2.set_title(f"(ii) eq_ens top-{top_pct:g}% tail enrichment by series roughness "
                  f"({m069.MODEL_LABEL[ns_model]}/{m069.PARENT_LABEL[parent]})", fontsize=11)
    fig2.tight_layout()
    p = os.path.join(out_dir, "oe_by_cv.png")
    fig2.savefig(p, dpi=150, bbox_inches="tight"); plt.close(); print(f"Saved: {p}")

    # ================= (iii) mediation: 2-D CV x uparea O/E =================
    ci4, c4_lab = qbins(cv, 4)
    oe3 = np.full((nuu, 4), np.nan)
    n3 = np.zeros((nuu, 4), int); k3 = np.zeros((nuu, 4), int)
    for bu in range(nuu):
        for bc in range(4):
            inb = (ui == bu) & (ci4 == bc)
            nd = int(inb.sum()); k = int((inb & tail).sum()); e = nd * p_tail
            n3[bu, bc] = nd; k3[bu, bc] = k
            if e >= 5:
                oe3[bu, bc] = k / e
    csv3 = os.path.join(out_dir, "oe_2d_cv_uparea.csv")
    with open(csv3, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["uparea_bin_km2", "cv_bin", "n_domain", "observed", "O/E"])
        for bu in range(nuu):
            for bc in range(4):
                w.writerow([u_lab[bu], c4_lab[bc], n3[bu, bc], k3[bu, bc],
                            "" if np.isnan(oe3[bu, bc]) else f"{oe3[bu, bc]:.3f}"])
    print(f"Saved: {csv3}")

    fig3, ax3 = plt.subplots(figsize=(8.4, 6.0))
    with np.errstate(invalid="ignore"):
        img = np.log10(oe3)
    cmap3 = plt.get_cmap("RdBu_r").copy(); cmap3.set_bad("#e0e0e0")
    im3 = ax3.imshow(img, origin="lower", cmap=cmap3, vmin=-1.0, vmax=1.0, aspect="auto")
    for bu in range(nuu):
        for bc in range(4):
            txt = (f"(n={n3[bu, bc]:,})" if np.isnan(oe3[bu, bc])
                   else f"{oe3[bu, bc]:.2f}\nn={n3[bu, bc]:,}")
            ax3.text(bc, bu, txt, ha="center", va="center", fontsize=8.5)
    ax3.set_xticks(range(4)); ax3.set_xticklabels(c4_lab, rotation=20, ha="right")
    ax3.set_yticks(range(nuu)); ax3.set_yticklabels(u_lab)
    ax3.set_xlabel("CV quartile  (smooth → rough)")
    ax3.set_ylabel("upstream area quartile [km$^2$]")
    cb3 = plt.colorbar(im3, ax=ax3)
    cb3.set_label("log10(O/E)   (blue = depleted, red = enriched)")
    ax3.set_title("(iii) Mediation: does CV absorb the uparea effect?\n"
                  "prediction: gradient along CV columns, ~flat uparea rows",
                  fontsize=11)
    fig3.tight_layout()
    p = os.path.join(out_dir, "oe_2d_cv_uparea.png")
    fig3.savefig(p, dpi=150, bbox_inches="tight"); plt.close(); print(f"Saved: {p}")


if __name__ == "__main__":
    main()
