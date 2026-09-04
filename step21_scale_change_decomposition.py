"""
step21_scale_change_decomposition.py
Decompose the CV -> tail link of 501 into its components and identify the TRUE
driver of eq_ens tail entry. Candidate drivers, all computed per cell from the
064 ln-parent GU coefficients (mu_neg(t)=mu0+mu1*t identity link; log sigma(t)
=sig0+sig1*t) and the 070 mean AMAX:

  sig_chg   = |sigma_1| * 119          relative (log) SCALE change over 120 yr
  mu_chg    = |mu_1| * 119 / mean_amax relative MEAN-trend amplitude
  noise_rel = sigma(mid) / mean_amax   static relative noise level (no change)

User's recollection to test: cells with HIGH CV but NO scale change gained
little -> the tail should follow sig_chg, not raw CV / static noise.

Analyses (top-1% eq_ens tail, O/E with exact Poisson CIs as 069/500/501):
  1. oe_by_drivers.png    : 1D O/E across sextiles of each driver side by side.
  2. oe_2d_sigchg_noise.png : mediation table sig_chg x noise_rel (expected>=5).
     Prediction (user memory): gradient along sig_chg, ~flat along noise.
  3. highcv_split.csv (+stdout): within the TOP CV quartile only, O/E split by
     sig_chg quartile -> "high CV without scale change stays out of the tail".
  4. sigchg_2d_qspec_uparea.png : median sig_chg in q_spec x uparea cells ->
     does aggregation damp the scale-change amplitude (reinterpreting 501-i)?

Scope mirrors 069/500/501 (flood-ensured, minus reverse-flow/Antarctica).
Outputs: <dat_dir>/502/<parent>/<ns_model>/

Usage:
  python3 step21_scale_change_decomposition.py [dat_dir] [parent st|ln|qd]
      [ns_model lin|qd|ad] [--top-pct P] [--nq N] [--uparea-path PATH]
      [--scope ensured|all]
"""
import os
import sys
import csv
import glob
import importlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

m066 = importlib.import_module("lib_equivalent_ensemble_loader")
m069 = importlib.import_module("lib_equivalent_ensemble_by_flow")
m069c = importlib.import_module("lib_attenuation_tests")
m501 = importlib.import_module("lib_cv_smoothing")
excess_lib = importlib.import_module("common_target_cells")

RES = m066.RES
NX, NY = 3600, 1800
YEARS = np.arange(1981, 2101)              # record years (covariate of the fit)
SPAN = 119.0                                # year_end - year_start
MID = 0.5 * (YEARS[0] + YEARS[-1])
UPAREA_DEFAULT = "/home/kk/jp_claude/gamlss/data/uparea.bin"


def fmt(v):
    return f"{v:.2g}"


def load_ln_coeffs(dat_dir):
    """cell_id-indexed DataFrame of the ln-parent truth coefficients (064)."""
    files = sorted(glob.glob(os.path.join(dat_dir, "064", "ln", "mc_truth", "truth_*.csv")))
    if not files:
        sys.exit(f"No 064/ln/mc_truth chunks in {dat_dir} (run 065 first)")
    cols = ["cell_id", "truth_mu_0", "truth_mu_1", "truth_sigma_0", "truth_sigma_1"]
    parts = [pd.read_csv(f, usecols=cols) for f in files]
    df = pd.concat(parts, ignore_index=True).dropna()
    df = df.set_index("cell_id")
    print(f"ln coefficients: {len(df)} cells from {len(files)} chunks")
    return df


def qbins(vals, k):
    e = np.quantile(vals, np.linspace(0, 1, k + 1))
    e[0] -= 1e-12; e[-1] += 1e-12
    idx = np.clip(np.searchsorted(e, vals, side="right") - 1, 0, k - 1)
    labs = [f"{fmt(e[b])}–{fmt(e[b+1])}" for b in range(k)]
    return idx, labs


def oe_rows(idx, labs, tail, p_tail):
    out = []
    for b in range(len(labs)):
        inb = idx == b
        nd = int(inb.sum()); k = int((inb & tail).sum()); e = nd * p_tail
        oe = k / e if e > 0 else np.nan
        pv = m069c.binom_p(k, nd, p_tail) if nd > 0 else np.nan
        lo, hi = m069c.poisson_ci(k)
        out.append((labs[b], nd, k, e, oe, pv,
                    lo / e if e > 0 else np.nan, hi / e if e > 0 else np.nan))
    return out


def draw_oe(ax, rows, xlabel, title):
    xs = np.arange(len(rows))
    ax.axhline(1.0, color="#666666", lw=0.8, ls="--")
    ax.plot(xs, [r[4] for r in rows], "-", color="#555555", lw=1.0)
    for i, r in enumerate(rows):
        col = "#9e9e9e" if r[5] >= 0.05 else ("#2166ac" if r[4] < 1 else "#b2182b")
        ax.plot([i, i], [r[6], r[7]], color=col, lw=1.4)
        ax.plot(i, r[4], "o", color=col, markersize=6)
        ax.text(i, r[7] * 1.15, f"{r[4]:.2f}", ha="center", fontsize=8)
    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([r[0] for r in rows], rotation=35, ha="right", fontsize=8)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_title(title, fontsize=10)


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

    out_dir = os.path.join(dat_dir, "502", parent, ns_model)
    os.makedirs(out_dir, exist_ok=True)

    # ---- join eq_ens + mean_outflow + uparea + ln coeffs + CV over scope ----
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
    up_grid = np.fromfile(uparea_path, dtype="<f4").reshape(NY, NX) / 1.0e6
    co = load_ln_coeffs(dat_dir)

    cid_l, eq_l, mo_l, up_l = [], [], [], []
    for cid_, (iy, ix, eqv) in eqmap.items():
        if keep_ids is not None and cid_ not in keep_ids:
            continue
        if (iy, ix) in bad_yx:
            continue
        if 90.0 - (iy + 0.5) * RES < m066.LAT_CUT:
            continue
        mo = momap.get(cid_, np.nan)
        upv = up_grid[iy, ix]
        if not (np.isfinite(eqv) and np.isfinite(mo) and np.isfinite(upv) and upv > 0):
            continue
        cid_l.append(cid_); eq_l.append(eqv); mo_l.append(mo); up_l.append(upv)
    cid = np.array(cid_l, dtype=np.int64)
    eq = np.array(eq_l); mo = np.array(mo_l); up = np.array(up_l)

    sub = co.reindex(cid)
    mu1 = sub["truth_mu_1"].to_numpy()
    sg0 = sub["truth_sigma_0"].to_numpy()
    sg1 = sub["truth_sigma_1"].to_numpy()
    ok = np.isfinite(mu1) & np.isfinite(sg0) & np.isfinite(sg1) & (mo > 0)
    cid, eq, mo, up, mu1, sg0, sg1 = cid[ok], eq[ok], mo[ok], up[ok], mu1[ok], sg0[ok], sg1[ok]

    # drivers
    sig_chg = np.abs(sg1) * SPAN                    # |Delta log sigma| over 120 yr
    mu_chg = np.abs(mu1) * SPAN / mo                # relative mean-trend amplitude
    noise_rel = np.exp(sg0 + sg1 * MID) / mo        # static relative noise (mid-record)
    cv = m501.cell_cv(amax_path, cid)               # raw CV for the memory check
    qs = mo / up

    n = eq.size
    eq_thr = float(np.percentile(eq, 100.0 - top_pct))
    tail = eq >= eq_thr
    p_tail = tail.sum() / float(n)
    print(f"usable cells: {n}; tail: {int(tail.sum())} (p_tail={p_tail:.5f})")
    print(f"medians: sig_chg={np.median(sig_chg):.3f}  mu_chg={np.median(mu_chg):.3f}  "
          f"noise_rel={np.median(noise_rel):.3f}  CV={np.nanmedian(cv):.3f}")

    # ============ 1) 1D O/E for the three drivers ============
    drivers = [("sig_chg", sig_chg, "|Δlog σ| over 120 yr  (scale change)"),
               ("mu_chg", mu_chg, "|Δμ| / mean AMAX  (mean-trend amplitude)"),
               ("noise_rel", noise_rel, "σ(mid) / mean AMAX  (static noise)")]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    csv1 = os.path.join(out_dir, "oe_by_drivers.csv")
    with open(csv1, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["driver", "bin", "n_domain", "observed", "expected", "O/E", "p"])
        for ax, (name, vals, lab) in zip(axes, drivers):
            idx, labs = qbins(vals, nq)
            rows = oe_rows(idx, labs, tail, p_tail)
            for r in rows:
                w.writerow([name, r[0], r[1], r[2], f"{r[3]:.1f}", f"{r[4]:.3f}", f"{r[5]:.3g}"])
            draw_oe(ax, rows, lab, f"O/E by {name}")
            print(f"{name}: " + "  ".join(f"{r[4]:.2f}" for r in rows))
    axes[0].set_ylabel("observed / expected (O/E)")
    fig.suptitle(f"Which component drives eq_ens tail entry? "
                 f"({m069.MODEL_LABEL[ns_model]}/{m069.PARENT_LABEL[parent]}, "
                 f"sextiles, n≈{n // nq:,}/bin)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    p = os.path.join(out_dir, "oe_by_drivers.png")
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved: {csv1}\nSaved: {p}")

    # ============ 2) mediation: sig_chg x noise_rel ============
    si, s_lab = qbins(sig_chg, 4)
    ni, n_lab = qbins(noise_rel, 4)
    oe2 = np.full((4, 4), np.nan)
    n2 = np.zeros((4, 4), int)
    for bn in range(4):
        for bs in range(4):
            inb = (ni == bn) & (si == bs)
            nd = int(inb.sum()); k = int((inb & tail).sum()); e = nd * p_tail
            n2[bn, bs] = nd
            if e >= 5:
                oe2[bn, bs] = k / e
    fig2, ax2 = plt.subplots(figsize=(8.4, 6.0))
    with np.errstate(invalid="ignore"):
        img = np.log10(oe2)
    cmap = plt.get_cmap("RdBu_r").copy(); cmap.set_bad("#e0e0e0")
    im = ax2.imshow(img, origin="lower", cmap=cmap, vmin=-1.0, vmax=1.0, aspect="auto")
    for bn in range(4):
        for bs in range(4):
            txt = (f"(n={n2[bn, bs]:,})" if np.isnan(oe2[bn, bs])
                   else f"{oe2[bn, bs]:.2f}\nn={n2[bn, bs]:,}")
            ax2.text(bs, bn, txt, ha="center", va="center", fontsize=8.5)
    ax2.set_xticks(range(4)); ax2.set_xticklabels(s_lab, rotation=20, ha="right")
    ax2.set_yticks(range(4)); ax2.set_yticklabels(n_lab)
    ax2.set_xlabel("|Δlog σ| quartile  (scale change →)")
    ax2.set_ylabel("static noise quartile (σ/mean AMAX)")
    cb = plt.colorbar(im, ax=ax2); cb.set_label("log10(O/E)")
    ax2.set_title("Mediation: scale change (columns) vs static noise (rows)\n"
                  "memory prediction: gradient along columns, ~flat rows", fontsize=11)
    fig2.tight_layout()
    p = os.path.join(out_dir, "oe_2d_sigchg_noise.png")
    fig2.savefig(p, dpi=150, bbox_inches="tight"); plt.close(); print(f"Saved: {p}")

    # ============ 3) user-memory check: high CV without scale change ============
    okc = np.isfinite(cv)
    cv_hi = okc & (cv >= np.nanquantile(cv, 0.75))       # top CV quartile only
    si_all, s_lab4 = qbins(sig_chg, 4)
    print("\nHigh-CV subgroup (top CV quartile) split by scale change:")
    csv3 = os.path.join(out_dir, "highcv_split.csv")
    with open(csv3, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sig_chg_quartile", "n", "observed", "expected", "O/E", "p"])
        for b in range(4):
            inb = cv_hi & (si_all == b)
            nd = int(inb.sum()); k = int((inb & tail).sum()); e = nd * p_tail
            oe = k / e if e > 0 else np.nan
            pv = m069c.binom_p(k, nd, p_tail) if nd > 0 else np.nan
            print(f"  sig_chg {s_lab4[b]:>12}: n={nd:6d}  O/E={oe:5.2f}  p={pv:.3g}")
            w.writerow([s_lab4[b], nd, k, f"{e:.1f}", f"{oe:.3f}", f"{pv:.3g}"])
    print(f"Saved: {csv3}")

    # ============ 4) does aggregation damp the scale-change amplitude? ============
    qi, q_lab = qbins(qs, 4)
    ui, u_lab = qbins(up, 4)
    a2 = np.full((4, 4), np.nan)
    m2 = np.zeros((4, 4), int)
    for bu in range(4):
        for bq in range(4):
            inb = (ui == bu) & (qi == bq)
            m2[bu, bq] = int(inb.sum())
            if m2[bu, bq] >= 100:
                a2[bu, bq] = float(np.median(sig_chg[inb]))
    csv4 = os.path.join(out_dir, "sigchg_2d_qspec_uparea.csv")
    with open(csv4, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["uparea_bin_km2", "qspec_bin", "n", "median_abs_dlog_sigma"])
        for bu in range(4):
            for bq in range(4):
                w.writerow([u_lab[bu], q_lab[bq], m2[bu, bq],
                            "" if np.isnan(a2[bu, bq]) else f"{a2[bu, bq]:.4f}"])
    print(f"Saved: {csv4}")

    fig4, ax4 = plt.subplots(figsize=(8.4, 6.0))
    cmap4 = plt.get_cmap("YlOrBr").copy(); cmap4.set_bad("#e0e0e0")
    im4 = ax4.imshow(a2, origin="lower", cmap=cmap4, aspect="auto")
    for bu in range(4):
        for bq in range(4):
            txt = ("--" if np.isnan(a2[bu, bq])
                   else f"{a2[bu, bq]:.3f}\nn={m2[bu, bq]:,}")
            ax4.text(bq, bu, txt, ha="center", va="center", fontsize=8.5)
    ax4.set_xticks(range(4)); ax4.set_xticklabels(q_lab, rotation=20, ha="right")
    ax4.set_yticks(range(4)); ax4.set_yticklabels(u_lab)
    ax4.set_xlabel("specific discharge quartile (dry → wet)")
    ax4.set_ylabel("upstream area quartile [km$^2$]")
    cb4 = plt.colorbar(im4, ax=ax4); cb4.set_label("median |Δlog σ|")
    ax4.set_title("Does aggregation damp the SCALE-CHANGE amplitude?\n"
                  "(median |Δlog σ| in q_spec x uparea cells)", fontsize=11)
    fig4.tight_layout()
    p = os.path.join(out_dir, "sigchg_2d_qspec_uparea.png")
    fig4.savefig(p, dpi=150, bbox_inches="tight"); plt.close(); print(f"Saved: {p}")


if __name__ == "__main__":
    main()
