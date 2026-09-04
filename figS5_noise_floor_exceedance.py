"""
figS5_noise_floor_exceedance.py
Floor-corrected scale-change statistics: how much of the observed |Delta log
sigma| (502) is REAL signal, once the estimation noise floor (503) is removed?

Uses the stationary-truth null draws from 503 (universal for all cells; the
mu-trend config is only an invariance check). Two floor-corrected metrics per
q_spec x uparea quartile cell (and per q_spec quartile alone):

  frac_above  = fraction of cells with sig_chg > null 95th percentile
                (expected 5% if the group had NO true scale change)
  rms_true    = sqrt(max(0, mean(sig_chg^2) - mean(null^2)))
                (RMS of the true signal, by variance subtraction: the fitted
                 slope is ~ true + independent noise, so second moments add)

Answers: "is the true sigma-change genuinely small in humid basins?" and
"does the dry-column damping with uparea survive floor correction?"

Inputs : <dat_dir>/503/null_dlogsigma.csv (run 503 first), 064/ln coefficients,
         063/070/uparea + scope as 502.
Outputs: <dat_dir>/504/<parent>/<ns_model>/floor_corrected_sigchg.{csv,png}

Usage:
  python3 figS5_noise_floor_exceedance.py [dat_dir] [parent] [ns_model]
      [--uparea-path PATH] [--scope ensured|all]
"""
import os
import sys
import csv
import importlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

m066 = importlib.import_module("lib_equivalent_ensemble_loader")
m069 = importlib.import_module("lib_equivalent_ensemble_by_flow")
m502 = importlib.import_module("step21_scale_change_decomposition")
excess_lib = importlib.import_module("common_target_cells")

RES = m066.RES
NX, NY = 3600, 1800
SPAN = 119.0
UPAREA_DEFAULT = "/home/kk/jp_claude/gamlss/data/uparea.bin"


def main():
    argv = sys.argv[1:]
    uparea_path = UPAREA_DEFAULT; scope = "ensured"

    def take(flag, cast=str):
        nonlocal argv
        if flag in argv:
            i = argv.index(flag); val = argv[i + 1]; del argv[i:i + 2]
            return cast(val)
        return None

    v = take("--uparea-path"); uparea_path = v if v is not None else uparea_path
    v = take("--scope");       scope = v if v is not None else scope

    dat_dir = argv[0] if len(argv) > 0 else \
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
    parent = argv[1] if len(argv) > 1 else "ln"
    ns_model = argv[2] if len(argv) > 2 else "lin"
    out_dir = os.path.join(dat_dir, "504", parent, ns_model)
    os.makedirs(out_dir, exist_ok=True)

    # ---- null floor from 503 ----
    null_csv = os.path.join(dat_dir, "503", "null_dlogsigma.csv")
    if not os.path.exists(null_csv):
        sys.exit(f"Not found: {null_csv} (run 503 first)")
    nd = pd.read_csv(null_csv)
    d_st = nd.loc[nd["config"] == "stationary", "dlogsigma"].to_numpy()
    d_mt = nd.loc[nd["config"] == "mu_trend", "dlogsigma"].to_numpy()
    null_med = float(np.median(d_st))
    null_q95 = float(np.quantile(d_st, 0.95))
    null_ms = float(np.mean(d_st ** 2))
    print(f"null (stationary): median={null_med:.3f}  q95={null_q95:.3f}  "
          f"rms={np.sqrt(null_ms):.3f}  (B={d_st.size})")
    if d_mt.size:
        print(f"null (mu-trend invariance check): median={np.median(d_mt):.3f}  "
              f"q95={np.quantile(d_mt, 0.95):.3f}  (B={d_mt.size})")

    # ---- join (as 502): eq/mo/uparea + ln coefficients over scope ----
    eq_path = os.path.join(dat_dir, "063", f"eq_ens_allgrid_{parent}_{ns_model}.csv")
    mo_path = os.path.join(dat_dir, "070", "timeseries_tests.csv")
    eqmap = m069.load_eq_ens(eq_path)
    momap = m069.load_mean_outflow(mo_path)
    bad_yx = excess_lib.load_bad_yx(dat_dir)
    keep_ids = None
    if scope == "ensured":
        keep_ids = excess_lib.load_flood_ensured_ids(dat_dir)
        if not keep_ids:
            keep_ids = None
    up_grid = np.fromfile(uparea_path, dtype="<f4").reshape(NY, NX) / 1.0e6
    co = m502.load_ln_coeffs(dat_dir)

    cid_l, mo_l, up_l = [], [], []
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
        cid_l.append(cid_); mo_l.append(mo); up_l.append(upv)
    cid = np.array(cid_l, dtype=np.int64)
    mo = np.array(mo_l); up = np.array(up_l)
    sub = co.reindex(cid)
    sg1 = sub["truth_sigma_1"].to_numpy()
    ok = np.isfinite(sg1) & (mo > 0)
    mo, up, sg1 = mo[ok], up[ok], sg1[ok]
    sig_chg = np.abs(sg1) * SPAN
    qs = mo / up
    print(f"usable cells: {sig_chg.size}")

    qi, q_lab = m502.qbins(qs, 4)
    ui, u_lab = m502.qbins(up, 4)

    def metrics(mask):
        x = sig_chg[mask]
        if x.size < 100:
            return np.nan, np.nan, x.size
        frac = float((x > null_q95).mean())
        rms_true = float(np.sqrt(max(0.0, np.mean(x ** 2) - null_ms)))
        return frac, rms_true, x.size

    # ---- per q_spec quartile (1D) ----
    print("\nby q_spec quartile (dry -> wet):")
    print(f"{'q_spec':>16} {'n':>8} {'frac>q95':>9} {'rms_true':>9}  (null exp 5%)")
    for b in range(4):
        fr, rt, nn = metrics(qi == b)
        print(f"{q_lab[b]:>16} {nn:>8d} {fr:>9.3f} {rt:>9.3f}")

    # ---- 2D q_spec x uparea ----
    frac2 = np.full((4, 4), np.nan); rms2 = np.full((4, 4), np.nan)
    n2 = np.zeros((4, 4), int)
    for bu in range(4):
        for bq in range(4):
            fr, rt, nn = metrics((ui == bu) & (qi == bq))
            frac2[bu, bq] = fr; rms2[bu, bq] = rt; n2[bu, bq] = nn

    out_csv = os.path.join(out_dir, "floor_corrected_sigchg.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["uparea_bin_km2", "qspec_bin", "n",
                    "frac_above_null_q95", "rms_true_signal",
                    "null_med", "null_q95"])
        for bu in range(4):
            for bq in range(4):
                w.writerow([u_lab[bu], q_lab[bq], n2[bu, bq],
                            f"{frac2[bu, bq]:.4f}", f"{rms2[bu, bq]:.4f}",
                            f"{null_med:.4f}", f"{null_q95:.4f}"])
    print(f"Saved: {out_csv}")

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8))
    for ax, mat, lab, cm in [
            (axes[0], frac2, "fraction of cells above null q95\n(5% = no true signal)", "YlOrBr"),
            (axes[1], rms2, "RMS of TRUE |Δlog σ| (floor-subtracted)", "YlOrBr")]:
        cmap = plt.get_cmap(cm).copy(); cmap.set_bad("#e0e0e0")
        im = ax.imshow(mat, origin="lower", cmap=cmap, aspect="auto")
        for bu in range(4):
            for bq in range(4):
                txt = ("--" if np.isnan(mat[bu, bq])
                       else f"{mat[bu, bq]:.2f}\nn={n2[bu, bq]:,}")
                ax.text(bq, bu, txt, ha="center", va="center", fontsize=8.5)
        ax.set_xticks(range(4)); ax.set_xticklabels(q_lab, rotation=20, ha="right")
        ax.set_yticks(range(4)); ax.set_yticklabels(u_lab)
        ax.set_xlabel("specific discharge quartile (dry → wet)")
        ax.set_ylabel("upstream area quartile [km$^2$]")
        plt.colorbar(im, ax=ax)
        ax.set_title(lab, fontsize=11)
    fig.suptitle(f"Floor-corrected scale change (null med={null_med:.2f}, q95={null_q95:.2f})",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    p = os.path.join(out_dir, "floor_corrected_sigchg.png")
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(); print(f"Saved: {p}")


if __name__ == "__main__":
    main()
