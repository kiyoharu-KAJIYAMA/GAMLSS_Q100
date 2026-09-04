"""
lib_attenuation_tests.py
Attribution test for the eq_ens-tail flow gradient found by 069.

NOTE: an earlier version also contained a "signal-occurrence" test A (change-type
detection rate by flow/uparea bins). It was REMOVED as invalid: a detection RATE
confounds occurrence with detection POWER (smoother large-basin series raise the
power of the AIC/Levene classification), so a flat rate cannot establish that
variance signals persist downstream. The smoothing mechanism is tested directly
in lib_cv_smoothing.py instead.

B) AGGREGATION-vs-FLOW attribution for the eq_ens tail:
   The 069 gradient uses flow bins; flow and uparea are correlated but decouple
   in arid basins (large uparea, small flow). O/E is computed across uparea bins
   (same exact-Poisson-CI methodology as 069) and in a coarse 2-D flow x uparea
   table (cells with expected < 5 masked):
     -> 069c/<parent>/<ns_model>/tail_enrichment_uparea.csv + oe_by_uparea.png
        + oe_2d_flow_uparea.png / .csv
   See also 500_oe_by_specific_discharge.py, which shows the common driver is
   specific discharge (aridity), with a secondary uparea effect.

Scope mirrors 069 (flood-ensured, minus reverse-flow/Antarctica).

Usage:
  python3 lib_attenuation_tests.py [dat_dir] [parent st|ln|qd] [ns_model lin|qd|ad]
      [--top-pct P] [--uparea-path PATH] [--scope ensured|all]
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
excess_lib = importlib.import_module("common_target_cells")

RES = m066.RES
NX, NY = 3600, 1800
UPAREA_EDGES = np.array([50, 1e3, 1e4, 1e5, 1e6, np.inf])   # km2
UPAREA_LABELS = ["0.05–1k", "1–10k", "10–100k", "0.1–1M", "≥1M"]
UPAREA_DEFAULT = "/home/kk/jp_claude/gamlss/data/uparea.bin"

# coarse bins for the 2-D attribution table
FLOW2 = np.array([0, 100, 1000, 5000, np.inf]);  FLOW2_LAB = ["<100", "100–1k", "1–5k", "≥5k"]
UP2 = np.array([50, 1e3, 1e4, 1e5, np.inf]);     UP2_LAB = ["<1k", "1–10k", "10–100k", "≥100k"]


def poisson_ci(k, alpha=0.05):
    try:
        from scipy.stats import chi2
        lo = 0.0 if k == 0 else 0.5 * chi2.ppf(alpha / 2, 2 * k)
        hi = 0.5 * chi2.ppf(1 - alpha / 2, 2 * (k + 1))
        return lo, hi
    except Exception:
        z = 1.959964
        lo = k * (1 - 1.0 / (9 * k) - z / (3 * np.sqrt(k))) ** 3 if k > 0 else 0.0
        hi = (k + 1) * (1 - 1.0 / (9 * (k + 1)) + z / (3 * np.sqrt(k + 1))) ** 3
        return lo, hi


def binom_p(k, n, p0):
    try:
        from scipy.stats import binomtest
        return float(binomtest(k, n, p0).pvalue)
    except Exception:
        mu = n * p0
        sd = np.sqrt(n * p0 * (1 - p0))
        if sd == 0:
            return 1.0
        from math import erfc
        return float(erfc(abs((k - mu) / sd) / np.sqrt(2.0)))


def bin_index(vals, edges):
    """index of the half-open bin [e_i, e_{i+1}); -1 outside."""
    idx = np.searchsorted(edges, vals, side="right") - 1
    idx[(vals < edges[0]) | ~np.isfinite(vals)] = -1
    return np.clip(idx, -1, len(edges) - 2)


def main():
    argv = sys.argv[1:]
    top_pct = 1.0
    uparea_path = UPAREA_DEFAULT
    scope = "ensured"

    def take(flag, cast=str):
        nonlocal argv
        if flag in argv:
            i = argv.index(flag); val = argv[i + 1]; del argv[i:i + 2]
            return cast(val)
        return None

    v = take("--top-pct", float); top_pct = v if v is not None else top_pct
    v = take("--uparea-path");    uparea_path = v if v is not None else uparea_path
    v = take("--scope");          scope = v if v is not None else scope

    dat_dir = argv[0] if len(argv) > 0 else \
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
    parent = argv[1] if len(argv) > 1 else "ln"
    ns_model = argv[2] if len(argv) > 2 else "lin"

    out_bm = os.path.join(dat_dir, "069c", parent, ns_model)   # own dir, separate from 069
    os.makedirs(out_bm, exist_ok=True)

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
    if not os.path.exists(uparea_path):
        sys.exit(f"uparea not found: {uparea_path} (needed for B)")
    up_grid = np.fromfile(uparea_path, dtype="<f4").reshape(NY, NX) / 1.0e6

    eq_l, mo_l, up_l2 = [], [], []
    for cid, (iy, ix, eqv) in eqmap.items():
        if keep_ids is not None and cid not in keep_ids:
            continue
        if (iy, ix) in bad_yx:
            continue
        if 90.0 - (iy + 0.5) * RES < m066.LAT_CUT:
            continue
        mo = momap.get(cid, np.nan)
        if not (np.isfinite(eqv) and np.isfinite(mo)):
            continue
        eq_l.append(eqv); mo_l.append(mo); up_l2.append(up_grid[iy, ix])
    eq = np.array(eq_l); mo = np.array(mo_l); up = np.array(up_l2)
    n = eq.size
    eq_thr = float(np.percentile(eq, 100.0 - top_pct))
    tail = eq >= eq_thr
    p_tail = tail.sum() / float(n)
    print(f"B) scope cells: {n}; tail (top {top_pct:g}%): {int(tail.sum())} (p_tail={p_tail:.5f})")

    # --- O/E by uparea bins (same methodology as 069's flow version) ---
    idx_up = bin_index(up, UPAREA_EDGES)
    enr_csv = os.path.join(out_bm, "tail_enrichment_uparea.csv")
    rows_oe = []
    with open(enr_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bin_km2", "n_domain", "observed_tail", "expected_tail",
                    "obs_over_exp", "p_binomial"])
        print(f"{'uparea bin':>10} {'n_domain':>9} {'obs':>7} {'exp':>9} {'O/E':>6} {'p':>10}")
        for b, lab in enumerate(UPAREA_LABELS):
            inb = idx_up == b
            nd = int(inb.sum()); k = int((inb & tail).sum()); e = nd * p_tail
            oe = k / e if e > 0 else np.nan
            pv = binom_p(k, nd, p_tail) if nd > 0 else np.nan
            lo, hi = poisson_ci(k)
            rows_oe.append((lab, nd, k, e, oe, pv, lo / e if e > 0 else np.nan,
                            hi / e if e > 0 else np.nan))
            print(f"{lab:>10} {nd:>9d} {k:>7d} {e:>9.1f} {oe:>6.2f} {pv:>10.3g}")
            w.writerow([lab, nd, k, f"{e:.2f}", f"{oe:.3f}", f"{pv:.4g}"])
    print(f"Saved: {enr_csv}")

    figo, axo = plt.subplots(figsize=(7.5, 5.0))
    xs = np.arange(len(UPAREA_LABELS))
    oes = [r[4] for r in rows_oe]
    axo.axhline(1.0, color="#666666", lw=0.8, ls="--")
    axo.plot(xs, oes, "-", color="#555555", lw=1.0)
    for i, r in enumerate(rows_oe):
        col = "#9e9e9e" if r[5] >= 0.05 else ("#2166ac" if r[4] < 1 else "#b2182b")
        axo.plot([i, i], [r[6], r[7]], color=col, lw=1.4)
        axo.plot(i, r[4], "o", color=col, markersize=7)
        axo.text(i, r[7] * 1.15, f"{r[4]:.2f}", ha="center", fontsize=9)
    axo.set_yscale("log")
    axo.set_xticks(xs); axo.set_xticklabels(UPAREA_LABELS)
    axo.set_xlabel("upstream area [km2]")
    axo.set_ylabel("observed / expected (O/E)")
    axo.grid(axis="y", alpha=0.3)
    axo.set_title(f"eq_ens top-{top_pct:g}% tail enrichment by AGGREGATION (uparea)\n"
                  f"({m069.MODEL_LABEL[ns_model]}/{m069.PARENT_LABEL[parent]})", fontsize=11)
    figo.tight_layout()
    p = os.path.join(out_bm, "oe_by_uparea.png")
    figo.savefig(p, dpi=150, bbox_inches="tight"); plt.close(); print(f"Saved: {p}")

    # --- 2-D attribution: O/E in coarse flow x uparea cells ---
    fi = bin_index(mo, FLOW2)
    ui = bin_index(up, UP2)
    nf, nu = len(FLOW2_LAB), len(UP2_LAB)
    oe2 = np.full((nu, nf), np.nan)
    n2 = np.zeros((nu, nf), int); k2 = np.zeros((nu, nf), int)
    for b_u in range(nu):
        for b_f in range(nf):
            inb = (ui == b_u) & (fi == b_f)
            nd = int(inb.sum()); k = int((inb & tail).sum()); e = nd * p_tail
            n2[b_u, b_f] = nd; k2[b_u, b_f] = k
            if e >= 5:                                 # mask unstable cells
                oe2[b_u, b_f] = k / e

    csv2 = os.path.join(out_bm, "oe_2d_flow_uparea.csv")
    with open(csv2, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["uparea_bin", "flow_bin", "n_domain", "observed", "O/E"])
        for b_u in range(nu):
            for b_f in range(nf):
                w.writerow([UP2_LAB[b_u], FLOW2_LAB[b_f], n2[b_u, b_f], k2[b_u, b_f],
                            "" if np.isnan(oe2[b_u, b_f]) else f"{oe2[b_u, b_f]:.3f}"])
    print(f"Saved: {csv2}")

    fig2, ax2 = plt.subplots(figsize=(7.6, 5.6))
    with np.errstate(invalid="ignore"):
        img = np.log10(oe2)
    cmap = plt.get_cmap("RdBu_r").copy(); cmap.set_bad("#e0e0e0")
    im = ax2.imshow(img, origin="lower", cmap=cmap, vmin=-1.0, vmax=1.0, aspect="auto")
    for b_u in range(nu):
        for b_f in range(nf):
            if np.isnan(oe2[b_u, b_f]):
                txt = f"(n={n2[b_u, b_f]:,})"
            else:
                txt = f"{oe2[b_u, b_f]:.2f}\nn={n2[b_u, b_f]:,}"
            ax2.text(b_f, b_u, txt, ha="center", va="center", fontsize=8.5)
    ax2.set_xticks(range(nf)); ax2.set_xticklabels(FLOW2_LAB)
    ax2.set_yticks(range(nu)); ax2.set_yticklabels(UP2_LAB)
    ax2.set_xlabel("mean AMAX flow [m3/s]")
    ax2.set_ylabel("upstream area [km2]")
    cb = plt.colorbar(im, ax=ax2)
    cb.set_label("log10(O/E)   (blue = depleted, red = enriched)")
    ax2.set_title("Which drives the tail depletion: FLOW (columns) or AGGREGATION (rows)?\n"
                  "read down a column (uparea effect at fixed flow) and along a row",
                  fontsize=11)
    fig2.tight_layout()
    p = os.path.join(out_bm, "oe_2d_flow_uparea.png")
    fig2.savefig(p, dpi=150, bbox_inches="tight"); plt.close(); print(f"Saved: {p}")


if __name__ == "__main__":
    main()
