"""
step09_bias_under_linear_parent.py
Compute the five per-cell error statistics of the Q100 estimate for the three
fitted models, from the Monte Carlo outputs of 040. No re-run of the MC is
needed: 040 stores every ensemble member's Q100 in mc_q100/*.bin and the truth
in mc_truth/*.csv. All five metrics are derived from a SINGLE delta-corrected
relative-error vector e, so their delta convention is guaranteed consistent
(and matches the 047 lightweight 3x3 definitions and manuscript Table 2).

Definitions (relative error e_m, member m):
  e_m  = (Q100_hat_m - Q100_truth) / Q100_truth
  bias = mean(e_m)                 systematic offset  -- NOT visible in the SD
  sd   = sd(e_m)                   spread (cross-check vs 040 mc_summary st/ln/qd_err_sd)
  rmse = sqrt(bias^2 + sd^2)       total error
  iqr  = q75(e_m) - q25(e_m)       robust spread
  tail = q95(e_m) - q05(e_m)       tail spread

Why bias/rmse: SD/IQR/tail are translation-invariant, so a model can have a
small SD while being systematically wrong. bias closes that gap; rmse combines
the two. If |bias| << sd for all models, the SD comparison is a valid proxy.

delta correction: 040 stores the fitted Q100 with an extra +delta relative to
truth_q100 (the fit is on the delta-shifted series, which already carries delta,
and 040 adds delta once more). SD/IQR/tail are unaffected (translation-
invariant), but the bias would be inflated by delta/truth (~25%). We subtract
the stored per-cell delta from the estimate here, so e is clean; the
denominator stays truth_q100 = truth_phys + delta (the Table-2 normalization).

mc_q100/q100_S_E.bin layout (from 040): 3 matrices [n_batch x n_mc] float32,
column-major (R), concatenated in order st, ln, qd. n_mc inferred from size.

Input:
  <dat_dir>/040/mc_q100/q100_SSSSSS_EEEEEE.bin
  <dat_dir>/040/mc_truth/truth_SSSSSS_EEEEEE.csv

This is the FAST, TABLE-ONLY step: it reads the heavy mc_q100 binaries once and
writes the per-cell CSV + console digest, with NO map rendering and no
matplotlib/cartopy import (faster startup). The bias/RMSE maps are produced
separately by 054b_plot_bias_maps.py, which reads only the small CSV below.

Output: <dat_dir>/054/
  bias_allgrid.csv   cell_id, iy, ix, truth_q100,
                     {st,ln,qd}_{bias,sd,rmse,iqr,tail}
  console: global median bias / sd / rmse / iqr / tail per model, plus a
           cross-check that 054's sd/iqr/tail reproduce 040/mc_summary
           cell-by-cell (max/median |rel diff| per model x stat; PASS/WARNING).
           Reads 040/mc_summary directly, so it does not depend on 050.

Usage:
  python3 step09_bias_under_linear_parent.py [dat_dir] [--force]
    reuses an existing 054/bias_allgrid.csv (no binary read); --force recomputes.
  python3 054b_plot_bias_maps.py [dat_dir] [vlim_pct]   # the maps
"""
import os
import sys
import csv
import glob
import time
import numpy as np
import importlib
excess_lib = importlib.import_module("common_target_cells")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RES = 0.1
MODELS = ["st", "ln", "qd"]
MODEL_NAME = {"st": "stationary", "ln": "linear", "qd": "quadratic"}
METRICS = ["bias", "sd", "rmse", "iqr", "tail"]


def read_chunk(q100_bin, truth_csv):
    """Return (records, n_mc). records: list of dicts per cell.

    delta correction: 040 adds delta to the fitted Q100 although the fit (made
    on the delta-shifted series) already contains it, so the stored estimates
    carry a spurious +delta offset relative to truth_q100. SD/IQR/tail are
    unaffected (translation-invariant), but the bias would be inflated by
    delta/truth (~25%). We subtract the stored per-cell delta here."""
    cells = []
    with open(truth_csv, "r", newline="") as f:
        for row in csv.DictReader(f):
            def g(key):
                v = row.get(key, "")
                return float(v) if v not in ("", "NA") else np.nan
            cells.append((int(row["cell_id"]), int(row["iy"]), int(row["ix"]),
                          g("truth_q100"), g("delta")))
    n_batch = len(cells)
    if n_batch == 0:
        return [], 0
    size = os.path.getsize(q100_bin)
    n_mc = size // (3 * n_batch * 4)
    if n_mc * 3 * n_batch * 4 != size:
        print(f"WARNING: size mismatch, skip {q100_bin}")
        return [], 0
    raw = np.fromfile(q100_bin, dtype="<f4")
    # 3 concatenated column-major matrices [n_batch x n_mc]
    mats = {}
    per = n_batch * n_mc
    for k, mod in enumerate(MODELS):
        mats[mod] = raw[k * per:(k + 1) * per].reshape((n_mc, n_batch)).T
    recs = []
    for i, (cid, iy, ix, tq, dlt) in enumerate(cells):
        if not np.isfinite(tq) or tq == 0.0:
            continue
        if not np.isfinite(dlt):
            dlt = 0.0
        rec = dict(cell_id=cid, iy=iy, ix=ix, truth_q100=tq)
        ok = True
        for mod in MODELS:
            e = (mats[mod][i].astype(float) - dlt - tq) / tq
            e = e[np.isfinite(e)]
            if len(e) < 2:
                ok = False
                break
            bias = float(np.mean(e))
            sd = float(np.std(e, ddof=1))
            q05, q25, q75, q95 = np.percentile(e, [5, 25, 75, 95])
            rec[f"{mod}_bias"] = bias
            rec[f"{mod}_sd"] = sd
            rec[f"{mod}_rmse"] = float(np.hypot(bias, sd))   # sqrt(bias^2 + sd^2)
            rec[f"{mod}_iqr"] = float(q75 - q25)
            rec[f"{mod}_tail"] = float(q95 - q05)
        if ok:
            recs.append(rec)
    return recs, n_mc


# Translation-invariant stats that 040 also stores in mc_summary (as
# {model}_err_{stat}); used to verify 054 reproduces 040 exactly.
CHECK_STATS = ["sd", "iqr", "tail"]


def cross_check_vs_mc_summary(all_recs, summary_dir, rtol=1e-3):
    """Verify 054's sd/iqr/tail match 040's mc_summary cell-by-cell.

    040 computes sd/iqr/tail during the MC (translation-invariant, so the
    spurious +delta on the estimate does not affect them). 054 recomputes them
    from the raw mc_q100 binaries after subtracting delta. They must agree to
    floating-point precision; any disagreement signals a delta/indexing bug.
    Reports max and median |relative difference| per model x stat and the number
    of cells exceeding rtol. Read directly from 040/mc_summary (no 050 needed).
    """
    files = sorted(glob.glob(os.path.join(summary_dir, "summary_*.csv")))
    if not files:
        print(f"\n[cross-check] no mc_summary in {summary_dir} -- skipped.")
        return
    ref = {}  # cell_id -> {f"{model}_{stat}": value}
    for fp in files:
        with open(fp, "r", newline="") as f:
            for row in csv.DictReader(f):
                try:
                    cid = int(row["cell_id"])
                except (KeyError, ValueError):
                    continue
                d = {}
                for m in MODELS:
                    for s in CHECK_STATS:
                        v = row.get(f"{m}_err_{s}", "")
                        d[f"{m}_{s}"] = float(v) if v not in ("", "NA") else np.nan
                ref[cid] = d

    print(f"\n[cross-check] 054 (from binaries) vs 040 mc_summary "
          f"({len(ref)} ref cells, rtol={rtol:g})")
    print(f"{'model':11s} {'stat':5s} {'matched':>8s} {'max |rel|':>11s} "
          f"{'med |rel|':>11s} {'n>rtol':>7s}")
    worst = 0.0
    for m in MODELS:
        for s in CHECK_STATS:
            a, b = [], []  # 054 value, 040 value
            for r in all_recs:
                rv = ref.get(r["cell_id"])
                if rv is None:
                    continue
                x, y = r[f"{m}_{s}"], rv[f"{m}_{s}"]
                # Skip arid garbage (truth_q100 ~ 0 -> spread ~1e137): both 054
                # and 040 compute it, but float roundoff makes the RELATIVE diff
                # explode there, which is not a reproduction error. 5.0 = 500%
                # relative-error spread, far above any physical cell.
                if np.isfinite(x) and np.isfinite(y) and abs(y) <= 5.0:
                    a.append(x); b.append(y)
            if not a:
                print(f"{MODEL_NAME[m]:11s} {s:5s} {0:8d} {'--':>11s} {'--':>11s} {'--':>7s}")
                continue
            a = np.array(a); b = np.array(b)
            denom = np.where(np.abs(b) > 0, np.abs(b), np.nan)
            rel = np.abs(a - b) / denom
            n_bad = int(np.nansum(rel > rtol))
            worst = max(worst, np.nanmax(rel))
            print(f"{MODEL_NAME[m]:11s} {s:5s} {len(a):8d} {np.nanmax(rel):11.2e} "
                  f"{np.nanmedian(rel):11.2e} {n_bad:7d}")
    verdict = "PASS: 054 reproduces 040 mc_summary" if worst <= rtol \
        else f"WARNING: max |rel diff| = {worst:.2e} > rtol -- investigate delta/indexing"
    print(f"[cross-check] {verdict}")


def print_digest_from_csv(out_csv, dat_dir):
    """Re-print the global median digest from an existing bias_allgrid.csv,
    without touching the heavy mc_q100 binaries (used on the reuse path).
    Excludes 031 negative-AMAX (reverse-flow) cells."""
    bad = excess_lib.load_bad_cells(dat_dir)
    cols = {f"{m}_{s}": [] for m in MODELS for s in METRICS}
    with open(out_csv, newline="") as f:
        for row in csv.DictReader(f):
            try:
                if int(row["cell_id"]) in bad:
                    continue
            except (KeyError, ValueError):
                pass
            for k in cols:
                v = row.get(k, "")
                cols[k].append(float(v) if v not in ("", "NA", "nan") else np.nan)
    hdr = f"\n{'model':11s}" + "".join(f"{'med ' + s + '%':>12s}" for s in METRICS)
    hdr += f"{'med |bias|/sd':>15s}"
    print(hdr)
    for m in MODELS:
        vals = {s: np.array(cols[f"{m}_{s}"]) * 100.0 for s in METRICS}
        ratio = np.abs(vals["bias"]) / np.where(vals["sd"] > 0, vals["sd"], np.nan)
        line = f"{MODEL_NAME[m]:11s}" + "".join(f"{np.nanmedian(vals[s]):12.2f}" for s in METRICS)
        line += f"{np.nanmedian(ratio):15.2f}"
        print(line)


def main():
    force = "--force" in sys.argv
    pos = [a for a in sys.argv[1:] if not a.startswith("--")]
    dat_dir = pos[0] if pos else os.path.join(SCRIPT_DIR, "..", "data")
    out_dir = os.path.join(dat_dir, "054")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "bias_allgrid.csv")

    # Reuse: bias_allgrid.csv is the cached result of the heavy ~55 GB mc_q100
    # read. If it already exists, skip the recompute entirely (no binary read) --
    # pass --force to rebuild it from the binaries (e.g. after 040 changed).
    if os.path.exists(out_csv) and not force:
        print(f"{out_csv} exists -> reusing (no binary read; pass --force to recompute).")
        print_digest_from_csv(out_csv, dat_dir)
        return

    q100_dir = os.path.join(dat_dir, "040", "mc_q100")
    truth_dir = os.path.join(dat_dir, "040", "mc_truth")
    q_files = sorted(glob.glob(os.path.join(q100_dir, "q100_*.bin")))
    if not q_files:
        sys.exit(f"No q100 chunks in {q100_dir} (run 040/043 first)")
    print(f"Chunks: {len(q_files)}")

    all_recs = []
    n_mc = None
    bad_chunks = []   # chunks that errored on read (corrupt / unreadable .bin)
    for qf in q_files:
        suffix = os.path.basename(qf)[len("q100_"):-len(".bin")]
        tf = os.path.join(truth_dir, f"truth_{suffix}.csv")
        if not os.path.exists(tf):
            print(f"WARNING: no truth csv for {suffix}, skipped")
            continue
        # Lustre/NFS can throw a transient Errno-5 on an otherwise intact file
        # (confirmed: the file reads fine on retry). Retry a few times before
        # giving up so a glitch does not silently drop 100 good cells.
        recs = nm = None
        for attempt in range(1, 4):
            try:
                recs, nm = read_chunk(qf, tf)
                break
            except OSError as e:
                print(f"WARNING: I/O error on {os.path.basename(qf)} "
                      f"(attempt {attempt}/3: {e})")
                time.sleep(2 * attempt)
        if recs is None:
            print(f"WARNING: giving up on {os.path.basename(qf)} after 3 tries; skipped")
            bad_chunks.append(suffix)
            continue
        all_recs.extend(recs)
        if nm and n_mc is None:
            n_mc = nm
    print(f"Cells with bias computed: {len(all_recs)} (n_mc = {n_mc})")
    if bad_chunks:
        bad_path = os.path.join(out_dir, "bad_chunks.txt")
        with open(bad_path, "w") as f:
            f.write("\n".join(bad_chunks) + "\n")
        print(f"WARNING: {len(bad_chunks)} unreadable chunk(s); list -> {bad_path}\n"
              f"         regenerate them with 040 (delete the .bin and re-run that "
              f"start_id range), then re-run 054.")
    if not all_recs:
        sys.exit("Nothing to do.")

    bad = excess_lib.load_bad_cells(dat_dir)
    if bad:
        before = len(all_recs)
        all_recs = [r for r in all_recs if r["cell_id"] not in bad]
        print(f"Excluding {before - len(all_recs)} negative-AMAX cells (031).")

    # CSV: cell_id, iy, ix, truth_q100, then {model}_{metric} for all 5 metrics
    cols = (["cell_id", "iy", "ix", "truth_q100"]
            + [f"{m}_{stat}" for m in MODELS for stat in METRICS])
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in all_recs:
            w.writerow([r["cell_id"], r["iy"], r["ix"], f"{r['truth_q100']:.6g}"]
                       + [f"{r[f'{m}_{stat}']:.6g}" for m in MODELS for stat in METRICS])
    print(f"Saved: {out_csv}")

    # console digest: median of each metric per model (in %), plus |bias|/sd ratio
    hdr = f"\n{'model':11s}" + "".join(f"{'med ' + s + '%':>12s}" for s in METRICS)
    hdr += f"{'med |bias|/sd':>15s}"
    print(hdr)
    for m in MODELS:
        vals = {s: np.array([r[f"{m}_{s}"] for r in all_recs]) * 100.0 for s in METRICS}
        ratio = np.abs(vals["bias"]) / np.where(vals["sd"] > 0, vals["sd"], np.nan)
        line = f"{MODEL_NAME[m]:11s}" + "".join(f"{np.median(vals[s]):12.2f}" for s in METRICS)
        line += f"{np.nanmedian(ratio):15.2f}"
        print(line)

    # cross-check: 054's sd/iqr/tail must reproduce 040's mc_summary exactly
    cross_check_vs_mc_summary(all_recs, os.path.join(dat_dir, "040", "mc_summary"))
    print("\nTable done. For the bias/RMSE maps run: "
          "python3 054b_plot_bias_maps.py")


if __name__ == "__main__":
    main()
