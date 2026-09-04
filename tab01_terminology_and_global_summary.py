"""
tab01_terminology_and_global_summary.py
Global (flood-relevant) version of manuscript Table 1, for the truth=linear row.

Aggregates the per-cell error statistics of 040/043 (via 050) over all
FLOOD-RELEVANT cells, keeping exactly the Table-1 structure:
  3 models (stationary / linear / quadratic) x 3 statistics (SD / IQR / tail),
  each entry = median [q25, q75] across cells, in percent (errors are relative,
  hence dimensionless and directly comparable across cells).

In addition, PAIRED win rates are computed per statistic (the medians compare
marginal distributions; the win rate shows the within-cell ranking):
  - linear best   : linear smallest of the three models
  - lin < stat    : linear beats stationary
  - lin < quad    : linear beats quadratic
and the SD win rate is stratified by change type x climate (from 095) to show
that the ranking is systematic in every stratum.

NOTE: 040 generates the truth from the LINEAR fit only, so this is the
truth=linear row of the 3x3 design. The truth=stationary / truth=quadratic
rows require a separate run (045) or are illustrated by the single-cell Table 1.

Inputs:
  <dat_dir>/050/summary_allgrid.csv      (stat/lin/quad _err_sd/_iqr/_tail)
  <dat_dir>/091/change_type_cells.csv    (q90, max_amax, uparea_km2, type, climate)

Output: <dat_dir>/055/
  global_table1.csv       median [q25,q75] per model x statistic + win rates
  winrate_stratified.csv  SD win rate per change type x climate
  console: the same tables

Usage:
  python3 tab01_terminology_and_global_summary.py [dat_dir] [summary_csv] [X1] [X2] [A]
    X1, X2, A : flood-relevance thresholds q90 / max / uparea (default 50/100/50)
"""
import os
import sys
import csv
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS = ["stat", "lin", "quad"]
STATS = ["sd", "iqr", "tail"]
TYPES = ["stationary", "trend", "step", "variance"]
CLIMATES = ["low", "high"]


def fnum(s):
    try:
        v = float(s)
        return v if np.isfinite(v) else np.nan
    except (ValueError, TypeError):
        return np.nan


def main():
    dat_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(SCRIPT_DIR, "..", "data")
    summ_csv = sys.argv[2] if len(sys.argv) > 2 else os.path.join(dat_dir, "050", "summary_allgrid.csv")
    x1 = float(sys.argv[3]) if len(sys.argv) > 3 else 50.0
    x2 = float(sys.argv[4]) if len(sys.argv) > 4 else 100.0
    a_up = float(sys.argv[5]) if len(sys.argv) > 5 else 50.0

    cells_csv = os.path.join(dat_dir, "091", "change_type_cells.csv")
    out_dir = os.path.join(dat_dir, "055")
    os.makedirs(out_dir, exist_ok=True)
    for p in (summ_csv, cells_csv):
        if not os.path.exists(p):
            sys.exit(f"Not found: {p}")

    # flood-relevance + stratification from 095
    meta = {}
    with open(cells_csv, "r", newline="") as f:
        for row in csv.DictReader(f):
            cid = int(row["cell_id"])
            q90 = fnum(row.get("q90", "")); mx = fnum(row.get("max_amax", ""))
            up = fnum(row.get("uparea_km2", ""))
            keep = ((q90 >= x1) or (mx >= x2)) and (up >= a_up)
            meta[cid] = (keep, row.get("type", "?"), row.get("climate", "?"))

    vals = {(m, s): [] for m in MODELS for s in STATS}
    typ, clim = [], []
    n_total = n_flood = 0
    with open(summ_csv, "r", newline="") as f:
        for row in csv.DictReader(f):
            cid = int(row["cell_id"])
            if cid not in meta:
                continue
            n_total += 1
            keep, t, c = meta[cid]
            if not keep:
                continue
            v = {(m, s): fnum(row.get(f"{m}_err_{s}", "")) for m in MODELS for s in STATS}
            if not all(np.isfinite(x) for x in v.values()):
                continue
            n_flood += 1
            for k, x in v.items():
                vals[k].append(x)
            typ.append(t); clim.append(c)
    if n_flood == 0:
        sys.exit("No flood-relevant cells with complete statistics.")
    arr = {k: np.array(v) * 100.0 for k, v in vals.items()}   # -> percent
    typ = np.array(typ); clim = np.array(clim)
    print(f"Cells: {n_total} classified, {n_flood} flood-relevant with all 9 stats "
          f"({100.0 * n_flood / max(n_total, 1):.1f}%)")

    # --- Table-1-style medians + paired win rates ---
    tab_csv = os.path.join(out_dir, "global_table1.csv")
    with open(tab_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([f"# truth=linear; flood-relevant: (q90>={x1:g} OR max>={x2:g}) "
                    f"AND uparea>={a_up:g}; n={n_flood}; values in % of true Q100"])
        w.writerow(["statistic", "model", "median", "q25", "q75"])
        print(f"\nGlobal Table 1 (truth=linear, n={n_flood}) -- median [q25, q75] %:")
        for s in STATS:
            line = f"  {s.upper():5s}"
            for m in MODELS:
                d = arr[(m, s)]
                md, q25, q75 = (np.percentile(d, p) for p in (50, 25, 75))
                w.writerow([s, m, f"{md:.3f}", f"{q25:.3f}", f"{q75:.3f}"])
                line += f"  {m}: {md:6.2f} [{q25:6.2f},{q75:6.2f}]"
            print(line)

        w.writerow([])
        w.writerow(["statistic", "winrate_linear_best", "winrate_lin_lt_stat",
                    "winrate_lin_lt_quad"])
        print("\nPaired win rates (fraction of cells):")
        for s in STATS:
            st_, ln_, qd_ = arr[("stat", s)], arr[("lin", s)], arr[("quad", s)]
            best = np.mean((ln_ < st_) & (ln_ < qd_))
            w_st = np.mean(ln_ < st_)
            w_qd = np.mean(ln_ < qd_)
            w.writerow([s, f"{best:.4f}", f"{w_st:.4f}", f"{w_qd:.4f}"])
            print(f"  {s.upper():5s}  linear best: {100*best:5.1f}%   "
                  f"lin<stat: {100*w_st:5.1f}%   lin<quad: {100*w_qd:5.1f}%")
    print(f"Saved: {tab_csv}")

    # --- SD win rate stratified by change type x climate ---
    strat_csv = os.path.join(out_dir, "winrate_stratified.csv")
    st_, ln_, qd_ = arr[("stat", "sd")], arr[("lin", "sd")], arr[("quad", "sd")]
    best = (ln_ < st_) & (ln_ < qd_)
    with open(strat_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["type", "climate", "n", "winrate_linear_best_sd"])
        print("\nSD win rate (linear best) by change type x climate:")
        for t in TYPES:
            line = f"  {t:11s}"
            for c in CLIMATES:
                m = (typ == t) & (clim == c)
                n = int(np.sum(m))
                wr = float(np.mean(best[m])) if n else np.nan
                w.writerow([t, c, n, f"{wr:.4f}" if n else ""])
                line += f"  {c}: {100*wr:5.1f}% (n={n})" if n else f"  {c}:    -- (n=0)"
            print(line)
    print(f"Saved: {strat_csv}")


if __name__ == "__main__":
    main()
