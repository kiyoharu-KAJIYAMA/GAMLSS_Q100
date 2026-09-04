"""
step13_timeseries_diagnostics.py
Compute statistical tests and diagnostics on 120-year AMAX time series
for all land grid cells. All tests implemented with numpy only (no extra packages).

Tests (with p-values):
  1. Mann-Kendall     : monotonic trend
  2. Pettitt          : change point detection
  3. SNHT             : homogeneity of mean
  4. Levene (median)  : equality of variance (first 60yr vs last 60yr)
  5. Runs test        : randomness around median

Metrics (no p-value):
  6.  Sen's slope          : robust trend slope [unit/year]
  7.  CUSUM max deviation  : magnitude of change point
  8.  Variance ratio       : var(first 60yr) / var(last 60yr)
  9.  ACF lag-1            : autocorrelation at lag 1
  10. Hurst exponent       : long-range dependence (R/S method)
  11. CV                   : coefficient of variation
  12. Skewness             : distribution asymmetry
  13. Kurtosis             : tail heaviness (excess, normal=0)
  14. Gini coefficient     : flow concentration
  15. Zero flow rate       : fraction of years with flow < 1 m3/s

Input:
  ../data/030/amax_all.bin   (float32, n_cells x 120)
  ../data/010/land_cells.csv
  uparea.bin                         (float32, 1800 x 3600, upstream area in km2)

Output:
  ../data/070/timeseries_tests.csv

Usage:
  python3 step13_timeseries_diagnostics.py [dat_dir] [uparea_path]
"""
import os
import sys
import csv
import numpy as np
from math import sqrt, erfc
import importlib
excess_lib = importlib.import_module("common_target_cells")

# ====================================================================
# Parameters
# ====================================================================
N_YEARS = 120
MIN_FLOW = 1.0   # m3/s threshold for valid cells
ZERO_THRESHOLD = 1.0  # m3/s for zero flow rate
# flood-relevance MAGNITUDE thresholds: single source of truth in common_target_cells.
# Used only with --flood-only to skip the heavy tests on non-flood cells (computed
# from the AMAX itself, reproducing the 091 flood-relevant set).
FLOOD_Q90 = excess_lib.FLOOD_Q90
FLOOD_MAX = excess_lib.FLOOD_MAX
FLOOD_UPAREA = excess_lib.FLOOD_UPAREA

# ====================================================================
# Statistical test implementations (numpy only)
# ====================================================================

def mann_kendall(x):
    """Mann-Kendall trend test. Returns (S, z, p_value)."""
    n = len(x)
    s = 0
    for i in range(n - 1):
        diff = x[i + 1:] - x[i]
        s += np.sum(np.sign(diff))
    # Variance of S (no ties correction for simplicity)
    var_s = n * (n - 1) * (2 * n + 5) / 18.0
    if s > 0:
        z = (s - 1) / sqrt(var_s)
    elif s < 0:
        z = (s + 1) / sqrt(var_s)
    else:
        z = 0.0
    p = erfc(abs(z) / sqrt(2))  # two-sided
    return float(s), float(z), float(p)


def sens_slope(x):
    """Sen's slope estimator (median of pairwise slopes)."""
    n = len(x)
    slopes = []
    for i in range(n):
        for j in range(i + 1, n):
            slopes.append((x[j] - x[i]) / (j - i))
    return float(np.median(slopes))


def pettitt_test(x):
    """Pettitt change point test. Returns (K, change_point_index, p_value)."""
    n = len(x)
    # Rank-based U statistic
    u = np.zeros(n, dtype=np.float64)
    for t in range(n):
        for j in range(n):
            u[t] += np.sign(x[t] - x[j])
    cum_u = np.cumsum(u)
    k_idx = np.argmax(np.abs(cum_u))
    k_val = abs(cum_u[k_idx])
    # Approximate p-value
    p = 2.0 * np.exp(-6.0 * k_val ** 2 / (n ** 3 + n ** 2))
    p = min(p, 1.0)
    return float(k_val), int(k_idx), float(p)


def snht_test(x):
    """Standard Normal Homogeneity Test. Returns (T_max, change_point_index, p_value)."""
    n = len(x)
    z = (x - np.mean(x)) / np.std(x, ddof=1) if np.std(x, ddof=1) > 0 else np.zeros(n)
    t_vals = np.zeros(n - 1)
    cum_z = np.cumsum(z)
    for k in range(1, n):
        z1_mean = cum_z[k - 1] / k
        z2_mean = (cum_z[-1] - cum_z[k - 1]) / (n - k)
        t_vals[k - 1] = k * z1_mean ** 2 + (n - k) * z2_mean ** 2
    t_max_idx = np.argmax(t_vals)
    t_max = t_vals[t_max_idx]
    # Approximate critical values (Khaliq & Ouarda 2007 approximation)
    # For n=120, critical value at 5% is ~9.0
    # Simple p-value approximation using exponential fit
    p = np.exp(-0.5 * t_max + 1.0) if t_max > 2 else 1.0
    p = min(max(p, 0.0), 1.0)
    return float(t_max), int(t_max_idx + 1), float(p)


def levene_test_median(x, split=60):
    """Levene test (median-based) for equal variance. Returns (W, p_value)."""
    g1 = x[:split]
    g2 = x[split:]
    n1, n2 = len(g1), len(g2)
    n = n1 + n2
    k = 2  # number of groups

    # Deviations from group medians
    d1 = np.abs(g1 - np.median(g1))
    d2 = np.abs(g2 - np.median(g2))

    d1_mean = np.mean(d1)
    d2_mean = np.mean(d2)
    d_mean = (np.sum(d1) + np.sum(d2)) / n

    # Between-group and within-group sum of squares
    ss_between = n1 * (d1_mean - d_mean) ** 2 + n2 * (d2_mean - d_mean) ** 2
    ss_within = np.sum((d1 - d1_mean) ** 2) + np.sum((d2 - d2_mean) ** 2)

    if ss_within == 0:
        return 0.0, 1.0

    w = ((n - k) / (k - 1)) * ss_between / ss_within

    # Approximate p-value using F-distribution (df1=1, df2=n-2)
    # Use simple approximation: for df1=1, F -> chi2(1) for large df2
    df1, df2 = k - 1, n - k
    # Beta regularized incomplete function approximation
    p = f_distribution_p(w, df1, df2)
    return float(w), float(p)


def f_distribution_p(f_val, df1, df2):
    """Approximate p-value for F distribution using normal approximation."""
    if f_val <= 0:
        return 1.0
    # Satterthwaite approximation
    z = (f_val ** (1.0 / 3) * (1 - 2.0 / (9 * df2)) -
         (1 - 2.0 / (9 * df1))) / sqrt(2.0 / (9 * df1) + f_val ** (2.0 / 3) * 2.0 / (9 * df2))
    p = erfc(abs(z) / sqrt(2))
    return min(max(p, 0.0), 1.0)


def runs_test(x):
    """Runs test for randomness around median. Returns (n_runs, z, p_value)."""
    med = np.median(x)
    binary = (x > med).astype(int)  # 1 if above median, 0 if below
    # Remove exact medians for cleaner test
    mask = x != med
    binary = binary[mask]
    n = len(binary)
    if n < 10:
        return 0, 0.0, 1.0

    n1 = np.sum(binary)
    n0 = n - n1
    if n1 == 0 or n0 == 0:
        return 0, 0.0, 1.0

    # Count runs
    runs = 1 + np.sum(binary[1:] != binary[:-1])

    # Expected runs and variance
    e_runs = 1 + 2 * n0 * n1 / n
    var_runs = 2 * n0 * n1 * (2 * n0 * n1 - n) / (n ** 2 * (n - 1))

    if var_runs <= 0:
        return int(runs), 0.0, 1.0

    z = (runs - e_runs) / sqrt(var_runs)
    p = erfc(abs(z) / sqrt(2))
    return int(runs), float(z), float(p)


def cusum_max(x):
    """CUSUM maximum deviation from mean."""
    cum = np.cumsum(x - np.mean(x))
    return float(np.max(np.abs(cum)))


def variance_ratio(x, split=60):
    """Variance ratio: var(first half) / var(second half)."""
    v1 = np.var(x[:split], ddof=1)
    v2 = np.var(x[split:], ddof=1)
    if v2 == 0:
        return float("nan")
    return float(v1 / v2)


def acf_lag1(x):
    """Autocorrelation at lag 1."""
    n = len(x)
    xm = x - np.mean(x)
    c0 = np.sum(xm ** 2) / n
    if c0 == 0:
        return 0.0
    c1 = np.sum(xm[:-1] * xm[1:]) / n
    return float(c1 / c0)


def hurst_exponent(x):
    """Hurst exponent via R/S analysis."""
    n = len(x)
    if n < 20:
        return float("nan")

    max_k = int(np.log2(n))
    rs_list = []
    ns_list = []

    for exp in range(2, max_k + 1):
        chunk_size = 2 ** exp
        n_chunks = n // chunk_size
        if n_chunks == 0:
            continue
        rs_vals = []
        for i in range(n_chunks):
            chunk = x[i * chunk_size:(i + 1) * chunk_size]
            mean_c = np.mean(chunk)
            dev = chunk - mean_c
            cum_dev = np.cumsum(dev)
            r = np.max(cum_dev) - np.min(cum_dev)
            s = np.std(chunk, ddof=1)
            if s > 0:
                rs_vals.append(r / s)
        if rs_vals:
            rs_list.append(np.mean(rs_vals))
            ns_list.append(chunk_size)

    if len(rs_list) < 2:
        return float("nan")

    log_n = np.log(ns_list)
    log_rs = np.log(rs_list)
    # Linear regression: log(R/S) = H * log(n) + c
    h = np.polyfit(log_n, log_rs, 1)[0]
    return float(h)


def gini_coefficient(x):
    """Gini coefficient of flow values."""
    x_sorted = np.sort(x)
    n = len(x)
    mean_x = np.mean(x)
    if mean_x == 0:
        return float("nan")
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * x_sorted) / (n * np.sum(x_sorted))) - (n + 1) / n)


def skewness(x):
    """Sample skewness."""
    n = len(x)
    m = np.mean(x)
    s = np.std(x, ddof=1)
    if s == 0:
        return 0.0
    return float(np.mean(((x - m) / s) ** 3) * n * n / ((n - 1) * (n - 2))) if n > 2 else 0.0


def kurtosis_excess(x):
    """Excess kurtosis (normal = 0)."""
    n = len(x)
    m = np.mean(x)
    s = np.std(x, ddof=1)
    if s == 0 or n < 4:
        return 0.0
    m4 = np.mean((x - m) ** 4)
    return float(m4 / (s ** 4) - 3.0)


# ====================================================================
# Per-cell worker (parallel). The big arrays are module globals so that, on Linux
# fork, worker processes share them copy-on-write (no per-worker 720 MB copy).
# ====================================================================
_AMAX = _CELLS = _UPAREA = None
_FLOOD_ONLY = False


def compute_cell(cid):
    """Return the output row for one cell, or None if the cell is skipped."""
    x = _AMAX[cid].astype(np.float64)
    if np.max(x) < MIN_FLOW or np.min(x) < 0:      # invalid / fill / reverse-flow
        return None
    iy, ix = _CELLS[cid]
    mean_outflow = float(np.mean(x))
    uparea = float(_UPAREA[iy, ix]) / 1.0e6        # m^2 -> km^2
    if _FLOOD_ONLY:
        if not ((float(np.percentile(x, 90)) >= FLOOD_Q90
                 or float(np.max(x)) >= FLOOD_MAX) and uparea >= FLOOD_UPAREA):
            return None
    mk_s, mk_z, mk_p = mann_kendall(x)
    ss = sens_slope(x)
    pt_k, pt_cp, pt_p = pettitt_test(x)
    sn_t, sn_cp, sn_p = snht_test(x)
    lv_w, lv_p = levene_test_median(x)
    rn_n, rn_z, rn_p = runs_test(x)
    cs = cusum_max(x)
    vr = variance_ratio(x)
    a1 = acf_lag1(x)
    hu = hurst_exponent(x)
    mean_x = np.mean(x)
    cv = float(np.std(x, ddof=1) / mean_x) if mean_x > 0 else float("nan")
    sk = skewness(x)
    ku = kurtosis_excess(x)
    gi = gini_coefficient(x)
    zf = float(np.mean(x < ZERO_THRESHOLD))
    return [
        cid, iy, ix,
        f"{mean_outflow:.4f}", f"{uparea:.0f}",
        f"{mk_s:.0f}", f"{mk_z:.4f}", f"{mk_p:.6f}",
        f"{ss:.6f}",
        f"{pt_k:.0f}", pt_cp, f"{pt_p:.6f}",
        f"{sn_t:.4f}", sn_cp, f"{sn_p:.6f}",
        f"{lv_w:.4f}", f"{lv_p:.6f}",
        rn_n, f"{rn_z:.4f}", f"{rn_p:.6f}",
        f"{cs:.4f}",
        f"{vr:.6f}",
        f"{a1:.6f}",
        f"{hu:.6f}",
        f"{cv:.6f}",
        f"{sk:.6f}",
        f"{ku:.6f}",
        f"{gi:.6f}",
        f"{zf:.6f}",
    ]


# ====================================================================
# Main
# ====================================================================
def main():
    # --flood-only: compute the tests only for flood-relevant cells (the ones used
    # downstream), skipping the heavy O(n^2) change-point tests on the rest (~2.4x
    # fewer cells). The flood filter is applied per cell from the AMAX itself, so no
    # 091 file is needed and the kept set matches 098/103/107.
    flood_only = "--flood-only" in sys.argv
    pos = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(pos) > 0:
        dat_dir = pos[0]
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        dat_dir = os.path.join(script_dir, "..", "data")
    uparea_path = pos[1] if len(pos) > 1 else \
        "/home/kk/jp_claude/gamlss/data/uparea.bin"
    if flood_only:
        print("FLOOD-ONLY mode: tests computed for flood-relevant cells only.")

    amax_bin = os.path.join(dat_dir, "030", "amax_all.bin")
    cell_csv = os.path.join(dat_dir, "010", "land_cells.csv")
    out_dir  = os.path.join(dat_dir, "070")
    out_csv  = os.path.join(out_dir, "timeseries_tests.csv")

    os.makedirs(out_dir, exist_ok=True)

    # Load cell info
    cells = []
    with open(cell_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cells.append((int(row["iy"]), int(row["ix"])))

    n_cells = len(cells)
    print(f"Total cells: {n_cells}")

    # Load all AMAX data
    amax_all = np.fromfile(amax_bin, dtype="float32").reshape(n_cells, N_YEARS)
    print(f"AMAX shape: {amax_all.shape}")

    # Load upstream area (1800 x 3600 global grid)
    NY, NX = 1800, 3600
    uparea_grid = np.fromfile(uparea_path, dtype="float32").reshape(NY, NX)
    print(f"Upstream area loaded: {uparea_path}")

    # Output header
    header = [
        "cell_id", "iy", "ix",
        # Basic properties
        "mean_outflow", "uparea_km2",
        # Mann-Kendall
        "mk_S", "mk_z", "mk_p",
        # Sen's slope
        "sens_slope",
        # Pettitt
        "pettitt_K", "pettitt_cp", "pettitt_p",
        # SNHT
        "snht_T", "snht_cp", "snht_p",
        # Levene
        "levene_W", "levene_p",
        # Runs
        "runs_n", "runs_z", "runs_p",
        # CUSUM
        "cusum_max",
        # Variance ratio
        "var_ratio",
        # ACF lag-1
        "acf1",
        # Hurst
        "hurst",
        # CV
        "cv",
        # Skewness
        "skewness",
        # Kurtosis (excess)
        "kurtosis",
        # Gini
        "gini",
        # Zero flow rate
        "zero_flow_rate",
    ]

    # Share the big arrays with the worker processes via module globals (fork =>
    # copy-on-write, so no 720 MB is duplicated per worker).
    global _AMAX, _CELLS, _UPAREA, _FLOOD_ONLY
    _AMAX, _CELLS, _UPAREA, _FLOOD_ONLY = amax_all, cells, uparea_grid, flood_only

    import multiprocessing as mp
    try:
        n_workers = len(os.sched_getaffinity(0))     # respects the PBS cpuset (e.g. 40)
    except AttributeError:
        n_workers = mp.cpu_count()
    n_workers = max(1, int(os.environ.get("N_WORKERS", n_workers)))
    print(f"Parallel: {n_workers} workers over {n_cells} cells")

    n_valid = n_skip = 0
    with open(out_csv, "w", newline="") as fout:
        writer = csv.writer(fout)
        writer.writerow(header)
        with mp.Pool(n_workers) as pool:
            for row in pool.imap_unordered(compute_cell, range(n_cells), chunksize=500):
                if row is None:
                    n_skip += 1
                    continue
                writer.writerow(row)
                n_valid += 1
                if n_valid % 50000 == 0:
                    print(f"  Processed {n_valid} valid cells ({n_skip} skipped)")

    print(f"Done: {n_valid} valid, {n_skip} skipped")
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()
