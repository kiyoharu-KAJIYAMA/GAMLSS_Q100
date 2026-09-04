"""
common_target_cells.py
Shared helper for the "excess SD improvement" analyses (073/074/075/076b).

Raw improvement is dominated by a near-constant BASELINE -- the sample-size effect
of fitting the linear model on 120 yr vs the stationary model on 30 yr, which
reduces the SD even when there is NO nonstationarity. That baseline (~15-18%) swamps
the nonstationarity-driven signal, so correlations with raw improvement come out
weak. To isolate the signal we subtract the baseline:

    baseline      = median improvement over Slater "stationary"-type cells (091)
    excess[cell]  = improvement[cell] - baseline          [percentage points]

excess > 0  => the linear model helps MORE than the pure sample-size effect, i.e.
there is a genuine nonstationarity benefit at that cell.
"""
import os
import csv
import numpy as np

CAP_PCT = 300.0          # |improvement| beyond this is arid blow-up (matches 050/052)
BASELINE_TYPE = "stationary"
DRY_MAX = 0.5            # flood-relevance: a cell must have a flood in >50% of years
                        # (dry_year_fraction <= DRY_MAX); above this Q100 is not
                        # estimable (no flood regime) and is dropped from aggregations.
# flood-relevance MAGNITUDE criteria (matches 098/103/107): keep a cell if
# (q90 >= FLOOD_Q90 OR max_amax >= FLOOD_MAX) AND uparea_km2 >= FLOOD_UPAREA.
# Single source of truth -- consumers should import these rather than re-hardcode.
FLOOD_Q90, FLOOD_MAX, FLOOD_UPAREA = 50.0, 100.0, 50.0


def load_slater_type(cells_csv):
    """cell_id -> Slater change type from 091/change_type_cells.csv ({} if absent)."""
    d = {}
    if not os.path.exists(cells_csv):
        return d
    with open(cells_csv, newline="") as f:
        for row in csv.DictReader(f):
            try:
                d[int(row["cell_id"])] = row.get("type", "")
            except (ValueError, KeyError):
                pass
    return d


def baseline_excess(cell_ids, improve, stype_map, baseline_type=BASELINE_TYPE):
    """Return (baseline, excess_array).

    baseline = median improvement over the Slater stationary-type cells present in
    `cell_ids` (the pure sample-size effect). excess = improve - baseline. Falls
    back to the global median improvement if the stationary class is missing/small.
    """
    cell_ids = np.asarray(cell_ids)
    improve = np.asarray(improve, dtype=float)
    finite = np.isfinite(improve)
    is_base = np.array([stype_map.get(int(c), "") == baseline_type for c in cell_ids])
    pool = improve[is_base & finite]
    if pool.size < 10:                       # no/too-few stationary cells -> global
        pool = improve[finite]
    baseline = float(np.median(pool)) if pool.size else 0.0
    return baseline, improve - baseline


def load_bad_cells(dat_dir):
    """Set of cell_ids flagged by step04_flag_reverse_flow_cells.py (negative-AMAX reverse/backwater
    cells that are unusable for flood frequency). Empty set if the file is absent."""
    path = os.path.join(dat_dir, "010", "bad_cells.csv")
    bad = set()
    if os.path.exists(path):
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                try:
                    bad.add(int(row["cell_id"]))
                except (ValueError, KeyError):
                    pass
    return bad


def load_bad_yx(dat_dir):
    """Set of (iy, ix) tuples for the 031 bad cells, for scripts keyed by iy/ix
    rather than cell_id (e.g. 066). Empty set if absent."""
    path = os.path.join(dat_dir, "010", "bad_cells.csv")
    bad = set()
    if os.path.exists(path):
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                try:
                    bad.add((int(row["iy"]), int(row["ix"])))
                except (ValueError, KeyError):
                    pass
    return bad


def load_degenerate_cells(dat_dir, dry_max=DRY_MAX):
    """Cell_ids with no flood regime: dry_year_fraction (070 zero_flow_rate) > dry_max,
    i.e. no flood in more than `dry_max` of the years, so Q100 is ill-conditioned
    (tiny denominator). This is the flood-relevance redefinition. Empty set if 070
    output is absent."""
    path = os.path.join(dat_dir, "070", "timeseries_tests.csv")
    deg = set()
    if os.path.exists(path):
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                try:
                    if float(row["zero_flow_rate"]) > dry_max:
                        deg.add(int(row["cell_id"]))
                except (ValueError, KeyError):
                    pass
    return deg


def load_excluded_cells(dat_dir, dry_max=DRY_MAX):
    """Union of cells to drop from flood-frequency aggregations / maps: reverse-flow
    (031 bad_cells, unconditionally invalid) PLUS no-flood-regime (dry_year > dry_max,
    the flood-relevance redefinition)."""
    return load_bad_cells(dat_dir) | load_degenerate_cells(dat_dir, dry_max)


def load_flood_relevant_ids(dat_dir, x1=FLOOD_Q90, x2=FLOOD_MAX, a_up=FLOOD_UPAREA):
    """cell_ids passing the flood MAGNITUDE criteria
    (q90 >= x1 OR max_amax >= x2) AND uparea_km2 >= a_up,
    read from 091/change_type_cells.csv. Empty set if the file is absent."""
    path = os.path.join(dat_dir, "091", "change_type_cells.csv")
    ids = set()
    if not os.path.exists(path):
        return ids
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            def num(k):
                v = row.get(k, "")
                try:
                    return float(v)
                except (ValueError, TypeError):
                    return float("nan")
            if ((num("q90") >= x1) or (num("max_amax") >= x2)) and (num("uparea_km2") >= a_up):
                try:
                    ids.add(int(row["cell_id"]))
                except (ValueError, KeyError):
                    pass
    return ids


def load_flood_ensured_ids(dat_dir, x1=FLOOD_Q90, x2=FLOOD_MAX, a_up=FLOOD_UPAREA,
                           dry_max=DRY_MAX):
    """The flood-ENSURED cell set = flood-relevant MAGNITUDE
    ((q90>=x1 OR max>=x2) AND uparea>=a_up) AND a flood regime (dry_year<=dry_max)
    AND not reverse-flow. This is the single canonical definition (A): one call gives
    exactly the cells used in the flood analyses/figures."""
    return load_flood_relevant_ids(dat_dir, x1, x2, a_up) - load_excluded_cells(dat_dir, dry_max)
