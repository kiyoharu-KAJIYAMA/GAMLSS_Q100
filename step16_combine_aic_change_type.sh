#!/bin/bash
# step16_combine_aic_change_type.sh
# Combine the per-chunk 077b outputs into one whole-globe classification table
# for the Figure-2 map (115). 078b only PRINTS the awk command; this script runs
# it, plus a completeness check and a row count. Safe to re-run (idempotent).
#
# Usage:
#   bash step16_combine_aic_change_type.sh [dat_dir]
#   (default dat_dir: ../data, relative to this script)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DAT_DIR="${1:-$SCRIPT_DIR/../data}"
AIC_DIR="$DAT_DIR/077b/aic"
CELL_CSV="$DAT_DIR/010/land_cells.csv"
OUT="$DAT_DIR/077b/aic_allgrid.csv"
BATCH_SIZE=1000

if [ ! -d "$AIC_DIR" ]; then echo "ERROR: not found: $AIC_DIR" >&2; exit 1; fi

# --- completeness check: chunk count vs expected batches ---
N_CELLS=$(($(wc -l < "$CELL_CSV") - 1))
EXPECTED=$(( (N_CELLS + BATCH_SIZE - 1) / BATCH_SIZE ))
ACTUAL=$(ls "$AIC_DIR"/chunk_*.csv 2>/dev/null | wc -l)
echo "land cells: $N_CELLS ; expected chunks: $EXPECTED ; actual chunks: $ACTUAL"
if [ "$ACTUAL" -lt "$EXPECTED" ]; then
    echo "WARNING: $((EXPECTED - ACTUAL)) chunk(s) missing -> 078b not finished."
    echo "         Re-submit 'qsub 078b_pbs.sh' (checkpoint fills the gaps), then re-run this."
    echo "         Combining what exists anyway..."
fi

# --- combine (header once) ---
awk 'FNR==1 && NR!=1{next}{print}' "$AIC_DIR"/chunk_*.csv > "$OUT"
ROWS=$(($(wc -l < "$OUT") - 1))
echo "Saved: $OUT"
echo "Classified cells (rows): $ROWS"
