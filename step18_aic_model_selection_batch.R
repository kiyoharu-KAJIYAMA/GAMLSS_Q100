#!/usr/bin/env Rscript
#
# step18_aic_model_selection_batch.R
# Per-grid AIC-based model selection among the three GAMLSS models used in the
# paper, fitted to the OBSERVED 120-yr AMAX series of each land cell:
#
#   st : stationary        outflow_neg ~ 1                  (mu,sigma const)   2 par
#   ln : linear nonstat.   outflow_neg ~ year               (mu,sigma linear)  4 par
#   qd : quadratic nonstat outflow_neg ~ poly(year,2)       (mu,sigma quad)    6 par
#
# All three are fitted to the SAME 120 data points so their AIC is directly
# comparable (a valid model-selection comparison; this is why the stationary
# model here uses all 120 yr, NOT the 2071-2100 window that 040/045b use for the
# Q100 estimate). AIC = G.deviance + 2*df penalises the extra parameters, so a
# model only "wins" where the data genuinely support the added flexibility --
# the parsimony argument behind recommending the linear model when the true
# parent distribution is unknown (quadratic has low bias but overfits).
#
# GU family on negated flow (outflow_neg = -AMAX), identical to 040/045b.
#
# Output (per chunk, atomic write):
#   <dat_dir>/100/summary/aic_SSSSSS_EEEEEE.csv
#       cell_id, iy, ix, aic_st, aic_ln, aic_qd   (NA where a fit failed)
#
# Checkpoint: skips a chunk whose summary CSV already exists.
#
# Usage:
#   Rscript step18_aic_model_selection_batch.R <start_id> <end_id> <dat_dir>

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3)
  stop("Usage: Rscript step18_aic_model_selection_batch.R <start_id> <end_id> <dat_dir>")
start_id <- as.integer(args[1])
end_id   <- as.integer(args[2])
dat_dir  <- args[3]

suppressMessages(suppressWarnings({
  library(gamlss)
  library(gamlss.dist)
}))

n_years    <- 120
year_start <- 1981
min_flow_threshold <- 1.0   # m3/s

amax_bin    <- file.path(dat_dir, "030", "amax_all.bin")
cell_csv    <- file.path(dat_dir, "010", "land_cells.csv")
out_root    <- file.path(dat_dir, "100")
summary_dir <- file.path(out_root, "summary")
fail_dir    <- file.path(out_root, "fail_log")
dir.create(summary_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(fail_dir,    showWarnings = FALSE, recursive = TRUE)

# Checkpoint: skip this chunk if its summary already exists.
out_summary <- file.path(summary_dir, sprintf("aic_%06d_%06d.csv", start_id, end_id))
if (file.exists(out_summary)) {
  cat(sprintf("Exists, skip: %s\n", out_summary))
  quit(save = "no")
}
fail_file <- file.path(fail_dir, sprintf("failed_%06d_%06d.txt", start_id, end_id))

cells <- read.csv(cell_csv)
batch_ids <- start_id:end_id
batch_ids <- batch_ids[batch_ids < nrow(cells)]
n_batch <- length(batch_ids)
if (n_batch == 0) { cat("No cells to process.\n"); quit(save = "no") }

cat(sprintf("AIC model selection | cells %d to %d (%d cells)\n",
            start_id, end_id, n_batch))

years <- year_start:(year_start + n_years - 1)

# --- AIC of each model fitted to the full 120-yr observed series ---
# Each returns the GAMLSS AIC, or NA if the fit fails / does not converge.
aic_of <- function(mu_fo, sigma_fo, df) {
  tryCatch({
    m <- gamlss(formula = mu_fo, sigma.fo = sigma_fo, family = "GU",
                data = df, trace = FALSE)
    val <- m$aic
    if (is.null(val) || !is.finite(val)) NA_real_ else as.numeric(val)
  }, error = function(e) NA_real_, warning = function(w) NA_real_)
}

aic_stationary <- function(df) aic_of(outflow_neg ~ 1, ~ 1, df)
aic_linear     <- function(df) aic_of(outflow_neg ~ year, ~ year, df)
aic_quadratic  <- function(df) aic_of(outflow_neg ~ poly(year, 2, raw = TRUE),
                                      ~ poly(year, 2, raw = TRUE), df)

# ====================================================================
# Main loop
# ====================================================================
con_amax <- file(amax_bin, "rb")
rows <- vector("list", n_batch)
n_ok <- 0; n_fail <- 0

for (i in seq_along(batch_ids)) {
  cid <- batch_ids[i]
  iy  <- cells$iy[cid + 1]
  ix  <- cells$ix[cid + 1]

  seek(con_amax, where = as.numeric(cid) * n_years * 4, origin = "start")
  amax_values <- readBin(con_amax, what = "numeric", size = 4, n = n_years, endian = "little")
  if (length(amax_values) != n_years || max(amax_values) < min_flow_threshold) {
    cat(sprintf("DATA_SKIP cell_id=%d\n", cid), file = fail_file, append = TRUE)
    n_fail <- n_fail + 1
    next
  }
  amax_df <- data.frame(year = years, outflow_neg = -amax_values)

  a_st <- aic_stationary(amax_df)
  a_ln <- aic_linear(amax_df)
  a_qd <- aic_quadratic(amax_df)

  if (!any(is.finite(c(a_st, a_ln, a_qd)))) {
    cat(sprintf("ALL_FAIL cell_id=%d\n", cid), file = fail_file, append = TRUE)
    n_fail <- n_fail + 1
    next
  }

  rows[[i]] <- data.frame(cell_id = cid, iy = iy, ix = ix,
                          aic_st = a_st, aic_ln = a_ln, aic_qd = a_qd)
  n_ok <- n_ok + 1
  if (i %% 25 == 0) cat(sprintf("  [%d/%d] ok=%d fail=%d\n", i, n_batch, n_ok, n_fail))
}
close(con_amax)

# Atomic write: .tmp then rename, so a killed job never leaves a half-written
# CSV that the checkpoint would treat as complete.
df_out <- do.call(rbind, rows[!vapply(rows, is.null, logical(1))])
if (is.null(df_out)) df_out <- data.frame()
tmp <- paste0(out_summary, ".tmp")
write.csv(df_out, tmp, row.names = FALSE); file.rename(tmp, out_summary)

cat(sprintf("Saved: %s (%d cells, %d fail)\n", out_summary, n_ok, n_fail))
