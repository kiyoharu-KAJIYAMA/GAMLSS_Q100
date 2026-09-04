#!/usr/bin/env Rscript
#
# step12_fitted_parent_parameters.R
# Lightweight: fit truth GAMLSS model only (no MC loop) for each grid cell.
# Generates mc_truth CSV files compatible with 040/mc_truth output format.
#
# Used by 060 to access truth coefficients without waiting for full 040 run.
#
# The parent (data-generating truth) is selectable (matches 108's parents):
#   st : stationary  mu/sigma ~ 1
#   ln : linear      mu/sigma ~ year            (default; original behaviour)
#   qd : quadratic   mu/sigma ~ poly(year, 2)
# Coefficients are stored up to quadratic (mu/sigma _0,_1,_2), NA-padded for the
# lower-order parents, so 060 reconstructs mu(t)/sigma(t) uniformly.
#
# Usage: Rscript step12_fitted_parent_parameters.R <start_id> <end_id> <dat_dir> [parent: st|ln|qd]
#
# Output: <dat_dir>/064/<parent>/
#   mc_truth/truth_SSSSSS_EEEEEE.csv : truth coefficients (cell_id, iy, ix, parent,
#                                       truth_q100, truth_mu_0..2, truth_sigma_0..2,
#                                       delta)
#   fail_log/failed_SSSSSS_EEEEEE.txt : error log

# --- Parse arguments ---
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) stop("Usage: Rscript step12_fitted_parent_parameters.R <start_id> <end_id> <dat_dir> [parent: st|ln|qd]")

start_id <- as.integer(args[1])
end_id   <- as.integer(args[2])
dat_dir  <- args[3]
parent   <- if (length(args) >= 4) args[4] else "ln"
if (!parent %in% c("st", "ln", "qd")) stop("parent must be one of st, ln, qd")

# Store coefficients with full precision (quadratic terms multiply year^2 ~ 4e6,
# so 7-sig-fig default would lose accuracy on reconstruction).
options(digits = 15)

# --- Load libraries ---
suppressMessages(suppressWarnings({
  library(gamlss)
  library(gamlss.dist)
}))

# --- Parameters ---
# Independent RNG stream per parent (delta draws), mirroring 108's offsets.
truth_offset <- c(st = 0L, ln = 1000000L, qd = 2000000L)[[parent]]
set.seed(123 + start_id + truth_offset)
n_years    <- 120
year_start <- 1981
min_flow_threshold <- 1.0   # m3/s

# --- Paths (per parent) ---
amax_bin    <- file.path(dat_dir, "030", "amax_all.bin")
cell_csv    <- file.path(dat_dir, "010", "land_cells.csv")
truth_dir   <- file.path(dat_dir, "064", parent, "mc_truth")
fail_dir    <- file.path(dat_dir, "064", parent, "fail_log")

dir.create(truth_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(fail_dir, showWarnings = FALSE, recursive = TRUE)

# --- Load cell information (typed read: faster on the ~1.7M-row csv) ---
cells <- read.csv(cell_csv, colClasses = c(iy = "integer", ix = "integer"))
batch_ids <- start_id:end_id
batch_ids <- batch_ids[batch_ids < nrow(cells)]
n_batch <- length(batch_ids)

if (n_batch == 0) {
  cat("No cells to process.\n")
  quit(save = "no")
}

cat(sprintf("Processing cells %d to %d (%d cells) | parent=%s\n", start_id, end_id, n_batch, parent))

years <- year_start:(year_start + n_years - 1)
fail_file <- file.path(fail_dir, sprintf("failed_%06d_%06d.txt", start_id, end_id))

# ====================================================================
# Pre-allocate
# ====================================================================
res_cell_id   <- integer(n_batch)
res_iy        <- integer(n_batch)
res_ix        <- integer(n_batch)
truth_q100    <- rep(NA_real_, n_batch)
truth_mu0     <- rep(NA_real_, n_batch)
truth_mu1     <- rep(NA_real_, n_batch)
truth_mu2     <- rep(NA_real_, n_batch)
truth_sig0    <- rep(NA_real_, n_batch)
truth_sig1    <- rep(NA_real_, n_batch)
truth_sig2    <- rep(NA_real_, n_batch)
truth_delta   <- rep(NA_real_, n_batch)

# ====================================================================
# Main loop: read AMAX, fit truth model, save coefficients
# ====================================================================
con_amax <- file(amax_bin, "rb")
n_success <- 0
n_fail    <- 0

for (i in seq_along(batch_ids)) {
  cid <- batch_ids[i]
  res_cell_id[i] <- cid
  res_iy[i] <- cells$iy[cid + 1]
  res_ix[i] <- cells$ix[cid + 1]

  # Read 120 AMAX values
  offset <- as.numeric(cid) * n_years * 4
  seek(con_amax, where = offset, origin = "start")
  amax_values <- readBin(con_amax, what = "numeric", size = 4, n = n_years, endian = "little")

  # Skip cells with negligible flow
  if (length(amax_values) != n_years || max(amax_values) < min_flow_threshold) {
    cat(sprintf("DATA_SKIP cell_id=%d max=%.4g\n", cid, max(amax_values)),
        file = fail_file, append = TRUE)
    n_fail <- n_fail + 1
    next
  }

  # Fit truth model (GU with negated flow, same family as 040); formula by parent.
  amax_df <- data.frame(year = years, outflow_neg = -amax_values)

  truth_result <- tryCatch({
    fo_mu  <- switch(parent, st = outflow_neg ~ 1, ln = outflow_neg ~ year,
                     qd = outflow_neg ~ poly(year, 2, raw = TRUE))
    fo_sig <- switch(parent, st = ~ 1, ln = ~ year, qd = ~ poly(year, 2, raw = TRUE))
    m <- gamlss(fo_mu, sigma.fo = fo_sig, family = GU, data = amax_df, trace = FALSE)
    mu_neg <- fitted(m, "mu")
    sig    <- fitted(m, "sigma")
    q100   <- -qGU(1 - 0.99, mu = mu_neg, sigma = sig)
    late   <- years >= 2071 & years <= 2100        # eval window (matches 040/060)
    # Coefficients up to quadratic; NA-pad the lower-order parents to 3 terms each.
    padc <- function(cc) c(as.numeric(cc), rep(NA_real_, 3 - length(cc)))[1:3]
    cm <- padc(coef(m, what = "mu"))
    cs <- padc(coef(m, what = "sigma"))
    list(mu_neg = mu_neg, sigma = sig,
         mu0 = cm[1], mu1 = cm[2], mu2 = cm[3],
         sig0 = cs[1], sig1 = cs[2], sig2 = cs[3],
         q100_uncorrected = mean(q100[late]))   # truth Q100 over the late window
  }, error = function(e) conditionMessage(e))

  if (is.character(truth_result)) {
    cat(sprintf("TRUTH_FAIL cell_id=%d err=%s\n", cid, truth_result),
        file = fail_file, append = TRUE)
    n_fail <- n_fail + 1
    next
  }

  truth_mu0[i]  <- truth_result$mu0
  truth_mu1[i]  <- truth_result$mu1
  truth_mu2[i]  <- truth_result$mu2
  truth_sig0[i] <- truth_result$sig0
  truth_sig1[i] <- truth_result$sig1
  truth_sig2[i] <- truth_result$sig2

  # Compute delta correction (100 MC samples for stability).
  # Vectorized: one rGU call for all 120 x 100 draws instead of 12,000
  # scalar calls via replicate/mapply (statistically identical, ~100x faster).
  mu_true_neg <- truth_result$mu_neg
  sigma_true  <- truth_result$sigma
  mc_for_delta <- -rGU(n_years * 100,
                       mu = rep(mu_true_neg, 100), sigma = rep(sigma_true, 100))
  min_val <- min(mc_for_delta)
  q1_val  <- quantile(mc_for_delta, 0.25)
  delta   <- abs(min_val) + q1_val

  truth_delta[i] <- delta
  truth_q100[i]  <- truth_result$q100_uncorrected + delta

  n_success <- n_success + 1

  if (i %% 20 == 0) {
    cat(sprintf("  [%d/%d] success=%d, fail=%d\n", i, n_batch, n_success, n_fail))
  }
}

close(con_amax)

# ====================================================================
# Write CSV (same format as 040/mc_truth)
# ====================================================================
out_df <- data.frame(
  cell_id       = res_cell_id,
  iy            = res_iy,
  ix            = res_ix,
  parent        = parent,
  truth_q100    = truth_q100,
  truth_mu_0    = truth_mu0,
  truth_mu_1    = truth_mu1,
  truth_mu_2    = truth_mu2,
  truth_sigma_0 = truth_sig0,
  truth_sigma_1 = truth_sig1,
  truth_sigma_2 = truth_sig2,
  delta         = truth_delta
)

out_path <- file.path(truth_dir, sprintf("truth_%06d_%06d.csv", start_id, end_id))
write.csv(out_df, out_path, row.names = FALSE)
cat(sprintf("Saved: %s\n", out_path))
cat(sprintf("All done: %d success, %d fail out of %d cells\n", n_success, n_fail, n_batch))
