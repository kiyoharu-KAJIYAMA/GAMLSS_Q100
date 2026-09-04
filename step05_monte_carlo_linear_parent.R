#!/usr/bin/env Rscript
#
# step05_monte_carlo_linear_parent.R
# Batch processing (2000-grid chunks):
#   1) Fit GAMLSS(GU, linear nonstationary) to observed AMAX -> true mu/sigma
#   2) Generate 3,000 MC samples one by one (in-memory only, not saved to disk)
#   3) Fit 3 GAMLSS models (stationary/linear/quadratic) to each MC sample
#   4) Save fitted coefficients to mc_results/*.bin (flat binary, float32)
#   5) Save mean Q100 (1981-2100) to mc_q100/*.bin
#   6) Save ground truth to mc_truth/*.csv
#   7) Compute error = estimated_Q100 - truth_Q100 for each MC,
#      then save SD, IQR, tail spread (Q95-Q5) to mc_summary/*.csv
#
# Error definition:
#   error = estimated_Q100 - reference_Q100
#   Both include the same delta correction, so delta cancels out.
#   Perfect prediction -> error = 0.
#
# Usage: Rscript step05_monte_carlo_linear_parent.R <start_cell_id> <end_cell_id> <dat_dir>
#
# Output: <dat_dir>/040/
#   mc_results/chunk_SSSSSS_EEEEEE.bin  : 12 coef matrices [n_batch x n_mc] float32
#   mc_q100/q100_SSSSSS_EEEEEE.bin     : 3 Q100 matrices [n_batch x n_mc] float32
#   mc_truth/truth_SSSSSS_EEEEEE.csv   : ground truth per grid (CSV)
#   mc_summary/summary_SSSSSS_EEEEEE.csv : error statistics per grid (CSV)

# --- Parse arguments ---
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) stop("Usage: Rscript step05_monte_carlo_linear_parent.R <start_cell_id> <end_cell_id> <dat_dir>")

start_id <- as.integer(args[1])
end_id   <- as.integer(args[2])
dat_dir  <- args[3]

# --- Load libraries ---
suppressMessages(suppressWarnings({
  library(gamlss)
  library(gamlss.dist)
}))

# --- Parameters ---
set.seed(123 + start_id)
n_years    <- 120
n_mc       <- 3000
year_start <- 1981

# --- Paths ---
amax_bin    <- file.path(dat_dir, "030", "amax_all.bin")
cell_csv    <- file.path(dat_dir, "010", "land_cells.csv")
result_dir  <- file.path(dat_dir, "040", "mc_results")
q100_dir    <- file.path(dat_dir, "040", "mc_q100")
truth_dir   <- file.path(dat_dir, "040", "mc_truth")
summary_dir <- file.path(dat_dir, "040", "mc_summary")
fail_dir    <- file.path(dat_dir, "040", "fail_log")

dir.create(result_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(q100_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(truth_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(summary_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(fail_dir, showWarnings = FALSE, recursive = TRUE)

# --- Load cell information ---
cells <- read.csv(cell_csv)
batch_ids <- start_id:end_id
batch_ids <- batch_ids[batch_ids < nrow(cells)]
n_batch <- length(batch_ids)

if (n_batch == 0) {
  cat("No cells to process.\n")
  quit(save = "no")
}

cat(sprintf("Processing cells %d to %d (%d cells)\n", start_id, end_id, n_batch))

years <- year_start:(year_start + n_years - 1)
fail_file <- file.path(fail_dir, sprintf("failed_%06d_%06d.txt", start_id, end_id))

# ====================================================================
# Pre-allocate matrices
# ====================================================================

# 12 coefficient matrices (grid x ensemble)
mat_st_mu0  <- matrix(NA_real_, nrow = n_batch, ncol = n_mc)
mat_st_sig0 <- matrix(NA_real_, nrow = n_batch, ncol = n_mc)
mat_ln_mu0  <- matrix(NA_real_, nrow = n_batch, ncol = n_mc)
mat_ln_mu1  <- matrix(NA_real_, nrow = n_batch, ncol = n_mc)
mat_ln_sig0 <- matrix(NA_real_, nrow = n_batch, ncol = n_mc)
mat_ln_sig1 <- matrix(NA_real_, nrow = n_batch, ncol = n_mc)
mat_qd_mu0  <- matrix(NA_real_, nrow = n_batch, ncol = n_mc)
mat_qd_mu1  <- matrix(NA_real_, nrow = n_batch, ncol = n_mc)
mat_qd_mu2  <- matrix(NA_real_, nrow = n_batch, ncol = n_mc)
mat_qd_sig0 <- matrix(NA_real_, nrow = n_batch, ncol = n_mc)
mat_qd_sig1 <- matrix(NA_real_, nrow = n_batch, ncol = n_mc)
mat_qd_sig2 <- matrix(NA_real_, nrow = n_batch, ncol = n_mc)

# 3 Q100 matrices (grid x ensemble)
mat_st_q100 <- matrix(NA_real_, nrow = n_batch, ncol = n_mc)
mat_ln_q100 <- matrix(NA_real_, nrow = n_batch, ncol = n_mc)
mat_qd_q100 <- matrix(NA_real_, nrow = n_batch, ncol = n_mc)

# Truth values (grid)
truth_mu0   <- rep(NA_real_, n_batch)
truth_mu1   <- rep(NA_real_, n_batch)
truth_sig0  <- rep(NA_real_, n_batch)
truth_sig1  <- rep(NA_real_, n_batch)
truth_delta <- rep(NA_real_, n_batch)
truth_q100  <- rep(NA_real_, n_batch)

# Summary statistics (grid): 3 models x 3 stats = 9 values
sum_st_sd   <- rep(NA_real_, n_batch)
sum_st_iqr  <- rep(NA_real_, n_batch)
sum_st_tail <- rep(NA_real_, n_batch)
sum_ln_sd   <- rep(NA_real_, n_batch)
sum_ln_iqr  <- rep(NA_real_, n_batch)
sum_ln_tail <- rep(NA_real_, n_batch)
sum_qd_sd   <- rep(NA_real_, n_batch)
sum_qd_iqr  <- rep(NA_real_, n_batch)
sum_qd_tail <- rep(NA_real_, n_batch)

# Cell ID and coordinates
cell_ids <- integer(n_batch)
cell_iy  <- integer(n_batch)
cell_ix  <- integer(n_batch)

# ====================================================================
# Fit functions: return list(coefs, q100_mean)
# ====================================================================

# End-of-century evaluation window (matches manuscript 2.4 and 060).
# Stationary is FITTED only to this window (30 yr); nonstationary models are
# fitted to all 120 yr but their Q100 is AVERAGED over this window.
eval_lo <- 2071
eval_hi <- 2100

fit_stationary <- function(df) {
  df    <- df[df$year >= eval_lo & df$year <= eval_hi, , drop = FALSE]   # 30-yr fit
  model <- gamlss(outflow_neg ~ 1, sigma.fo = ~ 1, family = "GU", data = df, trace = FALSE)
  mu_f  <- fitted(model, "mu")
  sig_f <- fitted(model, "sigma")
  q100  <- -qGU(1 - 0.99, mu = mu_f, sigma = sig_f)
  list(coefs = c(coef(model, what = "mu")[1], coef(model, what = "sigma")[1]),
       q100_mean = mean(q100))
}

fit_linear <- function(df) {
  model <- gamlss(outflow_neg ~ year, sigma.fo = ~ year, family = "GU", data = df, trace = FALSE)
  mu_f  <- fitted(model, "mu")
  sig_f <- fitted(model, "sigma")
  q100  <- -qGU(1 - 0.99, mu = mu_f, sigma = sig_f)
  late  <- df$year >= eval_lo & df$year <= eval_hi
  mu_c  <- coef(model, what = "mu")
  sig_c <- coef(model, what = "sigma")
  list(coefs = c(mu_c[1], mu_c[2], sig_c[1], sig_c[2]),
       q100_mean = mean(q100[late]))                                     # eval over late window
}

fit_quadratic <- function(df) {
  model <- gamlss(outflow_neg ~ poly(year, 2, raw = TRUE),
                  sigma.fo = ~ poly(year, 2, raw = TRUE),
                  family = "GU", data = df, trace = FALSE)
  mu_f  <- fitted(model, "mu")
  sig_f <- fitted(model, "sigma")
  q100  <- -qGU(1 - 0.99, mu = mu_f, sigma = sig_f)
  late  <- df$year >= eval_lo & df$year <= eval_hi
  mu_c  <- coef(model, what = "mu")
  sig_c <- coef(model, what = "sigma")
  list(coefs = c(mu_c[1], mu_c[2], mu_c[3], sig_c[1], sig_c[2], sig_c[3]),
       q100_mean = mean(q100[late]))                                     # eval over late window
}

# Helper: compute error distribution statistics
compute_error_stats <- function(q100_vec, truth_val) {
  # Relative error: dimensionless, scale-independent
  errors <- (q100_vec - truth_val) / truth_val
  valid  <- errors[!is.na(errors) & is.finite(errors)]
  if (length(valid) < 2) return(c(NA_real_, NA_real_, NA_real_))
  sd_val   <- sd(valid)
  iqr_val  <- IQR(valid)
  tail_val <- as.numeric(quantile(valid, 0.95) - quantile(valid, 0.05))
  c(sd_val, iqr_val, tail_val)
}

# ====================================================================
# Main processing loop
# ====================================================================
con_amax <- file(amax_bin, "rb")
n_success <- 0
n_fail    <- 0

for (i in seq_along(batch_ids)) {
  cid <- batch_ids[i]
  cell_ids[i] <- cid
  cell_iy[i]  <- cells$iy[cid + 1]
  cell_ix[i]  <- cells$ix[cid + 1]

  # --- Read 120 AMAX values ---
  offset <- as.numeric(cid) * n_years * 4
  seek(con_amax, where = offset, origin = "start")
  amax_values <- readBin(con_amax, what = "numeric", size = 4, n = n_years, endian = "little")

  # Skip cells with no meaningful flow:
  #   - missing/short data
  #   - all zeros (ocean, desert)
  #   - max AMAX < 1 m3/s (negligible flow, causes GAMLSS numerical failure)
  min_flow_threshold <- 1.0  # m3/s
  if (length(amax_values) != n_years || max(amax_values) < min_flow_threshold) {
    cat(sprintf("DATA_SKIP cell_id=%d len=%d max=%.4g\n",
                cid, length(amax_values), max(amax_values)),
        file = fail_file, append = TRUE)
    n_fail <- n_fail + 1
    next
  }

  # --- Fit linear GAMLSS to observed data (ground truth) ---
  amax_df <- data.frame(year = years, outflow = amax_values, outflow_neg = -amax_values)

  truth_result <- tryCatch({
    model_true <- gamlss(outflow_neg ~ year, sigma.fo = ~ year, family = GU,
                         data = amax_df, trace = FALSE)
    mu_neg <- fitted(model_true, "mu")
    sig    <- fitted(model_true, "sigma")
    q100   <- -qGU(1 - 0.99, mu = mu_neg, sigma = sig)
    late   <- amax_df$year >= eval_lo & amax_df$year <= eval_hi
    tmu    <- coef(model_true, what = "mu")
    tsig   <- coef(model_true, what = "sigma")
    list(mu_neg = mu_neg, sigma = sig,
         mu0 = tmu[1], mu1 = tmu[2], sig0 = tsig[1], sig1 = tsig[2],
         q100_uncorrected = mean(q100[late]))   # truth Q100 over the late window
  }, error = function(e) conditionMessage(e))

  if (is.character(truth_result)) {
    cat(sprintf("TRUTH_FAIL cell_id=%d err=%s range=[%.4g,%.4g] sd=%.4g\n",
                cid, truth_result,
                min(amax_values), max(amax_values), sd(amax_values)),
        file = fail_file, append = TRUE)
    n_fail <- n_fail + 1
    next
  }

  mu_true_neg <- truth_result$mu_neg
  sigma_true  <- truth_result$sigma
  truth_mu0[i]  <- truth_result$mu0
  truth_mu1[i]  <- truth_result$mu1
  truth_sig0[i] <- truth_result$sig0
  truth_sig1[i] <- truth_result$sig1

  # --- Compute delta correction ---
  mc_for_delta <- replicate(100, {
    -mapply(function(m, s) rGU(1, mu = m, sigma = s), mu_true_neg, sigma_true)
  })
  min_val <- min(mc_for_delta)
  q1_val  <- quantile(as.vector(mc_for_delta), 0.25)
  delta   <- abs(min_val) + q1_val
  truth_delta[i] <- delta
  truth_q100[i]  <- truth_result$q100_uncorrected + delta

  # --- MC loop ---
  for (m in 1:n_mc) {
    mc_series <- -mapply(function(mu, sig) rGU(1, mu = mu, sigma = sig),
                         mu_true_neg, sigma_true) + delta
    df_mc <- data.frame(year = years, outflow = mc_series, outflow_neg = -mc_series)

    tryCatch({
      r_st <- fit_stationary(df_mc)
      mat_st_mu0[i, m]  <- r_st$coefs[1]
      mat_st_sig0[i, m] <- r_st$coefs[2]
      mat_st_q100[i, m] <- r_st$q100_mean + delta

      r_ln <- fit_linear(df_mc)
      mat_ln_mu0[i, m]  <- r_ln$coefs[1]
      mat_ln_mu1[i, m]  <- r_ln$coefs[2]
      mat_ln_sig0[i, m] <- r_ln$coefs[3]
      mat_ln_sig1[i, m] <- r_ln$coefs[4]
      mat_ln_q100[i, m] <- r_ln$q100_mean + delta

      r_qd <- fit_quadratic(df_mc)
      mat_qd_mu0[i, m]  <- r_qd$coefs[1]
      mat_qd_mu1[i, m]  <- r_qd$coefs[2]
      mat_qd_mu2[i, m]  <- r_qd$coefs[3]
      mat_qd_sig0[i, m] <- r_qd$coefs[4]
      mat_qd_sig1[i, m] <- r_qd$coefs[5]
      mat_qd_sig2[i, m] <- r_qd$coefs[6]
      mat_qd_q100[i, m] <- r_qd$q100_mean + delta
    }, error = function(e) {
      # Leave as NA (pre-allocated)
    })
  }

  # --- Compute error statistics for this cell ---
  # error = estimated_Q100 - truth_Q100 (delta included in both, cancels out)
  st_stats <- compute_error_stats(mat_st_q100[i, ], truth_q100[i])
  ln_stats <- compute_error_stats(mat_ln_q100[i, ], truth_q100[i])
  qd_stats <- compute_error_stats(mat_qd_q100[i, ], truth_q100[i])

  sum_st_sd[i]   <- st_stats[1]
  sum_st_iqr[i]  <- st_stats[2]
  sum_st_tail[i] <- st_stats[3]
  sum_ln_sd[i]   <- ln_stats[1]
  sum_ln_iqr[i]  <- ln_stats[2]
  sum_ln_tail[i] <- ln_stats[3]
  sum_qd_sd[i]   <- qd_stats[1]
  sum_qd_iqr[i]  <- qd_stats[2]
  sum_qd_tail[i] <- qd_stats[3]

  n_success <- n_success + 1

  if (i %% 5 == 0) {
    cat(sprintf("  [%d/%d] success=%d, fail=%d\n", i, n_batch, n_success, n_fail))
  }
}

close(con_amax)

cat(sprintf("Fitting done: %d success, %d fail. Writing output...\n", n_success, n_fail))

# ====================================================================
# Helper: write a matrix as flat float32 binary
# Layout: column-major (R default), n_batch x n_mc per matrix
# Multiple matrices are concatenated in order.
# ====================================================================
write_bin_matrices <- function(filepath, ...) {
  mats <- list(...)
  con <- file(filepath, "wb")
  for (m in mats) {
    writeBin(as.numeric(m), con, size = 4, endian = "little")
  }
  close(con)
}

# ====================================================================
# Write binary: mc_results (12 coefficient matrices)
# File layout: 12 matrices concatenated, each [n_batch x n_mc] float32
# Order: st_mu0, st_sig0, ln_mu0, ln_mu1, ln_sig0, ln_sig1,
#        qd_mu0, qd_mu1, qd_mu2, qd_sig0, qd_sig1, qd_sig2
# ====================================================================
result_path <- file.path(result_dir, sprintf("chunk_%06d_%06d.bin", start_id, end_id))
write_bin_matrices(result_path,
  mat_st_mu0, mat_st_sig0,
  mat_ln_mu0, mat_ln_mu1, mat_ln_sig0, mat_ln_sig1,
  mat_qd_mu0, mat_qd_mu1, mat_qd_mu2, mat_qd_sig0, mat_qd_sig1, mat_qd_sig2)
cat(sprintf("Saved: %s\n", result_path))

# ====================================================================
# Write binary: mc_q100 (3 Q100 matrices)
# File layout: 3 matrices concatenated, each [n_batch x n_mc] float32
# Order: st_q100, ln_q100, qd_q100
# ====================================================================
q100_path <- file.path(q100_dir, sprintf("q100_%06d_%06d.bin", start_id, end_id))
write_bin_matrices(q100_path, mat_st_q100, mat_ln_q100, mat_qd_q100)
cat(sprintf("Saved: %s\n", q100_path))

# ====================================================================
# Write CSV: mc_summary (error statistics per grid, lightweight)
# ====================================================================
summary_df <- data.frame(
  cell_id     = cell_ids,
  iy          = cell_iy,
  ix          = cell_ix,
  truth_q100  = truth_q100,
  st_err_sd   = sum_st_sd,
  st_err_iqr  = sum_st_iqr,
  st_err_tail = sum_st_tail,
  ln_err_sd   = sum_ln_sd,
  ln_err_iqr  = sum_ln_iqr,
  ln_err_tail = sum_ln_tail,
  qd_err_sd   = sum_qd_sd,
  qd_err_iqr  = sum_qd_iqr,
  qd_err_tail = sum_qd_tail
)

sum_path <- file.path(summary_dir, sprintf("summary_%06d_%06d.csv", start_id, end_id))
write.csv(summary_df, sum_path, row.names = FALSE)
cat(sprintf("Saved: %s\n", sum_path))

# ====================================================================
# Write CSV: mc_truth (ground truth coefficients + Q100)
# ====================================================================
truth_df <- data.frame(
  cell_id       = cell_ids,
  iy            = cell_iy,
  ix            = cell_ix,
  truth_q100    = truth_q100,
  truth_mu_0    = truth_mu0,
  truth_mu_1    = truth_mu1,
  truth_sigma_0 = truth_sig0,
  truth_sigma_1 = truth_sig1,
  delta         = truth_delta
)

truth_path <- file.path(truth_dir, sprintf("truth_%06d_%06d.csv", start_id, end_id))
write.csv(truth_df, truth_path, row.names = FALSE)
cat(sprintf("Saved: %s\n", truth_path))

cat(sprintf("All done: %d success, %d fail out of %d cells\n", n_success, n_fail, n_batch))
