#!/usr/bin/env Rscript
#
# step06_monte_carlo_stationary_and_quadratic_parents.R
# All-grid version of the 3x3 truth-robustness MC (045), run one PARENT (truth)
# at a time so that parents already covered elsewhere can be skipped.
#
#   parent (truth) in {st, ln, qd}  x  fitted model in {st, ln, qd}
#
# Relationship to existing scripts:
#   * 040 : linear parent only, all grid, n_mc=3000, saves binaries,
#           SD/IQR/tail (NO bias). Its bias is added afterwards by 054.
#   * 045 : full 3x3 but on the 044 subsample, n_mc=500, summary CSV only.
#   * 045b: full 3x3 logic on ALL land cells (040's cell indexing), one parent
#           per invocation, n_mc configurable, outputs bias + SD + IQR + tail.
#
# Cell indexing follows 040 (cell_id = 0-based row of 010/land_cells.csv;
# AMAX read by seek into 030/amax_all.bin). Fit/truth/stat conventions and the
# delta handling follow 045 (the fits are made on the delta-shifted series so
# their Q100 already contains delta; truth_q100 = uncorrected + delta; NO second
# delta is added -- this is the clean convention and matches 054's bias).
#
# Output (per parent T and chunk):
#   <dat_dir>/045b/<T>/summary/summary_SSSSSS_EEEEEE.csv
#       cell_id, iy, ix, truth_q100,
#       st_bias, st_sd, st_iqr, st_tail, ln_..., qd_...
#   <dat_dir>/045b/<T>/truth/truth_SSSSSS_EEEEEE.csv
#       cell_id, iy, ix, truth_q100, delta,
#       truth_mu_0..2, truth_sigma_0..2   (GU-negated coefs; NA where absent)
#   <dat_dir>/045b/<T>/fail_log/failed_*.txt
#
# Usage:
#   Rscript step06_monte_carlo_stationary_and_quadratic_parents.R <start_id> <end_id> <dat_dir> <n_mc> <truth>
#     truth in {st, ln, qd}

# --- Parse arguments ---
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 5) {
  stop("Usage: Rscript step06_monte_carlo_stationary_and_quadratic_parents.R <start_id> <end_id> <dat_dir> <n_mc> <truth{st,ln,qd}>")
}
start_id <- as.integer(args[1])
end_id   <- as.integer(args[2])
dat_dir  <- args[3]
n_mc     <- as.integer(args[4])
truth    <- args[5]
if (!truth %in% c("st", "ln", "qd")) stop("truth must be one of st, ln, qd")

# --- Load libraries ---
suppressMessages(suppressWarnings({
  library(gamlss)
  library(gamlss.dist)
}))

# --- Parameters ---
# Decorrelate the random streams of different parents (st/ln/qd) for the same
# chunk so that running two parents does not reuse identical noise.
truth_offset <- c(st = 0L, ln = 1000000L, qd = 2000000L)[[truth]]
set.seed(123 + start_id + truth_offset)

n_years    <- 120
year_start <- 1981
eval_lo    <- 2071
eval_hi    <- 2100
min_flow_threshold <- 1.0   # m3/s

# --- Paths ---
amax_bin    <- file.path(dat_dir, "030", "amax_all.bin")
cell_csv    <- file.path(dat_dir, "010", "land_cells.csv")
out_root    <- file.path(dat_dir, "045b", truth)
summary_dir <- file.path(out_root, "summary")
truth_dir   <- file.path(out_root, "truth")
fail_dir    <- file.path(out_root, "fail_log")
dir.create(summary_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(truth_dir,   showWarnings = FALSE, recursive = TRUE)
dir.create(fail_dir,    showWarnings = FALSE, recursive = TRUE)

# Checkpoint: skip this chunk if its summary already exists.
out_summary <- file.path(summary_dir, sprintf("summary_%06d_%06d.csv", start_id, end_id))
if (file.exists(out_summary)) {
  cat(sprintf("Exists, skip: %s\n", out_summary))
  quit(save = "no")
}
fail_file <- file.path(fail_dir, sprintf("failed_%06d_%06d.txt", start_id, end_id))

# --- Load cell information ---
cells <- read.csv(cell_csv)
batch_ids <- start_id:end_id
batch_ids <- batch_ids[batch_ids < nrow(cells)]
n_batch <- length(batch_ids)
if (n_batch == 0) { cat("No cells to process.\n"); quit(save = "no") }

cat(sprintf("Parent=%s | cells %d to %d (%d cells) | n_mc=%d\n",
            truth, start_id, end_id, n_batch, n_mc))

years <- year_start:(year_start + n_years - 1)

# ====================================================================
# Fitted models (identical conventions to 040/045; GU on negated flow).
# Stationary is FITTED to the 2071-2100 window only; nonstationary models are
# fitted to all 120 yr and their Q100 is averaged over the window.
# ====================================================================
fit_stationary <- function(df) {
  d     <- df[df$year >= eval_lo & df$year <= eval_hi, , drop = FALSE]
  model <- gamlss(outflow_neg ~ 1, sigma.fo = ~ 1, family = "GU", data = d, trace = FALSE)
  mean(-qGU(1 - 0.99, mu = fitted(model, "mu"), sigma = fitted(model, "sigma")))
}
# Linear / quadratic return BOTH the window-mean Q100 and the model AIC (needed
# for the AIC-adaptive estimator below). All on the full 120 yr -> AICs comparable.
fit_linear <- function(df) {
  model <- gamlss(outflow_neg ~ year, sigma.fo = ~ year, family = "GU", data = df, trace = FALSE)
  q100  <- -qGU(1 - 0.99, mu = fitted(model, "mu"), sigma = fitted(model, "sigma"))
  late  <- df$year >= eval_lo & df$year <= eval_hi
  list(q = mean(q100[late]), aic = as.numeric(AIC(model)))
}
fit_quadratic <- function(df) {
  model <- gamlss(outflow_neg ~ poly(year, 2, raw = TRUE),
                  sigma.fo = ~ poly(year, 2, raw = TRUE),
                  family = "GU", data = df, trace = FALSE)
  q100  <- -qGU(1 - 0.99, mu = fitted(model, "mu"), sigma = fitted(model, "sigma"))
  late  <- df$year >= eval_lo & df$year <= eval_hi
  list(q = mean(q100[late]), aic = as.numeric(AIC(model)))
}

# --- Truth builder: fit the chosen parent to the observed 120-yr series. ---
# Returns the GU-negated mu/sigma series, the (uncorrected) window-mean Q100,
# and the coefficients padded to (mu0,mu1,mu2, sig0,sig1,sig2).
build_truth <- function(truth_name, amax_df) {
  fo_mu  <- switch(truth_name, st = outflow_neg ~ 1, ln = outflow_neg ~ year,
                   qd = outflow_neg ~ poly(year, 2, raw = TRUE))
  fo_sig <- switch(truth_name, st = ~ 1, ln = ~ year, qd = ~ poly(year, 2, raw = TRUE))
  model  <- gamlss(fo_mu, sigma.fo = fo_sig, family = GU, data = amax_df, trace = FALSE)
  mu_neg <- fitted(model, "mu")
  sig    <- fitted(model, "sigma")
  q100   <- -qGU(1 - 0.99, mu = mu_neg, sigma = sig)
  late   <- amax_df$year >= eval_lo & amax_df$year <= eval_hi
  mc <- coef(model, what = "mu");    mc <- c(mc, rep(NA_real_, 3 - length(mc)))
  sc <- coef(model, what = "sigma"); sc <- c(sc, rep(NA_real_, 3 - length(sc)))
  list(mu_neg = mu_neg, sigma = sig, q100_uncorrected = mean(q100[late]),
       mu = mc, sig = sc)
}

# --- Error statistics: bias (mean), SD, IQR, tail (q95 - q5) ---
error_stats <- function(q100_vec, truth_val) {
  e <- (q100_vec - truth_val) / truth_val
  v <- e[is.finite(e)]
  if (length(v) < 2) return(rep(NA_real_, 4))
  c(mean(v), sd(v), IQR(v),
    as.numeric(quantile(v, 0.95) - quantile(v, 0.05)))
}

# Models compared per realization. "ad" = AIC-ADAPTIVE: take the quadratic Q100
# only when AIC justifies QD over LIN by more than DELTA_AIC, else the linear Q100
# (the LIN-default / QD-when-justified rule). Adding it lets the Fig-4 comparison
# include the adaptive estimator alongside LIN-only (ln) and QD-only (qd).
DELTA_AIC <- 2.0
MODELS <- c("st", "ln", "qd", "ad")

# ====================================================================
# Main loop
# ====================================================================
con_amax <- file(amax_bin, "rb")
sum_rows   <- vector("list", n_batch)
truth_rows <- vector("list", n_batch)
n_ok <- 0; n_fail <- 0

for (i in seq_along(batch_ids)) {
  cid <- batch_ids[i]
  iy  <- cells$iy[cid + 1]
  ix  <- cells$ix[cid + 1]

  # --- Read 120 AMAX values for this cell ---
  seek(con_amax, where = as.numeric(cid) * n_years * 4, origin = "start")
  amax_values <- readBin(con_amax, what = "numeric", size = 4, n = n_years, endian = "little")
  if (length(amax_values) != n_years || max(amax_values) < min_flow_threshold) {
    cat(sprintf("DATA_SKIP cell_id=%d\n", cid), file = fail_file, append = TRUE)
    n_fail <- n_fail + 1
    next
  }
  amax_df <- data.frame(year = years, outflow_neg = -amax_values)

  # --- Build the parent (truth) ---
  tr <- tryCatch(build_truth(truth, amax_df), error = function(e) conditionMessage(e))
  if (is.character(tr)) {
    cat(sprintf("TRUTH_FAIL cell_id=%d truth=%s err=%s\n", cid, truth, tr),
        file = fail_file, append = TRUE)
    n_fail <- n_fail + 1
    next
  }
  # Generation needs sigma > 0 in every year (rGU requires positive scale).
  if (any(!is.finite(tr$sigma)) || any(tr$sigma <= 0)) {
    cat(sprintf("SIGMA_NONPOS cell_id=%d truth=%s\n", cid, truth),
        file = fail_file, append = TRUE)
    n_fail <- n_fail + 1
    next
  }

  # --- delta correction (as 040/045): 100 pre-draws, floor at Q1 ---
  mc_for_delta <- replicate(100, {
    -mapply(function(m, s) rGU(1, mu = m, sigma = s), tr$mu_neg, tr$sigma)
  })
  delta <- abs(min(mc_for_delta)) + quantile(as.vector(mc_for_delta), 0.25)
  truth_q100 <- tr$q100_uncorrected + delta

  # --- Monte Carlo: each trial is one delta-shifted 120-yr realization ---
  q100_mat <- matrix(NA_real_, nrow = length(MODELS), ncol = n_mc,
                     dimnames = list(MODELS, NULL))
  for (m in seq_len(n_mc)) {
    mc_series <- -mapply(function(mu, sig) rGU(1, mu = mu, sigma = sig),
                         tr$mu_neg, tr$sigma) + delta
    df_mc <- data.frame(year = years, outflow_neg = -mc_series)
    # NOTE: no "+ delta" on the fitted Q100 (clean convention, matches 045/054).
    tryCatch({
      q100_mat["st", m] <- fit_stationary(df_mc)
      rl <- fit_linear(df_mc)
      rq <- fit_quadratic(df_mc)
      q100_mat["ln", m] <- rl$q
      q100_mat["qd", m] <- rq$q
      # AIC-adaptive: QD's Q100 iff AIC prefers QD over LIN by > DELTA_AIC.
      q100_mat["ad", m] <- if (is.finite(rq$aic) && is.finite(rl$aic) &&
                               rq$aic < rl$aic - DELTA_AIC) rq$q else rl$q
    }, error = function(e) {})
  }

  stats <- unlist(lapply(MODELS, function(mod) error_stats(q100_mat[mod, ], truth_q100)))
  names(stats) <- as.vector(t(outer(MODELS, c("bias", "sd", "iqr", "tail"),
                                     function(a, b) paste0(a, "_", b))))

  sum_rows[[i]] <- data.frame(
    cell_id = cid, iy = iy, ix = ix, truth_q100 = truth_q100,
    as.list(stats), check.names = FALSE)
  truth_rows[[i]] <- data.frame(
    cell_id = cid, iy = iy, ix = ix, truth_q100 = truth_q100, delta = delta,
    truth_mu_0 = tr$mu[1], truth_mu_1 = tr$mu[2], truth_mu_2 = tr$mu[3],
    truth_sigma_0 = tr$sig[1], truth_sigma_1 = tr$sig[2], truth_sigma_2 = tr$sig[3])

  n_ok <- n_ok + 1
  if (i %% 5 == 0) cat(sprintf("  [%d/%d] ok=%d fail=%d\n", i, n_batch, n_ok, n_fail))
}
close(con_amax)

# ====================================================================
# Write outputs (atomic: write to .tmp then rename, so a killed job never
# leaves a half-written CSV that the checkpoint would treat as complete).
# ====================================================================
sum_df   <- do.call(rbind, sum_rows[!vapply(sum_rows, is.null, logical(1))])
truth_df <- do.call(rbind, truth_rows[!vapply(truth_rows, is.null, logical(1))])
if (is.null(sum_df))   sum_df   <- data.frame()
if (is.null(truth_df)) truth_df <- data.frame()

tmp <- paste0(out_summary, ".tmp")
write.csv(sum_df, tmp, row.names = FALSE); file.rename(tmp, out_summary)

out_truth <- file.path(truth_dir, sprintf("truth_%06d_%06d.csv", start_id, end_id))
tmp <- paste0(out_truth, ".tmp")
write.csv(truth_df, tmp, row.names = FALSE); file.rename(tmp, out_truth)

cat(sprintf("Saved: %s (%d cells, %d fail)\n", out_summary, n_ok, n_fail))
