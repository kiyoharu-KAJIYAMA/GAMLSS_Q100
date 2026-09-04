#!/usr/bin/env Rscript
#
# step19_aic_adaptive_estimator_batch.R
# All-grid AIC-ADAPTIVE estimator (the "ad" column of 045b) as a standalone run,
# so it can be added to the existing 3x3 WITHOUT recomputing st/ln/qd (those are
# already in 045b/summary_allgrid_{st,qd}.csv and 054/bias_allgrid.csv) and
# WITHOUT being skipped by 045b's checkpoint. Writes to its own 108/ tree.
#
# Adaptive rule, per Monte Carlo realization (identical to 045b's "ad"):
#   fit LIN and QD (both on 120 yr -> AICs comparable);
#   use QD's window-mean Q100 iff aic_qd < aic_lin - DELTA_AIC, else LIN's.
# Stationary is NOT fitted here (the adaptive estimator never uses it), so this
# does ~2/3 of 045b's fits per realization.
#
# Generation / delta / truth conventions mirror 045b exactly, so the resulting
# ad bias/SD/IQR/tail are on the same scale as the st/ln/qd columns and merge by
# cell_id (use step20_combine_aic_adaptive.py, then 107 picks up the 'ad' fit automatically).
#
# Output (per parent T and chunk, atomic write):
#   <dat_dir>/108/<T>/summary/ad_SSSSSS_EEEEEE.csv
#       cell_id, iy, ix, truth_q100,
#       ad_{bias,sd,iqr,tail}   AIC-adaptive estimator
#       ln_{bias,sd,iqr,tail}   LIN-only,  SAME realizations (self-contained 3-way)
#       qd_{bias,sd,iqr,tail}   QD-only,   SAME realizations
#       ad_qd_frac    = fraction of realizations where QD beat LIN by > DELTA_AIC
#                       (curvature detection FREQUENCY, ΔAIC=2 rule)
#       ad_qd_frac_d0 = same with ΔAIC=0 (pure lowest-AIC selection)
#       ad_mean_daic  = mean(aic_lin - aic_qd) (curvature support STRENGTH; >0=QD)
#       ad_n_valid    = number of realizations with a successful LIN+QD fit
#
# Checkpoint: skips a chunk whose summary CSV already exists.
#
# Usage:
#   Rscript step19_aic_adaptive_estimator_batch.R <start_id> <end_id> <dat_dir> <n_mc> <truth{st,ln,qd}>

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 5)
  stop("Usage: Rscript step19_aic_adaptive_estimator_batch.R <start_id> <end_id> <dat_dir> <n_mc> <truth{st,ln,qd}>")
start_id <- as.integer(args[1])
end_id   <- as.integer(args[2])
dat_dir  <- args[3]
n_mc     <- as.integer(args[4])
truth    <- args[5]
if (!truth %in% c("st", "ln", "qd")) stop("truth must be one of st, ln, qd")

suppressMessages(suppressWarnings({
  library(gamlss)
  library(gamlss.dist)
}))

# Match 045b's RNG streams so the adaptive run reuses the same noise per parent.
truth_offset <- c(st = 0L, ln = 1000000L, qd = 2000000L)[[truth]]
set.seed(123 + start_id + truth_offset)

n_years    <- 120
year_start <- 1981
eval_lo    <- 2071
eval_hi    <- 2100
min_flow_threshold <- 1.0
DELTA_AIC  <- 2.0
years      <- year_start:(year_start + n_years - 1)

amax_bin    <- file.path(dat_dir, "030", "amax_all.bin")
cell_csv    <- file.path(dat_dir, "010", "land_cells.csv")
out_root    <- file.path(dat_dir, "108", truth)
summary_dir <- file.path(out_root, "summary")
fail_dir    <- file.path(out_root, "fail_log")
dir.create(summary_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(fail_dir,    showWarnings = FALSE, recursive = TRUE)

out_summary <- file.path(summary_dir, sprintf("ad_%06d_%06d.csv", start_id, end_id))
if (file.exists(out_summary)) { cat(sprintf("Exists, skip: %s\n", out_summary)); quit(save = "no") }
fail_file <- file.path(fail_dir, sprintf("failed_%06d_%06d.txt", start_id, end_id))

cells <- read.csv(cell_csv)
batch_ids <- start_id:end_id
batch_ids <- batch_ids[batch_ids < nrow(cells)]
n_batch <- length(batch_ids)
if (n_batch == 0) { cat("No cells to process.\n"); quit(save = "no") }
cat(sprintf("ADAPTIVE parent=%s | cells %d to %d (%d cells) | n_mc=%d\n",
            truth, start_id, end_id, n_batch, n_mc))

# --- LIN / QD fits return window-mean Q100 + AIC (mirror 045b) ---
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

build_truth <- function(truth_name, amax_df) {
  fo_mu  <- switch(truth_name, st = outflow_neg ~ 1, ln = outflow_neg ~ year,
                   qd = outflow_neg ~ poly(year, 2, raw = TRUE))
  fo_sig <- switch(truth_name, st = ~ 1, ln = ~ year, qd = ~ poly(year, 2, raw = TRUE))
  model  <- gamlss(fo_mu, sigma.fo = fo_sig, family = GU, data = amax_df, trace = FALSE)
  mu_neg <- fitted(model, "mu")
  sig    <- fitted(model, "sigma")
  q100   <- -qGU(1 - 0.99, mu = mu_neg, sigma = sig)
  late   <- amax_df$year >= eval_lo & amax_df$year <= eval_hi
  list(mu_neg = mu_neg, sigma = sig, q100_uncorrected = mean(q100[late]))
}

error_stats <- function(q100_vec, truth_val) {
  e <- (q100_vec - truth_val) / truth_val
  v <- e[is.finite(e)]
  if (length(v) < 2) return(rep(NA_real_, 4))
  c(mean(v), sd(v), IQR(v), as.numeric(quantile(v, 0.95) - quantile(v, 0.05)))
}

# ====================================================================
# Main loop
# ====================================================================
con_amax <- file(amax_bin, "rb")
rows <- vector("list", n_batch)
n_ok <- 0; n_fail <- 0

for (i in seq_along(batch_ids)) {
  cid <- batch_ids[i]
  iy  <- cells$iy[cid + 1]; ix <- cells$ix[cid + 1]

  seek(con_amax, where = as.numeric(cid) * n_years * 4, origin = "start")
  amax_values <- readBin(con_amax, what = "numeric", size = 4, n = n_years, endian = "little")
  if (length(amax_values) != n_years || max(amax_values) < min_flow_threshold) {
    cat(sprintf("DATA_SKIP cell_id=%d\n", cid), file = fail_file, append = TRUE)
    n_fail <- n_fail + 1; next
  }
  amax_df <- data.frame(year = years, outflow_neg = -amax_values)

  tr <- tryCatch(build_truth(truth, amax_df), error = function(e) conditionMessage(e))
  if (is.character(tr)) {
    cat(sprintf("TRUTH_FAIL cell_id=%d truth=%s err=%s\n", cid, truth, tr),
        file = fail_file, append = TRUE)
    n_fail <- n_fail + 1; next
  }
  if (any(!is.finite(tr$sigma)) || any(tr$sigma <= 0)) {
    cat(sprintf("SIGMA_NONPOS cell_id=%d truth=%s\n", cid, truth), file = fail_file, append = TRUE)
    n_fail <- n_fail + 1; next
  }

  mc_for_delta <- replicate(100, -mapply(function(m, s) rGU(1, mu = m, sigma = s), tr$mu_neg, tr$sigma))
  delta <- abs(min(mc_for_delta)) + quantile(as.vector(mc_for_delta), 0.25)
  truth_q100 <- tr$q100_uncorrected + delta

  ad_vec    <- rep(NA_real_, n_mc)     # adaptive Q100
  ln_vec    <- rep(NA_real_, n_mc)     # LIN-only Q100 (same realizations)
  qd_vec    <- rep(NA_real_, n_mc)     # QD-only  Q100 (same realizations)
  chose_qd  <- rep(NA, n_mc)           # ΔAIC=2 rule: TRUE=QD, FALSE=LIN, NA=fail
  chose_qd0 <- rep(NA, n_mc)           # ΔAIC=0 rule (pure lowest AIC)
  daic_vec  <- rep(NA_real_, n_mc)     # aic_lin - aic_qd (>0 = QD favoured)
  for (m in seq_len(n_mc)) {
    mc_series <- -mapply(function(mu, sig) rGU(1, mu = mu, sigma = sig), tr$mu_neg, tr$sigma) + delta
    df_mc <- data.frame(year = years, outflow_neg = -mc_series)
    tryCatch({
      rl <- fit_linear(df_mc)
      rq <- fit_quadratic(df_mc)
      ln_vec[m] <- rl$q
      qd_vec[m] <- rq$q
      if (is.finite(rl$aic) && is.finite(rq$aic)) {
        daic_vec[m]  <- rl$aic - rq$aic
        chose_qd[m]  <- rq$aic < rl$aic - DELTA_AIC
        chose_qd0[m] <- rq$aic < rl$aic
      }
      ad_vec[m] <- if (isTRUE(chose_qd[m])) rq$q else rl$q   # LIN default
    }, error = function(e) {})
  }

  # Per-cell diagnostics on the AIC selection (under the known parent):
  #   qd_frac    = fraction of realizations where QD beat LIN by > DELTA_AIC
  #                -> how OFTEN curvature is detected (ΔAIC=2 rule)
  #   qd_frac_d0 = same with ΔAIC=0 (pure lowest-AIC; the looser selection)
  #   mean_daic  = mean(aic_lin - aic_qd) -> how STRONGLY curvature is supported
  # These complement 100/103 (a single decision on the OBSERVED series) with the
  # sampling view: e.g. high qd_frac under a LINEAR parent = false positives.
  n_valid    <- sum(!is.na(chose_qd))
  qd_frac    <- if (n_valid > 0) sum(chose_qd, na.rm = TRUE) / n_valid else NA_real_
  qd_frac_d0 <- if (any(!is.na(chose_qd0))) mean(chose_qd0, na.rm = TRUE) else NA_real_
  mean_daic  <- if (any(is.finite(daic_vec))) mean(daic_vec, na.rm = TRUE) else NA_real_

  # ad / ln / qd error stats from the SAME realizations (self-contained 3-way).
  a <- error_stats(ad_vec, truth_q100)
  l <- error_stats(ln_vec, truth_q100)
  q <- error_stats(qd_vec, truth_q100)
  rows[[i]] <- data.frame(cell_id = cid, iy = iy, ix = ix, truth_q100 = truth_q100,
    ad_bias = a[1], ad_sd = a[2], ad_iqr = a[3], ad_tail = a[4],
    ln_bias = l[1], ln_sd = l[2], ln_iqr = l[3], ln_tail = l[4],
    qd_bias = q[1], qd_sd = q[2], qd_iqr = q[3], qd_tail = q[4],
    ad_qd_frac = qd_frac, ad_qd_frac_d0 = qd_frac_d0,
    ad_mean_daic = mean_daic, ad_n_valid = n_valid)
  n_ok <- n_ok + 1
  if (i %% 10 == 0) cat(sprintf("  [%d/%d] ok=%d fail=%d\n", i, n_batch, n_ok, n_fail))
}
close(con_amax)

df_out <- do.call(rbind, rows[!vapply(rows, is.null, logical(1))])
if (is.null(df_out)) df_out <- data.frame()
tmp <- paste0(out_summary, ".tmp")
write.csv(df_out, tmp, row.names = FALSE); file.rename(tmp, out_summary)
cat(sprintf("Saved: %s (%d cells, %d fail)\n", out_summary, n_ok, n_fail))
