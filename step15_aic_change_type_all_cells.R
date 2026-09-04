#!/usr/bin/env Rscript
#
# step15_aic_change_type_all_cells.R
# Same FOUR-GU-model AIC decomposition as 077 (stationary / mean-only / var-only /
# both -> class stationary/mean/var/both), but over ALL land cells, NOT only the
# flood-relevant subset. The only gate is max(AMAX) >= min_flow (1 m3/s): cells
# whose annual maxima never exceed 1 m3/s are skipped (no flood signal). This
# produces a whole-globe classification for the Figure-2 map (115).
#
# Usage : Rscript step15_aic_change_type_all_cells.R <start_id> <end_id> <dat_dir>
# Output: <dat_dir>/077b/aic/chunk_SSSSSS_EEEEEE.csv
#   columns: cell_id, iy, ix, aic_st, aic_mean, aic_var, aic_both,
#            daic_mean, daic_var, daic_both, sig_slope, best_model, class
# Checkpoint: skips a chunk whose CSV already exists.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3)
  stop("Usage: Rscript step15_aic_change_type_all_cells.R <start_id> <end_id> <dat_dir>")
start_id <- as.integer(args[1]); end_id <- as.integer(args[2]); dat_dir <- args[3]

suppressMessages(suppressWarnings({ library(gamlss); library(gamlss.dist) }))

n_years <- 120; year_start <- 1981
years   <- year_start:(year_start + n_years - 1)
min_flow <- 1.0
ctrl <- gamlss.control(n.cyc = 50, trace = FALSE)
DELTA_AIC <- 2.0

amax_bin <- file.path(dat_dir, "030", "amax_all.bin")
cell_csv <- file.path(dat_dir, "010", "land_cells.csv")
out_dir  <- file.path(dat_dir, "077b", "aic")
fail_dir <- file.path(dat_dir, "077b", "fail_log")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(fail_dir, showWarnings = FALSE, recursive = TRUE)

out_path <- file.path(out_dir, sprintf("chunk_%06d_%06d.csv", start_id, end_id))
if (file.exists(out_path)) { cat(sprintf("Exists, skip: %s\n", out_path)); quit(save = "no") }
fail_file <- file.path(fail_dir, sprintf("failed_%06d_%06d.txt", start_id, end_id))

cells <- read.csv(cell_csv)
batch_ids <- start_id:end_id
batch_ids <- batch_ids[batch_ids < nrow(cells)]
n_batch <- length(batch_ids)
if (n_batch == 0) { cat("No cells.\n"); quit(save = "no") }

aic_of <- function(fo_mu, fo_sig, df) {
  m <- try(gamlss(fo_mu, sigma.fo = fo_sig, family = "GU", data = df, control = ctrl), silent = TRUE)
  if (inherits(m, "try-error")) return(NA_real_)
  as.numeric(AIC(m))
}
fit_both <- function(df) {
  m <- try(gamlss(outflow_neg ~ year, sigma.fo = ~ year, family = "GU", data = df, control = ctrl), silent = TRUE)
  if (inherits(m, "try-error")) return(list(aic = NA_real_, slope = NA_real_))
  sc <- coef(m, what = "sigma")
  list(aic = as.numeric(AIC(m)), slope = if (length(sc) >= 2) as.numeric(sc[2]) else NA_real_)
}

res_cid <- integer(n_batch); res_iy <- integer(n_batch); res_ix <- integer(n_batch)
a_st <- a_mn <- a_vr <- a_bo <- rep(NA_real_, n_batch)
sig_slope <- rep(NA_real_, n_batch)
best <- rep(NA_character_, n_batch); klass <- rep(NA_character_, n_batch)
keep_row <- logical(n_batch)

con <- file(amax_bin, "rb")
n_ok <- 0; n_fail <- 0; n_skip <- 0
for (i in seq_along(batch_ids)) {
  cid <- batch_ids[i]; res_cid[i] <- cid
  res_iy[i] <- cells$iy[cid + 1]; res_ix[i] <- cells$ix[cid + 1]
  seek(con, where = as.numeric(cid) * n_years * 4, origin = "start")
  flow <- readBin(con, what = "numeric", size = 4, n = n_years, endian = "little")
  if (length(flow) != n_years || max(flow) < min_flow) { n_skip <- n_skip + 1; next }  # AMAX<=1 omitted
  df <- data.frame(year = years, outflow_neg = -flow)
  a_st[i] <- aic_of(outflow_neg ~ 1,    ~ 1,    df)
  a_mn[i] <- aic_of(outflow_neg ~ year, ~ 1,    df)
  a_vr[i] <- aic_of(outflow_neg ~ 1,    ~ year, df)
  fb <- fit_both(df); a_bo[i] <- fb$aic; sig_slope[i] <- fb$slope
  aics <- c(stationary = a_st[i], mean = a_mn[i], var = a_vr[i], both = a_bo[i])
  if (all(is.na(aics))) { cat(sprintf("ALL_FAIL cell_id=%d\n", cid), file = fail_file, append = TRUE); n_fail <- n_fail + 1; next }
  amin <- min(aics, na.rm = TRUE)
  best[i] <- names(aics)[which.min(replace(aics, is.na(aics), Inf))]
  klass[i] <- if (!is.na(a_st[i]) && a_st[i] <= amin + DELTA_AIC) "stationary" else best[i]
  keep_row[i] <- TRUE; n_ok <- n_ok + 1
  if (i %% 100 == 0) cat(sprintf("  [%d/%d] ok=%d skip=%d\n", i, n_batch, n_ok, n_skip))
}
close(con)

out <- data.frame(cell_id = res_cid, iy = res_iy, ix = res_ix,
                  aic_st = a_st, aic_mean = a_mn, aic_var = a_vr, aic_both = a_bo,
                  daic_mean = a_st - a_mn, daic_var = a_st - a_vr, daic_both = a_st - a_bo,
                  sig_slope = sig_slope, best_model = best, class = klass)
out <- out[keep_row, , drop = FALSE]
tmp <- paste0(out_path, ".tmp"); write.csv(out, tmp, row.names = FALSE); file.rename(tmp, out_path)
cat(sprintf("Saved: %s (%d ok, %d fail, %d skipped <=1 m3/s)\n", out_path, n_ok, n_fail, n_skip))
