#!/usr/bin/env Rscript
#
# step14_aic_change_type_batch.R
# Per cell, fit FOUR GU models to the 120-yr simulated AMAX and compare by AIC to
# decide WHICH kind of nonstationarity is statistically justified:
#   stationary   mu ~ 1     sigma ~ 1       (k = 2)
#   mean-only    mu ~ year  sigma ~ 1       (k = 3)  -- mean trend only
#   var-only     mu ~ 1     sigma ~ year    (k = 3)  -- variance trend only (GU log link)
#   both         mu ~ year  sigma ~ year    (k = 4)
#
# Motivation: 076c found that the VARIANCE non-stationarity (var_ratio) is the
# strongest INDEPENDENT driver of the excess SD improvement, while the mean trend
# (trend_pct) was collinear with the Slater clarity (rho = 0.73) and added nothing
# once controlled. The observational predictors therefore cannot cleanly separate
# "mean" from "variance" nonstationarity. This AIC decomposition does it directly
# inside the GU framework: all four models are fit to the SAME 120-yr record, so
# their AICs are comparable and tell us, per cell, whether the data justify a
# time-varying mean, a time-varying variance, both, or neither.
#
# Q100 conventions match 080: GU on negated flow, no delta (positive AMAX).
#
# Runs on FLOOD-RELEVANT cells only (same definition as 098/076b, read from
# 091/change_type_cells.csv), not all ~1.3M land cells -- this is the manuscript's
# flood domain and halves the cost. Requires 091 (run 094/095 first).
#
# Usage : Rscript step14_aic_change_type_batch.R <start_id> <end_id> <dat_dir>
# Output: <dat_dir>/077/aic/chunk_SSSSSS_EEEEEE.csv   (flood-relevant cells only)
#         <dat_dir>/077/fail_log/failed_SSSSSS_EEEEEE.txt
#   columns: cell_id, iy, ix, aic_st, aic_mean, aic_var, aic_both,
#            daic_mean, daic_var, daic_both, sig_slope, best_model, class
#   daic_* = aic_st - aic_*  (>0 => that nonstationary model beats stationary)
#   sig_slope = year coefficient of sigma in the 'both' model (GU log link); its
#               SIGN is the variance-trend direction (>0 INCREASING, <0 DECREASING),
#               so 079 can test whether only increasing variance drives the excess.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3)
  stop("Usage: Rscript step14_aic_change_type_batch.R <start_id> <end_id> <dat_dir>")
start_id <- as.integer(args[1])
end_id   <- as.integer(args[2])
dat_dir  <- args[3]

suppressMessages(suppressWarnings({
  library(gamlss)
  library(gamlss.dist)
}))

n_years    <- 120
year_start <- 1981
years      <- year_start:(year_start + n_years - 1)
min_flow   <- 1.0                                  # m3/s, as 040/080
ctrl       <- gamlss.control(n.cyc = 50, trace = FALSE)
DELTA_AIC  <- 2.0                                  # parsimony margin for the class label

amax_bin <- file.path(dat_dir, "030", "amax_all.bin")
cell_csv <- file.path(dat_dir, "010", "land_cells.csv")
out_dir  <- file.path(dat_dir, "077", "aic")
fail_dir <- file.path(dat_dir, "077", "fail_log")
dir.create(out_dir,  showWarnings = FALSE, recursive = TRUE)
dir.create(fail_dir, showWarnings = FALSE, recursive = TRUE)

cells <- read.csv(cell_csv)
batch_ids <- start_id:end_id
batch_ids <- batch_ids[batch_ids < nrow(cells)]
n_batch <- length(batch_ids)
if (n_batch == 0) { cat("No cells to process.\n"); quit(save = "no") }
fail_file <- file.path(fail_dir, sprintf("failed_%06d_%06d.txt", start_id, end_id))

# Flood-relevance (same definition as 098/076b): keep only cells with
# (q90 >= X1 OR max_amax >= X2) AND uparea >= A, read from 091. The expensive
# 4-model fit is then skipped for all non-flood cells.
X1_Q90 <- 50.0; X2_MAX <- 100.0; A_UP <- 50.0
flood_csv <- file.path(dat_dir, "091", "change_type_cells.csv")
if (!file.exists(flood_csv))
  stop(sprintf("Need %s for flood-relevance (run 094/095 first).", flood_csv))
ct <- read.csv(flood_csv)
fr <- ((ct$q90 >= X1_Q90) | (ct$max_amax >= X2_MAX)) & (ct$uparea_km2 >= A_UP)
fr[is.na(fr)] <- FALSE
is_flood <- logical(nrow(cells))
is_flood[ct$cell_id[fr] + 1] <- TRUE
cat(sprintf("Flood-relevant cells: %d of %d total\n", sum(is_flood), nrow(cells)))

# AIC of one GU(mu ~ fo_mu, sigma ~ fo_sig) fit; NA on failure.
aic_of <- function(fo_mu, fo_sig, df) {
  m <- try(gamlss(fo_mu, sigma.fo = fo_sig, family = "GU", data = df,
                  control = ctrl), silent = TRUE)
  if (inherits(m, "try-error")) return(NA_real_)
  as.numeric(AIC(m))
}

# Fit the 'both' model (mu~year, sigma~year) and return its AIC AND the sigma
# year-slope (log link). The slope's SIGN gives the variance-trend direction.
fit_both <- function(df) {
  m <- try(gamlss(outflow_neg ~ year, sigma.fo = ~ year, family = "GU", data = df,
                  control = ctrl), silent = TRUE)
  if (inherits(m, "try-error")) return(list(aic = NA_real_, slope = NA_real_))
  sc <- coef(m, what = "sigma")
  list(aic = as.numeric(AIC(m)),
       slope = if (length(sc) >= 2) as.numeric(sc[2]) else NA_real_)
}

res_cid <- integer(n_batch); res_iy <- integer(n_batch); res_ix <- integer(n_batch)
a_st <- a_mn <- a_vr <- a_bo <- rep(NA_real_, n_batch)
sig_slope <- rep(NA_real_, n_batch)     # sigma year-slope (both model); sign = direction
best <- rep(NA_character_, n_batch)
klass <- rep(NA_character_, n_batch)
keep_row <- logical(n_batch)            # TRUE only for flood-relevant cells that fit

con <- file(amax_bin, "rb")
n_ok <- 0; n_fail <- 0; n_skip <- 0
for (i in seq_along(batch_ids)) {
  cid <- batch_ids[i]
  res_cid[i] <- cid
  res_iy[i] <- cells$iy[cid + 1]; res_ix[i] <- cells$ix[cid + 1]

  if (!is_flood[cid + 1]) { n_skip <- n_skip + 1; next }   # not flood-relevant -> no fit

  seek(con, where = as.numeric(cid) * n_years * 4, origin = "start")
  flow <- readBin(con, what = "numeric", size = 4, n = n_years, endian = "little")
  if (length(flow) != n_years || max(flow) < min_flow) {
    cat(sprintf("DATA_SKIP cell_id=%d\n", cid), file = fail_file, append = TRUE)
    n_fail <- n_fail + 1; next
  }

  df <- data.frame(year = years, outflow_neg = -flow)
  a_st[i] <- aic_of(outflow_neg ~ 1,    ~ 1,    df)
  a_mn[i] <- aic_of(outflow_neg ~ year, ~ 1,    df)
  a_vr[i] <- aic_of(outflow_neg ~ 1,    ~ year, df)
  fb <- fit_both(df)
  a_bo[i] <- fb$aic; sig_slope[i] <- fb$slope

  aics <- c(stationary = a_st[i], mean = a_mn[i], var = a_vr[i], both = a_bo[i])
  if (all(is.na(aics))) {
    cat(sprintf("FIT_FAIL cell_id=%d\n", cid), file = fail_file, append = TRUE)
    n_fail <- n_fail + 1; next
  }
  amin <- min(aics, na.rm = TRUE)
  best[i] <- names(aics)[which.min(replace(aics, is.na(aics), Inf))]
  # Parsimony: call it "stationary" unless a nonstationary model beats stationary
  # by more than DELTA_AIC; otherwise the best (mean/var/both) is the class.
  klass[i] <- if (!is.na(a_st[i]) && a_st[i] <= amin + DELTA_AIC) "stationary" else best[i]

  keep_row[i] <- TRUE
  n_ok <- n_ok + 1
  if (i %% 50 == 0)
    cat(sprintf("  [%d/%d] ok=%d fail=%d skip=%d\n", i, n_batch, n_ok, n_fail, n_skip))
}
close(con)

out <- data.frame(
  cell_id = res_cid, iy = res_iy, ix = res_ix,
  aic_st = a_st, aic_mean = a_mn, aic_var = a_vr, aic_both = a_bo,
  daic_mean = a_st - a_mn,   # > 0 => mean-nonstationary beats stationary
  daic_var  = a_st - a_vr,   # > 0 => variance-nonstationary beats stationary
  daic_both = a_st - a_bo,
  sig_slope = sig_slope,     # sigma year-slope (both model); >0 incr, <0 decr variance
  best_model = best, class = klass
)
out <- out[keep_row, , drop = FALSE]    # flood-relevant cells that fit only
out_path <- file.path(out_dir, sprintf("chunk_%06d_%06d.csv", start_id, end_id))
write.csv(out, out_path, row.names = FALSE)
cat(sprintf("Saved: %s (%d ok, %d fail, %d non-flood skipped)\n",
            out_path, n_ok, n_fail, n_skip))
