#!/usr/bin/env Rscript
#
# step10_equivalent_ensemble_size_mc.R
# Find equivalent ensemble size for each grid cell.
#
# For each grid:
#   1) Read truth model coefficients from 064/mc_truth (no re-fitting needed)
#   2) Reconstruct mu(t), sigma(t) from saved coefficients
#   3) Run MC trials: nonstationary (e=3) vs stationary (varying e)
#   4) Find eq_ens where stationary SD <= nonstationary SD
#   5) Save results to CSV
#
# SPEED-UP (optional): if 050/summary_allgrid.csv is present, the cheap e=1 SDs
# it holds (stat_err_sd, lin_err_sd from 040) predict the crossing analytically
#   e* = 3 * (stat_err_sd / lin_err_sd)^2     (sampling SD scales ~ 1/sqrt(e))
# and the stationary sweep is restricted to a +/-PRED_MARGIN window around e*
# instead of the full [3,10] grid. Guards re-extend/fall back if the crossing is
# outside the window, so eq_ens is unchanged -- only far fewer MC fits are run
# (the console reports the realised evals/cell vs the full-sweep 71). Without the
# file every cell uses the full sweep, exactly as before.
#
# 040/064 used GU (Gumbel min) with negated flow. GU links: mu=identity,
# sigma=LOG, so the stored coefficients give:
#   mu_neg(t) = mu0 + mu1*t,           sigma(t) = exp(sig0 + sig1*t)
#   outflow ~ -GU(mu_neg, sigma)  =>  equivalent to RG(-mu_neg, sigma)
# So: mu_RG(t) = -mu0 - mu1*t,  sigma_RG(t) = exp(sig0 + sig1*t)
#
# The nonstationary side (the e=3 ensemble whose SD the stationary sweep must
# match) is SELECTABLE via the optional 4th argument ns_model:
#   lin (default) : linear  mu/sigma ~ year                  (original behaviour)
#   qd            : quadratic mu/sigma ~ poly(year, 2)        (matches 108 QD)
#   ad            : AIC-adaptive -- fit LIN and QD per trial, use QD iff
#                   aic_qd < aic_lin - DELTA_AIC, else LIN     (matches 108 'ad')
# This lets the equivalent-ensemble experiment compare the stationary fit against
# stationary-vs-linear, stationary-vs-quadratic, or stationary-vs-adaptive.
#
# The data-generating truth (parent) is also selectable via the optional 6th
# argument parent (st|ln|qd), read from 064/<parent>/mc_truth. Together with
# ns_model this spans the full parent x estimator matrix.
#
# Usage: Rscript step10_equivalent_ensemble_size_mc.R <start_id> <end_id> <dat_dir> [ns_model] [n_trials] [parent]
#
# Input: <dat_dir>/064/<parent>/mc_truth (truth coefficients; run 065 first)
#        <dat_dir>/050/summary_allgrid.csv  (optional e=1 SDs; speeds up sweep)
# Output: <dat_dir>/060/<parent>/<ns_model>/
#   eq_ens/eq_ens_SSSSSS_EEEEEE.csv : equivalent ensemble per grid
#   fail_log/failed_SSSSSS_EEEEEE.txt : error log

# --- Parse arguments ---
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) stop("Usage: Rscript step10_equivalent_ensemble_size_mc.R <start_id> <end_id> <dat_dir> [ns_model: lin|qd|ad] [n_trials]")

start_id <- as.integer(args[1])
end_id   <- as.integer(args[2])
dat_dir  <- args[3]
ns_model <- if (length(args) >= 4) args[4] else "lin"
if (!ns_model %in% c("lin", "qd", "ad")) stop("ns_model must be one of lin, qd, ad")
parent   <- if (length(args) >= 6) args[6] else "ln"   # data-generating truth
if (!parent %in% c("st", "ln", "qd")) stop("parent must be one of st, ln, qd")

# --- Load libraries ---
suppressMessages(suppressWarnings({
  library(gamlss)
  library(gamlss.dist)
  library(boot)
}))

# --- Parameters ---
set.seed(456 + start_id)
n_years    <- 120
# n_trials = number of Monte Carlo realizations per cell (the "MC ensemble
# count"); optional 5th arg, default 300. The 062 PBS script encodes it in its
# file name (nmcNNN) so the MC count is explicit.
n_trials   <- if (length(args) >= 5) as.integer(args[5]) else 300L
boot_R     <- 1000       # bootstrap replicates for SD CI
year_start <- 1981
e_fixed    <- 3          # nonstationary ensemble count (fixed)
p_qpos     <- 0.99       # Q100
DELTA_AIC  <- 2.0        # ns_model="ad": QD chosen iff aic_qd < aic_lin - DELTA_AIC
ctrl       <- gamlss.control(n.cyc = 50, trace = FALSE)

# Equivalent-ensemble sweep range and the SD-based narrowing (see find_eq).
# qd|ad have a higher e=3 SD than lin, so the stationary crossing can fall below
# 3 (down to ~1 in high-variance quadratic cells, e.g. vertex-inside-record);
# lower the floor for them. e<1 is meaningless (a single 120-yr draw), so 1.0 is
# the natural minimum.
E_MIN_SWEEP <- if (ns_model == "lin") 3.0 else 1.0
E_MAX_SWEEP <- 10.0
E_STEP      <- 0.1
PRED_MARGIN <- 1.5       # +/- ensembles searched around the predicted crossing

# Flood-relevance filter (matches 095/097/098): restrict the expensive MC to
# flood-relevant cells only. Default ON (env FLOOD_ONLY=0 to process all land
# cells). Non-flood cells are skipped before the sweep, ~halving runtime and
# matching the cells used in the manuscript maps. q90/max from amax_all.bin,
# uparea from the MERIT uparea grid (m2 -> km2).
FLOOD_ONLY   <- Sys.getenv("FLOOD_ONLY", "1") != "0"
FLOOD_Q90    <- 50.0
FLOOD_MAX    <- 100.0
FLOOD_UPAREA <- 50.0
NX <- 3600L; NY <- 1800L
uparea_bin   <- Sys.getenv("UPAREA_PATH",
                           "/home/kk/jp_claude/gamlss/data/uparea.bin")

# --- Paths (per parent x ns_model) ---
cell_csv    <- file.path(dat_dir, "010", "land_cells.csv")
amax_bin    <- file.path(dat_dir, "030", "amax_all.bin")
truth_dir   <- file.path(dat_dir, "064", parent, "mc_truth")
base060     <- file.path(dat_dir, "060", parent, ns_model)
eq_dir      <- file.path(base060, "eq_ens")
fail_dir    <- file.path(base060, "fail_log")

dir.create(eq_dir, showWarnings = FALSE, recursive = TRUE)
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

cat(sprintf("Processing cells %d to %d (%d cells) | parent=%s ns_model=%s | flood_only=%s\n",
            start_id, end_id, n_batch, parent, ns_model, FLOOD_ONLY))

# Flood filter inputs: uparea grid (loaded once) + amax (read per cell below).
up_grid <- NULL
if (FLOOD_ONLY) {
  if (file.exists(uparea_bin)) {
    con_up  <- file(uparea_bin, "rb")
    up_grid <- readBin(con_up, what = "numeric", size = 4, n = NX * NY, endian = "little")
    close(con_up)
  } else {
    cat(sprintf("WARN: uparea not found (%s) -> flood filter uses q90/max only\n", uparea_bin))
  }
}
con_flood_amax <- if (FLOOD_ONLY) file(amax_bin, "rb") else NULL

years <- year_start:(year_start + n_years - 1)
fail_file <- file.path(fail_dir, sprintf("failed_%06d_%06d.txt", start_id, end_id))

# ====================================================================
# Load truth from 064/mc_truth.
# Batch-size agnostic: the chunk layout is read from the file names
# (truth_SSSSSS_EEEEEE.csv), so it works for any 065 BATCH_SIZE (100, 1000,
# and the partial last chunk). The matched file is cached, so consecutive
# cells from the same chunk do not re-read the CSV.
# ====================================================================
truth_index <- local({
  files <- Sys.glob(file.path(truth_dir, "truth_*.csv"))
  if (length(files) == 0) stop(sprintf("No truth files in %s (run 065 first)", truth_dir))
  parts <- t(sapply(basename(files), function(b) {
    s <- strsplit(sub("\\.csv$", "", b), "_")[[1]]
    as.integer(s[2:3])
  }))
  data.frame(start = parts[, 1], end = parts[, 2], path = files,
             stringsAsFactors = FALSE)
})
cat(sprintf("Truth chunks indexed: %d (cells %d-%d)\n",
            nrow(truth_index), min(truth_index$start), max(truth_index$end)))

.truth_cache <- new.env()
load_truth <- function(cell_id) {
  j <- which(truth_index$start <= cell_id & cell_id <= truth_index$end)
  if (length(j) == 0) return(NULL)
  csv_path <- truth_index$path[j[1]]
  if (is.null(.truth_cache$path) || .truth_cache$path != csv_path) {
    .truth_cache$df   <- read.csv(csv_path)
    .truth_cache$path <- csv_path
  }
  row <- .truth_cache$df[.truth_cache$df$cell_id == cell_id, ]
  if (nrow(row) == 0 || is.na(row$truth_q100[1])) return(NULL)
  row[1, ]
}

# --- Optional SD-based search narrowing (cheap e=1 SDs from 050 = 040 output) ---
# 040 measured, per cell, the relative-error SD of the stationary (sd_stat1) and
# of each nonstationary fit at a SINGLE ensemble (e=1). Because the sampling SD
# scales ~ 1/sqrt(e), the equivalent ensemble crosses where
#   sd_stat1/sqrt(e*) == sd_ns1/sqrt(3)  =>  e* = 3 * (sd_stat1 / sd_ns1)^2
# where sd_ns1 is the e=1 SD of the chosen ns_model:
#   lin -> lin_err_sd ; qd -> quad_err_sd ; ad -> lin_err_sd (ad defaults to LIN,
#   so the linear SD seeds the window well; guards correct the few QD-selected
#   cells). e* narrows the sweep to a +/-PRED_MARGIN window instead of the full
# range (the bulk of the cost). It is a ratio, hence scale-invariant and robust.
# Missing file/cell -> e_pred = NA and find_eq falls back to the full sweep.
sd1_path  <- file.path(dat_dir, "050", "summary_allgrid.csv")
ns_sd_col <- switch(ns_model, lin = "lin_err_sd", qd = "quad_err_sd", ad = "lin_err_sd")
sd1_tab  <- if (file.exists(sd1_path)) {
  tb <- read.csv(sd1_path)
  cat(sprintf("SD predictor: loaded %s (%d cells), ns col=%s\n", sd1_path, nrow(tb), ns_sd_col))
  data.frame(cell_id = tb$cell_id, sd_stat1 = tb$stat_err_sd, sd_ns1 = tb[[ns_sd_col]])
} else {
  cat(sprintf("SD predictor: %s not found -> full sweep per cell\n", sd1_path))
  NULL
}

predict_eq <- function(sd_stat1, sd_ns1) {
  if (is.na(sd_stat1) || is.na(sd_ns1) || sd_ns1 <= 0 || sd_stat1 <= 0) return(NA_real_)
  3 * (sd_stat1 / sd_ns1)^2
}

# ====================================================================
# Utility functions (RG family)
# ====================================================================

# Reconstruct mu_RG(t) and sigma_RG(t) from 064 GU coefficients.
# GU links (gamlss default): mu = identity, sigma = LOG. 064 stored the raw
# coefficients, so:
#   mu_neg(t) = mu0 + mu1*t          (identity)  -> mu_RG(t) = -(mu0 + mu1*t)
#   sigma(t)  = exp(sig0 + sig1*t)   (LOG link)  -> MUST exponentiate
# The earlier version used sig0 + sig1*t directly (sigma ~exp(6)x too small),
# which collapsed the generated spread and made ns_sd/eq_sd ~tens of x too low.
# Handles st/ln/qd: 064 stores raw-poly coefficients mu/sigma _0,_1,_2 with NA
# for absent terms; treat NA as 0 so mu_neg(t)=sum(c_k t^k), sigma(t)=exp(.).
reconstruct_params <- function(truth_row, years) {
  z <- function(v) if (is.na(v)) 0 else v
  mu_neg <- z(truth_row$truth_mu_0) + z(truth_row$truth_mu_1) * years +
            z(truth_row$truth_mu_2) * years^2
  sig_rg <- exp(z(truth_row$truth_sigma_0) + z(truth_row$truth_sigma_1) * years +
                z(truth_row$truth_sigma_2) * years^2)
  list(mu = -mu_neg, sigma = sig_rg)
}

# Generate weighted data frame for fractional ensemble size.
# delta: same positivity shift used for the truth (064) and in 040's synthetic
# generation; adding it here keeps the generated flows on the same scale as the
# truth Q100 so the relative error is unbiased (it cancels in eq_ens regardless).
build_weighted_df <- function(e, years, mu_year, sg_year, delta = 0) {
  k <- floor(e); f <- e - k
  draw_one <- function() qRG(runif(length(years)), mu = mu_year, sigma = sg_year) + delta

  flows_full <- if (k > 0) as.vector(sapply(seq_len(k), function(.) draw_one())) else numeric(0)
  years_full <- if (k > 0) rep(years, times = k) else integer(0)
  w_full     <- if (k > 0) rep(1, length(years_full)) else numeric(0)

  flows_frac <- if (f > 0) draw_one() else numeric(0)
  years_frac <- if (f > 0) years else integer(0)
  w_frac     <- if (f > 0) rep(f, length(years_frac)) else numeric(0)

  data.frame(year    = c(years_full, years_frac),
             outflow = c(flows_full, flows_frac),
             weight  = c(w_full, w_frac))
}

# Nonstationary Q100 (2071-2100 window mean, RG), selectable via ns_model.
# The window mean is taken over the fitted mu/sigma at the 2071-2100 data rows
# (equivalent to predicting on those years; works for poly() without newdata
# basis issues, and matches 108's Q100 convention). Each year appears the same
# number of times in the weighted ensemble, so the row-mean is the year-mean.
.q100_window <- function(fit, df) {
  late <- df$year >= 2071 & df$year <= 2100
  mean(qRG(p_qpos, mu = fitted(fit, "mu")[late], sigma = fitted(fit, "sigma")[late]))
}
ns_lin <- function(df) {
  fit <- try(gamlss(outflow ~ year, sigma.fo = ~ year, family = RG,
                    data = df, weights = df$weight, control = ctrl), silent = TRUE)
  if (inherits(fit, "try-error")) return(list(q = NA_real_, aic = NA_real_))
  list(q = .q100_window(fit, df), aic = as.numeric(AIC(fit)))
}
ns_qd <- function(df) {
  fit <- try(gamlss(outflow ~ poly(year, 2, raw = TRUE),
                    sigma.fo = ~ poly(year, 2, raw = TRUE), family = RG,
                    data = df, weights = df$weight, control = ctrl), silent = TRUE)
  if (inherits(fit, "try-error")) return(list(q = NA_real_, aic = NA_real_))
  list(q = .q100_window(fit, df), aic = as.numeric(AIC(fit)))
}
# Dispatcher: returns the chosen nonstationary window-mean Q100.
#   lin -> linear; qd -> quadratic; ad -> AIC-adaptive (QD iff it beats LIN by
#   more than DELTA_AIC, otherwise LIN -- identical rule to 108).
q100_nonstat <- function(df, p_qpos, ctrl) {
  if (ns_model == "lin") return(ns_lin(df)$q)
  if (ns_model == "qd")  return(ns_qd(df)$q)
  rl <- ns_lin(df); rq <- ns_qd(df)
  if (is.finite(rl$aic) && is.finite(rq$aic) && rq$aic < rl$aic - DELTA_AIC) rq$q else rl$q
}

# Stationary Q100 (last 30 years only, RG)
q100_stat <- function(df, p_qpos, ctrl) {
  df30 <- df[df$year >= 2071 & df$year <= 2100, , drop = FALSE]
  if (nrow(df30) < 2) return(NA_real_)
  fit <- try(gamlss(outflow ~ 1, sigma.fo = ~ 1, family = RG,
                    data = df30, weights = df30$weight, control = ctrl), silent = TRUE)
  if (inherits(fit, "try-error")) return(NA_real_)
  qRG(p_qpos, mu = fitted(fit, "mu")[1], sigma = fitted(fit, "sigma")[1])
}

# Compute SD with bootstrap 95% CI (n=300, RSE ~4%, bootstrap provides CI)
compute_sd_ci <- function(x) {
  x <- x[is.finite(x)]
  if (length(x) < 10) return(list(sd = NA_real_, lo = NA_real_, hi = NA_real_))
  b  <- boot(x, statistic = function(d, i) sd(d[i]), R = boot_R)
  ci <- boot.ci(b, type = "perc")
  list(sd = sd(x), lo = ci$percent[4], hi = ci$percent[5])
}

# Find the equivalent ensemble: lowest e in [E_MIN_SWEEP, E_MAX_SWEEP] (step
# E_STEP) where the stationary relative-error SD drops to the nonstationary (e=3)
# SD `ns_sd`. With a prediction `e_pred` the sweep is narrowed to a window around
# it; guards re-extend down/up (or fall back) whenever the crossing turns out to
# lie outside the window, so the answer matches the full sweep -- only cheaper.
# `st_eval(e)` returns list(sd, lo, hi): the per-cell stationary SD(e) with CI.
find_eq <- function(ns_sd, e_pred, st_eval) {
  full   <- seq(E_MIN_SWEEP, E_MAX_SWEEP, by = E_STEP)
  n_eval <- 0L
  done <- function(e, ci) list(eq_ens = e, sd = ci$sd, lo = ci$lo, hi = ci$hi, n_eval = n_eval)
  none <- function() list(eq_ens = NA_real_, sd = NA_real_, lo = NA_real_, hi = NA_real_, n_eval = n_eval)

  if (is.na(e_pred)) {
    grid <- full                                       # no prediction: full sweep
  } else if (e_pred >= E_MAX_SWEEP + PRED_MARGIN) {
    # Predicted beyond the cap. Confirm with ONE check at E_MAX before skipping:
    # only a genuine non-crossing at E_MAX is reported as "not reached".
    ci <- st_eval(E_MAX_SWEEP); n_eval <- n_eval + 1L
    if (is.na(ci$sd) || ci$sd > ns_sd) return(none())
    grid <- full                                       # rare: real crossing <= cap
  } else {
    lo   <- max(E_MIN_SWEEP, round(e_pred - PRED_MARGIN, 1))
    hi   <- min(E_MAX_SWEEP, round(e_pred + PRED_MARGIN, 1))
    grid <- full[full >= lo - 1e-9 & full <= hi + 1e-9]
  }

  # Sweep the (narrowed) grid low -> high, early exit at the first crossing.
  for (e in grid) {
    ci <- st_eval(e); n_eval <- n_eval + 1L
    if (!is.na(ci$sd) && ci$sd <= ns_sd) {
      # Crossed on the first (elevated) window point: the true crossing may be
      # lower -> sweep the skipped lower range and take ITS first crossing.
      if (e <= grid[1] + 1e-9 && grid[1] > E_MIN_SWEEP + 1e-9) {
        for (e2 in full[full >= E_MIN_SWEEP & full < grid[1] - 1e-9]) {
          ci2 <- st_eval(e2); n_eval <- n_eval + 1L
          if (!is.na(ci2$sd) && ci2$sd <= ns_sd) return(done(e2, ci2))
        }
      }
      return(done(e, ci))
    }
  }
  # No crossing inside the window: extend upward to E_MAX if it stopped short.
  top <- if (length(grid)) grid[length(grid)] else (E_MIN_SWEEP - E_STEP)
  if (top < E_MAX_SWEEP - 1e-9) {
    for (e in full[full > top + 1e-9]) {
      ci <- st_eval(e); n_eval <- n_eval + 1L
      if (!is.na(ci$sd) && ci$sd <= ns_sd) return(done(e, ci))
    }
  }
  none()
}

# ====================================================================
# Pre-allocate results
# ====================================================================
res_cell_id   <- integer(n_batch)
res_iy        <- integer(n_batch)
res_ix        <- integer(n_batch)
res_ns_sd     <- rep(NA_real_, n_batch)
res_ns_sd_lo  <- rep(NA_real_, n_batch)
res_ns_sd_hi  <- rep(NA_real_, n_batch)
res_eq_ens    <- rep(NA_real_, n_batch)
res_eq_sd     <- rep(NA_real_, n_batch)
res_eq_sd_lo  <- rep(NA_real_, n_batch)
res_eq_sd_hi  <- rep(NA_real_, n_batch)
res_q100_true <- rep(NA_real_, n_batch)
res_status    <- rep("SKIP", n_batch)

# ====================================================================
# Main processing loop
# ====================================================================
n_success    <- 0
n_fail       <- 0
n_skip_flood <- 0           # cells skipped by the flood-relevance filter
n_eval_total <- 0L          # stationary SD(e) evaluations (for the savings report)

for (i in seq_along(batch_ids)) {
  cid <- batch_ids[i]
  res_cell_id[i] <- cid
  iy <- cells$iy[cid + 1]; ix <- cells$ix[cid + 1]
  res_iy[i] <- iy; res_ix[i] <- ix

  # --- Flood-relevance filter: skip non-flood cells before the costly MC ---
  if (FLOOD_ONLY) {
    seek(con_flood_amax, where = as.numeric(cid) * n_years * 4, origin = "start")
    ax  <- readBin(con_flood_amax, what = "numeric", size = 4, n = n_years, endian = "little")
    q90 <- if (length(ax) == n_years) as.numeric(quantile(ax, 0.90)) else NA_real_
    mx  <- if (length(ax) == n_years) max(ax) else NA_real_
    up  <- if (!is.null(up_grid)) up_grid[iy * NX + ix + 1] / 1.0e6 else Inf
    flood_ok <- isTRUE(((q90 >= FLOOD_Q90) || (mx >= FLOOD_MAX)) && (up >= FLOOD_UPAREA))
    if (!flood_ok) { n_skip_flood <- n_skip_flood + 1; next }   # status stays SKIP
  }

  # --- Load truth from 064 ---
  truth_row <- load_truth(cid)
  if (is.null(truth_row)) {
    cat(sprintf("TRUTH_MISSING cell_id=%d\n", cid), file = fail_file, append = TRUE)
    n_fail <- n_fail + 1
    next
  }

  # --- Reconstruct mu_RG(t), sigma_RG(t) from GU coefficients ---
  params <- reconstruct_params(truth_row, years)
  mu_all    <- params$mu
  sg_all    <- params$sigma
  delta     <- truth_row$delta
  q100_true <- truth_row$truth_q100
  res_q100_true[i] <- q100_true

  # Skip cells where sigma_RG goes negative (invalid for RG distribution)
  if (any(sg_all <= 0)) {
    cat(sprintf("SIGMA_NEG cell_id=%d min_sigma=%.4g\n", cid, min(sg_all)),
        file = fail_file, append = TRUE)
    n_fail <- n_fail + 1
    next
  }

  # --- Nonstationary trials (e=3 fixed) -> the crossing target ---
  errors_ns <- numeric(n_trials)
  for (t in seq_len(n_trials)) {
    df_ns <- build_weighted_df(e_fixed, years, mu_all, sg_all, delta)
    q_est <- q100_nonstat(df_ns, p_qpos, ctrl)
    errors_ns[t] <- (q_est - q100_true) / q100_true
  }

  ns_ci <- compute_sd_ci(errors_ns)
  res_ns_sd[i]    <- ns_ci$sd
  res_ns_sd_lo[i] <- ns_ci$lo
  res_ns_sd_hi[i] <- ns_ci$hi

  # --- Stationary search for eq_ens, narrowed by the e=1 SD prediction ---
  st_eval <- function(e) {                       # per-cell stationary SD(e) + CI
    errs <- numeric(n_trials)
    for (t in seq_len(n_trials)) {
      df_st <- build_weighted_df(e, years, mu_all, sg_all, delta)
      errs[t] <- (q100_stat(df_st, p_qpos, ctrl) - q100_true) / q100_true
    }
    compute_sd_ci(errs)
  }

  # Narrow the sweep using the e=1 SD ratio (sd_ns1 column chosen per ns_model).
  e_pred <- NA_real_
  if (!is.null(sd1_tab)) {
    j <- match(cid, sd1_tab$cell_id)
    if (!is.na(j)) e_pred <- predict_eq(sd1_tab$sd_stat1[j], sd1_tab$sd_ns1[j])
  }

  eqr <- find_eq(ns_ci$sd, e_pred, st_eval)
  n_eval_total <- n_eval_total + eqr$n_eval
  if (is.na(eqr$eq_ens)) {
    res_eq_ens[i] <- E_MAX_SWEEP + 0.1           # exceeds search range / not reached
  } else {
    res_eq_ens[i]   <- eqr$eq_ens
    res_eq_sd[i]    <- eqr$sd
    res_eq_sd_lo[i] <- eqr$lo
    res_eq_sd_hi[i] <- eqr$hi
  }

  res_status[i] <- "OK"
  n_success <- n_success + 1

  if (i %% 5 == 0) {
    cat(sprintf("  [%d/%d] success=%d, fail=%d, avg st-evals/cell=%.1f\n",
                i, n_batch, n_success, n_fail,
                if (n_success > 0) n_eval_total / n_success else 0))
  }
}
if (!is.null(con_flood_amax)) close(con_flood_amax)

# A full sweep is length(seq(E_MIN_SWEEP, E_MAX_SWEEP, E_STEP)) stationary SD(e)
# evaluations per cell; report the actual average to quantify the narrowing.
full_n   <- length(seq(E_MIN_SWEEP, E_MAX_SWEEP, by = E_STEP))
avg_eval <- if (n_success > 0) n_eval_total / n_success else 0
cat(sprintf("Done: %d success, %d fail, %d skipped(non-flood). Stationary SD evals: %d total, %.1f/cell (full=%d, ~%.1fx fewer). Writing output...\n",
            n_success, n_fail, n_skip_flood, n_eval_total, avg_eval, full_n,
            if (avg_eval > 0) full_n / avg_eval else 1))

# ====================================================================
# Write CSV
# ====================================================================
out_df <- data.frame(
  cell_id    = res_cell_id,
  iy         = res_iy,
  ix         = res_ix,
  q100_true  = res_q100_true,
  ns_sd      = res_ns_sd,
  ns_sd_lo   = res_ns_sd_lo,
  ns_sd_hi   = res_ns_sd_hi,
  eq_ens     = res_eq_ens,
  eq_sd      = res_eq_sd,
  eq_sd_lo   = res_eq_sd_lo,
  eq_sd_hi   = res_eq_sd_hi,
  status     = res_status
)

out_path <- file.path(eq_dir, sprintf("eq_ens_%06d_%06d.csv", start_id, end_id))
write.csv(out_df, out_path, row.names = FALSE)
cat(sprintf("Saved: %s\n", out_path))

cat(sprintf("All done: %d success, %d fail out of %d cells\n", n_success, n_fail, n_batch))
