#!/usr/bin/env Rscript
#
# tabS2_aic_confusion_matrix.R
# Does the AIC-adaptive estimator (LIN by default, QD only when AIC justifies it
# by > DELTA_AIC) actually help, or is the "30% QD" just noise the classifier
# misreads as curvature?  We answer with Monte Carlo, NOT assertion.
#
# For each sampled FLOOD-RELEVANT cell:
#   1. fit the three manuscript parents to the observed AMAX (GU on negated flow):
#        stationary  mu~1,          sigma~1
#        linear      mu~year,       sigma~year
#        quadratic   mu~poly(2),    sigma~poly(2)
#   2. from each parent generate n_mc synthetic 120-yr records (delta-shifted, 056)
#   3. on every record fit the same three models (all on 120 yr -> AICs comparable)
#      and record their AIC and the 2071-2100 mean Q100.
#
# Outputs (<dat_dir>/104/):
#   confusion_prop.csv / confusion_counts.csv
#       true parent (row) x AIC-selected model (col). The key cell is
#       linear-parent -> quadratic = the FALSE-POSITIVE rate: if it is ~30% the
#       observed "30% QD" is reproducible in a world with NO real curvature, so
#       AIC-adaptive buys little; if it is small, the observed QD cells are real.
#   error_stats.csv  : SD / bias / RMSE [%] for LIN-only, QD-only, AIC-adaptive,
#                      per parent. AIC-adaptive is worthwhile iff it stays at or
#                      near the minimum RMSE for EVERY parent (robustness).
#   aic_adaptive_mc.png : SD / bias / RMSE barplots + observed-vs-MC classification.
#
# AIC selection (3-way) uses the AIC+DELTA parsimony rule (upgrade only if the
# more complex model beats the simpler by > DELTA_AIC). The adaptive ESTIMATOR
# follows the proposal: QD's Q100 iff aic_qd < aic_lin - DELTA_AIC, else LIN's.
#
# Usage: Rscript tabS2_aic_confusion_matrix.R [dat_dir] [n_cells] [n_mc] [seed] [n_cores]
#   defaults: ../data  100  100  1  (detectCores()-2)
# Cells are independent, so the per-cell work is fanned out with
# parallel::mclapply (fork); reproducible via the L'Ecuyer-CMRG RNG streams.

args    <- commandArgs(trailingOnly = TRUE)
dat_dir <- if (length(args) >= 1) args[1] else "../data"
n_cells <- if (length(args) >= 2) as.integer(args[2]) else 100L
n_mc    <- if (length(args) >= 3) as.integer(args[3]) else 100L
seed    <- if (length(args) >= 4) as.integer(args[4]) else 1L
n_cores <- if (length(args) >= 5) as.integer(args[5]) else
  max(1L, parallel::detectCores() - 2L)

suppressMessages(suppressWarnings({ library(gamlss); library(gamlss.dist); library(parallel) }))

RNGkind("L'Ecuyer-CMRG")   # forkable, reproducible parallel RNG streams
set.seed(seed)
n_years <- 120
years   <- 1981:2100
late    <- years >= 2071 & years <= 2100
p_gu    <- 0.01            # 1 - 0.99 -> Q100 via -qGU
DELTA_AIC <- 2.0
min_flow  <- 1.0
ctrl    <- gamlss.control(n.cyc = 50, trace = FALSE)

amax_bin <- file.path(dat_dir, "030", "amax_all.bin")
cell_csv <- file.path(dat_dir, "010", "land_cells.csv")
out_dir  <- file.path(dat_dir, "104")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

cells <- read.csv(cell_csv)

# sample flood-relevant cells (091 if available, else any valid cell)
cand <- NULL
flood_csv <- file.path(dat_dir, "091", "change_type_cells.csv")
if (file.exists(flood_csv)) {
  ct <- read.csv(flood_csv)
  fr <- ((ct$q90 >= 50) | (ct$max_amax >= 100)) & (ct$uparea_km2 >= 50)
  fr[is.na(fr)] <- FALSE
  cand <- ct$cell_id[fr]
}
if (is.null(cand) || !length(cand)) cand <- 0:(nrow(cells) - 1)
sample_ids <- sample(cand, min(n_cells, length(cand)))
cat(sprintf("Sampled %d flood-relevant cells; n_mc=%d per parent; seed=%d\n",
            length(sample_ids), n_mc, seed))

PARENTS <- list(
  stationary = list(mu = outflow_neg ~ 1,                         sg = ~ 1),
  linear     = list(mu = outflow_neg ~ year,                      sg = ~ year),
  quadratic  = list(mu = outflow_neg ~ poly(year, 2, raw = TRUE), sg = ~ poly(year, 2, raw = TRUE))
)
PNAMES <- names(PARENTS)

# Fit one model; return its AIC and the 2071-2100 mean Q100 (NA on failure).
fit_one <- function(fo_mu, fo_sg, df) {
  m <- try(gamlss(fo_mu, sigma.fo = fo_sg, family = "GU", data = df, control = ctrl),
           silent = TRUE)
  if (inherits(m, "try-error")) return(list(aic = NA_real_, q = NA_real_))
  q <- -qGU(p_gu, mu = fitted(m, "mu"), sigma = fitted(m, "sigma"))
  list(aic = as.numeric(AIC(m)), q = mean(q[late]))
}

# 3-way AIC class (parsimony: upgrade only if beat by > DELTA_AIC).
classify3 <- function(as, al, aq) {
  if (any(is.na(c(as, al, aq)))) return(NA_character_)
  cls <- "stationary"
  if (al < as - DELTA_AIC) cls <- "linear"
  if (identical(cls, "linear") && aq < al - DELTA_AIC) cls <- "quadratic"
  cls
}

est_names <- c("LIN", "QD", "adaptive")
err_keys  <- as.vector(outer(PNAMES, est_names, paste))

# ---- per-cell worker (parallel-safe: each fork opens its own file handle) ----
process_cell <- function(cid) {
  conf_c <- matrix(0, 3, 3, dimnames = list(PNAMES, PNAMES))
  obs_c  <- setNames(integer(3), PNAMES)
  err_c  <- setNames(vector("list", length(err_keys)), err_keys)
  for (k in err_keys) err_c[[k]] <- numeric(0)

  con <- file(amax_bin, "rb"); on.exit(close(con))
  seek(con, where = as.numeric(cid) * n_years * 4, origin = "start")
  amax <- readBin(con, what = "numeric", size = 4, n = n_years, endian = "little")
  if (length(amax) != n_years || max(amax) < min_flow)
    return(list(conf = conf_c, obs = obs_c, err = err_c))
  amax_df <- data.frame(year = years, outflow_neg = -amax)

  # observed-data classification (the real "30% QD") for direct comparison
  os <- fit_one(outflow_neg ~ 1,                         ~ 1,                         amax_df)
  ol <- fit_one(outflow_neg ~ year,                      ~ year,                      amax_df)
  oq <- fit_one(outflow_neg ~ poly(year, 2, raw = TRUE), ~ poly(year, 2, raw = TRUE), amax_df)
  oc <- classify3(os$aic, ol$aic, oq$aic)
  if (!is.na(oc)) obs_c[oc] <- obs_c[oc] + 1L

  for (pn in PNAMES) {
    sp  <- PARENTS[[pn]]
    par <- try(gamlss(sp$mu, sigma.fo = sp$sg, family = GU, data = amax_df, control = ctrl),
               silent = TRUE)
    if (inherits(par, "try-error")) next
    mu_neg <- fitted(par, "mu"); sigma <- fitted(par, "sigma")
    if (any(!is.finite(sigma)) || any(sigma <= 0)) next
    q_curve <- -qGU(p_gu, mu = mu_neg, sigma = sigma)
    pre   <- replicate(50, -mapply(function(m, s) rGU(1, mu = m, sigma = s), mu_neg, sigma))
    delta <- abs(min(pre)) + as.numeric(quantile(as.vector(pre), 0.25))
    true_q100 <- mean(q_curve[late]) + delta

    for (m in seq_len(n_mc)) {
      raw   <- -mapply(function(mu, s) rGU(1, mu = mu, sigma = s), mu_neg, sigma) + delta
      df_mc <- data.frame(year = years, outflow_neg = -raw)
      fs <- fit_one(outflow_neg ~ 1,                         ~ 1,                         df_mc)
      fl <- fit_one(outflow_neg ~ year,                      ~ year,                      df_mc)
      fq <- fit_one(outflow_neg ~ poly(year, 2, raw = TRUE), ~ poly(year, 2, raw = TRUE), df_mc)
      cls <- classify3(fs$aic, fl$aic, fq$aic)
      if (is.na(cls)) next
      conf_c[pn, cls] <- conf_c[pn, cls] + 1
      q_adp <- if (!is.na(fq$aic) && !is.na(fl$aic) && fq$aic < fl$aic - DELTA_AIC) fq$q else fl$q
      re <- function(q) (q - true_q100) / true_q100
      err_c[[paste(pn, "LIN")]]      <- c(err_c[[paste(pn, "LIN")]],      re(fl$q))
      err_c[[paste(pn, "QD")]]       <- c(err_c[[paste(pn, "QD")]],       re(fq$q))
      err_c[[paste(pn, "adaptive")]] <- c(err_c[[paste(pn, "adaptive")]], re(q_adp))
    }
  }
  list(conf = conf_c, obs = obs_c, err = err_c)
}

cat(sprintf("Running %d cells on %d cores (parallel::mclapply)\n",
            length(sample_ids), n_cores))
res <- mclapply(sample_ids, process_cell, mc.cores = n_cores,
                mc.set.seed = TRUE, mc.preschedule = FALSE)
ok <- Filter(function(r) is.list(r) && !is.null(r$conf), res)
cat(sprintf("  %d/%d cells returned results\n", length(ok), length(sample_ids)))
conf      <- Reduce(`+`, lapply(ok, `[[`, "conf"))
obs_count <- Reduce(`+`, lapply(ok, `[[`, "obs"))
err <- setNames(vector("list", length(err_keys)), err_keys)
for (k in err_keys)
  err[[k]] <- unlist(lapply(ok, function(r) r$err[[k]]), use.names = FALSE)

# ---- confusion matrix ----
rs <- pmax(rowSums(conf), 1)
conf_prop <- conf / rs
write.csv(cbind(parent = rownames(conf), as.data.frame(conf)),
          file.path(out_dir, "confusion_counts.csv"), row.names = FALSE)
write.csv(cbind(parent = rownames(conf_prop), round(as.data.frame(conf_prop), 4)),
          file.path(out_dir, "confusion_prop.csv"), row.names = FALSE)
cat("\nConfusion matrix (row = TRUE parent, col = AIC-selected), proportions:\n")
print(round(conf_prop, 3))
cat(sprintf("\n>> linear-parent -> QD (FALSE POSITIVE)   = %.1f%%\n",
            100 * conf_prop["linear", "quadratic"]))
cat(sprintf(">> stationary-parent -> QD (false positive) = %.1f%%\n",
            100 * conf_prop["stationary", "quadratic"]))
cat(sprintf(">> quadratic-parent -> LIN (missed)         = %.1f%%\n",
            100 * conf_prop["quadratic", "linear"]))
obs_p <- obs_count / max(sum(obs_count), 1)
cat(sprintf("Observed-data class fractions: stat=%.1f%% lin=%.1f%% qd=%.1f%%\n",
            100 * obs_p["stationary"], 100 * obs_p["linear"], 100 * obs_p["quadratic"]))

# ---- error stats ----
rows <- data.frame()
for (p in PNAMES) for (e in est_names) {
  v <- err[[paste(p, e)]]; v <- v[is.finite(v)]
  if (!length(v)) next
  rows <- rbind(rows, data.frame(parent = p, estimator = e, n = length(v),
                SD = 100 * sd(v), bias = 100 * mean(v), RMSE = 100 * sqrt(mean(v^2))))
}
write.csv(rows, file.path(out_dir, "error_stats.csv"), row.names = FALSE)
cat("\nSD / bias / RMSE [%] by parent x estimator:\n"); print(rows, row.names = FALSE)
cat("\nMin-RMSE estimator per parent (adaptive should be best or near-best):\n")
for (p in PNAMES) {
  sub <- rows[rows$parent == p, ]
  if (nrow(sub)) cat(sprintf("  %-10s -> %s (RMSE=%.2f%%)\n",
      p, sub$estimator[which.min(sub$RMSE)], min(sub$RMSE)))
}

# ---- figure: SD / bias / RMSE barplots ----
mat <- function(metric) {
  m <- matrix(NA_real_, length(est_names), length(PNAMES),
              dimnames = list(est_names, PNAMES))
  for (p in PNAMES) for (e in est_names) {
    r <- rows[rows$parent == p & rows$estimator == e, ]
    if (nrow(r)) m[e, p] <- r[[metric]]
  }
  m
}
png(file.path(out_dir, "aic_adaptive_mc.png"), width = 1500, height = 500, res = 130)
par(mfrow = c(1, 3), mar = c(4, 4, 3, 1))
cols <- c(LIN = "#a8d7bb", QD = "#a4a1ff", adaptive = "#cd8b94")
for (metric in c("SD", "bias", "RMSE")) {
  barplot(mat(metric), beside = TRUE, col = cols, las = 1,
          ylab = sprintf("%s [%%]", metric), main = metric,
          legend.text = (metric == "SD"),
          args.legend = list(x = "topleft", bty = "n"))
}
dev.off()
cat(sprintf("Saved: %s\n", file.path(out_dir, "aic_adaptive_mc.png")))
