#!/usr/bin/env Rscript
# =====================================================================
# figS1_negative_discharge_correction.R          (run in `gam` env; needs gamlss)
#
# Visualises Text S1 ("Correction of negative synthetic discharge").
# The synthetic discharge ensemble is drawn from the fitted Gumbel (GU)
# distribution, which has UNBOUNDED support, so individual realizations can
# fall below zero (physically impossible for discharge). Instead of clipping
# individual members, a SINGLE uniform upward shift
#
#        delta = |X_min| + q1                                          (Text S1, S1)
#
# is applied to every value, raising the ensemble minimum to the first quartile
# q1 of the pooled annual-maximum distribution. This preserves shape, scale and
# temporal trend exactly, and cancels when the Q100 estimate is shifted back.
#
# This script reproduces the generation EXACTLY as in step05_monte_carlo_linear_parent.R
#   truth fit : gamlss(outflow_neg ~ year, sigma.fo = ~ year, family = GU)
#   member    : -rGU(mu_true_neg, sigma_true)          (can be negative)
#   corrected : member + delta                          (strictly positive)
# and draws the two-panel figure (raw ensemble with negatives -> shifted).
#
# Inputs : 030/amax_all.bin  (row = cell_id, 120 float32 yr each)
#          010/land_cells.csv (row order / cell count)
# Output : data/532/negative_correction_demo.png
#          data/532/negative_correction_demo_cell.txt   (chosen cell + delta)
# Usage  : Rscript figS1_negative_discharge_correction.R [dat_dir] [cell_id]
#          (cell_id optional; if omitted a suitable example is auto-selected)
# =====================================================================

suppressMessages(suppressWarnings({
  library(gamlss); library(gamlss.dist)
}))

args    <- commandArgs(trailingOnly = TRUE)
here    <- tryCatch(dirname(sub("--file=", "",
             grep("--file=", commandArgs(FALSE), value = TRUE))), error = function(e) ".")
dat_dir <- if (length(args) >= 1 && nchar(args[1]) > 0) args[1] else file.path(here, "..", "data")
force_id <- if (length(args) >= 2) as.integer(args[2]) else NA_integer_

n_years    <- 120
year_start <- 1981
years      <- year_start:(year_start + n_years - 1)
n_show     <- 40          # ensemble members drawn for the illustration
set.seed(2024)

amax_bin <- file.path(dat_dir, "030", "amax_all.bin")
cell_csv <- file.path(dat_dir, "010", "land_cells.csv")
out_dir  <- file.path(dat_dir, "532")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

n_cells <- nrow(read.csv(cell_csv))
con     <- file(amax_bin, "rb")
on.exit(close(con))

read_amax <- function(cid) {
  seek(con, where = as.numeric(cid) * n_years * 4, origin = "start")
  readBin(con, what = "numeric", size = 4, n = n_years, endian = "little")
}

# Fit the ground-truth GU exactly as 040 (Gumbel is fit to the NEGATED series
# so that upper-tail floods become the lower tail the GU family models).
fit_truth <- function(amax_values) {
  df <- data.frame(year = years, outflow_neg = -amax_values)
  m  <- gamlss(outflow_neg ~ year, sigma.fo = ~ year, family = GU,
               data = df, trace = FALSE)
  list(mu_neg = fitted(m, "mu"), sigma = fitted(m, "sigma"),
       mu1 = coef(m, "mu")[2], sig1 = coef(m, "sigma")[2])
}

# --------------------------------------------------------------------
# Select an illustrative cell: a real river with an UPWARD level+scale trend
# whose raw synthetic ensemble contains a visible fraction of negatives
# (matching the manuscript figure). Cheap prefilter, then gamlss on candidates.
# --------------------------------------------------------------------
pick_cell <- function() {
  cand <- sample.int(n_cells, 15000)          # deterministic under the seed
  best <- NULL; best_score <- Inf             # fallback: closest to target
  target <- 0.15
  for (cid in cand) {
    a <- read_amax(cid - 1L)                  # amax row is 0-based cell_id
    if (length(a) != n_years || any(!is.finite(a))) next
    if (max(a) < 800 || max(a) > 8000 || min(a) < 0) next
    sl <- coef(lm(a ~ years))[2]              # upward trend in the raw AMAX
    cv <- sd(a) / mean(a)
    if (sl <= 0 || cv < 0.55 || cv > 1.6) next
    tr <- tryCatch(fit_truth(a), error = function(e) NULL)
    if (is.null(tr)) next
    if (!(tr$sig1 > 0)) next                  # scale (spread) grows over time
    raw <- -mapply(function(mu, sig) rGU(1, mu, sig), tr$mu_neg, tr$sigma)
    frac_neg <- mean(raw < 0)
    if (frac_neg >= 0.08 && frac_neg <= 0.30) return(cid - 1L)
    sc <- abs(frac_neg - target)              # keep the closest as a fallback
    if (frac_neg > 0.02 && sc < best_score) { best_score <- sc; best <- cid - 1L }
  }
  if (is.null(best)) stop("no suitable example cell found; pass a cell_id explicitly")
  cat(sprintf("(no cell inside target band; using closest fallback, |frac_neg-%.2f|=%.3f)\n",
              target, best_score))
  best
}

cid <- if (is.na(force_id)) pick_cell() else force_id
amax_values <- read_amax(cid)
tr <- fit_truth(amax_values)
mu_true_neg <- tr$mu_neg; sigma_true <- tr$sigma

# --- delta correction, computed exactly as 040 (pooled over 100 replicates) ---
mc_for_delta <- replicate(100,
  -mapply(function(m, s) rGU(1, mu = m, sigma = s), mu_true_neg, sigma_true))
delta <- abs(min(mc_for_delta)) + as.numeric(quantile(as.vector(mc_for_delta), 0.25))

# --- ensemble shown in the figure (n_show members x 120 years) ---
raw <- sapply(seq_len(n_show), function(m)
         -mapply(function(mu, sig) rGU(1, mu = mu, sigma = sig), mu_true_neg, sigma_true))
cor_ens  <- raw + delta
frac_neg <- mean(raw < 0)

# ggplot's default qualitative hue palette, with transparency
hue <- function(n, a = 0.7) {
  h <- (seq(15, 375, length.out = n + 1))[1:n]
  grDevices::adjustcolor(grDevices::hcl(h = h, c = 100, l = 65), alpha.f = a)
}
cols <- hue(n_show)

# common y-range for BOTH panels, so the vertical delta shift is directly visible
ylim <- range(c(raw, cor_ens)); ylim[2] <- ylim[2] * 1.02

panel <- function(mat, title, show_ylab, hline0 = FALSE) {
  plot(NA, xlim = range(years), ylim = ylim, xlab = "Year",
       ylab = if (show_ylab) expression("Outflow ["*m^3*"/s]") else "",
       main = title, cex.main = 0.98, cex.lab = 1.2, las = 1, bty = "n")
  usr <- par("usr")
  if (hline0) {
    rect(usr[1], usr[3], usr[2], 0, col = grDevices::adjustcolor("red", 0.06), border = NA)
    abline(h = 0, col = "red2", lwd = 1.6, lty = 2)
  }
  for (m in seq_len(ncol(mat)))
    points(years, mat[, m], pch = 16, cex = 0.5, col = cols[m])
  box(col = "grey55")
}

png(file.path(out_dir, "negative_correction_demo.png"),
    width = 1650, height = 760, res = 140)
par(mfrow = c(1, 2), mar = c(4.3, 4.7, 3.4, 1.0))
panel(raw, sprintf("Raw synthetic ensemble (GAMLSS-GU)\n%.1f%% of draws negative (impossible)",
                   100 * frac_neg), show_ylab = TRUE, hline0 = TRUE)
panel(cor_ens, sprintf("After uniform shift  δ = |X_min| + q₁\n= %.0f m³/s ;  all draws positive",
                       delta), show_ylab = FALSE)
dev.off()

lat <- 90 - (cid %/% 3600 + 0.5) * 0.1
lon <- -180 + (cid %%  3600 + 0.5) * 0.1
writeLines(c(
  sprintf("cell_id      = %d", cid),
  sprintf("lat, lon     = %.2f, %.2f", lat, lon),
  sprintf("delta        = %.2f m3/s", delta),
  sprintf("frac_negative(raw) = %.3f", frac_neg),
  sprintf("n_show       = %d ; years = %d-%d", n_show, min(years), max(years)),
  sprintf("max AMAX     = %.1f m3/s", max(amax_values))
), file.path(out_dir, "negative_correction_demo_cell.txt"))

cat(sprintf("cell_id=%d  lat=%.2f lon=%.2f  delta=%.1f  frac_neg(raw)=%.1f%%\n",
            cid, lat, lon, delta, 100 * frac_neg))
cat(sprintf("Saved: %s\n", file.path(out_dir, "negative_correction_demo.png")))
