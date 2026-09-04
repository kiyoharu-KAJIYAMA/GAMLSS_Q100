#!/usr/bin/env Rscript
#
# fig03_monte_carlo_samborombon.R
# Manuscript Figure 3 for the 20260327_naam pipeline.
#
# Produces an INDEPENDENT figure for EACH of the 8 example cells selected by
# 048/049 (048/fig1_all8_cells.csv). For each cell and each of three parent
# (truth) distributions {stationary, linear, quadratic}, the parent is fitted to
# the cell's observed 120-yr AMAX, n_mc synthetic 120-yr realizations are drawn
# from it, and each realization is re-estimated with the three fitted models
# (stationary-30yr, linear-120yr, quadratic-120yr). The 2071-2100 mean Q100 of
# every estimate is collected; the LEFT panel shows the synthetic ensemble cloud
# with the parent's true Q100(t), the RIGHT panel the density of the three
# fitted-model Q100 estimates (lines only; SD shown in the legend).
#
# Conventions match 040/045/048: GU on negated flow, eval window 2071-2100,
# stationary fitted to the window only, nonstationary fitted to all 120 yr.
#
# Usage:
#   Rscript fig03_monte_carlo_samborombon.R [dat_dir] [n_mc=3000] [cell_id]
#     - no cell_id : one figure per cell in 048/fig1_all8_cells.csv (all 8)
#     - cell_id    : only that cell
#   Results cached per cell to 056/cache_<fname>_nmc<N>.rds (re-runs resume).

# ---------------- args ----------------
args        <- commandArgs(trailingOnly = TRUE)
dat_dir     <- if (length(args) >= 1) args[1] else "../data"
n_mc        <- if (length(args) >= 2) as.integer(args[2]) else 3000L
# 3rd arg: "fig2" (default) = all Figure-2 b-i cells from 115/example_cells.csv;
#          "all" = the 048 eight-cell set; or a single cell_id.
.cid_raw    <- if (length(args) >= 3) args[3] else "fig2"

suppressMessages(suppressWarnings({
  library(gamlss)
  library(gamlss.dist)
}))

n_years <- 120
years   <- 1981:2100
eval_lo <- 2071
eval_hi <- 2100
late    <- years >= eval_lo & years <= eval_hi
n_show  <- min(300L, n_mc)   # ensemble realizations drawn in the left panel

amax_bin  <- file.path(dat_dir, "030", "amax_all.bin")
cells_csv <- file.path(dat_dir, "048", "fig1_all8_cells.csv")
out_dir   <- file.path(dat_dir, "056")
# amax is always needed; the 048 cell list is only used by the "all" / single-cell
# paths (the default "fig2" path builds its cells from 115/example_cells.csv), so
# its existence is checked lazily by read.csv(cells_csv) in those branches only.
if (!file.exists(amax_bin)) stop(sprintf("Not found: %s", amax_bin))
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# ---------------- cells to process ----------------
# Build a sel row-set for the Figure-2 cells (115/example_cells.csv): ix/iy from
# land_cells, type/climate from the b-i label/class/band.
build_fig2_sel <- function() {
  ex_csv <- file.path(dat_dir, "115", "example_cells.csv")
  if (!file.exists(ex_csv)) stop(sprintf("Not found: %s (run 115 first)", ex_csv))
  ex <- read.csv(ex_csv, stringsAsFactors = FALSE)
  lc <- read.csv(file.path(dat_dir, "010", "land_cells.csv"))
  data.frame(cell_id = ex$cell_id,
             ix = as.integer(lc$ix[ex$cell_id + 1]),
             iy = as.integer(lc$iy[ex$cell_id + 1]),
             type = sprintf("%s(%s)", ex$label, ex$class),
             climate = ex$band, stringsAsFactors = FALSE)
}
if (identical(tolower(.cid_raw), "fig2")) {
  sel <- build_fig2_sel()                                   # all Figure-2 b-i cells
} else if (identical(tolower(.cid_raw), "all")) {
  sel <- read.csv(cells_csv, stringsAsFactors = FALSE)      # 048 eight-cell set
} else {
  cell_id_arg <- as.integer(.cid_raw)                       # single cell_id
  sel <- read.csv(cells_csv, stringsAsFactors = FALSE)
  hit <- sel[sel$cell_id == cell_id_arg, , drop = FALSE]
  if (nrow(hit) == 0) {                                     # not in 048 -> derive from land_cells (+115)
    lc <- read.csv(file.path(dat_dir, "010", "land_cells.csv"))
    if (cell_id_arg + 1 > nrow(lc)) stop(sprintf("cell_id %d out of range in land_cells", cell_id_arg))
    ixv <- as.integer(lc$ix[cell_id_arg + 1]); iyv <- as.integer(lc$iy[cell_id_arg + 1])
    ty <- "fig2"; cl <- ""
    ex_csv <- file.path(dat_dir, "115", "example_cells.csv")
    if (file.exists(ex_csv)) {
      ex <- read.csv(ex_csv, stringsAsFactors = FALSE); r <- ex[ex$cell_id == cell_id_arg, , drop = FALSE]
      if (nrow(r)) { ty <- sprintf("%s(%s)", r$label[1], r$class[1]); cl <- r$band[1] }
    }
    hit <- data.frame(cell_id = cell_id_arg, ix = ixv, iy = iyv, type = ty, climate = cl,
                      stringsAsFactors = FALSE)
    cat(sprintf("cell_id %d not in 048 list -> derived ix=%d iy=%d (%s/%s)\n",
                cell_id_arg, ixv, iyv, ty, cl))
  }
  sel <- hit
}
cat(sprintf("Cells to process: %d | n_mc=%d\n", nrow(sel), n_mc))

# ---------------- parents, fitted models, helpers (cell-independent) ----------
p_gu       <- 0.01           # 1 - 0.99  -> Q100
PARENT_COL <- "#ffc107"      # all parents drawn in the same yellow
FIT_COL    <- c(stationary = "#ffc2c7", linear = "#a8d7bb", quadratic = "#a4a1ff")
CEX_AX     <- 1.8            # tick-label font size
CEX_LG     <- 1.6            # legend font size

PARENTS <- list(
  stationary = list(mu = outflow_neg ~ 1,                         sg = ~ 1),
  linear     = list(mu = outflow_neg ~ year,                      sg = ~ year),
  quadratic  = list(mu = outflow_neg ~ poly(year, 2, raw = TRUE), sg = ~ poly(year, 2, raw = TRUE))
)

build_parent <- function(spec, amax_df) {
  m <- gamlss(spec$mu, sigma.fo = spec$sg, family = GU, data = amax_df, trace = FALSE)
  mu_neg <- fitted(m, "mu"); sg <- fitted(m, "sigma")
  list(mu_neg = mu_neg, sigma = sg, q100 = -qGU(p_gu, mu = mu_neg, sigma = sg))
}
fit_st30 <- function(df) {
  d <- df[late, , drop = FALSE]
  m <- gamlss(outflow_neg ~ 1, sigma.fo = ~ 1, family = "GU", data = d, trace = FALSE)
  mean(-qGU(p_gu, mu = fitted(m, "mu"), sigma = fitted(m, "sigma")))
}
fit_lin <- function(df) {
  m <- gamlss(outflow_neg ~ year, sigma.fo = ~ year, family = "GU", data = df, trace = FALSE)
  mean((-qGU(p_gu, mu = fitted(m, "mu"), sigma = fitted(m, "sigma")))[late])
}
fit_qd <- function(df) {
  m <- gamlss(outflow_neg ~ poly(year, 2, raw = TRUE),
              sigma.fo = ~ poly(year, 2, raw = TRUE), family = "GU", data = df, trace = FALSE)
  mean((-qGU(p_gu, mu = fitted(m, "mu"), sigma = fitted(m, "sigma")))[late])
}
dens_or_null <- function(v, lo, hi) {
  v <- v[is.finite(v)]
  if (length(v) >= 2 && max(v) > min(v)) density(v, from = lo, to = hi, n = 512) else NULL
}

# ---------------- per-cell loop ----------------
con <- file(amax_bin, "rb")

for (ri in seq_len(nrow(sel))) {
  row <- sel[ri, ]
  cid <- as.integer(row$cell_id)
  ix  <- as.integer(row$ix); iy <- as.integer(row$iy)
  fname <- sprintf("ix%04d_iy%04d", ix, iy)
  cat(sprintf("[%d/%d] %s | cell_id=%d | %s/%s\n",
              ri, nrow(sel), fname, cid, row$type, row$climate))

  # --- read this cell's AMAX (120 values) ---
  seek(con, where = as.numeric(cid) * n_years * 4, origin = "start")
  amax_values <- readBin(con, what = "numeric", size = 4, n = n_years, endian = "little")
  if (length(amax_values) != n_years) { cat("  AMAX read failed, skip\n"); next }
  amax <- data.frame(year = years, outflow = amax_values, outflow_neg = -amax_values)

  # --- generate or load cached MC results ---
  cache_path <- file.path(out_dir, sprintf("cache_%s_nmc%d.rds", fname, n_mc))
  if (file.exists(cache_path)) {
    RES <- readRDS(cache_path)
  } else {
    set.seed(123)
    RES <- list()
    for (pname in names(PARENTS)) {
      par <- build_parent(PARENTS[[pname]], amax)
      q100_true_unc <- mean(par$q100[late])

      raw_list <- vector("list", n_mc)
      for (m in seq_len(n_mc))
        raw_list[[m]] <- -mapply(function(mu, sig) rGU(1, mu = mu, sigma = sig),
                                 par$mu_neg, par$sigma)
      all_vals <- unlist(raw_list)
      delta <- abs(min(all_vals)) + as.numeric(quantile(all_vals, 0.25, type = 7))
      q100_true <- q100_true_unc + delta

      q_st <- q_ln <- q_qd <- rep(NA_real_, n_mc)
      sim_show <- matrix(NA_real_, nrow = n_years, ncol = n_show)
      for (m in seq_len(n_mc)) {
        s_shift <- raw_list[[m]] + delta
        if (m <= n_show) sim_show[, m] <- s_shift
        df_mc <- data.frame(year = years, outflow_neg = -s_shift)
        q_st[m] <- tryCatch(fit_st30(df_mc), error = function(e) NA_real_)
        q_ln[m] <- tryCatch(fit_lin(df_mc),  error = function(e) NA_real_)
        q_qd[m] <- tryCatch(fit_qd(df_mc),   error = function(e) NA_real_)
      }
      RES[[pname]] <- list(delta = delta, q100_true = q100_true,
                           q100_curve_shift = par$q100 + delta, sim_show = sim_show,
                           Q = list(stationary = q_st, linear = q_ln, quadratic = q_qd))
      cat(sprintf("    %s parent done\n", pname))
    }
    saveRDS(RES, cache_path)
  }

  # --- common y-range across the three rows ---
  # Everything is shown on the raw, pre-correction (no-delta) physical scale:
  # simulated discharge, synthetic ensembles, true Q100(t) and the fitted-Q100
  # densities. The discharge axis is clipped at 0 (the raw synthetic cloud has a
  # small unphysical negative tail that is hidden).
  y_all <- numeric(0)
  for (pname in names(RES)) {
    r <- RES[[pname]]
    y_all <- c(y_all, amax$outflow, as.vector(r$sim_show) - r$delta,
               r$q100_curve_shift - r$delta, unlist(r$Q) - r$delta, r$q100_true - r$delta)
  }
  ylim <- range(y_all, finite = TRUE)
  ylim[1] <- 0                             # clip the raw cloud's negative tail
  ylim[2] <- ylim[2] + 0.05 * diff(ylim)   # top headroom so the highest point is not clipped
  y_ticks <- pretty(ylim, n = 6)

  # --- plot: 3 rows (parents) x [scatter | density] ---
  png_path <- file.path(out_dir, sprintf("056_figure3_%s.png", fname))
  png(png_path, width = 1700, height = 1600, res = 150)
  layout(matrix(1:6, nrow = 3, byrow = TRUE), widths = c(2.2, 1.0))
  par(cex.axis = CEX_AX)
  ens_grey <- grDevices::adjustcolor("grey85", alpha.f = 0.8)

  for (pname in names(RES)) {
    r <- RES[[pname]]
    years_vec <- rep(years, times = ncol(r$sim_show))

    ## left: synthetic ensembles (delta-corrected) + simulated discharge (raw) + true Q100(t)
    par(mar = c(5, 5.8, 2.6, 3))
    plot(years, amax$outflow, type = "n", xlab = "", ylab = "",
         ylim = ylim, yaxt = "n", xaxs = "i", yaxs = "i")
    axis(2, at = y_ticks, labels = format(y_ticks), las = 1, cex.axis = CEX_AX)
    abline(h = y_ticks, col = grDevices::adjustcolor("grey70", 0.2), lwd = 0.8)
    points(years_vec, as.vector(r$sim_show) - r$delta, pch = 16, cex = 0.4, col = ens_grey)  # synthetic, raw
    points(years, amax$outflow, pch = 16, col = "#21a4ff", cex = 0.7)               # simulated discharge, raw (pre-correction)
    lines(years, r$q100_curve_shift - r$delta, lwd = 4, col = PARENT_COL)             # true Q100, raw (no delta)
    points(mean(c(eval_lo, eval_hi)), r$q100_true - r$delta, pch = 16, cex = 1.6, col = PARENT_COL)
    # outline the 2071-2100 evaluation window with a black frame (no fill, so the
    # synthetic-discharge points inside stay fully visible)
    rect(eval_lo, ylim[1], eval_hi, ylim[2], col = NA, border = "black", lwd = 2)

    ## right: horizontal density of the three fitted-model Q100 estimates
    par(mar = c(5, 3.8, 2.6, 6))
    dens <- lapply(r$Q, function(v) dens_or_null(v - r$delta, ylim[1], ylim[2]))
    dmax <- max(c(0, unlist(lapply(dens, function(d) if (is.null(d)) 0 else max(d$y)))))
    if (!is.finite(dmax) || dmax <= 0) dmax <- 1e-6
    plot(0, 0, type = "n", xlab = "", ylab = "", xaxt = "n", yaxt = "n",
         xlim = c(0, dmax * 1.05), ylim = ylim, xaxs = "i", yaxs = "i")
    axis(4, at = y_ticks, labels = format(y_ticks), las = 1, cex.axis = CEX_AX)
    # density x-axis: positive ticks only, so they never collide with year 2100
    xt <- pretty(c(0, dmax), n = 3); xt <- xt[xt > 0 & xt <= dmax * 1.02]
    axis(1, at = xt, las = 1, cex.axis = CEX_AX)
    abline(h = r$q100_true - r$delta, col = grDevices::adjustcolor(PARENT_COL, 0.7), lwd = 2, lty = 2)
    # outline-only densities (no fill) so the stationary curve is never hidden;
    # drawn quadratic -> linear -> stationary so stationary sits on top.
    sd_lab <- character(0)
    for (mod in names(dens)) {
      e <- (r$Q[[mod]] - r$q100_true) / r$q100_true
      sd_lab <- c(sd_lab, sprintf("%s (SD=%.1f%%)", mod, 100 * sd(e[is.finite(e)])))
    }
    for (mod in rev(names(dens))) {
      d <- dens[[mod]]
      if (!is.null(d)) lines(d$y, d$x, lwd = 3.5, col = FIT_COL[mod])
    }
    legend("topright", inset = 0.02, bty = "n", lty = 1, lwd = 3.5,
           col = FIT_COL[names(dens)], legend = sd_lab, cex = CEX_LG)
  }
  dev.off()
  cat(sprintf("  Saved: %s\n", png_path))
}

close(con)
cat("All cells done.\n")
