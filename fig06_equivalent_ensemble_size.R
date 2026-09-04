#!/usr/bin/env Rscript
#
# fig06_equivalent_ensemble_size.R
# Manuscript Figure 6: equivalent ensemble size at the Figure-2 EXAMPLE cells.
#
# Question answered: how many STATIONARY ensembles are needed for the sampling
# spread (SD of the relative Q100 error) to fall to the level achieved by just
# THREE LINEAR-NONSTATIONARY ensembles? The stationary SD declines as the
# ensemble size e grows; the linear-nonstationary (e=3) SD is a fixed reference
# band. Their crossing is the "equivalent ensemble size".
#
# SCOPE: by default this runs for ALL EIGHT Figure-2 example cells (panels b-i,
#   the same cell_ids / rivers as fig02b_example_cell_panels.R), producing the two figures
#   per river with the river name in the titles and file names, plus a printed
#   summary of e* per river. Pass a single cell_id (or "ix,iy") to run just one
#   cell (backward compatible).
#
# Self-contained: the truth (parent GU: st|ln|qd) is fitted directly from the
# observed 120-yr AMAX of the cell (as in 064), so it does not depend on 064. The MC,
# the RG generation with fractional ensembles (build_weighted_df), the
# stationary/nonstationary Q100 conventions and the bootstrap CI all mirror 060.
#
# Outputs (to <dat_dir>/067/), per Figure-2 cell (label b..i):
#   fig6_<label>_equiv_ensemble_<parent>_<ns>_ix####_iy####.png   SD vs stationary
#       ensemble size, with the linear-nonstationary (e=3) mean +/- 95% CI band
#       and the crossing (e*), titled with the river name.
#   fig6_<label>_density_<parent>_<ns>_ix####_iy####.png           relative-error
#       densities of the STATIONARY estimate at each ensemble size e=3..6
#       (coloured solid), with the NONSTATIONARY e=3 density as a black dashed line.
#
# The nonstationary reference model (ns_model) and the data-generating truth
# (parent) are both selectable (matches 060/062):
#   ns_model lin (default) : linear  mu/sigma ~ year
#            qd            : quadratic mu/sigma ~ poly(year, 2)
#            ad            : AIC-adaptive (QD iff aic_qd < aic_lin - DELTA_AIC, else LIN)
#   parent   st | ln(default) | qd : the GU truth fitted to the cell's AMAX.
#
# Usage:
#   Rscript fig06_equivalent_ensemble_size.R [dat_dir] [cell_id] [e_max] [n_trials] [ns_model] [parent]
#     defaults: ../data  <all 8 fig-2 cells>  8.0  500  lin  ln
#     cell_id : omit (or pass "fig2") to run all eight Figure-2 cells; pass an
#               integer cell_id or "ix,iy" to run a single cell.

args     <- commandArgs(trailingOnly = TRUE)
lbl <- ""
if ("--label" %in% args) { .i <- which(args == "--label"); lbl <- args[.i + 1]; args <- args[-c(.i, .i + 1)] }
dat_dir  <- if (length(args) >= 1) args[1] else "../data"
.cid_raw <- if (length(args) >= 2) args[2] else "fig2"      # integer / "ix,iy" / "fig2" (=all 8)
e_max    <- if (length(args) >= 3) as.numeric(args[3]) else 8.0
n_trials <- if (length(args) >= 4) as.integer(args[4]) else 500L
ns_model <- if (length(args) >= 5) args[5] else "lin"
if (!ns_model %in% c("lin", "qd", "ad")) stop("ns_model must be one of lin, qd, ad")
parent   <- if (length(args) >= 6) args[6] else "ln"
if (!parent %in% c("st", "ln", "qd")) stop("parent must be one of st, ln, qd")
NS_LABEL  <- c(lin = "linear", qd = "quadratic", ad = "adaptive")[[ns_model]]
P_LABEL   <- c(st = "stationary", ln = "linear", qd = "quadratic")[[parent]]
DELTA_AIC <- 2.0          # ns_model="ad": QD chosen iff aic_qd < aic_lin - DELTA_AIC

suppressMessages(suppressWarnings({
  library(gamlss)
  library(gamlss.dist)
  library(boot)
}))

ltag <- if (nzchar(lbl)) paste0("_", lbl) else ""
n_years    <- 120
year_start <- 1981
years      <- year_start:(year_start + n_years - 1)
eval_lo    <- 2071
eval_hi    <- 2100
e_fixed    <- 3            # nonstationary ensemble count (fixed reference)
p_qpos     <- 0.99         # Q100
boot_R     <- 1000
e_grid     <- seq(1.0, e_max, by = 0.1)         # stationary sweep (curve), e=1..e_max
density_e  <- c(3, 4, 5, 6)                       # STATIONARY sizes shown in density
ctrl       <- gamlss.control(n.cyc = 50, trace = FALSE)
mtag       <- paste0("_", parent, "_", ns_model)  # parent x estimator tag

# --- Colours (match the manuscript style) ---
ST_COL   <- "#cd8b94"                              # stationary points / line (pink)
NS_LINE  <- "#3f6f54"                              # nonstationary mean (dark green)
NS_BAND  <- grDevices::adjustcolor("#a8d7bb", 0.55) # nonstationary 95% CI band
EQ_COL   <- "#21618c"                              # equivalent-ensemble marker

out_dir <- file.path(dat_dir, "067")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# land_cells (cell_id = 0-based row) for grid position + ix,iy resolution
cells <- read.csv(file.path(dat_dir, "010", "land_cells.csv"))

# ---- the eight finalised Figure-2 example cells (b-i), as fig02b_example_cell_panels.R ----
FIG2 <- data.frame(
  label   = c("b", "c", "d", "e", "f", "g", "h", "i"),
  class   = c("stationary", "location", "scale", "location and scale",
              "stationary", "location", "scale", "location and scale"),
  band    = c("low", "low", "low", "low", "high", "high", "high", "high"),
  cell_id = c(901988, 1283512, 1538746, 1547062, 1406770, 788582, 891235, 549028),
  river   = c("Nolan River", "Mosala River", "Samborombon River", "Tutaekuri River",
              "Kimazimazy River", "Yarkand River", "Ying River", "Mouchalagane River"),
  stringsAsFactors = FALSE)

# ====================================================================
# Per-cell driver: computes e* and writes the two figures for ONE cell.
# All MC/estimation code mirrors 060 exactly; only titles / file names carry the
# river name so the eight Figure-2 panels are identifiable.
# ====================================================================
run_cell <- function(cell_id, panel_label = "", river_name = "", class_label = "") {
  cell_id <- as.integer(cell_id)
  set.seed(456 + cell_id)
  iy <- cells$iy[cell_id + 1]; ix <- cells$ix[cell_id + 1]
  lat <- 90.0 - (iy + 0.5) * 0.1
  lon <- -180.0 + (ix + 0.5) * 0.1
  fname <- sprintf("ix%04d_iy%04d", ix, iy)
  celltag <- if (nzchar(panel_label)) sprintf("%s_%s", panel_label, fname) else fname

  con <- file(file.path(dat_dir, "030", "amax_all.bin"), "rb")
  seek(con, where = as.numeric(cell_id) * n_years * 4, origin = "start")
  amax_values <- readBin(con, what = "numeric", size = 4, n = n_years, endian = "little")
  close(con)
  if (length(amax_values) != n_years || max(amax_values) < 1.0)
    stop(sprintf("Cell %d (%s) has no usable AMAX.", cell_id, fname))

  hdr <- if (nzchar(river_name))
    sprintf("[%s] %s (%s)", panel_label, river_name, class_label) else fname
  cat(sprintf("\n=== %s | cell %d (%s)  lon=%.2f lat=%.2f | parent=%s ns=%s n=%d e in [1,%.1f] ===\n",
              hdr, cell_id, fname, lon, lat, parent, ns_model, n_trials, e_max))

  # --- Truth: GU on negated flow (as 040/064), parent = st|ln|qd; RG params + delta
  amax_df <- data.frame(year = years, outflow_neg = -amax_values)
  fo_mu  <- switch(parent, st = outflow_neg ~ 1, ln = outflow_neg ~ year,
                   qd = outflow_neg ~ poly(year, 2, raw = TRUE))
  fo_sig <- switch(parent, st = ~ 1, ln = ~ year, qd = ~ poly(year, 2, raw = TRUE))
  truth <- gamlss(fo_mu, sigma.fo = fo_sig, family = GU, data = amax_df, control = ctrl)
  mu_neg <- fitted(truth, "mu")
  sigma  <- fitted(truth, "sigma")
  if (any(sigma <= 0)) stop("Fitted sigma(t) <= 0; cannot generate RG samples.")
  mu_rg <- -mu_neg

  mc_for_delta <- replicate(100, -mapply(function(m, s) rGU(1, mu = m, sigma = s),
                                          mu_neg, sigma))
  delta <- abs(min(mc_for_delta)) + as.numeric(quantile(as.vector(mc_for_delta), 0.25))

  late <- years >= eval_lo & years <= eval_hi
  q100_true <- mean(qRG(p_qpos, mu = mu_rg[late], sigma = sigma[late])) + delta

  # --- MC utilities (mirror 060); closed over this cell's mu_rg/sigma/delta -----
  build_weighted_df <- function(e) {
    k <- floor(e); f <- e - k
    draw_one <- function() qRG(runif(n_years), mu = mu_rg, sigma = sigma) + delta
    flows_full <- if (k > 0) as.vector(sapply(seq_len(k), function(.) draw_one())) else numeric(0)
    years_full <- if (k > 0) rep(years, times = k) else integer(0)
    w_full     <- if (k > 0) rep(1, length(years_full)) else numeric(0)
    flows_frac <- if (f > 0) draw_one() else numeric(0)
    years_frac <- if (f > 0) years else integer(0)
    w_frac     <- if (f > 0) rep(f, length(years_frac)) else numeric(0)
    data.frame(year = c(years_full, years_frac),
               outflow = c(flows_full, flows_frac),
               weight = c(w_full, w_frac))
  }
  .q100_window <- function(fit, df) {
    lt <- df$year >= eval_lo & df$year <= eval_hi
    mean(qRG(p_qpos, mu = fitted(fit, "mu")[lt], sigma = fitted(fit, "sigma")[lt]))
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
  q100_nonstat <- function(df) {
    if (ns_model == "lin") return(ns_lin(df)$q)
    if (ns_model == "qd")  return(ns_qd(df)$q)
    rl <- ns_lin(df); rq <- ns_qd(df)
    if (is.finite(rl$aic) && is.finite(rq$aic) && rq$aic < rl$aic - DELTA_AIC) rq$q else rl$q
  }
  q100_stat <- function(df) {
    d <- df[df$year >= eval_lo & df$year <= eval_hi, , drop = FALSE]
    if (nrow(d) < 2) return(NA_real_)
    fit <- try(gamlss(outflow ~ 1, sigma.fo = ~ 1, family = RG,
                      data = d, weights = d$weight, control = ctrl), silent = TRUE)
    if (inherits(fit, "try-error")) return(NA_real_)
    qRG(p_qpos, mu = fitted(fit, "mu")[1], sigma = fitted(fit, "sigma")[1])
  }
  rel_errors <- function(e, fitfun) {
    err <- numeric(n_trials)
    for (t in seq_len(n_trials)) {
      q_est <- fitfun(build_weighted_df(e))
      err[t] <- (q_est - q100_true) / q100_true
    }
    err[is.finite(err)]
  }
  sd_ci <- function(x) {
    if (length(x) < 10) return(c(sd = NA, lo = NA, hi = NA))
    b  <- boot(x, statistic = function(d, i) sd(d[i]), R = boot_R)
    ci <- boot.ci(b, type = "perc")
    c(sd = sd(x), lo = ci$percent[4], hi = ci$percent[5])
  }

  # --- Nonstationary reference (e = 3) and stationary sweep --------------------
  cat("  Nonstationary (e=3) reference...\n")
  err_ns <- rel_errors(e_fixed, q100_nonstat)
  ns <- sd_ci(err_ns)

  cat(sprintf("  Stationary sweep over %d ensemble sizes...\n", length(e_grid)))
  st_sd <- st_lo <- st_hi <- rep(NA_real_, length(e_grid))
  dens_stat <- list()
  for (j in seq_along(e_grid)) {
    e   <- e_grid[j]
    err <- rel_errors(e, q100_stat)
    s   <- sd_ci(err)
    st_sd[j] <- s["sd"]; st_lo[j] <- s["lo"]; st_hi[j] <- s["hi"]
    if (any(abs(e - density_e) < 1e-9))
      dens_stat[[as.character(round(e))]] <- err
    if (j %% 10 == 0) cat(sprintf("    e=%.1f  SD=%.4f\n", e, s["sd"]))
  }
  for (e in density_e[density_e > e_max]) {
    cat(sprintf("  Extra stationary density ensemble size e=%d...\n", e))
    dens_stat[[as.character(e)]] <- rel_errors(e, q100_stat)
  }

  # Equivalent ensemble size: first e where stationary SD <= nonstationary SD.
  eq_ens <- NA_real_
  below <- which(st_sd <= ns["sd"])
  if (length(below) > 0) {
    j <- below[1]
    if (j == 1) {
      eq_ens <- e_grid[1]
    } else {
      x1 <- e_grid[j - 1]; y1 <- st_sd[j - 1]
      x2 <- e_grid[j];     y2 <- st_sd[j]
      eq_ens <- x1 + (ns["sd"] - y1) * (x2 - x1) / (y2 - y1)
    }
  }
  cat(sprintf("  Nonstationary (e=3) SD = %.4f [%.4f, %.4f]\n", ns["sd"], ns["lo"], ns["hi"]))
  cat(sprintf("  Equivalent stationary ensemble size = %s\n",
              ifelse(is.na(eq_ens), sprintf("> %.1f (not reached)", e_max),
                     sprintf("%.2f", eq_ens))))

  # figure title (river name so each Figure-2 panel is identifiable)
  main_txt <- if (nzchar(river_name))
    sprintf("(%s) %s  —  %s", panel_label, river_name, class_label) else ""

  # Bare figures: BIG axis ticks, NO axis names, NO in-figure legend (the legends
  # are written to their own standalone files below).
  AX_FS <- 2.0                     # tick-label size (cex.axis)

  # --- Figure 6 (main): SD vs stationary ensemble size ------------------------
  png6 <- file.path(out_dir, sprintf("fig6%s_equiv_ensemble%s_%s.png", ltag, mtag, celltag))
  png(png6, width = 1500, height = 950, res = 160)
  par(mar = c(4.4, 5.2, 2.8, 1.2))
  ylim <- range(c(st_lo, st_hi, ns["lo"], ns["hi"]), na.rm = TRUE)
  plot(e_grid, st_sd, type = "n", xlim = range(e_grid), ylim = ylim,
       xlab = "", ylab = "", main = main_txt, cex.axis = AX_FS, cex.main = 1.35)
  rect(par("usr")[1], ns["lo"], par("usr")[2], ns["hi"], col = NS_BAND, border = NA)
  abline(h = ns["sd"], col = NS_LINE, lwd = 2.5, lty = 2)
  # NOTE: the equivalent-ensemble marker (e*) is intentionally NOT drawn on the
  # figure and NOT put in the legend (per request); e* stays in the printout/CSV.
  arrows(e_grid, st_lo, e_grid, st_hi, length = 0.03, angle = 90, code = 3,
         col = ST_COL, lwd = 1.5)
  lines(e_grid, st_sd, col = ST_COL, lwd = 1.5)
  points(e_grid, st_sd, pch = 16, col = ST_COL, cex = 1.0)
  dev.off()
  cat(sprintf("  Saved: %s\n", png6))

  # standalone legend for the SD figure -- VERTICAL (one column), compact, no e*
  leg6 <- file.path(out_dir, sprintf("fig6%s_equiv_ensemble%s_legend.png", ltag, mtag))
  png(leg6, width = 1000, height = 240, res = 160)
  par(mar = c(0, 0, 0, 0)); plot.new()
  legend("center", bty = "n", cex = 1.2, ncol = 1, y.intersp = 1.5, seg.len = 2.4,
         legend = c("Nonstationary mean ± 95% CI (3 ensembles)", "Stationary mean ± 95% CI"),
         col = c(NS_LINE, ST_COL), lty = c(2, 1), pch = c(NA, 16), lwd = c(2.5, 1.5))
  dev.off()
  cat(sprintf("  Saved: %s\n", leg6))

  # --- Figure 6 companion: relative-error densities by ensemble size ----------
  pngd <- file.path(out_dir, sprintf("fig6%s_density%s_%s.png", ltag, mtag, celltag))
  png(pngd, width = 1400, height = 950, res = 160)
  par(mar = c(4.4, 5.2, 2.8, 1.2))
  sizes <- sort(density_e)
  n_s   <- length(sizes)
  pal <- grDevices::hcl(h = seq(15, 375, length.out = n_s + 1)[seq_len(n_s)],
                        c = 60, l = 78)
  d_st <- lapply(sizes, function(e) {
    x <- dens_stat[[as.character(e)]]; if (is.null(x) || length(x) < 5) NULL else density(x)
  })
  d_ns3 <- if (length(err_ns) < 5) NULL else density(err_ns)
  allx <- unlist(lapply(c(d_st, list(d_ns3)), function(d) if (!is.null(d)) d$x))
  ally <- unlist(lapply(c(d_st, list(d_ns3)), function(d) if (!is.null(d)) d$y))
  xr <- range(allx, na.rm = TRUE); yr <- range(ally, na.rm = TRUE)
  plot(NA, xlim = xr, ylim = c(0, yr[2] * 1.05),
       xlab = "", ylab = "", main = main_txt, cex.axis = AX_FS, cex.main = 1.35)
  abline(v = 0, col = "grey50", lty = 3)
  for (k in seq_len(n_s)) if (!is.null(d_st[[k]]))
    lines(d_st[[k]], col = pal[k], lwd = 2.6, lty = 1)
  if (!is.null(d_ns3))
    lines(d_ns3, col = "black", lwd = 2.4, lty = 2)
  dev.off()
  cat(sprintf("  Saved: %s\n", pngd))

  # standalone legend for the density figure -- VERTICAL: the stationary-size
  # colour group stacked on top, the nonstationary dashed entry below it (as the
  # attached design), so the legend is portrait, not wide.
  legd <- file.path(out_dir, sprintf("fig6%s_density%s_legend.png", ltag, mtag))
  png(legd, width = 820, height = 520, res = 160)
  par(mar = c(0, 0, 0, 0)); plot.new()
  legend(0.02, 0.95, xjust = 0, yjust = 1, bty = "n", cex = 1.35, seg.len = 2.0,
         title = "Stationary ensemble size",
         legend = sprintf("e = %d", sizes), col = pal, lwd = 2.8, lty = 1)
  legend(0.02, 0.40, xjust = 0, yjust = 1, bty = "n", cex = 1.35, seg.len = 2.0,
         legend = sprintf("Nonstationary %s (e=3)", NS_LABEL), col = "black",
         lwd = 2.6, lty = 2)
  dev.off()
  cat(sprintf("  Saved: %s\n", legd))

  data.frame(label = panel_label, river = river_name, class = class_label,
             cell_id = cell_id, ix = ix, iy = iy, lon = lon, lat = lat,
             ns_sd = as.numeric(ns["sd"]), eq_ens = eq_ens, stringsAsFactors = FALSE)
}

# ====================================================================
# Driver: all eight Figure-2 cells (default) or a single requested cell
# ====================================================================
if (identical(tolower(.cid_raw), "fig2")) {
  cat(sprintf("Figure-6 equivalent ensemble for the 8 Figure-2 cells | parent=%s ns=%s n_trials=%d\n",
              parent, ns_model, n_trials))
  res <- do.call(rbind, lapply(seq_len(nrow(FIG2)), function(k)
    run_cell(FIG2$cell_id[k], FIG2$label[k], FIG2$river[k], FIG2$class[k])))
  sum_csv <- file.path(out_dir, sprintf("fig6%s_eqens_summary%s.csv", ltag, mtag))
  write.csv(res, sum_csv, row.names = FALSE)
  cat("\n================ Equivalent ensemble size per Figure-2 river ================\n")
  for (k in seq_len(nrow(res)))
    cat(sprintf("  (%s) %-20s %-20s e* = %s\n", res$label[k], res$river[k], res$class[k],
                ifelse(is.na(res$eq_ens[k]), sprintf("> %.1f", e_max), sprintf("%.2f", res$eq_ens[k]))))
  cat(sprintf("Saved: %s\n", sum_csv))
} else {
  # single cell: integer cell_id OR "ix,iy" looked up in land_cells
  if (grepl(",", .cid_raw)) {
    .p <- as.integer(strsplit(.cid_raw, ",")[[1]])
    .j <- which(cells$ix == .p[1] & cells$iy == .p[2])
    if (!length(.j)) stop(sprintf("ix,iy = %d,%d not found in land_cells", .p[1], .p[2]))
    cid <- as.integer(.j[1] - 1L)
  } else cid <- as.integer(.cid_raw)
  # attach the Figure-2 label/river/class if this cell is one of the eight
  .m <- match(cid, FIG2$cell_id)
  if (!is.na(.m)) run_cell(cid, FIG2$label[.m], FIG2$river[.m], FIG2$class[.m])
  else run_cell(cid)
}
