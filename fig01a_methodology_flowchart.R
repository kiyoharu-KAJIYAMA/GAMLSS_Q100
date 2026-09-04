#!/usr/bin/env Rscript
#
# fig01a_methodology_flowchart.R
# Concept figures (manuscript Figure 1 + SI): plot simulated AMAX with the
# Q100(t) of FOUR fits of the observed series, per change type x climate:
#
#   1. stationary, fitted to all 120 yr        (grey dashed, 1981-2100)
#   2. stationary, fitted to 2071-2100 only    (red, window only)
#   3. nonstationary linear    (green;  Q100(t) line)
#   4. nonstationary quadratic (blue;   Q100(t) curve)
#
# Cell source: 091/change_type_cells.csv (ALL classified cells), restricted to
# the flood-relevant subset (q90 >= 50 OR max_amax >= 100) AND uparea >= 50.
# Candidates are first spread over the globe with a greedy MAXIMIN
# (farthest-point) rule on lon/lat (seeded at the highest-clarity cell), then
# RANKED by the discrepancy between the stationary 30-yr estimate and the
# nonstationary linear estimate of the window-mean Q100,
#     score = |Q100_st30 - Q100_lin| / Q100_lin,
# so the panels emphasize the st30-vs-nonstationary contrast that the paper
# quantifies (Fig. 2 / Table 1), not the st120 failure. Candidates must also
# pass the display-sanity guards in ok_display() (see below): without them the
# score simply finds the cells where the fits are broken for other reasons
# (outlier-driven tails, exploding log-link sigma extrapolation).
#
# DISPLAY-ONLY restriction: figure candidates additionally require
# q90 >= 50 m3/s (the recurrent-flood branch of the OR criterion) and < 20%
# near-zero years. Intermittent flash-flood cells stay in the ANALYSIS, but
# on such zero-inflated series the four fits diverge because of Gumbel tail
# misspecification (e.g. exploding log-link sigma), which is not the
# st30-vs-linear mechanism these figures are meant to illustrate.
#
# Modes:
#   Rscript fig01a_methodology_flowchart.R [dat_dir] [per_cat=16] [seed unused, kept for compat]
#       -> one composite PNG per category (4 columns x ceiling(per_cat/4) rows)
#   Rscript fig01a_methodology_flowchart.R [dat_dir] all8 [type_climate=cell_id ...]
#       -> ONE 2x4 figure, one cell per category (Figure 1 candidate).
#          Default cells: hand-picked representatives (PREFERRED below);
#          override per category, e.g.  trend_high=904611
#
# Input:
#   <dat_dir>/091/change_type_cells.csv
#   <dat_dir>/030/amax_all.bin
# Output: <dat_dir>/048/fig1_<type>_<climate>.png  or  fig1_all8.png
#
# Runtime: 4 gamlss fits per cell (grid mode, per_cat=16: 8x16x4 = 512 fits,
# ~5-10 min; plus ~30 s to read the 1.3M-row cells csv).

args <- commandArgs(trailingOnly = TRUE)
script_dir <- tryCatch(dirname(sub("^--file=", "",
  grep("^--file=", commandArgs(FALSE), value = TRUE))), error = function(e) ".")
if (length(script_dir) == 0 || script_dir == "") script_dir <- "."
dat_dir <- if (length(args) >= 1) args[1] else file.path(script_dir, "..", "data")
mode <- "grid"
per_cat <- 16L
overrides <- character(0)
if (length(args) >= 2) {
  if (args[2] == "all8") {
    mode <- "all8"
    if (length(args) >= 3) overrides <- args[3:length(args)]
  } else {
    per_cat <- as.integer(args[2])
  }
}

# Hand-picked default cells for the all8 figure (ix, iy), chosen from the
# 20260612 composites: clean archetype behaviour, no quantized-flow artifacts.
PREFERRED <- list(
  stationary_low  = c(1089, 1436),  # Patagonia: flat, all four fits agree
  stationary_high = c(153,  210),   # Alaska: textbook agreement of all fits
  trend_low       = c(2068, 1236),  # South Africa: clear INCREASING trend
  trend_high      = c(2803, 576),   # E Tibet: DECREASING; st120 inflates ~2.8x
  step_low        = c(2113, 299),   # NW Russia: clear downward step
  step_high       = c(2328, 293),   # Urals: sharp step, clean fits
  variance_low    = c(2908, 973),   # Java: growing spread
  variance_high   = c(1424, 949)    # NE Brazil: growing spread + trend
)

# Flood-relevance thresholds (see 099 / Figure S1)
FLOOD_Q90 <- 50.0
FLOOD_MAX <- 100.0
FLOOD_UPAREA <- 50.0

suppressMessages(suppressWarnings({
  library(gamlss)
  library(gamlss.dist)
}))

n_years <- 120
years   <- 1981:2100
eval_lo <- 2071
eval_hi <- 2100
late    <- years >= eval_lo & years <= eval_hi

amax_bin  <- file.path(dat_dir, "030", "amax_all.bin")
cells_csv <- file.path(dat_dir, "091", "change_type_cells.csv")
out_dir   <- file.path(dat_dir, "048")
for (p in c(amax_bin, cells_csv))
  if (!file.exists(p)) stop(sprintf("Not found: %s", p))
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

cat("Reading change_type_cells.csv (~1.3M rows) ...\n")
cells <- read.csv(cells_csv, colClasses = c(
  cell_id = "integer", ix = "integer", iy = "integer",
  lon = "numeric", lat = "numeric", climate = "character",
  type = "character", mean = "numeric", clarity = "numeric",
  q90 = "numeric", max_amax = "numeric", uparea_km2 = "numeric"))
keep <- with(cells, ((!is.na(q90) & q90 >= FLOOD_Q90) |
                     (!is.na(max_amax) & max_amax >= FLOOD_MAX)) &
                    (!is.na(uparea_km2) & uparea_km2 >= FLOOD_UPAREA))
cells <- cells[keep, ]
cat(sprintf("Flood-relevant cells: %d\n", nrow(cells)))
# display-only: restrict FIGURE candidates to recurrent-flood (perennial) cells
cells <- cells[!is.na(cells$q90) & cells$q90 >= FLOOD_Q90, ]
cat(sprintf("Figure candidates (q90 >= %g): %d\n", FLOOD_Q90, nrow(cells)))

types    <- c("stationary", "trend", "step", "variance")
climates <- c("low", "high")

con <- file(amax_bin, "rb")
read_amax <- function(cid) {
  seek(con, where = as.numeric(cid) * n_years * 4, origin = "start")
  readBin(con, what = "numeric", size = 4, n = n_years, endian = "little")
}

q100_of <- function(model) {
  -qGU(0.01, mu = fitted(model, "mu"), sigma = fitted(model, "sigma"))
}

fit_cell <- function(x) {
  df  <- data.frame(year = years, outflow_neg = -x)
  dfw <- df[late, , drop = FALSE]
  fits <- list(
    st120 = tryCatch(q100_of(gamlss(outflow_neg ~ 1, sigma.fo = ~ 1, family = "GU",
                                    data = df, trace = FALSE)),
                     error = function(e) NULL),
    st30  = tryCatch(q100_of(gamlss(outflow_neg ~ 1, sigma.fo = ~ 1, family = "GU",
                                    data = dfw, trace = FALSE)),
                     error = function(e) NULL),
    ln    = tryCatch(q100_of(gamlss(outflow_neg ~ year, sigma.fo = ~ year, family = "GU",
                                    data = df, trace = FALSE)),
                     error = function(e) NULL),
    qd    = tryCatch(q100_of(gamlss(outflow_neg ~ poly(year, 2, raw = TRUE),
                                    sigma.fo = ~ poly(year, 2, raw = TRUE),
                                    family = "GU", data = df, trace = FALSE)),
                     error = function(e) NULL)
  )
  if (any(sapply(fits, is.null))) return(NULL)
  fits
}

# Greedy maximin (farthest-point) ordering on lon/lat: returns row indices,
# starting at the highest-clarity cell, each next index maximizing the minimum
# distance to those already chosen -> globally spread panels.
maximin_order <- function(lon, lat, clarity, k) {
  n <- length(lon)
  k <- min(k, n)
  sel <- which.max(clarity)
  mind <- sqrt((lon - lon[sel])^2 + (lat - lat[sel])^2)
  while (length(sel) < k) {
    cur <- which.max(mind)
    sel <- c(sel, cur)
    mind <- pmin(mind, sqrt((lon - lon[cur])^2 + (lat - lat[cur])^2))
  }
  sel
}

# colors matched to the manuscript ensemble figure:
# stationary = pink, linear = green, quadratic = purple; st120 stays grey dashed
COL <- c(st120 = "#737373", st30 = "#f06292", ln = "#66b28e", qd = "#9575cd")
LTY <- c(st120 = 2, st30 = 1, ln = 1, qd = 1)
LWD <- c(st120 = 1.8, st30 = 2.6, ln = 2.4, qd = 2.4)
LBL <- c(st120 = "stationary (120-yr)", st30 = "stationary (2071-2100)",
         ln = "nonstat. linear", qd = "nonstat. quadratic")
COL_PTS <- "#1f78b4"   # simulated AMAX: vivid blue (was pale #a9d5e6)

draw_panel <- function(x, fits, r, show_legend = FALSE, title = NULL) {
  if (is.null(title))
    title <- sprintf("ix%d iy%d  (%.2f, %.2f)", r$ix, r$iy, r$lon, r$lat)
  ylim <- range(c(x, fits$st120, fits$ln[late], fits$qd[late], fits$st30), finite = TRUE)
  plot(years, x, pch = 16, col = COL_PTS, cex = 0.85, ylim = ylim,
       xlab = "", ylab = "", cex.axis = 1.9,
       main = title, cex.main = 0.95)
  rect(eval_lo, par("usr")[3], eval_hi, par("usr")[4], col = "#88888826", border = NA)
  lines(years, fits$st120, col = COL["st120"], lwd = LWD["st120"], lty = LTY["st120"])
  lines(years[late], fits$st30, col = COL["st30"], lwd = LWD["st30"], lty = LTY["st30"])
  lines(years, fits$ln, col = COL["ln"], lwd = LWD["ln"], lty = LTY["ln"])
  lines(years, fits$qd, col = COL["qd"], lwd = LWD["qd"], lty = LTY["qd"])
  if (show_legend) {
    legend("topleft", bty = "n", cex = 0.75, legend = LBL, col = COL,
           lty = LTY, lwd = LWD)
  }
}

wm_of <- function(fits) {
  c(st120 = mean(fits$st120[late]), st30 = mean(fits$st30),
    ln = mean(fits$ln[late]), qd = mean(fits$qd[late]))
}

# Display-sanity guards: the score hunts for the largest st30-vs-linear
# discrepancy, which without guards selects exactly the cells where the fits
# are broken for OTHER reasons. A figure candidate must have:
#   (a) st30/ln/qd window means within a factor 2.5 of each other
#       (st120 is NOT constrained -- its failure under change is legitimate),
#   (b) no heavy spike tail: max(AMAX) <= 5 x its 90th percentile
#       (excludes outlier-driven 'stationary' cells),
#   (c) no exploding extrapolation: Q100_lin(t) <= 2 x max(AMAX) in the window
#       (excludes log-link sigma blow-ups on strongly growing series).
ok_display <- function(x, fits, wm) {
  if (any(!is.finite(wm)) || any(wm <= 0)) return(FALSE)
  v <- wm[c("st30", "ln", "qd")]
  if (max(v) / min(v) > 2.5) return(FALSE)
  if (max(x) > 5 * as.numeric(quantile(x, 0.9))) return(FALSE)
  if (max(fits$ln[late]) > 2 * max(x)) return(FALSE)
  TRUE
}

score_of <- function(wm) {
  abs(wm["st30"] - wm["ln"]) / wm["ln"]
}

if (mode == "all8") {
  # ---- ONE 2x4 figure: one representative cell per type x climate ----
  ov <- list()
  for (s in overrides) {
    kv <- strsplit(s, "=")[[1]]
    if (length(kv) == 2) ov[[kv[1]]] <- as.integer(kv[2])
  }
  panels <- list()
  for (cl in climates) for (t in types) {
    key <- paste(t, cl, sep = "_")
    cat_rows <- cells[cells$type == t & cells$climate == cl, , drop = FALSE]
    if (nrow(cat_rows) == 0) { cat(sprintf("WARNING: no cells for %s\n", key)); next }
    # override (if given) wins; otherwise evaluate a spread candidate pool
    # (PREFERRED + 20 maximin points) and take the cell with the LARGEST
    # st30-vs-linear discrepancy (the contrast the paper is about).
    if (!is.null(ov[[key]])) {
      hit <- cat_rows[cat_rows$cell_id == ov[[key]], , drop = FALSE]
      if (nrow(hit) == 1) {
        x <- read_amax(hit$cell_id[1])
        fits <- if (max(x) >= 1.0) fit_cell(x) else NULL
        if (!is.null(fits)) {
          wm <- wm_of(fits)
          cat(sprintf("  %s (override): cell %d  st120 %.0f, st30 %.0f, ln %.0f, qd %.0f\n",
                      key, hit$cell_id[1], wm["st120"], wm["st30"], wm["ln"], wm["qd"]))
          panels[[key]] <- list(x = x, fits = fits, r = hit[1, ])
          next
        }
        cat(sprintf("WARNING: override %s=%d failed to fit, falling back\n", key, ov[[key]]))
      } else {
        cat(sprintf("WARNING: override %s=%d not in this category\n", key, ov[[key]]))
      }
    }
    ord <- maximin_order(cat_rows$lon, cat_rows$lat, cat_rows$clarity, 20L)
    cand <- cat_rows[ord, , drop = FALSE]
    pref <- PREFERRED[[key]]
    if (!is.null(pref)) {
      hit <- cat_rows[cat_rows$ix == pref[1] & cat_rows$iy == pref[2], , drop = FALSE]
      if (nrow(hit) == 1) cand <- rbind(hit, cand)
    }
    best <- NULL
    for (i in seq_len(nrow(cand))) {
      r <- cand[i, ]
      x <- read_amax(r$cell_id)
      if (max(x) < 1.0 || mean(x < 1.0) > 0.2) next   # skip intermittent series
      fits <- fit_cell(x)
      if (is.null(fits)) next
      wm <- wm_of(fits)
      if (!ok_display(x, fits, wm)) next
      s <- score_of(wm)
      cat(sprintf("  %s: cell %d (ix%d iy%d)  st30 %.0f vs ln %.0f  score %.3f\n",
                  key, r$cell_id, r$ix, r$iy, wm["st30"], wm["ln"], s))
      if (is.null(best) || s > best$s) best <- list(x = x, fits = fits, r = r, s = s)
    }
    if (!is.null(best)) {
      panels[[key]] <- best
      cat(sprintf("  -> %s selected: cell %d (score %.3f)\n", key, best$r$cell_id, best$s))
    } else {
      cat(sprintf("WARNING: no usable cell for %s\n", key))
    }
  }
  png(file.path(out_dir, "fig1_all8.png"), width = 480 * 4, height = 400 * 2, res = 110)
  par(mfrow = c(2, 4), mar = c(3.0, 4.2, 3.0, 0.8))
  sel_rows <- list()
  k <- 0
  for (cl in climates) for (t in types) {
    k <- k + 1
    p <- panels[[paste(t, cl, sep = "_")]]
    if (is.null(p)) { plot.new(); next }
    # (a) is reserved for the locator map (049). Letters run ROW-MAJOR:
    # top row (low)  : (b) stationary (c) trend (d) step (e) variance
    # bottom row (high): (f) stationary (g) trend (h) step (i) variance
    lab <- letters[k + 1]
    draw_panel(p$x, p$fits, p$r, show_legend = FALSE,
               title = sprintf("(%s) %s / %s-flow   ix%d iy%d",
                               lab, t, cl, p$r$ix, p$r$iy))
    sel_rows[[length(sel_rows) + 1]] <- data.frame(
      label = lab, type = t, climate = cl, cell_id = p$r$cell_id,
      ix = p$r$ix, iy = p$r$iy, lon = p$r$lon, lat = p$r$lat)
  }
  dev.off()
  cat(sprintf("Saved: %s\n", file.path(out_dir, "fig1_all8.png")))
  # selected cells for the locator map (049)
  sel_csv <- file.path(out_dir, "fig1_all8_cells.csv")
  write.csv(do.call(rbind, sel_rows), sel_csv, row.names = FALSE, quote = FALSE)
  cat(sprintf("Saved: %s\n", sel_csv))
  close(con)
  quit(save = "no")
}

# ---- grid mode: one composite per category, panels spread by maximin ----
for (t in types) for (cl in climates) {
  cat_rows <- cells[cells$type == t & cells$climate == cl, , drop = FALSE]
  if (nrow(cat_rows) == 0) { cat(sprintf("no cells: %s/%s\n", t, cl)); next }
  ord <- maximin_order(cat_rows$lon, cat_rows$lat, cat_rows$clarity, 2L * per_cat)
  cand <- cat_rows[ord, , drop = FALSE]

  # fit ALL spread candidates, then keep the per_cat with the largest
  # st30-vs-linear discrepancy (drawn in descending score order)
  pool <- list()
  for (i in seq_len(nrow(cand))) {
    r <- cand[i, ]
    x <- read_amax(r$cell_id)
    if (max(x) < 1.0 || mean(x < 1.0) > 0.2) next   # skip intermittent series
    fits <- fit_cell(x)
    if (is.null(fits)) next
    wm <- wm_of(fits)
    if (!ok_display(x, fits, wm)) next
    s <- score_of(wm)
    cat(sprintf("  %s/%s cell %d (ix%d iy%d): st30 %.0f vs ln %.0f  score %.3f\n",
                t, cl, r$cell_id, r$ix, r$iy, wm["st30"], wm["ln"], s))
    pool[[length(pool) + 1]] <- list(x = x, fits = fits, r = r, s = s)
  }
  if (length(pool) > 0) {
    pool <- pool[order(-sapply(pool, function(p) p$s))]
    drawn <- head(pool, per_cat)
  } else {
    drawn <- list()
  }
  n <- length(drawn)
  if (n == 0) { cat(sprintf("all fits failed: %s/%s\n", t, cl)); next }

  ncol_ <- min(4, n); nrow_ <- ceiling(n / ncol_)
  png(file.path(out_dir, sprintf("fig1_%s_%s.png", t, cl)),
      width = 480 * ncol_, height = 380 * nrow_, res = 110)
  par(mfrow = c(nrow_, ncol_), mar = c(3.0, 4.2, 2.4, 0.8),
      oma = c(0, 0, 2.2, 0))
  for (i in seq_len(n)) {
    draw_panel(drawn[[i]]$x, drawn[[i]]$fits, drawn[[i]]$r, show_legend = FALSE)
  }
  mtext(sprintf("%s / %s-flow", t, cl), outer = TRUE, cex = 1.2, font = 2)
  dev.off()
  cat(sprintf("Saved: fig1_%s_%s.png (%d panels)\n", t, cl, n))
}
close(con)
cat(sprintf("Done. Output in %s\n", out_dir))
