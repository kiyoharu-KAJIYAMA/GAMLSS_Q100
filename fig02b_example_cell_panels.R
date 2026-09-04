#!/usr/bin/env Rscript
#
# fig02b_example_cell_panels.R
# Rebuild Figure 2 from the eight finalised example cells (b-i), under the
# RENAMED GU classes: stationary / location / scale / location and scale
# (old mean->location, var->scale, both->location and scale). Produces, as
# SEPARATE pieces for manual composition:
#   1. panels_<mode>/panel_<label>_<cell>.png  -- one stripped panel per cell
#        (NO title / axis-label text / in-panel legend; numeric ticks ONLY, big
#         font, wide spacing). ST120 dashed is drawn LAST (frontmost) so it is
#         never hidden where it coincides with linear-NS.
#   2. legend_estimators.png  -- the four-estimator legend as its own image,
#        renamed (stationary 1981-2100 / 2071-2100 / linear-nonstationary /
#        quadratic-nonstationary), large font, no "lines = Q100(t)".
#   3. global_class_map.png   -- every classified land cell coloured by its GU
#        class (4 colours); the eight example cells marked and labelled b-i.
#   4. class_pie.png          -- pie of the GU-class grid-cell fractions, with
#        the percentage (2 significant figures) printed inside each wedge.
#   5. example_rivers.xlsx (or .csv) -- river / basin of each example cell.
#
# Estimators per panel (GU on negated flow), colours matched to 048:
#   ST120 grey dashed | ST30 pink | linear-NS green | quadratic-NS purple.
#
# Usage:
#   Rscript fig02b_example_cell_panels.R [dat_dir] [--modes qt,q100,both] [--osm]

args <- commandArgs(trailingOnly = TRUE)
modes <- c("qt", "q100", "both")
if ("--modes" %in% args) { i <- which(args == "--modes"); modes <- strsplit(args[i + 1], ",")[[1]]; args <- args[-c(i, i + 1)] }
use_osm <- "--osm" %in% args; args <- args[args != "--osm"]
dat_dir <- if (length(args) >= 1) args[1] else "../data"

suppressMessages(suppressWarnings({ library(gamlss); library(gamlss.dist) }))

# ---- the eight finalised example cells (b-i) --------------------------------
EX <- data.frame(
  label   = c("b", "c", "d", "e", "f", "g", "h", "i"),
  class   = c("stationary", "location", "scale", "location and scale",
              "stationary", "location", "scale", "location and scale"),
  band    = c("low", "low", "low", "low", "high", "high", "high", "high"),
  cell_id = c(901988, 1283512, 1538746, 1547062, 1406770, 788582, 891235, 549028),
  lon     = c(-97.5, 27.35, -57.55, 176.75, 44.15, 79.4, 115.1, -66.8),   # d,e derived from land_cells too
  lat     = c(32.4, -3.05, -36.05, -39.45, -18.05, 39.8, 33.2, 52.1),
  region  = c("N.America", "Africa", "S.America", "N.Zealand", "Africa", "Asia", "Asia", "N.America"),
  # OSM-verified river / basin / admin (Overpass nearest named waterway + Nominatim reverse)
  river   = c("Nolan River", "Mosala River", "Samborombon River (El Tordillo Canal)", "Tutaekuri River",
              "Kimazimazy River", "Yarkand River", "Ying River (Shaying)", "Mouchalagane River"),
  basin   = c("Brazos River basin", "Congo River basin", "Salado del Sur / Samborombon (Atlantic)",
              "Tutaekuri (to Hawke Bay)", "W. Madagascar coastal (Maintirano)",
              "Tarim basin (endorheic)", "Huai River basin", "Manicouagan basin"),
  admin   = c("Johnson Co., Texas, USA", "Shabunda, South Kivu, DR Congo", "Castelli, Buenos Aires, Argentina",
              "Hastings, Hawke's Bay, NZ", "Maintirano, Melaky, Madagascar",
              "Maralbexi (Bachu), Kashgar, Xinjiang, China", "Shenqiu, Zhoukou, Henan, China",
              "Caniapiscau, Cote-Nord, Quebec, Canada"),
  stringsAsFactors = FALSE)

# lon/lat authoritative from land_cells by cell_id (fills e; corrects any drift)
.lc_path <- file.path(dat_dir, "010", "land_cells.csv")
if (file.exists(.lc_path)) {
  .lc <- read.csv(.lc_path)
  EX$lon <- -180 + (.lc$ix[EX$cell_id + 1] + 0.5) * 0.1
  EX$lat <- 90 - (.lc$iy[EX$cell_id + 1] + 0.5) * 0.1
} else if (anyNA(EX$lon)) {
  stop("land_cells.csv not found and EX has NA lon/lat (e=1475208 needs derivation).")
}

# old 077 class -> renamed class
RENAME <- c(stationary = "stationary", mean = "location", var = "scale",
            both = "location and scale")
CLASS_LEVELS <- c("stationary", "location", "scale", "location and scale")
# Okabe-Ito: maximally separated hues, colour-blind safe (grey/blue/orange/green).
CLASS_COL <- c("stationary" = "#BBBBBB", "location" = "#0072B2",
               "scale" = "#D55E00", "location and scale" = "#009E73")
# pie %-label colour: black on the light-grey wedge, white on the saturated ones.
CLASS_TXT <- c("stationary" = "black", "location" = "white",
               "scale" = "white", "location and scale" = "white")

n_years <- 120; year_start <- 1981
years   <- year_start:(year_start + n_years - 1)
late    <- years >= 2071 & years <= 2100
p_gu <- 0.01; min_flow <- 1.0; DELTA_AIC <- 2.0
ctrl <- gamlss.control(n.cyc = 50, trace = FALSE)

amax_bin  <- file.path(dat_dir, "030", "amax_all.bin")
out_dir   <- file.path(dat_dir, "115")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
if (!file.exists(amax_bin)) stop(sprintf("Not found: %s", amax_bin))

# ---- estimator styling (048) + renamed legend labels ----
PAT     <- c("ST120", "ST30", "linear", "quad")
COL     <- c(ST120 = "#737373", ST30 = "#f06292", linear = "#66b28e", quad = "#9575cd")
LTY     <- c(ST120 = 2,         ST30 = 1,         linear = 1,         quad = 1)
LWD     <- c(ST120 = 2.2,       ST30 = 2.8,       linear = 2.6,       quad = 2.6)
ALPHA   <- c(ST120 = 0.10,      ST30 = 0.08,      linear = 0.20,      quad = 0.18)
PTS_COL <- "#1f78b4"; WIN_COL <- "#88888826"
BAND_PATS <- c("linear", "quad")
LEG_LAB <- c("stationary (1981-2100)", "stationary (2071-2100)",
             "linear-nonstationary", "quadratic-nonstationary")

read_amax <- function(con, cid) {
  seek(con, where = as.numeric(cid) * n_years * 4, origin = "start")
  v <- readBin(con, what = "numeric", size = 4, n = n_years, endian = "little")
  if (length(v) != n_years || max(v) < min_flow) return(NULL)
  v
}
fit_const <- function(df) {
  m <- try(gamlss(outflow_neg ~ 1, sigma.fo = ~ 1, family = "GU", data = df, control = ctrl), silent = TRUE)
  if (inherits(m, "try-error")) return(NULL)
  list(mu = rep(fitted(m, "mu")[1], n_years), sigma = rep(fitted(m, "sigma")[1], n_years), aic = as.numeric(AIC(m)))
}
fit_trend <- function(df, fo_mu, fo_sg) {
  m <- try(gamlss(fo_mu, sigma.fo = fo_sg, family = "GU", data = df, control = ctrl), silent = TRUE)
  if (inherits(m, "try-error")) return(NULL)
  s <- fitted(m, "sigma"); if (any(!is.finite(s)) || any(s <= 0)) return(NULL)
  list(mu = fitted(m, "mu"), sigma = s, aic = as.numeric(AIC(m)))
}

# ---- fit the four estimators for each example cell --------------------------
fits <- list()
con <- file(amax_bin, "rb")
for (k in seq_len(nrow(EX))) {
  cid <- EX$cell_id[k]
  amax <- read_amax(con, cid)
  if (is.null(amax)) { next }
  df <- data.frame(year = years, outflow_neg = -amax)
  fits[[as.character(cid)]] <- list(
    amax = amax, ST120 = fit_const(df), ST30 = fit_const(df[late, , drop = FALSE]),
    linear = fit_trend(df, outflow_neg ~ year, ~ year),
    quad   = fit_trend(df, outflow_neg ~ poly(year, 2, raw = TRUE), ~ poly(year, 2, raw = TRUE)))
}
close(con)

# ---- 1. stripped panels (no text; big ticks; ST120 frontmost) ---------------
draw_curves <- function(nm, pp, mode, front = FALSE) {
  col <- COL[nm]; lt <- LTY[nm]; band <- nm %in% BAND_PATS
  if (mode != "q100" && band && !front) {
    lower <- pmax(pp$loc - pp$sig, 0)
    polygon(c(years, rev(years)), c(pp$loc + pp$sig, rev(lower)),
            col = adjustcolor(col, alpha.f = ALPHA[nm]), border = NA)
  }
  if (mode != "q100") lines(years, pp$loc, col = col, lwd = if (mode == "both") max(1.2, LWD[nm] - 1.0) else LWD[nm], lty = lt)
  if (mode != "qt")   lines(years, pp$q100, col = col, lwd = LWD[nm], lty = lt)
}
panel_one <- function(r, mode) {
  f <- fits[[as.character(r$cell_id)]]
  if (is.null(f)) { plot.new(); return(invisible()) }
  amax <- f$amax
  cur <- list()
  for (nm in PAT) { fit <- f[[nm]]; if (is.null(fit)) next
    cur[[nm]] <- list(loc = -fit$mu, sig = fit$sigma, q100 = -qGU(p_gu, mu = fit$mu, sigma = fit$sigma)) }
  ys <- amax
  for (nm in names(cur)) { pp <- cur[[nm]]; band <- nm %in% BAND_PATS
    if (mode != "q100") { ys <- c(ys, pp$loc); if (band) ys <- c(ys, pp$loc + pp$sig, pmax(pp$loc - pp$sig, 0)) }
    if (mode != "qt") ys <- c(ys, pp$q100) }
  yl <- range(ys, finite = TRUE); yl[2] <- yl[2] + 0.04 * diff(yl)
  par(mar = c(4.8, 5.6, 1.0, 2.4))   # wider right margin so the 2100 x-tick label is not clipped
  plot(years, amax, type = "n", ylim = yl, xlab = "", ylab = "", main = "", xaxt = "n", yaxt = "n")
  rect(2071, yl[1], 2100, yl[2], col = WIN_COL, border = NA)
  axis(1, at = pretty(years, 4), cex.axis = 2.1, lwd = 0, lwd.ticks = 1, padj = 0.3)
  axis(2, at = pretty(yl, 4),    cex.axis = 2.1, las = 1, lwd = 0, lwd.ticks = 1)
  box()
  for (nm in setdiff(names(cur), "ST120")) draw_curves(nm, cur[[nm]], mode)   # bands + lines
  points(years, amax, pch = 16, cex = 0.7, col = PTS_COL)
  if (!is.null(cur[["ST120"]])) draw_curves("ST120", cur[["ST120"]], mode, front = TRUE)  # ST120 on top
}
for (mode in modes) {
  pdir <- file.path(out_dir, sprintf("panels_%s", mode)); dir.create(pdir, showWarnings = FALSE)
  for (k in seq_len(nrow(EX))) {
    r <- EX[k, ]
    png(file.path(pdir, sprintf("panel_%s_%d.png", r$label, r$cell_id)), width = 760, height = 620, res = 130)
    panel_one(r, mode); dev.off()
  }
  cat(sprintf("Saved: panels_%s/ (8 panels)\n", mode))
}

# ---- 2. estimator legend as its own image ----------------------------------
png(file.path(out_dir, "legend_estimators.png"), width = 900, height = 360, res = 150)
par(mar = c(0, 0, 0, 0)); plot.new()
legend("center", bty = "n", cex = 1.6, lwd = c(LWD["ST120"], LWD["ST30"], LWD["linear"], LWD["quad"]),
       lty = LTY[PAT], col = COL[PAT], legend = LEG_LAB, seg.len = 3)
dev.off(); cat("Saved: legend_estimators.png\n")

# ---- load WHOLE-GLOBE GU class for map + pie -------------------------------
SCOPE <- c("077b" = "WHOLE-GLOBE (077b, all AMAX>1 cells)",
           "079"  = "flood-relevant only (079)", "077" = "flood-relevant only (077)")
load_classes <- function() {
  # prefer the all-cells table (077b/aic_allgrid.csv); then 079 / 077 combined
  # (flood-relevant only); finally concatenate per-chunk files (slow).
  for (sub in c("077b", "079", "077")) {
    ag <- file.path(dat_dir, sub, "aic_allgrid.csv")
    if (file.exists(ag)) {
      cat(sprintf("Reading classes: %s  [%s]\n", ag, SCOPE[sub]))
      d <- read.csv(ag); return(d[, c("cell_id", "iy", "ix", "class")])
    }
  }
  for (sub in c("077b", "077")) {
    fs <- Sys.glob(file.path(dat_dir, sub, "aic", "chunk_*.csv"))
    if (length(fs)) {
      cat(sprintf("Reading classes: %d chunks in %s/ (slow)  [%s]\n", length(fs), sub, SCOPE[sub]))
      return(do.call(rbind, lapply(fs, function(p) { d <- read.csv(p); d[, c("cell_id", "iy", "ix", "class")] })))
    }
  }
  NULL
}
cls <- load_classes()
if (!is.null(cls)) {
  cls <- cls[cls$class %in% names(RENAME), ]
  # WHOLE-GLOBE: AMAX<=1 already omitted by 077/077b. Drop only reverse-flow
  # (negative-AMAX) cells for data integrity; no flood-ensured / dry exclusion.
  bad_csv <- file.path(dat_dir, "010", "bad_cells.csv")
  if (file.exists(bad_csv)) { bc <- read.csv(bad_csv); cls <- cls[!(cls$cell_id %in% bc$cell_id), ] }
  cls$newclass <- factor(RENAME[cls$class], levels = CLASS_LEVELS)
  cls$lon <- -180 + (cls$ix + 0.5) * 0.1
  cls$lat <- 90 - (cls$iy + 0.5) * 0.1
  cat(sprintf("Classified cells for map/pie: %d\n", nrow(cls)))

  # ---- 3. global class map: Robinson RASTER (terra), no graticule labels, no legend ----
  # (the class legend lives in its own image, legend_classes.png)
  ROBIN <- "+proj=robin +datum=WGS84 +units=m +no_defs"
  png(file.path(out_dir, "global_class_map.png"), width = 2400, height = 1280, res = 150)
  if (requireNamespace("terra", quietly = TRUE)) {
    # rasterise the 0.1deg classification onto a regular lon/lat grid (imshow-like,
    # gap-free), then reproject to Robinson with nearest-neighbour (keeps classes).
    r <- terra::rast(xmin = -180, xmax = 180, ymin = -90, ymax = 90,
                     resolution = 0.1, crs = "EPSG:4326")
    v <- rep(NA_integer_, terra::ncell(r))
    v[terra::cellFromXY(r, cbind(cls$lon, cls$lat))] <- match(as.character(cls$newclass), CLASS_LEVELS)
    r <- terra::setValues(r, v)
    r <- terra::crop(r, terra::ext(-180, 180, -60, 90))           # omit Antarctica band
    levels(r) <- data.frame(id = seq_along(CLASS_LEVELS), class = CLASS_LEVELS)
    rp <- terra::project(r, ROBIN, method = "near")
    terra::plot(rp, type = "classes", col = CLASS_COL[CLASS_LEVELS], levels = CLASS_LEVELS,
                legend = FALSE, axes = FALSE, mar = c(0.4, 0.4, 0.4, 0.4))
    # faint coastlines, reprojected to Robinson (NA breaks preserve polylines)
    cl <- maps::map("world", plot = FALSE)
    keep <- is.finite(cl$x) & is.finite(cl$y) & cl$y > -60
    pj <- matrix(NA_real_, length(cl$x), 2)
    pj[keep, ] <- terra::project(cbind(cl$x, cl$y)[keep, , drop = FALSE],
                                 from = "EPSG:4326", to = ROBIN)
    lines(pj[, 1], pj[, 2], col = "grey55", lwd = 0.4)
    ep <- terra::project(cbind(EX$lon, EX$lat), from = "EPSG:4326", to = ROBIN)
    points(ep[, 1], ep[, 2], pch = 21, bg = "white", col = "black", cex = 2.8, lwd = 2.4)
    text(ep[, 1], ep[, 2], EX$label, pos = 3, offset = 0.5, cex = 2.2, font = 2)
  } else if (requireNamespace("mapproj", quietly = TRUE)) {
    cat("NOTE: 'terra' not installed -> Robinson point markers (mapproj), may show gaps.\n")
    par(mar = c(0.3, 0.3, 0.3, 0.3))
    pp  <- mapproj::mapproject(cls$lon, cls$lat, projection = "robinson")
    fin <- is.finite(pp$x) & is.finite(pp$y) & cls$lat > -60
    plot(pp$x[fin], pp$y[fin], col = CLASS_COL[as.character(cls$newclass[fin])],
         pch = 15, cex = 0.24, asp = 1, axes = FALSE, xlab = "", ylab = "")
    try(maps::map("world", projection = "robinson", wrap = TRUE, add = TRUE,
                  ylim = c(-60, 90), col = "grey55", lwd = 0.4), silent = TRUE)
    ep <- mapproj::mapproject(EX$lon, EX$lat)
    points(ep$x, ep$y, pch = 21, bg = "white", col = "black", cex = 2.8, lwd = 2.4)
    text(ep$x, ep$y, EX$label, pos = 3, offset = 0.5, cex = 2.2, font = 2)
  } else {
    cat("NOTE: neither 'terra' nor 'mapproj' installed -> equirectangular fallback.\n")
    par(mar = c(0.3, 0.3, 0.3, 0.3))
    plot(cls$lon, cls$lat, col = CLASS_COL[as.character(cls$newclass)], pch = 15, cex = 0.24,
         xlim = c(-180, 180), ylim = c(-60, 84), axes = FALSE, xlab = "", ylab = "",
         xaxs = "i", yaxs = "i", asp = 1)
    try(maps::map("world", add = TRUE, ylim = c(-60, 90), col = "grey55", lwd = 0.4), silent = TRUE)
    points(EX$lon, EX$lat, pch = 21, bg = "white", col = "black", cex = 2.8, lwd = 2.4)
    text(EX$lon, EX$lat, EX$label, pos = 3, offset = 0.5, cex = 2.2, font = 2)
  }
  dev.off(); cat("Saved: global_class_map.png\n")

  # ---- 4. class pie: integer % that sum to EXACTLY 100 (largest-remainder) ----
  tab <- table(cls$newclass); p <- as.numeric(tab); frac <- 100 * p / sum(p)
  pct <- floor(frac); k <- 100L - sum(pct)                      # remaining points to distribute
  if (k > 0) { ord <- order(frac - pct, decreasing = TRUE); pct[ord[seq_len(k)]] <- pct[ord[seq_len(k)]] + 1 }
  png(file.path(out_dir, "class_pie.png"), width = 1200, height = 1200, res = 150)
  par(mar = c(0.5, 0.5, 0.5, 0.5))
  pie(p, col = CLASS_COL[CLASS_LEVELS], border = "white", radius = 0.95, labels = NA)
  ang <- 2 * pi * cumsum(c(0, p)) / sum(p)
  for (i in seq_along(p)) {
    mid <- (ang[i] + ang[i + 1]) / 2
    text(0.62 * cos(mid), 0.62 * sin(mid), sprintf("%d%%", pct[i]),
         cex = 3.0, font = 2, col = CLASS_TXT[CLASS_LEVELS[i]])
  }
  dev.off(); cat(sprintf("Saved: class_pie.png  (%s ; sum=%d%%)\n",
                         paste(sprintf("%s=%d%%", CLASS_LEVELS, pct), collapse = ", "), sum(pct)))

  # ---- class legend as its own image (for the pie / map) ----
  png(file.path(out_dir, "legend_classes.png"), width = 1000, height = 420, res = 150)
  par(mar = c(0, 0, 0, 0)); plot.new()
  legend("center", bty = "n", fill = CLASS_COL[CLASS_LEVELS], legend = CLASS_LEVELS, cex = 1.9)
  dev.off(); cat("Saved: legend_classes.png\n")
} else cat("NOTE: no GU classes found -> map/pie skipped.\n")

# ---- overall example-cell table (the eight Figure-2 panels b-i), incl. excess ----
BASELINE <- 17.25
imp_csv <- file.path(dat_dir, "050", "summary_allgrid.csv")
EX$excess <- NA_real_
if (file.exists(imp_csv)) {
  im <- read.csv(imp_csv); m <- setNames(im$improve_lin_sd, as.character(im$cell_id))
  EX$excess <- round(unname(m[as.character(EX$cell_id)]) - BASELINE, 1)
} else cat("NOTE: 050/summary_allgrid.csv not found -> excess = NA in table.\n")
# column order for the table / Excel
EX <- EX[, c("label", "class", "band", "cell_id", "lon", "lat",
             "river", "basin", "admin", "region", "excess")]
write.csv(EX, file.path(out_dir, "example_cells.csv"), row.names = FALSE)
cat("Saved: example_cells.csv\n")

# ---- Excel with the SAME contents (river / basin / admin / excess) ----
xlsx_path <- file.path(out_dir, "example_cells.xlsx")
if (requireNamespace("writexl", quietly = TRUE)) {
  writexl::write_xlsx(EX, xlsx_path); cat(sprintf("Saved: %s (writexl)\n", basename(xlsx_path)))
} else if (requireNamespace("openxlsx", quietly = TRUE)) {
  openxlsx::write.xlsx(EX, xlsx_path); cat(sprintf("Saved: %s (openxlsx)\n", basename(xlsx_path)))
} else {
  cat("NOTE: neither 'writexl' nor 'openxlsx' installed -> Excel skipped (CSV still written).\n")
}
print(EX)

cat("Done.\n")
