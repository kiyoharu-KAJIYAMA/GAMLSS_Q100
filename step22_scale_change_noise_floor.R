#!/usr/bin/env Rscript
#
# step22_scale_change_noise_floor.R
# Noise floor of the scale-change estimate |Delta log sigma| = |sigma_1| * 119.
#
# 502 uses sigma_1 from the ln-parent GAMLSS fit (GU, mu ~ year, log sigma ~
# year, n = 120). Even for a TRULY STATIONARY series this fit returns a nonzero
# slope, so small observed |Delta log sigma| values (~0.4-0.5 in wet/large
# basins) may be pure estimation noise. This script simulates that null:
#
#   truth: stationary GU(mu, sigma), 120 yr  ->  fit the ln model  ->  sigma_1
#
# KEY INVARIANCE: sigma uses a LOG link, so sigma_1 is dimensionless (relative
# change per year). Scaling the flows shifts sigma_0 only; shifting them moves
# mu only. Hence ONE configuration gives the universal null for every cell.
# A second configuration with a strong mu trend is run to verify that the mu
# trend (absorbed by the mu ~ year term) does not alter the sigma_1 null.
#
# Output: <dat_dir>/503/null_dlogsigma.csv   (one |sigma_1|*119 draw per row,
#         config = "stationary" | "mu_trend")
#         + printed quantiles (50/75/90/95/99%).
#
# Usage: Rscript step22_scale_change_noise_floor.R [dat_dir] [B_stationary] [B_mutrend]
#        defaults: ../data  2000  1000

args    <- commandArgs(trailingOnly = TRUE)
dat_dir <- if (length(args) >= 1) args[1] else "../data"
B1      <- if (length(args) >= 2) as.integer(args[2]) else 2000L
B2      <- if (length(args) >= 3) as.integer(args[3]) else 1000L

suppressMessages(suppressWarnings({
  library(gamlss)
  library(gamlss.dist)
}))

set.seed(503)
years <- 1981:2100
n     <- length(years)
SPAN  <- diff(range(years))          # 119
ctrl  <- gamlss.control(n.cyc = 50, trace = FALSE)

# arbitrary representative parameters (null is invariant to them; see header)
mu0    <- -1000                       # GU on negated flow (as 040/064)
sigma0 <- 200
mu_slope <- 5                         # config 2: strong mu trend (~60% of range)

fit_s1 <- function(y) {
  df <- data.frame(year = years, outflow_neg = y)
  m  <- try(gamlss(outflow_neg ~ year, sigma.fo = ~ year, family = GU,
                   data = df, control = ctrl), silent = TRUE)
  if (inherits(m, "try-error")) return(NA_real_)
  as.numeric(coef(m, what = "sigma")[2])
}

run_cfg <- function(B, mu_t, label) {
  out <- rep(NA_real_, B)
  for (b in seq_len(B)) {
    y <- rGU(n, mu = mu_t, sigma = sigma0)
    out[b] <- fit_s1(y)
    if (b %% 200 == 0) cat(sprintf("  %s: %d/%d\n", label, b, B))
  }
  abs(out) * SPAN
}

cat(sprintf("Simulating null |Delta log sigma| (n=%d yr): stationary B=%d, mu-trend B=%d\n",
            n, B1, B2))
d1 <- run_cfg(B1, rep(mu0, n), "stationary")
d2 <- run_cfg(B2, mu0 + mu_slope * (years - years[1]), "mu_trend")

qs <- c(0.50, 0.75, 0.90, 0.95, 0.99)
cat("\nNull quantiles of |Delta log sigma| (stationary truth):\n")
print(round(quantile(d1, qs, na.rm = TRUE), 4))
cat("Null quantiles (mu-trend truth; should match => invariance):\n")
print(round(quantile(d2, qs, na.rm = TRUE), 4))

out_dir <- file.path(dat_dir, "503")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
out_csv <- file.path(out_dir, "null_dlogsigma.csv")
df_out <- rbind(data.frame(config = "stationary", dlogsigma = d1),
                data.frame(config = "mu_trend",  dlogsigma = d2))
write.csv(df_out[is.finite(df_out$dlogsigma), ], out_csv, row.names = FALSE)
cat(sprintf("\nSaved: %s (%d draws)\n", out_csv, sum(is.finite(df_out$dlogsigma))))
