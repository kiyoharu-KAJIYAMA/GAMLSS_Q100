#!/usr/bin/env Rscript

# --- 引数取得 ---
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) stop("Usage: Rscript osse.R <iy> <ix>")

iy <- as.integer(args[1])
ix <- as.integer(args[2])

# --- Load Libraries ---
suppressMessages(suppressWarnings({
  library(gamlss)
  library(gamlss.dist)
  library(dplyr)
}))

# --- Parameters ---
set.seed(123)
n_years <- 120
n_ensemble <- 50000
threshold <- 10000
output_root <- sprintf("/your/directory/path/ACC_ssp245_amax/basin_%d", threshold)

savedir <- sprintf("/your/directory/path/stationary_basin_%d", threshold)
delta_dir <- sprintf("/your/directory/path/delta/stationary_basin_%d", threshold)  # delta保存用
dir.create(savedir, showWarnings = FALSE, recursive = TRUE)
dir.create(delta_dir, showWarnings = FALSE, recursive = TRUE)

# --- 読み込み ---
fname <- sprintf("ix%04d_iy%04d", ix, iy)
fpath <- file.path(output_root, fname)
con <- file(paste0(fpath, ".bin"), "rb")
data <- readBin(con, what = "numeric", size = 4, n = n_years * 2, endian = "little")
close(con)

amax <- data.frame(matrix(data, ncol = 2, byrow = TRUE))
colnames(amax) <- c("year", "outflow")
amax$outflow_neg <- -amax$outflow

# --- GAMLSS Fit but stationary---
model_nonstat_true <- gamlss(outflow_neg ~ 1, sigma.fo = ~ 1, family = GU, data = amax, trace = FALSE)
mu_nonstat_true_neg <- fitted(model_nonstat_true, "mu")
sigma_nonstat_true <- fitted(model_nonstat_true, "sigma")
q100_true <- -qGU(1 - 0.99, mu = mu_nonstat_true_neg, sigma = sigma_nonstat_true)

# --- Ensemble Generation ---
osse_matrix <- replicate(n_ensemble, {
  -mapply(function(m, s) rGU(1, mu = m, sigma = s), mu_nonstat_true_neg, sigma_nonstat_true)
})
ensemble_cols <- paste0("outflow_", 1:n_ensemble)
osse_df <- as.data.frame(osse_matrix)
colnames(osse_df) <- ensemble_cols
osse_df$year <- 1981:(1980 + n_years)

# --- Correction by constant delta ---
all_values <- unlist(osse_df[, ensemble_cols])
min_val <- min(all_values)
q1_val <- quantile(all_values, 0.25)
delta <- abs(min_val) + q1_val

osse_df_corrected <- osse_df
osse_df_corrected[, ensemble_cols] <- lapply(osse_df[, ensemble_cols], function(x) x + delta)

cat(sprintf("✔ iy %d, ix %d | Correction delta = %.4f\n", iy, ix, delta))

# --- バイナリ保存 ---
save_fname <- sprintf("ix%04d_iy%04d", ix, iy)
save_fpath <- file.path(savedir, save_fname)
con <- file(paste0(save_fpath, ".bin"), "wb")
writeBin(as.numeric(as.matrix(osse_df_corrected[, ensemble_cols])), con, size = 4)
close(con)

# --- delta 保存 ---
delta_df <- data.frame(iy = iy, ix = ix, delta = delta)
delta_path <- file.path(delta_dir, sprintf("delta_iy%04d_ix%04d.csv", iy, ix))
write.csv(delta_df, delta_path, row.names = FALSE)
