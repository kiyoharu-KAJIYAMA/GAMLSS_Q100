"""
fig08_kurtosis_vs_scale_slope_pathways.py   (run in `gam` env)
Does the signal-to-noise LAW hold for the WHOLE flood-ensured sample (not just the 5
hotspot regions)?

    improvement = 1 - lin_err_sd/stat_err_sd = 1 - 1/sqrt(1 + SNR^2)
    SNR = Delta / sigma_int,   Delta = sqrt(stat_err_sd^2 - lin_err_sd^2)  (trend signal),
                               sigma_int = lin_err_sd                       (intrinsic noise)

This is an algebraic IDENTITY, so it must hold exactly for every cell -- the experiment
verifies that (collapse onto the theoretical curve, R^2 ~ 1) AND shows the substantive
consequence: improvement is governed jointly by the SIGNAL (Delta, driven by |sigma_1|)
and the NOISE (sigma_int, driven by N_eff), so on the (ln sigma_int, ln Delta) plane the
improvement forms perfect DIAGONAL bands (constant Delta/sigma_int).  The five hotspots
lie on the SAME high-improvement diagonal but at opposite ends: arid = high signal+high
noise (upper-right), boreal = moderate signal+low noise (lower-left).

Inputs: 050 (stat_err_sd, lin_err_sd), 079 (sig_slope, iy, ix), 519 (neff_var)
Outputs (<dat_dir>/525/): snr_law.csv, snr_law_plane.png, snr_collapse.png
Usage: python3 fig08_kurtosis_vs_scale_slope_pathways.py [dat_dir]
"""
import os
import sys
import importlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

excess_lib = importlib.import_module("common_target_cells")

RES = 0.1
REGIONS = [
    ("Quebec-Labrador",           -78.0, -60.0,  47.0,  54.0, "#1a5fa8"),
    ("West Siberia",               40.0,  95.0,  55.0,  66.0, "#4393c3"),
    ("North China Plain",         110.0, 120.0,  32.0,  41.0, "#b2182b"),
    ("West+Central Africa",       -12.0,  30.0,  -6.0,  14.0, "#d6604d"),
    ("Southern Africa (Kalahari)", 19.0,  29.0, -27.0, -18.0, "#e8843c"),
]


def rank_r2(y, Xcols):
    yr = stats.rankdata(y)
    G = np.column_stack([np.ones(len(y))] + [stats.rankdata(c) for c in Xcols])
    beta, *_ = np.linalg.lstsq(G, yr, rcond=None)
    pred = G @ beta
    return 1 - np.sum((yr - pred) ** 2) / np.sum((yr - yr.mean()) ** 2)


def partial_spearman(y, x, z):
    ry, rx, rz = (stats.rankdata(v) for v in (y, x, z))
    def resid(a, b):
        A = np.column_stack([np.ones(a.size), b]); be, *_ = np.linalg.lstsq(A, a, rcond=None)
        return a - A @ be
    return np.corrcoef(resid(ry, rz), resid(rx, rz))[0, 1]


def main():
    dat_dir = sys.argv[1] if len(sys.argv) > 1 else \
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
    out_dir = os.path.join(dat_dir, "525"); os.makedirs(out_dir, exist_ok=True)

    d50 = pd.read_csv(os.path.join(dat_dir, "050", "summary_allgrid.csv"),
                      usecols=["cell_id", "stat_err_sd", "lin_err_sd"])
    d79 = pd.read_csv(os.path.join(dat_dir, "079", "aic_allgrid.csv"),
                      usecols=["cell_id", "iy", "ix", "sig_slope"])
    d519 = pd.read_csv(os.path.join(dat_dir, "519", "mechanism_vars.csv"),
                       usecols=["cell_id", "neff_var"])
    df = d50.merge(d79, on="cell_id").merge(d519, on="cell_id")
    keep = excess_lib.load_flood_ensured_ids(dat_dir)
    if keep:
        df = df[df["cell_id"].isin(keep)]
    df = df[(df["stat_err_sd"] > 0) & (df["lin_err_sd"] > 0) &
            np.isfinite(df["sig_slope"]) & np.isfinite(df["neff_var"])].copy()
    df["imp"] = (df["stat_err_sd"] - df["lin_err_sd"]) / df["stat_err_sd"] * 100.0
    df = df[np.abs(df["imp"]) <= excess_lib.CAP_PCT]
    n_all = len(df)
    neg = int((df["imp"] <= 0).sum())
    print(f"flood-ensured cells={n_all:,}   (improvement<=0: {neg:,} = {100*neg/n_all:.1f}%, "
          f"model overfits -> no signal; excluded from SNR)")

    # SNR only defined where stat >= lin (improvement>0)
    df = df[df["stat_err_sd"] > df["lin_err_sd"]].copy()
    df["delta"] = np.sqrt(df["stat_err_sd"] ** 2 - df["lin_err_sd"] ** 2)
    df["sig_int"] = df["lin_err_sd"]
    df["snr"] = df["delta"] / df["sig_int"]
    df["asig"] = np.abs(df["sig_slope"])
    n = len(df)

    # ---- (A) identity check: improvement == 100*(1 - 1/sqrt(1+SNR^2)) ----
    pred = 100.0 * (1.0 - 1.0 / np.sqrt(1.0 + df["snr"].to_numpy() ** 2))
    err = np.abs(pred - df["imp"].to_numpy())
    ss = 1 - np.sum((df["imp"] - pred) ** 2) / np.sum((df["imp"] - df["imp"].mean()) ** 2)
    print(f"\n[A] IDENTITY improvement = 1-1/sqrt(1+SNR^2) over {n:,} cells:")
    print(f"    R^2={ss:.6f}   max|err|={err.max():.3g}%   median|err|={np.median(err):.2g}%")
    print("    => the law is EXACT for every flood-ensured cell (holds universally).")

    imp = df["imp"].to_numpy(); dl = df["delta"].to_numpy(); si = df["sig_int"].to_numpy()
    asig = df["asig"].to_numpy(); neff = df["neff_var"].to_numpy()

    # ---- (B) both signal and noise govern improvement, sample-wide ----
    r2_sig = rank_r2(imp, [np.log(dl)])
    r2_noi = rank_r2(imp, [np.log(si)])
    r2_both = rank_r2(imp, [np.log(dl), np.log(si)])
    print(f"\n[B] rank-R^2 of improvement from  ln(signal Delta)={100*r2_sig:.1f}%   "
          f"ln(noise sigma_int)={100*r2_noi:.1f}%   both={100*r2_both:.1f}%")
    print(f"    Spearman(Delta,|sig1|)={stats.spearmanr(dl,asig).correlation:+.3f} "
          f"(signal tracks the trend)   "
          f"Spearman(sigma_int,N_eff)={stats.spearmanr(si,neff).correlation:+.3f}")
    p1 = partial_spearman(imp, neff, asig)
    p2 = partial_spearman(imp, asig, neff)
    print(f"    PARTIAL(imp,N_eff | |sig1|)={p1:+.3f}  (higher N_eff -> lower noise -> "
          f"more improvement at fixed trend)")
    print(f"    PARTIAL(imp,|sig1| | N_eff)={p2:+.3f}")

    pd.DataFrame({"metric": ["n_cells", "identity_R2", "max_abs_err_pct", "r2_from_signal",
                             "r2_from_noise", "r2_from_both", "spearman_delta_asig",
                             "spearman_sigint_neff", "partial_imp_neff", "partial_imp_asig"],
                  "value": [n, ss, err.max(), r2_sig, r2_noi, r2_both,
                            stats.spearmanr(dl, asig).correlation,
                            stats.spearmanr(si, neff).correlation, p1, p2]}
                 ).to_csv(os.path.join(out_dir, "snr_law.csv"), index=False)

    # ---- (C1) collapse figure: improvement vs SNR + theory curve ----
    fig, ax = plt.subplots(figsize=(8.5, 6))
    ax.hexbin(df["snr"], imp, gridsize=70, bins="log", cmap="Greys", mincnt=1,
              extent=(0, np.percentile(df["snr"], 99.5), 0, 30))
    xs = np.linspace(0, np.percentile(df["snr"], 99.5), 200)
    ax.plot(xs, 100 * (1 - 1 / np.sqrt(1 + xs ** 2)), "-", color="#e6550d", lw=2.4,
            label="law: 1 - 1/√(1+SNR²)")
    ax.set_xlabel("SNR = Δ / σ_int  (trend signal / intrinsic noise)")
    ax.set_ylabel("improvement [%]")
    ax.set_title(f"The SNR law holds for ALL flood-ensured cells (n={n:,}, R²={ss:.4f})")
    ax.legend(); ax.grid(alpha=0.2)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "snr_collapse.png"), dpi=160)
    plt.close(fig)

    # ---- (C2) (ln sigma_int, ln Delta) plane: diagonal improvement bands + regions ----
    lon = -180.0 + (df["ix"].to_numpy() + 0.5) * RES
    lat = 90.0 - (df["iy"].to_numpy() + 0.5) * RES
    thr = np.percentile(imp, 80.0); top = imp >= thr
    lx, ly = np.log10(si), np.log10(dl)
    fig, ax = plt.subplots(figsize=(9.5, 8))
    xr = (np.percentile(lx, 1), np.percentile(lx, 99))
    yr = (np.percentile(ly, 1), np.percentile(ly, 99))
    hb = ax.hexbin(lx, ly, C=imp, gridsize=55, cmap="viridis", reduce_C_function=np.median,
                   extent=(xr[0], xr[1], yr[0], yr[1]), mincnt=20)
    plt.colorbar(hb, ax=ax, label="median improvement [%]")
    # diagonal iso-improvement lines: ln Delta = ln sigma_int + ln r(I)
    gg = np.linspace(xr[0], xr[1], 10)
    for I, c in [(18, "#ffffff"), (20, "#fdd0a2"), (22, "#e6550d")]:
        r = np.sqrt(1.0 / (1.0 - I / 100.0) ** 2 - 1.0)
        ax.plot(gg, gg + np.log10(r), "--", color=c, lw=1.5,
                label=f"improvement = {I}%")
    for name, lo, hi, la0, la1, col in REGIONS:
        m = (lon >= lo) & (lon < hi) & (lat >= la0) & (lat < la1) & top
        if m.sum() < 20:
            continue
        ax.plot(np.median(lx[m]), np.median(ly[m]), "o", ms=13, mfc=col, mec="black",
                mew=1.5, zorder=10)
        ax.annotate(name.split(" (")[0], (np.median(lx[m]), np.median(ly[m])),
                    fontsize=9, fontweight="bold", color=col, ha="center",
                    xytext=(0, 11), textcoords="offset points",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=col, alpha=0.9))
    ax.set_xlabel("log10  σ_int  (intrinsic NOISE = lin_err_sd;  ← low N_eff/heavy tails)")
    ax.set_ylabel("log10  Δ  (trend SIGNAL = √(stat²−lin²))")
    ax.set_title("improvement forms DIAGONAL bands in the (noise, signal) plane\n"
                 "(depends only on the ratio Δ/σ_int). Arid = high-signal/high-noise (upper-right),\n"
                 "boreal = moderate-signal/low-noise (lower-left): same band, opposite ends",
                 fontsize=10.5)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "snr_law_plane.png"), dpi=160)
    plt.close(fig)

    # ---- (C3) SCATTER x=SNR, y=improvement, COLOUR = noise sigma_int ----
    # improvement = f(SNR) is exact/1-D, so points collapse onto the curve; a tiny
    # vertical jitter (+/-0.25%, for VISIBILITY only) spreads them so the noise colour
    # is readable. Shows that at a given SNR/improvement, a whole RANGE of noise occurs.
    snr_all = df["snr"].to_numpy()
    neff_all = df["neff_var"].to_numpy(); sigint_all = df["sig_int"].to_numpy()
    sub = df.sample(min(70000, len(df)), random_state=0)
    rng = np.random.default_rng(0)
    jit = rng.normal(0.0, 0.25, len(sub))
    cval = np.log10(sub["sig_int"].to_numpy())
    xs = np.linspace(0, np.percentile(df["snr"], 99.7), 300)
    curve = 100 * (1 - 1 / np.sqrt(1 + xs ** 2))
    fig, ax = plt.subplots(figsize=(11.5, 7.6))
    sc = ax.scatter(sub["snr"], sub["imp"] + jit, c=cval, s=7, cmap="viridis",
                    vmin=np.percentile(cval, 2), vmax=np.percentile(cval, 98),
                    alpha=0.55, linewidths=0, rasterized=True)
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label("log10 σ_int   (intrinsic NOISE;  low ↓ … ↑ high)", fontsize=11)
    ax.plot(xs, curve, "-", color="#d62728", lw=2.6, zorder=6, label="law: 1−1/√(1+SNR²)")
    # region markers (white-edged so visible over the colour scatter)
    for name, lo, hi, la0, la1, col in REGIONS:
        m = (lon >= lo) & (lon < hi) & (lat >= la0) & (lat < la1) & top
        if m.sum() < 20:
            continue
        ax.plot(np.median(snr_all[m]), np.median(imp[m]), "o", ms=14, mfc=col,
                mec="white", mew=2.0, zorder=10)
    # compact table upper-left
    rows_tab = sorted([(name.split(" (")[0].split(" /")[0], col,
                        np.median(snr_all[(lon >= lo) & (lon < hi) & (lat >= la0) & (lat < la1) & top]),
                        np.median(imp[(lon >= lo) & (lon < hi) & (lat >= la0) & (lat < la1) & top]),
                        np.median(neff_all[(lon >= lo) & (lon < hi) & (lat >= la0) & (lat < la1) & top]),
                        np.median(sigint_all[(lon >= lo) & (lon < hi) & (lat >= la0) & (lat < la1) & top]))
                       for name, lo, hi, la0, la1, col in REGIONS], key=lambda r: -r[4])
    ax.text(0.03, 0.97, f"{'region':<20s}{'SNR':>5s}{'N_eff':>7s}{'σ_int':>7s}",
            transform=ax.transAxes, fontsize=9, family="monospace", fontweight="bold", va="top")
    for i, (nm, col, sx, sy, ne, si) in enumerate(rows_tab):
        ax.text(0.03, 0.925 - i * 0.043, f"{nm:<20s}{sx:>5.2f}{ne:>7.0f}{si:>7.3f}",
                transform=ax.transAxes, fontsize=9, family="monospace", color=col, va="top")
    ax.set_xlabel("SNR = Δ / σ_int  (signal / noise)", fontsize=12)
    ax.set_ylabel("improvement [%]   (y jittered ±0.25% for visibility)", fontsize=11)
    ax.set_xlim(0, np.percentile(df["snr"], 99.7)); ax.set_ylim(14, 30)
    ax.set_title("Scatter coloured by NOISE (σ_int): at any given SNR/improvement a full RANGE of\n"
                 "noise occurs (colour spreads vertically). The 5 hotspots (dots) share SNR≈0.8\n"
                 "but boreal = low noise (dark), arid = high noise (bright)", fontsize=11)
    ax.legend(loc="lower right", fontsize=10); ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "snr_vs_improvement_colored.png"), dpi=170)
    plt.close(fig)
    # ---- (C4) upstream-driver plane: x = N_eff (or kurtosis), y = SIGNED sig_slope,
    # colour = improvement.  4 variants: {N_eff, kurtosis} x {log10, linear}.
    # No axis labels / titles (journal-style, captions carry the text); big fonts.
    from matplotlib.colors import Normalize
    sig_signed = df["sig_slope"].to_numpy()
    # kurtosis of the AMAX series from 070 (rank-identical to 120/N_eff)
    t70k = pd.read_csv(os.path.join(dat_dir, "070", "timeseries_tests.csv"),
                       usecols=["cell_id", "kurtosis"])
    df2 = df.merge(t70k, on="cell_id", how="left")
    kurt_all = df2["kurtosis"].to_numpy()
    print(f"\n[C4] corr(N_eff, sig_slope signed) = "
          f"{stats.spearmanr(neff_all, sig_signed).correlation:+.3f}   "
          f"corr(N_eff, |sig_slope|) = {stats.spearmanr(neff_all, np.abs(sig_signed)).correlation:+.3f}")

    VMIN, VMAX = 16.5, 25.0
    norm = Normalize(VMIN, VMAX)
    cmap = plt.get_cmap("RdYlBu_r")
    FS_TICK, FS_CBAR, FS_LAB = 17, 17, 14
    # display names for the figure labels (paper wording)
    DISPLAY = {"Quebec-Labrador": "Quebec", "West Siberia": "West Siberia",
               "North China Plain": "North China Plain",
               "West+Central Africa": "Sahelian belt",
               "Southern Africa (Kalahari)": "Southern Africa"}
    LABOFF_NEFF = {"Quebec-Labrador": (-44, -40, "right"), "West Siberia": (-110, 18, "left"),
                   "North China Plain": (28, 18, "left"), "West+Central Africa": (-34, 40, "right"),
                   "Southern Africa (Kalahari)": (-48, -36, "right")}
    LABOFF_KURT = {"Quebec-Labrador": (44, -40, "left"), "West Siberia": (48, 24, "left"),
                   "North China Plain": (-50, 28, "right"), "West+Central Africa": (52, 38, "left"),
                   "Southern Africa (Kalahari)": (54, -32, "left")}

    variants = [("neff", neff_all, True,  "slope_vs_neff_plane.png"),
                ("neff", neff_all, False, "slope_vs_neff_plane_linear.png"),
                ("kurt", kurt_all, True,  "slope_vs_kurt_plane.png"),
                ("kurt", kurt_all, False, "slope_vs_kurt_plane_linear.png")]
    for vname, xraw, use_log, fname in variants:
        okv = np.isfinite(xraw) & (xraw > 0 if use_log else np.isfinite(xraw))
        xv = np.where(okv, xraw, np.nan)
        xplot = np.log10(xv) if use_log else xv
        # ALL finite cells, PAINTER'S ORDER by improvement: low imp drawn first,
        # high imp drawn last -> always on top where points overlap
        iv = np.nonzero(np.isfinite(xplot) & np.isfinite(imp) & np.isfinite(sig_signed))[0]
        order2 = iv[np.argsort(imp[iv], kind="stable")]
        xr2 = (np.nanpercentile(xplot, 0.5), np.nanpercentile(xplot, 99.5))
        yr2 = (np.percentile(sig_signed, 0.5), np.percentile(sig_signed, 99.5))
        fig, ax = plt.subplots(figsize=(10.8, 8))
        sc = ax.scatter(xplot[order2], sig_signed[order2], c=imp[order2], s=4,
                        cmap=cmap, vmin=VMIN, vmax=VMAX, alpha=0.75, linewidths=0,
                        rasterized=True)
        cb = plt.colorbar(sc, ax=ax)
        cb.ax.tick_params(labelsize=FS_CBAR)          # no colorbar label (caption text)
        ax.axhline(0.0, color="#444444", lw=1.0, ls=":")
        for name, lo, hi, la0, la1, _rcol in REGIONS:
            m = (lon >= lo) & (lon < hi) & (lat >= la0) & (lat < la1) & top & np.isfinite(xplot)
            if m.sum() < 20:
                continue
            px, py = np.median(xplot[m]), np.median(sig_signed[m])
            mcol = cmap(norm(np.median(imp[m])))         # marker colour = improvement
            ax.plot(px, py, "o", ms=17, mfc=mcol, mec="black", mew=1.8, zorder=10)
            laboff = LABOFF_KURT if vname == "kurt" else LABOFF_NEFF
            dx, dy, ha = laboff.get(name, (16, 16, "left"))
            ax.annotate(DISPLAY.get(name, name), (px, py),
                        fontsize=FS_LAB, fontweight="bold", color="black", ha=ha,
                        va="center", xytext=(dx, dy), textcoords="offset points",
                        zorder=11,
                        bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="black",
                                  alpha=0.95),
                        arrowprops=dict(arrowstyle="-", color="black", lw=1.1))
        ax.set_xlim(xr2); ax.set_ylim(yr2)
        ax.tick_params(labelsize=FS_TICK)
        ax.grid(alpha=0.15)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, fname), dpi=170)
        plt.close(fig)
        print(f"Saved: {fname}  (x={vname}, {'log10' if use_log else 'linear'})")

    print(f"\nSaved: snr_law.csv, snr_collapse.png, snr_law_plane.png, "
          f"snr_vs_improvement_colored.png, slope_vs_neff_plane.png")


if __name__ == "__main__":
    main()
