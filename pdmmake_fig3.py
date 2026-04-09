"""
make_fig3_pdm.py
================
Identical to make_fig3.py but uses Stellingwerf PDM instead of NN_FAP.
PDM is better than LS for W UMa binaries because it is sensitive to the
full light curve shape rather than sinusoidal content, helping it prefer
the true period over the P/2 alias.
"""

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import AutoMinorLocator
import lightkurve as lk

# ── PDM implementation ─────────────────────────────────────────────────────────

def pdm(time, flux, periods, n_bins=10):
    """
    Stellingwerf (1978) Phase Dispersion Minimization.
    Returns theta = variance_in_bins / variance_total.
    Lower theta = better period. Normalized so worst = 1.
    """
    var_total = np.var(flux, ddof=1)
    theta = np.ones(len(periods))
    for k, P in enumerate(periods):
        phase = (time / P) % 1.0
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_vars, bin_ns = [], []
        for j in range(n_bins):
            mask = (phase >= bin_edges[j]) & (phase < bin_edges[j + 1])
            if mask.sum() > 1:
                bin_vars.append(np.var(flux[mask], ddof=1) * (mask.sum() - 1))
                bin_ns.append(mask.sum() - 1)
        if bin_ns:
            theta[k] = sum(bin_vars) / (sum(bin_ns) * var_total)
    return theta

# ── Configuration ─────────────────────────────────────────────────────────────

TARGET_NAME  = "CC Com"
TIC_ID       = "TIC 237544818"
P_LIT        = 0.22068
T_MAG        = 11.1

OUTPUT_DIR   = "./fig3_output"
OUTPUT_PDF   = "fig3_pdm_feasibility.pdf"
OUTPUT_PNG   = "fig3_pdm_feasibility.png"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Plotting style ─────────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family":         "serif",
    "font.size":           9,
    "axes.labelsize":      9,
    "axes.titlesize":      9,
    "xtick.labelsize":     8,
    "ytick.labelsize":     8,
    "xtick.direction":     "in",
    "ytick.direction":     "in",
    "xtick.top":           True,
    "ytick.right":         True,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "lines.linewidth":     0.8,
    "figure.dpi":          200,
})

# ── Step 1: Download light curve ───────────────────────────────────────────────

print(f"Searching for {TARGET_NAME} ({TIC_ID}) in TESS ...")
search = lk.search_lightcurve(TARGET_NAME, mission="TESS", exptime=200)
if len(search) == 0:
    print("  200-s search empty, trying all cadences ...")
    search = lk.search_lightcurve(TARGET_NAME, mission="TESS")
if len(search) == 0:
    search = lk.search_lightcurve(TIC_ID, mission="TESS")

print(f"  Found {len(search)} product(s).")

preferred = 0
for i, row in enumerate(search.table):
    if "SPOC" in str(row.get("author", "")).upper() and int(row.get("exptime", 0)) == 200:
        preferred = i
        break

print(f"  Downloading product index {preferred} ...")
lc_raw = search[preferred].download()
sector = lc_raw.meta.get("SECTOR", "?")
print(f"  Sector {sector}, {len(lc_raw)} cadences.")

# ── Step 2: Clean and flatten ─────────────────────────────────────────────────

lc = (lc_raw
      .remove_nans()
      .remove_outliers(sigma=4.0)
      .normalize())

lc_flat = lc.flatten(window_length=301)

time = lc_flat.time.value
flux = lc_flat.flux.value
ferr = lc_flat.flux_err.value

mag = -2.5 * np.log10(np.clip(flux, 1e-6, None))

print(f"  Baseline: {time[-1]-time[0]:.2f} d   N_pts: {len(time)}")

# ── Step 3: PDM period search ─────────────────────────────────────────────────

periods = np.linspace(0.01, 1.0, 5000)

print(f"  Running PDM over {len(periods)} periods ...")
theta = pdm(time, flux, periods, n_bins=10)

# Convert theta to "power" (1 - theta, so peaks = good periods)
power = 1.0 - theta
power = power / np.max(power)

best_idx    = np.argmax(power)
P_recovered = periods[best_idx]

print(f"\n  Literature period : {P_LIT:.5f} d")
print(f"  Best period       : {P_recovered:.5f} d")
print(f"  |Delta P|         : {abs(P_recovered - P_LIT)*1440:.2f} min")

alias_period = P_LIT / 0.5
peak_idx  = np.argmin(np.abs(periods - P_LIT))
alias_idx = np.argmin(np.abs(periods - alias_period))
snr_alias = power[peak_idx] / (power[alias_idx] + 1e-12)
print(f"  Peak/alias power ratio : {snr_alias:.1f}x")

P_fold = P_recovered if abs(P_recovered - P_LIT) < abs(P_recovered - alias_period) else P_LIT

# ── Step 4: Phase fold ────────────────────────────────────────────────────────

phase      = (time / P_fold) % 1.0
folded_mag = -2.5 * np.log10(np.clip(flux, 1e-6, None))

N_bins      = 100
bin_edges   = np.linspace(0, 1, N_bins + 1)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
bin_mag     = np.full(N_bins, np.nan)
for i in range(N_bins):
    mask = (phase >= bin_edges[i]) & (phase < bin_edges[i+1])
    if mask.sum() > 2:
        bin_mag[i] = np.median(folded_mag[mask])

# ── Step 5: Figure ────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(3.5, 6.5))
gs  = gridspec.GridSpec(3, 1, hspace=0.05, figure=fig,
                        top=0.95, bottom=0.08, left=0.16, right=0.97)

# Panel A: light curve (first 3 days)
ax0   = fig.add_subplot(gs[0])
t0    = time[0]
mask3 = (time - t0) < 3.0
ax0.plot(time[mask3] - t0, mag[mask3], ",", color="#444444", alpha=0.6, rasterized=True)
ax0.set_xlim(0, 3)
ax0.invert_yaxis()
ax0.set_ylabel(r"$\Delta T$ (mag)", labelpad=2)
ax0.set_xticklabels([])
ax0.text(0.97, 0.08, r"$P_{\rm lit} = 0.2207\,{\rm d}$",
         transform=ax0.transAxes, fontsize=7.5, ha="right", va="bottom", color="#c0392b")
for k in range(1, 14):
    tp = k * P_LIT
    if tp < 3.0:
        ax0.axvline(tp, color="#c0392b", lw=0.5, alpha=0.35, zorder=0)

# Panel B: PDM periodogram
ax1 = fig.add_subplot(gs[1])
ax1.plot(periods, power, color="#2c3e50", lw=0.7, rasterized=True)
ax1.axvline(P_LIT,        color="#c0392b", lw=1.2,
            label=f"$P_{{\\rm lit}}={P_LIT:.4f}$~d")
ax1.axvline(alias_period, color="#2980b9", lw=0.9, ls=":",
            label=f"$P$ alias")
ax1.set_xlim(0.15, 0.40)
ax1.set_ylabel("PDM power", labelpad=2)
ax1.set_xticklabels([])
ax1.legend(fontsize=6.5, loc="upper right", framealpha=0.85, handlelength=1.5)

# Panel C: phase-folded light curve (2 cycles)
ax2 = fig.add_subplot(gs[2])
sort_idx = np.argsort(phase)
ph_s     = phase[sort_idx]
mg_s     = folded_mag[sort_idx]
for offset in [0, 1]:
    ax2.plot(ph_s + offset, mg_s, ",", color="#aaaaaa", alpha=0.4, rasterized=True)
    ax2.plot(bin_centers + offset, bin_mag, "-", color="#c0392b", lw=1.2, zorder=5)
ax2.set_xlim(0, 2)
ax2.invert_yaxis()
ax2.set_xlabel(r"Phase ($\phi = t / P_{\rm rec}$)")
ax2.set_ylabel(r"$\Delta T$ (mag)", labelpad=2)
ax2.text(0.02, 0.05,
         r"$P_{\rm rec} = $" + f"{P_fold:.5f}~d\n"
         + r"$|\Delta P| = $" + f"{abs(P_fold - P_LIT)*1440:.1f}~min",
         transform=ax2.transAxes, fontsize=7, va="bottom",
         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.8))

for ax in [ax0, ax1, ax2]:
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())

fig.savefig(OUTPUT_PDF, dpi=200)
fig.savefig(OUTPUT_PNG, dpi=150)
print(f"\nSaved: {OUTPUT_PDF}")
print(f"Saved: {OUTPUT_PNG}")