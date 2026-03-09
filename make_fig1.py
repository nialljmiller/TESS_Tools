"""
Figure 1: Kepler EW contact binary period distribution showing the 0.22d cutoff.

Tries to load the Kirk et al. (2016) Kepler EB catalog from a local file
(kepler_ebs_kirk2016.csv) if you have it, otherwise downloads it from the
Villanova Kepler EB catalog. Falls back to a statistically faithful synthetic
distribution if neither is available.

Output: fig1_period_distribution.pdf
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator
from scipy.stats import gaussian_kde
import os

# ── 0. Style ─────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':      'serif',
    'font.size':        10,
    'axes.labelsize':   11,
    'axes.titlesize':   11,
    'xtick.labelsize':  9,
    'ytick.labelsize':  9,
    'xtick.direction':  'in',
    'ytick.direction':  'in',
    'xtick.top':        True,
    'ytick.right':      True,
    'xtick.minor.visible': True,
    'ytick.minor.visible': True,
    'axes.linewidth':   1.1,
    'legend.fontsize':  8.5,
    'figure.dpi':       200,
})

P_CUT = 0.22   # days  – the short-period cutoff

# ── 1. Load data — Kirk et al. (2016) Kepler EB catalog ──────────────────────
# Columns (space-separated, no header):
#   0: KIC ID  1: Period(d)  2: Period_err  3: BJD0  4: Amplitude
#   5: Morphology (>0.7 = EW contact binary)  6-8: coords/Kp  9: Teff  10: Flag

catalog_path = 'catalog.dat'

data = np.genfromtxt(catalog_path, dtype=None, encoding='ascii',
                     usecols=(1, 5),           # period, morphology
                     names=['period', 'morph'])

# EW contact binaries: morphology > 0.7 (Kirk+2016 convention)
mask_ew    = data['morph'] > 0.7
periods_ew = data['period'][mask_ew]
print(f"Kirk+2016: {len(data)} total EBs → {len(periods_ew)} EW contact binaries (morph > 0.7)")

# Restrict to sensible EW range for the plot
mask = (periods_ew >= 0.15) & (periods_ew <= 1.4)
periods_ew = periods_ew[mask]

# ── 2. Build histogram + KDE ──────────────────────────────────────────────────
bins      = np.arange(0.15, 1.42, 0.02)   # 0.02-day bins
counts, _ = np.histogram(periods_ew, bins=bins)
bin_cens  = 0.5 * (bins[:-1] + bins[1:])

# KDE over full range for smooth envelope
kde_x   = np.linspace(0.15, 1.4, 1000)
kde_bw  = 0.025  # days
kde     = gaussian_kde(periods_ew, bw_method=kde_bw / np.std(periods_ew))
kde_y   = kde(kde_x) * len(periods_ew) * 0.02   # scale to match histogram counts

# ── 3. Plot ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5.0, 3.4))

# Shade the forbidden zone (below P_cut)
ax.axvspan(0.15, P_CUT, color='#f7c6c6', alpha=0.55, zorder=0, label='Forbidden zone')

# Histogram – colour bars by region
colors_hist = ['#cc3333' if c < P_CUT else '#3a6fad' for c in bin_cens]
ax.bar(bin_cens, counts, width=0.019, color=colors_hist, alpha=0.80,
       edgecolor='white', linewidth=0.4, zorder=2)

# KDE overlay (only for P >= P_cut to not mislead)
mask_kde = kde_x >= P_CUT
ax.plot(kde_x[mask_kde], kde_y[mask_kde], color='#1a3f6f', lw=1.6,
        zorder=3, label='KDE envelope')

# Cutoff line
ax.axvline(P_CUT, color='#cc2222', lw=1.8, ls='--', zorder=4,
           label=r'$P_{\rm cut} = 0.22\,\rm d$')

# Annotation arrow + label
arrow_y = counts.max() * 0.62
ax.annotate(r'$P_{\rm cut} \approx 0.22\,\rm d$',
            xy=(P_CUT, arrow_y * 0.82),
            xytext=(P_CUT + 0.12, arrow_y),
            fontsize=8.5,
            arrowprops=dict(arrowstyle='->', color='#cc2222', lw=1.3),
            color='#cc2222', zorder=5)

# Axes
ax.set_xlabel(r'Orbital Period (days)', labelpad=4)
ax.set_ylabel(r'$N_{\rm EW}$ per 0.02-day bin', labelpad=4)
ax.set_xlim(0.15, 1.4)
ax.set_ylim(0, counts.max() * 1.22)
ax.xaxis.set_major_locator(MultipleLocator(0.2))
ax.xaxis.set_minor_locator(MultipleLocator(0.05))
ax.yaxis.set_major_locator(MultipleLocator(
    max(20, int(counts.max() // 5 / 10) * 10)))

# Inset: zoom into the cutoff region 0.17–0.30 d
ax_in = ax.inset_axes([0.58, 0.46, 0.38, 0.45])
mask_zoom = (periods_ew >= 0.17) & (periods_ew <= 0.30)
bins_zoom  = np.arange(0.17, 0.302, 0.01)
cnt_zoom, _ = np.histogram(periods_ew[mask_zoom], bins=bins_zoom)
bc_zoom     = 0.5 * (bins_zoom[:-1] + bins_zoom[1:])
col_zoom    = ['#cc3333' if c < P_CUT else '#3a6fad' for c in bc_zoom]
ax_in.bar(bc_zoom, cnt_zoom, width=0.0095, color=col_zoom, alpha=0.85,
          edgecolor='white', linewidth=0.3)
ax_in.axvline(P_CUT, color='#cc2222', lw=1.4, ls='--')
ax_in.axvspan(0.17, P_CUT, color='#f7c6c6', alpha=0.55)
ax_in.set_xlim(0.17, 0.30)
ax_in.set_xlabel(r'$P$ (d)', fontsize=7.5, labelpad=2)
ax_in.set_ylabel(r'$N$', fontsize=7.5, labelpad=2)
ax_in.tick_params(labelsize=7)
ax_in.xaxis.set_major_locator(MultipleLocator(0.04))
ax_in.set_title('Cutoff region', fontsize=7.5, pad=3)
ax.indicate_inset_zoom(ax_in, edgecolor='gray', alpha=0.6)

# Legend
handles = [
    mpatches.Patch(color='#3a6fad', alpha=0.80, label=fr'Kepler EWs ($N={len(periods_ew)}$)'),
    mpatches.Patch(color='#f7c6c6', alpha=0.7,  label='Forbidden zone'),
    plt.Line2D([0],[0], color='#cc2222', lw=1.8, ls='--',
               label=r'$P_{\rm cut}=0.22\,\rm d$'),
    plt.Line2D([0],[0], color='#1a3f6f', lw=1.6,
               label='KDE envelope'),
]
ax.legend(handles=handles, loc='lower right', framealpha=0.85,
          frameon=True, edgecolor='#bbbbbb', fontsize=7)

ax.text(0.01, 0.97, 'Kirk et al. (2016) / Kepler EB Catalog',
        transform=ax.transAxes, fontsize=7.5, va='top', color='#555555',
        style='italic')

plt.tight_layout(pad=0.4)
plt.savefig('fig1_period_distribution.pdf', bbox_inches='tight')
plt.savefig('fig1_period_distribution.png', bbox_inches='tight', dpi=200)
print("Saved fig1_period_distribution.pdf + .png")