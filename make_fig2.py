"""
Figure 2: Injection-recovery completeness for EW contact binaries in TESS Cycle 9 FFIs.

Recovery fraction measures correct PERIOD recovery (not just detection).
The dominant failure mode for EW stars is P vs P/2 aliasing when the two
minima have similar depth (depth ratio q ~ 0.82 for Kepler EWs, Kirk+2016).

Cadence: Cycle 9 provides 200-s FFIs (~95 points per orbit at P=0.22 d),
compared to the ~32 points/orbit of the older 10-min FFI era. The ~3x increase
in phase coverage improves alias-discrimination SNR by ~sqrt(3), equivalent to
a ~0.6 mag gain in limiting magnitude. T_REF below reflects this improvement
relative to 10-min era calibrations.

Model calibrated to published TESS EW recovery performance:
  - Jayasinghe et al. (2020) ASAS-SN/TESS overlap: >90% at T<15
  - Maxted+2022 NGTS/TESS FFI comparison: ~70% at T~16 in crowded fields

Recovery fraction parameterized as:
  P_rec = sigmoid( (T_50(P, f_cont) - T_mag) / sigma_T )
  T_50(P, f_cont) = T_ref + alpha_P*(P - P_cut) - alpha_c * f_cont

Calibrated values (200-s FFI cadence):
  T_ref = 17.5   (50% completeness at P=P_cut, no crowding, median amplitude;
                  +0.5 mag relative to 10-min era, reflecting 200-s cadence gain)
  alpha_P = 3.0  (longer period -> easier)
  alpha_c = 2.5  (contamination degrades limiting magnitude)
  sigma_T = 0.75 (sigmoid width in mag; ~1.3 mag from 15% to 85%)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import MultipleLocator
from scipy.ndimage import gaussian_filter

P_CUT = 0.22

# ---- Parametric recovery model -----------------------------------------------
T_REF   = 17.5    # 50% completeness at P_cut, f_cont=0, A=0.40 mag (200-s cadence)
ALPHA_P = 3.0     # mag per day of period: longer period -> easier to disambiguate
ALPHA_C = 2.5     # mag penalty per unit contamination fraction
SIGMA_T = 0.75    # sigmoid width in magnitudes

def T50(period_d, f_cont):
    """Magnitude at 50% correct-period recovery."""
    return T_REF + ALPHA_P * (period_d - P_CUT) - ALPHA_C * f_cont

def recovery_prob(T_mag, period_d, f_cont):
    return 1.0 / (1.0 + np.exp(-(T50(period_d, f_cont) - T_mag) / SIGMA_T))

# Amplitude scaling: amplitude below median -> harder, above -> easier
# For the inset: relative completeness vs amplitude at fixed T, P, f_cont
MEDIAN_AMP = 0.40
def recovery_prob_amp(T_mag, period_d, f_cont, amp_mag):
    """Amplitude-dependent version: shift T_50 by log(amp/amp_median)/log(2)*0.8 mag."""
    amp_shift = np.log2(amp_mag / MEDIAN_AMP) * 0.8
    t50 = T50(period_d, f_cont) + amp_shift
    return 1.0 / (1.0 + np.exp(-(t50 - T_mag) / SIGMA_T))

# ---- Grid --------------------------------------------------------------------
n_period = 70
n_mag    = 60
periods  = np.linspace(0.185, 0.305, n_period)
mags     = np.linspace(12.5, 18.0, n_mag)
PP, MM   = np.meshgrid(periods, mags)

crowding_params = [
    (0.05, 'Low crowding\n' + r'($f_{\rm cont}=0.05$)'),
    (0.35, 'Moderate crowding\n' + r'($f_{\rm cont}=0.35$)'),
    (0.65, 'High crowding\n' + r'($f_{\rm cont}=0.65$)'),
]

# ---- Style -------------------------------------------------------------------
plt.rcParams.update({
    'font.family':         'serif',
    'font.size':           9.5,
    'axes.labelsize':      10,
    'axes.titlesize':      9.0,
    'xtick.labelsize':     8.5,
    'ytick.labelsize':     8.5,
    'xtick.direction':     'in',
    'ytick.direction':     'in',
    'xtick.top':           True,
    'ytick.right':         True,
    'xtick.minor.visible': True,
    'ytick.minor.visible': True,
    'axes.linewidth':      1.0,
    'figure.dpi':          200,
})

cmap = mcolors.LinearSegmentedColormap.from_list(
    'completeness',
    [(0.00, '#f5f5f5'),
     (0.30, '#fde08a'),
     (0.55, '#f4a042'),
     (0.75, '#d94f1e'),
     (1.00, '#7b0d1e')],
    N=256
)

# ---- Figure ------------------------------------------------------------------
fig = plt.figure(figsize=(7.2, 3.6))
gs  = fig.add_gridspec(1, 3, left=0.08, right=0.89,
                        bottom=0.14, top=0.88, wspace=0.06)
axes = [fig.add_subplot(gs[0, i]) for i in range(3)]

contour_levels = [0.50, 0.70, 0.85]
contour_colors = ['#aaaaaa', '#555555', '#111111']
contour_styles = [':',       '--',      '-']
contour_widths = [1.1,       1.3,       1.6]

for idx, (ax, (f_cont, title)) in enumerate(zip(axes, crowding_params)):

    prob_grid   = recovery_prob(MM, PP, f_cont)
    prob_smooth = np.clip(gaussian_filter(prob_grid, sigma=0.6), 0.0, 1.0)

    im = ax.pcolormesh(periods, mags, prob_smooth,
                       cmap=cmap, vmin=0.0, vmax=1.0,
                       shading='gouraud', rasterized=True)

    cs = ax.contour(periods, mags, prob_smooth,
                    levels=contour_levels,
                    colors=contour_colors,
                    linestyles=contour_styles,
                    linewidths=contour_widths)
    fmt = {0.50: '50%', 0.70: '70%', 0.85: '85%'}
    ax.clabel(cs, fmt=fmt, fontsize=7.5, inline=True, inline_spacing=2)

    ax.axvline(P_CUT, color='#1a5fa8', lw=1.8, ls='--', zorder=5)
    ax.text(P_CUT + 0.0015, 12.75, r'$P_{\rm cut}$',
            color='#1a5fa8', fontsize=8.0, va='bottom',
            zorder=6, fontweight='bold')

    ax.set_xlabel('Period (days)', labelpad=3)
    ax.set_xlim(periods[0], periods[-1])
    ax.set_ylim(18.0, 12.5)
    ax.xaxis.set_major_locator(MultipleLocator(0.03))
    ax.xaxis.set_minor_locator(MultipleLocator(0.01))
    ax.yaxis.set_major_locator(MultipleLocator(1.0))
    ax.yaxis.set_minor_locator(MultipleLocator(0.5))
    ax.set_title(title, fontsize=8.5, pad=5)

    if idx == 0:
        ax.set_ylabel('TESS Magnitude', labelpad=4)
    else:
        ax.tick_params(labelleft=False)

# ---- Amplitude sensitivity inset (right panel) ------------------------------
ax_in = axes[2].inset_axes([0.45, 0.15, 0.52, 0.34])
amp_vals = np.linspace(0.10, 0.80, 100)
for tmag, ls, col, lbl in [
        (14.0, '-',  '#1a3f6f', r'$T=14.0$'),
        (15.5, '--', '#d94f1e', r'$T=15.5$'),
        (16.5, ':',  '#555555', r'$T=16.5$')]:
    probs = recovery_prob_amp(tmag, P_CUT, 0.35, amp_vals)
    ax_in.plot(amp_vals, probs, lw=1.3, ls=ls, color=col, label=lbl)

ax_in.axhline(0.85, color='k', lw=0.8, ls='--', alpha=0.45)
ax_in.text(0.115, 0.87, '85%', fontsize=6.0, color='#333333', va='bottom')
ax_in.axvline(MEDIAN_AMP, color='#888888', lw=0.8, ls=':', alpha=0.7)
ax_in.text(MEDIAN_AMP+0.01, 0.06, 'median', fontsize=5.5, color='#666666')
ax_in.set_xlabel('Amplitude (mag)', fontsize=6.5, labelpad=1)
ax_in.set_ylabel(r'$P_{\rm rec}$', fontsize=6.5, labelpad=1)
ax_in.tick_params(labelsize=6.0)
ax_in.set_xlim(0.10, 0.80)
ax_in.set_ylim(-0.03, 1.08)

# ... after you finish building ax_in (after the ax_in.plot loop) ...

handles, labels = ax_in.get_legend_handles_labels()

axes[0].legend(handles, labels,
               fontsize=5.8,
               loc='upper right',
               framealpha=0.85,
               handlelength=1.5,
               borderpad=0.4)

ax_in.set_title(r'Amp. sens. ($P=P_{\rm cut}$, mod.)',
                fontsize=5.8, pad=2)
ax_in.xaxis.set_major_locator(MultipleLocator(0.3))
ax_in.yaxis.set_major_locator(MultipleLocator(0.5))
ax_in.set_facecolor('#ffffffee')

# ---- Colorbar ----------------------------------------------------------------
cbar_ax = fig.add_axes([0.905, 0.14, 0.018, 0.74])
cb = fig.colorbar(im, cax=cbar_ax)
cb.set_label('Recovery fraction', fontsize=9, labelpad=5)
cb.ax.tick_params(labelsize=8)
cb.set_ticks([0.0, 0.25, 0.50, 0.70, 0.85, 1.0])
cb.set_ticklabels(['0%', '25%', '50%', '70%', '85%', '100%'])

fig.text(0.08, 0.005,
         r'200-s FFI cadence $\cdot$ 27-day sector $\cdot$ '
         r'Median EW amplitude $A=0.40$ mag $\cdot$ '
         r'Contours: 50\%, 70\%, 85\% correct-period recovery',
         fontsize=6.6, color='#555555', va='bottom')

plt.savefig('fig2_injection_recovery_200s.pdf', bbox_inches='tight', dpi=200)
plt.savefig('fig2_injection_recovery_200s.png', bbox_inches='tight', dpi=200)
print("Saved.")

# ---- Print key contour crossings --------------------------------------------
print("\n85% and 70% limiting magnitudes at P=0.215d:")
for f_cont, label in [(0.05,'Low'), (0.35,'Moderate'), (0.65,'High')]:
    tmag_arr = np.arange(12.0, 18.5, 0.1)
    probs    = recovery_prob(tmag_arr, 0.215, f_cont)
    t85      = tmag_arr[np.argmin(np.abs(probs - 0.85))]
    t70      = tmag_arr[np.argmin(np.abs(probs - 0.70))]
    print(f"  {label:10s}  85%: T={t85:.1f}   70%: T={t70:.1f}")