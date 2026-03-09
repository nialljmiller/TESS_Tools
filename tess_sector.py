#!/usr/bin/env python3
"""
tess_sector_overlay_both.py

Produces plots for TESS Cycle 9 (S108–S121):
  1) Equatorial Mollweide — TESS fields + catalogue points (POINT_STYLE)
  2) Galactic  Mollweide — TESS fields + catalogue points (POINT_STYLE)
  3) Equatorial Mollweide — TESS fields + points coloured by n_tess_obs
  4) Galactic  Mollweide — TESS fields + points coloured by n_tess_obs
  5) Histogram of n_tess_obs across all catalogue targets

Also computes n_tess_obs for every catalogue target, appends the column,
and writes the result back out.

Install:
  pip install tess-point astropy matplotlib numpy pandas
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import BoundaryNorm

from astropy.coordinates import SkyCoord, BarycentricTrueEcliptic
import astropy.units as u
from astropy.io import fits

from tess_stars2px import (
    tess_stars2px_reverse_function_entry,
    tess_stars2px_function_entry,
    TESS_Spacecraft_Pointing_Data,
)


# ============================================================
# POINT STYLE DICTIONARY
# Controls how catalogue points look in the standard overlay plots.
# ============================================================
POINT_STYLE = {
    "size":           100.0,   # marker size (pts^2)
    "marker":         "*",    # "o" dot | "*" star | "+" plus | "x" cross | "s" square
    "color":          "red",  # fixed colour -- ignored when colorbar_col is set
    "alpha":          0.9,
    "colorbar_col":   None,   # column name to use as colourbar instead of fixed colour
    "colorbar_label": "Value",
    "colorbar_cmap":  "plasma",
}
# ============================================================


# ---------------------------
# Projection helpers
# ---------------------------

def ra_to_mollweide_x(ra_deg):
    ra = np.asarray(ra_deg, dtype=float) % 360.0
    ra_rad = np.deg2rad(ra)
    ra_rad = (ra_rad + np.pi) % (2 * np.pi) - np.pi
    return -ra_rad


def lon_to_mollweide_x(lon_deg):
    lon = np.asarray(lon_deg, dtype=float) % 360.0
    lon_rad = np.deg2rad(lon)
    lon_rad = (lon_rad + np.pi) % (2 * np.pi) - np.pi
    return -lon_rad


def lat_to_mollweide_y(lat_deg):
    return np.deg2rad(np.asarray(lat_deg, dtype=float))


def break_wrap(x, y, jump_rad=np.pi * 0.95):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) == 0:
        return x, y
    outx, outy = [x[0]], [y[0]]
    for i in range(1, len(x)):
        if abs(x[i] - x[i - 1]) > jump_rad:
            outx.append(np.nan)
            outy.append(np.nan)
        outx.append(x[i])
        outy.append(y[i])
    return np.array(outx), np.array(outy)


# ---------------------------
# TESS footprint helpers
# ---------------------------

def ccd_outline_radec(sector, cam, ccd, scinfo, n_edge=1):
    if n_edge <= 1:
        pix = [(1, 1), (2048, 1), (2048, 2048), (1, 2048), (1, 1)]
    else:
        n = int(n_edge)
        cols = np.linspace(1, 2048, n)
        rows = np.linspace(1, 2048, n)
        bottom = [(c, 1) for c in cols]
        right  = [(2048, r) for r in rows[1:]]
        top    = [(c, 2048) for c in cols[::-1][1:]]
        left   = [(1, r) for r in rows[::-1][1:]]
        pix    = bottom + right + top + left + [bottom[0]]

    ras, decs = [], []
    for col, row in pix:
        ra, dec, _ = tess_stars2px_reverse_function_entry(sector, cam, ccd, col, row, scInfo=scinfo)
        ras.append(ra)
        decs.append(dec)
    return np.array(ras), np.array(decs)


def write_cycle9_override(path):
    cycle9_table = [
        (108, 66.2663,  -25.1811, 261.1568),
        (109, 86.3820,  -22.6679, 249.8954),
        (110, 106.0743, -23.7944, 238.6947),
        (111, 125.4703, -28.2718, 228.4036),
        (112, 145.1712, -35.6981, 219.7678),
        (113, 166.4824, -45.5270, 213.9682),
        (110, 192.2167, -56.6197, 213.7675),
        (115, 219.6384, -16.0606, 251.3437),
        (116, 288.1479, -23.3419, 276.7904),
        (117, 251.5610,  32.2829, 347.6386),
        (118, 269.0677,  30.5688, 359.3655),
        (119, 274.1880,  61.9047,  19.4458),
        (120, 278.1539,  62.9604,  40.3192),
        (121, 281.2282,  64.6669,  62.6836),
    ]
    with path.open("w") as f:
        for sec, ra, dec, roll in cycle9_table:
            f.write(f"{sec:d} {ra:.6f} {dec:.6f} {roll:.6f}\n")
    return [r[0] for r in cycle9_table]


def get_scinfo_for_sector(sec, scinfo_global):
    if scinfo_global is not None:
        return scinfo_global
    _, _, scinfo = tess_stars2px_reverse_function_entry(sec, 1, 1, 1, 1, scInfo=None)
    return scinfo


# ---------------------------
# Catalogue loader
# Returns full DataFrame so we can append n_tess_obs and write it back.
# ---------------------------

def load_catalogue(cat_path, ra_col, dec_col, fits_ext=1):
    """Load catalogue as a DataFrame from FITS or CSV.
    Returns (df, ra_array, dec_array) or (None, None, None) if not found."""
    cat_path = Path(cat_path).expanduser()
    if not cat_path.exists():
        print(f"[warn] Catalogue not found: {cat_path}")
        return None, None, None

    try:
        with fits.open(cat_path) as hdul:
            data = hdul[fits_ext].data
            if data is None:
                raise ValueError("Empty HDU")
            df = pd.DataFrame({col: np.asarray(data[col]) for col in data.columns.names})
        print(f"Loaded FITS catalogue: {cat_path}  ({len(df):,} rows)")
    except Exception as e_fits:
        try:
            df = pd.read_csv(cat_path)
            print(f"Loaded CSV catalogue: {cat_path}  ({len(df):,} rows)")
        except Exception as e_csv:
            print(f"[warn] Could not load catalogue.\n  FITS error: {e_fits}\n  CSV error: {e_csv}")
            return None, None, None

    for col in (ra_col, dec_col):
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found. Available: {df.columns.tolist()}")

    ra  = np.asarray(df[ra_col],  dtype=float)
    dec = np.asarray(df[dec_col], dtype=float)
    mask = np.isfinite(ra) & np.isfinite(dec)
    return df[mask].reset_index(drop=True), ra[mask], dec[mask]


# ---------------------------
# Observation counting
# ---------------------------

def count_observations_for_catalogue(ra_arr, dec_arr, sectors, scinfo_global):
    """
    For every target in ra_arr/dec_arr, count how many unique Cycle 9
    sectors observe it.  Returns an int array of length N.
    """
    sector_set = set(sectors)
    n_obs = np.zeros(len(ra_arr), dtype=int)
    for i, (ra, dec) in enumerate(zip(ra_arr, dec_arr)):
        _, _, _, outSec, _, _, _, _, _ = tess_stars2px_function_entry(
            i, ra, dec, scInfo=scinfo_global
        )
        if outSec is not None and len(outSec) > 0:
            n_obs[i] = len({int(s) for s in outSec if int(s) in sector_set})
    return n_obs


# ---------------------------
# Internal field-drawing helpers
# ---------------------------

def _draw_tess_fields_equatorial(ax, sectors, scinfo_global, cmap, norm):
    lam  = np.linspace(0, 360, 1500) * u.deg
    beta = np.zeros(1500) * u.deg
    ecl  = SkyCoord(lon=lam, lat=beta, distance=1*u.au,
                    frame=BarycentricTrueEcliptic).transform_to("icrs")
    ax.plot(ra_to_mollweide_x(ecl.ra.deg), lat_to_mollweide_y(ecl.dec.deg),
            linewidth=5, alpha=0.35)
    for sec in sectors:
        color  = cmap(norm(sec))
        scinfo = get_scinfo_for_sector(sec, scinfo_global)
        for cam in range(1, 5):
            for ccd in range(1, 5):
                ras, decs = ccd_outline_radec(sec, cam, ccd, scinfo, n_edge=1)
                x, y = break_wrap(ra_to_mollweide_x(ras), lat_to_mollweide_y(decs))
                ax.plot(x, y, linewidth=0.8, color=color, alpha=0.9)


def _draw_tess_fields_galactic(ax, sectors, scinfo_global, cmap, norm):
    l = np.linspace(0, 360, 1500)
    ax.plot(lon_to_mollweide_x(l), lat_to_mollweide_y(np.zeros_like(l)),
            linewidth=5, alpha=0.35)
    for sec in sectors:
        color  = cmap(norm(sec))
        scinfo = get_scinfo_for_sector(sec, scinfo_global)
        for cam in range(1, 5):
            for ccd in range(1, 5):
                ras, decs = ccd_outline_radec(sec, cam, ccd, scinfo, n_edge=25)
                g = SkyCoord(ra=ras*u.deg, dec=decs*u.deg, frame="icrs").galactic
                x, y = break_wrap(lon_to_mollweide_x(g.l.deg), lat_to_mollweide_y(g.b.deg))
                ax.plot(x, y, linewidth=0.8, color=color, alpha=0.9)


def _add_stacked_colorbars(fig, ax, sectors, sec_cmap, sec_norm, sc_nobs, max_obs, nobs_cmap, nobs_norm):
    """Two colorbars stacked at half height on the right — sector on top, nobs on bottom."""
    # [left, bottom, width, height] in figure coordinates
    cax_top = fig.add_axes([0.935, 0.52, 0.018, 0.44])
    cax_bot = fig.add_axes([0.935, 0.05, 0.018, 0.44])

    sm = mpl.cm.ScalarMappable(norm=sec_norm, cmap=sec_cmap)
    sm.set_array([])
    cb_top = fig.colorbar(sm, cax=cax_top, orientation="vertical")
    cb_top.set_label("TESS Sector", fontsize=8)
    cb_top.set_ticks(sectors)
    cb_top.ax.tick_params(labelsize=7)

    cb_bot = fig.colorbar(sc_nobs, cax=cax_bot, norm=nobs_norm, orientation="vertical")
    cb_bot.set_label("N TESS obs", fontsize=8)
    cb_bot.set_ticks(np.arange(0, max_obs + 1))
    cb_bot.ax.tick_params(labelsize=7)


def add_sector_colorbar(fig, ax, sectors, cmap, norm):
    """Single full-height sector colorbar — used when there is no nobs colorbar."""
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cax = fig.add_axes([0.935, 0.05, 0.018, 0.91])
    cbar = fig.colorbar(sm, cax=cax, orientation="vertical")
    cbar.set_label("TESS Sector", fontsize=8)
    cbar.set_ticks(sectors)
    cbar.ax.tick_params(labelsize=7)


# ---------------------------
# Standard overlay plots (POINT_STYLE colouring)
# ---------------------------

def _scatter_point_style(ax, px, py, cvals):
    s      = POINT_STYLE["size"]
    marker = POINT_STYLE["marker"]
    alpha  = POINT_STYLE["alpha"]
    if cvals is not None and POINT_STYLE["colorbar_col"] is not None:
        sc = ax.scatter(px, py, s=s, c=cvals, cmap=POINT_STYLE["colorbar_cmap"],
                        marker=marker, alpha=alpha, linewidths=0)
        cb = plt.colorbar(sc, ax=ax, orientation="vertical", pad=0.02, fraction=0.03)
        cb.set_label(POINT_STYLE["colorbar_label"])
    else:
        ax.scatter(px, py, s=s, color=POINT_STYLE["color"],
                   marker=marker, alpha=alpha, linewidths=0)


def plot_equatorial(sectors, scinfo_global, cmap, norm, ra, dec, cvals):
    fig = plt.figure(figsize=(10, 4.8), dpi=220)
    ax  = fig.add_subplot(111, projection="mollweide")
    fig.subplots_adjust(left=0.03, right=0.92, top=0.97, bottom=0.03)
    ax.grid(True, linewidth=0.6, alpha=0.6)
    _draw_tess_fields_equatorial(ax, sectors, scinfo_global, cmap, norm)
    add_sector_colorbar(fig, ax, sectors, cmap, norm)
    if ra is not None:
        _scatter_point_style(ax, ra_to_mollweide_x(ra), lat_to_mollweide_y(dec), cvals)
    return fig


def plot_galactic(sectors, scinfo_global, cmap, norm, ra, dec, cvals):
    fig = plt.figure(figsize=(10, 4.8), dpi=220)
    ax  = fig.add_subplot(111, projection="mollweide")
    fig.subplots_adjust(left=0.03, right=0.92, top=0.97, bottom=0.03)
    ax.grid(True, linewidth=0.6, alpha=0.6)
    _draw_tess_fields_galactic(ax, sectors, scinfo_global, cmap, norm)
    add_sector_colorbar(fig, ax, sectors, cmap, norm)
    if ra is not None:
        g = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame="icrs").galactic
        _scatter_point_style(ax, lon_to_mollweide_x(g.l.deg), lat_to_mollweide_y(g.b.deg), cvals)
    return fig


# ---------------------------
# n_obs overlay plots (discrete colourbar)
# ---------------------------

def _make_nobs_cmap_norm(max_obs):
    nobs_cmap = plt.get_cmap("jet", max_obs + 1)
    nobs_norm = BoundaryNorm(np.arange(-0.5, max_obs + 1.5), nobs_cmap.N)
    return nobs_cmap, nobs_norm


def _scatter_nobs(ax, px, py, n_obs, nobs_cmap, nobs_norm):
    return ax.scatter(px, py,
                      s=POINT_STYLE["size"], c=n_obs,
                      cmap=nobs_cmap, norm=nobs_norm,
                      marker=POINT_STYLE["marker"],
                      alpha=POINT_STYLE["alpha"],
                      linewidths=0)


def plot_equatorial_nobs(sectors, scinfo_global, cmap, norm, ra, dec, n_obs):
    max_obs              = max(int(n_obs.max()), 1)
    nobs_cmap, nobs_norm = _make_nobs_cmap_norm(max_obs)

    fig = plt.figure(figsize=(10, 4.8), dpi=220)
    ax  = fig.add_subplot(111, projection="mollweide")
    fig.subplots_adjust(left=0.03, right=0.91, top=0.97, bottom=0.03)
    ax.grid(True, linewidth=0.6, alpha=0.6)
    _draw_tess_fields_equatorial(ax, sectors, scinfo_global, cmap, norm)
    sc = _scatter_nobs(ax, ra_to_mollweide_x(ra), lat_to_mollweide_y(dec),
                       n_obs, nobs_cmap, nobs_norm)
    _add_stacked_colorbars(fig, ax, sectors, cmap, norm, sc, max_obs, nobs_cmap, nobs_norm)
    return fig


def plot_galactic_nobs(sectors, scinfo_global, cmap, norm, ra, dec, n_obs):
    max_obs              = max(int(n_obs.max()), 1)
    nobs_cmap, nobs_norm = _make_nobs_cmap_norm(max_obs)

    fig = plt.figure(figsize=(10, 4.8), dpi=220)
    ax  = fig.add_subplot(111, projection="mollweide")
    fig.subplots_adjust(left=0.03, right=0.91, top=0.97, bottom=0.03)
    ax.grid(True, linewidth=0.6, alpha=0.6)
    _draw_tess_fields_galactic(ax, sectors, scinfo_global, cmap, norm)
    g  = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame="icrs").galactic
    sc = _scatter_nobs(ax, lon_to_mollweide_x(g.l.deg), lat_to_mollweide_y(g.b.deg),
                       n_obs, nobs_cmap, nobs_norm)
    _add_stacked_colorbars(fig, ax, sectors, cmap, norm, sc, max_obs, nobs_cmap, nobs_norm)
    return fig



# ---------------------------
# n_obs only plots (single nobs colorbar, no sector colorbar)
# ---------------------------

def plot_equatorial_nobs_only(sectors, scinfo_global, cmap, norm, ra, dec, n_obs):
    max_obs              = max(int(n_obs.max()), 1)
    nobs_cmap, nobs_norm = _make_nobs_cmap_norm(max_obs)

    fig = plt.figure(figsize=(10, 4.8), dpi=220)
    ax  = fig.add_subplot(111, projection="mollweide")
    fig.subplots_adjust(left=0.03, right=0.92, top=0.97, bottom=0.03)
    ax.grid(True, linewidth=0.6, alpha=0.6)
    _draw_tess_fields_equatorial(ax, sectors, scinfo_global, cmap, norm)
    sc = _scatter_nobs(ax, ra_to_mollweide_x(ra), lat_to_mollweide_y(dec),
                       n_obs, nobs_cmap, nobs_norm)
    cax = fig.add_axes([0.935, 0.05, 0.018, 0.91])
    cb  = fig.colorbar(sc, cax=cax, orientation="vertical")
    cb.set_label("N TESS obs", fontsize=8)
    cb.set_ticks(np.arange(0, max_obs + 1))
    cb.ax.tick_params(labelsize=7)
    return fig


def plot_galactic_nobs_only(sectors, scinfo_global, cmap, norm, ra, dec, n_obs):
    max_obs              = max(int(n_obs.max()), 1)
    nobs_cmap, nobs_norm = _make_nobs_cmap_norm(max_obs)

    fig = plt.figure(figsize=(10, 4.8), dpi=220)
    ax  = fig.add_subplot(111, projection="mollweide")
    fig.subplots_adjust(left=0.03, right=0.92, top=0.97, bottom=0.03)
    ax.grid(True, linewidth=0.6, alpha=0.6)
    _draw_tess_fields_galactic(ax, sectors, scinfo_global, cmap, norm)
    g  = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame="icrs").galactic
    sc = _scatter_nobs(ax, lon_to_mollweide_x(g.l.deg), lat_to_mollweide_y(g.b.deg),
                       n_obs, nobs_cmap, nobs_norm)
    cax = fig.add_axes([0.935, 0.05, 0.018, 0.91])
    cb  = fig.colorbar(sc, cax=cax, orientation="vertical")
    cb.set_label("N TESS obs", fontsize=8)
    cb.set_ticks(np.arange(0, max_obs + 1))
    cb.ax.tick_params(labelsize=7)
    return fig



def plot_nobs_histogram(n_obs):
    max_obs = int(n_obs.max())
    bins    = np.arange(-0.5, max_obs + 1.5)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=180)
    ax.hist(n_obs, bins=bins, color="steelblue", edgecolor="white", linewidth=0.6)
    ax.set_xlabel("N TESS Cycle 9 sectors observing target")
    ax.set_ylabel("Number of targets")
    ax.set_title("Distribution of TESS Cycle 9 observations")
    ax.set_xticks(np.arange(0, max_obs + 1))
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    total    = len(n_obs)
    observed1 = int((n_obs > 1).sum())
    observed2 = int((n_obs > 2).sum())
    observed3 = int((n_obs > 3).sum())
    ax.text(0.98, 0.97,
            f"Total targets : {total:,}\nObserved >=2x : {observed1:,}  ({100*observed1/total:.1f}%)\nObserved >=3x : {observed2:,}  ({100*observed2/total:.1f}%)\nObserved >=4x : {observed3:,}  ({100*observed3/total:.1f}%)",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9, bbox=dict(boxstyle="round", fc="white", alpha=0.8))
    plt.tight_layout()
    return fig


# ---------------------------
# Write catalogue with n_obs appended
# ---------------------------

def write_catalogue_with_nobs(df, n_obs, cat_path, out_col="n_tess_cycle9_obs"):
    df = df.copy()
    df[out_col] = n_obs
    cat_path = Path(cat_path).expanduser()
    out_path = cat_path.parent / (cat_path.stem + "_tess_nobs" + cat_path.suffix)

    if cat_path.suffix.lower() in (".fits", ".fit"):
        from astropy.table import Table
        t = Table.from_pandas(df)
        t.write(str(out_path), overwrite=True)
    else:
        df.to_csv(out_path, index=False)

    print(f"Wrote catalogue with n_obs column: {out_path.resolve()}")
    return out_path


# ============================================================
# MAIN — edit the params dict here, no CLI arguments
# ============================================================

def main():

    # -----------------------------------------------------------
    # PARAMS — everything you need to change lives here
    # -----------------------------------------------------------
    params = {
        # True  -> use tess-point's internal Cycle 9 pointings
        # False -> override with hand-tabulated PDF values
        "use_internal_pointings": True,

        # Input catalogue (FITS or CSV)
        "catalogue": "tess_holygrail_sectorcheck.csv",
        "fits_ext":  1,        # HDU index (ignored for CSV)
        "ra_col":    "ra1",    # column name for RA  [deg]
        "dec_col":   "dec1",   # column name for Dec [deg]

        # Output image base name (suffixes added automatically)
        "out": "tess_cycle9_pointings_overlay.png",
    }
    # -----------------------------------------------------------

    # Sector list and pointing info
    if params["use_internal_pointings"]:
        sectors       = list(range(108, 122))
        scinfo_global = None
    else:
        override_path = Path("cycle9_sector_override.txt")
        sectors       = write_cycle9_override(override_path)
        scinfo_global = TESS_Spacecraft_Pointing_Data(sectorOverrideFile=str(override_path))

    cmap = plt.get_cmap("viridis", len(sectors))
    norm = mpl.colors.Normalize(vmin=min(sectors), vmax=max(sectors))

    out_base = Path(params["out"])
    stem     = out_base.stem
    parent   = Path(str(out_base.parent)) if str(out_base.parent) != "." else Path(".")

    # Load catalogue
    df, ra, dec = load_catalogue(
        params["catalogue"], params["ra_col"], params["dec_col"], params["fits_ext"]
    )

    # cvals for POINT_STYLE colourbar
    cvals = None
    if df is not None and POINT_STYLE["colorbar_col"] is not None:
        col = POINT_STYLE["colorbar_col"]
        if col in df.columns:
            cvals = np.asarray(df[col], dtype=float)
        else:
            print(f"[warn] colorbar_col '{col}' not found in catalogue -- using fixed colour")

    # Count TESS observations for every target
    n_obs = None
    if ra is not None:
        print(f"Computing TESS Cycle 9 observations for {len(ra):,} targets ...")
        n_obs = count_observations_for_catalogue(ra, dec, sectors, scinfo_global)
        print(f"  Done.  {(n_obs > 0).sum():,} targets observed at least once.")

    # Standard overlay figures
    fig1 = plot_equatorial(sectors, scinfo_global, cmap, norm, ra, dec, cvals)
    p = parent / f"{stem}_equatorial.png"
    fig1.savefig(p, dpi=220);  plt.close(fig1);  print(f"Wrote: {p.resolve()}")

    fig2 = plot_galactic(sectors, scinfo_global, cmap, norm, ra, dec, cvals)
    p = parent / f"{stem}_galactic.png"
    fig2.savefig(p, dpi=220);  plt.close(fig2);  print(f"Wrote: {p.resolve()}")

    if n_obs is not None:
        # n_obs overlay figures
        fig3 = plot_equatorial_nobs(sectors, scinfo_global, cmap, norm, ra, dec, n_obs)
        p = parent / f"{stem}_equatorial_nobs.png"
        fig3.savefig(p, dpi=220);  plt.close(fig3);  print(f"Wrote: {p.resolve()}")

        fig4 = plot_galactic_nobs(sectors, scinfo_global, cmap, norm, ra, dec, n_obs)
        p = parent / f"{stem}_galactic_nobs.png"
        fig4.savefig(p, dpi=220);  plt.close(fig4);  print(f"Wrote: {p.resolve()}")

        # n_obs only figures (single nobs colorbar, no sector colorbar)
        fig6 = plot_equatorial_nobs_only(sectors, scinfo_global, cmap, norm, ra, dec, n_obs)
        p = parent / f"{stem}_equatorial_nobs_only.png"
        fig6.savefig(p, dpi=220);  plt.close(fig6);  print(f"Wrote: {p.resolve()}")

        fig7 = plot_galactic_nobs_only(sectors, scinfo_global, cmap, norm, ra, dec, n_obs)
        p = parent / f"{stem}_galactic_nobs_only.png"
        fig7.savefig(p, dpi=220);  plt.close(fig7);  print(f"Wrote: {p.resolve()}")

        # Histogram
        fig5 = plot_nobs_histogram(n_obs)
        p = parent / f"{stem}_nobs_histogram.png"
        fig5.savefig(p, dpi=180);  plt.close(fig5);  print(f"Wrote: {p.resolve()}")

        # Write catalogue with n_obs column appended
        if df is not None:
            write_catalogue_with_nobs(df, n_obs, params["catalogue"])


if __name__ == "__main__":
    main()