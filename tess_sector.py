#!/usr/bin/env python3
"""
tess_sector_overlay_both.py

Always produces TWO plots for TESS Cycle 9 (S108–S121):
  1) Equatorial (RA/Dec) Mollweide with ecliptic plane
  2) Galactic (l/b) Mollweide with galactic plane

Both plots:
- show CCD outlines (no convex hulls)
- avoid "teleport lines" by breaking polylines at wrap boundaries
- optionally overlay a FITS table of points (RA/Dec columns in degrees)

Install:
  python -m pip install --user --upgrade tess-point astropy matplotlib numpy

Run (defaults match your workflow):
  python tess_sector_overlay_both.py

Outputs (by default):
  tess_cycle9_pointings_overlay_equatorial.png
  tess_cycle9_pointings_overlay_galactic.png
"""

from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

from astropy.coordinates import SkyCoord, BarycentricTrueEcliptic
import astropy.units as u
from astropy.io import fits

from tess_stars2px import tess_stars2px_reverse_function_entry, tess_stars2px_function_entry, TESS_Spacecraft_Pointing_Data


# ============================================================
# POINT STYLE DICTIONARY
# Edit this block to control how catalogue points are plotted.
# ============================================================
POINT_STYLE = {
    # Fixed marker size (pts²)
    "size": 10.0,

    # Marker shape — any matplotlib marker string, e.g.:
    #   "o" dot, "*" star, "+" plus, "x" cross, "^" triangle, "s" square
    "marker": "*",

    # Fixed marker colour — used when colorbar_col is None
    "color": "red",

    # Marker transparency [0–1]
    "alpha": 0.6,

    # Set to a column name (string) to colour points by that column
    # and show a colourbar instead of a fixed colour.
    # Set to None to use the fixed colour above.
    "colorbar_col": None,          # e.g. "Ks_mag" or None

    # Colourbar label (only used when colorbar_col is not None)
    "colorbar_label": "Value",

    # Colourmap used for the colourbar (any matplotlib cmap name)
    "colorbar_cmap": "plasma",
}
# ============================================================


# ---------------------------
# Projection helpers
# ---------------------------

def ra_to_mollweide_x(ra_deg: np.ndarray) -> np.ndarray:
    """RA [deg] -> Mollweide x [rad], with RA increasing to the LEFT."""
    ra = np.asarray(ra_deg, dtype=float) % 360.0
    ra_rad = np.deg2rad(ra)
    ra_rad = (ra_rad + np.pi) % (2 * np.pi) - np.pi
    return -ra_rad


def lon_to_mollweide_x(lon_deg: np.ndarray) -> np.ndarray:
    """Longitude (l) [deg] -> Mollweide x [rad], increasing to the LEFT."""
    lon = np.asarray(lon_deg, dtype=float) % 360.0
    lon_rad = np.deg2rad(lon)
    lon_rad = (lon_rad + np.pi) % (2 * np.pi) - np.pi
    return -lon_rad


def lat_to_mollweide_y(lat_deg: np.ndarray) -> np.ndarray:
    """Latitude (Dec or b) [deg] -> Mollweide y [rad]."""
    return np.deg2rad(np.asarray(lat_deg, dtype=float))


def break_wrap(x: np.ndarray, y: np.ndarray, jump_rad: float = np.pi * 0.95) -> tuple[np.ndarray, np.ndarray]:
    """
    Insert NaNs when x jumps across the wrap boundary.
    Prevents Matplotlib drawing a straight line across the whole sky.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) == 0:
        return x, y
    outx = [x[0]]
    outy = [y[0]]
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

def ccd_outline_radec(sector: int, cam: int, ccd: int, scinfo, n_edge: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """
    CCD outline as RA,Dec arrays (deg).

    n_edge=1: corners only (fast)
    n_edge>1: sample edges for smoother outlines after coordinate transforms
    """
    if n_edge <= 1:
        pix = [(1, 1), (2048, 1), (2048, 2048), (1, 2048), (1, 1)]
    else:
        n = int(n_edge)
        cols = np.linspace(1, 2048, n)
        rows = np.linspace(1, 2048, n)

        bottom = [(c, 1) for c in cols]
        right = [(2048, r) for r in rows[1:]]
        top = [(c, 2048) for c in cols[::-1][1:]]
        left = [(1, r) for r in rows[::-1][1:]]
        pix = bottom + right + top + left + [bottom[0]]

    ras, decs = [], []
    for col, row in pix:
        ra, dec, _ = tess_stars2px_reverse_function_entry(sector, cam, ccd, col, row, scInfo=scinfo)
        ras.append(ra)
        decs.append(dec)
    return np.array(ras), np.array(decs)


def write_cycle9_override(path: Path) -> list[int]:
    """Cycle 9 table from the AGIGO PDF (sector, RA, Dec, Roll)."""
    cycle9_table = [
        (108, 66.2663, -25.1811, 261.1568),
        (109, 86.3820, -22.6679, 249.8954),
        (110, 106.0743, -23.7944, 238.6947),
        (111, 125.4703, -28.2718, 228.4036),
        (112, 145.1712, -35.6981, 219.7678),
        (113, 166.4824, -45.5270, 213.9682),
        (114, 192.2167, -56.6197, 213.7675),
        (115, 219.6384, -16.0606, 251.3437),
        (116, 288.1479, -23.3419, 276.7904),
        (117, 251.5614,  32.2829, 347.6386),
        (118, 269.0677,  30.5688, 359.3655),
        (119, 274.1880,  61.9047,  19.4458),
        (120, 278.1539,  62.9604,  40.3192),
        (121, 281.2282,  64.6669,  62.6836),
    ]
    with path.open("w") as f:
        for sec, ra, dec, roll in cycle9_table:
            f.write(f"{sec:d} {ra:.6f} {dec:.6f} {roll:.6f}\n")
    return [r[0] for r in cycle9_table]


def get_scinfo_for_sector(sec: int, scinfo_global):
    """Return a pointing object usable for this sector."""
    if scinfo_global is not None:
        return scinfo_global
    _, _, scinfo = tess_stars2px_reverse_function_entry(sec, 1, 1, 1, 1, scInfo=None)
    return scinfo


# ---------------------------
# Catalogue loaders
# ---------------------------

def load_points_from_fits(
    fits_path: Path,
    ra_col: str,
    dec_col: str,
    c_col: str | None,
    ext: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    with fits.open(fits_path) as hdul:
        data = hdul[ext].data
        if data is None:
            raise ValueError(f"No table data in HDU {ext} of {fits_path}")
        names = list(data.columns.names)
        if ra_col not in names:
            raise KeyError(f"RA column '{ra_col}' not found. Columns: {names}")
        if dec_col not in names:
            raise KeyError(f"Dec column '{dec_col}' not found. Columns: {names}")
        ra  = np.asarray(data[ra_col],  dtype=float)
        dec = np.asarray(data[dec_col], dtype=float)
        cvals = None
        if c_col is not None:
            if c_col not in names:
                raise KeyError(f"Color column '{c_col}' not found. Columns: {names}")
            cvals = np.asarray(data[c_col], dtype=float)
    m = np.isfinite(ra) & np.isfinite(dec)
    if cvals is not None:
        m &= np.isfinite(cvals)
        cvals = cvals[m]
    return ra[m], dec[m], cvals


def load_points_from_csv(
    csv_path: Path,
    ra_col: str,
    dec_col: str,
    c_col: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    df = pd.read_csv(csv_path)
    names = df.columns.tolist()
    if ra_col not in names:
        raise KeyError(f"RA column '{ra_col}' not found. Columns: {names}")
    if dec_col not in names:
        raise KeyError(f"Dec column '{dec_col}' not found. Columns: {names}")
    ra  = np.asarray(df[ra_col],  dtype=float)
    dec = np.asarray(df[dec_col], dtype=float)
    cvals = None
    if c_col is not None:
        if c_col not in names:
            raise KeyError(f"Color column '{c_col}' not found. Columns: {names}")
        cvals = np.asarray(df[c_col], dtype=float)
    m = np.isfinite(ra) & np.isfinite(dec)
    if cvals is not None:
        m &= np.isfinite(cvals)
        cvals = cvals[m]
    return ra[m], dec[m], cvals


# ---------------------------
# Plot helpers
# ---------------------------

def add_sector_colorbar(fig, ax, sectors, cmap, norm):
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation="horizontal", pad=0.08, fraction=0.06)
    cbar.set_label("TESS Sector (Cycle 9)")
    cbar.set_ticks(sectors)


def scatter_points(ax, px, py, cvals):
    """Draw catalogue points using POINT_STYLE."""
    s     = POINT_STYLE["size"]
    alpha = POINT_STYLE["alpha"]

    marker = POINT_STYLE["marker"]

    if cvals is not None and POINT_STYLE["colorbar_col"] is not None:
        sc = ax.scatter(
            px, py,
            s=s, c=cvals,
            cmap=POINT_STYLE["colorbar_cmap"],
            marker=marker,
            alpha=alpha,
            linewidths=0,
        )
        cb = plt.colorbar(sc, ax=ax, orientation="vertical", pad=0.02, fraction=0.03)
        cb.set_label(POINT_STYLE["colorbar_label"])
    else:
        ax.scatter(
            px, py,
            s=s,
            color=POINT_STYLE["color"],
            marker=marker,
            alpha=alpha,
            linewidths=0,
        )


def plot_equatorial(sectors, scinfo_global, cmap, norm, points=None):
    fig = plt.figure(figsize=(14, 6), dpi=220)
    ax  = fig.add_subplot(111, projection="mollweide")
    ax.grid(True, linewidth=0.6, alpha=0.6)

    # ecliptic plane
    lam = np.linspace(0, 360, 1500) * u.deg
    beta = np.zeros_like(lam.value) * u.deg
    ecl = SkyCoord(lon=lam, lat=beta, distance=1 * u.au, frame=BarycentricTrueEcliptic).transform_to("icrs")
    ax.plot(ra_to_mollweide_x(ecl.ra.deg), lat_to_mollweide_y(ecl.dec.deg), linewidth=5, alpha=0.35)

    for sec in sectors:
        color  = cmap(norm(sec))
        scinfo = get_scinfo_for_sector(sec, scinfo_global)
        for cam in range(1, 5):
            for ccd in range(1, 5):
                ras, decs = ccd_outline_radec(sec, cam, ccd, scinfo, n_edge=1)
                x, y = break_wrap(ra_to_mollweide_x(ras), lat_to_mollweide_y(decs))
                ax.plot(x, y, linewidth=0.8, color=color, alpha=0.9)

    add_sector_colorbar(fig, ax, sectors, cmap, norm)

    if points is not None:
        ra, dec, cvals = points
        scatter_points(ax, ra_to_mollweide_x(ra), lat_to_mollweide_y(dec), cvals)

    return fig


def plot_galactic(sectors, scinfo_global, cmap, norm, points=None):
    fig = plt.figure(figsize=(14, 6), dpi=220)
    ax  = fig.add_subplot(111, projection="mollweide")
    ax.grid(True, linewidth=0.6, alpha=0.6)

    # galactic plane (b=0)
    l = np.linspace(0, 360, 1500)
    b = np.zeros_like(l)
    ax.plot(lon_to_mollweide_x(l), lat_to_mollweide_y(b), linewidth=5, alpha=0.35)

    for sec in sectors:
        color  = cmap(norm(sec))
        scinfo = get_scinfo_for_sector(sec, scinfo_global)
        for cam in range(1, 5):
            for ccd in range(1, 5):
                ras, decs = ccd_outline_radec(sec, cam, ccd, scinfo, n_edge=25)
                g = SkyCoord(ra=ras * u.deg, dec=decs * u.deg, frame="icrs").galactic
                x, y = break_wrap(lon_to_mollweide_x(g.l.deg), lat_to_mollweide_y(g.b.deg))
                ax.plot(x, y, linewidth=0.8, color=color, alpha=0.9)

    add_sector_colorbar(fig, ax, sectors, cmap, norm)

    if points is not None:
        ra, dec, cvals = points
        g = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs").galactic
        scatter_points(ax, lon_to_mollweide_x(g.l.deg), lat_to_mollweide_y(g.b.deg), cvals)

    return fig


# ---------------------------
# Main
# ---------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-override", default=False, action="store_true",   # BUG FIX: was default=True
                    help="Use tess-point internal sector definitions.")
    ap.add_argument("--fits",     type=str, default="~/PRIMVS/PRIMVS_P.fits")
    ap.add_argument("--ext",      type=int, default=1)
    ap.add_argument("--ra-col",   type=str, default="ra1")
    ap.add_argument("--dec-col",  type=str, default="dec1")
    ap.add_argument("--out",          type=str, default="tess_cycle9_pointings_overlay.png")
    ap.add_argument("--query-target", type=str, default=None,
                    help="Report observations for a target: 'RA,Dec' in degrees, e.g. --query-target 266.4,-29.0")
    args = ap.parse_args()

    if args.no_override:
        sectors       = list(range(108, 122))
        scinfo_global = None
    else:
        override_path = Path("cycle9_sector_override.txt")
        sectors       = write_cycle9_override(override_path)
        scinfo_global = TESS_Spacecraft_Pointing_Data(sectorOverrideFile=str(override_path))

    # Optional: report observations for a single target
    if args.query_target is not None:
        try:
            t_ra, t_dec = [float(v.strip()) for v in args.query_target.split(",")]
        except ValueError:
            print("[error] --query-target must be 'RA,Dec' in degrees, e.g. 266.4,-29.0")
        else:
            query_target_observations(t_ra, t_dec, sectors, scinfo_global)

    cmap = plt.get_cmap("viridis", len(sectors))
    norm = mpl.colors.Normalize(vmin=min(sectors), vmax=max(sectors))

    # Load catalogue — colorbar_col from POINT_STYLE drives which extra column to read
    points    = None
    fits_path = Path(args.fits).expanduser()
    c_col     = POINT_STYLE["colorbar_col"]   # None → fixed colour; string → colourbar

    if fits_path.exists():
        try:
            points = load_points_from_fits(fits_path, args.ra_col, args.dec_col, c_col, args.ext)
        except Exception as e:
            print(f"[warn] FITS load failed ({e}), trying CSV …")
            try:
                points = load_points_from_csv(fits_path, args.ra_col, args.dec_col, c_col)
            except Exception as e2:
                print(f"[warn] CSV load also failed ({e2}) — plotting footprints only")
    else:
        print(f"[warn] catalogue not found: {fits_path} — plotting footprints only")

    out_base = Path(args.out)
    stem     = out_base.stem
    parent   = out_base.parent if str(out_base.parent) != "" else Path(".")

    fig1 = plot_equatorial(sectors, scinfo_global, cmap, norm, points=points)
    plt.tight_layout()
    fig1.savefig(parent / f"{stem}_equatorial.png", dpi=220)
    plt.close(fig1)
    print(f"Wrote: {(parent / f'{stem}_equatorial.png').resolve()}")

    fig2 = plot_galactic(sectors, scinfo_global, cmap, norm, points=points)
    plt.tight_layout()
    fig2.savefig(parent / f"{stem}_galactic.png", dpi=220)
    plt.close(fig2)
    print(f"Wrote: {(parent / f'{stem}_galactic.png').resolve()}")


def query_target_observations(ra_deg: float, dec_deg: float, sectors: list[int], scinfo_global) -> None:
    """
    Print how many TESS Cycle 9 observations a target at (ra_deg, dec_deg) receives.

    Reports:
      - Total number of sector-camera-CCD hits
      - Which sectors observe it, and on which camera/CCD
      - Total unique sectors
    """
    print(f"\n{'='*60}")
    print(f"Target: RA={ra_deg:.5f}  Dec={dec_deg:.5f}")
    print(f"Checking {len(sectors)} Cycle 9 sectors ({sectors[0]}–{sectors[-1]})")
    print(f"{'='*60}")

    hits = []

    outID, outEclipLong, outEclipLat, outSec, outCam, outCcd, outColPix, outRowPix, scinfo_out = \
        tess_stars2px_function_entry(0, ra_deg, dec_deg, scInfo=scinfo_global)

    if outSec is None or len(outSec) == 0:
        print("  Not observed in any sector (tess-point returned no hits).")
    else:
        for sec, cam, ccd, col, row in zip(outSec, outCam, outCcd, outColPix, outRowPix):
            if sec in sectors:
                hits.append((int(sec), int(cam), int(ccd), float(col), float(row)))

    if not hits:
        print(f"  Result: NOT observed in any Cycle 9 sector.")
    else:
        unique_sectors = sorted(set(h[0] for h in hits))
        print(f"  Total hits (sector × camera × CCD): {len(hits)}")
        print(f"  Unique sectors observed            : {len(unique_sectors)}")
        print(f"  Sector list                        : {unique_sectors}")
        print()
        print(f"  {'Sector':>8}  {'Camera':>6}  {'CCD':>4}  {'Col':>8}  {'Row':>8}")
        print(f"  {'-'*8}  {'-'*6}  {'-'*4}  {'-'*8}  {'-'*8}")
        for sec, cam, ccd, col, row in sorted(hits):
            print(f"  {sec:>8}  {cam:>6}  {ccd:>4}  {col:>8.1f}  {row:>8.1f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()