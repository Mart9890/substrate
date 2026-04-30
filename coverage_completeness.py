#!/usr/bin/env python3
"""
coverage_completeness.py  —  Coastal grid completeness analysis
===============================================================
Queries an already-built spearo_coastal_grid SQLite (.db) or Parquet file
and reports how complete each data domain is, broken down by coastal zone.

Outputs (written next to the DB, or to --output-dir if supplied):
  <stem>_completeness.json     Full per-zone, per-domain statistics
  <stem>_gaps.gpkg             100m cells with NO data in any domain
  <stem>_map.png               Visual map: completeness score per cell

Usage
-----
  python coverage_completeness.py output/spearo_coastal_grid_100m.db
  python coverage_completeness.py output/spearo_coastal_grid_100m.parquet
  python coverage_completeness.py output/spearo_coastal_grid_100m.db --output-dir reports/

Dependencies
------------
  pip install pandas geopandas matplotlib tabulate
  pip install duckdb          # optional — faster parquet loading
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

CRS_WORKING = "EPSG:27700"

# Domains and the column / test that determines whether a cell has data
DOMAINS = {
    "substrate":  lambda df: df["substrate_source"].isin(
                      ["BGS_observed", "BGS_predictive", "DEFR"]),
    "habitat":    lambda df: df["eunis_code"].notna(),
    "bathymetry": lambda df: df["depth_m"].notna(),
    "bedrock":    lambda df: df["bedrock_lex_rcs"].notna(),
}

# Colour palette for the domain strip bars on the map
DOMAIN_COLOURS = {
    "substrate":  "#4e9af1",
    "habitat":    "#2a9d8f",
    "bathymetry": "#e9c46a",
    "bedrock":    "#e76f51",
}

# ── Logging ───────────────────────────────────────────────────────────────────

def log(msg: str, indent: int = 0):
    print("  " * indent + msg, flush=True)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_grid(path: Path):
    """Load coastal_grid from a .db (SQLite) or .parquet file."""
    suffix = path.suffix.lower()
    t0 = time.time()

    if suffix in (".db", ".sqlite", ".sqlite3"):
        try:
            import sqlite3, pandas as pd
            conn = sqlite3.connect(str(path))
            df = pd.read_sql("SELECT * FROM coastal_grid", conn)
            conn.close()
        except Exception as e_sqlite:
            try:
                import duckdb
                conn = duckdb.connect()
                conn.execute("INSTALL sqlite; LOAD sqlite;")
                conn.execute(f"ATTACH '{path}' AS src (TYPE sqlite);")
                df = conn.execute("SELECT * FROM src.coastal_grid").df()
                conn.close()
            except Exception as e_duck:
                raise RuntimeError(
                    f"SQLite: {e_sqlite}  |  DuckDB fallback: {e_duck}"
                )

    elif suffix in (".parquet", ".geoparquet"):
        try:
            import duckdb
            conn = duckdb.connect()
            df = conn.execute(f"SELECT * FROM read_parquet('{path}')").df()
            conn.close()
        except Exception:
            import pandas as pd
            df = pd.read_parquet(str(path))

    else:
        raise ValueError(f"Unsupported file type '{suffix}' — expected .db or .parquet")

    # Ensure expected columns exist (graceful for older schema versions)
    defaults = {
        "coverage_flags": None, "substrate_source": "none",
        "has_observed_survey": False, "overall_confidence": 0.0,
        "substrate_confidence": 0.0, "habitat_confidence": 0.0,
        "eunis_code": None, "bedrock_lex_rcs": None, "depth_m": None,
        "easting_bng": None, "northing_bng": None,
        "zone": "unknown",
    }
    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default

    log(f"Loaded {len(df):,} cells in {time.time()-t0:.1f}s")
    return df


# ── Statistics ────────────────────────────────────────────────────────────────

def compute_stats(df):
    """
    Returns (results_dict, fully_gapped_bool_series).
    results_dict is keyed by zone (+ 'ALL' and '_summary').
    """
    zones   = sorted(df["zone"].dropna().unique().tolist())
    results = {}

    for zone in (["ALL"] + zones):
        sub = df if zone == "ALL" else df[df["zone"] == zone]
        n   = len(sub)
        if n == 0:
            continue

        zr = {"total_cells": n, "domains": {}}

        for domain, check in DOMAINS.items():
            covered = int(check(sub).sum())
            pct     = round(100 * covered / n, 1)
            zr["domains"][domain] = {
                "covered_cells": covered,
                "pct":           pct,
                "gap_cells":     n - covered,
            }

        # Substrate source breakdown
        src_counts = sub["substrate_source"].fillna("none").value_counts()
        zr["substrate_sources"] = {
            src: {"cells": int(cnt), "pct": round(100 * cnt / n, 1)}
            for src, cnt in src_counts.items()
        }

        # Confidence distribution
        conf = sub["overall_confidence"].dropna()
        zr["confidence"] = {
            "mean":          round(float(conf.mean()),   3) if len(conf) else 0.0,
            "median":        round(float(conf.median()), 3) if len(conf) else 0.0,
            "pct_high_conf": round(100 * float((conf >= 0.75).sum()) / n, 1),
            "pct_low_conf":  round(100 * float((conf <  0.40).sum()) / n, 1),
        }

        # Most common gap-flag combinations
        zr["top_gap_combinations"] = (
            sub["coverage_flags"]
            .fillna("none")
            .value_counts()
            .head(10)
            .to_dict()
        )

        results[zone] = zr

    # Cells missing ALL domains
    fully_gapped = ~(
        DOMAINS["substrate"](df)  |
        DOMAINS["habitat"](df)    |
        DOMAINS["bathymetry"](df) |
        DOMAINS["bedrock"](df)
    )
    results["_summary"] = {
        "total_cells":        len(df),
        "cells_with_no_data": int(fully_gapped.sum()),
        "pct_with_no_data":   round(100 * fully_gapped.sum() / len(df), 1),
    }

    return results, fully_gapped


# ── Console report ────────────────────────────────────────────────────────────

def print_report(results):
    try:
        from tabulate import tabulate as _tab
    except ImportError:
        def _tab(rows, headers, tablefmt, **kw):
            lines = ["  ".join(str(h) for h in headers)]
            for r in rows:
                lines.append("  ".join(str(c) for c in r))
            return "\n".join(lines)

    zones = [k for k in results if not k.startswith("_")]

    print("\n" + "=" * 72)
    print("COASTAL GRID COMPLETENESS REPORT")
    print("=" * 72)

    rows = []
    for zone in zones:
        zr = results[zone]
        n  = zr["total_cells"]
        for domain, dr in zr["domains"].items():
            rows.append([zone, domain, f"{dr['covered_cells']:,}",
                         f"{n:,}", f"{dr['pct']:.1f}%"])
    print(_tab(rows,
               headers=["Zone", "Domain", "Covered", "Total", "% Complete"],
               tablefmt="github"))

    print(f"\n{'─' * 72}")
    print("SUBSTRATE SOURCES  (all zones)")
    for src, v in results.get("ALL", {}).get("substrate_sources", {}).items():
        bar = "█" * int(v["pct"] / 2)
        print(f"  {src:<22s}  {v['cells']:>10,} cells  ({v['pct']:5.1f}%)  {bar}")

    print(f"\n{'─' * 72}")
    print("CONFIDENCE  (all zones)")
    c = results.get("ALL", {}).get("confidence", {})
    print(f"  Mean:                     {c.get('mean', 0):.3f}")
    print(f"  Median:                   {c.get('median', 0):.3f}")
    print(f"  High confidence (≥0.75):  {c.get('pct_high_conf', 0):.1f}%")
    print(f"  Low confidence  (<0.40):  {c.get('pct_low_conf', 0):.1f}%")

    print(f"\n{'─' * 72}")
    s = results["_summary"]
    print(f"Cells with NO data (any domain):  {s['cells_with_no_data']:,}  "
          f"({s['pct_with_no_data']:.1f}%  of  {s['total_cells']:,})")

    combos = results.get("ALL", {}).get("top_gap_combinations", {})
    if combos:
        print("\nMost common gap-flag combinations  (ALL zones):")
        n = results["ALL"]["total_cells"]
        for flags, cnt in combos.items():
            pct = round(100 * cnt / n, 1)
            print(f"  [{flags:<30s}]  {cnt:>10,} cells  ({pct:.1f}%)")
    print(f"{'─' * 72}\n")


# ── Gap GeoPackage ────────────────────────────────────────────────────────────

def write_gap_gpkg(df, fully_gapped, out_path: Path):
    """Export cells missing ALL domains as 100m square polygons."""
    try:
        import geopandas as gpd
        from shapely.geometry import box as sbox

        gap_df = df.loc[fully_gapped, [
            c for c in ["cell_id", "easting_bng", "northing_bng",
                        "zone", "coverage_flags", "overall_confidence"]
            if c in df.columns
        ]].copy()

        log(f"Writing gap GeoPackage ({len(gap_df):,} cells)...")
        t0 = time.time()

        half  = 50
        geoms = [
            sbox(e - half, n - half, e + half, n + half)
            for e, n in zip(gap_df["easting_bng"], gap_df["northing_bng"])
        ]
        gdf = gpd.GeoDataFrame(gap_df, geometry=geoms, crs=CRS_WORKING)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        gdf.to_file(str(out_path), driver="GPKG")
        log(f"✓ Gap GeoPackage: {out_path}  [{time.time()-t0:.1f}s]")
        log("  Style by 'coverage_flags' in QGIS to see which domains are missing.", indent=1)
    except Exception as e:
        log(f"  Warning: gap GeoPackage failed: {e}")


# ── Coverage map ──────────────────────────────────────────────────────────────

def write_map(df, results, fully_gapped, out_path: Path):
    """
    Multi-panel map: one subplot per zone + an ALL panel.
    Each cell is coloured by overall_confidence (YlGnBu, 0–1).
    Cells missing ALL domains are shown in red.
    A domain-completeness bar strip sits below each panel.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        import matplotlib.patches as mpatches
        from matplotlib.cm import ScalarMappable

        log("Rendering coverage map...")
        t0 = time.time()

        zones    = sorted(df["zone"].dropna().unique().tolist())
        panels   = ["ALL"] + zones
        n_panels = len(panels)

        cmap = plt.cm.YlGnBu
        norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
        BG   = "#1a1a2e"

        fig, axes = plt.subplots(
            1, n_panels,
            figsize=(5 * n_panels, 14),
            facecolor=BG,
        )
        if n_panels == 1:
            axes = [axes]

        for ax, panel in zip(axes, panels):
            ax.set_facecolor(BG)

            if panel == "ALL":
                sub  = df
                gap  = fully_gapped
                title = f"ALL  ({len(sub):,} cells)"
            else:
                mask = df["zone"] == panel
                sub  = df[mask]
                gap  = fully_gapped[mask]
                zr   = results.get(panel, {})
                conf_mean = zr.get("confidence", {}).get("mean", 0.0)
                title = f"{panel}\n{len(sub):,} cells · conf {conf_mean:.2f}"

            if len(sub) == 0:
                ax.set_title(panel, color="white", fontsize=9)
                ax.axis("off")
                continue

            e      = sub["easting_bng"].values
            n_c    = sub["northing_bng"].values
            conf   = sub["overall_confidence"].fillna(0.0).values
            has_data = ~gap.values

            # Confidence-coloured cells
            if has_data.any():
                ax.scatter(
                    e[has_data], n_c[has_data],
                    c=conf[has_data], cmap=cmap, norm=norm,
                    s=0.5, linewidths=0, alpha=0.85, rasterized=True,
                )

            # Fully-gapped cells in red
            if gap.values.any():
                ax.scatter(
                    e[gap.values], n_c[gap.values],
                    c="#e63946", s=0.5, linewidths=0, alpha=0.9, rasterized=True,
                )

            ax.set_aspect("equal")
            ax.set_title(title, color="white", fontsize=9, pad=6)
            ax.tick_params(colors="grey", labelsize=6)
            for spine in ax.spines.values():
                spine.set_edgecolor("#333355")

            # Domain-completeness bar strip below the panel
            zr   = results.get(panel, {})
            doms = zr.get("domains", {})
            if doms:
                xlim    = ax.get_xlim()
                ylim    = ax.get_ylim()
                h_range = ylim[1] - ylim[0]
                strip_y = ylim[0] - h_range * 0.055
                bar_w   = (xlim[1] - xlim[0]) / len(DOMAINS)
                bar_h   = h_range * 0.022

                for i, (domain, colour) in enumerate(DOMAIN_COLOURS.items()):
                    pct = doms.get(domain, {}).get("pct", 0) / 100
                    bx  = xlim[0] + i * bar_w
                    # Background track
                    ax.barh(strip_y, bar_w, left=bx, height=bar_h,
                            color="#333355", alpha=0.6, clip_on=False)
                    # Filled portion
                    ax.barh(strip_y, pct * bar_w, left=bx, height=bar_h,
                            color=colour, alpha=0.9, clip_on=False)
                    ax.text(
                        bx + bar_w * 0.5, strip_y,
                        f"{domain[0].upper()}  {doms.get(domain,{}).get('pct',0):.0f}%",
                        ha="center", va="center", fontsize=5.5,
                        color="white", clip_on=False,
                    )

        # Shared colourbar
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes, orientation="horizontal",
                            fraction=0.018, pad=0.06, shrink=0.55)
        cbar.set_label("Overall confidence  (0 = no data  →  1 = high confidence)",
                       color="white", fontsize=9)
        cbar.ax.xaxis.set_tick_params(color="white", labelcolor="white", labelsize=8)

        # Legend for red no-data cells
        no_data_patch = mpatches.Patch(color="#e63946",
                                       label="No data in any domain")
        fig.legend(handles=[no_data_patch], loc="lower center",
                   facecolor=BG, edgecolor="#444466",
                   labelcolor="white", fontsize=9,
                   bbox_to_anchor=(0.5, 0.0))

        fig.suptitle("Coastal Grid — Data Completeness by Zone",
                     color="white", fontsize=13, y=1.01)
        fig.patch.set_facecolor(BG)
        plt.tight_layout()

        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(out_path), dpi=150, bbox_inches="tight", facecolor=BG)
        plt.close()
        log(f"✓ Coverage map:  {out_path}  [{time.time()-t0:.1f}s]")

    except Exception as e:
        import traceback
        log(f"  Warning: map generation failed: {e}")
        traceback.print_exc()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Analyse completeness of an already-built coastal_grid DB/Parquet.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "db",
        help="Path to spearo_coastal_grid_*.db or .parquet",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Directory to write outputs into (default: same directory as the DB)",
    )
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        log(f"ERROR: file not found: {db_path}")
        sys.exit(1)

    out_dir = Path(args.output_dir) if args.output_dir else db_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    stem      = db_path.stem
    json_path = out_dir / f"{stem}_completeness.json"
    gap_path  = out_dir / f"{stem}_gaps.gpkg"
    map_path  = out_dir / f"{stem}_map.png"

    log(f"\nCoverage Completeness Analysis")
    log(f"  Input:      {db_path}")
    log(f"  Output dir: {out_dir}/")
    t_total = time.time()

    df               = load_grid(db_path)
    results, gapped  = compute_stats(df)

    print_report(results)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    log(f"✓ JSON report:   {json_path}")

    write_gap_gpkg(df, gapped, gap_path)
    write_map(df, results, gapped, map_path)

    log(f"\nDone in {time.time()-t_total:.1f}s")


if __name__ == "__main__":
    main()
