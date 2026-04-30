#!/usr/bin/env python3
"""
query_grid.py  —  Spearo coastal grid point query
==================================================
Query the pre-built coastal grid by lat/lon or easting/northing.
Returns a structured dict with all substrate, habitat, bathymetry
and data-quality fields ready for the visibility/species algo.

Usage
-----
  # Single point by lat/lon
  python query_grid.py --lat 50.614 --lon -1.195

  # Single point by BNG easting/northing
  python query_grid.py --easting 460500 --northing 80300

  # Radius query (all cells within N metres of a point)
  python query_grid.py --lat 50.614 --lon -1.195 --radius 500

  # Output as JSON (default) or pretty-print
  python query_grid.py --lat 50.614 --lon -1.195 --format pretty

  # Use a different database
  python query_grid.py --lat 50.614 --lon -1.195 --db path/to/spearo_coastal_grid.db

  # Pipe to jq or other tools
  python query_grid.py --lat 50.614 --lon -1.195 --format json | jq .substrate

As a Python module
------------------
  from query_grid import GridQuery

  q = GridQuery("output/spearo_coastal_grid.db")

  # Nearest cell
  result = q.query_latlon(50.614, -1.195)

  # All cells within 500m
  results = q.query_latlon(50.614, -1.195, radius_m=500)

  # By BNG
  result = q.query_bng(460500, 80300)

  q.close()
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from pathlib import Path
from typing import Any

DEFAULT_DB = "output/spearo_coastal_grid.db"

# ── Coordinate conversion ─────────────────────────────────────────────────────

def latlon_to_bng(lat: float, lon: float) -> tuple[float, float]:
    """Convert WGS84 lat/lon to British National Grid easting/northing."""
    try:
        from pyproj import Transformer
        t = Transformer.from_crs(4326, 27700, always_xy=True)
        e, n = t.transform(lon, lat)
        return e, n
    except ImportError:
        # Fallback: approximate conversion (good to ~5m for England)
        return _approx_latlon_to_bng(lat, lon)


def _approx_latlon_to_bng(lat: float, lon: float) -> tuple[float, float]:
    """Approximate WGS84 → BNG without pyproj. Accurate to ~10m for England."""
    # Helmert parameters WGS84 → OSGB36
    a, b    = 6378137.000, 6356752.3141
    F0      = 0.9996012717
    lat0    = math.radians(49.0)
    lon0    = math.radians(-2.0)
    N0, E0  = -100000.0, 400000.0

    e2   = 1 - (b/a)**2
    n    = (a - b) / (a + b)
    lat_r = math.radians(lat)
    lon_r = math.radians(lon)

    nu  = a * F0 / math.sqrt(1 - e2 * math.sin(lat_r)**2)
    rho = a * F0 * (1 - e2) / (1 - e2 * math.sin(lat_r)**2)**1.5
    eta2 = nu/rho - 1

    M = b * F0 * (
        (1 + n + 1.25*n**2 + 1.25*n**3) * (lat_r - lat0)
        - (3*n + 3*n**2 + 2.625*n**3) * math.sin(lat_r - lat0) * math.cos(lat_r + lat0)
        + (1.875*n**2 + 1.875*n**3) * math.sin(2*(lat_r - lat0)) * math.cos(2*(lat_r + lat0))
        - 0.729167*n**3 * math.sin(3*(lat_r - lat0)) * math.cos(3*(lat_r + lat0))
    )

    I    = M + N0
    II   = nu/2 * math.sin(lat_r) * math.cos(lat_r)
    III  = nu/24 * math.sin(lat_r) * math.cos(lat_r)**3 * (5 - math.tan(lat_r)**2 + 9*eta2)
    IIIA = nu/720 * math.sin(lat_r) * math.cos(lat_r)**5 * (61 - 58*math.tan(lat_r)**2 + math.tan(lat_r)**4)
    IV   = nu * math.cos(lat_r)
    V    = nu/6 * math.cos(lat_r)**3 * (nu/rho - math.tan(lat_r)**2)
    VI   = nu/120 * math.cos(lat_r)**5 * (5 - 18*math.tan(lat_r)**2 + math.tan(lat_r)**4 + 14*eta2 - 58*math.tan(lat_r)**2*eta2)

    dlon = lon_r - lon0
    N = I + II*dlon**2 + III*dlon**4 + IIIA*dlon**6
    E = E0 + IV*dlon + V*dlon**3 + VI*dlon**5
    return E, N


# ── Query result structure ────────────────────────────────────────────────────

def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    """Convert a sqlite3.Row to a structured dict grouped by category."""
    r = dict(row)
    return {
        "location": {
            "cell_id":       r.get("cell_id"),
            "easting_bng":   r.get("easting_bng"),
            "northing_bng":  r.get("northing_bng"),
            "lat":           r.get("lat"),
            "lon":           r.get("lon"),
            "zone":          r.get("zone"),
            "dist_to_hwm_m": r.get("dist_to_hwm_m"),
        },
        "bathymetry": {
            "depth_m":   r.get("depth_m"),
            "slope_deg": r.get("slope_deg"),
            "morphology": r.get("morphology"),
        },
        "substrate": {
            "primary_type":  r.get("substrate_primary"),
            "folk_code":     r.get("folk_code"),
            "folk_description": r.get("folk_description"),
            "pct_gravel":    r.get("pct_gravel"),
            "pct_sand":      r.get("pct_sand"),
            "pct_mud":       r.get("pct_mud"),
            "hardness":      r.get("hardness"),
            "source":        r.get("substrate_source"),
            "confidence":    r.get("substrate_confidence"),
        },
        "habitat": {
            "eunis_code":  r.get("eunis_code"),
            "eunis_name":  r.get("eunis_name"),
            "mhc_code":    r.get("mhc_code"),
            "source":      r.get("habitat_source"),
            "confidence":  r.get("habitat_confidence"),
        },
        "bedrock": {
            "lex_rcs":     r.get("bedrock_lex_rcs"),
            "description": r.get("bedrock_description"),
            "exposed":     bool(r.get("bedrock_exposed", 0)),
        },
        "coastal": {
            "foreshore_type": r.get("foreshore_type"),
        },
        "data_quality": {
            "has_observed_survey": bool(r.get("has_observed_survey", 0)),
            "coverage_flags":      r.get("coverage_flags"),
            "overall_confidence":  r.get("overall_confidence"),
        },
    }


# ── GridQuery class ───────────────────────────────────────────────────────────

class GridQuery:
    """
    Thin query interface over the spearo_coastal_grid SQLite database.

    All queries snap to the nearest 100m grid cell (or return multiple cells
    when radius_m is specified).
    """

    CELL_SIZE = 100   # metres

    def __init__(self, db_path: str | Path = DEFAULT_DB):
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(
                f"Database not found: {self.db_path}\n"
                f"Run build_coastal_grid.py first."
            )
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row

    def close(self):
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    # ── Core snapping ─────────────────────────────────────────────────────────

    @staticmethod
    def _snap(easting: float, northing: float, cell: int = 100) -> tuple[int, int]:
        """Snap a coordinate to the nearest grid cell centre."""
        e = int(round(easting  / cell) * cell) + cell // 2
        n = int(round(northing / cell) * cell) + cell // 2
        return e, n

    # ── Point queries ─────────────────────────────────────────────────────────

    def query_bng(
        self,
        easting: float,
        northing: float,
        radius_m: float = 0,
    ) -> dict | list[dict] | None:
        """
        Query by BNG easting/northing.
        If radius_m=0: returns nearest single cell (or None if outside strip).
        If radius_m>0: returns all cells within radius_m metres.
        """
        if radius_m > 0:
            return self._radius_query(easting, northing, radius_m)

        e, n = self._snap(easting, northing)
        row = self.conn.execute(
            "SELECT * FROM coastal_grid WHERE easting_bng=? AND northing_bng=?",
            (e, n)
        ).fetchone()

        if row:
            return _row_to_dict(row)

        # Cell not in strip — find nearest cell in the database
        return self._nearest(easting, northing)

    def query_latlon(
        self,
        lat: float,
        lon: float,
        radius_m: float = 0,
    ) -> dict | list[dict] | None:
        """Query by WGS84 lat/lon. Converts to BNG internally."""
        e, n = latlon_to_bng(lat, lon)
        return self.query_bng(e, n, radius_m=radius_m)

    # ── Radius query ──────────────────────────────────────────────────────────

    def _radius_query(self, easting: float, northing: float, radius_m: float) -> list[dict]:
        """Return all cells within radius_m metres of a point."""
        half = radius_m
        candidates = self.conn.execute(
            """
            SELECT * FROM coastal_grid
            WHERE easting_bng  BETWEEN ? AND ?
              AND northing_bng BETWEEN ? AND ?
            """,
            (easting - half, easting + half, northing - half, northing + half)
        ).fetchall()

        results = []
        for row in candidates:
            r = dict(row)
            dx = r["easting_bng"]  - easting
            dy = r["northing_bng"] - northing
            dist = math.sqrt(dx*dx + dy*dy)
            if dist <= radius_m:
                d = _row_to_dict(row)
                d["_query_distance_m"] = round(dist, 1)
                results.append(d)

        results.sort(key=lambda x: x["_query_distance_m"])
        return results

    # ── Nearest fallback ──────────────────────────────────────────────────────

    def _nearest(self, easting: float, northing: float, limit: int = 1) -> dict | None:
        """Find the nearest cell to a point (used when exact snap misses the strip)."""
        # Approximate: expand search box until we find something
        for radius in [200, 500, 1000, 2000, 5000]:
            rows = self.conn.execute(
                """
                SELECT *, 
                    ((easting_bng - ?) * (easting_bng - ?) +
                     (northing_bng - ?) * (northing_bng - ?)) AS dist2
                FROM coastal_grid
                WHERE easting_bng  BETWEEN ? AND ?
                  AND northing_bng BETWEEN ? AND ?
                ORDER BY dist2
                LIMIT ?
                """,
                (easting, easting, northing, northing,
                 easting-radius, easting+radius,
                 northing-radius, northing+radius,
                 limit)
            ).fetchall()
            if rows:
                result = _row_to_dict(rows[0])
                dist = math.sqrt(dict(rows[0])["dist2"])
                result["_nearest_fallback"] = True
                result["_query_distance_m"] = round(dist, 1)
                return result
        return None

    # ── Convenience: algo-ready flat dict ─────────────────────────────────────

    def algo_inputs(self, lat: float, lon: float) -> dict | None:
        """
        Returns a flat dict with only the fields needed by the visibility/species algo.
        Designed to be passed directly into your prediction pipeline.
        """
        result = self.query_latlon(lat, lon)
        if not result:
            return None

        return {
            # Zone context
            "zone":            result["location"]["zone"],
            "dist_to_hwm_m":   result["location"]["dist_to_hwm_m"],
            # Depth & morphology
            "depth_m":         result["bathymetry"]["depth_m"],
            "slope_deg":       result["bathymetry"]["slope_deg"],
            "morphology":      result["bathymetry"]["morphology"],
            # Substrate (key vis + species drivers)
            "substrate":       result["substrate"]["primary_type"],
            "pct_gravel":      result["substrate"]["pct_gravel"],
            "pct_sand":        result["substrate"]["pct_sand"],
            "pct_mud":         result["substrate"]["pct_mud"],
            "hardness":        result["substrate"]["hardness"],
            # Habitat
            "eunis_code":      result["habitat"]["eunis_code"],
            "mhc_code":        result["habitat"]["mhc_code"],
            # Bedrock
            "bedrock_exposed": result["bedrock"]["exposed"],
            # Quality weight
            "data_confidence": result["data_quality"]["overall_confidence"],
        }

    # ── Database stats ────────────────────────────────────────────────────────

    def stats(self) -> dict:
        """Return build statistics from the database."""
        row = self.conn.execute("SELECT * FROM build_stats").fetchone()
        return dict(row) if row else {}


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Query the Spearo coastal grid by lat/lon or BNG coordinates.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--lat",      type=float, help="Latitude (WGS84)")
    group.add_argument("--easting",  type=float, help="BNG Easting")

    p.add_argument("--lon",       type=float, help="Longitude (required with --lat)")
    p.add_argument("--northing",  type=float, help="BNG Northing (required with --easting)")
    p.add_argument("--radius",    type=float, default=0,
                   help="Return all cells within this radius (metres)")
    p.add_argument("--db",        default=DEFAULT_DB,
                   help=f"Path to SQLite database (default: {DEFAULT_DB})")
    p.add_argument("--format",    choices=["json", "pretty", "algo"],
                   default="json",
                   help="Output format: json (default), pretty (human), algo (flat inputs only)")
    p.add_argument("--stats",     action="store_true",
                   help="Print database build statistics and exit")
    return p.parse_args()


def main():
    args = parse_args()

    try:
        q = GridQuery(args.db)
    except FileNotFoundError as e:
        print(f"✗ {e}", file=sys.stderr)
        sys.exit(1)

    if args.stats:
        stats = q.stats()
        print(json.dumps(stats, indent=2))
        q.close()
        return

    # Resolve coordinates
    if args.lat is not None:
        if args.lon is None:
            print("✗ --lon required with --lat", file=sys.stderr)
            sys.exit(1)
        if args.format == "algo":
            result = q.algo_inputs(args.lat, args.lon)
        else:
            result = q.query_latlon(args.lat, args.lon, radius_m=args.radius)
    else:
        if args.northing is None:
            print("✗ --northing required with --easting", file=sys.stderr)
            sys.exit(1)
        if args.format == "algo":
            lat, lon = _bng_to_latlon(args.easting, args.northing)
            result = q.algo_inputs(lat, lon)
        else:
            result = q.query_bng(args.easting, args.northing, radius_m=args.radius)

    q.close()

    if result is None:
        print(json.dumps({"error": "No data found for this location — outside England coastal strip?"}))
        sys.exit(0)

    if args.format == "json" or args.format == "algo":
        print(json.dumps(result, indent=2, default=str))
    elif args.format == "pretty":
        _pretty_print(result)


def _bng_to_latlon(easting, northing):
    try:
        from pyproj import Transformer
        t = Transformer.from_crs(27700, 4326, always_xy=True)
        lon, lat = t.transform(easting, northing)
        return lat, lon
    except ImportError:
        return 0, 0  # fallback


def _pretty_print(result):
    if isinstance(result, list):
        print(f"Found {len(result)} cells:\n")
        for i, r in enumerate(result[:10]):
            print(f"  Cell {i+1}  (dist: {r.get('_query_distance_m','?')}m)")
            _pretty_print_single(r)
            print()
        if len(result) > 10:
            print(f"  ... and {len(result)-10} more")
    else:
        _pretty_print_single(result)


def _pretty_print_single(r: dict):
    if not isinstance(r, dict) or "location" not in r:
        print(json.dumps(r, indent=2, default=str))
        return

    loc  = r["location"]
    bath = r["bathymetry"]
    sub  = r["substrate"]
    hab  = r["habitat"]
    bed  = r["bedrock"]
    cst  = r["coastal"]
    dq   = r["data_quality"]

    def _v(val, suffix="", na="—"):
        return f"{val}{suffix}" if val is not None else na

    print(f"  ┌─ Location ──────────────────────────────────────")
    print(f"  │  cell_id       {loc['cell_id']}")
    print(f"  │  BNG           E{loc['easting_bng']} N{loc['northing_bng']}")
    print(f"  │  WGS84         {loc['lat']}, {loc['lon']}")
    print(f"  │  zone          {loc['zone']}  ({_v(loc['dist_to_hwm_m'],'m')} from HWM)")
    print(f"  ├─ Bathymetry ────────────────────────────────────")
    print(f"  │  depth         {_v(bath['depth_m'],'m')}")
    print(f"  │  slope         {_v(bath['slope_deg'],'°')}  ({_v(bath['morphology'])})")
    print(f"  ├─ Substrate ─────────────────────────────────────")
    print(f"  │  primary       {_v(sub['primary_type'])}")
    print(f"  │  folk          {_v(sub['folk_code'])}  {_v(sub['folk_description'])}")
    print(f"  │  % gravel/sand/mud  {_v(sub['pct_gravel'])} / {_v(sub['pct_sand'])} / {_v(sub['pct_mud'])}")
    print(f"  │  hardness      {_v(sub['hardness'])}")
    print(f"  │  source        {_v(sub['source'])}  (conf: {_v(sub['confidence'])})")
    print(f"  ├─ Habitat ───────────────────────────────────────")
    print(f"  │  EUNIS         {_v(hab['eunis_code'])}  {_v(hab['eunis_name'])}")
    print(f"  │  MHC           {_v(hab['mhc_code'])}")
    print(f"  │  source        {_v(hab['source'])}  (conf: {_v(hab['confidence'])})")
    print(f"  ├─ Bedrock ───────────────────────────────────────")
    print(f"  │  LEX_RCS       {_v(bed['lex_rcs'])}")
    print(f"  │  exposed       {bed['exposed']}")
    print(f"  ├─ Coastal ───────────────────────────────────────")
    print(f"  │  foreshore     {_v(cst['foreshore_type'])}")
    print(f"  └─ Data quality ──────────────────────────────────")
    print(f"     observed data {dq['has_observed_survey']}")
    print(f"     confidence    {dq['overall_confidence']}")
    if dq["coverage_flags"]:
        print(f"     gaps          {dq['coverage_flags']}")


if __name__ == "__main__":
    main()
