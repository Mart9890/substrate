#!/usr/bin/env python3
"""
check_hwl_completeness.py  —  OS High Water Line polyline completeness audit
=============================================================================
OPTIMISED VERSION — key changes vs original:
  1. test_components() returns pre-merged geometry; test_polygonize() reuses it
     (avoids a second unary_union + linemerge that cost ~52 s)
  2. snap step is skipped when data is already clean (Tests 1–3 proved 0 dangles)
     (snap was the 2-hour bottleneck — O(n²) GEOS vertex matching on clean data)
  3. simplify(SIMPLIFY_M) applied before polygonize to cut vertex count 5–20×
     (invisible at 1:250k, reduces polygonize time from hours → minutes)
  4. --no-snap flag lets you force-skip snap even when dangles are present
  5. --simplify flag (default 5 m) controls pre-polygonize simplification

Checks whether the OS high water polyline is suitable for polygonization into
a land mask by testing:

  TEST 1  Component analysis
          How many disconnected linestring components exist?
          A complete mainland + islands coastline will have a small number of
          closed rings (one per landmass/island). Open chains indicate gaps.

  TEST 2  Endpoint / dangle analysis
          Extracts all line endpoints. A closed ring has no free endpoints.
          A dangling endpoint (one that doesn't snap to any other endpoint
          within tolerance) indicates a gap in the polyline.

  TEST 3  Gap characterisation
          For each pair of dangling endpoints that are within a closeable
          distance of each other, reports the gap size and location.

  TEST 4  Polygonization attempt
          Attempts to polygonize the (optionally snapped) linestring.
          Reports how many valid polygons are produced and their areas.

Outputs:
  dangles.gpkg        — point layer of all dangling endpoints (open in QGIS)
  gaps.gpkg           — line layer connecting paired gap endpoints
  components.gpkg     — one feature per disconnected component
  [optional] land_polygon.gpkg  — polygonized result if --polygonize is set

Usage:
    python check_hwl_completeness.py
    python check_hwl_completeness.py --shp raw/os_coastline/high_water_polyline.shp
    python check_hwl_completeness.py --snap 10 --polygonize
    python check_hwl_completeness.py --polygonize --no-snap --simplify 5
    python check_hwl_completeness.py --snap 10 --polygonize --out-dir output/coastline_check
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

SEP  = "=" * 72
SEP2 = "-" * 72

def hdr(t):  print(f"\n{SEP}\n  {t}\n{SEP}")
def sub(t):  print(f"\n{SEP2}\n  {t}\n{SEP2}")
def ok(m):   print(f"  \u2713  {m}")
def warn(m): print(f"  \u26a0  {m}")
def fail(m): print(f"  \u2717  {m}")
def info(m): print(f"     {m}")


# ── TEST 1: Component analysis ────────────────────────────────────────────────

def test_components(gdf) -> dict:
    hdr("TEST 1 — Component analysis")

    from shapely.ops import unary_union, linemerge
    from shapely.geometry import MultiLineString, LineString

    info(f"Input: {len(gdf):,} raw features")

    t0 = time.time()
    info("Merging and linemerging all features …")
    merged_union = unary_union(gdf.geometry)
    merged = linemerge(merged_union)
    info(f"  Done ({time.time()-t0:.1f}s)")

    if isinstance(merged, LineString):
        components = [merged]
    elif isinstance(merged, MultiLineString):
        components = list(merged.geoms)
    else:
        components = list(merged.geoms)

    n_components = len(components)
    info(f"  Components after linemerge: {n_components:,}")

    closed_rings = []
    open_chains  = []
    for c in components:
        coords = list(c.coords)
        is_closed = (abs(coords[0][0] - coords[-1][0]) < 0.01 and
                     abs(coords[0][1] - coords[-1][1]) < 0.01)
        if is_closed:
            closed_rings.append(c)
        else:
            open_chains.append(c)

    info(f"\n  Closed rings (no free endpoints): {len(closed_rings):,}")
    info(f"  Open chains  (has free endpoints): {len(open_chains):,}")

    lengths = [c.length / 1000 for c in components]
    lengths.sort(reverse=True)
    info(f"\n  Length summary (km):")
    info(f"    Total:   {sum(lengths):,.1f} km")
    info(f"    Longest: {lengths[0]:,.1f} km")
    if len(lengths) > 1:  info(f"    2nd:     {lengths[1]:,.1f} km")
    if len(lengths) > 2:  info(f"    3rd:     {lengths[2]:,.1f} km")
    if len(lengths) > 9:  info(f"    10th:    {lengths[9]:,.1f} km")
    info(f"    Shortest: {lengths[-1]:.3f} km")

    tiny = sum(1 for l in lengths if l < 1.0)
    info(f"\n  Components < 1 km: {tiny:,}  (likely small islands or noise)")
    info(f"  Components >= 1 km: {sum(1 for l in lengths if l >= 1.0):,}")

    print()
    if len(open_chains) == 0:
        ok("All components are closed rings — polyline is topologically complete.")
        verdict = "COMPLETE"
    elif len(open_chains) <= 10:
        warn(f"{len(open_chains)} open chains — small number of gaps, likely estuaries/borders.")
        verdict = "MOSTLY_COMPLETE"
    elif len(open_chains) <= 50:
        warn(f"{len(open_chains)} open chains — moderate gaps, may need snapping.")
        verdict = "GAPPY"
    else:
        fail(f"{len(open_chains)} open chains — many gaps, dataset may be incomplete.")
        verdict = "INCOMPLETE"

    # ── OPT: return the pre-merged geometry so Test 4 can reuse it ────────────
    return {
        "n_components":   n_components,
        "n_closed":       len(closed_rings),
        "n_open":         len(open_chains),
        "total_length_km": sum(lengths),
        "components":     components,
        "merged_union":   merged_union,   # reused by test_polygonize
        "merged":         merged,         # reused by test_polygonize
        "verdict":        verdict,
    }


# ── TEST 2: Dangle analysis ───────────────────────────────────────────────────

def test_dangles(gdf, snap_m: float = 1.0) -> dict:
    hdr("TEST 2 — Dangle / free endpoint analysis")

    from shapely.geometry import Point, LineString
    import geopandas as gpd

    info(f"Extracting endpoints from {len(gdf):,} features (snap tolerance: {snap_m} m) …")
    t0 = time.time()

    endpoints = []
    for i, geom in enumerate(gdf.geometry):
        if geom is None:
            continue
        geoms = list(geom.geoms) if hasattr(geom, "geoms") else [geom]
        for g in geoms:
            if not isinstance(g, LineString) or len(g.coords) < 2:
                continue
            coords = list(g.coords)
            endpoints.append((coords[0][0],  coords[0][1],  i, "start"))
            endpoints.append((coords[-1][0], coords[-1][1], i, "end"))

    info(f"  {len(endpoints):,} total endpoints  ({time.time()-t0:.1f}s)")

    if not endpoints:
        warn("No endpoints found")
        return {"skipped": True}

    coords_arr = np.array([(e[0], e[1]) for e in endpoints])

    info(f"  Building spatial index for snap check ({snap_m} m tolerance) …")
    t0 = time.time()

    from scipy.spatial import cKDTree
    tree = cKDTree(coords_arr)
    pairs = tree.query_ball_point(coords_arr, r=snap_m)
    dangles_idx = [i for i, p in enumerate(pairs) if len(p) <= 1]

    info(f"  {len(dangles_idx):,} dangling endpoints  ({time.time()-t0:.1f}s)")

    dangle_points  = [endpoints[i] for i in dangles_idx]
    dangle_coords  = np.array([(d[0], d[1]) for d in dangle_points])

    if len(dangle_points) > 0:
        sub("Dangle summary")
        info(f"  Total dangles:          {len(dangle_points):,}")
        info(f"  (Each gap = 2 dangles, so gaps ≈ {len(dangle_points)//2:,})")

        n_scotland = sum(1 for d in dangle_points if d[1] > 550_000)
        n_wales    = sum(1 for d in dangle_points if d[1] < 360_000 and d[0] < 340_000)
        n_other    = len(dangle_points) - n_scotland - n_wales
        info(f"\n  Near Scotland border (N>550km): {n_scotland:,}")
        info(f"  Near Wales border:              {n_wales:,}")
        info(f"  Elsewhere (estuaries/gaps):     {n_other:,}")

    if dangle_points:
        dangle_gdf = gpd.GeoDataFrame(
            {"feat_idx": [d[2] for d in dangle_points],
             "position": [d[3] for d in dangle_points]},
            geometry=[Point(d[0], d[1]) for d in dangle_points],
            crs="EPSG:27700"
        )
    else:
        dangle_gdf = gpd.GeoDataFrame(
            {"feat_idx": [], "position": []},
            geometry=[], crs="EPSG:27700"
        )

    print()
    if len(dangle_points) == 0:
        ok("No dangling endpoints — polyline is fully connected.")
        verdict = "FULLY_CONNECTED"
    elif len(dangle_points) <= 10:
        warn(f"{len(dangle_points)} dangles ({len(dangle_points)//2} gaps) — very few, likely border clips only.")
        verdict = "NEAR_COMPLETE"
    elif len(dangle_points) <= 50:
        warn(f"{len(dangle_points)} dangles ({len(dangle_points)//2} gaps) — moderate, check gap sizes.")
        verdict = "MODERATE_GAPS"
    else:
        fail(f"{len(dangle_points)} dangles ({len(dangle_points)//2} gaps) — many gaps.")
        verdict = "MANY_GAPS"

    return {"n_dangles": len(dangle_points), "n_gaps": len(dangle_points) // 2,
            "dangle_gdf": dangle_gdf, "dangle_coords": dangle_coords,
            "verdict": verdict}


# ── TEST 3: Gap characterisation ─────────────────────────────────────────────

def test_gaps(dangle_coords: np.ndarray, dangle_gdf) -> dict:
    hdr("TEST 3 — Gap characterisation")

    from shapely.geometry import LineString
    import geopandas as gpd

    if len(dangle_coords) == 0:
        ok("No dangles — no gaps to characterise.")
        return {"n_gaps": 0}

    info(f"Pairing {len(dangle_coords):,} dangling endpoints into nearest-neighbour gap pairs …")

    from scipy.spatial import cKDTree
    tree = cKDTree(dangle_coords)
    dists, idxs = tree.query(dangle_coords, k=2)
    nearest_dist = dists[:, 1]
    nearest_idx  = idxs[:, 1]

    seen  = set()
    pairs = []
    for i, (d, j) in enumerate(zip(nearest_dist, nearest_idx)):
        key = (min(i, j), max(i, j))
        if key not in seen:
            seen.add(key)
            pairs.append((i, j, d))

    pairs.sort(key=lambda x: x[2])
    info(f"  {len(pairs):,} gap pairs found")
    print()

    tiny   = [(i,j,d) for i,j,d in pairs if d <    50]
    small  = [(i,j,d) for i,j,d in pairs if   50 <= d <  5_000]
    medium = [(i,j,d) for i,j,d in pairs if 5_000 <= d < 20_000]
    large  = [(i,j,d) for i,j,d in pairs if d >= 20_000]

    sub("Gap size distribution")
    info(f"  < 50 m      (snap/tolerance issues):  {len(tiny):>5,}")
    info(f"  50 m – 5 km (estuaries/river mouths): {len(small):>5,}")
    info(f"  5 – 20 km   (large estuaries/fjords):  {len(medium):>5,}")
    info(f"  > 20 km     (missing data?):           {len(large):>5,}")

    if pairs:
        sub("Largest gaps (top 20)")
        info(f"  {'Gap #':>5}  {'Distance':>10}  {'From (E, N)':>24}  {'To (E, N)':>24}")
        for rank, (i, j, d) in enumerate(reversed(pairs[-20:]), 1):
            a, b = dangle_coords[i], dangle_coords[j]
            info(f"  {rank:>5}  {d:>9.0f}m  ({a[0]:>10.0f}, {a[1]:>10.0f})  ({b[0]:>10.0f}, {b[1]:>10.0f})")

        sub("Smallest gaps (top 20, potential auto-close candidates)")
        info(f"  {'Gap #':>5}  {'Distance':>10}  {'From (E, N)':>24}  {'To (E, N)':>24}")
        for rank, (i, j, d) in enumerate(pairs[:20], 1):
            a, b = dangle_coords[i], dangle_coords[j]
            info(f"  {rank:>5}  {d:>9.0f}m  ({a[0]:>10.0f}, {a[1]:>10.0f})  ({b[0]:>10.0f}, {b[1]:>10.0f})")

    gap_rows = []
    for i, j, d in pairs:
        a, b = dangle_coords[i], dangle_coords[j]
        size_class = ("tiny" if d < 50 else
                      "estuary" if d < 5_000 else
                      "large_estuary" if d < 20_000 else
                      "missing_data")
        gap_rows.append({
            "gap_dist_m": round(d, 1), "size_class": size_class,
            "from_e": round(float(a[0]), 1), "from_n": round(float(a[1]), 1),
            "to_e":   round(float(b[0]), 1), "to_n":   round(float(b[1]), 1),
            "geometry": LineString([a, b]),
        })

    gaps_gdf = gpd.GeoDataFrame(gap_rows, crs="EPSG:27700") if gap_rows else \
               gpd.GeoDataFrame({"gap_dist_m": [], "size_class": [], "geometry": []}, crs="EPSG:27700")

    print()
    if len(large) == 0 and len(medium) == 0:
        ok(f"All {len(pairs)} gaps are < 5 km — all are estuary/tolerance scale, auto-closeable.")
        verdict = "AUTO_CLOSEABLE"
    elif len(large) == 0:
        warn(f"{len(medium)} medium gaps (5–20 km) — large estuaries, closeable but review.")
        verdict = "MOSTLY_CLOSEABLE"
    else:
        fail(f"{len(large)} large gaps > 20 km — likely missing data, needs investigation.")
        verdict = "HAS_LARGE_GAPS"

    return {"n_gaps": len(pairs), "n_tiny": len(tiny), "n_small": len(small),
            "n_medium": len(medium), "n_large": len(large),
            "gaps_gdf": gaps_gdf, "pairs": pairs, "verdict": verdict}


# ── TEST 4: Polygonization attempt ────────────────────────────────────────────

def test_polygonize(gdf, snap_m: float, close_gaps: bool = True,
                    no_snap: bool = False, simplify_m: float = 5.0,
                    premerged_union=None, premerged=None) -> dict:
    hdr("TEST 4 — Polygonization attempt")

    from shapely.ops import linemerge, polygonize_full
    from shapely.geometry import LineString, MultiLineString
    import geopandas as gpd

    # ── OPT 1: reuse geometry already computed in Test 1 ─────────────────────
    if premerged_union is not None:
        info("  Reusing merged geometry from Test 1 (skipping second unary_union) …")
        merged = premerged_union
        need_linemerge = (premerged is None)
    else:
        from shapely.ops import unary_union
        info(f"Attempting polygonization (snap={snap_m} m, close_gaps={close_gaps}) …")
        t0 = time.time()
        merged = unary_union(gdf.geometry)
        info(f"  unary_union done ({time.time()-t0:.1f}s)")
        need_linemerge = True

    # ── OPT 2: skip snap when data is clean, or --no-snap forced ─────────────
    if no_snap:
        info("  Skipping snap step (--no-snap flag set).")
    elif close_gaps and snap_m > 0:
        info(f"  Snapping geometry to {snap_m} m tolerance …")
        info("  (tip: if Tests 1–3 showed 0 dangles, re-run with --no-snap to skip this)")
        t0 = time.time()
        from shapely import snap as shapely_snap
        merged = shapely_snap(merged, merged, snap_m)
        info(f"  Snap done ({time.time()-t0:.1f}s)")
        need_linemerge = True   # snap may have changed topology

    # ── OPT 3: simplify to cut vertex count before expensive linemerge/polygonize
    if simplify_m > 0:
        info(f"  Simplifying geometry to {simplify_m} m tolerance …")
        t0 = time.time()
        merged = merged.simplify(simplify_m, preserve_topology=True)
        info(f"  Simplify done ({time.time()-t0:.1f}s)")
        need_linemerge = True

    # linemerge if we haven't reused it or geometry changed
    if need_linemerge:
        info("  linemerge …")
        t0 = time.time()
        lines = linemerge(merged)
        info(f"  linemerge done ({time.time()-t0:.1f}s)")
    else:
        lines = premerged

    info("  polygonize …")
    t0 = time.time()
    result, dangles_out, cut_edges, invalid = polygonize_full(lines)
    polys = list(result.geoms) if hasattr(result, "geoms") else \
            ([result] if not result.is_empty else [])
    info(f"  Polygonization done ({time.time()-t0:.1f}s)")

    info(f"  Polygons produced:  {len(polys):,}")
    info(f"  Dangling edges:     {len(list(dangles_out.geoms)) if hasattr(dangles_out,'geoms') else 0:,}")
    info(f"  Cut edges:          {len(list(cut_edges.geoms)) if hasattr(cut_edges,'geoms') else 0:,}")
    info(f"  Invalid rings:      {len(list(invalid.geoms)) if hasattr(invalid,'geoms') else 0:,}")

    if polys:
        areas_km2 = sorted([p.area / 1e6 for p in polys], reverse=True)
        sub("Polygon areas")
        info(f"  {'Rank':>5}  {'Area (km²)':>12}  {'Likely'}")
        labels = {0: "England mainland", 1: "Isle of Wight or large island",
                  2: "Medium island", 3: "Small island"}
        for i, a in enumerate(areas_km2[:15]):
            info(f"  {i+1:>5}  {a:>12,.1f}  {labels.get(i, 'Island/enclave')}")
        if len(areas_km2) > 15:
            info(f"  ... and {len(areas_km2)-15} more polygons")

        total_area = sum(areas_km2)
        info(f"\n  Total polygonized area: {total_area:,.0f} km²")
        info(f"  England land area ~130,279 km² — coverage: {100*total_area/130_279:.1f}%")

        poly_gdf = gpd.GeoDataFrame(
            {"area_km2": areas_km2, "rank": range(1, len(polys)+1)},
            geometry=sorted(polys, key=lambda p: p.area, reverse=True),
            crs="EPSG:27700"
        )
    else:
        poly_gdf = None

    print()
    if len(polys) == 0:
        fail("Polygonization produced no polygons — polyline cannot be closed as-is.")
        verdict = "FAILED"
    elif len(polys) == 1:
        warn("Only 1 polygon — islands may be missing or merged into mainland.")
        verdict = "PARTIAL"
    elif areas_km2[0] > 100_000:
        ok(f"Largest polygon {areas_km2[0]:,.0f} km² — England mainland likely captured.")
        ok(f"{len(polys)} total polygons — mainland + {len(polys)-1} islands.")
        verdict = "SUCCESS"
    else:
        warn(f"Largest polygon only {areas_km2[0]:,.0f} km² — mainland may not be closed.")
        verdict = "PARTIAL"

    return {"n_polygons": len(polys), "poly_gdf": poly_gdf, "verdict": verdict}


# ── SUMMARY ───────────────────────────────────────────────────────────────────

def print_summary(results: dict) -> None:
    hdr("OVERALL VERDICT")

    verdicts = {k: v.get("verdict") for k, v in results.items()
                if isinstance(v, dict) and "verdict" in v}

    for test, verdict in verdicts.items():
        good = verdict in ("COMPLETE", "FULLY_CONNECTED", "AUTO_CLOSEABLE",
                           "MOSTLY_CLOSEABLE", "SUCCESS", "NEAR_COMPLETE",
                           "MOSTLY_COMPLETE")
        bad  = verdict in ("INCOMPLETE", "MANY_GAPS", "HAS_LARGE_GAPS",
                           "FAILED")
        m = "\u2713" if good else "\u2717" if bad else "\u26a0"
        print(f"  {m}  {test}: {verdict}")

    print()
    r1 = results.get("TEST_1_components", {})
    r3 = results.get("TEST_3_gaps", {})
    r4 = results.get("TEST_4_polygonize", {})

    n_large = r3.get("n_large", 0)
    n_gaps  = r3.get("n_gaps",  0)
    n_polys = r4.get("n_polygons", 0)
    poly_v  = r4.get("verdict", "")

    if poly_v == "SUCCESS" and n_large == 0:
        ok("CONCLUSION: HWL polyline is suitable for polygonization as a land mask.")
        ok(f"  {n_polys} polygon(s) produced, no large gaps.")
        ok("  Safe to use as the seaward-side boundary in Stage 2.")
        print()
        print("  RECOMMENDED build_coastal_grid.py fix:")
        print("    Replace the symmetric buffer approach with:")
        print("    1. Polygonize the HWL → land polygon")
        print("    2. outer = coast_line.buffer(STRIP_M)")
        print("    3. strip = outer.difference(land_polygon)")
        print("    This gives a true seaward-only strip.")
    elif n_large == 0 and n_gaps <= 20:
        warn("CONCLUSION: HWL is nearly suitable — small/estuary gaps only.")
        warn(f"  {n_gaps} gaps, all < 5 km. Auto-closing and polygonizing should work.")
        warn("  Review dangles.gpkg in QGIS to confirm gap locations are expected.")
    elif n_large > 0:
        fail(f"CONCLUSION: HWL has {n_large} large gap(s) > 20 km — likely missing data.")
        fail("  Polygonization will not produce a reliable land mask.")
        fail("  Consider supplementing with OS Boundary-Line or Natural Earth land polygon.")
    else:
        warn(f"CONCLUSION: {n_gaps} gaps found — review gaps.gpkg before deciding.")


# ── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shp",        default="raw/os_coastline/high_water_polyline.shp",
                    help="Path to OS high water polyline shapefile")
    ap.add_argument("--snap",       type=float, default=1.0,
                    help="Snap tolerance in metres for dangle detection (default: 1.0)")
    ap.add_argument("--polygonize", action="store_true",
                    help="Attempt polygonization (TEST 4)")
    ap.add_argument("--out-dir",    default="output/coastline_check",
                    help="Directory for output GeoPackages")
    ap.add_argument("--no-export",  action="store_true",
                    help="Skip GeoPackage export (console output only)")
    # ── new optimisation flags ────────────────────────────────────────────────
    ap.add_argument("--no-snap",    action="store_true",
                    help="Skip shapely.snap in TEST 4 (safe when Tests 1–3 show 0 dangles)")
    ap.add_argument("--simplify",   type=float, default=5.0,
                    help="Pre-polygonize simplification tolerance in metres (default: 5.0; "
                         "set 0 to disable). Invisible at 1:250k but cuts vertex count 5-20×.")
    args = ap.parse_args()

    shp_path = Path(args.shp)
    out_dir  = Path(args.out_dir)

    print(SEP)
    print("  check_hwl_completeness.py  [OPTIMISED]")
    print("  OS High Water Line polyline completeness audit")
    print(SEP)
    info(f"Input:    {shp_path}")
    info(f"Snap:     {args.snap} m")
    info(f"No-snap:  {args.no_snap}  (skip snap step in TEST 4)")
    info(f"Simplify: {args.simplify} m  (pre-polygonize; 0 = disabled)")
    info(f"Out dir:  {out_dir}")

    if not shp_path.exists():
        print(f"\nERROR: {shp_path} not found", file=sys.stderr)
        print("  Try: --shp path/to/high_water_polyline.shp", file=sys.stderr)
        sys.exit(1)

    try:
        import geopandas as gpd
        from scipy.spatial import cKDTree
    except ImportError as e:
        print(f"\nERROR: Missing dependency: {e}", file=sys.stderr)
        print("  pip install geopandas scipy", file=sys.stderr)
        sys.exit(1)

    info(f"\nLoading {shp_path.name} …")
    t0  = time.time()
    gdf = gpd.read_file(shp_path)
    if gdf.crs is None or gdf.crs.to_epsg() != 27700:
        info(f"  Reprojecting from {gdf.crs} → EPSG:27700 …")
        gdf = gdf.set_crs(27700) if gdf.crs is None else gdf.to_crs(27700)
    info(f"  {len(gdf):,} features loaded  ({time.time()-t0:.1f}s)")
    info(f"  CRS: {gdf.crs}")
    info(f"  Bounds: E={gdf.total_bounds[0]:.0f}–{gdf.total_bounds[2]:.0f}  "
         f"N={gdf.total_bounds[1]:.0f}–{gdf.total_bounds[3]:.0f}")

    results = {}

    # TEST 1 — also returns premerged geometry for TEST 4 to reuse
    results["TEST_1_components"] = test_components(gdf)

    results["TEST_2_dangles"] = test_dangles(gdf, snap_m=args.snap)

    dangle_coords = results["TEST_2_dangles"].get("dangle_coords", np.array([]))
    dangle_gdf    = results["TEST_2_dangles"].get("dangle_gdf")

    if len(dangle_coords) > 0:
        results["TEST_3_gaps"] = test_gaps(dangle_coords, dangle_gdf)
    else:
        results["TEST_3_gaps"] = {"n_gaps": 0, "verdict": "AUTO_CLOSEABLE"}
        hdr("TEST 3 — Gap characterisation")
        ok("No dangles found — no gaps to characterise.")

    if args.polygonize:
        # Decide whether to auto-skip snap
        n_dangles   = results["TEST_2_dangles"].get("n_dangles", -1)
        auto_no_snap = (n_dangles == 0)

        if auto_no_snap and not args.no_snap:
            info("\n  ℹ  Tests 1–3 confirmed 0 dangles — auto-skipping snap step in TEST 4.")
            info("     (Pass --no-snap explicitly to suppress this message, or --snap 0 to keep old behaviour.)\n")

        results["TEST_4_polygonize"] = test_polygonize(
            gdf,
            snap_m         = args.snap,
            close_gaps     = True,
            no_snap        = args.no_snap or auto_no_snap,
            simplify_m     = args.simplify,
            premerged_union = results["TEST_1_components"].get("merged_union"),
            premerged       = results["TEST_1_components"].get("merged"),
        )
    else:
        info("\nSkipping TEST 4 (pass --polygonize to attempt polygonization)")
        results["TEST_4_polygonize"] = {"skipped": True}

    print_summary(results)

    # ── Export GeoPackages ────────────────────────────────────────────────────
    if not args.no_export:
        hdr("Exporting GeoPackages")
        out_dir.mkdir(parents=True, exist_ok=True)

        exports = {
            "dangles.gpkg": results["TEST_2_dangles"].get("dangle_gdf"),
            "gaps.gpkg":    results["TEST_3_gaps"].get("gaps_gdf"),
        }

        comp_list = results["TEST_1_components"].get("components", [])
        if comp_list:
            from shapely.geometry import LineString
            comp_gdf = gpd.GeoDataFrame(
                {"component_id": range(len(comp_list)),
                 "length_km":    [c.length/1000 for c in comp_list],
                 "is_closed":    [
                     abs(list(c.coords)[0][0] - list(c.coords)[-1][0]) < 0.01
                     for c in comp_list
                 ]},
                geometry=comp_list,
                crs="EPSG:27700"
            )
            exports["components.gpkg"] = comp_gdf

        poly_gdf = results.get("TEST_4_polygonize", {}).get("poly_gdf")
        if poly_gdf is not None:
            exports["land_polygon.gpkg"] = poly_gdf

        for fname, gdf_out in exports.items():
            if gdf_out is None or len(gdf_out) == 0:
                info(f"  (skipping {fname} — empty)")
                continue
            path = out_dir / fname
            gdf_out.to_file(path, driver="GPKG")
            size = path.stat().st_size / 1024
            ok(f"  {fname}  ({size:.0f} KB)  → {path}")

        print()
        info("Open outputs in QGIS:")
        for fname in exports:
            info(f"  {out_dir / fname}")


if __name__ == "__main__":
    main()