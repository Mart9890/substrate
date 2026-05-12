#!/usr/bin/env python3
"""
verify_strip_placement.py  —  Proves or disproves the ribbon-centred hypothesis
================================================================================
Tests whether the coastal grid strip extends correctly from the HWM *outward*
to 1 nm (0 → 1,852 m seaward), or whether the HWM runs through the *centre*
of the ribbon (~926 m either side), incorrectly including inland cells.

Four independent tests:

  TEST 1  Strip mask raster analysis (fast — no geometry unions)
          Reads the cached NPZ rasters directly. Measures how strip mask
          pixels distribute relative to HWM pixels.

  TEST 2  dist_to_hwm_m distribution
          Histogram of the distance column — checks range and median.

  TEST 3  Land/sea side classification (spatial, vectorised, sampled)
          Classifies sampled cell centroids as sea-side or land-side of the
          HWM using a fast vectorised nearest-segment signed cross-product.

  TEST 4  Depth signature cross-check
          Land-side cells should not have meaningful offshore depth values.

Usage:
    python verify_strip_placement.py
    python verify_strip_placement.py --sample 10000
    python verify_strip_placement.py --no-spatial
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SEP  = "=" * 72
SEP2 = "-" * 72


def hdr(title):  print(f"\n{SEP}\n  {title}\n{SEP}")
def sub(title):  print(f"\n{SEP2}\n  {title}\n{SEP2}")
def ok(msg):     print(f"  \u2713  {msg}")
def warn(msg):   print(f"  \u26a0  {msg}")
def fail(msg):   print(f"  \u2717  {msg}")
def info(msg):   print(f"     {msg}")


# ── TEST 1: strip mask raster analysis (fast) ─────────────────────────────────

def test_strip_raster(cache_dir: Path) -> dict:
    """
    Use the cached raster arrays directly — no geometry unions, no geopandas.

    The key question: does the strip mask include pixels on BOTH sides of the
    HWM line, or only the seaward side?

    We answer this by:
      (a) Looking at the dist_hwm values of strip cells — if the strip is
          seaward-only (0–1852 m), ~50% of cells should be within 926 m.
          A centred ribbon (±926 m) would have nearly 100% within 926 m.
      (b) Per-column row analysis: for each grid column (easting), compare
          the row range of strip pixels vs HWM pixels. A seaward strip should
          be consistently on ONE side; a ribbon straddles both.
    """
    hdr("TEST 1 — Strip mask raster analysis (fast, no geometry ops)")

    stage2_dir = cache_dir.parent / "stage2_strip"
    stage1_dir = cache_dir

    strip_path = stage2_dir / "strip_mask_100m.npz"
    dist_path  = stage2_dir / "dist_hwm_100m.npz"
    coast_path = stage1_dir / "coastline_raster_100m.npz"
    meta_path  = stage2_dir / "raster_meta_100m.json"

    missing = [p for p in [strip_path, dist_path, coast_path, meta_path]
               if not p.exists()]
    if missing:
        for p in missing:
            warn(f"Not found: {p}")
        warn("Skipping TEST 1 — check cache paths")
        return {"skipped": True}

    info(f"Loading raster meta …")
    with open(meta_path) as f:
        meta = json.load(f)
    info(f"  {meta}")

    info(f"Loading strip mask …")
    strip_data = np.load(strip_path)
    strip      = strip_data[list(strip_data.keys())[0]].astype(bool)
    info(f"  shape={strip.shape}  strip pixels={strip.sum():,}")

    info(f"Loading distance-to-HWM raster …")
    dist_data  = np.load(dist_path)
    dist       = dist_data[list(dist_data.keys())[0]].astype(np.float32)
    info(f"  shape={dist.shape}  range=[{np.nanmin(dist):.0f}, {np.nanmax(dist):.0f}] m")

    info(f"Loading coastline raster …")
    coast_data = np.load(coast_path)
    coast      = coast_data[list(coast_data.keys())[0]].astype(bool)
    info(f"  shape={coast.shape}  HWM pixels={coast.sum():,}")

    # ── 1a: dist_to_HWM of strip cells ────────────────────────────────────────
    sub("1a — Distance-to-HWM of strip cells")
    strip_dists = dist[strip]
    info(f"  min:    {strip_dists.min():.0f} m")
    info(f"  max:    {strip_dists.max():.0f} m")
    info(f"  mean:   {strip_dists.mean():.0f} m")
    info(f"  median: {np.median(strip_dists):.0f} m")

    pct_926  = 100 * (strip_dists <=  926).mean()
    pct_1852 = 100 * (strip_dists <= 1852).mean()
    pct_over = 100 * (strip_dists >  1852).mean()
    info(f"  within  926 m: {pct_926:.1f}%  (expect ~50% for seaward strip, ~100% for ribbon)")
    info(f"  within 1852 m: {pct_1852:.1f}%")
    info(f"  beyond 1852 m: {pct_over:.1f}%  (should be ~0%)")

    # ── 1b: per-column strip extent relative to HWM ───────────────────────────
    sub("1b — Per-column strip extent relative to HWM row")
    coast_rows, coast_cols = np.where(coast)
    strip_rows, strip_cols = np.where(strip)

    if coast.sum() == 0:
        warn("No HWM pixels found — skipping 1b")
        verdict_1b = "AMBIGUOUS"
    else:
        unique_cols = np.intersect1d(coast_cols, strip_cols)
        info(f"  Columns with both HWM and strip pixels: {len(unique_cols):,}")

        # Sample every Nth column for speed
        step         = max(1, len(unique_cols) // 500)
        sample_cols  = unique_cols[::step]
        info(f"  Sampling every {step}th column ({len(sample_cols):,} columns) …")

        above_counts, below_counts = [], []
        for col in sample_cols:
            hwm_r   = coast_rows[coast_cols == col]
            strip_r = strip_rows[strip_cols == col]
            if len(hwm_r) == 0 or len(strip_r) == 0:
                continue
            hwm_mid = hwm_r.mean()
            above_counts.append((strip_r < hwm_mid).sum())
            below_counts.append((strip_r > hwm_mid).sum())

        if not above_counts:
            warn("  No overlapping columns found for 1b")
            verdict_1b = "AMBIGUOUS"
        else:
            mean_above = np.mean(above_counts)
            mean_below = np.mean(below_counts)
            total      = mean_above + mean_below
            pct_above  = 100 * mean_above / total if total > 0 else 0
            pct_below  = 100 * mean_below / total if total > 0 else 0

            info(f"\n  Mean strip pixels per column:")
            info(f"    Above HWM row (lower row idx / more north): {mean_above:.1f}  ({pct_above:.1f}%)")
            info(f"    Below HWM row (higher row idx / more south): {mean_below:.1f}  ({pct_below:.1f}%)")
            info(f"\n  Row 0 = northernmost. 'Above' = north of HWM, 'below' = south.")
            info(f"  For England south/east coasts, seaward = south/east = higher row idx.")
            info(f"  A seaward-only strip: heavily skewed to one side.")
            info(f"  A centred ribbon:     ~50% / 50%.")

            smaller_pct = min(pct_above, pct_below)
            if smaller_pct > 35:
                verdict_1b = "CENTRED_RIBBON"
                fail(f"\n  {pct_above:.0f}% / {pct_below:.0f}% split — strongly centred ribbon.")
            elif smaller_pct > 20:
                verdict_1b = "LIKELY_CENTRED"
                warn(f"\n  {pct_above:.0f}% / {pct_below:.0f}% split — likely centred / partial inland bleed.")
            else:
                verdict_1b = "CORRECT_SEAWARD"
                ok(f"\n  {pct_above:.0f}% / {pct_below:.0f}% split — consistent with seaward-only strip.")

    # ── Overall TEST 1 verdict ────────────────────────────────────────────────
    print()
    if pct_926 > 85:
        verdict = "CENTRED_RIBBON"
        fail(f"TEST 1 VERDICT: {pct_926:.1f}% of strip cells within 926 m of HWM.")
        fail(f"  Seaward 0–1852 m strip would be ~50%. Ribbon hypothesis CONFIRMED.")
    elif pct_926 < 60:
        verdict = "CORRECT_SEAWARD"
        ok(f"TEST 1 VERDICT: {pct_926:.1f}% within 926 m — consistent with seaward strip.")
    else:
        verdict = "AMBIGUOUS"
        warn(f"TEST 1 VERDICT: {pct_926:.1f}% within 926 m — ambiguous (50% = correct, 100% = ribbon).")

    return {"pct_within_926": float(pct_926),
            "pct_within_1852": float(pct_1852),
            "max_strip_dist": float(strip_dists.max()),
            "median_strip_dist": float(np.median(strip_dists)),
            "verdict": verdict}


# ── TEST 2: dist_to_hwm_m distribution ───────────────────────────────────────

def test_distance_distribution(df: pd.DataFrame) -> dict:
    hdr("TEST 2 — dist_to_hwm_m distribution (parquet column)")

    if "dist_to_hwm_m" not in df.columns:
        warn("Column 'dist_to_hwm_m' not present — skipping")
        return {"skipped": True}

    d = df["dist_to_hwm_m"].dropna()
    info(f"  n={len(d):,}  min={d.min():.0f} m  max={d.max():.0f} m  "
         f"mean={d.mean():.0f} m  median={d.median():.0f} m")
    print()

    bins   = [0, 100, 250, 500, 750, 926, 1000, 1250, 1500, 1852, 2000, 9999]
    labels = [f"{bins[i]:>4}–{bins[i+1]:<5}" for i in range(len(bins)-1)]
    counts = pd.cut(d, bins=bins, labels=labels).value_counts().sort_index()
    max_c  = counts.max()
    for label, count in counts.items():
        bar = "\u2588" * int(count / max_c * 36)
        info(f"  {label} m: {count:>8,}  {bar}")

    print()
    if d.max() > 1800:
        ok(f"Max {d.max():.0f} m — consistent with 1852 m strip")
    else:
        warn(f"Max {d.max():.0f} m — strip narrower than expected")

    if d.median() < 500:
        warn(f"Median {d.median():.0f} m — low, consistent with narrow/centred strip")
    else:
        ok(f"Median {d.median():.0f} m — consistent with 1852 m seaward strip")

    return {"max_dist": float(d.max()), "median_dist": float(d.median())}


# ── TEST 3: land/sea side (vectorised) ───────────────────────────────────────

def test_land_sea_side(df: pd.DataFrame, cache_dir: Path, n_sample: int) -> dict:
    hdr("TEST 3 — Land / sea side classification (vectorised spatial)")

    try:
        import geopandas as gpd
    except ImportError:
        warn("geopandas not available — skipping TEST 3")
        return {"skipped": True}

    coast_path = cache_dir / "os_coastline_bng.gpkg"
    if not coast_path.exists():
        warn(f"Not found: {coast_path}")
        return {"skipped": True}

    # Stratified sample
    info(f"Sampling {n_sample:,} cells …")
    if "dist_to_hwm_m" in df.columns:
        df["_decile"] = pd.qcut(df["dist_to_hwm_m"], q=10, labels=False,
                                duplicates="drop")
        per_bin = max(1, n_sample // df["_decile"].nunique())
        sample  = (df.groupby("_decile", group_keys=False)
                     .apply(lambda g: g.sample(min(len(g), per_bin),
                                               random_state=42)))
        df.drop(columns=["_decile"], inplace=True)
    else:
        sample = df.sample(min(n_sample, len(df)), random_state=42)
    sample = sample.copy()
    info(f"  Actual sample: {len(sample):,} cells")

    info(f"Loading and extracting coastline segments …")
    coast_gdf = gpd.read_file(coast_path)
    if coast_gdf.crs and coast_gdf.crs.to_epsg() != 27700:
        coast_gdf = coast_gdf.to_crs("EPSG:27700")

    seg_A, seg_B = [], []
    for geom in coast_gdf.geometry:
        if geom is None:
            continue
        geoms = list(geom.geoms) if hasattr(geom, "geoms") else [geom]
        for g in geoms:
            coords = np.array(g.coords)
            seg_A.append(coords[:-1])
            seg_B.append(coords[1:])

    A = np.vstack(seg_A).astype(np.float64)
    B = np.vstack(seg_B).astype(np.float64)
    info(f"  {len(A):,} coastline segments")

    AB      = B - A
    AB_len2 = (AB ** 2).sum(axis=1).clip(min=1e-12)

    # Known sea reference (North Sea, clearly offshore England in BNG)
    SEA_REF = np.array([600_000.0, 400_000.0])

    def classify_chunk(px: np.ndarray, py: np.ndarray) -> np.ndarray:
        """Vectorised signed-distance classification. Returns +1=sea, -1=land."""
        pts    = np.column_stack([px, py])          # (C, 2)
        C      = len(pts)
        sides  = np.empty(C, dtype=np.int8)

        # Batch over segments to find nearest — do in sub-chunks to limit RAM
        SEG_CHUNK = 5000
        nn_dist   = np.full(C, np.inf)
        nn_near   = np.zeros((C, 2))
        nn_ab     = np.zeros((C, 2))

        for s0 in range(0, len(A), SEG_CHUNK):
            s1    = min(s0 + SEG_CHUNK, len(A))
            a_    = A[s0:s1]                        # (S, 2)
            ab_   = AB[s0:s1]                       # (S, 2)
            al2_  = AB_len2[s0:s1]                  # (S,)

            AP    = pts[:, None, :] - a_[None, :, :]     # (C, S, 2)
            t     = (AP * ab_[None, :, :]).sum(2) / al2_[None, :]  # (C, S)
            t     = np.clip(t, 0.0, 1.0)
            near  = a_[None, :, :] + t[:, :, None] * ab_[None, :, :]  # (C, S, 2)
            diff  = pts[:, None, :] - near                              # (C, S, 2)
            dists = np.sqrt((diff ** 2).sum(2))                         # (C, S)

            best  = dists.argmin(axis=1)            # (C,) — best seg in this chunk
            bd    = dists[np.arange(C), best]       # (C,) — best dist

            better = bd < nn_dist
            nn_dist[better]    = bd[better]
            nn_near[better]    = near[np.arange(C), best][better]
            nn_ab[better]      = ab_[best][better]

        # Signed classification
        v      = pts - nn_near                     # (C, 2) vector to query point
        v_ref  = SEA_REF - nn_near                 # (C, 2) vector to sea reference
        cross     = nn_ab[:, 0] * v[:, 1]     - nn_ab[:, 1] * v[:, 0]      # (C,)
        cross_ref = nn_ab[:, 0] * v_ref[:, 1] - nn_ab[:, 1] * v_ref[:, 0]  # (C,)
        sides = np.where(cross * cross_ref >= 0, 1, -1).astype(np.int8)
        return sides

    ex  = sample["easting_bng"].values.astype(np.float64)
    ny_ = sample["northing_bng"].values.astype(np.float64)

    info(f"Classifying {len(ex):,} points (fully vectorised) …")
    info(f"  Should complete in ~10–30 s …")

    POINT_CHUNK = 2000
    all_sides   = []
    for p0 in range(0, len(ex), POINT_CHUNK):
        p1 = min(p0 + POINT_CHUNK, len(ex))
        all_sides.append(classify_chunk(ex[p0:p1], ny_[p0:p1]))
        done = min(p1, len(ex))
        print(f"\r     {done:,} / {len(ex):,} points classified …", end="", flush=True)
    print()

    sides = np.concatenate(all_sides)
    sample["side"] = np.where(sides == 1, "sea", "land")

    n_sea  = (sample["side"] == "sea").sum()
    n_land = (sample["side"] == "land").sum()
    pct_land = 100 * n_land / len(sample)
    pct_sea  = 100 * n_sea  / len(sample)

    print()
    info(f"  Sea side (correct):  {n_sea:>6,}  ({pct_sea:.1f}%)")
    info(f"  Land side (PROBLEM): {n_land:>6,}  ({pct_land:.1f}%)")

    if "dist_to_hwm_m" in sample.columns:
        print()
        info("Breakdown by distance band:")
        bins = [0, 100, 250, 500, 926, 1852, 9999]
        sample["_bin"] = pd.cut(sample["dist_to_hwm_m"], bins=bins)
        tbl = (sample.groupby("_bin", observed=True)["side"]
                     .value_counts().unstack(fill_value=0))
        for c in ["land", "sea"]:
            if c not in tbl.columns:
                tbl[c] = 0
        tbl["pct_land"] = (100 * tbl["land"] /
                           tbl[["land", "sea"]].sum(axis=1).clip(lower=1))
        for idx, row in tbl.iterrows():
            info(f"  {str(idx):>16s}: sea={int(row['sea']):>5,}  "
                 f"land={int(row['land']):>5,}  ({row['pct_land']:.1f}% land)")

    print()
    if pct_land > 30:
        verdict = "CENTRED_RIBBON"
        fail(f"TEST 3 VERDICT: {pct_land:.1f}% on LAND side — ribbon hypothesis CONFIRMED.")
    elif pct_land > 10:
        verdict = "LIKELY_CENTRED"
        warn(f"TEST 3 VERDICT: {pct_land:.1f}% land-side — significant inland contamination.")
    elif pct_land > 5:
        verdict = "BORDERLINE"
        warn(f"TEST 3 VERDICT: {pct_land:.1f}% land-side — borderline.")
    else:
        verdict = "CORRECT_SEAWARD"
        ok(f"TEST 3 VERDICT: {pct_land:.1f}% land-side — placement looks correct.")

    return {"pct_land": pct_land, "pct_sea": pct_sea,
            "n_land": int(n_land), "n_sea": int(n_sea),
            "verdict": verdict, "sample_df": sample}


# ── TEST 4: depth signature ───────────────────────────────────────────────────

def test_depth_signature(df: pd.DataFrame, sample_df=None) -> dict:
    hdr("TEST 4 — Depth signature by land/sea side")

    if "depth_m" not in df.columns:
        warn("Column 'depth_m' not present — skipping")
        return {"skipped": True}

    if sample_df is None or "side" not in sample_df.columns:
        warn("No TEST 3 classification — using dist_to_hwm_m proxy only")
        if "dist_to_hwm_m" not in df.columns:
            return {"skipped": True}
        s = df[["dist_to_hwm_m", "depth_m"]].dropna().sample(
            min(50_000, len(df)), random_state=42)
        bins = [0, 100, 250, 500, 750, 926, 1000, 1250, 1500, 1852, 9999]
        s["_bin"] = pd.cut(s["dist_to_hwm_m"], bins=bins)
        tbl = s.groupby("_bin", observed=True)["depth_m"].agg(
            ["mean", "median", "min", "max", "count"])
        info(f"  {'Band':>16s}  {'n':>7s}  {'mean':>8s}  {'median':>7s}")
        for idx, row in tbl.iterrows():
            info(f"  {str(idx):>16s}  {row['count']:>7.0f}  "
                 f"{row['mean']:>8.1f}  {row['median']:>7.1f}")
        return {"proxy_only": True}

    merged = sample_df[["cell_id", "side"]].merge(
        df[["cell_id", "depth_m"]].dropna(), on="cell_id", how="inner")

    for side in ["sea", "land"]:
        s = merged[merged["side"] == side]["depth_m"]
        if len(s) == 0:
            info(f"  {side}: no cells")
            continue
        info(f"  {side.upper()} ({len(s):,}): mean={s.mean():.1f} m  "
             f"median={s.median():.1f} m  min={s.min():.1f} m  "
             f"pct>1m={100*(s>1).mean():.1f}%")

    land_d = merged[merged["side"] == "land"]["depth_m"]
    if len(land_d) == 0:
        ok("No land-side cells — depth cross-check not applicable")
        return {"n_land_with_depth": 0}

    land_mean = land_d.mean()
    print()
    if land_mean > 1.0:
        warn(f"Land-side cells have mean depth {land_mean:.1f} m — filled from offshore data,")
        warn(f"confirming inland cells exist in the grid.")
    else:
        ok(f"Land-side cells have mean depth {land_mean:.1f} m — near zero, as expected.")

    return {"land_mean_depth": float(land_mean), "n_land_with_depth": int(len(land_d))}


# ── SUMMARY ───────────────────────────────────────────────────────────────────

def print_summary(results: dict) -> None:
    hdr("OVERALL VERDICT")
    verdicts = {k: v.get("verdict") for k, v in results.items()
                if isinstance(v, dict) and "verdict" in v}
    confirmed_centred = sum(1 for v in verdicts.values()
                            if v in ("CENTRED_RIBBON", "LIKELY_CENTRED"))
    confirmed_correct = sum(1 for v in verdicts.values()
                            if v == "CORRECT_SEAWARD")
    for test, verdict in verdicts.items():
        m = ("\u2717" if verdict in ("CENTRED_RIBBON", "LIKELY_CENTRED")
             else "\u2713" if verdict == "CORRECT_SEAWARD" else "\u26a0")
        print(f"  {m}  {test}: {verdict}")
    print()
    if confirmed_centred >= 2:
        fail("CONCLUSION: Grid strip is CENTRED ON THE HWM — ribbon hypothesis CONFIRMED.")
        fail("  ~half of all cells are on land with no real substrate data.")
        print()
        print("  SUGGESTED FIX in build_coastal_grid.py Stage 2:")
        print("    The EDT must run only over sea pixels (not a symmetric buffer).")
        print("    1. Rasterise the land/foreshore polygon to get a land mask.")
        print("    2. Set HWM pixels as the seed; run EDT over sea pixels only.")
        print("    3. Strip mask = EDT result where dist <= 1852 m (sea side only).")
    elif confirmed_correct >= 2:
        ok("CONCLUSION: Grid strip is correctly seaward of the HWM.")
        ok("  Ribbon-centred hypothesis REFUTED.")
    else:
        warn("CONCLUSION: Results mixed — review individual tests above.")


# ── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parquet",    default="output/spearo_coastal_grid_100m_filled.parquet")
    ap.add_argument("--cache",      default="cache/stage1_coastline",
                    help="Stage 1 cache dir (stage2_strip is its sibling)")
    ap.add_argument("--sample",     type=int, default=5000)
    ap.add_argument("--no-spatial", action="store_true",
                    help="Skip TEST 3 (land/sea spatial classification)")
    ap.add_argument("--seed",       type=int, default=42)
    args = ap.parse_args()

    np.random.seed(args.seed)
    parquet_path = Path(args.parquet)
    cache_dir    = Path(args.cache)

    print(SEP)
    print("  verify_strip_placement.py")
    print("  Tests whether the coastal grid is correctly seaward of the HWM.")
    print(SEP)
    info(f"Parquet:   {parquet_path}")
    info(f"Cache:     {cache_dir}")
    info(f"Sample N:  {args.sample:,}")

    if not parquet_path.exists():
        print(f"\nERROR: {parquet_path} not found", file=sys.stderr)
        sys.exit(1)

    info("\nLoading parquet …")
    import pyarrow.parquet as pq
    schema_cols = pq.read_schema(parquet_path).names
    needed      = ["cell_id", "easting_bng", "northing_bng",
                   "dist_to_hwm_m", "depth_m", "zone"]
    load_cols   = [c for c in needed if c in schema_cols]
    df = pd.read_parquet(parquet_path, columns=load_cols)
    info(f"Loaded {len(df):,} rows, columns: {list(df.columns)}")

    results = {}
    results["TEST_1_strip_raster"]  = test_strip_raster(cache_dir)
    results["TEST_2_distance_dist"] = test_distance_distribution(df)

    sample_df = None
    if not args.no_spatial:
        r3 = test_land_sea_side(df, cache_dir, args.sample)
        results["TEST_3_land_sea"] = r3
        sample_df = r3.get("sample_df")
    else:
        info("\nSkipping TEST 3 (--no-spatial)")
        results["TEST_3_land_sea"] = {"skipped": True}

    results["TEST_4_depth"] = test_depth_signature(df, sample_df)

    print_summary(results)


if __name__ == "__main__":
    main()