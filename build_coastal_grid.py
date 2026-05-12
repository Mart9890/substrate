#!/usr/bin/env python3
"""
build_coastal_grid.py  —  Spearo coastal substrate grid builder  (v3)
======================================================================
Raster-first, tree-indexed pipeline.  All geometry work happens ONCE
during dataset rasterisation (Stage 3); every subsequent build stage
uses integer numpy array lookups — no spatial joins at runtime.

Architecture
------------
  Stage 1  Coastline raster      OS HWL → BNG raster (NPZ)              ~10s  once
  Stage 2  Strip + distance      EDT strip mask + dist-to-HWM (NPZ)      ~15s  once
  Stage 3  Dataset rasters       Each vector dataset → int-coded raster  slow  ONCE PER DATASET
  Stage 4  Tree index            Strip cells × raster lookup → tile NPZ  ~60s  once per cell size
  Stage 5  Export                Tile NPZ → SQLite + Parquet              ~30s  always (fast)

v3 changes vs v2
----------------
  Strip polygon is now SEAWARD-ONLY.  The old approach buffered the HWL
  symmetrically (±STRIP_M) then subtracted only a 5 m inner notch, leaving
  a ribbon of inland cells equal in width to the seaward strip.

  The fix: polygonize the OS HWL closed rings into a land polygon, then
  subtract the full land polygon from the outer buffer:

      outer = buffer(HWL_features, +STRIP_M)   # symmetric, as before
      strip = outer.difference(land_polygon)    # subtract ALL land → seaward only

  The OS HWL diagnostic confirmed 5,231 closed rings, 0 open chains,
  0 dangling endpoints — polygonization is safe and complete.

  The land polygon is cached as land_polygon_{cs}m.gpkg in stage1_coastline/.

Cache layout (per cell size, e.g. 100m)
-----------------------------------------
  cache/
    stage1_coastline/
      coastline_raster_{RES}m.npz          ← binary BNG raster of HWM line
      land_polygon_{RES}m.gpkg             ← polygonized land mask  [NEW v3]
    stage2_strip/
      strip_mask_{RES}m.npz                ← bool mask, cells inside HWM→STRIP_M band
      dist_hwm_{RES}m.npz                  ← float32 metres to nearest HWM pixel
      raster_meta_{RES}m.json              ← origin, cell_size, n_rows, n_cols
    stage3_rasters/
      ukash_{RES}m.npz  + ukash_lookup.json
      bgs_sbs_obs_{RES}m.npz  + bgs_sbs_obs_lookup.json
      bgs_sbs_pred_{RES}m.npz  + bgs_sbs_pred_lookup.json
      bgs_bedrock_{RES}m.npz  + bgs_bedrock_lookup.json
      defr_{RES}m.npz  + defr_lookup.json
      emodnet_depth_{RES}m.npz             ← float32 direct (no lookup)
      emodnet_slope_{RES}m.npz             ← float32 direct (no lookup)
    stage4_index_{RES}m/
      tiles/
        tile_E{:04d}_N{:04d}.npz           ← one file per 100km×100km tile
      tile_registry.json                   ← tile → path + dataset coverage
  output/
    spearo_coastal_grid_{RES}m.db          ← SQLite, full schema + indexes
    spearo_coastal_grid_{RES}m.parquet     ← GeoParquet flat table
    build_manifest_{RES}m.json

Usage
-----
  python build_coastal_grid.py --root raw
  python build_coastal_grid.py --root raw --cell-size 50
  python build_coastal_grid.py --root raw --sample 2000
  python build_coastal_grid.py --root raw --force stage3
  python build_coastal_grid.py --root raw --force stage3:ukash
  python build_coastal_grid.py --root raw --force all

Dependencies
------------
  pip install geopandas pyogrio rasterio numpy shapely pyproj pandas tqdm pyarrow scipy fiona
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sqlite3
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

# Force UTF-8 on Windows stdout/stderr so Unicode log chars don't crash
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore")

# ── Dependency check ──────────────────────────────────────────────────────────

_missing = []
for _pkg in ["geopandas", "pyogrio", "rasterio", "numpy", "shapely",
             "pandas", "pyproj", "tqdm", "pyarrow", "scipy"]:
    try:
        __import__(_pkg)
    except ImportError:
        _missing.append(_pkg)

if _missing:
    print(f"✗ Missing dependencies: pip install {' '.join(_missing)}")
    sys.exit(1)

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.features
import rasterio.transform
import rasterio.warp
from pyproj import Transformer
from scipy.ndimage import distance_transform_edt
from shapely.geometry import box
from shapely.ops import unary_union
from tqdm import tqdm


# ── Constants ─────────────────────────────────────────────────────────────────

EPSG_BNG     = 27700
STRIP_M      = 500        # strip outer edge (metres from HWM)  ← 500m spearfisher zone
NEARSHORE_M  = 200        # nearshore zone boundary (m) — roughly 0-200m / 200-500m
TILE_SIZE_M  = 100_000    # 100 km tiles — about 40 tiles for England coast

# Keep ONE_NM_M as alias in case anything references it
ONE_NM_M     = STRIP_M

# England BNG bounding box — coarse clip for all datasets on load
ENGLAND_BBOX_BNG  = (82000, 5340, 655000, 657000)   # minE, minN, maxE, maxN
# Same extent in WGS84 (lon/lat) — used when reading non-BNG datasets
ENGLAND_BBOX_WGS84 = (-6.5, 49.5, 2.2, 61.0)        # minLon, minLat, maxLon, maxLat

# ── Dataset registry ──────────────────────────────────────────────────────────
# Each entry:  key → { path, type, fields, nodata_id }
#   type: "vector_polygon" | "raster_float"
#   fields: list of attribute columns to encode in the lookup table
#   nodata_id: integer used in raster for "no data" pixels (0 = reserved for nodata)

DATASETS = {
    "ukash": {
        "path":      "UKASH_CombinedMap_v2025/UKASH_Combined_Map_2025.gdb",
        "layer":     "UKASH_Combined_Map_v2025_1",
        "type":      "vector_polygon",
        "fields":    ["EUNISCode", "MHCCode", "MESH_conf", "SNCB_UID", "EUNISTranR", "EUNISL3"],
        "nodata_id": 0,
    },
    "bgs_sbs_obs": {
        "path":      "bgs-offshore-sbs-250k-geopackage/BGS_250k_SeabedSediments_WGS84_v3_FOLK.gpkg",
        "type":      "vector_polygon",
        "fields":    ["RCS", "RCS_D"],
        "nodata_id": 0,
    },
    "bgs_sbs_pred": {
        "path":      "Predictive_SBS_UK_V1_GeoPackage/BGS_Predictive_Seabed_Sediments_UK_v1.gpkg",
        "type":      "vector_polygon",
        "fields":    ["RCS", "RCS_D", "pct_gravel", "pct_sand", "pct_mud"],
        "nodata_id": 0,
    },
    "bgs_bedrock": {
        "path":      "offshore-bedrock-250k-geopackage/BGS_BedrockOffshore_250k_WGS84_v3.gpkg",
        "type":      "vector_polygon",
        "fields":    ["LEX_RCS", "LEX_RCS_D"],
        "nodata_id": 0,
    },
    "defr": {
        "path":      "Intertidal Substrate Foreshore (England and Scotland)/DEFR00000009.shp",
        "type":      "vector_polygon",
        "fields":    ["FORE_DESC", "BACK_DESC"],   # actual field names in DEFR00000009.shp
        "crs_bng":   True,                         # file has unknown CRS but coords are BNG
        "nodata_id": 0,
    },
    "emodnet_depth": {
        "path":      "emodnet_bathymetry/emodnet_depth_england_bng.tif",
        "type":      "raster_float",
    },
    "emodnet_slope": {
        "path":      "emodnet_bathymetry/emodnet_slope_england_bng.tif",
        "type":      "raster_float",
    },
}

OS_COAST_PATH = "os_coastline/high_water_polyline.shp"

# ── Normalisation lookup tables ───────────────────────────────────────────────

FOLK_TO_PRIMARY = {
    "ROCK": "rock",  "BCR": "rock",   "BDRK": "rock",
    "G":    "gravel","sG":  "gravel", "mG":   "gravel","gmS":  "gravel","GV":   "gravel",
    "S":    "sand",  "gS":  "sand",   "msS":  "sand",  "SND":  "sand",
    "SLGVSD": "sand","SNDGM":"sand",
    "M":    "mud",   "sM":  "mud",    "gM":   "mud",   "MUD":  "mud",
    "mS":   "mixed", "gsM": "mixed",  "MSND": "mixed", "MXSD": "mixed",
    "(g)S": "sand",  "(g)M":"mud",    "(g)sM":"mud",   "(g)mS":"mixed",
    "msG":  "gravel","BIOM":"mixed",
}

FOLK_TO_HARDNESS = {
    "ROCK": "hard",  "BCR": "hard",  "BDRK": "hard",
    "G":    "hard",  "sG":  "hard",  "mG":   "hard",  "GV":   "hard",
    "S":    "soft",  "gS":  "soft",  "msS":  "soft",  "SND":  "soft",
    "(g)S": "soft",
    "M":    "soft",  "sM":  "soft",  "MUD":  "soft",  "(g)M": "soft",
    "mS":   "mixed", "MSND":"mixed", "gsM":  "mixed", "BIOM": "mixed",
    "msG":  "hard",
}

SOURCE_CONFIDENCE = {
    "BGS_observed":     0.90,
    "BGS_predictive":   0.65,
    "DEFR":             0.60,
    "UKASH_survey":     0.85,
    "UKASH_predictive": 0.55,
    "none":             0.0,
}


# ── Logging ───────────────────────────────────────────────────────────────────

class Logger:
    """
    Dual-output logger: writes timestamped lines to both stdout and a log file.

    Every line is prefixed with an absolute timestamp and a delta-since-last-log,
    making it easy to spot exactly where time is being consumed.

    Format:
        [HH:MM:SS +Xs]   <indent><message>  [elapsed_in_stage Xs]

    Log file: log/build_coastal_grid_<YYYYMMDD_HHMMSS>.log
    """

    def __init__(self, log_dir: Path | None = None):
        self._start      = time.time()
        self._last       = self._start
        self._file       = None
        self._log_path   = None

        if log_dir is not None:
            log_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._log_path = log_dir / f"build_coastal_grid_{ts}.log"
            self._file = open(self._log_path, "w", buffering=1, encoding="utf-8")

    def _emit(self, line: str):
        now     = time.time()
        wall    = datetime.now().strftime("%H:%M:%S")
        delta   = now - self._last
        self._last = now
        prefix  = f"[{wall} +{delta:5.1f}s]  "
        full    = prefix + line
        print(full, flush=True)
        if self._file:
            self._file.write(full + "\n")
            self._file.flush()

    def msg(self, msg: str, indent: int = 0, elapsed: float | None = None):
        suffix = f"  [{elapsed:.1f}s]" if elapsed is not None else ""
        self._emit("  " * indent + msg + suffix)

    def section(self, title: str, step: int | None = None, total: int = 5):
        tag  = f"[{step}/{total}] " if step else ""
        sep  = "=" * 60
        self._emit("")
        self._emit(sep)
        self._emit(f"{tag}{title}")
        self._emit(sep)

    def cached(self, path: Path, label: str | None = None) -> bool:
        if path.exists() and path.stat().st_size > 0:
            name = label or path.name
            size = path.stat().st_size / 1_048_576
            self.msg(f"✓ Cached: {name}  ({size:.1f} MB)  — delete to rebuild", 1)
            return True
        return False

    def mem(self, tag: str = ""):
        """Log current process RSS memory usage."""
        try:
            import psutil
            rss = psutil.Process().memory_info().rss / 1_048_576
            self.msg(f"  MEM {tag}: {rss:.0f} MB RSS", indent=0)
        except ImportError:
            pass

    def close(self):
        if self._file:
            self._file.close()
            self._file = None

    @property
    def log_path(self) -> Path | None:
        return self._log_path


# Module-level logger instance — replaced in main() once log dir is known
_LOG = Logger(log_dir=None)


def log(msg: str, indent: int = 0, elapsed: float | None = None):
    _LOG.msg(msg, indent=indent, elapsed=elapsed)

def section(title: str, step: int | None = None, total: int = 5):
    _LOG.section(title, step=step, total=total)

def cached(path: Path, label: str | None = None) -> bool:
    return _LOG.cached(path, label=label)


# ── CRS / geometry helpers ────────────────────────────────────────────────────

def force_bng(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.crs is None:
        log("⚠ No CRS — assuming BNG", indent=2)
        return gdf.set_crs(EPSG_BNG)
    if gdf.crs.to_epsg() != EPSG_BNG:
        return gdf.to_crs(EPSG_BNG)
    return gdf

def bbox_gdf(gdf: gpd.GeoDataFrame, bbox=ENGLAND_BBOX_BNG) -> gpd.GeoDataFrame:
    return gdf[gdf.geometry.intersects(box(*bbox))].copy()

def _safe_float(v):
    if v is None: return None
    try:
        f = float(v)
        return None if math.isnan(f) else round(f, 4)
    except (TypeError, ValueError):
        return None

def _safe_str(v):
    if v is None: return None
    s = str(v).strip()
    return s if s and s.lower() not in ("nan", "none", "") else None

def slope_to_morphology(deg) -> str | None:
    try:
        d = float(deg)
    except (TypeError, ValueError):
        return None
    if math.isnan(d): return None
    if d < 1.0:  return "flat"
    if d < 5.0:  return "gentle_slope"
    if d < 15.0: return "slope"
    if d < 30.0: return "steep"
    return "cliff"


# ── Raster grid metadata ──────────────────────────────────────────────────────

class RasterMeta:
    """Shared BNG raster grid parameters — all rasters are aligned to this."""

    def __init__(self, origin_e: float, origin_n: float,
                 cell_size: int, n_cols: int, n_rows: int):
        self.origin_e  = origin_e    # west edge of grid (BNG easting)
        self.origin_n  = origin_n    # north edge of grid (BNG northing)
        self.cell_size = cell_size
        self.n_cols    = n_cols
        self.n_rows    = n_rows

    # ── coordinate → pixel index ──────────────────────────────────────────────
    def e_to_col(self, e: np.ndarray) -> np.ndarray:
        return ((e - self.origin_e) / self.cell_size).astype(np.int32)

    def n_to_row(self, n: np.ndarray) -> np.ndarray:
        # rows run south: row 0 = northernmost
        return ((self.origin_n - n) / self.cell_size).astype(np.int32)

    def col_to_e(self, col: np.ndarray) -> np.ndarray:
        return self.origin_e + (col + 0.5) * self.cell_size

    def row_to_n(self, row: np.ndarray) -> np.ndarray:
        return self.origin_n - (row + 0.5) * self.cell_size

    def rasterio_transform(self):
        return rasterio.transform.from_origin(
            self.origin_e, self.origin_n,
            self.cell_size, self.cell_size,
        )

    def to_dict(self) -> dict:
        return {
            "origin_e":  self.origin_e,
            "origin_n":  self.origin_n,
            "cell_size": self.cell_size,
            "n_cols":    self.n_cols,
            "n_rows":    self.n_rows,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RasterMeta":
        return cls(d["origin_e"], d["origin_n"], d["cell_size"],
                   d["n_cols"], d["n_rows"])

    @classmethod
    def from_strip_geom(cls, strip_geom, cell_size: int) -> "RasterMeta":
        minx, miny, maxx, maxy = strip_geom.bounds
        origin_e = math.floor(minx / cell_size) * cell_size
        origin_n = math.ceil(maxy  / cell_size) * cell_size
        n_cols   = math.ceil((maxx - origin_e) / cell_size)
        n_rows   = math.ceil((origin_n - miny)  / cell_size)
        return cls(origin_e, origin_n, cell_size, n_cols, n_rows)


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1 helpers
# ─────────────────────────────────────────────────────────────────────────────

def _buf_geom(args):
    """
    Module-level buffer worker — MUST be at module scope so Windows
    multiprocessing (spawn) can pickle it.  Inner/nested functions can't
    be pickled on Windows.
    args: (geometry, dist, resolution)
    """
    g, dist, resolution = args
    return g.buffer(dist, resolution=resolution)


def _parallel_buffer(geoms, dist: float, resolution: int = 4,
                     n_workers: int | None = None) -> object:
    """
    Buffer a list of Shapely geometries in parallel using a process pool,
    then union the results.  Much faster than buffering a single merged
    MultiLineString on one thread.

    Returns a single (possibly Multi-) Polygon.
    """
    import concurrent.futures
    import os
    from shapely.ops import unary_union as _uu

    workers  = n_workers or min(16, os.cpu_count() or 4)
    job_args = [(g, dist, resolution) for g in geoms]
    chunk    = max(1, len(job_args) // workers)

    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
        buffered = list(ex.map(_buf_geom, job_args, chunksize=chunk))

    return _uu(buffered)


def _build_land_polygon(coast_geom, cache_dir: Path, cell_size: int) -> object:
    """
    Polygonize the OS HWL closed rings into a single land polygon and cache it.

    The OS HWL diagnostic confirmed:
      - 5,231 closed rings, 0 open chains, 0 dangling endpoints
    Polygonization is therefore safe and produces a complete land mask.

    Filters out slivers < 1 km² (polygonizer artefacts), then unions all
    valid polygons into one MultiPolygon.

    Cached as land_polygon_{cell_size}m.gpkg in stage1_coastline/.
    Returns a Shapely geometry (Polygon or MultiPolygon) in EPSG:27700.
    """
    from shapely.ops import polygonize_full, linemerge, unary_union as _uu

    s1_dir    = cache_dir / "stage1_coastline"
    land_gpkg = s1_dir / f"land_polygon_{cell_size}m.gpkg"

    if land_gpkg.exists() and land_gpkg.stat().st_size > 0:
        log(f"  ✓ Cached land polygon — loading {land_gpkg.name}...", indent=2)
        t1   = time.time()
        land = gpd.read_file(land_gpkg).geometry.iloc[0]
        log(f"    Loaded  ({time.time()-t1:.1f}s)", indent=2)
        return land

    log("  Polygonizing HWL into land polygon...", indent=2)
    t1 = time.time()

    merged = linemerge(coast_geom)
    result, dangles, cut_edges, invalid = polygonize_full(merged)

    polys = (list(result.geoms) if hasattr(result, "geoms")
             else ([result] if not result.is_empty else []))

    log(f"    Polygonizer: {len(polys):,} raw polygons  ({time.time()-t1:.1f}s)",
        indent=3)

    if not polys:
        raise RuntimeError(
            "Polygonization produced no polygons — HWL may not be fully closed. "
            "Run check_hwl_completeness.py --polygonize to diagnose."
        )

    # Filter slivers (< 1 km²)
    MIN_AREA_M2  = 1_000_000
    polys_valid  = [p for p in polys if p.area >= MIN_AREA_M2]
    log(f"    After sliver filter (≥1 km²): {len(polys_valid):,} polygons "
        f"({len(polys)-len(polys_valid):,} removed)", indent=3)

    if not polys_valid:
        raise RuntimeError(
            "No polygons ≥ 1 km² after polygonization — unexpected result."
        )

    t2   = time.time()
    land = _uu(polys_valid)
    log(f"    Union done  ({time.time()-t2:.1f}s)  "
        f"area={land.area/1e6:,.0f} km²", indent=3)

    t2 = time.time()
    gpd.GeoDataFrame({"id": [1]}, geometry=[land], crs=EPSG_BNG).to_file(
        land_gpkg, driver="GPKG"
    )
    log(f"    Cached → {land_gpkg.name}  "
        f"({land_gpkg.stat().st_size/1_048_576:.1f} MB)  "
        f"({time.time()-t2:.1f}s)", indent=3, elapsed=time.time()-t1)

    return land


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1 — Coastline raster
# ─────────────────────────────────────────────────────────────────────────────

def stage1_coastline(root: Path, cache_dir: Path, cell_size: int) -> tuple[gpd.GeoDataFrame, np.ndarray, RasterMeta]:
    """
    Load OS HWL (or DEFR fallback), compute the seaward-only strip polygon,
    build the shared RasterMeta, and save a binary raster of the coastline pixels.

    Cache layout (inside cache/stage1_coastline/):
      os_coastline_bng.gpkg          — raw features reprojected to BNG
      land_polygon_{cs}m.gpkg        — polygonized land mask  [v3]
      strip_polygon_{cs}m.gpkg       — seaward-only strip polygon  [v3 fix]
      coastline_raster_{cs}m.npz     — uint8 raster of HWM pixels
      (raster_meta_{cs}m.json lives in stage2_strip/ for legacy reasons)

    Returns: (coast_gdf, coast_raster, meta)
    """
    import os as _os

    s1_dir      = cache_dir / "stage1_coastline"
    rast_out    = s1_dir / f"coastline_raster_{cell_size}m.npz"
    meta_out    = cache_dir / "stage2_strip" / f"raster_meta_{cell_size}m.json"
    strip_gpkg  = s1_dir / f"strip_polygon_{cell_size}m.gpkg"
    gpkg_out    = s1_dir / "os_coastline_bng.gpkg"

    # ── Full cache hit: raster + meta → skip all geometry work ────────────────
    if rast_out.exists() and rast_out.stat().st_size > 0 and meta_out.exists():
        log("✓ Stage 1 fully cached — loading NPZ + meta", indent=1)
        meta      = RasterMeta.from_dict(json.loads(meta_out.read_text()))
        coast_arr = np.load(rast_out)["coast"]
        coast_gdf = gpd.read_file(gpkg_out) if gpkg_out.exists() else None
        log(f"  Grid: {meta.n_cols:,} × {meta.n_rows:,}  "
            f"({meta.n_cols*meta.n_rows:,} candidate cells)", indent=2)
        return coast_gdf, coast_arr, meta

    section("Stage 1: Coastline → raster", 1, 5)
    t0 = time.time()

    os_path   = root / OS_COAST_PATH
    defr_path = root / DATASETS["defr"]["path"]
    s1_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: load raw coastline features ───────────────────────────────────
    if gpkg_out.exists():
        log("  Loading cached coastline gpkg...", indent=1)
        t1 = time.time()
        coast_gdf = gpd.read_file(gpkg_out)
        source    = coast_gdf["source"].iloc[0] if "source" in coast_gdf.columns else "cached"
        log(f"    {len(coast_gdf):,} features  ({time.time()-t1:.1f}s)", indent=2)
    elif os_path.exists():
        log("  Loading OS high water line...", indent=1)
        t1  = time.time()
        raw = gpd.read_file(os_path)
        raw = force_bng(raw)
        raw = bbox_gdf(raw)
        log(f"    {len(raw):,} features loaded  ({time.time()-t1:.1f}s)", indent=2)
        source = "OS_HWL"
        log("  Saving coastline gpkg cache...", indent=1)
        t1 = time.time()
        raw["source"] = source
        raw.to_file(gpkg_out, driver="GPKG")
        coast_gdf = raw
        log(f"    Saved {gpkg_out.name}  ({time.time()-t1:.1f}s)", indent=2)
    else:
        log("  OS coastline not found — falling back to DEFR boundary", indent=1)
        t1  = time.time()
        raw = gpd.read_file(defr_path, engine="pyogrio", bbox=box(*ENGLAND_BBOX_BNG))
        raw = force_bng(raw)
        raw = bbox_gdf(raw)
        raw["source"] = "DEFR_proxy"
        raw.to_file(gpkg_out, driver="GPKG")
        coast_gdf = raw
        source    = "DEFR_proxy"
        log(f"    {len(raw):,} DEFR features  ({time.time()-t1:.1f}s)", indent=2)

    n_feats = len(coast_gdf)

    # ── Step 2: merge all geometries into one ─────────────────────────────────
    log("  Merging features → unary_union...", indent=1)
    t1 = time.time()
    coast_geom = unary_union(coast_gdf.geometry)
    log(f"    Done  ({time.time()-t1:.1f}s)", indent=2)

    # ── Step 3: build seaward-only strip polygon ───────────────────────────────
    #
    # v3 FIX: The old approach did outer.difference(inner_5m), which left a
    # symmetric ribbon of inland cells equal in width to the seaward strip.
    #
    # The correct approach:
    #   1. Polygonize the HWL closed rings → land polygon
    #   2. outer = buffer(HWL_features, +STRIP_M)   — symmetric, as before
    #   3. strip = outer.difference(land_polygon)    — subtract ALL land
    #
    # This gives a true seaward-only strip, 0–STRIP_M from the HWM.
    if strip_gpkg.exists() and strip_gpkg.stat().st_size > 0:
        log(f"  ✓ Cached strip polygon — loading {strip_gpkg.name}...", indent=1)
        t1    = time.time()
        strip = gpd.read_file(strip_gpkg).geometry.iloc[0]
        log(f"    Loaded  ({time.time()-t1:.1f}s)", indent=2)
    else:
        log("  Building seaward-only strip polygon...", indent=1)
        t1 = time.time()

        # Step 3a: polygonize HWL → land polygon (cached separately)
        land = _build_land_polygon(coast_geom, cache_dir, cell_size)

        # Step 3b: outer buffer (+STRIP_M, symmetric — land side trimmed next)
        log(f"    Simplifying {n_feats:,} features (50m tolerance)...", indent=2)
        t2 = time.time()
        simple_geoms = [g.simplify(50, preserve_topology=True)
                        for g in coast_gdf.geometry]
        log(f"    Simplified  ({time.time()-t2:.1f}s)", indent=2)

        n_workers = min(16, _os.cpu_count() or 4)
        log(f"    Buffering outer (+{STRIP_M}m) across {n_workers} workers...", indent=2)
        t2    = time.time()
        outer = _parallel_buffer(simple_geoms, STRIP_M, resolution=4,
                                 n_workers=n_workers)
        log(f"    Outer buffer done  ({time.time()-t2:.1f}s)", indent=2)

        # Step 3c: subtract land polygon → seaward-only strip
        log("    Subtracting land polygon → seaward-only strip...", indent=2)
        t2    = time.time()
        strip = outer.difference(land)
        log(f"    Difference done  ({time.time()-t2:.1f}s)", indent=2)

        # Cache
        t2 = time.time()
        gpd.GeoDataFrame({"id": [1]}, geometry=[strip], crs=EPSG_BNG).to_file(
            strip_gpkg, driver="GPKG")
        log(f"    Strip polygon cached → {strip_gpkg.name}  ({time.time()-t2:.1f}s)",
            indent=2, elapsed=time.time()-t1)

    # ── Step 4: derive RasterMeta from strip extent ───────────────────────────
    meta = RasterMeta.from_strip_geom(strip, cell_size)
    log(f"  Grid: {meta.n_cols:,} cols × {meta.n_rows:,} rows = "
        f"{meta.n_cols*meta.n_rows:,} candidate cells", indent=2)

    # ── Step 5: rasterise the HWM line ────────────────────────────────────────
    log("  Rasterising coastline...", indent=1)
    t1 = time.time()
    coast_arr = rasterio.features.rasterize(
        [(coast_geom, 1)],
        out_shape=(meta.n_rows, meta.n_cols),
        transform=meta.rasterio_transform(),
        fill=0,
        dtype=np.uint8,
        all_touched=True,
    )
    log(f"    {int(coast_arr.sum()):,} coastline pixels  ({time.time()-t1:.1f}s)", indent=2)

    # ── Step 6: save raster + meta ────────────────────────────────────────────
    log("  Saving raster cache...", indent=1)
    t1 = time.time()
    np.savez_compressed(rast_out, coast=coast_arr)
    (cache_dir / "stage2_strip").mkdir(parents=True, exist_ok=True)
    meta_out.write_text(json.dumps(meta.to_dict(), indent=2))
    log(f"    {rast_out.name}  ({rast_out.stat().st_size/1_048_576:.1f} MB)  "
        f"({time.time()-t1:.1f}s)", indent=2)

    log(f"✓ Stage 1 complete  ({source})", indent=1, elapsed=time.time()-t0)
    return coast_gdf, coast_arr, meta


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2 — Strip mask + distance-to-HWM raster
# ─────────────────────────────────────────────────────────────────────────────

def stage2_strip(coast_gdf, coast_arr: np.ndarray,
                 meta: RasterMeta, cache_dir: Path,
                 cell_size: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (strip_mask, dist_hwm) — both shape (n_rows, n_cols).
    strip_mask: uint8, 1 = cell centre is inside the seaward HWM→STRIP_M band
    dist_hwm:   float32, metres to nearest HWM pixel (Euclidean, always >= 0)

    The strip is seaward-only: the land polygon (polygonized from the fully-closed
    HWL rings) is subtracted from the symmetric outer buffer, so no inland cells
    are included.
    """
    s2_dir   = cache_dir / "stage2_strip"
    mask_out = s2_dir / f"strip_mask_{cell_size}m.npz"
    dist_out = s2_dir / f"dist_hwm_{cell_size}m.npz"

    if cached(mask_out, f"strip_mask_{cell_size}m.npz") and dist_out.exists():
        log(f"✓ Cached: dist_hwm_{cell_size}m.npz", indent=1)
        strip_mask = np.load(mask_out)["mask"]
        dist_hwm   = np.load(dist_out)["dist"]
        return strip_mask, dist_hwm

    section("Stage 2: Strip mask + distance raster", 2, 5)
    t0 = time.time()

    # ── Step 1: load strip polygon from Stage 1 cache ─────────────────────────
    # Stage 1 always builds and caches the strip polygon before Stage 2 runs.
    # The fallback path handles --force stage2 without --force stage1.
    log("Loading strip polygon...", indent=1)
    t1 = time.time()

    s1_dir     = cache_dir / "stage1_coastline"
    strip_gpkg = s1_dir / f"strip_polygon_{cell_size}m.gpkg"
    strip      = None

    if strip_gpkg.exists() and strip_gpkg.stat().st_size > 0:
        log(f"  Loading cached strip polygon from Stage 1...", indent=2)
        strip_gdf = gpd.read_file(strip_gpkg)
        strip     = strip_gdf.geometry.iloc[0] if len(strip_gdf) > 0 else None
        if strip is not None:
            log(f"  Loaded  ({time.time()-t1:.1f}s)", indent=2)

    if strip is None:
        # Rebuild seaward-only strip — same approach as Stage 1.
        log("  Strip polygon not cached — rebuilding (seaward-only)...", indent=2)
        if coast_gdf is not None:
            import os as _os2
            coast_geom_s2 = unary_union(coast_gdf.geometry)
            land          = _build_land_polygon(coast_geom_s2, cache_dir, cell_size)

            geoms        = list(coast_gdf.geometry)
            n_workers    = min(16, _os2.cpu_count() or 4)
            log(f"    Simplifying {len(geoms):,} features (50m tolerance)...", indent=3)
            simple_geoms = [g.simplify(50, preserve_topology=True) for g in geoms]
            log(f"    Buffering outer (+{STRIP_M}m) across {n_workers} workers...", indent=3)
            outer = _parallel_buffer(simple_geoms, STRIP_M, resolution=4,
                                     n_workers=n_workers)
            log(f"    Subtracting land polygon → seaward-only strip...", indent=3)
            strip = outer.difference(land)

            gpd.GeoDataFrame({"id": [1]}, geometry=[strip], crs=EPSG_BNG).to_file(
                strip_gpkg, driver="GPKG"
            )
            log(f"    Strip polygon cached → {strip_gpkg.name}", indent=3)
        else:
            raise RuntimeError(
                "No strip polygon available and no coast_gdf to rebuild it. "
                "Run with --force stage1 to rebuild from scratch."
            )

    # ── Step 2: rasterise strip polygon → strip mask ──────────────────────────
    log("Rasterising strip polygon...", indent=1)
    t1 = time.time()
    strip_mask = rasterio.features.rasterize(
        [(strip, 1)],
        out_shape=(meta.n_rows, meta.n_cols),
        transform=meta.rasterio_transform(),
        fill=0,
        dtype=np.uint8,
        all_touched=False,
    )
    n_cells = int(strip_mask.sum())
    log(f"  {n_cells:,} cells in strip  ({time.time()-t1:.1f}s)", indent=2)

    # ── Step 3: distance-to-HWM via EDT ───────────────────────────────────────
    # Runs over the full raster (both sides of HWL) — this is correct.
    # dist_hwm is the unsigned Euclidean distance to the nearest HWM pixel.
    # Used for zone classification and confidence decay.
    # The strip_mask enforces seaward-only selection, so dist_hwm values for
    # any inland pixels are never surfaced in the output.
    log("Computing distance-to-HWM (EDT)...", indent=1)
    t1 = time.time()
    dist_hwm = (distance_transform_edt(1 - coast_arr) * cell_size).astype(np.float32)
    log(f"  Done  ({time.time()-t1:.1f}s)", indent=2)

    # ── Step 4: save ──────────────────────────────────────────────────────────
    s2_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(mask_out, mask=strip_mask)
    np.savez_compressed(dist_out, dist=dist_hwm)
    log(f"✓ Strip mask + distance saved  ({n_cells:,} cells)",
        indent=1, elapsed=time.time()-t0)

    return strip_mask, dist_hwm


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 3 — Rasterise each dataset
# ─────────────────────────────────────────────────────────────────────────────

def _rasterise_vector(gdf: gpd.GeoDataFrame, fields: list[str],
                      meta: RasterMeta) -> tuple[np.ndarray, dict]:
    """
    Burn each polygon into the shared raster grid.
    Returns:
      id_raster  — int32 array, shape (n_rows, n_cols), 0=nodata, 1..N = feature index
      lookup     — dict: id → {field: value, ...}  (id matches non-zero raster values)
    """
    gdf = gdf.reset_index(drop=True)

    # Build lookup table: id=1..N for each feature (0 = nodata)
    lookup = {}
    for i, row in enumerate(gdf.itertuples(index=False), start=1):
        rec = {}
        for f in fields:
            v = getattr(row, f, None)
            if v is not None and not (isinstance(v, float) and math.isnan(v)):
                rec[f] = v if not isinstance(v, float) else round(v, 4)
            else:
                rec[f] = None
        lookup[i] = rec

    # Burn: each polygon burns its 1-based index into the raster
    # We use int32 to support up to ~2M features
    id_raster = np.zeros((meta.n_rows, meta.n_cols), dtype=np.int32)

    shapes = (
        (geom, int(i))
        for i, geom in zip(range(1, len(gdf)+1), gdf.geometry)
        if geom is not None and not geom.is_empty
    )

    rasterio.features.rasterize(
        shapes,
        out=id_raster,
        transform=meta.rasterio_transform(),
        fill=0,
        dtype=np.int32,
        merge_alg=rasterio.enums.MergeAlg.replace,
    )

    return id_raster, lookup


def _resample_raster_to_meta(tif_path: Path, meta: RasterMeta) -> np.ndarray:
    """
    Resample a GeoTIFF to the shared BNG raster grid.
    Returns float32 array (n_rows, n_cols), NaN where no data.
    """
    with rasterio.open(tif_path) as src:
        dst_transform = meta.rasterio_transform()
        dst_crs       = f"EPSG:{EPSG_BNG}"
        dst_arr       = np.full((meta.n_rows, meta.n_cols), np.nan, dtype=np.float32)

        rasterio.warp.reproject(
            source=rasterio.band(src, 1),
            destination=dst_arr,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=rasterio.warp.Resampling.bilinear,
            dst_nodata=np.nan,
        )

        nodata = src.nodata
        if nodata is not None:
            dst_arr = np.where(dst_arr == nodata, np.nan, dst_arr)

    return dst_arr


def stage3_rasters(root: Path, cache_dir: Path,
                   meta: RasterMeta, cell_size: int,
                   force_dataset: str | None = None) -> dict:
    """
    Rasterise each dataset once.  Returns dict: key → npz_path (or None if missing).
    Skips datasets whose npz already exists unless force_dataset matches.
    """
    section("Stage 3: Rasterising datasets", 3, 5)
    s3_dir = cache_dir / "stage3_rasters"
    s3_dir.mkdir(parents=True, exist_ok=True)

    result = {}

    for key, cfg in DATASETS.items():
        src_path  = root / cfg["path"]
        npz_out   = s3_dir / f"{key}_{cell_size}m.npz"
        lut_out   = s3_dir / f"{key}_lookup.json"

        # Check force
        if force_dataset and force_dataset != key:
            pass  # still check cache normally
        elif force_dataset == key:
            for p in [npz_out, lut_out]:
                if p.exists():
                    p.unlink()
                    log(f"  Force cleared: {p.name}", indent=1)

        if not src_path.exists():
            log(f"  ⚠ Not found (skipping): {cfg['path']}", indent=1)
            result[key] = None
            continue

        if cached(npz_out, f"{key}_{cell_size}m.npz"):
            result[key] = npz_out
            continue

        log(f"  Rasterising {key}...", indent=1)
        t0 = time.time()

        if cfg["type"] == "vector_polygon":
            fields = cfg.get("fields", [])
            layer  = cfg.get("layer", None)

            log(f"    Loading {src_path.name}...", indent=2)

            # Detect native CRS to choose the right bbox — passing BNG coords
            # to a WGS84 file returns 0 features.
            src_epsg = None
            try:
                import pyogrio as _pyogrio
                _info = _pyogrio.read_info(str(src_path), **({"layer": layer} if layer else {}))
                if _info.get("crs"):
                    from pyproj import CRS as _CRS
                    src_epsg = _CRS(_info["crs"]).to_epsg()
                elif cfg.get("crs_bng"):
                    # Dataset has no embedded CRS but we know coords are BNG
                    src_epsg = EPSG_BNG
                elif src_epsg is None:
                    # Unknown CRS — inspect the bbox extent to guess coordinate system.
                    # BNG coords are in the hundreds of thousands; WGS84 lon/lat are tiny.
                    _bbox = _info.get("total_bounds")  # (minx, miny, maxx, maxy)
                    if _bbox and _bbox[2] > 1000:
                        src_epsg = EPSG_BNG
            except Exception:
                if cfg.get("crs_bng"):
                    src_epsg = EPSG_BNG

            read_bbox = box(*ENGLAND_BBOX_BNG) if src_epsg == EPSG_BNG else box(*ENGLAND_BBOX_WGS84)
            load_kwargs: dict = {"bbox": read_bbox}
            if layer:
                load_kwargs["layer"] = layer
            try:
                gdf = gpd.read_file(src_path, engine="pyogrio", **load_kwargs)
            except Exception:
                gdf = gpd.read_file(src_path, **load_kwargs)

            gdf = force_bng(gdf)

            # Keep only fields that exist in this dataset
            existing_fields = [f for f in fields if f in gdf.columns]
            missing_fields  = [f for f in fields if f not in gdf.columns]
            if missing_fields:
                log(f"    ⚠ Missing fields (will be null): {missing_fields}", indent=2)

            gdf = gdf[["geometry"] + existing_fields].copy()
            log(f"    {len(gdf):,} features → rasterising...", indent=2)
            _LOG.mem(f"after load {key}")

            id_raster, lookup = _rasterise_vector(gdf, existing_fields, meta)

            np.savez_compressed(npz_out, data=id_raster)
            with open(lut_out, "w") as f:
                json.dump(lookup, f, separators=(",", ":"), default=str)

            covered = int((id_raster > 0).sum())
            log(f"    ✓ {covered:,} cells covered  ({100*covered/(meta.n_rows*meta.n_cols):.1f}% of grid)",
                indent=2, elapsed=time.time()-t0)

        elif cfg["type"] == "raster_float":
            log(f"    Resampling {src_path.name} → {cell_size}m grid...", indent=2)
            arr = _resample_raster_to_meta(src_path, meta)

            # EMODnet bathymetry uses negative-below-sea-level convention
            # (e.g. 10 m water depth → -10.0).  The grid schema and fill
            # pipeline both expect positive-downward (10 m depth → +10.0).
            # Negate depth here so the convention is consistent from storage
            # onward.  Slope is a magnitude — no sign change needed.
            if key == "emodnet_depth":
                arr = np.where(np.isfinite(arr), -arr, np.nan).astype(np.float32)
                log(f"    (negated: EMODnet → positive-downward)", indent=2)

            valid = int(np.isfinite(arr).sum())
            np.savez_compressed(npz_out, data=arr)
            log(f"    ✓ {valid:,} valid cells", indent=2, elapsed=time.time()-t0)

        result[key] = npz_out

    return result


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 4 — Tree index (tile NPZ files)
# ─────────────────────────────────────────────────────────────────────────────

def stage4_index(strip_mask: np.ndarray, dist_hwm: np.ndarray,
                 raster_paths: dict, meta: RasterMeta,
                 cache_dir: Path, cell_size: int,
                 sample_n: int = 0) -> Path:
    """
    Build the tile tree index.

    For each 100km tile, writes a .npz containing arrays for every cell in
    the strip that falls within that tile:
      easting, northing, dist_hwm, zone  (location)
      {dataset}_id                        (int32, 0=nodata — index into lookup)
      emodnet_depth, emodnet_slope        (float32)

    Also writes tile_registry.json — coverage summary per tile.
    Returns the index directory path.
    """
    idx_dir   = cache_dir / f"stage4_index_{cell_size}m"
    tile_dir  = idx_dir / "tiles"
    reg_out   = idx_dir / "tile_registry.json"

    if reg_out.exists() and reg_out.stat().st_size > 0:
        log(f"✓ Cached: stage4_index_{cell_size}m  — delete dir to rebuild", indent=1)
        return idx_dir

    section(f"Stage 4: Building tree index ({cell_size}m)", 4, 5)
    t0 = time.time()

    tile_dir.mkdir(parents=True, exist_ok=True)

    # ── Load all stage3 rasters into RAM ──────────────────────────────────────
    log("Loading rasters into RAM...", indent=1)
    loaded: dict[str, np.ndarray | None] = {}
    lookups: dict[str, dict]             = {}

    s3_dir = cache_dir / "stage3_rasters"

    for key, npz_path in raster_paths.items():
        if npz_path is None or not Path(npz_path).exists():
            loaded[key] = None
            continue
        npz_path = Path(npz_path)
        arr = np.load(npz_path)["data"]
        loaded[key] = arr

        lut_path = s3_dir / f"{key}_lookup.json"
        if lut_path.exists():
            with open(lut_path) as f:
                # JSON keys are always strings; convert to int
                raw = json.load(f)
                lookups[key] = {int(k): v for k, v in raw.items()}
        log(f"  {key}: {arr.shape}  dtype={arr.dtype}", indent=2)

    _LOG.mem("all rasters loaded")

    # ── Extract all in-strip cell indices ─────────────────────────────────────
    log("Extracting strip cells...", indent=1)
    row_idx, col_idx = np.where(strip_mask == 1)
    n_cells = len(row_idx)
    log(f"  {n_cells:,} cells in strip", indent=2)

    if sample_n > 0 and sample_n < n_cells:
        rng  = np.random.default_rng(42)
        keep = rng.choice(n_cells, sample_n, replace=False)
        keep.sort()
        row_idx = row_idx[keep]
        col_idx = col_idx[keep]
        n_cells = sample_n
        log(f"  ⚠ Sample mode: {n_cells:,} cells", indent=2)

    # ── Compute coordinates ───────────────────────────────────────────────────
    eastings  = meta.col_to_e(col_idx.astype(np.float64)).astype(np.int32)
    northings = meta.row_to_n(row_idx.astype(np.float64)).astype(np.int32)
    distances = dist_hwm[row_idx, col_idx].astype(np.float32)

    zones = np.where(
        distances <= 0,   "intertidal",
        np.where(distances <= NEARSHORE_M, "nearshore", "offshore")
    )

    # ── WGS84 ─────────────────────────────────────────────────────────────────
    transformer = Transformer.from_crs(EPSG_BNG, 4326, always_xy=True)
    lons, lats  = transformer.transform(eastings.astype(float), northings.astype(float))
    lats = np.round(lats, 6).astype(np.float32)
    lons = np.round(lons, 6).astype(np.float32)

    # ── Lookup dataset value for every cell ───────────────────────────────────
    log("Extracting dataset values per cell...", indent=1)
    cell_data: dict[str, np.ndarray] = {}
    for key, arr in loaded.items():
        if arr is None:
            cell_data[key] = None
            continue
        cell_data[key] = arr[row_idx, col_idx]
        n_covered = int((cell_data[key] != 0).sum()) if arr.dtype != np.float32 \
                    else int(np.isfinite(cell_data[key]).sum())
        log(f"  {key}: {n_covered:,} / {n_cells:,} cells covered "
            f"({100*n_covered//max(1,n_cells)}%)", indent=2)

    # ── Tile assignment ───────────────────────────────────────────────────────
    log(f"Assigning cells to {TILE_SIZE_M//1000}km tiles...", indent=1)
    tile_e = (eastings  // TILE_SIZE_M).astype(np.int16)
    tile_n = (northings // TILE_SIZE_M).astype(np.int16)
    tile_keys = np.array([f"E{e:04d}_N{n:04d}" for e, n in zip(tile_e, tile_n)])
    unique_tiles = np.unique(tile_keys)
    log(f"  {len(unique_tiles)} tiles", indent=2)

    # ── Write tile NPZ files ───────────────────────────────────────────────────
    log("Writing tile files...", indent=1)
    registry = {}

    for tk in tqdm(unique_tiles, desc="  Tiles", unit="tile"):
        mask = tile_keys == tk
        n_t  = int(mask.sum())

        tile_payload = {
            "easting":      eastings[mask],
            "northing":     northings[mask],
            "lat":          lats[mask],
            "lon":          lons[mask],
            "dist_hwm_m":   distances[mask].astype(np.int32),
            "zone":         zones[mask],
        }

        datasets_present = []
        for key, vals in cell_data.items():
            if vals is None:
                continue
            arr_slice = vals[mask]
            tile_payload[key] = arr_slice
            if arr_slice.dtype == np.float32:
                has_data = bool(np.isfinite(arr_slice).any())
            else:
                has_data = bool((arr_slice != 0).any())
            if has_data:
                datasets_present.append(key)

        npz_path = tile_dir / f"tile_{tk}.npz"
        np.savez_compressed(npz_path, **tile_payload)

        registry[tk] = {
            "path":             str(npz_path.relative_to(cache_dir)),
            "n_cells":          n_t,
            "datasets_present": datasets_present,
        }

    reg_out.write_text(json.dumps(registry, indent=2))

    elapsed = time.time() - t0
    log(f"✓ Index built: {n_cells:,} cells in {len(unique_tiles)} tiles",
        indent=1, elapsed=elapsed)

    # Save lookups into the index dir for query-time use
    for key, lut in lookups.items():
        out = idx_dir / f"{key}_lookup.json"
        with open(out, "w") as f:
            json.dump({str(k): v for k, v in lut.items()}, f,
                      separators=(",", ":"), default=str)

    return idx_dir


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 5 — Export to SQLite + Parquet  (vectorised)
# ─────────────────────────────────────────────────────────────────────────────

def _lut_to_df(lut: dict, key: str, fields: list[str]) -> pd.DataFrame:
    """
    Expand a lookup dict  {id: {field: value, ...}}  into a DataFrame with
    columns [f"{key}_id", field1, field2, ...].  Rows with id==0 are skipped.
    """
    rows = []
    for id_int, rec in lut.items():
        if id_int == 0:
            continue
        row = {f"{key}_id": id_int}
        for f in fields:
            row[f] = rec.get(f)
        rows.append(row)
    if not rows:
        schema = {f"{key}_id": pd.Series(dtype="int32")}
        for f in fields:
            schema[f] = pd.Series(dtype="object")
        return pd.DataFrame(schema)
    return pd.DataFrame(rows)


def _normalise_vectorised(df: pd.DataFrame, lookups: dict) -> pd.DataFrame:
    """
    Fully vectorised normalisation — no Python loop per cell.
    Works on the whole tile-concatenated DataFrame at once.
    """
    # ── Join lookup tables ────────────────────────────────────────────────────
    def join_lut(key: str, fields: list[str], prefix: str | None = None):
        if key not in lookups or lookups[key] is None:
            for f in fields:
                col = f"{prefix}_{f}" if prefix else f
                df[col] = None
            return
        ldf = _lut_to_df(lookups[key], key, fields)
        id_col = f"{key}_id"
        if id_col not in df.columns:
            for f in fields:
                col = f"{prefix}_{f}" if prefix else f
                df[col] = None
            return
        merged = df[[id_col]].merge(ldf, on=id_col, how="left")
        for f in fields:
            col = f"{prefix}_{f}" if prefix else f
            df[col] = merged[f].values

    join_lut("bgs_sbs_obs",  ["RCS", "RCS_D"],                       prefix="obs")
    join_lut("bgs_sbs_pred", ["RCS", "RCS_D",
                               "pct_gravel", "pct_sand", "pct_mud"], prefix="pred")
    join_lut("defr",         ["FORE_DESC", "BACK_DESC"],              prefix="defr")
    join_lut("bgs_bedrock",  ["LEX_RCS", "LEX_RCS_D"],               prefix="brk")
    join_lut("ukash",        ["EUNISCode", "EUNISDesc", "MHCCode",
                               "MESH_conf", "SNCB_UID", "EUNISTranR",
                               "EUNISL3"],                            prefix="uk")

    # ── Substrate priority: BGS_obs > BGS_pred > DEFR ────────────────────────
    obs_rcs  = df["obs_RCS"].where(df["obs_RCS"].notna() & (df["obs_RCS"] != ""), None)
    pred_rcs = df["pred_RCS"].where(df["pred_RCS"].notna() & (df["pred_RCS"] != ""), None)
    defr_sub = df["defr_FORE_DESC"].fillna(df["defr_BACK_DESC"])
    defr_sub = defr_sub.where(defr_sub.notna() & (defr_sub != ""), None)

    folk_code = obs_rcs.fillna(pred_rcs).fillna(defr_sub)
    folk_desc = df["obs_RCS_D"].where(obs_rcs.notna()).fillna(
                df["pred_RCS_D"].where(pred_rcs.notna()))

    sub_source = pd.Series("none", index=df.index)
    sub_source = sub_source.where(defr_sub.isna(),  "DEFR")
    sub_source = sub_source.where(pred_rcs.isna(),  "BGS_predictive")
    sub_source = sub_source.where(obs_rcs.isna(),   "BGS_observed")

    fu = folk_code.str.upper()
    substrate_primary = fu.map(FOLK_TO_PRIMARY).fillna("unknown")
    hardness          = fu.map(FOLK_TO_HARDNESS).fillna("unknown")

    # Pct composition — prefer pred, fall back to obs
    pct_gravel = pd.to_numeric(df["pred_pct_gravel"], errors="coerce")
    pct_sand   = pd.to_numeric(df["pred_pct_sand"],   errors="coerce")
    pct_mud    = pd.to_numeric(df["pred_pct_mud"],    errors="coerce")

    # Harden hardness from pct where unknown
    unk = hardness == "unknown"
    pg  = pct_gravel.fillna(0.0)
    pm  = pct_mud.fillna(0.0)
    hardness = hardness.copy()
    hardness[unk & (pg > 60)]             = "hard"
    hardness[unk & (pg <= 60) & (pm > 50)]= "soft"
    hardness[unk & (pg > 0) | (pm > 0) & unk] = np.where(
        (pg[unk & ((pg > 0) | (pm > 0))] > 0) | (pm[unk & ((pg > 0) | (pm > 0))] > 0),
        "mixed", hardness[unk & ((pg > 0) | (pm > 0))]
    ) if (unk & ((pg > 0) | (pm > 0))).any() else hardness[unk & ((pg > 0) | (pm > 0))]

    sub_conf = sub_source.map(SOURCE_CONFIDENCE).fillna(0.0)

    # ── Bedrock ───────────────────────────────────────────────────────────────
    bedrock_lex  = df["brk_LEX_RCS"].where(df["brk_LEX_RCS"].notna() & (df["brk_LEX_RCS"] != ""), None)
    bedrock_desc = df["brk_LEX_RCS_D"].where(bedrock_lex.notna(), None)
    bedrock_exp  = bedrock_lex.str.upper().str.contains("ROCK", na=False)

    # Bedrock overrides hardness and substrate_primary
    hardness          = hardness.copy()
    substrate_primary = substrate_primary.copy()
    hardness[bedrock_exp]                                       = "hard"
    substrate_primary[bedrock_exp & (substrate_primary == "unknown")] = "rock"

    # ── Habitat ───────────────────────────────────────────────────────────────
    eunis_code = df["uk_EUNISCode"].where(df["uk_EUNISCode"].notna() & (df["uk_EUNISCode"] != ""), None)
    eunis_name = df["uk_EUNISDesc"].where(eunis_code.notna(), None)
    mhc_code   = df["uk_MHCCode"].where(df["uk_MHCCode"].notna() & (df["uk_MHCCode"] != ""), None)
    sncb_uid   = df["uk_SNCB_UID"].where(df["uk_SNCB_UID"].notna() & (df["uk_SNCB_UID"] != ""), None)

    hab_source = pd.Series("none", index=df.index)
    hab_source[eunis_code.notna() &  sncb_uid.notna()] = "UKASH_survey"
    hab_source[eunis_code.notna() &  sncb_uid.isna()]  = "UKASH_predictive"

    mesh_raw = pd.to_numeric(df["uk_MESH_conf"], errors="coerce")
    hab_conf_from_mesh = (mesh_raw / 100.0).clip(0.0, 1.0)
    hab_conf_from_src  = hab_source.map(SOURCE_CONFIDENCE).fillna(0.0)
    hab_conf = hab_conf_from_mesh.fillna(hab_conf_from_src)

    # ── Foreshore ─────────────────────────────────────────────────────────────
    foreshore_type = df["defr_FORE_DESC"].fillna(df["defr_BACK_DESC"])
    foreshore_type = foreshore_type.where(foreshore_type.notna() & (foreshore_type != ""), None)

    # ── Bathymetry ────────────────────────────────────────────────────────────
    depth_col = "emodnet_depth" if "emodnet_depth" in df.columns else None
    slope_col = "emodnet_slope" if "emodnet_slope" in df.columns else None
    depth_m   = pd.to_numeric(df[depth_col], errors="coerce") if depth_col else pd.Series(np.nan, index=df.index)
    slope_deg = pd.to_numeric(df[slope_col], errors="coerce") if slope_col else pd.Series(np.nan, index=df.index)
    morphology = slope_deg.map(slope_to_morphology)

    # ── Confidence ────────────────────────────────────────────────────────────
    overall_conf = pd.concat([sub_conf, hab_conf], axis=1).where(
        pd.concat([sub_conf, hab_conf], axis=1) > 0
    ).mean(axis=1).fillna(0.0).round(3)

    # ── Coverage flags ────────────────────────────────────────────────────────
    def make_flags(row):
        flags = []
        if row["substrate_primary"] == "unknown": flags.append("substrate")
        if pd.isna(row["eunis_code"]):            flags.append("habitat")
        if pd.isna(row["depth_m"]):               flags.append("bathymetry")
        if pd.isna(row["bedrock_lex_rcs"]):       flags.append("bedrock")
        return ",".join(flags) if flags else None

    # Build output DataFrame directly
    out = pd.DataFrame({
        "cell_id":              df["cell_id"].astype("int32"),
        "easting_bng":          df["easting"].astype("int32"),
        "northing_bng":         df["northing"].astype("int32"),
        "lat":                  df["lat"].astype("float32"),
        "lon":                  df["lon"].astype("float32"),
        "zone":                 df["zone"].astype(str),
        "dist_to_hwm_m":        df["dist_hwm_m"].astype("int32"),
        "depth_m":              depth_m,
        "slope_deg":            slope_deg,
        "morphology":           morphology,
        "substrate_primary":    substrate_primary,
        "folk_code":            folk_code,
        "folk_description":     folk_desc,
        "pct_gravel":           pct_gravel,
        "pct_sand":             pct_sand,
        "pct_mud":              pct_mud,
        "hardness":             hardness,
        "substrate_source":     sub_source,
        "substrate_confidence": sub_conf.round(3),
        "eunis_code":           eunis_code,
        "eunis_name":           eunis_name,
        "mhc_code":             mhc_code,
        "habitat_source":       hab_source,
        "habitat_confidence":   hab_conf.round(3),
        "foreshore_type":       foreshore_type,
        "bedrock_lex_rcs":      bedrock_lex,
        "bedrock_description":  bedrock_desc,
        "bedrock_exposed":      bedrock_exp,
        "has_observed_survey":  (sub_source == "BGS_observed"),
        "overall_confidence":   overall_conf,
    })

    # Coverage flags — still per-row but only string concatenation
    out["coverage_flags"] = out.apply(make_flags, axis=1)

    return out


def stage5_export(idx_dir: Path, out_dir: Path,
                  cell_size: int, cache_dir: Path) -> pd.DataFrame:
    """
    Read all tile NPZ files, normalise each cell, and write the final
    SQLite + Parquet outputs.  Fully vectorised — no Python loop per cell.
    """
    section("Stage 5: Normalising + exporting", 5, 5)
    t0 = time.time()

    stem    = f"spearo_coastal_grid_{cell_size}m"
    db_path = out_dir / f"{stem}.db"
    pq_path = out_dir / f"{stem}.parquet"

    # Load lookup tables from the index dir
    log("Loading lookup tables...", indent=1)
    lookups: dict[str, dict] = {}
    for lut_path in sorted(idx_dir.glob("*_lookup.json")):
        key = lut_path.stem.replace("_lookup", "")
        with open(lut_path) as f:
            raw = json.load(f)
            lookups[key] = {int(k): v for k, v in raw.items()}
    log(f"  {len(lookups)} lookup tables loaded", indent=2)

    # Load registry
    reg_path = idx_dir / "tile_registry.json"
    registry = json.loads(reg_path.read_text())
    log(f"  {len(registry)} tiles to process", indent=2)

    # ── Load all tiles into a single DataFrame ────────────────────────────────
    log("Loading tiles into DataFrame...", indent=1)
    tile_dfs = []
    cell_id_counter = 0

    for tk, tile_info in tqdm(registry.items(), desc="  Loading tiles", unit="tile"):
        npz_path = cache_dir / tile_info["path"]
        if not npz_path.exists():
            log(f"  ⚠ Missing tile file: {npz_path}", indent=2)
            continue

        tile = np.load(npz_path, allow_pickle=True)
        n_t  = len(tile["easting"])

        tdf = pd.DataFrame({
            "cell_id":    np.arange(cell_id_counter, cell_id_counter + n_t, dtype=np.int32),
            "easting":    tile["easting"],
            "northing":   tile["northing"],
            "lat":        tile["lat"],
            "lon":        tile["lon"],
            "dist_hwm_m": tile["dist_hwm_m"],
            "zone":       tile["zone"],
        })

        for key in DATASETS:
            if key in tile:
                tdf[key] = tile[key]
                # For int raster IDs, rename to _id column for join
                if DATASETS[key]["type"] == "vector_polygon":
                    tdf[f"{key}_id"] = tile[key].astype(np.int32)

        tile_dfs.append(tdf)
        cell_id_counter += n_t

    df = pd.concat(tile_dfs, ignore_index=True)
    log(f"  {len(df):,} cells loaded", indent=2, elapsed=time.time()-t0)

    # ── Vectorised normalisation ──────────────────────────────────────────────
    log("Normalising (vectorised)...", indent=1)
    t1 = time.time()
    df = _normalise_vectorised(df, lookups)
    log(f"  {len(df):,} cells normalised", indent=2, elapsed=time.time()-t1)

    # ── SQLite ────────────────────────────────────────────────────────────────
    log("Writing SQLite...", indent=1)
    t1 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)

    if db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(db_path)
    df.to_sql("coastal_grid", conn, if_exists="replace", index=False)

    cur = conn.cursor()
    for sql in [
        "CREATE INDEX IF NOT EXISTS idx_cell_id   ON coastal_grid(cell_id)",
        "CREATE INDEX IF NOT EXISTS idx_bng       ON coastal_grid(easting_bng, northing_bng)",
        "CREATE INDEX IF NOT EXISTS idx_latlon    ON coastal_grid(lat, lon)",
        "CREATE INDEX IF NOT EXISTS idx_zone      ON coastal_grid(zone)",
        "CREATE INDEX IF NOT EXISTS idx_substrate ON coastal_grid(substrate_primary)",
        "CREATE INDEX IF NOT EXISTS idx_eunis     ON coastal_grid(eunis_code)",
        "CREATE INDEX IF NOT EXISTS idx_hardness  ON coastal_grid(hardness)",
    ]:
        cur.execute(sql)

    # Coverage view
    cur.execute(f"""
        CREATE VIEW IF NOT EXISTS coverage AS
        SELECT
            zone,
            COUNT(*)                                                    AS total_cells,
            SUM(CASE WHEN substrate_primary != 'unknown' THEN 1 END)   AS substrate_cells,
            SUM(CASE WHEN eunis_code IS NOT NULL THEN 1 END)            AS habitat_cells,
            SUM(CASE WHEN depth_m IS NOT NULL THEN 1 END)               AS depth_cells,
            SUM(CASE WHEN bedrock_lex_rcs IS NOT NULL THEN 1 END)       AS bedrock_cells,
            SUM(CASE WHEN substrate_source = 'BGS_observed' THEN 1 END) AS observed_cells,
            ROUND(AVG(overall_confidence), 3)                           AS mean_confidence
        FROM coastal_grid
        GROUP BY zone
    """)

    cur.execute("""
        CREATE VIEW IF NOT EXISTS algo_inputs AS
        SELECT cell_id, easting_bng, northing_bng, lat, lon,
               zone, dist_to_hwm_m,
               depth_m, slope_deg, morphology,
               substrate_primary, pct_gravel, pct_sand, pct_mud, hardness,
               eunis_code, mhc_code,
               bedrock_exposed,
               has_observed_survey, overall_confidence
        FROM coastal_grid
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS build_stats AS
        SELECT
            COUNT(*)                                                AS total_cells,
            SUM(CASE WHEN zone='intertidal'  THEN 1 END)           AS intertidal_cells,
            SUM(CASE WHEN zone='nearshore'   THEN 1 END)           AS nearshore_cells,
            SUM(CASE WHEN zone='offshore'    THEN 1 END)           AS coastal_cells,
            SUM(CASE WHEN substrate_primary!='unknown' THEN 1 END) AS cells_with_substrate,
            SUM(CASE WHEN eunis_code IS NOT NULL THEN 1 END)        AS cells_with_habitat,
            SUM(CASE WHEN depth_m IS NOT NULL THEN 1 END)           AS cells_with_depth,
            SUM(CASE WHEN has_observed_survey=1 THEN 1 END)         AS cells_with_observed,
            ROUND(AVG(overall_confidence), 3)                       AS mean_confidence
        FROM coastal_grid
    """)

    conn.commit()
    conn.close()
    log(f"  ✓ {db_path.name}  ({db_path.stat().st_size/1_048_576:.1f} MB)",
        indent=2, elapsed=time.time()-t1)

    # ── Parquet ───────────────────────────────────────────────────────────────
    log("Writing Parquet...", indent=1)
    t1 = time.time()
    df.to_parquet(pq_path, index=False)
    log(f"  ✓ {pq_path.name}  ({pq_path.stat().st_size/1_048_576:.1f} MB)",
        indent=2, elapsed=time.time()-t1)

    return df


# ── Manifest ──────────────────────────────────────────────────────────────────

def write_manifest(out_dir: Path, df: pd.DataFrame, args, cell_size: int):
    stats = {
        "total_cells":          int(len(df)),
        "intertidal_cells":     int((df["zone"] == "intertidal").sum()),
        "nearshore_cells":      int((df["zone"] == "nearshore").sum()),
        "coastal_cells":        int((df["zone"] == "offshore").sum()),
        "cells_with_substrate": int((df["substrate_primary"] != "unknown").sum()),
        "cells_with_habitat":   int(df["eunis_code"].notna().sum()),
        "cells_with_depth":     int(df["depth_m"].notna().sum()),
        "cells_with_observed":  int(df["has_observed_survey"].sum()),
        "mean_confidence":      round(float(df["overall_confidence"].mean()), 3),
        "substrate_sources":    df["substrate_source"].value_counts().to_dict(),
        "habitat_sources":      df["habitat_source"].value_counts().to_dict(),
        "morphology_dist":      df["morphology"].value_counts().to_dict(),
        "coverage_by_zone":     (
            df.groupby("zone").apply(lambda g: {
                "total":     int(len(g)),
                "substrate": int((g["substrate_primary"] != "unknown").sum()),
                "habitat":   int(g["eunis_code"].notna().sum()),
                "depth":     int(g["depth_m"].notna().sum()),
            }, include_groups=False).to_dict()
        ),
    }
    manifest = {
        "built_at":      datetime.now(timezone.utc).isoformat(),
        "cell_size_m":   cell_size,
        "strip_width_m": ONE_NM_M,
        "root_dir":      str(Path(args.root).resolve()),
        "sample_n":      args.sample,
        "stats":         stats,
    }
    out = out_dir / f"build_manifest_{cell_size}m.json"
    with open(out, "w") as f:
        json.dump(manifest, f, indent=2)
    return stats


# ── Force / cache clearing ────────────────────────────────────────────────────

def _apply_force(force: str, cache_dir: Path, out_dir: Path, cell_size: int):
    """
    Delete stage outputs to force rebuild.
    Supports:
      stage1  stage2  stage3  stage3:ukash  stage4  all
    """
    targets = {
        "stage1": [cache_dir / "stage1_coastline"],
        "stage2": [cache_dir / "stage2_strip"],
        "stage4": [cache_dir / f"stage4_index_{cell_size}m"],
        "all":    [cache_dir, out_dir],
    }

    if force.startswith("stage3"):
        if ":" in force:
            ds_key = force.split(":")[1]
            s3_dir = cache_dir / "stage3_rasters"
            targets[force] = [
                s3_dir / f"{ds_key}_{cell_size}m.npz",
                s3_dir / f"{ds_key}_lookup.json",
            ]
            # Also invalidate stage4 since a dataset changed
            idx_dir = cache_dir / f"stage4_index_{cell_size}m"
            if idx_dir.exists():
                shutil.rmtree(idx_dir)
                log(f"  Also cleared stage4 index (dataset changed)")
        else:
            targets["stage3"] = [cache_dir / "stage3_rasters"]

    for t in targets.get(force, []):
        t = Path(t)
        if t.exists():
            if t.is_dir():
                shutil.rmtree(t)
            else:
                t.unlink()
            log(f"  Cleared: {t}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Spearo coastal substrate grid builder (v3 — seaward-only strip).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--root",       default="raw",    help="Raw datasets directory")
    p.add_argument("--cache-dir",  default="cache",  help="Cache directory")
    p.add_argument("--output-dir", default="output", help="Output directory")
    p.add_argument("--cell-size",  type=int, default=100,
                   help="Grid cell size in metres (default: 100)")
    p.add_argument("--sample",     type=int, default=0,
                   help="Limit to N random strip cells (test mode)")
    p.add_argument("--force",      default=None,
                   help="Force rebuild: stage1 | stage2 | stage3 | stage3:DATASET | stage4 | all")
    return p.parse_args()


def main():
    args      = parse_args()
    root      = Path(args.root)
    cache_dir = Path(args.cache_dir)
    out_dir   = Path(args.output_dir)
    cs        = args.cell_size

    # ── Initialise logger ─────────────────────────────────────────────────────
    log_dir = Path("log")
    global _LOG
    _LOG = Logger(log_dir=log_dir)

    header = [
        "=" * 60,
        "Spearo Coastal Grid Builder  (v3 — seaward-only strip)",
        "=" * 60,
        f"  Root       : {root.resolve()}",
        f"  Cache      : {cache_dir.resolve()}",
        f"  Output     : {out_dir.resolve()}",
        f"  Log        : {_LOG.log_path}",
        f"  Cell size  : {cs}m",
        f"  Strip      : HWM -> {STRIP_M}m (seaward only)",
        f"  Tile size  : {TILE_SIZE_M//1000}km × {TILE_SIZE_M//1000}km",
        f"  Force      : {args.force or 'none (use cache)'}",
    ]
    if args.sample:
        header.append(f"  Sample     : {args.sample:,} cells  ⚠ TEST MODE")
    header.append("")
    for line in header:
        log(line)

    try:
        if args.force:
            _apply_force(args.force, cache_dir, out_dir, cs)

        t_total = time.time()

        coast_gdf, coast_arr, meta = stage1_coastline(root, cache_dir, cs)
        strip_mask, dist_hwm       = stage2_strip(coast_gdf, coast_arr, meta, cache_dir, cs)

        force_ds = None
        if args.force and args.force.startswith("stage3:"):
            force_ds = args.force.split(":")[1]

        raster_paths = stage3_rasters(root, cache_dir, meta, cs, force_dataset=force_ds)
        idx_dir      = stage4_index(strip_mask, dist_hwm, raster_paths, meta,
                                     cache_dir, cs, sample_n=args.sample)
        df           = stage5_export(idx_dir, out_dir, cs, cache_dir)
        stats        = write_manifest(out_dir, df, args, cs)

        elapsed = time.time() - t_total
        pct = lambda k: f"({100*stats[k]//max(1,stats['total_cells'])}%)"

        summary = [
            "",
            "=" * 60,
            "BUILD COMPLETE",
            "=" * 60,
            f"  Total time         : {elapsed:.0f}s  ({elapsed/60:.1f} min)",
            f"  Total cells        : {stats['total_cells']:,}",
            f"    intertidal        : {stats['intertidal_cells']:,}",
            f"    nearshore         : {stats['nearshore_cells']:,}",
            f"    offshore          : {stats['coastal_cells']:,}",
            f"  With substrate      : {stats['cells_with_substrate']:,}  {pct('cells_with_substrate')}",
            f"  With habitat        : {stats['cells_with_habitat']:,}  {pct('cells_with_habitat')}",
            f"  With depth          : {stats['cells_with_depth']:,}  {pct('cells_with_depth')}",
            f"  With observed data  : {stats['cells_with_observed']:,}",
            f"  Mean confidence     : {stats['mean_confidence']:.2f}",
            "",
            "  Outputs:",
        ]
        for f in sorted(out_dir.glob(f"spearo_coastal_grid_{cs}m*")):
            summary.append(f"    {f.name}  ({f.stat().st_size/1_048_576:.1f} MB)")
        summary += [
            "",
            "  Coverage quick-check (SQL):",
            f"    sqlite3 output/spearo_coastal_grid_{cs}m.db 'SELECT * FROM coverage'",
            "",
            "  Query example:",
            f"    python query_grid.py --lat 50.614 --lon -1.195 --db output/spearo_coastal_grid_{cs}m.db",
            "",
            f"  Log file: {_LOG.log_path}",
        ]
        for line in summary:
            log(line)

    except Exception as exc:
        log(f"\n✗ BUILD FAILED: {exc}")
        import traceback
        tb = traceback.format_exc()
        log(tb)
        raise
    finally:
        _LOG.close()


if __name__ == "__main__":
    main()