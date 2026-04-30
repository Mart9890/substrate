#!/usr/bin/env python3
"""
Dataset Inspection Script
=========================
Inspects all spatial datasets in a root directory and produces a structured
schema report for each layer/table found.

Supported formats:
  - GeoPackage (.gpkg)
  - ESRI FileGDB (.gdb)
  - Shapefile (.shp)
  - GeoJSON (.geojson / .json)

Usage:
  python inspect_datasets.py <root_directory> [--output report.json] [--md report.md]

Dependencies:
  pip install geopandas fiona pyogrio shapely tabulate
"""

import os
import sys
import json
import argparse
import warnings
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")

try:
    import fiona
    import geopandas as gpd
    import numpy as np
    from shapely.geometry import box, mapping
    from tabulate import tabulate
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Run: pip install geopandas fiona pyogrio shapely tabulate")
    sys.exit(1)


# ── Config ──────────────────────────────────────────────────────────────────

SUPPORTED_EXTENSIONS = {".gpkg", ".gdb", ".shp", ".geojson", ".json"}
SAMPLE_VALUES_N = 5          # number of unique sample values to show per field
MAX_FEATURES_FOR_SAMPLE = 10_000  # subsample if larger (for speed)
RASTER_LAYER_PREFIXES = (     # skip raster tables inside gpkgs
    "gpkg_tile_matrix",
    "gpkg_2d_gridded",
    "rtree_",
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def find_datasets(root: Path) -> list[dict]:
    """Walk root and find all spatial dataset paths."""
    datasets = []
    seen_gdbs = set()

    for path in sorted(root.rglob("*")):
        if path.suffix.lower() == ".gdb" and path.is_dir():
            if path not in seen_gdbs:
                seen_gdbs.add(path)
                datasets.append({"path": path, "format": "FileGDB"})
        elif path.suffix.lower() in SUPPORTED_EXTENSIONS and path.is_file():
            fmt_map = {
                ".gpkg": "GeoPackage",
                ".shp": "Shapefile",
                ".geojson": "GeoJSON",
                ".json": "GeoJSON",
            }
            datasets.append({"path": path, "format": fmt_map.get(path.suffix.lower(), "Unknown")})

    return datasets


def safe_crs_string(crs) -> str:
    if crs is None:
        return "Unknown"
    try:
        return crs.to_epsg() and f"EPSG:{crs.to_epsg()}" or crs.to_string()
    except Exception:
        return str(crs)


def bbox_to_dict(bounds) -> dict:
    """Convert (minx, miny, maxx, maxy) to a readable dict."""
    if bounds is None:
        return {}
    return {
        "minx": round(bounds[0], 6),
        "miny": round(bounds[1], 6),
        "maxx": round(bounds[2], 6),
        "maxy": round(bounds[3], 6),
    }


def describe_field(series) -> dict:
    """Return dtype, null count and sample unique values for a pandas Series."""
    dtype = str(series.dtype)
    null_count = int(series.isna().sum())
    total = len(series)
    non_null = series.dropna()

    samples = []
    try:
        unique_vals = non_null.unique()
        n = min(SAMPLE_VALUES_N, len(unique_vals))
        samples = [str(v) for v in unique_vals[:n]]
    except Exception:
        samples = []

    return {
        "dtype": dtype,
        "null_count": null_count,
        "null_pct": round(100 * null_count / total, 1) if total > 0 else 0,
        "n_unique": int(non_null.nunique()) if len(non_null) > 0 else 0,
        "sample_values": samples,
    }


def inspect_layer(dataset_path: Path, layer_name: str, fmt: str) -> dict:
    """Load a single layer and return its schema report."""
    result = {
        "layer": layer_name,
        "status": "ok",
        "error": None,
        "format": fmt,
        "geometry_type": None,
        "crs": None,
        "feature_count": None,
        "bbox": {},
        "fields": {},
        "notes": [],
    }

    try:
        # Use pyogrio engine for speed where available
        read_kwargs = dict(layer=layer_name, engine="pyogrio")
        if fmt == "FileGDB":
            read_kwargs = dict(layer=layer_name)

        gdf = gpd.read_file(dataset_path, **read_kwargs)

        result["feature_count"] = len(gdf)
        result["crs"] = safe_crs_string(gdf.crs)

        if gdf.geometry is not None and not gdf.geometry.empty:
            geom_types = gdf.geometry.geom_type.dropna().unique().tolist()
            result["geometry_type"] = geom_types
            try:
                bounds = gdf.total_bounds  # (minx, miny, maxx, maxy)
                result["bbox"] = bbox_to_dict(bounds)
            except Exception:
                pass
        else:
            result["notes"].append("No geometry column (attribute table only)")

        # Subsample for field analysis on large layers
        if len(gdf) > MAX_FEATURES_FOR_SAMPLE:
            sample_gdf = gdf.sample(MAX_FEATURES_FOR_SAMPLE, random_state=42)
            result["notes"].append(
                f"Field stats sampled from {MAX_FEATURES_FOR_SAMPLE:,} of {len(gdf):,} features"
            )
        else:
            sample_gdf = gdf

        # Describe each non-geometry column
        for col in sample_gdf.columns:
            if col == sample_gdf.geometry.name if sample_gdf.geometry is not None else False:
                continue
            if col.lower() in ("geometry", "geom", "shape", "shape_length", "shape_area"):
                continue
            result["fields"][col] = describe_field(sample_gdf[col])

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)

    return result


def inspect_dataset(ds: dict) -> dict:
    """Inspect all layers in a dataset."""
    path = ds["path"]
    fmt = ds["format"]

    report = {
        "path": str(path),
        "format": fmt,
        "file_size_mb": None,
        "inspected_at": datetime.utcnow().isoformat(),
        "layers": [],
        "error": None,
    }

    # File size (skip for .gdb dirs)
    try:
        if path.is_file():
            report["file_size_mb"] = round(path.stat().st_size / 1_048_576, 2)
        else:
            # sum .gdb directory
            total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
            report["file_size_mb"] = round(total / 1_048_576, 2)
    except Exception:
        pass

    # List layers
    try:
        layer_names = fiona.listlayers(str(path))
    except Exception as e:
        report["error"] = f"Could not list layers: {e}"
        return report

    # Filter out internal raster/metadata tables in gpkg
    if fmt == "GeoPackage":
        layer_names = [
            l for l in layer_names
            if not any(l.startswith(p) for p in RASTER_LAYER_PREFIXES)
        ]

    for layer_name in layer_names:
        print(f"    Layer: {layer_name} ...", flush=True)
        layer_report = inspect_layer(path, layer_name, fmt)
        report["layers"].append(layer_report)

    return report


# ── Reporting ────────────────────────────────────────────────────────────────

def report_to_markdown(all_reports: list[dict]) -> str:
    lines = [
        "# Dataset Inspection Report",
        f"*Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}*",
        "",
    ]

    for ds in all_reports:
        rel_path = ds["path"]
        lines += [
            f"---",
            f"## `{Path(rel_path).name}`",
            f"- **Format:** {ds['format']}",
            f"- **Path:** `{rel_path}`",
            f"- **File size:** {ds['file_size_mb']} MB" if ds['file_size_mb'] else "",
            "",
        ]

        if ds.get("error"):
            lines.append(f"> ⚠️ Error: {ds['error']}")
            lines.append("")
            continue

        for layer in ds["layers"]:
            lines += [
                f"### Layer: `{layer['layer']}`",
            ]

            if layer["status"] == "error":
                lines.append(f"> ⚠️ Error reading layer: {layer['error']}")
                lines.append("")
                continue

            meta_rows = [
                ["Geometry type", ", ".join(layer["geometry_type"]) if layer["geometry_type"] else "None"],
                ["CRS", layer["crs"]],
                ["Feature count", f"{layer['feature_count']:,}" if layer['feature_count'] is not None else "?"],
            ]
            if layer["bbox"]:
                b = layer["bbox"]
                meta_rows.append(["Bounding box", f"({b['minx']}, {b['miny']}) to ({b['maxx']}, {b['maxy']})"])

            lines.append(tabulate(meta_rows, tablefmt="github"))
            lines.append("")

            if layer["notes"]:
                for note in layer["notes"]:
                    lines.append(f"> 📝 {note}")
                lines.append("")

            if layer["fields"]:
                field_rows = []
                for fname, finfo in layer["fields"].items():
                    samples = ", ".join(finfo["sample_values"][:3])
                    field_rows.append([
                        fname,
                        finfo["dtype"],
                        finfo["n_unique"],
                        f"{finfo['null_pct']}%",
                        samples,
                    ])
                lines.append(
                    tabulate(
                        field_rows,
                        headers=["Field", "Type", "Unique values", "Null %", "Sample values"],
                        tablefmt="github",
                    )
                )
                lines.append("")

    return "\n".join(lines)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Inspect spatial datasets and report schema.")
    parser.add_argument("root", help="Root directory to search for datasets")
    parser.add_argument("--output", default="dataset_inspection.json", help="JSON output path")
    parser.add_argument("--md", default="dataset_inspection.md", help="Markdown report output path")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"Error: {root} does not exist.")
        sys.exit(1)

    print(f"Searching for datasets under: {root}")
    datasets = find_datasets(root)

    if not datasets:
        print("No supported spatial datasets found.")
        sys.exit(0)

    print(f"Found {len(datasets)} dataset(s):\n")
    for ds in datasets:
        print(f"  [{ds['format']}] {ds['path']}")

    print()
    all_reports = []
    for ds in datasets:
        print(f"\nInspecting: {ds['path']}")
        report = inspect_dataset(ds)
        all_reports.append(report)

    # Write JSON
    json_path = Path(args.output)
    with open(json_path, "w") as f:
        json.dump(all_reports, f, indent=2, default=str)
    print(f"\n✓ JSON report written to: {json_path}")

    # Write Markdown
    md_path = Path(args.md)
    md_content = report_to_markdown(all_reports)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"✓ Markdown report written to: {md_path}")

    # Quick terminal summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for ds_report in all_reports:
        name = Path(ds_report["path"]).name
        n_layers = len(ds_report["layers"])
        ok = sum(1 for l in ds_report["layers"] if l["status"] == "ok")
        err = n_layers - ok
        size = f"{ds_report['file_size_mb']} MB" if ds_report['file_size_mb'] else "?"
        print(f"  {name}: {n_layers} layer(s), {ok} ok, {err} errors — {size}")


if __name__ == "__main__":
    main()