# Spearo Coastal Substrate Grid

A pre-joined, 100 m resolution lookup table covering the England coastal strip from the High Water Mark (HWM) to 1 nautical mile offshore. The grid is intended as the primary environmental input for a visibility and species-distribution algorithm: given a coordinate, a single row lookup returns substrate type, seabed hardness, habitat classification, bathymetry, and bedrock geology for that cell — with no spatial queries at runtime.

The grid is built by combining five public-domain geospatial datasets into a single flat table, then gap-filling all remaining nulls to ensure every cell carries a usable value with an associated confidence score and provenance flag.

---

## Repository layout

```
.
├── build_coastal_grid.py          # Stage 1–5 grid builder
├── fill_coastal_grid.py           # Gap-fill pass (produces _filled.parquet)
├── query_grid.py                  # CLI point-lookup tool
├── check_hwl_completeness.py      # OS HWL topology audit (run before build)
├── verify_strip_placement.py      # Confirms strip is seaward-only (run after build)
├── spearo_coastal_grid_schema.md  # Full column reference for the raw grid
├── spearo_coastal_grid_100m_filled_README.md  # Column reference for the filled grid
├── inspect_datasets.py            # One-off dataset schema auditor
├── dataset_inspection.json        # Cached inspection output
├── dataset_inspection.md          # Human-readable inspection report
├── coverage_completeness.py       # Per-zone coverage statistics
├── spearo_coastal_grid_100m_completeness.json  # Coverage audit output
├── diagnose_grid.py               # Data quality / consistency auditor
└── tree.txt                       # Raw data directory listing
└── raw/                           # Source datasets (not tracked in git)
    ├── bgs-offshore-sbs-250k-geopackage/
    ├── offshore-bedrock-250k-geopackage/
    ├── Predictive_SBS_UK_V1_GeoPackage/
    ├── UKASH_CombinedMap_v2025/
    ├── Intertidal Substrate Foreshore (England and Scotland)/
    ├── emodnet_bathymetry/
    └── os_coastline/
```

Outputs land in `output/` (also not tracked):

```
output/
├── spearo_coastal_grid_100m.db            # SQLite — full schema + indexes + views
├── spearo_coastal_grid_100m.parquet       # GeoParquet flat table (raw)
├── spearo_coastal_grid_100m_filled.parquet  # Gap-filled version (primary product)
└── build_manifest_100m.json
```

---

## Source datasets

All source data sits under `raw/` and is **not committed to the repository**. Obtain each dataset from the sources listed below and place it at the indicated path before running the build.

| Key | Dataset | Publisher | Path under `raw/` |
|-----|---------|-----------|-------------------|
| `ukash` | UKASH Combined Map v2025.1 | JNCC | `UKASH_CombinedMap_v2025/UKASH_Combined_Map_2025.gdb` |
| `bgs_sbs_obs` | BGS Seabed Sediments 250k (observed) | British Geological Survey | `bgs-offshore-sbs-250k-geopackage/BGS_250k_SeabedSediments_WGS84_v3_FOLK.gpkg` |
| `bgs_sbs_pred` | BGS Predictive Seabed Sediments UK v1 | British Geological Survey | `Predictive_SBS_UK_V1_GeoPackage/BGS_Predictive_Seabed_Sediments_UK_v1.gpkg` |
| `bgs_bedrock` | BGS Offshore Bedrock 250k | British Geological Survey | `offshore-bedrock-250k-geopackage/BGS_BedrockOffshore_250k_WGS84_v3.gpkg` |
| `defr` | Intertidal Substrate Foreshore (England and Scotland) | DEFR / Natural England | `Intertidal Substrate Foreshore (England and Scotland)/DEFR00000009.shp` |
| `emodnet_depth` | EMODnet Bathymetry — depth | EMODnet | `emodnet_bathymetry/emodnet_depth_england_bng.tif` |
| `emodnet_slope` | EMODnet Bathymetry — slope | EMODnet | `emodnet_bathymetry/emodnet_slope_england_bng.tif` |

**UKASH** provides seabed habitat classification in EUNIS and Marine Habitat Classification (MHC) codes. It is a mosaic of ground-truthed survey maps (~12% of UK waters) infilled by the UKSeaMap predictive model.

**BGS observed sediments** contain Folk-classified substrate polygons derived from grab samples and side-scan sonar surveys conducted mainly in the 1970s–2000s at 1:250 000 scale.

**BGS predictive sediments** are a 2025 machine-learning model (Distributional Random Forest) trained on >38 000 legacy sediment samples. They supply the classified Folk map plus rasters of predicted %gravel, %sand and %mud at ~110 m resolution across the UKCS.

**BGS offshore bedrock** provides lithostratigraphic polygons with LEX/RCS codes, age and lithology at 1:250 000 scale. Used to infer bedrock exposure and override hardness classification.

**DEFR foreshore** supplies intertidal foreshore substrate polygons for England. It also serves as the HWM boundary fallback when the OS High Water Line is unavailable.

**EMODnet bathymetry** provides depth (metres below chart datum) and slope (degrees) as rasters, sampled at each cell centroid.

When multiple substrate datasets cover the same cell, priority is: BGS observed (confidence 0.90) → BGS predictive (0.65) → DEFR (0.60).

---

## check_hwl_completeness.py

Audits the topological integrity of the OS High Water Line polyline before running the build. The build pipeline polygonizes the HWL into a land mask to subtract from the outer buffer and produce a seaward-only strip; this requires the polyline to be fully closed. This script proves that it is, and must be run once against a new or updated HWL file before the first build.

Four tests are run in sequence:

| Test | What it checks |
|------|---------------|
| TEST 1 — Component analysis | How many disconnected linestring components exist after linemerge. A clean coastline is a set of closed rings (one per landmass/island); open chains indicate gaps. |
| TEST 2 — Dangle analysis | Extracts all line endpoints. A closed ring has no free endpoints. Any dangling endpoint not snapping to another within tolerance indicates a gap. |
| TEST 3 — Gap characterisation | For each pair of dangles close enough to close, reports gap size and location. Skipped automatically when TEST 2 finds zero dangles. |
| TEST 4 — Polygonization attempt | Attempts to polygonize the (optionally snapped and simplified) linestring. Reports polygon count and areas so you can verify the mainland is captured. |

Outputs written to `--out-dir` (default `output/coastline_check/`):

- `dangles.gpkg` — point layer of dangling endpoints (open in QGIS if present)
- `gaps.gpkg` — line layer connecting paired gap endpoints
- `components.gpkg` — one feature per disconnected linestring component
- `land_polygon.gpkg` — polygonized result (only written with `--polygonize`)

The OS HWL supplied with this project passed all four tests cleanly: 5,231 closed rings, 0 open chains, 0 dangling endpoints, 0 polygonization failures, mainland area 218,305 km².

### Usage

```bash
# Default run — uses raw/os_coastline/high_water_polyline.shp
python check_hwl_completeness.py

# Explicit path, with polygonization output
python check_hwl_completeness.py --shp raw/os_coastline/high_water_polyline.shp --polygonize

# Adjust snap tolerance and simplification (rarely needed)
python check_hwl_completeness.py --snap 10 --polygonize
python check_hwl_completeness.py --polygonize --no-snap --simplify 5

# Write outputs to a specific directory
python check_hwl_completeness.py --polygonize --out-dir output/coastline_check
```

The snap step is automatically skipped when Tests 1–3 confirm zero dangles (the clean-data case), avoiding an O(n²) GEOS vertex-matching operation that can take hours on an already-clean polyline. Pass `--no-snap` to force-skip it regardless.

### Dependencies

```bash
pip install geopandas shapely numpy
```

---

## build_coastal_grid.py

Builds the raw grid from the source datasets. The pipeline constructs the grid in five stages, with each stage writing its results to a `cache/` directory so individual stages can be re-run without repeating upstream work.

### Stages

| Stage | Task | Approx. time | Cached? |
|-------|------|-------------|---------|
| 1 | Load the OS HWL; polygonize into a land mask; build a seaward-only strip polygon (outer buffer minus land); rasterise the HWM line to BNG grid | ~10 s | Once |
| 2 | Generate the strip mask (HWM → 500 m seaward) and compute Euclidean distance-to-HWM via EDT | ~15 s | Once |
| 3 | Load and clip all source datasets; rasterise each as an int-coded NPZ | slow | Once per dataset |
| 4 | Build tile index — strip cells × raster lookup, tiled into 100 km × 100 km NPZ files | ~60 s | Once |
| 5 | Export — tile NPZ → normalised canonical schema → SQLite + Parquet with indexes and views | ~30 s | Always |

Stage 3 is the expensive step on first run (several minutes per dataset). Subsequent runs skip any stage whose cache files are already present unless `--force` is passed.

The strip is **seaward-only**: the land polygon polygonized from the HWL closed rings is subtracted from the symmetric outer buffer, so no inland cells are included. This is the v3 fix; earlier versions used a symmetric ±STRIP_M ribbon.

### Usage

```bash
# Standard build at 100 m resolution
python build_coastal_grid.py --root raw

# Different cell size
python build_coastal_grid.py --root raw --cell-size 50

# Quick smoke-test on 2 000 cells
python build_coastal_grid.py --root raw --sample 2000

# Re-run from Stage 3 onwards (all datasets)
python build_coastal_grid.py --root raw --force stage3

# Re-run Stage 3 for a single dataset only
python build_coastal_grid.py --root raw --force stage3:ukash

# Nuke cache and rebuild everything
python build_coastal_grid.py --root raw --force all
```

### Outputs

```
output/
├── spearo_coastal_grid_100m.db      # SQLite database
├── spearo_coastal_grid_100m.parquet # GeoParquet flat table
└── build_manifest_100m.json         # Build stats and parameters
```

The SQLite database includes six indexes on the most common query fields, a `coverage` view that summarises filled/gap counts by zone, and an `algo_inputs` view that projects the subset of columns consumed by the species/visibility algorithm.

A summary is printed at the end of a successful build including cell counts by zone, coverage percentages for each data domain, and mean confidence score. See `spearo_coastal_grid_schema.md` for the full column reference.

### Dependencies

```bash
pip install geopandas pyogrio rasterio numpy shapely pyproj pandas tqdm pyarrow scipy fiona
```

Python 3.12 · DuckDB 1.2.1 (for post-build queries; not required by the script itself)

---

## verify_strip_placement.py

Confirms that the built grid strip lies correctly seaward of the HWM (0 → 500 m offshore) rather than straddling it as a centred ribbon (~250 m either side). This is a post-build sanity check that should be run once after the first successful build, or after any change to Stage 1 or Stage 2.

Four independent tests are run:

| Test | What it checks |
|------|---------------|
| TEST 1 — Strip raster analysis | Reads the cached NPZ rasters directly. Measures how strip mask pixels distribute relative to HWM pixels. A seaward-only strip should have cells consistently on one side; a centred ribbon would straddle both. |
| TEST 2 — dist_to_hwm_m distribution | Histogram of the distance column. A correct strip has values in 0–500 m; a centred ribbon shows values from ~0 to ~250 m regardless of zone. |
| TEST 3 — Land/sea classification | Classifies a random sample of cell centroids as sea-side or land-side of the HWM using a fast vectorised nearest-segment signed cross-product. Passes if ≥95% of strip cells are on the seaward side. |
| TEST 4 — Depth signature cross-check | Land-side cells should not carry meaningful offshore depth values. Checks that any cells flagged as land-side by TEST 3 have null or negligible depth. |

TEST 3 reads the OS HWL shapefile and uses it to classify cell centroids geometrically — it is the most definitive test but also the slowest. Pass `--no-spatial` to skip it if you only want the raster-based checks.

### Usage

```bash
# Default run
python verify_strip_placement.py

# Larger spatial sample for TEST 3 (default 5 000)
python verify_strip_placement.py --sample 10000

# Skip TEST 3 (land/sea spatial classification)
python verify_strip_placement.py --no-spatial

# Explicit paths
python verify_strip_placement.py \
    --parquet output/spearo_coastal_grid_100m_filled.parquet \
    --cache cache/stage1_coastline
```

### Dependencies

```bash
pip install numpy pandas pyarrow geopandas shapely
```

---

## fill_coastal_grid.py

Reads the raw Parquet produced by `build_coastal_grid.py` and resolves all remaining nulls and unknown values, producing a gap-free `_filled.parquet`. Every cell in the output has a value for every field; the provenance of each value is recorded in `*_source` columns and fill distances are stored in `*_fill_distance_m` columns so downstream consumers can weight or discount approximated values.

### Fill passes

**Pass 0 — Pre-clean sentinels.** Removes known build-pipeline artefacts: the `ISIN-GN` bedrock sentinel, zero-fraction substrate entries where all three of %gravel/%sand/%mud are simultaneously zero (a BGS predictive raster no-data pattern), and empty substrate source labels.

**Pass 1 — Folk-code resolution.** A comprehensive lookup table maps every Folk code to `substrate_primary`, `hardness`, and heuristic %gravel/%sand/%mud/%rock fractions. This alone resolves ~42% of `unknown` substrate values without any spatial inference.

**Pass 2 — Bathymetry gap-fill.** Two-pass depth fill: first fits a linear `depth = dist_to_HWM × k` gradient from the nearest donors; falls back to straight nearest-neighbour where the gradient fit fails. Slope is filled by nearest-neighbour; morphology is re-derived from slope.

**Pass 3 — Categorical nearest-neighbour fills (parallelised).** Substrate, habitat (EUNIS/MHC), foreshore type, and bedrock are each filled independently in parallel threads using a `cKDTree` lookup against cells that already carry real data. Confidence scores are decayed linearly with fill distance: `confidence = donor_confidence × max(0.10, 1 − fill_distance / 5000 m)`.

**Pass 4 — Bedrock hardening.** Where `bedrock_exposed = True` and `substrate_primary` is still `unknown` after all NN passes, sets `substrate_primary = rock`. Runs last so legitimate sediment fills are never overwritten.

**Pass 5 — Percentage columns.** Computes `pct_rock` from the Folk-code heuristic table or as the residual of `100 − G − S − M`. Normalises all four percentage columns so they sum to 100 for rows that have at least one positive value.

**Pass 6 — Human-readable name enrichment.** Populates `eunis_name` from an embedded 167-entry EUNIS 2007-11 habitat dictionary, with ancestor fallback for unmapped deep codes. Populates `bedrock_description` by joining `bedrock_lex_rcs` against the BGS Offshore Bedrock GeoPackage attribute table (requires `--bedrock-gpkg`; if omitted, `bedrock_description` remains null).

### Usage

```bash
# Standard run (uses all available CPU cores up to 24)
python fill_coastal_grid.py output/spearo_coastal_grid_100m.parquet

# Explicit worker count
python fill_coastal_grid.py output/spearo_coastal_grid_100m.parquet --workers 24

# Write output to a specific directory
python fill_coastal_grid.py output/spearo_coastal_grid_100m.parquet --output-dir out/

# Populate bedrock_description (requires path to BGS Offshore Bedrock GeoPackage)
python fill_coastal_grid.py output/spearo_coastal_grid_100m.parquet \
    --bedrock-gpkg raw/offshore-bedrock-250k-geopackage/BGS_BedrockOffshore_250k_WGS84_v3.gpkg
```

The script is safe to re-run on an already-filled file; it strips any existing `_filled` suffix before naming the output.

### New columns added by the fill pass

| Column | Description |
|--------|-------------|
| `pct_rock` | Estimated rock fraction (0–100) |
| `bathymetry_source` | `emodnet` / `depth_gradient` / `proximal_nn` |
| `bedrock_source` | `bgs` / `proximal_nn` |
| `foreshore_source` | `defr` / `proximal_nn` |
| `substrate_fill_distance_m` | Distance to NN donor cell for substrate (0 = resolved from data) |
| `habitat_fill_distance_m` | Distance to NN donor cell for habitat |
| `fill_distance_m` | Worst-case fill distance across all domains |

See `spearo_coastal_grid_100m_filled_README.md` for the full schema of the filled dataset.

### Output coverage (100 m, England)

| Zone | Cells | Substrate coverage (raw) | Habitat coverage (raw) |
|------|-------|--------------------------|------------------------|
| Intertidal | 154,833 | 35.9% | ~36% |
| Nearshore | 314,674 | 38.4% | 32.7% |
| Offshore | 394,870 | 38.6% | 35.9% |
| **Total** | **864,377** | — | — |

After filling, all 864,377 cells carry a complete record. Mean overall confidence is ~0.31 reflecting the significant proportion of cells that depend on predictive or NN-filled data.

---

## Quick-start

```bash
# 1. Install dependencies
pip install geopandas pyogrio rasterio numpy shapely pyproj pandas tqdm pyarrow scipy fiona

# 2. Place raw datasets under raw/ (see Source datasets above)

# 3. Audit the OS High Water Line topology (required once before first build)
python check_hwl_completeness.py --polygonize

# 4. Build the raw grid
python build_coastal_grid.py --root raw

# 5. Verify strip placement (recommended after first build)
python verify_strip_placement.py

# 6. Gap-fill (add --bedrock-gpkg to populate bedrock_description)
python fill_coastal_grid.py output/spearo_coastal_grid_100m.parquet \
    --bedrock-gpkg raw/offshore-bedrock-250k-geopackage/BGS_BedrockOffshore_250k_WGS84_v3.gpkg

# 7. Point lookup
python query_grid.py --lat 50.614 --lon -1.195 --db output/spearo_coastal_grid_100m.db

# 8. Coverage check
sqlite3 output/spearo_coastal_grid_100m.db "SELECT * FROM coverage"
```

---

## Ancillary scripts

`inspect_datasets.py` walks the `raw/` directory, opens each spatial dataset, and writes a schema report to `dataset_inspection.json` and `dataset_inspection.md`. Used during initial data acquisition to verify field names and coverage before building.

`coverage_completeness.py` reads the built Parquet and produces `spearo_coastal_grid_100m_completeness.json`, a per-zone breakdown of how many cells have data for each domain (substrate, habitat, bathymetry, bedrock) and which gap combinations are most common.

`diagnose_grid.py` runs a consistency audit on a filled or unfilled Parquet: null counts, Folk→substrate mapping coverage, hardness/substrate disagreements, pct-column anomalies, and cross-field spot-checks. Useful after any change to the build or fill pipeline.

`query_grid.py` is a lightweight CLI tool that takes a lat/lon and returns the row for the nearest cell.

---

## Zone classification

Each cell is assigned to one of three coastal zones based on distance from the High Water Mark:

| Zone | Distance from HWM | Description |
|------|-------------------|-------------|
| `intertidal` | Within DEFR foreshore polygon | Exposed at low tide; covered by DEFR foreshore data |
| `nearshore` | 0 – 200 m | Shallow subtidal strip |
| `offshore` | 200 – 500 m | Outer coastal band to strip edge |

---

## Environment

Developed and tested on Windows 11 · Python 3.12.8 · DuckDB 1.2.1 · 24-core CPU · 192 GB RAM · RTX 3500 GPU. The build and fill scripts are CPU-bound and make no use of the GPU. The fill script parallelises across up to 24 workers (configurable via `--workers`). The parallel buffer step in Stage 1 uses up to 16 workers via `ProcessPoolExecutor`.