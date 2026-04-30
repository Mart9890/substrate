# Spearo Coastal Grid — Filled Dataset Reference

**File:** `spearo_coastal_grid_100m_filled.parquet`  
**Format:** Apache Parquet (Snappy compressed)  
**Rows:** 864,377  
**Columns:** 35  
**CRS:** British National Grid, EPSG:27700  
**Coverage:** England coastal strip, High Water Mark (HWM) to 1 nautical mile (1,852 m) offshore  
**Resolution:** 100 m × 100 m cells

---

## What this dataset is

A complete, gap-free lookup table of seabed and foreshore properties for every 100 m cell in the England coastal strip. Each row represents one grid cell and carries its location, bathymetry, substrate composition, habitat classification, and bedrock geology.

Every cell has a value for every field — there are no nulls in any data column. Cells where authoritative survey data existed retain that data unchanged. Cells where original data was absent have been filled using nearest-neighbour interpolation from adjacent cells (with bathymetry additionally using a distance-from-shoreline gradient model). The provenance of every value is recorded in source columns, and a `fill_distance_m` column records how far away the donor cell was for any filled value, allowing downstream algorithms to weight or discount approximated data as appropriate.

---

## Zones

Each cell is assigned to one of three coastal zones based on its distance from the High Water Mark:

| Zone | Distance from HWM | Description |
|------|-------------------|-------------|
| `intertidal` | Within foreshore polygon | Exposed at low tide |
| `nearshore` | 0 – 500 m | Shallow subtidal strip |
| `offshore` | 500 – 1,852 m | Outer coastal band to 1 nm |

---

## Data confidence

Two confidence scores are carried per cell:

- **`substrate_confidence`** and **`habitat_confidence`** — per-domain scores from 0.0 to 1.0 reflecting the quality of the underlying data source. For filled cells these are reduced by a linear decay function based on fill distance: `confidence = donor_confidence × max(0.10, 1 − fill_distance_m / 5000)`. A cell filled from 5 km away carries at minimum 10% of the donor's original confidence.
- **`overall_confidence`** — the mean of the two domain scores, excluding zero terms. This is the primary single-number quality indicator for a cell.

| Score range | Interpretation |
|-------------|----------------|
| 0.75 – 1.0 | High confidence — original survey data or close fill |
| 0.40 – 0.75 | Moderate confidence — predictive model or moderate-distance fill |
| 0.10 – 0.40 | Low confidence — distant fill or weak source |

---

## Schema

### Location

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| `cell_id` | INTEGER | No | Unique cell identifier. Primary key. |
| `easting_bng` | INTEGER | No | Cell centroid easting, British National Grid (EPSG:27700), metres. Snapped to the nearest 100 m node. |
| `northing_bng` | INTEGER | No | Cell centroid northing, British National Grid (EPSG:27700), metres. |
| `lat` | FLOAT | No | Centroid latitude, WGS84 decimal degrees. |
| `lon` | FLOAT | No | Centroid longitude, WGS84 decimal degrees. |
| `zone` | TEXT | No | Coastal zone. Values: `intertidal`, `nearshore`, `offshore`. |
| `dist_to_hwm_m` | INTEGER | No | Euclidean distance from cell centroid to nearest High Water Mark point, metres. |

---

### Bathymetry

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| `depth_m` | FLOAT | No | Water depth in metres, positive downward. Original values sampled from EMODnet bathymetry raster. Filled values estimated from nearest observed depth or from a local depth-gradient model (`depth = dist_to_hwm × k`) fitted from neighbouring cells. |
| `slope_deg` | FLOAT | No | Seabed slope in degrees. Original values from EMODnet slope raster. Filled by nearest-neighbour. |
| `morphology` | TEXT | No | Qualitative slope class derived from `slope_deg`. Values: `flat` (< 1°), `gentle_slope` (1–5°), `slope` (5–15°), `steep` (15–30°), `cliff` (≥ 30°). |
| `bathymetry_source` | TEXT | No | Origin of the depth value. Values: `emodnet` (direct raster sample), `depth_gradient` (gradient model from neighbouring cells), `proximal_nn` (nearest-neighbour from closest cell with real depth data). |

---

### Substrate

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| `substrate_primary` | TEXT | No | Primary substrate category. Values: `rock`, `gravel`, `sand`, `mud`, `mixed`, `unknown`. |
| `folk_code` | TEXT | No | BGS modified Folk classification code (e.g. `S` = sand, `G` = gravel, `mS` = muddy sand, `ROCK`). |
| `folk_description` | TEXT | No | Human-readable description of the Folk code (e.g. `SAND (SEA BED SEDIMENT, BASED ON FOLK)`). |
| `pct_gravel` | FLOAT | No | Percentage of gravel in the sediment, 0–100. |
| `pct_sand` | FLOAT | No | Percentage of sand in the sediment, 0–100. |
| `pct_mud` | FLOAT | No | Percentage of mud in the sediment, 0–100. |
| `pct_rock` | FLOAT | No | Percentage of rock/hard substrate, 0–100. Derived from bedrock exposure, Folk code, or as the residual `max(0, 100 − gravel − sand − mud)`. |
| `hardness` | TEXT | No | Broad hardness class. Values: `hard` (rock, gravel, or exposed bedrock), `soft` (sand, mud), `mixed`, `unknown`. |
| `substrate_source` | TEXT | No | Origin of the substrate data. Values: `BGS_observed` (physical sampling, confidence 0.90), `BGS_predictive` (statistical model, 0.65), `DEFR` (intertidal foreshore polygons, 0.60), `proximal_nn` (nearest-neighbour fill). |
| `substrate_confidence` | FLOAT | No | Confidence weight for substrate data, 0.0–1.0. See Data confidence section. |

**Note:** `pct_gravel + pct_sand + pct_mud + pct_rock = 100` for every row.

---

### Habitat

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| `eunis_code` | TEXT | No | EUNIS v2007-11 habitat code (e.g. `A5.2` = Sublittoral sediment, `A2.3` = Muddy sand shores). From UKASH Combined Map v2025.1. |
| `eunis_name` | TEXT | No | Human-readable EUNIS habitat name. |
| `mhc_code` | TEXT | No | Marine Habitat Classification for Britain and Ireland v22.04 code (e.g. `SS.SSa` = Sublittoral sand, `LS.LMu` = Littoral mud). From UKASH. |
| `habitat_source` | TEXT | No | Origin of habitat data. Values: `UKASH_survey` (ground-truthed survey, confidence 0.85), `UKASH_predictive` (UKSeaMap model, 0.55), `proximal_nn` (nearest-neighbour fill). |
| `habitat_confidence` | FLOAT | No | Confidence weight for habitat data, 0.0–1.0. See Data confidence section. |

---

### Intertidal / foreshore

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| `foreshore_type` | TEXT | No | Foreshore substrate classification from the DEFR Intertidal Foreshore dataset (e.g. `sand`, `rock`, `mud`). Populated for intertidal cells; filled by nearest-neighbour for intertidal cells not covered by a DEFR polygon. Non-intertidal cells carry the value of the nearest intertidal donor. |

---

### Bedrock

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| `bedrock_lex_rcs` | TEXT | No | BGS LEX_RCS code of the underlying bedrock polygon (e.g. `MSQH-SND`, `CR-LMST`). Lexicon (LEX) describes the age/stratigraphic unit; Rock Classification System (RCS) describes the lithology. |
| `bedrock_description` | TEXT | No | Human-readable description of the `bedrock_lex_rcs` code (e.g. `MARINE SEDIMENTS, HOLOCENE (UNDIFFERENTIATED) - SAND`). |
| `bedrock_exposed` | BOOLEAN | No | `True` if the bedrock outcrops at the seabed surface. When `True`, `hardness` is set to `hard` and `substrate_primary` is set to `rock` where otherwise unknown. |
| `bedrock_source` | TEXT | No | Origin of bedrock data. Values: `bgs` (BGS Offshore Bedrock 250k GeoPackage), `proximal_nn` (nearest-neighbour fill). |

---

### Data quality

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| `has_observed_survey` | BOOLEAN | No | `True` if `substrate_source == 'BGS_observed'` — i.e. the substrate is from direct physical sampling rather than modelling or fill. Use this to filter to the highest-confidence cells. |
| `coverage_flags` | TEXT | Yes | Comma-separated list of domains that could not be filled from any source. `NULL` if all domains are covered (expected to be `NULL` for all rows in this dataset). Possible tokens: `substrate`, `habitat`, `bathymetry`, `bedrock`. |
| `overall_confidence` | FLOAT | No | Mean of `substrate_confidence` and `habitat_confidence`, excluding zero terms. Primary single-number quality indicator, 0.0–1.0. |
| `fill_distance_m` | FLOAT | No | Distance in metres from this cell to the nearest donor cell used to fill any domain. `0.0` for cells where all data came from direct observation or modelling with no gap-fill. Useful for downstream weighting — a high value indicates the cell's data is extrapolated over a large distance. |

---

## Source data provenance

| Source | Publisher | What it provides |
|--------|-----------|-----------------|
| BGS Seabed Sediments 250k (observed) | British Geological Survey | Folk codes from physical grab samples and sonar surveys |
| BGS Predictive Seabed Sediments UK v1 | British Geological Survey | Statistically modelled Folk codes and pct_gravel / pct_sand / pct_mud rasters |
| UKASH Combined Map v2025.1 | JNCC | EUNIS and MHC habitat codes from survey and UKSeaMap model |
| BGS Offshore Bedrock 250k | British Geological Survey | Bedrock LEX_RCS codes and exposure |
| DEFR Intertidal Foreshore | DEFR / Natural England | Intertidal foreshore substrate polygons |
| EMODnet Bathymetry | EMODnet | Depth and slope rasters |

---

## Using the dataset

### Reading in Python

```python
import pandas as pd

df = pd.read_parquet("spearo_coastal_grid_100m_filled.parquet")

# All columns are populated — no null handling required for data columns
# Filter to high-confidence cells only
high_conf = df[df["overall_confidence"] >= 0.75]

# Filter to cells with original (non-filled) data only
original  = df[df["fill_distance_m"] == 0]

# Look up a specific location by BNG coordinates (nearest cell)
from scipy.spatial import cKDTree
import numpy as np

coords = np.column_stack([df["easting_bng"], df["northing_bng"]])
tree   = cKDTree(coords)
dist, idx = tree.query([[460500, 80300]])   # easting, northing
cell = df.iloc[idx[0]]
```

### Reading in DuckDB

```sql
-- Load the parquet directly
SELECT * FROM 'spearo_coastal_grid_100m_filled.parquet'
WHERE zone = 'nearshore'
  AND overall_confidence >= 0.5
  AND substrate_primary IN ('rock', 'gravel')
LIMIT 100;

-- Sediment composition for a bounding box
SELECT cell_id, lat, lon, depth_m, substrate_primary,
       pct_rock, pct_gravel, pct_sand, pct_mud,
       overall_confidence, fill_distance_m
FROM 'spearo_coastal_grid_100m_filled.parquet'
WHERE easting_bng BETWEEN 430000 AND 445000
  AND northing_bng BETWEEN 80000 AND 95000;
```

---

## Key column value reference

### substrate_primary
| Value | Meaning |
|-------|---------|
| `rock` | Hard substrate — bedrock or consolidated material |
| `gravel` | Coarse sediment, pebbles |
| `sand` | Sandy seabed |
| `mud` | Fine silty/muddy seabed |
| `mixed` | Mixed sediment types |
| `unknown` | Could not be classified |

### substrate_source / habitat_source
| Value | Meaning | Confidence |
|-------|---------|-----------|
| `BGS_observed` | Physical sampling (grabs, cores, sonar) | 0.90 |
| `BGS_predictive` | Statistical sediment model | 0.65 |
| `DEFR` | Intertidal foreshore polygon | 0.60 |
| `UKASH_survey` | Ground-truthed habitat survey | 0.85 |
| `UKASH_predictive` | UKSeaMap predictive habitat model | 0.55 |
| `proximal_nn` | Nearest-neighbour fill from adjacent cell | Decayed |

### bathymetry_source
| Value | Meaning |
|-------|---------|
| `emodnet` | Directly sampled from EMODnet raster |
| `depth_gradient` | Modelled from local depth/distance gradient |
| `proximal_nn` | Nearest-neighbour from closest cell with real depth |

### morphology
| Value | Slope range |
|-------|------------|
| `flat` | < 1° |
| `gentle_slope` | 1° – 5° |
| `slope` | 5° – 15° |
| `steep` | 15° – 30° |
| `cliff` | ≥ 30° |

### zone
| Value | Distance from HWM |
|-------|-------------------|
| `intertidal` | Within DEFR foreshore polygon |
| `nearshore` | 0 – 500 m |
| `offshore` | 500 – 1,852 m |
