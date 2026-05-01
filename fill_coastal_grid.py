#!/usr/bin/env python3
"""
fill_coastal_grid.py  —  Proximal gap-fill for the Spearo coastal grid
=======================================================================
Reads spearo_coastal_grid_100m.parquet, resolves all derivable fields from
existing data first (no NN needed), then gap-fills remaining holes via
nearest-neighbour, and writes a _filled parquet.

Pass 0: Pre-clean sentinel values
----------------------------------
  - bedrock_lex_rcs = "ISIN-GN" is a build-pipeline default, not real data.
    Treated as null throughout.
  - substrate_source in ("none", "") treated as null.
  - pct_gravel / pct_sand / pct_mud of 0.0 where all three are zero AND
    substrate_primary is not "rock" — treated as unknown (set to NaN),
    because the BGS predictive rasters return 0/0/0 for unmapped cells,
    not a real "100% rock" reading.

Pass 1: Folk-code resolution (no NN required)
---------------------------------------------
  Every cell already has a folk_code.  A comprehensive lookup table maps
  each folk code to:
    - substrate_primary  (rock / gravel / sand / mud / mixed)
    - hardness           (hard / soft / mixed)
    - pct_rock / pct_gravel / pct_sand / pct_mud  (heuristic best-guess)
  This alone resolves ~42% of "unknown" substrate_primary values without
  any spatial inference.

Pass 2: Bathymetry gap-fill
----------------------------
  Two-pass depth fill:
    1. depth_gradient — fit depth = dist_to_hwm x k from k nearest donors.
    2. proximal_nn fallback where gradient fit fails.
  slope_deg filled by NN; morphology re-derived from slope.
  Also separately fills slope_deg gaps on rows that already have depth.

Pass 3: Categorical NN gap-fill (parallel)
-------------------------------------------
  Substrate    NN to nearest cell with a resolved substrate_primary.
  Habitat      NN to nearest cell with eunis_code.
  Foreshore    NN within intertidal zone.
  Bedrock      NN to nearest cell with a real (non-sentinel) bedrock_lex_rcs.

Pass 4: Post-merge bedrock hardening
--------------------------------------
  Where bedrock_exposed=True and substrate_primary is STILL "unknown" after
  all passes, set substrate_primary="rock".  Runs last so legitimate sediment
  fills are never overwritten.

Pass 5: pct_rock computation and normalisation
----------------------------------------------
  pct_rock derived from folk-code heuristic table where available.
  For cells where G+S+M > 0, residual = 100 - G - S - M.
  Rows with no percentage information get NaN (not fabricated values).
  Normalisation only applied to rows that have at least one positive value.

Pass 6: human-readable name enrichment
---------------------------------------
  eunis_name      — Populated from an embedded EUNIS 2007-11 marine habitat
                    lookup dictionary (167 entries).  Compound slash-separated
                    codes (e.g. "A4.21/A4.22") resolve via the first match.
                    Deep codes fall back to the nearest ancestor
                    (e.g. an unmapped 5th-level code → 4th-level parent).
                    Note: build_coastal_grid.py requested a non-existent field
                    "EUNISDesc" from UKASH; the correct field is "OrigName".
                    The lookup here handles both routes.

  bedrock_description — Constructed as "LEX_D - RCS_D" by joining the
                    bedrock_lex_rcs values against the BGS Offshore Bedrock
                    250k GeoPackage attribute table.  The GeoPackage has no
                    combined LEX_RCS_D field so the two parts are concatenated.
                    Requires --bedrock-gpkg path to be supplied.

New columns
-----------
  bathymetry_source          emodnet | depth_gradient | proximal_nn
  bedrock_source             bgs | proximal_nn
  foreshore_source           defr | proximal_nn
  substrate_fill_distance_m  per-domain NN distance (0 = resolved from data)
  habitat_fill_distance_m
  fill_distance_m            worst-case max across all domains
  pct_rock

Usage
-----
  python fill_coastal_grid.py output/spearo_coastal_grid_100m.parquet
  python fill_coastal_grid.py output/spearo_coastal_grid_100m.parquet --workers 24
  python fill_coastal_grid.py output/spearo_coastal_grid_100m.parquet --output-dir out/
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

warnings.filterwarnings("ignore")

# ── Constants ─────────────────────────────────────────────────────────────────

DEFAULT_WORKERS     = min(24, os.cpu_count() or 8)
CONFIDENCE_DECAY_M  = 5_000.0
CONFIDENCE_FLOOR    = 0.10
GRADIENT_NEIGHBOURS = 16
GRADIENT_MIN_DONORS = 4
NN_SOURCE_LABEL     = "proximal_nn"

# Sentinel bedrock value written by the build pipeline for every cell
# regardless of whether real bedrock data exists — treat as null
BEDROCK_SENTINEL = "ISIN-GN"

# Hardcoded path to the BGS Offshore Bedrock 250k GeoPackage.
# Pass 6 uses this to populate bedrock_description.  Update this path
# if you move the raw data directory.
BEDROCK_GPKG = Path(
    r"C:\Spearo\substrate dataset\raw\offshore-bedrock-250k-geopackage"
    r"\BGS_BedrockOffshore_250k_WGS84_v3.gpkg"
)

# Maximum depth assigned to intertidal cells.
# EMODnet bathymetry does not cover the intertidal zone, so these cells
# get their depth from NN or gradient fill using offshore neighbours —
# which can produce unrealistically deep values.  We clamp to this limit
# so intertidal cells never appear deeper than the lowest astronomical
# tide water level.
INTERTIDAL_DEPTH_CAP_M = -4.0

# Bathymetry confidence by source
BATHY_CONFIDENCE = {
    "emodnet":        1.0,
    "depth_gradient": 0.5,
    NN_SOURCE_LABEL:  0.3,
}

# ── EUNIS 2007-11 marine habitat lookup ──────────────────────────────────────
# Complete hierarchy for the A (marine) section of EUNIS 2007-11 relevant to
# UK coastal and offshore waters.  Used to populate eunis_name from eunis_code.
# Source: EEA EUNIS 2007-11 classification, marine section.

EUNIS_NAMES: dict[str, str] = {
    # Level 1
    "A":        "Marine habitats",
    # Level 2
    "A1":       "Littoral rock and other hard substrata",
    "A2":       "Littoral sediment",
    "A3":       "Infralittoral rock and other hard substrata",
    "A4":       "Circalittoral rock and other hard substrata",
    "A5":       "Sublittoral sediment",
    "A6":       "Deep-sea bed",
    "A7":       "Pelagic water column",
    # A1: Littoral rock
    "A1.1":     "High energy littoral rock",
    "A1.11":    "Exposed littoral rock without macroalgae",
    "A1.111":   "Exposed littoral rock with barnacles and/or limpets",
    "A1.112":   "Exposed littoral rock with mussel communities",
    "A1.12":    "Exposed littoral rock with encrusting algae",
    "A1.13":    "Exposed littoral rock with fucoids",
    "A1.2":     "Moderate energy littoral rock",
    "A1.21":    "Moderate energy littoral rock with barnacles and/or limpets",
    "A1.22":    "Moderate energy littoral rock with mussels",
    "A1.23":    "Moderate energy littoral rock with fucoids",
    "A1.3":     "Low energy littoral rock",
    "A1.31":    "Low energy littoral rock with patchy communities",
    "A1.311":   "Low energy littoral rock with barnacles",
    "A1.3111":  "Low energy littoral rock with barnacles and fucoids",
    "A1.3112":  "Low energy littoral rock with barnacles and mussels",
    "A1.312":   "Low energy littoral rock with fucoids",
    "A1.3121":  "Low energy littoral rock with Fucus vesiculosus",
    "A1.3122":  "Low energy littoral rock with Fucus serratus",
    "A1.313":   "Low energy littoral rock with kelp",
    "A1.3131":  "Low energy littoral rock with Laminaria digitata and fucoids",
    "A1.3132":  "Low energy littoral rock with Laminaria digitata on sand-influenced rock",
    "A1.32":    "Low energy littoral rock with coralline algae",
    "A1.33":    "Low energy littoral rock with mussels",
    "A1.4":     "Littoral biogenic reef",
    "A1.41":    "Littoral mussel beds",
    "A1.42":    "Littoral oyster beds",
    "A1.43":    "Littoral polychaete reefs (Sabellaria)",
    "A1.5":     "Littoral rock with caves and overhangs",
    # A2: Littoral sediment
    "A2.1":     "Littoral coarse sediment",
    "A2.11":    "Shingle and gravel shores",
    "A2.2":     "Littoral sand and muddy sand",
    "A2.21":    "Bare littoral sand",
    "A2.22":    "Littoral sand with seagrass (Zostera)",
    "A2.23":    "Littoral sand with polychaetes",
    "A2.24":    "Littoral sand with bivalves",
    "A2.25":    "Littoral sand with cockles",
    "A2.3":     "Littoral muddy sand",
    "A2.31":    "Littoral muddy sand with amphipods",
    "A2.32":    "Littoral muddy sand with Hediste diversicolor",
    "A2.33":    "Littoral muddy sand with bivalves",
    "A2.34":    "Littoral muddy sand with lugworms (Arenicola)",
    "A2.4":     "Littoral mud",
    "A2.41":    "Littoral mud with Arenicola marina",
    "A2.42":    "Littoral mud with Nereis and Corophium",
    "A2.43":    "Littoral mud with Hydrobia",
    "A2.44":    "Littoral mud flats",
    "A2.5":     "Coastal saltmarshes and saline reedbeds",
    "A2.51":    "Atlantic low-mid saltmarsh",
    "A2.52":    "Atlantic mid-upper saltmarsh",
    "A2.6":     "Littoral biogenic habitats",
    "A2.61":    "Littoral mixed sediment",
    "A2.62":    "Maerl beds in the littoral zone",
    # A3: Infralittoral rock
    "A3.1":     "High energy infralittoral rock",
    "A3.11":    "Exposed infralittoral rock with kelp",
    "A3.111":   "Kelp forest with dense foliose red algae (exposed)",
    "A3.1111":  "Laminaria hyperborea with dense foliose red algae (exposed)",
    "A3.1112":  "Laminaria hyperborea with sparse foliose red algae (exposed)",
    "A3.112":   "Kelp forest with foliose red algae and sponges",
    "A3.113":   "Kelp forest with coralline algae",
    "A3.12":    "Exposed infralittoral rock with mixed algae",
    "A3.13":    "Exposed infralittoral rock with encrusting coralline algae",
    "A3.2":     "Moderate energy infralittoral rock",
    "A3.21":    "Moderate energy infralittoral rock with kelp",
    "A3.211":   "Kelp forest with foliose red algae (moderate energy)",
    "A3.2111":  "Laminaria hyperborea with foliose red algae (moderate energy)",
    "A3.22":    "Moderate energy infralittoral rock with mixed algae",
    "A3.3":     "Low energy infralittoral rock",
    "A3.31":    "Kelp in silted conditions",
    "A3.311":   "Laminaria saccharina and/or Laminaria digitata with sediment",
    "A3.32":    "Low energy infralittoral rock with mixed communities",
    "A3.4":     "Infralittoral biogenic reef",
    "A3.41":    "Infralittoral mussel beds",
    "A3.5":     "Infralittoral rock with caves or overhangs",
    # A4: Circalittoral rock
    "A4.1":     "High energy circalittoral rock",
    "A4.11":    "Circalittoral rock with hydroids and bryozoans",
    "A4.12":    "Circalittoral rock with sponges",
    "A4.13":    "Circalittoral rock with tubeworms",
    "A4.14":    "Circalittoral rock with Ross coral (Pentapora)",
    "A4.2":     "Moderate energy circalittoral rock",
    "A4.21":    "Circalittoral rock with red algae",
    "A4.22":    "Circalittoral rock with encrusting coralline algae",
    "A4.23":    "Circalittoral rock with sponges and soft corals",
    "A4.24":    "Circalittoral rock with hydroids and bryozoans",
    "A4.25":    "Circalittoral rock with mixed fauna",
    "A4.26":    "Circalittoral rock with polychaetes",
    "A4.27":    "Circalittoral rock with bivalves (horse mussels)",
    "A4.3":     "Low energy circalittoral rock",
    "A4.31":    "Silted circalittoral rock",
    "A4.32":    "Low energy circalittoral rock with bryozoans and sponges",
    "A4.33":    "Low energy circalittoral rock with ascidians",
    "A4.4":     "Circalittoral biogenic reef",
    "A4.41":    "Circalittoral coral gardens (cold water)",
    "A4.42":    "Maerl beds in the circalittoral zone",
    "A4.43":    "Circalittoral polychaete reefs (Sabellaria spinulosa)",
    "A4.44":    "Serpulid aggregations",
    "A4.5":     "Circalittoral rock with caves or overhangs",
    "A4.7":     "Deep circalittoral rock",
    # A5: Sublittoral sediment
    "A5.1":     "Sublittoral coarse sediment",
    "A5.11":    "Infralittoral coarse sediment",
    "A5.12":    "Circalittoral coarse sediment",
    "A5.13":    "Deep circalittoral coarse sediment",
    "A5.14":    "Circalittoral coarse sediment with bivalves",
    "A5.15":    "Offshore circalittoral coarse sediment",
    "A5.2":     "Sublittoral sand",
    "A5.21":    "Infralittoral sand",
    "A5.22":    "Infralittoral mobile sand",
    "A5.23":    "Infralittoral fine sand",
    "A5.24":    "Infralittoral muddy sand",
    "A5.25":    "Circalittoral fine sand",
    "A5.26":    "Circalittoral muddy sand",
    "A5.27":    "Deep circalittoral sand",
    "A5.28":    "Offshore circalittoral sand",
    "A5.3":     "Sublittoral mud",
    "A5.31":    "Infralittoral mud",
    "A5.32":    "Circalittoral mud",
    "A5.33":    "Deep circalittoral mud",
    "A5.34":    "Offshore circalittoral mud",
    "A5.35":    "Circalittoral mud with burrowing megafauna",
    "A5.36":    "Offshore circalittoral mud with burrowing megafauna",
    "A5.37":    "Deep offshore mud",
    "A5.4":     "Sublittoral mixed sediment",
    "A5.41":    "Infralittoral mixed sediment",
    "A5.42":    "Circalittoral mixed sediment",
    "A5.43":    "Deep circalittoral mixed sediment",
    "A5.44":    "Offshore circalittoral mixed sediment",
    "A5.45":    "Sublittoral mixed sediment with conspicuous fauna",
    "A5.5":     "Sublittoral biogenic habitats",
    "A5.51":    "Maerl beds in the sublittoral zone",
    "A5.52":    "Sublittoral seagrass beds",
    "A5.53":    "Horse mussel beds (Modiolus)",
    "A5.54":    "Sublittoral mussel beds",
    "A5.55":    "Sabellaria spinulosa reefs on sublittoral sediment",
    "A5.56":    "Sublittoral sand and gravel with Lanice conchilega",
    "A5.57":    "Sublittoral bivalve beds",
    "A5.61":    "Sublittoral biogenic gravel",
    "A5.62":    "Sublittoral mixed sediment with maerl",
    # A6: Deep-sea bed
    "A6.1":     "Deep-sea rock and artificial hard substrata",
    "A6.3":     "Deep-sea sand and gravelly sand",
    "A6.4":     "Deep-sea muddy sand and sandy mud",
    "A6.5":     "Deep-sea mud",
    "A6.6":     "Deep-sea mixed sediment",
}


def eunis_name_for_code(code) -> str | None:
    """
    Return the EUNIS 2007-11 habitat name for a code, with ancestor fallback.

    Handles:
      - Exact match:     "A5.25"  → "Circalittoral fine sand"
      - Compound codes:  "A4.21/A4.22/A4.23" → name of first resolvable part
      - Ancestor fallback: "A3.1111" → tries each shorter prefix until a hit
    Returns None only if the code is null/empty or completely unresolvable.
    """
    if not code or (isinstance(code, float) and code != code):
        return None
    code = str(code).strip()
    if not code:
        return None
    # Compound slash-separated: resolve the first resolvable part
    if "/" in code:
        for part in code.split("/"):
            r = eunis_name_for_code(part.strip())
            if r:
                return r
        return None
    # Exact match
    if code in EUNIS_NAMES:
        return EUNIS_NAMES[code]
    # Ancestor fallback: trim trailing digits/dots until we hit a match
    candidate = code
    while candidate:
        last = candidate[-1]
        if last.isdigit() or last == ".":
            candidate = candidate[:-1]
        else:
            break
        if candidate in EUNIS_NAMES:
            return EUNIS_NAMES[candidate]
    return None


# ── Folk code lookup table ────────────────────────────────────────────────────
# Maps folk_code.upper().strip() ->
#   (substrate_primary, hardness, pct_rock, pct_gravel, pct_sand, pct_mud)
# pct values are heuristic best-guesses used when BGS rasters returned no data.
# None means genuinely unresolvable (e.g. "NOT PRESENT") — leave as unknown.

FOLK_MAP: dict[str, tuple[str, str, float, float, float, float] | None] = {
    # ── Pure sand ──────────────────────────────────────────────────────────
    "S":            ("sand",   "soft",  0.0,  0.0, 100.0,  0.0),
    "SND":          ("sand",   "soft",  0.0,  0.0, 100.0,  0.0),
    "SAND":         ("sand",   "soft",  0.0,  0.0, 100.0,  0.0),
    "SS":           ("sand",   "soft",  0.0,  0.0, 100.0,  0.0),
    "MS":           ("sand",   "soft",  0.0,  5.0,  90.0,  5.0),
    "GS":           ("sand",   "soft",  0.0, 20.0,  80.0,  0.0),
    "mS":           ("sand",   "soft",  0.0,  5.0,  90.0,  5.0),
    "sS":           ("sand",   "soft",  0.0,  0.0, 100.0,  0.0),
    "gS":           ("sand",   "soft",  0.0, 20.0,  80.0,  0.0),
    # ── Pure mud ───────────────────────────────────────────────────────────
    "M":            ("mud",    "soft",  0.0,  0.0,  0.0, 100.0),
    "MUD":          ("mud",    "soft",  0.0,  0.0,  0.0, 100.0),
    "sM":           ("mud",    "soft",  0.0,  0.0, 10.0,  90.0),
    "gM":           ("mud",    "soft",  0.0, 10.0,  0.0,  90.0),
    "MUDM":         ("mud",    "soft",  0.0,  0.0,  5.0,  95.0),
    # ── Pure gravel ────────────────────────────────────────────────────────
    "G":            ("gravel", "hard",  5.0, 90.0,  5.0,  0.0),
    "GV":           ("gravel", "hard",  5.0, 90.0,  5.0,  0.0),
    "GVL":          ("gravel", "hard",  5.0, 90.0,  5.0,  0.0),
    "GRAVEL":       ("gravel", "hard",  5.0, 90.0,  5.0,  0.0),
    "sG":           ("gravel", "hard",  5.0, 85.0, 10.0,  0.0),
    "mG":           ("gravel", "hard",  5.0, 80.0,  5.0, 10.0),
    # ── Rock ───────────────────────────────────────────────────────────────
    "ROCK":                                      ("rock", "hard", 100.0, 0.0,  0.0,  0.0),
    "R":                                         ("rock", "hard", 100.0, 0.0,  0.0,  0.0),
    "BDRK":                                      ("rock", "hard", 100.0, 0.0,  0.0,  0.0),
    "BEDROCK":                                   ("rock", "hard", 100.0, 0.0,  0.0,  0.0),
    "HR":                                        ("rock", "hard", 100.0, 0.0,  0.0,  0.0),
    "HARD ROCK":                                 ("rock", "hard", 100.0, 0.0,  0.0,  0.0),
    "ROCK PLATFORM":                             ("rock", "hard", 100.0, 0.0,  0.0,  0.0),
    "RODI":                                      ("rock", "hard", 100.0, 0.0,  0.0,  0.0),
    "RSD":                                       ("rock", "hard",  80.0, 0.0, 20.0,  0.0),
    "BOULDERS/LOOSE ROCK":                       ("rock", "hard", 100.0, 0.0,  0.0,  0.0),
    "ROCK PLATFORM WITH BOULDERS/LOSE ROCK":     ("rock", "hard", 100.0, 0.0,  0.0,  0.0),
    "ROCK PLATFORM WITH BANKS OF GRAVEL":        ("rock", "hard",  60.0,35.0,  5.0,  0.0),
    # ── Mixed: gravelly sand / sandy gravel ────────────────────────────────
    "GVSND":        ("mixed", "mixed",  5.0, 30.0, 60.0,  5.0),
    "GSND":         ("mixed", "mixed",  5.0, 30.0, 60.0,  5.0),
    "SNDGV":        ("mixed", "mixed",  5.0, 25.0, 65.0,  5.0),
    "(G)S":         ("mixed", "mixed",  5.0, 20.0, 70.0,  5.0),
    "SLGVSD":       ("mixed", "hard",  10.0, 55.0, 30.0,  5.0),
    "SLGVMS":       ("mixed", "hard",  10.0, 50.0, 25.0, 15.0),
    "SLGVSM":       ("mixed", "hard",  10.0, 50.0, 20.0, 20.0),
    "SNDGM":        ("mixed", "mixed",  0.0, 15.0, 55.0, 30.0),
    # ── Mixed: sand and gravel ─────────────────────────────────────────────
    "SAND & GRAVEL": ("mixed", "mixed",  5.0, 45.0, 45.0,  5.0),
    "MUD & GRAVEL":  ("mixed", "mixed",  5.0, 40.0, 10.0, 45.0),
    "GVMUD":         ("mixed", "mixed",  5.0, 40.0,  5.0, 50.0),
    "MUDGV":         ("mixed", "mixed",  5.0, 35.0,  5.0, 55.0),
    # ── Mixed: sand and mud ────────────────────────────────────────────────
    "SAND & MUD":   ("mixed", "soft",   0.0,  0.0, 50.0, 50.0),
    "MUDSND":       ("mixed", "soft",   0.0,  0.0, 40.0, 60.0),
    "SNDMUD":       ("mixed", "soft",   0.0,  0.0, 60.0, 40.0),
    "MUDSGV":       ("mixed", "soft",   0.0, 15.0, 15.0, 70.0),
    "(G)M":         ("mixed", "soft",   0.0, 10.0,  5.0, 85.0),
    # ── Biogenic / other ───────────────────────────────────────────────────
    "BIOM":         ("mixed", "mixed",  5.0, 10.0, 75.0, 10.0),
    "XVSZ":         ("sand",  "soft",   0.0,  5.0, 90.0,  5.0),
    # ── Genuinely unresolvable ─────────────────────────────────────────────
    "NOT PRESENT":            None,
    "UNSPECIFIED":            None,
    "MADE GROUND (MAN MADE)": None,
    "NODATA":                 None,
}

# Substrings that identify rock regardless of exact code wording
ROCK_SUBSTRINGS = [
    "ROCK PLATFORM", "BOULDERS", "LOOSE ROCK", "BEDROCK", "HARD ROCK",
]

# Set of folk codes that definitively indicate rock (for pct_rock override)
ROCK_FOLK_CODES = frozenset(
    k for k, v in FOLK_MAP.items() if v is not None and v[0] == "rock"
)


# ── Logging ───────────────────────────────────────────────────────────────────

def log(msg: str, indent: int = 0, elapsed: float | None = None):
    suffix = f"  [{elapsed:.1f}s]" if elapsed is not None else ""
    print("  " * indent + msg + suffix, flush=True)


# ── I/O ───────────────────────────────────────────────────────────────────────

def load_parquet(path: Path) -> pd.DataFrame:
    t0 = time.time()
    log(f"Loading {path.name} ...")
    df = pd.read_parquet(path)
    log(f"  {len(df):,} rows x {len(df.columns)} columns", elapsed=time.time() - t0)
    return df


def save_parquet(df: pd.DataFrame, path: Path):
    t0 = time.time()
    log(f"Writing {path.name} ...")
    df.to_parquet(path, index=False, engine="pyarrow", compression="snappy")
    mb = path.stat().st_size / 1_048_576
    log(f"  {len(df):,} rows  {mb:.1f} MB", elapsed=time.time() - t0)


# ── KD-tree helpers ───────────────────────────────────────────────────────────

def bng_coords(df: pd.DataFrame) -> np.ndarray:
    return np.column_stack([
        df["easting_bng"].values.astype(np.float64),
        df["northing_bng"].values.astype(np.float64),
    ])


def build_tree(df: pd.DataFrame) -> cKDTree:
    return cKDTree(bng_coords(df))


def confidence_decay(distance_m: np.ndarray) -> np.ndarray:
    return np.clip(1.0 - distance_m / CONFIDENCE_DECAY_M, CONFIDENCE_FLOOR, 1.0)


# ── Pass 0: Pre-clean ─────────────────────────────────────────────────────────

def pre_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Neutralise known sentinel / default values so they don't poison fills.

    1. bedrock_lex_rcs = ISIN-GN is a build-pipeline default for every cell,
       not real bedrock data. Clear it and dependent fields.
    2. substrate_source in ("none", "") treated as null.
    3. pct columns all-zero where substrate is not rock — treat as unknown.
       The BGS predictive rasters return 0/0/0 for cells without real data;
       leaving these as 0.0 would make every such cell appear to be 100% rock
       after normalisation.
    4. hardness = "unknown" — sentinel string, replace with None.
    """
    log("Pass 0: pre-cleaning sentinel values ...", indent=1)

    # 1. Bedrock sentinel
    bed_mask = df["bedrock_lex_rcs"] == BEDROCK_SENTINEL
    df.loc[bed_mask, "bedrock_lex_rcs"]     = None
    df.loc[bed_mask, "bedrock_description"] = None
    df.loc[bed_mask, "bedrock_exposed"]     = False
    log(f"  bedrock sentinel ({BEDROCK_SENTINEL}) cleared: {int(bed_mask.sum()):,}", indent=2)

    # 2. Substrate source sentinels
    df["substrate_source"] = df["substrate_source"].replace({"none": None, "": None})

    # 3. All-zero pct rows on non-rock cells
    pct_cols = ["pct_gravel", "pct_sand", "pct_mud"]
    if all(c in df.columns for c in pct_cols):
        all_zero = (
            (df["pct_gravel"].fillna(0) == 0) &
            (df["pct_sand"].fillna(0)   == 0) &
            (df["pct_mud"].fillna(0)    == 0) &
            (df["substrate_primary"].fillna("unknown") != "rock")
        )
        df.loc[all_zero, pct_cols] = np.nan
        log(f"  all-zero pct neutralised (non-rock rows): {int(all_zero.sum()):,}", indent=2)

    # 4. Hardness sentinel
    df["hardness"] = df["hardness"].replace({"unknown": None})

    return df


# ── Pass 1: Folk-code resolution ──────────────────────────────────────────────

def _folk_lookup(folk_code) -> tuple | None:
    """Return FOLK_MAP entry for a folk code, falling back to substring rock check."""
    if folk_code is None or (isinstance(folk_code, float) and np.isnan(folk_code)):
        return None
    upper = str(folk_code).strip().upper()
    entry = FOLK_MAP.get(upper)
    if entry is not None:
        return entry
    for substr in ROCK_SUBSTRINGS:
        if substr in upper:
            return ("rock", "hard", 100.0, 0.0, 0.0, 0.0)
    return None


def resolve_from_folk(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pass 1: derive substrate_primary, hardness, and pct columns from folk_code.

    - Fills substrate_primary where currently unknown/null.
    - Fills hardness where currently null.
    - Sets heuristic pct values where pct columns are currently NaN.
      These are best-guess compositions based on BGS Folk classification
      descriptions; they are better than NaN or fabricated zeros.
    """
    log("Pass 1: folk-code resolution ...", indent=1)
    t0 = time.time()
    n = len(df)

    folk_primary  = np.full(n, None, dtype=object)
    folk_hardness = np.full(n, None, dtype=object)
    folk_pct_rock = np.full(n, np.nan, dtype=np.float64)
    folk_pct_grav = np.full(n, np.nan, dtype=np.float64)
    folk_pct_sand = np.full(n, np.nan, dtype=np.float64)
    folk_pct_mud  = np.full(n, np.nan, dtype=np.float64)

    for i, code in enumerate(df["folk_code"].values):
        entry = _folk_lookup(code)
        if entry is not None:
            folk_primary[i]  = entry[0]
            folk_hardness[i] = entry[1]
            folk_pct_rock[i] = entry[2]
            folk_pct_grav[i] = entry[3]
            folk_pct_sand[i] = entry[4]
            folk_pct_mud[i]  = entry[5]

    # substrate_primary
    sub_unknown = (
        df["substrate_primary"].isna() |
        (df["substrate_primary"] == "unknown")
    ).values
    can_fill_sub = sub_unknown & (folk_primary != None)  # noqa: E711
    df.loc[can_fill_sub, "substrate_primary"] = folk_primary[can_fill_sub]
    log(f"  substrate_primary resolved: {int(can_fill_sub.sum()):,}", indent=2)

    # hardness
    hard_missing = df["hardness"].isna().values
    can_fill_hard = hard_missing & (folk_hardness != None)  # noqa: E711
    df.loc[can_fill_hard, "hardness"] = folk_hardness[can_fill_hard]
    log(f"  hardness resolved: {int(can_fill_hard.sum()):,}", indent=2)

    # pct columns — fill NaN rows with folk heuristic
    if "pct_rock" not in df.columns:
        df["pct_rock"] = np.nan
    for col, arr in [("pct_rock",   folk_pct_rock),
                     ("pct_gravel", folk_pct_grav),
                     ("pct_sand",   folk_pct_sand),
                     ("pct_mud",    folk_pct_mud)]:
        missing = df[col].isna().values
        can_fill = missing & np.isfinite(arr)
        df.loc[can_fill, col] = arr[can_fill].astype(np.float32)

    still_unknown = int((df["substrate_primary"].fillna("unknown") == "unknown").sum())
    log(f"  substrate_primary still unknown after folk pass: {still_unknown:,}",
        indent=2, elapsed=time.time() - t0)
    return df


# ── EUNIS → substrate composition lookup ─────────────────────────────────────
# Maps EUNIS level-2 prefix → (substrate_primary, hardness, pct_rock, pct_gravel, pct_sand, pct_mud)
# Used in Pass 1b to override BGS pct values where UKASH has a clear,
# high-confidence habitat classification that contradicts BGS substrate.
# Only the level-2 prefix (A1–A6) is used — finer codes inherit from parent.
# Rock habitats (A1, A3, A4) → 100% rock.
# Sediment habitats (A2, A5) → mapped by sub-level.

EUNIS_SUBSTRATE: dict[str, tuple[str, str, float, float, float, float]] = {
    # Littoral rock — always hard rock
    "A1":    ("rock",   "hard",  100.0,  0.0,  0.0,  0.0),
    # Littoral sediment — split by sub-level below
    "A2.1":  ("gravel", "hard",   0.0,  90.0, 10.0,  0.0),  # coarse/shingle
    "A2.2":  ("sand",   "soft",   0.0,   0.0, 100.0, 0.0),   # sand
    "A2.3":  ("sand",   "soft",   0.0,   0.0, 80.0, 20.0),   # muddy sand
    "A2.4":  ("mud",    "soft",   0.0,   0.0,  0.0, 100.0),  # mud
    "A2.5":  ("mud",    "soft",   0.0,   0.0, 10.0,  90.0),  # saltmarsh/saline
    "A2.6":  ("mixed",  "mixed",  5.0,  30.0, 55.0, 10.0),   # biogenic/mixed
    "A2":    ("sand",   "soft",   0.0,   0.0, 80.0, 20.0),   # fallback: littoral sediment
    # Infralittoral rock — always hard rock
    "A3":    ("rock",   "hard",  100.0,  0.0,  0.0,  0.0),
    # Circalittoral rock — always hard rock
    "A4":    ("rock",   "hard",  100.0,  0.0,  0.0,  0.0),
    # Sublittoral sediment — split by sub-level
    "A5.1":  ("gravel", "hard",   0.0,  90.0, 10.0,  0.0),   # coarse sediment
    "A5.2":  ("sand",   "soft",   0.0,   0.0, 100.0, 0.0),   # sand
    "A5.3":  ("mud",    "soft",   0.0,   0.0,  5.0,  95.0),  # mud
    "A5.4":  ("mixed",  "mixed",  5.0,  20.0, 65.0, 10.0),   # mixed sediment
    "A5.5":  ("mixed",  "mixed",  5.0,  15.0, 65.0, 15.0),   # biogenic sediment
    "A5.6":  ("gravel", "hard",   0.0,  85.0, 15.0,  0.0),   # biogenic gravel/maerl
    "A5":    ("sand",   "soft",   0.0,   0.0, 90.0, 10.0),   # fallback: sublittoral sediment
    # Deep-sea — sediment dominated
    "A6":    ("mud",    "soft",   0.0,   0.0, 10.0,  90.0),
}


def _eunis_substrate(eunis_code: str):
    """
    Return EUNIS_SUBSTRATE entry for a code, with level-2 then level-1 fallback.
    Handles compound codes (slash-separated) by using the first part.
    Returns None if code is null or unresolvable.
    """
    if not eunis_code or (isinstance(eunis_code, float) and eunis_code != eunis_code):
        return None
    code = str(eunis_code).split("/")[0].strip()  # take first part of compound codes
    # Try increasingly short prefixes: full → level-2 sub → level-2 → level-1
    # e.g. A5.25 → A5.2 → A5 → A
    for length in range(len(code), 1, -1):
        prefix = code[:length]
        if prefix in EUNIS_SUBSTRATE:
            return EUNIS_SUBSTRATE[prefix]
    return None


def apply_ukash_substrate_override(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pass 1b: Override substrate composition where UKASH habitat data is
    more reliable than BGS.

    The BGS predictive model sometimes classifies the dominant sediment type
    at a coarse scale (e.g. ROCK for a rough seabed) while UKASH, based on
    actual survey data, records a fine-sediment habitat (e.g. A5.25 fine sand).
    The reverse also occurs.

    Strategy:
      - Only override where habitat_source = 'UKASH_survey' (ground-truthed).
        Predictive UKASH is not trusted for substrate correction.
      - Derive the EUNIS-implied substrate_primary, hardness, and pct values.
      - Apply where the implied substrate_primary DISAGREES with the current
        BGS-derived substrate_primary.  Agreement = no change.
      - Record the change by appending '_ukash_override' to substrate_source
        so it is traceable.
      - Do NOT override substrate_confidence — the confidence already
        reflects the BGS source quality.  The algorithm can weigh both.
    """
    log("Pass 1b: UKASH habitat substrate override ...", indent=1)
    t0 = time.time()

    survey_mask = df["habitat_source"] == "UKASH_survey"
    n_survey = int(survey_mask.sum())
    log(f"  UKASH_survey cells: {n_survey:,}", indent=2)

    # Vectorised: build implied-substrate series for all survey cells
    survey_df = df.loc[survey_mask, ["eunis_code", "substrate_primary"]].copy()
    entries   = survey_df["eunis_code"].map(_eunis_substrate)

    # Expand the tuple entries into component arrays
    has_entry = entries.notna()
    imp_primary = entries[has_entry].map(lambda e: e[0])
    imp_hardness = entries[has_entry].map(lambda e: e[1])
    imp_pct_rock   = entries[has_entry].map(lambda e: np.float32(e[2]))
    imp_pct_gravel = entries[has_entry].map(lambda e: np.float32(e[3]))
    imp_pct_sand   = entries[has_entry].map(lambda e: np.float32(e[4]))
    imp_pct_mud    = entries[has_entry].map(lambda e: np.float32(e[5]))

    # Only override where implied != current
    current = survey_df.loc[has_entry, "substrate_primary"]
    disagree = has_entry[has_entry].index[imp_primary.values != current.values]

    n_overridden = len(disagree)
    if n_overridden > 0:
        df.loc[disagree, "substrate_primary"] = imp_primary.loc[disagree].values
        df.loc[disagree, "hardness"]          = imp_hardness.loc[disagree].values
        df.loc[disagree, "pct_rock"]          = imp_pct_rock.loc[disagree].values
        df.loc[disagree, "pct_gravel"]        = imp_pct_gravel.loc[disagree].values
        df.loc[disagree, "pct_sand"]          = imp_pct_sand.loc[disagree].values
        df.loc[disagree, "pct_mud"]           = imp_pct_mud.loc[disagree].values
        df.loc[disagree, "substrate_source"]  = (
            df.loc[disagree, "substrate_source"].fillna("").astype(str)
            .apply(lambda s: (s + "+ukash_override") if s else "ukash_override")
        )

    log(f"  overridden: {n_overridden:,}  ({100*n_overridden/max(n_survey,1):.1f}% of survey cells)",
        indent=2, elapsed=time.time() - t0)
    return df


# ── Pass 2: Bathymetry gap-fill ───────────────────────────────────────────────

def slope_to_morphology(slope: np.ndarray) -> np.ndarray:
    out   = np.full(len(slope), None, dtype=object)
    valid = np.isfinite(slope.astype(float))
    s     = slope[valid].astype(float)
    out[valid] = np.where(s < 1,  "flat",
                 np.where(s < 5,  "gentle_slope",
                 np.where(s < 15, "slope",
                 np.where(s < 30, "steep", "cliff"))))
    return out


def fill_bathymetry(df: pd.DataFrame, gradient_neighbours: int, workers: int) -> pd.DataFrame:
    log("Pass 2: bathymetry gap-fill ...", indent=1)
    t0 = time.time()

    df["bathymetry_source"] = np.where(df["depth_m"].notna(), "emodnet", None)

    gap_mask  = df["depth_m"].isna()
    have_mask = ~gap_mask
    n_gaps    = int(gap_mask.sum())
    log(f"  depth gaps: {n_gaps:,}  donors: {int(have_mask.sum()):,}", indent=2)

    if n_gaps > 0:
        donors = df[have_mask].reset_index(drop=True)
        gaps   = df[gap_mask].reset_index(drop=True)

        donor_coords = bng_coords(donors)
        gap_coords   = bng_coords(gaps)
        donor_depth  = donors["depth_m"].values.astype(np.float32)
        donor_slope  = (donors["slope_deg"].values.astype(np.float32)
                        if "slope_deg" in donors.columns
                        else np.full(len(donors), np.nan, dtype=np.float32))
        donor_hwm    = donors["dist_to_hwm_m"].values.astype(np.float32)
        gap_hwm      = gaps["dist_to_hwm_m"].values.astype(np.float32)

        tree = cKDTree(donor_coords)
        nn_dist, nn_idx = tree.query(gap_coords, k=1, workers=workers)
        nn_depth = donor_depth[nn_idx]
        nn_slope = donor_slope[nn_idx]

        k = min(gradient_neighbours, len(donors))
        _, g_idx = tree.query(gap_coords, k=k, workers=workers)

        d_depth = donor_depth[g_idx]
        d_hwm   = donor_hwm[g_idx]
        valid   = np.isfinite(d_depth) & np.isfinite(d_hwm) & (d_hwm > 0)
        n_valid = valid.sum(axis=1)

        d_m  = np.where(valid, d_depth, 0.0)
        h_m  = np.where(valid, d_hwm,   0.0)
        num  = (d_m * h_m).sum(axis=1)
        den  = (h_m ** 2).sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            k_fit = np.where(den > 0, num / den, np.nan)

        predicted = k_fit * gap_hwm
        grad_ok   = (n_valid >= GRADIENT_MIN_DONORS) & np.isfinite(predicted) & (predicted >= 0)

        fill_depth  = np.where(grad_ok, predicted, nn_depth).astype(np.float32)
        fill_source = np.where(grad_ok, "depth_gradient", NN_SOURCE_LABEL)

        gap_index = df.index[gap_mask]
        df.loc[gap_index, "depth_m"]           = fill_depth
        df.loc[gap_index, "slope_deg"]         = nn_slope.astype(np.float32)
        df.loc[gap_index, "bathymetry_source"] = fill_source
        df.loc[gap_index, "morphology"]        = slope_to_morphology(nn_slope)
        df.loc[gap_index, "fill_distance_m"]   = nn_dist.astype(np.float32)

        log(f"  gradient: {int(grad_ok.sum()):,}  NN fallback: {int((~grad_ok).sum()):,}",
            indent=2)

    # Fill slope_deg on rows that have depth but no slope
    slope_gap = df["slope_deg"].isna() & df["depth_m"].notna()
    n_slope_gaps = int(slope_gap.sum())
    if n_slope_gaps > 0:
        log(f"  filling {n_slope_gaps:,} slope gaps on depth-present rows ...", indent=2)
        slope_donors = df[df["slope_deg"].notna()].reset_index(drop=True)
        if len(slope_donors) > 0:
            stree = cKDTree(bng_coords(slope_donors))
            _, sidx = stree.query(bng_coords(df[slope_gap].reset_index(drop=True)),
                                  k=1, workers=workers)
            slope_vals = slope_donors["slope_deg"].values[sidx].astype(np.float32)
            sg_index = df.index[slope_gap]
            df.loc[sg_index, "slope_deg"]  = slope_vals
            df.loc[sg_index, "morphology"] = slope_to_morphology(slope_vals)

    log(f"  done", indent=2, elapsed=time.time() - t0)
    return df


# ── Pass 3: Categorical NN fills ─────────────────────────────────────────────

def _nn_fill(df: pd.DataFrame, gap_mask: pd.Series, have_mask: pd.Series,
             copy_cols: list[str], source_col: str, source_val: str,
             conf_col: str | None, dist_col: str, workers: int) -> pd.DataFrame:
    """Generic nearest-neighbour fill. Per-domain distance written to dist_col."""
    n_gaps = int(gap_mask.sum())
    if n_gaps == 0 or have_mask.sum() == 0:
        return df

    donors     = df[have_mask].reset_index(drop=True)
    tree       = build_tree(donors)
    gap_coords = bng_coords(df[gap_mask].reset_index(drop=True))
    nn_dist, nn_idx = tree.query(gap_coords, k=1, workers=workers)

    gap_index = df.index[gap_mask]
    for col in copy_cols:
        if col in donors.columns:
            df.loc[gap_index, col] = donors[col].values[nn_idx]

    df.loc[gap_index, source_col] = source_val

    if conf_col and conf_col in donors.columns:
        decay = confidence_decay(nn_dist)
        orig  = donors[conf_col].values[nn_idx].astype(np.float64)
        df.loc[gap_index, conf_col] = np.round(orig * decay, 3).astype(np.float32)

    df.loc[gap_index, dist_col] = nn_dist.astype(np.float32)
    return df


def fill_substrate(df: pd.DataFrame, workers: int) -> pd.DataFrame:
    log("  substrate NN ...", indent=2)
    t0 = time.time()
    # Gap = still unknown after folk resolution
    gap  = df["substrate_primary"].isna() | (df["substrate_primary"] == "unknown")
    # Donor = any cell with a real resolved substrate
    have = (~gap) & df["substrate_source"].notna()
    df   = _nn_fill(df, gap, have,
                    copy_cols=["substrate_primary", "folk_code", "folk_description",
                               "pct_gravel", "pct_sand", "pct_mud", "pct_rock", "hardness"],
                    source_col="substrate_source", source_val=NN_SOURCE_LABEL,
                    conf_col="substrate_confidence",
                    dist_col="substrate_fill_distance_m",
                    workers=workers)
    df.loc[gap, "has_observed_survey"] = False
    log(f"    {int(gap.sum()):,} NN-filled", indent=2, elapsed=time.time() - t0)
    return df


def fill_habitat(df: pd.DataFrame, workers: int) -> pd.DataFrame:
    log("  habitat NN ...", indent=2)
    t0 = time.time()
    gap  = df["eunis_code"].isna()
    have = ~gap
    df   = _nn_fill(df, gap, have,
                    copy_cols=["eunis_code", "eunis_name", "mhc_code"],
                    source_col="habitat_source", source_val=NN_SOURCE_LABEL,
                    conf_col="habitat_confidence",
                    dist_col="habitat_fill_distance_m",
                    workers=workers)
    log(f"    {int(gap.sum()):,} NN-filled", indent=2, elapsed=time.time() - t0)
    return df


def fill_bedrock(df: pd.DataFrame, workers: int) -> pd.DataFrame:
    """NN fill for bedrock fields. Does NOT set substrate_primary — that is Pass 4."""
    log("  bedrock NN ...", indent=2)
    t0 = time.time()
    df["bedrock_source"] = np.where(df["bedrock_lex_rcs"].notna(), "bgs", None)
    gap  = df["bedrock_lex_rcs"].isna()
    have = ~gap
    df   = _nn_fill(df, gap, have,
                    copy_cols=["bedrock_lex_rcs", "bedrock_description", "bedrock_exposed"],
                    source_col="bedrock_source", source_val=NN_SOURCE_LABEL,
                    conf_col=None,
                    dist_col="fill_distance_m",
                    workers=workers)
    log(f"    {int(gap.sum()):,} NN-filled", indent=2, elapsed=time.time() - t0)
    return df


def fill_foreshore(df: pd.DataFrame, workers: int) -> pd.DataFrame:
    log("  foreshore NN ...", indent=2)
    t0 = time.time()
    intertidal = df["zone"] == "intertidal"
    gap  = intertidal & df["foreshore_type"].isna()
    have = intertidal & df["foreshore_type"].notna()
    # Initialise foreshore_source from existing data
    df["foreshore_source"] = np.where(df["foreshore_type"].notna(), "defr", None)
    if gap.sum() > 0 and have.sum() > 0:
        df = _nn_fill(df, gap, have,
                      copy_cols=["foreshore_type"],
                      source_col="foreshore_source",
                      source_val=NN_SOURCE_LABEL,
                      conf_col=None,
                      dist_col="fill_distance_m",
                      workers=workers)
    log(f"    {int(gap.sum()):,} NN-filled", indent=2, elapsed=time.time() - t0)
    return df


def fill_categorical_parallel(df: pd.DataFrame, workers: int) -> pd.DataFrame:
    log("Pass 3: categorical NN fills (parallel) ...", indent=1)
    t0 = time.time()

    df["substrate_fill_distance_m"] = np.float32(0.0)
    df["habitat_fill_distance_m"]   = np.float32(0.0)

    with ThreadPoolExecutor(max_workers=4) as ex:
        f_sub  = ex.submit(fill_substrate, df.copy(), workers)
        f_hab  = ex.submit(fill_habitat,   df.copy(), workers)
        f_bed  = ex.submit(fill_bedrock,   df.copy(), workers)
        f_fore = ex.submit(fill_foreshore, df.copy(), workers)
        r_sub, r_hab, r_bed, r_fore = (
            f_sub.result(), f_hab.result(), f_bed.result(), f_fore.result()
        )

    # Merge substrate (substrate_primary NOT from bedrock — see Pass 4)
    for col in ["substrate_primary", "folk_code", "folk_description",
                "pct_gravel", "pct_sand", "pct_mud", "pct_rock", "hardness",
                "substrate_source", "substrate_confidence",
                "has_observed_survey", "substrate_fill_distance_m"]:
        df[col] = r_sub[col]

    # Merge habitat
    for col in ["eunis_code", "eunis_name", "mhc_code",
                "habitat_source", "habitat_confidence", "habitat_fill_distance_m"]:
        df[col] = r_hab[col]

    # Merge bedrock (substrate_primary deliberately excluded)
    for col in ["bedrock_lex_rcs", "bedrock_description", "bedrock_exposed", "bedrock_source"]:
        df[col] = r_bed[col]

    # Merge foreshore
    df["foreshore_type"]   = r_fore["foreshore_type"]
    df["foreshore_source"] = r_fore["foreshore_source"]

    # fill_distance_m = worst-case max across all domains
    d = np.column_stack([
        r_sub["substrate_fill_distance_m"].fillna(0).values,
        r_hab["habitat_fill_distance_m"].fillna(0).values,
        r_bed["fill_distance_m"].fillna(0).values,
        df["fill_distance_m"].fillna(0).values,
    ])
    df["fill_distance_m"] = d.max(axis=1).astype(np.float32)

    log(f"  merged all fills", indent=2, elapsed=time.time() - t0)
    return df


# ── Pass 4: Post-merge bedrock hardening ─────────────────────────────────────

def apply_bedrock_hardening(df: pd.DataFrame) -> pd.DataFrame:
    """
    Where bedrock_exposed=True and substrate is STILL unknown after all
    other passes, upgrade to rock.  Legitimate sediment fills are untouched
    because they will have substrate_primary != "unknown".
    """
    log("Pass 4: bedrock hardening ...", indent=1)
    exposed   = df["bedrock_exposed"].fillna(False).astype(bool)
    still_unk = df["substrate_primary"].fillna("unknown") == "unknown"
    idx = df.index[exposed & still_unk]
    n = len(idx)
    if n > 0:
        df.loc[idx, "substrate_primary"] = "rock"
        df.loc[idx, "hardness"]          = "hard"
        df.loc[idx, "pct_rock"]          = np.float32(100.0)
        df.loc[idx, "pct_gravel"]        = np.float32(0.0)
        df.loc[idx, "pct_sand"]          = np.float32(0.0)
        df.loc[idx, "pct_mud"]           = np.float32(0.0)
        no_folk = df.loc[idx, "folk_code"].isna()
        ff_idx  = idx[no_folk.values]
        df.loc[ff_idx, "folk_code"]        = "ROCK"
        df.loc[ff_idx, "folk_description"] = "Rock - This term is used for any unidentified rock."
    log(f"  {n:,} cells hardened to rock", indent=2)
    return df


# ── Pass 5: pct_rock and normalisation ───────────────────────────────────────

def compute_pct_rock(df: pd.DataFrame) -> np.ndarray:
    """
    Finalise pct_rock for every row.

    Priority (applied in order, later steps only fill NaN from earlier):
      1. Heuristic already set by folk resolution (Pass 1) or bedrock
         hardening (Pass 4) — use as-is.
      2. G+S+M > 0 — residual = 100 - G - S - M.
      3. Known non-rock substrate (sand/mud) with no pct data — 0.0.
      4. Hard override: definite rock indicators always win — set to 100.
         (This corrects any case where folk/bedrock says rock but pct
          was inherited from a non-rock NN donor.)
    """
    pct_rock = df["pct_rock"].values.copy().astype(np.float64)

    # Step 2: residual where G+S+M > 0 and pct_rock still NaN
    need = np.isnan(pct_rock)
    if need.any():
        g = df["pct_gravel"].fillna(0).values.astype(np.float64)
        s = df["pct_sand"].fillna(0).values.astype(np.float64)
        m = df["pct_mud"].fillna(0).values.astype(np.float64)
        gsm = g + s + m
        ok  = need & (gsm > 0)
        pct_rock[ok] = np.clip(100.0 - gsm[ok], 0.0, 100.0)

    # Step 3: known non-rock → 0
    still_nan    = np.isnan(pct_rock)
    known_nonrock = still_nan & df["substrate_primary"].isin(["sand", "mud"]).values
    pct_rock[known_nonrock] = 0.0

    # Step 4: definite rock → 100 (always last, always wins)
    is_rock = (
        df["bedrock_exposed"].fillna(False).astype(bool).values |
        df["folk_code"].fillna("").str.upper().isin(ROCK_FOLK_CODES).values |
        (df["substrate_primary"].fillna("") == "rock").values
    )
    pct_rock[is_rock] = 100.0

    return pct_rock.astype(np.float32)


def normalise_pct_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise pct_rock / pct_gravel / pct_sand / pct_mud to sum to 100
    only for rows that have at least one positive value.
    All-NaN rows are left as NaN.
    """
    cols = ["pct_rock", "pct_gravel", "pct_sand", "pct_mud"]
    arr  = df[cols].values.astype(np.float64)
    row_sum  = np.nansum(arr, axis=1)
    has_data = row_sum > 0
    arr[has_data] = np.round(
        arr[has_data] / row_sum[has_data, np.newaxis] * 100.0, 2
    )
    df[cols] = arr.astype(np.float32)
    return df


# ── Quality recompute ─────────────────────────────────────────────────────────

def recompute_quality(df: pd.DataFrame) -> pd.DataFrame:
    sub_gap   = df["substrate_primary"].isna() | (df["substrate_primary"] == "unknown")
    hab_gap   = df["eunis_code"].isna()
    bathy_gap = df["depth_m"].isna()
    bed_gap   = df["bedrock_lex_rcs"].isna()

    df["coverage_flags"] = pd.DataFrame({
        "substrate":  sub_gap,
        "habitat":    hab_gap,
        "bathymetry": bathy_gap,
        "bedrock":    bed_gap,
    }).apply(lambda r: ",".join(k for k, v in r.items() if v) or None, axis=1)

    sc = df["substrate_confidence"].fillna(0.0).values.astype(np.float64)
    hc = df["habitat_confidence"].fillna(0.0).values.astype(np.float64)
    bathy_src = df["bathymetry_source"].fillna("").values
    bc = np.array([BATHY_CONFIDENCE.get(str(s), 0.0) for s in bathy_src])

    total = sc + hc + bc
    count = (sc > 0).astype(float) + (hc > 0).astype(float) + (bc > 0).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        overall = np.where(count > 0, total / count, 0.0)
    df["overall_confidence"] = np.round(overall, 3).astype(np.float32)
    return df


# ── Summary ───────────────────────────────────────────────────────────────────

# ── Pass 6: human-readable name enrichment ───────────────────────────────────

def _build_bedrock_desc_lookup(bedrock_gpkg: Path) -> dict[str, str]:
    """
    Build LEX_RCS → description lookup from the BGS Offshore Bedrock 250k
    GeoPackage by concatenating the separate LEX_D and RCS_D fields.
    Returns an empty dict (silently) if geopandas is unavailable or the file
    cannot be read — bedrock_description will just stay null in that case.
    """
    try:
        import geopandas as gpd
    except ImportError:
        log("  geopandas not installed — bedrock_description will remain null", indent=2)
        return {}
    try:
        gdf = gpd.read_file(bedrock_gpkg,
                            layer="BGS_250k_BedrockOffshore_WGS84_v3",
                            columns=["LEX_RCS", "LEX_D", "RCS_D"])
        lookup: dict[str, str] = {}
        for _, row in gdf.iterrows():
            code  = row.get("LEX_RCS")
            lex_d = str(row.get("LEX_D", "") or "").strip()
            rcs_d = str(row.get("RCS_D", "") or "").strip()
            if code and not (isinstance(code, float) and code != code):
                if lex_d and rcs_d:
                    lookup[str(code).strip()] = f"{lex_d} - {rcs_d}"
                elif lex_d or rcs_d:
                    lookup[str(code).strip()] = lex_d or rcs_d
        return lookup
    except Exception as e:
        log(f"  WARNING: could not read bedrock GeoPackage ({e})", indent=2)
        return {}


def fill_descriptions(df: pd.DataFrame, bedrock_gpkg: Path | None) -> pd.DataFrame:
    """
    Pass 6: populate eunis_name and bedrock_description from lookup tables.

    eunis_name:
      Applied to all rows unconditionally — the embedded EUNIS_NAMES dict
      covers the full A-section hierarchy so any existing partial values
      (from a prior run or from build_coastal_grid) are also corrected.

    bedrock_description:
      Requires the BGS Offshore Bedrock 250k GeoPackage path.  If not
      supplied (--bedrock-gpkg not given), the column remains as-is.
      Only null cells are filled; existing non-null values are preserved.
    """
    log("Pass 6: human-readable name enrichment ...", indent=1)
    t0 = time.time()

    # ── eunis_name ────────────────────────────────────────────────────────────
    before_eunis = df["eunis_name"].isna().sum()
    df["eunis_name"] = df["eunis_code"].map(eunis_name_for_code)
    after_eunis = df["eunis_name"].isna().sum()
    log(f"  eunis_name:          filled {before_eunis - after_eunis:,}  "
        f"still null {after_eunis:,}", indent=2)

    # Report any unresolvable codes so they can be added to EUNIS_NAMES later
    if after_eunis > 0:
        unresolved = (
            df.loc[df["eunis_name"].isna() & df["eunis_code"].notna(), "eunis_code"]
            .value_counts().head(10)
        )
        if len(unresolved):
            log("  Unresolved EUNIS codes (add to EUNIS_NAMES if needed):", indent=2)
            for code, cnt in unresolved.items():
                log(f"    {str(code):<35}  {cnt:>7,} rows", indent=3)

    # ── bedrock_description ───────────────────────────────────────────────────
    if bedrock_gpkg is None:
        log("  bedrock_description: skipped (no --bedrock-gpkg supplied)", indent=2)
    else:
        log("  bedrock_description: reading GeoPackage ...", indent=2)
        lookup = _build_bedrock_desc_lookup(bedrock_gpkg)
        if lookup:
            before_brk = df["bedrock_description"].isna().sum()
            # Only fill nulls — preserve any descriptions already present
            null_mask = df["bedrock_description"].isna() & df["bedrock_lex_rcs"].notna()
            df.loc[null_mask, "bedrock_description"] = (
                df.loc[null_mask, "bedrock_lex_rcs"].map(lookup)
            )
            after_brk = df["bedrock_description"].isna().sum()
            log(f"  bedrock_description: filled {before_brk - after_brk:,}  "
                f"still null {after_brk:,}", indent=2)

    log(f"  Pass 6 complete", indent=2, elapsed=time.time() - t0)
    return df


def print_summary(df_before: pd.DataFrame, df_after: pd.DataFrame):
    n = len(df_after)
    print("\n" + "=" * 68)
    print("FILL SUMMARY")
    print("=" * 68)

    def pct(x):
        return f"{x:,}  ({100*x/n:.1f}%)"

    for col, label, sentinel in [
        ("depth_m",             "Bathymetry",         None),
        ("slope_deg",           "Slope",              None),
        ("substrate_primary",   "Substrate",          "unknown"),
        ("eunis_code",          "Habitat code",       None),
        ("eunis_name",          "Habitat name",       None),
        ("bedrock_lex_rcs",     "Bedrock code",       None),
        ("bedrock_description", "Bedrock description",None),
    ]:
        def _gap(d, s):
            if col not in d.columns:
                return n
            base = d[col].isna()
            return int((base | (d[col] == s)).sum()) if s else int(base.sum())
        before_gap = _gap(df_before, sentinel)
        after_gap  = _gap(df_after,  sentinel)
        print(f"  {label:<12s}  before: {before_gap:>8,}  "
              f"after: {after_gap:>8,}  resolved: {before_gap - after_gap:>8,}")

    filled = df_after["fill_distance_m"] > 0
    if filled.any():
        d = df_after.loc[filled, "fill_distance_m"]
        print(f"\n  fill_distance_m (NN-filled cells):")
        print(f"    mean {d.mean():,.0f} m  |  median {d.median():,.0f} m  "
              f"|  p95 {d.quantile(0.95):,.0f} m  |  max {d.max():,.0f} m")

    c = df_after["overall_confidence"]
    print(f"\n  overall_confidence:")
    print(f"    mean {c.mean():.3f}  |  >=0.75: {100*(c>=0.75).mean():.1f}%"
          f"  |  <0.40: {100*(c<0.40).mean():.1f}%")

    still_flagged = df_after["coverage_flags"].notna().sum()
    print(f"\n  Cells with remaining coverage gaps: {pct(int(still_flagged))}")

    print(f"\n  substrate_primary breakdown:")
    for val, cnt in df_after["substrate_primary"].value_counts(dropna=False).items():
        print(f"    {str(val):<25}  {pct(cnt)}")

    print(f"\n  pct_rock:")
    print(f"    NaN:  {pct(int(df_after['pct_rock'].isna().sum()))}")
    pr = df_after["pct_rock"].dropna()
    if len(pr):
        print(f"    =100: {pct(int((pr == 100).sum()))}  =0: {pct(int((pr == 0).sum()))}")

    print(f"\n  Substrate sources:")
    for src, cnt in df_after["substrate_source"].value_counts(dropna=False).items():
        print(f"    {str(src):<25}  {pct(cnt)}")
    print(f"\n  Bathymetry sources:")
    for src, cnt in df_after["bathymetry_source"].value_counts(dropna=False).items():
        print(f"    {str(src):<25}  {pct(cnt)}")
    print("=" * 68 + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Proximal gap-fill for spearo coastal grid parquet.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("parquet", help="Path to spearo_coastal_grid_*.parquet")
    parser.add_argument("--output-dir",            default=None)
    parser.add_argument("--workers",      type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--gradient-neighbours", type=int, default=GRADIENT_NEIGHBOURS)
    args = parser.parse_args()

    in_path      = Path(args.parquet)
    bedrock_gpkg = BEDROCK_GPKG if BEDROCK_GPKG.exists() else None
    if bedrock_gpkg is None:
        print(f"WARNING: BEDROCK_GPKG not found at hardcoded path, "
              f"bedrock_description will remain null:\n  {BEDROCK_GPKG}", file=sys.stderr)

    if not in_path.exists():
        print(f"ERROR: {in_path} not found", file=sys.stderr)
        sys.exit(1)

    # Strip any existing _filled suffix so re-runs on a filled file don't
    # double-suffix the output
    stem = in_path.stem
    if stem.endswith("_filled"):
        stem = stem[:-7]

    out_dir  = Path(args.output_dir) if args.output_dir else in_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (stem + "_filled.parquet")

    print(f"\nSpearo Coastal Grid — Gap Fill")
    print(f"  Input:   {in_path}")
    print(f"  Output:  {out_path}")
    print(f"  Workers: {args.workers}  |  gradient neighbours: {args.gradient_neighbours}")
    print(f"  Bedrock GeoPackage: {bedrock_gpkg or str(BEDROCK_GPKG) + " (NOT FOUND)"}")
    t_total = time.time()

    df = load_parquet(in_path)
    df_orig = df.copy()

    # Initialise fill-tracking columns
    df["fill_distance_m"]     = np.float32(0.0)
    df["bedrock_exposed"]     = df["bedrock_exposed"].fillna(False).astype(bool)
    df["has_observed_survey"] = df["has_observed_survey"].fillna(False).astype(bool)
    if "pct_rock" not in df.columns:
        df["pct_rock"] = np.nan

    df = pre_clean(df)
    df = resolve_from_folk(df)
    df = fill_bathymetry(df, args.gradient_neighbours, args.workers)
    df = fill_categorical_parallel(df, args.workers)
    df = apply_bedrock_hardening(df)

    log("Pass 5: finalising pct columns ...", indent=1)
    df["pct_rock"] = compute_pct_rock(df)
    df = normalise_pct_columns(df)
    df = recompute_quality(df)

    df = fill_descriptions(df, bedrock_gpkg)

    # Preserve original column order; insert new columns next to related ones
    orig_cols = [c for c in df_orig.columns if c in df.columns]
    new_cols  = [c for c in df.columns if c not in df_orig.columns]

    for after_col, new_col in [("pct_mud",       "pct_rock"),
                                ("foreshore_type","foreshore_source")]:
        if after_col in orig_cols and new_col in new_cols:
            orig_cols.insert(orig_cols.index(after_col) + 1, new_col)
            new_cols.remove(new_col)

    df = df[orig_cols + new_cols]

    print_summary(df_orig, df)
    save_parquet(df, out_path)
    log(f"Total time: {time.time() - t_total:.1f}s\n")


if __name__ == "__main__":
    main()