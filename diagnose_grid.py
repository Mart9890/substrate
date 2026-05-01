#!/usr/bin/env python3
"""
diagnose_grid.py  —  Data quality audit for spearo coastal grid parquet
=======================================================================
Analyses a filled or unfilled parquet and reports:

  1. Null / unknown / none counts per column
  2. substrate_primary="unknown" breakdown — what folk_code / bedrock data IS present
  3. Folk code → substrate_primary mapping coverage (codes that have no mapping)
  4. pct_rock=NaN or pct columns all-zero breakdown
  5. Hardness consistency (e.g. hard substrate with soft hardness label)
  6. Cells where substrate_primary disagrees with folk_code
  7. Per-zone coverage summary
  8. Cross-field consistency spot-checks

Usage:
  python diagnose_grid.py path/to/grid.parquet
  python diagnose_grid.py path/to/grid.parquet --sample 20
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np

SEP = "=" * 72

FOLK_TO_PRIMARY = {
    # Sand types
    "S": "sand", "sS": "sand", "mS": "sand", "gS": "sand",
    "SND": "sand", "MS": "sand", "GS": "sand",
    # Mud types
    "M": "mud", "sM": "mud", "gM": "mud",
    "MUD": "mud",
    # Gravel types
    "G": "gravel", "sG": "gravel", "mG": "gravel",
    "GVL": "gravel",
    # Mixed
    "(g)M": "mixed", "(g)S": "mixed", "mG": "mixed",
    "GVSND": "mixed",  # gravelly sand — could argue sand or mixed
    "GSND": "mixed",
    "SGM": "mixed", "GSM": "mixed",
    # Rock
    "ROCK": "rock", "R": "rock", "BDRK": "rock", "BEDROCK": "rock",
    "HR": "rock", "HARD ROCK": "rock",
    # Special
    "ROCK PLATFORM": "rock",
    "NODATA": "unknown",
}

HARD_SUBSTRATES = {"rock", "gravel"}
SOFT_SUBSTRATES = {"sand", "mud"}


def pct(n, total):
    return f"{n:>8,}  ({100 * n / total:.1f}%)" if total > 0 else f"{n:>8,}  (-)"


def section(title):
    print(f"\n{SEP}\n  {title}\n{SEP}")


def load(path):
    print(f"\nLoading {path} ...")
    df = pd.read_parquet(path)
    print(f"  {len(df):,} rows × {len(df.columns)} columns")
    return df


def audit_nulls(df):
    section("1. NULL / UNKNOWN / NONE counts per column")
    n = len(df)
    rows = []
    sentinel_values = {"unknown", "none", "null", "", "nan", "nodata"}
    for col in df.columns:
        null_count = int(df[col].isna().sum())
        if df[col].dtype == object:
            sentinel_count = int(
                df[col].str.lower().isin(sentinel_values).sum()
                if df[col].notna().any() else 0
            )
        else:
            sentinel_count = 0
        total_bad = null_count + sentinel_count
        if total_bad > 0:
            rows.append((col, null_count, sentinel_count, total_bad))

    rows.sort(key=lambda x: -x[3])
    print(f"\n  {'Column':<35} {'NULLs':>8}  {'Sentinels':>10}  {'Total':>8}")
    print(f"  {'-' * 35} {'-' * 8}  {'-' * 10}  {'-' * 8}")
    for col, nulls, sents, total in rows:
        print(f"  {col:<35} {pct(nulls, n)}  {sents:>10,}  {pct(total, n)}")


def audit_unknown_substrate(df):
    section("2. substrate_primary='unknown' — what info IS available?")
    unk = df[df["substrate_primary"].fillna("unknown") == "unknown"].copy()
    n_unk = len(unk)
    n_total = len(df)
    print(f"\n  Total unknown substrate: {pct(n_unk, n_total)}")
    if n_unk == 0:
        return

    # How many have a non-null, non-rock-platform folk_code that could map?
    has_folk = unk["folk_code"].notna() & ~unk["folk_code"].str.upper().isin(["", "NODATA"])
    print(f"\n  Of those, have a folk_code:            {pct(int(has_folk.sum()), n_unk)}")

    # Which folk codes appear in unknown-substrate rows?
    folk_counts = unk.loc[has_folk, "folk_code"].str.upper().value_counts()
    print(f"\n  Folk codes present in unknown-substrate rows (top 20):")
    for code, cnt in folk_counts.head(20).items():
        mapped = FOLK_TO_PRIMARY.get(code, "— NO MAPPING —")
        print(f"    {code:<20}  {cnt:>7,}  → would map to: {mapped}")

    # How many have bedrock that could resolve them?
    has_bedrock = unk["bedrock_exposed"].fillna(False).astype(bool)
    print(f"\n  Have bedrock_exposed=True:             {pct(int(has_bedrock.sum()), n_unk)}")

    has_lex = unk["bedrock_lex_rcs"].notna()
    print(f"  Have bedrock_lex_rcs (any):            {pct(int(has_lex.sum()), n_unk)}")

    # How many have hardness that contradicts "unknown"?
    hard_count = int((unk["hardness"] == "hard").sum())
    print(f"  Have hardness='hard' despite unknown:  {pct(hard_count, n_unk)}")


def audit_folk_mapping(df):
    section("3. Folk code → substrate_primary mapping coverage")
    has_folk = df["folk_code"].notna()
    print(f"\n  Rows with a folk_code: {pct(int(has_folk.sum()), len(df))}")

    all_codes = df.loc[has_folk, "folk_code"].str.upper().value_counts()
    unmapped = [(code, cnt) for code, cnt in all_codes.items()
                if FOLK_TO_PRIMARY.get(code) is None]

    print(f"\n  Folk codes with no mapping in FOLK_TO_PRIMARY ({len(unmapped)} codes):")
    for code, cnt in sorted(unmapped, key=lambda x: -x[1])[:30]:
        print(f"    {code:<30}  {cnt:>8,}")

    if not unmapped:
        print("    (all codes have a mapping)")


def audit_pct_columns(df):
    section("4. Percentage column audit")
    n = len(df)

    # pct_rock
    if "pct_rock" in df.columns:
        rock_null = int(df["pct_rock"].isna().sum())
        rock_100 = int((df["pct_rock"] == 100.0).sum())
        rock_0 = int((df["pct_rock"] == 0.0).sum())
        print(f"\n  pct_rock:")
        print(f"    NULL:   {pct(rock_null, n)}")
        print(f"    = 100:  {pct(rock_100, n)}")
        print(f"    = 0:    {pct(rock_0, n)}")

    # All-zero pct rows (where not null)
    pct_cols = [c for c in ["pct_gravel", "pct_sand", "pct_mud"] if c in df.columns]
    if pct_cols:
        all_present = df[pct_cols].notna().all(axis=1)
        all_zero = (df[pct_cols].fillna(-1) == 0).all(axis=1) & all_present
        print(f"\n  Rows with pct_gravel/sand/mud all present but all zero:")
        print(f"    {pct(int(all_zero.sum()), n)}")
        # Break down by substrate_primary
        if all_zero.any():
            print(f"    By substrate_primary:")
            for sub, cnt in df.loc[all_zero, "substrate_primary"].value_counts().items():
                print(f"      {sub:<15}  {cnt:>8,}")


def audit_hardness_consistency(df):
    section("5. Hardness consistency")
    n = len(df)

    hard_sub_soft_hard = (
            df["substrate_primary"].isin(HARD_SUBSTRATES) &
            (df["hardness"] == "soft")
    )
    soft_sub_hard_hard = (
            df["substrate_primary"].isin(SOFT_SUBSTRATES) &
            (df["hardness"] == "hard")
    )
    print(f"\n  Hard substrate (rock/gravel) but hardness='soft':  {pct(int(hard_sub_soft_hard.sum()), n)}")
    print(f"  Soft substrate (sand/mud) but hardness='hard':     {pct(int(soft_sub_hard_hard.sum()), n)}")

    unk_sub_known_hard = (
            (df["substrate_primary"].fillna("unknown") == "unknown") &
            df["hardness"].isin(["hard", "soft"])
    )
    print(f"  Unknown substrate but known hardness (resolvable):  {pct(int(unk_sub_known_hard.sum()), n)}")
    if unk_sub_known_hard.any():
        print(f"    Breakdown:")
        for h, cnt in df.loc[unk_sub_known_hard, "hardness"].value_counts().items():
            print(f"      hardness={h:<8}  {cnt:>8,}")


def audit_folk_vs_primary(df):
    section("6. substrate_primary vs folk_code disagreement")
    n = len(df)
    has_both = df["folk_code"].notna() & df["substrate_primary"].notna()

    mismatches = []
    for _, row in df[has_both].iterrows():
        folk_upper = str(row["folk_code"]).upper()
        expected = FOLK_TO_PRIMARY.get(folk_upper)
        actual = row["substrate_primary"]
        if expected is not None and expected != actual and actual != "unknown":
            mismatches.append((folk_upper, actual, expected))

    print(f"\n  Rows with folk_code that maps to a different substrate_primary: {len(mismatches):,}")
    if mismatches:
        from collections import Counter
        counts = Counter(mismatches)
        print(f"  Top disagreements (folk → actual | expected):")
        for (folk, actual, expected), cnt in counts.most_common(15):
            print(f"    folk={folk:<20} actual={actual:<12} expected={expected:<12}  ×{cnt}")


def audit_zones(df):
    section("7. Per-zone coverage summary")
    for zone in ["intertidal", "nearshore", "offshore", "coastal"]:
        zdf = df[df["zone"] == zone]
        if len(zdf) == 0:
            continue
        nz = len(zdf)
        sub_ok = int((~zdf["substrate_primary"].isin(["unknown", "none"]) & zdf["substrate_primary"].notna()).sum())
        hab_ok = int(zdf["eunis_code"].notna().sum())
        dep_ok = int(zdf["depth_m"].notna().sum())
        unk_sub = int((zdf["substrate_primary"].fillna("unknown") == "unknown").sum())
        print(f"\n  zone={zone}  ({nz:,} cells)")
        print(f"    substrate known:  {pct(sub_ok, nz)}")
        print(f"    substrate unknown:{pct(unk_sub, nz)}")
        print(f"    habitat known:    {pct(hab_ok, nz)}")
        print(f"    depth known:      {pct(dep_ok, nz)}")


def audit_resolvable_unknowns(df):
    section("8. Resolvable unknowns — cells where we CAN derive substrate_primary")
    n = len(df)
    unk = df["substrate_primary"].fillna("unknown") == "unknown"

    # From folk_code
    folk_resolvable = unk & df["folk_code"].notna()
    folk_resolvable &= df["folk_code"].str.upper().map(
        lambda x: FOLK_TO_PRIMARY.get(x) not in (None, "unknown")
    )
    print(f"\n  Via folk_code mapping:            {pct(int(folk_resolvable.sum()), n)}")

    # From hardness=hard → at least rock or gravel
    hard_resolvable = unk & (df["hardness"] == "hard")
    print(f"  Via hardness='hard':              {pct(int(hard_resolvable.sum()), n)}")

    # From bedrock_exposed
    bedrock_resolvable = unk & df["bedrock_exposed"].fillna(False).astype(bool)
    print(f"  Via bedrock_exposed=True:         {pct(int(bedrock_resolvable.sum()), n)}")

    # pct columns resolvable: if pct_mud > 50 → mud, pct_sand > 70 → sand, etc
    if all(c in df.columns for c in ["pct_gravel", "pct_sand", "pct_mud"]):
        pct_g = df["pct_gravel"].fillna(0)
        pct_s = df["pct_sand"].fillna(0)
        pct_m = df["pct_mud"].fillna(0)
        gsm = pct_g + pct_s + pct_m
        pct_resolvable = unk & (gsm > 0)
        print(f"  Via pct_gravel/sand/mud > 0:      {pct(int(pct_resolvable.sum()), n)}")

    print(f"\n  (These overlap — fixing folk_code mapping first is most impactful)")


def main():
    parser = argparse.ArgumentParser(description="Diagnose coastal grid data quality.")
    parser.add_argument("parquet", help="Path to grid parquet file")
    parser.add_argument("--sample", type=int, default=0,
                        help="Print N example rows of unknown substrate")
    args = parser.parse_args()

    path = Path(args.parquet)
    if not path.exists():
        print(f"ERROR: {path} not found", file=sys.stderr)
        sys.exit(1)

    df = load(path)

    audit_nulls(df)
    audit_unknown_substrate(df)
    audit_folk_mapping(df)
    audit_pct_columns(df)
    audit_hardness_consistency(df)
    audit_folk_vs_primary(df)
    audit_zones(df)
    audit_resolvable_unknowns(df)

    if args.sample > 0:
        section(f"Sample: {args.sample} rows with unknown substrate_primary")
        unk = df[df["substrate_primary"].fillna("unknown") == "unknown"]
        cols = ["cell_id", "zone", "folk_code", "folk_description", "hardness",
                "bedrock_lex_rcs", "bedrock_exposed", "pct_gravel", "pct_sand", "pct_mud",
                "substrate_source", "eunis_code"]
        cols = [c for c in cols if c in unk.columns]
        print(unk[cols].head(args.sample).to_string(max_colwidth=40))

    print(f"\n{SEP}\n  Done\n{SEP}\n")


if __name__ == "__main__":
    main()