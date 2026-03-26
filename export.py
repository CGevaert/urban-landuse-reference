"""
export.py — Write the labelled buildings to the reference GeoPackage.

Public functions
----------------
export_reference_dataset(labelled_gdf) → None
"""

import sys
import uuid
from pathlib import Path

import geopandas as gpd
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from config import OUTPUT_GPKG  # noqa: E402
from acquire.osm import _drop_gpkg_layer  # noqa: E402

# ---------------------------------------------------------------------------
# Exact output column order
# ---------------------------------------------------------------------------

_OUTPUT_COLUMNS = [
    "footprint_id",
    "geometry",
    "lu_class",
    "lu_subclass",
    "lu_tier",
    "lu_source",
    "lu_confidence",
    "lu_class_components",
    "lu_mixed_use_poi_signal",
    "osm_building_id",
    "osm_building_tag",
    "osm_last_edit",
    "overture_category",
    "overture_confidence",
    "cccm_site_id",
    "cccm_site_type",
    "cccm_geometry_source",
    "osm_landuse",
]

# Candidate column names that may hold the original footprint ID
_ID_CANDIDATES = [
    "id", "ID", "fid", "FID", "osm_id", "building_id",
    "OBJECTID", "objectid", "gid", "uid", "GlobalID",
]

_LAYER_NAME = "reference_buildings"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_footprint_id(gdf: gpd.GeoDataFrame) -> pd.Series:
    """
    Return a Series of footprint IDs (str).
    Looks for a common ID column first; falls back to auto-generated UUIDs.
    """
    for candidate in _ID_CANDIDATES:
        if candidate in gdf.columns:
            return gdf[candidate].astype(str)
    # Auto-generate
    return pd.Series(
        [str(uuid.uuid4()) for _ in range(len(gdf))],
        index=gdf.index,
        dtype=str,
    )


def _normalise_tier(val) -> str:
    """Convert lu_tier to a plain string (int → '1', '2', …; '2+3' stays)."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "5"
    return str(val)


def _normalise_source(val) -> str:
    """Convert lu_source, replacing '+' with '|' for schema compliance."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    return str(val).replace("+", "|")


def _normalise_components(val) -> str:
    """
    Convert lu_class_components (list | str | None) to a pipe-separated
    string, or None if not applicable.
    """
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    if isinstance(val, list):
        return "|".join(str(v) for v in val)
    return str(val)


def _build_summary(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """Build a tidy summary DataFrame for CSV export."""
    rows = []
    n = len(gdf)

    rows.append({"metric": "total_buildings", "value": n, "pct": 100.0})

    # By class
    for cls, cnt in gdf["lu_class"].fillna("(unclassified)").value_counts().items():
        rows.append({
            "metric": f"class_{cls}",
            "value": cnt,
            "pct": round(100 * cnt / n, 2) if n else 0,
        })

    # By tier
    for tier, cnt in gdf["lu_tier"].value_counts().sort_index().items():
        rows.append({
            "metric": f"tier_{tier}",
            "value": cnt,
            "pct": round(100 * cnt / n, 2) if n else 0,
        })

    # High confidence
    n_high = int((gdf["lu_confidence"] == "high").sum())
    rows.append({
        "metric": "confidence_high",
        "value": n_high,
        "pct": round(100 * n_high / n, 2) if n else 0,
    })

    # Unclassified
    n_null = int(gdf["lu_class"].isna().sum())
    rows.append({
        "metric": "unclassified",
        "value": n_null,
        "pct": round(100 * n_null / n, 2) if n else 0,
    })

    return pd.DataFrame(rows)


def _print_final_summary(gdf: gpd.GeoDataFrame) -> None:
    n = len(gdf)
    print(f"\n{'=' * 56}")
    print(f"  Export summary  ({n:,} buildings → {OUTPUT_GPKG.name})")
    print(f"{'=' * 56}")

    print("\n  By lu_class:")
    for cls, cnt in gdf["lu_class"].fillna("(unclassified)").value_counts().items():
        pct = 100 * cnt / n if n else 0
        print(f"    {cls:<30}  {cnt:>7,}  ({pct:.1f} %)")

    print("\n  By lu_tier:")
    for tier, cnt in gdf["lu_tier"].value_counts().sort_index().items():
        pct = 100 * cnt / n if n else 0
        print(f"    Tier {str(tier):<7}  {cnt:>7,}  ({pct:.1f} %)")

    n_high = int((gdf["lu_confidence"] == "high").sum())
    n_null = int(gdf["lu_class"].isna().sum())
    print(f"\n  Confidence = high:  {n_high:>7,}  ({100*n_high/n:.1f} %)" if n else "")
    print(f"  Unclassified:       {n_null:>7,}  ({100*n_null/n:.1f} %)" if n else "")
    print(f"{'=' * 56}\n")


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

def export_reference_dataset(labelled_gdf: gpd.GeoDataFrame) -> None:
    """
    Write *labelled_gdf* to ``config.OUTPUT_GPKG`` as layer
    ``reference_buildings`` in EPSG:4326, enforcing the exact output schema.

    Column schema
    -------------
    footprint_id, geometry, lu_class, lu_subclass, lu_tier, lu_source,
    lu_confidence, lu_class_components, lu_mixed_use_poi_signal,
    osm_building_id, osm_building_tag, osm_last_edit,
    overture_category, overture_confidence,
    cccm_site_id, cccm_site_type, cccm_geometry_source, osm_landuse.

    Also writes a companion CSV summary to
    ``data/{project_name}_landuse_reference_summary.csv``.

    Parameters
    ----------
    labelled_gdf : gpd.GeoDataFrame
        Output of :func:`join.run_joins`.  Must be in EPSG:4326.
    """
    gdf = labelled_gdf.copy()

    # Ensure EPSG:4326
    if gdf.crs is None or gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")

    # ------------------------------------------------------------------
    # Build / normalise required columns
    # ------------------------------------------------------------------

    gdf["footprint_id"] = _resolve_footprint_id(gdf)

    gdf["lu_tier"] = gdf["lu_tier"].apply(_normalise_tier)
    gdf["lu_source"] = gdf["lu_source"].apply(_normalise_source)
    gdf["lu_class_components"] = gdf["lu_class_components"].apply(
        _normalise_components
    )

    # Rename osm_landuse_value → osm_landuse to match schema
    if "osm_landuse_value" in gdf.columns and "osm_landuse" not in gdf.columns:
        gdf = gdf.rename(columns={"osm_landuse_value": "osm_landuse"})

    # Ensure every required column exists (fill missing with None)
    for col in _OUTPUT_COLUMNS:
        if col not in gdf.columns:
            gdf[col] = None

    # Cast lu_mixed_use_poi_signal to nullable bool
    gdf["lu_mixed_use_poi_signal"] = gdf["lu_mixed_use_poi_signal"].astype(
        pd.BooleanDtype()
    )

    # Cast overture_confidence to float
    gdf["overture_confidence"] = pd.to_numeric(
        gdf["overture_confidence"], errors="coerce"
    )

    # Select and order columns
    gdf = gdf[_OUTPUT_COLUMNS]

    # ------------------------------------------------------------------
    # Write GeoPackage layer (overwrite if exists)
    # ------------------------------------------------------------------
    OUTPUT_GPKG.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT_GPKG.exists():
        _drop_gpkg_layer(OUTPUT_GPKG, _LAYER_NAME)

    print(f"Writing {len(gdf):,} features → {OUTPUT_GPKG} (layer: {_LAYER_NAME})…")
    gdf.to_file(str(OUTPUT_GPKG), driver="GPKG", layer=_LAYER_NAME, mode="a")
    print("  GeoPackage written.")

    # ------------------------------------------------------------------
    # Summary CSV
    # ------------------------------------------------------------------
    summary_csv = OUTPUT_GPKG.parent / f"{OUTPUT_GPKG.stem}_summary.csv"
    summary_df = _build_summary(gdf)
    summary_df.to_csv(str(summary_csv), index=False)
    print(f"  Summary CSV → {summary_csv.name}")

    _print_final_summary(gdf)
