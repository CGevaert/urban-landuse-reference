"""
export.py — Write the labelled buildings to the reference GeoPackage.

Public functions
----------------
export_reference_dataset(labelled_gdf) → None
export_open_space_layer(layers_dict)   → None
"""

import sys
import uuid
from pathlib import Path

import geopandas as gpd
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from config import AOI_GEOM, CRS_GEO, OUTPUT_GPKG  # noqa: E402
from acquire.osm import _drop_gpkg_layer  # noqa: E402
from classify.lookup import LU_CODE_MAP, OSM_TAG_CLASS, OVERTURE_L0_CLASS  # noqa: E402
from classify.open_space import OS_CLASS_MAP, OS_FILTER, build_open_space_gdf  # noqa: E402

# ---------------------------------------------------------------------------
# Exact output column order
# ---------------------------------------------------------------------------

_OUTPUT_COLUMNS = [
    "footprint_id",
    "lu_code",
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
    "os_override_tag",
]

# Candidate column names that may hold the original footprint ID
_ID_CANDIDATES = [
    "id", "ID", "fid", "FID", "osm_id", "building_id",
    "OBJECTID", "objectid", "gid", "uid", "GlobalID",
]

_LAYER_NAME = "reference_buildings"
_UNMATCHED_LAYER_NAME = "unmatched_points"
_UNMATCHED_COLUMNS = ["footprint_id", "geometry", "osm_building_id", "osm_building_tag", "osm_landuse"]

_OPEN_SPACE_LAYER_NAME = "open_space_polygons"
_UNMATCHED_POIS_LAYER_NAME = "unmatched_pois"

# Tag key priority order used to resolve lu_class_candidate for OSM POIs
_POI_TAG_PRIORITY = ["amenity", "shop", "office", "leisure", "public_transport"]

# OS_FILTER and OS_CLASS_MAP live in classify/open_space.py; imported above.


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


def _print_final_summary(gdf: gpd.GeoDataFrame, n_unmatched: int = 0) -> None:
    n = len(gdf)
    print(f"\n{'=' * 56}")
    print(f"  Export summary  ({OUTPUT_GPKG.name})")
    print(f"{'=' * 56}")
    print(f"\n  reference_buildings (labelled):   {n:>7,}")
    print(f"  unmatched_points    (unlabelled): {n_unmatched:>7,}")

    print("\n  By lu_code:")
    for code, cnt in gdf["lu_code"].fillna("—").value_counts().sort_index().items():
        pct = 100 * cnt / n if n else 0
        print(f"    {code:<6}  {cnt:>7,}  ({pct:.1f} %)")

    print("\n  By lu_class:")
    for cls, cnt in gdf["lu_class"].fillna("(unclassified)").value_counts().items():
        pct = 100 * cnt / n if n else 0
        print(f"    {cls:<30}  {cnt:>7,}  ({pct:.1f} %)")

    print("\n  By lu_tier:")
    for tier, cnt in gdf["lu_tier"].value_counts().sort_index().items():
        pct = 100 * cnt / n if n else 0
        print(f"    Tier {str(tier):<7}  {cnt:>7,}  ({pct:.1f} %)")

    n_high = int((gdf["lu_confidence"] == "high").sum())
    print(f"\n  Confidence = high:  {n_high:>7,}  ({100*n_high/n:.1f} %)" if n else "")
    print(f"{'=' * 56}\n")


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

def export_reference_dataset(labelled_gdf: gpd.GeoDataFrame) -> None:
    """
    Write labelled buildings and unlabelled centroids to ``config.OUTPUT_GPKG``.

    Two layers are written to the same GeoPackage:

    ``reference_buildings``
        Rows where ``lu_class`` is not null.  Full 18-column schema in
        EPSG:4326.

    ``unmatched_points``
        Rows where ``lu_class`` is null, written as centroid Points in
        EPSG:4326 with reduced columns: footprint_id, geometry,
        osm_building_id, osm_building_tag, osm_landuse.  This layer is
        intended for later manual review or algorithmic matching to street
        segments or block-level units where no per-building source evidence
        was available.

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

    # Derive lu_code from lu_class via LU_CODE_MAP
    gdf["lu_code"] = gdf["lu_class"].map(LU_CODE_MAP)

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

    # Select and order columns; split into labelled and unlabelled
    gdf = gdf[_OUTPUT_COLUMNS]

    labelled = gdf[gdf["lu_class"].notna()].copy()
    unlabelled = gdf[gdf["lu_class"].isna()].copy()

    # ------------------------------------------------------------------
    # Layer 1 — reference_buildings (labelled footprints only)
    # ------------------------------------------------------------------
    OUTPUT_GPKG.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT_GPKG.exists():
        _drop_gpkg_layer(OUTPUT_GPKG, _LAYER_NAME)

    print(f"Writing {len(labelled):,} labelled features → {OUTPUT_GPKG} (layer: {_LAYER_NAME})…")
    labelled.to_file(str(OUTPUT_GPKG), driver="GPKG", layer=_LAYER_NAME, mode="a")

    # ------------------------------------------------------------------
    # Layer 2 — unmatched_points (unlabelled footprint centroids)
    #
    # Centroid geometry is computed in the projected CRS (EPSG:32636) for
    # accuracy then reprojected back to EPSG:4326 for storage.  This layer
    # is intended for later manual review or algorithmic matching to street
    # segments or block-level units where no per-building source evidence
    # was available.
    # ------------------------------------------------------------------
    if OUTPUT_GPKG.exists():
        _drop_gpkg_layer(OUTPUT_GPKG, _UNMATCHED_LAYER_NAME)

    if not unlabelled.empty:
        unmatched = unlabelled[
            [c for c in _UNMATCHED_COLUMNS if c != "geometry"]
        ].copy()
        centroids = (
            gpd.GeoSeries(
                unlabelled.to_crs("EPSG:32636").geometry.centroid,
                crs="EPSG:32636",
            ).to_crs("EPSG:4326")
        )
        unmatched = gpd.GeoDataFrame(unmatched, geometry=centroids, crs="EPSG:4326")
        unmatched = unmatched[_UNMATCHED_COLUMNS]
        print(f"Writing {len(unmatched):,} unlabelled centroids → {OUTPUT_GPKG} (layer: {_UNMATCHED_LAYER_NAME})…")
        unmatched.to_file(str(OUTPUT_GPKG), driver="GPKG", layer=_UNMATCHED_LAYER_NAME, mode="a")
    else:
        print(f"  No unlabelled buildings — layer '{_UNMATCHED_LAYER_NAME}' not written.")

    # ------------------------------------------------------------------
    # Summary CSV
    # ------------------------------------------------------------------
    summary_csv = OUTPUT_GPKG.parent / f"{OUTPUT_GPKG.stem}_summary.csv"
    summary_df = _build_summary(labelled)
    summary_df.to_csv(str(summary_csv), index=False)
    print(f"  Summary CSV → {summary_csv.name}")

    _print_final_summary(labelled, n_unmatched=len(unlabelled))


def export_open_space_layer(layers_dict: dict) -> None:
    """
    Build and write an open-space polygon layer from OSM landuse polygons.

    This function operates on the raw ``osm_landuse`` layer from
    *layers_dict* — it does not involve building footprints.  Features are
    filtered to a curated set of tag values covering vegetation, water,
    recreation, agriculture, and cemetery land cover, then clipped to the
    study-area AOI and written to ``config.OUTPUT_GPKG`` as the layer
    ``open_space_polygons`` in EPSG:4326.

    Columns written
    ---------------
    osm_id         : str   — OSM element identifier
    geometry       : Polygon / MultiPolygon — in EPSG:4326
    os_class       : str   — one of Vegetation / Water / Recreation /
                             Agriculture / Cemetery
    osm_tag_key    : str   — the OSM key that triggered inclusion
                             (natural > leisure > landuse by priority)
    osm_tag_value  : str   — the corresponding OSM tag value

    Parameters
    ----------
    layers_dict : dict
        Output of :func:`preprocess.align.align_layers`.  All GeoDataFrames
        must be in CRS_PROJ (EPSG:32636).
    """
    # Build the filtered open-space GDF using the shared helper (CRS_PROJ).
    gdf = build_open_space_gdf(layers_dict)

    if gdf.empty:
        print("  [open_space] osm_landuse is empty or has no qualifying features — layer not written.")
        return

    # Reproject to EPSG:4326 for clipping against AOI_GEOM and final storage.
    gdf = gdf.to_crs(CRS_GEO).copy()

    # ------------------------------------------------------------------
    # Clip to AOI_GEOM (EPSG:4326)
    # ------------------------------------------------------------------
    try:
        gdf = gdf.clip(mask=AOI_GEOM).copy()
    except Exception:
        gdf = gdf[gdf.geometry.intersects(AOI_GEOM)].copy()

    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()

    if gdf.empty:
        print("  [open_space] No features remain after AOI clip — layer not written.")
        return

    # ------------------------------------------------------------------
    # Select output columns and write
    # ------------------------------------------------------------------
    out_cols = ["osm_id", "geometry", "os_class", "osm_tag_key", "osm_tag_value"]
    for col in out_cols:
        if col not in gdf.columns:
            gdf[col] = None
    out = gdf[out_cols].copy()

    OUTPUT_GPKG.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT_GPKG.exists():
        _drop_gpkg_layer(OUTPUT_GPKG, _OPEN_SPACE_LAYER_NAME)

    out.to_file(str(OUTPUT_GPKG), driver="GPKG", layer=_OPEN_SPACE_LAYER_NAME, mode="a")
    print(
        f"  Written {len(out):,} open-space polygons → "
        f"{OUTPUT_GPKG.name} (layer: {_OPEN_SPACE_LAYER_NAME})"
    )


def export_unmatched_pois(
    unmatched_osm_pois_gdf: gpd.GeoDataFrame,
    unmatched_overture_gdf: gpd.GeoDataFrame,
) -> None:
    """
    Write POI features that were not matched to any building footprint to
    ``config.OUTPUT_GPKG`` as the layer ``unmatched_pois`` in EPSG:4326.

    Both OSM POI nodes and Overture Places records that were not spatially
    associated with any footprint during the join phase are included.
    They are normalised to a common schema and concatenated before writing.

    Columns written
    ---------------
    source            : str  — 'osm_poi' or 'overture'
    geometry          : Point — in EPSG:4326
    name              : str  — feature name (or None)
    lu_class_candidate: str  — indicative land-use class resolved from tag/category
    raw_tag           : str  — the original tag value used to resolve the class
    confidence        : float — Overture confidence score (None for OSM rows)

    Parameters
    ----------
    unmatched_osm_pois_gdf : gpd.GeoDataFrame
        OSM POI nodes not matched to any footprint, in EPSG:4326.
    unmatched_overture_gdf : gpd.GeoDataFrame
        Overture Places records not matched to any footprint, in EPSG:4326.
    """
    _SHARED_COLS = ["source", "geometry", "name", "lu_class_candidate", "raw_tag", "confidence"]

    parts = []

    # ------------------------------------------------------------------
    # OSM POI rows
    # ------------------------------------------------------------------
    if not unmatched_osm_pois_gdf.empty:
        osm = unmatched_osm_pois_gdf.copy()

        lu_candidates = []
        raw_tags = []

        for _, row in osm.iterrows():
            matched_class = None
            matched_raw = None
            for key in _POI_TAG_PRIORITY:
                if key not in osm.columns:
                    continue
                val = row.get(key)
                if val is None or (isinstance(val, float) and pd.isna(val)):
                    continue
                val = str(val).strip()
                entry = OSM_TAG_CLASS.get((key, val)) or OSM_TAG_CLASS.get((key, "*"))
                if entry:
                    matched_class = entry["lu_class"]
                    matched_raw = val
                    break
            lu_candidates.append(matched_class)
            raw_tags.append(matched_raw)

        osm["source"] = "osm_poi"
        osm["lu_class_candidate"] = lu_candidates
        osm["raw_tag"] = raw_tags
        osm["confidence"] = None
        osm["name"] = osm["name"] if "name" in osm.columns else None

        for col in _SHARED_COLS:
            if col not in osm.columns:
                osm[col] = None
        parts.append(osm[_SHARED_COLS])

    # ------------------------------------------------------------------
    # Overture rows
    # ------------------------------------------------------------------
    if not unmatched_overture_gdf.empty:
        ov = unmatched_overture_gdf.copy()

        lu_candidates_ov = []
        for _, row in ov.iterrows():
            cat = row.get("category_primary") if "category_primary" in ov.columns else None
            if cat is None or (isinstance(cat, float) and pd.isna(cat)):
                lu_candidates_ov.append(None)
            else:
                l0 = str(cat).split(".")[0]
                lu_candidates_ov.append(OVERTURE_L0_CLASS.get(l0))

        ov["source"] = "overture"
        ov["lu_class_candidate"] = lu_candidates_ov
        ov["raw_tag"] = ov["category_primary"] if "category_primary" in ov.columns else None
        ov["name"] = ov["name"] if "name" in ov.columns else None
        ov["confidence"] = pd.to_numeric(
            ov["confidence"] if "confidence" in ov.columns else None,
            errors="coerce",
        )

        for col in _SHARED_COLS:
            if col not in ov.columns:
                ov[col] = None
        parts.append(ov[_SHARED_COLS])

    # ------------------------------------------------------------------
    # Concatenate and write
    # ------------------------------------------------------------------
    if not parts:
        print(f"  [unmatched_pois] No unmatched POIs — layer not written.")
        return

    combined = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), geometry="geometry", crs="EPSG:4326")
    combined = combined[combined.geometry.notna() & ~combined.geometry.is_empty].copy()

    if combined.empty:
        print(f"  [unmatched_pois] No valid geometries after filtering — layer not written.")
        return

    OUTPUT_GPKG.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT_GPKG.exists():
        _drop_gpkg_layer(OUTPUT_GPKG, _UNMATCHED_POIS_LAYER_NAME)

    combined.to_file(str(OUTPUT_GPKG), driver="GPKG", layer=_UNMATCHED_POIS_LAYER_NAME, mode="a")

    n_osm = int((combined["source"] == "osm_poi").sum())
    n_ov  = int((combined["source"] == "overture").sum())
    print(
        f"  Written {len(combined):,} unmatched POIs → "
        f"{OUTPUT_GPKG.name} (layer: {_UNMATCHED_POIS_LAYER_NAME})  "
        f"[osm_poi: {n_osm:,}  overture: {n_ov:,}]"
    )
