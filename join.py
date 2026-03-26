"""
join.py — Spatial join orchestration for the landuse_ref pipeline.

Joins all acquired layers to building footprints and assigns land-use labels.

Public functions
----------------
run_joins(layers_dict) → GeoDataFrame  (fully labelled buildings in EPSG:4326)
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional

import geopandas as gpd
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from classify.assign import assign_label  # noqa: E402
from classify.lookup import OVERTURE_PRIORITY_ORDER  # noqa: E402
from classify.mixed_use import detect_mixed_use_from_pois  # noqa: E402
from config import CRS_PROJ  # noqa: E402

_OVERTURE_ORDER_MAP = {l0: i for i, l0 in enumerate(OVERTURE_PRIORITY_ORDER)}

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get(row: Optional[pd.Series], field: str, default=None):
    """Safely read a field from a Series or return *default*."""
    if row is None:
        return default
    val = row.get(field) if hasattr(row, "get") else getattr(row, field, default)
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return default
    return val


def _centroids(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Return a copy of *gdf* with centroid geometry, same index."""
    c = gdf[[]].copy()
    c["geometry"] = gdf.geometry.centroid
    return gpd.GeoDataFrame(c, geometry="geometry", crs=gdf.crs)


# ---------------------------------------------------------------------------
# Step 1 — CCCM centroid join
# ---------------------------------------------------------------------------

def _join_cccm(
    footprints: gpd.GeoDataFrame,
    cccm_sites: gpd.GeoDataFrame,
) -> Dict[int, pd.Series]:
    """
    Left-join footprint centroids within CCCM site polygons.
    Returns {footprint_index → cccm_sites row}.
    Where a centroid falls in multiple CCCM sites, the first match is kept.
    """
    if cccm_sites.empty:
        return {}
    centroids = _centroids(footprints)
    joined = gpd.sjoin(centroids, cccm_sites, how="left", predicate="within")
    matched = joined[joined["index_right"].notna()]

    result: Dict = {}
    for fp_i, row in matched.iterrows():
        if fp_i in result:
            continue  # keep first match only
        cccm_i = int(row["index_right"])
        result[fp_i] = cccm_sites.loc[cccm_i]
    return result


# ---------------------------------------------------------------------------
# Step 2 — OSM building overlap join
# ---------------------------------------------------------------------------

def _join_osm_buildings(
    footprints: gpd.GeoDataFrame,
    osm_buildings: gpd.GeoDataFrame,
    threshold: float = 0.3,
) -> Dict[int, pd.Series]:
    """
    Match each footprint to the OSM building with the largest intersection
    fraction (intersection_area / footprint_area).  Only matches where this
    fraction ≥ *threshold* are retained.

    Uses the OSM buildings' spatial index for candidate selection, then
    computes intersection areas explicitly.

    Returns {footprint_index → osm_buildings row}.
    """
    if osm_buildings.empty:
        return {}

    sindex = osm_buildings.sindex
    result: Dict = {}

    for fp_i, fp_row in footprints.iterrows():
        fp_geom = fp_row.geometry
        fp_area = fp_geom.area
        if fp_area <= 0:
            continue

        candidate_pos = list(sindex.query(fp_geom, predicate="intersects"))
        if not candidate_pos:
            continue

        best_frac = -1.0
        best_pos = None

        for pos in candidate_pos:
            osm_geom = osm_buildings.geometry.iloc[pos]
            try:
                inter_area = fp_geom.intersection(osm_geom).area
            except Exception:
                continue
            frac = inter_area / fp_area
            if frac > best_frac:
                best_frac = frac
                best_pos = pos

        if best_pos is not None and best_frac >= threshold:
            result[fp_i] = osm_buildings.iloc[best_pos]

    return result


# ---------------------------------------------------------------------------
# Step 3 — Overture Places within-footprint join
# ---------------------------------------------------------------------------

def _join_overture(
    footprints: gpd.GeoDataFrame,
    overture_places: gpd.GeoDataFrame,
) -> Dict[int, List[pd.Series]]:
    """
    Spatial join Overture place points within footprint polygons.
    Returns {footprint_index → [overture row, ...]}.
    """
    if overture_places.empty:
        return {}

    joined = gpd.sjoin(
        overture_places,
        footprints[["geometry"]],
        how="inner",
        predicate="within",
    )

    result: Dict = {}
    for fp_i, group in joined.groupby("index_right"):
        result[fp_i] = [overture_places.loc[i] for i in group.index if i in overture_places.index]
    return result


# ---------------------------------------------------------------------------
# Step 4 — OSM landuse centroid join (prefer smallest polygon)
# ---------------------------------------------------------------------------

def _join_osm_landuse(
    footprints: gpd.GeoDataFrame,
    osm_landuse: gpd.GeoDataFrame,
) -> Dict[int, pd.Series]:
    """
    Left-join footprint centroids within OSM landuse polygons.
    Where a centroid falls in multiple overlapping polygons, the one with
    the smallest area (most specific) is chosen.

    Returns {footprint_index → osm_landuse row}.
    """
    if osm_landuse.empty:
        return {}

    lu = osm_landuse.copy()
    lu["_area"] = lu.geometry.area

    centroids = _centroids(footprints)
    joined = gpd.sjoin(centroids, lu, how="left", predicate="within")
    matched = joined[joined["index_right"].notna()].copy()

    if matched.empty:
        return {}

    # Sort ascending by area so the smallest polygon comes first, then dedup
    matched = matched.sort_values("_area", ascending=True)
    matched = matched[~matched.index.duplicated(keep="first")]

    result: Dict = {}
    for fp_i, row in matched.iterrows():
        lu_i = int(row["index_right"])
        result[fp_i] = osm_landuse.loc[lu_i]
    return result


# ---------------------------------------------------------------------------
# Step 4b — OSM POI within-footprint join (+ 5 m exterior fallback)
#            Used as the Tier 3b labelling signal.
# ---------------------------------------------------------------------------

def _join_osm_pois(
    footprints: gpd.GeoDataFrame,
    osm_pois: gpd.GeoDataFrame,
    buffer_m: float = 5.0,
) -> Dict[int, List[pd.Series]]:
    """
    Collect OSM POI nodes associated with each footprint for Tier 3b labelling.

    Two passes are performed:

    1. Strict within join (``sjoin predicate='within'``): POI points that fall
       entirely inside the footprint polygon.
    2. Exterior buffer fallback: POIs not matched in pass 1 that fall within
       *buffer_m* metres of the footprint boundary.  This catches entrance
       nodes and POIs snapped to a building edge, which are common in OSM.

    Returns
    -------
    dict
        ``{footprint_index: [osm_pois row, ...]}``.  Only footprints with at
        least one associated POI are included.
    """
    if osm_pois.empty:
        return {}

    # Pass 1 — strict within
    joined_within = gpd.sjoin(
        osm_pois,
        footprints[["geometry"]],
        how="inner",
        predicate="within",
    )

    result: Dict[int, List[pd.Series]] = {}
    matched_poi_indices: set = set()

    for fp_i, group in joined_within.groupby("index_right"):
        valid = [i for i in group.index if i in osm_pois.index]
        if valid:
            result[int(fp_i)] = [osm_pois.loc[i] for i in valid]
            matched_poi_indices.update(valid)

    # Pass 2 — 5 m exterior buffer fallback for unmatched POIs
    unmatched = osm_pois[~osm_pois.index.isin(matched_poi_indices)]
    if not unmatched.empty:
        sindex = unmatched.sindex
        for fp_i, fp_row in footprints.iterrows():
            search_zone = fp_row.geometry.buffer(buffer_m)
            candidate_pos = list(sindex.query(search_zone, predicate="within"))
            if not candidate_pos:
                continue
            rows = [unmatched.iloc[pos] for pos in candidate_pos]
            if fp_i in result:
                result[fp_i].extend(rows)
            else:
                result[fp_i] = rows

    return result


# ---------------------------------------------------------------------------
# Step 5 — OSM POI 5 m buffer join (mixed-use POI signal)
# ---------------------------------------------------------------------------

def _join_pois_buffer(
    footprints: gpd.GeoDataFrame,
    osm_pois: gpd.GeoDataFrame,
    buffer_m: float = 5.0,
) -> Dict[int, gpd.GeoDataFrame]:
    """
    For each footprint, find OSM POI nodes within *buffer_m* metres of the
    building exterior using the POI spatial index for efficiency.

    Returns {footprint_index → GeoDataFrame of nearby POIs}.
    """
    if osm_pois.empty:
        return {}

    sindex = osm_pois.sindex
    result: Dict = {}

    for fp_i, fp_row in footprints.iterrows():
        search_zone = fp_row.geometry.buffer(buffer_m)
        candidate_pos = list(sindex.query(search_zone, predicate="within"))
        if candidate_pos:
            result[fp_i] = osm_pois.iloc[candidate_pos]

    return result


# ---------------------------------------------------------------------------
# Row builder
# ---------------------------------------------------------------------------

def _best_overture_match(matches: List[pd.Series]) -> Optional[pd.Series]:
    """Return the highest-priority Overture match by OVERTURE_PRIORITY_ORDER."""
    if not matches:
        return None

    def _priority(m: pd.Series) -> int:
        cat = _get(m, "category_primary")
        if cat is None:
            return len(_OVERTURE_ORDER_MAP)
        l0 = cat.split(".")[0]
        return _OVERTURE_ORDER_MAP.get(l0, len(_OVERTURE_ORDER_MAP))

    return min(matches, key=_priority)


def _build_row(
    fp_i,
    fp_row: pd.Series,
    label: dict,
    poi_signal: Optional[dict],
    cccm_m: Optional[pd.Series],
    osm_bm: Optional[pd.Series],
    ov_ms: List[pd.Series],
    lu_m: Optional[pd.Series],
) -> dict:
    """Assemble one output row from join results and the assigned label."""
    best_ov = _best_overture_match(ov_ms)

    return {
        # Preserve original geometry (still in CRS_PROJ at this point)
        "geometry": fp_row.geometry,
        # ---- Classification label ----
        "lu_class": label.get("lu_class"),
        "lu_subclass": label.get("lu_subclass"),
        "lu_tier": label.get("lu_tier"),
        "lu_source": label.get("lu_source"),
        "lu_confidence": label.get("lu_confidence"),
        "lu_class_components": label.get("lu_class_components"),  # list or None
        "geometry_source": label.get("geometry_source"),
        # ---- Mixed-use POI signal ----
        "lu_mixed_use_poi_signal": (
            poi_signal.get("mixed_use_poi_signal") if poi_signal else None
        ),
        # ---- OSM building provenance ----
        "osm_building_id": _get(osm_bm, "osm_id"),
        "osm_building_tag": _get(osm_bm, "building"),
        "osm_last_edit": _get(osm_bm, "last_edit"),
        # ---- Overture provenance ----
        "overture_category": _get(best_ov, "category_primary"),
        "overture_confidence": _get(best_ov, "confidence"),
        # ---- CCCM provenance ----
        "cccm_site_id": _get(cccm_m, "site_id"),
        "cccm_site_type": _get(cccm_m, "site_type"),
        "cccm_geometry_source": _get(cccm_m, "geometry_source"),
        # ---- OSM landuse provenance ----
        "osm_landuse_value": _get(lu_m, "landuse"),
    }


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def _print_summary(gdf: gpd.GeoDataFrame) -> None:
    n = len(gdf)
    print(f"\n{'=' * 52}")
    print(f"  Label assignment summary  ({n:,} buildings)")
    print(f"{'=' * 52}")

    print("\n  By tier:")
    tier_counts = (
        gdf["lu_tier"].astype(str).value_counts().sort_index()
    )
    for tier, count in tier_counts.items():
        pct = 100 * count / n if n else 0
        print(f"    Tier {tier:<5}  {count:>7,}  ({pct:.1f} %)")

    print("\n  By class:")
    class_counts = gdf["lu_class"].fillna("(unclassified)").value_counts()
    for cls, count in class_counts.items():
        pct = 100 * count / n if n else 0
        print(f"    {cls:<30}  {count:>7,}  ({pct:.1f} %)")

    n_mixed = int((gdf["lu_class"] == "Mixed Use").sum())
    n_unlabelled = int(gdf["lu_class"].isna().sum())
    print(f"\n  Mixed Use:     {n_mixed:>7,}")
    print(f"  Unclassified:  {n_unlabelled:>7,}")
    print(f"{'=' * 52}\n")


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

def run_joins(layers_dict: dict) -> gpd.GeoDataFrame:
    """
    Orchestrate spatial joins and label assignment for all building footprints.

    Parameters
    ----------
    layers_dict : dict
        Output of :func:`preprocess.align.align_layers`.  All GeoDataFrames
        must be in CRS_PROJ (EPSG:32636).

    Returns
    -------
    gpd.GeoDataFrame
        Fully labelled buildings reprojected to EPSG:4326.  One row per
        footprint.  See :mod:`export` for the final column schema.
    """
    footprints     = layers_dict["footprints"].copy()
    osm_buildings  = layers_dict.get("osm_buildings",  gpd.GeoDataFrame())
    osm_landuse    = layers_dict.get("osm_landuse",    gpd.GeoDataFrame())
    osm_pois       = layers_dict.get("osm_pois",       gpd.GeoDataFrame())
    overture_places = layers_dict.get("overture_places", gpd.GeoDataFrame())
    cccm_sites     = layers_dict.get("cccm_sites",     gpd.GeoDataFrame())

    n = len(footprints)
    print(f"\nProcessing {n:,} building footprints…")

    print("  [1/6] CCCM centroid join…")
    cccm_by_fp = _join_cccm(footprints, cccm_sites)
    print(f"        {len(cccm_by_fp):,} footprints matched to CCCM sites.")

    print("  [2/6] OSM building overlap join (threshold = 0.30)…")
    osm_bld_by_fp = _join_osm_buildings(footprints, osm_buildings)
    print(f"        {len(osm_bld_by_fp):,} footprints matched to OSM buildings.")

    print("  [3/6] Overture Places within-footprint join…")
    overture_by_fp = _join_overture(footprints, overture_places)
    print(f"        {len(overture_by_fp):,} footprints have Overture matches.")

    print("  [4/6] OSM POI within-footprint join (+ 5 m fallback)…")
    osm_poi_by_fp = _join_osm_pois(footprints, osm_pois)
    print(f"        {len(osm_poi_by_fp):,} footprints have OSM POI matches (Tier 3b).")

    print("  [5/6] OSM landuse centroid join…")
    landuse_by_fp = _join_osm_landuse(footprints, osm_landuse)
    print(f"        {len(landuse_by_fp):,} footprints matched to OSM landuse zones.")

    print("  [6/6] OSM POI 5 m buffer join (mixed-use signal)…")
    pois_by_fp = _join_pois_buffer(footprints, osm_pois)
    print(f"        {len(pois_by_fp):,} footprints have nearby POIs.")

    print("\n  Assigning labels…")
    _empty_pois = gpd.GeoDataFrame()
    rows = []

    for fp_i, fp_row in footprints.iterrows():
        ov_ms = overture_by_fp.get(fp_i, [])

        label = assign_label(
            building_row=fp_row,
            osm_building_match=osm_bld_by_fp.get(fp_i),
            overture_matches=ov_ms,
            osm_landuse_match=landuse_by_fp.get(fp_i),
            cccm_match=cccm_by_fp.get(fp_i),
            osm_poi_matches=osm_poi_by_fp.get(fp_i, []),
        )

        pois_near = pois_by_fp.get(fp_i, _empty_pois)
        poi_signal = detect_mixed_use_from_pois(label.get("lu_class"), pois_near)

        rows.append(
            _build_row(
                fp_i, fp_row, label, poi_signal,
                cccm_by_fp.get(fp_i),
                osm_bld_by_fp.get(fp_i),
                ov_ms,
                landuse_by_fp.get(fp_i),
            )
        )

    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=CRS_PROJ)
    gdf = gdf.to_crs("EPSG:4326")

    _print_summary(gdf)
    return gdf
