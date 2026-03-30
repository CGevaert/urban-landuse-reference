"""
validate.py — Non-blocking pipeline validation checkpoints.

Public functions
----------------
validate_acquisition()                → dict[layer_name, feature_count]
validate_joins(labelled_gdf)          → dict (summary statistics)
validate_spatial_distribution(labelled_gdf) → None
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional

import geopandas as gpd
import numpy as np
from shapely import STRtree
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).parent))

import config  # noqa: E402
from config import AOI_GEOM, CRS_PROJ, SCRATCH_GPKG  # noqa: E402

_EXPECTED_LAYERS = [
    "osm_buildings",
    "osm_landuse",
    "osm_pois",
    "overture_places",
    "cccm_sites",
]

_SEP = "─" * 60
_WARN = "  [WARN]"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_layer(layer_name: str) -> Optional[gpd.GeoDataFrame]:
    """Attempt to load a layer from SCRATCH_GPKG; return None if absent."""
    try:
        return gpd.read_file(str(SCRATCH_GPKG), layer=layer_name)
    except Exception:
        return None


def _geom_types(gdf: gpd.GeoDataFrame) -> List[str]:
    """Return sorted list of distinct geometry type strings present in *gdf*."""
    return sorted(gdf.geometry.geom_type.dropna().unique().tolist())


def _null_or_empty_count(gdf: gpd.GeoDataFrame) -> int:
    """Count rows with null or empty (but non-null) geometry."""
    return int(gdf.geometry.isna().sum() + gdf.geometry[gdf.geometry.notna()].is_empty.sum())


def _bbox_in_geo(gdf: gpd.GeoDataFrame) -> Optional[object]:
    """
    Return the layer's bounding box as a shapely geometry in EPSG:4326.
    Reprojects if needed.
    """
    if gdf.empty:
        return None
    try:
        geo = gdf.to_crs("EPSG:4326") if (gdf.crs and gdf.crs.to_epsg() != 4326) else gdf
        return box(*geo.total_bounds)
    except Exception:
        return None


def _proj_centroid_to_geo(centroid_proj) -> tuple:
    """Convert a projected shapely Point to (lon, lat) in EPSG:4326."""
    pt_gdf = gpd.GeoDataFrame(geometry=[centroid_proj], crs=CRS_PROJ).to_crs("EPSG:4326")
    pt = pt_gdf.geometry.iloc[0]
    return round(pt.x, 6), round(pt.y, 6)


def _mean_nn_distance_m(geoms) -> Optional[float]:
    """
    Compute the mean nearest-neighbour distance (metres) among the centroids
    of *geoms* using shapely's STRtree.  Returns None if fewer than 2 features.
    """
    if len(geoms) < 2:
        return None
    pts = np.array([g.centroid for g in geoms])
    tree = STRtree(pts)
    dists = []
    for pt in pts:
        _, d = tree.query_nearest(pt, all_matches=False, return_distance=True, exclusive=True)
        if len(d) > 0:
            dists.append(d[0])
    return float(np.mean(dists)) if dists else None


# ---------------------------------------------------------------------------
# Checkpoint 1 — Acquisition
# ---------------------------------------------------------------------------

def validate_acquisition() -> Dict[str, int]:
    """
    Load every expected layer from SCRATCH_GPKG and print a formatted report.

    For each layer reports: feature count, geometry types, CRS, bounding box,
    and count of null/empty geometries.

    Raises
    ------
    ValueError
        If any expected layer is absent or has zero features, or if any
        layer's bounding box does not intersect config.AOI_GEOM.

    Returns
    -------
    dict[str, int]
        Mapping of layer name → feature count.
    """
    print(f"\n{_SEP}")
    print("  VALIDATE: Acquisition layers")
    print(_SEP)

    counts: Dict[str, int] = {}
    errors: List[str] = []

    for layer_name in _EXPECTED_LAYERS:
        # cccm_sites is optional — skip the check when CCCM acquisition is disabled
        if layer_name == "cccm_sites" and not config.USE_CCCM:
            print(f"\n  Layer: {layer_name}")
            print("    [SKIP] CCCM layer check disabled (--include-cccm not set).")
            continue

        print(f"\n  Layer: {layer_name}")

        if not SCRATCH_GPKG.exists():
            msg = f"SCRATCH_GPKG does not exist: {SCRATCH_GPKG}"
            print(f"    [ERROR] {msg}")
            errors.append(msg)
            counts[layer_name] = 0
            continue

        gdf = _load_layer(layer_name)

        if gdf is None:
            msg = f"Layer '{layer_name}' not found in {SCRATCH_GPKG.name}"
            print(f"    [ERROR] {msg}")
            errors.append(msg)
            counts[layer_name] = 0
            continue

        n = len(gdf)
        counts[layer_name] = n

        # Feature count
        print(f"    Features:        {n:,}")

        # Geometry types
        gtypes = _geom_types(gdf) if n > 0 else ["(empty)"]
        print(f"    Geometry types:  {', '.join(gtypes)}")

        # CRS
        crs_str = str(gdf.crs.to_epsg()) if (gdf.crs and gdf.crs.to_epsg()) else str(gdf.crs)
        print(f"    CRS:             EPSG:{crs_str}")

        # Bounding box
        if n > 0:
            bb = _bbox_in_geo(gdf)
            if bb is not None:
                minx, miny, maxx, maxy = bb.bounds
                print(
                    f"    Bbox (geo):      W={minx:.4f}  S={miny:.4f}  "
                    f"E={maxx:.4f}  N={maxy:.4f}"
                )
            else:
                print("    Bbox (geo):      (could not compute)")
                bb = None
        else:
            bb = None

        # Null/empty geometries
        n_bad = _null_or_empty_count(gdf) if n > 0 else 0
        if n_bad:
            print(f"    Null/empty geom: {n_bad:,}  {_WARN} non-zero null/empty count")
        else:
            print(f"    Null/empty geom: 0")

        # Hard checks
        if n == 0:
            if layer_name == "overture_places":
                print(
                    f"  {_WARN} Layer 'overture_places' is empty (0 features). "
                    "Sparse or zero Overture coverage is common in informal urban "
                    "areas where commercial POI data is limited. "
                    "The pipeline will continue using OSM and CCCM sources only."
                )
                continue
            msg = f"Layer '{layer_name}' is empty (0 features)."
            print(f"    [ERROR] {msg}")
            errors.append(msg)
            continue

        if bb is not None and not bb.intersects(AOI_GEOM):
            msg = (
                f"Layer '{layer_name}' bounding box does not intersect "
                f"AOI_GEOM — data may be for the wrong region."
            )
            print(f"    [ERROR] {msg}")
            errors.append(msg)

    print(f"\n{_SEP}")

    if errors:
        raise ValueError(
            "Acquisition validation failed with the following errors:\n"
            + "\n".join(f"  • {e}" for e in errors)
        )

    print("  Acquisition validation passed.\n")
    return counts


# ---------------------------------------------------------------------------
# Checkpoint 2 — Joins
# ---------------------------------------------------------------------------

def validate_joins(labelled_gdf: gpd.GeoDataFrame) -> Dict:
    """
    Validate the labelled GeoDataFrame produced by join.run_joins().

    Prints: tier breakdown, class breakdown, Mixed Use count, POI signal
    count, and approximate CCCM boundary count.

    Issues a printed warning (does not raise) if:
    * More than 70 % of buildings are unclassified (Tier 5).
    * No buildings were matched at Tier 1 (CCCM).
    * No buildings were classified as Mixed Use.

    Returns
    -------
    dict
        Summary statistics dictionary.
    """
    print(f"\n{_SEP}")
    print("  VALIDATE: Join results")
    print(_SEP)

    n = len(labelled_gdf)
    print(f"\n  Total footprints: {n:,}")

    summary: Dict = {"total": n}
    warnings_issued: List[str] = []

    # ------------------------------------------------------------------
    # Tier breakdown
    # ------------------------------------------------------------------
    print("\n  By tier:")
    tier_counts: Dict = {}
    tier_col = labelled_gdf["lu_tier"].astype(str)

    for tier_label in ["1", "2", "2+3", "3", "3b", "3+3b", "4", "5"]:
        cnt = int((tier_col == tier_label).sum())
        tier_counts[f"tier_{tier_label}"] = cnt
        pct = 100 * cnt / n if n else 0
        flag = ""
        if tier_label == "5" and pct > 70:
            flag = f"  {_WARN} > 70 % unclassified"
            warnings_issued.append(f"Tier 5 (unclassified) is {pct:.1f} % of all buildings.")
        if tier_label == "1" and cnt == 0:
            warnings_issued.append("No buildings matched at Tier 1 (CCCM) — expected if CCCM data covers this area.")
        print(f"    Tier {tier_label:<5}  {cnt:>7,}  ({pct:.1f} %){flag}")

    summary.update(tier_counts)

    # ------------------------------------------------------------------
    # Class breakdown
    # ------------------------------------------------------------------
    print("\n  By lu_class:")
    class_counts: Dict = {}
    for cls, cnt in labelled_gdf["lu_class"].fillna("(unclassified)").value_counts().items():
        pct = 100 * cnt / n if n else 0
        print(f"    {cls:<35}  {cnt:>7,}  ({pct:.1f} %)")
        class_counts[f"class_{cls}"] = cnt
    summary.update(class_counts)

    # ------------------------------------------------------------------
    # Mixed Use count
    # ------------------------------------------------------------------
    n_mixed = int((labelled_gdf["lu_class"] == "Mixed Use").sum())
    summary["mixed_use_count"] = n_mixed
    print(f"\n  Mixed Use:                  {n_mixed:>7,}")
    if n_mixed == 0:
        warnings_issued.append(
            "No buildings classified as Mixed Use — this may be correct for a low-density area "
            "but is unusual for an urban African context."
        )

    # ------------------------------------------------------------------
    # POI signal
    # ------------------------------------------------------------------
    if "lu_mixed_use_poi_signal" in labelled_gdf.columns:
        n_poi_signal = int(labelled_gdf["lu_mixed_use_poi_signal"].eq(True).sum())
        summary["poi_signal_count"] = n_poi_signal
        print(f"  POI mixed-use signal:       {n_poi_signal:>7,}")

    # ------------------------------------------------------------------
    # Approximate CCCM boundaries
    # ------------------------------------------------------------------
    if "cccm_geometry_source" in labelled_gdf.columns:
        n_approx = int(
            (labelled_gdf["cccm_geometry_source"] == "point_buffered").sum()
        )
        summary["cccm_point_buffered_count"] = n_approx
        if n_approx:
            print(
                f"  CCCM approx. boundaries:    {n_approx:>7,}  "
                f"(point_buffered — no polygon boundary available)"
            )

    # ------------------------------------------------------------------
    # Tier 1 == 0 warning (deferred so table prints cleanly first)
    # ------------------------------------------------------------------
    if tier_counts.get("tier_1", 0) == 0:
        print(f"\n{_WARN} Tier 1 count is 0 (no CCCM matches).")

    # ------------------------------------------------------------------
    # Summary warnings
    # ------------------------------------------------------------------
    if warnings_issued:
        print(f"\n  Warnings ({len(warnings_issued)}):")
        for w in warnings_issued:
            print(f"  {_WARN} {w}")
    else:
        print("\n  No warnings.")

    print(f"\n{_SEP}\n")
    return summary


# ---------------------------------------------------------------------------
# Checkpoint 3 — Spatial distribution
# ---------------------------------------------------------------------------

def validate_spatial_distribution(labelled_gdf: gpd.GeoDataFrame) -> None:
    """
    Analyse the spatial distribution of labelled buildings by lu_class.

    For each class with > 10 records, prints its centroid in geographic
    coordinates and mean nearest-neighbour distance between building
    centroids (a rough clustering indicator).

    Issues a warning if the Temporary Housing centroid is NOT more than
    5 km from the overall centroid of all labelled buildings — spatial
    isolation of TH clusters is expected; proximity to the general urban
    mass suggests possible mis-assignment.
    """
    print(f"\n{_SEP}")
    print("  VALIDATE: Spatial distribution")
    print(_SEP)

    # Work in projected CRS for distance computations
    if labelled_gdf.crs is None or labelled_gdf.crs.to_epsg() != int(CRS_PROJ.split(":")[1]):
        gdf = labelled_gdf.to_crs(CRS_PROJ)
    else:
        gdf = labelled_gdf.copy()

    labelled = gdf[gdf["lu_class"].notna()]
    if labelled.empty:
        print("  No labelled buildings — skipping spatial distribution check.\n")
        return

    # Overall centroid
    overall_centroid = labelled.geometry.unary_union.centroid
    oc_lon, oc_lat = _proj_centroid_to_geo(overall_centroid)
    print(f"\n  Overall centroid (all labelled):  lon={oc_lon}  lat={oc_lat}")

    print(f"\n  {'Class':<35}  {'N':>7}  {'Centroid (lon, lat)':<30}  {'Mean NN dist (m)'}")
    print(f"  {'─'*35}  {'─'*7}  {'─'*30}  {'─'*16}")

    th_centroid = None

    for cls, group in sorted(labelled.groupby("lu_class"), key=lambda x: x[0]):
        n = len(group)
        if n <= 10:
            continue

        # Class centroid
        cls_centroid = group.geometry.unary_union.centroid
        lon, lat = _proj_centroid_to_geo(cls_centroid)
        coord_str = f"({lon}, {lat})"

        # Mean NN distance
        nn_dist = _mean_nn_distance_m(group.geometry.values)
        nn_str = f"{nn_dist:>10.1f}" if nn_dist is not None else "         N/A"

        print(f"  {cls:<35}  {n:>7,}  {coord_str:<30}  {nn_str}")

        if cls == "Temporary Housing":
            th_centroid = cls_centroid

    # ------------------------------------------------------------------
    # Temporary Housing isolation check
    # ------------------------------------------------------------------
    print()
    if th_centroid is None:
        print(
            f"{_WARN} No Temporary Housing class found with > 10 buildings. "
            "Cannot perform isolation check."
        )
    else:
        dist_m = overall_centroid.distance(th_centroid)
        th_lon, th_lat = _proj_centroid_to_geo(th_centroid)
        print(
            f"  Temporary Housing centroid:  lon={th_lon}  lat={th_lat}\n"
            f"  Distance from overall centroid:  {dist_m:,.0f} m"
        )
        if dist_m < 5_000:
            print(
                f"{_WARN} Temporary Housing centroid is only {dist_m:,.0f} m from the "
                "overall urban centroid (< 5 km). TH sites are expected to be "
                "spatially peripheral. This may indicate mis-assignment — verify "
                "that CCCM/refugee-camp tagged features are genuinely displacement sites."
            )
        else:
            print(
                f"  ✓ Temporary Housing centroid is {dist_m:,.0f} m from the overall "
                "centroid (> 5 km) — expected spatial isolation confirmed."
            )

    print(f"\n{_SEP}\n")
