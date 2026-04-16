"""
preprocess/merge_footprints.py — Merge two building footprint datasets.

This is a standalone preprocessing tool that runs *before* the main run.py
pipeline.  Call it directly, verify the output in QGIS, then pass the result
to ``run.py --footprints``.

Usage
-----
  python preprocess/merge_footprints.py \\
      --path-base    data/input/buildings_base.gpkg \\
      --path-update  data/input/buildings_update.gpkg \\
      --aoi          data/input/juba_aoi.geojson \\
      --output       data/input/buildings_merged.gpkg \\
      --base-label   google_2023 \\
      --update-label ai_2025

The output is then passed to run.py via --footprints data/input/buildings_merged.gpkg.

Public functions
----------------
merge_building_footprints(path_base, path_update, aoi_path, output_path, *, ...) -> MergeResult
"""

import argparse
import sys
import warnings
from pathlib import Path
from typing import NamedTuple

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely.validation import make_valid

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import _read_vector_file  # noqa: E402


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------

class MergeResult(NamedTuple):
    """Return value of ``merge_building_footprints()``.

    Attributes
    ----------
    buildings : gpd.GeoDataFrame
        Merged, deduplicated, flagged building footprint layer.
    coverage_gap_zones : gpd.GeoDataFrame
        Concave hull polygons for each DBSCAN hotspot cluster of base-dataset-only
        orphan buildings.  Empty if no clusters were found.
    """

    buildings: gpd.GeoDataFrame
    coverage_gap_zones: gpd.GeoDataFrame


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _utm_crs_from_gdf(gdf: gpd.GeoDataFrame) -> str:
    """Return the best UTM projected CRS string for a GeoDataFrame.

    If the GDF is already in a UTM projected CRS (EPSG 32601–32660 for the
    northern hemisphere, 32701–32760 for the southern), that CRS is returned
    directly.  Otherwise the best UTM zone is derived from the centroid of the
    data reprojected to WGS 84.

    This mirrors the ``_choose_projected_crs`` logic in ``config.init()`` but
    operates on any GeoDataFrame, making it usable without calling
    ``config.init()`` first.
    """
    raw_crs = gdf.crs
    if raw_crs is not None and raw_crs.is_projected:
        epsg = raw_crs.to_epsg()
        if epsg is not None and (32601 <= epsg <= 32660 or 32701 <= epsg <= 32760):
            return f"EPSG:{epsg}"

    # Derive UTM zone from centroid in geographic coordinates
    if raw_crs is None or raw_crs.to_epsg() != 4326:
        geo_gdf = gdf.to_crs("EPSG:4326")
    else:
        geo_gdf = gdf
    centroid = geo_gdf.dissolve().geometry.iloc[0].centroid
    lon, lat = centroid.x, centroid.y
    zone = int((lon + 180.0) / 6.0) + 1
    epsg = 32600 + zone if lat >= 0 else 32700 + zone
    return f"EPSG:{epsg}"


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

def merge_building_footprints(
    path_base: str | Path,
    path_update: str | Path,
    aoi_path: str | Path,
    output_path: str | Path,
    *,
    min_area_m2: float = 5.0,
    overlap_threshold: float = 0.1,
    orphan_search_radius_m: float = 30.0,
    cluster_eps_m: float = 40.0,
    min_cluster_size: int = 5,
    max_area_m2: float = 5000.0,
    base_label: str = "base",
    update_label: str = "update",
) -> MergeResult:
    """
    Merge two building footprint datasets into a single deduplicated layer.

    The update dataset takes precedence where footprints overlap.  Footprints
    from the base dataset that do not overlap any update feature are retained as
    additions unless they are classified as isolated artefacts (orphans) by the
    DBSCAN cluster filter.

    Parameters
    ----------
    path_base : str | Path
        Path to the base (older) building footprints file.  Any vector format
        readable by geopandas is accepted (GeoPackage, Shapefile, GeoJSON, etc.).
    path_update : str | Path
        Path to the update (newer) building footprints file.  Takes spatial
        precedence over *path_base* wherever the two datasets overlap.
    aoi_path : str | Path
        Path to the Area of Interest polygon file (GeoJSON or GeoPackage).
        Both footprint datasets are clipped to the AOI before merging.
    output_path : str | Path
        Destination path for the merged GeoPackage.  The file is written with
        layer name ``"buildings_merged"``.
    min_area_m2 : float, optional
        Minimum footprint area in square metres.  Footprints smaller than this
        threshold are removed after reprojection to the UTM CRS.  Default 5.0.
    overlap_threshold : float, optional
        Intersection/area ratio in [0, 1] above which a base footprint is
        considered a duplicate of an update footprint and discarded.
        Default 0.1.
    orphan_search_radius_m : float, optional
        Search radius in metres used to identify base-only footprints that have
        no update building nearby, marking them as orphan candidates for the
        DBSCAN cluster filter.  Default 30.0.
    cluster_eps_m : float, optional
        Epsilon (neighbourhood radius) in metres for the DBSCAN cluster filter
        applied to orphan candidate footprints.  Footprints in clusters smaller
        than *min_cluster_size* are treated as noise.  Default 40.0.
    min_cluster_size : int, optional
        Minimum number of points (footprint centroids) for a DBSCAN group to be
        treated as a real hotspot cluster.  Default 5.
    max_area_m2 : float, optional
        Maximum footprint area in square metres.  Footprints larger than this
        threshold are flagged as ``"large"`` (probable non-building polygons).
        Default 5000.0.
    base_label : str, optional
        Short label written to the ``bf_dataset`` column for base buildings
        (e.g. ``"google_2023"``).  Default ``"base"``.
    update_label : str, optional
        Short label written to the ``bf_dataset`` column for new update buildings
        (e.g. ``"ai_2025"``).  Default ``"update"``.

    Returns
    -------
    MergeResult
        A named tuple with two fields:

        ``buildings`` : gpd.GeoDataFrame
            Merged, deduplicated, and flagged building footprint layer in the
            UTM projected CRS derived from the AOI.  Also written to
            *output_path* as layer ``"buildings_merged"``.
        ``coverage_gap_zones`` : gpd.GeoDataFrame
            Concave hull polygons for each DBSCAN hotspot cluster of base-only
            orphan buildings (columns: ``cluster_id``, ``building_count``,
            ``area_m2``, ``geometry``).  Empty GeoDataFrame if no clusters were
            found.  Also written to *output_path* as layer
            ``"coverage_gap_zones"``.
    """
    path_base = Path(path_base)
    path_update = Path(path_update)
    aoi_path = Path(aoi_path)

    # ------------------------------------------------------------------
    # 1. Load inputs
    # ------------------------------------------------------------------
    print("Loading inputs…")

    for desc, p in (
        (f"base footprints ({base_label})", path_base),
        (f"update footprints ({update_label})", path_update),
        ("AOI", aoi_path),
    ):
        if not p.exists():
            raise FileNotFoundError(
                f"{desc} file not found: {p}\n"
                "Provide the correct path and re-run merge_building_footprints()."
            )

    print(f"  base footprints ({base_label})    ← {path_base.name}")
    gdf_base = _read_vector_file(path_base)

    print(f"  update footprints ({update_label}) ← {path_update.name}")
    gdf_update = _read_vector_file(path_update)

    print(f"  AOI                               ← {aoi_path.name}")
    gdf_aoi = _read_vector_file(aoi_path)

    # ------------------------------------------------------------------
    # 2. Derive UTM CRS from AOI and reproject all layers
    # ------------------------------------------------------------------
    crs_proj = _utm_crs_from_gdf(gdf_aoi)
    print(f"\nProjected CRS: {crs_proj}")

    print("Reprojecting to projected CRS…")
    gdf_aoi = gdf_aoi.to_crs(crs_proj)
    gdf_base = gdf_base.to_crs(crs_proj)
    gdf_update = gdf_update.to_crs(crs_proj)

    # ------------------------------------------------------------------
    # 3. Clip building datasets to AOI
    # ------------------------------------------------------------------
    print("\nClipping to AOI…")
    aoi_geom = gdf_aoi.dissolve().geometry.iloc[0]

    n_base_pre = len(gdf_base)
    gdf_base = gdf_base.clip(aoi_geom).reset_index(drop=True)
    n_base_post = len(gdf_base)
    print(f"  base:   {n_base_pre:,} → {n_base_post:,} features after clip")

    n_update_pre = len(gdf_update)
    gdf_update = gdf_update.clip(aoi_geom).reset_index(drop=True)
    n_update_post = len(gdf_update)
    print(f"  update: {n_update_pre:,} → {n_update_post:,} features after clip")

    # ------------------------------------------------------------------
    # 4. Repair invalid geometries
    # ------------------------------------------------------------------
    print("\nValidating geometries…")

    for desc, gdf in ((base_label, gdf_base), (update_label, gdf_update)):
        invalid_mask = ~gdf.geometry.is_valid
        n_invalid = int(invalid_mask.sum())
        if n_invalid > 0:
            warnings.warn(
                f"[{desc}] {n_invalid:,} invalid geometries repaired with make_valid()."
            )
            gdf = gdf.copy()
            gdf.loc[invalid_mask, "geometry"] = (
                gdf.loc[invalid_mask, "geometry"].apply(make_valid)
            )
            if desc == base_label:
                gdf_base = gdf
            else:
                gdf_update = gdf

    # ------------------------------------------------------------------
    # 5. Add provenance columns
    # ------------------------------------------------------------------
    gdf_base["bf_source"] = "base"
    gdf_base["bf_dataset"] = base_label

    gdf_update["bf_source"] = "update_new"
    gdf_update["bf_dataset"] = update_label

    # ------------------------------------------------------------------
    # 6. Add placeholder columns for later pipeline stages
    # ------------------------------------------------------------------
    for gdf in (gdf_base, gdf_update):
        gdf["geom_flag"] = None
        gdf["overlap_partner_id"] = pd.array([pd.NA] * len(gdf), dtype="Int64")
        gdf["coverage_gap_cluster_id"] = pd.array([pd.NA] * len(gdf), dtype="Int64")
        gdf["needs_review"] = False

    # ------------------------------------------------------------------
    # Input summary
    # ------------------------------------------------------------------
    print(f"\nInput summary (after clip):")
    print(f"  {'base footprints (' + base_label + ')':<32} {n_base_post:>7,} features")
    print(f"  {'update footprints (' + update_label + ')':<32} {n_update_post:>7,} features")

    # ------------------------------------------------------------------
    # 7. Identify genuinely new update buildings via STRtree overlap filter
    # ------------------------------------------------------------------
    print(f"\nFiltering {update_label} buildings for genuinely new footprints…")

    # Build STRtree from base geometries (numpy array, not centroids)
    base_geoms = gdf_base.geometry.values
    tree_base = shapely.STRtree(base_geoms)

    is_new = np.zeros(len(gdf_update), dtype=bool)

    for i, geom_update in enumerate(gdf_update.geometry):
        candidate_idxs = tree_base.query(geom_update)
        if len(candidate_idxs) == 0:
            is_new[i] = True
            continue

        area_update = geom_update.area
        if area_update == 0.0:
            # Zero-area geometry: treat as duplicate to avoid division by zero
            continue

        max_overlap = 0.0
        for idx in candidate_idxs:
            intersection_area = geom_update.intersection(base_geoms[idx]).area
            ratio = intersection_area / area_update
            if ratio > max_overlap:
                max_overlap = ratio
                if max_overlap >= overlap_threshold:
                    break  # No need to check remaining candidates

        if max_overlap < overlap_threshold:
            is_new[i] = True

    new_update = gdf_update.loc[is_new].copy()
    n_new = int(is_new.sum())
    n_dropped = len(gdf_update) - n_new

    print(f"  base buildings ({base_label}):        {n_base_post:>7,}")
    print(f"  update buildings ({update_label}) evaluated: {n_update_post:>7,}")
    print(f"  new update buildings added:          {n_new:>7,}")
    print(f"  update duplicates dropped:           {n_dropped:>7,}")

    # ------------------------------------------------------------------
    # 8. Concatenate and assign clean bf_id
    # ------------------------------------------------------------------
    merged = pd.concat([gdf_base, new_update], ignore_index=True)
    merged = gpd.GeoDataFrame(merged, geometry="geometry", crs=crs_proj)
    merged["bf_id"] = merged.index  # 0-based integer, matches positional index

    # ------------------------------------------------------------------
    # 9. Remove small buildings (area < min_area_m2)
    # ------------------------------------------------------------------
    area = merged.geometry.area
    small_mask = area < min_area_m2
    n_small = int(small_mask.sum())
    if n_small > 0:
        print(f"\nRemoving {n_small:,} buildings smaller than {min_area_m2} m²…")
        merged = merged.loc[~small_mask].reset_index(drop=True)
        merged["bf_id"] = merged.index
    else:
        print(f"\nNo buildings smaller than {min_area_m2} m² found.")

    # ------------------------------------------------------------------
    # 10. Geometry flagging
    # ------------------------------------------------------------------
    print("\nFlagging geometry issues…")

    # Work on a fresh area series now that small buildings are removed
    area = merged.geometry.area

    # -- 10a. Invalid geometries -----------------------------------------
    invalid_mask = ~merged.geometry.is_valid
    n_invalid = int(invalid_mask.sum())
    if n_invalid > 0:
        warnings.warn(
            f"[merged] {n_invalid:,} invalid geometries repaired with make_valid()."
        )
        merged.loc[invalid_mask, "geometry"] = (
            merged.loc[invalid_mask, "geometry"].apply(make_valid)
        )
        # Refresh area after repair before compactness/size checks
        area = merged.geometry.area
    merged.loc[invalid_mask, "geom_flag"] = "invalid"

    # -- 10b. Elongated shapes (compactness = area / perimeter²) ---------
    perimeter = merged.geometry.length
    # Avoid division by zero for degenerate geometries
    safe_perimeter = perimeter.where(perimeter > 0, other=np.nan)
    compactness = area / (safe_perimeter ** 2)
    elongated_mask = compactness < 0.01
    merged["geom_flag"] = merged["geom_flag"].where(
        merged["geom_flag"].notna() | ~elongated_mask, "elongated"
    )

    # -- 10c. Oversized buildings (area > max_area_m2) -------------------
    large_mask = area > max_area_m2
    merged["geom_flag"] = merged["geom_flag"].where(
        merged["geom_flag"].notna() | ~large_mask, "large"
    )

    # -- 10d. Overlapping pairs within merged dataset --------------------
    geoms_merged = list(merged.geometry)
    tree_merged = shapely.STRtree(geoms_merged)
    bf_ids = merged["bf_id"].to_numpy()
    # Work with numpy arrays for flag and partner writes to avoid repeated
    # iloc/loc overhead inside the loop.
    geom_flag_arr = merged["geom_flag"].to_numpy(dtype=object)
    partner_arr = merged["overlap_partner_id"].to_numpy(dtype=object)

    for i, geom in enumerate(geoms_merged):
        own_area = area.iloc[i]
        if own_area == 0.0:
            continue

        candidate_idxs = tree_merged.query(geom)
        for j in candidate_idxs:
            if j == i:  # exclude self-comparison
                continue
            intersection_area = geom.intersection(geoms_merged[j]).area
            if intersection_area > 0.05 * own_area:
                if geom_flag_arr[i] is None or (isinstance(geom_flag_arr[i], float) and np.isnan(geom_flag_arr[i])):
                    geom_flag_arr[i] = "overlap"
                if geom_flag_arr[j] is None or (isinstance(geom_flag_arr[j], float) and np.isnan(geom_flag_arr[j])):
                    geom_flag_arr[j] = "overlap"
                if partner_arr[i] is pd.NA or partner_arr[i] is None:
                    partner_arr[i] = int(bf_ids[j])
                if partner_arr[j] is pd.NA or partner_arr[j] is None:
                    partner_arr[j] = int(bf_ids[i])
                break  # Only record the first partner per building

    merged["geom_flag"] = geom_flag_arr
    merged["overlap_partner_id"] = pd.array(
        [pd.NA if (v is None or (isinstance(v, float) and np.isnan(v))) else v for v in partner_arr],
        dtype="Int64",
    )

    # -- 10e. Set needs_review where any flag is set ---------------------
    merged["needs_review"] = merged["geom_flag"].notna()

    # -- Summary ---------------------------------------------------------
    flag_counts = merged["geom_flag"].value_counts(dropna=True)
    n_needs_review = int(merged["needs_review"].sum())
    print(f"  Geometry flags:")
    if flag_counts.empty:
        print("    (none)")
    else:
        for flag_val, count in flag_counts.items():
            print(f"    {flag_val:<20} {count:>6,}")
    print(f"  Buildings needing review: {n_needs_review:,}")

    # ------------------------------------------------------------------
    # 11. Coverage gap detection — orphan clusters in base-only buildings
    # ------------------------------------------------------------------
    print("\nDetecting coverage gaps in base-only buildings…")

    # Empty result GDF used as default when no clusters are found
    coverage_gap_zones = gpd.GeoDataFrame(
        columns=["cluster_id", "building_count", "area_m2", "geometry"],
        geometry="geometry",
        crs=crs_proj,
    )

    # Phase A: Identify orphan base buildings --------------------------------
    # A base building is an "orphan" if no update building centroid falls within
    # orphan_search_radius_m of its centroid.

    mask_base = merged["bf_source"] == "base"
    mask_update = merged["bf_source"] == "update_new"

    # Integer index arrays aligned with the merged RangeIndex
    idx_base = merged.index[mask_base].to_numpy()
    # Centroids as shapely Point objects, in the same order as idx_base
    cents_base = list(merged.loc[mask_base, "geometry"].centroid)
    cents_update = list(merged.loc[mask_update, "geometry"].centroid)

    # Build STRtree from update footprint geometries; query with buffered base
    # centroid to get bbox candidates, then verify exact centroid distance.
    geoms_update = list(merged.loc[mask_update, "geometry"])
    tree_update = shapely.STRtree(geoms_update)

    is_orphan = np.zeros(len(idx_base), dtype=bool)
    for k, centroid in enumerate(cents_base):
        query_geom = centroid.buffer(orphan_search_radius_m)
        candidate_pos = tree_update.query(query_geom)
        if len(candidate_pos) == 0:
            is_orphan[k] = True
            continue
        has_nearby = any(
            centroid.distance(cents_update[j]) <= orphan_search_radius_m
            for j in candidate_pos
        )
        if not has_nearby:
            is_orphan[k] = True

    orphan_merged_idxs = idx_base[is_orphan]
    n_orphans = int(is_orphan.sum())
    print(f"  Base-only orphan buildings found: {n_orphans:,}")

    n_clusters = 0
    n_isolated = 0

    if n_orphans == 0:
        print("  No orphan buildings — skipping DBSCAN and concave hull steps.")
    else:
        # Phase B: DBSCAN clustering on orphan centroids ---------------------
        from sklearn.cluster import DBSCAN  # lazy import — not in base requirements
        orphan_cents = [cents_base[k] for k in np.where(is_orphan)[0]]
        coords = np.array([[c.x, c.y] for c in orphan_cents])

        db = DBSCAN(
            eps=cluster_eps_m, min_samples=min_cluster_size, metric="euclidean"
        ).fit(coords)
        labels = db.labels_  # shape (n_orphans,); -1 = noise

        # Write cluster labels back into the merged GDF.
        # Isolated noise points (label == -1) do not get needs_review set here;
        # they may already have it from geometry flags in step 10.
        for k, mi in enumerate(orphan_merged_idxs):
            lbl = int(labels[k])
            merged.at[mi, "coverage_gap_cluster_id"] = lbl
            if lbl >= 0:
                merged.at[mi, "needs_review"] = True

        unique_cluster_labels = sorted(int(l) for l in np.unique(labels) if l >= 0)
        n_clusters = len(unique_cluster_labels)
        n_isolated = int((labels == -1).sum())

        print(f"  Hotspot clusters found:           {n_clusters:,}")
        print(f"  Isolated orphans (label -1):      {n_isolated:,}")
        if n_clusters > 0:
            print("  Cluster sizes:")
            for lbl in unique_cluster_labels:
                size = int((labels == lbl).sum())
                print(f"    cluster {lbl:>3d}: {size:>5,} buildings")

        # Phase C: Concave hull polygon per cluster --------------------------
        if n_clusters > 0:
            hull_rows = []
            for lbl in unique_cluster_labels:
                member_pos = np.where(labels == lbl)[0]
                member_merged_idxs = orphan_merged_idxs[member_pos]
                member_geoms = list(merged.loc[member_merged_idxs, "geometry"])
                hull = shapely.concave_hull(shapely.unary_union(member_geoms), ratio=0.3)
                hull_rows.append({
                    "cluster_id": lbl,
                    "building_count": len(member_geoms),
                    "area_m2": float(hull.area),
                    "geometry": hull,
                })
            coverage_gap_zones = gpd.GeoDataFrame(
                hull_rows, geometry="geometry", crs=crs_proj
            )

    # ------------------------------------------------------------------
    # 12. Write output GeoPackage (both layers in EPSG:4326)
    # ------------------------------------------------------------------
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nWriting output to {output_path}…")

    # GeoPackage reserves the name 'fid' for its internal row identifier.
    # Drop it (and any other case variant) if it was carried in from the inputs.
    _reserved = {"fid"}
    def _drop_reserved(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        cols_to_drop = [c for c in gdf.columns if c.lower() in _reserved and c != gdf.geometry.name]
        if cols_to_drop:
            warnings.warn(f"Dropping reserved column(s) before GPKG write: {cols_to_drop}")
            return gdf.drop(columns=cols_to_drop)
        return gdf

    merged_geo = _drop_reserved(merged.to_crs("EPSG:4326"))
    merged_geo.to_file(str(output_path), layer="buildings_merged", driver="GPKG")

    zones_geo = _drop_reserved(coverage_gap_zones.to_crs("EPSG:4326"))
    zones_geo.to_file(str(output_path), layer="coverage_gap_zones", driver="GPKG", mode="a")

    print(f"  buildings_merged:    {len(merged_geo):,} features")
    print(f"  coverage_gap_zones:  {len(zones_geo):,} features")

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    _DIVIDER = "─" * 60
    n_needs_review_final = int(merged["needs_review"].sum())
    _area_label = str(min_area_m2).rstrip("0").rstrip(".")

    print(f"\n{_DIVIDER}")
    print("  merge_building_footprints — summary")
    print(_DIVIDER)
    print(f"  {'Input base buildings (' + base_label + '):':<36} {n_base_post:>7,}")
    print(f"  {'Input update buildings (' + update_label + '):':<36} {n_update_post:>7,}")
    print(f"  {'New update buildings added:':<36} {n_new:>7,}")
    print(f"  {'Buildings removed (< ' + _area_label + ' m²):':<36} {n_small:>7,}")
    print(f"  {'─── Geometry flags ───'}")
    if flag_counts.empty:
        print(f"  {'(none)'}")
    else:
        for flag_val, count in flag_counts.items():
            print(f"  {flag_val + ':':<36} {count:>7,}")
    print(f"  {'Buildings needing review:':<36} {n_needs_review_final:>7,}   (needs_review = True)")
    print(f"  {'─── Coverage gap detection ───'}")
    print(f"  {'Base-only orphan buildings:':<36} {n_orphans:>7,}")
    print(f"  {'Hotspot clusters:':<36} {n_clusters:>7,}")
    print(f"  {'Isolated orphans (noise):':<36} {n_isolated:>7,}")
    print(f"  {'Output:':<36} {output_path}")
    print(_DIVIDER)

    return MergeResult(buildings=merged, coverage_gap_zones=coverage_gap_zones)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import datetime
    import time

    _DIVIDER = "─" * 60

    class _Tee:
        """Write to both the original stdout and a log file simultaneously."""

        def __init__(self, log_path: Path):
            self._file = open(log_path, "w", encoding="utf-8")
            self._stdout = sys.stdout

        def write(self, text: str) -> None:
            self._stdout.write(text)
            self._file.write(text)

        def flush(self) -> None:
            self._stdout.flush()
            self._file.flush()

        def close(self) -> None:
            self._file.close()

    parser = argparse.ArgumentParser(
        prog="merge_footprints.py",
        description="Merge two building footprint datasets into a single deduplicated layer.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Required inputs
    parser.add_argument(
        "--path-base",
        required=True,
        metavar="PATH",
        help="Path to the base (older/reference) building footprints file (GPKG, SHP, GeoJSON, …).",
    )
    parser.add_argument(
        "--path-update",
        required=True,
        metavar="PATH",
        help="Path to the update (newer) building footprints file (GPKG, SHP, GeoJSON, …).",
    )
    parser.add_argument(
        "--aoi",
        required=True,
        metavar="PATH",
        help="Path to the Area of Interest polygon file (GeoJSON or GeoPackage).",
    )
    parser.add_argument(
        "--output",
        required=True,
        metavar="PATH",
        help="Destination GeoPackage path for the merged output.",
    )

    # Dataset labels
    parser.add_argument(
        "--base-label",
        default="base",
        metavar="STR",
        help=(
            "Short label written to bf_dataset for base buildings "
            "(e.g. 'google_2023'). Default: 'base'."
        ),
    )
    parser.add_argument(
        "--update-label",
        default="update",
        metavar="STR",
        help=(
            "Short label written to bf_dataset for new update buildings "
            "(e.g. 'ai_2025'). Default: 'update'."
        ),
    )

    # Optional tuning — mirrors keyword parameters of merge_building_footprints()
    parser.add_argument(
        "--min-area-m2",
        type=float,
        default=5.0,
        metavar="FLOAT",
        help="Minimum footprint area in m². Smaller buildings are dropped. Default: 5.0.",
    )
    parser.add_argument(
        "--overlap-threshold",
        type=float,
        default=0.1,
        metavar="FLOAT",
        help=(
            "Intersection/area ratio above which a base building is considered a "
            "duplicate of an update building and discarded. Range [0, 1]. Default: 0.1."
        ),
    )
    parser.add_argument(
        "--orphan-search-radius-m",
        type=float,
        default=30.0,
        metavar="FLOAT",
        help=(
            "Search radius in metres for identifying base buildings with no nearby "
            "update building (orphan detection). Default: 30.0."
        ),
    )
    parser.add_argument(
        "--cluster-eps-m",
        type=float,
        default=40.0,
        metavar="FLOAT",
        help="DBSCAN neighbourhood radius in metres for orphan cluster detection. Default: 40.0.",
    )
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=5,
        metavar="INT",
        help=(
            "Minimum number of orphan buildings to form a DBSCAN hotspot cluster. "
            "Smaller groups are treated as noise. Default: 5."
        ),
    )
    parser.add_argument(
        "--max-area-m2",
        type=float,
        default=5000.0,
        metavar="FLOAT",
        help="Maximum footprint area in m². Larger polygons are flagged as 'large'. Default: 5000.0.",
    )

    args = parser.parse_args()

    # Set up log file next to the output GeoPackage
    _output_path = Path(args.output)
    _output_path.parent.mkdir(parents=True, exist_ok=True)
    _timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    _log_path = _output_path.parent / f"{_output_path.stem}_{_timestamp}.log"
    _tee = _Tee(_log_path)
    sys.stdout = _tee

    t_start = time.perf_counter()
    print(_DIVIDER)
    print("  merge_footprints.py")
    print(f"  Log: {_log_path}")
    print(_DIVIDER)

    try:
        merge_building_footprints(
            path_base=args.path_base,
            path_update=args.path_update,
            aoi_path=args.aoi,
            output_path=args.output,
            min_area_m2=args.min_area_m2,
            overlap_threshold=args.overlap_threshold,
            orphan_search_radius_m=args.orphan_search_radius_m,
            cluster_eps_m=args.cluster_eps_m,
            min_cluster_size=args.min_cluster_size,
            max_area_m2=args.max_area_m2,
            base_label=args.base_label,
            update_label=args.update_label,
        )
    except Exception:
        import traceback
        traceback.print_exc()
        sys.stdout = _tee._stdout
        _tee.close()
        raise

    elapsed = time.perf_counter() - t_start
    m, s = divmod(int(elapsed), 60)
    elapsed_str = f"{m}m {s}s" if m else f"{s}s"
    print(_DIVIDER)
    print(f"  Done — total elapsed: {elapsed_str}")
    print(f"  Log saved: {_log_path}")
    print(_DIVIDER)

    sys.stdout = _tee._stdout
    _tee.close()
