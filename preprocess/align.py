"""
preprocess/align.py — Load, validate, and align all acquired layers.

Public functions
----------------
align_layers(footprints_path) → dict of cleaned, projected GeoDataFrames
"""

import sys
import warnings
from pathlib import Path
from typing import Dict

import geopandas as gpd
import shapely
from shapely.validation import make_valid

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import AOI_GEOM_PROJ, CRS_PROJ, SCRATCH_GPKG, _read_vector_file  # noqa: E402

# Layer names as stored in SCRATCH_GPKG
_GPKG_LAYERS = (
    "osm_buildings",
    "osm_landuse",
    "osm_pois",
    "overture_places",
    "cccm_sites",
)

# Geometry types that should be validated/repaired (not Points)
_POLYGON_TYPES = {"Polygon", "MultiPolygon", "GeometryCollection"}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _repair_invalid(gdf: gpd.GeoDataFrame, layer_name: str) -> gpd.GeoDataFrame:
    """
    Run shapely.validation.make_valid() on any invalid polygon geometries.
    Logs a warning with the count of repaired geometries; returns the GDF
    with all geometries valid.  Point layers are returned unchanged.
    """
    if gdf.empty:
        return gdf

    geom_types = gdf.geometry.geom_type.dropna().unique()
    if not any(t in _POLYGON_TYPES for t in geom_types):
        return gdf  # Point layer — nothing to repair

    invalid_mask = ~gdf.geometry.is_valid
    n_invalid = int(invalid_mask.sum())
    if n_invalid > 0:
        warnings.warn(
            f"[{layer_name}] {n_invalid:,} invalid geometries repaired with make_valid()."
        )
        gdf = gdf.copy()
        gdf.loc[invalid_mask, "geometry"] = (
            gdf.loc[invalid_mask, "geometry"].apply(make_valid)
        )
    return gdf


def _normalize_crs(gdf: gpd.GeoDataFrame, layer_name: str) -> gpd.GeoDataFrame:
    """
    Ensure the loaded layer has an appropriate CRS before reprojection.

    Some user-supplied vector files are mislabeled as geographic CRS but
    contain already-projected coordinates. In that case we preserve the
    projected coordinate values and avoid an invalid reprojection.
    Empty layers with no CRS are assigned the project CRS so they can still
    be reprojected safely.
    """
    if gdf.empty:
        if gdf.crs is None:
            return gdf.set_crs(CRS_PROJ)
        return gdf

    minx, miny, maxx, maxy = gdf.total_bounds
    if gdf.crs is None:
        if abs(minx) > 180 or abs(maxx) > 180 or abs(miny) > 90 or abs(maxy) > 90:
            warnings.warn(
                f"[{layer_name}] has no CRS and coordinates look projected "
                f"(bounds: {minx:.2f}, {miny:.2f}, {maxx:.2f}, {maxy:.2f}). "
                f"Assuming {CRS_PROJ}."
            )
            return gdf.set_crs(CRS_PROJ)

        warnings.warn(
            f"[{layer_name}] has no CRS defined — assuming EPSG:4326."
        )
        return gdf.set_crs("EPSG:4326")

    if gdf.crs.is_geographic and (
        abs(minx) > 180 or abs(maxx) > 180 or abs(miny) > 90 or abs(maxy) > 90
    ):
        warnings.warn(
            f"[{layer_name}] is labeled with geographic CRS {gdf.crs} but "
            f"coordinates are projected (bounds: {minx:.2f}, {miny:.2f}, "
            f"{maxx:.2f}, {maxy:.2f}). Overriding CRS to {CRS_PROJ}."
        )
        return gdf.set_crs(CRS_PROJ, allow_override=True)

    return gdf


def _dedup_geometries(gdf: gpd.GeoDataFrame, layer_name: str) -> gpd.GeoDataFrame:
    """
    Remove rows with duplicate geometries.

    Geometries are compared by exact WKB representation. For datasets where
    precision rounding can produce invalid coordinates, we attempt per-geometry
    rounding and fall back to the original geometry if necessary.
    """
    if gdf.empty:
        return gdf

    def _geom_key(geom):
        if geom is None or geom.is_empty:
            return b""
        try:
            rounded = shapely.set_precision(geom, grid_size=0.01)
            if rounded.is_empty or not rounded.is_valid:
                raise ValueError("rounded geometry invalid")
            return shapely.to_wkb(rounded)
        except Exception:
            return shapely.to_wkb(geom)

    import pandas as pd
    wkb_keys = [_geom_key(geom) for geom in gdf.geometry]
    dup_mask = pd.Series(wkb_keys).duplicated()
    n_dups = int(dup_mask.sum())
    if n_dups > 0:
        print(f"  [{layer_name}] Removed {n_dups:,} duplicate geometries.")
    return gdf.loc[~dup_mask.values].copy()


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

def align_layers(footprints_path: str | Path) -> Dict[str, gpd.GeoDataFrame]:
    """
    Load, validate, deduplicate, and reproject all acquired layers.

    This function is a pure in-memory transformation: the source files and
    SCRATCH_GPKG are never modified.

    Parameters
    ----------
    footprints_path : str | Path
        Path to the building footprints file (GeoPackage, Shapefile, etc.).
        Equivalent to ``config.FOOTPRINTS_PATH`` by convention, but accepted
        as a parameter so callers can substitute alternative inputs.

    Steps
    -----
    1. Load the footprint file and all five layers from SCRATCH_GPKG.
    2. Reproject every layer to ``config.CRS_PROJ`` (projected UTM CRS, metres).
    3. Repair invalid polygon geometries with ``make_valid()``; log count.
    4. Remove exact duplicate geometries within each layer (1 cm precision).

    Returns
    -------
    dict[str, gpd.GeoDataFrame]
        Keys: ``footprints``, ``osm_buildings``, ``osm_landuse``,
        ``osm_pois``, ``overture_places``, ``cccm_sites``.
        All GeoDataFrames are in CRS_PROJ (appropriate UTM zone for the AOI).
    """
    footprints_path = Path(footprints_path)
    if not footprints_path.exists():
        raise FileNotFoundError(
            f"Building footprints file not found: {footprints_path}\n"
            f"Place the file at that path or pass the correct path to align_layers()."
        )

    # ------------------------------------------------------------------
    # 1. Load layers
    # ------------------------------------------------------------------
    print("Loading layers…")

    print(f"  footprints  ← {footprints_path.name}")
    footprints = _read_vector_file(footprints_path)

    layers: Dict[str, gpd.GeoDataFrame] = {"footprints": footprints}

    gpkg_path = str(SCRATCH_GPKG)
    for layer_name in _GPKG_LAYERS:
        try:
            gdf = gpd.read_file(gpkg_path, layer=layer_name)
            print(f"  {layer_name:<20} ← {len(gdf):,} features")
        except Exception as exc:
            warnings.warn(
                f"Could not load layer '{layer_name}' from {SCRATCH_GPKG.name}: {exc}\n"
                "Substituting an empty GeoDataFrame."
            )
            gdf = gpd.GeoDataFrame(geometry=gpd.GeoSeries([], crs=None))
        layers[layer_name] = gdf

    # ------------------------------------------------------------------
    # 2. Normalise layer CRS and reproject all layers to CRS_PROJ
    # ------------------------------------------------------------------
    print(f"\nNormalizing layer CRS and reprojecting all layers to {CRS_PROJ}…")
    for name, gdf in layers.items():
        gdf = _normalize_crs(gdf, name)
        layers[name] = gdf.to_crs(CRS_PROJ)
        print(f"  {name}")

    # ------------------------------------------------------------------
    # 2b. Verify footprints overlap the AOI (fail fast before expensive steps)
    # ------------------------------------------------------------------
    fp = layers["footprints"]
    if not fp.empty and AOI_GEOM_PROJ is not None:
        fp_bbox = shapely.geometry.box(*fp.total_bounds)
        if not fp_bbox.intersects(AOI_GEOM_PROJ):
            fp_b = fp.total_bounds
            aoi_b = AOI_GEOM_PROJ.bounds
            raise ValueError(
                "\n[ERROR] Building footprints do not overlap the AOI.\n"
                f"  Footprints extent ({CRS_PROJ}): "
                f"W={fp_b[0]:.0f}  S={fp_b[1]:.0f}  E={fp_b[2]:.0f}  N={fp_b[3]:.0f}\n"
                f"  AOI extent       ({CRS_PROJ}): "
                f"W={aoi_b[0]:.0f}  S={aoi_b[1]:.0f}  E={aoi_b[2]:.0f}  N={aoi_b[3]:.0f}\n"
                "Ensure the footprints file covers the same geographic area as the AOI.\n"
                "If the footprints file has a wrong or missing CRS, fix the CRS before running."
            )

    # ------------------------------------------------------------------
    # 3. Repair invalid geometries (polygon layers only)
    # ------------------------------------------------------------------
    print("\nValidating geometries…")
    for name in list(layers):
        layers[name] = _repair_invalid(layers[name], name)

    # ------------------------------------------------------------------
    # 4. Remove duplicate geometries within each layer
    # ------------------------------------------------------------------
    print("\nRemoving duplicate geometries…")
    for name in list(layers):
        before = len(layers[name])
        layers[name] = _dedup_geometries(layers[name], name)
        after = len(layers[name])
        if before != after:
            pass  # message already printed by _dedup_geometries
        else:
            print(f"  [{name}] no duplicates found.")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\nAlign complete. Layer sizes (features in CRS_PROJ):")
    for name, gdf in layers.items():
        print(f"  {name:<20} {len(gdf):>7,} features")

    return layers
