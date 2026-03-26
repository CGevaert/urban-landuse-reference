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

from config import CRS_PROJ, SCRATCH_GPKG  # noqa: E402

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


def _dedup_geometries(gdf: gpd.GeoDataFrame, layer_name: str) -> gpd.GeoDataFrame:
    """
    Remove rows with duplicate geometries.

    Coordinates are snapped to a 1 cm grid (grid_size=0.01 m in the
    projected CRS) before comparison so that floating-point noise does not
    prevent detection of logically identical geometries.
    """
    if gdf.empty:
        return gdf

    # Snap to 1 cm grid, then serialise to WKB for exact comparison
    rounded = shapely.set_precision(gdf.geometry.values, grid_size=0.01)
    wkb_keys = shapely.to_wkb(rounded)

    import pandas as pd
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
    2. Reproject every layer to ``config.CRS_PROJ`` (EPSG:32636, metres).
    3. Repair invalid polygon geometries with ``make_valid()``; log count.
    4. Remove exact duplicate geometries within each layer (1 cm precision).

    Returns
    -------
    dict[str, gpd.GeoDataFrame]
        Keys: ``footprints``, ``osm_buildings``, ``osm_landuse``,
        ``osm_pois``, ``overture_places``, ``cccm_sites``.
        All GeoDataFrames are in CRS_PROJ (EPSG:32636).
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
    footprints = gpd.read_file(str(footprints_path))

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
    # 2. Reproject all layers to CRS_PROJ
    # ------------------------------------------------------------------
    print(f"\nReprojecting all layers to {CRS_PROJ}…")
    for name, gdf in layers.items():
        if gdf.crs is None:
            warnings.warn(
                f"[{name}] has no CRS defined — assuming EPSG:4326 before reprojection."
            )
            gdf = gdf.set_crs("EPSG:4326")
        layers[name] = gdf.to_crs(CRS_PROJ)
        print(f"  {name}")

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
    print("\nAlign complete. Layer sizes (features in EPSG:32636):")
    for name, gdf in layers.items():
        print(f"  {name:<20} {len(gdf):>7,} features")

    return layers
