"""
classify/open_space.py — Open-space polygon layer builder.

Provides a single shared helper, ``build_open_space_gdf``, used by:

* ``join.py``   — to identify residential buildings that fall within open-space
                  polygons and reclassify them as ``Residential-Informal``
                  (spatial override step).
* ``export.py`` — to construct and write the ``open_space_polygons`` output
                  layer without duplicating the tag-filter logic.

Public API
----------
build_open_space_gdf(layers_dict) → gpd.GeoDataFrame
"""

import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Tag filter constants
# ---------------------------------------------------------------------------

# Tag values that qualify a landuse polygon as open space, keyed by OSM tag.
# Priority when a row matches multiple keys: natural > leisure > landuse.
OS_FILTER: dict = {
    "landuse": {
        "grass", "meadow", "forest", "cemetery", "recreation_ground",
        "village_green", "allotments", "orchard", "farmland", "farmyard",
        "basin", "reservoir", "flowerbed",
    },
    "leisure": {
        "park", "pitch", "playground", "nature_reserve", "garden",
        "common", "dog_park",
    },
    "natural": {
        "water", "wetland", "wood", "scrub", "heath", "grassland",
        "beach", "sand",
    },
}

# Mapping from individual tag values to five os_class categories.
OS_CLASS_MAP: dict = {
    # Vegetation
    "grass": "Vegetation",     "meadow": "Vegetation",    "forest": "Vegetation",
    "flowerbed": "Vegetation", "allotments": "Vegetation", "orchard": "Vegetation",
    "nature_reserve": "Vegetation", "garden": "Vegetation", "wood": "Vegetation",
    "scrub": "Vegetation",     "heath": "Vegetation",     "grassland": "Vegetation",
    "sand": "Vegetation",
    # Water
    "water": "Water",    "wetland": "Water", "basin": "Water",
    "reservoir": "Water",
    # Recreation
    "recreation_ground": "Recreation", "village_green": "Recreation",
    "park": "Recreation",  "pitch": "Recreation",  "playground": "Recreation",
    "common": "Recreation", "dog_park": "Recreation", "beach": "Recreation",
    # Agriculture
    "farmland": "Agriculture", "farmyard": "Agriculture",
    # Cemetery
    "cemetery": "Cemetery",
}


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

def build_open_space_gdf(layers_dict: dict) -> gpd.GeoDataFrame:
    """
    Filter ``osm_landuse`` to open-space features and assign os_class.

    Applies the tag priority rule natural > leisure > landuse: when a row
    matches multiple tag keys, the highest-priority key determines the
    assigned class and tag provenance.

    Parameters
    ----------
    layers_dict : dict
        Output of :func:`preprocess.align.align_layers`.  Expected to contain
        key ``"osm_landuse"`` with a GeoDataFrame in CRS_PROJ (AOI-specific projected CRS).

    Returns
    -------
    gpd.GeoDataFrame
        Filtered GeoDataFrame **in the same CRS as the input**, with columns:
        ``osm_id``, ``geometry``, ``os_class``, ``osm_tag_key``,
        ``osm_tag_value``.  Returns an empty GeoDataFrame with those columns
        if the input layer is missing or empty.
    """
    _COLS = ["osm_id", "geometry", "os_class", "osm_tag_key", "osm_tag_value"]

    raw = layers_dict.get("osm_landuse", gpd.GeoDataFrame())
    if raw.empty:
        return gpd.GeoDataFrame(columns=_COLS)

    tag_keys: list = []
    tag_vals: list = []
    keep: list = []

    for _, row in raw.iterrows():
        matched_key = None
        matched_val = None

        for key in ("natural", "leisure", "landuse"):
            if key not in raw.columns:
                continue
            val = row.get(key)
            if val is None or (isinstance(val, float) and pd.isna(val)):
                continue
            val = str(val).strip()
            if val in OS_FILTER.get(key, set()):
                matched_key = key
                matched_val = val
                break  # highest-priority match found; stop checking further keys

        tag_keys.append(matched_key)
        tag_vals.append(matched_val)
        keep.append(matched_key is not None)

    out = raw[keep].copy()
    if out.empty:
        return gpd.GeoDataFrame(columns=_COLS, crs=raw.crs)

    out["osm_tag_key"] = [k for k, flag in zip(tag_keys, keep) if flag]
    out["osm_tag_value"] = [v for v, flag in zip(tag_vals, keep) if flag]
    out["os_class"] = out["osm_tag_value"].map(OS_CLASS_MAP)

    for col in _COLS:
        if col not in out.columns:
            out[col] = None

    return out[_COLS].copy()