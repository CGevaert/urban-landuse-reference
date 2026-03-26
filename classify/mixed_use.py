"""
classify/mixed_use.py — Supplementary mixed-use signal from nearby OSM POIs.

This module handles the specific case where OSM POI nodes co-located with a
building (within 5 m of its exterior) imply a different land-use class from
the building's own tags — a pattern common in dense urban areas where
ground-floor commercial activity occupies residential structures.

The output of detect_mixed_use_from_pois() is stored as metadata only and
does NOT override the tier assignment produced by assign.assign_label().

Public functions
----------------
detect_mixed_use_from_pois(building_geom, osm_pois_in_buffer) → dict | None
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional

import geopandas as gpd
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from classify.lookup import MIXED_USE_TRIGGER_PAIRS, OSM_TAG_CLASS  # noqa: E402

# Tag keys to probe on POI nodes, same priority order as assign.py
_TAG_PRIORITY = [
    "amenity",
    "shop",
    "office",
    "building",
    "landuse",
    "leisure",
    "public_transport",
    "military",
    "aeroway",
    "social_facility",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_field(row: pd.Series, field: str) -> Optional[str]:
    """Return a field from a pandas Series, or None if missing / NaN."""
    val = row.get(field) if hasattr(row, "get") else getattr(row, field, None)
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    s = str(val).strip()
    return s or None


def _resolve_poi_class(poi_row: pd.Series) -> Optional[str]:
    """
    Return the lu_class resolved for a single POI row by walking
    *_TAG_PRIORITY* against OSM_TAG_CLASS (exact match, then wildcard).
    Returns None if no matching entry is found.
    """
    for key in _TAG_PRIORITY:
        value = _get_field(poi_row, key)
        if value is None:
            continue
        entry = OSM_TAG_CLASS.get((key, value)) or OSM_TAG_CLASS.get((key, "*"))
        if entry:
            return entry["lu_class"]
    return None


def _triggers_mixed_use(classes: List[str]) -> bool:
    """
    Return True if any pair of class names in *classes* forms a
    MIXED_USE_TRIGGER_PAIR.
    """
    for i, a in enumerate(classes):
        for b in classes[i + 1:]:
            if frozenset({a, b}) in MIXED_USE_TRIGGER_PAIRS:
                return True
    return False


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

def detect_mixed_use_from_pois(
    building_class: Optional[str],
    osm_pois_in_buffer: gpd.GeoDataFrame,
) -> Optional[Dict]:
    """
    Detect a supplementary mixed-use signal from OSM POI nodes co-located
    with a building.

    This function is called *after* assign_label() and its output is stored
    in the ``lu_mixed_use_poi_signal`` column of the output dataset.  It does
    **not** change the tier assignment.

    Parameters
    ----------
    building_class : str or None
        The lu_class already assigned to this building by assign_label()
        (e.g. ``"Residential"``).  Used to check whether POI classes differ.
    osm_pois_in_buffer : gpd.GeoDataFrame
        OSM POI nodes within 5 m of this building's exterior (i.e. very
        close, suggesting co-location rather than mere proximity).  Expected
        columns match the schema from acquire/osm.py:
        osm_id, geometry, amenity, shop, office, leisure, tourism,
        public_transport, name, last_edit.

    Returns
    -------
    dict or None
        If a mixed-use POI signal is detected::

            {
                "mixed_use_poi_signal": True,
                "poi_classes": ["Commercial", "Residential"],  # distinct classes
            }

        Returns ``None`` if:
        * *osm_pois_in_buffer* is empty,
        * all POIs resolve to the same class as *building_class*, or
        * the resolved POI classes do not form any MIXED_USE_TRIGGER_PAIR.
    """
    if osm_pois_in_buffer is None or osm_pois_in_buffer.empty:
        return None

    # Resolve a class for every POI that has a recognisable tag
    poi_classes: List[str] = []
    for _, row in osm_pois_in_buffer.iterrows():
        cls = _resolve_poi_class(row)
        if cls is not None and cls not in poi_classes:
            poi_classes.append(cls)

    if not poi_classes:
        return None

    # Include the building's own class in the set to check for trigger pairs
    all_classes = list(poi_classes)
    if building_class is not None and building_class not in all_classes:
        all_classes = [building_class] + all_classes

    if len(all_classes) < 2:
        return None

    if not _triggers_mixed_use(all_classes):
        return None

    # At least one trigger pair exists — report the POI classes that fired it
    return {
        "mixed_use_poi_signal": True,
        "poi_classes": poi_classes,
    }
