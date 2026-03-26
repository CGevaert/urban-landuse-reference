"""
classify/assign.py — Rule-based land-use label assignment for building footprints.

Public functions
----------------
assign_label(building_row, osm_building_match, overture_matches,
             osm_landuse_match, cccm_match, osm_poi_matches) → dict
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from classify.lookup import (  # noqa: E402
    MIXED_USE_TRIGGER_PAIRS,
    OSM_TAG_CLASS,
    OVERTURE_L0_CLASS,
    OVERTURE_PRIORITY_ORDER,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# OSM tag keys to probe, in priority order (highest → lowest)
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

# Tag keys used for Tier 3b (POI nodes — building/landuse tags excluded)
_POI_TAG_PRIORITY = [
    "amenity",
    "shop",
    "office",
    "leisure",
    "public_transport",
]

_UNCLASSIFIED: Dict[str, Any] = {
    "lu_class": None,
    "lu_subclass": None,
    "lu_tier": 5,
    "lu_source": None,
    "lu_confidence": None,
    "lu_class_components": None,
    "geometry_source": None,
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_field(row: pd.Series, field: str) -> Optional[str]:
    """Return a field from a pandas Series, or None if missing / NaN."""
    val = row.get(field) if hasattr(row, "get") else getattr(row, field, None)
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    return str(val).strip() or None


def _lookup_entry(key: str, value: str) -> Optional[Dict]:
    """
    Return the OSM_TAG_CLASS entry for (key, value).
    Checks the exact pair first, then the wildcard (key, "*").
    Returns None if neither is found.
    """
    return OSM_TAG_CLASS.get((key, value)) or OSM_TAG_CLASS.get((key, "*"))


def _resolve_osm_tags(row: pd.Series) -> List[Dict]:
    """
    Walk *_TAG_PRIORITY* on *row* and collect all matching OSM_TAG_CLASS
    entries.  Returns a list of entry dicts (may be empty; may contain
    duplicates by lu_class if multiple tags point to the same class).
    """
    entries = []
    for key in _TAG_PRIORITY:
        value = _get_field(row, key)
        if value is None:
            continue
        entry = _lookup_entry(key, value)
        if entry:
            entries.append(entry)
    return entries


def _distinct_classes(entries: List[Dict]) -> List[str]:
    """Return deduplicated lu_class values preserving first-seen order."""
    seen: List[str] = []
    for e in entries:
        cls = e["lu_class"]
        if cls not in seen:
            seen.append(cls)
    return seen


def _triggers_mixed_use(classes: List[str]) -> bool:
    """
    Return True if any pair of class names in *classes* appears in
    MIXED_USE_TRIGGER_PAIRS.
    """
    for i, a in enumerate(classes):
        for b in classes[i + 1:]:
            if frozenset({a, b}) in MIXED_USE_TRIGGER_PAIRS:
                return True
    return False


def _mixed_use_result(
    components: List[str],
    tier: Any,
    source: str,
    confidence: str,
    geometry_source: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "lu_class": "Mixed Use",
        "lu_subclass": None,
        "lu_tier": tier,
        "lu_source": source,
        "lu_confidence": confidence,
        "lu_class_components": sorted(components),
        "geometry_source": geometry_source,
    }


def _single_class_result(
    entry: Dict,
    tier: Any,
    source: str,
    geometry_source: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "lu_class": entry["lu_class"],
        "lu_subclass": entry["lu_subclass"],
        "lu_tier": tier,
        "lu_source": source,
        "lu_confidence": entry["lu_confidence"],
        "lu_class_components": None,
        "geometry_source": geometry_source,
    }


# ---------------------------------------------------------------------------
# Overture-specific helpers
# ---------------------------------------------------------------------------

def _overture_l0(match: pd.Series) -> Optional[str]:
    """Extract the L0 taxonomy string from an Overture match row."""
    cat = _get_field(match, "category_primary")
    if cat is None:
        return None
    return cat.split(".")[0]


def _best_overture_entry(
    l0_class_pairs: List[tuple],
) -> Optional[tuple]:
    """
    Given a list of (l0_category, lu_class) pairs, return the pair whose
    L0 category appears earliest in OVERTURE_PRIORITY_ORDER.
    Falls back to first-seen order for categories not in the list.
    """
    if not l0_class_pairs:
        return None
    order_map = {l0: i for i, l0 in enumerate(OVERTURE_PRIORITY_ORDER)}
    return min(
        l0_class_pairs,
        key=lambda p: order_map.get(p[0], len(OVERTURE_PRIORITY_ORDER)),
    )


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

def assign_label(
    building_row: pd.Series,
    osm_building_match: Optional[pd.Series],
    overture_matches: List[pd.Series],
    osm_landuse_match: Optional[pd.Series],
    cccm_match: Optional[pd.Series],
    osm_poi_matches: Optional[List[pd.Series]] = None,
) -> Dict[str, Any]:
    """
    Assign a land-use label to a single building footprint using a six-tier
    rule hierarchy.

    Parameters
    ----------
    building_row : pd.Series
        One row from the building footprints GeoDataFrame (not used for
        classification logic itself, reserved for caller context).
    osm_building_match : pd.Series or None
        The OSM building feature spatially joined to this footprint, if any.
    overture_matches : list of pd.Series
        All Overture Places records whose point geometry falls within this
        footprint.
    osm_landuse_match : pd.Series or None
        The OSM landuse polygon whose area contains this footprint's centroid.
    cccm_match : pd.Series or None
        The CCCM site polygon containing this footprint's centroid.
    osm_poi_matches : list of pd.Series or None
        OSM POI nodes that fall within the footprint polygon or within 5 m of
        its exterior.  Used for Tier 3b labelling.

    Returns
    -------
    dict
        Keys: lu_class (str|None), lu_subclass (str|None), lu_tier (int|str),
        lu_source (str|None), lu_confidence (str|None),
        lu_class_components (list|None), geometry_source (str|None).

    Tier hierarchy
    --------------
    1    CCCM site match   → Temporary Housing (authoritative ground-truth)
    2    OSM building tags → resolve class(es) from OSM_TAG_CLASS
    3    Overture Places   → resolve class(es) from OVERTURE_L0_CLASS
    2+3  Cross-tier        → Mixed Use when Tier 2 and Tier 3 disagree on a
                             trigger pair
    3b   OSM POI nodes     → resolve class from OSM_TAG_CLASS using POI tags
                             (amenity, shop, office, leisure, public_transport)
    3+3b Cross-tier        → Mixed Use when Tier 3 and Tier 3b disagree on a
                             trigger pair
    4    OSM landuse zone  → low-confidence class from the enclosing polygon
    5    Unclassified      → no evidence available
    """
    if osm_poi_matches is None:
        osm_poi_matches = []

    # ------------------------------------------------------------------
    # Tier 1 — CCCM ground-truth
    # ------------------------------------------------------------------
    if cccm_match is not None:
        return {
            "lu_class": "Temporary Housing",
            "lu_subclass": None,
            "lu_tier": 1,
            "lu_source": "cccm",
            "lu_confidence": "high",
            "lu_class_components": None,
            "geometry_source": _get_field(cccm_match, "geometry_source"),
        }

    # ------------------------------------------------------------------
    # Tier 2 — OSM building tags
    # ------------------------------------------------------------------
    tier2_class: Optional[str] = None
    tier2_entry: Optional[Dict] = None

    if osm_building_match is not None:
        entries = _resolve_osm_tags(osm_building_match)
        distinct = _distinct_classes(entries)

        if len(distinct) == 1:
            # Single unambiguous class
            tier2_class = distinct[0]
            # Return the entry with highest priority (first match in tag order)
            tier2_entry = next(e for e in entries if e["lu_class"] == tier2_class)

        elif len(distinct) >= 2:
            if _triggers_mixed_use(distinct):
                return _mixed_use_result(
                    components=distinct[:2],  # report the triggering pair
                    tier=2,
                    source="osm_building",
                    confidence="high",
                )
            else:
                # Multiple classes but no trigger pair — take highest-priority
                tier2_class = distinct[0]
                tier2_entry = next(e for e in entries if e["lu_class"] == tier2_class)

    # ------------------------------------------------------------------
    # Tier 3 — Overture Places
    # ------------------------------------------------------------------
    tier3_class: Optional[str] = None
    tier3_l0: Optional[str] = None

    if overture_matches:
        l0_class_pairs: List[tuple] = []
        for match in overture_matches:
            l0 = _overture_l0(match)
            if l0 is None:
                continue
            cls = OVERTURE_L0_CLASS.get(l0)
            if cls:
                l0_class_pairs.append((l0, cls))

        distinct_overture = list(dict.fromkeys(cls for _, cls in l0_class_pairs))

        if len(distinct_overture) == 1:
            tier3_class = distinct_overture[0]
            best = _best_overture_entry(l0_class_pairs)
            tier3_l0 = best[0] if best else None

        elif len(distinct_overture) >= 2:
            if _triggers_mixed_use(distinct_overture):
                return _mixed_use_result(
                    components=distinct_overture[:2],
                    tier=3,
                    source="overture",
                    confidence="medium",
                )
            else:
                best = _best_overture_entry(l0_class_pairs)
                if best:
                    tier3_class = best[1]
                    tier3_l0 = best[0]

    # ------------------------------------------------------------------
    # Cross-tier check (2 + 3) — upgrade to Mixed Use if they disagree
    # ------------------------------------------------------------------
    if tier2_class is not None and tier3_class is not None:
        if tier2_class != tier3_class:
            if frozenset({tier2_class, tier3_class}) in MIXED_USE_TRIGGER_PAIRS:
                return _mixed_use_result(
                    components=[tier2_class, tier3_class],
                    tier="2+3",
                    source="osm_building+overture",
                    confidence="high",
                )

    # Return Tier 2 result if available
    if tier2_class is not None and tier2_entry is not None:
        return _single_class_result(tier2_entry, tier=2, source="osm_building")

    # ------------------------------------------------------------------
    # Tier 3b — OSM POI nodes within or near the footprint
    # ------------------------------------------------------------------
    tier3b_class: Optional[str] = None
    tier3b_entry: Optional[Dict] = None

    if osm_poi_matches:
        poi_entries: List[Dict] = []
        for poi_row in osm_poi_matches:
            # Take the highest-priority tag from each POI (one entry per node)
            for key in _POI_TAG_PRIORITY:
                value = _get_field(poi_row, key)
                if value is None:
                    continue
                entry = _lookup_entry(key, value)
                if entry:
                    poi_entries.append(entry)
                    break

        distinct_3b = _distinct_classes(poi_entries)

        if len(distinct_3b) == 1:
            tier3b_class = distinct_3b[0]
            tier3b_entry = next(e for e in poi_entries if e["lu_class"] == tier3b_class)

        elif len(distinct_3b) >= 2:
            if _triggers_mixed_use(distinct_3b):
                return _mixed_use_result(
                    components=distinct_3b[:2],
                    tier="3b",
                    source="osm_poi",
                    confidence="medium",
                )
            else:
                # Multiple classes but no trigger pair — take highest-priority
                tier3b_class = distinct_3b[0]
                tier3b_entry = next(e for e in poi_entries if e["lu_class"] == tier3b_class)

    # ------------------------------------------------------------------
    # Cross-tier check (3 + 3b) — upgrade to Mixed Use if they disagree
    # ------------------------------------------------------------------
    if tier3_class is not None and tier3b_class is not None:
        if tier3_class != tier3b_class:
            if frozenset({tier3_class, tier3b_class}) in MIXED_USE_TRIGGER_PAIRS:
                return _mixed_use_result(
                    components=[tier3_class, tier3b_class],
                    tier="3+3b",
                    source="overture+osm_poi",
                    confidence="medium",
                )

    # Return Tier 3 result if available (takes precedence over 3b)
    if tier3_class is not None:
        return {
            "lu_class": tier3_class,
            "lu_subclass": None,
            "lu_tier": 3,
            "lu_source": "overture",
            "lu_confidence": "medium",
            "lu_class_components": None,
            "geometry_source": None,
        }

    # Return Tier 3b result if available
    if tier3b_class is not None and tier3b_entry is not None:
        return {
            "lu_class": tier3b_entry["lu_class"],
            "lu_subclass": tier3b_entry["lu_subclass"],
            "lu_tier": "3b",
            "lu_source": "osm_poi",
            "lu_confidence": "medium",
            "lu_class_components": None,
            "geometry_source": None,
        }

    # ------------------------------------------------------------------
    # Tier 4 — OSM landuse zone
    # ------------------------------------------------------------------
    if osm_landuse_match is not None:
        landuse_val = _get_field(osm_landuse_match, "landuse")
        if landuse_val:
            entry = _lookup_entry("landuse", landuse_val)
            if entry:
                return _single_class_result(entry, tier=4, source="osm_landuse")

    # ------------------------------------------------------------------
    # Tier 5 — Unclassified
    # ------------------------------------------------------------------
    return dict(_UNCLASSIFIED)
