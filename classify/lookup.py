"""
classify/lookup.py — Static lookup tables for land-use classification.

This file contains only data — no logic, no I/O, no side effects.

Contents
--------
OSM_TAG_CLASS          : (tag_key, tag_value) → classification dict
MIXED_USE_TRIGGER_PAIRS: frozensets of class pairs that imply Mixed Use
OVERTURE_L0_CLASS      : Overture L0 taxonomy string → lu_class
OVERTURE_PRIORITY_ORDER: L0 categories ordered highest → lowest priority
"""

from typing import Dict, FrozenSet, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------

_Entry = Dict[str, Optional[str]]  # {lu_class, lu_subclass, lu_confidence}


def _e(lu_class: str, lu_subclass: Optional[str], lu_confidence: str) -> _Entry:
    """Convenience constructor for a lookup entry."""
    return {
        "lu_class": lu_class,
        "lu_subclass": lu_subclass,
        "lu_confidence": lu_confidence,
    }


# ---------------------------------------------------------------------------
# OSM_TAG_CLASS
# ---------------------------------------------------------------------------
# Keys are (tag_key, tag_value) tuples. Use "*" as the value to match any
# non-null value for that key (wildcard). The assign module checks for an
# exact match first, then falls back to the wildcard entry.
#
# Confidence semantics:
#   high   — the tag reliably identifies this land use
#   medium — the tag is suggestive but ambiguous in some contexts
#   low    — inferred from an area/zone tag; individual buildings may differ
# ---------------------------------------------------------------------------

OSM_TAG_CLASS: Dict[Tuple[str, str], _Entry] = {

    # -----------------------------------------------------------------------
    # RESIDENTIAL
    # -----------------------------------------------------------------------
    ("building", "residential"):      _e("Residential", None,             "medium"),
    ("building", "house"):            _e("Residential", None,             "medium"),
    ("building", "detached"):         _e("Residential", None,             "medium"),
    ("building", "semi-detached"):    _e("Residential", None,             "medium"),
    ("building", "terrace"):          _e("Residential", None,             "medium"),
    ("building", "bungalow"):         _e("Residential", None,             "medium"),
    ("building", "villa"):            _e("Residential", None,             "medium"),
    ("building", "hut"):              _e("Residential", "Traditional",    "medium"),
    ("building", "cabin"):            _e("Residential", "Traditional",    "low"),
    ("building", "farm"):             _e("Residential", "Traditional",    "low"),
    ("building", "farm_auxiliary"):   _e("Residential", "Traditional",    "low"),
    ("building", "apartments"):       _e("Residential", "Multi-family",   "high"),
    ("building", "dormitory"):        _e("Residential", "Multi-family",   "high"),
    ("building", "block_of_flats"):   _e("Residential", "Multi-family",   "high"),
    ("building", "static_caravan"):   _e("Residential", "Temporary",      "medium"),
    ("landuse", "residential"):       _e("Residential", None,             "low"),

    # -----------------------------------------------------------------------
    # COMMERCIAL
    # -----------------------------------------------------------------------
    ("building", "commercial"):        _e("Commercial", None,             "high"),
    ("building", "retail"):            _e("Commercial", None,             "high"),
    ("building", "supermarket"):       _e("Commercial", "Retail",         "high"),
    ("building", "kiosk"):             _e("Commercial", "Retail",         "high"),
    ("building", "market"):            _e("Commercial", "Market",         "high"),
    ("building", "mall"):              _e("Commercial", "Retail",         "high"),
    ("building", "hotel"):             _e("Commercial", "Hospitality",    "high"),
    ("building", "guest_house"):       _e("Commercial", "Hospitality",    "high"),
    ("building", "office"):            _e("Commercial", None,             "high"),
    ("shop", "*"):                     _e("Commercial", None,             "high"),
    ("office", "*"):                   _e("Commercial", None,             "high"),
    ("amenity", "marketplace"):        _e("Commercial", "Market",         "high"),
    ("amenity", "bank"):               _e("Commercial", "Finance",        "high"),
    ("amenity", "atm"):                _e("Commercial", "Finance",        "high"),
    ("amenity", "bureau_de_change"):   _e("Commercial", "Finance",        "high"),
    ("amenity", "money_transfer"):     _e("Commercial", "Finance",        "high"),
    ("amenity", "fuel"):               _e("Commercial", "Service",        "high"),
    ("amenity", "car_wash"):           _e("Commercial", "Service",        "high"),
    ("amenity", "car_rental"):         _e("Commercial", "Service",        "high"),
    ("amenity", "restaurant"):         _e("Commercial", "Food & Drink",   "high"),
    ("amenity", "fast_food"):          _e("Commercial", "Food & Drink",   "high"),
    ("amenity", "bar"):                _e("Commercial", "Food & Drink",   "high"),
    ("amenity", "pub"):                _e("Commercial", "Food & Drink",   "high"),
    ("amenity", "cafe"):               _e("Commercial", "Food & Drink",   "high"),
    ("amenity", "nightclub"):          _e("Commercial", "Entertainment",  "high"),
    ("amenity", "cinema"):             _e("Commercial", "Entertainment",  "high"),
    ("amenity", "arts_centre"):        _e("Commercial", "Entertainment",  "high"),
    ("amenity", "casino"):             _e("Commercial", "Entertainment",  "high"),
    ("tourism", "hotel"):              _e("Commercial", "Hospitality",    "high"),
    ("tourism", "motel"):              _e("Commercial", "Hospitality",    "high"),
    ("tourism", "guest_house"):        _e("Commercial", "Hospitality",    "high"),
    ("tourism", "hostel"):             _e("Commercial", "Hospitality",    "high"),
    ("tourism", "camp_site"):          _e("Commercial", "Hospitality",    "medium"),
    ("landuse", "commercial"):         _e("Commercial", None,             "low"),
    ("landuse", "retail"):             _e("Commercial", None,             "low"),

    # -----------------------------------------------------------------------
    # INDUSTRIAL
    # -----------------------------------------------------------------------
    ("building", "industrial"):       _e("Industrial", None,             "high"),
    ("building", "warehouse"):        _e("Industrial", "Storage",        "high"),
    ("building", "storage_tank"):     _e("Industrial", "Storage",        "high"),
    ("building", "factory"):          _e("Industrial", "Manufacturing",  "high"),
    ("building", "manufacture"):      _e("Industrial", "Manufacturing",  "high"),
    ("building", "slaughterhouse"):   _e("Industrial", "Manufacturing",  "high"),
    ("building", "water_tower"):      _e("Industrial", "Utility",        "medium"),
    ("building", "pumping_station"):  _e("Industrial", "Utility",        "medium"),
    ("building", "power_substation"): _e("Industrial", "Utility",        "medium"),
    ("man_made", "water_tower"):      _e("Industrial", "Utility",        "medium"),
    ("man_made", "storage_tank"):     _e("Industrial", "Storage",        "medium"),
    ("man_made", "wastewater_plant"): _e("Industrial", "Utility",        "high"),
    ("man_made", "works"):            _e("Industrial", "Manufacturing",  "medium"),
    ("power", "plant"):               _e("Industrial", "Utility",        "high"),
    ("landuse", "industrial"):        _e("Industrial", None,             "low"),
    ("landuse", "quarry"):            _e("Industrial", "Extractive",     "medium"),
    ("landuse", "landfill"):          _e("Industrial", "Waste",          "medium"),

    # -----------------------------------------------------------------------
    # PUBLIC / INSTITUTIONAL
    # -----------------------------------------------------------------------
    # Education
    ("amenity", "school"):            _e("Public/Institutional", "Education",   "high"),
    ("amenity", "university"):        _e("Public/Institutional", "Education",   "high"),
    ("amenity", "college"):           _e("Public/Institutional", "Education",   "high"),
    ("amenity", "kindergarten"):      _e("Public/Institutional", "Education",   "high"),
    ("amenity", "library"):           _e("Public/Institutional", "Education",   "high"),
    ("amenity", "training"):          _e("Public/Institutional", "Education",   "medium"),
    ("building", "school"):           _e("Public/Institutional", "Education",   "high"),
    ("building", "university"):       _e("Public/Institutional", "Education",   "high"),
    ("building", "college"):          _e("Public/Institutional", "Education",   "high"),
    ("building", "kindergarten"):     _e("Public/Institutional", "Education",   "high"),
    # Health
    ("amenity", "hospital"):          _e("Public/Institutional", "Health",      "high"),
    ("amenity", "clinic"):            _e("Public/Institutional", "Health",      "high"),
    ("amenity", "health_post"):       _e("Public/Institutional", "Health",      "high"),
    ("amenity", "pharmacy"):          _e("Public/Institutional", "Health",      "high"),
    ("amenity", "dentist"):           _e("Public/Institutional", "Health",      "high"),
    ("amenity", "doctors"):           _e("Public/Institutional", "Health",      "high"),
    ("amenity", "veterinary"):        _e("Public/Institutional", "Health",      "medium"),
    ("building", "hospital"):         _e("Public/Institutional", "Health",      "high"),
    # Religious
    ("amenity", "place_of_worship"):  _e("Public/Institutional", "Religious",   "high"),
    ("building", "church"):           _e("Public/Institutional", "Religious",   "high"),
    ("building", "cathedral"):        _e("Public/Institutional", "Religious",   "high"),
    ("building", "chapel"):           _e("Public/Institutional", "Religious",   "high"),
    ("building", "mosque"):           _e("Public/Institutional", "Religious",   "high"),
    ("building", "temple"):           _e("Public/Institutional", "Religious",   "high"),
    ("building", "shrine"):           _e("Public/Institutional", "Religious",   "high"),
    # Government / civic
    ("amenity", "police"):            _e("Public/Institutional", "Government",  "high"),
    ("amenity", "fire_station"):      _e("Public/Institutional", "Government",  "high"),
    ("amenity", "townhall"):          _e("Public/Institutional", "Government",  "high"),
    ("amenity", "courthouse"):        _e("Public/Institutional", "Government",  "high"),
    ("amenity", "prison"):            _e("Public/Institutional", "Government",  "high"),
    ("amenity", "immigration"):       _e("Public/Institutional", "Government",  "high"),
    ("amenity", "post_office"):       _e("Public/Institutional", "Government",  "medium"),
    ("amenity", "embassy"):           _e("Public/Institutional", "Government",  "high"),
    ("office", "government"):         _e("Public/Institutional", "Government",  "high"),
    ("office", "ngo"):                _e("Public/Institutional", "Government",  "medium"),
    ("office", "diplomatic"):         _e("Public/Institutional", "Government",  "high"),
    ("office", "association"):        _e("Public/Institutional", "Government",  "medium"),
    ("building", "government"):       _e("Public/Institutional", "Government",  "high"),
    ("building", "public"):           _e("Public/Institutional", None,          "medium"),
    # Community / social
    ("amenity", "community_centre"):  _e("Public/Institutional", "Community",   "high"),
    ("amenity", "social_centre"):     _e("Public/Institutional", "Community",   "high"),
    ("amenity", "social_facility"):   _e("Public/Institutional", "Community",   "high"),
    ("amenity", "grave_yard"):        _e("Public/Institutional", "Cemetery",    "high"),
    ("landuse", "cemetery"):          _e("Public/Institutional", "Cemetery",    "high"),
    # Military
    ("landuse", "military"):          _e("Public/Institutional", "Military",    "medium"),
    ("military", "*"):                _e("Public/Institutional", "Military",    "high"),
    ("building", "barracks"):         _e("Public/Institutional", "Military",    "high"),
    # General institutional zone
    ("landuse", "institutional"):     _e("Public/Institutional", None,          "low"),

    # -----------------------------------------------------------------------
    # TRANSPORT
    # -----------------------------------------------------------------------
    ("aeroway", "*"):                  _e("Transport", "Airport",         "high"),
    ("amenity", "bus_station"):        _e("Transport", "Road",            "high"),
    ("amenity", "ferry_terminal"):     _e("Transport", "Water",           "high"),
    ("amenity", "taxi"):               _e("Transport", "Road",            "high"),
    ("amenity", "motorcycle_taxi"):    _e("Transport", "Road",            "high"),
    ("amenity", "parking"):            _e("Transport", "Road",            "medium"),
    ("amenity", "bicycle_parking"):    _e("Transport", "Road",            "low"),
    ("public_transport", "station"):   _e("Transport", None,              "high"),
    ("public_transport", "stop_position"): _e("Transport", None,          "medium"),
    ("public_transport", "platform"):  _e("Transport", None,              "medium"),
    ("railway", "station"):            _e("Transport", "Rail",            "high"),
    ("railway", "halt"):               _e("Transport", "Rail",            "high"),
    ("building", "train_station"):     _e("Transport", "Rail",            "high"),
    ("building", "transportation"):    _e("Transport", None,              "high"),
    ("landuse", "port"):               _e("Transport", "Water",           "high"),

    # -----------------------------------------------------------------------
    # TEMPORARY HOUSING
    # -----------------------------------------------------------------------
    ("amenity", "refugee_camp"):       _e("Temporary Housing", None,      "high"),
    ("social_facility", "shelter"):    _e("Temporary Housing", None,      "medium"),
    ("social_facility", "refugee"):    _e("Temporary Housing", None,      "high"),
    ("landuse", "camp_site"):          _e("Temporary Housing", None,      "low"),
    ("building", "shelter"):           _e("Temporary Housing", None,      "medium"),

    # -----------------------------------------------------------------------
    # OPEN SPACE / ENVIRONMENTAL
    # -----------------------------------------------------------------------
    ("landuse", "grass"):              _e("Open Space/Environmental", None,            "high"),
    ("landuse", "meadow"):             _e("Open Space/Environmental", None,            "high"),
    ("landuse", "forest"):             _e("Open Space/Environmental", "Vegetation",    "high"),
    ("landuse", "farmland"):           _e("Open Space/Environmental", "Agricultural",  "medium"),
    ("landuse", "farmyard"):           _e("Open Space/Environmental", "Agricultural",  "medium"),
    ("landuse", "orchard"):            _e("Open Space/Environmental", "Agricultural",  "medium"),
    ("landuse", "allotments"):         _e("Open Space/Environmental", "Agricultural",  "medium"),
    ("landuse", "garden"):             _e("Open Space/Environmental", None,            "medium"),
    ("landuse", "recreation_ground"):  _e("Open Space/Environmental", "Recreation",   "high"),
    ("landuse", "village_green"):      _e("Open Space/Environmental", None,            "medium"),
    ("landuse", "greenfield"):         _e("Open Space/Environmental", None,            "low"),
    ("leisure", "park"):               _e("Open Space/Environmental", "Recreation",   "high"),
    ("leisure", "pitch"):              _e("Open Space/Environmental", "Recreation",   "high"),
    ("leisure", "playground"):         _e("Open Space/Environmental", "Recreation",   "high"),
    ("leisure", "garden"):             _e("Open Space/Environmental", None,            "high"),
    ("leisure", "sports_centre"):      _e("Open Space/Environmental", "Recreation",   "high"),
    ("leisure", "stadium"):            _e("Open Space/Environmental", "Recreation",   "high"),
    ("leisure", "swimming_pool"):      _e("Open Space/Environmental", "Recreation",   "high"),
    ("leisure", "golf_course"):        _e("Open Space/Environmental", "Recreation",   "high"),
    ("leisure", "nature_reserve"):     _e("Open Space/Environmental", "Vegetation",   "high"),
    ("natural", "water"):              _e("Open Space/Environmental", "Water",         "high"),
    ("natural", "wetland"):            _e("Open Space/Environmental", "Wetland",       "high"),
    ("natural", "wood"):               _e("Open Space/Environmental", "Vegetation",    "high"),
    ("natural", "scrub"):              _e("Open Space/Environmental", "Vegetation",    "medium"),
    ("natural", "heath"):              _e("Open Space/Environmental", "Vegetation",    "medium"),
    ("natural", "grassland"):          _e("Open Space/Environmental", None,            "medium"),
    ("natural", "sand"):               _e("Open Space/Environmental", None,            "medium"),
    ("natural", "beach"):              _e("Open Space/Environmental", None,            "medium"),
    ("waterway", "riverbank"):         _e("Open Space/Environmental", "Water",         "high"),
    ("landuse", "basin"):              _e("Open Space/Environmental", "Water",         "high"),
    ("landuse", "reservoir"):          _e("Open Space/Environmental", "Water",         "high"),
}

# ---------------------------------------------------------------------------
# MIXED_USE_TRIGGER_PAIRS
# ---------------------------------------------------------------------------
# A frozenset of two class names whose co-occurrence on a single building
# triggers Mixed Use labelling. The assign module checks whether any pair of
# resolved classes appears in this list.

MIXED_USE_TRIGGER_PAIRS: List[FrozenSet[str]] = [
    frozenset({"Residential", "Commercial"}),
    frozenset({"Residential", "Public/Institutional"}),
    frozenset({"Commercial", "Public/Institutional"}),
    frozenset({"Residential", "Industrial"}),
    frozenset({"Commercial", "Industrial"}),
    frozenset({"Residential", "Transport"}),
]

# ---------------------------------------------------------------------------
# OVERTURE_L0_CLASS
# ---------------------------------------------------------------------------
# Maps an Overture Maps L0 taxonomy string to an lu_class name.
# The L0 value is obtained by splitting category_primary on "." and taking [0].

OVERTURE_L0_CLASS: Dict[str, str] = {
    "accommodation":          "Residential",
    "food_and_drink":         "Commercial",
    "retail":                 "Commercial",
    "civic_and_social":       "Public/Institutional",
    "health_and_medical":     "Public/Institutional",
    "education":              "Public/Institutional",
    "transportation":         "Transport",
    "arts_and_entertainment": "Commercial",
    "government_and_community": "Public/Institutional",
    "religious":              "Public/Institutional",
    "natural_and_geographic": "Open Space/Environmental",
    "landforms":              "Open Space/Environmental",
    "financial_services":     "Commercial",
    "business_to_business":   "Commercial",
    "professional_services":  "Commercial",
    "mass_media_and_information": "Commercial",
    "activity":               "Open Space/Environmental",
    "landmarks_and_outdoors": "Open Space/Environmental",
}

# ---------------------------------------------------------------------------
# OVERTURE_PRIORITY_ORDER
# ---------------------------------------------------------------------------
# L0 categories in descending priority order. When multiple Overture records
# within a building footprint resolve to different classes, the record whose
# L0 appears earliest in this list takes precedence (unless a Mixed Use
# trigger pair applies, in which case Mixed Use is assigned instead).

OVERTURE_PRIORITY_ORDER: List[str] = [
    "civic_and_social",
    "health_and_medical",
    "education",
    "government_and_community",
    "religious",
    "transportation",
    "retail",
    "food_and_drink",
    "accommodation",
    "arts_and_entertainment",
    "financial_services",
    "professional_services",
    "business_to_business",
    "mass_media_and_information",
    "activity",
    "landmarks_and_outdoors",
    "natural_and_geographic",
    "landforms",
]
