"""
config.py — Central configuration for the landuse_ref pipeline.

All path constants, CRS identifiers, and AOI-derived spatial constants are
defined here.  Derived constants are populated by calling ``init()`` once at
the very start of ``run.py``'s ``main()`` before any other module is imported.

Static constants (CRS_GEO, CRS_PROJ, OSM_TIMEOUT, POINT_BUFFER_M) are always
available at import time.  Dynamic constants (AOI_GEOM, SCRATCH_GPKG, etc.)
are ``None`` until ``init()`` is called.
"""

from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Package root (always available)
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).parent
"""Absolute path to the pipeline package root directory."""

_DATA = _ROOT / "data"
"""Absolute path to the data/ directory (package-root-relative)."""

# ---------------------------------------------------------------------------
# Static constants — always available before init()
# ---------------------------------------------------------------------------

CRS_GEO: str = "EPSG:4326"
"""Geographic CRS (WGS 84, decimal degrees). Used for data exchange and bbox operations."""

CRS_PROJ: str = "EPSG:32636"
"""Projected CRS (WGS 84 / UTM Zone 36N, metres). Suitable for spatial
operations covering East Africa.  Use for distance, area, and buffer work."""

POINT_BUFFER_M: int = 250
"""Buffer distance in metres applied around point geometries when converting
them to polygonal features for spatial analysis."""

OSM_TIMEOUT: int = 300
"""Timeout in seconds for Overpass API queries (OpenStreetMap data acquisition)."""

# ---------------------------------------------------------------------------
# Dynamic constants — None until init() is called
# ---------------------------------------------------------------------------

PROJECT_NAME: str = "project"
"""Short label used in output filenames.  Set by init()."""

FOOTPRINTS_PATH: Optional[Path] = None
"""Path to the user-supplied building footprints file.  Set by init()."""

INPUT_DIR: Optional[Path] = None
"""Path to data/input/ for user-supplied and downloaded files.  Set by init()."""

SCRATCH_GPKG: Optional[Path] = None
"""Path to the intermediate GeoPackage ({project_name}_scratch.gpkg).  Set by init()."""

OUTPUT_GPKG: Optional[Path] = None
"""Path to the final output GeoPackage ({project_name}_landuse_reference.gpkg).  Set by init()."""

OVERTURE_CONFIDENCE_MIN: float = 0.7
"""Minimum confidence score (0–1) for retaining Overture Maps features.  Set by init()."""

AOI_GEOM = None
"""Dissolved shapely geometry of the AOI in EPSG:4326.  Set by init()."""

AOI_GEOM_PROJ = None
"""Dissolved shapely geometry of the AOI in CRS_PROJ (metres).  Set by init()."""

AOI_BBOX: Optional[dict] = None
"""Bounding box of the AOI as a dict with keys west/south/east/north (EPSG:4326).  Set by init()."""

AOI_BBOX_STR: Optional[str] = None
"""Bounding box of the AOI as 'west,south,east,north' string (EPSG:4326).  Set by init()."""

AOI_BBOX_OVERPASS: Optional[tuple] = None
"""Bounding box in Overpass API format: (south, west, north, east).  Set by init()."""


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

def init(
    aoi_path: "str | Path",
    footprints_path: "str | Path",
    project_name: Optional[str] = None,
    confidence_min: float = 0.7,
) -> None:
    """
    Initialise all derived pipeline constants from the supplied AOI and
    footprints paths.

    Must be called once at the very start of ``main()`` before importing any
    ``acquire``, ``preprocess``, ``classify``, or ``validate`` module.

    Parameters
    ----------
    aoi_path : str | Path
        Path to a GeoJSON file defining the study-area polygon(s).
    footprints_path : str | Path
        Path to the building footprints file (GeoPackage, Shapefile, etc.).
    project_name : str | None
        Short label used in output filenames.  Defaults to the stem of
        *aoi_path* (e.g. ``"juba"`` from ``"juba_aoi.geojson"``).
    confidence_min : float
        Minimum Overture Maps confidence threshold (0–1).  Default 0.7.

    Raises
    ------
    FileNotFoundError
        If *aoi_path* does not exist.
    """
    import geopandas as gpd
    from shapely.validation import make_valid

    global PROJECT_NAME, FOOTPRINTS_PATH, INPUT_DIR, SCRATCH_GPKG, OUTPUT_GPKG
    global OVERTURE_CONFIDENCE_MIN
    global AOI_GEOM, AOI_GEOM_PROJ, AOI_BBOX, AOI_BBOX_STR, AOI_BBOX_OVERPASS

    aoi_path = Path(aoi_path)
    footprints_path = Path(footprints_path)

    if not aoi_path.exists():
        raise FileNotFoundError(
            f"AOI file not found: {aoi_path}\n"
            "Please provide a GeoJSON file defining the study area of interest."
        )

    # Derive project name from AOI filename if not supplied
    if project_name is None:
        project_name = aoi_path.stem

    PROJECT_NAME = project_name
    FOOTPRINTS_PATH = footprints_path
    OVERTURE_CONFIDENCE_MIN = confidence_min

    # Ensure data directories exist
    _DATA.mkdir(parents=True, exist_ok=True)
    (_DATA / "input").mkdir(parents=True, exist_ok=True)
    (_DATA / "scratch").mkdir(parents=True, exist_ok=True)

    INPUT_DIR = _DATA / "input"
    SCRATCH_GPKG = _DATA / f"{project_name}_scratch.gpkg"
    OUTPUT_GPKG = _DATA / f"{project_name}_landuse_reference.gpkg"

    # AOI geometry
    _aoi_raw = gpd.read_file(str(aoi_path))

    if _aoi_raw.crs is None or _aoi_raw.crs.to_epsg() != 4326:
        _aoi_geo = _aoi_raw.to_crs(CRS_GEO)
    else:
        _aoi_geo = _aoi_raw.copy()

    _aoi_dissolved = _aoi_geo.dissolve()
    AOI_GEOM = make_valid(_aoi_dissolved.geometry.iloc[0])

    _aoi_proj_dissolved = _aoi_dissolved.to_crs(CRS_PROJ)
    AOI_GEOM_PROJ = make_valid(_aoi_proj_dissolved.geometry.iloc[0])

    _minx, _miny, _maxx, _maxy = AOI_GEOM.bounds
    AOI_BBOX = {"west": _minx, "south": _miny, "east": _maxx, "north": _maxy}
    AOI_BBOX_STR = f"{_minx},{_miny},{_maxx},{_maxy}"
    AOI_BBOX_OVERPASS = (_miny, _minx, _maxy, _maxx)

    print(f"  config.init() — project: '{project_name}'")
    print(f"    AOI:         {aoi_path}")
    print(f"    footprints:  {footprints_path}")
    print(f"    scratch:     {SCRATCH_GPKG.name}")
    print(f"    output:      {OUTPUT_GPKG.name}")
    print(f"    bbox (geo):  W={_minx:.4f}  S={_miny:.4f}  E={_maxx:.4f}  N={_maxy:.4f}")
    print(f"    confidence:  >= {confidence_min}")
