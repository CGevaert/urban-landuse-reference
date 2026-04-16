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
"""Projected CRS used for metric spatial operations.
This value is updated at runtime from the AOI centroid to a suitable UTM
zone for the study area."""

OSM_PBF_SOURCE: Optional[str] = None
"""Local file path or download URL for the OSM PBF to use. Set by init()."""


def _read_vector_file(path: "str | Path") -> "gpd.GeoDataFrame":
    """Read a vector file from GeoJSON or GeoPackage with robust fallback.

    The pipeline accepts both GeoJSON and GeoPackage inputs for AOI and
    building footprints. GeoJSON is read with an explicit driver if possible,
    and a Fiona fallback is attempted on unusually small read results.
    """
    from pathlib import Path
    import warnings
    import geopandas as gpd

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Vector file not found: {path}")

    suffix = path.suffix.lower()
    driver = None
    if suffix in {".geojson", ".json"}:
        driver = "GeoJSON"
    elif suffix == ".gpkg":
        driver = "GPKG"

    try:
        gdf = gpd.read_file(str(path), driver=driver) if driver else gpd.read_file(str(path))
    except Exception as exc:
        if suffix in {".geojson", ".json"}:
            warnings.warn(
                f"Reading {path.name} via geopandas failed: {exc}. "
                "Falling back to Fiona."
            )
            try:
                import fiona
                with fiona.open(str(path), "r") as src:
                    gdf = gpd.GeoDataFrame.from_features(src, crs=src.crs)
            except Exception as exc2:
                raise RuntimeError(
                    f"Failed to read GeoJSON fallback for {path}: {exc2}"
                ) from exc2
        else:
            raise

    if suffix in {".geojson", ".json"} and len(gdf) <= 1 and path.stat().st_size > 100_000_000:
        warnings.warn(
            f"GeoJSON {path.name} was read as {len(gdf)} feature(s); "
            "retrying with Fiona fallback."
        )
        try:
            import fiona
            with fiona.open(str(path), "r") as src:
                fallback_gdf = gpd.GeoDataFrame.from_features(src, crs=src.crs)
            if len(fallback_gdf) > len(gdf):
                gdf = fallback_gdf
        except Exception:
            pass

    return gdf

POINT_BUFFER_M: int = 100
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

USE_CCCM: bool = False
"""Whether CCCM/IDP site data acquisition and Tier 1 classification are enabled.
Set by init() via the ``--include-cccm`` CLI flag.  Defaults to False so that
the pipeline does not require an HDX account or network access to the CCCM
dataset for basic city-agnostic runs."""


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

def init(
    aoi_path: "str | Path",
    footprints_path: "str | Path",
    project_name: Optional[str] = None,
    confidence_min: float = 0.7,
    use_cccm: bool = False,
    osm_pbf: Optional[str] = None,
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
    osm_pbf : str | None
        Optional local path or HTTP(S) URL to an OSM PBF file.
        If omitted, the pipeline uses the default South Sudan Geofabrik extract.

    Raises
    ------
    FileNotFoundError
        If *aoi_path* does not exist.
    """

    def _choose_projected_crs(aoi_raw_gdf) -> str:
        """Return the best projected CRS string for the AOI.

        If the AOI is already in a UTM projected CRS (EPSG 32601-32660 for
        northern hemisphere, 32701-32760 for southern), that CRS is returned
        directly.  Otherwise the best UTM zone is derived from the AOI centroid
        reprojected to WGS 84.
        """
        raw_crs = aoi_raw_gdf.crs
        if raw_crs is not None and raw_crs.is_projected:
            epsg = raw_crs.to_epsg()
            if epsg is not None and (32601 <= epsg <= 32660 or 32701 <= epsg <= 32760):
                return f"EPSG:{epsg}"

        # Derive UTM zone from centroid in geographic coordinates
        if raw_crs is None or raw_crs.to_epsg() != 4326:
            geo_gdf = aoi_raw_gdf.to_crs("EPSG:4326")
        else:
            geo_gdf = aoi_raw_gdf
        centroid = geo_gdf.dissolve().geometry.iloc[0].centroid
        lon, lat = centroid.x, centroid.y
        zone = int((lon + 180.0) / 6.0) + 1
        epsg = 32600 + zone if lat >= 0 else 32700 + zone
        return f"EPSG:{epsg}"

    import geopandas as gpd
    from shapely.validation import make_valid

    global PROJECT_NAME, FOOTPRINTS_PATH, INPUT_DIR, SCRATCH_GPKG, OUTPUT_GPKG
    global OVERTURE_CONFIDENCE_MIN, USE_CCCM, CRS_PROJ, OSM_PBF_SOURCE
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
    USE_CCCM = use_cccm
    OSM_PBF_SOURCE = osm_pbf

    # Ensure data directories exist
    _DATA.mkdir(parents=True, exist_ok=True)
    (_DATA / "input").mkdir(parents=True, exist_ok=True)
    (_DATA / "scratch").mkdir(parents=True, exist_ok=True)

    INPUT_DIR = _DATA / "input"
    SCRATCH_GPKG = _DATA / f"{project_name}_scratch.gpkg"
    OUTPUT_GPKG = _DATA / f"{project_name}_landuse_reference.gpkg"

    # AOI geometry
    _aoi_raw = _read_vector_file(aoi_path)

    if _aoi_raw.crs is None or _aoi_raw.crs.to_epsg() != 4326:
        _aoi_geo = _aoi_raw.to_crs(CRS_GEO)
    else:
        _aoi_geo = _aoi_raw.copy()

    _aoi_dissolved = _aoi_geo.dissolve()
    AOI_GEOM = make_valid(_aoi_dissolved.geometry.iloc[0])

    # Select the best projected CRS: use the AOI's own CRS if it is already a
    # UTM zone, otherwise derive the best UTM zone from the AOI centroid.
    CRS_PROJ = _choose_projected_crs(_aoi_raw)
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
    print(f"    projected CRS: {CRS_PROJ}")
    print(f"    confidence:  >= {confidence_min}")
    print(f"    CCCM/IDP:    {'enabled (--include-cccm)' if use_cccm else 'disabled (use --include-cccm to enable)'}")
