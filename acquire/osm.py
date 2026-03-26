"""
acquire/osm.py — Fetch OSM features for the study area from Geofabrik.

Downloads the South Sudan Geofabrik PBF extract to data/input/ (skipped if the
file is already present) and uses pyrosm to extract buildings, land-use
polygons, and POI nodes clipped precisely to config.AOI_GEOM.

Public functions
----------------
fetch_osm_buildings()  → GeoDataFrame of building polygons (EPSG:4326)
fetch_osm_landuse()    → GeoDataFrame of land-use polygons (EPSG:4326)
fetch_osm_pois()       → GeoDataFrame of POI points (EPSG:4326)
"""

import sqlite3
import sys
from pathlib import Path
from typing import Optional

import geopandas as gpd
import pandas as pd
import pyrosm
import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import AOI_BBOX, AOI_GEOM, CRS_GEO, CRS_PROJ, SCRATCH_GPKG  # noqa: E402

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

_GEOFABRIK_URL = (
    "https://download.geofabrik.de/africa/south-sudan-latest.osm.pbf"
)
_PBF_PATH: Path = (
    Path(__file__).parent.parent / "data" / "input" / "south-sudan-latest.osm.pbf"
)

# pyrosm osm_type string → osm_id prefix
_TYPE_PREFIX = {"node": "n", "way": "w", "relation": "r"}

# Cached pyrosm.OSM reader (initialised once on first use)
_osm_reader: Optional[pyrosm.OSM] = None

# ---------------------------------------------------------------------------
# Internal helpers — PBF download & reader
# ---------------------------------------------------------------------------

def _ensure_pbf() -> None:
    """
    Download the Geofabrik South Sudan PBF to *_PBF_PATH* if not already
    present.  Streams with a tqdm progress bar; skips the download entirely
    if the file exists.
    """
    if _PBF_PATH.exists():
        size_mb = _PBF_PATH.stat().st_size / 1_000_000
        print(
            f"  PBF already present: {_PBF_PATH.name} "
            f"({size_mb:.1f} MB) — skipping download."
        )
        return

    _PBF_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {_GEOFABRIK_URL} …")

    resp = requests.get(_GEOFABRIK_URL, stream=True, timeout=120)
    resp.raise_for_status()

    total = int(resp.headers.get("content-length", 0))
    with open(_PBF_PATH, "wb") as fh:
        with tqdm(
            total=total, unit="B", unit_scale=True,
            desc=_PBF_PATH.name, ncols=80,
        ) as pbar:
            for chunk in resp.iter_content(chunk_size=65_536):
                fh.write(chunk)
                pbar.update(len(chunk))

    size_mb = _PBF_PATH.stat().st_size / 1_000_000
    print(f"  Downloaded → {_PBF_PATH.name} ({size_mb:.1f} MB)")


def _get_reader() -> pyrosm.OSM:
    """
    Return a cached pyrosm.OSM reader initialised with the AOI bounding box.
    Downloads the PBF on first call if necessary.
    """
    global _osm_reader
    if _osm_reader is None:
        _ensure_pbf()
        bbox = [
            AOI_BBOX["west"],
            AOI_BBOX["south"],
            AOI_BBOX["east"],
            AOI_BBOX["north"],
        ]
        print(f"  Initialising pyrosm reader (bbox-filtered to AOI)…")
        _osm_reader = pyrosm.OSM(str(_PBF_PATH), bounding_box=bbox)
    return _osm_reader


# ---------------------------------------------------------------------------
# Internal helpers — data normalisation
# ---------------------------------------------------------------------------

def _col(gdf: gpd.GeoDataFrame, name: str) -> pd.Series:
    """
    Return column *name* from *gdf* with NaN/NA replaced by None.
    Returns a None-filled object Series if the column is absent.
    """
    if name in gdf.columns:
        s = gdf[name].copy().astype(object)
        return s.where(s.notna(), other=None)
    return pd.Series([None] * len(gdf), index=gdf.index, dtype=object)


def _make_ids(gdf: gpd.GeoDataFrame) -> pd.Series:
    """
    Build 'w{id}' / 'r{id}' / 'n{id}' strings from pyrosm's 'id' and
    'osm_type' columns.
    """
    ids = gdf["id"].astype(str)
    if "osm_type" in gdf.columns:
        prefixes = gdf["osm_type"].map(_TYPE_PREFIX).fillna("x")
    else:
        prefixes = pd.Series(["x"] * len(gdf), index=gdf.index)
    return prefixes + ids


def _fmt_ts(ts) -> Optional[str]:
    """
    Format a pyrosm timestamp (pd.Timestamp, datetime, str, or None) as an
    ISO 8601 string, returning None if the value is missing or unparseable.
    """
    if ts is None:
        return None
    try:
        return pd.Timestamp(ts).isoformat()
    except Exception:
        return None


def _clip_to_aoi(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Clip *gdf* precisely to AOI_GEOM (EPSG:4326).
    pyrosm's bbox filter is a rectangle; this step applies the actual AOI
    polygon boundary.  Handles reprojection if the layer is not in EPSG:4326.

    Falls back to a simple intersection filter (no geometry clipping) if
    shapely raises a GEOSException during the clip operation.
    """
    if gdf.empty:
        return gdf
    if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(CRS_GEO)
    try:
        return gdf.clip(mask=AOI_GEOM).copy()
    except Exception as exc:
        from shapely.errors import GEOSException
        if isinstance(exc, GEOSException) or "GEOSException" in type(exc).__name__:
            print(
                f"  [WARN] clip() raised a topology error ({exc}); "
                "falling back to intersection filter — geometries are not clipped to AOI boundary."
            )
            return gdf[gdf.geometry.intersects(AOI_GEOM)].copy()
        raise


def _empty_gdf(columns: list, crs: str = CRS_GEO) -> gpd.GeoDataFrame:
    """Return an empty GeoDataFrame with *columns* and *crs*."""
    return gpd.GeoDataFrame(
        {c: pd.Series([], dtype=object) for c in columns},
        geometry="geometry",
        crs=crs,
    )


# ---------------------------------------------------------------------------
# GeoPackage helpers (imported by acquire/overture.py, acquire/unhcr.py,
# export.py — keep signatures unchanged)
# ---------------------------------------------------------------------------

def _drop_gpkg_layer(gpkg_path: Path, layer_name: str) -> None:
    """
    Remove a single layer from a GeoPackage via raw SQLite.

    Drops the layer table and removes its entries from the GeoPackage
    metadata tables (gpkg_contents, gpkg_geometry_columns) so that a
    subsequent write creates a clean layer rather than appending.
    """
    conn = sqlite3.connect(str(gpkg_path))
    try:
        cur = conn.cursor()
        cur.execute(f'DROP TABLE IF EXISTS "{layer_name}"')
        for meta_table in ("gpkg_contents", "gpkg_geometry_columns"):
            cur.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (meta_table,),
            )
            if cur.fetchone():
                cur.execute(
                    f"DELETE FROM {meta_table} WHERE table_name = ?",
                    (layer_name,),
                )
        conn.commit()
    finally:
        conn.close()


def _save_layer(gdf: gpd.GeoDataFrame, layer: str) -> None:
    """
    Write *gdf* to SCRATCH_GPKG as *layer*, overwriting any existing layer
    of the same name.  Creates the parent directory if necessary.
    """
    SCRATCH_GPKG.parent.mkdir(parents=True, exist_ok=True)
    if SCRATCH_GPKG.exists():
        _drop_gpkg_layer(SCRATCH_GPKG, layer)
    gdf.to_file(str(SCRATCH_GPKG), driver="GPKG", layer=layer, mode="a")
    print(
        f"  Saved {len(gdf):,} features → "
        f"layer '{layer}' in {SCRATCH_GPKG.name}"
    )


# ---------------------------------------------------------------------------
# Public fetch functions
# ---------------------------------------------------------------------------

def fetch_osm_buildings() -> gpd.GeoDataFrame:
    """
    Extract OSM building polygons for the study area from the Geofabrik PBF.

    Reads ways and multipolygon relations tagged ``building=*`` using
    pyrosm, then clips precisely to ``config.AOI_GEOM``.

    Returns
    -------
    gpd.GeoDataFrame
        CRS: EPSG:4326.
        Columns: osm_id (str), geometry (Polygon/MultiPolygon),
        building (str|None), amenity (str|None), shop (str|None),
        office (str|None), landuse (str|None), name (str|None),
        last_edit (str|None — ISO 8601 OSM edit timestamp).
        Also written to layer ``osm_buildings`` in SCRATCH_GPKG.
    """
    _COLS = ["osm_id", "geometry", "building", "amenity",
             "shop", "office", "landuse", "name", "last_edit"]

    print("Reading OSM buildings from Geofabrik PBF…")
    osm = _get_reader()

    raw = osm.get_buildings()
    if raw is None or raw.empty:
        print("  No buildings found in AOI bbox.")
        return _empty_gdf(_COLS)

    print(f"  Raw features in bbox: {len(raw):,}")
    raw = _clip_to_aoi(raw)
    print(f"  After precise AOI clip: {len(raw):,}")

    if raw.empty:
        return _empty_gdf(_COLS)

    gdf = gpd.GeoDataFrame(
        {
            "osm_id":    _make_ids(raw),
            "geometry":  raw.geometry,
            "building":  _col(raw, "building"),
            "amenity":   _col(raw, "amenity"),
            "shop":      _col(raw, "shop"),
            "office":    _col(raw, "office"),
            "landuse":   _col(raw, "landuse"),
            "name":      _col(raw, "name"),
            "last_edit": _col(raw, "timestamp").apply(_fmt_ts),
        },
        geometry="geometry",
        crs=CRS_GEO,
    )

    # Drop null/empty geometries
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()

    print(f"  Built {len(gdf):,} building polygons.")
    _save_layer(gdf, "osm_buildings")
    return gdf


def fetch_osm_landuse() -> gpd.GeoDataFrame:
    """
    Extract OSM land-use polygons for the study area from the Geofabrik PBF.

    Reads ways and multipolygon relations tagged ``landuse=*`` using pyrosm,
    clips to ``config.AOI_GEOM``, then drops features smaller than 100 m²
    (projected area in EPSG:32636) as tagging errors.

    Returns
    -------
    gpd.GeoDataFrame
        CRS: EPSG:4326.
        Columns: osm_id (str), geometry (Polygon/MultiPolygon),
        landuse (str|None), name (str|None),
        last_edit (str|None — ISO 8601 OSM edit timestamp).
        Also written to layer ``osm_landuse`` in SCRATCH_GPKG.
    """
    _COLS = ["osm_id", "geometry", "landuse", "name", "last_edit"]

    print("Reading OSM land use from Geofabrik PBF…")
    osm = _get_reader()

    raw = osm.get_landuse()
    if raw is None or raw.empty:
        print("  No landuse features found in AOI bbox.")
        return _empty_gdf(_COLS)

    print(f"  Raw features in bbox: {len(raw):,}")
    raw = _clip_to_aoi(raw)
    print(f"  After precise AOI clip: {len(raw):,}")

    if raw.empty:
        return _empty_gdf(_COLS)

    gdf = gpd.GeoDataFrame(
        {
            "osm_id":    _make_ids(raw),
            "geometry":  raw.geometry,
            "landuse":   _col(raw, "landuse"),
            "name":      _col(raw, "name"),
            "last_edit": _col(raw, "timestamp").apply(_fmt_ts),
        },
        geometry="geometry",
        crs=CRS_GEO,
    )

    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()

    # Drop features smaller than 100 m² (tagging errors)
    area_m2 = gdf.to_crs(CRS_PROJ).geometry.area
    mask = area_m2 >= 100.0
    n_dropped = int((~mask).sum())
    if n_dropped:
        print(f"  Dropped {n_dropped:,} features with area < 100 m².")
    gdf = gdf.loc[mask].copy()

    print(f"  Built {len(gdf):,} land-use polygons.")
    _save_layer(gdf, "osm_landuse")
    return gdf


def fetch_osm_pois() -> gpd.GeoDataFrame:
    """
    Extract OSM POI nodes for the study area from the Geofabrik PBF.

    Reads nodes tagged with any of: amenity, shop, office, leisure, tourism,
    or public_transport.  Only Point geometries (nodes) are retained.

    Returns
    -------
    gpd.GeoDataFrame
        CRS: EPSG:4326.
        Columns: osm_id (str), geometry (Point),
        amenity (str|None), shop (str|None), office (str|None),
        leisure (str|None), tourism (str|None),
        public_transport (str|None), name (str|None),
        last_edit (str|None — ISO 8601 OSM edit timestamp).
        Also written to layer ``osm_pois`` in SCRATCH_GPKG.
    """
    _COLS = [
        "osm_id", "geometry", "amenity", "shop", "office",
        "leisure", "tourism", "public_transport", "name", "last_edit",
    ]

    print("Reading OSM POIs from Geofabrik PBF…")
    osm = _get_reader()

    custom_filter = {
        "amenity":          True,
        "shop":             True,
        "office":           True,
        "leisure":          True,
        "tourism":          True,
        "public_transport": True,
    }

    raw = osm.get_pois(custom_filter=custom_filter)
    if raw is None or raw.empty:
        print("  No POI features found in AOI bbox.")
        return _empty_gdf(_COLS)

    print(f"  Raw features in bbox: {len(raw):,}")

    # Keep only node geometries (Points) — ways/relations are out of scope
    raw = raw[raw.geometry.geom_type == "Point"].copy()
    print(f"  After filtering to Point geometries: {len(raw):,}")

    raw = _clip_to_aoi(raw)
    print(f"  After precise AOI clip: {len(raw):,}")

    if raw.empty:
        return _empty_gdf(_COLS)

    gdf = gpd.GeoDataFrame(
        {
            "osm_id":           _make_ids(raw),
            "geometry":         raw.geometry,
            "amenity":          _col(raw, "amenity"),
            "shop":             _col(raw, "shop"),
            "office":           _col(raw, "office"),
            "leisure":          _col(raw, "leisure"),
            "tourism":          _col(raw, "tourism"),
            "public_transport": _col(raw, "public_transport"),
            "name":             _col(raw, "name"),
            "last_edit":        _col(raw, "timestamp").apply(_fmt_ts),
        },
        geometry="geometry",
        crs=CRS_GEO,
    )

    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()

    print(f"  Built {len(gdf):,} POI points.")
    _save_layer(gdf, "osm_pois")
    return gdf
