"""
acquire/unhcr.py — Fetch CCCM displacement-site data from HDX (UNHCR).

Public functions
----------------
fetch_cccm_sites()              → GeoDataFrame of site point locations
build_site_polygons(sites_gdf)  → GeoDataFrame with polygon boundaries where available
"""

import sys
import warnings
from pathlib import Path
from typing import Optional

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (  # noqa: E402
    CRS_GEO,
    CRS_PROJ,
    INPUT_DIR,
    POINT_BUFFER_M,
    PROJECT_NAME,
    SCRATCH_GPKG,
)
from acquire.osm import _drop_gpkg_layer  # noqa: E402

# Input directory is resolved from config at import time (set by config.init()).
# This ensures the path reflects the working directory chosen at runtime rather
# than a path hardcoded relative to this source file.
_INPUT = INPUT_DIR

# ---------------------------------------------------------------------------
# Column-name normalisation maps
# ---------------------------------------------------------------------------

# Recognised variants for each canonical field (all compared case-insensitively).
# Covers both CCCM masterlist conventions and IOM DTM Site Assessment naming.
_LAT_VARIANTS = {
    "lat", "latitude", "y", "y_wgs84", "ylat", "lat_dd",
    # DTM variants
    "gps latitude", "gps_latitude",
    # DTM structured field codes
    "b11.gps.lat",
}
_LON_VARIANTS = {
    "lon", "lng", "long", "longitude", "x", "x_wgs84", "xlon", "lon_dd",
    # DTM variants
    "gps longitude", "gps_longitude",
    # DTM structured field codes
    "b10.gps.lon",
}
_SITE_ID_VARIANTS = {
    "site_id", "siteid", "site id", "id", "p_code", "pcode",
    # DTM variants
    "dtm id", "dtm_id", "location id", "location_id",
    # DTM structured field codes
    "b01.location.ssid",
}
_SITE_NAME_VARIANTS = {
    "site_name", "sitename", "site name", "name", "location name",
    # DTM variants
    "location_name",
    # DTM structured field codes
    "b02.location.name",
}
_SITE_TYPE_VARIANTS = {
    "site_type", "sitetype", "site type", "type", "settlement type",
    # DTM variants
    "type of site", "type_of_site", "settlement_type",
    # DTM structured field codes
    "b14.settlement.type",
}
_MGMT_STATUS_VARIANTS = {
    "management_status", "mgmt_status", "management status",
    "status", "operational status",
    # DTM variants
    "managed", "site management", "site_management",
    # DTM structured field codes
    "b39.site.management.agency",
}


def _find_col(df: pd.DataFrame, variants: set) -> Optional[str]:
    """Return the first column whose lower-cased name is in *variants*, or None."""
    lower_map = {c.lower().strip(): c for c in df.columns}
    for v in variants:
        if v in lower_map:
            return lower_map[v]
    return None


def _save_layer(gdf: gpd.GeoDataFrame, layer: str) -> None:
    """Write *gdf* to SCRATCH_GPKG as *layer*, overwriting if it exists."""
    SCRATCH_GPKG.parent.mkdir(parents=True, exist_ok=True)
    if SCRATCH_GPKG.exists():
        _drop_gpkg_layer(SCRATCH_GPKG, layer)
    gdf.to_file(str(SCRATCH_GPKG), driver="GPKG", layer=layer, mode="a")
    print(
        f"  Saved {len(gdf):,} features → "
        f"layer '{layer}' in {SCRATCH_GPKG.name}"
    )


# ---------------------------------------------------------------------------
# Internal: parse a flat tabular file (Excel or CSV) into a site GeoDataFrame
# ---------------------------------------------------------------------------

def _parse_masterlist(path: Path) -> gpd.GeoDataFrame:
    """
    Read an Excel or CSV masterlist and return a GeoDataFrame with columns:
    site_id, site_name, site_type, management_status, geometry (Point EPSG:4326).

    Latitude/longitude columns are detected automatically from common name variants.

    Raises
    ------
    ValueError
        If latitude or longitude columns cannot be found.
    """
    suffix = path.suffix.lower()
    if suffix in (".xlsx", ".xls"):
        df = pd.read_excel(path, dtype=str)
    elif suffix == ".csv":
        df = pd.read_csv(path, dtype=str)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

    # ------------------------------------------------------------------
    # Detect columns and print the resolved mapping
    # ------------------------------------------------------------------
    lat_col     = _find_col(df, _LAT_VARIANTS)
    lon_col     = _find_col(df, _LON_VARIANTS)
    id_col      = _find_col(df, _SITE_ID_VARIANTS)
    name_col    = _find_col(df, _SITE_NAME_VARIANTS)
    type_col    = _find_col(df, _SITE_TYPE_VARIANTS)
    mgmt_col    = _find_col(df, _MGMT_STATUS_VARIANTS)

    print(f"  Column mapping detected in {path.name}:")
    print(f"    latitude         → {lat_col!r}")
    print(f"    longitude        → {lon_col!r}")
    print(f"    site_id          → {id_col!r}  {'(will auto-generate DTM_ prefix)' if id_col is None else ''}")
    print(f"    site_name        → {name_col!r}")
    print(f"    site_type        → {type_col!r}")
    print(f"    management_status→ {mgmt_col!r}  {'(will default to Unknown)' if mgmt_col is None else ''}")

    if lat_col is None or lon_col is None:
        raise ValueError(
            f"Cannot find lat/lon columns in {path.name}. "
            f"Columns present: {list(df.columns)}"
        )

    # ------------------------------------------------------------------
    # Build geometry
    # ------------------------------------------------------------------
    lats = pd.to_numeric(df[lat_col], errors="coerce")
    lons = pd.to_numeric(df[lon_col], errors="coerce")
    geoms = [
        Point(lon, lat) if pd.notna(lon) and pd.notna(lat) else None
        for lon, lat in zip(lons, lats)
    ]

    # ------------------------------------------------------------------
    # Map canonical columns; apply fallbacks for optional fields
    # ------------------------------------------------------------------
    def _get_col(col_name):
        """Return the named column with NaN replaced by None."""
        if col_name is None:
            return pd.Series([None] * len(df), dtype=object, index=df.index)
        return df[col_name].where(df[col_name].notna(), other=None)

    # site_id: fall back to row-index-based DTM_ identifier
    if id_col is not None:
        site_ids = _get_col(id_col).astype(str)
    else:
        site_ids = pd.Series(
            [f"DTM_{i}" for i in range(len(df))],
            index=df.index,
            dtype=str,
        )

    # management_status: fall back to "Unknown" when column is absent
    if mgmt_col is not None:
        mgmt_vals = _get_col(mgmt_col).fillna("Unknown").astype(str)
    else:
        mgmt_vals = pd.Series(["Unknown"] * len(df), dtype=str, index=df.index)

    out = gpd.GeoDataFrame(
        {
            "site_id":           site_ids,
            "site_name":         _get_col(name_col).astype(str),
            "site_type":         _get_col(type_col).astype(str),
            "management_status": mgmt_vals,
            "geometry":          geoms,
        },
        geometry="geometry",
        crs=CRS_GEO,
    )

    # Drop rows with no geometry
    n_before = len(out)
    out = out.dropna(subset=["geometry"]).copy()
    n_dropped = n_before - len(out)
    if n_dropped:
        print(f"  Dropped {n_dropped:,} rows with missing coordinates.")

    return out


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def fetch_cccm_sites() -> gpd.GeoDataFrame:
    """
    Download the CCCM South Sudan site masterlist from HDX and return it as
    a GeoDataFrame of site point locations.

    Searches HDX for keyword "South Sudan CCCM site masterlist" under country
    ``ssd``.  Downloads the first Excel or CSV resource found to
    ``data/input/cccm_masterlist.xlsx`` (or ``.csv``).

    If the HDX download fails for any reason, falls back to
    ``data/input/cccm_masterlist_manual.xlsx`` if that file exists, allowing
    the user to place a manually downloaded copy as a fallback.

    Returns
    -------
    gpd.GeoDataFrame
        CRS: EPSG:4326.
        Columns: site_id (str), site_name (str), site_type (str),
        management_status (str), geometry (Point).

    Raises
    ------
    FileNotFoundError
        If neither HDX download nor the manual fallback file is available.
    """
    _INPUT.mkdir(parents=True, exist_ok=True)
    download_path: Optional[Path] = None

    # --- Attempt HDX download ---
    try:
        from hdx.api.configuration import Configuration
        from hdx.data.dataset import Dataset

        print("Connecting to HDX to find CCCM South Sudan site masterlist…")
        try:
            Configuration.create(
                hdx_site="prod",
                user_agent=f"landuse_ref_{PROJECT_NAME}",
                hdx_read_only=True,
            )
        except Exception:
            # Configuration already exists from a previous call — safe to continue
            pass

        datasets = Dataset.search_in_hdx(
            "South Sudan CCCM site masterlist",
            fq="groups:ssd",
            rows=5,
        )

        if not datasets:
            raise RuntimeError("No datasets returned by HDX search.")

        dataset = datasets[0]
        print(f"  Found dataset: {dataset.get('title', dataset['name'])}")

        resources = dataset.get_resources()
        resource = next(
            (r for r in resources if r.get_format().lower() in ("xlsx", "xls", "csv")),
            None,
        )
        if resource is None:
            raise RuntimeError("No Excel/CSV resource found in dataset.")

        fmt = resource.get_format().lower()
        dest_name = f"cccm_masterlist.{'xlsx' if 'xls' in fmt else 'csv'}"
        print(f"  Downloading resource '{resource.get('name')}' …")
        _, raw_path = resource.download(folder=str(_INPUT))
        download_path = Path(raw_path)
        # Rename to canonical name if needed
        canonical = _INPUT / dest_name
        if download_path != canonical:
            canonical.write_bytes(download_path.read_bytes())
            download_path = canonical
        print(f"  Downloaded → {download_path.name}")

    except Exception as hdx_exc:
        warnings.warn(
            f"HDX download failed: {hdx_exc}\n"
            "Checking for manual fallback at data/input/cccm_masterlist_manual.xlsx …"
        )
        fallback = _INPUT / "cccm_masterlist_manual.xlsx"
        if fallback.exists():
            print(f"  Using manual fallback: {fallback.name}")
            download_path = fallback
        else:
            raise FileNotFoundError(
                "HDX download failed and no manual fallback found.\n"
                "Place a CCCM site masterlist at:\n"
                f"  {_INPUT / 'cccm_masterlist_manual.xlsx'}\n"
                "and re-run fetch_cccm_sites()."
            ) from hdx_exc

    gdf = _parse_masterlist(download_path)
    print(f"  Parsed {len(gdf):,} sites with valid coordinates.")
    return gdf


def build_site_polygons(sites_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Attach polygon boundaries to CCCM site points where available, and buffer
    the remaining points to approximate site extents.

    Processing steps
    ~~~~~~~~~~~~~~~~
    1. Look for a polygon file at ``data/input/cccm_site_boundaries.*``
       (GeoPackage or Shapefile).  If found, left-join its geometries onto
       *sites_gdf* on ``site_id``.
    2. For sites without a polygon boundary, buffer the point geometry by
       ``POINT_BUFFER_M`` metres (reprojects to EPSG:32636, buffers, reprojects
       back to EPSG:4326).
    3. Adds a ``geometry_source`` column: ``"polygon"``,
       ``"point_buffered"``, or ``"point_buffered_osm"`` (when the boundary
       was cross-referenced from an OSM ``amenity=refugee_camp`` feature).

    Parameters
    ----------
    sites_gdf : gpd.GeoDataFrame
        Output of :func:`fetch_cccm_sites`.

    Returns
    -------
    gpd.GeoDataFrame
        CRS: EPSG:4326.  Same columns as input plus ``geometry_source``.
        Also written to layer ``cccm_sites`` in SCRATCH_GPKG.
    """
    gdf = sites_gdf.copy()
    gdf["geometry_source"] = "point_buffered"

    # --- Try to load polygon boundaries ---
    boundary_file: Optional[Path] = None
    for ext in ("*.gpkg", "*.shp", "*.geojson"):
        matches = list(_INPUT.glob(f"cccm_site_boundaries{ext[1:]}"))
        if matches:
            boundary_file = matches[0]
            break
    # Handle glob pattern correctly
    if boundary_file is None:
        for pattern in ("cccm_site_boundaries.gpkg",
                        "cccm_site_boundaries.shp",
                        "cccm_site_boundaries.geojson"):
            candidate = _INPUT / pattern
            if candidate.exists():
                boundary_file = candidate
                break

    if boundary_file is not None:
        print(f"  Loading polygon boundaries from {boundary_file.name}…")
        bounds_gdf = gpd.read_file(str(boundary_file))

        # Normalise site_id column in boundaries
        bid_col = _find_col(bounds_gdf, _SITE_ID_VARIANTS)
        if bid_col is None:
            warnings.warn(
                f"No site_id column found in {boundary_file.name}; "
                "skipping polygon join."
            )
        else:
            bounds_gdf = bounds_gdf.rename(columns={bid_col: "site_id"})
            bounds_gdf = bounds_gdf[["site_id", "geometry"]].to_crs(CRS_GEO)
            bounds_gdf = bounds_gdf.rename(columns={"geometry": "poly_geometry"})

            gdf = gdf.merge(bounds_gdf, on="site_id", how="left")

            poly_mask = gdf["poly_geometry"].notna()
            n_matched = poly_mask.sum()
            print(f"  Matched {n_matched:,} sites to polygon boundaries.")

            # Replace point geometry with polygon where available
            gdf.loc[poly_mask, "geometry"] = gdf.loc[poly_mask, "poly_geometry"]
            gdf.loc[poly_mask, "geometry_source"] = "polygon"
            gdf = gdf.drop(columns=["poly_geometry"])
            gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs=CRS_GEO)
    else:
        print("  No cccm_site_boundaries.* file found in data/input/ — buffering all points.")

    # --- Buffer remaining point geometries ---
    point_mask = gdf["geometry_source"] != "polygon"
    n_to_buffer = point_mask.sum()
    if n_to_buffer > 0:
        print(
            f"  Buffering {n_to_buffer:,} point sites by {POINT_BUFFER_M} m…"
        )
        point_rows = gdf[point_mask].copy()
        point_proj = point_rows.to_crs(CRS_PROJ)
        point_proj["geometry"] = point_proj.geometry.buffer(POINT_BUFFER_M)
        buffered_geo = point_proj.to_crs(CRS_GEO)
        gdf.loc[point_mask, "geometry"] = buffered_geo["geometry"].values
        gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs=CRS_GEO)

    print(
        f"  geometry_source breakdown:\n"
        + "\n".join(
            f"    {k}: {v}"
            for k, v in gdf["geometry_source"].value_counts().items()
        )
    )

    _save_layer(gdf, "cccm_sites")
    return gdf
