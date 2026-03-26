"""
acquire/overture.py — Fetch Overture Maps Places data for the study area.

Queries Overture's public S3 Parquet release directly via DuckDB + httpfs.
No CLI tool or local file download is required.

Public functions
----------------
fetch_overture_places()  → GeoDataFrame of place points in EPSG:4326
"""

import sys
from pathlib import Path

import duckdb
import geopandas as gpd
import pandas as pd
import shapely
import shapely.wkb
import shapely.wkt
from shapely import from_wkb

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (  # noqa: E402
    CRS_GEO,
    AOI_BBOX,
    OVERTURE_CONFIDENCE_MIN,
    SCRATCH_GPKG,
)

# Re-use the _drop_gpkg_layer / _save_layer helpers from osm.py
from acquire.osm import _drop_gpkg_layer  # noqa: E402

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_geom(val):
    """
    Convert a geometry value returned by DuckDB to a Shapely geometry.

    Handles all three formats that DuckDB may return:
      * bytes / bytearray  — raw WKB
      * str starting with '00' or '01'  — hex-encoded WKB
      * anything else (DuckDB geometry object, memoryview)  — attempt WKB
        conversion via bytes()
    """
    if val is None:
        return None
    try:
        if isinstance(val, (bytes, bytearray)):
            return shapely.wkb.loads(bytes(val))
        if isinstance(val, str) and val.startswith(("00", "01")):
            return shapely.wkb.loads(bytes.fromhex(val))
        return from_wkb(bytes(val))
    except Exception:
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
# Public fetch function
# ---------------------------------------------------------------------------

def fetch_overture_places() -> gpd.GeoDataFrame:
    """
    Download Overture Maps Places for the study-area bbox via DuckDB + httpfs.

    The release path is resolved dynamically at call time by querying the
    Overture STAC catalog (https://stac.overturemaps.org/catalog.json) so
    the function always targets the latest available release without any
    manual config updates.

    Filters applied
    ~~~~~~~~~~~~~~~
    * Bounding-box intersection using the Parquet ``bbox`` struct.
    * ``confidence >= OVERTURE_CONFIDENCE_MIN``.

    Returns
    -------
    gpd.GeoDataFrame
        CRS: EPSG:4326.
        Columns: id (str), geometry (Point), name (str|None),
        category_primary (str|None), category_alternate (str|None — pipe-separated list),
        confidence (float).
        Also written to layer ``overture_places`` in SCRATCH_GPKG.

    Raises
    ------
    RuntimeError
        If the STAC lookup or the S3 Parquet query fails.
    """
    west  = AOI_BBOX["west"]
    east  = AOI_BBOX["east"]
    south = AOI_BBOX["south"]
    north = AOI_BBOX["north"]

    print("Querying Overture Maps Places via DuckDB + S3…")

    con = duckdb.connect()
    try:
        con.execute("INSTALL httpfs; LOAD httpfs;")
        con.execute("SET s3_region = 'us-west-2';")
        con.execute("SET s3_use_ssl = true;")

        # ------------------------------------------------------------------
        # Step 1 — resolve the latest release tag from the Overture STAC
        # ------------------------------------------------------------------
        _S3_BASE = "s3://overturemaps-us-west-2/release/"

        print("  Resolving latest Overture release from STAC catalog…")
        try:
            con.execute("""
                SET VARIABLE latest = (
                    SELECT latest FROM 'https://stac.overturemaps.org/catalog.json'
                )
            """)
            release_tag = con.execute(
                "SELECT getvariable('latest')"
            ).fetchone()[0]
        except Exception as exc:
            raise RuntimeError(
                f"Failed to resolve Overture release from STAC catalog: {exc}"
            ) from exc

        release_path = f"{_S3_BASE}{release_tag}/theme=places/type=place/*"
        print(f"  Release tag:  {release_tag}")
        print(f"  Release path: {release_path}")

        if not release_path.startswith("s3://"):
            raise ValueError(
                f"Constructed Overture release path does not start with 's3://':\n"
                f"  {release_path}\n"
                f"The STAC catalog returned an unexpected release tag: {release_tag!r}"
            )

        # ------------------------------------------------------------------
        # Step 2 — query the Places Parquet files on S3
        # ------------------------------------------------------------------
        query = f"""
            SELECT
                id,
                names['primary']                           AS name,
                basic_category                             AS category_primary,
                array_to_string(categories['alternate'], '|') AS category_alternate,
                confidence,
                geometry
            FROM read_parquet('{release_path}', hive_partitioning = true)
            WHERE
                bbox.xmin BETWEEN {west} AND {east}
                AND bbox.ymin BETWEEN {south} AND {north}
                AND confidence >= {OVERTURE_CONFIDENCE_MIN}
        """

        try:
            df: pd.DataFrame = con.execute(query).df()
        except Exception as exc:
            raise RuntimeError(
                f"Overture S3 read failed for path:\n  {release_path}\n"
                f"Original error: {exc}"
            ) from exc

        # ------------------------------------------------------------------
        # Diagnostic — only runs when the main query returns 0 results.
        # Re-queries with no confidence filter and a 1-degree wider bbox to
        # distinguish between: low confidence scores, bbox mismatch, or
        # genuine absence of Overture data for this area.
        # Results are not saved — informational only.
        # ------------------------------------------------------------------
        if len(df) == 0:
            diag_query = f"""
                SELECT count(*) AS n
                FROM read_parquet('{release_path}', hive_partitioning = true)
                WHERE
                    bbox.xmin BETWEEN {west  - 1.0} AND {east  + 1.0}
                    AND bbox.ymin BETWEEN {south - 1.0} AND {north + 1.0}
            """
            try:
                diag_count = con.execute(diag_query).fetchone()[0]
            except Exception:
                diag_count = "unknown (diagnostic query failed)"
            print(
                f"  Diagnostic (no confidence filter, wider bbox): "
                f"{diag_count} records found."
            )
    finally:
        con.close()

    print(f"  Retrieved {len(df):,} rows from S3.")

    # Diagnostic — show raw geometry dtype and first value so format is visible
    if not df.empty:
        print(f"  geometry dtype: {df['geometry'].dtype}")
        print(f"  geometry[0] repr: {repr(df['geometry'].iloc[0])}")

    # Convert geometry column to Shapely geometries (handles bytes, hex-WKB, or DuckDB objects)
    df["geometry"] = df["geometry"].apply(_parse_geom)

    # Report parse results before dropping failures
    n_ok   = int(df["geometry"].notna().sum())
    n_fail = int(df["geometry"].isna().sum())
    print(f"  Geometry parsed OK: {n_ok:,}  |  failed (None): {n_fail:,}")

    # Drop rows where geometry could not be parsed
    n_before = len(df)
    df = df.dropna(subset=["geometry"])
    n_dropped = n_before - len(df)
    if n_dropped:
        print(f"  Dropped {n_dropped:,} rows with unparseable geometry.")

    if df.empty:
        gdf = gpd.GeoDataFrame(
            columns=[
                "id", "name", "category_primary", "category_alternate",
                "confidence", "geometry",
            ],
            geometry="geometry",
            crs=CRS_GEO,
        )
    else:
        gdf = gpd.GeoDataFrame(df, geometry="geometry", crs=CRS_GEO)

    print(f"  Built {len(gdf):,} place features.")
    _save_layer(gdf, "overture_places")
    return gdf
