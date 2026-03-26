"""
run.py — CLI entry point for the landuse_ref pipeline.

Usage
-----
  python run.py --aoi data/input/juba_aoi.geojson --footprints data/input/building_footprints.gpkg
  python run.py --aoi ... --footprints ... --project-name juba --confidence 0.7
  python run.py --aoi ... --footprints ... --skip-osm
  python run.py --aoi ... --footprints ... --skip-osm --skip-overture --skip-cccm
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _elapsed(start: float) -> str:
    """Format seconds since *start* as 'Xm Ys' or 'Ys'."""
    secs = int(time.perf_counter() - start)
    m, s = divmod(secs, 60)
    return f"{m}m {s}s" if m else f"{s}s"


def _has_layers(gpkg_path: Path, layer_names: list) -> bool:
    """Return True only if *gpkg_path* exists and contains every named layer."""
    if not gpkg_path.exists():
        return False
    try:
        import fiona
        existing = set(fiona.listlayers(str(gpkg_path)))
        return all(name in existing for name in layer_names)
    except Exception:
        return False


def _step(name: str, func, *args, **kwargs):
    """
    Run *func* as a named pipeline step.
    Prints elapsed time on success; re-raises with step name on failure.
    """
    print(f"\n{'─' * 60}")
    print(f"  STEP: {name}")
    print(f"{'─' * 60}")
    t = time.perf_counter()
    try:
        result = func(*args, **kwargs)
    except Exception as exc:
        print(f"\n[FAIL] Step '{name}' raised: {type(exc).__name__}: {exc}")
        raise
    print(f"\n  ✓ {name} completed in {_elapsed(t)}")
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run.py",
        description=(
            "Run the full landuse_ref pipeline: "
            "acquire → preprocess → classify → export."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Required inputs
    parser.add_argument(
        "--aoi",
        required=True,
        metavar="PATH",
        help=(
            "Path to the GeoJSON file defining the study-area polygon(s). "
            "Any CRS is accepted and will be reprojected to EPSG:4326."
        ),
    )
    parser.add_argument(
        "--footprints",
        required=True,
        metavar="PATH",
        help=(
            "Path to the building footprints file "
            "(GeoPackage, Shapefile, GeoJSON, etc.)."
        ),
    )

    # Optional tuning
    parser.add_argument(
        "--project-name",
        default=None,
        metavar="NAME",
        help=(
            "Short label used in output filenames "
            "(e.g. 'juba' produces juba_landuse_reference.gpkg). "
            "Defaults to the stem of the --aoi filename."
        ),
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.7,
        metavar="FLOAT",
        help="Minimum Overture Maps confidence threshold (0–1). Default: 0.7.",
    )

    # Skip flags
    parser.add_argument(
        "--skip-osm",
        action="store_true",
        help=(
            "Skip OSM data download and use cached layers "
            "(osm_buildings, osm_landuse, osm_pois) from the scratch GeoPackage."
        ),
    )
    parser.add_argument(
        "--skip-overture",
        action="store_true",
        help=(
            "Skip Overture download and use cached layer "
            "(overture_places) from the scratch GeoPackage."
        ),
    )
    parser.add_argument(
        "--skip-cccm",
        action="store_true",
        help=(
            "Skip CCCM download and use cached layer "
            "(cccm_sites) from the scratch GeoPackage."
        ),
    )
    return parser


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    pipeline_start = time.perf_counter()

    # ------------------------------------------------------------------
    # Initialise config BEFORE importing any other pipeline module.
    # All derived constants (SCRATCH_GPKG, OUTPUT_GPKG, AOI_GEOM, …)
    # are set here; lazy imports below will see the correct values.
    # ------------------------------------------------------------------
    import config
    config.init(
        aoi_path=args.aoi,
        footprints_path=args.footprints,
        project_name=args.project_name,
        confidence_min=args.confidence,
    )

    from config import FOOTPRINTS_PATH, PROJECT_NAME, SCRATCH_GPKG  # noqa: E402

    print("=" * 60)
    print(f"  landuse_ref — reference land-use classification ({PROJECT_NAME})")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Lazy imports (keep startup fast; also ensures config.init() has run
    # before any module-level constants are bound)
    # ------------------------------------------------------------------
    from acquire.osm import fetch_osm_buildings, fetch_osm_landuse, fetch_osm_pois
    from acquire.overture import fetch_overture_places
    from acquire.unhcr import build_site_polygons, fetch_cccm_sites
    from export import export_reference_dataset
    from join import run_joins
    from preprocess.align import align_layers
    from validate import (
        validate_acquisition,
        validate_joins,
        validate_spatial_distribution,
    )

    # ------------------------------------------------------------------
    # Step 1 — OSM acquisition
    # ------------------------------------------------------------------
    _OSM_LAYERS = ["osm_buildings", "osm_landuse", "osm_pois"]

    if args.skip_osm and _has_layers(SCRATCH_GPKG, _OSM_LAYERS):
        print(f"\n[SKIP] OSM acquisition — using cached layers in {SCRATCH_GPKG.name}")
    else:
        if args.skip_osm:
            print(
                f"\n[WARN] --skip-osm requested but cached layers not found in "
                f"{SCRATCH_GPKG.name}. Running OSM acquisition."
            )

        def _run_osm():
            fetch_osm_buildings()
            fetch_osm_landuse()
            fetch_osm_pois()

        _step("OSM acquisition", _run_osm)

    # ------------------------------------------------------------------
    # Step 2 — Overture acquisition
    # ------------------------------------------------------------------
    _OVERTURE_LAYERS = ["overture_places"]

    if args.skip_overture and _has_layers(SCRATCH_GPKG, _OVERTURE_LAYERS):
        print(f"\n[SKIP] Overture acquisition — using cached layer in {SCRATCH_GPKG.name}")
    else:
        if args.skip_overture:
            print(
                f"\n[WARN] --skip-overture requested but cached layer not found. "
                "Running Overture acquisition."
            )
        _step("Overture acquisition", fetch_overture_places)

    # ------------------------------------------------------------------
    # Step 3 — CCCM acquisition + site polygon construction
    # ------------------------------------------------------------------
    _CCCM_LAYERS = ["cccm_sites"]

    if args.skip_cccm and _has_layers(SCRATCH_GPKG, _CCCM_LAYERS):
        print(f"\n[SKIP] CCCM acquisition — using cached layer in {SCRATCH_GPKG.name}")
    else:
        if args.skip_cccm:
            print(
                "\n[WARN] --skip-cccm requested but cached layer not found. "
                "Running CCCM acquisition."
            )

        def _run_cccm():
            sites = fetch_cccm_sites()
            build_site_polygons(sites)

        _step("CCCM acquisition + site polygons", _run_cccm)

    # ------------------------------------------------------------------
    # Checkpoint 1 — Validate acquired layers
    # (raises ValueError for empty/missing layers; otherwise non-blocking)
    # ------------------------------------------------------------------
    _step("Validation: acquisition", validate_acquisition)

    # ------------------------------------------------------------------
    # Step 4 — Layer alignment / preprocessing
    # ------------------------------------------------------------------
    layers = _step("Layer alignment", align_layers, FOOTPRINTS_PATH)

    # ------------------------------------------------------------------
    # Step 5 — Spatial joins + label assignment
    # ------------------------------------------------------------------
    labelled = _step("Spatial joins + label assignment", run_joins, layers)

    # ------------------------------------------------------------------
    # Checkpoints 2 & 3 — Validate joins and spatial distribution
    # Both are fully non-blocking: printed warnings only, never raise.
    # ------------------------------------------------------------------
    for _val_name, _val_fn in [
        ("Validation: join results",         lambda: validate_joins(labelled)),
        ("Validation: spatial distribution", lambda: validate_spatial_distribution(labelled)),
    ]:
        try:
            _step(_val_name, _val_fn)
        except Exception as _val_exc:  # pragma: no cover
            print(f"\n[WARN] {_val_name} raised an unexpected error and was skipped: {_val_exc}")

    # ------------------------------------------------------------------
    # Step 6 — Export
    # ------------------------------------------------------------------
    _step("Export reference dataset", export_reference_dataset, labelled)

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    print("=" * 60)
    print(f"  Pipeline complete — total elapsed: {_elapsed(pipeline_start)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
