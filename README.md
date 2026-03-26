<<<<<<< HEAD
# landuse_ref — Multi-Source Reference Land-Use Dataset for Urban Areas

## 1. Overview

`landuse_ref` is a Python pipeline that assembles a building-level reference land-use dataset by integrating three openly available geospatial data sources: OpenStreetMap (OSM), Overture Maps Places, and the IOM/UNHCR CCCM Displacement Site Masterlist. The pipeline is designed for data-scarce urban contexts — particularly cities in sub-Saharan Africa and other low-income regions where ground-truth land-use surveys are absent — and produces a labelled GeoPackage suitable for use as training data for spatial AI/ML classification models. Each building footprint in a user-supplied input layer is assigned a land-use class, a confidence tier, and full provenance metadata through a deterministic five-tier priority hierarchy that resolves conflicts between sources. The output is reproducible, city-agnostic, and fully documented at the feature level.

---

## 2. Data Sources

| Source | Theme | Coverage type | Licence | Access method | Known limitations |
|---|---|---|---|---|---|
| OpenStreetMap | Buildings, land-use zones, POI nodes | Point, Polygon | ODbL 1.0 | Geofabrik PBF extract (HTTP download, cached locally) | In HOT-mapped cities, building tags are often sparse or limited to `building=yes`, reducing the proportion of classifiable features; attribute completeness varies strongly by mapping campaign |
| Overture Maps Places | Commercial and civic point-of-interest features | Point | CDLA Permissive 2.0 | DuckDB + httpfs querying public S3 Parquet release | POI coverage in low-income urban areas is sparse; the dataset aggregates commercial data sources and therefore systematically under-represents informal markets, kiosks, and non-monetised community services |
| IOM DTM / CCCM Site Masterlist | Displacement sites (IDP camps, collective centres, spontaneous settlements) | Point (buffered to polygon) or Polygon | Open (HDX/OCHA) | HDX API (hdx-python-api); manual download fallback | Site boundaries are frequently represented as centroids only; the pipeline approximates extents by buffering the point geometry to a configurable radius (default 250 m), which may over- or under-estimate the true site boundary |

*Note on reliability:* In HOT-mapped urban areas, OSM provides the densest and most carefully quality-controlled data layer, but tag richness varies significantly between mapping campaigns and data collectors. Overture Maps Places provides a useful supplementary signal for formal commercial activity but cannot be treated as comprehensive in informal urban economies. The CCCM Masterlist is the most authoritative source for displacement-site identification but is updated episodically and may lag field conditions.

---

## 3. Land-Use Classification Schema

### 3.1 Tier Hierarchy

Labels are assigned through a five-tier priority hierarchy. A higher tier is only applied when no data is available from a higher-priority source. Tier 2+3 is a special cross-tier case: it is applied when a building matches both an OSM-derived class (Tier 2) and an Overture-derived class (Tier 3) that together constitute a recognised mixed-use combination.

| Tier | Source | Description |
|---|---|---|
| 1 | CCCM / IOM DTM | Building centroid falls within a known displacement-site boundary (highest confidence; overrides all lower tiers) |
| 2 | OSM building or POI tag | Building has an informative OSM tag on the building object or an associated POI node |
| 2+3 | OSM + Overture combined | OSM and Overture signals resolve to two classes that together constitute a Mixed Use trigger pair |
| 3 | Overture Maps Places | A place record with confidence ≥ threshold falls within the building footprint; no usable OSM tag was present |
| 4 | OSM landuse zone | Building centroid lies within a land-use zone polygon (e.g. `landuse=residential`); individual tag evidence absent |
| 5 | Unclassified | No source provides sufficient evidence to assign a class |

### 3.2 Classification Table

| lu_class | Example lu_subclass values | Primary assignment source | lu_confidence |
|---|---|---|---|
| Residential | *(none)*, Traditional, Multi-family, Temporary | OSM `building=residential/house/hut/apartments`, `landuse=residential` zone | high / medium / low |
| Commercial | Retail, Market, Finance, Hospitality, Food & Drink, Service, Entertainment | OSM `building=commercial/retail`, `shop=*`, `office=*`, `amenity=restaurant`, etc.; Overture `food_and_drink`, `retail`, `financial_services` | high / low (zone) |
| Industrial | Storage, Manufacturing, Utility, Extractive, Waste | OSM `building=industrial/warehouse/factory`, `landuse=industrial` | high / medium / low |
| Public/Institutional | Education, Health, Religious, Government, Community, Military, Cemetery | OSM `amenity=school/hospital/place_of_worship/police`, `building=government`, `landuse=institutional`; Overture `civic_and_social`, `health_and_medical`, `education` | high / medium / low |
| Transport | Airport, Road, Rail, Water | OSM `aeroway=*`, `amenity=bus_station`, `railway=station`, `public_transport=*`; Overture `transportation` | high / medium |
| Temporary Housing | *(none)* | OSM `amenity=refugee_camp`, `social_facility=shelter`; CCCM site boundary (Tier 1) | high / medium / low |
| Open Space/Environmental | Vegetation, Agricultural, Recreation, Water, Wetland | OSM `landuse=grass/forest/farmland`, `leisure=park/pitch`, `natural=wood/water` | high / medium |
| Mixed Use | *(pipe-separated component classes)* | Co-occurrence of Residential + Commercial, or other trigger pair (see `classify/lookup.py`) | medium |
| Unclassified | — | No source evidence; includes buildings with `building=yes` only | — |

Mixed Use is assigned when a building footprint accumulates evidence from two distinct classes that form a recognised trigger pair (e.g. Residential + Commercial). The component classes are preserved in the `lu_class_components` field. Mixed Use is also flagged via a supplementary POI signal (`lu_mixed_use_poi_signal`) when an OSM POI node lies within 5 m of the building exterior and implies a different class from the building tag.

---

## 4. Prerequisites

- Python 3.10 or later
- Git
- An active internet connection for the first run (the pipeline downloads the Geofabrik OSM PBF extract and queries Overture Maps directly over Amazon S3 via DuckDB; both are cached after the first run)

### Python dependencies

```
geopandas
shapely
pyproj
fiona
overpy
pyrosm
duckdb
hdx-python-api
pandas
tqdm
requests
```

All dependencies are listed in `requirements.txt` and are available from PyPI.

---

## 5. Installation

### Windows

```bat
git clone https://github.com/your-org/landuse_ref.git
cd landuse_ref
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### macOS / Linux

```bash
git clone https://github.com/your-org/landuse_ref.git
cd landuse_ref
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 6. Input Data

Two input files are required. Both must be placed or referenced before running the pipeline; no default paths are assumed.

### 6.1 Area of Interest (AOI) — required

A GeoJSON file containing one or more polygons defining the study area. All polygons are dissolved to a single geometry at runtime. Any projected or geographic CRS is accepted; the pipeline reprojects internally to EPSG:4326 for data exchange and EPSG:32636 (WGS 84 / UTM Zone 36N) for metric operations. If your study area falls outside UTM Zone 36N, verify that `CRS_PROJ` in `config.py` is appropriate for your region and update accordingly.

Accepted formats: GeoJSON (`.geojson`, `.json`). The file may contain additional attribute columns, which are ignored.

### 6.2 Building footprints — required

A polygon layer containing the building footprints to be classified. This is the primary unit of analysis; every output record corresponds to one footprint. Any CRS is accepted. Accepted formats: GeoPackage (`.gpkg`), Shapefile (`.shp`), GeoJSON, or any format readable by `geopandas.read_file()`.

Recommended sources: Microsoft Building Footprints, Google Open Buildings, OSM-derived footprints (via Geofabrik or Humanitarian Data Exchange).

### 6.3 CCCM site masterlist — optional manual fallback

The pipeline attempts to download the CCCM South Sudan Site Masterlist automatically from HDX at runtime. If the download fails (e.g. due to network restrictions or HDX API changes), place a manually downloaded copy at:

```
data/input/cccm_masterlist_manual.xlsx
```

The masterlist can be obtained from the Humanitarian Data Exchange (HDX) at [https://data.humdata.org](https://data.humdata.org) by searching for "South Sudan CCCM site masterlist", or from the IOM Displacement Tracking Matrix (DTM) at [https://dtm.iom.int](https://dtm.iom.int). Both Excel (`.xlsx`) and CSV formats are accepted. Column names are detected automatically from a broad set of recognised variants, including IOM DTM structured field codes (`b01.location.ssid`, `b11.gps.lat`, etc.).

Polygon boundary files for CCCM sites, if available, should be placed at:

```
data/input/cccm_site_boundaries.gpkg
```

(GeoJSON and Shapefile are also accepted.) If no boundary file is present, all site extents are approximated by buffering the point location by `POINT_BUFFER_M` metres (default: 250 m).

---

## 7. Usage

### 7.1 Basic run

```bash
python run.py \
  --aoi data/input/juba_aoi.geojson \
  --footprints data/input/building_footprints.gpkg
```

### 7.2 Full run with all options

```bash
python run.py \
  --aoi data/input/juba_aoi.geojson \
  --footprints data/input/building_footprints.gpkg \
  --project-name juba \
  --confidence 0.7
```

### 7.3 CLI flags

| Flag | Type | Required | Default | Description |
|---|---|---|---|---|
| `--aoi` | path | Yes | — | Path to the AOI GeoJSON file defining the study area |
| `--footprints` | path | Yes | — | Path to the building footprints file |
| `--project-name` | string | No | Stem of `--aoi` filename | Short label used in all output filenames (e.g. `juba` → `juba_landuse_reference.gpkg`) |
| `--confidence` | float | No | `0.7` | Minimum Overture Maps confidence score (0–1); records below this threshold are discarded during acquisition |
| `--skip-osm` | flag | No | off | Skip OSM acquisition and use cached layers from the scratch GeoPackage |
| `--skip-overture` | flag | No | off | Skip Overture acquisition and use the cached layer from the scratch GeoPackage |
| `--skip-cccm` | flag | No | off | Skip CCCM acquisition and use the cached layer from the scratch GeoPackage |

### 7.4 Incremental re-runs

After a full pipeline run, intermediate layers are cached in `data/{project_name}_scratch.gpkg`. The `--skip-*` flags allow individual acquisition steps to be bypassed, which is useful when iterating on the classification or join logic without re-downloading data.

```bash
# Re-run classification only — reuse all cached acquisition layers
python run.py \
  --aoi data/input/juba_aoi.geojson \
  --footprints data/input/building_footprints.gpkg \
  --project-name juba \
  --skip-osm --skip-overture --skip-cccm
```

```bash
# Re-run with a stricter Overture confidence threshold — reuse OSM and CCCM
python run.py \
  --aoi data/input/juba_aoi.geojson \
  --footprints data/input/building_footprints.gpkg \
  --project-name juba \
  --confidence 0.85 \
  --skip-osm --skip-cccm
```

If a `--skip-*` flag is supplied but the corresponding cached layer is not found in the scratch GeoPackage, the pipeline logs a warning and runs that acquisition step regardless.

---

## 8. Output Schema

The pipeline writes a single GeoPackage at `data/{project_name}_landuse_reference.gpkg` containing the layer `reference_buildings` in EPSG:4326. A companion summary CSV is written to `data/{project_name}_landuse_reference_summary.csv`.

| Field | Type | Description |
|---|---|---|
| `footprint_id` | String | Identifier carried over from the input footprints file; auto-generated UUID if no recognised ID column is present |
| `geometry` | Polygon / MultiPolygon | Building footprint geometry in EPSG:4326 |
| `lu_class` | String / null | Primary land-use class (see Section 3); null for Tier 5 (Unclassified) |
| `lu_subclass` | String / null | Secondary classification within the primary class; null when not applicable |
| `lu_tier` | String | Tier assignment: `"1"`, `"2"`, `"2+3"`, `"3"`, `"4"`, or `"5"` |
| `lu_source` | String / null | Pipe-separated list of source identifiers that contributed to the label (e.g. `osm_building\|overture`) |
| `lu_confidence` | String / null | Confidence level of the assigned class: `"high"`, `"medium"`, or `"low"` |
| `lu_class_components` | String / null | Pipe-separated component classes for Mixed Use features (e.g. `"Residential\|Commercial"`); null for all other classes |
| `lu_mixed_use_poi_signal` | Boolean / null | `True` if a supplementary OSM POI node within 5 m of the building exterior implies a different class from the building tag |
| `osm_building_id` | String / null | OSM element identifier of the matched building feature (prefixed `w` for way, `r` for relation) |
| `osm_building_tag` | String / null | Value of the primary OSM tag used for classification (e.g. `amenity=school`) |
| `osm_last_edit` | String / null | ISO 8601 timestamp of the last OSM edit to the matched building feature |
| `overture_category` | String / null | Overture Maps `basic_category` value of the matched place record |
| `overture_confidence` | Float / null | Overture Maps confidence score of the matched place record (0–1) |
| `cccm_site_id` | String / null | CCCM / IOM DTM site identifier of the matched displacement site |
| `cccm_site_type` | String / null | Site type as reported in the CCCM masterlist (e.g. `Collective Centre`, `Spontaneous Settlement`) |
| `cccm_geometry_source` | String / null | How the CCCM site boundary was derived: `"polygon"` (from supplied boundary file), `"point_buffered"` (centroid + buffer), or null |
| `osm_landuse` | String / null | OSM `landuse=*` zone value if the building centroid falls within an OSM land-use polygon (Tier 4 signal; present even for buildings classified at a higher tier) |

---

## 9. Validation

The pipeline includes three non-blocking validation checkpoints that run automatically and print structured reports to standard output.

**Checkpoint 1 — Acquisition (`validate_acquisition`)** runs after all three data sources have been fetched. It verifies that each expected scratch layer exists and contains features, reports geometry types, CRS, bounding box, and null geometry counts, and confirms that every layer's spatial extent intersects the AOI. An empty `overture_places` layer triggers a warning rather than an error, since sparse Overture coverage is common in informal urban contexts; the pipeline continues using OSM and CCCM only. All other empty layers raise a `ValueError` and halt the pipeline.

**Checkpoint 2 — Join results (`validate_joins`)** runs after the spatial join and label assignment step. It reports the distribution of features across all five tiers and all land-use classes, flags if more than 70 % of buildings are unclassified (Tier 5), and notes the absence of Mixed Use assignments or Tier 1 (CCCM) matches. These flags indicate either data sparsity or a possible configuration error and should be reviewed before using the output as training data.

**Checkpoint 3 — Spatial distribution (`validate_spatial_distribution`)** computes, for each class with more than ten buildings, the class centroid in geographic coordinates and the mean nearest-neighbour distance between building centroids. This provides a coarse indicator of spatial clustering. For contexts containing displacement sites, it additionally checks whether the Temporary Housing centroid is more than 5 km from the overall urban centroid, which is the expected pattern for peripheral camp sites; proximity to the general urban mass may indicate mis-assignment.

---

## 10. Known Limitations and Assumptions

**OSM attribute completeness.** In cities mapped primarily through Humanitarian OpenStreetMap Team (HOT) activations, building digitisation typically precedes attribute tagging. A large proportion of buildings may carry only `building=yes`, providing no information to the classifier and resulting in Tier 5 assignment. The unclassified fraction therefore reflects the state of OSM attribute completeness at the time of data extraction, not necessarily the underlying land-use diversity. Users should treat the Tier 5 proportion as an indicator of data quality rather than a property of the study area.

**Overture Maps POI sparsity.** The Overture Maps Places dataset aggregates commercial data providers including Foursquare, Meta, and Microsoft. These providers have systematically lower coverage in low-income urban areas, informal economies, and cities outside North America and Western Europe. Confidence scores are computed by Overture internally and reflect data-source agreement rather than ground-truth accuracy. The pipeline applies a minimum confidence threshold (default 0.7), but even high-confidence records may represent the formal commercial sector only. The Tier 3 classification signal should be interpreted as indicative, not comprehensive.

**CCCM geometry approximation.** CCCM and IOM DTM site masterlists predominantly record site locations as a single centroid point rather than a surveyed polygon boundary. Where no boundary file is provided, the pipeline buffers this point by a fixed radius (default 250 m) to approximate the site footprint. This approximation does not account for site shape, topography, or actual extent; it will over-include buildings outside camp boundaries and may under-include buildings in large or irregularly shaped sites. Users with access to surveyed site boundaries should provide them as `data/input/cccm_site_boundaries.gpkg` to replace this approximation.

**Temporal inconsistency between sources.** The three data sources are extracted at different points in time and are updated on different schedules. OSM edits are continuous; Geofabrik extracts are updated daily but the pipeline caches a downloaded PBF and does not re-download unless the cache is cleared. Overture Maps releases a new snapshot approximately monthly; the pipeline queries the latest release at run time. CCCM masterlists are updated episodically. Buildings constructed, demolished, or repurposed between data extract dates will be misclassified or absent. The `osm_last_edit` field provides a partial indicator of OSM data currency at the feature level, but no equivalent timestamp is available from Overture or CCCM.

**Inability to distinguish residential subtypes without morphological analysis.** The classification schema includes a Residential class with subclasses (e.g. Traditional, Multi-family) but these depend on explicit OSM tags that are frequently absent. The pipeline cannot infer residential subtype from building footprint morphology alone. Informal settlements, compound housing, and traditional structures are generally recorded as `building=yes` in HOT-mapped datasets and therefore receive Tier 5 labels rather than a specific residential subtype. This is a structural limitation of the tag-based approach and cannot be resolved without additional morphological or satellite imagery features.

**Mixed Use as an undercount.** Mixed Use assignment requires co-occurrence of evidence from two distinct sources or the simultaneous presence of multiple OSM tag signals on a single building. In data-scarce contexts, the majority of mixed-use buildings carry insufficient tagging for this pattern to be detected. The `Mixed Use` class and the `lu_mixed_use_poi_signal` flag should therefore be understood as lower bounds on actual mixed-use prevalence; the true proportion is likely substantially higher in dense urban markets.

**Non-random spatial pattern of unclassified buildings.** Tier 5 (Unclassified) buildings are not spatially random. In HOT-mapped cities, attribute completeness is typically higher in central areas mapped by experienced contributors and lower in peripheral or informally settled zones mapped later. The spatial distribution of unclassified buildings therefore correlates with mapping completeness, which in turn correlates with urban morphology and socioeconomic gradients. Any model trained on this dataset should account for this spatial structure in its validation design; random train/test splits that ignore spatial autocorrelation will overestimate generalisation accuracy.

---

## 11. Citation and Acknowledgements

### Citation

If you use this pipeline or the datasets it produces in published research, please cite:

> Gevaert, C. M. (2025). *landuse_ref: A multi-source reference land-use dataset pipeline for building-level classification in data-scarce urban contexts* [Software]. GitHub. https://github.com/your-org/landuse_ref

### Acknowledgements

This pipeline relies on data and infrastructure maintained by the following organisations:

**OpenStreetMap contributors** — The OpenStreetMap dataset is made available under the Open Database Licence (ODbL) 1.0. Building, land-use, and POI data are accessed via Geofabrik GmbH regional extracts. © OpenStreetMap contributors.

**Overture Maps Foundation** — Places data are sourced from the Overture Maps Foundation open data release, available under the Community Data Licence Agreement — Permissive 2.0. Overture Maps Foundation is a project of the Linux Foundation.

**IOM Displacement Tracking Matrix (DTM)** — Displacement site locations and attributes are sourced from the IOM DTM Site Assessment and CCCM Cluster South Sudan Site Masterlist, distributed via the Humanitarian Data Exchange (HDX). © International Organization for Migration.

**OCHA / Humanitarian Data Exchange** — Displacement-related datasets are distributed through the UN OCHA Humanitarian Data Exchange platform (data.humdata.org). OCHA is not responsible for the content or analysis presented in outputs produced by this pipeline.
=======
# urban-landuse-reference
Code to generate reference data for land use classification in cities. This is done by taking data from: OSM, Overture, and CCCM for refugee campls.
>>>>>>> 142125d023ae119d0cad307ff0001e4347a07104
