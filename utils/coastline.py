"""Coastline loading utilities for Plotly maps.

Loads shapefiles from improverviz/CartopyFiles and converts to GeoJSON for Plotly.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import shapefile
from shapely.geometry import shape, mapping
from shapely.ops import unary_union
from pyproj import Transformer

import config


# IMPROVER Albers Equal Area projection string (for Australia)
ALBERS_PROJ4 = "+proj=aea +lat_0=-24.75 +lon_0=134.0 +lat_1=-10.0 +lat_2=-40.0 +x_0=0 +y_0=0 +ellps=GRS80 +units=m +no_defs"


# Module-level cache for loaded coastlines
_coastline_cache: Dict[str, Any] = {}


def load_shapefile(shapefile_path: Path) -> List[Dict]:
    """Load a shapefile and convert to GeoJSON features.

    Args:
        shapefile_path: Path to the .shp file

    Returns:
        List of GeoJSON feature dictionaries
    """
    features = []

    with shapefile.Reader(str(shapefile_path)) as shp:
        for sr in shp.shapeRecords():
            geom = shape(sr.shape.__geo_interface__)
            feature = {
                "type": "Feature",
                "geometry": mapping(geom),
                "properties": dict(zip([f[0] for f in shp.fields[1:]], sr.record)),
            }
            features.append(feature)

    return features


def load_coastlines(
    source: str = "gfe",
    simplify_tolerance: Optional[float] = None,
) -> Dict:
    """Load coastline data for Plotly display.

    Args:
        source: Which coastline source to use ("gfe" or "natural_earth")
        simplify_tolerance: Optional tolerance for geometry simplification

    Returns:
        GeoJSON FeatureCollection dict
    """
    cache_key = f"{source}_{simplify_tolerance}"

    if cache_key in _coastline_cache:
        return _coastline_cache[cache_key]

    if source == "gfe":
        shapefile_path = config.CARTOPY_PATH / "gfe_public_weather.shp"
    elif source == "natural_earth":
        shapefile_path = config.CARTOPY_PATH / "ne_10m_coastline.shp"
    else:
        raise ValueError(f"Unknown coastline source: {source}")

    if not shapefile_path.exists():
        raise FileNotFoundError(f"Shapefile not found: {shapefile_path}")

    features = load_shapefile(shapefile_path)

    # Optionally simplify geometries for better performance
    if simplify_tolerance:
        simplified_features = []
        for feature in features:
            geom = shape(feature["geometry"])
            simplified_geom = geom.simplify(simplify_tolerance)
            feature["geometry"] = mapping(simplified_geom)
            simplified_features.append(feature)
        features = simplified_features

    geojson = {
        "type": "FeatureCollection",
        "features": features,
    }

    _coastline_cache[cache_key] = geojson
    return geojson


def coastlines_to_plotly_traces(
    geojson: Dict,
    line_color: str = "black",
    line_width: float = 0.5,
) -> List[Dict]:
    """Convert GeoJSON coastlines to Plotly scattergeo trace data.

    Args:
        geojson: GeoJSON FeatureCollection
        line_color: Color for coastline lines
        line_width: Width of coastline lines

    Returns:
        List of trace dictionaries for Plotly
    """
    traces = []

    for feature in geojson["features"]:
        geom = feature["geometry"]
        geom_type = geom["type"]

        if geom_type == "LineString":
            coords = geom["coordinates"]
            lons = [c[0] for c in coords]
            lats = [c[1] for c in coords]
            traces.append({
                "lons": lons,
                "lats": lats,
            })

        elif geom_type == "MultiLineString":
            for line in geom["coordinates"]:
                lons = [c[0] for c in line]
                lats = [c[1] for c in line]
                traces.append({
                    "lons": lons,
                    "lats": lats,
                })

        elif geom_type == "Polygon":
            # For polygons, extract the exterior ring
            exterior = geom["coordinates"][0]
            lons = [c[0] for c in exterior]
            lats = [c[1] for c in exterior]
            traces.append({
                "lons": lons,
                "lats": lats,
            })

        elif geom_type == "MultiPolygon":
            for polygon in geom["coordinates"]:
                exterior = polygon[0]
                lons = [c[0] for c in exterior]
                lats = [c[1] for c in exterior]
                traces.append({
                    "lons": lons,
                    "lats": lats,
                })

    return traces


def get_coastline_scatter_data(
    source: str = "gfe",
    simplify_tolerance: Optional[float] = 0.01,
) -> tuple:
    """Get flattened lon/lat arrays for a single scattergeo trace.

    Uses None values to create breaks between separate line segments.

    Args:
        source: Coastline source
        simplify_tolerance: Simplification tolerance

    Returns:
        Tuple of (lons, lats) arrays with None separators
    """
    geojson = load_coastlines(source, simplify_tolerance)
    traces = coastlines_to_plotly_traces(geojson)

    all_lons = []
    all_lats = []

    for trace in traces:
        all_lons.extend(trace["lons"])
        all_lons.append(None)  # Break between segments
        all_lats.extend(trace["lats"])
        all_lats.append(None)

    return all_lons, all_lats


def clear_cache():
    """Clear the coastline cache."""
    global _coastline_cache
    _coastline_cache = {}


def get_coastline_scatter_data_albers(
    source: str = "gfe",
    simplify_tolerance: Optional[float] = 0.01,
    proj4str: Optional[str] = None,
) -> Tuple[List, List]:
    """Get coastline data transformed to Albers Equal Area projection.

    Args:
        source: Coastline source
        simplify_tolerance: Simplification tolerance
        proj4str: Optional proj4 string for the target projection. If None, uses default IMPROVER Albers.

    Returns:
        Tuple of (x_coords, y_coords) in km with None separators
    """
    # Use the projection in the cache key so different projections are cached separately
    target_proj = proj4str if proj4str else ALBERS_PROJ4
    # Create a short hash of the projection string for cache key
    proj_hash = hash(target_proj) % 10000
    cache_key = f"{source}_{simplify_tolerance}_albers_{proj_hash}"

    if cache_key in _coastline_cache:
        return _coastline_cache[cache_key]

    # Get lat/lon coastlines first
    lons, lats = get_coastline_scatter_data(source, simplify_tolerance)

    # Create transformer from WGS84 to the target projection
    transformer = Transformer.from_crs("EPSG:4326", target_proj, always_xy=True)

    # Transform coordinates
    x_coords = []
    y_coords = []

    for lon, lat in zip(lons, lats):
        if lon is None or lat is None:
            x_coords.append(None)
            y_coords.append(None)
        else:
            x, y = transformer.transform(lon, lat)
            # Convert from m to km for readability
            x_coords.append(x / 1000)
            y_coords.append(y / 1000)

    _coastline_cache[cache_key] = (x_coords, y_coords)
    return x_coords, y_coords
