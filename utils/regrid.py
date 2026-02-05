"""Regridding utilities for converting IMPROVER projections to lat/lon.

Transforms data coordinates directly to WGS84 lat/lon without interpolation.
"""

import numpy as np
import xarray as xr
from pyproj import CRS, Transformer
from typing import Tuple, Optional

import config


def regrid_to_latlon(
    data_loader,
    grid: Optional[xr.DataArray] = None,
) -> xr.DataArray:
    """Transform data from IMPROVER Albers projection to WGS84 lat/lon.

    This function transforms the native coordinates to lat/lon without
    interpolation, preserving all data values.

    Args:
        data_loader: DataLoader instance with projection info
        grid: Optional DataArray to transform (defaults to data_loader.data)

    Returns:
        DataArray with WGS84 lat/lon coordinates
    """
    # Use provided grid or default to data_loader.data
    if grid is None:
        grid = data_loader.data

    # Get the source CRS
    if data_loader.proj4str:
        source_crs = CRS(data_loader.proj4str)
    else:
        # Fallback to default Australian Albers projection
        source_crs = CRS(
            "+proj=aea +lat_1=-18 +lat_2=-36 +lat_0=0 +lon_0=132 "
            "+x_0=0 +y_0=0 +ellps=GRS80 +units=m +no_defs"
        )

    wgs84_crs = CRS("EPSG:4326")

    # Create transformer
    transformer = Transformer.from_crs(source_crs, wgs84_crs, always_xy=True)

    # Get original coordinates
    x_coord_name = None
    y_coord_name = None

    for name in ["projection_x_coordinate", "x"]:
        if name in grid.coords:
            x_coord_name = name
            break

    for name in ["projection_y_coordinate", "y"]:
        if name in grid.coords:
            y_coord_name = name
            break

    if x_coord_name is None or y_coord_name is None:
        raise ValueError("Cannot find x/y coordinates in data")

    x_coords = grid.coords[x_coord_name].values
    y_coords = grid.coords[y_coord_name].values

    # Create 2D meshgrid of original coordinates
    x_2d, y_2d = np.meshgrid(x_coords, y_coords)

    # Transform all points to lat/lon
    lon_2d, lat_2d = transformer.transform(x_2d, y_2d)

    # Get the data values
    values = grid.values.copy()

    # Handle extra dimensions (squeeze if needed)
    if values.ndim > 2:
        values = np.squeeze(values)

    # Create new DataArray with lat/lon coordinates
    # For plotting with Plotly heatmap, we need 1D coordinate arrays
    # Use the center row/column to get representative lat/lon values
    center_row = lat_2d.shape[0] // 2
    center_col = lon_2d.shape[1] // 2

    # Get 1D arrays from the 2D transformed coordinates
    # Use actual transformed values along center lines
    lon_1d = lon_2d[center_row, :]
    lat_1d = lat_2d[:, center_col]

    # Create the output DataArray
    result = xr.DataArray(
        values,
        dims=["lat", "lon"],
        coords={
            "lat": lat_1d,
            "lon": lon_1d,
        },
        attrs=grid.attrs.copy(),
    )

    # Store the full 2D lat/lon grids as data variables for reference
    result.attrs["lon_2d"] = lon_2d
    result.attrs["lat_2d"] = lat_2d

    return result


def get_latlon_bounds(data_loader) -> Tuple[float, float, float, float]:
    """Get the lat/lon bounds of the data.

    Args:
        data_loader: DataLoader instance

    Returns:
        Tuple of (lon_min, lon_max, lat_min, lat_max)
    """
    regridded = regrid_to_latlon(data_loader)
    return (
        float(regridded.lon.min()),
        float(regridded.lon.max()),
        float(regridded.lat.min()),
        float(regridded.lat.max()),
    )


def create_latlon_grid(data_loader) -> Tuple[np.ndarray, np.ndarray]:
    """Create a lat/lon meshgrid for the data domain.

    Args:
        data_loader: DataLoader instance

    Returns:
        Tuple of (lon_grid, lat_grid) 2D arrays
    """
    regridded = regrid_to_latlon(data_loader)
    return np.meshgrid(regridded.lon.values, regridded.lat.values)


# Keep GFEGrid class for backward compatibility but it's not used by default
class GFEGrid:
    """GFE Grid class for defining the target Mercator grid.

    Note: This is kept for reference but not used by the default regridding.
    """

    @staticmethod
    def axes_from_origin_and_extent(
        origin: Tuple[float, float],
        extent: Tuple[float, float],
        size: Tuple[int, int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        x_values = np.linspace(origin[0], origin[0] + extent[0], size[0])
        y_values = np.linspace(origin[1], origin[1] + extent[1], size[1])
        return x_values, y_values

    @staticmethod
    def get_axes() -> Tuple[np.ndarray, np.ndarray]:
        return GFEGrid.axes_from_origin_and_extent(
            config.GFE_ORIGIN, config.GFE_EXTENT, config.GFE_SIZE
        )
