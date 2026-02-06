"""Data loader for IMPROVER NetCDF files, adapted from improverviz.retrievegrid."""

import numpy as np
import pandas as pd
import xarray as xr
from typing import Optional, Dict, Any, Tuple

import config

# Prevent issues with attrs being lost with every computation
xr.set_options(keep_attrs=True)

# Parameter frequency mapping
PARAM_FREQUENCY = {
    "precipacc24h": "24h",
    "precipacc03h": "3h",
    "tempscreen": "instantaneous",
    "tempdaymax": "24h",
    "tempnightmin": "24h",
    "tempdewscreen": "instantaneous",
    "windspd10m": "instantaneous",
    "winddir10m": "instantaneous",
}


class DataLoader:
    """Load and process IMPROVER NetCDF forecast data.

    Adapted from improverviz.retrievegrid.ImproverGrid class.
    """

    def __init__(
        self,
        source: str,
        parameter: str,
        validtime: Optional[str] = None,
        name: str = "forecast",
    ):
        """Initialize the data loader.

        Args:
            source: Path to the NetCDF file
            parameter: Parameter name (e.g., "precipacc24h")
            validtime: Optional valid time to select from merged files
            name: Optional name for the forecast
        """
        self.source = source
        self.parameter = parameter
        self.name = name

        # Load the dataset (decode_timedelta=False to avoid FutureWarning)
        self.dataset = xr.open_dataset(source, decode_timedelta=False)

        # If merged file with multiple times, select the specified validtime
        if "time" in self.dataset.dims and validtime:
            self.dataset = self._select_time(validtime)

        # Extract projection info
        self.proj_info = self._get_projection_info()
        self.proj4str = self._build_proj4str()

        # Get metadata
        self.basetime = self._get_basetime()
        self.validtime = self._get_validtime()
        self.leadhour = self._calculate_leadhour()
        self.frequency = PARAM_FREQUENCY.get(parameter, "instantaneous")

        # Get forecast bounds if available
        if "time_bnds" in self.dataset.variables:
            self.fcst_bnds = self.dataset.time_bnds.values
        else:
            self.fcst_bnds = [self.validtime, self.validtime]

        # Extract the main variable
        self.variable_name = self._get_main_variable()
        self.data = self.dataset[self.variable_name]

        # Determine data type
        self.data_type = self._get_data_type()

        # Apply unit conversions
        self.data = self._adjust_units(self.data)
        self.units = self.data.attrs.get("units", "")

    def _select_time(self, validtime: str) -> xr.Dataset:
        """Select a specific time from a merged dataset."""
        try:
            vt = pd.to_datetime(validtime)
            return self.dataset.sel(time=vt, method="nearest")
        except Exception:
            return self.dataset.isel(time=0)

    def _get_projection_info(self) -> Dict[str, Any]:
        """Extract projection information from the dataset."""
        # Look for common projection variable names
        proj_vars = [
            "albers_conical_equal_area",
            "projection",
            "crs",
            "spatial_ref",
        ]

        for var in proj_vars:
            if var in self.dataset:
                return dict(self.dataset[var].attrs)

        # Return empty dict if no projection found
        return {}

    def _build_proj4str(self) -> str:
        """Build proj4 string from projection info."""
        if not self.proj_info:
            return ""

        # Handle Albers Equal Area projection
        if "standard_parallel" in self.proj_info:
            components = {
                "proj": "aea",
                "lat_1": self.proj_info.get("standard_parallel", [0, 0])[0],
                "lat_2": self.proj_info.get("standard_parallel", [0, 0])[1],
                "lat_0": self.proj_info.get("latitude_of_projection_origin", 0),
                "lon_0": self.proj_info.get("longitude_of_central_meridian", 0),
                "x_0": self.proj_info.get("false_easting", 0),
                "y_0": self.proj_info.get("false_northing", 0),
                "a": self.proj_info.get("semi_major_axis", 6378137),
                "b": self.proj_info.get("semi_minor_axis", 6356752.314),
            }
            parts = [f"+{key}={value}" for key, value in components.items()]
            return " ".join(parts)

        return ""

    def _get_basetime(self) -> np.datetime64:
        """Get the forecast reference time."""
        if "forecast_reference_time" in self.dataset:
            return self.dataset.forecast_reference_time.values
        return np.datetime64("NaT")

    def _get_validtime(self) -> np.datetime64:
        """Get the forecast valid time."""
        if "time" in self.dataset:
            return self.dataset.time.values
        return np.datetime64("NaT")

    def _calculate_leadhour(self) -> Optional[int]:
        """Calculate lead hour from basetime and validtime."""
        try:
            bt = pd.to_datetime(self.basetime)
            vt = pd.to_datetime(self.validtime)
            return int((vt - bt).total_seconds() / 3600)
        except Exception:
            return None

    def _get_main_variable(self) -> str:
        """Identify the main data variable in the dataset."""
        all_vars = list(self.dataset.data_vars)

        # First, try parameter-specific lookup from config
        expected_var = config.PARAMETER_VARIABLE_NAMES.get(self.parameter)
        if expected_var:
            # Exact match
            if expected_var in all_vars:
                return expected_var
            # Partial match (variable name contains expected)
            for var in all_vars:
                if expected_var in var:
                    return var

        # Fallback to original heuristic
        exclude_patterns = [
            "projection",
            "time",
            "time_bnds",
            "forecast",
            "albers",
            "crs",
            "spatial_ref",
        ]

        pick_vars = [
            v
            for v in all_vars
            if not any(pattern in v.lower() for pattern in exclude_patterns)
        ]

        if len(pick_vars) == 1:
            return pick_vars[0]
        elif len(pick_vars) > 1:
            # Return first match
            return pick_vars[0]
        else:
            raise ValueError(f"Cannot determine main variable from: {all_vars}")

    def _get_data_type(self) -> str:
        """Identify the type of output: expected values, percentiles, or probabilities."""
        if "percentile" in self.data.dims:
            return "percentile"
        elif "threshold" in self.data.dims:
            return "threshold"
        else:
            return "expectedvalue"

    def _adjust_units(self, data: xr.DataArray) -> xr.DataArray:
        """Adjust units to user-friendly values.

        Converts:
        - Probabilities: 0-1 to 0-100%
        - Precipitation: meters to mm
        - Temperature: Kelvin to Celsius
        - Wind speed: m/s to knots
        """
        var_name = data.name.lower() if data.name else ""
        original_attrs = dict(data.attrs)

        if "probability" in var_name:
            data = data * 100
            data.attrs.update(original_attrs)
            data.attrs["units"] = "%"
            return data

        if "precipitation" in var_name:
            data = data * 1000
            data.attrs.update(original_attrs)
            data.attrs["units"] = "mm"
            return data

        if "temperature" in var_name or "dew_point" in var_name or "dewpoint" in var_name:
            data = data - 273.15
            data.attrs.update(original_attrs)
            data.attrs["units"] = "°C"
            return data

        if "wind" in var_name and "direction" not in var_name:
            # Convert m/s to knots
            data = data * 1.943844
            data.attrs.update(original_attrs)
            data.attrs["units"] = "knots"
            return data

        return data

    def get_coordinates(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get the x and y coordinates of the data.

        Returns:
            Tuple of (x_coords, y_coords) arrays
        """
        x_coord_names = ["projection_x_coordinate", "x", "lon", "longitude"]
        y_coord_names = ["projection_y_coordinate", "y", "lat", "latitude"]

        x_coords = None
        y_coords = None

        for name in x_coord_names:
            if name in self.data.coords:
                x_coords = self.data.coords[name].values
                break

        for name in y_coord_names:
            if name in self.data.coords:
                y_coords = self.data.coords[name].values
                break

        return x_coords, y_coords

    def get_values(self, percentile: Optional[float] = None, threshold: Optional[float] = None) -> np.ndarray:
        """Get the data values as a 2D numpy array.

        Args:
            percentile: For percentile data, which percentile to select
            threshold: For probability data, which threshold to select (in meters for precip)

        Returns:
            2D numpy array of values
        """
        data = self.data

        # Handle percentile dimension
        if "percentile" in data.dims and percentile is not None:
            data = data.sel(percentile=percentile, method="nearest")

        # Handle threshold dimension
        if "threshold" in data.dims and threshold is not None:
            data = data.sel(threshold=threshold, method="nearest")

        # Handle realization/ensemble dimension
        if "realization" in data.dims:
            data = data.mean(dim="realization")

        # Drop any remaining singleton dimensions
        data = data.squeeze()

        return data.values

    def get_thresholds(self) -> Optional[np.ndarray]:
        """Get available threshold values (in meters for precip).

        Returns:
            Array of threshold values or None if no threshold dimension
        """
        if "threshold" in self.data.coords:
            return self.data.coords["threshold"].values
        return None

    def get_thresholds_mm(self) -> Optional[np.ndarray]:
        """Get available threshold values converted to mm.

        Returns:
            Array of threshold values in mm or None
        """
        thresholds = self.get_thresholds()
        if thresholds is not None:
            return thresholds * 1000
        return None

    def get_percentiles(self) -> Optional[np.ndarray]:
        """Get available percentile values.

        Returns:
            Array of percentile values or None if no percentile dimension
        """
        if "percentile" in self.data.coords:
            return self.data.coords["percentile"].values
        return None

    def select_threshold(self, threshold_m: float) -> xr.DataArray:
        """Select data for a specific threshold and return as DataArray.

        Args:
            threshold_m: Threshold value in meters

        Returns:
            DataArray with the threshold dimension removed
        """
        if "threshold" not in self.data.dims:
            return self.data

        selected = self.data.sel(threshold=threshold_m, method="nearest")
        return selected

    def select_percentile(self, percentile: float) -> xr.DataArray:
        """Select data for a specific percentile and return as DataArray.

        Args:
            percentile: Percentile value (e.g., 50.0)

        Returns:
            DataArray with the percentile dimension removed
        """
        if "percentile" not in self.data.dims:
            return self.data

        selected = self.data.sel(percentile=percentile, method="nearest")
        return selected

    def get_basetime_str(self) -> str:
        """Get basetime as formatted string."""
        try:
            return pd.to_datetime(self.basetime).strftime("%Y%m%dT%H%MZ")
        except Exception:
            return str(self.basetime)

    def get_validtime_str(self) -> str:
        """Get validtime as formatted string."""
        try:
            return pd.to_datetime(self.validtime).strftime("%Y%m%dT%H%MZ")
        except Exception:
            return str(self.validtime)

    def info(self) -> Dict[str, Any]:
        """Return a dictionary of forecast metadata."""
        return {
            "name": self.name,
            "source": self.source,
            "variable": self.variable_name,
            "parameter": self.parameter,
            "basetime": str(self.basetime),
            "validtime": str(self.validtime),
            "leadhour": self.leadhour,
            "data_type": self.data_type,
            "units": self.units,
            "frequency": self.frequency,
            "shape": self.data.shape,
        }

    def get_global_attrs(self) -> Dict[str, Any]:
        """Return global attributes of the NetCDF file."""
        return dict(self.dataset.attrs)

    def get_variable_attrs(self) -> Dict[str, Any]:
        """Return attributes of the main variable."""
        return dict(self.data.attrs)

    def close(self):
        """Close the dataset."""
        self.dataset.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
