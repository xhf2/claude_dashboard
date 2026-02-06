"""Configuration for the Claude Dashboard.

This file contains all configurable settings for the dashboard including:
- Realm paths (data directories)
- Parameter definitions and display names
- Threshold/probability field definitions
- Colorscale mappings
- Unit definitions

To configure for your environment, update the REALM_PATHS dictionary
to point to your data directories.
"""

import os
from pathlib import Path

# =============================================================================
# BASE PATHS
# =============================================================================

# Base directory of the dashboard
BASE_DIR = Path(__file__).parent

# Data directories (colorscales, shapefiles)
DATA_PATH = BASE_DIR / "data"
COLORSCALE_PATH = DATA_PATH / "colorscales"
CARTOPY_PATH = DATA_PATH / "shapefiles"


# =============================================================================
# REALM CONFIGURATION
# Configure these paths for your data locations
# Each realm should point to a directory containing cycle directories (e.g., 20250308T1200Z)
# =============================================================================

REALM_PATHS = {
    "test_realm": str(BASE_DIR),  # Default: dashboard directory containing cycle dirs
    "user_realm": str(BASE_DIR),  # Second realm for comparison (configure as needed)
    # Add more realms as needed:
    # "production": "/path/to/production/data",
    # "experimental": "/path/to/experimental/data",
}


# =============================================================================
# FORECAST MODELS
# Available forecast models in the data pipeline
# =============================================================================

MODELS = [
    "blend",
    "ecmwf_ens",
    "ecmwf_hres",
    "bom_access_g3",
    "bom_access_g4",
    "bom_access_ge3",
    "bom_access_ge4",
]


# =============================================================================
# OUTPUT TYPES
# Types of outputs to look for in the data directories
# =============================================================================

OUTPUT_TYPES = [
    "blendgrids",
    "level2",
    "level3",
    "expectedvalues_extract",
    "merge_expectedvalues",
    "merge_percentiles",
    "merge_probabilities",
    "percentiles_extract",
    "probabilities_extract",
    "recfilter",
    "apply_rainforests_calibration",
    "nb_topographic",
    "calc_ens_mean",
    "calc_ens_std",
    "regrid",
]

# Output types that contain threshold/probability data
PROBABILITY_OUTPUT_TYPES = [
    "blendgrids",
    "level2",
    "level3",
    "recfilter",
    "probabilities_extract",
    "merge_probabilities",
    "apply_rainforests_calibration",
    "nb_topographic",
]

# Output types that contain percentile data
PERCENTILE_OUTPUT_TYPES = [
    "percentiles_extract",
    "merge_percentiles",
]

# Output types that contain expected value data
EXPECTED_VALUE_OUTPUT_TYPES = [
    "expectedvalues_extract",
    "merge_expectedvalues",
    "calc_ens_mean",
    "calc_ens_std",
    "regrid",
]


# =============================================================================
# PARAMETER DEFINITIONS
# Available parameters and their display names
# =============================================================================

PARAMETERS = {
    # Precipitation
    "precipacc24h": "24h Precipitation",
    "precipacc03h": "3h Precipitation",
    # Temperature - Surface
    "tempscreen": "Screen Temperature",
    "tempdaymax": "Day Maximum Temperature",
    "tempnightmin": "Night Minimum Temperature",
    # Temperature - Pressure Levels
    "temp200hPa": "200 hPa Temperature",
    "temp300hPa": "300 hPa Temperature",
    "temp400hPa": "400 hPa Temperature",
    "temp500hPa": "500 hPa Temperature",
    "temp700hPa": "700 hPa Temperature",
    "temp800hPa": "800 hPa Temperature",
    "temp850hPa": "850 hPa Temperature",
    "temp900hPa": "900 hPa Temperature",
    "temp950hPa": "950 hPa Temperature",
    # Dew Point - Surface
    "tempdewscreen": "Screen Dew Point",
    # Dew Point - Pressure Levels
    "tempdew200hPa": "200 hPa Dew Point",
    "tempdew300hPa": "300 hPa Dew Point",
    "tempdew400hPa": "400 hPa Dew Point",
    "tempdew500hPa": "500 hPa Dew Point",
    "tempdew700hPa": "700 hPa Dew Point",
    "tempdew800hPa": "800 hPa Dew Point",
    "tempdew850hPa": "850 hPa Dew Point",
    "tempdew900hPa": "900 hPa Dew Point",
    "tempdew950hPa": "950 hPa Dew Point",
    # Wind Speed - Surface
    "windspd10m": "10m Wind Speed",
    "winddir10m": "10m Wind Direction",
    # Wind Speed - Pressure Levels
    "windspd700hPa": "700 hPa Wind Speed",
    "windspd800hPa": "800 hPa Wind Speed",
    "windspd850hPa": "850 hPa Wind Speed",
    "windspd900hPa": "900 hPa Wind Speed",
    "windspd950hPa": "950 hPa Wind Speed",
}


# =============================================================================
# THRESHOLD/PROBABILITY FIELD DEFINITIONS
# Define the probability fields (PoP for precipitation, PoW for wind)
# Format: {display_label: threshold_value_in_SI_units}
# =============================================================================

# Daily Precipitation Probability Fields (threshold in meters)
DAILY_POP_FIELDS = {
    "PoP (0.2mm)": 0.0002,
    "PoP1 (1mm)": 0.001,
    "PoP5 (5mm)": 0.005,
    "PoP10 (10mm)": 0.010,
    "PoP15 (15mm)": 0.015,
    "PoP25 (25mm)": 0.025,
    "PoP50 (50mm)": 0.050,
}

# 3-Hourly Precipitation Probability Fields (threshold in meters)
HOURLY_POP_FIELDS = {
    "PoP (0.2mm)": 0.0002,
    "PoP1 (1mm)": 0.001,
    "PoP2 (2mm)": 0.002,
    "PoP5 (5mm)": 0.005,
    "PoP10 (10mm)": 0.010,
}

# Wind Speed Probability Fields (threshold in m/s)
# PoW = Probability of Wind exceeding threshold
# Thresholds match values in windspd data files
POW_FIELDS = {
    "PoW1 (1 kt)": 0.514,      # ~1 knot
    "PoW4 (4 kts)": 2.06,      # ~4 knots
    "PoW7 (7 kts)": 3.6,       # ~7 knots
    "PoW11 (11 kts)": 5.66,    # ~11 knots
    "PoW17 (17 kts)": 8.75,    # ~17 knots
    "PoW22 (22 kts)": 11.3,    # ~22 knots
    "PoW26 (26 kts)": 13.4,    # ~26 knots
    "PoW34 (34 kts)": 17.5,    # ~34 knots (Gale threshold)
    "PoW48 (48 kts)": 24.7,    # ~48 knots (Storm threshold)
    "PoW64 (64 kts)": 32.9,    # ~64 knots (Hurricane threshold)
}

# Temperature Probability Fields (threshold in Kelvin)
# POT = Probability of Temperature exceeding threshold
# Configure thresholds as needed for your use case
POT_FIELDS = {
    "POT0 (0C)": 273.15,       # 0°C (freezing)
    "POT5 (5C)": 278.15,       # 5°C
    "POT10 (10C)": 283.15,     # 10°C
    "POT15 (15C)": 288.15,     # 15°C
    "POT20 (20C)": 293.15,     # 20°C
    "POT25 (25C)": 298.15,     # 25°C
    "POT30 (30C)": 303.15,     # 30°C
    "POT35 (35C)": 308.15,     # 35°C (heat threshold)
    "POT40 (40C)": 313.15,     # 40°C (extreme heat)
}

# Dew Point Probability Fields (threshold in Kelvin)
# POTd = Probability of Dew Point exceeding threshold
# Configure thresholds as needed for your use case
POTD_FIELDS = {
    "POTd-10 (-10C)": 263.15,  # -10°C (very dry)
    "POTd-5 (-5C)": 268.15,    # -5°C
    "POTd0 (0C)": 273.15,      # 0°C
    "POTd5 (5C)": 278.15,      # 5°C
    "POTd10 (10C)": 283.15,    # 10°C (comfortable)
    "POTd15 (15C)": 288.15,    # 15°C
    "POTd20 (20C)": 293.15,    # 20°C (humid)
    "POTd25 (25C)": 298.15,    # 25°C (very humid)
}


# =============================================================================
# PERCENTILE FIELD DEFINITIONS
# Format: {display_label: percentile_value}
# =============================================================================

# Daily Precipitation Percentile Fields
DAILY_PRECIP_PERCENTILE_FIELDS = {
    "DailyPrecip10Pct (90th pctl)": 90.0,
    "DailyPrecip25Pct (75th pctl)": 75.0,
    "DailyPrecip50Pct (50th pctl)": 50.0,
    "DailyPrecip75Pct (25th pctl)": 25.0,
    "DailyPrecip90Pct (10th pctl)": 10.0,
}

# Wind Speed Percentile Fields
WINDMAG_PERCENTILE_FIELDS = {
    "WindMag10Pct (90th pctl)": 90.0,
    "WindMag25Pct (75th pctl)": 75.0,
    "WindMag50Pct (50th pctl)": 50.0,
    "WindMag75Pct (25th pctl)": 25.0,
    "WindMag90Pct (10th pctl)": 10.0,
}

# Temperature Percentile Fields (screen temperature)
T_PERCENTILE_FIELDS = {
    "T10Pct (90th pctl)": 90.0,
    "T25Pct (75th pctl)": 75.0,
    "T50Pct (50th pctl)": 50.0,
    "T75Pct (25th pctl)": 25.0,
    "T90Pct (10th pctl)": 10.0,
}

# Max Temperature Percentile Fields
MAXT_PERCENTILE_FIELDS = {
    "maxT10Pct (90th pctl)": 90.0,
    "maxT25Pct (75th pctl)": 75.0,
    "maxT50Pct (50th pctl)": 50.0,
    "maxT75Pct (25th pctl)": 25.0,
    "maxT90Pct (10th pctl)": 10.0,
}

# Min Temperature Percentile Fields
MINT_PERCENTILE_FIELDS = {
    "minT10Pct (90th pctl)": 90.0,
    "minT25Pct (75th pctl)": 75.0,
    "minT50Pct (50th pctl)": 50.0,
    "minT75Pct (25th pctl)": 25.0,
    "minT90Pct (10th pctl)": 10.0,
}

# Dew Point Percentile Fields
TD_PERCENTILE_FIELDS = {
    "Td10Pct (90th pctl)": 90.0,
    "Td25Pct (75th pctl)": 75.0,
    "Td50Pct (50th pctl)": 50.0,
    "Td75Pct (25th pctl)": 25.0,
    "Td90Pct (10th pctl)": 10.0,
}


# =============================================================================
# COLORSCALE MAPPINGS
# Map parameters to their GFE colorscale types
# =============================================================================

PARAMETER_COLORSCALE = {
    # Precipitation
    "precipacc24h": "DailyPrecip",
    "precipacc03h": "Precip",
    # Temperature - Surface
    "tempscreen": "T",
    "tempdaymax": "T",
    "tempnightmin": "T",
    # Temperature - Pressure Levels
    "temp200hPa": "T",
    "temp300hPa": "T",
    "temp400hPa": "T",
    "temp500hPa": "T",
    "temp700hPa": "T",
    "temp800hPa": "T",
    "temp850hPa": "T",
    "temp900hPa": "T",
    "temp950hPa": "T",
    # Dew Point - Surface
    "tempdewscreen": "Td",
    # Dew Point - Pressure Levels
    "tempdew200hPa": "Td",
    "tempdew300hPa": "Td",
    "tempdew400hPa": "Td",
    "tempdew500hPa": "Td",
    "tempdew700hPa": "Td",
    "tempdew800hPa": "Td",
    "tempdew850hPa": "Td",
    "tempdew900hPa": "Td",
    "tempdew950hPa": "Td",
    # Wind Speed
    "windspd10m": "Wind",
    "windspd700hPa": "Wind",
    "windspd800hPa": "Wind",
    "windspd850hPa": "Wind",
    "windspd900hPa": "Wind",
    "windspd950hPa": "Wind",
    "winddir10m": None,  # Wind direction uses circular colorscale
}


# =============================================================================
# PARAMETER TO VARIABLE NAME MAPPING
# Maps parameter names to expected NetCDF variable names
# =============================================================================

PARAMETER_VARIABLE_NAMES = {
    # Temperature - Surface
    "tempscreen": "temperature_at_screen_level",
    "tempdaymax": "temperature_at_screen_level_daytime_max",
    "tempnightmin": "temperature_at_screen_level_nighttime_min",
    # Dew Point - Surface
    "tempdewscreen": "temperature_of_dew_point_at_screen_level",
    # Precipitation
    "precipacc24h": "precipitation_accumulation",
    "precipacc03h": "precipitation_accumulation",
    # Wind
    "windspd10m": "wind_speed",
    "winddir10m": "wind_from_direction",
}


# =============================================================================
# UNIT DEFINITIONS
# Display units after conversion (conversions handled in data_loader)
# =============================================================================

PARAMETER_UNITS = {
    # Precipitation
    "precipacc24h": "mm",
    "precipacc03h": "mm",
    # Temperature - Surface
    "tempscreen": "degC",
    "tempdaymax": "degC",
    "tempnightmin": "degC",
    # Temperature - Pressure Levels
    "temp200hPa": "degC",
    "temp300hPa": "degC",
    "temp400hPa": "degC",
    "temp500hPa": "degC",
    "temp700hPa": "degC",
    "temp800hPa": "degC",
    "temp850hPa": "degC",
    "temp900hPa": "degC",
    "temp950hPa": "degC",
    # Dew Point - Surface
    "tempdewscreen": "degC",
    # Dew Point - Pressure Levels
    "tempdew200hPa": "degC",
    "tempdew300hPa": "degC",
    "tempdew400hPa": "degC",
    "tempdew500hPa": "degC",
    "tempdew700hPa": "degC",
    "tempdew800hPa": "degC",
    "tempdew850hPa": "degC",
    "tempdew900hPa": "degC",
    "tempdew950hPa": "degC",
    # Wind Speed
    "windspd10m": "knots",
    "windspd700hPa": "knots",
    "windspd800hPa": "knots",
    "windspd850hPa": "knots",
    "windspd900hPa": "knots",
    "windspd950hPa": "knots",
    "winddir10m": "degrees",
}


# =============================================================================
# PROJECTION PARAMETERS
# =============================================================================

# GFE Mercator projection (legacy)
GFE_PROJ4STR = "+proj=merc +ellps=sphere +x_0=0 +y_0=0 +lon_0=135"
GFE_EXTENT = (5111745.5, 4668484.5)
GFE_ORIGIN = (-2771347.5, -5616770)
GFE_SIZE = (772, 705)

# IMPROVER Albers Equal Area projection for Australia
ALBERS_PROJ4STR = "+proj=aea +lat_0=-24.75 +lon_0=134.0 +lat_1=-10.0 +lat_2=-40.0 +x_0=0 +y_0=0 +ellps=GRS80 +units=m +no_defs"


# =============================================================================
# MAP DISPLAY DEFAULTS
# =============================================================================

DEFAULT_MAP_CENTER = {"lat": -25.0, "lon": 135.0}
DEFAULT_MAP_ZOOM = 3


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_probability_fields(parameter: str) -> dict:
    """Get the appropriate probability field definitions for a parameter.

    Args:
        parameter: Parameter name (e.g., 'precipacc24h', 'windspd700hPa', 'tempscreen')

    Returns:
        Dict of {label: threshold_value} for the parameter
    """
    param_lower = parameter.lower()
    if "precip" in param_lower:
        if "24h" in parameter:
            return DAILY_POP_FIELDS
        else:
            return HOURLY_POP_FIELDS
    elif "windspd" in param_lower:
        return POW_FIELDS
    elif "tempdew" in param_lower:
        return POTD_FIELDS
    elif "temp" in param_lower:
        return POT_FIELDS
    return {}


def get_percentile_fields(parameter: str) -> dict:
    """Get the appropriate percentile field definitions for a parameter.

    Args:
        parameter: Parameter name

    Returns:
        Dict of {label: percentile_value} for the parameter
    """
    param_lower = parameter.lower()
    if "precip" in param_lower:
        return DAILY_PRECIP_PERCENTILE_FIELDS
    elif "windspd" in param_lower:
        return WINDMAG_PERCENTILE_FIELDS
    elif "tempdew" in param_lower:
        return TD_PERCENTILE_FIELDS
    elif "tempdaymax" in param_lower:
        return MAXT_PERCENTILE_FIELDS
    elif "tempnightmin" in param_lower:
        return MINT_PERCENTILE_FIELDS
    elif "temp" in param_lower:
        return T_PERCENTILE_FIELDS
    return {}
