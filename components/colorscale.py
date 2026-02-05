"""Colorscale handling for the dashboard.

Provides GFE-style colorscales with non-linear breakpoints for precipitation.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable

import numpy as np

import config


# Path to GFE colorscale files
GFE_COLORSCALE_DIR = Path(__file__).parent.parent / "data" / "colorscales"


def load_gfe_colorscale_file(filepath: Path) -> List[str]:
    """Load all RGB colors from a GFE colorscale file.

    Args:
        filepath: Path to the GFE colorscale txt file

    Returns:
        List of rgb(r,g,b) color strings
    """
    colors = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            # Skip header line
            if line.startswith("R") or not line:
                continue
            parts = line.split()
            if len(parts) >= 3:
                r, g, b = int(parts[0]), int(parts[1]), int(parts[2])
                colors.append(f"rgb({r},{g},{b})")
    return colors


# GFE colorscale properties
# Note: cbar_label shortened for comparison plots
GFE_COLORSCALE_PROPERTIES = {
    "PoP": {
        "cbar_labels": [0, 10, 15, 35, 65, 85, 100],
        "vmin": 0,
        "vmax": 100,
        "cbar_label": "%",
    },
    "DailyPrecip": {
        "cbar_labels": [0, 0.2, 2, 5, 10, 20, 50, 100, 200],
        "vmin": 0,
        "vmax": 200,
        "cbar_label": "mm",
    },
    "Precip": {
        "cbar_labels": [0, 0.2, 0.5, 1, 2, 5, 10, 20, 40, 100],
        "vmin": 0,
        "vmax": 100,
        "cbar_label": "mm",
    },
    "T": {
        "cbar_labels": [-15, -5, 5, 15, 25, 35, 45],
        "vmin": -15,
        "vmax": 50,
        "cbar_label": "°C",
    },
    "Td": {
        "cbar_labels": [-20, -10, 0, 10, 20, 30],
        "vmin": -20,
        "vmax": 30,
        "cbar_label": "°C",
    },
    "Wind": {
        "cbar_labels": [0, 15, 33, 47, 63, 75],
        "vmin": 0,
        "vmax": 75,
        "cbar_label": "kts",
    },
}


def create_daily_precip_colorscale() -> List[List]:
    """Create the GFE Daily Precipitation colorscale.

    Uses ALL RGB colors from gfe_A_PrecipDaily.txt file (256 colors total)
    to create smooth gradients within each band.

    Breakpoints (mm) and line ranges:
    - 0-0.2: gray (lines 0-2, 3 colors)
    - 0.2-2: yellow (lines 3-25, 23 colors)
    - 2-5: light green (lines 26-52, 27 colors)
    - 5-10: dark green (lines 53-81, 29 colors)
    - 10-20: blue (lines 82-116, 35 colors)
    - 20-50: purple/magenta (lines 117-168, 52 colors)
    - 50-100: red (lines 169-211, 43 colors)
    - 100-200: orange (lines 212-255, 44 colors)

    Returns:
        Plotly colorscale list with full gradient
    """
    vmax = 200.0

    # Load all colors from file
    colorscale_file = GFE_COLORSCALE_DIR / "gfe_A_PrecipDaily.txt"
    all_colors = load_gfe_colorscale_file(colorscale_file)

    # Define bands: (value_start, value_end, line_start, line_end)
    # Line indices are 0-based (after skipping header)
    bands = [
        (0, 0.2, 0, 3),       # gray: lines 0-2 (3 colors)
        (0.2, 2, 3, 26),      # yellow: lines 3-25 (23 colors)
        (2, 5, 26, 53),       # light green: lines 26-52 (27 colors)
        (5, 10, 53, 82),      # dark green: lines 53-81 (29 colors)
        (10, 20, 82, 117),    # blue: lines 82-116 (35 colors)
        (20, 50, 117, 169),   # purple: lines 117-168 (52 colors)
        (50, 100, 169, 212),  # red: lines 169-211 (43 colors)
        (100, 200, 212, 256), # orange: lines 212-255 (44 colors)
    ]

    colorscale = []
    for v_start, v_end, line_start, line_end in bands:
        band_colors = all_colors[line_start:line_end]
        n_colors = len(band_colors)

        if n_colors == 0:
            continue

        # Map each color to a position within the band's value range
        for i, color in enumerate(band_colors):
            # Interpolate position within the band
            if n_colors == 1:
                frac = 0.5
            else:
                frac = i / (n_colors - 1)

            value = v_start + frac * (v_end - v_start)
            position = value / vmax
            colorscale.append([position, color])

    # Ensure colorscale starts at 0 and ends at 1
    if colorscale and colorscale[0][0] > 0:
        colorscale.insert(0, [0, colorscale[0][1]])
    if colorscale and colorscale[-1][0] < 1:
        colorscale.append([1, colorscale[-1][1]])

    return colorscale


def create_daily_pop_colorscale() -> List[List]:
    """Create the GFE Daily PoP (Probability of Precipitation) colorscale.

    Returns:
        Plotly colorscale for probability 0-100%
    """
    # PoP colorscale: 0-100%
    breakpoints = [
        (0, "rgb(255, 255, 255)"),     # white (0%)
        (10, "rgb(200, 255, 200)"),    # very light green
        (15, "rgb(150, 255, 150)"),    # light green
        (35, "rgb(100, 200, 100)"),    # green
        (65, "rgb(100, 150, 255)"),    # light blue
        (85, "rgb(50, 50, 255)"),      # blue
        (100, "rgb(150, 0, 200)"),     # purple
    ]

    colorscale = []
    for value, color in breakpoints:
        position = value / 100.0
        colorscale.append([position, color])

    return colorscale


def create_precip_3hr_colorscale() -> Tuple[List[List], List[float], List[str]]:
    """Create 3-hourly precipitation colorscale.

    Returns:
        Tuple of (colorscale, tickvals, ticktext)
    """
    vmax = 100.0

    color_bands = [
        (0, 0.2, "rgb(80, 80, 80)"),         # gray
        (0.2, 1, "rgb(255, 255, 0)"),        # yellow
        (1, 2, "rgb(144, 238, 144)"),        # light green
        (2, 5, "rgb(0, 200, 0)"),            # green
        (5, 10, "rgb(0, 128, 0)"),           # dark green
        (10, 20, "rgb(0, 100, 255)"),        # blue
        (20, 40, "rgb(148, 0, 211)"),        # purple
        (40, 100, "rgb(255, 0, 0)"),         # red
    ]

    colorscale = []
    for v_start, v_end, color in color_bands:
        pos_start = v_start / vmax
        pos_end = v_end / vmax
        colorscale.append([pos_start, color])
        colorscale.append([pos_end, color])

    boundaries = [0, 0.2, 1, 2, 5, 10, 20, 40, 100]
    n_bands = len(boundaries) - 1
    tickvals = [i * vmax / n_bands for i in range(len(boundaries))]
    ticktext = ["0", "0.2", "1", "2", "5", "10", "20", "40", "100"]

    return colorscale, tickvals, ticktext


def create_temperature_colorscale() -> List[List]:
    """Create temperature colorscale from -15 to 50°C."""
    vmin = -15.0
    vmax = 50.0
    vrange = vmax - vmin

    # Cold to hot: blue -> cyan -> green -> yellow -> orange -> red
    breakpoints = [
        (-15, "rgb(0, 0, 180)"),       # dark blue (very cold)
        (-10, "rgb(50, 50, 255)"),     # blue
        (-5, "rgb(100, 150, 255)"),    # light blue
        (0, "rgb(150, 220, 255)"),     # cyan
        (5, "rgb(150, 255, 200)"),     # cyan-green
        (10, "rgb(100, 255, 100)"),    # green
        (15, "rgb(180, 255, 50)"),     # yellow-green
        (20, "rgb(255, 255, 0)"),      # yellow
        (25, "rgb(255, 200, 0)"),      # orange-yellow
        (30, "rgb(255, 150, 0)"),      # orange
        (35, "rgb(255, 80, 0)"),       # red-orange
        (40, "rgb(255, 0, 0)"),        # red
        (45, "rgb(200, 0, 50)"),       # dark red
        (50, "rgb(150, 0, 100)"),      # maroon
    ]

    colorscale = []
    for value, color in breakpoints:
        position = (value - vmin) / vrange
        colorscale.append([position, color])

    return colorscale


def create_wind_colorscale() -> List[List]:
    """Create wind speed colorscale 0-75 knots using GFE colors.

    Uses ALL RGB colors from gfe_windthresholds.txt file (150 colors total)
    to create smooth gradients within each band.

    Breakpoints (knots) and line ranges:
    - 0-5: gray (lines 0-8, 9 colors) - calm
    - 5-10: olive/khaki (lines 9-18, 10 colors) - light
    - 10-15: yellow (lines 19-32, 14 colors)
    - 15-25: gold (lines 33-50, 18 colors)
    - 25-33: green (lines 51-66, 16 colors) - strong breeze
    - 33-47: blue (lines 67-94, 28 colors) - gale
    - 47-63: red (lines 95-126, 32 colors) - storm
    - 63-75: magenta (lines 127-149, 23 colors) - hurricane force

    Returns:
        Plotly colorscale list with full gradient
    """
    vmax = 75.0

    # Load all colors from file
    colorscale_file = GFE_COLORSCALE_DIR / "gfe_windthresholds.txt"
    all_colors = load_gfe_colorscale_file(colorscale_file)

    # Define bands: (value_start, value_end, line_start, line_end)
    # Line indices are 0-based (after skipping header)
    bands = [
        (0, 5, 0, 9),         # gray: lines 0-8 (9 colors)
        (5, 10, 9, 19),       # olive: lines 9-18 (10 colors)
        (10, 15, 19, 33),     # yellow: lines 19-32 (14 colors)
        (15, 25, 33, 51),     # gold: lines 33-50 (18 colors)
        (25, 33, 51, 67),     # green: lines 51-66 (16 colors)
        (33, 47, 67, 95),     # blue: lines 67-94 (28 colors)
        (47, 63, 95, 127),    # red: lines 95-126 (32 colors)
        (63, 75, 127, 150),   # magenta: lines 127-149 (23 colors)
    ]

    colorscale = []
    for v_start, v_end, line_start, line_end in bands:
        band_colors = all_colors[line_start:line_end]
        n_colors = len(band_colors)

        if n_colors == 0:
            continue

        # Map each color to a position within the band's value range
        for i, color in enumerate(band_colors):
            if n_colors == 1:
                frac = 0.5
            else:
                frac = i / (n_colors - 1)

            value = v_start + frac * (v_end - v_start)
            position = value / vmax
            colorscale.append([position, color])

    # Ensure colorscale starts at 0 and ends at 1
    if colorscale and colorscale[0][0] > 0:
        colorscale.insert(0, [0, colorscale[0][1]])
    if colorscale and colorscale[-1][0] < 1:
        colorscale.append([1, colorscale[-1][1]])

    return colorscale


def create_dewpoint_colorscale() -> List[List]:
    """Create dew point temperature colorscale from -20 to 30°C."""
    vmin = -20.0
    vmax = 30.0
    vrange = vmax - vmin

    # Similar to temperature but narrower range
    # Cold to warm: blue -> cyan -> green -> yellow -> orange
    breakpoints = [
        (-20, "rgb(0, 0, 180)"),       # dark blue (very dry/cold)
        (-15, "rgb(50, 50, 255)"),     # blue
        (-10, "rgb(100, 150, 255)"),   # light blue
        (-5, "rgb(150, 200, 255)"),    # pale blue
        (0, "rgb(150, 220, 220)"),     # cyan
        (5, "rgb(150, 255, 200)"),     # cyan-green
        (10, "rgb(100, 255, 100)"),    # green
        (15, "rgb(180, 255, 50)"),     # yellow-green
        (20, "rgb(255, 255, 0)"),      # yellow
        (25, "rgb(255, 180, 0)"),      # orange-yellow
        (30, "rgb(255, 100, 0)"),      # orange (humid)
    ]

    colorscale = []
    for value, color in breakpoints:
        position = (value - vmin) / vrange
        colorscale.append([position, color])

    return colorscale


# Cache for colorscales
_colorscale_cache: Dict[str, List] = {}


def get_colorscale(
    colorscale_type: str,
    use_preset: bool = True,
) -> Tuple[List, float, float]:
    """Get a Plotly colorscale for a given type.

    Args:
        colorscale_type: Type of colorscale (e.g., "DailyPrecip", "T", "Wind")
        use_preset: Whether to use preset GFE colorscale or auto

    Returns:
        Tuple of (colorscale, vmin, vmax)
    """
    if not use_preset:
        return "Viridis", None, None

    if colorscale_type not in GFE_COLORSCALE_PROPERTIES:
        return "Viridis", None, None

    props = GFE_COLORSCALE_PROPERTIES[colorscale_type]

    # Check cache
    if colorscale_type not in _colorscale_cache:
        if colorscale_type == "DailyPrecip":
            _colorscale_cache[colorscale_type] = create_daily_precip_colorscale()
        elif colorscale_type == "Precip":
            cs, tickvals, ticktext = create_precip_3hr_colorscale()
            _colorscale_cache[colorscale_type] = cs
            _colorscale_cache[f"{colorscale_type}_tickvals"] = tickvals
            _colorscale_cache[f"{colorscale_type}_ticktext"] = ticktext
        elif colorscale_type == "T":
            _colorscale_cache[colorscale_type] = create_temperature_colorscale()
        elif colorscale_type == "Td":
            _colorscale_cache[colorscale_type] = create_dewpoint_colorscale()
        elif colorscale_type == "Wind":
            _colorscale_cache[colorscale_type] = create_wind_colorscale()
        elif colorscale_type == "PoP":
            _colorscale_cache[colorscale_type] = create_daily_pop_colorscale()
        else:
            return "Viridis", props["vmin"], props["vmax"]

    colorscale = _colorscale_cache[colorscale_type]
    return colorscale, props["vmin"], props["vmax"]


def get_colorscale_properties(colorscale_type: str) -> Dict[str, Any]:
    """Get properties for a colorscale type.

    Args:
        colorscale_type: Type of colorscale

    Returns:
        Dict with colorscale properties including tick positions for equal-spaced colorscales
    """
    if colorscale_type not in GFE_COLORSCALE_PROPERTIES:
        return {}

    props = GFE_COLORSCALE_PROPERTIES[colorscale_type].copy()

    # Add tick positions for equal-spaced colorbar display
    tickvals_key = f"{colorscale_type}_tickvals"
    ticktext_key = f"{colorscale_type}_ticktext"
    if tickvals_key in _colorscale_cache:
        props["tickvals"] = _colorscale_cache[tickvals_key]
        props["ticktext"] = _colorscale_cache[ticktext_key]

    return props


def transform_data_for_equal_spacing(data: np.ndarray, colorscale_type: str) -> np.ndarray:
    """Transform data values to work with equal-spaced colorscales.

    Maps actual values to positions in the colorscale where each color band
    occupies equal space.

    Args:
        data: Original data array
        colorscale_type: Type of colorscale

    Returns:
        Transformed data array
    """
    if colorscale_type == "DailyPrecip":
        boundaries = [0, 0.2, 2, 5, 10, 20, 50, 100, 200]
        vmax = 200
    elif colorscale_type == "Precip":
        boundaries = [0, 0.2, 1, 2, 5, 10, 20, 40, 100]
        vmax = 100
    else:
        return data

    n_bands = len(boundaries) - 1
    result = np.zeros_like(data, dtype=float)

    for i in range(n_bands):
        v_min = boundaries[i]
        v_max = boundaries[i + 1]
        # Target range for this band (equal spacing)
        t_min = i * vmax / n_bands
        t_max = (i + 1) * vmax / n_bands

        # Find values in this band
        mask = (data >= v_min) & (data < v_max) if i < n_bands - 1 else (data >= v_min) & (data <= v_max)

        # Linear interpolation within the band
        if v_max > v_min:
            result[mask] = t_min + (data[mask] - v_min) / (v_max - v_min) * (t_max - t_min)

    # Handle NaN
    result[np.isnan(data)] = np.nan

    return result


class ColorscaleManager:
    """Manage colorscales for the dashboard."""

    def __init__(self, parameter: str):
        """Initialize the colorscale manager.

        Args:
            parameter: Parameter name (e.g., "precipacc24h")
        """
        self.parameter = parameter
        self.colorscale_type = config.PARAMETER_COLORSCALE.get(parameter)
        self.use_preset = True

    def get_colorscale(self) -> Tuple[List, Optional[float], Optional[float]]:
        """Get the current colorscale settings.

        Returns:
            Tuple of (colorscale, vmin, vmax)
        """
        return get_colorscale(self.colorscale_type, self.use_preset)

    def get_auto_colorscale(
        self, data: np.ndarray, percentile_clip: float = 99,
    ) -> Tuple[str, float, float]:
        """Get auto-scaled colorscale based on data.

        Args:
            data: Data array to base colorscale on
            percentile_clip: Percentile for clipping max value

        Returns:
            Tuple of (colorscale_name, vmin, vmax)
        """
        valid_data = data[~np.isnan(data)]
        if len(valid_data) == 0:
            return "Viridis", 0, 1

        vmin = float(np.nanmin(valid_data))
        vmax = float(np.nanpercentile(valid_data, percentile_clip))

        if "precip" in self.parameter.lower():
            return "YlGnBu", max(0, vmin), vmax
        elif "temp" in self.parameter.lower():
            return "RdBu_r", vmin, vmax
        elif "wind" in self.parameter.lower():
            return "Plasma", max(0, vmin), vmax
        else:
            return "Viridis", vmin, vmax

    def toggle_preset(self, use_preset: bool):
        """Toggle between preset and auto colorscale."""
        self.use_preset = use_preset

    def get_colorbar_settings(self) -> Dict[str, Any]:
        """Get colorbar settings for Plotly.

        Returns:
            Dict with colorbar configuration
        """
        if self.colorscale_type and self.use_preset:
            props = get_colorscale_properties(self.colorscale_type)
            return {
                "title": props.get("cbar_label", ""),
                "tickvals": props.get("cbar_labels", []),
                "ticktext": [str(v) for v in props.get("cbar_labels", [])],
            }
        return {"title": config.PARAMETER_UNITS.get(self.parameter, "")}


def get_wind_direction_colorscale() -> Tuple[List, float, float]:
    """Get a circular colorscale for wind direction (0-360 degrees).

    Returns:
        Tuple of (colorscale, vmin, vmax)
    """
    n_colors = 360
    colors = []

    for i in range(n_colors):
        h = i / 360.0
        if h < 1/6:
            r, g, b = 255, int(255 * h * 6), 0
        elif h < 2/6:
            r, g, b = int(255 * (2 - h * 6)), 255, 0
        elif h < 3/6:
            r, g, b = 0, 255, int(255 * (h * 6 - 2))
        elif h < 4/6:
            r, g, b = 0, int(255 * (4 - h * 6)), 255
        elif h < 5/6:
            r, g, b = int(255 * (h * 6 - 4)), 0, 255
        else:
            r, g, b = 255, 0, int(255 * (6 - h * 6))

        colors.append((r, g, b))

    colorscale = []
    for i, (r, g, b) in enumerate(colors):
        position = i / (len(colors) - 1)
        colorscale.append([position, f"rgb({r},{g},{b})"])

    return colorscale, 0, 360
