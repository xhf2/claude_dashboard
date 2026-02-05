"""Dashboard components for the Claude Dashboard."""

from .map_plot import create_map_figure, MapPlot
from .colorscale import get_colorscale, ColorscaleManager
from .dropdowns import create_dropdown_options, DropdownBuilder
from .wind_arrows import create_wind_arrows, WindArrowOverlay
from .obs_overlay import create_obs_overlay, ObservationOverlay

__all__ = [
    "create_map_figure",
    "MapPlot",
    "get_colorscale",
    "ColorscaleManager",
    "create_dropdown_options",
    "DropdownBuilder",
    "create_wind_arrows",
    "WindArrowOverlay",
    "create_obs_overlay",
    "ObservationOverlay",
]
