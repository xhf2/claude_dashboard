"""Map plotting component for the dashboard.

Creates Plotly figures with forecast data overlaid on maps with coastlines.
"""

from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import config
from utils.coastline import get_coastline_scatter_data
from components.colorscale import ColorscaleManager, get_colorscale


class MapPlot:
    """Create map plots with forecast data and coastlines."""

    def __init__(
        self,
        parameter: str,
        use_preset_colorscale: bool = True,
    ):
        """Initialize the map plot.

        Args:
            parameter: Parameter name for colorscale selection
            use_preset_colorscale: Whether to use preset GFE colorscale
        """
        self.parameter = parameter
        self.colorscale_manager = ColorscaleManager(parameter)
        self.colorscale_manager.toggle_preset(use_preset_colorscale)

        # Pre-load coastlines
        self._coastline_lons, self._coastline_lats = get_coastline_scatter_data(
            source="gfe", simplify_tolerance=0.01
        )

    def create_figure(
        self,
        lon: np.ndarray,
        lat: np.ndarray,
        values: np.ndarray,
        title: str = "",
        show_coastlines: bool = True,
        colorscale: Optional[List] = None,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
    ) -> go.Figure:
        """Create a Plotly figure with forecast data.

        Args:
            lon: 1D or 2D array of longitudes
            lat: 1D or 2D array of latitudes
            values: 2D array of data values
            title: Figure title
            show_coastlines: Whether to show coastlines
            colorscale: Optional custom colorscale
            vmin: Optional minimum value for colorscale
            vmax: Optional maximum value for colorscale

        Returns:
            Plotly Figure object
        """
        # Get colorscale settings
        if colorscale is None:
            colorscale, default_vmin, default_vmax = self.colorscale_manager.get_colorscale()
            if vmin is None:
                vmin = default_vmin
            if vmax is None:
                vmax = default_vmax

        # Auto-scale if no preset values
        if vmin is None or vmax is None:
            auto_colorscale, auto_vmin, auto_vmax = self.colorscale_manager.get_auto_colorscale(values)
            if vmin is None:
                vmin = auto_vmin
            if vmax is None:
                vmax = auto_vmax
            if not self.colorscale_manager.use_preset:
                colorscale = auto_colorscale

        # Create figure
        fig = go.Figure()

        # Add heatmap trace
        fig.add_trace(
            go.Heatmap(
                x=lon if lon.ndim == 1 else lon[0, :],
                y=lat if lat.ndim == 1 else lat[:, 0],
                z=values,
                colorscale=colorscale,
                zmin=vmin,
                zmax=vmax,
                colorbar=dict(
                    title=self.colorscale_manager.get_colorbar_settings().get("title", ""),
                    tickvals=self.colorscale_manager.get_colorbar_settings().get("tickvals"),
                    ticktext=self.colorscale_manager.get_colorbar_settings().get("ticktext"),
                ),
                hovertemplate="Lon: %{x:.2f}<br>Lat: %{y:.2f}<br>Value: %{z:.2f}<extra></extra>",
            )
        )

        # Add coastlines
        if show_coastlines:
            fig.add_trace(
                go.Scatter(
                    x=self._coastline_lons,
                    y=self._coastline_lats,
                    mode="lines",
                    line=dict(color="black", width=0.5),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

        # Update layout
        fig.update_layout(
            title=dict(text=title, x=0.5),
            xaxis=dict(
                title="Longitude",
                scaleanchor="y",
                scaleratio=1,
                range=[lon.min() if hasattr(lon, 'min') else min(lon),
                       lon.max() if hasattr(lon, 'max') else max(lon)],
            ),
            yaxis=dict(
                title="Latitude",
                range=[lat.min() if hasattr(lat, 'min') else min(lat),
                       lat.max() if hasattr(lat, 'max') else max(lat)],
            ),
            margin=dict(l=60, r=60, t=60, b=60),
            height=600,
        )

        return fig


def create_map_figure(
    lon: np.ndarray,
    lat: np.ndarray,
    values: np.ndarray,
    parameter: str,
    title: str = "",
    use_preset_colorscale: bool = True,
    show_coastlines: bool = True,
) -> go.Figure:
    """Convenience function to create a map figure.

    Args:
        lon: Longitude array
        lat: Latitude array
        values: Data values array
        parameter: Parameter name
        title: Figure title
        use_preset_colorscale: Whether to use preset colorscale
        show_coastlines: Whether to show coastlines

    Returns:
        Plotly Figure object
    """
    map_plot = MapPlot(parameter, use_preset_colorscale)
    return map_plot.create_figure(
        lon, lat, values, title, show_coastlines
    )


def create_comparison_figure(
    lon: np.ndarray,
    lat: np.ndarray,
    values1: np.ndarray,
    values2: np.ndarray,
    parameter: str,
    title1: str = "Realm 1",
    title2: str = "Realm 2",
    show_difference: bool = True,
    use_preset_colorscale: bool = True,
) -> go.Figure:
    """Create a comparison figure with two realms and optional difference.

    Args:
        lon: Longitude array
        lat: Latitude array
        values1: Data values for first realm
        values2: Data values for second realm
        parameter: Parameter name
        title1: Title for first realm plot
        title2: Title for second realm plot
        show_difference: Whether to show difference plot
        use_preset_colorscale: Whether to use preset colorscale

    Returns:
        Plotly Figure with subplots
    """
    n_cols = 3 if show_difference else 2
    titles = [title1, title2]
    if show_difference:
        titles.append("Difference (1 - 2)")

    fig = make_subplots(
        rows=1,
        cols=n_cols,
        subplot_titles=titles,
        horizontal_spacing=0.05,
    )

    # Get colorscale settings
    cm = ColorscaleManager(parameter)
    cm.toggle_preset(use_preset_colorscale)
    colorscale, vmin, vmax = cm.get_colorscale()

    if vmin is None or vmax is None:
        all_values = np.concatenate([values1.flatten(), values2.flatten()])
        _, vmin, vmax = cm.get_auto_colorscale(all_values)
        if not use_preset_colorscale:
            colorscale, _, _ = cm.get_auto_colorscale(all_values)

    # Load coastlines
    coast_lons, coast_lats = get_coastline_scatter_data(source="gfe", simplify_tolerance=0.01)

    # Common heatmap settings
    lon_1d = lon if lon.ndim == 1 else lon[0, :]
    lat_1d = lat if lat.ndim == 1 else lat[:, 0]

    # Add first realm
    fig.add_trace(
        go.Heatmap(
            x=lon_1d,
            y=lat_1d,
            z=values1,
            colorscale=colorscale,
            zmin=vmin,
            zmax=vmax,
            showscale=False,
            hovertemplate="Lon: %{x:.2f}<br>Lat: %{y:.2f}<br>Value: %{z:.2f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # Add second realm
    fig.add_trace(
        go.Heatmap(
            x=lon_1d,
            y=lat_1d,
            z=values2,
            colorscale=colorscale,
            zmin=vmin,
            zmax=vmax,
            showscale=True,
            colorbar=dict(x=0.65 if show_difference else 1.02),
            hovertemplate="Lon: %{x:.2f}<br>Lat: %{y:.2f}<br>Value: %{z:.2f}<extra></extra>",
        ),
        row=1,
        col=2,
    )

    # Add difference plot
    if show_difference:
        diff = values1 - values2
        diff_max = max(abs(np.nanmin(diff)), abs(np.nanmax(diff)))

        fig.add_trace(
            go.Heatmap(
                x=lon_1d,
                y=lat_1d,
                z=diff,
                colorscale="RdBu_r",
                zmid=0,
                zmin=-diff_max,
                zmax=diff_max,
                showscale=True,
                colorbar=dict(x=1.02, title="Diff"),
                hovertemplate="Lon: %{x:.2f}<br>Lat: %{y:.2f}<br>Diff: %{z:.2f}<extra></extra>",
            ),
            row=1,
            col=3,
        )

    # Add coastlines to all subplots
    for col in range(1, n_cols + 1):
        fig.add_trace(
            go.Scatter(
                x=coast_lons,
                y=coast_lats,
                mode="lines",
                line=dict(color="black", width=0.5),
                hoverinfo="skip",
                showlegend=False,
            ),
            row=1,
            col=col,
        )

    # Update layout
    fig.update_layout(
        height=500,
        margin=dict(l=40, r=40, t=60, b=40),
    )

    # Update axes for all subplots
    for i in range(1, n_cols + 1):
        fig.update_xaxes(
            title_text="Longitude" if i == 2 else "",
            scaleanchor=f"y{i}" if i > 1 else "y",
            row=1,
            col=i,
        )
        fig.update_yaxes(
            title_text="Latitude" if i == 1 else "",
            row=1,
            col=i,
        )

    return fig
