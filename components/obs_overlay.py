"""Observation overlay component for displaying weather observations on maps.

Placeholder implementation - format TBD based on observation data format.
"""

from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go


class ObservationOverlay:
    """Create observation overlays for Plotly figures.

    This is a placeholder implementation. The actual implementation will
    depend on the format of observation data when it becomes available.
    """

    def __init__(
        self,
        marker_size: int = 8,
        show_values: bool = True,
    ):
        """Initialize the observation overlay.

        Args:
            marker_size: Size of observation markers
            show_values: Whether to show values as text
        """
        self.marker_size = marker_size
        self.show_values = show_values

    def create_overlay_from_dataframe(
        self,
        df: pd.DataFrame,
        lon_col: str = "longitude",
        lat_col: str = "latitude",
        value_col: str = "value",
        station_col: Optional[str] = "station_id",
        colorscale: str = "Viridis",
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
    ) -> List[go.Scatter]:
        """Create observation overlay from a pandas DataFrame.

        Args:
            df: DataFrame with observation data
            lon_col: Column name for longitude
            lat_col: Column name for latitude
            value_col: Column name for observation values
            station_col: Optional column name for station IDs
            colorscale: Colorscale for markers
            vmin: Minimum value for colorscale
            vmax: Maximum value for colorscale

        Returns:
            List of Plotly Scatter traces
        """
        if df.empty:
            return []

        lons = df[lon_col].values
        lats = df[lat_col].values
        values = df[value_col].values

        if vmin is None:
            vmin = np.nanmin(values)
        if vmax is None:
            vmax = np.nanmax(values)

        # Create hover text
        if station_col and station_col in df.columns:
            hover_text = [
                f"Station: {s}<br>Value: {v:.2f}"
                for s, v in zip(df[station_col], values)
            ]
        else:
            hover_text = [f"Value: {v:.2f}" for v in values]

        traces = []

        # Main marker trace
        traces.append(
            go.Scatter(
                x=lons,
                y=lats,
                mode="markers",
                marker=dict(
                    size=self.marker_size,
                    color=values,
                    colorscale=colorscale,
                    cmin=vmin,
                    cmax=vmax,
                    showscale=True,
                    colorbar=dict(title="Obs Value", x=1.1),
                    line=dict(color="black", width=0.5),
                ),
                hovertext=hover_text,
                hoverinfo="text",
                name="Observations",
            )
        )

        # Optional value labels
        if self.show_values:
            traces.append(
                go.Scatter(
                    x=lons,
                    y=lats,
                    mode="text",
                    text=[f"{v:.1f}" for v in values],
                    textposition="top center",
                    textfont=dict(size=8),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

        return traces

    def create_overlay_from_csv(
        self,
        csv_path: str,
        **kwargs,
    ) -> List[go.Scatter]:
        """Create observation overlay from a CSV file.

        Args:
            csv_path: Path to CSV file
            **kwargs: Additional arguments passed to create_overlay_from_dataframe

        Returns:
            List of Plotly Scatter traces
        """
        df = pd.read_csv(csv_path)
        return self.create_overlay_from_dataframe(df, **kwargs)

    def create_overlay_from_arrays(
        self,
        lons: np.ndarray,
        lats: np.ndarray,
        values: np.ndarray,
        station_ids: Optional[np.ndarray] = None,
        colorscale: str = "Viridis",
    ) -> List[go.Scatter]:
        """Create observation overlay from numpy arrays.

        Args:
            lons: Array of longitudes
            lats: Array of latitudes
            values: Array of observation values
            station_ids: Optional array of station IDs
            colorscale: Colorscale for markers

        Returns:
            List of Plotly Scatter traces
        """
        data = {
            "longitude": lons,
            "latitude": lats,
            "value": values,
        }
        if station_ids is not None:
            data["station_id"] = station_ids

        df = pd.DataFrame(data)
        return self.create_overlay_from_dataframe(
            df, colorscale=colorscale
        )


def create_obs_overlay(
    lons: np.ndarray,
    lats: np.ndarray,
    values: np.ndarray,
    marker_size: int = 8,
    colorscale: str = "Viridis",
) -> List[go.Scatter]:
    """Convenience function to create observation overlay.

    Args:
        lons: Longitude array
        lats: Latitude array
        values: Observation values
        marker_size: Marker size
        colorscale: Colorscale name

    Returns:
        List of Plotly traces
    """
    overlay = ObservationOverlay(marker_size=marker_size)
    return overlay.create_overlay_from_arrays(lons, lats, values, colorscale=colorscale)


def add_obs_to_figure(
    fig: go.Figure,
    lons: np.ndarray,
    lats: np.ndarray,
    values: np.ndarray,
    marker_size: int = 8,
) -> go.Figure:
    """Add observations to an existing figure.

    Args:
        fig: Plotly figure
        lons: Longitude array
        lats: Latitude array
        values: Observation values
        marker_size: Marker size

    Returns:
        Updated figure
    """
    traces = create_obs_overlay(lons, lats, values, marker_size)
    for trace in traces:
        fig.add_trace(trace)

    return fig
