"""Wind arrow overlay component for visualizing wind direction.

Creates quiver/arrow overlays for wind direction data.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go

import config


class WindArrowOverlay:
    """Create wind arrow overlays for Plotly figures."""

    def __init__(
        self,
        subsample: int = 10,
        arrow_scale: float = 0.5,
        color_by_speed: bool = True,
    ):
        """Initialize the wind arrow overlay.

        Args:
            subsample: Subsample factor for arrow density
            arrow_scale: Scale factor for arrow size
            color_by_speed: Whether to color arrows by wind speed
        """
        self.subsample = subsample
        self.arrow_scale = arrow_scale
        self.color_by_speed = color_by_speed

    def create_arrows(
        self,
        lon: np.ndarray,
        lat: np.ndarray,
        u_component: np.ndarray,
        v_component: np.ndarray,
        speed: Optional[np.ndarray] = None,
    ) -> List[go.Scatter]:
        """Create arrow traces for wind direction.

        Uses U and V components to calculate direction.

        Args:
            lon: 1D or 2D longitude array
            lat: 1D or 2D latitude array
            u_component: U wind component (east-west)
            v_component: V wind component (north-south)
            speed: Optional wind speed for coloring

        Returns:
            List of Plotly Scatter traces
        """
        # Ensure 2D arrays
        if lon.ndim == 1 and lat.ndim == 1:
            lon_2d, lat_2d = np.meshgrid(lon, lat)
        else:
            lon_2d, lat_2d = lon, lat

        # Subsample for performance
        lon_sub = lon_2d[:: self.subsample, :: self.subsample]
        lat_sub = lat_2d[:: self.subsample, :: self.subsample]
        u_sub = u_component[:: self.subsample, :: self.subsample]
        v_sub = v_component[:: self.subsample, :: self.subsample]

        if speed is not None:
            speed_sub = speed[:: self.subsample, :: self.subsample]
        else:
            speed_sub = np.sqrt(u_sub ** 2 + v_sub ** 2)

        # Flatten arrays
        lons = lon_sub.flatten()
        lats = lat_sub.flatten()
        us = u_sub.flatten()
        vs = v_sub.flatten()
        speeds = speed_sub.flatten()

        # Remove NaN values
        valid = ~(np.isnan(us) | np.isnan(vs))
        lons = lons[valid]
        lats = lats[valid]
        us = us[valid]
        vs = vs[valid]
        speeds = speeds[valid]

        # Normalize for display
        max_speed = np.nanmax(speeds) if len(speeds) > 0 else 1
        if max_speed == 0:
            max_speed = 1

        # Calculate arrow endpoints
        # Scale factor for arrow length
        scale = self.arrow_scale * (lon_2d.max() - lon_2d.min()) / 50

        dx = us / max_speed * scale
        dy = vs / max_speed * scale

        traces = []

        # Create arrows as individual line segments with arrowheads
        # For efficiency, batch into single trace with None separators
        arrow_lons = []
        arrow_lats = []

        for i in range(len(lons)):
            # Arrow shaft
            x0, y0 = lons[i], lats[i]
            x1, y1 = lons[i] + dx[i], lats[i] + dy[i]

            arrow_lons.extend([x0, x1, None])
            arrow_lats.extend([y0, y1, None])

        # Main arrow shafts
        traces.append(
            go.Scatter(
                x=arrow_lons,
                y=arrow_lats,
                mode="lines",
                line=dict(color="black", width=1),
                hoverinfo="skip",
                showlegend=False,
            )
        )

        # Arrow heads as markers at endpoints
        head_lons = []
        head_lats = []
        head_angles = []

        for i in range(len(lons)):
            x1 = lons[i] + dx[i]
            y1 = lats[i] + dy[i]
            angle = np.arctan2(dy[i], dx[i]) * 180 / np.pi

            head_lons.append(x1)
            head_lats.append(y1)
            head_angles.append(angle)

        # Add arrow heads using markers with rotation
        # Plotly doesn't support rotated markers directly, so we use annotations
        # For simplicity, we'll use circle markers to indicate endpoints

        if self.color_by_speed and len(speeds) > 0:
            traces.append(
                go.Scatter(
                    x=head_lons,
                    y=head_lats,
                    mode="markers",
                    marker=dict(
                        size=5,
                        color=speeds,
                        colorscale="Plasma",
                        cmin=0,
                        cmax=max_speed,
                        showscale=True,
                        colorbar=dict(title="Speed (kts)", x=1.15),
                    ),
                    hovertemplate="Speed: %{marker.color:.1f} kts<extra></extra>",
                    showlegend=False,
                )
            )
        else:
            traces.append(
                go.Scatter(
                    x=head_lons,
                    y=head_lats,
                    mode="markers",
                    marker=dict(size=4, color="black"),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

        return traces

    def create_from_direction(
        self,
        lon: np.ndarray,
        lat: np.ndarray,
        direction: np.ndarray,
        speed: np.ndarray,
    ) -> List[go.Scatter]:
        """Create arrow traces from wind direction and speed.

        Args:
            lon: Longitude array
            lat: Latitude array
            direction: Wind direction in degrees (meteorological convention)
            speed: Wind speed

        Returns:
            List of Plotly Scatter traces
        """
        # Convert meteorological direction to math convention
        # Meteorological: direction FROM which wind blows (0=N, 90=E)
        # Math: direction TO which wind blows
        dir_rad = np.deg2rad(270 - direction)

        # Calculate U and V components
        u = speed * np.cos(dir_rad)
        v = speed * np.sin(dir_rad)

        return self.create_arrows(lon, lat, u, v, speed)


def create_wind_arrows(
    lon: np.ndarray,
    lat: np.ndarray,
    direction: np.ndarray,
    speed: np.ndarray,
    subsample: int = 10,
    arrow_scale: float = 0.5,
) -> List[go.Scatter]:
    """Convenience function to create wind arrows.

    Args:
        lon: Longitude array
        lat: Latitude array
        direction: Wind direction (degrees, met convention)
        speed: Wind speed
        subsample: Subsample factor
        arrow_scale: Arrow size scale

    Returns:
        List of Plotly traces
    """
    overlay = WindArrowOverlay(subsample=subsample, arrow_scale=arrow_scale)
    return overlay.create_from_direction(lon, lat, direction, speed)


def add_wind_arrows_to_figure(
    fig: go.Figure,
    lon: np.ndarray,
    lat: np.ndarray,
    direction: np.ndarray,
    speed: np.ndarray,
    subsample: int = 10,
) -> go.Figure:
    """Add wind arrows to an existing figure.

    Args:
        fig: Plotly figure to add arrows to
        lon: Longitude array
        lat: Latitude array
        direction: Wind direction
        speed: Wind speed
        subsample: Subsample factor

    Returns:
        Updated figure
    """
    traces = create_wind_arrows(lon, lat, direction, speed, subsample)
    for trace in traces:
        fig.add_trace(trace)

    return fig
