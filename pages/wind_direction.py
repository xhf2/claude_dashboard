"""Wind Direction page with arrow overlays."""

import dash
from dash import dcc, html, callback, Input, Output, State
import plotly.graph_objects as go
import numpy as np

import config
from utils.file_scanner import FileScanner
from utils.data_loader import DataLoader
from utils.regrid import regrid_to_latlon
from components.map_plot import MapPlot
from components.wind_arrows import add_wind_arrows_to_figure
from components.colorscale import get_wind_direction_colorscale
from components.dropdowns import (
    DropdownBuilder,
    create_dropdown_options,
)

# Register page
dash.register_page(__name__, path="/wind-direction", name="Wind Direction")

# Page-specific parameters - need both direction and speed
PAGE_PARAMS = ["winddir10m", "windspd10m"]
PAGE_ID = "winddir"

# Layout
layout = html.Div(
    [
        html.H2("Wind Direction"),
        html.Hr(),
        # Controls - simpler layout without parameter dropdown
        html.Div(
            [
                html.Div(
                    [
                        html.Label("Realm"),
                        dcc.Dropdown(
                            id=f"{PAGE_ID}-realm-dropdown",
                            options=[],
                            value=None,
                            clearable=False,
                        ),
                    ],
                    style={"width": "15%", "display": "inline-block", "marginRight": "10px"},
                ),
                html.Div(
                    [
                        html.Label("Cycle"),
                        dcc.Dropdown(
                            id=f"{PAGE_ID}-cycle-dropdown",
                            options=[],
                            value=None,
                            clearable=False,
                        ),
                    ],
                    style={"width": "18%", "display": "inline-block", "marginRight": "10px"},
                ),
                html.Div(
                    [
                        html.Label("Step"),
                        dcc.Dropdown(
                            id=f"{PAGE_ID}-step-dropdown",
                            options=[],
                            value=None,
                            clearable=False,
                        ),
                    ],
                    style={"width": "12%", "display": "inline-block", "marginRight": "10px"},
                ),
                html.Div(
                    [
                        html.Label("Output"),
                        dcc.Dropdown(
                            id=f"{PAGE_ID}-output-dropdown",
                            options=[],
                            value=None,
                            clearable=False,
                        ),
                    ],
                    style={"width": "18%", "display": "inline-block", "marginRight": "10px"},
                ),
                html.Div(
                    [
                        html.Label("Valid Time"),
                        dcc.Dropdown(
                            id=f"{PAGE_ID}-validtime-dropdown",
                            options=[],
                            value=None,
                            clearable=False,
                        ),
                    ],
                    style={"width": "18%", "display": "inline-block"},
                ),
            ],
            style={"marginBottom": "20px"},
        ),
        # Arrow density slider
        html.Div(
            [
                html.Label("Arrow Density: "),
                dcc.Slider(
                    id=f"{PAGE_ID}-density-slider",
                    min=5,
                    max=30,
                    step=5,
                    value=15,
                    marks={i: str(i) for i in [5, 10, 15, 20, 25, 30]},
                    tooltip={"placement": "bottom"},
                ),
            ],
            style={"width": "300px", "marginBottom": "20px"},
        ),
        # Refresh button
        html.Button("Refresh Data", id=f"{PAGE_ID}-refresh-btn", n_clicks=0),
        html.Hr(),
        # Loading wrapper
        dcc.Loading(
            id=f"{PAGE_ID}-loading",
            type="default",
            children=[
                # Main plot
                dcc.Graph(id=f"{PAGE_ID}-map", style={"height": "650px"}),
                # Info display
                html.Div(id=f"{PAGE_ID}-info"),
            ],
        ),
    ]
)


# Callbacks for dropdown cascading
@callback(
    Output(f"{PAGE_ID}-realm-dropdown", "options"),
    Output(f"{PAGE_ID}-realm-dropdown", "value"),
    Input(f"{PAGE_ID}-refresh-btn", "n_clicks"),
)
def update_realm_options(n_clicks):
    builder = DropdownBuilder()
    options = builder.get_realm_options()
    return create_dropdown_options(options, "test_realm")


@callback(
    Output(f"{PAGE_ID}-cycle-dropdown", "options"),
    Output(f"{PAGE_ID}-cycle-dropdown", "value"),
    Input(f"{PAGE_ID}-realm-dropdown", "value"),
    Input(f"{PAGE_ID}-refresh-btn", "n_clicks"),
)
def update_cycle_options(realm, n_clicks):
    if not realm:
        return [], None
    builder = DropdownBuilder(realm)
    options = builder.get_cycle_options()
    return create_dropdown_options(options)


@callback(
    Output(f"{PAGE_ID}-step-dropdown", "options"),
    Output(f"{PAGE_ID}-step-dropdown", "value"),
    Input(f"{PAGE_ID}-cycle-dropdown", "value"),
    State(f"{PAGE_ID}-realm-dropdown", "value"),
)
def update_step_options(cycle, realm):
    if not cycle or not realm:
        return [], None
    builder = DropdownBuilder(realm)
    options = builder.get_step_options(cycle)
    return create_dropdown_options(options, "blend")


@callback(
    Output(f"{PAGE_ID}-output-dropdown", "options"),
    Output(f"{PAGE_ID}-output-dropdown", "value"),
    Input(f"{PAGE_ID}-step-dropdown", "value"),
    State(f"{PAGE_ID}-cycle-dropdown", "value"),
    State(f"{PAGE_ID}-realm-dropdown", "value"),
)
def update_output_options(step, cycle, realm):
    if not step or not cycle or not realm:
        return [], None
    builder = DropdownBuilder(realm)
    # Check for wind direction parameter
    params = builder.scanner.get_parameters(cycle, step)
    if "winddir10m" in params:
        options = builder.get_output_options(cycle, step, "winddir10m")
    else:
        options = []
    return create_dropdown_options(options, "expectedvalues_extract")


@callback(
    Output(f"{PAGE_ID}-validtime-dropdown", "options"),
    Output(f"{PAGE_ID}-validtime-dropdown", "value"),
    Input(f"{PAGE_ID}-output-dropdown", "value"),
    State(f"{PAGE_ID}-step-dropdown", "value"),
    State(f"{PAGE_ID}-cycle-dropdown", "value"),
    State(f"{PAGE_ID}-realm-dropdown", "value"),
)
def update_validtime_options(output, step, cycle, realm):
    if not output or not step or not cycle or not realm:
        return [], None
    builder = DropdownBuilder(realm)
    options = builder.get_validtime_options(cycle, step, "winddir10m", output)
    return create_dropdown_options(options)


# Main plot callback
@callback(
    Output(f"{PAGE_ID}-map", "figure"),
    Output(f"{PAGE_ID}-info", "children"),
    Input(f"{PAGE_ID}-validtime-dropdown", "value"),
    Input(f"{PAGE_ID}-density-slider", "value"),
    State(f"{PAGE_ID}-output-dropdown", "value"),
    State(f"{PAGE_ID}-step-dropdown", "value"),
    State(f"{PAGE_ID}-cycle-dropdown", "value"),
    State(f"{PAGE_ID}-realm-dropdown", "value"),
)
def update_plot(validtime, density, output, step, cycle, realm):
    empty_fig = go.Figure()
    empty_fig.update_layout(
        title="Select data to display",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )

    if not all([validtime, output, step, cycle, realm]):
        return empty_fig, ""

    scanner = FileScanner(realm)

    # Load wind direction
    dir_file = scanner.get_file_for_validtime(cycle, step, "winddir10m", output, validtime)
    # Load wind speed
    spd_file = scanner.get_file_for_validtime(cycle, step, "windspd10m", output, validtime)

    if not dir_file:
        return empty_fig, f"Wind direction file not found for {validtime}"

    try:
        # Load direction data
        dir_loader = DataLoader(dir_file, "winddir10m", validtime)
        dir_regridded = regrid_to_latlon(dir_loader)

        lon = dir_regridded.lon.values
        lat = dir_regridded.lat.values
        direction = dir_regridded.values

        # Load speed if available
        if spd_file:
            spd_loader = DataLoader(spd_file, "windspd10m", validtime)
            spd_regridded = regrid_to_latlon(spd_loader)
            speed = spd_regridded.values
            spd_loader.close()
        else:
            # Default speed for arrow length
            speed = np.ones_like(direction) * 10

        # Create base figure with wind direction colormap
        colorscale, vmin, vmax = get_wind_direction_colorscale()

        map_plot = MapPlot("winddir10m", use_preset_colorscale=False)
        fig = map_plot.create_figure(
            lon, lat, direction,
            title=f"Wind Direction - {validtime}",
            show_coastlines=True,
            colorscale=colorscale,
            vmin=vmin,
            vmax=vmax,
        )

        # Update colorbar for direction
        fig.update_traces(
            colorbar=dict(
                title="Direction (°)",
                tickvals=[0, 90, 180, 270, 360],
                ticktext=["N", "E", "S", "W", "N"],
            ),
            selector=dict(type="heatmap"),
        )

        # Add wind arrows
        subsample = max(5, density)  # Ensure minimum subsample
        fig = add_wind_arrows_to_figure(fig, lon, lat, direction, speed, subsample=subsample)

        dir_loader.close()

        info_text = f"{realm} | {cycle} | {step} | Wind Direction | {output} | {validtime}"

        return fig, html.Div([
            html.P(info_text, style={"color": "gray", "fontSize": "12px"}),
            html.P(
                "Arrows show wind direction (from), colored by speed. "
                "Background shows direction degrees.",
                style={"color": "gray", "fontSize": "11px"},
            ),
        ])

    except Exception as e:
        error_fig = go.Figure()
        error_fig.update_layout(
            title=f"Error loading data: {str(e)}",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
        )
        return error_fig, f"Error: {str(e)}"
