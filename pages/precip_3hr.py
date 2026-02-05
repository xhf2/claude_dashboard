"""3-Hourly Precipitation page for short-range precipitation visualization."""

import dash
from dash import dcc, html, callback, Input, Output, State
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

import config
from utils.file_scanner import FileScanner
from utils.data_loader import DataLoader
from utils.regrid import regrid_to_latlon
from components.colorscale import get_colorscale, get_colorscale_properties
from components.dropdowns import (
    DropdownBuilder,
    create_dropdown_options,
    create_colorscale_toggle,
    create_comparison_toggle,
    create_projection_toggle,
)
from utils.coastline import get_coastline_scatter_data

# Register page
dash.register_page(__name__, path="/precip-3hr", name="3hr Precipitation")

# Page-specific parameters
PAGE_PARAMS = ["precipacc03h"]
PAGE_ID = "precip3hr"

# Output types
PROBABILITY_OUTPUTS = ["blendgrids", "recfilter", "probabilities_extract", "merge_probabilities"]
PERCENTILE_OUTPUTS = ["percentiles_extract", "merge_percentiles"]

# Field definitions for probability data (from np_fields) - 3hr PoP thresholds
# Values are in meters
POP_FIELDS = {
    "PoP (0.2mm)": 0.0002,
    "PoP1 (1mm)": 0.001,
    "PoP5 (5mm)": 0.005,
    "PoP10 (10mm)": 0.010,
    "PoP15 (15mm)": 0.015,
    "PoP25 (25mm)": 0.025,
    "PoP50 (50mm)": 0.050,
}

# Field definitions for percentile data (from np_fields)
PRECIP_PERCENTILE_FIELDS = {
    "Precip10Pct (90th pctl)": 90.0,
    "Precip25Pct (75th pctl)": 75.0,
    "Precip50Pct (50th pctl)": 50.0,
    "Precip75Pct (25th pctl)": 25.0,
}

# Layout
layout = html.Div(
    [
        html.H2("3-Hourly Precipitation"),
        html.Hr(),
        html.Div(
            [
                html.Div(
                    [
                        html.Label("Realm"),
                        dcc.Dropdown(id=f"{PAGE_ID}-realm-dropdown", options=[], value=None, clearable=False),
                    ],
                    style={"width": "12%", "display": "inline-block", "marginRight": "10px"},
                ),
                html.Div(
                    [
                        html.Label("Cycle"),
                        dcc.Dropdown(id=f"{PAGE_ID}-cycle-dropdown", options=[], value=None, clearable=False),
                    ],
                    style={"width": "15%", "display": "inline-block", "marginRight": "10px"},
                ),
                html.Div(
                    [
                        html.Label("Step"),
                        dcc.Dropdown(id=f"{PAGE_ID}-step-dropdown", options=[], value=None, clearable=False),
                    ],
                    style={"width": "10%", "display": "inline-block", "marginRight": "10px"},
                ),
                html.Div(
                    [
                        html.Label("Parameter"),
                        dcc.Dropdown(id=f"{PAGE_ID}-parameter-dropdown", options=[], value=None, clearable=False),
                    ],
                    style={"width": "15%", "display": "inline-block", "marginRight": "10px"},
                ),
                html.Div(
                    [
                        html.Label("Output"),
                        dcc.Dropdown(id=f"{PAGE_ID}-output-dropdown", options=[], value=None, clearable=False),
                    ],
                    style={"width": "18%", "display": "inline-block", "marginRight": "10px"},
                ),
                html.Div(
                    [
                        html.Label("Valid Time"),
                        dcc.Dropdown(id=f"{PAGE_ID}-validtime-dropdown", options=[], value=None, clearable=False),
                    ],
                    style={"width": "15%", "display": "inline-block"},
                ),
            ],
            style={"marginBottom": "10px"},
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.Label("Field"),
                        dcc.Dropdown(id=f"{PAGE_ID}-field-dropdown", options=[], value=None, clearable=False),
                    ],
                    id=f"{PAGE_ID}-field-container",
                    style={"width": "250px", "display": "none"},
                ),
            ],
            style={"marginBottom": "10px"},
        ),
        html.Div(
            [
                create_colorscale_toggle(PAGE_ID),
                create_projection_toggle(PAGE_ID),
                create_comparison_toggle(PAGE_ID),
            ],
            style={"marginBottom": "10px"},
        ),
        html.Button("Refresh Data", id=f"{PAGE_ID}-refresh-btn", n_clicks=0),
        html.Hr(),
        dcc.Loading(
            id=f"{PAGE_ID}-loading",
            type="default",
            children=[
                dcc.Graph(id=f"{PAGE_ID}-map", style={"height": "650px"}),
                html.Div(id=f"{PAGE_ID}-info"),
            ],
        ),
    ]
)


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
    Output(f"{PAGE_ID}-parameter-dropdown", "options"),
    Output(f"{PAGE_ID}-parameter-dropdown", "value"),
    Input(f"{PAGE_ID}-step-dropdown", "value"),
    State(f"{PAGE_ID}-cycle-dropdown", "value"),
    State(f"{PAGE_ID}-realm-dropdown", "value"),
)
def update_parameter_options(step, cycle, realm):
    if not step or not cycle or not realm:
        return [], None
    builder = DropdownBuilder(realm)
    options = builder.get_parameter_options(cycle, step, PAGE_PARAMS)
    return create_dropdown_options(options, "precipacc03h")


@callback(
    Output(f"{PAGE_ID}-output-dropdown", "options"),
    Output(f"{PAGE_ID}-output-dropdown", "value"),
    Input(f"{PAGE_ID}-parameter-dropdown", "value"),
    State(f"{PAGE_ID}-step-dropdown", "value"),
    State(f"{PAGE_ID}-cycle-dropdown", "value"),
    State(f"{PAGE_ID}-realm-dropdown", "value"),
)
def update_output_options(parameter, step, cycle, realm):
    if not parameter or not step or not cycle or not realm:
        return [], None
    builder = DropdownBuilder(realm)
    options = builder.get_output_options(cycle, step, parameter)
    return create_dropdown_options(options, "expectedvalues_extract")


@callback(
    Output(f"{PAGE_ID}-validtime-dropdown", "options"),
    Output(f"{PAGE_ID}-validtime-dropdown", "value"),
    Input(f"{PAGE_ID}-output-dropdown", "value"),
    State(f"{PAGE_ID}-parameter-dropdown", "value"),
    State(f"{PAGE_ID}-step-dropdown", "value"),
    State(f"{PAGE_ID}-cycle-dropdown", "value"),
    State(f"{PAGE_ID}-realm-dropdown", "value"),
)
def update_validtime_options(output, parameter, step, cycle, realm):
    if not output or not parameter or not step or not cycle or not realm:
        return [], None
    builder = DropdownBuilder(realm)
    options = builder.get_validtime_options(cycle, step, parameter, output)
    return create_dropdown_options(options)


@callback(
    Output(f"{PAGE_ID}-field-container", "style"),
    Output(f"{PAGE_ID}-field-dropdown", "options"),
    Output(f"{PAGE_ID}-field-dropdown", "value"),
    Input(f"{PAGE_ID}-output-dropdown", "value"),
    Input(f"{PAGE_ID}-validtime-dropdown", "value"),
    State(f"{PAGE_ID}-parameter-dropdown", "value"),
    State(f"{PAGE_ID}-step-dropdown", "value"),
    State(f"{PAGE_ID}-cycle-dropdown", "value"),
    State(f"{PAGE_ID}-realm-dropdown", "value"),
)
def update_field_options(output, validtime, parameter, step, cycle, realm):
    hidden_style = {"width": "250px", "display": "none"}
    visible_style = {"width": "250px", "display": "inline-block", "marginRight": "10px"}

    if not output or not validtime:
        return hidden_style, [], None

    # Probability outputs - use predefined PoP fields for 3hr precip
    if output in PROBABILITY_OUTPUTS:
        options = [{"label": k, "value": str(v)} for k, v in POP_FIELDS.items()]
        return visible_style, options, str(POP_FIELDS["PoP (0.2mm)"])

    # Percentile outputs
    if output in PERCENTILE_OUTPUTS:
        options = [{"label": k, "value": str(v)} for k, v in PRECIP_PERCENTILE_FIELDS.items()]
        return visible_style, options, "50.0"

    return hidden_style, [], None


@callback(
    Output(f"{PAGE_ID}-realm2-container", "style"),
    Input(f"{PAGE_ID}-comparison-toggle", "value"),
)
def toggle_realm2_visibility(compare_value):
    if "compare" in compare_value:
        return {"display": "block", "marginTop": "10px"}
    return {"display": "none"}


@callback(
    Output(f"{PAGE_ID}-realm2-dropdown", "options"),
    Input(f"{PAGE_ID}-realm-dropdown", "value"),
)
def update_realm2_options(realm):
    builder = DropdownBuilder()
    options = builder.get_realm_options()
    options = [o for o in options if o["value"] != realm]
    return options


def load_and_process_data(realm, cycle, step, parameter, output, validtime, field_value):
    """Load and process data for a single realm.

    Returns:
        Tuple of (lon, lat, values, data_type, colorscale_type, field_label, cycle_str, lead_hour) or None on error
    """
    scanner = FileScanner(realm)
    file_path = scanner.get_file_for_validtime(cycle, step, parameter, output, validtime)

    if not file_path:
        return None

    loader = DataLoader(file_path, parameter, validtime)
    data_type = loader.data_type
    field_label = ""
    colorscale_type = "Precip"

    if data_type == "threshold" and field_value:
        threshold_m = float(field_value)
        threshold_mm = threshold_m * 1000
        data_to_plot = loader.select_threshold(threshold_m)
        field_label = f" - PoP >{threshold_mm:.1f}mm"
        colorscale_type = "PoP"
    elif data_type == "percentile" and field_value:
        percentile = float(field_value)
        data_to_plot = loader.select_percentile(percentile)
        field_name = None
        for name, pct in PRECIP_PERCENTILE_FIELDS.items():
            if abs(pct - percentile) < 0.1:
                field_name = name.split(" (")[0]
                break
        field_label = f" - {field_name}" if field_name else f" - {int(percentile)}th Percentile"
    else:
        data_to_plot = loader.data

    class TempLoader:
        pass
    temp_loader = TempLoader()
    temp_loader.proj4str = loader.proj4str
    temp_loader.data = data_to_plot

    regridded = regrid_to_latlon(temp_loader)

    cycle_str = loader.get_basetime_str()
    lead_hour = loader.leadhour
    loader.close()

    return (regridded.lon.values, regridded.lat.values, regridded.values,
            data_type, colorscale_type, field_label, cycle_str, lead_hour)


@callback(
    Output(f"{PAGE_ID}-map", "figure"),
    Output(f"{PAGE_ID}-info", "children"),
    Input(f"{PAGE_ID}-validtime-dropdown", "value"),
    Input(f"{PAGE_ID}-field-dropdown", "value"),
    Input(f"{PAGE_ID}-colorscale-toggle", "value"),
    Input(f"{PAGE_ID}-comparison-toggle", "value"),
    Input(f"{PAGE_ID}-realm2-dropdown", "value"),
    State(f"{PAGE_ID}-output-dropdown", "value"),
    State(f"{PAGE_ID}-parameter-dropdown", "value"),
    State(f"{PAGE_ID}-step-dropdown", "value"),
    State(f"{PAGE_ID}-cycle-dropdown", "value"),
    State(f"{PAGE_ID}-realm-dropdown", "value"),
)
def update_plot(validtime, field_value, colorscale_mode, compare_value, realm2,
                output, parameter, step, cycle, realm):
    empty_fig = go.Figure()
    empty_fig.update_layout(title="Select data to display", xaxis=dict(visible=False), yaxis=dict(visible=False))

    if not all([validtime, output, parameter, step, cycle, realm]):
        return empty_fig, ""

    # Check if comparison mode is enabled
    compare_mode = "compare" in (compare_value or []) and realm2

    try:
        # Load data for primary realm
        result1 = load_and_process_data(realm, cycle, step, parameter, output, validtime, field_value)
        if result1 is None:
            return empty_fig, f"File not found for {realm}"

        lon1, lat1, values1, data_type, colorscale_type, field_label, cycle_str, lead_hour = result1

        # Determine colorscale
        use_preset = colorscale_mode == "preset"
        if use_preset:
            colorscale, vmin, vmax = get_colorscale(colorscale_type, True)
        else:
            colorscale = "Plasma"
            vmin = float(np.nanmin(values1))
            vmax = float(np.nanpercentile(values1[~np.isnan(values1)], 99))

        # Get coastlines
        coast_lons, coast_lats = get_coastline_scatter_data(source="gfe", simplify_tolerance=0.01)

        # Build type label
        type_label = {
            "expectedvalue": "Expected Value",
            "percentile": "Percentile",
            "threshold": "Probability",
        }.get(data_type, data_type)

        if compare_mode:
            # Load data for second realm
            result2 = load_and_process_data(realm2, cycle, step, parameter, output, validtime, field_value)

            if result2 is None:
                return empty_fig, f"File not found for {realm2}"

            lon2, lat2, values2, _, _, _, _, _ = result2

            # Update vmin/vmax to encompass both datasets if using auto colorscale
            if not use_preset:
                vmin = min(vmin, float(np.nanmin(values2)))
                vmax = max(vmax, float(np.nanpercentile(values2[~np.isnan(values2)], 99)))

            # Create side-by-side subplots
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=[f"{realm}", f"{realm2}"],
                horizontal_spacing=0.1,
            )

            # Add heatmap for realm 1
            fig.add_trace(
                go.Heatmap(
                    x=lon1, y=lat1, z=values1,
                    colorscale=colorscale, zmin=vmin, zmax=vmax,
                    colorbar=dict(
                        title=get_colorscale_properties(colorscale_type).get("cbar_label", ""),
                        tickvals=get_colorscale_properties(colorscale_type).get("cbar_labels", []),
                        x=0.45, len=0.9,
                    ),
                    hovertemplate="Lon: %{x:.2f}<br>Lat: %{y:.2f}<br>Value: %{z:.2f}<extra></extra>",
                ),
                row=1, col=1
            )

            # Add coastlines for realm 1
            fig.add_trace(
                go.Scatter(
                    x=coast_lons, y=coast_lats, mode="lines",
                    line=dict(color="black", width=0.5),
                    hoverinfo="skip", showlegend=False,
                ),
                row=1, col=1
            )

            # Add heatmap for realm 2
            fig.add_trace(
                go.Heatmap(
                    x=lon2, y=lat2, z=values2,
                    colorscale=colorscale, zmin=vmin, zmax=vmax,
                    colorbar=dict(
                        title=get_colorscale_properties(colorscale_type).get("cbar_label", ""),
                        tickvals=get_colorscale_properties(colorscale_type).get("cbar_labels", []),
                        x=1.0, len=0.9,
                    ),
                    hovertemplate="Lon: %{x:.2f}<br>Lat: %{y:.2f}<br>Value: %{z:.2f}<extra></extra>",
                ),
                row=1, col=2
            )

            # Add coastlines for realm 2
            fig.add_trace(
                go.Scatter(
                    x=coast_lons, y=coast_lats, mode="lines",
                    line=dict(color="black", width=0.5),
                    hoverinfo="skip", showlegend=False,
                ),
                row=1, col=2
            )

            # Calculate common axis range from data extent
            lon_min = min(lon1.min(), lon2.min())
            lon_max = max(lon1.max(), lon2.max())
            lat_min = min(lat1.min(), lat2.min())
            lat_max = max(lat1.max(), lat2.max())

            # Update axes for both subplots
            fig.update_xaxes(title_text="Longitude", range=[lon_min, lon_max], row=1, col=1)
            fig.update_yaxes(title_text="Latitude", range=[lat_min, lat_max], scaleanchor="x", row=1, col=1)
            fig.update_xaxes(title_text="Longitude", range=[lon_min, lon_max], row=1, col=2)
            fig.update_yaxes(title_text="Latitude", range=[lat_min, lat_max], scaleanchor="x2", row=1, col=2)

            title = f"{config.PARAMETERS.get(parameter, parameter)}{field_label}<br>"
            title += f"<sub>Cycle: {cycle_str} | Lead: T+{lead_hour}h | Type: {type_label}</sub>"

            fig.update_layout(
                title=dict(text=title, x=0.5),
                height=600,
                margin=dict(l=60, r=60, t=100, b=60),
            )

            info_text = f"Comparing: {realm} vs {realm2} | {cycle} | {step} | {parameter} | {output} | {validtime}"

        else:
            # Single plot mode
            fig = go.Figure()

            fig.add_trace(
                go.Heatmap(
                    x=lon1, y=lat1, z=values1,
                    colorscale=colorscale, zmin=vmin, zmax=vmax,
                    colorbar=dict(
                        title=get_colorscale_properties(colorscale_type).get("cbar_label", ""),
                        tickvals=get_colorscale_properties(colorscale_type).get("cbar_labels", []),
                    ),
                    hovertemplate="Lon: %{x:.2f}<br>Lat: %{y:.2f}<br>Value: %{z:.2f}<extra></extra>",
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=coast_lons, y=coast_lats, mode="lines",
                    line=dict(color="black", width=0.5),
                    hoverinfo="skip", showlegend=False,
                )
            )

            title = f"{config.PARAMETERS.get(parameter, parameter)}{field_label}<br>"
            title += f"<sub>Cycle: {cycle_str} | Lead: T+{lead_hour}h | Type: {type_label}</sub>"

            fig.update_layout(
                title=dict(text=title, x=0.5),
                xaxis=dict(title="Longitude", range=[lon1.min(), lon1.max()], scaleanchor="y"),
                yaxis=dict(title="Latitude", range=[lat1.min(), lat1.max()]),
                margin=dict(l=60, r=60, t=80, b=60),
                height=650,
            )

            info_text = f"{realm} | {cycle} | {step} | {parameter} | {output} | {validtime}"

        if field_label:
            info_text += f" | {field_label.strip(' - ')}"

        return fig, html.P(info_text, style={"color": "gray", "fontSize": "12px"})

    except Exception as e:
        import traceback
        error_fig = go.Figure()
        error_fig.update_layout(title=f"Error: {str(e)}", xaxis=dict(visible=False), yaxis=dict(visible=False))
        return error_fig, f"Error: {str(e)}\n{traceback.format_exc()}"
