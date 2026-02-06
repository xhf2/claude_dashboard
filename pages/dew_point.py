"""Dew Point page for screen and pressure level dew point temperature visualization."""

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
from utils.coastline import get_coastline_scatter_data, get_coastline_scatter_data_albers

# Register page
dash.register_page(__name__, path="/dew-point", name="Dew Point")

# Page-specific parameters - surface and pressure levels
PAGE_PARAMS = [
    "tempdewscreen",
    "tempdew200hPa",
    "tempdew300hPa",
    "tempdew400hPa",
    "tempdew500hPa",
    "tempdew700hPa",
    "tempdew800hPa",
    "tempdew850hPa",
    "tempdew900hPa",
    "tempdew950hPa",
]
PAGE_ID = "dewpoint"

# Use output types from config
PROBABILITY_OUTPUTS = config.PROBABILITY_OUTPUT_TYPES
PERCENTILE_OUTPUTS = config.PERCENTILE_OUTPUT_TYPES

# Layout
layout = html.Div(
    [
        html.H2("Dew Point Temperature"),
        html.Hr(),
        # Controls row 1: main dropdowns
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
                    style={"width": "12%", "display": "inline-block", "marginRight": "10px"},
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
                    style={"width": "15%", "display": "inline-block", "marginRight": "10px"},
                ),
                html.Div(
                    [
                        html.Label("Model"),
                        dcc.Dropdown(
                            id=f"{PAGE_ID}-model-dropdown",
                            options=[],
                            value=None,
                            clearable=False,
                        ),
                    ],
                    style={"width": "10%", "display": "inline-block", "marginRight": "10px"},
                ),
                html.Div(
                    [
                        html.Label("Parameter"),
                        dcc.Dropdown(
                            id=f"{PAGE_ID}-parameter-dropdown",
                            options=[],
                            value=None,
                            clearable=False,
                        ),
                    ],
                    style={"width": "15%", "display": "inline-block", "marginRight": "10px"},
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
                    style={"width": "15%", "display": "inline-block"},
                ),
            ],
            style={"marginBottom": "10px"},
        ),
        # Controls row 2: Field dropdown
        html.Div(
            [
                html.Div(
                    [
                        html.Label("Field"),
                        dcc.Dropdown(
                            id=f"{PAGE_ID}-field-dropdown",
                            options=[],
                            value=None,
                            clearable=False,
                        ),
                    ],
                    id=f"{PAGE_ID}-field-container",
                    style={"width": "200px", "display": "none"},
                ),
            ],
            style={"marginBottom": "10px"},
        ),
        # Controls row 3: toggles
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
    Output(f"{PAGE_ID}-model-dropdown", "options"),
    Output(f"{PAGE_ID}-model-dropdown", "value"),
    Input(f"{PAGE_ID}-cycle-dropdown", "value"),
    State(f"{PAGE_ID}-realm-dropdown", "value"),
)
def update_model_options(cycle, realm):
    if not cycle or not realm:
        return [], None
    builder = DropdownBuilder(realm)
    options = builder.get_model_options(cycle)
    return create_dropdown_options(options, "blend")


@callback(
    Output(f"{PAGE_ID}-parameter-dropdown", "options"),
    Output(f"{PAGE_ID}-parameter-dropdown", "value"),
    Input(f"{PAGE_ID}-model-dropdown", "value"),
    State(f"{PAGE_ID}-cycle-dropdown", "value"),
    State(f"{PAGE_ID}-realm-dropdown", "value"),
)
def update_parameter_options(model, cycle, realm):
    if not model or not cycle or not realm:
        return [], None
    builder = DropdownBuilder(realm)
    options = builder.get_parameter_options(cycle, model, PAGE_PARAMS)
    return create_dropdown_options(options, "tempdewscreen")


@callback(
    Output(f"{PAGE_ID}-output-dropdown", "options"),
    Output(f"{PAGE_ID}-output-dropdown", "value"),
    Input(f"{PAGE_ID}-parameter-dropdown", "value"),
    State(f"{PAGE_ID}-model-dropdown", "value"),
    State(f"{PAGE_ID}-cycle-dropdown", "value"),
    State(f"{PAGE_ID}-realm-dropdown", "value"),
)
def update_output_options(parameter, model, cycle, realm):
    if not parameter or not model or not cycle or not realm:
        return [], None
    builder = DropdownBuilder(realm)
    options = builder.get_output_options(cycle, model, parameter)
    return create_dropdown_options(options, "expectedvalues_extract")


@callback(
    Output(f"{PAGE_ID}-validtime-dropdown", "options"),
    Output(f"{PAGE_ID}-validtime-dropdown", "value"),
    Input(f"{PAGE_ID}-output-dropdown", "value"),
    State(f"{PAGE_ID}-parameter-dropdown", "value"),
    State(f"{PAGE_ID}-model-dropdown", "value"),
    State(f"{PAGE_ID}-cycle-dropdown", "value"),
    State(f"{PAGE_ID}-realm-dropdown", "value"),
)
def update_validtime_options(output, parameter, model, cycle, realm):
    if not output or not parameter or not model or not cycle or not realm:
        return [], None
    builder = DropdownBuilder(realm)
    options = builder.get_validtime_options(cycle, model, parameter, output)
    return create_dropdown_options(options)


# Field dropdown based on output type
@callback(
    Output(f"{PAGE_ID}-field-container", "style"),
    Output(f"{PAGE_ID}-field-dropdown", "options"),
    Output(f"{PAGE_ID}-field-dropdown", "value"),
    Input(f"{PAGE_ID}-output-dropdown", "value"),
    Input(f"{PAGE_ID}-validtime-dropdown", "value"),
    State(f"{PAGE_ID}-parameter-dropdown", "value"),
    State(f"{PAGE_ID}-model-dropdown", "value"),
    State(f"{PAGE_ID}-cycle-dropdown", "value"),
    State(f"{PAGE_ID}-realm-dropdown", "value"),
)
def update_field_options(output, validtime, parameter, model, cycle, realm):
    hidden_style = {"width": "200px", "display": "none"}
    visible_style = {"width": "250px", "display": "inline-block", "marginRight": "10px"}

    if not output or not validtime:
        return hidden_style, [], None

    # Probability outputs - use POTd fields from config
    if output in PROBABILITY_OUTPUTS:
        potd_fields = config.get_probability_fields(parameter)
        if potd_fields:
            options = [{"label": k, "value": str(v)} for k, v in potd_fields.items()]
            default_value = str(list(potd_fields.values())[len(potd_fields)//2])
            return visible_style, options, default_value
        return hidden_style, [], None

    # Percentile outputs - use config fields
    if output in PERCENTILE_OUTPUTS:
        pctl_fields = config.get_percentile_fields(parameter)
        if pctl_fields:
            options = [{"label": k, "value": str(v)} for k, v in pctl_fields.items()]
            return visible_style, options, "50.0"
        return hidden_style, [], None

    return hidden_style, [], None


# Comparison mode callbacks
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


def load_and_process_data(realm, cycle, model, parameter, output, validtime, field_value, use_native_proj=False):
    """Load and process data for a single realm.

    Args:
        use_native_proj: If True, return native Albers coordinates instead of lat/lon

    Returns:
        Tuple of (x, y, values, data_type, colorscale_type, field_label, cycle_str, lead_hour, proj4str) or None on error
    """
    scanner = FileScanner(realm)
    file_path = scanner.get_file_for_validtime(cycle, model, parameter, output, validtime)

    if not file_path:
        return None

    loader = DataLoader(file_path, parameter, validtime)
    data_type = loader.data_type
    field_label = ""
    colorscale_type = "Td"
    proj4str = loader.proj4str

    if data_type == "threshold" and field_value:
        threshold_k = float(field_value)
        threshold_c = threshold_k - 273.15
        data_to_plot = loader.select_threshold(threshold_k)
        # Look up field name from config
        potd_fields = config.get_probability_fields(parameter)
        field_name = None
        for name, val in potd_fields.items():
            if abs(val - threshold_k) < 0.1:
                field_name = name.split(" (")[0]
                break
        field_label = f" - {field_name}" if field_name else f" - POTd >{threshold_c:.1f}C"
        colorscale_type = "PoP"
    elif data_type == "percentile" and field_value:
        percentile = float(field_value)
        data_to_plot = loader.select_percentile(percentile)
        pctl_fields = config.get_percentile_fields(parameter)
        field_name = None
        for name, pct in pctl_fields.items():
            if abs(pct - percentile) < 0.1:
                field_name = name.split(" (")[0]
                break
        field_label = f" - {field_name}" if field_name else f" - {int(percentile)}th Percentile"
    else:
        data_to_plot = loader.data

    cycle_str = loader.get_basetime_str()
    lead_hour = loader.leadhour

    if use_native_proj:
        x_coords = data_to_plot.projection_x_coordinate.values / 1000
        y_coords = data_to_plot.projection_y_coordinate.values / 1000
        values = data_to_plot.values
        loader.close()
        return (x_coords, y_coords, values, data_type, colorscale_type, field_label, cycle_str, lead_hour, proj4str)
    else:
        class TempLoader:
            pass
        temp_loader = TempLoader()
        temp_loader.proj4str = loader.proj4str
        temp_loader.data = data_to_plot

        regridded = regrid_to_latlon(temp_loader)
        loader.close()

        return (regridded.lon.values, regridded.lat.values, regridded.values,
                data_type, colorscale_type, field_label, cycle_str, lead_hour, proj4str)


# Main plot callback
@callback(
    Output(f"{PAGE_ID}-map", "figure"),
    Output(f"{PAGE_ID}-info", "children"),
    Input(f"{PAGE_ID}-validtime-dropdown", "value"),
    Input(f"{PAGE_ID}-field-dropdown", "value"),
    Input(f"{PAGE_ID}-colorscale-toggle", "value"),
    Input(f"{PAGE_ID}-projection-toggle", "value"),
    Input(f"{PAGE_ID}-comparison-toggle", "value"),
    Input(f"{PAGE_ID}-realm2-dropdown", "value"),
    State(f"{PAGE_ID}-output-dropdown", "value"),
    State(f"{PAGE_ID}-parameter-dropdown", "value"),
    State(f"{PAGE_ID}-model-dropdown", "value"),
    State(f"{PAGE_ID}-cycle-dropdown", "value"),
    State(f"{PAGE_ID}-realm-dropdown", "value"),
)
def update_plot(validtime, field_value, colorscale_mode, projection_mode, compare_value, realm2,
                output, parameter, model, cycle, realm):
    empty_fig = go.Figure()
    empty_fig.update_layout(
        title="Select data to display",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )

    if not all([validtime, output, parameter, model, cycle, realm]):
        return empty_fig, ""

    # Check if comparison mode is enabled
    compare_mode = "compare" in (compare_value or []) and realm2
    use_native_proj = projection_mode == "albers"

    try:
        # Load data for primary realm
        result1 = load_and_process_data(realm, cycle, model, parameter, output, validtime, field_value, use_native_proj)
        if result1 is None:
            return empty_fig, f"File not found for {realm}"

        x1, y1, values1, data_type, colorscale_type, field_label, cycle_str, lead_hour, proj4str = result1

        # Determine colorscale
        use_preset = colorscale_mode == "preset"
        if use_preset:
            colorscale, vmin, vmax = get_colorscale(colorscale_type, True)
            cbar_props = get_colorscale_properties(colorscale_type)
        else:
            colorscale = "RdBu_r"
            vmin = float(np.nanmin(values1))
            vmax = float(np.nanmax(values1))
            cbar_props = {}

        # Get coastlines
        if use_native_proj:
            coast_x, coast_y = get_coastline_scatter_data_albers(source="gfe", simplify_tolerance=0.01, proj4str=proj4str)
        else:
            coast_x, coast_y = get_coastline_scatter_data(source="gfe", simplify_tolerance=0.01)

        # Build type label
        type_label = {
            "expectedvalue": "Expected Value",
            "percentile": "Percentile",
            "threshold": "Probability",
        }.get(data_type, data_type)

        # Axis labels based on projection
        x_label = "X (km)" if use_native_proj else "Longitude"
        y_label = "Y (km)" if use_native_proj else "Latitude"
        hover_template = "X: %{x:.0f} km<br>Y: %{y:.0f} km<br>Value: %{z:.2f}<extra></extra>" if use_native_proj else "Lon: %{x:.2f}<br>Lat: %{y:.2f}<br>Value: %{z:.2f}<extra></extra>"

        if compare_mode:
            # Load data for second realm
            result2 = load_and_process_data(realm2, cycle, model, parameter, output, validtime, field_value, use_native_proj)

            if result2 is None:
                return empty_fig, f"File not found for {realm2}"

            x2, y2, values2, _, _, _, _, _, _ = result2

            # Update vmin/vmax to encompass both datasets if using auto colorscale
            if not use_preset:
                vmin = min(vmin, float(np.nanmin(values2)))
                vmax = max(vmax, float(np.nanmax(values2)))

            # Calculate difference (realm1 - realm2)
            diff_values = values1 - values2
            diff_max = max(abs(np.nanmin(diff_values)), abs(np.nanmax(diff_values)))
            if diff_max == 0:
                diff_max = 1

            # Create 2 rows: maps on top, histograms on bottom
            fig = make_subplots(
                rows=2, cols=3,
                subplot_titles=[f"{realm}", f"{realm2}", f"Difference ({realm} - {realm2})",
                               f"{realm} Histogram", f"{realm2} Histogram", "Difference Histogram"],
                row_heights=[0.7, 0.3],
                horizontal_spacing=0.08,
                vertical_spacing=0.1,
            )

            # Colorbar settings
            cbar_len = 0.4
            cbar_y = 0.82
            cbar_thickness = 12
            cbar_base = {"len": cbar_len, "y": cbar_y, "thickness": cbar_thickness, "title": cbar_props.get("cbar_label", "")}

            # Add heatmap for realm 1
            fig.add_trace(
                go.Heatmap(
                    x=x1, y=y1, z=values1,
                    colorscale=colorscale, zmin=vmin, zmax=vmax,
                    colorbar=dict(**cbar_base, x=0.30),
                    hovertemplate=hover_template,
                ),
                row=1, col=1
            )

            # Add heatmap for realm 2
            fig.add_trace(
                go.Heatmap(
                    x=x2, y=y2, z=values2,
                    colorscale=colorscale, zmin=vmin, zmax=vmax,
                    colorbar=dict(**cbar_base, x=0.66),
                    hovertemplate=hover_template,
                ),
                row=1, col=2
            )

            # Add difference heatmap
            fig.add_trace(
                go.Heatmap(
                    x=x1, y=y1, z=diff_values,
                    colorscale="RdBu_r", zmin=-diff_max, zmax=diff_max, zmid=0,
                    colorbar=dict(
                        title="Diff",
                        x=1.02, len=cbar_len, y=cbar_y, thickness=cbar_thickness,
                    ),
                    hovertemplate=hover_template,
                ),
                row=1, col=3
            )

            # Add coastlines
            for col in [1, 2, 3]:
                fig.add_trace(
                    go.Scatter(
                        x=coast_x, y=coast_y, mode="lines",
                        line=dict(color="black", width=0.5),
                        hoverinfo="skip", showlegend=False,
                    ),
                    row=1, col=col
                )

            # Add histograms
            flat_values1 = values1.flatten()
            flat_values1 = flat_values1[~np.isnan(flat_values1)]
            flat_values2 = values2.flatten()
            flat_values2 = flat_values2[~np.isnan(flat_values2)]
            flat_diff = diff_values.flatten()
            flat_diff = flat_diff[~np.isnan(flat_diff)]

            fig.add_trace(
                go.Histogram(x=flat_values1, nbinsx=50, marker_color="steelblue", showlegend=False),
                row=2, col=1
            )
            fig.add_trace(
                go.Histogram(x=flat_values2, nbinsx=50, marker_color="steelblue", showlegend=False),
                row=2, col=2
            )
            fig.add_trace(
                go.Histogram(x=flat_diff, nbinsx=50, marker_color="gray", showlegend=False),
                row=2, col=3
            )

            # Calculate common axis range from data extent
            x_min = min(x1.min(), x2.min())
            x_max = max(x1.max(), x2.max())
            y_min = min(y1.min(), y2.min())
            y_max = max(y1.max(), y2.max())

            # Update axes for map subplots
            for col in [1, 2, 3]:
                fig.update_xaxes(title_text=x_label, range=[x_min, x_max], row=1, col=col)
                fig.update_yaxes(title_text=y_label, range=[y_min, y_max], row=1, col=col)

            # Update histogram axes
            fig.update_xaxes(title_text="Value", row=2, col=1)
            fig.update_xaxes(title_text="Value", row=2, col=2)
            fig.update_xaxes(title_text="Difference", row=2, col=3)
            fig.update_yaxes(title_text="Count", row=2, col=1)
            fig.update_yaxes(title_text="Count", row=2, col=2)
            fig.update_yaxes(title_text="Count", row=2, col=3)

            title = f"{config.PARAMETERS.get(parameter, parameter)}{field_label}<br>"
            title += f"<sub>Cycle: {cycle_str} | Lead: T+{lead_hour}h | Type: {type_label}"
            if use_native_proj:
                title += " | Albers Equal Area"
            title += "</sub>"

            fig.update_layout(
                title=dict(text=title, x=0.5),
                height=900,
                margin=dict(l=50, r=80, t=100, b=50),
            )

            info_text = f"Comparing: {realm} vs {realm2} | {cycle} | {model} | {parameter} | {output} | {validtime}"

        else:
            # Single plot mode with histogram to the right
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=[f"{config.PARAMETERS.get(parameter, parameter)}{field_label}", "Distribution"],
                column_widths=[0.75, 0.25],
                horizontal_spacing=0.08,
            )

            # Build colorbar settings
            single_cbar = {"title": cbar_props.get("cbar_label", ""), "len": 0.9, "y": 0.5, "thickness": 18}

            fig.add_trace(
                go.Heatmap(
                    x=x1, y=y1, z=values1,
                    colorscale=colorscale, zmin=vmin, zmax=vmax,
                    colorbar=single_cbar,
                    hovertemplate=hover_template,
                ),
                row=1, col=1
            )

            # Add coastlines
            fig.add_trace(
                go.Scatter(
                    x=coast_x, y=coast_y, mode="lines",
                    line=dict(color="black", width=0.5),
                    hoverinfo="skip", showlegend=False,
                ),
                row=1, col=1
            )

            # Add histogram (vertical)
            flat_values = values1.flatten()
            flat_values = flat_values[~np.isnan(flat_values)]
            fig.add_trace(
                go.Histogram(y=flat_values, nbinsy=50, marker_color="steelblue", showlegend=False),
                row=1, col=2
            )

            title = f"{config.PARAMETERS.get(parameter, parameter)}{field_label}<br>"
            title += f"<sub>Cycle: {cycle_str} | Lead: T+{lead_hour}h | Type: {type_label}"
            if use_native_proj:
                title += " | Albers Equal Area"
            title += "</sub>"

            # Use actual data extent for axis range
            x_range = [float(x1.min()), float(x1.max())]
            y_range = [float(y1.min()), float(y1.max())]

            fig.update_layout(
                title=dict(text=title, x=0.5),
                margin=dict(l=50, r=50, t=80, b=50),
                height=650,
            )

            fig.update_xaxes(title_text=x_label, range=x_range, row=1, col=1)
            fig.update_yaxes(title_text=y_label, range=y_range, scaleanchor="x", row=1, col=1)
            fig.update_xaxes(title_text="Count", row=1, col=2)
            fig.update_yaxes(title_text="Value", row=1, col=2)

            info_text = f"{realm} | {cycle} | {model} | {parameter} | {output} | {validtime}"

        if field_label:
            info_text += f" | {field_label.strip(' - ')}"

        return fig, html.P(info_text, style={"color": "gray", "fontSize": "12px"})

    except Exception as e:
        import traceback
        error_fig = go.Figure()
        error_fig.update_layout(title=f"Error: {str(e)}", xaxis=dict(visible=False), yaxis=dict(visible=False))
        return error_fig, f"Error: {str(e)}\n{traceback.format_exc()}"


