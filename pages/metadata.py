"""Metadata page for viewing NetCDF file attributes and structure."""

import dash
from dash import dcc, html, callback, Input, Output, State, dash_table
import pandas as pd

import config
from utils.file_scanner import FileScanner
from utils.data_loader import DataLoader
from components.dropdowns import (
    DropdownBuilder,
    create_dropdown_options,
    create_standard_dropdowns,
)

# Register page
dash.register_page(__name__, path="/metadata", name="Metadata")

PAGE_ID = "meta"

# Layout
layout = html.Div(
    [
        html.H2("File Metadata"),
        html.Hr(),
        # Controls
        create_standard_dropdowns(PAGE_ID),
        # Refresh button
        html.Button("Refresh Data", id=f"{PAGE_ID}-refresh-btn", n_clicks=0),
        html.Hr(),
        # Loading wrapper
        dcc.Loading(
            id=f"{PAGE_ID}-loading",
            type="default",
            children=[
                # File info section
                html.Div(id=f"{PAGE_ID}-file-info"),
                html.Hr(),
                # Global attributes table
                html.H4("Global Attributes"),
                html.Div(id=f"{PAGE_ID}-global-attrs"),
                html.Hr(),
                # Variable attributes table
                html.H4("Variable Attributes"),
                html.Div(id=f"{PAGE_ID}-var-attrs"),
                html.Hr(),
                # Forecast info
                html.H4("Forecast Information"),
                html.Div(id=f"{PAGE_ID}-forecast-info"),
                html.Hr(),
                # File listing
                html.H4("Available Files"),
                html.Div(id=f"{PAGE_ID}-file-list"),
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
    options = builder.get_parameter_options(cycle, step)
    return create_dropdown_options(options)


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
    return create_dropdown_options(options)


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


# Main metadata display callback
@callback(
    Output(f"{PAGE_ID}-file-info", "children"),
    Output(f"{PAGE_ID}-global-attrs", "children"),
    Output(f"{PAGE_ID}-var-attrs", "children"),
    Output(f"{PAGE_ID}-forecast-info", "children"),
    Output(f"{PAGE_ID}-file-list", "children"),
    Input(f"{PAGE_ID}-validtime-dropdown", "value"),
    State(f"{PAGE_ID}-output-dropdown", "value"),
    State(f"{PAGE_ID}-parameter-dropdown", "value"),
    State(f"{PAGE_ID}-step-dropdown", "value"),
    State(f"{PAGE_ID}-cycle-dropdown", "value"),
    State(f"{PAGE_ID}-realm-dropdown", "value"),
)
def update_metadata(validtime, output, parameter, step, cycle, realm):
    empty = html.P("Select data to view metadata")

    if not all([validtime, output, parameter, step, cycle, realm]):
        return empty, empty, empty, empty, empty

    scanner = FileScanner(realm)
    file_path = scanner.get_file_for_validtime(cycle, step, parameter, output, validtime)

    if not file_path:
        return (
            html.P(f"File not found for {validtime}", style={"color": "red"}),
            empty,
            empty,
            empty,
            empty,
        )

    try:
        loader = DataLoader(file_path, parameter, validtime)

        # File info
        file_info = html.Div([
            html.P(f"File: {file_path}"),
            html.P(f"Data shape: {loader.data.shape}"),
            html.P(f"Data type: {loader.data_type}"),
        ])

        # Global attributes table
        global_attrs = loader.get_global_attrs()
        if global_attrs:
            global_df = pd.DataFrame([
                {"Attribute": k, "Value": str(v)[:200]}  # Truncate long values
                for k, v in global_attrs.items()
            ])
            global_table = dash_table.DataTable(
                data=global_df.to_dict("records"),
                columns=[{"name": c, "id": c} for c in global_df.columns],
                style_cell={"textAlign": "left", "padding": "5px"},
                style_header={"fontWeight": "bold"},
                style_data={"whiteSpace": "normal", "height": "auto"},
                page_size=10,
            )
        else:
            global_table = html.P("No global attributes found")

        # Variable attributes table
        var_attrs = loader.get_variable_attrs()
        if var_attrs:
            var_df = pd.DataFrame([
                {"Attribute": k, "Value": str(v)[:200]}
                for k, v in var_attrs.items()
            ])
            var_table = dash_table.DataTable(
                data=var_df.to_dict("records"),
                columns=[{"name": c, "id": c} for c in var_df.columns],
                style_cell={"textAlign": "left", "padding": "5px"},
                style_header={"fontWeight": "bold"},
                style_data={"whiteSpace": "normal", "height": "auto"},
            )
        else:
            var_table = html.P("No variable attributes found")

        # Forecast info
        info = loader.info()
        info_df = pd.DataFrame([
            {"Property": k, "Value": str(v)}
            for k, v in info.items()
        ])
        info_table = dash_table.DataTable(
            data=info_df.to_dict("records"),
            columns=[{"name": c, "id": c} for c in info_df.columns],
            style_cell={"textAlign": "left", "padding": "5px"},
            style_header={"fontWeight": "bold"},
        )

        # File listing
        all_files = scanner.get_files(cycle, step, parameter, output)
        if all_files:
            files_df = pd.DataFrame([
                {
                    "Filename": f["filename"],
                    "Valid Time": f.get("validtime", "N/A"),
                    "Lead Hour": f.get("leadhour", "N/A"),
                }
                for f in all_files
            ])
            files_table = dash_table.DataTable(
                data=files_df.to_dict("records"),
                columns=[{"name": c, "id": c} for c in files_df.columns],
                style_cell={"textAlign": "left", "padding": "5px"},
                style_header={"fontWeight": "bold"},
                page_size=10,
            )
        else:
            files_table = html.P("No files found")

        loader.close()

        return file_info, global_table, var_table, info_table, files_table

    except Exception as e:
        error_msg = html.P(f"Error loading file: {str(e)}", style={"color": "red"})
        return error_msg, empty, empty, empty, empty
