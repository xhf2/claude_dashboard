"""Home page showing data availability overview."""

import dash
from dash import dcc, html, callback, Input, Output, State, dash_table
from datetime import datetime
import pandas as pd

import config
from utils.file_scanner import FileScanner
from components.dropdowns import DropdownBuilder, create_dropdown_options

dash.register_page(__name__, path="/", name="Home")
PAGE_ID = "home"

layout = html.Div([
    html.H2("IMPROVER Forecast Dashboard"),
    html.Hr(),
    # Realm selector
    html.Div([
        html.Div([
            html.Label("Realm"),
            dcc.Dropdown(
                id=f"{PAGE_ID}-realm-dropdown",
                options=[],
                value=None,
                clearable=False,
                style={"width": "200px"},
            ),
        ], style={"display": "inline-block", "marginRight": "20px"}),
        html.Button("Refresh", id=f"{PAGE_ID}-refresh-btn", n_clicks=0),
    ], style={"marginBottom": "20px"}),
    html.Hr(),
    dcc.Loading([
        # Summary section
        html.Div(id=f"{PAGE_ID}-summary"),
        html.Hr(),
        # Most recent cycle section
        html.H4("Most Recent Cycle"),
        html.Div(id=f"{PAGE_ID}-latest-cycle-info"),
        html.Div(id=f"{PAGE_ID}-latest-structure"),
        html.Hr(),
        # Other cycles section
        html.H4("Browse Other Cycles"),
        html.Div([
            html.Label("Select Cycle: "),
            dcc.Dropdown(
                id=f"{PAGE_ID}-cycle-dropdown",
                options=[],
                value=None,
                clearable=True,
                placeholder="Select a cycle to view...",
                style={"width": "300px"},
            ),
        ], style={"marginBottom": "20px"}),
        html.Div(id=f"{PAGE_ID}-selected-cycle-info"),
        html.Div(id=f"{PAGE_ID}-selected-structure"),
    ]),
])


@callback(
    Output(f"{PAGE_ID}-realm-dropdown", "options"),
    Output(f"{PAGE_ID}-realm-dropdown", "value"),
    Input(f"{PAGE_ID}-refresh-btn", "n_clicks"),
)
def update_realm_options(n_clicks):
    builder = DropdownBuilder()
    options = builder.get_realm_options()
    default = "test_realm"
    return create_dropdown_options(options, default)


@callback(
    Output(f"{PAGE_ID}-summary", "children"),
    Output(f"{PAGE_ID}-cycle-dropdown", "options"),
    Input(f"{PAGE_ID}-realm-dropdown", "value"),
    Input(f"{PAGE_ID}-refresh-btn", "n_clicks"),
)
def update_summary(realm, n_clicks):
    if not realm:
        return html.P("Select a realm to view data"), []

    scanner = FileScanner(realm)
    cycles = scanner.get_cycles()

    if not cycles:
        return html.P(f"No cycles found in realm: {realm}"), []

    # Summary section
    summary = html.Div([
        html.P([html.Strong("Realm: "), realm]),
        html.P([html.Strong("Realm Path: "), config.REALM_PATHS.get(realm, "N/A")]),
        html.P([html.Strong("Total Cycles: "), str(len(cycles))]),
    ])

    # Cycle options for dropdown (exclude the most recent one)
    cycle_options = [{"label": c, "value": c} for c in cycles[1:]] if len(cycles) > 1 else []

    return summary, cycle_options


def build_structure_table(scanner, cycle):
    """Build a table showing the directory structure with timestamps."""
    rows = []

    models = scanner.get_models(cycle)
    for model in models:
        parameters = scanner.get_parameters(cycle, model)
        for param in parameters:
            param_display = config.PARAMETERS.get(param, param)
            outputs = scanner.get_outputs(cycle, model, param)
            for output in outputs:
                files = scanner.get_files(cycle, model, param, output)
                file_count = len(files)

                # Get modification times
                mtimes = [f.get("mtime") for f in files if f.get("mtime")]
                if mtimes:
                    latest_mtime = max(mtimes)
                    oldest_mtime = min(mtimes)
                else:
                    latest_mtime = "N/A"
                    oldest_mtime = "N/A"

                # Get valid time range
                validtimes = [f.get("validtime", "") for f in files if f.get("validtime")]
                if validtimes:
                    vt_range = f"{validtimes[0]} - {validtimes[-1]}"
                else:
                    vt_range = "N/A"

                rows.append({
                    "Model": model,
                    "Parameter": param,
                    "Parameter Name": param_display,
                    "Output Type": output,
                    "Files": file_count,
                    "Valid Time Range": vt_range,
                    "Oldest Modified": oldest_mtime,
                    "Latest Modified": latest_mtime,
                })

    if not rows:
        return html.P("No data found for this cycle")

    df = pd.DataFrame(rows)

    return dash_table.DataTable(
        data=df.to_dict("records"),
        columns=[{"name": c, "id": c} for c in df.columns],
        style_cell={
            "textAlign": "left",
            "padding": "8px",
            "fontSize": "12px",
            "whiteSpace": "normal",
            "height": "auto",
        },
        style_header={
            "fontWeight": "bold",
            "backgroundColor": "#f8f9fa",
            "borderBottom": "2px solid #dee2e6",
        },
        style_data_conditional=[
            {
                "if": {"row_index": "odd"},
                "backgroundColor": "#f8f9fa",
            }
        ],
        style_table={
            "overflowX": "auto",
        },
        page_size=20,
        sort_action="native",
        filter_action="native",
    )


@callback(
    Output(f"{PAGE_ID}-latest-cycle-info", "children"),
    Output(f"{PAGE_ID}-latest-structure", "children"),
    Input(f"{PAGE_ID}-realm-dropdown", "value"),
    Input(f"{PAGE_ID}-refresh-btn", "n_clicks"),
)
def update_latest_structure(realm, n_clicks):
    if not realm:
        return "", ""

    scanner = FileScanner(realm)
    cycles = scanner.get_cycles()

    if not cycles:
        return "", ""

    latest_cycle = cycles[0]

    info = html.Div([
        html.P([html.Strong("Cycle: "), latest_cycle], style={"marginBottom": "10px"}),
    ])

    table = build_structure_table(scanner, latest_cycle)

    return info, table


@callback(
    Output(f"{PAGE_ID}-selected-cycle-info", "children"),
    Output(f"{PAGE_ID}-selected-structure", "children"),
    Input(f"{PAGE_ID}-cycle-dropdown", "value"),
    State(f"{PAGE_ID}-realm-dropdown", "value"),
)
def update_selected_structure(cycle, realm):
    if not cycle or not realm:
        return "", ""

    scanner = FileScanner(realm)

    info = html.Div([
        html.P([html.Strong("Cycle: "), cycle], style={"marginBottom": "10px"}),
    ])

    table = build_structure_table(scanner, cycle)

    return info, table
