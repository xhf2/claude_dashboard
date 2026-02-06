"""Verification page for computing and displaying verification metrics."""

import dash
from dash import dcc, html, callback, Input, Output, State, dash_table
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

import config
from utils.file_scanner import FileScanner
from utils.data_loader import DataLoader
from utils.regrid import regrid_to_latlon
from utils.verification import VerificationMetrics, calculate_all_metrics, SCORES_AVAILABLE
from components.dropdowns import DropdownBuilder, create_dropdown_options

# Register page
dash.register_page(__name__, path="/verification", name="Verification")

PAGE_ID = "verif"

# Output types that contain probability data
PROBABILITY_OUTPUTS = ["blendgrids", "recfilter", "probabilities_extract", "level3", "merge_probabilities"]

# Reliability diagram thresholds for precipitation (in mm)
RELIABILITY_THRESHOLDS_MM = [0.2, 10, 25, 50]

# Layout
layout = html.Div(
    [
        html.H2("Verification Metrics"),
        html.Hr(),
        # Status message for scores package
        html.Div(
            id=f"{PAGE_ID}-scores-status",
            children=[
                html.P(
                    "scores package available" if SCORES_AVAILABLE
                    else "Warning: scores package not installed. Install with: pip install scores",
                    style={"color": "green" if SCORES_AVAILABLE else "orange"},
                )
            ],
        ),
        # Controls for forecast data
        html.H4("Forecast Data"),
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
            style={"marginBottom": "20px"},
        ),
        # Observation data placeholder
        html.H4("Observation Data"),
        html.Div(
            [
                html.P(
                    "Observation data format TBD. Currently using synthetic observations for demonstration.",
                    style={"color": "gray", "fontStyle": "italic"},
                ),
                dcc.Checklist(
                    id=f"{PAGE_ID}-synthetic-obs",
                    options=[{"label": "Use synthetic observations (for demo)", "value": "synthetic"}],
                    value=["synthetic"],
                ),
            ],
            style={"marginBottom": "20px"},
        ),
        # Threshold input for categorical metrics
        html.Div(
            [
                html.Label("Threshold for categorical metrics: "),
                dcc.Input(
                    id=f"{PAGE_ID}-threshold-input",
                    type="number",
                    value=1.0,
                    step=0.1,
                    style={"width": "100px", "marginRight": "20px"},
                ),
                html.Span("(mm for precip, °C for temp, knots for wind)", style={"color": "gray"}),
            ],
            style={"marginBottom": "20px"},
        ),
        # Calculate button
        html.Button("Calculate Metrics", id=f"{PAGE_ID}-calc-btn", n_clicks=0),
        html.Button("Refresh Data", id=f"{PAGE_ID}-refresh-btn", n_clicks=0, style={"marginLeft": "10px"}),
        html.Hr(),
        # Results section
        dcc.Loading(
            id=f"{PAGE_ID}-loading",
            type="default",
            children=[
                # Data info
                html.Div(id=f"{PAGE_ID}-data-info", style={"marginBottom": "20px"}),
                # Continuous metrics
                html.H4("Continuous Metrics"),
                html.Div(id=f"{PAGE_ID}-continuous-metrics"),
                html.Hr(),
                # Probabilistic metrics (CRPS) - shown for probability outputs
                html.Div(
                    id=f"{PAGE_ID}-probabilistic-section",
                    children=[
                        html.H4("Probabilistic Metrics"),
                        html.Div(id=f"{PAGE_ID}-probabilistic-metrics"),
                        html.Hr(),
                    ],
                ),
                # Categorical metrics
                html.H4("Categorical Metrics"),
                html.Div(id=f"{PAGE_ID}-categorical-metrics"),
                html.Hr(),
                # Spatial metrics (FSS)
                html.H4("Spatial Metrics (FSS)"),
                html.Div(id=f"{PAGE_ID}-spatial-metrics"),
                html.Hr(),
                # Reliability diagrams
                html.H4("Reliability Diagrams"),
                html.Div(id=f"{PAGE_ID}-reliability-section"),
                dcc.Graph(id=f"{PAGE_ID}-reliability-plot", style={"height": "500px"}),
                html.Hr(),
                # ROC curve
                html.H4("ROC Curve"),
                dcc.Graph(id=f"{PAGE_ID}-roc-plot", style={"height": "400px"}),
            ],
        ),
    ]
)


def calculate_reliability_diagram(forecast_prob, obs_binary, n_bins=10):
    """Calculate reliability diagram data with 90% confidence intervals.

    Args:
        forecast_prob: Forecast probabilities (0-1 or 0-100)
        obs_binary: Binary observations (0 or 1)
        n_bins: Number of probability bins

    Returns:
        Dict with forecast_prob, observed_freq, sample_count, ci_lower, ci_upper
    """
    # Ensure probability is 0-1
    if np.nanmax(forecast_prob) > 1:
        forecast_prob = forecast_prob / 100.0

    # Flatten arrays
    prob_flat = forecast_prob.flatten()
    obs_flat = obs_binary.flatten()

    # Remove NaNs
    mask = ~(np.isnan(prob_flat) | np.isnan(obs_flat))
    prob_clean = prob_flat[mask]
    obs_clean = obs_flat[mask]

    if len(prob_clean) == 0:
        return {"forecast_prob": [], "observed_freq": [], "sample_count": [], "ci_lower": [], "ci_upper": []}

    # Create bins
    bin_edges = np.linspace(0, 1, n_bins + 1)

    forecast_probs = []
    observed_freqs = []
    sample_counts = []
    ci_lowers = []
    ci_uppers = []

    for i in range(n_bins):
        in_bin = (prob_clean >= bin_edges[i]) & (prob_clean < bin_edges[i + 1])
        if i == n_bins - 1:  # Include upper edge in last bin
            in_bin = (prob_clean >= bin_edges[i]) & (prob_clean <= bin_edges[i + 1])

        n_samples = np.sum(in_bin)
        if n_samples > 0:
            mean_prob = np.mean(prob_clean[in_bin])
            obs_freq = np.mean(obs_clean[in_bin])

            # 90% confidence interval using Wilson score interval
            # More accurate for proportions, especially near 0 or 1
            z = 1.645  # 90% CI
            n = n_samples
            p = obs_freq

            denominator = 1 + z**2 / n
            center = (p + z**2 / (2 * n)) / denominator
            spread = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denominator

            ci_lower = max(0, center - spread)
            ci_upper = min(1, center + spread)

            forecast_probs.append(mean_prob)
            observed_freqs.append(obs_freq)
            sample_counts.append(int(n_samples))
            ci_lowers.append(ci_lower)
            ci_uppers.append(ci_upper)

    return {
        "forecast_prob": forecast_probs,
        "observed_freq": observed_freqs,
        "sample_count": sample_counts,
        "ci_lower": ci_lowers,
        "ci_upper": ci_uppers,
    }


def create_reliability_plot(reliability_data_dict, thresholds, title="Reliability Diagrams"):
    """Create a multi-panel reliability diagram with 90% confidence intervals.

    Args:
        reliability_data_dict: Dict mapping threshold to reliability data
        thresholds: List of threshold values
        title: Plot title

    Returns:
        Plotly figure
    """
    n_thresholds = len(thresholds)
    cols = min(2, n_thresholds)
    rows = (n_thresholds + cols - 1) // cols

    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=[f"Threshold: {t}mm" for t in thresholds],
        horizontal_spacing=0.12,
        vertical_spacing=0.15,
    )

    for idx, threshold in enumerate(thresholds):
        row = idx // cols + 1
        col = idx % cols + 1

        data = reliability_data_dict.get(threshold, {})
        forecast_prob = data.get("forecast_prob", [])
        observed_freq = data.get("observed_freq", [])
        sample_count = data.get("sample_count", [])
        ci_lower = data.get("ci_lower", [])
        ci_upper = data.get("ci_upper", [])

        if forecast_prob:
            # 90% Confidence interval shading
            if ci_lower and ci_upper:
                fig.add_trace(
                    go.Scatter(
                        x=forecast_prob + forecast_prob[::-1],
                        y=ci_upper + ci_lower[::-1],
                        fill="toself",
                        fillcolor="rgba(100, 149, 237, 0.3)",
                        line=dict(color="rgba(255,255,255,0)"),
                        hoverinfo="skip",
                        showlegend=False,
                        name="90% CI",
                    ),
                    row=row, col=col
                )

            # Reliability curve
            fig.add_trace(
                go.Scatter(
                    x=forecast_prob,
                    y=observed_freq,
                    mode="lines+markers",
                    name=f"{threshold}mm",
                    marker=dict(size=8, color="blue"),
                    line=dict(color="blue"),
                    hovertemplate="Forecast: %{x:.2f}<br>Observed: %{y:.2f}<br>Samples: %{text}<extra></extra>",
                    text=sample_count,
                    showlegend=False,
                ),
                row=row, col=col
            )

        # Perfect reliability line
        fig.add_trace(
            go.Scatter(
                x=[0, 1], y=[0, 1],
                mode="lines",
                line=dict(dash="dash", color="gray", width=1),
                showlegend=False,
            ),
            row=row, col=col
        )

        fig.update_xaxes(title_text="Forecast Probability", range=[0, 1], row=row, col=col)
        fig.update_yaxes(title_text="Observed Frequency", range=[0, 1], row=row, col=col)

    fig.update_layout(
        title=title,
        height=300 * rows + 50,
    )

    return fig


def generate_synthetic_precip(shape, seed=42):
    """Generate synthetic precipitation data with realistic distribution.

    Most values near 0, exponentially decreasing frequency toward higher values,
    with occasional extreme events >100mm.

    Args:
        shape: Shape of output array
        seed: Random seed

    Returns:
        Numpy array with synthetic precipitation values in mm
    """
    np.random.seed(seed)

    # Start with zeros (dry areas - about 40% of domain)
    precip = np.zeros(shape)

    # Determine which cells have precipitation
    has_precip = np.random.random(shape) > 0.4

    # For cells with precip, use mixture of distributions:
    # - Most are light (exponential with small scale)
    # - Some are moderate
    # - Few are heavy

    n_precip = np.sum(has_precip)

    if n_precip > 0:
        # Light precip (0.2-5mm): 70% of rainy cells
        # Moderate precip (5-25mm): 25% of rainy cells
        # Heavy precip (25-200mm): 5% of rainy cells

        precip_values = np.zeros(n_precip)

        # Assign categories
        rand_cat = np.random.random(n_precip)

        light_mask = rand_cat < 0.70
        moderate_mask = (rand_cat >= 0.70) & (rand_cat < 0.95)
        heavy_mask = rand_cat >= 0.95

        # Light: exponential with scale=2, shifted by 0.2
        precip_values[light_mask] = 0.2 + np.random.exponential(scale=2.0, size=np.sum(light_mask))

        # Moderate: exponential with scale=8, shifted by 5
        precip_values[moderate_mask] = 5 + np.random.exponential(scale=8.0, size=np.sum(moderate_mask))

        # Heavy: exponential with scale=40, shifted by 25
        precip_values[heavy_mask] = 25 + np.random.exponential(scale=40.0, size=np.sum(heavy_mask))

        # Cap at 200mm
        precip_values = np.minimum(precip_values, 200)

        precip[has_precip] = precip_values

    return precip


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
    options = builder.get_parameter_options(cycle, model)
    return create_dropdown_options(options)


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


# Main calculation callback
@callback(
    Output(f"{PAGE_ID}-data-info", "children"),
    Output(f"{PAGE_ID}-continuous-metrics", "children"),
    Output(f"{PAGE_ID}-probabilistic-section", "style"),
    Output(f"{PAGE_ID}-probabilistic-metrics", "children"),
    Output(f"{PAGE_ID}-categorical-metrics", "children"),
    Output(f"{PAGE_ID}-spatial-metrics", "children"),
    Output(f"{PAGE_ID}-reliability-section", "children"),
    Output(f"{PAGE_ID}-reliability-plot", "figure"),
    Output(f"{PAGE_ID}-roc-plot", "figure"),
    Input(f"{PAGE_ID}-calc-btn", "n_clicks"),
    State(f"{PAGE_ID}-validtime-dropdown", "value"),
    State(f"{PAGE_ID}-output-dropdown", "value"),
    State(f"{PAGE_ID}-parameter-dropdown", "value"),
    State(f"{PAGE_ID}-model-dropdown", "value"),
    State(f"{PAGE_ID}-cycle-dropdown", "value"),
    State(f"{PAGE_ID}-realm-dropdown", "value"),
    State(f"{PAGE_ID}-threshold-input", "value"),
    State(f"{PAGE_ID}-synthetic-obs", "value"),
)
def calculate_metrics(n_clicks, validtime, output, parameter, model, cycle, realm, threshold, synthetic):
    empty_fig = go.Figure()
    empty_fig.update_layout(title="No data", xaxis=dict(visible=False), yaxis=dict(visible=False))
    empty_msg = html.P("Click 'Calculate Metrics' to compute")
    hidden_style = {"display": "none"}
    visible_style = {"display": "block"}

    if n_clicks == 0:
        return "", empty_msg, hidden_style, empty_msg, empty_msg, empty_msg, "", empty_fig, empty_fig

    if not all([validtime, output, parameter, model, cycle, realm]):
        return (
            html.P("Please select all data options", style={"color": "orange"}),
            empty_msg, hidden_style, empty_msg, empty_msg, empty_msg, "", empty_fig, empty_fig,
        )

    scanner = FileScanner(realm)
    file_path = scanner.get_file_for_validtime(cycle, model, parameter, output, validtime)

    if not file_path:
        return (
            html.P("File not found", style={"color": "red"}),
            empty_msg, hidden_style, empty_msg, empty_msg, empty_msg, "", empty_fig, empty_fig,
        )

    is_probability_output = output in PROBABILITY_OUTPUTS

    try:
        # Load forecast
        loader = DataLoader(file_path, parameter, validtime)
        data_type = loader.data_type

        # Data info
        info_text = f"Data type: {data_type} | File: {file_path}"
        data_info = html.P(info_text, style={"color": "gray", "fontSize": "12px"})

        # Initialize variables
        verifier = VerificationMetrics()
        prob_style = visible_style if is_probability_output else hidden_style
        prob_metrics_content = empty_msg
        reliability_content = ""
        reliability_fig = empty_fig
        forecast = None
        observation = None

        # For probability data (threshold dimension), handle differently
        if is_probability_output and data_type == "threshold":
            thresholds_in_file = loader.get_thresholds()
            if thresholds_in_file is not None:
                thresholds_mm = thresholds_in_file * 1000  # Convert m to mm
                info_text += f" | Available thresholds: {[f'{t:.1f}mm' for t in thresholds_mm]}"
                data_info = html.P(info_text, style={"color": "gray", "fontSize": "12px"})

                # For probability data, we need synthetic observations based on a reference
                # Use the first threshold's probability to create a reference grid
                first_thresh_data = loader.select_threshold(thresholds_in_file[0])

                class TempLoader:
                    pass
                temp_loader = TempLoader()
                temp_loader.proj4str = loader.proj4str
                temp_loader.data = first_thresh_data

                ref_regridded = regrid_to_latlon(temp_loader)

                # Create synthetic precipitation observations (for demo)
                if "synthetic" in (synthetic or []):
                    obs_values = generate_synthetic_precip(ref_regridded.shape, seed=42)
                    observation = ref_regridded.copy()
                    observation.values = obs_values
                else:
                    # No real obs - use zeros
                    observation = ref_regridded.copy()
                    observation.values = np.zeros_like(ref_regridded.values)

                # Calculate probabilistic metrics for each threshold
                prob_metrics_rows = []
                reliability_data = {}

                for thresh_m in thresholds_in_file:
                    thresh_mm = thresh_m * 1000

                    # Select probability data for this threshold
                    prob_data = loader.select_threshold(thresh_m)

                    temp_loader = TempLoader()
                    temp_loader.proj4str = loader.proj4str
                    temp_loader.data = prob_data

                    prob_regridded = regrid_to_latlon(temp_loader)
                    prob_values = prob_regridded.values

                    # Ensure probability is 0-1
                    if np.nanmax(prob_values) > 1:
                        prob_values = prob_values / 100.0

                    # Binary observations at this threshold (obs >= threshold)
                    obs_binary = (observation.values >= thresh_mm).astype(float)

                    # Brier score for this threshold
                    valid_mask = ~(np.isnan(prob_values) | np.isnan(obs_binary))
                    if np.sum(valid_mask) > 0:
                        brier = np.mean((prob_values[valid_mask] - obs_binary[valid_mask]) ** 2)
                        prob_metrics_rows.append({
                            "Threshold": f"{thresh_mm:.1f}mm",
                            "Brier Score": f"{brier:.4f}",
                        })

                    # Reliability diagram for selected thresholds
                    if any(abs(thresh_mm - t) < 1.0 for t in RELIABILITY_THRESHOLDS_MM):
                        closest_thresh = min(RELIABILITY_THRESHOLDS_MM, key=lambda x: abs(x - thresh_mm))
                        if closest_thresh not in reliability_data:
                            rel_data = calculate_reliability_diagram(prob_values, obs_binary)
                            reliability_data[closest_thresh] = rel_data

                # Calculate overall CRPS approximation (mean of Brier scores)
                if prob_metrics_rows:
                    brier_values = [float(row["Brier Score"]) for row in prob_metrics_rows]
                    crps_approx = np.mean(brier_values)

                    # Threshold-weighted CRPS for >50mm
                    high_thresh_briers = [float(row["Brier Score"]) for row in prob_metrics_rows
                                          if float(row["Threshold"].replace("mm", "")) >= 50]
                    tw_crps_50 = np.mean(high_thresh_briers) if high_thresh_briers else np.nan

                    prob_df = pd.DataFrame([
                        {"Metric": "CRPS (approx)", "Value": f"{crps_approx:.4f}"},
                        {"Metric": "TW-CRPS (>50mm)", "Value": f"{tw_crps_50:.4f}" if not np.isnan(tw_crps_50) else "N/A"},
                    ])

                    prob_metrics_content = html.Div([
                        dash_table.DataTable(
                            data=prob_df.to_dict("records"),
                            columns=[{"name": c, "id": c} for c in prob_df.columns],
                            style_cell={"textAlign": "left", "padding": "10px"},
                            style_header={"fontWeight": "bold"},
                        ),
                        html.Br(),
                        html.H5("Brier Scores by Threshold"),
                        dash_table.DataTable(
                            data=prob_metrics_rows,
                            columns=[{"name": "Threshold", "id": "Threshold"},
                                     {"name": "Brier Score", "id": "Brier Score"}],
                            style_cell={"textAlign": "left", "padding": "10px"},
                            style_header={"fontWeight": "bold"},
                            page_size=10,
                        ),
                    ])

                # Create reliability diagrams
                if reliability_data:
                    available_thresholds = sorted(reliability_data.keys())
                    reliability_content = html.P(
                        f"Reliability diagrams at thresholds: {', '.join([f'{t}mm' for t in available_thresholds])}",
                        style={"color": "gray"}
                    )
                    reliability_fig = create_reliability_plot(reliability_data, available_thresholds)

                # For probability output, continuous metrics don't apply directly
                cont_table = html.P(
                    "Continuous metrics (RMSE, MAE, Bias) not applicable for probability data. "
                    "See Probabilistic Metrics below.",
                    style={"color": "gray", "fontStyle": "italic"}
                )

        else:
            # Non-probability data (expected values, percentiles)
            # Get regridded data
            regridded = regrid_to_latlon(loader)
            forecast = regridded

            # Create synthetic observations (for demo)
            if "synthetic" in (synthetic or []):
                if "precip" in parameter.lower():
                    # Use realistic precipitation distribution
                    obs_values = generate_synthetic_precip(forecast.shape, seed=42)
                else:
                    # For non-precip, add noise to forecast
                    np.random.seed(42)
                    noise = np.random.normal(0, 0.1 * np.nanstd(forecast.values), forecast.shape)
                    obs_values = forecast.values + noise
                observation = forecast.copy()
                observation.values = obs_values
            else:
                observation = forecast.copy()

            # Continuous metrics (RMSE, MAE, Bias)
            cont_metrics = verifier.continuous_metrics(forecast, observation)
            cont_df = pd.DataFrame([
                {"Metric": "RMSE", "Value": f"{cont_metrics['rmse']:.4f}"},
                {"Metric": "MAE", "Value": f"{cont_metrics['mae']:.4f}"},
                {"Metric": "Bias", "Value": f"{cont_metrics['bias']:.4f}"},
            ])
            cont_table = dash_table.DataTable(
                data=cont_df.to_dict("records"),
                columns=[{"name": c, "id": c} for c in cont_df.columns],
                style_cell={"textAlign": "left", "padding": "10px"},
                style_header={"fontWeight": "bold"},
            )

            # Reliability diagrams for expected values at multiple thresholds
            # For expected values, we convert forecast values to "pseudo-probabilities"
            # by normalizing based on the max expected value
            reliability_data = {}
            fcst_max = np.nanmax(forecast.values)

            for thresh_mm in RELIABILITY_THRESHOLDS_MM:
                # Normalize forecast to 0-1 based on threshold
                # P(exceed threshold) approximated by clipping forecast/threshold
                if fcst_max > 0:
                    # Create pseudo-probability: how much of threshold is forecast to occur
                    # Clip at 1.0 for values exceeding threshold
                    pseudo_prob = np.minimum(forecast.values / (thresh_mm * 2), 1.0)
                    pseudo_prob = np.maximum(pseudo_prob, 0.0)
                else:
                    pseudo_prob = np.zeros_like(forecast.values)

                # Binary observation at this threshold
                obs_binary = (observation.values >= thresh_mm).astype(float)

                # Calculate reliability
                rel_data = calculate_reliability_diagram(pseudo_prob, obs_binary, n_bins=10)
                reliability_data[thresh_mm] = rel_data

            # Create reliability plot for expected values
            if reliability_data:
                reliability_content = html.P(
                    f"Reliability diagrams at thresholds: {', '.join([f'{t}mm' for t in RELIABILITY_THRESHOLDS_MM])} "
                    "(forecast values normalized to pseudo-probabilities)",
                    style={"color": "gray"}
                )
                reliability_fig = create_reliability_plot(
                    reliability_data,
                    RELIABILITY_THRESHOLDS_MM,
                    title="Reliability Diagrams (Expected Values)"
                )

        # Categorical metrics and ROC curve - only for non-probability data
        if forecast is not None and observation is not None:
            # Calculate ROC curve over many thresholds (0-200)
            roc_thresholds = np.linspace(0, 200, 51)  # 51 thresholds from 0 to 200

            hit_rates = []
            false_alarm_rates = []

            fcst_flat = forecast.values.flatten()
            obs_flat = observation.values.flatten()
            valid_mask = ~(np.isnan(fcst_flat) | np.isnan(obs_flat))
            fcst_clean = fcst_flat[valid_mask]
            obs_clean = obs_flat[valid_mask]

            for thresh in roc_thresholds:
                fcst_binary = (fcst_clean >= thresh).astype(int)
                obs_binary = (obs_clean >= thresh).astype(int)

                hits = np.sum((fcst_binary == 1) & (obs_binary == 1))
                misses = np.sum((fcst_binary == 0) & (obs_binary == 1))
                false_alarms = np.sum((fcst_binary == 1) & (obs_binary == 0))
                correct_negatives = np.sum((fcst_binary == 0) & (obs_binary == 0))

                hr = hits / (hits + misses) if (hits + misses) > 0 else 0
                far = false_alarms / (false_alarms + correct_negatives) if (false_alarms + correct_negatives) > 0 else 0

                hit_rates.append(hr)
                false_alarm_rates.append(far)

            # Calculate AUC using trapezoidal rule
            # Sort by FAR for proper integration
            sorted_indices = np.argsort(false_alarm_rates)
            far_sorted = np.array(false_alarm_rates)[sorted_indices]
            hr_sorted = np.array(hit_rates)[sorted_indices]
            auc = np.trapezoid(hr_sorted, far_sorted)

            # CSI at user-specified threshold
            if threshold is not None:
                cat_metrics = verifier.categorical_metrics(forecast, observation, threshold)
                cat_df = pd.DataFrame([
                    {"Metric": "AUC (Area Under ROC)", "Value": f"{auc:.4f}"},
                    {"Metric": f"CSI at {threshold}mm", "Value": f"{cat_metrics['csi']:.4f}"},
                ])
            else:
                cat_df = pd.DataFrame([
                    {"Metric": "AUC (Area Under ROC)", "Value": f"{auc:.4f}"},
                ])

            cat_table = dash_table.DataTable(
                data=cat_df.to_dict("records"),
                columns=[{"name": c, "id": c} for c in cat_df.columns],
                style_cell={"textAlign": "left", "padding": "10px"},
                style_header={"fontWeight": "bold"},
            )

            # Spatial metrics (FSS) - only if threshold specified
            if threshold is not None:
                spatial = verifier.spatial_metrics(forecast, observation, threshold)
                fss_data = spatial.get("fss", {})
                fss_df = pd.DataFrame([
                    {"Window Size": k, "FSS": f"{v:.4f}" if not np.isnan(v) else "N/A"}
                    for k, v in fss_data.items()
                ])
                fss_table = dash_table.DataTable(
                    data=fss_df.to_dict("records"),
                    columns=[{"name": c, "id": c} for c in fss_df.columns],
                    style_cell={"textAlign": "left", "padding": "10px"},
                    style_header={"fontWeight": "bold"},
                )
            else:
                fss_table = html.P("Set threshold to calculate FSS")

            # ROC curve plot
            roc_fig = go.Figure()
            roc_fig.add_trace(
                go.Scatter(
                    x=false_alarm_rates,
                    y=hit_rates,
                    mode="lines+markers",
                    name="ROC Curve",
                    text=[f"{t:.0f}mm" for t in roc_thresholds],
                    hovertemplate="Threshold: %{text}<br>FAR: %{x:.3f}<br>POD: %{y:.3f}<extra></extra>",
                )
            )
            roc_fig.add_trace(
                go.Scatter(
                    x=[0, 1], y=[0, 1],
                    mode="lines",
                    line=dict(dash="dash", color="gray"),
                    name="No Skill",
                )
            )
            roc_fig.update_layout(
                title=f"ROC Curve (thresholds 0-200mm, AUC={auc:.3f})",
                xaxis_title="False Alarm Rate",
                yaxis_title="Hit Rate (POD)",
                xaxis=dict(range=[0, 1]),
                yaxis=dict(range=[0, 1]),
            )
        elif is_probability_output:
            cat_table = html.P(
                "Categorical metrics calculated from probability thresholds - see Brier scores above.",
                style={"color": "gray", "fontStyle": "italic"}
            )
            fss_table = html.P(
                "FSS not applicable for probability data.",
                style={"color": "gray", "fontStyle": "italic"}
            )
            roc_fig = empty_fig
        else:
            cat_table = html.P("Select data to calculate categorical metrics")
            fss_table = html.P("Set threshold to calculate FSS")
            roc_fig = empty_fig

        loader.close()

        return (data_info, cont_table, prob_style, prob_metrics_content,
                cat_table, fss_table, reliability_content, reliability_fig, roc_fig)

    except Exception as e:
        import traceback
        error_msg = html.P(f"Error: {str(e)}\n{traceback.format_exc()}", style={"color": "red", "whiteSpace": "pre-wrap"})
        return ("", error_msg, hidden_style, error_msg, error_msg, error_msg, "", empty_fig, empty_fig)


