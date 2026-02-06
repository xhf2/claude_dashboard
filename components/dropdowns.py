"""Dropdown builder utilities for the dashboard.

Creates dynamic dropdown options based on available data.
"""

from typing import Dict, List, Optional, Tuple, Any

from dash import dcc, html

import config
from utils.file_scanner import FileScanner


class DropdownBuilder:
    """Build dropdown components for the dashboard."""

    def __init__(self, realm: str = "test_realm"):
        """Initialize the dropdown builder.

        Args:
            realm: Which realm to scan for options
        """
        self.scanner = FileScanner(realm)

    def set_realm(self, realm: str):
        """Change the realm for scanning.

        Args:
            realm: Realm name
        """
        self.scanner.set_realm(realm)

    def get_realm_options(self) -> List[Dict[str, str]]:
        """Get dropdown options for realm selection.

        Returns:
            List of option dicts
        """
        options = []
        for realm_name, realm_path in config.REALM_PATHS.items():
            if realm_path:  # Only include configured realms
                options.append({
                    "label": realm_name,
                    "value": realm_name,
                })
            elif realm_name == "test_realm":
                # Always include test_realm
                options.append({
                    "label": realm_name,
                    "value": realm_name,
                })
        return options

    def get_cycle_options(self) -> List[Dict[str, str]]:
        """Get dropdown options for available cycles.

        Returns:
            List of option dicts
        """
        cycles = self.scanner.get_cycles()
        return [{"label": c, "value": c} for c in cycles]

    def get_model_options(self, cycle: str) -> List[Dict[str, str]]:
        """Get dropdown options for available models.

        Args:
            cycle: Selected cycle

        Returns:
            List of option dicts
        """
        models = self.scanner.get_models(cycle)
        return [{"label": m, "value": m} for m in models]

    def get_parameter_options(
        self, cycle: str, model: str, filter_params: Optional[List[str]] = None
    ) -> List[Dict[str, str]]:
        """Get dropdown options for available parameters.

        Args:
            cycle: Selected cycle
            model: Selected model
            filter_params: Optional list of parameters to filter to

        Returns:
            List of option dicts
        """
        params = self.scanner.get_parameters(cycle, model)

        if filter_params:
            params = [p for p in params if p in filter_params]

        options = []
        for p in params:
            label = config.PARAMETERS.get(p, p)
            options.append({"label": label, "value": p})

        return options

    def get_output_options(
        self, cycle: str, model: str, parameter: str
    ) -> List[Dict[str, str]]:
        """Get dropdown options for available output types.

        Args:
            cycle: Selected cycle
            model: Selected model
            parameter: Selected parameter

        Returns:
            List of option dicts
        """
        outputs = self.scanner.get_outputs(cycle, model, parameter)
        return [{"label": o, "value": o} for o in outputs]

    def get_validtime_options(
        self, cycle: str, model: str, parameter: str, output: str
    ) -> List[Dict[str, str]]:
        """Get dropdown options for available valid times.

        Args:
            cycle: Selected cycle
            model: Selected model
            parameter: Selected parameter
            output: Selected output type

        Returns:
            List of option dicts
        """
        validtimes = self.scanner.get_validtimes(cycle, model, parameter, output)
        options = []
        for vt, lh in validtimes:
            if "Lead" in vt:
                label = vt
            else:
                label = f"{vt} (T+{lh}h)"
            options.append({"label": label, "value": vt})

        return options

    def refresh(self):
        """Refresh the file scanner cache."""
        self.scanner.refresh()


def create_dropdown_options(
    options: List[Dict[str, str]],
    default_value: Optional[str] = None,
) -> Tuple[List[Dict], Optional[str]]:
    """Create dropdown options with a default value.

    Args:
        options: List of option dicts with "label" and "value" keys
        default_value: Optional default value to select

    Returns:
        Tuple of (options, selected_value)
    """
    if not options:
        return [], None

    # If default_value is provided and exists in options, use it
    if default_value:
        for opt in options:
            if opt["value"] == default_value:
                return options, default_value

    # Otherwise use the first option
    return options, options[0]["value"]


def create_standard_dropdowns(
    page_id: str,
    filter_params: Optional[List[str]] = None,
) -> html.Div:
    """Create the standard set of dropdowns for a page.

    Args:
        page_id: Unique ID prefix for this page's dropdowns
        filter_params: Optional list of parameters to filter to

    Returns:
        Div containing all dropdown components
    """
    return html.Div(
        [
            html.Div(
                [
                    html.Label("Realm"),
                    dcc.Dropdown(
                        id=f"{page_id}-realm-dropdown",
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
                        id=f"{page_id}-cycle-dropdown",
                        options=[],
                        value=None,
                        clearable=False,
                    ),
                ],
                style={"width": "18%", "display": "inline-block", "marginRight": "10px"},
            ),
            html.Div(
                [
                    html.Label("Model"),
                    dcc.Dropdown(
                        id=f"{page_id}-model-dropdown",
                        options=[],
                        value=None,
                        clearable=False,
                    ),
                ],
                style={"width": "12%", "display": "inline-block", "marginRight": "10px"},
            ),
            html.Div(
                [
                    html.Label("Parameter"),
                    dcc.Dropdown(
                        id=f"{page_id}-parameter-dropdown",
                        options=[],
                        value=None,
                        clearable=False,
                    ),
                ],
                style={"width": "18%", "display": "inline-block", "marginRight": "10px"},
            ),
            html.Div(
                [
                    html.Label("Output"),
                    dcc.Dropdown(
                        id=f"{page_id}-output-dropdown",
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
                        id=f"{page_id}-validtime-dropdown",
                        options=[],
                        value=None,
                        clearable=False,
                    ),
                ],
                style={"width": "18%", "display": "inline-block"},
            ),
        ],
        style={"marginBottom": "20px"},
    )


def create_colorscale_toggle(page_id: str) -> html.Div:
    """Create colorscale toggle component.

    Args:
        page_id: Unique ID prefix

    Returns:
        Div containing the toggle
    """
    return html.Div(
        [
            html.Label("Colorscale: "),
            dcc.RadioItems(
                id=f"{page_id}-colorscale-toggle",
                options=[
                    {"label": "Preset (GFE)", "value": "preset"},
                    {"label": "Auto", "value": "auto"},
                ],
                value="preset",
                inline=True,
            ),
        ],
        style={"marginBottom": "10px"},
    )


def create_comparison_toggle(page_id: str) -> html.Div:
    """Create comparison mode toggle.

    Args:
        page_id: Unique ID prefix

    Returns:
        Div containing the toggle
    """
    return html.Div(
        [
            dcc.Checklist(
                id=f"{page_id}-comparison-toggle",
                options=[{"label": "Compare Realms", "value": "compare"}],
                value=[],
                inline=True,
            ),
            html.Div(
                [
                    html.Label("Second Realm: "),
                    dcc.Dropdown(
                        id=f"{page_id}-realm2-dropdown",
                        options=[],
                        value=None,
                        clearable=False,
                        style={"width": "200px"},
                    ),
                ],
                id=f"{page_id}-realm2-container",
                style={"display": "none", "marginTop": "10px"},
            ),
        ],
        style={"marginBottom": "10px"},
    )


def create_projection_toggle(page_id: str) -> html.Div:
    """Create projection toggle component.

    Args:
        page_id: Unique ID prefix

    Returns:
        Div containing the toggle
    """
    return html.Div(
        [
            html.Label("Projection: "),
            dcc.RadioItems(
                id=f"{page_id}-projection-toggle",
                options=[
                    {"label": "Lat/Lon", "value": "latlon"},
                    {"label": "Albers Equal Area", "value": "albers"},
                ],
                value="latlon",
                inline=True,
            ),
        ],
        style={"marginBottom": "10px", "marginLeft": "20px", "display": "inline-block"},
    )
