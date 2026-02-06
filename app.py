"""Main Dash application entry point for the Claude Dashboard.

A multi-page dashboard for visualizing meteorological NetCDF forecast data.
"""

import dash
from dash import Dash, html, dcc, page_container

# Initialize the Dash app with multi-page support
app = Dash(
    __name__,
    use_pages=True,
    suppress_callback_exceptions=True,
    title="IMPROVER Forecast Dashboard",
)

# Define the navigation layout
nav_links = [
    {"name": "Home", "path": "/"},
    {"name": "Precipitation", "path": "/precipitation"},
    {"name": "3hr Precip", "path": "/precip-3hr"},
    {"name": "Temperature", "path": "/temperature"},
    {"name": "Dew Point", "path": "/dew-point"},
    {"name": "Wind Speed", "path": "/wind-speed"},
    {"name": "Wind Direction", "path": "/wind-direction"},
    {"name": "Metadata", "path": "/metadata"},
    {"name": "Verification", "path": "/verification"},
]

# Main layout with navigation
app.layout = html.Div(
    [
        # Header
        html.Div(
            [
                html.H1(
                    "IMPROVER Forecast Dashboard",
                    style={
                        "display": "inline-block",
                        "marginRight": "30px",
                    },
                ),
                # Navigation links
                html.Div(
                    [
                        dcc.Link(
                            link["name"],
                            href=link["path"],
                            className="nav-link",
                            style={
                                "marginRight": "20px",
                                "padding": "10px 15px",
                                "textDecoration": "none",
                                "color": "#2c3e50",
                                "borderRadius": "5px",
                                "backgroundColor": "#ecf0f1",
                            },
                        )
                        for link in nav_links
                    ],
                    style={"display": "inline-block"},
                ),
            ],
            style={
                "padding": "20px",
                "backgroundColor": "#f8f9fa",
                "borderBottom": "2px solid #dee2e6",
                "marginBottom": "20px",
            },
        ),
        # Page content container
        html.Div(
            page_container,
            style={"padding": "0 20px 20px 20px"},
        ),
        # Footer
        html.Div(
            [
                html.Hr(),
                html.P(
                    "IMPROVER Forecast Visualization Dashboard",
                    style={"color": "gray", "fontSize": "12px"},
                ),
            ],
            style={"padding": "20px", "textAlign": "center"},
        ),
    ]
)

# Server reference for deployment
server = app.server

if __name__ == "__main__":
    # Run the development server
    app.run(debug=True, host="0.0.0.0", port=8050)
