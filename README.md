# Weather Visualization Dashboard

A multi-page Dash application for visualizing IMPROVER weather forecast data with interactive maps, comparison modes, and data distribution histograms.

## Features

- **Precipitation** (daily/3-hourly) with GFE colorscales
- **Temperature** and **Dew Point** with pressure level support (200-950 hPa)
- **Wind Speed** with probability thresholds (PoW)
- **Comparison mode** - side-by-side realm comparison with difference plots
- **Data histograms** - distribution visualization for all parameters
- **Projection toggle** - switch between lat/lon and Albers Equal Area
- **Colorscale toggle** - GFE preset or auto-scaled

## Setup (Linux)

```bash
# Clone the repository
git clone https://github.com/xhf2/claude_dashboard.git
cd claude_dashboard

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Configuration

Edit `config.py` to set your data paths:

```python
REALM_PATHS = {
    "test_realm": "/path/to/your/data",
    "user_realm": "/path/to/comparison/data",
}
```

Data directories should contain cycle folders (e.g., `20250308T1200Z/`) with the IMPROVER output structure.

## Running the Dashboard

```bash
# Activate venv (if not already active)
source .venv/bin/activate

# Run the app
python app.py

# Open in browser
# http://localhost:8050
```

## Project Structure

```
claude_dashboard/
├── app.py                 # Main Dash application
├── config.py              # Configuration (paths, parameters, thresholds)
├── requirements.txt       # Python dependencies
├── assets/
│   └── style.css          # Dashboard styling
├── components/
│   ├── colorscale.py      # GFE colorscale handling
│   ├── dropdowns.py       # Dropdown builders
│   └── ...
├── data/
│   ├── colorscales/       # GFE colorscale definitions
│   └── shapefiles/        # Australian coastline shapefiles
├── pages/
│   ├── precipitation.py   # Daily precipitation page
│   ├── precip_3hr.py      # 3-hourly precipitation page
│   ├── temperature.py     # Temperature page
│   ├── dew_point.py       # Dew point page
│   ├── wind_speed.py      # Wind speed page
│   └── ...
└── utils/
    ├── data_loader.py     # NetCDF data loading
    ├── file_scanner.py    # Directory scanning
    ├── coastline.py       # Coastline loading/projection
    └── regrid.py          # Coordinate regridding
```

## Supported Parameters

| Category | Parameters |
|----------|------------|
| Precipitation | `precipacc24h`, `precipacc03h` |
| Temperature | `tempscreen`, `tempdaymax`, `tempnightmin`, `temp{200-950}hPa` |
| Dew Point | `tempdewscreen`, `tempdew{200-950}hPa` |
| Wind Speed | `windspd10m`, `windspd{700-950}hPa` |

## Output Types

- **expectedvalues_extract** - Ensemble mean
- **percentiles_extract** - Percentile fields (10th, 25th, 50th, 75th, 90th)
- **probabilities_extract** - Threshold exceedance probabilities (PoP, PoW, POT, POTd)
