# panel_app.py
import panel as pn
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import yaml
import os

# Import your local modules
from ai_predictor import predict_next_n
import jobs_app
import fetch_data
import utils

pn.extension('plotly')

# ------------------------------
# Paths / constants
# ------------------------------
DATA_DIR = "data"
ACTUAL_DATA_FILE = os.path.join(DATA_DIR, "actual_data.csv")
WEIGHT_FILE = "weight.yaml"

# ------------------------------
# Load weight.yaml indicators
# ------------------------------
def load_weights(asset_name):
    if not os.path.exists(WEIGHT_FILE):
        return {}
    try:
        with open(WEIGHT_FILE, "r") as f:
            weights = yaml.safe_load(f) or {}
        return weights.get(asset_name.lower(), {}) or {}
    except Exception as e:
        print(f"[panel_app] Warning loading weights: {e}")
        return {}

# ------------------------------
# Load actual historical prices
# ------------------------------
def load_actual_prices(asset_name):
    if not os.path.exists(ACTUAL_DATA_FILE):
        return pd.Series(dtype=float)
    
    df = pd.read_csv(ACTUAL_DATA_FILE, parse_dates=["timestamp"])
    col_map = {"gold": "gold_actual", "bitcoin": "bitcoin_actual"}
    col = col_map.get(asset_name.lower())
    if col is None or col not in df.columns:
        return pd.Series(dtype=float)
    
    df = df.sort_values("timestamp")
    
    # Use daily average to speed up
    df['date'] = df['timestamp'].dt.date
    daily_avg = df.groupby('date')[col].mean().reset_index()
    daily_avg['date'] = pd.to_datetime(daily_avg['date'])
    return daily_avg

# ------------------------------
# Plot actual vs predicted
# ------------------------------
def plot_prices(asset_name, n_steps=5):
    actual_df = load_actual_prices(asset_name)
    predicted_df = predict_next_n(asset_name=asset_name, n_steps=n_steps)

    fig = go.Figure()
    if not actual_df.empty:
        fig.add_trace(go.Scatter(
            x=actual_df['date'],
            y=actual_df.iloc[:,1],
            mode='lines+markers',
            name='Historical'
        ))
    if not predicted_df.empty:
        fig.add_trace(go.Scatter(
            x=predicted_df['timestamp'],
            y=predicted_df['predicted_price'],
            mode='lines+markers',
            name='Predicted'
        ))

    fig.update_layout(
        title=f"{asset_name.capitalize()} Prices",
        xaxis_title="Date",
        yaxis_title="Price",
        template="plotly_white"
    )
    return fig

# ------------------------------
# Create asset panel
# ------------------------------
def create_asset_panel(asset_name):
    n_steps_slider = pn.widgets.IntSlider(name="Forecast Horizon (days)", start=1, end=30, value=5)

    @pn.depends(n_steps_slider)
    def update_plot(n_steps):
        fig = plot_prices(asset_name, n_steps=n_steps)
        return pn.pane.Plotly(fig, sizing_mode='stretch_width', height=500)

    weights = load_weights(asset_name)
    weights_table = pn.widgets.DataFrame(pd.DataFrame(weights.items(), columns=['Indicator', 'Value']),
                                         name=f"{asset_name.capitalize()} Indicators",
                                         autosize_mode=True)

    return pn.Column(
        f"## {asset_name.capitalize()} Dashboard",
        n_steps_slider,
        update_plot,
        "### Current Indicators",
        weights_table
    )

# ------------------------------
# Build Tabs for multiple assets
# ------------------------------
tabs = pn.Tabs(
    ("Gold", create_asset_panel("gold")),
    ("Bitcoin", create_asset_panel("bitcoin")),
)

# ------------------------------
# Include other modules / jobs panel
# ------------------------------
def create_jobs_panel():
    return pn.Column(
        "## Jobs Panel",
        pn.pane.Markdown("This panel can integrate your `jobs_app` outputs."),
        # Example: call a function from jobs_app
        pn.pane.DataFrame(jobs_app.list_jobs(), autosize_mode=True)
    )

# ------------------------------
# Main App
# ------------------------------
main_tabs = pn.Tabs(
    ("Assets", tabs),
    ("Jobs", create_jobs_panel()),
    ("Data Fetch", pn.pane.Markdown("Use `fetch_data` module here to fetch external data.")),
)

# ------------------------------
# Serve Panel
# ------------------------------
if __name__.startswith("bokeh"):
    main_tabs.servable(title="Gold & Bitcoin Dashboard")
else:
    pn.serve(main_tabs, port=5006, show=True)
