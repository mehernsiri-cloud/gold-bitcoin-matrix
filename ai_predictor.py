# ai_predictor.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from datetime import timedelta
import os
import yaml
import streamlit as st
import plotly.graph_objects as go

DATA_DIR = "data"
PREDICTION_FILE = os.path.join(DATA_DIR, "predictions_log.csv")
ACTUAL_FILE = os.path.join(DATA_DIR, "actual_data.csv")
WEIGHT_FILE = "weight.yaml"

# ------------------------------
# Load macro indicators from weight.yaml
# ------------------------------
def load_macro_indicators(asset_name):
    if not os.path.exists(WEIGHT_FILE):
        return {}
    with open(WEIGHT_FILE, "r") as f:
        weights = yaml.safe_load(f)
    return weights.get(asset_name.lower(), {})

# ------------------------------
# Load historical data including macro indicators
# ------------------------------
def load_data(asset_name, actual_col):
    if not os.path.exists(PREDICTION_FILE) or not os.path.exists(ACTUAL_FILE):
        return pd.DataFrame()

    df_pred = pd.read_csv(PREDICTION_FILE, parse_dates=["timestamp"])
    df_actual = pd.read_csv(ACTUAL_FILE, parse_dates=["timestamp"])

    df_asset = df_pred[df_pred["asset"] == asset_name].copy()

    # Merge with actuals
    if actual_col in df_actual.columns:
        df_asset = pd.merge_asof(
            df_asset.sort_values("timestamp"),
            df_actual.sort_values("timestamp")[["timestamp", actual_col]].rename(columns={actual_col: "actual"}),
            on="timestamp",
            direction="backward"
        )
    else:
        df_asset["actual"] = np.nan

    # Load macro indicators from weight.yaml
    macro_cols = ["inflation", "usd_strength", "energy_prices", "tail_risk_event"]
    macros = load_macro_indicators(asset_name)
    for col in macro_cols:
        df_asset[col] = macros.get(col, 0.0)

    df_asset["predicted_price"] = pd.to_numeric(df_asset["predicted_price"], errors="coerce")
    df_asset["actual"] = pd.to_numeric(df_asset["actual"], errors="coerce")
    df_asset.dropna(subset=["predicted_price", "actual"], inplace=True)
    df_asset = df_asset.reset_index(drop=True)
    return df_asset

# ------------------------------
# Create rolling window features
# ------------------------------
def create_features(df, window=3):
    X, y = [], []
    macro_cols = ["inflation", "usd_strength", "energy_prices", "tail_risk_event"]
    n = len(df)
    for i in range(window, n):
        price_feat = np.concatenate([df["actual"].values[i-window:i],
                                     df["predicted_price"].values[i-window:i]])
        macro_feat = df[macro_cols].iloc[i-window:i].values.flatten()
        X.append(np.concatenate([price_feat, macro_feat]))
        y.append(df["actual"].values[i])
    return np.array(X), np.array(y)

# ------------------------------
# Predict next n steps
# ------------------------------
def predict_next_n(df_actual, df_pred, asset_name="Gold", n_steps=7, window=3):
    df = load_data(asset_name, f"{asset_name.lower()}_actual")
    if df.empty or len(df) < window + 1:
        return pd.DataFrame(columns=["timestamp", "predicted_price"])

    X_train, y_train = create_features(df, window=window)
    model = RandomForestRegressor(n_estimators=300, random_state=42)
    model.fit(X_train, y_train)

    last_actuals = df["actual"].values[-window:].tolist()
    last_preds = df["predicted_price"].values[-window:].tolist()
    last_macro = df[["inflation", "usd_strength", "energy_prices", "tail_risk_event"]].values[-window:].tolist()
    future_dates = [df["timestamp"].max() + timedelta(days=i) for i in range(1, n_steps+1)]
    predictions = []

    for _ in range(n_steps):
        macro_feat = np.array(last_macro).flatten()
        features = np.array(last_actuals + last_preds + macro_feat.tolist()).reshape(1, -1)
        next_pred = model.predict(features)[0]
        predictions.append(next_pred)

        last_actuals = last_actuals[1:] + [next_pred]
        last_preds = last_preds[1:] + [next_pred]
        last_macro = last_macro[1:] + [last_macro[-1]]  # keep macro indicators constant

    df_future = pd.DataFrame({
        "timestamp": future_dates,
        "predicted_price": predictions
    })
    return df_future

# ------------------------------
# Helper functions for dashboard layout
# ------------------------------
ASSET_THEMES = {
    "Gold": {"chart_actual":"#FBC02D", "chart_pred":"#FFCC80", "chart_ai":"#FF6F61",
             "buy":"#FFF9C4","sell":"#FFE0B2","hold":"#E0E0E0",
             "target_bg":"#FFFDE7","target_text":"black","assumption_pos":"#FFD54F","assumption_neg":"#FFAB91"},
    "Bitcoin": {"chart_actual":"#42A5F5","chart_pred":"#81D4FA","chart_ai":"#FF6F61",
                "buy":"#BBDEFB","sell":"#FFCDD2","hold":"#CFD8DC",
                "target_bg":"#E3F2FD","target_text":"black","assumption_pos":"#64B5F6","assumption_neg":"#EF9A9A"}
}

INDICATOR_ICONS = {
    "inflation": "💹", "real_rates": "🏦", "bond_yields": "📈", "energy_prices": "🛢️",
    "usd_strength": "💵", "liquidity": "💧", "equity_flows": "📊", "regulation": "🏛️",
    "adoption": "🤝", "currency_instability": "⚖️", "recession_probability": "📉",
    "tail_risk_event": "🚨", "geopolitics": "🌍"
}

def alert_badge(signal, asset_name):
    theme = ASSET_THEMES[asset_name]
    color = theme["buy"] if signal=="Buy" else theme["sell"] if signal=="Sell" else theme["hold"]
    return f'<div style="background-color:{color};color:black;padding:8px;font-size:20px;text-align:center;border-radius:8px">{signal.upper()}</div>'

def target_price_card(price, asset_name, horizon):
    theme = ASSET_THEMES[asset_name]
    st.markdown(f"""
        <div style='background-color:{theme["target_bg"]};color:{theme["target_text"]};
        padding:12px;font-size:22px;text-align:center;border-radius:12px;margin-bottom:10px'>
        💰 {asset_name} Target Price: {price} <br>⏳ Horizon: {horizon}
        </div>
        """, unsafe_allow_html=True)

def explanation_card(asset_df, asset_name):
    if asset_df.empty: return
    st.markdown(f"<div style='background-color:#FAFAFA;padding:12px;border-radius:10px;margin-bottom:10px'>AI forecast for {asset_name}</div>", unsafe_allow_html=True)

def assumptions_card(asset_df, asset_name):
    theme = ASSET_THEMES[asset_name]
    st.markdown(f"<div style='background-color:#FAFAFA;padding:12px;border-radius:10px;margin-bottom:10px'>Assumptions loaded for {asset_name}</div>", unsafe_allow_html=True)

# ------------------------------
# Render AI Forecast dashboard section
# ------------------------------
def render_ai_forecast(df_actual, df_pred, n_steps=7):
    col1, col2 = st.columns(2)
    for col, asset, actual_col in zip([col1, col2], ["Gold", "Bitcoin"], ["gold_actual","bitcoin_actual"]):
        with col:
            st.subheader(asset)
            df_ai = predict_next_n(df_actual, df_pred, asset, n_steps)
            if df_ai.empty:
                st.info(f"No AI prediction available for {asset}.")
                continue

            last_actual = df_actual[actual_col].dropna().iloc[-1] if not df_actual[actual_col].dropna().empty else None
            last_pred = df_ai["predicted_price"].iloc[-1]
            signal = "Buy" if last_pred > last_actual else ("Sell" if last_pred < last_actual else "Hold") if last_actual else "Hold"

            trend = ""
            if len(df_ai) >= 3:
                last3 = df_ai["predicted_price"].tail(3)
                if last3.is_monotonic_increasing:
                    trend = "Bullish 📈"
                elif last3.is_monotonic_decreasing:
                    trend = "Bearish 📉"
                else:
                    trend = "Neutral ⚖️"

            st.markdown(alert_badge(signal, asset), unsafe_allow_html=True)
            st.markdown(f"**Market Trend:** {trend}")
            target_price_card(last_pred, asset, "Days")
            explanation_card(df_ai, asset)
            assumptions_card(df_ai, asset)

            # Plot actual + AI predictions
            df_hist = df_actual[["timestamp", actual_col]].rename(columns={actual_col:"actual"})
            theme = ASSET_THEMES[asset]
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_hist["timestamp"], y=df_hist["actual"],
                mode="lines+markers", name="Actual",
                line=dict(color=theme["chart_actual"], width=2)
            ))
            fig.add_trace(go.Scatter(
                x=df_ai["timestamp"], y=df_ai["predicted_price"],
                mode="lines+markers", name="AI Forecast",
                line=dict(color=theme["chart_ai"], dash="dot")
            ))
            fig.update_layout(
                title=f"{asset} AI Forecast vs Actual",
                xaxis_title="Date", yaxis_title="Price",
                plot_bgcolor="#FAFAFA", paper_bgcolor="#FAFAFA"
            )
            st.plotly_chart(fig, use_container_width=True)
