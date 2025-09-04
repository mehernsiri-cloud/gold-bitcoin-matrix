import streamlit as st
import plotly.graph_objects as go
from ai_predictor import predict_next_n

def render_ai_forecast(df_actual, df_pred, n_steps=7):
    """
    Renders the AI Forecast dashboard with the same layout style 
    as the Gold & Bitcoin main dashboard.
    """
    assets = [("Gold", "gold_actual"), ("Bitcoin", "bitcoin_actual")]

    for asset, actual_col in assets:
        st.subheader(asset)

        # Run AI predictor
        df_ai = predict_next_n(df_actual, df_pred, asset, n_steps)

        if df_ai.empty:
            st.info(f"No AI forecast available for {asset}.")
            continue

        # Historical actuals
        df_hist = df_actual[["timestamp", actual_col]].rename(columns={actual_col: "actual"})

        # Plot chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_hist["timestamp"], y=df_hist["actual"],
            mode="lines+markers", name="Actual",
            line=dict(color="#42A5F5", width=2)
        ))
        fig.add_trace(go.Scatter(
            x=df_ai["timestamp"], y=df_ai["predicted_price"],
            mode="lines+markers", name="AI Forecast",
            line=dict(color="#FF6F61", dash="dash")
        ))

        fig.update_layout(
            title=f"{asset} AI Forecast vs Actual",
            xaxis_title="Date", yaxis_title="Price",
            plot_bgcolor="#FAFAFA", paper_bgcolor="#FAFAFA"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Show table
        st.dataframe(df_ai)
