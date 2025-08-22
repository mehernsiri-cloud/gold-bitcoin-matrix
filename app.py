import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

@st.cache_data
def load_predictions():
    url = "https://raw.githubusercontent.com/YOUR_USERNAME/YOUR_REPO/main/predictions_log.csv"
    return pd.read_csv(url, parse_dates=["date"])

st.title("📊 Gold, Bitcoin & Real Estate Prediction Dashboard")

df = load_predictions()

# Latest predictions with risk
st.subheader("🔮 Latest Predictions")
latest = df.groupby("asset").tail(1)
st.dataframe(latest)

# Color-coded risk
def risk_color(r):
    if r=="Low": return "green"
    if r=="Medium": return "orange"
    return "red"

for _, row in latest.iterrows():
    st.markdown(f"**{row['asset']}** → Predicted: {row['prediction']} | Risk: "
                f"<span style='color:{risk_color(row['risk'])}'>{row['risk']}</span>", 
                unsafe_allow_html=True)

# Plot comparison chart
asset_choice = st.selectbox("Choose an asset:", df["asset"].unique())
df_asset = df[df["asset"]==asset_choice]

fig, ax = plt.subplots()
ax.plot(df_asset["date"], df_asset["actual"], label="Actual", marker="o")
ax.plot(df_asset["date"], df_asset["prediction"], label="Prediction", linestyle="--")
ax.set_title(f"{asset_choice}: Actual vs Prediction")
ax.legend()
st.pyplot(fig)
