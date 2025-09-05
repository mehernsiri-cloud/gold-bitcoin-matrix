
# Gold & Bitcoin Predictive Matrix — Streamlit Web App


[![Launch Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/<your-github-mehernsiri-cloud>/<gold-bitcoin-matrix>/HEAD?filepath=run_app.py)



This is a **no-install web app** to score macro drivers and predict 30‑day directional bias and magnitude for **Gold** and **Bitcoin**.

## Files
- `app.py` — main Streamlit app
- `requirements.txt` — dependencies
- `weights.yaml` — default weights (you can upload your own from the sidebar)

---

## Deploy on Streamlit Community Cloud (Free)
1. Create a free account at https://streamlit.io/cloud
2. Push these files to a **public GitHub repo** (e.g., `gold-bitcoin-matrix`).
3. In Streamlit Cloud, click **New app** → select your repo and `app.py`.
4. Deploy. The app will be live at a shareable URL.

Notes:
- Streamlit Cloud automatically installs packages from `requirements.txt`.
- You can update the app by pushing new commits to GitHub.

---

## Deploy on Hugging Face Spaces (Free)
1. Create a free account at https://huggingface.co
2. Click **Spaces** → **Create New Space** → **Streamlit**.
3. Upload `app.py` and `requirements.txt` (and optionally `weights.yaml`).
4. Click **Create** and wait for the build. Your app will be served at a public URL.

---

## How to Use
- Choose **Manual** mode and set driver values (−2..+2) with sliders.
- Or switch to **Auto-placeholder** (replace with real API logic later inside `app.py`).
- Download or upload **custom weights** via the sidebar.

## Plugging Real-Time Data (later)
- Replace the `placeholder_auto_fetch()` with API calls:
  - Geopolitics (GDELT, NewsAPI)
  - Inflation & Real rates (FRED API)
  - USD/DXY (market data API)
  - Liquidity (Fed/ECB balance sheet endpoints)
  - Equity/Bond flows (ETF/EPFR/alternative providers)
  - Crypto adoption & regulation (ETF approvals, addresses, volumes)

## Calibration
- Adjust `weights.yaml` to reflect your view or backtests.
- Mapping from score→return is linear: Gold ±15% max, BTC ±30% max (edit inside `app.py`).

Generated: 2025-08-21T14:22:20.088516Z
