# run_app.py
import panel as pn
import panel_app

pn.extension(sizing_mode="stretch_width")

# Get the dashboard
dashboard = panel_app.get_dashboard()

# Serve the dashboard in Binder
pn.serve(dashboard, show=True, start=True)
