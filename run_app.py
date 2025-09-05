# run_app.py
import panel as pn
import webbrowser
import socket
from panel_app import app  # Make sure your panel_app.py defines 'app'

# -----------------------------
# Helper: find free port
# -----------------------------
def find_free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port

# -----------------------------
# Main function
# -----------------------------
def main():
    pn.extension(sizing_mode="stretch_width")

    # Find a free port
    port = find_free_port()
    url = f"http://localhost:{port}"

    print(f"Launching Panel dashboard at {url} ...")

    # Open browser automatically
    webbrowser.open(url)

    # Serve the app
    pn.serve(app, port=port, show=False, threaded=True)

if __name__ == "__main__":
    main()
