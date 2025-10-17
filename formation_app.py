import streamlit as st

# Must be first
st.set_page_config(
    page_title="Formation Word / Excel / PowerPoint",
    page_icon="🎓",
    layout="wide"
)

from Formation_Word_Excel_PowerPoint import render_training_dashboard


# --------------------------
# 🎯 Header / Branding
# --------------------------
with st.container():
    col1, col2 = st.columns([0.2, 0.8])
    with col1:
        st.image(
            "https://upload.wikimedia.org/wikipedia/commons/4/45/Microsoft_Office_logo_%282013%E2%80%932019%29.svg",
            width=80,
        )
    with col2:
        st.markdown(
            """
            <div style='padding-top:10px'>
            <h1 style='color:#2E86C1; font-size:32px;'>Formation Microsoft Word, Excel & PowerPoint</h1>
            <p style='font-size:18px; color:#555;'>Cours gratuits, vidéos, exercices et examens – en français 🇫🇷</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("---")

# --------------------------
# 📚 Content Rendering
# --------------------------
try:
    render_training_dashboard()
except Exception as e:
    st.error(f"⚠️ Une erreur est survenue lors du chargement de la formation : {e}")

# --------------------------
# 🧩 Footer
# --------------------------
st.markdown("---")
st.markdown(
    """
    <div style='text-align:center; color:gray; font-size:14px;'>
    © 2025 Formation Word / Excel / PowerPoint — propulsé par votre projet Streamlit 🎓
    </div>
    """,
    unsafe_allow_html=True,
)
