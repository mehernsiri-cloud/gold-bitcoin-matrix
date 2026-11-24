"""Toyota RAV4 comparison dashboard. Automatic La Centrale scraping is intentionally disabled."""
import streamlit as st


def cars_dashboard():
    st.title("🚗 Cars – Toyota RAV4")
    st.markdown(
        "Comparateur HEV / PHEV Toyota RAV4 XA50. Les annonces sont saisies manuellement ; "
        "le lien La Centrale ouvre directement la recherche avec les filtres sélectionnés."
    )
    st.info(
        "La récupération automatique des annonces La Centrale est désactivée car elle ne fournissait pas "
        "de résultats suffisamment fiables. Utilise le lien direct dans le comparateur pour consulter les annonces à jour."
    )
    try:
        from cars_simulator import render_rav4_simulator
        render_rav4_simulator()
    except Exception as exc:
        st.error(f"Le comparateur RAV4 est temporairement indisponible : {exc}")
