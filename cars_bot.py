"""Toyota RAV4 ownership-cost simulator dashboard. La Centrale scraping removed."""
import streamlit as st


def cars_dashboard():
    st.title("🚗 Cars – Toyota RAV4")
    st.markdown(
        "Comparateur HEV / PHEV Toyota RAV4. "
        "Les véhicules sont saisis manuellement : aucune annonce n'est récupérée automatiquement depuis La Centrale."
    )
    st.info(
        "La recherche automatique La Centrale a été retirée : l'extraction ne fournissait pas "
        "de résultats suffisamment fiables. Le simulateur reste disponible indépendamment des annonces."
    )

    try:
        from cars_simulator import render_rav4_simulator
        render_rav4_simulator()
    except Exception as exc:
        st.error(f"Le simulateur RAV4 est temporairement indisponible : {exc}")
