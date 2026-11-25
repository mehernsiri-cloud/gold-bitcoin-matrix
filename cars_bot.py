"""Cars section with legacy RAV4 HEV/PHEV and expanded family SUV comparator."""
import streamlit as st


def cars_dashboard():
    st.title("🚗 Cars — Comparateurs SUV familiaux")
    page = st.radio(
        "Choisir le comparateur",
        ["Toyota RAV4 — HEV / PHEV (historique)", "SUV familial — RAV4 & Audi Q5 quattro"],
        horizontal=True,
        key="cars_comparator_menu",
    )

    if page == "SUV familial — RAV4 & Audi Q5 quattro":
        try:
            from cars_family_comparator import render_family_suv_comparator
            render_family_suv_comparator()
        except Exception as exc:
            st.error(f"Le comparateur SUV familial est temporairement indisponible : {exc}")
        return

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
