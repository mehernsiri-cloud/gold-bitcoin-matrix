import streamlit as st

def render_word_training():
    st.title("🎓 Formation Microsoft Word — Niveau Débutant")

    st.markdown("""
    Suivez le parcours complet pour débutants sur Microsoft Word.  
    Chaque module contient des sous-modules avec un cours et un exercice pratique.  
    Les liens renvoient vers les pages correspondantes sur [CoursInfo.fr](https://www.coursinfo.fr/word/).
    """)

    # --- Module 1: Prise en main ---
    with st.expander("Module 1 : Prise en main de Word"):
        st.markdown("**1.1 Ouvrir Word sur Windows**")
        st.markdown("- Description : Comment lancer Word et créer un document vierge.")
        st.markdown("- Cours en ligne : [CoursInfo - Ouvrir Word](https://www.coursinfo.fr/word/comment-ouvrir-le-logiciel-word-sur-windows)")
        st.markdown("- Exercice : Créez et enregistrez un document sur votre bureau.\n")

        st.markdown("**1.2 Enregistrer et fermer un document**")
        st.markdown("- Description : Sauvegarder vos documents correctement.")
        st.markdown("- Cours en ligne : [CoursInfo - Enregistrer et fermer](https://www.coursinfo.fr/word/enregistrer-et-fermer-un-document)")
        st.markdown("- Exercice : Enregistrez votre document et fermez-le.\n")

    # --- Module 2: Mise en forme et styles ---
    with st.expander("Module 2 : Mise en forme et styles"):
        st.markdown("**2.1 Appliquer des styles et formats**")
        st.markdown("- Description : Police, taille, couleurs, alignement et styles.")
        st.markdown("- Cours en ligne : [CoursInfo - Styles et formats](https://www.coursinfo.fr/word/appliquer-des-styles-et-formats)")
        st.markdown("- Exercice : Formatez un texte avec différentes polices et styles.\n")

        st.markdown("**2.2 Créer des listes et tableaux**")
        st.markdown("- Description : Listes à puces, numérotées et insertion de tableaux.")
        st.markdown("- Cours en ligne : [CoursInfo - Listes et tableaux](https://www.coursinfo.fr/word/creer-des-listes-et-tableaux)")
        st.markdown("- Exercice : Créez une liste et un tableau dans votre document.\n")

    # --- Module 3: En-têtes et pagination ---
    with st.expander("Module 3 : En-têtes, pieds de page et numérotation"):
        st.markdown("**3.1 Ajouter en-têtes et pieds de page**")
        st.markdown("- Description : Personnaliser vos documents avec en-têtes/pieds de page.")
        st.markdown("- Cours en ligne : [CoursInfo - En-têtes et pieds de page](https://www.coursinfo.fr/word/ajouter-en-tetes-et-pieds-de-page)")
        st.markdown("- Exercice : Ajoutez un en-tête et pied de page à votre document.\n")

        st.markdown("**3.2 Numérotation des pages**")
        st.markdown("- Description : Numéroter vos pages automatiquement.")
        st.markdown("- Cours en ligne : [CoursInfo - Numérotation des pages](https://www.coursinfo.fr/word/numerotation-des-pages)")
        st.markdown("- Exercice : Numérotez toutes les pages de votre document.\n")

    st.markdown("---")
    st.caption("© 2025 Formation IA & Bureautique — Parcours Word débutant avec ressources publiques.")
