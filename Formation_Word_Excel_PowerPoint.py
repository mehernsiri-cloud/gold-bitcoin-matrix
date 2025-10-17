import streamlit as st

def render_training_dashboard():
    st.title("🎓 Formation Microsoft Word — Niveau Débutant (LMS-Style)")

    st.markdown("""
    Bienvenue dans votre parcours de formation **Microsoft Word** pour débutants.  
    Suivez les modules, sous-modules et cours pour progresser étape par étape.  
    Les liens renvoient vers des ressources fiables et publiques sur [CoursInfo.fr](https://www.coursinfo.fr/word/).
    """)

    # --- Learning Path Modules ---
    st.header("📝 Parcours de formation Word - Débutants")

    # Module 1
    with st.expander("Module 1 : Prise en main de Word"):
        st.markdown("Découvrir l’interface et créer ses premiers documents Word.")
        
        # Submodule 1.1
        with st.expander("1.1 Ouvrir Word et créer un document"):
            st.markdown("""
            - **Cours:** Comment ouvrir Word sur Windows  
            - **Ressource en ligne:** [CoursInfo - Ouvrir Word](https://www.coursinfo.fr/word/)  
            - **Exercice:** Créez un document vierge, enregistrez-le sur votre bureau.
            """)
        
        # Submodule 1.2
        with st.expander("1.2 Enregistrer et fermer un document"):
            st.markdown("""
            - **Cours:** Enregistrer, sauvegarder et fermer un document Word  
            - **Ressource en ligne:** [CoursInfo - Sauvegarder Word](https://www.coursinfo.fr/word/)  
            - **Exercice:** Enregistrez et fermez votre document créé.
            """)

    # Module 2
    with st.expander("Module 2 : Mise en forme et styles"):
        st.markdown("Apprendre à formater le texte, utiliser les styles, listes et tableaux.")
        
        # Submodule 2.1
        with st.expander("2.1 Formater le texte et les paragraphes"):
            st.markdown("""
            - **Cours:** Styles, polices et alignements  
            - **Ressource en ligne:** [CoursInfo - Formater le texte](https://www.coursinfo.fr/word/)  
            - **Exercice:** Formatez un texte avec différentes polices et couleurs.
            """)
        
        # Submodule 2.2
        with st.expander("2.2 Listes et tableaux"):
            st.markdown("""
            - **Cours:** Créer des listes à puces, numérotées et insérer un tableau  
            - **Ressource en ligne:** [CoursInfo - Listes et tableaux](https://www.coursinfo.fr/word/)  
            - **Exercice:** Créez une liste et un tableau dans votre document.
            """)

    # Module 3
    with st.expander("Module 3 : En-têtes, pieds de page et pagination"):
        st.markdown("Ajouter en-têtes, pieds de page et numérotation pour un document professionnel.")
        
        # Submodule 3.1
        with st.expander("3.1 En-têtes et pieds de page"):
            st.markdown("""
            - **Cours:** Ajouter en-têtes et pieds de page  
            - **Ressource en ligne:** [CoursInfo - En-têtes & pieds de page](https://www.coursinfo.fr/word/)  
            - **Exercice:** Ajoutez un en-tête et un pied de page à votre document.
            """)
        
        # Submodule 3.2
        with st.expander("3.2 Numérotation des pages"):
            st.markdown("""
            - **Cours:** Numérotation des pages  
            - **Ressource en ligne:** [CoursInfo - Numérotation](https://www.coursinfo.fr/word/)  
            - **Exercice:** Numérotez toutes les pages de votre document.
            """)

    st.markdown("---")
    st.caption("© 2025 Formation IA & Bureautique — LMS-style Word parcours pour débutants. Les liens renvoient vers des ressources publiques fiables.")
