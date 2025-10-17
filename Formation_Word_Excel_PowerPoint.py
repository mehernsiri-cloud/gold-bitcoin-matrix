import streamlit as st

# Example JSON-like offline Word content
WORD_JSON = {
    "Module 1 : Découverte de Word": {
        "1.1 - Ouvrir et naviguer dans Word": {
            "text": """
Apprendre à ouvrir Word et naviguer dans les menus principaux.
- Accéder au logiciel depuis le bureau ou la barre des tâches
- Comprendre les onglets et le ruban
            """,
            "images": [],
            "url": "https://www.coursinfo.fr/word/ouvrir-word"
        },
        "1.2 - Interface et rubans": {
            "text": """
Identifier les rubans, onglets, et barres d'outils.
- Accéder aux fonctionnalités rapidement
- Utiliser les icônes pour mise en forme
            """,
            "images": [],
            "url": "https://www.coursinfo.fr/word/interface-rubans"
        },
        "1.3 - Créer un document simple": {
            "text": """
Créer et enregistrer un document simple.
- Rédiger un texte
- Enregistrer et rouvrir un fichier
            """,
            "images": [],
            "url": "https://www.coursinfo.fr/word/creer-document"
        }
    },
    "Module 2 : Mise en forme": {
        "2.1 - Polices et paragraphes": {
            "text": "Modifier les polices, tailles, et alignement des paragraphes.",
            "images": [],
            "url": "https://www.coursinfo.fr/word/polices-paragraphes"
        },
        "2.2 - Styles et thèmes": {
            "text": "Appliquer des styles prédéfinis et thèmes de document.",
            "images": [],
            "url": "https://www.coursinfo.fr/word/styles-themes"
        },
        "2.3 - Listes et tableaux": {
            "text": "Créer des listes à puces/numérotées et des tableaux simples.",
            "images": [],
            "url": "https://www.coursinfo.fr/word/listes-tableaux"
        }
    },
    "Module 3 : Documents avancés": {
        "3.1 - En-têtes et pieds de page": {
            "text": "Ajouter et personnaliser les en-têtes et pieds de page.",
            "images": [],
            "url": "https://www.coursinfo.fr/word/en-tetes-pieds"
        },
        "3.2 - Table des matières automatique": {
            "text": "Créer et mettre à jour automatiquement une table des matières.",
            "images": [],
            "url": "https://www.coursinfo.fr/word/table-matieres"
        },
        "3.3 - Insertion d’images et graphiques": {
            "text": "Insérer et formater des images et graphiques dans Word.",
            "images": [],
            "url": "https://www.coursinfo.fr/word/images-graphiques"
        }
    }
}

def render_training_dashboard():
    st.title("🎓 Formation Bureautique — Word, Excel & PowerPoint (Débutants)")

    st.markdown("""
    Bienvenue dans votre espace de formation continue !  
    Ici, vous trouverez un parcours structuré pour apprendre **Microsoft Word, Excel et PowerPoint**.  
    Les ressources sont gratuites, accessibles hors ligne, et organisées par modules et sous-modules.
    """)

    # --- Sidebar menu for navigation ---
    section = st.sidebar.radio(
        "📘 Choisissez un module de formation :",
        ["Introduction", "Microsoft Word", "Microsoft Excel", "Microsoft PowerPoint", "Tests & Exercices"]
    )

    # --- Introduction ---
    if section == "Introduction":
        st.subheader("🧭 Objectifs de la formation")
        st.markdown("""
        - Acquérir les bases de Word, Excel et PowerPoint  
        - Créer, formater et présenter des documents professionnels  
        - Maîtriser les outils bureautiques pour le travail en entreprise
        """)
        st.info("💡 Avancez module par module et testez vos connaissances à la fin de chaque partie.")

    # --- Microsoft Word ---
    elif section == "Microsoft Word":
        st.header("📝 Parcours Word — Débutant")

        for module_name, submodules in WORD_JSON.items():
            with st.expander(module_name):
                submodule_names = list(submodules.keys())
                selected_sub = st.selectbox(f"Sélectionnez un cours dans {module_name}", submodule_names, key=module_name)
                content = submodules[selected_sub]

                # Display lesson content
                st.markdown(content["text"])
                for img_url in content["images"]:
                    st.image(img_url, use_column_width=True)
                st.markdown(f"[Voir sur coursinfo.fr]({content['url']})")

        # Exercises
        st.subheader("📚 Exercices pratiques Word")
        st.markdown("""
        - Rédigez un courrier professionnel avec en-tête et pied de page  
        - Créez une page de garde et appliquez un style uniforme  
        - Insérez une table des matières automatique
        """)

        # Mini quiz
        st.subheader("🧠 Mini-quiz Word")
        st.markdown("**Question 1:** Quelle option permet de créer un en-tête dans Word ?")
        st.checkbox("Insertion > En-tête", key="q1a")
        st.checkbox("Disposition > Bordures", key="q1b")
        st.checkbox("Accueil > Styles", key="q1c")
        st.success("Réponse correcte : Insertion > En-tête")

    # --- Microsoft Excel ---
    elif section == "Microsoft Excel":
        st.header("📊 Parcours Excel — Débutant")
        st.info("💡 Contenu à compléter hors ligne comme pour Word, avec modules et sous-modules.")

    # --- Microsoft PowerPoint ---
    elif section == "Microsoft PowerPoint":
        st.header("📈 Parcours PowerPoint — Débutant")
        st.info("💡 Contenu à compléter hors ligne comme pour Word, avec modules et sous-modules.")

    # --- Tests & Exercices ---
    elif section == "Tests & Exercices":
        st.header("🧩 Tests & Exercices finaux")
        st.markdown("""
        Testez vos compétences à travers ces mini-projets :
        - **Word :** Créez une lettre professionnelle avec en-tête et pied de page  
        - **Excel :** Créez un tableau de suivi de dépenses avec un graphique  
        - **PowerPoint :** Présentez un sujet de votre choix en 5 slides
        """)
        st.success("✅ Astuce : Comparez vos fichiers avec les exemples disponibles en ligne.")

    st.markdown("---")
    st.caption("© 2025 Formation IA & Bureautique — Ressources gratuites pour l'apprentissage continu.")
