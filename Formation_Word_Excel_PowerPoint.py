import streamlit as st

def render_training_dashboard():
    st.title("🎓 Formation Bureautique — Word, Excel & PowerPoint (Débutants)")

    st.markdown("""
    Bienvenue dans votre espace de formation continue !  
    Ici, vous trouverez un parcours structuré pour apprendre **Microsoft Word, Excel et PowerPoint**.  
    Les ressources sont gratuites, accessibles en ligne, et organisées par modules et sous-modules.
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

        modules = {
            "Module 1 : Découverte de Word": {
                "1.1 - Ouvrir et naviguer dans Word": "https://www.coursinfo.fr/word/ouvrir-word",
                "1.2 - Interface et rubans": "https://www.coursinfo.fr/word/interface-rubans",
                "1.3 - Créer un document simple": "https://www.coursinfo.fr/word/creer-document"
            },
            "Module 2 : Mise en forme": {
                "2.1 - Polices et paragraphes": "https://www.coursinfo.fr/word/polices-paragraphes",
                "2.2 - Styles et thèmes": "https://www.coursinfo.fr/word/styles-themes",
                "2.3 - Listes et tableaux": "https://www.coursinfo.fr/word/listes-tableaux"
            },
            "Module 3 : Documents avancés": {
                "3.1 - En-têtes et pieds de page": "https://www.coursinfo.fr/word/en-tetes-pieds",
                "3.2 - Table des matières automatique": "https://www.coursinfo.fr/word/table-matieres",
                "3.3 - Insertion d’images et graphiques": "https://www.coursinfo.fr/word/images-graphiques"
            }
        }

        for module_name, submodules in modules.items():
            with st.expander(module_name):
                for sub_name, link in submodules.items():
                    st.markdown(f"- [{sub_name}]({link})")

        # Exercises
        st.subheader("📚 Exercices pratiques")
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

        modules = {
            "Module 1 : Découverte d’Excel": {
                "1.1 - Ouvrir et naviguer dans Excel": "https://www.coursinfo.fr/excel/ouvrir-excel",
                "1.2 - Cellules et feuilles": "https://www.coursinfo.fr/excel/cellules-feuilles",
                "1.3 - Saisie et formatage de données": "https://www.coursinfo.fr/excel/saisie-formatage"
            },
            "Module 2 : Formules et calculs": {
                "2.1 - Somme, Moyenne, Min, Max": "https://www.coursinfo.fr/excel/formules-simples",
                "2.2 - Références absolues et relatives": "https://www.coursinfo.fr/excel/references",
                "2.3 - Fonctions logiques SI": "https://www.coursinfo.fr/excel/fonction-si"
            },
            "Module 3 : Graphiques et tableaux": {
                "3.1 - Créer un graphique simple": "https://www.coursinfo.fr/excel/graphique",
                "3.2 - Mise en forme conditionnelle": "https://www.coursinfo.fr/excel/mise-en-forme-conditionnelle",
                "3.3 - Filtres et tris": "https://www.coursinfo.fr/excel/filtres-tris"
            }
        }

        for module_name, submodules in modules.items():
            with st.expander(module_name):
                for sub_name, link in submodules.items():
                    st.markdown(f"- [{sub_name}]({link})")

        # Exercises
        st.subheader("📚 Exercices pratiques")
        st.markdown("""
        - Créez un tableau de budget personnel  
        - Appliquez des formules simples (SOMME, MOYENNE)  
        - Créez un graphique pour visualiser vos données
        """)

        # Mini quiz
        st.subheader("🧠 Mini-quiz Excel")
        st.markdown("**Question 1:** Quelle fonction permet de calculer la moyenne d’une série de nombres ?")
        st.checkbox("SOMME()", key="ex_q1a")
        st.checkbox("MOYENNE()", key="ex_q1b")
        st.checkbox("MAX()", key="ex_q1c")
        st.success("Réponse correcte : MOYENNE()")

    # --- Microsoft PowerPoint ---
    elif section == "Microsoft PowerPoint":
        st.header("📈 Parcours PowerPoint — Débutant")

        modules = {
            "Module 1 : Découverte de PowerPoint": {
                "1.1 - Ouvrir et naviguer dans PowerPoint": "https://www.coursinfo.fr/powerpoint/ouvrir-powerpoint",
                "1.2 - Création d’une présentation simple": "https://www.coursinfo.fr/powerpoint/presentation-simple"
            },
            "Module 2 : Mise en forme": {
                "2.1 - Thèmes et modèles": "https://www.coursinfo.fr/powerpoint/themes",
                "2.2 - Transitions et animations": "https://www.coursinfo.fr/powerpoint/transitions-animations",
                "2.3 - Insertion de médias": "https://www.coursinfo.fr/powerpoint/insertion-medias"
            }
        }

        for module_name, submodules in modules.items():
            with st.expander(module_name):
                for sub_name, link in submodules.items():
                    st.markdown(f"- [{sub_name}]({link})")

        # Exercises
        st.subheader("📚 Exercices pratiques")
        st.markdown("""
        - Créez une présentation de 5 diapositives  
        - Appliquez un thème et des transitions  
        - Insérez des images et graphiques  
        - Animez du texte ou des objets
        """)

        # Mini quiz
        st.subheader("🧠 Mini-quiz PowerPoint")
        st.markdown("**Question 1:** Quelle fonctionnalité permet d’animer un texte ?")
        st.checkbox("Transitions", key="ppt_q1a")
        st.checkbox("Animations", key="ppt_q1b")
        st.checkbox("Diapositive", key="ppt_q1c")
        st.success("Réponse correcte : Animations")

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
