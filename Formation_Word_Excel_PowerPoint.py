import streamlit as st

def render_training_dashboard():
    st.title("🎓 Formation Bureautique — Word, Excel & PowerPoint (Débutants)")

    st.markdown("""
    Bienvenue dans votre espace de formation continue !  
    Ici, vous trouverez un parcours structuré pour apprendre **Microsoft Word, Excel et PowerPoint**.  
    Tous les modules sont **offline**, avec exercices et mini-quizzes intégrés.
    """)

    # --- Sidebar menu ---
    section = st.sidebar.radio(
        "📘 Choisissez un module de formation :",
        ["Introduction", "Microsoft Word", "Tests & Exercices"]
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

        # --- Offline course structure ---
        modules = {
            "Module 1 : Découverte de Word": [
                {"title": "1.1 - Ouvrir Word et naviguer dans l'interface",
                 "content": """Pour ouvrir Word :
1. Cliquez sur Démarrer > Microsoft Office > Word
2. Créez un nouveau document ou ouvrez un existant
3. Découvrez le ruban, les menus et l'interface principale."""},

                {"title": "1.2 - Création et enregistrement d'un document",
                 "content": """Pour créer un document :
- Fichier > Nouveau
- Choisissez un modèle ou document vierge
Pour enregistrer :
- Fichier > Enregistrer sous > Choisissez l'emplacement et le nom."""}
            ],

            "Module 2 : Mise en forme du texte": [
                {"title": "2.1 - Police et paragraphe",
                 "content": """- Modifier la police et la taille depuis l'onglet Accueil
- Ajuster l'alignement, interligne, et retrait des paragraphes"""},

                {"title": "2.2 - Styles et thèmes",
                 "content": """- Appliquer des styles prédéfinis
- Modifier le thème pour uniformiser le document"""},

                {"title": "2.3 - Listes et tableaux",
                 "content": """- Créer des listes à puces ou numérotées
- Insérer des tableaux et ajuster leurs dimensions"""},
            ],

            "Module 3 : Documents avancés": [
                {"title": "3.1 - En-têtes et pieds de page",
                 "content": """- Insertion > En-tête / Pied de page
- Ajouter numérotation, titre, ou date"""},
                {"title": "3.2 - Table des matières automatique",
                 "content": """- Références > Table des matières
- Sélectionner le style et mettre à jour automatiquement"""},
                {"title": "3.3 - Insertion d'images et graphiques",
                 "content": """- Insertion > Images / SmartArt / Graphiques
- Redimensionner et positionner les éléments"""
                 },
            ]
        }

        # Display modules and submodules
        for module_name, submodules in modules.items():
            st.subheader(module_name)
            for course in submodules:
                exp = st.expander(course["title"])
                exp.write(course["content"])

        # --- Exercises ---
        st.subheader("📚 Exercices pratiques")
        st.markdown("""
        - Rédigez un courrier professionnel avec en-tête et pied de page  
        - Créez une page de garde et appliquez un style uniforme  
        - Insérez une table des matières automatique
        """)

        # --- Mini quiz ---
        st.subheader("🧠 Mini-quiz Word")
        q1 = st.radio("1. Quelle option permet de créer un en-tête dans Word ?", 
                      ("Disposition > Bordures", "Accueil > Styles", "Insertion > En-tête"), key="q1")
        if q1 == "Insertion > En-tête":
            st.success("✅ Correct !")
        else:
            st.error("❌ Réponse incorrecte")

        q2 = st.radio("2. Quelle fonctionnalité permet d'insérer un tableau ?", 
                      ("Insertion > Tableau", "Mise en page > Bordures", "Références > Table des matières"), key="q2")
        if q2 == "Insertion > Tableau":
            st.success("✅ Correct !")
        else:
            st.error("❌ Réponse incorrecte")

    # --- Tests & Exercises ---
    elif section == "Tests & Exercices":
        st.header("🧩 Tests & Exercices finaux")
        st.markdown("""
        Testez vos compétences :
        - **Word :** Créez une lettre professionnelle avec en-tête et pied de page  
        - **Word :** Créez un tableau simple avec bordures et mise en forme  
        """)
        st.success("✅ Comparez vos fichiers avec vos corrections personnelles.")
    
    st.markdown("---")
    st.caption("© 2025 Formation IA & Bureautique — Ressources gratuites pour l'apprentissage continu.")
