import streamlit as st

def render_training_dashboard():
    st.title("🎓 Formation Word, Excel & PowerPoint — Débutants")

    st.markdown("""
    Bienvenue dans votre espace de formation continue !  
    Ici, vous trouverez des **cours gratuits, structurés et progressifs** pour apprendre à utiliser **Microsoft Word**, **Excel** et **PowerPoint**.  
    Les ressources sont issues de plateformes publiques et toujours accessibles en ligne.
    """)

    # --- Menu latéral de navigation ---
    section = st.sidebar.radio(
        "📘 Choisissez un module de formation :",
        ["Introduction", "Microsoft Word", "Microsoft Excel", "Microsoft PowerPoint", "Tests & Exercices"]
    )

    # --- Introduction ---
    if section == "Introduction":
        st.subheader("🧭 Objectifs de la formation")
        st.markdown("""
        - Acquérir les bases de Word, Excel et PowerPoint.  
        - Apprendre à créer, formater et présenter des documents professionnels.  
        - Maîtriser les outils de bureautique pour le travail en entreprise.  
        """)
        st.info("💡 Conseil : Avancez module par module, à votre rythme, et testez vos connaissances à la fin de chaque partie.")

    # --- Microsoft Word ---
    elif section == "Microsoft Word":
        st.header("📝 Formation Microsoft Word — Niveau Débutant")

        with st.expander("📺 Cours Vidéos (YouTube)"):
            st.video("https://www.youtube.com/watch?v=3bU9W1E5YqY")  # base Word
            st.video("https://www.youtube.com/watch?v=GdJZBjaCxMw")  # mise en forme

        with st.expander("📚 Exercices Pratiques"):
            st.markdown("""
            - Rédiger un courrier professionnel  
            - Créer une page de garde  
            - Utiliser les styles et la mise en page  
            - Insérer une table des matières automatique  
            """)
            st.checkbox("Exercice 1 terminé ?")
            st.checkbox("Exercice 2 terminé ?")
            st.checkbox("Exercice 3 terminé ?")
            st.checkbox("Exercice 4 terminé ?")

        with st.expander("🧠 Mini Quiz"):
            st.markdown("""
            1. Comment insérer une table des matières automatique dans Word ?  
            2. Quels sont les raccourcis pour mettre en gras, italique et souligné ?  
            3. Comment créer un en-tête et un pied de page ?  
            """)
            st.radio("Question 1", ["Option A", "Option B", "Option C"])
            st.radio("Question 2", ["Option A", "Option B", "Option C"])
            st.radio("Question 3", ["Option A", "Option B", "Option C"])

    # --- Microsoft Excel ---
    elif section == "Microsoft Excel":
        st.header("📊 Formation Microsoft Excel — Niveau Débutant")

        with st.expander("📺 Cours Vidéos (YouTube)"):
            st.video("https://www.youtube.com/watch?v=9JpNY-XAseg")  # base Excel
            st.video("https://www.youtube.com/watch?v=QbmM1U4kRrw")  # formules simples

        with st.expander("📚 Exercices Pratiques"):
            st.markdown("""
            - Créer un tableau de budget personnel  
            - Appliquer des formules de base (SOMME, MOYENNE, MAX, MIN)  
            - Utiliser les filtres et tris  
            - Créer un graphique simple  
            """)
            st.checkbox("Exercice 1 terminé ?")
            st.checkbox("Exercice 2 terminé ?")
            st.checkbox("Exercice 3 terminé ?")
            st.checkbox("Exercice 4 terminé ?")

        with st.expander("🧠 Mini Quiz"):
            st.markdown("""
            1. Comment créer un graphique à partir d'un tableau ?  
            2. Quels sont les raccourcis pour copier, coller et remplir automatiquement ?  
            3. Comment trier et filtrer des données ?  
            """)
            st.radio("Question 1", ["Option A", "Option B", "Option C"])
            st.radio("Question 2", ["Option A", "Option B", "Option C"])
            st.radio("Question 3", ["Option A", "Option B", "Option C"])

    # --- Microsoft PowerPoint ---
    elif section == "Microsoft PowerPoint":
        st.header("📈 Formation Microsoft PowerPoint — Niveau Débutant")

        with st.expander("📺 Cours Vidéos (YouTube)"):
            st.video("https://www.youtube.com/watch?v=GdpXHgycvU0")  # base PowerPoint
            st.video("https://www.youtube.com/watch?v=RGvhDyc8jA4")  # animations

        with st.expander("📚 Exercices Pratiques"):
            st.markdown("""
            - Créer une présentation de 5 diapositives  
            - Appliquer un thème et des transitions  
            - Insérer des images et graphiques  
            - Animer des objets et du texte  
            """)
            st.checkbox("Exercice 1 terminé ?")
            st.checkbox("Exercice 2 terminé ?")
            st.checkbox("Exercice 3 terminé ?")
            st.checkbox("Exercice 4 terminé ?")

        with st.expander("🧠 Mini Quiz"):
            st.markdown("""
            1. Comment ajouter une transition entre deux diapositives ?  
            2. Comment insérer un graphique dans PowerPoint ?  
            3. Comment animer un texte ou un objet ?  
            """)
            st.radio("Question 1", ["Option A", "Option B", "Option C"])
            st.radio("Question 2", ["Option A", "Option B", "Option C"])
            st.radio("Question 3", ["Option A", "Option B", "Option C"])

    # --- Tests & Exercices ---
    elif section == "Tests & Exercices":
        st.header("🧩 Tests & Exercices Finaux")
        st.markdown("""
        Testez vos compétences à travers ces mini-projets :
        - **Word :** Créez une lettre professionnelle avec en-tête et pied de page.  
        - **Excel :** Créez un tableau de suivi de dépenses avec un graphique.  
        - **PowerPoint :** Présentez un sujet de votre choix en 5 slides claires.  
        """)
        st.success("✅ Astuce : Enregistrez vos fichiers et comparez-les avec les modèles disponibles en ligne.")

    st.markdown("---")
    st.caption("© 2025 Formation IA & Bureautique — Ressources gratuites pour l'apprentissage continu.")
