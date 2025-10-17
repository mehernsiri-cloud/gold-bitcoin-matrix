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

        st.subheader("📺 Cours Vidéos (YouTube)")
        st.video("https://www.youtube.com/watch?v=3bU9W1E5YqY")  # base Word
        st.video("https://www.youtube.com/watch?v=GdJZBjaCxMw")  # mise en forme

        st.subheader("📚 Exercices Pratiques")
        st.markdown("""
        - Rédiger un courrier professionnel  
        - Créer une page de garde  
        - Utiliser les styles et la mise en page  
        - Insérer une table des matières automatique  
        """)

        st.subheader("🧠 Quiz en ligne")
        st.markdown("[Quiz Microsoft Word – Débutant](https://forms.office.com/)")

    # --- Microsoft Excel ---
    elif section == "Microsoft Excel":
        st.header("📊 Formation Microsoft Excel — Niveau Débutant")

        st.subheader("📺 Cours Vidéos (YouTube)")
        st.video("https://www.youtube.com/watch?v=9JpNY-XAseg")  # base Excel
        st.video("https://www.youtube.com/watch?v=QbmM1U4kRrw")  # formules simples

        st.subheader("📚 Exercices Pratiques")
        st.markdown("""
        - Créer un tableau de budget personnel  
        - Appliquer des formules de base (SOMME, MOYENNE, MAX, MIN)  
        - Utiliser les filtres et tris  
        - Créer un graphique simple  
        """)

        st.subheader("🧠 Quiz en ligne")
        st.markdown("[Quiz Microsoft Excel – Débutant](https://forms.office.com/)")

    # --- Microsoft PowerPoint ---
    elif section == "Microsoft PowerPoint":
        st.header("📈 Formation Microsoft PowerPoint — Niveau Débutant")

        st.subheader("📺 Cours Vidéos (YouTube)")
        st.video("https://www.youtube.com/watch?v=GdpXHgycvU0")  # base PowerPoint
        st.video("https://www.youtube.com/watch?v=RGvhDyc8jA4")  # animations

        st.subheader("📚 Exercices Pratiques")
        st.markdown("""
        - Créer une présentation de 5 diapositives  
        - Appliquer un thème et des transitions  
        - Insérer des images et graphiques  
        - Animer des objets et du texte  
        """)

        st.subheader("🧠 Quiz en ligne")
        st.markdown("[Quiz PowerPoint – Débutant](https://forms.office.com/)")

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
