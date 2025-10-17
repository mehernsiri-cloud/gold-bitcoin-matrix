import streamlit as st

def render_training_dashboard():
    st.title("🎓 Formation Bureautique – Word, Excel & PowerPoint")
    st.markdown(
        """
        **Bienvenue dans votre espace de formation gratuite à la bureautique.**  
        Apprenez à maîtriser Microsoft **Word**, **Excel** et **PowerPoint** à votre rythme grâce à des vidéos, exercices pratiques et quiz.  
        *(Tout le contenu est gratuit et issu de ressources publiques fiables en français.)*
        """
    )

    # Sidebar menu
    menu = st.sidebar.radio(
        "📘 Sélectionnez un module de formation :",
        ["Microsoft Word – Débutant", "Microsoft Excel – Débutant à Intermédiaire", "Microsoft PowerPoint – Débutant"]
    )

    # --- WORD ---
    if menu == "Microsoft Word – Débutant":
        st.header("📝 Formation Microsoft Word – Niveau Débutant")
        st.markdown("Apprenez les bases du traitement de texte et la mise en forme professionnelle de vos documents.")

        st.subheader("🎥 Cours vidéo complet")
        st.video("https://www.youtube.com/watch?v=zbZ4GYt8i_0")  # vidéo de formation Word débutant
        st.video("https://www.youtube.com/watch?v=jbQDbhg4qek")  # les bases Word en 35 min

        st.subheader("📂 Exercices & supports")
        st.markdown("""
        - 📘 [Cours Word niveau 1 – CoursInfo.fr](https://www.coursinfo.fr/word/les-fonctions-de-base-word-niveau-1/)
        - 📗 [Télécharger exercices Word débutant (WordPratique.com)](https://www.word-pratique.com/)
        - 📄 [Formation Microsoft officielle (niveau base)](https://support.microsoft.com/fr-fr/training)
        """)

        st.subheader("🧩 Quiz & Auto-évaluation")
        st.markdown("""
        👉 Testez vos connaissances sur [Cursa.app – Word Débutant](https://cursa.app/cours-gratuits-productivite-bureautique-online)
        """)

    # --- EXCEL ---
    elif menu == "Microsoft Excel – Débutant à Intermédiaire":
        st.header("📊 Formation Microsoft Excel – Débutant à Intermédiaire")
        st.markdown("Découvrez comment manipuler des données, créer des formules et des graphiques, et automatiser vos tâches Excel.")

        st.subheader("🎥 Cours vidéo complet")
        st.video("https://www.youtube.com/watch?v=aZ-SY0ORaoA")  # 2h30 formation Excel débutant
        st.video("https://www.youtube.com/watch?v=qzVnA_rfjvQ")  # Formules Excel

        st.subheader("📂 Exercices & supports")
        st.markdown("""
        - 📘 [Cours Excel – Excel-Pratique.com](https://excel-pratique.com/fr/formation-excel)
        - 📗 [Formation Excel – Excel-Formation.com](https://www.excel-formation.com/post/formation-excel-d%C3%A9butant)
        - 📄 [Télécharger fichiers d’exercices gratuits (Excel-Pratique.com)](https://excel-pratique.com/fr/formation-excel/exercices)
        """)

        st.subheader("🧩 Quiz & Auto-évaluation")
        st.markdown("""
        👉 Essayez les quiz gratuits sur [Cursa.app – Excel Formation](https://cursa.app/cours-gratuits-productivite-bureautique-online)
        """)

        st.subheader("💡 Astuce du jour")
        st.info("💡 Utilisez **Ctrl + ;** pour insérer la date du jour automatiquement dans une cellule Excel.")

    # --- POWERPOINT ---
    elif menu == "Microsoft PowerPoint – Débutant":
        st.header("📽️ Formation Microsoft PowerPoint – Débutant")
        st.markdown("Apprenez à concevoir des présentations efficaces et professionnelles.")

        st.subheader("🎥 Cours vidéo complet")
        st.video("https://www.youtube.com/watch?v=HY0P1b_hJqg")  # Formation PowerPoint débutant
        st.video("https://www.youtube.com/watch?v=6S-3Th_rpGc")  # Créer un diaporama

        st.subheader("📂 Exercices & supports")
        st.markdown("""
        - 📘 [Cours PowerPoint – OfficePPT.com](https://www.officepourtous.com/formation-powerpoint-debutant/)
        - 📗 [Formation Microsoft officielle – PowerPoint](https://support.microsoft.com/fr-fr/training)
        """)

        st.subheader("🧩 Quiz & Auto-évaluation")
        st.markdown("""
        👉 Quiz en ligne sur [Cursa.app – PowerPoint](https://cursa.app/cours-gratuits-productivite-bureautique-online)
        """)

        st.success("🎯 Objectif : savoir créer une présentation claire, animée et cohérente pour vos réunions ou vos projets.")

    # Footer
    st.markdown("---")
    st.caption("📚 Ressources publiques – Formation bureautique gratuite (Microsoft, CoursInfo, Excel-Pratique, YouTube)")

