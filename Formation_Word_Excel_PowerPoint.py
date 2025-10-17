import streamlit as st
import json

def load_word_courses():
    try:
        with open("word_courses.json", "r", encoding="utf-8") as f:
            data = json.load(f)
            # Ensure the top-level JSON is a list
            if isinstance(data, dict):
                return [data]
            elif isinstance(data, list):
                return data
            else:
                st.error("⚠️ Le format du fichier JSON n'est pas correct.")
                return []
    except FileNotFoundError:
        st.error("⚠️ Le fichier word_courses.json est manquant.")
        return []
    except json.JSONDecodeError:
        st.error("⚠️ Le fichier JSON est invalide.")
        return []

def render_training_dashboard():

    st.title("🎓 Formation Bureautique — Word, Excel & PowerPoint (Débutants)")
    st.markdown("""
    Bienvenue dans votre espace de formation continue !  
    Ici, vous trouverez un parcours structuré pour apprendre **Microsoft Word, Excel et PowerPoint**.  
    Les ressources sont gratuites, accessibles en ligne, et organisées par modules et sous-modules.
    """)

    # Sidebar menu
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
        courses = load_word_courses()

        if courses:
            for course in courses:
                title = course.get("title", "Module sans titre")
                text_content = course.get("text", [])
                images = course.get("images", [])
                exercises = course.get("exercises", [])
                quiz = course.get("quiz", [])
                url = course.get("url", None)

                with st.expander(title):
                    # Display course text
                    for paragraph in text_content:
                        st.markdown(paragraph)

                    # Display images
                    for img_url in images:
                        st.image(img_url, use_column_width=True)

                    # Link to external page if exists
                    if url:
                        st.markdown(f"[Voir le cours complet sur le site]({url})")

                    # Exercises
                    if exercises:
                        st.subheader("📚 Exercices pratiques")
                        for ex in exercises:
                            st.markdown(f"- {ex}")

                    # Mini quiz
                    if quiz:
                        st.subheader("🧠 Mini-quiz")
                        for i, q in enumerate(quiz, 1):
                            st.markdown(f"**Question {i}:** {q}")
                            st.checkbox("Réponse A", key=f"{title}_q{i}_a")
                            st.checkbox("Réponse B", key=f"{title}_q{i}_b")
                            st.checkbox("Réponse C", key=f"{title}_q{i}_c")
                            st.success("Réponse correcte à vérifier dans vos notes ou corrigé fourni.")

        else:
            st.warning("⚠️ Aucun cours Word disponible. Assurez-vous que 'word_courses.json' existe et est valide.")

    # --- Microsoft Excel ---
    elif section == "Microsoft Excel":
        st.header("📊 Parcours Excel — Débutant")
        st.markdown("Contenu Excel à compléter... (vous pouvez réutiliser le même format que Word avec un JSON séparé)")

    # --- Microsoft PowerPoint ---
    elif section == "Microsoft PowerPoint":
        st.header("📈 Parcours PowerPoint — Débutant")
        st.markdown("Contenu PowerPoint à compléter... (vous pouvez réutiliser le même format que Word avec un JSON séparé)")

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
