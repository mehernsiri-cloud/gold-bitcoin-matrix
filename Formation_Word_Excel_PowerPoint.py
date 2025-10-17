import streamlit as st
import json
import os

# --- Utility function to load Word courses JSON ---
def load_word_courses():
    try:
        with open("word_courses.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("⚠️ Le fichier word_courses.json est manquant.")
        return []
    except json.JSONDecodeError:
        st.error("⚠️ Le fichier JSON est invalide.")
        return []

# --- Word submenu rendering ---
def render_word_submenu():
    st.header("📝 Parcours Word — Débutant")
    
    courses = load_word_courses()
    if not courses:
        st.warning("Aucun cours disponible pour Word pour le moment.")
        return

    # Iterate over the list of courses
    for course in courses:
        title = course.get("title", "Cours sans titre")
        url = course.get("url", "")
        text = course.get("text", [])
        images = course.get("images", [])
        exercises = course.get("exercises", [])
        quiz = course.get("quiz", [])

        with st.expander(title):
            if url:
                st.markdown(f"**Lien du cours :** [{url}]({url})")
            
            # Display text content
            for paragraph in text:
                st.write(paragraph)
            
            # Display images
            for img_url in images:
                st.image(img_url)
            
            # Exercises
            if exercises:
                st.subheader("📚 Exercices")
                for ex in exercises:
                    st.markdown(f"- {ex}")
            
            # Mini-quiz
            if quiz:
                st.subheader("🧠 Mini-quiz")
                for q in quiz:
                    st.markdown(f"- {q}")

# --- Excel submenu ---
def render_excel_submenu():
    st.header("📊 Parcours Excel — Débutant")
    st.markdown("Contenu Excel ici... (à compléter avec vos cours/exercices/quiz)")

# --- PowerPoint submenu ---
def render_powerpoint_submenu():
    st.header("📈 Parcours PowerPoint — Débutant")
    st.markdown("Contenu PowerPoint ici... (à compléter avec vos cours/exercices/quiz)")

# --- Main training dashboard ---
def render_training_dashboard():
    st.title("🎓 Formation Bureautique — Word, Excel & PowerPoint (Débutants)")

    st.markdown("""
    Bienvenue dans votre espace de formation continue !  
    Apprenez **Microsoft Word, Excel et PowerPoint** à travers des modules structurés et interactifs.
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
        render_word_submenu()

    # --- Microsoft Excel ---
    elif section == "Microsoft Excel":
        render_excel_submenu()

    # --- Microsoft PowerPoint ---
    elif section == "Microsoft PowerPoint":
        render_powerpoint_submenu()

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
