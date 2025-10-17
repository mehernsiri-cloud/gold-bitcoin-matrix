import streamlit as st
import json
import os

def render_training_dashboard():
    st.title("🎓 Formation Bureautique — Word, Excel & PowerPoint (Débutants)")

    st.markdown("""
    Bienvenue dans votre espace de formation continue !  
    Ici, vous trouverez un parcours structuré pour apprendre **Microsoft Word, Excel et PowerPoint**.  
    Les ressources sont gratuites, accessibles hors ligne via JSON et organisées par modules et sous-modules.
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

        # Load scraped JSON
        json_path = os.path.join("data", "word_courses.json")
        if not os.path.exists(json_path):
            st.error("⚠️ Une erreur est survenue : le fichier word_courses.json est manquant.")
            return

        with open(json_path, "r", encoding="utf-8") as f:
            word_courses = json.load(f)

        # Organize by modules
        modules_dict = {}
        for course in word_courses:
            # Assuming the module is first part of title "Module X : ..."
            if ":" in course["title"]:
                module_name = course["title"].split(":")[0].strip()
            else:
                module_name = "Autres cours"
            modules_dict.setdefault(module_name, []).append(course)

        # Display modules and courses
        for module_name, courses in modules_dict.items():
            with st.expander(module_name):
                for course in courses:
                    st.subheader(course["title"])
                    # Display text content
                    for paragraph in course["text"]:
                        st.write(paragraph)
                    # Display images
                    for img_url in course["images"]:
                        st.image(img_url)
                    # Link to original course page
                    st.markdown(f"[Voir sur coursinfo.fr]({course['url']})")
                    st.markdown("---")

        # --- Exercises ---
        st.subheader("📚 Exercices pratiques Word")
        st.markdown("""
        - Rédigez un courrier professionnel avec en-tête et pied de page  
        - Créez une page de garde et appliquez un style uniforme  
        - Insérez une table des matières automatique  
        """)

        # --- Mini-quiz ---
        st.subheader("🧠 Mini-quiz Word")
        st.markdown("**Question 1:** Quelle option permet de créer un en-tête dans Word ?")
        st.checkbox("Insertion > En-tête", key="q1a")
        st.checkbox("Disposition > Bordures", key="q1b")
        st.checkbox("Accueil > Styles", key="q1c")
        st.success("Réponse correcte : Insertion > En-tête")

    # --- Microsoft Excel ---
    elif section == "Microsoft Excel":
        st.header("📊 Parcours Excel — Débutant")
        st.markdown("Module en préparation…")

    # --- Microsoft PowerPoint ---
    elif section == "Microsoft PowerPoint":
        st.header("📈 Parcours PowerPoint — Débutant")
        st.markdown("Module en préparation…")

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
