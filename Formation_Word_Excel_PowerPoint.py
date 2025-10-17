import streamlit as st

def render_training_dashboard():
    st.title("Formation Word, Excel & PowerPoint")
    st.write("**Bienvenue dans l’espace de formation bureautique (niveau débutant)** 💻")

    st.markdown("""
    ### 📘 Contenu disponible :
    - **Cours Microsoft Word** (vidéos, exercices, tests)
    - **Cours Microsoft Excel** (vidéos, exercices, tests)
    - **Cours Microsoft PowerPoint** (vidéos, exercices, tests)
    """)

    choice = st.sidebar.selectbox(
        "Choisissez un module", 
        ["Microsoft Word", "Microsoft Excel", "Microsoft PowerPoint"]
    )

    if choice == "Microsoft Word":
        st.header("Formation Microsoft Word - Débutant")
        st.video("https://www.youtube.com/watch?v=9Q3BO-44sVs")  # example video
        st.markdown("""
        #### Exercices :
        - Créer un document Word avec mise en forme (titres, listes, styles)
        - Insérer un tableau et une image
        """)
        st.markdown("""
        #### Test :
        - QCM : notions de base, mise en page, insertion
        """)

    elif choice == "Microsoft Excel":
        st.header("Formation Microsoft Excel - Débutant")
        st.video("https://www.youtube.com/watch?v=0W4F8b1mA9Y")
        st.markdown("""
        #### Exercices :
        - Créer un tableau de suivi de dépenses
        - Utiliser les formules SOMME, MOYENNE, MAX
        """)
        st.markdown("""
        #### Test :
        - QCM : formules, tri, filtres, graphiques simples
        """)

    else:
        st.header("Formation Microsoft PowerPoint - Débutant")
        st.video("https://www.youtube.com/watch?v=JhY7CLtN1sA")
        st.markdown("""
        #### Exercices :
        - Créer une présentation de 5 diapositives
        - Ajouter des transitions et animations
        """)
        st.markdown("""
        #### Test :
        - QCM : diapos, transitions, thème, insertion d’images
        """)
