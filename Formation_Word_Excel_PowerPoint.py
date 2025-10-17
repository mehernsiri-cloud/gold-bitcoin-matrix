import streamlit as st
import requests
from bs4 import BeautifulSoup
import json

# --- Scraping function for Word content ---
def scrape_word_lessons():
    base_url = "https://www.coursinfo.fr/word/"
    lessons_pages = [
        "ouvrir-word",
        "interface-rubans",
        "creer-document",
        "polices-paragraphes",
        "styles-themes",
        "listes-tableaux",
        "en-tetes-pieds",
        "table-matieres",
        "images-graphiques"
    ]

    modules = {
        "Module 1 : Découverte de Word": ["ouvrir-word", "interface-rubans", "creer-document"],
        "Module 2 : Mise en forme": ["polices-paragraphes", "styles-themes", "listes-tableaux"],
        "Module 3 : Documents avancés": ["en-tetes-pieds", "table-matieres", "images-graphiques"]
    }

    word_data = {}

    for module_name, pages in modules.items():
        word_data[module_name] = {}
        for page_slug in pages:
            url = f"{base_url}{page_slug}"
            try:
                response = requests.get(url)
                if response.status_code != 200:
                    word_data[module_name][page_slug] = {
                        "title": page_slug,
                        "text": "Contenu indisponible",
                        "images": []
                    }
                    continue

                soup = BeautifulSoup(response.text, "html.parser")
                content_div = soup.find("div", class_="entry-content")
                text_content = content_div.get_text(separator="\n", strip=True) if content_div else "Contenu indisponible"

                images = []
                for img_tag in content_div.find_all("img") if content_div else []:
                    img_url = img_tag.get("src")
                    if img_url:
                        images.append(img_url)

                word_data[module_name][page_slug] = {
                    "title": page_slug.replace("-", " ").capitalize(),
                    "text": text_content,
                    "images": images,
                    "url": url
                }

            except Exception as e:
                word_data[module_name][page_slug] = {
                    "title": page_slug,
                    "text": f"Erreur lors du chargement: {e}",
                    "images": [],
                    "url": url
                }

    return word_data

# --- Generate JSON once ---
try:
    WORD_JSON = scrape_word_lessons()
except:
    WORD_JSON = {}

# --- Render LMS Dashboard ---
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
        if not WORD_JSON:
            st.error("⚠️ Une erreur est survenue lors du chargement de la formation.")
            return

        for module_name, submodules in WORD_JSON.items():
            with st.expander(module_name):
                for sub_slug, content in submodules.items():
                    course_exp = st.expander(content["title"])
                    with course_exp:
                        st.markdown(content["text"])
                        for img_url in content["images"]:
                            st.image(img_url, use_column_width=True)
                        st.markdown(f"[Voir sur coursinfo.fr]({content['url']})")

        # Exercises
        st.subheader("📚 Exercices pratiques")
        st.markdown("""
        - Rédigez un courrier professionnel avec en-tête et pied de page  
        - Créez une page de garde et appliquez un style uniforme  
        - Insérez une table des matières automatique
        """)

        # Mini-quiz
        st.subheader("🧠 Mini-quiz Word")
        st.markdown("**Question 1:** Quelle option permet de créer un en-tête dans Word ?")
        st.checkbox("Insertion > En-tête", key="q1a")
        st.checkbox("Disposition > Bordures", key="q1b")
        st.checkbox("Accueil > Styles", key="q1c")
        st.success("Réponse correcte : Insertion > En-tête")

    # --- Microsoft Excel ---
    elif section == "Microsoft Excel":
        st.header("📊 Parcours Excel — Débutant")
        st.info("Contenu Excel à venir...")

    # --- Microsoft PowerPoint ---
    elif section == "Microsoft PowerPoint":
        st.header("📈 Parcours PowerPoint — Débutant")
        st.info("Contenu PowerPoint à venir...")

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
    st.caption("© 2025 Formation IA & Bureautique — Contenu intégré depuis coursinfo.fr")

