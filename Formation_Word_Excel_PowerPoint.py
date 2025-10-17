import streamlit as st
import requests
from bs4 import BeautifulSoup
import json
import os

# --- Constants ---
WORD_BASE_URL = "https://www.coursinfo.fr/word/"
WORD_JSON_FILE = "word_courses.json"
IMAGES_DIR = "word_images"
os.makedirs(IMAGES_DIR, exist_ok=True)

# --- Scraper ---
def get_page_content(url):
    r = requests.get(url)
    r.raise_for_status()
    return r.text

def scrape_word_courses(base_url=WORD_BASE_URL):
    """Scrapes Word modules/submodules from coursinfo.fr and saves locally as JSON"""
    courses = {}
    html = get_page_content(base_url)
    soup = BeautifulSoup(html, "html.parser")

    # Menu links (adjust selector if site structure changes)
    menu_items = soup.select("nav a")
    for item in menu_items:
        module_name = item.get_text(strip=True)
        module_url = item['href']
        if not module_url.startswith("http"):
            module_url = f"https://www.coursinfo.fr{module_url}"

        # Scrape submodules
        module_html = get_page_content(module_url)
        module_soup = BeautifulSoup(module_html, "html.parser")
        submodules = {}
        headers = module_soup.find_all(["h2", "h3"])
        for h in headers:
            sub_name = h.get_text(strip=True)
            content_text = ""
            images = []
            for sib in h.find_next_siblings():
                if sib.name in ["h2", "h3"]:
                    break
                if sib.name == "p":
                    content_text += sib.get_text(strip=True) + "\n"
                if sib.name == "img":
                    img_url = sib['src']
                    if not img_url.startswith("http"):
                        img_url = f"https://www.coursinfo.fr{img_url}"
                    images.append(img_url)
            if content_text.strip() or images:
                submodules[sub_name] = {
                    "text": content_text.strip(),
                    "images": images,
                    "url": module_url
                }
        if submodules:
            courses[module_name] = submodules

    # Save JSON
    with open(WORD_JSON_FILE, "w", encoding="utf-8") as f:
        json.dump(courses, f, ensure_ascii=False, indent=2)

# --- Load JSON ---
def load_word_json():
    if not os.path.exists(WORD_JSON_FILE):
        st.info("⚡ Scraping Word courses from coursinfo.fr, please wait...")
        scrape_word_courses()
    with open(WORD_JSON_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

WORD_JSON = load_word_json()

# --- Streamlit LMS ---
def render_training_dashboard():
    st.title("🎓 Formation Bureautique — Word, Excel & PowerPoint (Débutants)")

    st.markdown("""
    Bienvenue dans votre espace de formation continue !  
    Ici, vous trouverez un parcours structuré pour apprendre **Microsoft Word, Excel et PowerPoint**.  
    Les ressources sont gratuites et organisées par modules et sous-modules.
    """)

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
        for module_name, submodules in WORD_JSON.items():
            module_expander = st.expander(module_name)
            with module_expander:
                for sub_name, content in submodules.items():
                    st.subheader(sub_name)
                    st.markdown(content["text"])
                    for img_url in content["images"]:
                        st.image(img_url, use_column_width=True)
                    st.markdown(f"[Voir sur coursinfo.fr]({content['url']})")
                    st.markdown("---")

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
        st.markdown("Contenu Excel sera ajouté ici (modules, exercices et quiz).")

    # --- Microsoft PowerPoint ---
    elif section == "Microsoft PowerPoint":
        st.header("📈 Parcours PowerPoint — Débutant")
        st.markdown("Contenu PowerPoint sera ajouté ici (modules, exercices et quiz).")

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
