import streamlit as st
import requests
from bs4 import BeautifulSoup
import os
import json
from PIL import Image

# --- Constants ---
WORD_URL = "https://www.coursinfo.fr/word/"
JSON_FILE = "word_courses.json"
IMG_FOLDER = "word_images"

os.makedirs(IMG_FOLDER, exist_ok=True)

# --- Scraper function ---
def scrape_word(force=False):
    """Scrape Word courses from coursinfo.fr and save locally"""
    if os.path.exists(JSON_FILE) and not force:
        return  # Use cached data

    r = requests.get(WORD_URL)
    soup = BeautifulSoup(r.content, "html.parser")
    courses = []

    # Adjust selector for course links
    for link in soup.select("a.cours-link"):  
        title = link.get_text(strip=True)
        href = link.get("href")
        if not href.startswith("http"):
            href = "https://www.coursinfo.fr" + href
        course_data = scrape_course_page(href)
        courses.append({"title": title, "url": href, **course_data})

    with open(JSON_FILE, "w", encoding="utf-8") as f:
        json.dump(courses, f, ensure_ascii=False, indent=4)

def scrape_course_page(url):
    """Scrape a single course page"""
    try:
        r = requests.get(url)
        soup = BeautifulSoup(r.content, "html.parser")

        # Extract main text
        content_div = soup.find("div", {"class": "cours-content"})
        text = content_div.get_text(separator="\n").strip() if content_div else ""

        # Extract images
        images = []
        for img in soup.find_all("img"):
            src = img.get("src")
            if src:
                img_data = requests.get(src).content
                img_name = os.path.join(IMG_FOLDER, os.path.basename(src))
                with open(img_name, "wb") as f:
                    f.write(img_data)
                images.append(img_name)
        return {"text": text, "images": images}
    except Exception as e:
        return {"text": f"⚠️ Impossible de charger la page : {e}", "images": []}

# --- Word submenu ---
def render_word_submenu():
    st.header("📝 Parcours Word — Débutant")
    try:
        if not os.path.exists(JSON_FILE):
            st.info("⚡ Premier lancement : récupération du contenu Word depuis le site coursinfo.fr ...")
            scrape_word()
        with open(JSON_FILE, "r", encoding="utf-8") as f:
            courses_list = json.load(f)
    except Exception as e:
        st.error(f"⚠️ Erreur lors du chargement de la formation : {e}")
        return

    # Iterate over the list of courses safely
    for course in courses_list:
        if isinstance(course, dict):
            title = course.get("title", "Cours sans titre")
            text = course.get("text", "")
            images = course.get("images", [])

            with st.expander(title):
                st.text(text)
                for img_path in images:
                    try:
                        img = Image.open(img_path)
                        st.image(img)
                    except:
                        pass

    # Example exercises and mini-quiz
    st.subheader("📚 Exercices pratiques")
    st.markdown("""
    - Rédigez un courrier professionnel avec en-tête et pied de page  
    - Créez une page de garde et appliquez un style uniforme  
    - Insérez une table des matières automatique
    """)

    st.subheader("🧠 Mini-quiz Word")
    st.markdown("**Question 1:** Quelle option permet de créer un en-tête dans Word ?")
    st.checkbox("Insertion > En-tête", key="q1a")
    st.checkbox("Disposition > Bordures", key="q1b")
    st.checkbox("Accueil > Styles", key="q1c")
    st.success("Réponse correcte : Insertion > En-tête")

# --- Excel & PowerPoint submenus (static for now) ---
def render_excel_submenu():
    st.header("📊 Parcours Excel — Débutant")
    st.markdown("**Exercices et quiz Excel ici (statique ou futur scraping)**")

def render_powerpoint_submenu():
    st.header("📈 Parcours PowerPoint — Débutant")
    st.markdown("**Exercices et quiz PowerPoint ici (statique ou futur scraping)**")

# --- Main Training Dashboard ---
def render_training_dashboard():
    st.title("🎓 Formation Bureautique — Word, Excel & PowerPoint (Débutants)")
    st.markdown("""
    Bienvenue dans votre espace de formation continue !  
    Ici, vous trouverez un parcours structuré pour apprendre **Microsoft Word, Excel et PowerPoint**.  
    Les ressources sont organisées par modules et sous-modules.
    """)

    section = st.sidebar.radio(
        "📘 Choisissez un module de formation :",
        ["Introduction", "Microsoft Word", "Microsoft Excel", "Microsoft PowerPoint"]
    )

    if section == "Introduction":
        st.subheader("🧭 Objectifs de la formation")
        st.markdown("""
        - Acquérir les bases de Word, Excel et PowerPoint  
        - Créer, formater et présenter des documents professionnels  
        - Maîtriser les outils bureautiques pour le travail en entreprise
        """)
        st.info("💡 Avancez module par module et testez vos connaissances à la fin de chaque partie.")

    elif section == "Microsoft Word":
        render_word_submenu()
    elif section == "Microsoft Excel":
        render_excel_submenu()
    elif section == "Microsoft PowerPoint":
        render_powerpoint_submenu()

    st.markdown("---")
    st.caption("© 2025 Formation IA & Bureautique — Ressources gratuites pour l'apprentissage continu.")

# --- Run ---
if __name__ == "__main__":
    render_training_dashboard()
