import streamlit as st
import os
import json
import requests
from bs4 import BeautifulSoup
from pathlib import Path
import base64

WORD_JSON_PATH = "word_courses.json"

def scrape_word_courses():
    """Scrape Word lessons from coursinfo.fr/word/ and save to JSON."""
    base_url = "https://www.coursinfo.fr/word/"
    response = requests.get(base_url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    # Find all course links on main Word page
    lessons = {}
    for a in soup.select("a[href*='/word/']"):
        href = a.get("href")
        title = a.get_text(strip=True)
        if href and title and not href.endswith("#"):
            lessons[title] = base_url.rstrip("/") + "/" + href.split("/word/")[-1]

    # For each lesson, fetch content and images
    courses = []
    for lesson_title, lesson_url in lessons.items():
        lesson_resp = requests.get(lesson_url)
        lesson_resp.raise_for_status()
        lesson_soup = BeautifulSoup(lesson_resp.text, "html.parser")
        # Extract text
        paragraphs = [p.get_text(strip=True) for p in lesson_soup.select("p")]
        # Extract images
        images = [img["src"] for img in lesson_soup.select("img") if img.get("src")]
        courses.append({
            "title": lesson_title,
            "url": lesson_url,
            "text": paragraphs,
            "images": images,
            "exercises": ["Exercice 1: ...", "Exercice 2: ..."],
            "quiz": ["Question 1: ...", "Question 2: ..."]
        })

    # Save JSON
    with open(WORD_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(courses, f, ensure_ascii=False, indent=2)

def load_word_courses():
    if not os.path.exists(WORD_JSON_PATH):
        scrape_word_courses()
    with open(WORD_JSON_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def render_word_submenu():
    st.header("📝 Parcours Word — Débutant")
    courses = load_word_courses()

    for course in courses:
        with st.expander(course["title"]):
            st.markdown(f"**URL du cours :** [{course['url']}]({course['url']})")
            # Display text
            for paragraph in course["text"]:
                st.write(paragraph)
            # Display images
            for img_url in course["images"]:
                st.image(img_url)
            # Exercises
            st.subheader("📚 Exercices")
            for ex in course["exercises"]:
                st.markdown(f"- {ex}")
            # Mini quiz
            st.subheader("🧠 Quiz")
            for q in course["quiz"]:
                st.markdown(f"- {q}")

def render_training_dashboard():
    st.title("🎓 Formation Bureautique — Word, Excel & PowerPoint (Débutants)")
    st.markdown("Bienvenue dans votre espace de formation continue !")

    section = st.sidebar.radio(
        "📘 Choisissez un module de formation :",
        ["Introduction", "Microsoft Word", "Microsoft Excel", "Microsoft PowerPoint", "Tests & Exercices"]
    )

    if section == "Introduction":
        st.subheader("🧭 Objectifs de la formation")
        st.markdown("""
        - Acquérir les bases de Word, Excel et PowerPoint  
        - Créer, formater et présenter des documents professionnels  
        - Maîtriser les outils bureautiques pour le travail en entreprise
        """)
    elif section == "Microsoft Word":
        render_word_submenu()
    else:
        st.info("Sections Excel / PowerPoint / Tests & Exercices restent inchangées pour l'instant.")

