import streamlit as st
from io import BytesIO
import base64

def render_training_dashboard():
    st.title("📚 Formation Word & Excel - Débutants")
    st.markdown("Bienvenue dans votre espace de formation interactif. Suivez les cours, faites les exercices et testez vos connaissances !")

    # Tabs for LMS-like structure
    tabs = st.tabs(["Cours", "Exercices Pratiques", "Mini-Quiz"])

    # --- Cours Tab ---
    with tabs[0]:
        st.header("📝 Cours")
        
        # WORD Lesson
        st.subheader("Microsoft Word - Débutant")
        st.markdown("""
**Objectifs :**
- Apprendre à créer, formater et sauvegarder un document Word
- Maîtriser les bases de la mise en page et du texte
- Utiliser les styles et listes

**Contenu du cours :**
1. Interface Word et ruban
2. Création et sauvegarde d’un document
3. Mise en forme du texte : gras, italique, souligné
4. Paragraphes, alignement et interligne
5. Listes à puces et numérotation
6. Insertion d’images et tableaux
7. En-têtes, pieds de page et pagination
""")
        # Option to download PDF
        word_pdf_bytes = b"%PDF-1.4\n%PDF content for Word beginner course (replace with real PDF bytes)"
        b64_word_pdf = base64.b64encode(word_pdf_bytes).decode()
        st.markdown(f"[📄 Télécharger le PDF du cours Word](data:application/pdf;base64,{b64_word_pdf})")

        # EXCEL Lesson
        st.subheader("Microsoft Excel - Débutant")
        st.markdown("""
**Objectifs :**
- Comprendre l'interface Excel et les cellules
- Saisir et formater les données
- Utiliser les formules de base

**Contenu du cours :**
1. Interface et ruban Excel
2. Cellules, lignes, colonnes
3. Saisie et formatage de données
4. Formules de base : SUM, AVERAGE, MIN, MAX
5. Gestion de feuilles et de classeurs
6. Mise en forme conditionnelle
7. Création de graphiques simples
""")
        excel_pdf_bytes = b"%PDF-1.4\n%PDF content for Excel beginner course (replace with real PDF bytes)"
        b64_excel_pdf = base64.b64encode(excel_pdf_bytes).decode()
        st.markdown(f"[📄 Télécharger le PDF du cours Excel](data:application/pdf;base64,{b64_excel_pdf})")

    # --- Exercises Tab ---
    with tabs[1]:
        st.header("🧩 Exercices pratiques")

        st.subheader("Exercice Word")
        st.markdown("Créez un document Word avec :")
        st.markdown("- Un titre centré")
        st.markdown("- Deux paragraphes avec mise en forme (gras, italique)")
        st.markdown("- Une liste à puces")
        completed_word = st.checkbox("J'ai complété l'exercice Word ✅")

        st.subheader("Exercice Excel")
        st.markdown("Dans Excel :")
        st.markdown("- Créez un tableau avec 5 lignes et 3 colonnes")
        st.markdown("- Ajoutez les valeurs numériques de votre choix")
        st.markdown("- Calculez la somme et la moyenne d'une colonne")
        completed_excel = st.checkbox("J'ai complété l'exercice Excel ✅")

        if completed_word and completed_excel:
            st.success("🎉 Bravo ! Vous avez complété tous les exercices pratiques.")

    # --- Mini-Quiz Tab ---
    with tabs[2]:
        st.header("📝 Mini-Quiz")

        st.subheader("Question 1 - Word")
        q1 = st.radio("Quel raccourci permet de copier un texte ?", ["Ctrl+V", "Ctrl+C", "Ctrl+X"], key="q1")
        st.subheader("Question 2 - Excel")
        q2 = st.radio("Quelle formule calcule la moyenne ?", ["=SUM()", "=AVERAGE()", "=MAX()"], key="q2")

        st.subheader("Question 3 - Word")
        q3 = st.radio("Comment insérer une liste à puces ?", ["Onglet Accueil > Puces", "Onglet Insertion > Image", "Onglet Fichier > Nouveau"], key="q3")

        if st.button("Valider mes réponses"):
            score = 0
            score += 1 if q1 == "Ctrl+C" else 0
            score += 1 if q2 == "=AVERAGE()" else 0
            score += 1 if q3 == "Onglet Accueil > Puces" else 0
            st.info(f"Votre score : {score}/3")
            if score == 3:
                st.success("🎉 Parfait ! Vous avez répondu correctement à toutes les questions.")
            elif score == 2:
                st.warning("👍 Presque parfait, 2 bonnes réponses !")
            else:
                st.error("😕 Révisez le cours et réessayez !")

    st.markdown("---")
    st.markdown("© 2025 Formation Word & Excel - Débutants | Tous droits réservés")
