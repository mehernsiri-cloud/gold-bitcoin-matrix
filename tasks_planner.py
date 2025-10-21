# tasks_planner.py (Enhanced with Gantt-like & planning features)

import streamlit as st
import json
import os
from datetime import datetime, date, time, timedelta
import calendar
import uuid
from typing import List, Dict, Any
import pandas as pd
import plotly.express as px

DATA_DIR = "data"
TASKS_FILE = os.path.join(DATA_DIR, "tasks.json")

os.makedirs(DATA_DIR, exist_ok=True)

# -------------------------
# Persistence helpers
# -------------------------
def load_tasks() -> List[Dict[str, Any]]:
    if not os.path.exists(TASKS_FILE):
        return []
    try:
        with open(TASKS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except Exception:
        return []

def save_tasks(tasks: List[Dict[str, Any]]):
    try:
        with open(TASKS_FILE, "w", encoding="utf-8") as f:
            json.dump(tasks, f, ensure_ascii=False, indent=2, default=str)
    except Exception as e:
        st.error(f"Erreur en sauvegardant les tâches : {e}")

def add_task(task: Dict[str, Any]):
    tasks = load_tasks()
    tasks.append(task)
    save_tasks(tasks)

def update_task(task_id: str, updates: Dict[str, Any]):
    tasks = load_tasks()
    for t in tasks:
        if t.get("id") == task_id:
            t.update(updates)
            break
    save_tasks(tasks)

def delete_task(task_id: str):
    tasks = [t for t in load_tasks() if t.get("id") != task_id]
    save_tasks(tasks)

# -------------------------
# Utility helpers
# -------------------------
def parse_iso(s: str) -> date:
    return datetime.fromisoformat(s).date()

def tasks_for_date(target_date: date) -> List[Dict[str, Any]]:
    tasks = load_tasks()
    day_tasks = []
    for t in tasks:
        try:
            t_date = datetime.fromisoformat(t["date"]).date()
            if t_date == target_date:
                day_tasks.append(t)
        except Exception:
            continue
    day_tasks.sort(key=lambda x: x.get("time") or "")
    return day_tasks

# -------------------------
# Interactive Calendar & Gantt
# -------------------------
def render_interactive_calendar(selected_date: date):
    st.markdown("### 🗓️ Calendrier interactif")
    month = selected_date.month
    year = selected_date.year
    cal = calendar.Calendar(firstweekday=0)
    month_days = cal.monthdatescalendar(year, month)

    clicked_date = None

    # Weekday headers
    header_cols = st.columns(7)
    for i, wd in enumerate(["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"]):
        header_cols[i].markdown(f"<div style='text-align:center;font-weight:bold'>{wd}</div>", unsafe_allow_html=True)

    # Calendar grid
    for week in month_days:
        week_cols = st.columns(7)
        for i, day in enumerate(week):
            day_tasks = tasks_for_date(day)
            badge = f" ({len(day_tasks)})" if day_tasks else ""
            is_today = day == date.today()
            style = ""
            if is_today:
                style += "border: 2px solid #FF5733; background-color: #FFF0E0; border-radius: 8px; padding: 10px; text-align:center"
            else:
                style += "border: 1px solid #ccc; border-radius: 8px; padding: 10px; text-align:center"

            # Only show current month days as bold
            label_color = "#000" if day.month == month else "#aaa"

            day_label = f"<div style='color:{label_color}; font-weight:bold'>{day.day}{badge}</div>"
            # Add up to 2 task titles inside the box
            task_lines = ""
            for t in day_tasks[:2]:
                task_lines += f"<div style='font-size:10px;text-align:left'>- {t.get('title')[:15]}</div>"

            if week_cols[i].button("", key=f"cal_{day.isoformat()}"):
                clicked_date = day

            week_cols[i].markdown(f"<div style='{style}'>{day_label}{task_lines}</div>", unsafe_allow_html=True)

    return clicked_date

def render_gantt_chart():
    tasks = load_tasks()
    if not tasks:
        st.info("Aucune tâche pour le Gantt.")
        return
    df = pd.DataFrame(tasks)
    df["start"] = pd.to_datetime(df["date"])
    df["end"] = df["start"] + pd.to_timedelta(1, unit="d")
    df["Task"] = df["title"]
    df["Status"] = df["done"].apply(lambda x: "Done" if x else "Pending")
    fig = px.timeline(df, x_start="start", x_end="end", y="Task", color="Status", hover_data=["category","priority"])
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, use_container_width=True)

# -------------------------
# Task editor & forms
# -------------------------
def render_task_editor(selected_day: date):
    st.markdown(f"### Tâches pour le {selected_day.isoformat()}")

    day_tasks = tasks_for_date(selected_day)
    if day_tasks:
        st.markdown("**Tâches existantes**")
        for t in day_tasks:
            cols = st.columns([4,1,1])
            with cols[0]:
                status = "✅" if t.get("done") else "🕘"
                st.write(f"{status} **{t.get('title')}** — {t.get('category')} — {t.get('priority')}")
                if t.get("description"):
                    st.caption(t.get("description"))
                if t.get("time"):
                    st.caption(f"Heure: {t.get('time')}")
            with cols[1]:
                new_done = st.checkbox("Fait", value=bool(t.get("done")), key=f"done_{t['id']}")
                if new_done != bool(t.get("done")):
                    update_task(t["id"], {"done": new_done})
            with cols[2]:
                if st.button("Supprimer", key=f"del_{t['id']}"):
                    delete_task(t["id"])
                    st.experimental_rerun()

    st.markdown("---")
    st.markdown("### Ajouter une nouvelle tâche")
    with st.form(key=f"add_task_form_{selected_day.isoformat()}"):
        title = st.text_input("Titre")
        description = st.text_area("Description (optionnelle)")
        time_input = st.time_input("Heure (optionnelle)", value=None)
        time_str = time_input.strftime("%H:%M") if time_input else ""
        priority = st.selectbox("Priorité", ["Low","Medium","High"])
        category = st.selectbox("Catégorie", ["Integration","Support","Project","Maintenance","Personal"])
        submit = st.form_submit_button("Ajouter la tâche")
        if submit:
            new_task = {
                "id": str(uuid.uuid4()),
                "title": title or "Tâche sans titre",
                "description": description or "",
                "date": selected_day.isoformat(),
                "time": time_str,
                "priority": priority,
                "category": category,
                "done": False,
                "created_at": datetime.utcnow().isoformat()
            }
            add_task(new_task)
            st.success("Tâche ajoutée ✅")
            st.experimental_rerun()

# -------------------------
# Daily notification
# -------------------------
def render_daily_notification():
    today = date.today()
    todays_tasks = tasks_for_date(today)
    if not todays_tasks:
        st.info("🔔 Aucune tâche pour aujourd'hui.")
    else:
        st.success(f"🔔 {len(todays_tasks)} tâche(s) aujourd'hui :")
        for t in todays_tasks:
            status = "✅" if t.get("done") else "🕘"
            time_str = f" @ {t.get('time')}" if t.get("time") else ""
            st.write(f"{status} **{t.get('title')}**{time_str} — {t.get('category')} — {t.get('priority')}")

# -------------------------
# Main render function
# -------------------------
def render_task_planner():
    st.header("📅 Task Planner — Calendrier interactif & tâches")

    # Top bar
    left_col, right_col = st.columns([3,1])
    with left_col:
        render_daily_notification()
    with right_col:
        st.markdown("**Actions**")
        if st.button("Nouvelle tâche rapide"):
            st.session_state.setdefault("selected_day", date.today().isoformat())
        # Download / Upload
        tasks = load_tasks()
        st.download_button("Télécharger tâches (JSON)", data=json.dumps(tasks, ensure_ascii=False, indent=2),
                           file_name=f"tasks_{date.today().isoformat()}.json", mime="application/json")
        uploaded = st.file_uploader("Importer tasks.json", type=["json"])
        if uploaded:
            try:
                data = json.load(uploaded)
                if isinstance(data, list):
                    save_tasks(data)
                    st.success("Fichier importé et sauvegardé.")
                    st.experimental_rerun()
                else:
                    st.error("Format JSON invalide.")
            except Exception as e:
                st.error(f"Erreur import: {e}")

    st.markdown("---")

    # Selected month & day
    today = date.today()
    st.session_state.setdefault("calendar_month", today.month)
    st.session_state.setdefault("calendar_year", today.year)
    st.session_state.setdefault("selected_day", today.isoformat())

    # Render calendar
    clicked = render_interactive_calendar(parse_iso(st.session_state["selected_day"]))
    if clicked:
        st.session_state["selected_day"] = clicked.isoformat()

    # Task editor
    st.markdown("---")
    render_task_editor(parse_iso(st.session_state["selected_day"]))

    # Gantt Chart
    st.markdown("---")
    st.markdown("### 📈 Vue Gantt / Planning des tâches")
    render_gantt_chart()

    # Sidebar stats
    all_tasks = load_tasks()
    total = len(all_tasks)
    done = sum(1 for t in all_tasks if t.get("done"))
    pending = total - done
    st.sidebar.markdown("### 📊 Statuts des tâches")
    st.sidebar.write(f"- Total: **{total}**")
    st.sidebar.write(f"- Terminées: **{done}**")
    st.sidebar.write(f"- En attente: **{pending}**")
    st.sidebar.write(f"- Fichier: `{TASKS_FILE}`")

# -------------------------
# Run standalone
# -------------------------
if __name__ == "__main__":
    render_task_planner()
