# tasks_planner.py - Modern Project Management Dashboard

import streamlit as st
import json
import os
from datetime import datetime, date, timedelta
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
# Calendar grid rendering
# -------------------------
def render_interactive_calendar(selected_date: date):
    st.markdown("### 🗓️ Calendrier interactif")

    # --- Month navigation ---
    col1, col2, col3 = st.columns([1,3,1])
    prev_month = (selected_date.replace(day=1) - timedelta(days=1)).replace(day=1)
    next_month = (selected_date.replace(day=28) + timedelta(days=4)).replace(day=1)
    
    with col1:
        if st.button("⬅", key="prev_month"):
            st.session_state["selected_day"] = prev_month.isoformat()
            st.experimental_rerun()
    with col2:
        st.markdown(f"<h3 style='text-align:center'>{selected_date.strftime('%B %Y')}</h3>", unsafe_allow_html=True)
    with col3:
        if st.button("➡", key="next_month"):
            st.session_state["selected_day"] = next_month.isoformat()
            st.experimental_rerun()

    month = selected_date.month
    year = selected_date.year
    cal = calendar.Calendar(firstweekday=0)
    month_days = cal.monthdatescalendar(year, month)

    clicked_date = None

    # --- Weekday headers ---
    header_cols = st.columns(7)
    for i, wd in enumerate(["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"]):
        header_cols[i].markdown(f"<div style='text-align:center;font-weight:bold'>{wd}</div>", unsafe_allow_html=True)

    # --- Calendar grid ---
    for week in month_days:
        week_cols = st.columns(7)
        for i, day in enumerate(week):
            day_tasks = tasks_for_date(day)
            num_tasks = len(day_tasks)
            
            # Color coding by task priority
            if num_tasks > 0:
                if any(t["priority"]=="High" for t in day_tasks):
                    bg_color = "#FFB3B3"  # light red
                elif any(t["priority"]=="Medium" for t in day_tasks):
                    bg_color = "#FFF4B3"  # light yellow
                else:
                    bg_color = "#B3FFB3"  # light green
            else:
                bg_color = "#f0f0f0"

            # Today highlight
            border = "2px solid #FF5733" if day == date.today() else "1px solid #ccc"
            label_color = "#000" if day.month == month else "#aaa"

            # Day box HTML
            day_label = f"<div style='color:{label_color}; font-weight:bold'>{day.day}</div>"
            task_lines = ""
            for t in day_tasks[:3]:
                task_color = "#000"
                if t["priority"]=="High":
                    task_color = "#C70039"
                elif t["priority"]=="Medium":
                    task_color = "#FF8C00"
                task_lines += f"<div style='font-size:10px;text-align:left;color:{task_color}'>- {t.get('title')[:15]}</div>"

            day_html = f"""
            <div style='border:{border}; background-color:{bg_color}; border-radius:8px; padding:5px; min-height:90px; text-align:center'>
                {day_label}
                {task_lines}
            </div>
            """

            if week_cols[i].button("", key=f"cal_{day.isoformat()}"):
                clicked_date = day

            week_cols[i].markdown(day_html, unsafe_allow_html=True)

    return clicked_date

# -------------------------
# Gantt / Timeline rendering
# -------------------------
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
    fig = px.timeline(df, x_start="start", x_end="end", y="Task", color="Status", hover_data=["category","priority","time"])
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
                    st.caption(t.get('description'))
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
    st.header("📅 Task Planner — Dashboard Moderne")

    # Sidebar filters & stats
    all_tasks = load_tasks()
    st.sidebar.markdown("### 📊 Statuts et filtres")
    total = len(all_tasks)
    done = sum(1 for t in all_tasks if t.get("done"))
    pending = total - done
    st.sidebar.write(f"- Total: **{total}**")
    st.sidebar.write(f"- Terminées: **{done}**")
    st.sidebar.write(f"- En attente: **{pending}**")
    st.sidebar.markdown("---")
    category_filter = st.sidebar.multiselect("Filtrer par catégorie", ["Integration","Support","Project","Maintenance","Personal"], default=[])
    priority_filter = st.sidebar.multiselect("Filtrer par priorité", ["Low","Medium","High"], default=[])
    status_filter = st.sidebar.multiselect("Filtrer par statut", ["Pending","Done"], default=[])

    # Top notifications
    render_daily_notification()

    # Selected day in session
    today = date.today()
    st.session_state.setdefault("selected_day", today.isoformat())
    selected_day = parse_iso(st.session_state["selected_day"])

    # --- Layout ---
    left_col, right_col = st.columns([2,3])
    with left_col:
        clicked = render_interactive_calendar(selected_day)
        if clicked:
            st.session_state["selected_day"] = clicked.isoformat()
        st.markdown("---")
        render_task_editor(selected_day)

    with right_col:
        st.markdown("### 📈 Gantt / Planning des tâches")
        # Apply filters to Gantt
        gantt_tasks = all_tasks
        if category_filter:
            gantt_tasks = [t for t in gantt_tasks if t["category"] in category_filter]
        if priority_filter:
            gantt_tasks = [t for t in gantt_tasks if t["priority"] in priority_filter]
        if status_filter:
            gantt_tasks = [t for t in gantt_tasks if ("Done" if t["done"] else "Pending") in status_filter]

        if gantt_tasks:
            df = pd.DataFrame(gantt_tasks)
            df["start"] = pd.to_datetime(df["date"])
            df["end"] = df["start"] + pd.to_timedelta(1, unit="d")
            df["Task"] = df["title"]
            df["Status"] = df["done"].apply(lambda x: "Done" if x else "Pending")
            fig = px.timeline(df, x_start="start", x_end="end", y="Task", color="Status",
                              hover_data=["category","priority","time"])
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Aucune tâche correspondante pour le Gantt avec vos filtres.")

# -------------------------
# Run app
# -------------------------
if __name__ == "__main__":
    render_task_planner()
