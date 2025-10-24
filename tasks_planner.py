# tasks_planner.py
"""
Monday.com-like Task Planner for Streamlit
Features:
 - Views: Grid (editable), Calendar (month grid), Kanban (columns)
 - Task model: id, title, description, start_datetime, end_datetime, priority, category, status, created_at
 - Pastel colors per category
 - Auto-save to CSV (data/tasks.csv) and JSON (data/tasks.json)
 - Inline editing via st.experimental_data_editor in Grid view
"""
import streamlit as st
import os
import json
import uuid
from datetime import datetime, date, time, timedelta
from typing import List, Dict, Any
import pandas as pd
import calendar
import plotly.express as px

# -------------------------
# Config / Files
# -------------------------
DATA_DIR = "data"
JSON_FILE = os.path.join(DATA_DIR, "tasks.json")
CSV_FILE = os.path.join(DATA_DIR, "tasks.csv")
os.makedirs(DATA_DIR, exist_ok=True)

# -------------------------
# Categories & Colors (pastel)
# -------------------------
CATEGORY_COLORS = {
    "Integration": "#A8D5BA",  # pastel green
    "Support": "#F6C8C8",      # pastel pink
    "Project": "#F9E79F",      # pastel yellow
    "Maintenance": "#C3AED6",  # pastel purple
    "Personal": "#B3E5FC"      # pastel blue
}

PRIORITIES = ["Low", "Medium", "High"]
STATUSES = ["To Do", "In Progress", "Blocked", "Done"]

# -------------------------
# Persistence helpers
# -------------------------
def load_tasks() -> List[Dict[str, Any]]:
    """Load tasks from CSV if present, else JSON, else return []"""
    if os.path.exists(CSV_FILE):
        try:
            df = pd.read_csv(CSV_FILE)
            tasks = df.to_dict(orient="records")
            # ensure fields present
            for t in tasks:
                # convert nan to empty strings where appropriate
                for k in ["description", "category", "priority", "status"]:
                    if pd.isna(t.get(k, "")):
                        t[k] = ""
                # ensure datetimes are strings
                if pd.isna(t.get("start_datetime", "")):
                    t["start_datetime"] = ""
                if pd.isna(t.get("end_datetime", "")):
                    t["end_datetime"] = ""
            return tasks
        except Exception:
            pass
    if os.path.exists(JSON_FILE):
        try:
            with open(JSON_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, list) else []
        except Exception:
            pass
    return []

def save_tasks(tasks: List[Dict[str, Any]]):
    """Save tasks to both CSV and JSON (auto-save)"""
    try:
        # Normalize tasks for CSV: ensure keys consistent
        df = pd.DataFrame(tasks)
        # Ensure columns exist
        cols = ["id", "title", "description", "start_datetime", "end_datetime", "priority", "category", "status", "created_at"]
        for c in cols:
            if c not in df.columns:
                df[c] = ""
        # Save CSV (index False)
        df.to_csv(CSV_FILE, index=False)
        # Save JSON pretty
        with open(JSON_FILE, "w", encoding="utf-8") as f:
            json.dump(tasks, f, ensure_ascii=False, indent=2)
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
def iso_now():
    return datetime.utcnow().replace(microsecond=0).isoformat()

def parse_dt(s: str) -> datetime:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s)
    except Exception:
        try:
            # try common alternative parsing
            return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
        except Exception:
            return None

def ensure_task_model(t: Dict[str, Any]) -> Dict[str, Any]:
    # Ensure all required keys exist with defaults
    model = {
        "id": t.get("id") or str(uuid.uuid4()),
        "title": t.get("title") or "Untitled",
        "description": t.get("description") or "",
        "start_datetime": t.get("start_datetime") or "",
        "end_datetime": t.get("end_datetime") or "",
        "priority": t.get("priority") or "Low",
        "category": t.get("category") or "Project",
        "status": t.get("status") or "To Do",
        "created_at": t.get("created_at") or iso_now()
    }
    # If start exists but end missing, set default +1h
    sd = parse_dt(model["start_datetime"])
    ed = parse_dt(model["end_datetime"])
    if sd and not ed:
        model["end_datetime"] = (sd + timedelta(hours=1)).isoformat()
    return model

def tasks_df(tasks: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame([ensure_task_model(t) for t in tasks])
    if df.empty:
        # create default columns
        df = pd.DataFrame(columns=["id","title","description","start_datetime","end_datetime","priority","category","status","created_at"])
    return df

# -------------------------
# UI Components
# -------------------------
def top_bar():
    st.title("📋 Task Planner — Dashboard (Monday.com style)")
    col1, col2, col3 = st.columns([1,3,1])
    with col1:
        if st.button("Nouvelle tâche rapide"):
            # set session selected day to today and open add form (handled later)
            st.session_state.setdefault("show_add", True)
            st.session_state.setdefault("selected_day", date.today().isoformat())
            st.experimental_rerun()
    with col2:
        st.markdown("**Affichage**")
    with col3:
        st.write("")  # placeholder for alignment

def add_task_form(default_date: date):
    st.markdown("### ➕ Ajouter / Importer une tâche")
    with st.form("add_task_form"):
        title = st.text_input("Titre")
        description = st.text_area("Description")
        task_date = st.date_input("Date de début", value=default_date)
        task_time = st.time_input("Heure de début", value=datetime.now().time())
        start_dt = datetime.combine(task_date, task_time)
        duration_hours = st.number_input("Durée (heures)", min_value=0.25, max_value=168.0, value=1.0, step=0.25)
        end_dt = start_dt + timedelta(hours=duration_hours)
        priority = st.selectbox("Priorité", PRIORITIES, index=1)
        category = st.selectbox("Catégorie", list(CATEGORY_COLORS.keys()), index=0)
        status = st.selectbox("Statut", STATUSES, index=0)
        submitted = st.form_submit_button("Ajouter")
        if submitted:
            new_task = {
                "id": str(uuid.uuid4()),
                "title": title or "Tâche sans titre",
                "description": description or "",
                "start_datetime": start_dt.isoformat(),
                "end_datetime": end_dt.isoformat(),
                "priority": priority,
                "category": category,
                "status": status,
                "created_at": iso_now()
            }
            add_task(new_task)
            st.success("Tâche ajoutée ✅")
            st.session_state["show_add"] = False
            st.experimental_rerun()

# -------------------------
# Views
# -------------------------
def grid_view():
    st.subheader("Grid — Edition rapide")
    tasks = load_tasks()
    df = tasks_df(tasks)
    # Use experimental_data_editor for inline editing
    # Hide id/created_at in editable view or show as readonly
    if "grid_editor_df" not in st.session_state:
        st.session_state["grid_editor_df"] = df.copy()
    edited = st.experimental_data_editor(
        st.session_state["grid_editor_df"],
        num_rows="dynamic",
        use_container_width=True
    )
    # After edit, show save controls
    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        if st.button("Enregistrer modifications (Grid)"):
            # Turn edited dataframe back to tasks list
            tasks_out = []
            for _, row in edited.iterrows():
                t = row.to_dict()
                t = ensure_task_model(t)
                tasks_out.append(t)
            save_tasks(tasks_out)
            st.success("Modifications sauvegardées ✅")
            st.session_state["grid_editor_df"] = edited
            st.experimental_rerun()
    with col2:
        if st.button("Importer CSV"):
            uploaded = st.file_uploader("Charger un CSV", type=["csv"], key="import_csv")
            if uploaded:
                try:
                    df_in = pd.read_csv(uploaded)
                    tasks_out = []
                    for _, row in df_in.fillna("").iterrows():
                        t = row.to_dict()
                        t = ensure_task_model(t)
                        tasks_out.append(t)
                    save_tasks(tasks_out)
                    st.success("CSV importé ✅")
                    st.experimental_rerun()
                except Exception as e:
                    st.error(f"Erreur import CSV: {e}")
    with col3:
        st.download_button("Télécharger CSV", data=open(CSV_FILE, "rb"), file_name=f"tasks_{date.today().isoformat()}.csv")

def calendar_view():
    st.subheader("Calendar — Vue mensuelle")
    tasks = load_tasks()
    # controls: month/year navigation
    if "calendar_month" not in st.session_state:
        today = date.today()
        st.session_state["calendar_month"] = today.month
        st.session_state["calendar_year"] = today.year
    ccol1, ccol2, ccol3 = st.columns([1,3,1])
    with ccol1:
        if st.button("⬅", key="cal_prev"):
            # go to previous month
            m = st.session_state["calendar_month"]
            y = st.session_state["calendar_year"]
            prev = (date(y, m, 1) - timedelta(days=1)).replace(day=1)
            st.session_state["calendar_month"] = prev.month
            st.session_state["calendar_year"] = prev.year
    with ccol2:
        st.markdown(f"### {date(st.session_state['calendar_year'], st.session_state['calendar_month'], 1).strftime('%B %Y')}")
    with ccol3:
        if st.button("➡", key="cal_next"):
            m = st.session_state["calendar_month"]
            y = st.session_state["calendar_year"]
            # increment month
            nextm = (date(y, m, 28) + timedelta(days=4)).replace(day=1)
            st.session_state["calendar_month"] = nextm.month
            st.session_state["calendar_year"] = nextm.year

    month = st.session_state["calendar_month"]
    year = st.session_state["calendar_year"]
    cal = calendar.Calendar(firstweekday=0)
    month_days = cal.monthdatescalendar(year, month)

    # Weekday headers
    cols = st.columns(7)
    for i, wd in enumerate(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]):
        cols[i].markdown(f"**{wd}**")

    clicked = None
    tasks_list = tasks_df(tasks).to_dict(orient="records")
    for week in month_days:
        cols = st.columns(7)
        for i, day in enumerate(week):
            day_tasks = [t for t in tasks_list if parse_dt(t["start_datetime"]) and parse_dt(t["start_datetime"]).date() == day]
            # background color: if no tasks => light gray else color of first task category
            if day_tasks:
                bg = CATEGORY_COLORS.get(day_tasks[0]["category"], "#eaeaea")
            else:
                bg = "#f8f8f8"
            border = "3px solid #2b8fff" if day == date.today() else "1px solid #ddd"
            label_color = "#000" if day.month == month else "#aaa"
            day_label = f"<div style='font-weight:bold;color:{label_color}'>{day.day}</div>"
            tasks_html = ""
            for t in day_tasks[:4]:
                dt = parse_dt(t["start_datetime"])
                time_str = dt.strftime("%H:%M") if dt else ""
                tasks_html += f"<div style='font-size:11px;text-align:left;padding:2px 0'>• <b>{t['title']}</b> <span style='font-size:10px;color:#555'>({time_str})</span></div>"
            html = f"""
                <div style='border:{border}; background-color:{bg}; border-radius:6px; min-height:100px; padding:6px'>
                    {day_label}
                    {tasks_html}
                </div>
            """
            if cols[i].button("", key=f"cal_{day.isoformat()}"):
                clicked = day
            cols[i].markdown(html, unsafe_allow_html=True)
    if clicked:
        st.session_state["selected_day"] = clicked.isoformat()
        st.info(f"Jour sélectionné : {clicked.isoformat()}")

def kanban_view():
    st.subheader("Kanban — Vue tableau (Trello style)")
    tasks = tasks_df(load_tasks())
    # Create columns for statuses
    cols = st.columns(len(STATUSES))
    status_columns = {s: cols[i] for i, s in enumerate(STATUSES)}
    # Build a mapping from status -> tasks
    tasks_by_status = {s: [] for s in STATUSES}
    for _, row in tasks.iterrows():
        r = row.to_dict()
        tasks_by_status.setdefault(r["status"], []).append(ensure_task_model(r))

    # For each column, render tasks and allow status change
    for status, col in status_columns.items():
        with col:
            st.markdown(f"### {status}")
            for t in tasks_by_status.get(status, []):
                cat_color = CATEGORY_COLORS.get(t["category"], "#eee")
                start = parse_dt(t["start_datetime"])
                when = start.strftime("%Y-%m-%d %H:%M") if start else ""
                card_html = f"""
                <div style='background-color:#fff;border-radius:8px;padding:8px;margin-bottom:8px;box-shadow:0 1px 3px rgba(0,0,0,0.08)'>
                    <div style='font-weight:bold'>{t['title']}</div>
                    <div style='font-size:12px;color:#444'>{t['category']} • {t['priority']} • {when}</div>
                    <div style='font-size:12px;margin-top:6px'>{t['description'][:120]}</div>
                </div>
                """
                st.markdown(card_html, unsafe_allow_html=True)
                # Controls to edit/move/delete
                key_move = f"move_{t['id']}"
                key_edit = f"edit_{t['id']}"
                # Move
                new_status = st.selectbox("Déplacer vers", STATUSES, index=STATUSES.index(t["status"]), key=key_move)
                if new_status != t["status"]:
                    update_task(t["id"], {"status": new_status})
                    st.experimental_rerun()
                # Quick edit open modal via expander
                with st.expander("Modifier"):
                    new_title = st.text_input("Titre", value=t["title"], key=f"title_{t['id']}")
                    new_desc = st.text_area("Description", value=t["description"], key=f"desc_{t['id']}")
                    new_cat = st.selectbox("Catégorie", list(CATEGORY_COLORS.keys()), index=list(CATEGORY_COLORS.keys()).index(t["category"]), key=f"cat_{t['id']}")
                    new_prio = st.selectbox("Priorité", PRIORITIES, index=PRIORITIES.index(t["priority"]), key=f"prio_{t['id']}")
                    # start / end
                    s_date = parse_dt(t["start_datetime"])
                    default_date = s_date.date() if s_date else date.today()
                    default_time = s_date.time() if s_date else datetime.now().time()
                    d_date = st.date_input("Date début", value=default_date, key=f"sdate_{t['id']}")
                    d_time = st.time_input("Heure début", value=default_time, key=f"stime_{t['id']}")
                    dur = (parse_dt(t["end_datetime"]) - parse_dt(t["start_datetime"])).total_seconds()/3600 if parse_dt(t["end_datetime"]) and parse_dt(t["start_datetime"]) else 1
                    duration_hours = st.number_input("Durée (heures)", min_value=0.25, max_value=168.0, value=max(0.25, float(dur)), key=f"dur_{t['id']}")
                    if st.button("Sauvegarder", key=f"save_{t['id']}"):
                        sdt = datetime.combine(d_date, d_time)
                        edt = sdt + timedelta(hours=duration_hours)
                        update_task(t["id"], {
                            "title": new_title,
                            "description": new_desc,
                            "category": new_cat,
                            "priority": new_prio,
                            "start_datetime": sdt.isoformat(),
                            "end_datetime": edt.isoformat()
                        })
                        st.success("Mis à jour")
                        st.experimental_rerun()
                if st.button("Supprimer cette tâche", key=f"del_{t['id']}"):
                    delete_task(t["id"])
                    st.success("Supprimé")
                    st.experimental_rerun()

# -------------------------
# Extra: Gantt mini (optional)
# -------------------------
def gantt_mini():
    st.subheader("Gantt — Vue synthétique")
    tasks = tasks_df(load_tasks())
    if tasks.empty:
        st.info("Aucune tâche pour le Gantt.")
        return
    df = tasks.copy()
    # Ensure datetime columns
    df["start"] = pd.to_datetime(df["start_datetime"])
    df["end"] = pd.to_datetime(df["end_datetime"])
    df["TaskLabel"] = df["title"] + " (" + df["status"] + ")"
    fig = px.timeline(df, x_start="start", x_end="end", y="TaskLabel", color="category",
                      color_discrete_map=CATEGORY_COLORS, hover_data=["priority","description"])
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, use_container_width=True)

# -------------------------
# Main UI
# -------------------------
def render_task_planner():
    # top
    top_bar()

    # left sidebar: view selector & filters
    st.sidebar.markdown("### Vue / Filtres")
    view = st.sidebar.radio("Choisir une vue", ["Grid", "Calendar", "Kanban"], index=0)
    # Filters
    categories = st.sidebar.multiselect("Catégories", options=list(CATEGORY_COLORS.keys()), default=list(CATEGORY_COLORS.keys()))
    priorities = st.sidebar.multiselect("Priorités", options=PRIORITIES, default=PRIORITIES)
    statuses = st.sidebar.multiselect("Statuts", options=STATUSES, default=STATUSES)

    # Show add form if requested
    selected_day = parse_dt(st.session_state.get("selected_day", date.today().isoformat()))
    default_date = selected_day.date() if selected_day else date.today()
    if st.session_state.get("show_add", False):
        add_task_form(default_date)

    # Main area
    if view == "Grid":
        grid_view()
    elif view == "Calendar":
        calendar_view()
    elif view == "Kanban":
        kanban_view()

    # Bottom: Gantt mini and global stats
    st.markdown("---")
    st.markdown("### Statistiques & Gantt")
    all_tasks = tasks_df(load_tasks())
    total = len(all_tasks)
    done = sum(1 for t in all_tasks.to_dict(orient="records") if t.get("status") == "Done")
    pending = total - done
    c1, c2, c3 = st.columns(3)
    c1.metric("Total tâches", total)
    c2.metric("Terminées", done)
    c3.metric("En attente", pending)

    gantt_mini()

# -------------------------
# Run App
# -------------------------
if __name__ == "__main__":
    render_task_planner()
