# tasks_planner.py
import streamlit as st
import json
import os
from datetime import datetime, date, timedelta
import calendar
import uuid
from typing import List, Dict, Any

DATA_DIR = "data"
TASKS_FILE = os.path.join(DATA_DIR, "tasks.json")

# Ensure data folder exists
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
            if isinstance(data, list):
                return data
            else:
                return []
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
    changed = False
    for t in tasks:
        if t.get("id") == task_id:
            t.update(updates)
            changed = True
            break
    if changed:
        save_tasks(tasks)

def delete_task(task_id: str):
    tasks = load_tasks()
    tasks = [t for t in tasks if t.get("id") != task_id]
    save_tasks(tasks)

# -------------------------
# Utility helpers
# -------------------------
def iso(d: date) -> str:
    return d.isoformat()

def parse_iso(s: str) -> date:
    return datetime.fromisoformat(s).date()

def tasks_for_date(target_date: date) -> List[Dict[str, Any]]:
    tasks = load_tasks()
    out = []
    for t in tasks:
        try:
            t_date = datetime.fromisoformat(t["date"]).date()
            if t_date == target_date:
                out.append(t)
        except Exception:
            continue
    # Sort by time field if present
    def sort_key(x):
        try:
            return x.get("time") or ""
        except:
            return ""
    out.sort(key=sort_key)
    return out

def count_tasks_by_sentiment(tasks: List[Dict[str, Any]]):
    # placeholder - not used here but kept for parity with other dashboards
    return len(tasks)

# -------------------------
# Calendar rendering
# -------------------------
def render_month_calendar(year: int, month: int, selected_date: date):
    """
    Renders a calendar grid for given year/month using Streamlit columns.
    Returns the date that user clicked (or None).
    """
    cal = calendar.Calendar(firstweekday=0)  # Monday=0? calendar module default is Monday=0; adjust if needed
    month_days = cal.monthdatescalendar(year, month)

    st.markdown(f"### {calendar.month_name[month]} {year}")
    clicked_date = None

    # Weekday headers
    cols = st.columns(7)
    for i, wd in enumerate(["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"]):
        cols[i].markdown(f"**{wd}**")

    # For each week row, render 7 columns
    for week in month_days:
        cols = st.columns(7)
        for i, day in enumerate(week):
            col = cols[i]
            is_current_month = (day.month == month)
            day_tasks = tasks_for_date(day)
            badge = ""
            if len(day_tasks) > 0:
                badge = f" — {len(day_tasks)}"
            # style background depending on selected / today / other month
            is_today = (day == date.today())
            label = f"**{day.day}**{badge}"
            if not is_current_month:
                # muted tone for days not in the month
                if col.button(label, key=f"day_{day.isoformat()}", help=str(day)):
                    clicked_date = day
            else:
                # Show color-coded small indicators for priority
                if col.button(label, key=f"day_{day.isoformat()}", help=str(day)):
                    clicked_date = day
                # Add small summary below — we cannot control spacing much, but we can add a small line
                if len(day_tasks) > 0:
                    # show top 2 task titles truncated
                    for t in day_tasks[:2]:
                        try:
                            title = t.get("title", "")[:30]
                            col.caption(f"- {title}")
                        except:
                            pass
            # Highlight today's date (can't style button easily; show caption)
            if is_today:
                col.markdown("_Aujourd'hui_")
    return clicked_date

# -------------------------
# Task forms & UI
# -------------------------
def render_task_editor(selected_day: date):
    st.markdown(f"#### Tâches pour le {selected_day.isoformat()}")

    # Existing tasks for the day
    day_tasks = tasks_for_date(selected_day)
    if day_tasks:
        st.markdown("**Tâches existantes**")
        for t in day_tasks:
            cols = st.columns([4, 1, 1])
            with cols[0]:
                status = "✅" if t.get("done") else "🕘"
                st.write(f"{status} **{t.get('title')}** — {t.get('category','')} — {t.get('priority','')}")
                if t.get("description"):
                    st.caption(t.get("description"))
                if t.get("time"):
                    st.caption(f"Heure: {t.get('time')}")
            with cols[1]:
                # Toggle done
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
        title = st.text_input("Titre", "")
        description = st.text_area("Description (optionnelle)", "", height=80)
        time = st.time_input("Heure (optionnelle)", value=None)
        if time is not None:
            time_str = time.strftime("%H:%M")
        else:
            time_str = ""
        priority = st.selectbox("Priorité", ["Low", "Medium", "High"])
        category = st.selectbox("Catégorie", ["Integration", "Support", "Project", "Maintenance", "Personal"])
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
# Notifications
# -------------------------
def render_daily_notification():
    today = date.today()
    todays = tasks_for_date(today)
    if not todays:
        st.info("🔔 Aucune tâche planifiée pour aujourd'hui. Profitez-en pour avancer sur vos priorités !")
    else:
        st.success(f"🔔 Vous avez {len(todays)} tâche(s) aujourd'hui :")
        for t in todays:
            time = t.get("time") or ""
            done = "✅" if t.get("done") else "🕘"
            st.write(f"{done} **{t.get('title')}** {f'@{time}' if time else ''} — {t.get('category')} — {t.get('priority')}")

# -------------------------
# Export / Import helpers
# -------------------------
def download_tasks_button():
    tasks = load_tasks()
    st.download_button(
        label="Télécharger tâches (JSON)",
        data=json.dumps(tasks, ensure_ascii=False, indent=2),
        file_name=f"tasks_{date.today().isoformat()}.json",
        mime="application/json"
    )

def upload_tasks_area():
    uploaded = st.file_uploader("Importer fichier tasks.json", type=["json"])
    if uploaded is not None:
        try:
            data = json.load(uploaded)
            if isinstance(data, list):
                save_tasks(data)
                st.success("Fichier importé et sauvegardé.")
                st.experimental_rerun()
            else:
                st.error("Format JSON invalide: attendu une liste de tâches.")
        except Exception as e:
            st.error(f"Impossible d'importer le fichier : {e}")

# -------------------------
# Public render function (call from your app.py)
# -------------------------
def render_task_planner():
    st.header("📅 Task Planner — Calendrier mensuel & tâches")

    # Top bar: quick controls and daily notification
    left_col, right_col = st.columns([3, 1])
    with left_col:
        render_daily_notification()
    with right_col:
        st.markdown("**Actions**")
        if st.button("Nouvelle tâche rapide"):
            # open a small modal-like flow by redirecting to today cell
            st.session_state.setdefault("selected_day", date.today().isoformat())
        download_tasks_button()
        upload_tasks_area()

    st.markdown("---")

    # month navigation in sidebar or top
    now = datetime.now()
    # Persist selected month in session state
    if "calendar_month" not in st.session_state:
        st.session_state["calendar_month"] = now.month
    if "calendar_year" not in st.session_state:
        st.session_state["calendar_year"] = now.year
    if "selected_day" not in st.session_state:
        st.session_state["selected_day"] = date.today().isoformat()

    # Month navigation
    nav_col1, nav_col2, nav_col3 = st.columns([1, 2, 1])
    with nav_col1:
        if st.button("◀️ Prev"):
            # decrement month
            y = st.session_state["calendar_year"]
            m = st.session_state["calendar_month"] - 1
            if m < 1:
                m = 12
                y -= 1
            st.session_state["calendar_month"] = m
            st.session_state["calendar_year"] = y
    with nav_col2:
        # center: show current month/year and a month picker
        month_selection = st.selectbox(
            "Mois",
            list(range(1,13)),
            index=st.session_state["calendar_month"] - 1,
            format_func=lambda m: f"{calendar.month_name[m]} {st.session_state['calendar_year']}"
        )
        # update month selection
        st.session_state["calendar_month"] = month_selection
    with nav_col3:
        if st.button("Next ▶️"):
            y = st.session_state["calendar_year"]
            m = st.session_state["calendar_month"] + 1
            if m > 12:
                m = 1
                y += 1
            st.session_state["calendar_month"] = m
            st.session_state["calendar_year"] = y

    # Render calendar grid
    year = st.session_state["calendar_year"]
    month = st.session_state["calendar_month"]
    selected_click = render_month_calendar(year, month, parse_iso(st.session_state["selected_day"]))
    if selected_click:
        st.session_state["selected_day"] = selected_click.isoformat()

    # Selected day details on the right (or below)
    st.markdown("---")
    selected_day = parse_iso(st.session_state["selected_day"])
    st.markdown(f"## Détails : {selected_day.isoformat()}")
    render_task_editor(selected_day)

    # Quick stats
    all_tasks = load_tasks()
    total = len(all_tasks)
    done = sum(1 for t in all_tasks if t.get("done"))
    pending = total - done
    st.sidebar.markdown("### 📊 Statuts des tâches")
    st.sidebar.write(f"- Total: **{total}**")
    st.sidebar.write(f"- Terminées: **{done}**")
    st.sidebar.write(f"- En attente: **{pending}**")
    st.sidebar.write(f"- Fichier: `{TASKS_FILE}`")

# If run directly (for testing)
if __name__ == "__main__":
    render_task_planner()
