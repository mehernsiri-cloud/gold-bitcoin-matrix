import os
import pandas as pd
import streamlit as st
import plotly.express as px
from datetime import datetime, date
import calendar

# --- FILE SETUP ---
DATA_DIR = "data"
CSV_FILE = os.path.join(DATA_DIR, "tasks_log.csv")
os.makedirs(DATA_DIR, exist_ok=True)

# --- TASK UTILITIES ---
def load_tasks():
    """Load tasks from CSV if it exists, else return empty DataFrame."""
    if os.path.exists(CSV_FILE):
        return pd.read_csv(CSV_FILE, parse_dates=["Start Date", "End Date"])
    else:
        columns = ["Task Name", "Description", "Status", "Priority", "Start Date", "End Date"]
        return pd.DataFrame(columns=columns)

def save_tasks(df):
    """Save tasks to CSV."""
    df.to_csv(CSV_FILE, index=False)

# --- COLOR SCHEME ---
STATUS_COLORS = {
    "To Do": "#A5D8FF",      # light blue
    "In Progress": "#FFE066",# soft yellow
    "Done": "#B2F2BB",       # light green
    "Blocked": "#FFC9C9"     # pastel red
}

PRIORITY_COLORS = {
    "Low": "#D0EBFF",
    "Medium": "#FFD8A8",
    "High": "#FFA8A8",
    "Critical": "#FF8787"
}

# --- TASK CREATION FORM ---
def add_task_form():
    st.subheader("➕ Add a New Task")

    with st.form("add_task"):
        col1, col2 = st.columns(2)
        with col1:
            task_name = st.text_input("Task Name")
            status = st.selectbox("Status", list(STATUS_COLORS.keys()))
            priority = st.selectbox("Priority", list(PRIORITY_COLORS.keys()))
        with col2:
            start_date = st.date_input("Start Date", date.today())
            end_date = st.date_input("End Date", date.today())
        description = st.text_area("Description", height=100)

        submitted = st.form_submit_button("Save Task")
        if submitted:
            if not task_name:
                st.error("⚠️ Please enter a task name.")
            else:
                new_task = {
                    "Task Name": task_name,
                    "Description": description,
                    "Status": status,
                    "Priority": priority,
                    "Start Date": pd.to_datetime(start_date),
                    "End Date": pd.to_datetime(end_date)
                }

                df = load_tasks()
                df = pd.concat([df, pd.DataFrame([new_task])], ignore_index=True)
                save_tasks(df)
                st.success("✅ Task added successfully and saved to CSV!")

# --- GRID VIEW (TABLE) ---
def grid_view():
    st.subheader("📋 Grid View (All Tasks)")
    df = load_tasks()

    if df.empty:
        st.info("No tasks yet — add your first task above.")
    else:
        colored_df = df.copy()
        colored_df["Status Color"] = colored_df["Status"].map(STATUS_COLORS)
        st.dataframe(df, use_container_width=True)

        with st.expander("📥 Download Tasks CSV"):
            if os.path.exists(CSV_FILE):
                with open(CSV_FILE, "rb") as f:
                    st.download_button("Download CSV", f, file_name="tasks_log.csv", mime="text/csv")
            else:
                st.warning("No CSV file found yet.")

# --- TRELLO BOARD ---
def trello_view():
    st.subheader("🗂️ Kanban Board")

    df = load_tasks()
    if df.empty:
        st.info("No tasks available yet.")
        return

    statuses = list(STATUS_COLORS.keys())
    cols = st.columns(len(statuses))

    for i, status in enumerate(statuses):
        with cols[i]:
            st.markdown(f"### {status}")
            st.markdown(f'<div style="background-color:{STATUS_COLORS[status]};height:5px;border-radius:4px;margin-bottom:8px;"></div>', unsafe_allow_html=True)
            tasks = df[df["Status"] == status]
            for _, task in tasks.iterrows():
                st.markdown(f"""
                <div style="background-color:white;border:1px solid #ddd;border-radius:10px;
                padding:10px;margin-bottom:10px;box-shadow:2px 2px 5px rgba(0,0,0,0.05)">
                    <b>{task['Task Name']}</b><br>
                    <small>{task['Description']}</small><br>
                    <span style="font-size:12px;color:gray;">
                        ⏱ {task['Start Date'].strftime('%Y-%m-%d')} → {task['End Date'].strftime('%Y-%m-%d')}
                    </span><br>
                    <span style="background-color:{PRIORITY_COLORS[task['Priority']]};padding:2px 6px;
                    border-radius:5px;font-size:12px;">{task['Priority']}</span>
                </div>
                """, unsafe_allow_html=True)

# --- CALENDAR VIEW ---
def calendar_view():
    st.subheader("🗓️ Calendar View")
    df = load_tasks()

    if df.empty:
        st.info("No tasks yet.")
        return

    year = st.selectbox("Select Year", sorted(df["Start Date"].dt.year.unique()), index=0)
    month = st.selectbox("Select Month", list(calendar.month_name)[1:], index=date.today().month - 1)

    month_idx = list(calendar.month_name).index(month)
    cal = calendar.Calendar()

    days = list(cal.itermonthdates(year, month_idx))
    grid = ""
    for i, day in enumerate(days):
        if i % 7 == 0:
            grid += "<div style='display:flex;'>"
        if day.month == month_idx:
            daily_tasks = df[df["Start Date"].dt.date == day]
            color = "#f8f9fa"
            if not daily_tasks.empty:
                color = STATUS_COLORS.get(daily_tasks.iloc[0]["Status"], "#f8f9fa")
            grid += f"<div style='flex:1;text-align:center;padding:8px;margin:2px;background-color:{color};border-radius:8px;'>{day.day}</div>"
        else:
            grid += "<div style='flex:1;padding:8px;margin:2px;color:#ccc;'> </div>"
        if i % 7 == 6:
            grid += "</div>"
    st.markdown(grid, unsafe_allow_html=True)

# --- GANTT CHART ---
def gantt_view():
    st.subheader("📊 Gantt Chart")
    df = load_tasks()

    if df.empty:
        st.info("No tasks available yet.")
        return

    fig = px.timeline(
        df,
        x_start="Start Date",
        x_end="End Date",
        y="Task Name",
        color="Status",
        color_discrete_map=STATUS_COLORS,
        hover_data=["Priority", "Description"]
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(height=500, margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig, use_container_width=True)

# --- MAIN RENDER FUNCTION ---
def render_task_planner():
    st.title("🧭 Project Task Planner")
    st.markdown("A Monday.com-style dashboard to manage your projects — with calendar, Gantt chart, and Kanban board.")

    add_task_form()

    st.markdown("---")
    view = st.radio(
        "Choose your view:",
        ["Grid", "Kanban Board", "Calendar", "Gantt Chart"],
        horizontal=True
    )

    if view == "Grid":
        grid_view()
    elif view == "Kanban Board":
        trello_view()
    elif view == "Calendar":
        calendar_view()
    elif view == "Gantt Chart":
        gantt_view()
