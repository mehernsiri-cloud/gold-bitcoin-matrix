def jobs_dashboard():
    st.title("💼 Jobs Dashboard (Trello Style)")

    # Sidebar filters
    st.sidebar.header("Job Filters")
    location_choice = st.sidebar.selectbox("🌐 Select Location", list(LOCATIONS.keys()))
    remote_only = st.sidebar.checkbox("🏠 Only remote jobs", value=False)
    company_filter = st.sidebar.text_input("🏢 Filter by company (optional)", "")

    st.markdown(f"### 🌍 Showing jobs in **{location_choice}**")

    # Trello-style columns for categories
    cols = st.columns(len(CATEGORIES))
    for col, (cat, kws) in zip(cols, CATEGORIES.items()):
        with col:
            st.markdown(
                f"<div style='background-color:#004080;color:white;padding:10px;border-radius:8px;text-align:center;font-weight:bold'>📌 {cat}</div>",
                unsafe_allow_html=True
            )
            found_any = False
            for kw in kws:
                df_jobs = fetch_jobs_adzuna(
                    kw,
                    LOCATIONS[location_choice],
                    location_choice,
                    max_results=5,
                    remote_only=remote_only,
                    company_filter=company_filter if company_filter else None
                )
                if not df_jobs.empty:
                    found_any = True
                    for _, job in df_jobs.iterrows():
                        st.markdown(f"""
                            <div style='background-color:#f8f9fa;padding:10px;border-radius:8px;margin-top:8px;box-shadow:0 1px 3px rgba(0,0,0,0.1)'>
                                <b><a href="{job['link']}" target="_blank" style="text-decoration:none;color:#004080">{job['title']}</a></b><br>
                                <span style='color:gray'>{job['company']} | {job['location']}</span>
                            </div>
                        """, unsafe_allow_html=True)
            if not found_any:
                st.info(f"No jobs found for {cat} in {location_choice}.")
