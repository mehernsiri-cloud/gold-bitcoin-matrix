# real_estate_bot.py
import streamlit as st

def real_estate_dashboard():
    st.title("🏠 Real Estate Agent Bot – Dubai Sales Assistant")

    st.markdown("""
    This assistant helps **Real Estate Sales Teams in Dubai** with:
    - 🕵️ Client prospection & qualification  
    - 📞 Responding to initial client queries  
    - 📝 Collecting client requirements  
    - 📅 Scheduling meetings with agents  
    - 📊 Sharing investment options and ROI projections  
    """)

    # --- Chat state ---
    if "real_estate_chat" not in st.session_state:
        st.session_state.real_estate_chat = []

    # --- User input ---
    user_input = st.text_input("💬 Ask me anything about Dubai Real Estate:")

    if user_input:
        # Save user question
        st.session_state.real_estate_chat.append({"role": "user", "content": user_input})

        # --- Simple rules for responses ---
        if "apartment" in user_input.lower():
            response = "We have several luxury apartments in Downtown Dubai. Would you like floor plans or to schedule a visit?"
        elif "villa" in user_input.lower():
            response = "Palm Jumeirah and Dubai Hills offer excellent villas. Can I know your budget range?"
        elif "roi" in user_input.lower() or "investment" in user_input.lower():
            response = "Dubai properties provide ROI between **6-9% annually**, depending on location. Should I prepare a projection for you?"
        elif "meeting" in user_input.lower() or "schedule" in user_input.lower():
            response = "Great! Please share your preferred date and time to schedule a meeting with our agent."
        else:
            response = "Thanks for your interest! Could you tell me more about your preferred property type and budget?"

        # Save bot response
        st.session_state.real_estate_chat.append({"role": "bot", "content": response})

    # --- Display conversation ---
    st.markdown("### 💬 Conversation")
    for msg in st.session_state.real_estate_chat:
        if msg["role"] == "user":
            st.markdown(f"👤 **Client:** {msg['content']}")
        else:
            st.markdown(f"🤖 **Agent Bot:** {msg['content']}")
