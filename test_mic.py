# test_mic.py - This is just for testing

import streamlit as st
from streamlit_mic_recorder import speech_to_text

st.set_page_config(page_title="Mic Test", page_icon="🎤")
st.title("🎤 Testing Hindi Microphone")

# Simple mic button
hindi_text = speech_to_text(
    language='hi',
    start_prompt="🎙️ Click and Speak Hindi",
    stop_prompt="⏹️ Stop",
    just_once=True,
    key="test_mic"
)

# Show the result
if hindi_text:
    st.success(f"✅ You said: {hindi_text}")
    st.write(f"**Text length:** {len(hindi_text)} characters")
else:
    st.info("👆 Click the mic button above and speak in Hindi")

# Instructions
with st.expander("📖 How to use"):
    st.markdown("""
    1. Click the **🎙️ Click and Speak Hindi** button
    2. **Allow microphone access** when browser asks
    3. Speak clearly in **Hindi**
    4. Click **⏹️ Stop** when done
    5. Your spoken text will appear above!
    """)