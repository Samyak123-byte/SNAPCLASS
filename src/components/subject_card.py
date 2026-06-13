import streamlit as st
import re

def clean_html(raw_text):
    """Agar data ke andar galti se HTML code store ho gaya hai, toh yeh use saaf kar dega"""
    if not isinstance(raw_text, str):
        return str(raw_text)
    # Yeh regex kisi bhi <...> tag ko remove kar deta hai
    clean_text = re.sub(r'<[^>]*>', '', raw_text)
    # Agar text ke andar 'Attended' ya extra inline css bacha ho toh use hatayein
    if "style=" in clean_text or "Attended" in clean_text:
        return "N/A"
    return clean_text.strip()

def subject_card(name, code, section, stats=None, footer_callback=None):
    with st.container(border=True):
        
        # Database text ko render karne se pehle clean karenge
        clean_name = clean_html(name)
        clean_code = clean_html(code)
        clean_section = clean_html(section)
        
        # 1. Header aur Details (Pure Streamlit, No HTML)
        st.subheader(clean_name)
        st.caption(f"Code: {clean_code}  |  Section: {clean_section}")
        
        # 2. Attendance Stats
        if stats and isinstance(stats, list):
            stat_cols = st.columns(len(stats))
            for idx, item in enumerate(stats):
                if len(item) == 3:
                    icon, label, value = item
                    with stat_cols[idx]:
                        st.metric(label=f"{idx+1}. {icon} {label}", value=int(value))
        
        st.write("")
        
        # 3. Action Button
        if footer_callback is not None:
            footer_callback()
